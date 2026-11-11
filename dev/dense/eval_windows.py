"""Оценка ансамбля по пакету окон (`winpack.py`) — та же мера, что у `eval_control.py`, но без геномов на машине счёта
(круг улучшений, решение владельца 21). На контрольной части числа обязаны совпасть с `eval_control.py`.

Для пары «белок + запись генома»: оценка сайта — наибольшая оценка основания на отрезке сайта; оценка окна фона —
наибольшая оценка основания на отрезке той же ширины с центром в центре фона; у ансамбля — средняя z-оценок членов (z —
по фону пары у каждого члена). Считается со своим белком, с нулевым вектором (`--null`) и с вектором чужого белка
(`--foreign K`: белки того же пакета из списка годных чужих с близким составом главного генома; на каждом третьем центре).
Сохраняются оценки сайтов и фона по парам — разбор (AUROC по полосам, суммы с переносом мотива) делается дома.

Запуск: eval_windows.py ИМЯ=ПРОГОН[@ТОЧКА][+ПРОГОН@ТОЧКА…] [...] --pack PACK.pkl --out OUT.pkl [--device cuda] [--threads N]
        [--foreign K] [--null] [--part K/N]
`--part K/N` — считать только пары с номером ≡ K (mod N): N процессов разом вместо одного (счёт идёт парой за парой и
упирается в один поток); выбор чужих белков от части не зависит. Части (OUT.pkl.partK) сшивает
`eval_windows.py --merge OUT.pkl N`: словари пар объединяются, таблица по белкам пересчитывается из пар."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from prokabind.model import build

argv = sys.argv[1:]
opt = lambda name, default=None: argv[argv.index(name) + 1] if name in argv else default
K = int(opt('--foreign', 0))
NULL = '--null' in argv
torch.set_num_threads(int(opt('--threads', 12)))
DEV = torch.device(opt('--device', 'cpu'))
pack_path = opt('--pack')
pack_path = pack_path if os.path.isabs(pack_path) else os.path.join(PACKS, pack_path)
WP = pickle.load(open(pack_path, 'rb'))
items, band, groups = WP['items'], WP['band'], WP['groups']
out_path = opt('--out', opt('--merge'))
out_path = out_path if os.path.isabs(out_path) else os.path.join(NIGHT, out_path)
FEATS = Features(json.load(open(CFG_BAC)))


def score_segments(model, vec, segs, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(segs), batch):
            x = onehot_codes(segs[i:i + batch], DEV)
            v = vec.expand(x.shape[0], -1)
            y = (model._trunk(x, v) + model._trunk(x.flip(1).flip(2), v).flip(-1)) / 2
            out.append(y[:, CTX:x.shape[2] - CTX].float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def item_scores(model, vec, it, thin=False):
    """(оценки сайтов, список оценок фона по сайтам) — как `eval_control.window_scores_model`"""
    keep, n_c = it['bg_keep'], it['n_centres']
    rows = np.arange(keep.size) if not thin else np.flatnonzero(keep % 3 == 0)
    wmax = it['wmax']
    bg_scores = score_segments(model, vec, it['bg_segs'][rows]) if rows.size else np.zeros((0, wmax), dtype=np.float32)
    pos, bg = [], []
    by_w = collections.defaultdict(list)
    for k, seg in enumerate(it['site_segs']):
        if seg is not None:
            by_w[it['w'][k]].append(k)
    site_score = {}
    for w, ks in by_w.items():
        sc = score_segments(model, vec, np.stack([it['site_segs'][k] for k in ks]))
        for k, row in zip(ks, sc):
            site_score[k] = float(row.max())
    for k, w in enumerate(it['w']):
        pos.append(site_score.get(k, np.nan))
        off = (wmax - 1) // 2 - (w - 1) // 2
        # как у eval_control: по всем годным центрам сайта, NaN — у центра без окна (тогда массивы совпадают по длине с
        # оценками соперника `pwm_full.py`, и сумму z можно считать дома)
        vals = np.full(n_c, np.nan, dtype=np.float32)
        if rows.size:
            vals[keep[rows]] = bg_scores[:, off:off + w].max(axis=1)
        sel = it['ok'][k] if not thin else (it['ok'][k] & (np.arange(n_c) % 3 == 0))
        bg.append(vals[sel])
    return np.asarray(pos, dtype=float), bg


def zavg(parts):
    if len(parts) == 1:
        return parts[0]
    pos_z, bg_z = [], []
    for pos, bg in parts:
        B_ = np.concatenate(bg).astype(float)
        m, sd = np.nanmean(B_), np.nanstd(B_) + 1e-9
        pos_z.append((pos - m) / sd); bg_z.append([(b - m) / sd for b in bg])
    return np.mean(pos_z, axis=0), [np.mean([bz[k] for bz in bg_z], axis=0) for k in range(len(bg_z[0]))]


def auroc_of(scores):
    pos, bg = scores
    b = np.concatenate(bg).astype(float) if len(bg) else np.zeros(0)
    p, b = pos[~np.isnan(pos)], b[~np.isnan(b)]
    return float(T._auroc(p.astype(np.float64), b.astype(np.float64))) if p.size and b.size else None


def load(run, update=None):
    cfg = json.load(open(os.path.join(run, 'config.json')))
    journal = json.load(open(os.path.join(run, 'journal.json')))
    cfg['protein_paths'] = (journal.get('choices') or {}).get('protein_paths', 'both')
    model = build(cfg, protein_input=bool(journal.get('protein_input', True)))
    path = os.path.join(run, 'model.pt') if update is None else os.path.join(run, 'checkpoints', '%s.pt' % update)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    if model.retrieval is not None:
        raise ValueError('модель с полем retrieval этим оценщиком не считается')
    return model.to(DEV).eval()


def vector(model, protein):
    with torch.no_grad():
        if model.protein is None or protein is None:
            if model.protein is None:
                return model.protein_vector(None, None, 1)
            return torch.zeros(1, 128, device=DEV)
        f = FEATS.get(protein).unsqueeze(0).to(DEV)
        return model.protein_vector(f, torch.ones(1, f.shape[1], dtype=torch.bool, device=DEV), 1)


# чужие белки: главный геном (где сайтов больше) близкого состава
count = collections.Counter()
at_of_item = {}
for it in items:
    count[(it['protein'], it['genome'])] += len(it['w'])
    c = it['composition']
    at_of_item[(it['protein'], it['genome'])] = (c[0] + c[3]) / max(1, sum(c))
main = {}
for (p, g), n in sorted(count.items()):
    if p not in main or n > main[p][1]:
        main[p] = (g, n)
at = {p: at_of_item[(p, main[p][0])] for p in main}
foreign = {p: [] for p in main}
if K:
    frng = np.random.default_rng(SPLIT_NUMBER + 11)
    for p in sorted(main):
        cand = [x for x in WP['allowed'].get(p, ()) if x in at and abs(at[x] - at[p]) < 0.05]
        foreign[p] = [cand[int(i)] for i in frng.permutation(len(cand))[:K]]


def by_band(per, base=None):
    out = {}
    for bname in ('all', '>=40', '>=70', '40-70', '<40'):
        sel = [p for p in per if (bname == 'all' or band[p] == bname or (bname == '>=40' and band[p] in ('>=70', '40-70'))) and (base is None or p in base)]
        byg = collections.defaultdict(list)
        for p in sel:
            byg[groups[p]].append(per[p] - (base[p] if base is not None else 0.0))
        d = [float(np.mean(v)) for v in byg.values()]
        if not d:
            continue
        if base is None:
            out[bname] = '%.3f (%d)' % (float(np.mean(d)), len(d))
        elif len(d) >= 4:
            m, lo, hi = boot(d)
            out[bname] = '%+.3f [%+.3f; %+.3f] %d/%d' % (m, lo, hi, sum(1 for x in d if x > 0), len(d))
    return out


PART = opt('--part')
if PART:
    _k, _n = (int(x) for x in PART.split('/'))
    assert 0 <= _k < _n
    items = [it for i, it in enumerate(items) if i % _n == _k]
    out_path = '%s.part%d' % (out_path, _k)


def per_protein_of(res):
    """таблица по белкам из словаря пар — тем же счётом, что в цикле ниже"""
    tmp = collections.defaultdict(lambda: collections.defaultdict(list))
    for (p, g), rec in res.items():
        v = auroc_of(rec['own'])
        if v is not None:
            tmp['own'][p].append(v)
        v1 = auroc_of(rec['own_thin']) if 'own_thin' in rec else None
        if 'null' in rec:
            v0 = auroc_of(rec['null'])
            if v0 is not None and v1 is not None:
                tmp['null'][p].append(v0); tmp['own_thin_n'][p].append(v1)
        if rec.get('foreign'):
            fv = [x for x in (auroc_of(x) for x in rec['foreign']) if x is not None]
            if fv and v1 is not None:
                tmp['own_thin_f'][p].append(v1); tmp['other'][p].append(float(np.mean(fv)))
    return {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in tmp.items()}


def report(name, per, seconds, n_members):
    print('%s  (%d с, членов %d)' % (name, seconds, n_members))
    print('   own AUROC: ' + ' | '.join('%s %s' % kv for kv in by_band(per['own']).items()))
    if 'null' in per:
        print('   null: ' + ' | '.join('%s %s' % kv for kv in by_band(per['null']).items()))
        print('   own - null: ' + ' | '.join('%s %s' % kv for kv in by_band(per['own_thin_n'], per['null']).items()))
    if 'other' in per:
        print('   own - foreign: ' + ' | '.join('%s %s' % kv for kv in by_band(per['own_thin_f'], per['other']).items()), flush=True)
    sys.stdout.flush()


if '--merge' in argv:
    target = argv[argv.index('--merge') + 1]
    target = target if os.path.isabs(target) else os.path.join(NIGHT, target)
    n_parts = int(argv[argv.index('--merge') + 2])
    parts = [pickle.load(open('%s.part%d' % (target, k), 'rb')) for k in range(n_parts)]
    store = pickle.load(open(target, 'rb')) if os.path.exists(target) else {}
    for name in parts[0]:
        res = {}
        for part in parts:
            assert not (set(res) & set(part[name]['pairs'])), 'пара в двух частях'
            res.update(part[name]['pairs'])
        assert len({part[name]['members'] for part in parts}) == 1, 'части посчитаны разными ансамблями'
        want = {(it['protein'], it['genome']) for it in WP['items']}
        assert set(res) == want, 'в частях не все пары пакета: %d из %d' % (len(res), len(want))
        per = per_protein_of(res)
        store[name] = {'pairs': res, 'per_protein': per, 'members': parts[0][name]['members'], 'pack': pack_path,
                       'seconds': max(part[name]['seconds'] for part in parts), 'parts': n_parts}
        report(name, per, store[name]['seconds'], len(parts[0][name]['members'].split('+')))
    pickle.dump(store, open(target, 'wb'), protocol=4)
    sys.exit(0)

store = pickle.load(open(out_path, 'rb')) if os.path.exists(out_path) else {}
for arg in [x for x in argv if '=' in x and not x.startswith('--')]:
    name, spec = arg.split('=', 1)
    t0 = time.time()
    members = []
    for one in spec.split('+'):
        run, update = (one.split('@') + [None])[:2]
        members.append((load(run, update), {}))

    def vec_of(k, protein):
        model, cache = members[k]
        if protein not in cache:
            cache[protein] = vector(model, protein)
        return cache[protein]

    res, tmp = {}, collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        p, g = it['protein'], it['genome']
        own = zavg([item_scores(m, vec_of(k, p), it) for k, (m, _) in enumerate(members)])
        rec = {'own': own}
        v = auroc_of(own)
        if v is not None:
            tmp['own'][p].append(v)
        if NULL or foreign[p]:
            rec['own_thin'] = zavg([item_scores(m, vec_of(k, p), it, thin=True) for k, (m, _) in enumerate(members)])
            v1 = auroc_of(rec['own_thin'])
        if NULL:
            rec['null'] = zavg([item_scores(m, vec_of(k, None), it, thin=True) for k, (m, _) in enumerate(members)])
            v0 = auroc_of(rec['null'])
            if v0 is not None and v1 is not None:
                tmp['null'][p].append(v0); tmp['own_thin_n'][p].append(v1)
        if foreign[p]:
            rec['foreign'] = [zavg([item_scores(m, vec_of(k, x), it, thin=True) for k, (m, _) in enumerate(members)]) for x in foreign[p]]
            fv = [auroc_of(x) for x in rec['foreign']]
            fv = [x for x in fv if x is not None]
            if fv and v1 is not None:
                tmp['own_thin_f'][p].append(v1); tmp['other'][p].append(float(np.mean(fv)))
        res[(p, g)] = rec
    per = {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in tmp.items()}
    store[name] = {'pairs': res, 'per_protein': per, 'members': spec, 'pack': pack_path, 'seconds': round(time.time() - t0)}
    pickle.dump(store, open(out_path, 'wb'), protocol=4)
    report(name, per, time.time() - t0, len(members))
