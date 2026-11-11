"""Оценка модели на контрольной части мерой остановки SPEC п. 4.8 с выборочным фоном (разработка, решение владельца 20).

Сайты, свои метки и запретные основания — кодом обучения (`_TrainingData._control_sites`, пакет `e0_setup.pkl`); фон — 3 000
закреплённых случайных межгенных центров на пару «белок + геном» вместо всех (сверено с журналами T30 до 0,001).
Для модели считается:
  own        — AUROC чистых точных сайтов против фона со своим белком, по полосам родства (с обучающей частью захода 0);
  own − PWM  — парная разница с соперником «перенос мотива» SPEC п. 3.3.1 (оценки окон соперника — из `e0_hybrid.py`);
  hybrid     — сумма z-оценок модели и переноса мотива (z — по неразмеченному фону пары; вес 1 : 1) и её разница с PWM;
  own − foreign (`--foreign K`) — мера вида числа 11 (а): та же AUROC с вектором чужого белка контрольной части
               (другая группа родства и семейство, основной геном того же состава), на каждом третьем центре фона.
Запуск: ./.venv/bin/python -I -B dev/dense/eval_control.py NAME=RUN_DIR[@UPDATE] [...] [--foreign K] [--threads N]
Итог дописывается в {NIGHT}/eval_control.json."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from prokabind.features import load_protein
from prokabind.scan import onehot
from prokabind.model import build

argv = sys.argv[1:]
K = int(argv[argv.index('--foreign') + 1]) if '--foreign' in argv else 0
NULL = '--null' in argv          # ещё и проход с нулевым вектором белка («средний белок» после нормировки) — слепой режим модели
THREADS = int(argv[argv.index('--threads') + 1]) if '--threads' in argv else 12
torch.set_num_threads(THREADS)
DEV = torch.device(argv[argv.index('--device') + 1] if '--device' in argv else 'cpu')
S = pickle.load(open(os.path.join(PACKS, 'e0_setup.pkl'), 'rb'))
items = S['items']
# соперник: 'pwm' — перенос мотива по точным сайтам и PRODORIC родственника (e0_hybrid.py); 'pwmfull' — по всем обучающим
# меткам родственника при дорожках BacRegDB, библиотеки и пиков-сайтов (pwm_full.py)
COMP = argv[argv.index('--competitor') + 1] if '--competitor' in argv else 'pwm'
WS = pickle.load(open(os.path.join(PACKS, 'pwmfull_window_scores.pkl' if COMP == 'pwmfull' else 'e0_window_scores_t14.pkl'), 'rb'))
C = pickle.load(open(os.path.join(PACKS, 'e2_control.pkl'), 'rb'))
band, groups = C['band'], C['groups']
base_cfg = json.load(open(os.path.join(ROOT, 'work/tasks/T502/configs/updates_1600.json')))
dirs = D.genome_dirs(base_cfg)
_seq = {}


def genome_seq(g):
    if g not in _seq:
        _seq[g] = D.genome_sequence(dirs, g).upper()
    return _seq[g]


def blocked_cumsum(it):
    blocked = np.zeros(it['length'], dtype=bool)
    blocked[it['blocked_idx']] = True
    return np.concatenate(([0], np.cumsum(blocked)))


def cut(seq, s, e, circular):
    n = len(seq)
    if s >= 0 and e <= n:
        return seq[s:e]
    if not circular:
        return None
    return ''.join(seq[i % n] for i in range(s, e))


RET = None                                                     # банк мотивов родственников — у модели с полем retrieval


def score_segments(model, vec, segs, batch=256):
    vec, pick = vec if isinstance(vec, tuple) else (vec, None)
    out = []
    with torch.no_grad():
        for i in range(0, len(segs), batch):
            x = torch.cat([onehot(t) for t in segs[i:i + batch]], dim=0).to(DEV)
            v = vec.expand(x.shape[0], -1)
            if RET is None:
                y = (model._trunk(x, v) + model._trunk(x.flip(1).flip(2), v).flip(-1)) / 2
            else:
                r = RET.tensors([pick] * x.shape[0], DEV)
                y = (model._trunk(x, v, r) + model._trunk(x.flip(1).flip(2), v, r).flip(-1)) / 2
            out.append(y[:, CTX:x.shape[2] - CTX].float().cpu().numpy())
    return np.concatenate(out, axis=0)


def window_scores_model(model, vec, it):
    """оценки сайтов и фона пары — в том же порядке, что у соперника (`e0_hybrid.window_scores`); NaN — оценки нет"""
    seq = genome_seq(it['genome']); n = it['length']; circ = it['circular']
    closed = blocked_cumsum(it)
    wins = []
    for (s0, e0) in it['sites']:
        w = e0 - s0
        if w < 1 or w > n:
            continue
        st = it['centres'] - (w - 1) // 2
        ok = (st >= 0) & (st + w <= n)
        okk = np.zeros(it['centres'].size, dtype=bool)
        okk[ok] = (closed[st[ok] + w] - closed[st[ok]]) == 0
        wins.append((s0, w, okk))
    if not wins:
        return None
    wmax = max(w for _, w, _ in wins)
    cen = it['centres']
    seg_start = cen - (wmax - 1) // 2
    segs, keep = [], []
    for k, s in enumerate(seg_start.tolist()):
        t = cut(seq, s - CTX, s + wmax + CTX, circ)
        if t is None or set(t) - set('ACGT'):
            continue
        segs.append(t); keep.append(k)
    bg_scores = np.full((cen.size, wmax), np.nan, dtype=np.float32)
    if segs:
        bg_scores[np.asarray(keep)] = score_segments(model, vec, segs)
    pos, bg = [], []
    by_w = collections.defaultdict(list)
    for s0, w, _ in wins:
        by_w[w].append(s0)
    site_score = {}
    for w, starts in by_w.items():
        ss, kk = [], []
        for s0 in starts:
            t = cut(seq, s0 - CTX, s0 + w + CTX, circ)
            if t is None or set(t) - set('ACGT'):
                continue
            ss.append(t); kk.append(s0)
        if ss:
            sc = score_segments(model, vec, ss)
            for s0, row in zip(kk, sc):
                site_score[(s0, w)] = float(row.max())
    for s0, w, okk in wins:
        pos.append(site_score.get((s0, w), np.nan))
        off = (wmax - 1) // 2 - (w - 1) // 2
        bg.append(bg_scores[okk][:, off:off + w].max(axis=1))
    return np.asarray(pos, dtype=float), bg


def auroc(pos, bg):
    return float(T._auroc(np.asarray(pos, dtype=np.float64), np.asarray(bg, dtype=np.float64)))


def auroc_of(scores):
    pos, bg = scores
    b = np.concatenate(bg).astype(float)
    p, b = pos[~np.isnan(pos)], b[~np.isnan(b)]
    return auroc(p, b) if p.size and b.size else None


def zsum(first, second, lam=1.0):
    (p1, b1), (p2, b2) = first, second
    B1, B2 = np.concatenate(b1).astype(float), np.concatenate(b2).astype(float)
    ok = ~np.isnan(B1) & ~np.isnan(B2); okp = ~np.isnan(p1) & ~np.isnan(p2)
    if not ok.any() or not okp.any():
        return None
    z = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-9)
    return auroc(z(p1[okp], B1[ok]) + lam * z(p2[okp], B2[ok]), z(B1[ok], B1[ok]) + lam * z(B2[ok], B2[ok]))


def load(run, update=None):
    cpath = os.path.join(run, 'config.json')
    cfg = json.load(open(cpath if os.path.exists(cpath) else os.path.join(ROOT, 'work/config/train.json')))   # обучения до T34 — без копии
    journal = json.load(open(os.path.join(run, 'journal.json')))
    cfg['protein_paths'] = (journal.get('choices') or {}).get('protein_paths', 'both')
    model = build(cfg, protein_input=bool(journal.get('protein_input', True)))
    path = os.path.join(run, 'model.pt') if update is None else os.path.join(run, 'checkpoints', '%s.pt' % update)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.to(DEV).eval(), cfg


def vector(model, cfg, protein):
    """вектор белка, а у модели с полем retrieval — пара (вектор, ближайший обучающий родственник с матрицей)"""
    with torch.no_grad():
        if model.protein is None:
            v = model.protein_vector(None, None, 1)
        else:
            f, m = load_protein(cfg, protein)
            v = model.protein_vector(f.to(DEV), m.to(DEV), 1)
    return v if RET is None else (v, RET.pick(protein))


# чужие белки (число 11 (а)): основной геном того же состава
count = collections.Counter()
for it in items:
    count[(it['protein'], it['genome'])] += len(it['sites'])
main = {}
for (p, g), n in sorted(count.items()):
    if p not in main or n > main[p][1]:
        main[p] = (g, n)


def at_share(g):
    s = genome_seq(g); x = s.count('A') + s.count('T')
    return x / (x + s.count('C') + s.count('G'))


foreign = {p: [] for p in main}
if K:
    at = {p: at_share(main[p][0]) for p in main}
    frng = np.random.default_rng(SPLIT_NUMBER + 11)
    for p in sorted(main):
        cand = [x for x in C['allowed'].get(p, ()) if x in at and abs(at[x] - at[p]) < 0.05]
        foreign[p] = [cand[int(i)] for i in frng.permutation(len(cand))[:K]]


def by_band(per, label, base=None):
    """средняя по группам родства по полосам; с `base` — парная разница и интервал"""
    out = {}
    for bname in ('all', '>=40', '>=70', '40-70', '<40'):
        sel = [p for p in per if bname == 'all' or band[p] == bname or (bname == '>=40' and band[p] in ('>=70', '40-70'))]
        if base is not None:
            sel = [p for p in sel if p in base]
        if not sel:
            continue
        byg = collections.defaultdict(list)
        for p in sel:
            byg[groups[p]].append(per[p] - (base[p] if base is not None else 0.0))
        d = [float(np.mean(v)) for v in byg.values()]
        if base is None:
            out[bname] = {'mean': float(np.mean(d)), 'groups': len(d)}
        elif len(d) >= 4:
            m, lo, hi = boot(d)
            out[bname] = {'mean': m, 'lo': lo, 'hi': hi, 'wins': sum(1 for x in d if x > 0), 'groups': len(d)}
    return out


store_path = os.path.join(NIGHT, argv[argv.index('--out') + 1] if '--out' in argv else 'eval_control.json')
store = json.load(open(store_path)) if os.path.exists(store_path) else {}
def zavg(parts):
    """ансамбль: средняя z-оценок моделей (z — по фону пары у каждой модели отдельно); NaN, если нет хоть у одной"""
    if len(parts) == 1:
        return parts[0]
    pos_z, bg_z = [], []
    for pos, bg in parts:
        B_ = np.concatenate(bg).astype(float)
        m, sd = np.nanmean(B_), np.nanstd(B_) + 1e-9
        pos_z.append((pos - m) / sd); bg_z.append([(b - m) / sd for b in bg])
    return np.mean(pos_z, axis=0), [np.mean([bz[k] for bz in bg_z], axis=0) for k in range(len(bg_z[0]))]


for arg in [x for x in argv if '=' in x]:
    name, spec = arg.split('=', 1)
    t0 = time.time()
    members = []
    for one in spec.split('+'):
        run, update = (one.split('@') + [None])[:2]
        model, cfg = load(run, update)
        ret = Retrieval(json.load(open(os.path.join(run, 'train_proteins.json')))) if model.retrieval is not None else None
        members.append((model, cfg, ret, {}))
    model, cfg = members[0][0], members[0][1]
    RET = members[0][2]
    vec_cache = members[0][3]

    def vec(p):
        if p not in vec_cache:
            vec_cache[p] = vector(model, cfg, p)
        return vec_cache[p]

    _single = window_scores_model

    def window_scores_model(model_, v_, it_, _members=members, _single=_single):
        """у ансамбля — средняя по моделям; `v_` — вектор первой модели или белок (тогда вектор считает каждая модель)"""
        global RET
        if len(_members) == 1:
            return _single(model_, v_, it_)
        parts = []
        for m_, c_, r_, cache_ in _members:
            RET = r_
            key = it_['_protein_for_vector']
            if key not in cache_:
                cache_[key] = vector(m_, c_, key) if key is not None else None
            vv = cache_[key]
            if vv is None:
                z0 = torch.zeros(1, 128, device=DEV)
                vv = (z0, None) if r_ is not None else z0
            got = _single(m_, vv, it_)
            if got is None:
                return None
            parts.append(got)
        RET = _members[0][2]
        return zavg(parts)
    own, pwm, hyb, blindhyb, own_thin, other = ({} for _ in range(6))
    tmp = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        p, g = it['protein'], it['genome']
        it = dict(it, _protein_for_vector=p)
        sc = window_scores_model(model, vec(p), it)
        if sc is None:
            continue
        v = auroc_of(sc)
        if v is not None:
            tmp['own'][p].append(v)
        ref = WS.get((p, g), {})
        if COMP in ref:
            v1, v2, v3 = auroc_of(ref[COMP]), zsum(ref[COMP], sc, 1.0), zsum(ref[COMP], ref['ctx'], 0.5)
            if v1 is not None and v2 is not None:
                tmp['pwm'][p].append(v1); tmp['hyb'][p].append(v2); tmp['own_p'][p].append(v)
                tmp['blindhyb'][p].append(v3)
        if NULL:
            thin = dict(it, centres=it['centres'][::3])
            zero = torch.zeros_like(vec(p)[0] if isinstance(vec(p), tuple) else vec(p))
            zero = (zero, None) if RET is not None else zero
            v0 = auroc_of(window_scores_model(model, zero, dict(thin, _protein_for_vector=None)))
            v1 = auroc_of(window_scores_model(model, vec(p), thin))
            if v0 is not None and v1 is not None:
                tmp['null'][p].append(v0); tmp['own_thin2'][p].append(v1)
        if foreign[p]:
            thin = dict(it, centres=it['centres'][::3])
            base_v = auroc_of(window_scores_model(model, vec(p), thin))
            fv = [auroc_of(window_scores_model(model, vec(x), dict(thin, _protein_for_vector=x))) for x in foreign[p]]
            fv = [x for x in fv if x is not None]
            if fv and base_v is not None:
                tmp['own_thin'][p].append(base_v); tmp['other'][p].append(float(np.mean(fv)))
    per = {k: {p: float(np.mean(v)) for p, v in d.items()} for k, d in tmp.items()}
    res = {'own': by_band(per['own'], 'own'), 'seconds': round(time.time() - t0)}
    if 'pwm' in per:
        res['pwm'] = by_band(per['pwm'], 'pwm')
        res['own_minus_pwm'] = by_band(per['own_p'], 'd', per['pwm'])
        res['hybrid'] = by_band(per['hyb'], 'hyb')
        res['hybrid_minus_pwm'] = by_band(per['hyb'], 'd', per['pwm'])
        res['hybrid_minus_blindhybrid'] = by_band(per['hyb'], 'd', per['blindhyb'])
    if 'other' in per:
        res['own_minus_foreign'] = by_band(per['own_thin'], 'd', per['other'])
    if 'null' in per:
        res['null'] = by_band(per['null'], 'null')
        res['own_minus_null'] = by_band(per['own_thin2'], 'd', per['null'])
    res['per_protein'] = per
    window_scores_model = _single
    store[name] = res
    json.dump(store, open(store_path, 'w'), indent=1)
    fm = lambda d, k: ('%.3f' % d[k]['mean']) if k in d else '  -  '
    fd = lambda d, k: ('%+.3f [%+.3f; %+.3f] %d/%d' % (d[k]['mean'], d[k]['lo'], d[k]['hi'], d[k]['wins'], d[k]['groups'])) if k in d else '-'
    print('%s  (%d s)' % (name, res['seconds']))
    print('   own AUROC: all %s | >=70 %s | 40-70 %s | <40 %s' % tuple(fm(res['own'], k) for k in ('all', '>=70', '40-70', '<40')))
    if 'pwm' in res:
        print('   motif transfer: >=40 %s | 40-70 %s ;  own - PWM: >=40 %s | 40-70 %s' % (fm(res['pwm'], '>=40'), fm(res['pwm'], '40-70'), fd(res['own_minus_pwm'], '>=40'), fd(res['own_minus_pwm'], '40-70')))
        print('   hybrid (model + PWM): >=40 %s | 40-70 %s ;  hybrid - PWM: >=40 %s | 40-70 %s' % (fm(res['hybrid'], '>=40'), fm(res['hybrid'], '40-70'), fd(res['hybrid_minus_pwm'], '>=40'), fd(res['hybrid_minus_pwm'], '40-70')))
        print('   hybrid - (PWM + blind T14): >=40 %s | 40-70 %s' % (fd(res['hybrid_minus_blindhybrid'], '>=40'), fd(res['hybrid_minus_blindhybrid'], '40-70')))
    if 'null' in res:
        print('   null vector AUROC: all %s | >=70 %s | 40-70 %s | <40 %s ;  own - null: all %s | 40-70 %s | <40 %s' % (
            tuple(fm(res['null'], k) for k in ('all', '>=70', '40-70', '<40')) + tuple(fd(res['own_minus_null'], k) for k in ('all', '40-70', '<40'))))
    if 'own_minus_foreign' in res:
        print('   own - foreign: all %s | >=70 %s | 40-70 %s | <40 %s' % tuple(fd(res['own_minus_foreign'], k) for k in ('all', '>=70', '40-70', '<40')), flush=True)
