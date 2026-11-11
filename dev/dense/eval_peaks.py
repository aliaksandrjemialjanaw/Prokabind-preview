"""Оценка на пиках DAP-seq белков контрольной части (круг улучшений, решение владельца 21).

Шаг `build` (дома, нужен пакет `peaks.pkl`): закреплённый набор — у каждого контрольного белка с ≥ 10 пиками до 150 пиков
(по хэшу), у каждого пика: окно пика (вершина ± 170), неразмеченное окно того же состава вне пиков белка и его вариантов,
пик чужого контрольного белка (другая группа родства и семейство), не задевающий пики своего. -> {PACKS}/peaks_eval.pkl
Шаг `score` (на машине): ансамбль считает оценку интервала (логарифм средней экспоненты по части пика в оцениваемой
области, границы своего пика у всех окон) и печатает по белкам, в среднем по группам родства:
  pk_bg      — AUROC «свой пик против неразмеченного окна» (свой белок);
  pk_foreign — AUROC «свой пик против чужого пика» (свой белок) — вид меры Б6 «свои пики против чужих»;
  pk_swap    — доля своих пиков, где оценка со своим белком выше, чем с вектором чужого контрольного белка
               (средняя по `--foreign K` чужим); 0,5 — белок не различается.
Запуск: eval_peaks.py build
        eval_peaks.py score ИМЯ=ПРОГОН@ТОЧКА+… [...] [--device cuda] [--threads N] [--foreign 2] [--out FILE.json]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

stage = sys.argv[1]
argv = sys.argv[2:]
opt = lambda name, default=None: argv[argv.index(name) + 1] if name in argv else default
EVAL = os.path.join(PACKS, opt('--eval', 'peaks_eval.pkl'))
MULT = int(opt('--mult', 0))                  # build: только пики, чью вершину накрывает не больше стольких пиков (отсев «горячих» мест)

if stage == 'build':
    pk = pickle.load(open(os.path.join(PACKS, 'peaks.pkl'), 'rb'))
    d, seq, H, RP = pk['control'], pk['seq'], (pk['WP'] - 1) // 2, pk['RP']
    rows_of = collections.defaultdict(list)
    for i, q in enumerate(d['protein']):
        if not MULT or d['mult'][i] <= MULT:
            rows_of[q].append(i)
    at = np.asarray(d['at'])
    order = np.argsort(at, kind='stable'); at_sorted = at[order]

    def touches(protein, start, end, margin=100):
        starts, ends = pk['marks'][protein]
        k = int(np.searchsorted(ends, start - margin, 'right'))
        return starts.size > 0 and k < starts.size and starts[k] < end + margin

    win = lambda c: seq[c - H:c + H + 1]
    items = []
    for p in sorted(rows_of):
        if len(rows_of[p]) < 10:
            continue
        rng = np.random.default_rng([SPLIT_NUMBER, int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)])
        pick = sorted(rows_of[p], key=lambda i: hashlib.sha256(('%s:%d' % (p, d['summit'][i])).encode()).hexdigest())[:150]
        allowed = set(pk['allowed'].get(p, ()))
        pos, bg, fp, span = [], [], [], []
        for i in pick:
            lo = int(np.searchsorted(pk['bg_at'], at[i] - 0.02, 'left')); hi = int(np.searchsorted(pk['bg_at'], at[i] + 0.02, 'right'))
            c = None
            for _ in range(60):
                c = int(pk['bg_centres'][int(rng.integers(lo, hi))]) if hi > lo else int(pk['bg_centres'][int(rng.integers(pk['bg_centres'].size))])
                if not touches(p, c - RP, c + RP + 1):
                    break
            j = None
            lo2 = int(np.searchsorted(at_sorted, at[i] - 0.03, 'left')); hi2 = int(np.searchsorted(at_sorted, at[i] + 0.03, 'right'))
            for _ in range(60):
                if hi2 <= lo2:
                    break
                t = int(order[int(rng.integers(lo2, hi2))])
                x = d['protein'][t]
                if x in allowed and not touches(p, d['start'][t], d['end'][t]):
                    j = t
                    break
            pos.append(win(d['summit'][i])); bg.append(win(c)); span.append((d['lo'][i], d['hi'][i]))
            fp.append(win(d['summit'][j]) if j is not None else None)
        items.append({'protein': p, 'pos': np.stack(pos), 'bg': np.stack(bg), 'foreign_peak': fp, 'span': span,
                      'allowed': sorted(allowed & {q for q in rows_of if len(rows_of[q]) >= 10})})
    pickle.dump({'items': items, 'groups': pk['groups'], 'family': pk['family']}, open(EVAL, 'wb'), protocol=4)
    print('набор оценки на пиках: белков %d, пиков %d, с чужим пиком %d -> %s (%.1f МБ)' % (
        len(items), sum(len(it['span']) for it in items), sum(1 for it in items for x in it['foreign_peak'] if x is not None), EVAL,
        os.path.getsize(EVAL) / 1e6))
    sys.exit(0)

from prokabind.model import build
torch.set_num_threads(int(opt('--threads', 12)))
DEV = torch.device(opt('--device', 'cpu'))
K = int(opt('--foreign', 2))
E_ = pickle.load(open(EVAL, 'rb'))
items, groups = E_['items'], E_['groups']
FEATS = Features(json.load(open(CFG_BAC)))
out_path = opt('--out', 'eval_peaks.json')
out_path = out_path if os.path.isabs(out_path) else os.path.join(NIGHT, out_path)
store = json.load(open(out_path)) if os.path.exists(out_path) else {}


def load(run, update=None):
    cfg = json.load(open(os.path.join(run, 'config.json')))
    journal = json.load(open(os.path.join(run, 'journal.json')))
    cfg['protein_paths'] = (journal.get('choices') or {}).get('protein_paths', 'both')
    model = build(cfg, protein_input=bool(journal.get('protein_input', True)))
    path = os.path.join(run, 'model.pt') if update is None else os.path.join(run, 'checkpoints', '%s.pt' % update)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model.to(DEV).eval()


def vector(model, protein):
    with torch.no_grad():
        if model.protein is None:
            return model.protein_vector(None, None, 1)
        f = FEATS.get(protein).unsqueeze(0).to(DEV)
        return model.protein_vector(f, torch.ones(1, f.shape[1], dtype=torch.bool, device=DEV), 1)


def scores(model, vec, wins, span, batch=256):
    """оценка интервала у каждого окна: логарифм средней экспоненты по [lo, hi) своей строки"""
    out = []
    with torch.no_grad():
        for i in range(0, len(wins), batch):
            x = onehot_codes(wins[i:i + batch], DEV)
            s = window_scores(model, x, vec.expand(x.shape[0], -1))
            lo = torch.tensor([a_ for a_, _ in span[i:i + batch]], device=DEV); hi = torch.tensor([b_ for _, b_ in span[i:i + batch]], device=DEV)
            out.append(interval_scores(s, lo, hi).float().cpu().numpy())
    return np.concatenate(out)


def auroc(pos, neg):
    return float(T._auroc(np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)))


frng = np.random.default_rng(SPLIT_NUMBER + 13)
foreign = {it['protein']: [it['allowed'][int(i)] for i in frng.permutation(len(it['allowed']))[:K]] for it in items}
for arg in [x for x in argv if '=' in x and not x.startswith('--')]:
    name, spec = arg.split('=', 1)
    t0 = time.time()
    members = [load(*((one.split('@') + [None])[:2])) for one in spec.split('+')]
    per = collections.defaultdict(dict)
    for it in items:
        p = it['protein']
        zs = {'pos': [], 'bg': [], 'fp': [], 'swap': []}
        has_fp = np.asarray([x is not None for x in it['foreign_peak']])
        fp_w = np.stack([x for x in it['foreign_peak'] if x is not None]) if has_fp.any() else None
        fp_span = [s for s, h in zip(it['span'], has_fp) if h]
        for m in members:
            v = vector(m, p)
            pos, bg = scores(m, v, it['pos'], it['span']), scores(m, v, it['bg'], it['span'])
            mu, sd = bg.mean(), bg.std() + 1e-9
            zs['pos'].append((pos - mu) / sd); zs['bg'].append((bg - mu) / sd)
            if fp_w is not None:
                zs['fp'].append((scores(m, v, fp_w, fp_span) - mu) / sd)
            if foreign[p] and m.protein is not None:
                zs['swap'].append(np.mean([(scores(m, vector(m, x), it['pos'], it['span']) - mu) / sd for x in foreign[p]], axis=0))
        pos, bg = np.mean(zs['pos'], axis=0), np.mean(zs['bg'], axis=0)
        per['pk_bg'][p] = auroc(pos, bg)
        if zs['fp']:
            per['pk_foreign'][p] = auroc(pos[has_fp], np.mean(zs['fp'], axis=0))
        if zs['swap']:
            sw = np.mean(zs['swap'], axis=0)
            per['pk_swap'][p] = float(np.mean((pos > sw) + 0.5 * (pos == sw)))
    res = {}
    for key, d in per.items():
        byg = collections.defaultdict(list)
        for p, v in d.items():
            byg[groups.get(p, p)].append(v)
        vals = [float(np.mean(v)) for v in byg.values()]
        m, lo, hi = boot(vals)
        res[key] = {'mean': m, 'lo': lo, 'hi': hi, 'groups': len(vals), 'per_protein': d}
    store[name] = {'members': spec, 'seconds': round(time.time() - t0), **res}
    json.dump(store, open(out_path, 'w'), indent=1)
    print('%s (%d с, членов %d): ' % (name, time.time() - t0, len(members)) + ' | '.join(
        '%s %.3f [%.3f; %.3f] (%d групп)' % (k, v['mean'], v['lo'], v['hi'], v['groups']) for k, v in res.items()), flush=True)
