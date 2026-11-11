"""Честный соперник при дорожке библиотеки (разработка, решение владельца 20): «перенос мотива» по ВСЕМ обучающим меткам
ближайшего родственника, которые видит модель — точные сайты и PRODORIC (как в SPEC п. 3.3.1), места BacRegDB, сайты
библиотеки SigmoID того же белка, сайты мотива MEME в его пиках, — вместе с его вариантами ≥ 90% (группа вариантов наших
белков; у белков библиотеки — их группа вариантов). SPEC п. 3.3.1: «модель и соперник видят одни данные».

Соперник выбирается тем же правилом (`eval_defs.competitor_relative`: ближайший обучающий родственник не ниже границы
полосы, у которого есть метки), матрица — `eval_defs.build_pwm` с составом генома, оценка — `eval_defs.motif_transfer`.
Обучающие белки — заход 0 без контрольной части и отложенного набора; белки библиотеки — допущенные к модели захода 0.
Выход: {PACKS}/pwmfull_window_scores.pkl — оценки сайтов и фона каждой пары «белок + геном» в том же порядке, что у
`e0_window_scores_t14.pkl` (ключ 'pwm' — прежний соперник, 'pwmfull' — новый, 'ctx' — модель без белка T14).
Запуск: ./.venv/bin/python -I -B dev/dense/pwm_full.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import eval_defs as E

S = pickle.load(open(os.path.join(PACKS, 'e0_setup.pkl'), 'rb'))
items, ident = S['items'], S['identity']
WS = pickle.load(open(os.path.join(PACKS, 'e0_window_scores_t14.pkl'), 'rb'))
C = pickle.load(open(os.path.join(PACKS, 'e2_control.pkl'), 'rb'))
band, groups = C['band'], C['groups']
RT = pickle.load(open(os.path.join(PACKS, 'retrieval.pkl'), 'rb'))
lib = pickle.load(open(os.path.join(PACKS, 'library.pkl'), 'rb'))
P = pickle.load(open(os.path.join(PACKS, 'coord_bacpeaks.pkl'), 'rb'))
split = {r['protein']: r for r in read_tsv('data/derived/split.tsv')}
training = set(P['training'])
closed = {'fold0', 'control', 'holdout'}
lib_ok = {q for q, m in lib['meta'].items() if not (m['related_parts'] & closed)}
letters = 'ACGT'
sites = collections.defaultdict(lambda: collections.defaultdict(set))       # белок -> источник -> сайты
for i, p in enumerate(P['protein']):
    w = P['end'][i] - P['start'][i]
    if w > W:
        continue
    lo = HALF - (w - 1) // 2
    s = ''.join(letters[c] if c < 4 else 'N' for c in P['pos'][i][lo:lo + w])
    if 'N' not in s:
        sites[p][P['lane'][i]].add(s)
for p, site in zip(lib['rows']['protein'], lib['rows']['site']):
    if p in lib_ok:
        sites[p]['library'].add(''.join(letters[c] for c in site))
def group_of(p):
    if p in split:
        return ('ours', split[p]['variant_group'])
    if p in lib['meta']:
        return ('lib', lib['meta'][p]['variant_group'])
    return ('bac', p)                                  # белок только BacRegDB вне таблицы библиотеки


members = collections.defaultdict(set)
for p in training | lib_ok:
    members[group_of(p)].add(p)


cfg = json.load(open(os.path.join(ROOT, 'work/tasks/T502/configs/updates_1600.json')))
dirs = D.genome_dirs(cfg)
_seq = {}


def genome_seq(g):
    if g not in _seq:
        _seq[g] = D.genome_sequence(dirs, g).upper()
    return _seq[g]


def blocked_cumsum(it):
    b = np.zeros(it['length'], dtype=bool); b[it['blocked_idx']] = True
    return np.concatenate(([0], np.cumsum(b)))


def windows(it):
    closed_ = blocked_cumsum(it); n = it['length']; out = []
    for (a0, c0) in it['sites']:
        w = c0 - a0
        if w < 1 or w > n:
            continue
        st = it['centres'] - (w - 1) // 2
        st = st[(st >= 0) & (st + w <= n)]
        st = st[(closed_[st + w] - closed_[st]) == 0]
        out.append((a0, w, st))
    return out


def window_scores(it, track):
    pos, bg = [], []
    for a0, w, st in windows(it):
        seg = track[a0:a0 + w]
        vals = track[st[:, None] + np.arange(w)[None, :]] if st.size else np.zeros((0, w))
        pos.append(np.nan if np.all(np.isnan(seg)) else np.nanmax(seg))
        allnan = np.all(np.isnan(vals), axis=1)
        bg.append(np.where(allnan, np.nan, np.nanmax(np.where(np.isnan(vals), -np.inf, vals), axis=1)))
    return np.asarray(pos, dtype=float), bg


def auroc_of(sc):
    pos, bg = sc
    b = np.concatenate(bg).astype(float); p = pos[~np.isnan(pos)]; b = b[~np.isnan(b)]
    return float(T._auroc(p, b)) if p.size and b.size else None


rel = RT['rel']
out = dict(WS)
per = collections.defaultdict(lambda: collections.defaultdict(list))
info = {}
for it in items:
    p, g = it['protein'], it['genome']
    if band[p] == '<40':
        continue
    low = 0.7 if band[p] == '>=70' else 0.4
    cand = {}
    for q, v in rel.get(p, {}).items():
        if (q in training or q in lib_ok) and q in sites and v >= low:
            k = group_of(q)
            if k not in cand or v > cand[k][0]:
                cand[k] = (v, q)
    if not cand:
        continue
    k, (v, q) = max(cand.items(), key=lambda kv: (kv[1][0], kv[1][1]))
    pooled = collections.defaultdict(set)
    for m in members[k]:
        for src, ss in sites.get(m, {}).items():
            pooled[src] |= ss
    full = sorted(set().union(*pooled.values()))
    if len(full) > 60:                                   # выравнивание build_pwm квадратично: не больше 60 сайтов, отбор по sha256
        full = sorted(full, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:60]
    track = E.motif_transfer(full, genome_seq(g))
    sc = window_scores(it, track)
    out[(p, g)] = dict(out.get((p, g), {}), pwmfull=sc)
    a_full = auroc_of(sc)
    if a_full is not None:
        per['pwmfull'][p].append(a_full)
    if 'pwm' in WS.get((p, g), {}) and auroc_of(WS[(p, g)]['pwm']) is not None:
        per['pwm'][p].append(auroc_of(WS[(p, g)]['pwm']))
    info[p] = {'relative_identity': v, 'sites_by_source': {s: len(x) for s, x in pooled.items()}, 'used': len(full)}
    print(p[:8], g, band[p], 'ident %.2f' % v, info[p]['sites_by_source'], 'PWM %s -> full %s' % (
        '%.3f' % per['pwm'][p][-1] if per['pwm'].get(p) else '-', '%.3f' % a_full if a_full is not None else '-'), flush=True)
pickle.dump(out, open(os.path.join(PACKS, 'pwmfull_window_scores.pkl'), 'wb'), protocol=4)
mean = {k: {p: float(np.mean(v)) for p, v in d.items() if v} for k, d in per.items()}
json.dump({'per_protein': mean, 'info': info}, open(os.path.join(NIGHT, 'pwm_full.json'), 'w'), indent=1)
for k in ('pwm', 'pwmfull'):
    for bname in ('>=70', '40-70'):
        sel = {p: v for p, v in mean[k].items() if band[p] == bname}
        print('%-8s %-6s proteins %2d  mean over groups %.3f' % (k, bname, len(sel), group_mean(sel, groups)[0]))
both = [p for p in mean['pwmfull'] if p in mean['pwm']]
byg = collections.defaultdict(list)
for p in both:
    byg[groups[p]].append(mean['pwmfull'][p] - mean['pwm'][p])
d = [float(np.mean(v)) for v in byg.values()]
m, lo, hi = boot(d)
print('pwmfull - pwm over %d groups: %+.3f [%+.3f; %+.3f], wins %d' % (len(d), m, lo, hi, sum(1 for x in d if x > 0)))
