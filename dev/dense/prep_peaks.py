"""Пакет дорожки интервалов для плотного обучения: пики DAP-seq StrepTRN (круг улучшений, решение владельца 21).

Пик — интервал «сайт где-то здесь» (SPEC п. 4.5): пример — окно `WP` оснований вокруг вершины пика, оценка примера —
логарифм средней экспоненты оценок оснований на части пика внутри оцениваемой области (вершина ± `RP`). Обучающие пики —
строки дорожки `interval` обучающей части захода 0 по коду проекта (`prokabind.train._TrainingData`: те же исключения,
что у прежнего обучения); пики белков контрольной части — отдельно, только для оценки. Пики отложенного набора и
проверочных белков захода не читаются (берутся только строки `_TrainingData` и белки части `control`).

В пакете: геном (коды), пики (белок, границы, вершина, место интервала в оцениваемой области, вес — 0,5 × нормированная
сила по п. 4.6), собственные метки каждого белка и его вариантов ≥ 90% (слитые интервалы — запрет для неразмеченных
окон и для пары «чужой»), годные чужие белки по правилу проекта (`_allowed_foreign`: другая группа родства и другое
семейство), запас центров неразмеченных окон с долей AT (вне «липких» мест).
Запуск: ./.venv/bin/python -I -B dev/dense/prep_peaks.py  ->  {PACKS}/peaks.pkl"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

WP, RP = 341, 100                       # окно и половина оцениваемой области: 341 − 2·70 = 201 = 2·100 + 1
HP = (WP - 1) // 2
GENOME = 'NC_003888.3'
cfg = json.load(open(CFG_OURS))
b = T._TrainingData(cfg, CHOICES, 0, device='cpu')
split = {r['protein']: r for r in read_tsv('data/derived/split.tsv') if r['part'] != 'holdout'}
seq = D.genome_sequence(b.dirs, GENOME).upper()
L = len(seq)
code = codes(seq)
is_at = np.concatenate(([0], np.cumsum((code == 0) | (code == 3))))
is_n = np.concatenate(([0], np.cumsum(code == 4)))
sticky = D.sticky_mask(b.labels, GENOME, L)
sticky_c = np.concatenate(([0], np.cumsum(sticky)))
print('геном %s: %d оснований, AT %.3f, липких оснований %d' % (GENOME, L, is_at[-1] / L, int(sticky.sum())))

# `b.labels` — обучающие метки до обрезки широких пиков (`_build_rows` режет их под окно прежнего обучения); здесь окно своё
train_keys = {(r['protein'], int(r['start']), int(r['end'])) for r in b.labels if r['lane'] == 'interval' and r['genome'] == GENOME}
control = {p for p, r in split.items() if r['part'] == 'control'}
rows = [r for r in read_tsv('data/derived/peaks.tsv') if r['protein'] in split and r['genome'] == GENOME]
drop = collections.Counter()


def take(part_rows):
    out = collections.defaultdict(list)
    for r in part_rows:
        s, e, c, st = int(r['start']), int(r['end']), int(r['summit']), float(r['strength'])
        if c - HP < 0 or c + HP + 1 > L:
            drop['edge'] += 1; continue
        if is_n[c + HP + 1] - is_n[c - HP]:
            drop['unknown_base'] += 1; continue
        lo, hi = max(s, c - RP) - (c - RP), min(e, c + RP + 1) - (c - RP)
        if hi - lo < 20:
            drop['interval_outside_scored'] += 1; continue
        if sticky_c[c - RP + hi] - sticky_c[c - RP + lo]:
            drop['sticky'] += 1; continue
        for k, v in (('protein', r['protein']), ('start', s), ('end', e), ('summit', c), ('strength', st), ('lo', lo), ('hi', hi),
                     ('at', float(is_at[c + HP + 1] - is_at[c - HP]) / WP)):
            out[k].append(v)
    return out


tr = take([r for r in rows if (r['protein'], int(r['start']), int(r['end'])) in train_keys])
ct = take([r for r in rows if r['protein'] in control])
# кратность места: сколько пиков (обучающих и контрольных белков; пики проверочных белков захода сюда не входят)
# накрывают вершину пика. «Горячие» места, где садится всё подряд, белок не различают — их можно отсеять (`--peak-mult`)
_cov = np.zeros(L + 1, dtype=np.int32)
for d in (tr, ct):
    np.add.at(_cov, np.asarray(d['start']), 1); np.add.at(_cov, np.asarray(d['end']), -1)
_depth = np.cumsum(_cov)[:L]
for d in (tr, ct):
    d['mult'] = _depth[np.asarray(d['summit'])].astype(int).tolist()
for d in (tr, ct):
    v = np.log1p(np.asarray(d['strength']))
    d['weight'] = (0.5 * np.clip(v / v.mean(), 0.5, 1.0)).astype(np.float32).tolist()      # вес дорожки 0,5 × сила в пределах 0,5–1
print('обучающих пиков %d у %d белков; контрольных %d у %d белков; снято: %s' % (
    len(tr['protein']), len(set(tr['protein'])), len(ct['protein']), len(set(ct['protein'])), dict(drop)))

# собственные метки: пики белка и его вариантов ≥ 90% (все пики таблицы без отложенного набора — метка есть метка),
# и его сайты в этом геноме
vg_members = collections.defaultdict(list)
for p, r in split.items():
    vg_members[r['variant_group']].append(p)
own = collections.defaultdict(list)
for r in rows:
    own[r['protein']].append((int(r['start']), int(r['end'])))
for r in read_tsv('data/derived/sites.tsv'):
    if r['protein'] in split and r['genome'] == GENOME:
        own[r['protein']].append((int(r['start']), int(r['end'])))
marks = {}
for p in set(tr['protein']) | set(ct['protein']):
    iv = []
    for q in vg_members[split[p]['variant_group']]:
        iv += own.get(q, [])
    m = D.merge_intervals(iv)
    marks[p] = (np.asarray([x for x, _ in m], dtype=np.int64), np.asarray([y for _, y in m], dtype=np.int64))

train_set, ctrl_set = set(tr['protein']), set(ct['protein'])
allowed = {}
for p in sorted(train_set):
    keys, _ = b._allowed_foreign(p)
    allowed[p] = sorted(set(keys) & train_set)
for p in sorted(ctrl_set):
    allowed[p] = sorted(q for q in ctrl_set if q != p and b.relatedness.get(q) != b.relatedness.get(p)
                        and T.is_other_family(b.families.get(q), b.families.get(p)))
print('годных чужих: у обучающих белков медиана %d, без чужих %d; у контрольных медиана %d, без чужих %d' % (
    np.median([len(allowed[p]) for p in train_set]), sum(1 for p in train_set if not allowed[p]),
    np.median([len(allowed[p]) for p in ctrl_set]), sum(1 for p in ctrl_set if not allowed[p])))

rng = np.random.default_rng([SPLIT_NUMBER, 2110])
c = rng.integers(HP, L - HP - 1, size=600000)
ok = ((is_n[c + HP + 1] - is_n[c - HP]) == 0) & ((sticky_c[c + RP + 1] - sticky_c[c - RP]) == 0)
c = np.unique(c[ok])[:400000] if ok.sum() > 400000 else np.unique(c[ok])
at = (is_at[c + HP + 1] - is_at[c - HP]) / WP
order = np.argsort(at, kind='stable')
pack = {'genome': GENOME, 'seq': code, 'WP': WP, 'RP': RP, 'train': dict(tr), 'control': dict(ct), 'marks': marks, 'allowed': allowed,
        'bg_centres': c[order].astype(np.int64), 'bg_at': at[order].astype(np.float32),
        'groups': {p: b.relatedness.get(p) for p in train_set | ctrl_set}, 'family': {p: str(b.families.get(p)) for p in train_set | ctrl_set}}
path = os.path.join(PACKS, 'peaks.pkl')
pickle.dump(pack, open(path, 'wb'), protocol=4)
n = collections.Counter(ct['protein'])
print('контрольных белков с ≥ 10 пиками: %d; с ≥ 30: %d' % (sum(1 for v in n.values() if v >= 10), sum(1 for v in n.values() if v >= 30)))
print('запас неразмеченных центров %d; пакет %s, %.1f МБ' % (c.size, path, os.path.getsize(path) / 1e6))
