"""Маска «чистых» сайтов белков разработки BacRegDB (ревизия круга 10.11.2026, находка Н1): сайт оценки, у которого нет
общего отрезка ≥ 12 оснований (любая цепь — критерий SPEC п. 2.7.2, функции T511 `kmer_index`, `shares_kmer`) ни с одной
обучающей строкой рук круга: координатные строки белков вне набора разработки (`e2_data_bac.pkl`) и строки дорожки
RegPrecise регуляторов, допущенных правилом `rp_ok` (`train_dense.py`), во всех названных пакетах.

Для наших проверочных белков такую защиту даёт п. 2.7.2 при сборке данных; у белков разработки её нет, поэтому рядом
с числом «все сайты» `analyze_round.py` печатает число «чистые сайты».
Выход: {PACKS}/bacdev_clean_sites.json — «белок|запись» -> список 0/1 по сайтам пары; счётчики — на экран.
Запуск: ./.venv/bin/python -I -B dev/dense/prep_bacdev_clean.py [ПАКЕТ_REGPRECISE.pkl …]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
sys.path.insert(0, os.path.join(ROOT, 'work/tasks/T511'))
import stage1_tables as S1                                               # noqa: E402

packs = sys.argv[1:] or ['regprecise_q1.pkl', 'regprecise_h1.pkl']
H = json.load(open(os.path.join(PACKS, 'bacdev_hold.json')))
HOLD = set(H['hold']) | set(H.get('hold_variants', ()))
letters = 'ACGT'


def site_of(window, start, end):
    w = end - start
    lo = HALF - (w - 1) // 2
    return ''.join(letters[c] if c < 4 else 'N' for c in window[lo:lo + w])


train = []
P = pickle.load(open(os.path.join(PACKS, 'e2_data_bac.pkl'), 'rb'))
n_coord = 0
for i, q in enumerate(P['protein']):
    if q in HOLD or P['end'][i] - P['start'][i] > W:
        continue
    train.append(site_of(P['pos'][i], P['start'][i], P['end'][i])); n_coord += 1
closed = {'fold0', 'control', 'holdout'}
n_rp = {}
for name in packs:
    rp = pickle.load(open(os.path.join(PACKS, name), 'rb'))
    ok = {q for q in rp['proteins'] if not (rp['meta'][q]['related_parts'] & closed) and q not in HOLD
          and not (set(rp['meta'][q]['related_bacregdb']) & HOLD)}
    n_rp[name] = 0
    for i, q in enumerate(rp['protein']):
        if q in ok:
            train.append(site_of(rp['pos'][i], rp['start'][i], rp['end'][i])); n_rp[name] += 1
index = S1.kmer_index([s for s in train if 'N' not in s])
WP = pickle.load(open(os.path.join(PACKS, 'bacdev_win.pkl'), 'rb'))
out, n, n_clean, prot_all, prot_clean = {}, 0, 0, set(), set()
for it in WP['items']:
    flags = []
    for seg in it['site_segs']:
        if seg is None:
            flags.append(0); continue
        s = ''.join(letters[c] if c < 4 else 'N' for c in seg[CTX:len(seg) - CTX])
        flags.append(0 if S1.shares_kmer(s, index) else 1)
    out['%s|%s' % (it['protein'], it['genome'])] = flags
    n += len(flags); n_clean += sum(flags); prot_all.add(it['protein'])
    if any(flags):
        prot_clean.add(it['protein'])
json.dump({'packs': packs, 'coord_rows': n_coord, 'regprecise_rows': n_rp, 'sites': n, 'clean_sites': n_clean, 'mask': out},
          open(os.path.join(PACKS, 'bacdev_clean_sites.json'), 'w'))
print('обучающих строк: координатных %d, RegPrecise %s; сайтов оценки %d, чистых %d (белков %d из %d)' % (
    n_coord, n_rp, n, n_clean, len(prot_clean), len(prot_all)))
