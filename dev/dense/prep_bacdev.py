"""Набор разработки из белков BacRegDB (решение владельца 21, п. 2; `work/PLAN.md`).

Кто годится: белок BacRegDB, которого нет в нашем разбиении и который не родственен (идентичность ≥ 40% при покрытии
≥ 80%) ни одному нашему белку (столбец `related` таблицы T511 пуст — он считан против всех наших белков, включая
отложенный набор), с местами в геномах (строки пакета `e2_data_bac.pkl`). Такой белок не проверочный ни в одном заходе
и не «прокси» ни одного проверочного белка (SPEC п. 3.10), поэтому расчёты на нём проверочных данных не раскрывают.

Половина годных белков (по хэшу группы вариантов, закреплённое семя) убирается из обучений разработки (`--hold-file`
у `train_dense.py`) и служит для оценки: AUROC мест белка в геноме против межгенного фона той же записи, как мера
остановки на контрольной части. Сайты этих белков — операторы BacRegDB (около половины найдены поиском по профилю), а
не опыты с очищенным белком: набор годится сравнивать рецепты; заглавные числа остаются за контрольной частью.

Выход в {PACKS}: bacdev_hold.json (кого убирать из обучения), bacdev_setup.pkl (пары «белок + запись генома»: сайты,
центры фона, запретные основания — формат `e0_setup.pkl`), bacdev_control.pkl (полосы родства с обучающей частью
разработки, группы родства, годные «чужие»).
Запуск: ./.venv/bin/python -I -B dev/dense/prep_bacdev.py [--share 0.5] [--seed 21] [--max-genomes 8] [--centres 500]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

argv = sys.argv[1:]
arg = lambda name, default, cast=float: cast(argv[argv.index(name) + 1]) if name in argv else default
SHARE, SEED = arg('--share', 0.5), arg('--seed', 21, int)
MAXG, NC = arg('--max-genomes', 8, int), arg('--centres', 500, int)
P = pickle.load(open(os.path.join(PACKS, 'e2_data_bac.pkl'), 'rb'))
lib = pickle.load(open(os.path.join(PACKS, 'library.pkl'), 'rb'))
split = {r['protein'] for r in read_tsv('data/derived/split.tsv')}
table = {r['protein']: r for r in read_tsv('data/derived/sigmoid/proteins.tsv')}
rows_of = collections.defaultdict(list)
for i, p in enumerate(P['protein']):
    rows_of[p].append(i)
cands = sorted(p for p in rows_of if p not in split and p in table and not table[p]['related'].strip())
vg = {p: table[p]['variant_group'] for p in cands}


def held_hash(key):
    return int(hashlib.sha256(('%d:bacdev:%s' % (SEED, key)).encode()).hexdigest()[:8], 16) / 16.0 ** 8 < SHARE


held = sorted(p for p in cands if held_hash(vg[p]))
held_set = set(held)
# вместе с белком уходят из обучения его варианты ≥ 90% (та же группа вариантов) — как у настоящего разбиения
held_groups = {vg[p] for p in held}
also = sorted(p for p in rows_of if p not in held_set and p in table and table[p]['variant_group'] in held_groups)
train = sorted(set(rows_of) - held_set - set(also))
json.dump({'seed': SEED, 'share': SHARE, 'candidates': cands, 'hold': held, 'hold_variants': also},
          open(os.path.join(PACKS, 'bacdev_hold.json'), 'w'), indent=0)
print('годных белков %d, отложено %d (строк пакета %d), с ними варианты %d; обучающих белков разработки %d (строк %d)'
      % (len(cands), len(held), sum(len(rows_of[p]) for p in held), len(also), len(train), sum(len(rows_of[p]) for p in train)))

# полосы: наибольшая идентичность с обучающим белком разработки (наши ему не родственны по условию отбора)
train_set = set(train)
band, best = {}, {}
for p in held:
    v = max([x for q, x in lib['rel'].get(p, {}).items() if q in train_set] + [0.0])
    best[p] = v
    band[p] = '>=70' if v >= 0.7 else '40-70' if v >= 0.4 else '<40'
# группы родства среди отложенных (связные компоненты ≥ 40%)
parent = {p: p for p in held}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for p in held:
    for q in lib['rel'].get(p, {}):
        if q in held_set:
            parent[find(p)] = find(q)
groups = {p: find(p) for p in held}
fam = {p: lib['meta'][p]['tf_hmm'] for p in held}
allowed = {p: sorted(q for q in held if q != p and groups[q] != groups[p] and fam[q] != fam[p] and fam[q] and fam[p]) for p in held}
print('полосы:', dict(collections.Counter(band.values())), '| групп родства', len(set(groups.values())), '| семейств', len(set(fam.values())))
pickle.dump({'band': band, 'groups': groups, 'allowed': allowed, 'identity': best, 'family': fam},
            open(os.path.join(PACKS, 'bacdev_control.pkl'), 'wb'), protocol=4)

cfg = json.load(open(CFG_BAC))
dirs = D.genome_dirs(cfg)
marks_by_genome = collections.defaultdict(list)
for (q, g), iv in P['marks'].items():
    marks_by_genome[g].extend(iv)
items, skipped = [], collections.Counter()
t0 = time.time()
for n, p in enumerate(held):
    by_g = collections.defaultdict(list)
    for i in rows_of[p]:
        by_g[P['genome'][i]].append((int(P['start'][i]), int(P['end'][i])))
    order = sorted(by_g, key=lambda g: hashlib.sha256(('%s:%s' % (p, g)).encode()).hexdigest())
    took = 0
    for g in order:
        if took >= MAXG:
            break
        try:
            seq = D.genome_sequence(dirs, g)
            genes = D.gene_mask(dirs, g, len(seq))
            circ = D.topology(dirs, g) == 'circular'
        except Exception as err:
            skipped['no_record'] += 1
            continue
        L = len(seq)
        blocked = np.zeros(L, dtype=bool)
        for s0, e0 in marks_by_genome.get(g, ()):
            blocked[max(s0, 0):min(e0, L)] = True
        inter = np.flatnonzero(~genes)
        inter = inter[(inter > 200) & (inter < L - 200)] if not circ else inter
        if inter.size < 300 or L < 5000:
            skipped['short_or_no_intergenic'] += 1
            continue
        seedv = int(hashlib.sha256(('%s:%s' % (p, g)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng([SPLIT_NUMBER, seedv])
        centres = np.sort(rng.choice(inter, size=min(NC, inter.size), replace=False))
        items.append({'protein': p, 'genome': g, 'length': L, 'circular': circ, 'sites': sorted(by_g[g]), 'centres': centres,
                      'blocked_idx': np.flatnonzero(blocked).astype(np.int64), 'n_intergenic': int(inter.size)})
        took += 1
    if n % 20 == 0:
        print('  белков %d из %d, пар %d, %.0f с' % (n, len(held), len(items), time.time() - t0), flush=True)
pickle.dump({'items': items, 'training': train, 'hold': held}, open(os.path.join(PACKS, 'bacdev_setup.pkl'), 'wb'), protocol=4)
have = {it['protein'] for it in items}
print('пар «белок + запись генома»: %d у %d белков; сайтов %d; пропущено: %s; полосы белков с парами: %s' % (
    len(items), len(have), sum(len(it['sites']) for it in items), dict(skipped), dict(collections.Counter(band[p] for p in have))))
