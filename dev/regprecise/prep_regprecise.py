"""Дорожка RegPrecise для плотного обучения: сайты на своих местах у своих белков (круг улучшений, решение владельца 21).

Вход — `place_sites.py`: места сайтов в записях геномов и белки-регуляторы этих геномов. Здесь:
  1. повторы мест сняты; правило утечки п. 2.7.2: место, чья последовательность делит ≥ 12 оснований подряд с любым
     нашим точным сайтом, сайтом PRODORIC или исправленной записью T511, снимается (`kmer_index`, `shares_kmer` T511);
  2. правило утечки п. 2.7.3: родство регулятора с нашими белками (команда CONTRACTS, в обе стороны; `relatives` T511) —
     записываются наши белки ≥ 40% и их части разбиения; кого к какой модели допускать, решает обучение;
  3. родство регуляторов между собой и с белками BacRegDB (≥ 40%) — для запрета пар «чужой» и для белков разработки;
  4. пакет строк в формате координатного пакета (`CoordSet` в `train_dense.py`): окно сайта, восемь неразмеченных окон
     той же записи близкого состава вне меток (места RegPrecise, наши и BacRegDB метки пакета `e2_data_bac.pkl`),
     годные «чужие» — регуляторы другого семейства RegPrecise без родства ≥ 40%.
Это только обучение. Отложенный набор не читается: родство с его белками считает `relatives` по `proteins.fasta`, как
у сборок T511 и библиотеки.

Выход: {DERIVED}/proteins.tsv, ladder.json; {PACKS}/regprecise.pkl; {NIGHT}/regprecise_queue.tsv — очередь признаков ESM-2.
Запуск: ./.venv/bin/python -I -B dev/regprecise/prep_regprecise.py [ИМЯ_ПАКЕТА [DERIVED_DIR]]
(по умолчанию regprecise.pkl и data/derived/regprecise)"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'dense'))
from common import *
sys.path.insert(0, os.path.join(ROOT, 'work/tasks/T511'))
import t511lib as L                                                      # noqa: E402
import stage1_tables as S1                                               # noqa: E402

DER = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'data/derived/regprecise')
t0 = time.time()
ladder = collections.OrderedDict()
places = read_tsv(os.path.join(DER, 'places.tsv'))
ladder['places_read'] = len(places)
seen, uniq = set(), []
for r in places:
    key = (r['protein'], r['genome'], r['start'], r['end'])
    if key not in seen:
        seen.add(key); uniq.append(r)
ladder['places_unique'] = len(uniq)
seqs = {}
name = None
for line in open(os.path.join(DER, 'regulators.fasta')):
    if line.startswith('>'):
        name = line[1:].strip()
    else:
        seqs[name] = line.strip()
drop = collections.Counter()
ok = []
for r in uniq:
    s = seqs.get(r['protein'])
    if not s or set(s) - S1.AMINO or len(s) > L.ESM_MAX:
        drop['regulator_sequence'] += 1; continue
    if not (8 <= len(r['seq']) <= 60):
        drop['site_length'] += 1; continue
    ok.append(r)
our_rows = [r for r in L.read_tsv(L.R / 'data/derived/sites.tsv') if r['lane'] in ('exact', 'prodoric')]
corr = L.read_tsv(L.R / 'data/derived/sigmoid/corrections.tsv')
index = S1.kmer_index([r['seq'].strip().upper() for r in our_rows] + [r['new_seq'] for r in corr])
kept = [r for r in ok if not S1.shares_kmer(r['seq'], index)]
ladder['places_dropped_12bp'] = len(ok) - len(kept)
for k, v in drop.items():
    ladder['places_dropped_' + k] = v
ladder['places_kept'] = len(kept)
used = sorted({r['protein'] for r in kept})
ladder['regulators'] = len(used)
print('мест %d -> без повторов %d -> после правила 12 оснований %d; регуляторов %d; %.0f с' % (
    len(places), len(uniq), len(kept), len(used), time.time() - t0), flush=True)

ours = L.read_fasta(L.R / 'data/derived/proteins.fasta')
part = {r['protein']: r for r in L.read_tsv(L.R / 'data/derived/split.tsv')}
rel, group = S1.relatives(seqs, used, ours)
print('родство с нашими белками посчитано, %.0f с' % (time.time() - t0), flush=True)
bac = {}
name = None
for line in open(os.path.join(ROOT, 'data/derived/sigmoid/proteins.fasta')):
    if line.startswith('>'):
        name = line[1:].strip().split()[0]
    else:
        bac[name] = bac.get(name, '') + line.strip()
rel_self, rel_bac = collections.defaultdict(dict), collections.defaultdict(dict)
with tempfile.TemporaryDirectory(dir='/var/tmp') as tmp:
    mine, bfa = os.path.join(tmp, 'reg.fasta'), os.path.join(tmp, 'bac.fasta')
    with open(mine, 'w') as f:
        for k in used:
            f.write('>%s\n%s\n' % (k, seqs[k]))
    with open(bfa, 'w') as f:
        for k in sorted(bac):
            f.write('>%s\n%s\n' % (k, bac[k]))
    for q, t, fid, cov in L.mmseqs_rows(mine, mine, os.path.join(tmp, 'self')):
        if q != t and fid >= L.IDENT_RELATED and cov >= L.COV_RELATED:
            rel_self[q][t] = max(fid, rel_self[q].get(t, 0.0)); rel_self[t][q] = max(fid, rel_self[t].get(q, 0.0))
    for a_, b_, flip in ((mine, bfa, False), (bfa, mine, True)):
        for q, t, fid, cov in L.mmseqs_rows(a_, b_, os.path.join(tmp, 'bac%d' % flip)):
            if fid >= L.IDENT_RELATED and cov >= L.COV_RELATED:
                x, y = (t, q) if flip else (q, t)
                rel_bac[x][y] = max(fid, rel_bac[x].get(y, 0.0))
for k in used:
    if k in bac:
        rel_bac[k][k] = 1.0
print('родство регуляторов между собой и с BacRegDB посчитано, %.0f с' % (time.time() - t0), flush=True)
family = {}
for r in kept:
    family.setdefault(r['protein'], collections.Counter())[r['family']] += 1
family = {k: v.most_common(1)[0][0] for k, v in family.items()}
n_places = collections.Counter(r['protein'] for r in kept)
rows = []
for k in used:
    rl = rel.get(k, {})
    rows.append({'protein': k, 'length': len(seqs[k]), 'identical': 'yes' if k in ours else 'no',
                 'related': ';'.join('%s:%.3f' % (t, rl[t]) for t in sorted(rl)),
                 'related_parts': ';'.join(sorted({part[t]['part'] for t in rl})), 'variant_group': group[k],
                 'family': family[k], 'n_places': n_places[k], 'related_bacregdb': ';'.join('%s:%.3f' % (t, v) for t, v in sorted(rel_bac.get(k, {}).items()))})
with open(os.path.join(DER, 'proteins.tsv'), 'w') as f:
    cols = ['protein', 'length', 'identical', 'related', 'related_parts', 'variant_group', 'family', 'n_places', 'related_bacregdb']
    f.write('\t'.join(cols) + '\n')
    for r in rows:
        f.write('\t'.join(str(r[c]) for c in cols) + '\n')
closed = {'fold0', 'control', 'holdout'}
adm0 = [r for r in rows if not (set(x for x in r['related_parts'].split(';') if x) & closed)]
ladder['regulators_identical_to_ours'] = sum(1 for r in rows if r['identical'] == 'yes')
ladder['regulators_related_to_ours'] = sum(1 for r in rows if r['related'])
ladder['regulators_admitted_fold0_model'] = len(adm0)
ladder['places_admitted_fold0_model'] = sum(r['n_places'] for r in adm0)
ladder['variant_groups'] = len(set(group.values()))
ladder['families'] = len(set(family.values()))

# ---------------------------------------------------------------- пакет строк
Pb = pickle.load(open(os.path.join(PACKS, 'e2_data_bac.pkl'), 'rb'))
marks_by_genome = collections.defaultdict(list)
for (q, g), iv in Pb['marks'].items():
    marks_by_genome[g].extend(iv)
for r in kept:
    marks_by_genome[r['genome']].append((int(r['start']), int(r['end'])))
dirs = D.genome_dirs(json.load(open(CFG_BAC))) + ['data/reference_cache/nuccore_regprecise']      # третья папка — геномы fetch_genomes.py
by_genome = collections.defaultdict(list)
for i, r in enumerate(kept):
    by_genome[r['genome']].append(i)
NEG = 8
out = collections.defaultdict(list)
pos_w, neg_w = [], []
skipped = collections.Counter()
own_marks = collections.defaultdict(list)
for gi, g in enumerate(sorted(by_genome)):
    try:
        seq = D.genome_sequence(dirs, g).upper()
        genes = D.gene_mask(dirs, g, len(seq))
    except Exception:
        skipped['record_unreadable'] += len(by_genome[g]); continue
    Lg = len(seq)
    code = codes(seq)
    at_c = np.concatenate(([0], np.cumsum((code == 0) | (code == 3))))
    n_c = np.concatenate(([0], np.cumsum(code == 4)))
    blocked = np.zeros(Lg, dtype=bool)
    for s0, e0 in marks_by_genome[g]:
        blocked[max(0, s0 - 100):min(Lg, e0 + 100)] = True
    b_c = np.concatenate(([0], np.cumsum(blocked)))
    rng = np.random.default_rng([SPLIT_NUMBER, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)])
    inter = np.flatnonzero(~genes)
    inter = inter[(inter >= HALF) & (inter < Lg - HALF)]
    cand = inter if inter.size <= 40000 else np.sort(rng.choice(inter, size=40000, replace=False))
    if cand.size:
        okc = ((n_c[cand + HALF + 1] - n_c[cand - HALF]) == 0) & ((b_c[cand + HALF + 1] - b_c[cand - HALF]) == 0)
        cand = cand[okc]
    cand_at = (at_c[cand + HALF + 1] - at_c[cand - HALF]) / W if cand.size else np.zeros(0)
    order = np.argsort(cand_at, kind='stable'); cand, cand_at = cand[order], cand_at[order]
    for i in by_genome[g]:
        r = kept[i]
        s0, e0 = int(r['start']), int(r['end'])
        c = s0 + (e0 - s0 - 1) // 2
        if c - HALF < 0 or c + HALF + 1 > Lg or n_c[c + HALF + 1] - n_c[c - HALF]:
            skipped['edge_or_unknown_base'] += 1; continue
        if e0 - s0 < L.KMER and S1.shares_kmer(seq[max(0, e0 - L.KMER):s0 + L.KMER], index):
            # сайт короче 12 оснований правило п. 2.7.2 по самому сайту проверить не может: проверяются все отрезки по
            # 12 оснований записи, целиком накрывающие сайт (ревизия круга 10.11)
            skipped['short_site_shares_12bp_with_context'] += 1; continue
        at = (at_c[c + HALF + 1] - at_c[c - HALF]) / W
        lo_i, hi_i = np.searchsorted(cand_at, at - 0.03, 'left'), np.searchsorted(cand_at, at + 0.03, 'right')
        if hi_i - lo_i < NEG:
            lo_i, hi_i = np.searchsorted(cand_at, at - 0.08, 'left'), np.searchsorted(cand_at, at + 0.08, 'right')
        if hi_i - lo_i < 1:
            skipped['no_unlabeled_window'] += 1; continue
        pick = cand[rng.integers(lo_i, hi_i, size=NEG)]
        pos_w.append(code[c - HALF:c + HALF + 1])
        neg_w.append(np.stack([code[x - HALF:x + HALF + 1] for x in pick]))
        lo, hi = span_of(e0 - s0)
        for k, v in (('protein', r['protein']), ('genome', g), ('start', s0), ('end', e0), ('lane', 'regprecise'), ('at', float(at)),
                     ('lo', lo), ('hi', hi), ('n_neg', NEG), ('weight', 1.0), ('row', len(out['protein']))):
            out[k].append(v)
        own_marks[(r['protein'], g)].append((s0, e0))
    if gi % 50 == 0:
        print('  записей %d из %d, строк %d, %.0f с' % (gi, len(by_genome), len(out['protein']), time.time() - t0), flush=True)
proteins = sorted(set(out['protein']))
pset = set(proteins)
allowed = {}
fam_members = collections.defaultdict(list)
for p in proteins:
    fam_members[family[p]].append(p)
undefined = {'', '[Other]', 'Other', 'None'}
for p in proteins:
    if family[p] in undefined:
        allowed[p] = []
        continue
    relp = set(rel_self.get(p, {}))
    allowed[p] = sorted(q for f_, ms in fam_members.items() if f_ != family[p] and f_ not in undefined for q in ms if q not in relp)
pack = {k: v for k, v in out.items()}
pack.update({'pos': np.stack(pos_w), 'neg': np.stack(neg_w), 'proteins': proteins, 'allowed': allowed,
             'marks': {k: D.merge_intervals(v) for k, v in own_marks.items()},
             'meta': {r['protein']: {'related_parts': set(x for x in r['related_parts'].split(';') if x), 'related': {x.split(':')[0]: float(x.split(':')[1]) for x in r['related'].split(';') if x},
                                     'related_bacregdb': {x.split(':')[0]: float(x.split(':')[1]) for x in r['related_bacregdb'].split(';') if x},
                                     'variant_group': r['variant_group'], 'family': r['family'], 'identical': r['identical'] == 'yes'} for r in rows if r['protein'] in pset},
             'rel': {p: dict(v) for p, v in rel_self.items() if p in pset}})
path = os.path.join(PACKS, sys.argv[1] if len(sys.argv) > 1 else 'regprecise.pkl')
pickle.dump(pack, open(path, 'wb'), protocol=4)
with open(os.path.join(NIGHT, 'regprecise_queue.tsv'), 'w') as f:
    for p in proteins:
        f.write('%s\t%s\n' % (p, seqs[p]))
ladder['pack_rows'] = len(out['protein']); ladder['pack_proteins'] = len(proteins); ladder['pack_records'] = len(set(out['genome']))
for k, v in skipped.items():
    ladder['pack_skipped_' + k] = v
ladder['proteins_without_foreign'] = sum(1 for p in proteins if not allowed[p])
ladder['seconds'] = round(time.time() - t0, 1)
json.dump(ladder, open(os.path.join(DER, 'ladder.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(ladder, indent=1, ensure_ascii=False))
print('пакет %s, %.1f МБ' % (path, os.path.getsize(path) / 1e6))
