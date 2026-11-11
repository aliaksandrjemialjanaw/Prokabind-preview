"""Сайты RegPrecise — на свои места в геномах и к своим белкам (круг улучшений, решение владельца 21).

Вход: таблицы `parse.py` (регулоны: локус-тег регулятора, геном; сайты: первый ген оперона, положение, последовательность)
и указатель локус-тегов записей геномов на диске (`index_loci.py`). Для каждого регулона выбирается запись генома, где
нашлось больше всего его локус-тегов; сайт ищется точным совпадением последовательности (обе цепи) перед первым геном
оперона (от −600 до +150 оснований относительно начала гена, с учётом цепи гена) и ставится, только если место одно.
Белок сайта — белок регулятора ЭТОГО генома (его последовательность из аннотации), а не белок-образец профиля.

Выход (вне git): {OUT}/places.tsv (protein — sha256 последовательности регулятора, как ключ белка в проекте; genome —
запись; start, end с нуля, конец не включён; strand; seq; regulon_id, regulog_id, family, tf_locus_tag, genome_name),
{OUT}/regulators.fasta, {OUT}/place_report.json.
Запуск: python3 -I -B dev/regprecise/place_sites.py [OUT_DIR]"""
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from prokabind import data as D                                          # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/derived/regprecise')
DIRS = [os.path.join(ROOT, 'data/reference_cache/nuccore'), os.path.join(ROOT, 'data/reference_cache/nuccore_bacregdb'),
        os.path.join(ROOT, 'data/reference_cache/nuccore_regprecise')]                    # третья — геномы fetch_genomes.py
RC = str.maketrans('ACGT', 'TGCA')
# SPEC п. 2.4 к регуляторам RegPrecise, как к белкам любого источника (п. 2.7.4): сигма-факторы (семейство RegPrecise
# «Sigma…» и имена регулогов Rpo*, Sig*, FliA, HrpL, WhiG) и белки, садящиеся на ДНК только с другим белком (список
# `PARTNER_ONLY` из work/tasks/T511/stage1_tables.py)
SIGMA = re.compile(r'^(Rpo[A-Z]|Sig[A-Z0-9]|FliA|HrpL|WhiG)')
PARTNER_ONLY = {'DinJ', 'YafN', 'AtaR', 'ParD', 'Kis', 'RcsA'}
t0 = time.time()
regulons = list(csv.DictReader(open(os.path.join(OUT, 'regulons.tsv')), delimiter='\t'))
sites = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(OUT, 'sites.tsv')), delimiter='\t'):
    sites[r['regulon_id']].append(r)


def variants(tag):
    """написания локус-тега: как на странице; с подчёркиванием между буквами и цифрами и без него (RegPrecise пишет
    SO0817 и PP4489, записи — SO_0817 и PP_4489). Список «A , B» (регулятор из двух генов) не делится: такой регулон
    не ставится вовсе — см. ниже"""
    out = []
    t = tag.strip()
    for v in (t, re.sub(r'^([A-Za-z]+)(\d+)$', r'\1_\2', t), t.replace('_', '', 1) if re.match(r'^[A-Za-z]+_\d+$', t) else t):
        if v and v not in out:
            out.append(v)
    return out


def find(tag):
    for v in variants(tag):
        if v in index:
            return index[v]
    return []


need = {v for r in regulons for v in variants(r['tf_locus_tag'])} | {v for ss in sites.values() for s in ss if s['first_gene'] for v in variants(s['first_gene'])}
index = collections.defaultdict(list)
for iname in ('locus_index.tsv', 'locus_index_extra.tsv'):            # второй — геномы fetch_genomes.py (index_loci.py extra)
    if not os.path.exists(os.path.join(OUT, iname)):
        continue
    with open(os.path.join(OUT, iname)) as f:
        next(f)
        for line in f:
            tag = line.split('\t', 1)[0]
            if tag in need:
                t = line.rstrip('\n').split('\t')
                index[tag].append({'accession': t[2], 'start': int(t[3]), 'end': int(t[4]), 'strand': t[5], 'protein': t[6]})
print('регулонов %d, локус-тегов нужно %d, найдено в записях на диске %d, %.0f с' % (len(regulons), len(need), len(index), time.time() - t0), flush=True)
# Канонная запись нашего генома — самая свежая версия (CONTRACTS § 1); прежние версии (NC_000913.2 рядом с NC_000913.3) и
# записи T513, помеченные копией нашей (`same_as` таблицы genomes_bacregdb.tsv), местом сайта не становятся, пока тот же
# ген есть в канонной записи: метки проекта стоят на ней, и неразмеченные окна обходят их только там (ревизия круга
# 10.11, находка Н3). Запись-близнец среди геномов fetch_genomes.py (та же последовательность, что у нашей канонной)
# узнаётся по длине и sha256 последовательности.
_files0 = set(x[:-6] for x in os.listdir(DIRS[0]) if x.endswith('.fasta'))
_latest = {}
for _a in _files0:
    _b, _v = (_a.rsplit('.', 1) + ['0'])[:2] if '.' in _a else (_a, '0')
    if _b not in _latest or int(_v) > int(_latest[_b][1]):
        _latest[_b] = (_a, _v)
ours = {a for a, _ in _latest.values()}
copy_of = {}
for _r in csv.DictReader(open(os.path.join(ROOT, 'data/derived/sigmoid/genomes_bacregdb.tsv')), delimiter='\t'):
    _t = [x for x in _r['same_as'].split(';') if x in ours]
    if _t:
        copy_of[_r['genome']] = _t[0]
for _a in _files0 - ours:
    copy_of[_a] = _latest[_a.rsplit('.', 1)[0]][0]            # прежняя версия: канонная — последняя (места сверяются ниже)


def seq_key(folder, acc):
    h, n = hashlib.sha256(), 0
    with open(os.path.join(folder, acc + '.fasta')) as f:
        for line in f:
            if not line.startswith('>'):
                t = line.strip().upper().encode()
                h.update(t); n += len(t)
    return n, h.hexdigest()


if os.path.isdir(DIRS[2]):
    _extra = sorted(x[:-6] for x in os.listdir(DIRS[2]) if x.endswith('.fasta'))
    _cache = os.path.join(OUT, 'sequence_keys.json')
    _keys = json.load(open(_cache)) if os.path.exists(_cache) else {}
    for _folder, _names in ((DIRS[0], sorted(ours)), (DIRS[2], _extra)):
        for _a in _names:
            if _a not in _keys:
                _keys[_a] = list(seq_key(_folder, _a))
    json.dump(_keys, open(_cache, 'w'))
    _by_key = {tuple(_keys[a]): a for a in sorted(ours)}
    for _a in _extra:
        if tuple(_keys[_a]) in _by_key and _a not in ours:
            copy_of[_a] = _by_key[tuple(_keys[_a])]
print('канонных наших записей %d; копий и прежних версий %d' % (len(ours), len(copy_of)), flush=True)
rank = lambda a: 0 if a in ours else 2 if a in copy_of else 1


def canon(acc, start, end):
    """(запись, начало): место на копии переносится в канонную запись, если там в тех же координатах та же последовательность"""
    t = copy_of.get(acc)
    if t is None:
        return acc, start
    try:
        if genome(t)[start:end] == genome(acc)[start:end]:
            return t, start
    except Exception:
        pass
    report['sites_left_on_copy_record'] += 1
    return acc, start
_seq = collections.OrderedDict()


def genome(acc):
    if acc not in _seq:
        if len(_seq) > 12:
            _seq.popitem(last=False)
        _seq[acc] = D.genome_sequence(DIRS, acc).upper()
    return _seq[acc]


report = collections.Counter()
rows, prot_need = [], {}
offsets = collections.Counter()
for r in sorted(regulons, key=lambda x: (x['genome'], int(x['regulon_id']))):
    report['regulons'] += 1
    if r.get('domain') != 'bacteria':                 # археи и регулоны без коллекции в дорожку не идут (проект — о бактериях)
        report['regulon_not_bacteria'] += 1
        continue
    name = r['regulog'].split(' - ')[0].strip()
    if 'sigma' in r['family'].lower() or SIGMA.match(name):       # SPEC п. 2.4.4: сигма-факторы не берутся
        report['regulon_sigma_factor'] += 1
        continue
    if name in PARTNER_ONLY:                                      # SPEC п. 2.4.2: белок садится только с партнёром
        report['regulon_partner_only'] += 1
        continue
    if ',' in r['tf_locus_tag']:
        # регулятор из двух генов: сайт нельзя приписать одному белку, а правило п. 2.7.3 проверяло бы родство только у
        # него (ревизия круга 10.11) — такие регулоны не ставятся
        report['regulon_two_regulator_genes'] += 1
        continue
    tf = find(r['tf_locus_tag'])
    if not tf:
        report['regulon_tf_not_on_disk'] += 1
        continue
    # запись генома: где больше локус-тегов этого регулона; при равенстве — наша запись, затем по имени
    votes = collections.Counter()
    for s in sites[r['regulon_id']]:
        for g in find(s['first_gene']):
            votes[g['accession']] += 1
    for g in tf:
        votes[g['accession']] += 1
    cands = {g['accession'] for g in tf}
    acc = sorted(cands, key=lambda a: (-votes[a], rank(a), a))[0]
    tf_rec = [g for g in tf if g['accession'] == acc][0]
    try:
        seq = genome(acc)
    except Exception:
        report['regulon_genome_unreadable'] += 1
        continue
    placed_here = 0
    for s in sites[r['regulon_id']]:
        report['sites'] += 1
        found = find(s['first_gene'])
        gene = [g for g in found if g['accession'] == acc]
        other = False
        if not gene and found:
            # ген — в другой записи (другой репликон того же генома: плазмида, вторая хромосома): берётся запись, где ген
            # найден, — наша запись прежде прочих, затем по имени; место всё равно ставится только точным совпадением
            gene = sorted(found, key=lambda x: (rank(x['accession']), x['accession']))[:1]
            other = True
        if not gene:
            report['site_gene_not_in_record'] += 1
            continue
        g = gene[0]
        site = s['sequence'].upper()
        if set(site) - set('ACGT') or len(site) < 8:
            report['site_bad_letters_or_short'] += 1
            continue
        try:
            gseq = seq if not other else genome(g['accession'])
        except Exception:
            report['site_gene_not_in_record'] += 1
            continue
        if g['strand'] == '+':
            lo, hi = max(0, g['start'] - 600), min(len(gseq), g['start'] + 150)
        else:
            lo, hi = max(0, g['end'] - 150), min(len(gseq), g['end'] + 600)
        region = gseq[lo:hi]
        hits = []
        for strand, q in (('+', site), ('-', site.translate(RC)[::-1])):
            at = region.find(q)
            while at >= 0:
                hits.append((lo + at, strand))
                at = region.find(q, at + 1)
        hits = sorted(set(hits))
        if len({h[0] for h in hits}) != 1:
            report['site_not_found' if not hits else 'site_ambiguous'] += 1
            continue
        start = hits[0][0]
        if other:
            report['sites_placed_in_other_record'] += 1
        # положение RegPrecise — от начала гена до сайта; запомним, какой конец сайта оно называет (для отчёта)
        rel = (start - g['start']) if g['strand'] == '+' else (g['end'] - (start + len(site)))
        offsets[rel - int(s['position'])] += 1
        place_acc, start = canon(g['accession'], start, start + len(site))
        rows.append((tf_rec['protein'], place_acc, start, start + len(site), hits[0][1], site, r['regulon_id'], r['regulog_id'], r['family'],
                     r['tf_locus_tag'], r['genome'], s['position'], s['score']))
        placed_here += 1
        report['sites_placed'] += 1
    if placed_here:
        report['regulons_with_places'] += 1
        prot_need[tf_rec['protein']] = None
for fname in ('locus_proteins.fasta', 'locus_proteins_extra.fasta'):
    if not os.path.exists(os.path.join(OUT, fname)):
        continue
    with open(os.path.join(OUT, fname)) as f:
        name = None
        for line in f:
            if line.startswith('>'):
                name = line[1:].strip()
            elif name in prot_need:
                prot_need[name] = line.strip()
with open(os.path.join(OUT, 'places.tsv'), 'w') as f:
    f.write('protein\tgenome\tstart\tend\tstrand\tseq\tregulon_id\tregulog_id\tfamily\ttf_locus_tag\tgenome_name\tposition\tscore\n')
    for row in rows:
        f.write('\t'.join(str(x) for x in row) + '\n')
with open(os.path.join(OUT, 'regulators.fasta'), 'w') as f:
    for k, v in sorted(prot_need.items()):
        if v:
            assert hashlib.sha256(v.encode()).hexdigest() == k
            f.write('>%s\n%s\n' % (k, v))
report = dict(report)
report.update({'regulator_proteins': len(prot_need), 'genome_records_used': len({r[1] for r in rows}),
               'offset_site_start_minus_position_top': {str(k): v for k, v in offsets.most_common(5)}})
json.dump(report, open(os.path.join(OUT, 'place_report.json'), 'w'), indent=1)
print(json.dumps(report, ensure_ascii=False))
print('%.0f с' % (time.time() - t0))
