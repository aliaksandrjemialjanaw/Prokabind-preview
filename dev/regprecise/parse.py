"""Разбор скачанных страниц регулонов RegPrecise 3.2 (`download.py`) в таблицы (круг улучшений, решение владельца 21).

Со страницы регулона берутся: вид регулятора (берутся только «Transcription factor»), локус-тег регулятора в этом
геноме и номер локуса MicrobesOnline, семейство, регулог, геном (из заголовка страницы), и по каждому оперону — сайты
(положение относительно начала первого гена оперона, оценка, последовательность) и локус-теги генов оперона.

Выход (вне git): {OUT}/regulons.tsv, {OUT}/sites.tsv, {OUT}/parse_report.json.
Запуск: python3 -I -B dev/regprecise/parse.py [IN_DIR] [OUT_DIR]"""
import collections
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/incoming/regprecise_3.2')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'data/derived/regprecise')
os.makedirs(OUT, exist_ok=True)
tag = re.compile(r'<[^>]+>')
clean = lambda s: html.unescape(re.sub(r'\s+', ' ', tag.sub(' ', s))).strip()


def prop(page, name):
    m = re.search(r'<td[^>]*>\s*%s\s*</td>\s*<td>(.*?)</td>' % re.escape(name), page, re.S)
    return (clean(m.group(1)), m.group(1)) if m else (None, '')


# таксономическая коллекция каждого регулона (страницы коллекций); девять коллекций — археи: проект — о бактериях,
# их регулоны в дорожку не идут (отметка domain, отсев — в place_sites.py)
ARCHAEA = {'Archaeoglobales', 'Halobacteriales', 'Methanobacteriales', 'Methanococcales', 'Methanomicrobiales',
           'Methanosarcinales', 'Sulfolobales', 'Thermococcales', 'Thermoproteales'}
cdir = os.path.join(IN, 'collections')
cnames = {}
for cid, cname in re.findall(r'collection_tax\.jsp\?collection_id=(\d+)[^>]*>([^<]+)<', open(os.path.join(cdir, 'index.html'), errors='replace').read()):
    cnames.setdefault(cid, cname.strip())
assert ARCHAEA <= set(cnames.values()), 'названия коллекций архей не найдены в списке коллекций'
collection_of = {}
for cid, cname in sorted(cnames.items(), key=lambda kv: int(kv[0])):
    path = os.path.join(cdir, 'tax_%s.html' % cid)
    if os.path.exists(path):
        for x in re.findall(r'regulon\.jsp\?regulon_id=(\d+)', open(path, errors='replace').read()):
            collection_of.setdefault(int(x), cname)

regulons, sites = [], []
report = collections.Counter()
for name in sorted(os.listdir(os.path.join(IN, 'regulons')), key=lambda x: int(x.split('.')[0])):
    rid = int(name.split('.')[0])
    page = open(os.path.join(IN, 'regulons', name), errors='replace').read()
    report['pages'] += 1
    rtype, _ = prop(page, 'Regulator type:')
    if rtype != 'Transcription factor':
        report['not_tf:%s' % rtype] += 1
        continue
    locus, raw = prop(page, 'TF locus tag:')
    mo = re.search(r'fetchLocus\.cgi\?locus=(\d+)', raw)
    family, _ = prop(page, 'Regulator family:')
    regulog, raw_rg = prop(page, 'Regulog:')
    rg = re.search(r'regulog_id=(\d+)', raw_rg)
    m = re.search(r'<h1>\s*Regulon of\s*<span class="titleItem">(.*?)</span>\s*in\s*<span class="titleItem">(.*?)</span>', page, re.S)
    genome = clean(m.group(2)) if m else ''
    if not locus or not genome:
        report['no_locus_or_genome'] += 1
        continue
    n = 0
    for block in re.findall(r'<div class="operon">(.*?)</div>\s*</td>', page, re.S):
        tips = re.findall(r'<span>(.*?)</span>', block, re.S)
        ss = [t for t in tips if 'Position:' in t]
        genes = [re.search(r'Locus tag:\s*([^<\s]+)', t).group(1) for t in tips if 'Locus tag:' in t and re.search(r'Locus tag:\s*([^<\s]+)', t)]
        for t in ss:
            pos = re.search(r'Position:\s*(-?\d+)', t); sc = re.search(r'Score:\s*([-\d.]+)', t); sq = re.search(r'Sequence:\s*([A-Za-z]+)', t)
            if not (pos and sq):
                report['site_unparsed'] += 1
                continue
            sites.append((rid, int(pos.group(1)), sc.group(1) if sc else '', sq.group(1).upper(), genes[0] if genes else '', ';'.join(genes)))
            n += 1
    coll = collection_of.get(rid, '')
    domain = 'archaea' if coll in ARCHAEA else 'bacteria' if coll else ''
    regulons.append((rid, rg.group(1) if rg else '', regulog or '', genome, locus, mo.group(1) if mo else '', family or '', n, coll, domain))
    report['tf_regulons'] += 1
    report['tf_regulons_%s' % (domain or 'no_collection')] += 1
with open(os.path.join(OUT, 'regulons.tsv'), 'w') as f:
    f.write('regulon_id\tregulog_id\tregulog\tgenome\ttf_locus_tag\tmo_locus\tfamily\tn_sites\tcollection\tdomain\n')
    for r in regulons:
        f.write('\t'.join(str(x) for x in r) + '\n')
with open(os.path.join(OUT, 'sites.tsv'), 'w') as f:
    f.write('regulon_id\tposition\tscore\tsequence\tfirst_gene\toperon_genes\n')
    for r in sites:
        f.write('\t'.join(str(x) for x in r) + '\n')
report.update({'sites': len(sites), 'genomes': len({r[3] for r in regulons}), 'regulogs': len({r[1] for r in regulons}),
               'families': len({r[6] for r in regulons}), 'regulons_without_sites': sum(1 for r in regulons if r[7] == 0)})
json.dump(dict(report), open(os.path.join(OUT, 'parse_report.json'), 'w'), indent=1)
print(json.dumps(dict(report), ensure_ascii=False))
