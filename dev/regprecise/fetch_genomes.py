"""Записи геномов RegPrecise, которых нет на диске (круг улучшений, решение владельца 21): по названию генома со
страницы регулона — сборка NCBI (Datasets API v2), её репликоны — записи GenBank (`efetch`, как записи проекта:
`gbwithparts` и `fasta`). Только чтение открытых источников; наружу уходит название организма и номер записи.

Зачем: у трети регулонов RegPrecise локус-тег регулятора не найден в записях на диске (`place_sites.py`,
`regulon_tf_not_on_disk`), чаще всего потому, что генома нет вовсе. Берутся геномы, у которых не найден ни один
локус-тег регулятора. Берётся сборка GenBank (GCA): в её записях локус-теги исходные, как у RegPrecise.

Выбор сборки: отчёты по названию (таксон NCBI); годна сборка с ≤ MAXSEQ последовательностями; первая по порядку
(сначала сборки, в названии или штамме которых есть последнее слово названия RegPrecise; уровень «Complete Genome» выше
прочих, затем «current», затем меньшее число контигов). Проверка выбора —
не здесь, а ниже по пути: `index_loci.py` и `place_sites.py` ставят сайт только при точном совпадении последовательности
у названного гена, так что чужой штамм даёт «локус-тег не найден», а не ложное место.

Выход: {DIR}/ACC.gbff, ACC.fasta и ACC.*.json (адрес, время UTC, sha256, байты) в data/reference_cache/nuccore_regprecise;
{OUT}/genomes_fetch.tsv — по строке на геном: название, сборка, записи, итог.
Запуск: python3 -I -B dev/regprecise/fetch_genomes.py [--limit N] [--retry] [--out DERIVED_DIR] [--shard K/N]
`--shard K/N` — несколько потоков разом (каждый берёт свой N-й геном списка); NCBI без ключа допускает 3 запроса в секунду.
Докачка: готовые файлы пропускаются; геном с итогом в genomes_fetch.tsv повторно не спрашивается.
`--retry` — второй проход по геномам с итогом no_assembly / too_many_sequences: название ищется в таксономии NCBI
(`esearch db=taxonomy` знает прежние названия и синонимы), сборки спрашиваются по номеру таксона, предел числа
последовательностей — MAXSEQ_RETRY (черновые сборки); прежняя строка генома в таблице заменяется."""
import csv
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'data/derived/regprecise')
DIR = os.path.join(ROOT, 'data/reference_cache/nuccore_regprecise')
AGENT = 'prokabind-research-client'
PAUSE = 0.5
MAXSEQ = 12
LIMIT = int(sys.argv[sys.argv.index('--limit') + 1]) if '--limit' in sys.argv else None
RETRY = '--retry' in sys.argv
SHARD = [int(x) for x in sys.argv[sys.argv.index('--shard') + 1].split('/')] if '--shard' in sys.argv else None   # K/N: каждый N-й геном, начиная с K
MAXSEQ_RETRY = 60
os.makedirs(DIR, exist_ok=True)
TABLE = os.path.join(OUT, 'genomes_fetch.tsv')


def get(url, tries=6, timeout=120):
    wait = 5.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': AGENT, 'Accept': 'application/json, text/plain, */*'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            time.sleep(PAUSE)
            return body
        except urllib.error.HTTPError as err:
            if err.code in (400, 404):                     # такого таксона или записи нет — повторять незачем
                time.sleep(PAUSE)
                return None
            print('  повтор %d: %s — HTTP %d' % (attempt + 1, url[:110], err.code), flush=True)
        except Exception as err:                           # обрыв связи
            print('  повтор %d: %s — %s' % (attempt + 1, url[:110], str(err)[:100]), flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 120.0)
    # все попытки сорвались (сеть или отказ сервера): не выдавать это за «такой записи нет» — остановиться, геном
    # останется неразобранным и будет спрошен при следующем запуске
    raise SystemExit('нет ответа после %d попыток: %s' % (tries, url[:110]))


def fetch_record(acc, rettype, ext, must):
    path = os.path.join(DIR, acc + ext)
    if os.path.exists(path) and os.path.getsize(path) > 200:
        return True
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=%s&rettype=%s&retmode=text' % (acc, rettype)
    body = get(url, timeout=600)
    if body is None or must not in body[:20000] or (rettype == 'gbwithparts' and not body.rstrip().endswith(b'//')):
        return False
    tmp = path + '.tmp%d' % os.getpid()
    with open(tmp, 'wb') as f:
        f.write(body)
    os.replace(tmp, path)
    json.dump({'accession_version': acc, 'database': 'nuccore', 'request_url': url, 'response_bytes': len(body),
               'response_sha256': hashlib.sha256(body).hexdigest(), 'retrieved_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
              open(path + '.json', 'w'), indent=1)
    return True


# геномы без единого найденного локус-тега регулятора
tags = set()
with open(os.path.join(OUT, 'locus_index.tsv')) as f:
    f.readline()
    for line in f:
        tags.add(line.split('\t', 1)[0])
found, missing = {}, {}
for r in csv.DictReader(open(os.path.join(OUT, 'regulons.tsv')), delimiter='\t'):
    (found if r['tf_locus_tag'] in tags else missing).setdefault(r['genome'], []).append(r['tf_locus_tag'])
want = sorted(g for g in missing if g not in found)
done = {}
if os.path.exists(TABLE):
    for r in csv.DictReader(open(TABLE), delimiter='\t'):
        done[r['genome']] = r
else:
    with open(TABLE, 'w') as f:
        f.write('genome\tregulons\tassembly\torganism\tlevel\trecords\tstatus\n')
todo = [g for g in want if g not in done]
if RETRY:
    todo = [g for g in want if g in done and done[g]['status'].split(':')[0] in ('no_assembly', 'too_many_sequences', 'no_sequences')]
    MAXSEQ = MAXSEQ_RETRY


if SHARD:
    todo = todo[SHARD[0]::SHARD[1]]


def taxon_ids(name):
    """номера таксонов NCBI по названию (с прежними названиями и синонимами)"""
    out = []
    for term in (name, name.replace("'", ''), name.replace(' str. ', ' '), name.replace(' sp ', ' sp. ')):
        body = get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=taxonomy&retmode=json&term=%s' % urllib.parse.quote('"%s"' % term))
        try:
            ids = json.loads(body)['esearchresult']['idlist'] if body else []
        except Exception:
            ids = []
        out += [i for i in ids if i not in out]
        if out:
            break
    return out


print('геномов без записей на диске: %d; уже разобрано: %d; к разбору: %d' % (len(want), len(want) - len(todo), len(todo)), flush=True)
LEVEL = {'Complete Genome': 0, 'Chromosome': 1, 'Scaffold': 2, 'Contig': 3}
t0 = time.time()
for n, g in enumerate(todo[:LIMIT] if LIMIT else todo):
    row = {'genome': g, 'regulons': len(missing[g]), 'assembly': '', 'organism': '', 'level': '', 'records': '', 'status': ''}
    body = get('https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/%s/dataset_report?page_size=50' % urllib.parse.quote(g))
    reports = (json.loads(body).get('reports') or []) if body else []
    if RETRY and not reports:
        for tid in taxon_ids(g)[:2]:
            body = get('https://api.ncbi.nlm.nih.gov/datasets/v2/genome/taxon/%s/dataset_report?page_size=50' % tid)
            reports += (json.loads(body).get('reports') or []) if body else []
    cands = []
    for rep in reports:
        info, stats = rep.get('assembly_info', {}), rep.get('assembly_stats', {})
        nseq = int(stats.get('total_number_of_chromosomes') or 0) + int(stats.get('number_of_component_sequences') or 0) * 0
        ncontig = int(stats.get('number_of_contigs') or 0)
        acc = rep.get('accession', '')
        gca = acc if acc.startswith('GCA_') else rep.get('paired_accession', '')
        if not gca.startswith('GCA_'):
            continue
        org = rep.get('organism', {})
        names = (org.get('organism_name', '') + ' ' + ' '.join(str(v) for v in (org.get('infraspecific_names') or {}).values())).lower()
        strain = g.replace("'", '').split()[-1].lower()            # последнее слово названия RegPrecise — обычно штамм
        cands.append((0 if strain in names.replace("'", '') else 1, LEVEL.get(info.get('assembly_level'), 9),
                      0 if info.get('assembly_status') == 'current' else 1, ncontig, gca,
                      org.get('organism_name', ''), info.get('assembly_level', '')))
    if not cands:
        row['status'] = 'no_assembly'
    else:
        ok = False
        for _, level, _, _, gca, org, lname in sorted(cands)[:3]:
            body = get('https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/%s/sequence_reports' % gca)
            seqs = (json.loads(body).get('reports') or []) if body else []
            accs = [s.get('genbank_accession') for s in seqs if s.get('genbank_accession')]
            row.update(assembly=gca, organism=org, level=lname)
            if not accs or len(accs) > MAXSEQ:
                row['status'] = 'too_many_sequences:%d' % len(accs) if accs else 'no_sequences'
                continue
            good = [a for a in accs if fetch_record(a, 'gbwithparts', '.gbff', b'LOCUS') and fetch_record(a, 'fasta', '.fasta', b'>')]
            row['records'] = ';'.join(good)
            row['status'] = 'ok' if len(good) == len(accs) else 'partial:%d_of_%d' % (len(good), len(accs))
            ok = True
            break
    cols = ('genome', 'regulons', 'assembly', 'organism', 'level', 'records', 'status')
    if RETRY:                                              # строка генома заменяется
        done[g] = {k: str(row[k]) for k in cols}
        tmp = TABLE + '.tmp%d' % os.getpid()
        with open(tmp, 'w') as f:
            f.write('\t'.join(cols) + '\n')
            for r in done.values():
                f.write('\t'.join(str(r[k]) for k in cols) + '\n')
        os.replace(tmp, TABLE)
    else:
        with open(TABLE, 'a') as f:
            f.write('\t'.join(str(row[k]) for k in cols) + '\n')
    print('%3d из %d  %-50s %-16s %-28s %s  (%.0f мин)' % (n + 1, len(todo), g[:50], row['assembly'], row['status'], row['records'][:60], (time.time() - t0) / 60), flush=True)
print('готово, %.0f мин' % ((time.time() - t0) / 60), flush=True)
