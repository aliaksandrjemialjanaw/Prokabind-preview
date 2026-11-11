"""Откуда операторы BacRegDB в обучающем пакете (круг улучшений, решение владельца 21: «сравнить операторы из поиска по
профилю с опытными»).

У каждого сайта кураторского профиля SigmoID есть заголовок. Три вида заголовков:
  search     — выход поиска по профилю в геноме («…_site_1 offset= 379 RC»);
  regprecise — запись регулона RegPrecise («DVU2247(ahpC) Score=5.4 Pos=-117 [геном]») — вывод сравнительной геномики;
  named      — всё остальное: сайт назван по гену, часто с опытом и ссылкой («gdhA … DNAse footprinting [PMID: …]»).
Пара «белок-образец + сайт» получает множество видов по всем профилям, где она есть. Строка пакета `e2_data_bac.pkl`
(дорожка bacregdb) сопоставляется паре по белку и канонической записи сайта (сайт или его обратный комплемент).
Выход: {PACKS}/bacregdb_origin.json — номера строк пакета по классам: только поиск; только RegPrecise; есть «названный».
Запуск: ./.venv/bin/python -I -B dev/bacregdb_origin.py"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dense'))
from common import *

src = open(os.path.join(ROOT, 'work/scripts/extract.py')).read()
ns = {}
exec(src[:src.index('OUT = sys.argv[1]')], ns)                     # только чтение контейнера .sig (`parse_vfs`)
parse_vfs = ns['parse_vfs']
RC = str.maketrans('ACGT', 'TGCA')
canon = lambda s: min(s, s.translate(RC)[::-1])


def kind(head):
    if 'offset=' in head:
        return 'search'
    if re.search(r'Score=\S+\s+Pos=', head) or 'VIMSSId=' in head:      # второй вид заголовка — ревизия круга 10.11, Н5
        return 'regprecise'
    return 'named'


pairs = collections.defaultdict(set)
files = bad = 0
for line in open(os.path.join(ROOT, 'data/derived/sigmoid_extract/raw.jsonl')):
    r = json.loads(line)
    if r.get('error') or not r.get('cur') or not r.get('seed_seq'):
        continue
    try:
        fs = parse_vfs(r['path'])
    except Exception:
        bad += 1
        continue
    fa = None
    for name, blob in fs.items():
        if name.rsplit('.', 1)[-1].lower() in ('fasta', 'fa'):
            fa = blob.decode('utf-8', 'replace')
    if fa is None:
        continue
    files += 1
    protein = hashlib.sha256(r['seed_seq'].encode()).hexdigest()
    head, seq = None, []
    for ln in fa.splitlines() + ['>']:
        if ln.startswith('>'):
            if head is not None and seq:
                s = ''.join(seq).upper()
                if not set(s) - set('ACGT'):
                    pairs[(protein, canon(s))].add(kind(head))
            head, seq = ln[1:].strip(), []
        elif ln.strip():
            seq.append(ln.strip())
print('кураторских файлов прочитано %d (не прочитано %d); пар «белок + сайт» %d: %s' % (
    files, bad, len(pairs), dict(collections.Counter('+'.join(sorted(v)) for v in pairs.values()))))

P = pickle.load(open(os.path.join(PACKS, 'e2_data_bac.pkl'), 'rb'))
letters = 'ACGT'
out = collections.defaultdict(list)
for i, (p, lane) in enumerate(zip(P['protein'], P['lane'])):
    if lane != 'bacregdb':
        continue
    w = P['end'][i] - P['start'][i]
    lo = HALF - (w - 1) // 2
    s = ''.join(letters[c] if c < 4 else 'N' for c in P['pos'][i][lo:lo + w])
    kinds = pairs.get((p, canon(s)))
    if kinds is None:
        out['unmatched'].append(i)
    elif kinds == {'search'}:
        out['search_only'].append(i)
    elif kinds == {'regprecise'}:
        out['regprecise_only'].append(i)
    elif 'named' in kinds:
        out['named'].append(i)
    else:
        out['search_and_regprecise'].append(i)
split = {r['protein'] for r in read_tsv('data/derived/split.tsv')}
for k, v in sorted(out.items()):
    print('  %-22s строк %5d, белков %3d (из них наших %d)' % (k, len(v), len({P['protein'][i] for i in v}), len({P['protein'][i] for i in v} & split)))
json.dump({k: v for k, v in out.items()}, open(os.path.join(PACKS, 'bacregdb_origin.json'), 'w'))
