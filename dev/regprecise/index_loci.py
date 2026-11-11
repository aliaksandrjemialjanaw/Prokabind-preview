"""Указатель локус-тегов по записям геномов, которые уже лежат на диске (наши геномы и геномы T513), — чтобы поставить
сайты RegPrecise на свои места без нового скачивания там, где запись уже есть (круг улучшений, решение владельца 21).

Из каждого файла .gbff берутся признаки CDS: /locus_tag и /old_locus_tag, положение, цепь, белок (/translation).
Выход (вне git): {OUT}/locus_index.tsv — tag, kind (locus_tag | old_locus_tag), accession, start, end, strand, protein_sha256,
length; {OUT}/locus_proteins.fasta — белки по sha256 (без повторов).
Запуск: python3 -I -B dev/regprecise/index_loci.py [OUT_DIR [extra]]
`extra` — указатель только по папке data/reference_cache/nuccore_regprecise (геномы `fetch_genomes.py`): файлы
locus_index_extra.tsv и locus_proteins_extra.fasta; `place_sites.py` читает оба указателя."""
import hashlib
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data/derived/regprecise')
os.makedirs(OUT, exist_ok=True)
DIRS = [os.path.join(ROOT, 'data/reference_cache/nuccore'), os.path.join(ROOT, 'data/reference_cache/nuccore_bacregdb')]
SUFFIX = ''
if len(sys.argv) > 2 and sys.argv[2] == 'extra':       # только геномы, взятые для RegPrecise (fetch_genomes.py): отдельный указатель
    DIRS = [os.path.join(ROOT, 'data/reference_cache/nuccore_regprecise')]
    SUFFIX = '_extra'
loc_re = re.compile(r'(\d+)\.\.>?(\d+)')
t0 = time.time()
seen_p = set()
n_files = n_cds = 0
with open(os.path.join(OUT, 'locus_index%s.tsv' % SUFFIX), 'w') as idx, open(os.path.join(OUT, 'locus_proteins%s.fasta' % SUFFIX), 'w') as fa:
    idx.write('tag\tkind\taccession\tstart\tend\tstrand\tprotein\tlength\n')
    for d in DIRS:
        for name in sorted(os.listdir(d)):
            if not name.endswith('.gbff'):
                continue
            acc = name[:-5]
            n_files += 1
            feat = None                                   # текущий признак CDS: [положение, {квалификаторы}]

            def flush():
                global n_cds
                if feat is None:
                    return
                loc, q = feat
                spans = [(int(a), int(b)) for a, b in loc_re.findall(loc)]
                if not spans or 'translation' not in q:
                    return
                start, end = min(a for a, _ in spans) - 1, max(b for _, b in spans)
                strand = '-' if 'complement' in loc else '+'
                prot = re.sub(r'[^A-Za-z]', '', q['translation']).upper()
                key = hashlib.sha256(prot.encode()).hexdigest()
                if key not in seen_p:
                    seen_p.add(key)
                    fa.write('>%s\n%s\n' % (key, prot))
                for kind in ('locus_tag', 'old_locus_tag'):
                    for tag in q.get(kind + '_all', ()):
                        idx.write('%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\n' % (tag, kind, acc, start, end, strand, key, len(prot)))
                n_cds += 1
            key_q = None
            in_features = False
            for line in open(os.path.join(d, name), errors='replace'):
                if line.startswith('FEATURES'):
                    in_features = True
                    continue
                if line.startswith('ORIGIN') or line.startswith('CONTIG'):
                    flush(); feat = None; in_features = False
                    continue
                if not in_features:
                    continue
                if len(line) > 5 and line[5] != ' ':                     # начало признака
                    flush()
                    kind, loc = line[5:21].strip(), line[21:].strip()
                    feat = [loc, {}] if kind == 'CDS' else None
                    key_q = None
                    continue
                if feat is None:
                    continue
                body = line[21:].rstrip('\n')
                m = re.match(r'/(\w+)=?"?(.*)', body) if body.startswith('/') else None
                if m:
                    key_q, val = m.group(1), m.group(2)
                    closed = val.endswith('"')
                    val = val.rstrip('"')
                    if key_q in ('locus_tag', 'old_locus_tag'):
                        feat[1].setdefault(key_q + '_all', []).append(val)
                    elif key_q == 'translation':
                        feat[1]['translation'] = val
                    if closed and key_q != 'translation':
                        key_q = None
                elif key_q == 'translation':
                    feat[1]['translation'] += body.rstrip('"')
                elif key_q is None and feat[1].get('translation') is None and not body.startswith('/'):
                    feat[0] += body.strip()                              # перенос длинного положения
            flush()
            if n_files % 500 == 0:
                print('  файлов %d, CDS %d, %.0f с' % (n_files, n_cds, time.time() - t0), flush=True)
print('готово: файлов %d, CDS %d, разных белков %d, %.0f с' % (n_files, n_cds, len(seen_p), time.time() - t0))
