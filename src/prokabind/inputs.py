"""Вход программы: белок, геном, аннотация и топология (SPEC п. 5.1).

Пишет задача прототипа T16. Без PyTorch. Ошибка входа — **отказ с кодом и причиной** (`Refusal`); молчаливое
обрезание, замена букв и подстановка нулей запрещены (п. 5.7). Отказ по белку отменяет запуск, отказ одного
репликона — нет (п. 5.1).

Координаты внутри — с нуля, конец не включается (`CONTRACTS.md` § 1): аннотация переводится один раз, при чтении.
"""
import hashlib
import re

#: Предел белковой сети (SPEC пп. 4.2, 5.1).
MAX_PROTEIN = 1022

#: 20 стандартных аминокислот и пять нестандартных букв словаря белковой сети (SPEC п. 5.1).
STANDARD = set('ACDEFGHIKLMNPQRSTVWY')
NONCANONICAL = set('XBZUO')

#: Буквы генома: A, C, G, T и коды IUPAC; `U` читается как `T` с пометкой (п. 5.1).
IUPAC = set('ACGTRYSWKMBDHVN')

#: Записи аннотации, внутри которых основание — «внутри гена» (SPEC п. 3.4; те же типы, что у обучения).
GENE_TYPES = ('gene', 'CDS', 'rRNA', 'tRNA', 'ncRNA', 'tmRNA')


class Refusal(Exception):
    """Отказ запуска с кодом SPEC п. 5.1 и причиной."""

    def __init__(self, code, reason):
        super().__init__('%s: %s' % (code, reason))
        self.code = code
        self.reason = reason


def protein_key(sequence):
    """Ключ белка (`CONTRACTS.md` § 1): sha256 последовательности заглавными буквами без конечной `*`."""
    return hashlib.sha256(sequence.upper().rstrip('*').encode('ascii')).hexdigest()


def _fasta_records(path):
    """Записи FASTA: `(имя, заголовок, [строки последовательности])`; строки — без перевода строки и без пробелов
    по краям (пробел **внутри** строки остаётся и ловится проверкой букв)."""
    records = []
    with open(path, encoding='ascii', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\r\n')
            if line.startswith('>'):
                header = line[1:].strip()
                records.append((header.split()[0] if header else '', header, []))
            elif line.strip() == '' or line.startswith(';'):
                continue
            elif not records:
                raise Refusal('bad_fasta', '%s: последовательность до первого заголовка «>»' % path)
            else:
                records[-1][2].append(line.strip())
    return records


def read_protein(path):
    """Белок п. 5.1: одна запись FASTA, от 1 до 1 022 остатков. Ответ — `{'name', 'sequence', 'key', 'length',
    'notes'}`; ошибка — `Refusal`.

    Конечная `*` снимается с пометкой; `*` внутри и `-` — отказ `bad_letter`; строчные буквы переводятся в
    заглавные с пометкой (это не подмена); `X`, `B`, `Z`, `U`, `O` передаются словарю белковой сети как есть
    с пометкой `noncanonical_residues`; `J`, цифры, пробелы внутри — отказ `bad_letter`."""
    records = _fasta_records(path)
    if not records:
        raise Refusal('no_record', '%s: в файле белка нет ни одной записи FASTA' % path)
    if len(records) > 1:
        raise Refusal('many_records', '%s: записей %d, а один запуск — один белок (SPEC п. 5.1)' % (path, len(records)))
    name, header, lines = records[0]
    raw = ''.join(lines)
    notes = []
    if raw.endswith('*'):
        raw = raw[:-1]
        notes.append('terminal_stop_removed')
    if raw != raw.upper():
        notes.append('lowercase_to_uppercase')
    seq = raw.upper()
    bad = sorted({ch for ch in seq if ch not in STANDARD and ch not in NONCANONICAL})
    if bad:
        raise Refusal('bad_letter', 'в белке недопустимые знаки %s (SPEC п. 5.1)' % ''.join(repr(c) for c in bad))
    if not 1 <= len(seq) <= MAX_PROTEIN:
        raise Refusal('unsupported_length', 'длина белка %d, допустимо от 1 до %d (предел белковой сети, SPEC п. 4.2); '
                      'белок не обрезается' % (len(seq), MAX_PROTEIN))
    odd = sorted({ch for ch in seq if ch in NONCANONICAL})
    if odd:
        notes.append('noncanonical_residues:' + ''.join(odd))
    return {'name': name, 'header': header, 'sequence': seq, 'key': protein_key(seq), 'length': len(seq),
            'notes': notes}


def read_genome(path):
    """Геном п. 5.1: FASTA одного или нескольких репликонов. Ответ — список `{'name', 'sequence', 'notes'}` в порядке
    файла; ошибка — `Refusal`. Строчные буквы — заглавные (мягкая маска сборщика, не подмена), `U` — `T` с пометкой,
    буквы вне A, C, G, T и кодов IUPAC — отказ; повторяющееся имя — отказ `duplicate_replicon`."""
    records = _fasta_records(path)
    if not records:
        raise Refusal('no_record', '%s: в файле генома нет ни одной записи FASTA' % path)
    seen, out = set(), []
    for name, header, lines in records:
        if not name:
            raise Refusal('bad_fasta', '%s: запись генома без имени' % path)
        if name in seen:
            raise Refusal('duplicate_replicon', 'имя репликона %r повторяется в файле генома (SPEC п. 5.1)' % name)
        seen.add(name)
        raw = ''.join(lines)
        notes = []
        if raw != raw.upper():
            notes.append('lowercase_to_uppercase')
        seq = raw.upper()
        if 'U' in seq:
            seq = seq.replace('U', 'T')
            notes.append('u_read_as_t')
        bad = sorted({ch for ch in seq if ch not in IUPAC})
        if bad:
            raise Refusal('bad_letter', 'в репликоне %s недопустимые знаки %s (SPEC п. 5.1)'
                          % (name, ''.join(repr(c) for c in bad)))
        ambiguous = sum(1 for ch in set(seq) if ch not in 'ACGT')
        if ambiguous:
            notes.append('ambiguous_letters')
        out.append({'name': name, 'sequence': seq, 'notes': notes})
    return out


# ---- аннотация ------------------------------------------------------------------------------
def _gbff_ranges(location):
    """Интервалы положения записи GBFF: с единицы, конец включён → `start − 1`, `end` (`CONTRACTS.md` § 1)."""
    out = [(int(a) - 1, int(b)) for a, b in re.findall(r'<?(\d+)\s*\.\.\s*>?(\d+)', location)]
    if not out:
        single = re.search(r'(\d+)', location)
        if single:
            a = int(single.group(1))
            out = [(a - 1, a)]
    return out


def _qualifier(text, key):
    found = re.search(r'/%s="([^"]*)"' % key, text)
    return found.group(1).replace('\n', ' ').strip() if found else None


def _read_gbff(path):
    """Записи GBFF: имя — по полю VERSION (SPEC п. 5.1), длина и топология — по строке LOCUS."""
    records = {}
    current = None
    feature = None

    def close_feature():
        if current is not None and feature is not None and feature['kind'] in GENE_TYPES:
            text = '\n'.join(feature['lines'])
            location = ''
            for piece in feature['lines']:
                if piece.lstrip().startswith('/'):
                    break
                location += piece.strip()
            name = _qualifier(text, 'gene') or _qualifier(text, 'locus_tag')
            for a, b in _gbff_ranges(location):
                current['genes'].append((a, b, feature['kind'], name, 'complement' in location))

    with open(path, encoding='utf-8', errors='replace') as handle:
        section = None
        for line in handle:
            line = line.rstrip('\n')
            if line.startswith('LOCUS'):
                words = line.split()
                length = None
                for i, w in enumerate(words):
                    if w in ('bp', 'aa') and i > 0 and words[i - 1].isdigit():
                        length = int(words[i - 1])
                topo = 'circular' if re.search(r'\bcircular\b', line) else (
                    'linear' if re.search(r'\blinear\b', line) else None)
                current = {'name': None, 'length': length, 'topology': topo, 'genes': []}
                section, feature = None, None
            elif current is None:
                continue
            elif line.startswith('VERSION'):
                parts = line.split()
                current['name'] = parts[1] if len(parts) > 1 else None
            elif line.startswith('FEATURES'):
                section = 'features'
            elif line.startswith('ORIGIN') or line.startswith('CONTIG'):
                close_feature()
                feature, section = None, None
            elif line.startswith('//'):
                close_feature()
                feature, section = None, None
                if current['name']:
                    if current['name'] in records:
                        raise Refusal('duplicate_replicon', 'имя %r повторяется в аннотации' % current['name'])
                    records[current['name']] = current
                current = None
            elif section == 'features':
                if len(line) > 5 and line[5] != ' ' and line[:5] == '     ':
                    close_feature()
                    feature = {'kind': line[5:21].strip(), 'lines': [line[21:]]}
                elif feature is not None:
                    feature['lines'].append(line[21:])
    if current is not None and current['name'] and current['name'] not in records:
        close_feature()
        records[current['name']] = current
    return records


def _read_gff3(path):
    """Записи GFF3: длина — по `##sequence-region` или записи `region`, кольцо — `Is_circular=true` (SPEC п. 5.1)."""
    records = {}

    def record(name):
        return records.setdefault(name, {'name': name, 'length': None, 'topology': None, 'genes': []})

    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\n')
            if line.startswith('##FASTA'):
                break
            if line.startswith('##sequence-region'):
                parts = line.split()
                if len(parts) >= 4:
                    record(parts[1])['length'] = int(parts[3]) - int(parts[2]) + 1
                continue
            if not line or line.startswith('#'):
                continue
            f = line.split('\t')
            if len(f) < 9:
                continue
            rec = record(f[0])
            attrs = {}
            for item in f[8].split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    attrs[k.strip()] = v.strip()
            if f[2] == 'region':
                if attrs.get('Is_circular', '').lower() == 'true':
                    rec['topology'] = 'circular'
                if rec['length'] is None and f[3] == '1':
                    rec['length'] = int(f[4])
            elif f[2] in GENE_TYPES:
                name = attrs.get('gene') or attrs.get('Name') or attrs.get('locus_tag') or attrs.get('ID')
                rec['genes'].append((int(f[3]) - 1, int(f[4]), f[2], name, f[6] == '-'))
    return records


def read_annotation(path):
    """Аннотация GBFF или GFF3 (п. 5.1): `{имя репликона: {'name', 'length', 'topology', 'genes'}}`; гены —
    `(start, end, тип, имя, на обратной цепи)` с нуля, конец не включён."""
    with open(path, encoding='utf-8', errors='replace') as handle:
        head = handle.read(4096)
    if head.lstrip().startswith('##gff-version') or '\t' in head.split('\n', 1)[0]:
        return 'gff3', _read_gff3(path)
    if head.startswith('LOCUS'):
        return 'gbff', _read_gbff(path)
    raise Refusal('bad_annotation', '%s: ни GBFF (строка LOCUS), ни GFF3 (##gff-version)' % path)


def match_annotation(replicons, annotation, topology_keys):
    """Сопоставление аннотации с FASTA по имени записи и выбор топологии каждого репликона (п. 5.1).

    `topology_keys` — `{имя: 'circular' | 'linear'}` из ключей командной строки. Порядок источников: ключ
    командной строки, затем аннотация; **не задана — линейный** (это всегда безопасно). Длина записи аннотации не
    равна длине записи FASTA — отказ `annotation_mismatch`: аннотация другой сборки хуже, чем никакой.

    Ответ — по репликону `{'topology', 'topology_source', 'genes' (или None — аннотации нет)}`."""
    names = [r['name'] for r in replicons]
    unknown = sorted(set(topology_keys) - set(names))
    if unknown:
        raise Refusal('unknown_replicon', 'ключ топологии называет репликоны, которых нет в FASTA: %s' % ', '.join(unknown))
    out = {}
    for rep in replicons:
        name, n = rep['name'], len(rep['sequence'])
        ann = annotation.get(name) if annotation else None
        if ann is not None and ann['length'] is not None and int(ann['length']) != n:
            raise Refusal('annotation_mismatch', 'репликон %s: в аннотации длина %d, в FASTA %d (SPEC п. 5.1)'
                          % (name, int(ann['length']), n))
        if name in topology_keys:
            topo, source = topology_keys[name], 'command_line'
        elif ann is not None and ann['topology'] in ('circular', 'linear'):
            topo, source = ann['topology'], 'annotation'
        else:
            topo, source = 'linear', 'default_linear'
        genes = None
        if ann is not None:
            genes = sorted(ann['genes'], key=lambda g: (g[0], g[1], g[2], str(g[3])))
        out[name] = {'topology': topo, 'topology_source': source, 'genes': genes}
    return out
