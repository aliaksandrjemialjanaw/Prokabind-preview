"""Чтение таблиц данных для обучения (`CONTRACTS.md` § 2.2–2.6, § 2.8) и правила, которые считаются
по ним: обучающая часть модели, запретные основания, «липкие» места, чистые сайты.

Без PyTorch: это разбор таблиц и арифметика интервалов. Координаты внутри проекта — с нуля, конец
не включается (`CONTRACTS.md` § 1).

Этот модуль контракт не называл — он был внутренним делом кода (§ 4). С T512 `CONTRACTS.md` § 4.11 называет его
имена при поле `bacregdb`: `bacregdb_of`, `genome_dirs`, чтение геномов из списка папок, `RECORD_SEQUENCES` и
`held_records`, `corrected_rows`, `training_proteins`, `training_labels`. Остальное наружу не обещано.

Отложенный набор этот модуль не читает: его строки — не таблица меток обучения (SPEC п. 3.2).
"""
import csv
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

#: Записи аннотации, вне которых основание считается межгенным (SPEC п. 3.4).
GENE_TYPES = ('gene', 'CDS', 'rRNA', 'tRNA', 'ncRNA', 'tmRNA')

#: «общий отрезок ≥15 оснований» правила «чистого» сайта (SPEC п. 2.9).
KMER = 15

#: «перекрытие по координатам не меньше чем на min(8, длина сайта)» (SPEC п. 2.9).
MIN_OVERLAP = 8

#: «места, где садятся 20 и больше разных белков» (SPEC пп. 2.6, 4.7).
STICKY_PROTEINS = 20

#: Ключ 15-меры с буквой вне ACGT: настоящий ключ не больше 4**15 − 1, поэтому не совпадёт.
INVALID = np.uint32(0xFFFFFFFF)

#: Сколько записей второй папки (поле `bacregdb`, `CONTRACTS.md` § 4.11) процесс держит в памяти разом:
#: последовательность и маска генов записи — вместе, в очереди последнего обращения; `None` — без предела. Значение
#: читается при каждом обращении (тест 24 его подменяет).
RECORD_SEQUENCES = 64

#: Пометки сайта от самой точной к наименее точной: при сведении строк с одинаковыми новыми границами остаётся
#: строка с самой точной (SPEC п. 2.5.2, `CONTRACTS.md` § 4.11).
LABEL_ORDER = ('footprint', 'method_not_recorded', 'motif_in_emsa', 'longer_than_30', 'prodoric')

#: «вариант ≥90%»: доля идентичности столбца `related` § 2.10, с которой белок BacRegDB — вариант нашего белка.
VARIANT_IDENTITY = 0.900

#: Ключи поля `bacregdb` (`CONTRACTS.md` § 2.9).
BACREGDB_KEYS = ('tables', 'genomes', 'cache', 'label_weight')

_CACHE = {}

#: Записи второй папки: (папка, номер) → {'seq', 'stat', 'genes': {длина: маска}, 'genes_stat'}; от давней к последней.
_RECORDS = OrderedDict()


def read_tsv(path):
    """Строки TSV с заголовком (`CONTRACTS.md` § 1)."""
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


# ---- папки геномов: наши и записи T513 (поле `bacregdb`, `CONTRACTS.md` § 4.11) --------------------------------------
def _dirs(genomes_dir):
    """Папка или список папок → список строк путей в том же порядке."""
    if isinstance(genomes_dir, (list, tuple)):
        return [str(d) for d in genomes_dir]
    return [str(genomes_dir)]


def genome_folder(genomes_dir, accession):
    """`(номер папки в списке, папка)`, где лежит запись (`.fasta` или `.gbff`), — первая по порядку списка; нет ни в
    одной — `(None, первая папка)`: тогда чтение отказывает, как прежде."""
    dirs = _dirs(genomes_dir)
    for number, folder in enumerate(dirs):
        if os.path.exists(os.path.join(folder, accession + '.fasta')) or \
                os.path.exists(os.path.join(folder, accession + '.gbff')):
            return number, folder
    return None, dirs[0]


def is_record(genomes_dir, accession):
    """Запись второй папки: её нет в первой папке списка, а есть в следующей (`CONTRACTS.md` § 4.11)."""
    number, _ = genome_folder(genomes_dir, accession)
    return bool(number)


def _stat(path):
    try:
        found = os.stat(path)
    except OSError:
        return None
    return found.st_size, found.st_mtime_ns


def _trim_records():
    """Сверх `RECORD_SEQUENCES` уходят давние записи — последовательность и маска вместе. Зовётся при каждом
    обращении к записи, поэтому уменьшенный предел действует с ближайшего обращения."""
    bound = RECORD_SEQUENCES
    if bound is None:
        return
    while len(_RECORDS) > max(int(bound), 0):
        _RECORDS.popitem(last=False)


def _record(folder, accession):
    """Запись второй папки из ограниченного кэша: прочитана заново, если её нет или файл изменился с чтения."""
    key = (folder, accession)
    path = os.path.join(folder, accession + '.fasta')
    stat = _stat(path)
    entry = _RECORDS.get(key)
    if entry is None or entry['stat'] != stat:
        if stat is None:
            raise FileNotFoundError('нет файла генома %s' % path)
        _RECORDS.pop(key, None)
        with open(path) as handle:
            sequence = ''.join(l.strip() for l in handle if not l.startswith('>')).upper()
        entry = {'seq': sequence, 'stat': stat, 'genes': {}, 'genes_stat': None}
        del sequence
        _RECORDS[key] = entry
    else:
        _RECORDS.move_to_end(key)
    return entry


def held_records():
    """Номера записей второй папки, которые процесс держит сейчас, от давней к последней (`CONTRACTS.md` § 4.11)."""
    return [accession for _, accession in _RECORDS]


def genome_dirs(config):
    """Папки геномов обучения (`CONTRACTS.md` § 4.11): `[genomes_dir]` или, при включённом поле `bacregdb`,
    `[genomes_dir, bacregdb.genomes]` — запись ищется по порядку."""
    field = bacregdb_of(config)
    out = [config['genomes_dir']]
    if field is not None:
        out.append(field['genomes'])
    return out


def genome_sequence(genomes_dir, accession):
    """Последовательность репликона (`CONTRACTS.md` § 2.5), в верхнем регистре. `genomes_dir` — папка или список
    папок (§ 4.11): запись ищется по порядку. Наши геномы (первая папка) держатся всё время процесса, записи
    второй папки — в ограниченном кэше (`RECORD_SEQUENCES`)."""
    key = ('seq', _dirs(genomes_dir)[0], accession)
    if key in _CACHE:                         # наш геном, уже прочитанный: без обращения к диску, как прежде
        return _CACHE[key]
    number, folder = genome_folder(genomes_dir, accession)
    if number:
        try:
            return _record(folder, accession)['seq']
        finally:
            _trim_records()
    key = ('seq', folder, accession)
    if key not in _CACHE:
        path = Path(folder) / (accession + '.fasta')
        if not path.exists():
            raise FileNotFoundError('нет файла генома %s' % path)
        with open(path) as handle:
            _CACHE[key] = ''.join(l.strip() for l in handle if not l.startswith('>')).upper()
    return _CACHE[key]


def _features(path):
    """Пары «тип записи, строка положения» из раздела FEATURES файла GBFF."""
    kind, location, qualifiers = None, [], False
    with open(path) as handle:
        inside = False
        for line in handle:
            if line.startswith('FEATURES'):
                inside = True
                continue
            if line.startswith('ORIGIN') or line.startswith('//'):
                break
            if not inside:
                continue
            body = line[21:].rstrip('\n') if len(line) > 21 else ''
            if len(line) > 5 and line[5] != ' ':
                if kind is not None:
                    yield kind, ''.join(location)
                kind, location, qualifiers = line[5:21].strip(), [body.strip()], False
            elif kind is not None:
                if body.lstrip().startswith('/'):
                    qualifiers = True
                elif not qualifiers:
                    location.append(body.strip())
    if kind is not None:
        yield kind, ''.join(location)


def _ranges(location):
    """Интервалы положения записи GBFF в координатах проекта: с единицы и концом включительно
    в источнике → `start − 1`, `end` без изменения (`CONTRACTS.md` § 1)."""
    out = [(int(a) - 1, int(b)) for a, b in re.findall(r'<?(\d+)\s*\.\.\s*>?(\d+)', location)]
    if not out:
        single = re.search(r'(\d+)', location)
        if single:
            a = int(single.group(1))
            out = [(a - 1, a)]
    return out


def gene_mask(genomes_dir, accession, length):
    """Маска «основание внутри гена» (SPEC п. 3.4): вне записей `gene`, `CDS`, `rRNA`, `tRNA`,
    `ncRNA`, `tmRNA` основание межгенное.

    Файла аннотации нет — отказ, а не «генов нет»: тип примера входит в подбор отрицательных окон
    (SPEC п. 4.5), и молчаливое «всё межгенное» изменило бы обучение.

    `genomes_dir` — папка или список папок (§ 4.11). Маска записи второй папки держится вместе с её
    последовательностью в ограниченном кэше (`RECORD_SEQUENCES`) и уходит вместе с ней.
    """
    key = ('genes', _dirs(genomes_dir)[0], accession, int(length))
    if key in _CACHE:
        return _CACHE[key]
    number, folder = genome_folder(genomes_dir, accession)
    if number:
        try:
            entry = _record(folder, accession)
            path = os.path.join(folder, accession + '.gbff')
            stat = _stat(path)
            if entry['genes_stat'] != stat:
                entry['genes'], entry['genes_stat'] = {}, stat
            mask = entry['genes'].get(int(length))
            if mask is None:
                mask = _gene_mask(Path(path), length)
                entry['genes'][int(length)] = mask
            return mask
        finally:
            _trim_records()
    key = ('genes', folder, accession, int(length))
    if key not in _CACHE:
        _CACHE[key] = _gene_mask(Path(folder) / (accession + '.gbff'), length)
    return _CACHE[key]


def _gene_mask(path, length):
    if not path.exists():
        raise FileNotFoundError('нет аннотации %s: тип примера по п. 3.4 посчитать нечем' % path)
    mask = np.zeros(int(length), dtype=bool)
    for kind, location in _features(path):
        if kind in GENE_TYPES:
            for a, b in _ranges(location):
                mask[max(a, 0):min(b, int(length))] = True
    return mask


def topology(genomes_dir, accession):
    """`circular` или `linear` по **тексту** строки LOCUS файла GBFF (`CONTRACTS.md` § 2.5).
    Слова нет — репликон считается линейным: это всегда безопасно (SPEC п. 5.1). `genomes_dir` — папка или
    список папок (§ 4.11)."""
    key = ('topo', _dirs(genomes_dir)[0], accession)
    if key in _CACHE:
        return _CACHE[key]
    _, folder = genome_folder(genomes_dir, accession)
    key = ('topo', folder, accession)
    if key not in _CACHE:
        path = Path(folder) / (accession + '.gbff')
        word = None
        if path.exists():
            with open(path) as handle:
                found = re.search(r'\b(circular|linear)\b', handle.readline())
                word = found.group(1) if found else None
        _CACHE[key] = word or 'linear'
    return _CACHE[key]


def fold_column(choices):
    """Какой столбец разбиения действует (`CONTRACTS.md` § 2.6, § 2.9)."""
    return 'part' if int(choices.get('folds', 5)) == 5 else 'part3'


# ---- поле `bacregdb` (`CONTRACTS.md` § 2.9–2.11, § 4.11; SPEC пп. 2.7.3, 2.7.8, 4.5) ---------------------------------
def bacregdb_of(config):
    """Значение поля `bacregdb` (§ 2.9): `None` при отсутствии поля и при `null`. Объект проверяется: ровно четыре
    ключа (`tables`, `genomes`, `cache`, `label_weight`), три папки существуют — иначе отказ, а не догадка."""
    field = config.get('bacregdb') if config is not None else None
    if field is None:
        return None
    if not isinstance(field, dict) or sorted(field) != sorted(BACREGDB_KEYS):
        raise ValueError('поле bacregdb — null или объект с ключами %s (CONTRACTS § 2.9), получено %r'
                         % (', '.join(BACREGDB_KEYS), field))
    for name in ('tables', 'genomes', 'cache'):
        if not os.path.isdir(str(field[name])):
            raise ValueError('поле bacregdb: папки %s = %r нет (CONTRACTS § 4.11)' % (name, field[name]))
    weight = field['label_weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not weight > 0:
        raise ValueError('поле bacregdb: label_weight — положительное число, получено %r' % (weight,))
    return field


def _table(field, name):
    """Таблица § 2.10 / § 2.11 папки поля; читается при каждом вызове (таблицы могут переписать между двумя
    обучениями одного процесса). Нет таблицы — отказ (`FileNotFoundError`), а не «мест нет»: без `genome_labels.tsv`
    обучение молча шло бы без мест T513."""
    return read_tsv(Path(field['tables']) / name)


def _parts(row, choices):
    """Части разбиения родственников белка BacRegDB в действующем столбце (`related_parts` или `related_parts3`)."""
    column = 'related_parts' if fold_column(choices) == 'part' else 'related_parts3'
    return {x for x in (row.get(column) or '').split(';') if x}


def _admitted(proteins, choices, fold):
    """Белки BacRegDB, допущенные к модели по SPEC п. 2.7.3 (решение владельца «per model»): к заходу `k` — если в
    частях родственников нет ни `fold{k}`, ни `control`, ни `holdout`; к итоговой модели — нет ни `control`, ни
    `holdout`."""
    bad = {'control', 'holdout'}
    if fold is not None:
        bad.add('fold%d' % int(fold))
    return {r['protein'] for r in proteins if not _parts(r, choices) & bad}


def _bacregdb_places(field):
    """Места BacRegDB: строки `labels.tsv` с `kind` = `coord` (T511) и все строки `genome_labels.tsv` (T513)."""
    labels = _table(field, 'labels.tsv')
    found = _table(field, 'genome_labels.tsv')
    return [r for r in labels if r['kind'] == 'coord' and r['genome']] + list(found), labels, found


def _bacregdb_rule(config, choices, fold):
    """Разбор поля для одной модели: строки `proteins.tsv`, допущенные правилом, места и пробы."""
    field = bacregdb_of(config)
    proteins = _table(field, 'proteins.tsv')
    places, labels, found = _bacregdb_places(field)
    rule = _admitted(proteins, choices, fold)
    placed = {r['protein'] for r in places}
    return field, proteins, places, labels, found, rule, placed


def _place_row(row):
    return {'protein': row['protein'], 'genome': row['genome'], 'start': int(row['start']), 'end': int(row['end']),
            'lane': 'bacregdb', 'label': 'bacregdb', 'strength': None, 'summit': None,
            'n_places': int(row['n_places'])}


def bacregdb_kin(config):
    """Белки BacRegDB, чьи места — собственные метки нашего белка (SPEC п. 3.4, `own_labels`): сам белок
    (`identical` = `yes`) и белки BacRegDB, у которых в `related` он стоит с долей ≥ 0,900 (его варианты ≥90%).
    Ответ — словарь «наш белок → множество ключей белков BacRegDB»; при выключенном поле — пустой."""
    field = bacregdb_of(config)
    if field is None:
        return {}
    out = {}
    for row in _table(field, 'proteins.tsv'):
        if row.get('identical') == 'yes':
            out.setdefault(row['protein'], set()).add(row['protein'])
        for item in (row.get('related') or '').split(';'):
            if not item:
                continue
            ours, _, share = item.rpartition(':')
            if round(float(share), 3) >= VARIANT_IDENTITY:
                out.setdefault(ours, set()).add(row['protein'])
    return out


def bacregdb_places(config):
    """Все места BacRegDB поля (§ 2.10 `coord` и § 2.11) строками `{'protein', 'genome', 'start', 'end'}`, без
    правила допуска: собственные метки белка убираются из его фона, даже если они не учат ни одну модель."""
    field = bacregdb_of(config)
    if field is None:
        return []
    return [{'protein': r['protein'], 'genome': r['genome'], 'start': int(r['start']), 'end': int(r['end'])}
            for r in _bacregdb_places(field)[0]]


def _corrections(field):
    return {(r['protein'], r['genome'], int(r['start']), int(r['end']), r['lane']): r
            for r in _table(field, 'corrections.tsv')}


def _merge(rows, table):
    """Исправленные границы и сведение (SPEC пп. 2.7.8, 2.5.2; `CONTRACTS.md` § 4.11): строка дорожки `exact` или
    `prodoric` со строкой `corrections.tsv` получает `start`, `end`, `seq` ← `new_start`, `new_end`, `new_seq`; затем
    строки одного белка на одной записи с одинаковыми границами — одна, с самой точной пометкой (`LABEL_ORDER`), при
    равенстве — с меньшими прежними `start`, `end`. Строки других дорожек не трогаются. Порядок — по `genome`,
    `start`, `end`, `protein` (как § 2.2), устойчиво. Ответ — `(строки, исправленных строк после сведения, групп
    совпадений)`; исправленная — у которой хоть одна строка группы есть в `corrections.tsv` (класс `same` тоже)."""
    groups, order, other = {}, [], []
    for position, row in enumerate(rows):
        if row['lane'] not in ('exact', 'prodoric'):
            other.append((position, row))
            continue
        old = (int(row['start']), int(row['end']))
        fix = table.get((row['protein'], row['genome'], old[0], old[1], row['lane']))
        if fix is None:
            new, seq = old, row.get('seq')
        else:
            new, seq = (int(fix['new_start']), int(fix['new_end'])), fix['new_seq']
        key = (row['protein'], row['genome'], new[0], new[1])
        rank = (LABEL_ORDER.index(row['label']) if row['label'] in LABEL_ORDER else len(LABEL_ORDER), old[0], old[1])
        if key not in groups:
            groups[key] = {'position': position, 'best': None, 'size': 0, 'fixed': False}
            order.append(key)
        group = groups[key]
        group['size'] += 1
        group['fixed'] = group['fixed'] or fix is not None
        if group['best'] is None or rank < group['best'][0]:
            if fix is None:
                made = row
            else:
                as_text = isinstance(row['start'], str)
                made = dict(row, start=str(new[0]) if as_text else new[0], end=str(new[1]) if as_text else new[1])
                if 'seq' in row:
                    made['seq'] = seq
            group['best'] = (rank, made)
    out = [(groups[k]['position'], groups[k]['best'][1]) for k in order] + other
    out.sort(key=lambda x: x[0])
    rows_out = [row for _, row in out]
    rows_out.sort(key=lambda r: (r['genome'], int(r['start']), int(r['end']), r['protein']))
    corrected = sum(1 for k in order if groups[k]['fixed'])
    collisions = sum(1 for k in order if groups[k]['size'] > 1)
    return rows_out, corrected, collisions


def corrected_rows(config, rows):
    """Строки § 2.2 с исправленными границами и последовательностью нового отрезка, сведённые (`_merge`;
    `CONTRACTS.md` § 4.11). При выключенном поле — строки без изменений."""
    field = bacregdb_of(config)
    if field is None:
        return rows
    return _merge(rows, _corrections(field))[0]


def training_proteins(config, choices, fold):
    """Белки, чьи метки входят в обучающую часть **этой** модели, по возрастанию ключа.

    Контрольная часть и отложенный набор не учатся никогда (SPEC пп. 3.1, 3.2, 3.10); у модели захода
    не учатся и проверочные белки её захода (п. 2.8). Белок, у которого нет обучающих меток — только
    роль `galagan`, — ни на чём не учится (`CONTRACTS.md` § 2.7); белок только с ролью `prodoric`
    учится, лишь когда PRODORIC в обучении (п. 2.5.3).

    При включённом поле `bacregdb` (§ 4.11) к ним добавляются белки BacRegDB, допущенные к этой модели правилом
    п. 2.7.3 и имеющие хоть одно место; допущенный без места меток не имеет и обучающим не считается. При
    выключенном поле таблицы BacRegDB не читаются.
    """
    out = _our_training(config, choices, fold)
    if bacregdb_of(config) is None:
        return out
    _, _, _, _, _, rule, placed = _bacregdb_rule(config, choices, fold)
    return sorted(set(out) | (rule & placed))


def bacregdb_counts(config, choices, fold):
    """Числа поля одной модели (журнал § 3.0.2, отчёт T512): `proteins` — допущенные с местом (ключи по
    возрастанию), `excluded` — недопущенные правилом, `no_place` — допущенные без места, `rows` — строки меток
    BacRegDB, `corrected` и `collisions` — исправленные строки и группы совпадений обучающих сайтов после
    сведения, `unused_probes` — пробы допущенных правилом белков без места T513. `None` при выключенном поле."""
    if bacregdb_of(config) is None:
        return None
    field, proteins, places, labels, found, rule, placed = _bacregdb_rule(config, choices, fold)
    inside = set(training_proteins(config, choices, fold))
    lanes = ('exact', 'prodoric') if bool(choices.get('prodoric_in_training', True)) else ('exact',)
    ours = [r for r in read_tsv(Path(config['data_root']) / 'sites.tsv')
            if r['protein'] in inside and r['lane'] in lanes]
    _, corrected, collisions = _merge(ours, _corrections(field))
    pairs = {(r['protein'], r['site']) for r in found}
    return OrderedDict([
        ('proteins', sorted(rule & placed)), ('excluded', len(proteins) - len(rule)),
        ('no_place', len(rule - placed)), ('rows', sum(1 for r in places if r['protein'] in rule)),
        ('corrected', corrected), ('collisions', collisions),
        ('unused_probes', sum(1 for r in labels if r['kind'] == 'probe' and r['protein'] in rule
                              and (r['protein'], r['site']) not in pairs))])


def _our_training(config, choices, fold):
    """Обучающая часть по таблицам проекта (прежнее правило, без поля `bacregdb`)."""
    root = Path(config['data_root'])
    column = fold_column(choices)
    own = None if fold is None else 'fold%d' % int(fold)
    roles = {r['protein']: {x for x in r['roles'].split(';') if x} for r in read_tsv(root / 'proteins.tsv')}
    wanted = {'exact', 'streptrn'}
    if bool(choices.get('prodoric_in_training', True)):
        wanted.add('prodoric')
    out = []
    for row in read_tsv(root / 'split.tsv'):
        part = row[column]
        if part in ('control', 'holdout') or (own is not None and part == own):
            continue
        if roles.get(row['protein'], set()) & wanted:
            out.append(row['protein'])
    return sorted(out)


def training_labels(config, choices, fold, proteins=None):
    """Обучающие метки этой модели: точные сайты, сайты PRODORIC при `prodoric_in_training` и пики
    StrepTRN белков обучающей части (SPEC п. 4.5). Одна запись — словарь с `protein`, `genome`,
    `start`, `end`, `lane`, `label`, `strength`, `summit`.

    При включённом поле `bacregdb` (§ 4.11) точные сайты и сайты PRODORIC — в исправленных границах после
    сведения (`corrected_rows`), и к ним — строки дорожки `bacregdb` (пометка `bacregdb`, ключ `n_places`) белков,
    допущенных к модели правилом п. 2.7.3: места `labels.tsv` (`kind` = `coord`) и `genome_labels.tsv`; пробы не
    входят. Порядок прежний."""
    root = Path(config['data_root'])
    field = bacregdb_of(config)
    inside = set(training_proteins(config, choices, fold) if proteins is None else proteins)
    prodoric = bool(choices.get('prodoric_in_training', True))
    out = []
    sites = [row for row in read_tsv(root / 'sites.tsv') if row['protein'] in inside and
             (row['lane'] == 'exact' or (row['lane'] == 'prodoric' and prodoric))]
    if field is not None:
        sites = _merge(sites, _corrections(field))[0]
    for row in sites:
        out.append({'protein': row['protein'], 'genome': row['genome'], 'start': int(row['start']),
                    'end': int(row['end']), 'lane': row['lane'], 'label': row['label'],
                    'strength': None, 'summit': None})
    for row in read_tsv(root / 'peaks.tsv'):
        if row['protein'] not in inside:
            continue
        out.append({'protein': row['protein'], 'genome': row['genome'], 'start': int(row['start']),
                    'end': int(row['end']), 'lane': 'interval', 'label': 'streptrn',
                    'strength': float(row['strength']),
                    'summit': int(row['summit']) if row['summit'] not in ('', None) else None})
    if field is not None:
        _, _, places, _, _, rule, _ = _bacregdb_rule(config, choices, fold)
        out.extend(_place_row(row) for row in places if row['protein'] in rule and row['protein'] in inside)
    out.sort(key=lambda r: (r['genome'], r['start'], r['end'], r['protein'], r['lane']))
    return out


def merge_intervals(intervals):
    """Объединение перекрывающихся и соприкасающихся интервалов `[start, end)`."""
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def sticky_mask(labels, genome, length):
    """Маска «липких» мест (SPEC пп. 2.6, 4.7): основания, где садятся 20 и больше **разных** белков
    StrepTRN. Число белков считается по обучающей части этой модели: иначе место девятнадцати
    обучающих белков выпадало бы из-за двадцатого — проверочного (правка SPEC 29.10)."""
    by_protein = {}
    for row in labels:
        if row['lane'] == 'interval' and row['genome'] == genome:
            by_protein.setdefault(row['protein'], []).append((row['start'], row['end']))
    diff = np.zeros(int(length) + 1, dtype=np.int32)
    for protein in sorted(by_protein):
        for a, b in merge_intervals(by_protein[protein]):
            diff[max(a, 0)] += 1
            diff[min(b, int(length))] -= 1
    return np.cumsum(diff[:-1]) >= STICKY_PROTEINS


def kmer_keys(sequence):
    """Канонические ключи всех 15-мер строки: `key[i]` — окно `[i, i + 15)`, `INVALID` — окно с буквой
    вне ACGT. Каноническая форма — меньший из ключа окна и ключа его обратного комплемента, поэтому
    «любая цепь» правила SPEC п. 2.9 выполняется сравнением ключей."""
    raw = np.frombuffer(sequence.upper().encode('ascii', 'replace'), dtype=np.uint8)
    code = np.full(raw.size, 255, dtype=np.uint8)
    for value, base in enumerate('ACGT'):
        code[raw == ord(base)] = value
    n = code.size
    if n < KMER:
        return np.zeros(0, dtype=np.uint32)
    width = n - KMER + 1
    forward = np.zeros(width, dtype=np.uint32)
    backward = np.zeros(width, dtype=np.uint32)
    bad = np.zeros(width, dtype=bool)
    values = code.astype(np.uint32)
    complement = (np.uint32(3) - values) & np.uint32(3)
    unknown = code == 255
    for shift in range(KMER):
        window = slice(shift, shift + width)
        forward = (forward << np.uint32(2)) | values[window]
        backward = backward | (complement[window] << np.uint32(2 * shift))
        bad |= unknown[window]
    keys = np.minimum(forward, backward)
    keys[bad] = INVALID
    return keys


def unclean_sites(sites, labels, genomes_dir):
    """Номера строк `sites`, которые по правилу «чистого» сайта (SPEC п. 2.9) перекрываются с
    обучающей меткой **другого** белка: общий отрезок ≥15 оснований (любая цепь) **либо** перекрытие
    по координатам не меньше чем на `min(8, длина сайта)`.

    Оба правила нужны: у сайта короче 15 оснований общего отрезка ≥15 нет физически, а перекрытие по
    координатам проверяет именно то, что под сайтом лежит обучающая метка другого белка (SPEC п. 2.9).
    """
    bad = set()
    by_genome = {}
    for row in labels:
        by_genome.setdefault(row['genome'], []).append((row['start'], row['end'], row['protein']))
    prepared = {}
    for genome, items in by_genome.items():
        items.sort()
        prepared[genome] = (items, np.array([x[0] for x in items]),
                            max(b - a for a, b, _ in items))

    # ---- перекрытие по координатам
    for number, site in enumerate(sites):
        found = prepared.get(site['genome'])
        if found is None:
            continue
        items, starts, longest = found
        a, b = int(site['start']), int(site['end'])
        need = min(MIN_OVERLAP, b - a)
        lo = int(np.searchsorted(starts, a - longest, 'left'))
        hi = int(np.searchsorted(starts, b, 'left'))
        for la, lb, protein in items[lo:hi]:
            if protein != site['protein']:
                if min(b, lb) - max(a, la) >= need:
                    bad.add(number)
                    break

    # ---- общий отрезок не меньше 15 оснований, любая цепь
    keys, owners = [], []
    for number, site in enumerate(sites):
        a, b = int(site['start']), int(site['end'])
        if b - a < KMER:
            continue
        piece = kmer_keys(genome_sequence(genomes_dir, site['genome'])[a:b])
        piece = piece[piece != INVALID]
        if piece.size:
            keys.append(piece)
            owners.append(np.full(piece.size, number, dtype=np.int64))
    if not keys:
        return bad
    keys = np.concatenate(keys)
    owners = np.concatenate(owners)
    order = np.argsort(keys, kind='stable')
    keys, owners = keys[order], owners[order]
    owner_protein = [s['protein'] for s in sites]
    for genome in sorted(prepared):
        items, _, _ = prepared[genome]
        sequence = genome_sequence(genomes_dir, genome)
        covered = sum(b - a for a, b, _ in items)
        whole = kmer_keys(sequence) if covered * 5 > len(sequence) else None
        for la, lb, protein in items:
            if lb - la < KMER:
                continue
            if whole is None:
                segment = kmer_keys(sequence[la:lb])
            else:
                segment = whole[la:min(lb - KMER + 1, whole.size)]
            segment = segment[segment != INVALID]
            if segment.size == 0:
                continue
            lo = np.searchsorted(keys, segment, 'left')
            hi = np.searchsorted(keys, segment, 'right')
            hit = hi > lo
            for start, stop in zip(lo[hit], hi[hit]):
                for number in owners[start:stop]:
                    if owner_protein[int(number)] != protein:
                        bad.add(int(number))
    return bad
