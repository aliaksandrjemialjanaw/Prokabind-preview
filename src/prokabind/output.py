"""Выход программы: дорожка оценок, маска, таблицы (SPEC пп. 5.2, 5.7; `CONTRACTS.md` § 4.9).

Пишет задача прототипа T16 по тексту SPEC и контракта. Без PyTorch: модуль только переводит массивы в файлы.

Два правила координат живут здесь, и только здесь:

* **таблицы** (TSV дорожки, маски, списка мест) — **от 1, оба конца включительно**: основание с внутренним
  номером `i` печатается как `i + 1` (SPEC п. 5.2);
* **bigWig** — по своему формату, **с нуля, правый конец не включён**: основание `i` занимает интервал
  `[i, i + 1)`; просмотрщик генома сам показывает его от 1 (уточнение SPEC 29.10).

Оценки печатаются **девятью значащими знаками** (`%.9g`) — столько нужно, чтобы текст восстанавливал float32 без
потерь. Отсутствие оценки — **пустое поле** в TSV и отсутствие интервала в bigWig, а не ноль: ноль в этой шкале —
обычная оценка (п. 5.2).

Сжатые таблицы пишутся с `mtime=0` и без имени файла внутри: иначе gzip кладёт в заголовок время записи, и байты
двух одинаковых запусков расходятся (п. 5.7).
"""
import gzip
import io

import numpy as np

#: Заголовок табличной дорожки (`CONTRACTS.md` § 4.9).
TRACK_HEADER = ('replicon', 'position', 'score')

#: Заголовок маски: отрезки подряд идущих оснований с оценкой и без неё, от 1 и оба конца включительно.
MASK_HEADER = ('replicon', 'start', 'end', 'scored')

#: Уровень сжатия постоянен: от него зависят байты файла (п. 5.7).
GZIP_LEVEL = 6


def fmt9(value):
    """Оценка девятью значащими знаками (п. 5.2); `NaN` — пустое поле.

    Это `%.9g`, но у целого значения остаётся десятичная точка: ноль печатается `0.0`, а не `0` (так требует
    замороженный тест `work/tests/program/test_bigwig.py`: «a score of zero is written, not dropped»). Знаков от этого
    не прибавляется, и float32 восстанавливается так же."""
    v = float(value)
    if v != v:
        return ''
    text = '%.9g' % v
    return text if ('.' in text or 'e' in text or 'n' in text) else text + '.0'


class _Gzip:
    """Сжатый текстовый файл без времени и имени внутри (п. 5.7)."""

    def __init__(self, path):
        self._raw = open(path, 'wb')
        self._gz = gzip.GzipFile(filename='', mode='wb', fileobj=self._raw, mtime=0, compresslevel=GZIP_LEVEL)
        self._text = io.TextIOWrapper(self._gz, encoding='utf-8', newline='\n')

    def write(self, text):
        self._text.write(text)

    def close(self):
        self._text.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_gzip(path):
    """Открыть сжатую таблицу на запись детерминированно (п. 5.7)."""
    return _Gzip(path)


def _check(scores, mask):
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if s.size != m.size:
        raise ValueError('дорожка %d оснований, маска %d' % (s.size, m.size))
    return s, m


def write_track(path, replicon, scores, mask):
    """Табличная дорожка п. 5.2 (`CONTRACTS.md` § 4.9): gzip, заголовок `replicon`, `position`, `score`; строка на
    основание; `position` от 1; `score` — девять значащих знаков (`fmt9`); где `mask` ложна — пустое поле."""
    s, m = _check(scores, mask)
    name = str(replicon)
    chunk = 1 << 16
    with open_gzip(path) as out:
        out.write('\t'.join(TRACK_HEADER) + '\n')
        for lo in range(0, s.size, chunk):
            hi = min(lo + chunk, s.size)
            values = s[lo:hi].tolist()          # float32 → float без изменения значения
            flags = m[lo:hi].tolist()
            out.write(''.join('%s\t%d\t%s\n' % (name, lo + k + 1, fmt9(v) if f else '')
                              for k, (v, f) in enumerate(zip(values, flags))))


def read_track(path):
    """Обратное чтение табличной дорожки (`CONTRACTS.md` § 4.9): те же float32 побайтно, пустое поле — `mask`
    ложна. Файл одного репликона; строки идут подряд с позиции 1."""
    replicon = None
    values, flags = [], []
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        header = handle.readline().rstrip('\n').split('\t')
        if tuple(header) != TRACK_HEADER:
            raise ValueError('%s: заголовок %r, ожидался %r' % (path, header, TRACK_HEADER))
        for number, line in enumerate(handle, start=1):
            line = line.rstrip('\n')
            if not line:
                continue
            name, position, score = line.split('\t')
            if replicon is None:
                replicon = name
            elif name != replicon:
                raise ValueError('%s: второй репликон %r в файле одного репликона %r' % (path, name, replicon))
            if int(position) != len(values) + 1:
                raise ValueError('%s, строка %d: позиция %s, ожидалась %d' % (path, number, position, len(values) + 1))
            if score == '':
                values.append(0.0)
                flags.append(False)
            else:
                values.append(float(score))
                flags.append(True)
    scores = np.asarray(values, dtype=np.float64).astype(np.float32)
    mask = np.asarray(flags, dtype=bool)
    scores[~mask] = np.nan
    return replicon, scores, mask


def _runs(mask):
    """Отрезки подряд идущих оснований одного состояния маски: `(начало, конец, состояние)`, `[начало, конец)`."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size == 0:
        return []
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [m.size]))
    return [(int(a), int(b), bool(m[a])) for a, b in zip(starts, ends)]


def write_mask(path, replicon, mask):
    """Маска пригодности отдельным файлом (п. 5.2) теми же координатами, что таблицы: отрезки подряд идущих оснований
    с оценкой (`scored = yes`) и без неё (`no`), от 1, оба конца включительно."""
    with open_gzip(path) as out:
        out.write('\t'.join(MASK_HEADER) + '\n')
        for a, b, state in _runs(mask):
            out.write('%s\t%d\t%d\t%s\n' % (replicon, a + 1, b, 'yes' if state else 'no'))


def write_bigwig(path, replicon, scores, mask):
    """Та же дорожка в bigWig (SPEC п. 5.2, `CONTRACTS.md` § 4.9): основание `i` — интервал `[i, i + 1)` в координатах
    bigWig, его значение — `scores[i]`; где `mask` ложна, интервала нет; ноль — обычное значение и пишется. Длина
    хромосомы в заголовке — `n`.

    Отрезки подряд идущих оснований с оценкой пишутся записями с шагом 1 и пролётом 1 — это ровно по интервалу на
    основание, только компактнее записанный."""
    import pyBigWig
    s, m = _check(scores, mask)
    bw = pyBigWig.open(str(path), 'w')
    try:
        bw.addHeader([(str(replicon), int(s.size))], maxZooms=10)
        for a, b, state in _runs(m):
            if state:
                bw.addEntries(str(replicon), a, values=s[a:b].astype(np.float64).tolist(), span=1, step=1)
    finally:
        bw.close()


def read_bigwig(path):
    """Обратное чтение bigWig той же программой (`CONTRACTS.md` § 4.9): основание `i` получает значение интервала
    `[i, i + 1)`, без интервала — `mask` ложна. Файл одного репликона."""
    import pyBigWig
    bw = pyBigWig.open(str(path))
    try:
        chroms = bw.chroms()
        if len(chroms) != 1:
            raise ValueError('%s: репликонов %d, ожидался один' % (path, len(chroms)))
        (replicon, n), = chroms.items()
        scores = np.full(int(n), np.nan, dtype=np.float32)
        mask = np.zeros(int(n), dtype=bool)
        for start, end, value in (bw.intervals(replicon) or ()):
            scores[start:end] = np.float32(value)
            mask[start:end] = True
    finally:
        bw.close()
    return replicon, scores, mask


def write_table(path, columns, rows, comments=()):
    """Простая таблица TSV: строки — словари; значение `None` — пустое поле. Сжимается, если имя кончается на `.gz`.
    `comments` — строки пояснения перед заголовком, каждая с `# ` (их печатаются рядом с таблицей, п. 5.2)."""
    text = ''.join('# %s\n' % c for c in comments)
    text += '\t'.join(columns) + '\n'
    for row in rows:
        text += '\t'.join(_cell(row.get(c)) for c in columns) + '\n'
    if str(path).endswith('.gz'):
        with open_gzip(path) as out:
            out.write(text)
    else:
        with open(path, 'w', encoding='utf-8', newline='\n') as out:
            out.write(text)


def _cell(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (float, np.floating)):
        return fmt9(value)
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)
