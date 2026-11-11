"""Признаки белка ESM-2 для программы — **всегда через кэш** (SPEC пп. 4.2, 5.7; `CONTRACTS.md` § 2.8).

Пишет задача прототипа T16. Если признаков белка в кэше нет, они считаются тем же путём, что кэш проекта
(`CONTRACTS.md` § 2.8: пакет `fair-esm`, `esm.pretrained.esm2_t33_650M_UR50D()` в режиме `eval`, выход
`repr_layers=[33]`, строки служебных токенов начала и конца отброшены, расчёт в FP32, запись в float16),
записываются в кэш и **дальше читаются из него**: кэш хранит FP16, а посчитанные в памяти — полную точность, и без
этого первый и повторный запуски дали бы разные байты (п. 5.7).

Кэшей может быть несколько: сначала смотрятся кэши только для чтения (например, кэш проекта T08), затем кэш
записи; новые признаки пишутся только в кэш записи — чужой кэш программа не трогает.

**Белковый вход модели из кэша проекта — только `load_protein`** (`CONTRACTS.md` § 4.10, с 04.11.2026): обучение,
проверка остановки, диагностика и применение обученной модели (`train.load_trained`) берут признаки и маску сбора
отсюда, чтобы маска домена (поле `protein_mask` § 2.9) не осталась в одном месте. Других чтений кэша § 2.8 в пакете
нет. (До аудита тестов функция звалась `protein_input`; имя занято логическим аргументом `train` и `step_plan`.)
"""
import hashlib
import os

import numpy as np

#: Модель и слой признаков (`CONTRACTS.md` § 2.8).
ESM_MODEL = 'esm2_t33_650M_UR50D'
ESM_LAYER = 33
ESM_DIM = 1280

#: Виды маски сбора (`CONTRACTS.md` § 2.9, поле `protein_mask`); отсутствующее поле — `whole`.
PROTEIN_MASKS = ('whole', 'dbd')

#: Прочитанные таблицы: путь → (sha256 содержимого, разобранное). Ключ — содержимое, а не время изменения: таблицу
#: могут переписать между двумя обучениями одного процесса (так делает тест 20).
_TABLES = {}


def mask_kind(config):
    """Вид маски сбора из конфигурации (`CONTRACTS.md` § 2.9): нет поля или `null` — `whole`."""
    kind = config.get('protein_mask')
    kind = 'whole' if kind is None else str(kind)
    if kind not in PROTEIN_MASKS:
        raise ValueError('protein_mask — `whole` или `dbd` (CONTRACTS § 2.9), получено %r' % (kind,))
    return kind


def _table(data_root, name, parse):
    import csv
    import io
    path = os.path.abspath(os.path.join(str(data_root), name))
    with open(path, 'rb') as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    found = _TABLES.get(path)
    if found is None or found[0] != digest:
        found = (digest, parse(csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='\t')))
        _TABLES[path] = found
    return found[1]


def _domain_rows(rows):
    """Строки `domains.tsv` (`CONTRACTS.md` § 2.3.1) по белкам: `(env_start, env_end, dbd)`, с нуля, конец не включён —
    перевод из счёта HMMER сделан при записи таблицы, здесь его нет."""
    out = {}
    for row in rows:
        out.setdefault(row['protein'], []).append((int(row['env_start']), int(row['env_end']), row['dbd']))
    return out


def _held_out(rows):
    """Белки части `holdout` разбиения (`CONTRACTS.md` § 2.6, столбцы `part` и `part3`): их домены не искали (§ 2.3.1)."""
    return {row['protein'] for row in rows if 'holdout' in (row.get('part'), row.get('part3'))}


def load_protein(config, protein, device=None):
    """Белковый вход модели (`CONTRACTS.md` § 4.10): признаки белка из кэша § 2.8 (`config['cache_dir']`) —
    float32 `[1, L, 1 280]` — и маска сбора — bool `[1, L]`.

    При `protein_mask` = `whole` (и без поля) маска — все остатки. При `dbd` — объединение отрезков
    `[env_start, env_end)` строк `domains.tsv` этого белка с `dbd` = `yes` (§ 2.3.1, в `config['data_root']`):
    остаток `i` в маске, если `env_start ≤ i < env_end` хоть у одной такой строки; белок без таких строк берётся
    целиком (SPEC п. 4.2: пропуск в разметке возвращает прежний вектор). Признаки от маски не зависят — ESM-2
    читает всю последовательность, маска убирает остатки только из сбора.

    Отказ (`ValueError`): строка этого белка вне `0 ≤ env_start < env_end ≤ L` — ошибка таблицы, а не обрезка; при
    `dbd` — белок части `holdout`: его домены не искали, и «нет строк» у него не значит «нет домена».

    При включённом поле `bacregdb` (§ 2.9) признак ищется в `cache_dir`, затем в `bacregdb.cache` (тот же формат
    § 2.8); белок BacRegDB при маске `dbd` берётся целиком — строк § 2.3.1 у него нет. При выключенном поле второй
    кэш не читается.
    """
    import torch
    path = os.path.join(str(config['cache_dir']), protein + '.npy')
    field = config.get('bacregdb')
    if field is not None and not os.path.exists(path):
        # поле `bacregdb` (§ 2.9, § 4.10): признак ищется в кэше проекта, затем во втором кэше того же формата
        from .data import bacregdb_of
        second = os.path.join(str(bacregdb_of(config)['cache']), protein + '.npy')
        if os.path.exists(second):
            path = second
    array = np.load(path).astype(np.float32)
    if array.ndim != 2 or array.shape[1] != ESM_DIM:
        raise RuntimeError('%s: форма %s, ожидалась (длина, %d) (CONTRACTS § 2.8)' % (path, array.shape, ESM_DIM))
    length = int(array.shape[0])
    features = torch.from_numpy(array).unsqueeze(0)
    if mask_kind(config) == 'whole':
        mask = torch.ones(1, length, dtype=torch.bool)
    else:
        if protein in _table(config['data_root'], 'split.tsv', _held_out):
            raise ValueError('белок %s — из части holdout: его домены не искали (CONTRACTS § 2.3.1, § 4.10)'
                             % protein[:12])
        rows = _table(config['data_root'], 'domains.tsv', _domain_rows).get(protein, ())
        for start, end, _ in rows:
            if not 0 <= start < end <= length:
                raise ValueError('строка домена [%d, %d) белка %s вне его длины %d: ошибка таблицы, а не обрезка '
                                 '(CONTRACTS § 2.3.1, § 4.10)' % (start, end, protein[:12], length))
        spans = [(start, end) for start, end, dbd in rows if dbd == 'yes']
        if spans:
            # дословно § 2.3.1: «код остаток i (с нуля) относит к домену, если env_start ≤ i < env_end»
            residue = np.arange(length)
            keep = np.zeros(length, dtype=bool)
            for start, end in spans:
                keep |= (start <= residue) & (residue < end)
            mask = torch.from_numpy(keep).unsqueeze(0)
        else:
            mask = torch.ones(1, length, dtype=torch.bool)
    if device is not None:
        features, mask = features.to(device), mask.to(device)
    return features, mask


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def esm_weights_path():
    """Файл весов белковой сети в кэше `torch.hub` этого компьютера (так их загружает `fair-esm`)."""
    import torch
    return os.path.join(torch.hub.get_dir(), 'checkpoints', ESM_MODEL + '.pt')


def _compute(sequence):
    """Признаки одного белка путём `CONTRACTS.md` § 2.8: float32 «длина × 1 280»."""
    import esm
    import torch
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    convert = alphabet.get_batch_converter()
    _, _, tokens = convert([('protein', sequence)])
    with torch.no_grad():
        rep = model(tokens, repr_layers=[ESM_LAYER])['representations'][ESM_LAYER]
    out = rep[0, 1:len(sequence) + 1].to(torch.float32).numpy()
    if out.shape != (len(sequence), ESM_DIM):
        raise RuntimeError('признаки ESM-2: форма %s, ожидалась (%d, %d)' % (out.shape, len(sequence), ESM_DIM))
    return out


def protein_features(sequence, key, read_dirs, write_dir):
    """Признаки белка из кэша; нет — посчитать, записать в `write_dir` и прочитать оттуда (п. 5.7).

    Ответ — `(features float32 [L, 1280], info)`; `info` — источник и ключ кэша для описания запуска: `cache_key`
    (sha256 последовательности, § 1), `model`, `layer`, `file`, `file_sha256`, `origin` (`read_cache` — найдены в
    кэше, `computed_now` — посчитаны этим запуском и прочитаны из записанного файла)."""
    name = key + '.npy'
    found = None
    for folder in [d for d in read_dirs if d] + ([write_dir] if write_dir else []):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            found = path
            break
    origin = 'read_cache'
    if found is None:
        if not write_dir:
            raise FileNotFoundError('признаков белка %s нет в кэшах %s, а кэша записи не задано' % (key[:12], read_dirs))
        import torch
        array = _compute(sequence)
        half = array.astype(np.float16)
        if not np.isfinite(half).all():
            raise RuntimeError('признаки ESM-2 вышли за пределы float16: наибольшее |x| = %g' % float(np.abs(array).max()))
        os.makedirs(write_dir, exist_ok=True)
        found = os.path.join(write_dir, name)
        tmp = found + '.tmp'
        with open(tmp, 'wb') as handle:
            np.save(handle, half)
        os.replace(tmp, found)
        origin = 'computed_now'
        threads = torch.get_num_threads()
    else:
        threads = None
    half = np.load(found)
    if half.dtype != np.float16 or half.shape != (len(sequence), ESM_DIM):
        raise RuntimeError('%s: %s %s, ожидался float16 (%d, %d) (CONTRACTS § 2.8)'
                           % (found, half.dtype, half.shape, len(sequence), ESM_DIM))
    info = {'cache_key': key, 'model': ESM_MODEL, 'layer': ESM_LAYER, 'file': found, 'file_sha256': sha256_file(found),
            'origin': origin}
    if threads is not None:
        info['threads'] = threads
    return half.astype(np.float32), info
