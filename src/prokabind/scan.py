"""Проход генома: дорожка оценок и маска пригодности (SPEC пп. 4.1, 5.2; `CONTRACTS.md` § 4.2).

Основание получает оценку только тогда, когда все 141 основание его поля видимости известны;
неоднозначные буквы дают **пропуск, а не ноль**. У кольцевого репликона контекст замыкается через
начало координат, у линейного край даёт пропуск.

Проход идёт целиком в FP32, и сокращённая точность матричных операций (TF32) выключается явно:
у библиотеки она включена по умолчанию для свёрток, поэтому слово «FP32» без этого запрета ничего
не гарантирует (SPEC п. 5.2). Этот же запрет и детерминированные ядра проверяет тест 10 п. 4.11.
"""
import os

import numpy as np
import torch

from .model import receptive_field

#: Окно прохода генома (SPEC п. 4.1): «проход генома — окнами 4 096, из которых полезны 3 956».
WINDOW = 4096


def enforce_determinism():
    """Детерминированные ядра включены, сокращённая точность запрещена (SPEC п. 5.2, тест 10 п. 4.11).

    Проход **сам** выставляет флаги, а не полагается на вызывающего: иначе тест может пройти случайно.

    `use_deterministic_algorithms(True)` на видеокарте требует переменной `CUBLAS_WORKSPACE_CONFIG`
    (иначе отказ при первом матричном умножении), поэтому её ставит **сам код**, а не описание
    запуска (`CONTRACTS.md` § 4.5, записано 31.10 по рецензии кода T09). Уже заданное значение не
    перебивается: его мог поставить тот, кто запускает.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def select_device(name=None):
    """Устройство счёта, выбранное **в одном месте** (`CONTRACTS.md` § 4.5): `None` значит
    «видеокарта, если она есть, иначе процессор», имя (`cpu`, `cuda`, `cuda:1`) берётся как есть.

    Выбор живёт здесь, рядом с флагами детерминизма, потому что его спрашивают и обучение, и проход
    генома, и карта вклада: два места выбора разошлись бы молча. Окружение владельца собрано для
    процессора, видеокарта берётся в аренду (§ 4, SPEC пп. 4.9, 6.4).
    """
    if name is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def device_name(unit):
    """Имя устройства для журнала обучения (`CONTRACTS.md` § 3.0.2): `cpu` или имя видеокарты."""
    unit = torch.device(unit)
    if unit.type == 'cuda':
        return torch.cuda.get_device_name(unit)
    return unit.type


def onehot(sequence):
    """ДНК в float32 `[1, 4, L]` в порядке A, C, G, T (`CONTRACTS.md` § 4.1); прочие буквы — нули
    во всех четырёх каналах (неоднозначное основание не подменяется ни одним из четырёх)."""
    code = np.frombuffer(sequence.upper().encode('ascii', 'replace'), dtype=np.uint8)
    out = np.zeros((1, 4, code.size), dtype=np.float32)
    for channel, base in enumerate('ACGT'):
        out[0, channel] = (code == ord(base))
    return torch.from_numpy(out)


def _known(sequence):
    code = np.frombuffer(sequence.upper().encode('ascii', 'replace'), dtype=np.uint8)
    known = np.zeros(code.size, dtype=bool)
    for base in 'ACGT':
        known |= (code == ord(base))
    return known


def valid_mask(sequence, rf, circular):
    """Маска «оценка есть»: у основания всё поле видимости известно (SPEC пп. 4.1, 5.1)."""
    known = _known(sequence)
    n = known.size
    out = np.zeros(n, dtype=bool)
    context = (rf - 1) // 2
    if n < rf:
        return out                                  # репликон короче поля видимости пропускается
    if circular:
        ext = np.concatenate((known[-context:], known, known[:context]))
        count = np.concatenate(([0], np.cumsum(ext)))
        base = np.arange(n)
        out[:] = (count[base + rf] - count[base]) == rf
    else:
        count = np.concatenate(([0], np.cumsum(known)))
        base = np.arange(context, n - context)
        out[base] = (count[base + context + 1] - count[base - context]) == rf
    return out


def score_genome(model, protein, sequence, circular):
    """Дорожка оценок длины генома и маска «оценка есть» (`CONTRACTS.md` § 4.2).

    `protein` — пара «признаки float32 `[1, P, 1280]`, маска bool `[1, P]`» из кэша § 2.8; у модели
    без белкового входа пара тоже передаётся и на ответ не влияет (SPEC п. 4.4).

    Окна берутся с перекрытием в поле видимости, чтобы у каждого основания был полный контекст, и
    каждое основание оценивается **ровно один раз** — из окна, где его поле видимости целиком внутри.
    Поэтому оценка не зависит от нарезки (тест 3 п. 4.11). Основания без оценки — `NaN`: ноль в этой
    шкале — обычная оценка (SPEC п. 5.2).
    """
    enforce_determinism()
    rf = receptive_field(model.config)
    context = (rf - 1) // 2
    sequence = sequence.upper()
    n = len(sequence)
    scores = np.full(n, np.nan, dtype=np.float32)
    mask = valid_mask(sequence, rf, bool(circular))
    if n < rf:
        return scores, mask
    if circular:
        extended, shift, first, last = sequence[-context:] + sequence + sequence[:context], context, 0, n
    else:
        extended, shift, first, last = sequence, 0, context, n - context
    features, protein_mask = protein
    # Окна и признаки белка идут на то устройство, где лежит модель (`CONTRACTS.md` § 4.5): проход
    # не выбирает устройство сам — он исполняет выбор обучения или запуска. На процессоре переносы
    # ничего не меняют, и байты прохода остаются те же (тест 10 п. 4.11).
    unit = next(model.parameters()).device
    features, protein_mask = features.to(unit), protein_mask.to(unit)
    stride = WINDOW - 2 * context
    if stride <= 0:
        raise ValueError('окно прохода %d не вмещает поле видимости %d' % (WINDOW, rf))
    with torch.no_grad():
        for start in range(first, last, stride):
            stop = min(start + stride, last)
            window = extended[start + shift - context:stop + shift + context]
            out = model(onehot(window).to(unit), features, protein_mask)
            scores[start:stop] = (out[0, context:context + (stop - start)]
                                  .detach().to('cpu', torch.float32).numpy())
    scores[~mask] = np.nan
    return scores, mask
