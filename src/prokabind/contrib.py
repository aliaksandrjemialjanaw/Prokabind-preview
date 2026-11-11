"""Карта вклада оснований пробными заменами (SPEC пп. 1.4, 5.4; `CONTRACTS.md` § 4.4).

Карта — **матрица «позиция × основание»**, а не одно число на позицию: для каждой позиции записываются
оценки всех четырёх оснований за вычетом их среднего. Одномерный профиль с матрицей мотива сравнить
нельзя, а мера сходства п. 5.4 сравнивает матрицы.

Считается пересчётом **только поля видимости** каждой замены (141 основание), а не окна прохода 4 096:
при 423 заменах на место дорогой способ стоил бы больше всего главного прохода генома (п. 5.4).
Берётся **итоговая оценка места — среднее двух проходов** (`model(...)`, п. 4.1), по которому ранжируется
геном: карта объясняет ту оценку, которую видит пользователь, и подписывается в выходе «вклад в оценку места»
(решение владельца 03.11.2026; до того — оценка одного, большего прохода). Цена — второй проход на каждую замену.
Ориентацию для логотипа функция не выбирает: её берёт вызывающий код по большему проходу в точке привязки (п. 5.4).
"""
import numpy as np
import torch

from .model import receptive_field
from .scan import enforce_determinism, onehot

#: Сколько замен считается одним проходом. Размер постоянен: от него зависят байты ответа (тест 10
#: п. 4.11), а память при 141 × 4 заменах на место остаётся малой.
CHUNK = 64


def contribution_map(model, protein, sequence, anchor, half=70, circular=False):
    """Карта вклада вокруг точки привязки `anchor` (`CONTRACTS.md` § 4.4): `float32[2*half+1, 4]`.

    Окно замены — поле видимости точки привязки; у кольцевого репликона оно замыкается через начало
    координат (`circular=True`), у линейного основание ближе поля видимости к краю оценки не имеет
    (SPEC п. 4.1), и карта для него не определена — весь ответ `NaN`, а не выдуманный контекст.
    """
    enforce_determinism()
    rf = receptive_field(model.config)
    context = (rf - 1) // 2
    half = int(half)
    out = np.full((2 * half + 1, 4), np.nan, dtype=np.float32)
    n = len(sequence)
    anchor = int(anchor)
    lo, hi = anchor - context, anchor + context + 1
    if n < rf:
        return out
    if circular:
        index = np.arange(lo, hi) % n
        window = ''.join(sequence[int(i)].upper() for i in index)
    else:
        if lo < 0 or hi > n:
            return out
        window = sequence[lo:hi].upper()

    features, protein_mask = protein
    # то же, что в проходе генома: окно и признаки идут на устройство модели (`CONTRACTS.md` § 4.5)
    unit = next(model.parameters()).device
    features, protein_mask = features.to(unit), protein_mask.to(unit)
    with torch.no_grad():
        plain = onehot(window).to(unit)
        # каждая замена — итоговая оценка места, среднее двух проходов (п. 5.4, тест 16 п. 4.11), пачками
        # постоянного размера: размер постоянен, поэтому ответ повторяем побайтно (тест 10), а память ограничена
        batch, places, values = [], [], []
        for offset in range(-half, half + 1):
            position = context + offset
            for channel in range(4):
                column = plain.clone()
                if 0 <= position < rf:
                    column[0, :, position] = 0.0
                    column[0, channel, position] = 1.0
                batch.append(column)
                places.append((offset + half, channel))
        for start in range(0, len(batch), CHUNK):
            part = torch.cat(batch[start:start + CHUNK], dim=0)
            scores = model(part, features, protein_mask)
            values.append(scores[:, context].detach().to('cpu', torch.float32).numpy())
    for (row, channel), value in zip(places, np.concatenate(values)):
        out[row, channel] = value
    out -= out.mean(axis=1, keepdims=True)
    return out
