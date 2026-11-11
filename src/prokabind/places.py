"""Места: единица списка кандидатов — место, а не максимум (SPEC п. 5.3; `CONTRACTS.md` § 4.3).

Без PyTorch (§ 4.3): модуль зовёт и код оценки, работающий системным Python.

Два правила живут здесь целиком:

* **что считается максимумом** — «основание, чья оценка не ниже оценок обоих соседей, а подряд идущие
  основания с равной оценкой — плато — считаются одним максимумом; его положение — середина плато
  (при чётной длине — меньшая координата)»;
* **как строится место** — «жадно от сильнейшего, а не цепочкой»: максимумы, чьё положение ближе
  `radius` оснований к точке привязки (строго: расстояние `< radius`), причисляются к этому месту
  и выбывают. Склейка по цепочке запрещена прямо (SPEC п. 5.3).
"""
import numpy as np

#: Радиус подавления соседей, взятый как определение единицы (SPEC п. 5.3): записан заранее и не
#: подбирается по результатам.
RADIUS = 20

_EMPTY = (np.zeros(0, np.int64), np.zeros(0, np.float64), np.zeros(0, bool))


class Place:
    """Место списка кандидатов (`CONTRACTS.md` § 4.3).

    Поля: `anchor` — точка привязки, лучшее основание места, с нуля; `score` — её оценка;
    `n_merged` — число **причисленных** к месту максимумов, то есть кроме самой точки привязки
    (SPEC п. 5.3: «число причисленных максимумов печатается»); `edge` — у места сосед «нет данных»
    (край пропуска или край линейного репликона), такое место остаётся в списке с этой пометкой.
    """

    __slots__ = ('anchor', 'score', 'n_merged', 'edge')

    def __init__(self, anchor, score, n_merged, edge):
        self.anchor = int(anchor)
        self.score = float(score)
        self.n_merged = int(n_merged)
        self.edge = bool(edge)

    def __repr__(self):
        return 'Place(anchor=%d, score=%r, n_merged=%d, edge=%r)' % (
            self.anchor, self.score, self.n_merged, self.edge)

    def __eq__(self, other):
        if not isinstance(other, Place):
            return NotImplemented
        return (self.anchor, self.score, self.n_merged, self.edge) == (
            other.anchor, other.score, other.n_merged, other.edge)


def _maxima_padded(vv, mm, n):
    """Ядро правила плато на массиве с двумя служебными основаниями по краям: оценки репликона —
    `vv[1:n+1]`, маска «оценка есть» — `mm[1:n+1]`, а `vv[0]`, `vv[n+1]` и их маски — соседи края
    (пропуск у линейного репликона, основание по кольцу у кольцевого).

    Плато ищется только среди оснований репликона: служебные в него не входят, иначе сосед по кольцу
    стал бы частью плато и его середина съехала бы.
    """
    idx = np.flatnonzero(mm[1:n + 1]) + 1
    if idx.size == 0:
        return _EMPTY
    new = np.empty(idx.size, dtype=bool)
    new[0] = True
    if idx.size > 1:
        new[1:] = (idx[1:] != idx[:-1] + 1) | (vv[idx[1:]] != vv[idx[:-1]])
    at = np.flatnonzero(new)
    starts = idx[at]
    ends = np.empty(at.size, dtype=idx.dtype)
    ends[:-1] = idx[at[1:] - 1]
    ends[-1] = idx[-1]
    left_scored, right_scored = mm[starts - 1], mm[ends + 1]
    # «оценка не ниже оценок обоих соседей»; сосед «нет данных» условия не ставит
    is_max = ((~left_scored | (vv[starts - 1] <= vv[starts])) &
              (~right_scored | (vv[ends + 1] <= vv[ends])))
    pos = starts + (ends - starts) // 2            # середина плато, при чётной длине — меньшая
    keep = np.flatnonzero(is_max)
    return (pos[keep] - 1, vv[starts[keep]], ~(left_scored[keep] & right_scored[keep]))


def _maxima(scores, mask, circular):
    """Максимумы репликона, по возрастанию положения. У кольцевого плато через начало координат —
    одно плато (SPEC п. 5.3)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = s.size
    if n == 0:
        return _EMPTY
    if not circular or n < 2:
        vv = np.empty(n + 2, dtype=np.float64)
        mm = np.zeros(n + 2, dtype=bool)
        vv[1:n + 1], mm[1:n + 1] = s, m
        vv[0] = vv[n + 1] = np.nan                 # края линейного репликона — пропуск
        return _maxima_padded(vv, mm, n)
    # кольцо разрезается там, где разрез не разрубит плато
    joined = m & np.roll(m, 1) & (s == np.roll(s, 1))
    breaks = np.flatnonzero(~joined)
    if breaks.size == 0:
        # всё кольцо — одно плато, оценённое целиком и одной оценкой: ровно одно место
        return np.array([(n - 1) // 2]), np.array([s[0]]), np.array([False])
    r = int(breaks[0])
    sr, mr = np.roll(s, -r), np.roll(m, -r)
    vv = np.empty(n + 2, dtype=np.float64)
    mm = np.empty(n + 2, dtype=bool)
    vv[1:n + 1], mm[1:n + 1] = sr, mr
    vv[0], mm[0] = sr[-1], mr[-1]                  # соседи — по кольцу, а не пропуск
    vv[n + 1], mm[n + 1] = sr[0], mr[0]
    pos, val, edge = _maxima_padded(vv, mm, n)
    pos = (pos + r) % n
    order = np.argsort(pos, kind='stable')         # по возрастанию положения: так ищет build_places
    return pos[order], val[order], edge[order]


def local_maxima(scores, mask):
    """Максимумы с правилом плато (SPEC п. 5.3; `CONTRACTS.md` § 4.3): плато — один максимум в его
    середине. Репликон здесь линейный: кольцевой разбирает `build_places`, которому топология задана."""
    return _maxima(scores, mask, False)[0]


def build_places(scores, mask, circular, radius=RADIUS):
    """Список мест (SPEC п. 5.3; `CONTRACTS.md` § 4.3).

    Жадно от сильнейшего: берётся максимум с наибольшей оценкой (при равных — с меньшей координатой),
    он становится местом, и все максимумы, чьё положение ближе `radius` оснований к его точке привязки
    (расстояние **строго** меньше `radius`: максимум ровно в `radius` основаниях начинает своё место),
    причисляются к этому месту и выбывают; затем то же для сильнейшего из оставшихся.

    Ответ отсортирован по оценке, при равных — по координате (SPEC п. 5.3). Расстояния на кольцевом
    репликоне считаются по кольцу.
    """
    pos, val, edge = _maxima(scores, mask, bool(circular))
    if pos.size == 0:
        return []
    n = int(np.asarray(scores).reshape(-1).size)
    order = np.lexsort((pos, -val))                # по убыванию оценки, при равных — меньшая координата
    taken = np.zeros(pos.size, dtype=bool)
    out = []
    for j in order:
        if taken[j]:
            continue
        anchor = int(pos[j])
        lo, hi = anchor - radius + 1, anchor + radius - 1
        if circular and (lo < 0 or hi >= n):
            near = np.flatnonzero(np.minimum((pos - anchor) % n, (anchor - pos) % n) < radius)
        else:
            near = np.arange(int(np.searchsorted(pos, lo, 'left')), int(np.searchsorted(pos, hi, 'right')))
        merged = sum(1 for i in near if not taken[i] and i != j)
        taken[near] = True
        taken[j] = True
        out.append(Place(anchor, val[j], merged, bool(edge[j])))
    return out
