"""Логотип: обрезка, мера сходства, случайный уровень (SPEC п. 5.4; `CONTRACTS.md` § 4.6).

Без PyTorch (§ 4.3): модуль зовёт и код оценки, работающий системным Python.

Логотип — усреднённая по местам матрица «позиция × основание» (п. 5.4). Столбцы — A, C, G, T.
"""
import numpy as np

#: «оставляются столбцы, чья норма не меньше 20% наибольшей нормы столбца, остальные считаются
#: неопределёнными» (SPEC п. 5.4). Порог записан заранее и не подбирается, поэтому он не параметр.
TRIM_SHARE = 0.2

#: Наименьшее перекрытие двух матриц в столбцах с ненулевым разбросом (SPEC п. 5.4).
MIN_OVERLAP = 6


def trim(logo):
    """Обрезка логотипа по норме столбца (SPEC п. 5.4): столбцы с нормой меньше 20% наибольшей нормы
    становятся **неопределёнными** (`NaN`), а не нулевыми.

    Зачем: без обрезки мера ломается от систематического предпочтения сети на краях поля видимости,
    сколь угодно слабого — при постоянном перевесе A и T каждый из примерно 120 крайних столбцов
    становится одним и тем же столбцом (SPEC п. 5.4).
    """
    a = np.array(logo, dtype=float, copy=True)
    if a.size == 0:
        return a
    norms = np.linalg.norm(np.nan_to_num(a, nan=0.0), axis=1)
    out = a.copy()
    out[norms < TRIM_SHARE * float(norms.max())] = np.nan
    return out


def _revcomp(matrix):
    """Обратная цепь матрицы: столбцы A, C, G, T читаются как T, G, C, A в обратном порядке позиций."""
    return np.asarray(matrix, dtype=float)[::-1, ::-1]


def _mean_pearson(a, b, min_overlap):
    """Средний по столбцам коэффициент Пирсона при совмещении без сдвига; столбцы с нулевым разбросом
    и неопределённые пропускаются, перекрытие считается по определённым (SPEC п. 5.4)."""
    good = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not good.any():
        return None
    x = a[good] - a[good].mean(axis=1, keepdims=True)
    y = b[good] - b[good].mean(axis=1, keepdims=True)
    sx, sy = (x * x).sum(axis=1), (y * y).sum(axis=1)
    defined = (sx > 0) & (sy > 0)                  # в столбце из четырёх равных значений Пирсон не определён
    if int(defined.sum()) < min_overlap:
        return None
    r = (x[defined] * y[defined]).sum(axis=1) / np.sqrt(sx[defined] * sy[defined])
    return float(r.mean())


def similarity(a, b, min_overlap=MIN_OVERLAP):
    """Мера сходства двух матриц (SPEC п. 5.4): лучшее по сдвигу и цепи совпадение — средний по
    столбцам коэффициент Пирсона при перекрытии не меньше `min_overlap` столбцов с ненулевым разбросом.

    Нет ни одного совмещения с таким перекрытием — `NaN`: сходство не определено, и выдавать за него
    число нельзя.
    """
    A = np.asarray(a, dtype=float)
    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError('матрица «позиция × основание»: ожидалось [K, 4], получено %s' % (A.shape,))
    best = None
    for cand in (np.asarray(b, dtype=float), _revcomp(b)):
        if cand.ndim != 2 or cand.shape[1] != 4:
            raise ValueError('матрица «позиция × основание»: ожидалось [K, 4], получено %s' % (cand.shape,))
        for shift in range(-(len(cand) - 1), len(A)):
            lo, hi = max(0, shift), min(len(A), shift + len(cand))
            if hi - lo < min_overlap:
                continue
            got = _mean_pearson(A[lo:hi], cand[lo - shift:hi - shift], min_overlap)
            if got is not None and (best is None or got > best):
                best = got
    return float('nan') if best is None else best


def random_level(logo, others):
    """Случайный уровень (SPEC п. 5.4): та же мера между логотипом и матрицами **других** белков
    набора, по одному значению на пару; печатается 95-й перцентиль этого распределения (по n − 1
    значениям, считая от наибольшего вниз — то есть обычная линейная интерполяция на n значениях).

    Сходство со своим мотивом читается против этого уровня.
    """
    vals = [similarity(logo, o) for o in others]
    if not vals or not np.isfinite(vals).any():
        return float('nan')
    return float(np.nanpercentile(np.asarray(vals, dtype=float), 95))
