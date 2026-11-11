"""Код оценки и соперников ProkaBind (T-eval-2, SPEC п. 3.11; сигнатуры — `CONTRACTS.md` § 4.8).

Определения живут здесь (п. 3.11). Те из них, которых в таблице обязательств не видно, дописаны
словами — каждое в строке своей функции, а список и их перенос в текст SPEC — отчёт T10
(`work/tasks/T10/report.md`, SPEC п. 3.10: «Определения, которые п. 3.11 велит дописать словами в
`eval_defs.py`, до депонирования переносятся в текст SPEC»).

**Как он импортируется** (`CONTRACTS.md` § 4.8). Замороженные тесты `work/tests/eval/` и проверки
готовности кладут в путь только `lib` и делают `import eval_defs`; поэтому модуль сам находит
корень проекта от своего файла и добавляет `src` в путь. Работает **системным Python без PyTorch**:
`prokabind.places`, `prokabind.logo` и `prokabind.families` его не требуют (§ 4.3).

**Чего здесь нет.** Склейка максимумов в место (п. 5.3), мера сходства логотипа (п. 5.4) и правило
семейств (п. 3.6) **не повторяются**, а импортируются из `prokabind.places`, `prokabind.logo` и
`prokabind.families` (§ 4.3): одно правило на обучение, программу и оценку. Код из `work/review/`
не взят: тесты T-eval-3 написала сессия спецификации, и общий с ними код сделал бы их проверкой
самих себя (`CONSTITUTION.md`, ст. 2 и 5).

Соглашения счёта — `CONTRACTS.md` § 1: координаты с нуля, `[start, end)`, центр интервала —
`start + (end − start − 1) // 2`, окно ширины `w` с центром `p` — `[p − (w − 1) // 2, … + w)`.
Перебор по множествам идёт по отсортированным ключам: ответ не зависит от порядка словаря
(`CONSTITUTION.md`, ст. 6).
"""
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, 'src'))

from prokabind.families import family_labels, is_other_family       # noqa: E402  (§ 4.3, § 4.8)
from prokabind import logo as _logo                                 # noqa: E402  (§ 4.6)
from prokabind import places as _places                             # noqa: E402  (§ 4.3)

#: Корень проекта: его знают скрипты `work/eval/b*.py`, чтобы не считать путь заново.
ROOT = _ROOT

#: Какие источники раздаются, а какие нет (SPEC п. 3.10) — для `release_tables`.
OPEN_SOURCES = ('prodoric', 'pa_dapseq')
#: У этих источников остаются идентификаторы записей и наши решения, но не координаты (п. 3.10).
ID_ONLY_SOURCES = ('regulondb', 'collectf')
#: Эти не раздаются вовсе: строка выпадает целиком (п. 3.10).
CLOSED_SOURCES = ('streptrn', 'galagan', 'sigmoid')

_RC = str.maketrans('ACGT', 'TGCA')
_BASES = 'ACGT'


def _revcomp(s):
    return s.translate(_RC)[::-1]


def _clean(x, what):
    """Массив float без `NaN`: позиции без оценки выпадают **до** вызова (§ 4.8, «Позиции без оценки»)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size and not np.isfinite(a).all():
        raise ValueError('%s: позиция без оценки (NaN) доехала до меры; она должна выпасть у всех '
                         'сравниваемых одинаково — eval_defs.common_positions (CONTRACTS § 4.8)' % what)
    return a


# ---- мера и общий набор позиций ------------------------------------------------------------------
def auroc_background(pos, bg):
    """AUROC: доля пар «положительный выше фона», ничья — половина (§ 4.8, SPEC п. 3.4).

    Та же мера, что правило остановки обучения (п. 4.8): `prokabind.train.protein_auroc` считает её по
    каждому репликону и усредняет по репликонам белка, а здесь она считается на одном наборе. Проверка
    T10 сверяет два счёта численно на одном входе.

    Вход с `NaN` — `ValueError` (§ 4.8): молчаливый `NaN` встал бы выше или ниже всех оценок. Пустой
    вход — тоже `ValueError`: мерить нечем, а 0,5 здесь выглядело бы как измеренная ничья.
    """
    p = _clean(pos, 'auroc_background(pos)')
    b = _clean(bg, 'auroc_background(bg)')
    if p.size == 0 or b.size == 0:
        raise ValueError('auroc_background: нет положительных (%d) или нет фона (%d)' % (p.size, b.size))
    order = np.sort(b)
    low = np.searchsorted(order, p, 'left')
    high = np.searchsorted(order, p, 'right')
    return float(((low + 0.5 * (high - low)) / order.size).mean())


def common_positions(keys, scores):
    """Общий набор позиций сравнения и его отпечаток (§ 3.3, § 4.8).

    `keys` — ключи позиций `{genome}:{start}-{end}`, `scores` — оценки каждой сравниваемой модели или
    соперника по тем же позициям в том же порядке. Ответ — индексы позиций, оценённых **всеми**
    моделями словаря, в порядке входа, и `kept_sha256` — sha256 строки UTF-8 из **различных** ключей
    этих позиций, отсортированных как строки, каждый с `\\n` на конце; пустой набор — sha256 пустой
    строки.

    Обе величины отдаёт одна функция (правка 29.10 по рецензии готовности, R7): так отпечаток берётся
    из тех позиций, на которых посчитан AUROC, а не из отдельного списка, который мог бы совпасть и при
    разных наборах.
    """
    n = len(keys)
    keep = np.ones(n, dtype=bool)
    for name in sorted(scores):
        a = np.asarray(scores[name], dtype=np.float64).ravel()
        if a.size != n:
            raise ValueError('common_positions: у %r %d оценок на %d позиций' % (name, a.size, n))
        keep &= np.isfinite(a)
    idx = np.nonzero(keep)[0]
    text = ''.join(k + '\n' for k in sorted({keys[i] for i in idx}))
    return idx, hashlib.sha256(text.encode('utf-8')).hexdigest()


# ---- критерий полосы 40–70% (SPEC пп. 1.5, 3.6) ---------------------------------------------------
def _signed_ranks(diffs):
    """Ранги знакового теста: средние ранги при равных модулях, **ноль остаётся в рангах со знаком
    минус** (SPEC п. 1.5: «Разница группы, равная нулю, считается проигрышем»; § 4.8).

    Ответ — (ранги ×2 целыми, положительная сумма ×2, полная сумма ×2). Умножение на два точное:
    средний ранг группы равных модулей кратен ½, и дальше распределение считается целыми.
    """
    d = np.asarray(diffs, dtype=np.float64).ravel()
    if d.size == 0:
        return np.zeros(0, dtype=np.int64), 0, 0
    mod = np.abs(d)
    order = np.argsort(mod, kind='stable')
    ranks2 = np.zeros(d.size, dtype=np.int64)
    i = 0
    while i < d.size:
        j = i
        while j + 1 < d.size and mod[order[j + 1]] == mod[order[i]]:
            j += 1
        # средний ранг группы равных модулей: (i + 1 + j + 1) / 2, взятый ×2
        ranks2[order[i:j + 1]] = (i + 1) + (j + 1)
        i = j + 1
    plus2 = int(ranks2[d > 0].sum())
    return ranks2, plus2, int(ranks2.sum())


def signed_rank_test(diffs):
    """Точный двусторонний тест знаковых рангов по группам (SPEC п. 3.6, § 4.8).

    Ноль — проигрыш: он остаётся в рангах с наименьшим рангом и знаком минус, а не отбрасывается, как
    по умолчанию у `scipy.stats.wilcoxon`; при равных модулях ранги средние. Точное распределение
    считается перечислением сумм рангов (динамическое программирование по целым ранга ×2), а не
    приближением: при 13–28 группах нормальное приближение здесь не годится.

    Крайние случаи: пусто — 1,0 (свидетельства нет); одна группа — 1,0 при выигрыше (2/2 = 1) и 1,0 при
    проигрыше; все разницы нулевые — 1,0.
    """
    ranks2, plus2, total2 = _signed_ranks(diffs)
    if ranks2.size == 0:
        return 1.0
    counts = np.zeros(total2 + 1, dtype=np.float64)
    counts[0] = 1.0
    for r in ranks2:
        counts[r:] += counts[:total2 + 1 - r]
    whole = counts.sum()
    ge = counts[plus2:].sum()
    le = counts[:plus2 + 1].sum()
    return float(min(1.0, 2.0 * min(ge, le) / whole))


def group_bootstrap(diffs, seed, n, se=None):
    """Студентизированный бутстреп по группам, двусторонний 95% интервал (SPEC п. 3.6, § 4.8).

    В каждом повторе `t* = (среднее* − среднее) / (s*/√n)`; `lo` — среднее минус квантиль 0,975
    распределения `t*`, умноженный на `s/√n`, `hi` — то же с квантилем 0,025. При заданной `se`
    (поправленная ошибка `shared_shift_se`) вместо `s/√n` берётся она.

    Вырождения (§ 4.8): все разницы равны (s = 0) — интервал вырождается в точку `lo = hi` = среднее;
    но при заданной положительной `se` точки нет — `t*` не студентизируется ни в одном повторе, а
    внешняя ошибка говорит, что неопределённость не нулевая, поэтому `lo = hi = NaN` и сравнение не
    проходит.

    **Дописано словами (T10): повтор с `s* = 0`.** Если в повторе все вытянутые разницы равны, а сами
    разницы нет, `t*` — предел `(среднее* − среднее) / 0`: **+∞ или −∞ по знаку сдвига** (0, если сдвига
    нет). Такой повтор не выбрасывается: при двух группах половина повторов — одна и та же группа
    дважды, и без них распределение `t*` вырождалось бы в точку 0, интервал — в среднее, и слово полосы
    ≥70% печаталось бы по двум группам с ложной уверенностью. С пределом интервал при двух-трёх группах
    честно бесконечен, и слово — «не решено». **Одна группа** — интервал не определён (`NaN`, `NaN`):
    разброса по одной единице нет вовсе. На настоящих данных (13 и больше групп) повтор с `s* = 0` не
    встречается: его вероятность — n · n⁻ⁿ.
    """
    d = np.asarray(diffs, dtype=np.float64).ravel()
    k = d.size
    if k == 0:
        return float('nan'), float('nan')
    mean = float(d.mean())
    if k == 1:
        return float('nan'), float('nan')
    s = float(d.std(ddof=1))
    width = float(se) if se is not None else s / math.sqrt(k)
    if s == 0.0:
        if se is not None and float(se) > 0:
            return float('nan'), float('nan')
        return mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(int(n), k))
    sample = d[idx]
    m_star = sample.mean(axis=1)
    s_star = sample.std(axis=1, ddof=1)
    shift = m_star - mean
    t_star = np.empty(shift.size, dtype=np.float64)
    good = s_star > 0
    t_star[good] = shift[good] / (s_star[good] / math.sqrt(k))
    t_star[~good] = np.sign(shift[~good]) * np.inf
    t_star[~good & (shift == 0)] = 0.0
    hi_q, lo_q = _quantile(t_star, 0.975), _quantile(t_star, 0.025)
    return float(mean - hi_q * width), float(mean - lo_q * width)


def _quantile(values, q):
    """Квантиль `q` с линейной интерполяцией (как `numpy.quantile` по умолчанию), который не даёт `NaN`
    у бесконечных краёв: между −∞ и конечным значением ответ −∞, между конечным и +∞ — +∞, между −∞ и
    +∞ — ближний из двух. `numpy` в первом случае дал бы `NaN` (−∞ + ∞)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    pos = q * (v.size - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    frac = pos - lo
    a, b = float(v[lo]), float(v[hi])
    if frac == 0 or a == b:
        return a
    if math.isinf(a) and math.isinf(b):
        return a if frac < 0.5 else b
    if math.isinf(a):
        return a
    if math.isinf(b):
        return b
    return a + frac * (b - a)


def shared_shift_se(diffs, ecoli, folds, tau_genome, tau_fold):
    """Ошибка среднего с поправкой на общий сдвиг групп (SPEC п. 3.6, § 4.8).

    **Формула, дописанная словами** (п. 3.11 требует её здесь): к дисперсии среднего `s²/n`
    прибавляются `τ_г² · (доля групп E. coli)²` и `τ_з² · Σ_k (доля групп захода k)²`, и ответ —
    корень суммы. Доля захода `k` — средняя по группам доля этого захода внутри группы: оценка группы
    невзвешенная средняя по её белкам (п. 3.6), поэтому группа из двух белков двух заходов даёт каждому
    по половине. `ecoli[i]` — у группы `i` большинство сайтов в E. coli K-12; `folds[i]` — заходы
    моделей, оценивших белки группы `i`, по элементу на белок.

    τ, равное `None` (не измерено) или не больше 0,01, слагаемого не даёт: п. 3.6 поправляет ошибку,
    «если измеренный общий сдвиг больше 0,01 AUROC». Неизмеренная величина не превращается в ноль
    молча — она называется при результате как ограничение (п. 3.12).
    """
    d = np.asarray(diffs, dtype=np.float64).ravel()
    k = d.size
    if k == 0:
        return float('nan')
    var = (float(np.var(d, ddof=1)) / k) if k > 1 else 0.0
    if tau_genome is not None and float(tau_genome) > 0.01:
        share = float(np.mean(np.asarray(ecoli, dtype=bool))) if len(ecoli) else 0.0
        var += float(tau_genome) ** 2 * share ** 2
    if tau_fold is not None and float(tau_fold) > 0.01:
        weight = {}
        for group in folds:
            members = list(group)
            if not members:
                continue
            for f in members:
                weight[f] = weight.get(f, 0.0) + 1.0 / len(members)
        var += float(tau_fold) ** 2 * sum((w / k) ** 2 for w in (weight[f] for f in sorted(weight)))
    return math.sqrt(var)


def verdict(diffs_by_competitor, seed, shift=None):
    """Слово итога полосы 40–70% (SPEC пп. 1.5, 3.6; § 4.8).

    На входе разницы по группам против каждого обязательного соперника (`motif_transfer`, `blind`) — на
    одних группах в одном порядке. Сравнение проходит при трёх условиях вместе: `signed_rank_test`
    даёт p < 0,05, **положительные ранги перевешивают** (сумма рангов выигравших групп больше половины
    всей суммы — иначе значимость говорит о проигрыше) и нижняя граница `group_bootstrap` выше нуля.
    «Подтверждено» требует **обоих** сравнений; иначе «не подтверждено» с названием непрошедших.

    `k_of_n` — (k, N): k — группы с разницей больше нуля (ноль — проигрыш, п. 1.5).
    `worse_than_motif` — весь интервал против переноса мотива ниже нуля (п. 3.6).
    `shift` — `{'ecoli', 'folds', 'tau_genome', 'tau_fold'}` тех же групп: при нём граница каждого
    сравнения считается с `se = shared_shift_se(…)`, а тест знаковых рангов поправки не получает.
    Число повторов — 10 000 (п. 3.6) и одно и то же `seed` у всех сравнений.
    """
    per, failed = {}, []
    for name in sorted(diffs_by_competitor):
        d = np.asarray(diffs_by_competitor[name], dtype=np.float64).ravel()
        se = None
        if shift is not None:
            se = shared_shift_se(d, shift['ecoli'], shift['folds'], shift.get('tau_genome'), shift.get('tau_fold'))
        p = signed_rank_test(d)
        lo, hi = group_bootstrap(d, seed, 10000, se)
        _, plus2, total2 = _signed_ranks(d)
        wins = int((d > 0).sum())
        per[name] = {'p': p, 'lo': lo, 'hi': hi, 'mean': float(d.mean()) if d.size else float('nan'),
                     'k_of_n': (wins, int(d.size))}
        if not (p < 0.05 and 2 * plus2 > total2 and lo > 0):
            failed.append(name)
    mt = per.get('motif_transfer')
    return {'word': 'подтверждено' if not failed else 'не подтверждено', 'failed': failed,
            'per_competitor': per,
            'worse_than_motif': bool(mt is not None and mt['hi'] < 0)}


def band70_word(diffs, seed, n=10000, tolerance=0.02):
    """Слово полосы ≥70% (SPEC п. 3.6, Б4; § 4.8).

    Интервал — `group_bootstrap(diffs, seed, n)`. «Не хуже допуска» — нижняя граница **не ниже**
    `−tolerance` (граница, равная допуску, ещё «не хуже»); «хуже допуска» — верхняя граница ниже
    `−tolerance`; «не решено» — остальное. Объявлено заранее, что самый вероятный исход здесь —
    «не решено»: при тринадцати группах и равном качестве он выпадает примерно в 80% прогонов.
    """
    d = np.asarray(diffs, dtype=np.float64).ravel()
    lo, hi = group_bootstrap(d, seed, n)
    if lo >= -tolerance:
        word = 'не хуже допуска'
    elif hi < -tolerance:
        word = 'хуже допуска'
    else:
        word = 'не решено'
    return {'word': word, 'lo': lo, 'hi': hi,
            'mean': float(d.mean()) if d.size else float('nan'), 'n_groups': int(d.size)}


def group_scores(per_protein, groups):
    """Оценка группы — **невзвешенная средняя по её белкам** (SPEC п. 3.6).

    **Определение, дописанное словами** (п. 3.11: «чем усредняются белки внутри группы, когда у них
    разное число сайтов»): средняя по белкам, а не по сайтам, и белок с двадцатью сайтами весит ровно
    столько же, сколько белок с пятью. Иначе единицей разброса стала бы не группа родства, а сайт, и
    одна хорошо изученная пара «белок + геном» решала бы за всю группу.

    Белок, которого нет в `groups`, — отказ: молчаливый общий «ящик» сшил бы такие белки в одну
    единицу (то же правило, что у `prokabind.families.family_labels`).
    """
    acc = {}
    for protein in sorted(per_protein):
        if protein not in groups:
            raise ValueError('group_scores: белка %r нет в группах родства (SPEC п. 3.6)' % protein)
        acc.setdefault(groups[protein], []).append(float(per_protein[protein]))
    return {g: float(np.mean(v)) for g, v in sorted(acc.items(), key=lambda kv: str(kv[0]))}


def usable_groups(groups, min_sites=5):
    """Учитываемые группы: группа родства с не меньше чем `min_sites` сайтами (SPEC п. 3.6, порог 5).

    **Единица — пара «полоса + группа»** (определение, дописанное словами, п. 3.11): группа, белки
    которой попали в обе полосы, считается отдельной единицей в каждой из них, поэтому ключ здесь —
    пара, а не одна группа; на наших данных таких групп 6–8 на разбиение. Порог закреплён 22.10 и после
    измерения σ заново не выбирается.
    """
    return sorted((g for g, n in groups.items() if int(n) >= int(min_sites)), key=str)


def leave_family_out(diffs_by_group, families, seed=0, n=10000):
    """Та же оценка без каждого семейства по очереди (SPEC п. 3.6).

    **Определение, дописанное словами** (п. 3.11: «как исключается семейство у групп, охватывающих два
    семейства»): исключение семейства — удаление из набора **всех групп, в которых есть хотя бы один
    белок этого семейства**, целиком, а не разбор группы по белкам: иначе группа теряет порог ≥5 сайтов
    и число единиц зависит от порядка удаления. Семейство здесь — объединение меток Pfam
    (`prokabind.families.family_labels`), а не сырая метка. Значение `families[группа]` — одно семейство
    или набор семейств её белков: группа с белками двух семейств удаляется при исключении **каждого**
    из них. Метка `undetermined` семейства не образует (п. 1.5) и сама не исключается.

    `seed` и число повторов — у границы (п. 3.6, студентизированный бутстреп); скрипт Б3 передаёт
    номер разбиения, как и главному числу. Колонка описательная (п. 3.12).
    """
    member = {}
    for g in diffs_by_group:
        value = families.get(g)
        if isinstance(value, (list, tuple, set, frozenset)):
            member[g] = {str(f) for f in value}
        elif value is not None:
            member[g] = {str(value)}
        else:
            member[g] = set()
    out = {}
    names = sorted({f for fs in member.values() for f in fs if f != 'undetermined'})
    for family in names:
        kept = [float(diffs_by_group[g]) for g in sorted(diffs_by_group, key=str) if family not in member[g]]
        arr = np.asarray(kept, dtype=np.float64)
        lo, hi = group_bootstrap(arr, seed, n)
        out[family] = {'mean': float(arr.mean()) if arr.size else float('nan'),
                       'lo': lo, 'hi': hi, 'n_groups': int(arr.size)}
    return out


def family_share(diffs_by_group, families):
    """Доля семейства X (SPEC пп. 1.5, 3.6; § 4.8): сумма разниц групп семейства X **со знаком**,
    делённая на сумму разниц всех групп той же колонки.

    Значение `families[группа]` — семейство или набор семейств её белков; группа двух объединённых
    семейств входит в обе суммы; метка `undetermined` семейства не образует (п. 1.5). Сумма всех разниц
    не больше нуля — `None`: доля считается только при положительной сумме, то есть только при
    «подтверждено» (п. 3.6). Слово п. 1.5 печатает Б3 рядом со словом итога: «больше трети преимущества
    даёт семейство X» для каждой доли больше ⅓ (называются все), иначе «ни одно семейство не даёт больше
    трети преимущества». Слово решается по доле преимущества, а не по интервалу на подмножестве, и
    называет ровно то, что меряет правило (п. 1.5). Записано 31.10 по рецензии кода T10, Р6.
    """
    total = float(sum(float(diffs_by_group[g]) for g in diffs_by_group))
    if not total > 0:
        return None
    out = {}
    for g in sorted(diffs_by_group, key=str):
        value = families.get(g)
        if isinstance(value, (list, tuple, set, frozenset)):
            names = {str(f) for f in value}
        elif value is None:
            names = set()
        else:
            names = {str(value)}
        for family in sorted(names - {'undetermined', ''}):
            out[family] = out.get(family, 0.0) + float(diffs_by_group[g])
    return {f: v / total for f, v in sorted(out.items())}


def family_share_word(shares, threshold=1.0 / 3.0):
    """Слово оговорки по семействам (SPEC п. 1.5) по долям `family_share`: все семейства с долей больше
    `threshold` называются; пустой ответ `family_share` (`None`) слова не даёт — печатается только при
    «подтверждено»."""
    if shares is None:
        return ''
    named = sorted(f for f, v in shares.items() if v > threshold)
    if named:
        return 'больше трети преимущества даёт семейство ' + ', '.join(named)
    return 'ни одно семейство не даёт больше трети преимущества'


#: Слово оговорки по силе соперника (SPEC п. 1.5), печатается только при «подтверждено».
STRENGTH_CAVEAT_WORD = 'против сильного соперника преимущество меньше, чем против слабого'


def strength_caveat(diffs, seed, n=10000):
    """Оговорка SPEC п. 1.5 по силе соперника (§ 4.8).

    На входе `{группа: {ячейка: разница}}` с ячейками `'1'` и `'>=5'` (`strength_cells`); преимущество
    ячейки — средняя разница её групп; оцениваемая величина — «преимущество в ячейке ≥5 минус
    преимущество в ячейке 1». Интервал — студентизированный бутстреп по группам (п. 3.6): **группа
    берётся целиком со всеми своими ячейками**, `n` повторов, генератор `numpy.random.default_rng(seed)`;
    ошибка разницы в каждом повторе считается по группам целиком — `√(s₁²/n₁ + s₅²/n₅)` по группам
    повтора в каждой ячейке. Ответ `{'diff', 'lo', 'hi', 'caveat'}`: `caveat` истинно, когда весь 95%
    интервал ниже нуля; в одной из двух ячеек меньше двух групп — `lo` и `hi` равны `NaN`, `caveat`
    ложно.

    Вырождения записаны: повтор, где в ячейке меньше двух групп, даёт `s* = 0` и `t* = ±∞` по знаку
    сдвига, как у `group_bootstrap`; повтор, где ячейка пуста вовсе, разницы не имеет и в распределение
    `t*` не входит (при десяти группах в ячейке таких повторов ожидается около 2 · 2⁻²⁰ от `n`).
    Правило печатают Б3 (слово — только при «подтверждено», п. 1.5) и Б7. Записано 31.10 по рецензии
    кода T10, Р6.
    """
    groups = sorted(diffs, key=str)
    a = np.asarray([float(diffs[g]['1']) if '1' in diffs[g] else np.nan for g in groups], dtype=np.float64)
    b = np.asarray([float(diffs[g]['>=5']) if '>=5' in diffs[g] else np.nan for g in groups], dtype=np.float64)
    nan = float('nan')

    def estimate(sa, sb):
        """Разница и её ошибка по строкам матриц повторов (`NaN` — группы нет в ячейке)."""
        na = np.sum(~np.isnan(sa), axis=1)
        nb = np.sum(~np.isnan(sb), axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            ma = np.nansum(sa, axis=1) / np.where(na > 0, na, 1)
            mb = np.nansum(sb, axis=1) / np.where(nb > 0, nb, 1)
            va = np.nansum((sa - ma[:, None]) ** 2, axis=1) / np.where(na > 1, na - 1, 1)
            vb = np.nansum((sb - mb[:, None]) ** 2, axis=1) / np.where(nb > 1, nb - 1, 1)
            se = np.sqrt(np.where(na > 1, va / np.where(na > 0, na, 1), 0.0)
                         + np.where(nb > 1, vb / np.where(nb > 0, nb, 1), 0.0))
        return mb - ma, se, na, nb

    d0, se0, na0, nb0 = estimate(a[None, :], b[None, :])
    if na0[0] < 2 or nb0[0] < 2:
        return {'diff': float(d0[0]) if na0[0] and nb0[0] else nan, 'lo': nan, 'hi': nan, 'caveat': False}
    d0, se0 = float(d0[0]), float(se0[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(int(n), len(groups)))
    d_star, se_star, na, nb = estimate(a[idx], b[idx])
    defined = (na > 0) & (nb > 0)
    shift = d_star[defined] - d0
    se_star = se_star[defined]
    t_star = np.empty(shift.size, dtype=np.float64)
    good = se_star > 0
    t_star[good] = shift[good] / se_star[good]
    t_star[~good] = np.sign(shift[~good]) * np.inf
    t_star[~good & (shift == 0)] = 0.0
    if t_star.size == 0 or se0 == 0.0:
        return {'diff': d0, 'lo': nan, 'hi': nan, 'caveat': False}
    lo = d0 - _quantile(t_star, 0.975) * se0
    hi = d0 - _quantile(t_star, 0.025) * se0
    return {'diff': d0, 'lo': float(lo), 'hi': float(hi), 'caveat': bool(hi < 0)}


# ---- позиции, окна и фон (SPEC пп. 3.4, 3.5, 3.7) -------------------------------------------------
def _window(centre, width):
    """Окно ширины `width` с центром `centre` по `CONTRACTS.md` § 1: `[c − (w − 1) // 2, … + w)`."""
    lo = int(centre) - (int(width) - 1) // 2
    return lo, lo + int(width)


def _free(lo, hi, labels):
    """Окно `[lo, hi)` не задевает ни одной метки, **включая окна, задевающие их краем** (SPEC п. 3.4)."""
    for a, b in labels:
        if lo < int(b) and int(a) < hi:
            return False
    return True


def intergenic_mask(length, genes):
    """`True` вне всех генов; гены — полуоткрытые интервалы (§ 1; SPEC п. 3.4).

    Межгенная ДНК — основания вне записей `gene`, `CDS`, `rRNA`, `tRNA`, `ncRNA`, `tmRNA` (для GFF3 —
    и `pseudogene`); какие записи попали в `genes`, решает читатель аннотации, а не эта функция.
    """
    m = np.ones(int(length), dtype=bool)
    for a, b in genes:
        m[max(0, int(a)):max(0, int(b))] = False
    return m


def background_positions(intergenic, own_labels, width):
    """Позиции фона белка по SPEC п. 3.4 — **центры окон** («Позиция — центр окна»).

    Основание `p` маски `intergenic`, у которого окно ширины `width`, центрированное по § 1, целиком
    лежит в геноме и не перекрывает опытных меток белка и его вариантов ≥90%, включая окна, задевающие
    их краем. «Целиком в геноме» — по линейным координатам: окно через начало кольцевого репликона в фон
    не берётся (меньше `width` позиций на репликон). Маска — межгенные основания для главного фона и
    **основания внутри генов для генного фона** (п. 3.4: «фон своего типа для примеров с центром внутри
    гена — все позиции генома с центром внутри гена»): правило одно, меняется только маска (записано
    31.10 по рецензии кода T10, Р11).

    Крайние случаи: метка вплотную к окну — позиция остаётся (окно кончается там, где метка
    начинается); меток нет — остаются все позиции, чьё окно внутри генома; одна позиция — ответ из
    одного элемента или пустой.
    """
    inter = np.asarray(intergenic, dtype=bool)
    n = inter.size
    w = int(width)
    half = (w - 1) // 2
    ok = inter.copy()
    ok[:half] = False
    if w - 1 - half:
        ok[n - (w - 1 - half):] = False
    for a, b in own_labels:
        a, b = int(a), int(b)
        ok[max(0, a - (w - 1 - half)):min(n, b + half)] = False
    return np.nonzero(ok)[0]


def flank_background(site, half, length, width, own_labels=()):
    """Центры окон флангового фона SPEC п. 3.5 — фланги того же промотора ±`half` оснований.

    Центр — основание фланга `[start − half, start)` или `[end, end + half)`; окно ширины `width`,
    центрированное по § 1, целиком лежит в `[0, length)` и не задевает ни сам сайт, ни метки
    `own_labels` — опытные метки белка и его вариантов ≥90% (пп. 3.4, 3.5: «включая окна, задевающие
    их краем»).
    """
    start, end = int(site[0]), int(site[1])
    labels = [(start, end)] + [(int(a), int(b)) for a, b in own_labels]
    out = []
    for p in list(range(max(0, start - int(half)), start)) + list(range(end, min(int(length), end + int(half)))):
        lo, hi = _window(p, width)
        if lo >= 0 and hi <= int(length) and _free(lo, hi, labels):
            out.append(p)
    return np.asarray(out, dtype=np.int64)


def flank_subwindow(site, neighbour, half, length):
    """Фланговый фон без соседнего сайта того же белка (SPEC п. 3.7).

    У 236 из 575 футпринтов другой записанный сайт того же белка лежит во фланге, у 194 — внутри
    широкого допуска; в таких случаях положение оценивается на подокне без этого соседнего сайта.
    Ответ — основания флангов `[start − half, start)` и `[end, end + half)` внутри `[0, length)`,
    из которых вырезан соседний сайт. Это **фон** (сам сайт в него не входит, как и в
    `flank_background`); ширину окна учитывает `flank_background(..., own_labels=[neighbour])`, а
    область поиска положения модели скрипт Б5 собирает как сайт вместе с этим подокном.
    """
    start, end = int(site[0]), int(site[1])
    na, nb = int(neighbour[0]), int(neighbour[1])
    out = [p for p in (list(range(max(0, start - int(half)), start))
                       + list(range(end, min(int(length), end + int(half)))))
           if not (na <= p < nb)]
    return np.asarray(out, dtype=np.int64)


def other_protein_background(protein, sites, variants, width, own_labels, length):
    """Центры фона «сайты других белков» SPEC п. 3.5, по правилу позиций п. 3.4.

    Позиция — центр (§ 1) сайта белка **другой группы вариантов**, чем у `protein`; окно ширины
    `width`, центрированное по § 1, целиком лежит в `[0, length)` и не задевает `own_labels` —
    опытные метки самого белка и его вариантов ≥90% (точные сайты, сайты PRODORIC, пики StrepTRN).
    Сайты SigmoID из фона не убираются: это находки соперника, их удаление подняло бы его результат
    (п. 3.4).

    Обязывающее число — против **всех** других белков; «против белков другого семейства» —
    диагностическая колонка (п. 3.5), и её скрипт Б2 собирает тем же вызовом по подмножеству белков.
    """
    own_group = variants.get(protein)
    out = set()
    for other in sorted(sites):
        if other == protein or variants.get(other) == own_group:
            continue
        for a, b in sites[other]:
            a, b = int(a), int(b)
            centre = a + (b - a - 1) // 2
            lo, hi = _window(centre, width)
            if lo >= 0 and hi <= int(length) and _free(lo, hi, [(int(x), int(y)) for x, y in own_labels]):
                out.add(centre)
    return np.asarray(sorted(out), dtype=np.int64)


def training_label_mask(length, labels):
    """Маска обучающих меток (SPEC п. 4.5), интервалы полуоткрыты (§ 1).

    Ею строится **обязывающий фон без чужих обучающих меток** (п. 3.4): из межгенного фона убираются
    окна, задевающие любую обучающую метку этого захода — точные сайты, сайты PRODORIC, пики StrepTRN.
    Иначе модель с белковым входом может выигрывать, узнавая «это место чужого белка».
    """
    m = np.zeros(int(length), dtype=bool)
    for a, b in labels:
        m[max(0, int(a)):max(0, int(b))] = True
    return m


def position_type(base, genes):
    """Тип примера по его **центральному основанию** (SPEC п. 3.4): `intergenic` или `genic`.

    Тип определяется одним основанием — центром сайта, вершиной пика, центром участка, — и остаточное
    смешение внутри колонки «межгенные» печатается рядом с колонкой (п. 3.4): медиана межгенного
    промежутка у E. coli 95 оснований, а медиана участка Galagan 384.
    """
    return 'genic' if bool(np.asarray(genes, dtype=bool)[int(base)]) else 'intergenic'


def position_hit(best_base, site, tolerance):
    """Попадание SPEC п. 3.7 при узком и широком допуске.

    **Определение, дописанное словами** (п. 3.11: «что считается позицией модели и попаданием при узком
    допуске»): позиция модели — основание с наибольшей оценкой в области поиска (фланги вместе с самим
    сайтом, п. 3.7; у мотива — центр лучшего окна, п. 3.3.1); при равных оценках берётся меньшая
    координата, как у плато п. 5.3. **Узкий допуск** — позиция внутри записанных границ сайта
    `[start, end)`, концы полуоткрыты (§ 1). **Широкий** — не дальше 40 оснований от **центра**
    записанного сайта (§ 1), а не от его начала.

    Узкий допуск меряет согласие с расстановкой куратора, а не физическую точность: у большинства
    футпринтов границы поставлены по мотиву внутри более длинного футпринта (п. 2.3), и это печатается
    рядом с числом.
    """
    b, (start, end) = int(best_base), (int(site[0]), int(site[1]))
    if tolerance == 'narrow':
        return start <= b < end
    if tolerance == 'wide':
        return abs(b - (start + (end - start - 1) // 2)) <= 40
    raise ValueError('position_hit: допуск %r — только narrow или wide (SPEC п. 3.7)' % (tolerance,))


def label_distance(centre, labels, length, circular):
    """Расстояние от основания `centre` до ближайшего основания метки (SPEC п. 3.1, § 4.8).

    0, если метка его накрывает; у кольцевого репликона — наименьшее из двух направлений, с переходом
    через начало координат; пустой список меток — длина репликона (ближе ничего нет). Топология берётся
    из текста строки `LOCUS` файла GBFF (§ 2.5), а не из догадки читателя.
    """
    c, n = int(centre), int(length)
    best = n
    for a, b in labels:
        a, b = int(a), int(b)
        if a <= c < b:
            return 0
        if circular:
            for edge in (a, b - 1):
                d = abs(c - edge)
                best = min(best, min(d, n - d))
        else:
            best = min(best, a - c if c < a else c - (b - 1))
    return int(best)


def new_dna(centre, labels, radius, cluster_in_training, length, circular):
    """Сайт на **новой ДНК** (SPEC п. 3.1): кластера ДНК нет среди обучающих меток захода **и**
    `label_distance` не меньше `radius`; иначе — на виденной.

    Радиус основной строки — 70 (радиус поля видимости п. 4.1), строгой — 141 (всё поле видимости);
    край принадлежит новой ДНК («расстояние **не меньше** радиуса»), поэтому `new_141 = yes` влечёт
    `new_70 = yes`. Кластеры считает задача разбиения T07 тем же выравнивателем, что родство
    (§ 2.7.1), поэтому здесь они приходят готовым признаком.
    """
    if bool(cluster_in_training):
        return False
    return label_distance(centre, labels, length, circular) >= int(radius)


def level_999(scores, mask):
    """Справочный уровень SPEC п. 5.3 — 99,9-й перцентиль оценок оснований **этого белка в этом
    геноме**, считая только основания с оценкой.

    **Определение, дописанное словами** (п. 3.11): уровень определён для любого белка и любого входа, с
    аннотацией и без, и **не совпадает** с уровнем колонки верха списка п. 3.4 (там окно длины каждого
    известного сайта и фон из межгенных позиций). Полнота при этом уровне считается **той же**
    функцией: прочтение уровня меняет полноту в 1,34 раза (0,506 по основаниям всего генома против
    0,378 по межгенным), поэтому уровень и полнота всегда берутся одним определением (п. 5.6).
    """
    s = np.asarray(scores, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if s.size == 0:
        raise ValueError('level_999: ни одного основания с оценкой (SPEC п. 5.3)')
    return float(np.percentile(s, 99.9))


def type_weighted_auroc(by_type, shares):
    """Сведение AUROC по типам примеров (SPEC пп. 3.4, 3.9): среднее `by_type` с весами `shares`.

    Веса — число примеров каждого типа («Веса усреднения — число примеров каждого типа») или их доли;
    веса делятся на свою сумму. Сравнение задач между собой ведётся по **однотипным** числам, а
    взвешенное среднее печатается как сводка (п. 3.4): состав типов в задачах разный — межгенных
    примеров 88% на точных сайтах, 36% у вершин P. aeruginosa, 31% у участков Galagan.
    """
    keys = sorted(by_type)
    w = np.asarray([float(shares.get(k, 0.0)) for k in keys], dtype=np.float64)
    v = np.asarray([float(by_type[k]) for k in keys], dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError('type_weighted_auroc: сумма весов %r не положительна' % total)
    return float((v * w).sum() / total)


def budget_subsample(n, size, seed):
    """Одна фиксированная общая подвыборка для соперника вне бюджета (SPEC п. 3.4).

    То же семя — те же позиции, и это печатается в таблице рядом с его числом. Остальные соперники
    (ProkaBind, перенос мотива, модель без белкового входа, SigmoID) оцениваются по всему геному,
    поэтому подвыборка касается только того, кого нельзя оценить целиком в пределах бюджета.
    """
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(int(n), size=int(size), replace=False))


def coverage_cell(n_groups, n_sites, min_groups=3, min_sites=15):
    """Оценивается ли ячейка таблицы покрытия семейств Б15 (SPEC п. 3.12).

    Не меньше `min_groups` групп родства **и** не меньше `min_sites` чистых сайтов; иначе в ячейке
    печатается «не оценивается», а не число. Пороги записаны 29.10, до данных, и после них не меняются:
    это честная граница применимости — о семействе в полосе, где ячейка не оценивается, мы не знаем
    ничего.
    """
    return bool(int(n_groups) >= int(min_groups) and int(n_sites) >= int(min_sites))


def split_number(checksums_file):
    """Номер разбиения (SPEC п. 3.10): целое из первых восьми шестнадцатеричных знаков SHA-256 файла
    контрольных сумм зафиксированных таблиц `data/` (`data/SHA256SUMS`, § 2.6).

    Одно вычисление, без повторов: номер получается механически, без выбора.
    """
    with open(checksums_file, 'rb') as fh:
        return int(hashlib.sha256(fh.read()).hexdigest()[:8], 16)


# ---- перенос мотива: соперник п. 3.3.1 ------------------------------------------------------------
def _composition(genome):
    """Состав (нулевого порядка) генома, где идёт поиск: доля каждого основания среди A, C, G, T."""
    up = genome.upper()
    counts = {b: up.count(b) for b in _BASES}
    total = sum(counts.values())
    if total == 0 or min(counts.values()) == 0:
        raise ValueError('build_pwm: в геноме нет всех четырёх оснований (%r) — рецепт п. 3.3.1 '
                         'делит на долю основания в фоне' % counts)
    return {b: counts[b] / total for b in _BASES}


def _canonical(site):
    """Каноническая форма сайта (SPEC п. 3.3.1): меньшая в порядке A < C < G < T из него самого и его
    обратного комплемента."""
    return min(site, _revcomp(site))


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _matches(cand, offset, placed):
    """Совпадения кандидата с **собранным выравниванием**: по всем уже положенным сайтам сумма
    оснований, совпавших с кандидатом в общих столбцах."""
    total = 0
    for off, text in placed:
        lo, hi = max(offset, off), min(offset + len(cand), off + len(text))
        for col in range(lo, hi):
            if cand[col - offset] == text[col - off]:
                total += 1
    return total


def build_pwm(sites, genome):
    """Матрица переноса мотива по рецепту SPEC п. 3.3.1 (§ 4.8).

    Сайты выравниваются с поиском ориентации; ядро — столбцы, покрытые **не меньше чем половиной**
    сайтов; псевдосчёт 0,1 × доля основания в фоне; элемент — **натуральный** логарифм отношения
    частоты к доле основания в составе (нулевого порядка) генома, где идёт поиск. Числами: в столбце
    ядра с `n_b` основаниями `b` из `N = Σ n_b` частота — `(n_b + 0,1 · q_b) / (N + 0,1)`, элемент —
    `ln(частота / q_b)`. Столбцы — A, C, G, T; строки — в ориентации выравнивания. Из одного сайта
    мотив — сам сайт.

    **Выравнивание детерминировано и не зависит ни от цепи, которой подан сайт, ни от порядка сайтов**
    (п. 3.3.1; у RegulonDB столбец цепи — цепь промотора, а не сайта): сайты берутся в порядке
    возрастания sha256 канонической формы, первый — в канонической форме, каждый следующий — той цепью
    и тем сдвигом, при которых совпадений с собранным выравниванием больше; при равенстве — кандидат с
    меньшим sha256 строки, при той же строке — с меньшим **со знаком** сдвигом относительно первого
    сайта (−3 меньше −2, а не наоборот). Параметры записаны 22.10, до любого прогона на настоящих
    данных: от них результат соперника зависит на величину того же порядка, что ожидаемый выигрыш.
    """
    rows = [s.upper() for s in sites]
    if not rows:
        raise ValueError('build_pwm: ни одного сайта (SPEC п. 3.3.1)')
    q = _composition(genome)
    order = sorted(rows, key=lambda s: (_sha(_canonical(s)), _canonical(s)))
    placed = [(0, _canonical(order[0]))]
    for site in order[1:]:
        span_lo = min(off for off, _ in placed)
        span_hi = max(off + len(text) for off, text in placed)
        best_key, best = None, None
        for cand in sorted({site, _revcomp(site)}):
            for offset in range(span_lo - len(cand) + 1, span_hi):
                key = (-_matches(cand, offset, placed), _sha(cand), offset)
                if best_key is None or key < best_key:
                    best_key, best = key, (offset, cand)
        placed.append(best)
    span_lo = min(off for off, _ in placed)
    span_hi = max(off + len(text) for off, text in placed)
    half = len(placed) / 2.0
    matrix = []
    for col in range(span_lo, span_hi):
        bases = [text[col - off] for off, text in placed if off <= col < off + len(text)]
        if len(bases) < half:
            continue
        n = len(bases)
        matrix.append([math.log((bases.count(b) + 0.1 * q[b]) / (n + 0.1) / q[b]) for b in _BASES])
    return np.asarray(matrix, dtype=np.float64)


def motif_transfer(sites, genome):
    """Оценка мотива в каждом основании генома по SPEC п. 3.3.1 (§ 4.8).

    **Лучшее размещение мотива с центром в этом основании**, обе цепи, матрицей `build_pwm`. Окно
    мотива длины `K` с центром в основании `i` — `[i − (K − 1) // 2, … + K)` (центр по § 1); где окно
    выходит за край генома или содержит основание не из A, C, G, T, оценки нет (`NaN`). Так же
    определена и оценка окна («максимум по основаниям»), когда пример короче мотива, — таких сайтов в
    полосе около трети; ничья сайта с окном фона входит в AUROC за ½.
    """
    pwm = build_pwm(sites, genome)
    k = pwm.shape[0]
    n = len(genome)
    out = np.full(n, np.nan, dtype=np.float64)
    if k == 0 or k > n:
        return out
    idx = np.full(n, -1, dtype=np.int64)
    up = np.frombuffer(genome.upper().encode('ascii', 'replace'), dtype=np.uint8)
    for value, base in enumerate(_BASES):
        idx[up == ord(base)] = value
    starts = np.arange(0, n - k + 1)
    forward = np.zeros(starts.size, dtype=np.float64)
    reverse = np.zeros(starts.size, dtype=np.float64)
    good = np.ones(starts.size, dtype=bool)
    for j in range(k):
        col = idx[j:j + starts.size]
        good &= col >= 0
        safe = np.where(col >= 0, col, 0)
        forward += pwm[j][safe]
        reverse += pwm[k - 1 - j][3 - safe]
    best = np.maximum(forward, reverse)
    best[~good] = np.nan
    out[starts + (k - 1) // 2] = best
    return out


# ---- мотив по пикам: MEME (SPEC п. 3.3.1, записано 31.10; CONTRACTS § 4.8) -------------------------
#: Аргументы командной строки MEME — флаги SPEC п. 3.3.1 **ровно в том порядке**; входной файл, папка
#: вывода и `-bfile` с файлом фона подставляются при запуске (§ 4.8): по этой константе проверка видит,
#: с чем MEME запускается. Параметры записаны до любого прогона и меняются только как отклонение (п. 3.10).
MEME_ARGS = ('-dna', '-revcomp', '-mod', 'zoops', '-nmotifs', '1', '-minw', '6', '-maxw', '30',
             '-markov_order', '0', '-nostatus')
#: Технический порог MEME (п. 3.3.1): из меньшего числа пиков мотив не строится вовсе.
MEME_MIN_SEQUENCES = 5
#: Центральные основания пика, которые идут в MEME (п. 3.3.1).
PEAK_CORE = 100
#: Программа по умолчанию — окружение conda `.meme/` в корне проекта (§ 4, «Среда»).
MEME_DEFAULT = '.meme/bin/meme'


def peak_core(start, end, summit, sequence, circular):
    """Центральные 100 оснований пика (SPEC п. 3.3.1; § 4.8).

    Окно `[c − 50, c + 50)`, где `c` — вершина пика (`summit`, с нуля, § 2.4), а у пика без вершины —
    середина интервала по § 1: `start + (end − start − 1) // 2`. У линейного репликона окно, выходящее за
    край, сдвигается внутрь (у начала — `[0, 100)`, у конца — последние 100 оснований); у кольцевого оно
    идёт через начало координат. Репликон короче 100 оснований или центр вне репликона — `ValueError`.
    """
    n = len(sequence)
    if n < PEAK_CORE:
        raise ValueError('peak_core: репликон из %d оснований короче окна %d (SPEC п. 3.3.1)' % (n, PEAK_CORE))
    start, end = int(start), int(end)
    if summit is None or summit == '':
        centre = start + (end - start - 1) // 2
    else:
        centre = int(summit)
    if not 0 <= centre < n:
        raise ValueError('peak_core: центр %d вне репликона из %d оснований' % (centre, n))
    lo = centre - PEAK_CORE // 2
    if circular:
        if lo < 0 or lo + PEAK_CORE > n:
            return sequence[lo % n:] + sequence[:(lo + PEAK_CORE) % n]
        return sequence[lo:lo + PEAK_CORE]
    lo = min(max(lo, 0), n - PEAK_CORE)
    return sequence[lo:lo + PEAK_CORE]


def meme_sites(cores, background, meme=MEME_DEFAULT):
    """Мотив по пикам программой MEME (SPEC п. 3.3.1; § 4.8).

    `cores` — центральные 100 оснований пиков (`peak_core`) **в поданном порядке** (п. 3.3.1: порядок
    `{genome}:{start}` задаёт вызывающий); `background` — доли `A`, `C`, `G`, `T` генома, из которого
    взяты пики, для `-bfile` (состав нулевого порядка). Один процесс MEME с аргументами `MEME_ARGS`;
    из вывода (`meme.xml`) берётся **единственный найденный мотив без отбора по E-значению** — отбор по
    силе мотива был бы выбором соперника по его качеству; E-значение отдаётся рядом. Ответ —
    `{'sites': [строки сайтов мотива, как их выдал MEME — цепью, которую он выбрал], 'evalue', 'width',
    'version'}`; сайты идут в `build_pwm` теми же правилами, что сайты.

    Меньше пяти последовательностей — `ValueError` (технический порог MEME, п. 3.3.1: из одного-двух
    пиков мотив не строится); программы нет — `FileNotFoundError`, а не тихая замена соперника; MEME
    завершился с ошибкой или не выдал мотива — `RuntimeError`. Относительный путь к программе считается от
    корня проекта. Временная папка с входом и выводом MEME удаляется.
    """
    cores = [str(c).upper() for c in cores]
    if len(cores) < MEME_MIN_SEQUENCES:
        raise ValueError('meme_sites: %d последовательностей, а порог MEME — %d (SPEC п. 3.3.1: из одного-двух '
                         'пиков мотив не строится)' % (len(cores), MEME_MIN_SEQUENCES))
    if any(not c for c in cores):
        raise ValueError('meme_sites: пустая последовательность среди пиков')
    program = meme if os.path.isabs(meme) else os.path.join(_ROOT, meme)
    if not os.path.exists(program):
        raise FileNotFoundError('meme_sites: программы %s нет (CONTRACTS § 4: MEME Suite 5.5.9 в .meme/)' % program)
    try:
        freq = {b: float(background[b]) for b in _BASES}
    except (KeyError, TypeError, ValueError):
        raise ValueError('meme_sites: фон должен содержать доли A, C, G, T, дано %r' % (background,))
    total = sum(freq.values())
    if total <= 0 or min(freq.values()) <= 0:
        raise ValueError('meme_sites: доли фона должны быть положительными, дано %r' % (freq,))
    folder = tempfile.mkdtemp(prefix='prokabind_meme_')
    try:
        fasta = os.path.join(folder, 'peaks.fa')
        with open(fasta, 'w') as fh:
            for i, core in enumerate(cores):
                fh.write('>peak_%d\n%s\n' % (i, core))
        bfile = os.path.join(folder, 'background.txt')
        with open(bfile, 'w') as fh:
            fh.write('# order 0\n')
            for b in _BASES:
                fh.write('%s %.8f\n' % (b, freq[b] / total))
        out = os.path.join(folder, 'out')
        command = [program, fasta, '-oc', out, '-bfile', bfile] + list(MEME_ARGS)
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError('meme_sites: MEME завершился с кодом %d: %s'
                               % (run.returncode, (run.stderr or run.stdout).strip()[-400:]))
        root = ET.parse(os.path.join(out, 'meme.xml')).getroot()
        motif = root.find('./motifs/motif')
        if motif is None:
            raise RuntimeError('meme_sites: MEME не выдал ни одного мотива на %d последовательностях' % len(cores))
        sites = [''.join(ref.get('letter_id') for ref in site.findall('letter_ref'))
                 for site in motif.iter('site')]
        return {'sites': sites, 'evalue': float(motif.get('e_value')), 'width': int(motif.get('width')),
                'version': root.get('version', '')}
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def competitor_relative(relatives, band_low):
    """Соперник SPEC п. 3.3.1: ближайший обучающий родственник **вместе с вариантами ≥90%**.

    **Определение, дописанное словами** (п. 3.11: «какой родственник даёт соперника и что делается с
    белком без соперника»): сайты и пики ближайшего обучающего родственника складываются **по его
    группе вариантов** — по п. 2.8 это один белок, и его варианты всегда лежат в той же обучающей
    части. Если у этой группы только пики и их меньше пяти, берётся следующий по родству — ближайший
    белок **другой** группы вариантов, — но **не ниже границы полосы** проверочного белка (`band_low`;
    для полосы 40–70% — не ниже 40%, край полосы внутри неё). Пять пиков — технический порог MEME, а не
    требование к силе: на сайты он не распространяется, мотив строится и из одного сайта.

    **Белок, у которого такого родственника в его полосе нет, в критерий не входит** — ответ `None`, и
    он показывается отдельной строкой: соперник ниже 40% был бы переносом мотива из другой полосы, а
    пустой соперник — даровым выигрышем. В полосе 40–70% таких белков 2–4 на разбиение, учитываемых
    групп теряется 0–1. На слабость соперника отвечает разбивка по силе (`strength_breakdown`), а не
    подмена соперника.

    **Граница полосы проверяется по ближайшему белку группы вариантов, а не по каждой строке** (§ 4.8;
    записано 31.10 по рецензии кода T10, Р4): вариант входит в сумму своей группы при любой своей
    идентичности — п. 3.3.1 своей идентичности варианта не называет вовсе, а вариант по п. 2.8 всегда
    лежит в той же обучающей части, что и его родственник. Группы идут по убыванию идентичности
    ближайшего белка; первая группа ниже границы останавливает перебор — дальше все ниже.
    """
    groups = {}
    for r in relatives:
        key = r['variant_group']
        g = groups.setdefault(key, {'protein': None, 'identity': -1.0, 'n_sites': 0, 'n_peaks': 0,
                                    'variant_group': key})
        g['n_sites'] += int(r.get('n_sites') or 0)
        g['n_peaks'] += int(r.get('n_peaks') or 0)
        if float(r['identity']) > g['identity']:
            g['identity'], g['protein'] = float(r['identity']), r['protein']
    for g in sorted(groups.values(), key=lambda g: (-g['identity'], str(g['protein']))):
        if g['identity'] < float(band_low):
            return None
        if g['n_sites'] > 0 or g['n_peaks'] >= 5:
            return dict(g)
    return None


def competitor_strength(relative):
    """Сила соперника — число меток, **из которых построен его мотив** (SPEC п. 3.3.1).

    Число сайтов, если они есть; число пиков — только если сайтов нет (тогда мотив ищется MEME по
    центральным 100 основаниям пиков). Сайты и пики **не складываются**: мотив из одного сайта — это
    почти сам сайт, и именно поэтому результат печатается разбивкой по силе.
    """
    sites = int(relative.get('n_sites') or 0)
    return sites if sites > 0 else int(relative.get('n_peaks') or 0)


def strength_breakdown(strengths):
    """Ячейки разбивки по силе соперника (SPEC п. 3.3): `1`, `2-4`, `>=5` и внутри неё `>=10`.

    Отдельной ячейки 5–9 нет: в ней ~5 групп, а при пяти группах двусторонняя значимость недостижима
    вообще. В каждой ячейке печатаются только оценка, интервал, число групп и число выигрышей «k из N»
    — без порога значимости и без вывода «прошла или не прошла» (п. 3.3); старшинство разбивки
    относительно главного числа задано в п. 1.5. Ячейка `>=10` — подмножество `>=5`, а не отдельная
    полоса, поэтому суммы ячеек не складываются в число белков.
    """
    s = [int(x) for x in strengths]
    return {'1': sum(1 for x in s if x == 1), '2-4': sum(1 for x in s if 2 <= x <= 4),
            '>=5': sum(1 for x in s if x >= 5), '>=10': sum(1 for x in s if x >= 10)}


#: Ячейки разбивки по силе соперника (SPEC п. 3.3) в порядке печати.
STRENGTH_CELLS = ('1', '2-4', '>=5', '>=10')


def strength_cells(strengths, sites, min_sites=5):
    """В какие ячейки разбивки по силе соперника входит группа родства (SPEC п. 3.3).

    **Определение, дописанное словами** (одно правило на Б3 и Б7; записано 31.10 по рецензии кода T10,
    Р7): «Группа родства, части которой попали в разные ячейки, входит в ячейку той частью, где у неё
    ≥5 сайтов». `strengths` — белок группы → сила его соперника (`competitor_strength`), `sites` —
    белок → число его чистых сайтов в колонке. Часть группы в ячейке — её белки, чья сила лежит в
    ячейке; группа входит в ячейку, если сайтов у этой части не меньше `min_sites`. Ответ —
    `{ячейка: [белки части]}` только для ячеек, куда группа входит; ячейка `>=10` — подмножество `>=5`.
    Разница группы в ячейке — невзвешенная средняя по белкам части (п. 3.6), её считает скрипт.
    """
    out = {}
    for cell in STRENGTH_CELLS:
        part = sorted(p for p in strengths if strength_breakdown([strengths[p]])[cell])
        if part and sum(int(sites.get(p, 0)) for p in part) >= int(min_sites):
            out[cell] = part
    return out


def sigmoid_competitor(profiles, band_low):
    """Соперник SigmoID по SPEC п. 3.3.4 (решение владельца 25.10).

    Профиль допускается, только если его белок-образец — вариант обучающего белка этого захода
    (идентичность ≥90% при покрытии ≥80%) и **не** проверочный белок и не его вариант ≥90%. Из
    допустимых — ближайший к проверочному белку, **не ниже границы полосы** `band_low`; нет такого —
    `None`, и соперник «неприменим», а не выигрыш. При широком прочтении («любой образец, кроме самого
    белка») у 26–36 белков полосы 40–70% образец оказался бы родственен проверочному белку на ≥70% —
    соперник получил бы более близкого родственника, чем допускает сама полоса.
    """
    allowed = [p for p in profiles
               if bool(p.get('sample_in_training')) and not bool(p.get('sample_is_test_variant'))
               and float(p['identity']) >= float(band_low)]
    if not allowed:
        return None
    return dict(sorted(allowed, key=lambda p: (-float(p['identity']), str(p['profile'])))[0])


# ---- скореры без белкового входа (SPEC п. 3.3.5) --------------------------------------------------
def at_fraction(seq):
    """Доля A и T среди оснований A, C, G, T строки (SPEC п. 3.3.5); прочие буквы не считаются.

    Строка без A, C, G, T — `NaN`. Это слабейший из трёх скореров и печатается строкой в Б1, Б6 и Б8:
    в `prokabind-auto` он дал 0,53, а `markov5` — 0,689 при 0,794 у переноса мотива, и без такой
    строки модель с 0,72 выглядела бы результатом.
    """
    up = seq.upper()
    at = up.count('A') + up.count('T')
    total = at + up.count('C') + up.count('G')
    return float('nan') if total == 0 else at / total


def _kmer_features():
    """44 признака SPEC п. 3.3.5: k-меры k = 1…3, где k-мер и его обратный комплемент — один признак."""
    import itertools
    feats = []
    for k in (1, 2, 3):
        for parts in itertools.product(_BASES, repeat=k):
            canon = _canonical(''.join(parts))
            if canon not in feats:
                feats.append(canon)
    return feats


_KMERS = _kmer_features()
_KMER_INDEX = {t: i for i, t in enumerate(_KMERS)}


def _kmer_vector(seq):
    """Частоты k-меров окна: счёт, делённый на число k-меров этой длины в окне (из одних A, C, G, T;
    при нуле таких — 0). Частоты, а не счёта, — чтобы окна разной длины были сравнимы (п. 3.3.5)."""
    up = seq.upper()
    v = np.zeros(len(_KMERS), dtype=np.float64)
    for k in (1, 2, 3):
        words = [up[i:i + k] for i in range(len(up) - k + 1)]
        words = [w for w in words if all(c in _BASES for c in w)]
        if not words:
            continue
        share = 1.0 / len(words)
        for w in words:
            v[_KMER_INDEX[_canonical(w)]] += share
    return v


# ---- проход скорера по всему геному (SPEC п. 3.4: «Фон — все межгенные позиции генома») -----------
def _encode(sequence):
    """Коды оснований строки: A, C, G, T → 0…3, любая другая буква → 4 («не основание»)."""
    raw = np.frombuffer(sequence.upper().encode('ascii', 'replace'), dtype=np.uint8)
    code = np.full(raw.size, 4, dtype=np.int64)
    for value, base in enumerate(_BASES):
        code[raw == ord(base)] = value
    return code


def _code_of(word):
    """Код строки из A, C, G, T: первая буква — старший разряд."""
    out = 0
    for ch in word:
        out = out * 4 + _BASES.index(ch)
    return out


def _window_sum(values, m):
    """`out[s] = Σ_{j<m} values[s + j]` для каждого начала `s` (длина `n − m + 1`).

    Прямое суммирование `m` слагаемых (`numpy.convolve`), а не разность накопленных сумм: у накопленной
    суммы по миллионам оснований ошибка округления доходит до 10⁻⁸ на окно, у прямой — 10⁻¹⁴, и оценка
    окна совпадает с вызовом скорера на его строке до погрешности сложения.
    """
    v = np.asarray(values, dtype=np.float64)
    m = int(m)
    if m <= 0 or m > v.size:
        return np.zeros(max(0, v.size - m + 1), dtype=np.float64)
    if m == 1:
        return v.copy()
    return np.convolve(v, np.ones(m, dtype=np.float64), mode='valid')


def _kmer_codes(code, k):
    """Код k-мера, начинающегося в каждом основании (длина `n − k + 1`), и его годность — все `k` букв
    из A, C, G, T."""
    n = code.size - k + 1
    if n <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    out = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for j in range(k):
        piece = code[j:j + n]
        valid &= piece < 4
        out = out * 4 + np.where(piece < 4, piece, 0)
    return out, valid


def _feature_tables():
    """Для каждой длины k = 1…3 — таблица «код k-мера (4ᵏ значений) → номер канонического признака»."""
    tables = {}
    for k in (1, 2, 3):
        table = np.zeros(4 ** k, dtype=np.int64)
        for code in range(4 ** k):
            word = ''.join(_BASES[(code >> (2 * (k - 1 - j))) & 3] for j in range(k))
            table[code] = _KMER_INDEX[_canonical(word)]
        tables[k] = table
    return tables


_FEATURE_OF = _feature_tables()


class _WindowScorer(object):
    """Скорер без белкового входа (SPEC п. 3.3.5): вызов — оценка одного окна по определению § 4.8;
    `track(sequence, width, key)` — те же оценки для **всех** окон ширины `width` строки `sequence` одним
    проходом: `out[s]` — оценка окна `[s, s + width)`.

    **Зачем проход** (п. 3.4: «Фон — все межгенные позиции генома, а не подвыборка»; записано 31.10 при
    втором заходе T10, п. 12): скореры оцениваются по всему фону, как ProkaBind, а по окну в цикле
    Python полмиллиона межгенных и четыре миллиона генных окон E. coli на каждую ширину не посчитать.
    Формула та же, что при вызове, — отличается только порядок суммирования; **окно с буквой вне A, C,
    G, T считается вызовом по окну**, поэтому ответ прохода совпадает с вызовом на любом окне (таких окон
    мало: буквы вне A, C, G, T есть в 18 геномах из 168, от 1 до 200 на геном). Прогон T10 сверяет оба пути
    на одних окнах, с буквами N и без, и измеряет цену прохода на настоящем геноме. Промежуточные массивы
    одного репликона держатся в памяти, пока не придёт другой; репликон узнаётся по sha256 строки, а не по
    имени (`key` оставлен для совместимости и в ключ не входит).
    """

    def __init__(self):
        self._base_key, self._base = None, None

    def __call__(self, seq):
        raise NotImplementedError

    def track(self, sequence, width, key=None):
        raise NotImplementedError

    def _cached(self, key, sequence, build):
        digest = hashlib.sha256(sequence.encode('ascii', 'replace')).hexdigest()
        if self._base_key != digest:
            self._base_key, self._base = digest, build(sequence)
        return self._base

    def _by_window(self, out, base_code, sequence, width):
        """Окна с буквой вне A, C, G, T — вызовом по окну: ответ прохода равен вызову и там."""
        w = int(width)
        count = out.size
        odd = np.nonzero(_window_sum((base_code == 4).astype(np.float64), w)[:count] > 0)[0]
        for s in odd:
            out[s] = self(sequence[s:s + w])
        return out


class _KmerScorer(_WindowScorer):
    """Логистическая регрессия на частотах k-меров (`kmer_lr`): оценка окна — линейный предиктор."""

    def __init__(self, weights, mu, scale, keep):
        _WindowScorer.__init__(self)
        self.weights, self.mu, self.scale, self.keep = weights, mu, scale, keep
        # предиктор = w₀ + Σ_f w_f · keep_f · (x_f − μ_f) / σ_f = const + Σ_f coef_f · x_f
        self.coef = np.where(keep, weights[1:] / scale, 0.0)
        self.constant = float(weights[0] - np.sum(np.where(keep, weights[1:] * mu / scale, 0.0)))

    def _design(self, matrix):
        z = np.where(self.keep, (matrix - self.mu) / self.scale, 0.0)
        return np.hstack([np.ones((z.shape[0], 1)), z])

    def __call__(self, seq):
        return float((self._design(_kmer_vector(seq)[None, :]) @ self.weights)[0])

    def _build(self, sequence):
        code = _encode(sequence)
        out = {'code': code}
        for k in (1, 2, 3):
            codes, valid = _kmer_codes(code, k)
            weighted = np.where(valid, self.coef[_FEATURE_OF[k][codes]], 0.0)
            out[k] = (weighted, valid.astype(np.float32))
        return out

    def track(self, sequence, width, key=None):
        n, w = len(sequence), int(width)
        count = n - w + 1
        if w <= 0 or count <= 0:
            return np.zeros(max(0, count), dtype=np.float64)
        base = self._cached(key, sequence, self._build)
        out = np.full(count, self.constant, dtype=np.float64)
        for k in (1, 2, 3):
            m = w - k + 1                       # k-меров в окне ширины w
            if m <= 0:
                continue
            weighted, valid = base[k]
            total = _window_sum(weighted, m)[:count]
            words = _window_sum(valid, m)[:count]
            out += np.where(words > 0, total / np.where(words > 0, words, 1.0), 0.0)
        return self._by_window(out, base['code'], sequence, w)


def kmer_lr(pos, bg, lam=1.0):
    """Скорер без белкового входа SPEC п. 3.3.5: логистическая регрессия на частотах k-меров k = 1…3.

    Признаки — 44 (k-мер и его обратный комплемент — один), стандартизованные средним и стандартным
    отклонением обучающих окон (признак с нулевым отклонением — 0); штраф `½ · lam · ‖w‖²` к **сумме**
    логарифма правдоподобия, свободный член без штрафа. Обучается только на обучающей части захода
    (для Б8 — итоговой модели). Оценка окна — линейный предиктор; она не зависит от того, какой цепью
    подано окно, потому что канонический признак у окна и его обратного комплемента один.

    Решается методом Ньютона: матрица вторых производных с штрафом положительно определена по весам,
    поэтому оптимум один, и ответ не зависит от начальной точки. Ответ — вызываемый объект; его `track`
    даёт те же оценки для всех окон генома одним проходом (`_WindowScorer`, п. 3.4).
    """
    x = np.asarray([_kmer_vector(s) for s in list(pos) + list(bg)], dtype=np.float64)
    y = np.asarray([1.0] * len(list(pos)) + [0.0] * len(list(bg)), dtype=np.float64)
    if x.shape[0] == 0:
        raise ValueError('kmer_lr: ни одного обучающего окна (SPEC п. 3.3.5)')
    mu, sd = x.mean(axis=0), x.std(axis=0)
    keep = sd > 0
    scale = np.where(keep, sd, 1.0)

    def design(matrix):
        z = np.where(keep, (matrix - mu) / scale, 0.0)
        return np.hstack([np.ones((z.shape[0], 1)), z])

    a = design(x)
    penalty = np.eye(a.shape[1]) * float(lam)
    penalty[0, 0] = 0.0
    w = np.zeros(a.shape[1], dtype=np.float64)
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(a @ w)))
        grad = a.T @ (y - p) - penalty @ w
        hess = a.T @ (a * (p * (1.0 - p))[:, None]) + penalty
        step = np.linalg.solve(hess, grad)
        w = w + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return _KmerScorer(w, mu, scale, keep)


class _MarkovScorer(_WindowScorer):
    """Две интерполированные марковские модели (`markov5`): оценка окна — средняя по цепям сумма
    `ln P_pos − ln P_bg`."""

    def __init__(self, pos_counts, bg_counts, order, alpha):
        _WindowScorer.__init__(self)
        self.pos, self.bg = pos_counts, bg_counts
        self.order, self.alpha = int(order), float(alpha)
        self._tables = None

    def _logp(self, text, counts):
        n, total = counts
        out = 0.0
        for i, base in enumerate(text):
            p = 0.25
            for k in range(min(self.order, i) + 1):
                ctx = text[i - k:i]
                p = (n.get((ctx, base), 0) + self.alpha * p) / (total.get(ctx, 0) + self.alpha)
            out += math.log(p)
        return out

    def __call__(self, seq):
        text = seq.upper()
        values = [self._logp(s, self.pos) - self._logp(s, self.bg) for s in (text, _revcomp(text))]
        return float(sum(values) / 2.0)

    def _arrays(self):
        """Счёты по кодам контекстов для прохода: `n_k[ctx, b]` и `N_k[ctx]` для k = 0…order. Ключи с
        буквами вне A, C, G, T (возможны только в обучающих окнах с такими буквами) в таблицы не входят:
        окна генома с такими буквами проход считает вызовом по окну (`_by_window`), где эти ключи есть."""
        if self._tables is None:
            tables = {}
            for name, (n, total) in (('pos', self.pos), ('bg', self.bg)):
                per_k = []
                for k in range(self.order + 1):
                    nk = np.zeros((4 ** k, 4), dtype=np.float64)
                    totals = np.zeros(4 ** k, dtype=np.float64)
                    for ctx, count in total.items():
                        if len(ctx) == k and all(ch in _BASES for ch in ctx):
                            totals[_code_of(ctx)] = count
                    for (ctx, base), count in n.items():
                        if len(ctx) == k and base in _BASES and all(ch in _BASES for ch in ctx):
                            nk[_code_of(ctx), _BASES.index(base)] = count
                    per_k.append((nk, totals))
                tables[name] = per_k
            self._tables = tables
        return self._tables

    def _strand(self, code, base, reverse):
        """По одной цепи: `head[s] = Σ_{j<order} D_j[s + j]` (прямая) или `head[e] = Σ_{m<order} D_m[e − m]`
        (обратная, по концу окна `e`) и `tail = D_order` — разность `ln P_pos − ln P_bg` при полном
        контексте. Контексты строятся по порядкам от 0 до `order` без хранения всех шести массивов."""
        n = code.size
        tables = self._arrays()
        base_ok = base < 4
        b = np.where(base_ok, base, 0)
        p = {'pos': np.full(n, 0.25), 'bg': np.full(n, 0.25)}
        ctx = np.zeros(n, dtype=np.int64)
        valid = np.ones(n, dtype=bool)
        head = np.zeros(n, dtype=np.float64)
        tail = None
        for k in range(self.order + 1):
            if k > 0:
                shifted = np.full(n, 4, dtype=np.int64)
                if k < n:
                    if reverse:
                        following = code[k:]
                        shifted[:n - k] = np.where(following < 4, 3 - following, 4)
                    else:
                        shifted[k:] = code[:n - k]
                good = shifted < 4
                valid &= good
                ctx = ctx + np.where(good, shifted, 0) * 4 ** (k - 1)
            for name in ('pos', 'bg'):
                nk, totals = tables[name][k]
                counts = np.where(valid & base_ok, nk[ctx, b], 0.0)
                total = np.where(valid, totals[ctx], 0.0)
                p[name] = (counts + self.alpha * p[name]) / (total + self.alpha)
            d = np.log(p['pos']) - np.log(p['bg'])
            if k == self.order:
                tail = d
            elif reverse:
                head[k:] += d[:n - k]                 # head[e] += D_k[e − k]
            else:
                head[:n - k] += d[k:]                 # head[s] += D_k[s + k]
        return head, tail

    def _build(self, sequence):
        code = _encode(sequence)
        comp = np.where(code < 4, 3 - code, 4)
        return {'code': code, 'forward': self._strand(code, code, reverse=False),
                'reverse': self._strand(code, comp, reverse=True)}

    def track(self, sequence, width, key=None):
        n, w = len(sequence), int(width)
        count = n - w + 1
        if w <= 0 or count <= 0:
            return np.zeros(max(0, count), dtype=np.float64)
        if w <= self.order:
            # окно короче полного контекста: по окну, таких ширин в данных нет (сайт не короче 8 оснований)
            return np.asarray([self(sequence[s:s + w]) for s in range(count)], dtype=np.float64)
        base = self._cached(key, sequence, self._build)
        head_f, tail_f = base['forward']
        head_r, tail_r = base['reverse']
        h = self.order
        total = head_f[:count] + head_r[w - 1:w - 1 + count]
        tail_sum = _window_sum(tail_f, w - h)
        total += tail_sum[h:h + count]
        tail_sum = _window_sum(tail_r, w - h)
        total += tail_sum[:count]
        return self._by_window(total / 2.0, base['code'], sequence, w)


def markov5(pos, bg, order=5, alpha=10.0):
    """Скорер без белкового входа SPEC п. 3.3.5: две интерполированные марковские модели порядка
    `order` — по `pos` и по `bg`, — обе по **обеим цепям** каждого окна.

    Вероятность основания после контекста длины k — `P_k = (n_k + alpha · P_(k−1)) / (N_k + alpha)`, где
    `n_k` — счёт пары «контекст + основание», `N_k` — счёт контекста, `P_(−1) = ¼`; первые основания
    окна берут столько контекста, сколько есть. Оценка окна — средняя по двум цепям сумма
    `ln P_pos − ln P_bg`. Значения по умолчанию — порядок 5 и α = 10 (п. 3.3.5). Ответ — вызываемый
    объект; его `track` даёт те же оценки для всех окон генома одним проходом (`_WindowScorer`, п. 3.4).
    """
    order, alpha = int(order), float(alpha)

    def counts(windows):
        n, total = {}, {}
        for window in windows:
            text = window.upper()
            for strand in (text, _revcomp(text)):
                for i, base in enumerate(strand):
                    for k in range(min(order, i) + 1):
                        ctx = strand[i - k:i]
                        n[(ctx, base)] = n.get((ctx, base), 0) + 1
                        total[ctx] = total.get(ctx, 0) + 1
        return n, total

    return _MarkovScorer(counts(pos), counts(bg), order, alpha)


def _at_fraction_track(sequence, width):
    """`at_fraction` для всех окон ширины `width` одним проходом: доля A и T среди A, C, G, T окна,
    `NaN` там, где таких букв нет."""
    code = _encode(sequence)
    w = int(width)
    count = code.size - w + 1
    if w <= 0 or count <= 0:
        return np.zeros(max(0, count), dtype=np.float64)
    at = _window_sum(((code == 0) | (code == 3)).astype(np.float64), w)
    acgt = _window_sum((code < 4).astype(np.float64), w)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(acgt > 0, at / np.where(acgt > 0, acgt, 1.0), np.nan)


def scorer_track(scorer, sequence, width, key=None):
    """Оценки скорера без белкового входа для всех окон ширины `width` строки (п. 3.3.5, п. 3.4):
    `out[s]` — оценка окна `[s, s + width)`, длина `len(sequence) − width + 1`.

    `at_fraction` считается своим проходом, `kmer_lr` и `markov5` — своим `track`; любой другой скорер —
    по окну в цикле. Так скрипты Б1 и Б8 оценивают скореры по **всему** фону, как ProkaBind (п. 3.4), а
    не на подвыборке. Ответ на каждом окне равен вызову скорера на его строке (окна с буквами вне A, C,
    G, T считаются вызовом); `key` оставлен для совместимости — кэш узнаёт репликон по содержимому.
    """
    if isinstance(scorer, _WindowScorer):
        return scorer.track(sequence, width, key)
    if scorer is at_fraction:
        return _at_fraction_track(sequence, width)
    w = int(width)
    return np.asarray([scorer(sequence[s:s + w]) for s in range(len(sequence) - w + 1)], dtype=np.float64)


# ---- Б6: свои пики против чужих (SPEC п. 3.8) -----------------------------------------------------
def own_vs_other_peaks(own, other, w_own=None, w_other=None):
    """Оценка белка в Б6 (SPEC п. 3.8): AUROC «свои пики против пиков фона» при одном белковом входе —
    своём или чужом; ничья — половина.

    Оценка пика — максимум по всему пику (п. 3.8: вершина — лишь подсказка, границ пиков в части данных
    нет); здесь она приходит готовой. Веса — кратности пиков в повторе блочного бутстрепа (`None` — по
    единице): пара «свой, чужой» входит с весом `w_own · w_other`, поэтому пик, вытянутый трижды, и
    весит втрое.
    """
    a = _clean(own, 'own_vs_other_peaks(own)')
    b = _clean(other, 'own_vs_other_peaks(other)')
    wa = np.ones(a.size) if w_own is None else np.asarray(w_own, dtype=np.float64).ravel()
    wb = np.ones(b.size) if w_other is None else np.asarray(w_other, dtype=np.float64).ravel()
    total = wa.sum() * wb.sum()
    if a.size == 0 or b.size == 0 or total <= 0:
        raise ValueError('own_vs_other_peaks: нет своих пиков или нет фона (SPEC п. 3.8)')
    order = np.argsort(b, kind='stable')
    bs, wbs = b[order], wb[order]
    cum = np.concatenate([[0.0], np.cumsum(wbs)])
    low = cum[np.searchsorted(bs, a, 'left')]
    high = cum[np.searchsorted(bs, a, 'right')]
    return float((wa * (low + 0.5 * (high - low))).sum() / total)


def b6_share(rows, scores, background, groups, weights=None, min_peaks=10):
    """Доля выигравших групп Б6 (SPEC п. 3.8).

    `rows[i]` — пик `{'protein', …}`; `scores[X]` — `{'own': …, 'foreign': [… пять …]}` — оценки всех
    пиков со своим белком X и с каждым из пяти чужих; `background[X]` — номера строк фона X: пики
    других белков, не перекрывающие его пиков, и **только тех белков, на которых модель этого захода не
    обучалась** (иначе 69–73% чужих пиков были бы обучающими положительными примерами, и выигрыш мог бы
    означать память о них). `groups[X]` — группа родства.

    Разница белка — `own_vs_other_peaks` со своим белком **минус средняя по пяти чужим**; группа
    выигрывает, если средняя по её белкам разница больше нуля (**ноль — проигрыш**, против ProkaBind);
    доля — выигравшие группы среди групп, где остался хоть один белок. Белок, у которого сумма
    кратностей своих пиков меньше `min_peaks` (п. 3.8 — 10), в повтор не входит.
    """
    w = np.ones(len(rows), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    protein_rows = {}
    for i, row in enumerate(rows):
        protein_rows.setdefault(row['protein'], []).append(i)
    by_group = {}
    for protein in sorted(scores):
        own_idx = np.asarray(protein_rows.get(protein, ()), dtype=np.int64)
        bg_idx = np.asarray(sorted(background.get(protein, ())), dtype=np.int64)
        if own_idx.size == 0 or bg_idx.size == 0:
            continue
        if w[own_idx].sum() < float(min_peaks) or w[bg_idx].sum() <= 0:
            continue
        own_track = np.asarray(scores[protein]['own'], dtype=np.float64)
        mine = own_vs_other_peaks(own_track[own_idx], own_track[bg_idx],
                                  w_own=w[own_idx], w_other=w[bg_idx])
        foreign = [own_vs_other_peaks(np.asarray(t, dtype=np.float64)[own_idx],
                                      np.asarray(t, dtype=np.float64)[bg_idx],
                                      w_own=w[own_idx], w_other=w[bg_idx])
                   for t in scores[protein]['foreign']]
        by_group.setdefault(groups[protein], []).append(mine - float(np.mean(foreign)))
    if not by_group:
        return float('nan')
    wins = sum(1 for diffs in by_group.values() if float(np.mean(diffs)) > 0)
    return wins / len(by_group)


def block_bootstrap(peaks, lengths, share, block_len, seed, n):
    """Блочный бутстреп Б6 по SPEC п. 3.8.

    Каждый репликон режется на подряд идущие отрезки длины `block_len` от начала координат (последний
    короче); повтор тянет отрезки **равновероятно с возвращением** — короткий последний отрезок с той
    же вероятностью, что и полный, — пока сумма их длин не достигнет длины генома (то есть сумма длин
    повтора не меньше длины генома и меньше её плюс `block_len`). **Кратность пика** — сколько раз
    вытянут отрезок, где лежит его середина, поэтому пики одного отрезка всегда имеют одну кратность, и
    блок не рвётся. Ответ — перцентили 2,5 и 97,5 по `n` повторам: граница перцентильная, и это
    записанное исключение из правила п. 3.6 — студентизация потребовала бы вложенного бутстрепа на
    каждый повтор. Длина отрезка записана заранее (100 тыс.), рядом печатаются 50 и 200 тыс.
    """
    block_len = int(block_len)
    seg_len, offset = [], {}
    for replicon in sorted(lengths):
        offset[replicon] = len(seg_len)
        length = int(lengths[replicon])
        seg_len.extend(min(block_len, length - start) for start in range(0, length, block_len))
    seg_len = np.asarray(seg_len, dtype=np.int64)
    if seg_len.size == 0:
        raise ValueError('block_bootstrap: ни одного отрезка (SPEC п. 3.8)')
    genome = int(sum(int(lengths[r]) for r in lengths))
    seg_of_peak = np.asarray([offset[r] + min(int(m) // block_len,
                                              (int(lengths[r]) - 1) // block_len) for r, m in peaks],
                             dtype=np.int64)
    rng = np.random.default_rng(seed)
    # сколько отрезков тянуть за раз: по полной длине отрезка, а не по самому короткому — у репликона с
    # остатком 1 при делении на `block_len` пачка иначе была бы размером с геном (Р11 рецензии кода T10);
    # недобор восполняет цикл ниже, ответ от размера пачки не зависит
    batch = int(genome // max(1, min(block_len, genome)) + 32)
    values = []
    for _ in range(int(n)):
        draws = rng.integers(0, seg_len.size, size=batch)
        reached = np.nonzero(np.cumsum(seg_len[draws]) >= genome)[0]
        while reached.size == 0:
            draws = np.concatenate([draws, rng.integers(0, seg_len.size, size=batch)])
            reached = np.nonzero(np.cumsum(seg_len[draws]) >= genome)[0]
        counts = np.bincount(draws[:reached[0] + 1], minlength=seg_len.size)
        values.append(float(share(counts[seg_of_peak])))
    lo, hi = np.percentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return float(lo), float(hi)


# ---- внешние проверки и контроли (SPEC пп. 3.9, 2.8) ----------------------------------------------
def _seeded_order(items, seed, key):
    """Механический порядок с семенем: по возрастанию `sha256('{семя}:{ключ}')`, как раздача групп по
    заходам (§ 2.6). Выбор не зависит ни от порядка входа, ни от версии генератора случайных чисел."""
    return sorted(items, key=lambda x: (_sha('%s:%s' % (seed, key(x))), str(key(x))))


def foreign_protein(target, candidates, seed):
    """«Чужой» белок Y для контроля Б13 (SPEC п. 3.9) и для дорожки п. 4.5.

    **Правило и семя, дописанные словами** (п. 3.11: «по какому правилу и с каким семенем выбирается
    „чужой“ белок Y»): годен кандидат, у которого идентичность с X **ниже 40%** (по всему белку,
    выравнивателем § 2.6) **и** семейство другое в смысле п. 1.5 —
    `prokabind.families.is_other_family`, то есть оба семейства известны и различны; «не определено»
    хотя бы с одной стороны годным не делает. Из годных берётся первый в порядке возрастания
    `sha256('{seed}:{белок}')`: порядок механический, одно семя — один белок, и он не зависит ни от
    порядка входа, ни от версии генератора. Нет годного — `None`.

    У паралогов вроде MarA и SoxS «чужой» белок может оказаться правильным ответом, поэтому порог
    родства и другое семейство требуются вместе. Читается контроль как Б6 (п. 3.8): такие пары входят в
    обучение, поэтому он показывает перенос выученной зависимости на невиденные белки, а не её
    самостоятельное возникновение; слова «прямое свидетельство» не употребляются.
    """
    fit = [c for c in candidates
           if float(c['identity']) < 40.0 and is_other_family(target.get('family'), c.get('family'))]
    if not fit:
        return None
    return dict(_seeded_order(fit, seed, lambda c: c['protein'])[0])


def foreign_proteins(target, pool, n, seed):
    """Пять чужих белков на проверочный белок для Б6 (SPEC пп. 3.8, 4.10).

    Те же два условия и то же семя, что у `foreign_protein`: годные упорядочены по
    `sha256('{seed}:{белок}')`, берутся первые `n`. Годных меньше `n` — возвращаются все; оценка «с
    чужим белком» в Б6 — средняя по тем, что есть.
    """
    fit = [c for c in pool
           if float(c['identity']) < 40.0 and is_other_family(target.get('family'), c.get('family'))]
    return [dict(c) for c in _seeded_order(fit, seed, lambda c: c['protein'])[:int(n)]]


def galagan_regions(rows, weight):
    """Участки Galagan с нужным уровнем белка — подмножество, несущее вес (SPEC п. 2.6, § 2.4.1).

    Остаются строки, у которых значение `level`, разобранное по `+` на составляющие, содержит `weight`:
    при `weight = 'native'` — `native`, `native+overexpression`, `native+vitro`. Сверхэкспрессия
    показывается отдельно и веса не несёт. Порядок строк сохраняется.
    """
    return [r for r in rows if str(weight) in str(r['level']).split('+')]


def holdout_pa(rows, min_peaks):
    """Единицы отложенного набора: вершины регуляторов, у которых не меньше `min_peaks` вершин
    (SPEC п. 3.9, порог 10); остальные регуляторы показываются отдельно.

    Учитываются регуляторы с ≥10 пиками — 40 из 48, остальные 8 отдельной строкой. Порядок строк
    сохраняется.
    """
    counts = {}
    for r in rows:
        counts[r['protein']] = counts.get(r['protein'], 0) + 1
    return [r for r in rows if counts[r['protein']] >= int(min_peaks)]


def no_leak_model(protein, models):
    """Модель, не видевшая этого белка и его вариантов (SPEC п. 2.8).

    Элемент списка — `{'name', 'fold', 'training'}`; `training` — белки обучающей части **вместе с их
    вариантами ≥90%** (по группам вариантов § 2.6), так что «не видела» значит «белка нет в
    `training`». Из подходящих — итоговая модель (`fold` равен `None`), если подходит она; иначе
    подходящая модель захода с наименьшим номером; нет подходящей — `None`.

    Правило без утечек действует по всем источникам: белок Galagan оценивается только моделью, которая
    не видела ни его данных, ни данных его вариантов (п. 3.2).
    """
    fit = [m for m in models if protein not in set(m['training'])]
    final = [m for m in fit if m['fold'] is None]
    if final:
        return dict(final[0])
    if not fit:
        return None
    return dict(sorted(fit, key=lambda m: (int(m['fold']), str(m['name'])))[0])


def release_tables(rows):
    """Собственный слой SPEC п. 3.10 — то, что раздаётся вместе с разбиением и кодом оценки.

    Открыты только `prodoric` и `pa_dapseq`: их строки остаются целиком. У строки, среди источников
    которой есть `regulondb` или `collectf`, убираются координаты и последовательность — `start`,
    `end`, `strand`, `seq`; остаются белок, геном, `sources`, `source_ids`, `names` и наши решения
    (пометка, дорожка). Строка StrepTRN выпадает целиком: её данные не раздаются, а идентификатора
    записи и нашего решения у пика нет. То же для источников, которые не раздаются вовсе
    (`galagan`, `sigmoid`) и для незнакомого имени источника: строка выпадает, а не раздаётся на
    всякий случай. Порядок оставшихся строк сохраняется.
    """
    out = []
    for row in rows:
        if 'sources' in row:
            sources = [s for s in str(row['sources']).split(';') if s]
        else:
            sources = [str(row['source'])] if row.get('source') else []
        if not sources or any(s not in OPEN_SOURCES + ID_ONLY_SOURCES for s in sources):
            continue
        if all(s in OPEN_SOURCES for s in sources):
            out.append(row)
        else:
            out.append({k: v for k, v in row.items() if k not in ('start', 'end', 'strand', 'seq')})
    return out


# ---- правила программы, которые оценка не повторяет, а зовёт (CONTRACTS § 4.3) --------------------
def local_maxima(scores, mask):
    """Максимумы дорожки с правилом плато — `prokabind.places.local_maxima` (SPEC п. 5.3, § 4.3).

    **Определение, дописанное словами** (п. 3.11, «склейка максимумов в место»): максимум — основание,
    чья оценка не ниже оценок обоих соседей, а подряд идущие основания с равной оценкой (плато)
    считаются **одним** максимумом, его положение — середина плато (при чётной длине — меньшая
    координата); плато через начало координат кольцевого репликона — тоже одно. Правило живёт в
    `prokabind.places`, и оценка его зовёт, а не повторяет.
    """
    return _places.local_maxima(scores, mask)


def build_places(scores, mask, circular, radius=20):
    """Склейка максимумов в место — `prokabind.places.build_places` (SPEC п. 5.3, § 4.3).

    **Определение, дописанное словами** (п. 3.11): место строится **жадно от сильнейшего**, а не
    цепочкой. Берётся максимум с наибольшей оценкой (при равных — с меньшей координатой), он становится
    местом, и все максимумы, чьё положение ближе `radius` = 20 оснований к его точке привязки
    (расстояние **строго меньше** 20: максимум ровно в 20 основаниях начинает своё место), причисляются
    к нему и выбывают; затем то же для сильнейшего из оставшихся. Склейка по цепочке запрещена прямо:
    прочитанная как цепочка, она даёт одно место на весь геном. Проверка масштаба: на заменителе
    дорожки правило даёт 165 380–172 564 места на геном E. coli — одно место на 27–28 оснований.
    """
    return _places.build_places(scores, mask, circular, radius)


def logo_trim(logo):
    """Обрезка логотипа перед сравнением — `prokabind.logo.trim` (SPEC п. 5.4, § 4.6).

    Столбцы с нормой меньше 20% наибольшей нормы столбца становятся неопределёнными. Порог записан
    заранее и не подбирается: без обрезки мера ломается от систематического перевеса A и T на краях
    поля видимости (переворачивает вывод у 41 мотива из 87), с обрезкой — ни у одного.
    """
    return _logo.trim(logo)


def logo_similarity(a, b, min_overlap=6):
    """Мера сходства логотипа с матрицей мотива — `prokabind.logo.similarity` (SPEC п. 5.4, § 4.6).

    **Определение, дописанное словами** (п. 3.11, «ориентация кандидатов и мера сходства логотипа»):
    мера — лучшее по сдвигу и цепи совпадение двух матриц «позиция × основание»: средний по столбцам
    коэффициент Пирсона при перекрытии не меньше шести столбцов, **у которых разброс значений не
    нулевой** (в столбце с четырьмя равными значениями Пирсон не определён; такие столбцы пропускаются,
    и перекрытие считается по определённым).

    **Ориентация кандидатов** (там же): дорожка оценок направления не имеет (п. 4.1), поэтому карты
    кандидатов ориентируются **до** усреднения — по тому из двух проходов п. 4.1, который дал в точке
    привязки большую оценку, и карты обратной ориентации разворачиваются с дополнением. Без ориентации
    7 из 51 непалиндромных мотивов E. coli оказываются ниже своего случайного уровня, с ней — ни один.
    Если согласие ориентации на пилоте не отличается от половины, печатается логотип без ориентации с
    пометкой «ориентация не подтверждена», а сравнение берёт лучшую из двух ориентаций.
    """
    return _logo.similarity(a, b, min_overlap)


def logo_random_level(logo, others):
    """Случайный уровень сходства — `prokabind.logo.random_level` (SPEC п. 5.4, § 4.6).

    Та же мера между логотипом белка и матрицами **других** белков набора, по одному значению на пару;
    печатается 95-й перцентиль этого распределения (по n − 1 значениям, считая от наибольшего вниз), и
    сходство со своим мотивом читается против него. Б14 — показ, а не слово итога: низкое сходство не
    означает, что предсказания неверны, — оно означает, что модель объясняет их иначе, чем матрица.
    """
    return _logo.random_level(logo, others)
