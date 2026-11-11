"""Обучение: из чего составляется пример, чему учим, пределы вклада, что закреплено
(SPEC пп. 4.5–4.8). Имена и сигнатуры — `CONTRACTS.md` § 4.5, журнал обучения — § 3.0.2.

Метки SigmoID в это обучение не входят (SPEC п. 4.5): по п. 2.7.7 это необязательный поздний опыт со
смешиванием слабым весом, у меток нет координат, и они не участвуют в выборе отрицательных окон. Вес
дорожки `sigmoid` функция `example_weights` считает (он закреплён п. 4.6 и равен 0,1), а сами пробы
строит задача T24 — здесь их нет.

**Поля исправляющих кругов** (SPEC пп. 4.2, 4.5, 4.8, 6.6; `CONTRACTS.md` § 2.9, с 04.11.2026; код — T34):
`coord_share`, `ema_decay`, `smooth_checks`, `protein_mask`, `sampler`, `example_workers`, `pool_cache`,
`diagnostics`. Каждое выключено по умолчанию, и при умолчаниях обучение побайтно прежнее (тесты 9 и 18–22 п. 4.11):
последовательный путь генератора (`sampler` = `sequential`) — прежний цикл без изменений, а выбор точки при
`ema_decay` = `null` и `smooth_checks` = 1 — прежнее «лучшее сырое значение, при равенстве — раннее».

**Подмена белка и нормировка вектора белка** (решение владельца 15; SPEC пп. 4.2, 4.6; `CONTRACTS.md` § 2.9; код —
T515): поля `swap_share` (по умолчанию 0) и `protein_norm` (по умолчанию `null`). При умолчаниях обучение побайтно
прежнее (тест 25 п. 4.11): план шага без ключа `swap`, формула потери прежняя, побочного пакета и буферов нет.
"""
import copy
import fcntl
import hashlib
import json
import math
import multiprocessing
import numbers
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from . import data
from .families import family_labels, is_other_family
from .features import load_protein, mask_kind
from .model import _Norm, build, covariance_penalty, protein_norm_of, receptive_field, standardise
from .scan import device_name, enforce_determinism, onehot, score_genome, select_device

#: Окно обучения (SPEC п. 4.1): «Обучение идёт окнами 512 оснований, из которых полезны 372».
TRAIN_WINDOW = 512

#: «четыре подобранных неразмеченных окна на положительный пример» (SPEC п. 4.8).
UNLABELLED_PER_EXAMPLE = 4

#: «Длина не меньше 8 оснований» — наименьшая ширина сравниваемого интервала (SPEC п. 2.2.3).
SHORTEST_INTERVAL = 8

#: «с долей A+T в пределах двух процентных пунктов от окна положительного примера» (SPEC п. 4.5).
AT_TOLERANCE = 0.02

#: «±40 оснований от центра записанного сайта, а если сайт длиннее 81 основания, то сам сайт» (п. 4.6).
POSITION_HALF = 40
POSITION_LONGEST = 81

#: «вероятность выбрать белок внутри дорожки пропорциональна min(число его интервалов, 200)» (п. 4.7).
PROTEIN_CAP = 200

#: «половина примеров — из дорожек с координатами, половина — из дорожки интервалов» (SPEC п. 4.5). С 04.11.2026 —
#: значение по умолчанию поля `coord_share` (`CONTRACTS.md` § 2.9), а не постоянная.
COORD_SHARE = 0.5

#: Поля исправляющих кругов и их значения по умолчанию (`CONTRACTS.md` § 2.9): отсутствующее поле (или `null`) —
#: значение по умолчанию, и при всех умолчаниях обучение побайтно прежнее.
ROUND_DEFAULTS = OrderedDict([('coord_share', COORD_SHARE), ('ema_decay', None), ('smooth_checks', 1),
                              ('protein_mask', 'whole'), ('sampler', 'sequential'), ('example_workers', 0),
                              ('pool_cache', None), ('diagnostics', None), ('swap_share', 0.0),
                              ('protein_norm', None)])

#: Третье число семени генератора побочного пакета нормировки (`CONTRACTS.md` § 2.9): `default_rng([семя, попытка,
#: 515])` — свой генератор, поэтому выбор примеров шага от поля `protein_norm` не меняется.
SIDE_BATCH_SALT = 515

#: Версия формата пулов на диске (`pool_cache`); входит в имя папки пула, поэтому другой формат не читается.
POOL_FORMAT = 1

#: Массивы пула одного генома, которые лежат на диске (`negative_pool`); указатель `boxes` — отдельно.
POOL_ARRAYS = ('centre', 'at', 'genic', 'free_left', 'free_right', 'at_count')

#: Счётчики `_TrainingData`, которые меняет сборка примера: при `sampler` = `per_step` пример может собираться в
#: рабочем процессе, и его прибавки к счётчикам возвращаются в основной процесс вместе с примером.
EXAMPLE_COUNTERS = ('unscorable', 'short_replicons', 'foreign_not_built', 'foreign_without_candidate',
                    'pool_builds', 'pool_seconds')

#: Дорожки п. 4.6, к которым приводятся значения столбца `lane` (`CONTRACTS.md` § 2.9); метки BacRegDB (поле
#: `bacregdb`) — координатные, их пометки нормируются вместе с точными сайтами и PRODORIC (SPEC п. 4.5).
LANE_OF = {'exact': 'coord', 'prodoric': 'coord', 'interval': 'interval', 'sigmoid': 'sigmoid', 'bacregdb': 'coord'}

#: Шаг кандидатов `negative_pool` в записи второй папки (поле `bacregdb`, `CONTRACTS.md` § 4.5): в пул входят только
#: центры, кратные ему. Читается при каждом обращении.
RECORD_STRIDE = 10

#: Сколько пулов записей второй папки данные обучения держат в памяти разом (`None` — без предела; § 4.5). Читается
#: при каждом обращении.
RECORD_POOLS = 32

#: Признаки скольких белков держатся в памяти (кэш § 2.8 читается с диска).
FEATURE_CACHE = 128

#: Сколько кандидатов отрицательных окон держится в памяти разом (около 22 байт на кандидата, то есть
#: порядка 3 ГБ). Пулы всех 168 геномов данных заняли бы около 12 ГБ, поэтому редко нужные вытесняются
#: и при надобности собираются заново; сколько раз и сколько на это ушло — в журнале обучения.
POOL_BUDGET = 140_000_000


# ---- потери (SPEC п. 4.6) -----------------------------------------------------------------------
def interval_score(scores):
    """Оценка интервала — логарифм среднего экспонент оценок его оснований (SPEC п. 4.6); при одном
    основании это ровно его оценка."""
    values = scores if torch.is_tensor(scores) else torch.as_tensor(scores, dtype=torch.float32)
    return torch.logsumexp(values, dim=-1) - math.log(values.shape[-1])


def pair_loss(s_pos, s_neg, margin=1.0):
    """Сравнение пары (SPEC п. 4.6): softplus от разности с отступом 1. Известный сайт должен стоять
    выше подобранного неразмеченного окна и выше сайта чужого неродственного белка."""
    x = margin - (s_pos - s_neg)
    if torch.is_tensor(x):
        return torch.nn.functional.softplus(x)
    return math.log1p(math.exp(x)) if x <= 0 else x + math.log1p(math.exp(-x))


def position_region(start, end, length):
    """Область сайта для потери положения (SPEC п. 4.6): ±40 оснований от центра записанного сайта,
    а если сайт длиннее 81 основания — сам сайт. Центр интервала — по `CONTRACTS.md` § 1."""
    out = np.zeros(int(length), dtype=bool)
    start, end = int(start), int(end)
    if end - start > POSITION_LONGEST:
        low, high = start, end
    else:
        centre = start + (end - start - 1) // 2
        low, high = centre - POSITION_HALF, centre + POSITION_HALF + 1
    out[max(low, 0):max(min(high, int(length)), 0)] = True
    return out


def position_loss(scores_window, region):
    """Потеря положения (SPEC п. 4.6): минус логарифм доли softmax-распределения оценок полезной части
    окна внутри области сайта. Внутри области положение не наказывается, поэтому модель не учится
    ставить пик туда, куда куратор поставил мотив."""
    values = scores_window if torch.is_tensor(scores_window) else torch.as_tensor(scores_window)
    mask = region if torch.is_tensor(region) else torch.as_tensor(np.asarray(region, dtype=bool))
    mask = mask.to(torch.bool)
    if torch.is_tensor(values):
        mask = mask.to(values.device)            # оценки могут лежать на видеокарте (§ 4.5)
    if not bool(mask.any()):
        raise ValueError('область сайта пуста: потерю положения считать не от чего (SPEC п. 4.6)')
    return torch.logsumexp(values, dim=-1) - torch.logsumexp(values[mask], dim=-1)


def has_position_loss(lane, label):
    """Есть ли у примера потеря положения (SPEC п. 4.6): у точных сайтов и меток BacRegDB да (граница оператора
    BacRegDB задана мотивом, п. 4.5), у интервалов StrepTRN, сайтов PRODORIC и проб SigmoID нет. Пометка
    точности на это не влияет — правило о дорожке."""
    return str(lane) in ('exact', 'bacregdb')


def example_weights(rows, config):
    """Веса примеров (SPEC п. 4.6): вес дорожки × вес пометки, нормированный на среднее 1 внутри
    дорожки. Координатная дорожка — точные сайты и PRODORIC вместе (`CONTRACTS.md` § 2.9); у
    интервальной место пометки занимает нормированная сила пика, ограниченная `strength_clip`.

    Нормировка зависит от состава дорожки, поэтому функция получает дорожку целиком.
    """
    lane_weights = config['lane_weights']
    label_weights = config['label_weights']
    low, high = (float(x) for x in config['strength_clip'])
    out = [0.0] * len(rows)
    lanes = {}
    for number, row in enumerate(rows):
        lane = LANE_OF.get(row['lane'])
        if lane is None:
            raise ValueError('неизвестная дорожка %r (словарь — CONTRACTS § 2.2)' % (row['lane'],))
        lanes.setdefault(lane, []).append(number)
    coord = lanes.get('coord', [])
    if coord:
        def label_weight(row):
            # метка BacRegDB: вес пометки поля, делённый на число мест пары — один сайт в нескольких штаммах весит
            # как один сайт (SPEC п. 4.6, `CONTRACTS.md` § 2.9)
            if row['lane'] == 'bacregdb':
                return float(config['bacregdb']['label_weight']) / float(row['n_places'])
            return float(label_weights[row['label']])
        values = np.array([label_weight(rows[i]) for i in coord])
        mean = float(values.mean())
        for i, value in zip(coord, values):
            out[i] = float(lane_weights['coord']) * float(value) / mean
    interval = lanes.get('interval', [])
    if interval:
        # «нормированная сила пика: log(1 + сила), делённое на среднее log(1 + сила) по дорожке,
        # с ограничением в пределах от 0,5 до 1» (SPEC п. 4.6)
        values = np.log1p(np.array([float(rows[i]['strength']) for i in interval]))
        mean = float(values.mean())
        for i, value in zip(interval, values):
            out[i] = float(lane_weights['interval']) * min(max(float(value) / mean, low), high)
    for i in lanes.get('sigmoid', ()):
        out[i] = float(lane_weights['sigmoid'])        # у проб SigmoID пометка одна, её вес 1
    return out


def example_loss(pos, unlabelled, foreign, position, shares, swap=None):
    """Полная потеря одного примера (SPEC п. 4.6):

        L = a_u · (средняя pair_loss по неразмеченным парам) + a_f · pair_loss с чужим белком
            + a_p · position_loss + a_s · pair_loss подмены

    Четыре неразмеченные пары входят **средней**, а не суммой. Если какого-то слагаемого у примера
    нет, его доля раздаётся остальным пропорционально их долям, так что сумма долей остаётся прежней.

    Подмена (T515, `CONTRACTS.md` § 4.5): `swap` — оценка того же положительного интервала с вектором белка чужой
    пары, s(I | X), или `None`; слагаемое — `pair_loss(pos, swap)`. Доля a_s — `shares['swap']`: нет ключа — формула
    прежняя побайтно (аргумент `swap` тогда не читается); ключ есть, а пары нет — доля раздаётся остальным, сумма
    долей `2,5 + a_s`.
    """
    if len(unlabelled) == 0:
        raise ValueError('пример без неразмеченных пар в шаг не входит (SPEC п. 4.5)')
    total = float(shares['unlabelled']) + float(shares['foreign']) + float(shares['position'])
    swap_share = shares.get('swap')
    if swap_share is not None:
        total = total + float(swap_share)
    mean = None
    for value in unlabelled:
        term = pair_loss(pos, value)
        mean = term if mean is None else mean + term
    mean = mean / len(unlabelled)
    parts = [(float(shares['unlabelled']), mean)]
    if foreign is not None:
        parts.append((float(shares['foreign']), pair_loss(pos, foreign)))
    if position is not None:
        parts.append((float(shares['position']), position))
    if swap_share is not None and swap is not None:
        parts.append((float(swap_share), pair_loss(pos, swap)))
    present = sum(share for share, _ in parts)
    scale = total / present if present else 1.0
    out = None
    for share, value in parts:
        term = (share * scale) * value
        out = term if out is None else out + term
    return out


def protein_choice_weights(counts, limit=PROTEIN_CAP):
    """Вероятность выбрать белок внутри дорожки (SPEC п. 4.7, `CONTRACTS.md` § 4.5): пропорционально
    min(число его интервалов, `limit`), сумма 1.

    **Одна функция на положительные дорожки и на дорожку «чужой белок»** (п. 4.5): предел один и тот
    же. Иначе обязательство Б10 «один белок» выполнялось бы для положительных примеров и нарушалось
    для отрицательных того же шага — 20 белков с тысячами пиков забирали бы 60% чужих отрицательных
    вместо 21% (рецензия кода T09, решение владельца 31.10).

    Перебор идёт по отсортированным ключам: веса не должны зависеть от порядка словаря
    (`CONSTITUTION.md`, ст. 6).
    """
    keys = sorted(counts)
    capped = {k: float(min(int(counts[k]), int(limit))) for k in keys}
    total = sum(capped.values())
    if total <= 0:
        raise ValueError('ни у одного белка нет интервалов: выбирать не из чего (SPEC п. 4.7)')
    return {k: capped[k] / total for k in keys}


def protein_auroc(sites, background):
    """Мера остановки для одного белка (SPEC п. 4.8, `CONTRACTS.md` § 4.5). Ключи обоих словарей —
    репликоны, значения — оценки сайтов и фона **этого** репликона.

    AUROC считается по каждому репликону отдельно (п. 3.4: «против фона того же генома»), и значение
    белка — **среднее по его репликонам**, а не один AUROC по сложенным вместе оценкам: у белка с
    сайтами в нескольких геномах сложение сравнивало бы сайт одного генома с фоном другого
    (записано 31.10 по рецензии кода T09). Репликон, где нет сайтов или нет фона, в среднее не входит.
    """
    values = []
    for replicon in sorted(sites):
        pos = np.asarray(sites[replicon], dtype=np.float64).ravel()
        bg = np.asarray(background.get(replicon, ()), dtype=np.float64).ravel()
        if pos.size == 0 or bg.size == 0:
            continue
        values.append(_auroc(pos, bg))
    if not values:
        raise ValueError('ни одного репликона с сайтами и фоном: мерить нечем (SPEC п. 4.8)')
    return float(np.mean(values))


def step_plan(config, protein_input):
    """План шага (SPEC пп. 4.4, 4.8): сколько положительных примеров и окон в пакете, доли трёх потерь
    и включена ли дорожка «чужой белок».

    У модели без белкового входа дорожка «чужой белок» выключена (п. 4.4: при постоянном векторе её
    пары противоречат друг другу), поэтому пример тянет пять окон, а не шесть, и две оставшиеся доли
    умножаются на 5/3, чтобы их сумма осталась той же.

    Доля подмены (`swap_share`, п. 4.6; T515): ключ `swap` в `loss_shares` — только при `swap_share` > 0 и белковом
    входе, иначе словарь прежний. Подмена берёт то же положительное окно вторым проходом, поэтому окон шага столько же.
    """
    positives = int(config['batch_positives'])
    unlabelled, foreign, position = (float(x) for x in config['loss_shares'])
    swap = swap_share_of(config)
    if protein_input:
        shares = {'unlabelled': unlabelled, 'foreign': foreign, 'position': position}
        if swap > 0:
            shares['swap'] = swap
        windows = positives * (1 + UNLABELLED_PER_EXAMPLE + 1)
    else:
        scale = (unlabelled + foreign + position) / (unlabelled + position)
        shares = {'unlabelled': unlabelled * scale, 'foreign': 0.0, 'position': position * scale}
        windows = positives * (1 + UNLABELLED_PER_EXAMPLE)
    return {'positives': positives, 'windows': windows, 'loss_shares': shares,
            'foreign_lane': bool(protein_input)}


def ema_beta(n, beta_max):
    """Коэффициент скользящего среднего после обновления `n` (с нуля): min(`beta_max`, (1 + n)/(10 + n)) (SPEC
    п. 4.8, `CONTRACTS.md` § 4.5). Среднее w̄ ← β·w̄ + (1 − β)·w — в этом порядке множителей; разгон нужен, чтобы
    на первой проверке среднее не держало почти целиком начальные веса."""
    return min(float(beta_max), (1.0 + int(n)) / (10.0 + int(n)))


def swap_share_of(config):
    """Значение поля `swap_share` (`CONTRACTS.md` § 2.9, SPEC п. 4.6): доля a_s слагаемого подмены, число ≥ 0; нет поля
    или `null` — 0, как у прочих полей кругов (`round_fields`). Не число (логическое значение числом не считается),
    бесконечность, NaN или число меньше 0 — отказ `ValueError`."""
    value = config.get('swap_share')
    if value is None:
        return float(ROUND_DEFAULTS['swap_share'])
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) \
            or float(value) < 0.0:
        raise ValueError('swap_share — число ≥ 0 (CONTRACTS § 2.9, SPEC п. 4.6), получено %r' % (value,))
    return float(value)


def side_batch(pool, size, seed, attempt):
    """Побочный пакет шага для нормировки вектора белка (`CONTRACTS.md` § 2.9, SPEC п. 4.2): `size` = min(K, n) белков
    из `pool` — отсортированных ключей белков обучающих строк захода, — в порядке, который даёт
    `numpy.random.default_rng([семя, попытка, 515]).choice(n, size, replace=False)`; попытка — номер прохода цикла
    обучения с нуля, как у `sampler` = `per_step`."""
    picked = np.random.default_rng([int(seed), int(attempt), SIDE_BATCH_SALT]).choice(len(pool), int(size),
                                                                                       replace=False)
    return [pool[int(k)] for k in picked]


def round_fields(config):
    """Значения полей исправляющих кругов (`CONTRACTS.md` § 2.9), с которыми идёт обучение: нет поля или `null` —
    значение по умолчанию (`ROUND_DEFAULTS`). Значение вне допустимого — отказ, а не догадка. С T515 — и поля
    `swap_share` (`swap_share_of`) и `protein_norm` (`model.protein_norm_of`)."""
    def get(name):
        value = config.get(name)
        return ROUND_DEFAULTS[name] if value is None else value
    out = OrderedDict()
    share = float(get('coord_share'))
    if not 0.0 <= share <= 1.0:
        raise ValueError('coord_share — доля от 0 до 1 (CONTRACTS § 2.9), получено %r' % (share,))
    out['coord_share'] = share
    decay = config.get('ema_decay')
    if decay is not None:
        decay = float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError('ema_decay — null или β_max в [0, 1) (CONTRACTS § 2.9), получено %r' % (decay,))
    out['ema_decay'] = decay
    smooth = get('smooth_checks')
    if int(smooth) != smooth or int(smooth) < 1:
        raise ValueError('smooth_checks — целое не меньше 1 (CONTRACTS § 2.9), получено %r' % (smooth,))
    out['smooth_checks'] = int(smooth)
    out['protein_mask'] = mask_kind(config)
    sampler = str(get('sampler'))
    if sampler not in ('sequential', 'per_step'):
        raise ValueError('sampler — `sequential` или `per_step` (CONTRACTS § 2.9), получено %r' % (sampler,))
    out['sampler'] = sampler
    workers = get('example_workers')
    if int(workers) != workers or int(workers) < 0:
        raise ValueError('example_workers — целое не меньше 0 (CONTRACTS § 2.9), получено %r' % (workers,))
    if sampler == 'sequential' and int(workers) > 0:
        raise ValueError('рабочие процессы сборки — только при sampler = per_step (CONTRACTS § 2.9): у '
                         'последовательного пути один генератор на всё обучение')
    out['example_workers'] = int(workers)
    for name in ('pool_cache', 'diagnostics'):
        value = config.get(name)
        out[name] = None if value is None else str(value)
    out['swap_share'] = swap_share_of(config)
    out['protein_norm'] = protein_norm_of(config)
    return out


# ---- подбор отрицательных окон (SPEC п. 4.5) ----------------------------------------------------
def negative_pool(config, choices, fold, genome, labels=None):
    """Кандидаты в неразмеченные интервалы одного генома для обучения захода `fold` (`None` — итоговая
    модель), по SPEC п. 4.5 и `CONTRACTS.md` § 4.5.

    Кандидат — **центр**, шаг одно основание. Запретное основание — основание метки обучающей части
    этой модели или «липкого» места п. 4.7, а также основания −1 и L сразу за концами репликона:
    интервал через начало кольцевого репликона не строится. Метки проверочных белков захода,
    контрольной части, отложенного набора и Galagan запрета не создают (п. 4.5).

    В пул входит каждый центр, для которого интервал наименьшей ширины 8 вместе с контекстом не
    задевает запретных оснований, и только такие; годность для настоящей ширины решает
    `matched_negatives`.

    `labels` — уже прочитанные метки обучающей части (их даёт `data.training_labels`); без него
    функция читает таблицы сама. Числа от этого не меняются, меняется только время.

    Типы массивов выбраны узкими, но **точными**: `centre` и расстояния — int32 (репликоны короче
    2³¹), доля A+T — float32 (разницы долей кратны 1/141, поэтому к допуску 0,02 они не подходят
    вплотную). Пул одного генома иначе занимает втрое больше памяти, а обучение держит их несколько.

    **Запись второй папки** (поле `bacregdb`; найдена не в `genomes_dir`, а в `bacregdb.genomes`): в пул входят
    только центры, кратные `RECORD_STRIDE` (читается при каждом вызове), — пул шага 1 этой записи, прореженный до
    них; расстояния, доля A+T и тип те же. У наших геномов шаг 1 при любом значении поля (SPEC п. 4.5).
    """
    rf = receptive_field(config)
    context = (rf - 1) // 2
    dirs = data.genome_dirs(config)
    sequence = data.genome_sequence(dirs, genome)
    length = len(sequence)
    if labels is None:
        labels = data.training_labels(config, choices, fold)
    forbidden = np.zeros(length, dtype=bool)
    for row in labels:
        if row['genome'] == genome:
            forbidden[max(row['start'], 0):min(row['end'], length)] = True
    forbidden |= data.sticky_mask(labels, genome, length)

    candidates = np.flatnonzero(~forbidden)
    marks = np.concatenate(([-1], np.flatnonzero(forbidden), [length]))
    place = np.searchsorted(marks, candidates)
    left = candidates - marks[place - 1]
    right = marks[place] - candidates
    usable = ((left > (SHORTEST_INTERVAL - 1) // 2 + context) &
              (right > SHORTEST_INTERVAL // 2 + context))
    if data.is_record(dirs, genome):
        stride = int(RECORD_STRIDE)
        if stride > 1:
            usable &= (candidates % stride) == 0
    centre = candidates[usable].astype(np.int32)

    letters = np.frombuffer(sequence.encode('ascii', 'replace'), dtype=np.uint8)
    at = ((letters == ord('A')) | (letters == ord('T'))).astype(np.int64)
    count = np.concatenate(([0], np.cumsum(at)))
    index = candidates[usable]
    number = (count[index + context + 1] - count[index - context])
    genes = data.gene_mask(dirs, genome, length)
    genic = genes[index]
    pool = {'genome': np.broadcast_to(np.array(genome), (centre.size,)), 'centre': centre,
            'at': (number / rf).astype(np.float32), 'genic': genic,
            'free_left': left[usable].astype(np.int32), 'free_right': right[usable].astype(np.int32),
            'context': int(context), 'one_genome': genome}
    # Указатель по типу примера и числу оснований A+T в поле видимости: `matched_negatives` смотрит
    # только в подходящие ящики вместо всего генома. Правило от этого не меняется — предикат
    # применяется к собранному подмножеству, и ответ тот же (ниже).
    if rf <= np.iinfo(np.uint8).max:
        pool['at_count'] = number.astype(np.uint8)
        order = np.lexsort((pool['at_count'], genic))
        keys = list(zip(genic[order].tolist(), pool['at_count'][order].tolist()))
        boxes, start = {}, 0
        for position in range(1, len(keys) + 1):
            if position == len(keys) or keys[position] != keys[start]:
                boxes[keys[start]] = order[start:position].astype(np.int32)
                start = position
        pool['boxes'] = boxes
    return pool


def matched_negatives(positive, pool, rng, n=UNLABELLED_PER_EXAMPLE):
    """Индексы **различных** кандидатов пула по SPEC п. 4.5 и `CONTRACTS.md` § 4.5: тот же геном, тот
    же тип по центру, доля A+T в пределах ±2 процентных пунктов и интервал **ширины сравниваемого
    интервала положительного примера** вокруг кандидата годен вместе с контекстом.

    Подходящих больше `n` — `n` из них выбираются через `rng` без повторов; меньше — возвращаются все
    (полосу допуска по составу не расширяют, SPEC п. 4.5); ни одного — пустой массив, и тогда пример
    в шаг не входит.
    """
    centre = np.asarray(pool['centre'])
    if centre.size == 0:
        return np.zeros(0, dtype=np.int64)
    width = int(positive['width'])
    context = int(pool['context'])
    share = float(positive['at'])
    boxes = pool.get('boxes')
    if boxes is None:
        look = np.arange(centre.size)
    else:
        # ящики, где доля A+T вообще может подойти; границы берутся с запасом, а решает предикат ниже,
        # поэтому ответ совпадает с полным перебором
        rf = 2 * context + 1
        low = int(math.floor((share - AT_TOLERANCE) * rf)) - 1
        high = int(math.ceil((share + AT_TOLERANCE) * rf)) + 1
        parts = [boxes[(bool(positive['genic']), value)]
                 for value in range(max(low, 0), high + 1)
                 if (bool(positive['genic']), value) in boxes]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        look = np.concatenate(parts)
    # каждый отбор идёт по уже выбранным местам, а не по всему геному: приведение целого массива
    # к другому типу до выборки стоило бы десятков миллисекунд на каждый пример
    one = pool.get('one_genome')
    if one is not None:
        if one != positive['genome']:
            return np.zeros(0, dtype=np.int64)
        ok = np.ones(look.size, dtype=bool)
    else:
        ok = np.asarray(pool['genome'])[look] == positive['genome']
    ok &= np.asarray(pool['genic'])[look].astype(bool) == bool(positive['genic'])
    ok &= np.abs(np.asarray(pool['at'])[look].astype(float) - share) <= AT_TOLERANCE + 1e-12
    ok &= np.asarray(pool['free_left'])[look] > (width - 1) // 2 + context
    ok &= np.asarray(pool['free_right'])[look] > width // 2 + context
    found = np.sort(look[ok])
    if found.size <= n:
        return found
    return np.sort(rng.choice(found, size=int(n), replace=False))


# ---- окна и геометрия пары ----------------------------------------------------------------------
def _window(sequence, centre, circular):
    """Физический вход примера: окно обучения 512 оснований с центром в центре сравниваемого интервала
    (SPEC п. 4.5). Ответ — (окно, индекс центра в окне); `None`, если репликон короче окна."""
    n = len(sequence)
    if n < TRAIN_WINDOW:
        return None
    start = centre - (TRAIN_WINDOW - 1) // 2
    if circular:
        begin = start % n
        if begin + TRAIN_WINDOW <= n:
            return sequence[begin:begin + TRAIN_WINDOW], (TRAIN_WINDOW - 1) // 2
        return sequence[begin:] + sequence[:begin + TRAIN_WINDOW - n], (TRAIN_WINDOW - 1) // 2
    start = min(max(start, 0), n - TRAIN_WINDOW)
    return sequence[start:start + TRAIN_WINDOW], centre - start


def crop_peak(start, end, summit, useful):
    """Пик шире полезной части окна обучения обрезается до неё (SPEC п. 4.6): вокруг вершины, а если
    вершины нет — вокруг середины. Окно той же ширины **сдвигается внутрь пика**, чтобы в метку не
    попала неразмеченная ДНК: `start = min(max(вершина − (useful − 1) // 2, начало), конец − useful)`.
    """
    if end - start <= useful:
        return start, end
    middle = summit if summit is not None else start + (end - start - 1) // 2
    begin = min(max(middle - (useful - 1) // 2, start), end - useful)
    return begin, begin + useful


def _sparse_max(track, widest):
    """Уровни для максимума по окну: `levels[k]` — максимум по окнам ширины 2**k. Уровней ровно
    столько, сколько нужно для окон до `widest`: у генома в 4,6 млн оснований каждый лишний уровень
    стоит ещё 18 МБ."""
    levels, size = [np.asarray(track, dtype=np.float32)], 1
    while size * 2 <= min(levels[0].size, max(int(widest), 1)):
        previous = levels[-1]
        levels.append(np.maximum(previous[:previous.size - size], previous[size:]))
        size *= 2
    return levels


def _range_max(levels, start, width):
    """Максимум по окнам `[start, start + width)`. `NaN` в окне делает ответ неопределённым — оценка
    окна, в котором есть основание без оценки, не определена (`CONTRACTS.md` § 4.8)."""
    k = int(width).bit_length() - 1
    size = 1 << k
    return np.maximum(levels[k][start], levels[k][start + width - size])


def _auroc(positive, background):
    """AUROC: доля пар «положительный выше фона», ничья — половина (`CONTRACTS.md` § 4.8)."""
    order = np.sort(np.asarray(background, dtype=np.float64))
    values = np.asarray(positive, dtype=np.float64)
    low = np.searchsorted(order, values, 'left')
    high = np.searchsorted(order, values, 'right')
    return float(((low + 0.5 * (high - low)) / order.size).mean())


# ---- рабочие процессы (`sampler` = `per_step`, `pool_cache`) -------------------------------------
#: Состояние, которое рабочие процессы получают копированием основного (`fork`): обучающий набор, семя, план шага.
#: Через аргументы оно не передаётся — иначе пулы и геномы копировались бы в каждый процесс.
_WORK = None


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _fork_pool(workers, state):
    """Пул из `workers` процессов, созданных копированием основного: обучающий набор, геномы и пулы на диске они
    читают общими (страницы памяти не копируются, пока их не пишут)."""
    global _WORK
    _WORK = state
    return multiprocessing.get_context('fork').Pool(int(workers))


def _release_work():
    global _WORK
    _WORK = None


def _ensure_task(genome):
    bundle = _WORK[0]
    before = bundle.pool_seconds
    got = bundle.ensure_pool(genome, blocking=False)
    return genome, got, bundle.pool_seconds - before


def _step_example(bundle, seed, plan, attempt, number):
    """Пример `number` попытки шага `attempt` при `sampler` = `per_step` (SPEC п. 4.8, правка 4 исправляющих кругов;
    `CONTRACTS.md` § 2.9): генератор примера — семя (семя обучения, номер попытки шага, номер примера); попытка — каждый
    проход цикла обучения с нуля, в том числе шаг, где все примеры выпали. Поэтому что собрано, не зависит от того, в
    каком процессе и в каком порядке. Ответ — номер строки положительного примера (`None` — дорожек нет), сам пример
    (`None` — в шаг не входит) и прибавки к счётчикам набора; счётчики самого `bundle` при этом не меняются — прибавки
    складывает основной процесс."""
    rng = np.random.default_rng([int(seed), int(attempt), int(number)])
    before = bundle.counters()
    try:
        picked = bundle.sample(rng)
        built = None if picked is None else bundle.build_example(picked, rng, plan)
        after = bundle.counters()
    finally:
        bundle.set_counters(before)
    return picked, built, {name: after[name] - before[name] for name in after}


def _example_task(attempt, number):
    bundle, seed, plan = _WORK
    return _step_example(bundle, seed, plan, attempt, number)


class _Examples:
    """Примеры попыток шага при `sampler` = `per_step`. При `example_workers` = 0 — в основном процессе; иначе рабочие
    процессы собирают примеры следующих попыток, пока основной учит текущую (сборка упиралась в процессор, а карта была
    занята на 3%, SPEC п. 4.8). Номер попытки растёт на единицу на каждом проходе цикла, поэтому каждую попытку можно
    собрать заранее: её примеры от исхода прежних попыток не зависят."""

    def __init__(self, bundle, seed, plan, workers, limit):
        self.bundle, self.seed, self.plan = bundle, int(seed), plan
        self.positives = int(plan['positives'])
        self.limit = int(limit)
        self.depth = max(2, 2 * int(workers))
        self.pending = {}
        self.pool = _fork_pool(workers, (bundle, self.seed, plan)) if int(workers) > 0 else None

    def _submit(self, attempt):
        if attempt in self.pending or attempt >= self.limit:
            return
        self.pending[attempt] = [self.pool.apply_async(_example_task, (attempt, k)) for k in range(self.positives)]

    def get(self, attempt):
        if self.pool is None:
            return [_step_example(self.bundle, self.seed, self.plan, attempt, k) for k in range(self.positives)]
        for ahead in range(attempt, attempt + self.depth + 1):
            self._submit(ahead)
        return [result.get() for result in self.pending.pop(attempt)]

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
            _release_work()


# ---- диагностика на каждой проверке (SPEC п. 6.6; `CONTRACTS.md` § 3.0.2, § 3.0.4) -----------------
def _average_ranks(values):
    """Ранги с единицы, ничья — средний ранг."""
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(np.float64)
    return (ends - (counts - 1) / 2.0)[inverse]


def rank_share(matrix):
    """Доля дисперсии рангов, зависящая от поданного белка (`CONTRACTS.md` § 3.0.2, поле `diagnostics`): строки —
    пики белка и его фон, столбцы — белковые входы (свой и чужие); ранги — внутри каждого входа; доля — сумма
    квадратов отклонений рангов от средней по входам **строки** в полной сумме квадратов отклонений от средней
    **столбца**. Все входы одинаково ранжируют пики — 0; ранги не зависят от пика — 1. Строка, где у какого-то входа
    нет оценки, выпадает целиком, поэтому у всех столбцов одна средняя ранга и доля лежит в [0, 1]; обрезка до
    [0, 1] снимает только ошибку округления."""
    m = np.asarray(matrix, dtype=np.float64)
    m = m[~np.isnan(m).any(axis=1)]
    if m.shape[0] < 2 or m.shape[1] < 2:
        return None
    ranks = np.column_stack([_average_ranks(m[:, j]) for j in range(m.shape[1])])
    total = float(((ranks - ranks.mean(axis=0, keepdims=True)) ** 2).sum())
    if total <= 0:
        return 0.0
    within = float(((ranks - ranks.mean(axis=1, keepdims=True)) ** 2).sum())
    return min(1.0, max(0.0, within / total))


class _Diagnostics:
    """Диагностика п. 6.6 на подвыборке § 3.0.4: число 11 (б′) с ошибкой и доля дисперсии рангов, зависящая от
    белка. Это не мера выбора: веса по ней не выбираются, остановка её не читает (SPEC п. 6.6).

    Оценка пика — наибольшая оценка по его основаниям; пик проходится с 70 основаниями по краям (полем видимости):
    вырезки всех пиков, которые нужны одному белковому входу, склеиваются через `N` и проходятся
    `scan.score_genome` — основание, чьё поле задело стык, оценки не получает, а у каждого основания пика поле
    целиком внутри его вырезки. Белковый вход — через `load_protein` (`_TrainingData.features`), с маской модели."""

    def __init__(self, path, bundle):
        self.path = Path(path)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        sub = json.loads(raw.decode('utf-8'))
        self.proteins = list(sub['proteins'])
        self.peaks = {p: [tuple(x) for x in sub['peaks'][p]] for p in self.proteins}
        self.background = {p: [tuple(x) for x in sub['background'].get(p, [])] for p in self.proteins}
        self.foreign = {p: list(sub['foreign'].get(p, [])) for p in self.proteins}
        need = {}
        for p in self.proteins:
            rows = self.peaks[p] + self.background[p]
            for q in [p] + self.foreign[p]:
                need.setdefault(q, set()).update(rows)
        self.need = OrderedDict((q, sorted(need[q])) for q in sorted(need))
        self.context = bundle.context
        self.genomes_dir = bundle.dirs

    def _cut(self, genome, start, end):
        sequence = data.genome_sequence(self.genomes_dir, genome)
        n = len(sequence)
        lo, hi = start - self.context, end + self.context
        if data.topology(self.genomes_dir, genome) == 'circular' and n >= hi - lo:
            if lo < 0:
                return sequence[lo % n:] + sequence[:hi], self.context
            if hi > n:
                return sequence[lo:] + sequence[:hi - n], self.context
            return sequence[lo:hi], self.context
        lo, hi = max(lo, 0), min(hi, n)
        return sequence[lo:hi], start - lo

    def peak_scores(self, model, pair, rows):
        pieces, places, offset = [], [], 0
        for genome, start, end in rows:
            text, inside = self._cut(genome, int(start), int(end))
            places.append((offset + inside, int(end) - int(start)))
            pieces.append(text)
            offset += len(text) + 1
        track, _ = score_genome(model, pair, 'N'.join(pieces), False)
        out = np.full(len(rows), np.nan, dtype=np.float64)
        for k, (begin, width) in enumerate(places):
            values = track[begin:begin + width]
            values = values[~np.isnan(values)]
            if values.size:
                out[k] = float(values.max())
        return out

    def measure(self, model, bundle):
        """Число 11 (б′) на подвыборке, его ошибка и доля дисперсии рангов (`CONTRACTS.md` § 3.0.2). У модели без
        белкового входа оба числа не определены (ответ не зависит от белка) — `None`."""
        if not getattr(model, 'protein_input', True):
            return {'number11': None, 'number11_se': None, 'protein_share': None, 'by_protein': {}}
        was_training = model.training
        model.eval()
        scores = {}
        try:
            for q, rows in self.need.items():
                values = self.peak_scores(model, bundle.features(q), rows)
                scores[q] = dict(zip(rows, values.tolist()))
        finally:
            if was_training:
                model.train()
        diffs, shares, by = [], [], OrderedDict()
        for p in self.proteins:
            inputs = [p] + self.foreign[p]
            own = np.array([[scores[q][r] for q in inputs] for r in self.peaks[p]],
                           dtype=np.float64).reshape(-1, len(inputs))
            bg = np.array([[scores[q][r] for q in inputs] for r in self.background[p]],
                          dtype=np.float64).reshape(-1, len(inputs))
            aurocs = []
            for j in range(len(inputs)):
                a, b = own[:, j], bg[:, j]
                a, b = a[~np.isnan(a)], b[~np.isnan(b)]
                aurocs.append(_auroc(a, b) if a.size and b.size else None)
            record = {'own_auroc': aurocs[0], 'foreign_aurocs': aurocs[1:]}
            if len(inputs) > 1 and all(x is not None for x in aurocs):
                record['diff'] = aurocs[0] - float(np.mean(aurocs[1:]))
                diffs.append(record['diff'])
            share = rank_share(np.vstack([own, bg])) if len(inputs) > 1 else None
            record['protein_share'] = share
            if share is not None:
                shares.append(share)
            by[p] = record
        n = len(diffs)
        return {'number11': float(np.mean(diffs)) if n else None,
                'number11_se': float(np.std(diffs, ddof=1) / math.sqrt(n)) if n > 1 else None,
                'protein_share': float(np.mean(shares)) if shares else None,
                'proteins': n, 'by_protein': by}


# ---- обучающий набор ----------------------------------------------------------------------------
class _TrainingData:
    """Обучающая часть модели, её примеры, веса и пулы отрицательных окон (SPEC пп. 4.5–4.7)."""

    def __init__(self, config, choices, fold, device=None):
        self.config = dict(config)
        #: Устройство счёта: признаки белка и окна кладутся сюда же, где лежит модель (§ 4.5).
        self.device = torch.device('cpu') if device is None else torch.device(device)
        self.choices = dict(choices)
        self.fold = fold
        self.rf = receptive_field(config)
        self.context = (self.rf - 1) // 2
        self.useful = TRAIN_WINDOW - 2 * self.context
        self.genomes_dir = config['genomes_dir']
        #: Поле `bacregdb` (§ 2.9; `None` — выключено) и папки геномов: наши и записи T513 (§ 4.11).
        self.field = data.bacregdb_of(config)
        self.dirs = data.genome_dirs(config)
        root = Path(config['data_root'])
        self.protein_rows = {r['protein']: r for r in data.read_tsv(root / 'proteins.tsv')}
        self.split_rows = {r['protein']: r for r in data.read_tsv(root / 'split.tsv')}
        self.column = data.fold_column(choices)
        self.training = data.training_proteins(config, choices, fold)
        self.labels = data.training_labels(config, choices, fold, self.training)
        self.relatedness = {p: r['relatedness_group'] for p, r in self.split_rows.items()}
        self.families = family_labels({p: r['pfam'] for p, r in self.protein_rows.items()},
                                      self.relatedness)
        fields = round_fields(config)
        #: Доля координатных дорожек (`coord_share`, § 2.9; по умолчанию ½ — прежняя постоянная).
        self.coord_share = fields['coord_share']
        #: Папка общих пулов на диске (`pool_cache`); `None` — пулы в памяти, как прежде.
        self.pool_dir = None if fields['pool_cache'] is None else Path(fields['pool_cache'])
        self._pool_names = {}
        self._pool_key = None
        self._genome_keys = {}
        self._features = OrderedDict()
        self._pools = OrderedDict()
        #: Пулы записей второй папки: (запись, шаг) → пул, от давнего обращения к последнему (`RECORD_POOLS`).
        self._record_pools = OrderedDict()
        self._records = {}
        self._held = 0
        self.pool_builds = 0
        self.pool_seconds = 0.0
        self.unscorable = 0
        self.short_replicons = 0
        self.foreign_not_built = 0
        self.foreign_without_candidate = 0
        self._own_labels = {}
        self._bacregdb_kin = None
        self._control = None
        self._sticky = None
        #: Метки белка и его вариантов ≥90% по геномам (`_kin_by_genome`) и их объединение по кольцу для правила пары
        #: подмены (`swap_blocked`), по (белок, геном).
        self._kin_genomes = {}
        self._swap_marks = {}
        self._build_rows()
        self._build_lanes()
        self._build_foreign()

    # -- примеры ---------------------------------------------------------------------------------
    def _build_rows(self):
        self.rows = []
        self.clipped_peaks = 0
        for row in self.labels:
            start, end = row['start'], row['end']
            if row['lane'] == 'interval':
                start, end = crop_peak(start, end, row['summit'], self.useful)
                if (end - start) != (row['end'] - row['start']):
                    self.clipped_peaks += 1
            made = {'protein': row['protein'], 'genome': row['genome'], 'start': start,
                    'end': end, 'lane': row['lane'], 'label': row['label'],
                    'strength': row['strength']}
            if 'n_places' in row:                    # метка BacRegDB: вес пометки делится на число мест пары (п. 4.6)
                made['n_places'] = row['n_places']
            self.rows.append(made)
        self.weights = example_weights(self.rows, self.config)

    def _build_lanes(self):
        self.lanes = {}
        for number, row in enumerate(self.rows):
            self.lanes.setdefault(LANE_OF[row['lane']], {}).setdefault(row['protein'], []).append(number)
        self.lane_pick = {}
        for lane, by_protein in sorted(self.lanes.items()):
            weights = protein_choice_weights({k: len(v) for k, v in by_protein.items()})
            keys = sorted(weights)
            self.lane_pick[lane] = (keys, np.array([weights[k] for k in keys], dtype=float))
        self.empty_lanes = [lane for lane in ('coord', 'interval') if lane not in self.lanes]
        self.lane_sizes = {lane: sum(len(v) for v in by.values()) for lane, by in self.lanes.items()}

    def _build_foreign(self):
        """Пул чужих отрицательных примеров (SPEC п. 4.5). Исключаются сайты, которые по правилу
        «чистого» сайта п. 2.9 перекрываются с меткой другого белка: именно такой сайт стал бы
        отрицательным примером для белка, который там и правда садится. Пики, перекрывающиеся
        с собственными пиками белка Y, исключаются при выборе — это зависит от Y.

        Строки дорожки `bacregdb` (поле `bacregdb`) в пул не входят вовсе (SPEC п. 4.5); метки BacRegDB этой
        модели при этом — обучающие метки других белков для правила «чистого» сайта."""
        sites = [i for i, row in enumerate(self.rows) if row['lane'] not in ('interval', 'bacregdb')]
        unclean = data.unclean_sites([self.rows[i] for i in sites], self.labels, self.dirs)
        dropped = {sites[k] for k in unclean}
        self.unclean_dropped = len(dropped)
        self.foreign_rows = np.array([i for i in range(len(self.rows))
                                      if i not in dropped and self.rows[i]['lane'] != 'bacregdb'],
                                     dtype=np.int64)
        # «Белок X выбирается по тому же правилу п. 4.7, что и положительный пример» (SPEC п. 4.5):
        # строки пула разложены по белкам, вес белка — min(число его строк, 200).
        self.foreign_by_protein = {}
        for number in self.foreign_rows.tolist():
            self.foreign_by_protein.setdefault(self.rows[number]['protein'], []).append(number)
        self._foreign_allowed = {}
        # Подбор по составу (SPEC п. 4.5, дописано 04.11.2026): у каждой строки пула — доля A+T на поле видимости
        # вокруг центра её интервала (§ 1), тем же счётом, что у положительного примера. Окно чужого — той же
        # ширины, что у положительного примера, вокруг центра сайта чужого белка, и его центр — тот же центр.
        rows = [self.rows[i] for i in self.foreign_rows.tolist()]
        self._foreign_names = sorted(self.foreign_by_protein)
        self._foreign_place = {name: k for k, name in enumerate(self._foreign_names)}
        self._foreign_pidx = np.array([self._foreign_place[r['protein']] for r in rows], dtype=np.int64)
        self._foreign_centre = np.array([r['start'] + (r['end'] - r['start'] - 1) // 2 for r in rows],
                                        dtype=np.int64)
        self._foreign_genome = np.array([r['genome'] for r in rows], dtype=object)
        self._foreign_interval = np.array([r['lane'] == 'interval' for r in rows], dtype=bool)
        self._foreign_start = np.array([r['start'] for r in rows], dtype=np.int64)
        self._foreign_end = np.array([r['end'] for r in rows], dtype=np.int64)
        self._foreign_at = np.array([self._at_share(r['genome'], int(c))
                                     for r, c in zip(rows, self._foreign_centre.tolist())], dtype=np.float64)
        self._foreign_base = {}
        self._own_context = {}
        self._kin_marks = None

    def _own_peaks(self, protein):
        if protein not in self._own_labels:
            by_genome = {}
            for row in self.labels:
                if row['protein'] == protein and row['lane'] == 'interval':
                    by_genome.setdefault(row['genome'], []).append((row['start'], row['end']))
            self._own_labels[protein] = {g: data.merge_intervals(v) for g, v in by_genome.items()}
        return self._own_labels[protein]

    def _allowed_foreign(self, protein):
        """Белки, годные в чужие отрицательные для белка `protein`, и их веса по п. 4.7.

        «Идентичность ниже 40% **и** другое семейство» (SPEC п. 4.5): группа родства и есть порог
        40% при покрытии ≥80% (`CONTRACTS.md` § 2.6), поэтому белки разных групп заведомо ниже 40%,
        а семейство — объединение меток Pfam п. 1.5, а не сырая метка. Семейство «не определено»
        другим семейством не считается ни с какой стороны — `is_other_family`.

        Считается один раз на белок: правило зависит от белка, а не от примера.

        Чужая пара — по белку положительного примера, а не по дорожке строки (SPEC п. 4.5): у белка только BacRegDB
        (его нет в `proteins.tsv` § 2.3) семейства п. 1.5 нет — годных чужих нет, пара не строится; наш белок
        получает их и для своих строк BacRegDB.
        """
        found = self._foreign_allowed.get(protein)
        if found is None and protein not in self.protein_rows:
            found = ((), None)
            self._foreign_allowed[protein] = found
        if found is None:
            group = self.relatedness.get(protein)
            family = self.families.get(protein)
            good = {}
            for other in sorted(self.foreign_by_protein):
                if other == protein or self.relatedness.get(other) == group:
                    continue
                if not is_other_family(self.families.get(other), family):
                    continue
                good[other] = len(self.foreign_by_protein[other])
            if good:
                weights = protein_choice_weights(good)
                keys = sorted(weights)
                found = (keys, np.array([weights[k] for k in keys], dtype=float))
            else:
                found = ((), None)
            self._foreign_allowed[protein] = found
        return found

    def _foreign_row_ok(self, row, own):
        """Годна ли строка годного белка: пик, перекрывающийся с собственными пиками белка Y, в
        чужие отрицательные не идёт (SPEC п. 4.5; Б6 это уже исключает, и обучающая дорожка обязана
        исключать тоже). Для сайтов условия нет: их отбирает правило «чистого» сайта при сборке пула.
        Дорожка считает это же правило разом по всему пулу (`foreign_base`).
        """
        if row['lane'] == 'interval':
            for a, b in own.get(row['genome'], ()):
                if min(row['end'], b) - max(row['start'], a) > 0:
                    return False
        return True

    def foreign_base(self, protein):
        """Строки пула, годные белку `protein` **по правилам до 04.11**: белок неродственный и другого семейства
        (`_allowed_foreign`), пик не перекрывается с собственными пиками белка (`_foreign_row_ok`). Ответ — маска
        по `foreign_rows`; считается один раз на белок. Подбор по составу и запрет контекста с меткой самого
        белка накладываются сверху (`foreign_candidates`)."""
        found = self._foreign_base.get(protein)
        if found is None:
            keys, _ = self._allowed_foreign(protein)
            allowed = np.zeros(len(self._foreign_names), dtype=bool)
            for name in keys:
                allowed[self._foreign_place[name]] = True
            found = allowed[self._foreign_pidx]
            for genome, marks in sorted(self._own_peaks(protein).items()):
                if not marks:
                    continue
                a = np.array([x for x, _ in marks], dtype=np.int64)
                b = np.array([y for _, y in marks], dtype=np.int64)
                rows = np.flatnonzero(found & self._foreign_interval & (self._foreign_genome == genome))
                if rows.size == 0:
                    continue
                # объединённые пики отсортированы и не пересекаются: первый с концом правее начала строки —
                # единственный, кто может её задеть
                first = np.searchsorted(b, self._foreign_start[rows], 'right')
                hit = first < b.size
                hit[hit] = a[first[hit]] < self._foreign_end[rows[hit]]
                found[rows[hit]] = False
            self._foreign_base[protein] = found
        return found

    def _kin_labels(self):
        """Опытные метки каждого белка по таблицам сайтов и пиков (все дорожки), по белкам: запрет контекста
        чужого окна (SPEC п. 4.5, 04.11.2026) берёт метки самого белка и его вариантов ≥90%. Отложенный набор
        (`holdout_summits.tsv`) здесь не читается: его строки — не таблица меток обучения (SPEC п. 3.2).

        При включённом поле `bacregdb` точные сайты и сайты PRODORIC — в исправленных границах после сведения
        (`data.corrected_rows`), как в мере остановки и в фоне оценки: одни границы во всех трёх местах."""
        if self._kin_marks is None:
            root = Path(self.config['data_root'])
            marks = {}
            for name in ('sites.tsv', 'peaks.tsv'):
                rows = data.read_tsv(root / name)
                if name == 'sites.tsv':
                    rows = data.corrected_rows(self.config, rows)
                for r in rows:
                    marks.setdefault(r['protein'], []).append((r['genome'], int(r['start']), int(r['end'])))
            self._kin_marks = marks
        return self._kin_marks

    def _kin_places(self, protein):
        """Места BacRegDB, которые запрет контекста чужого окна считает метками белка `protein` (SPEC п. 4.5, при
        включённом поле): строки дорожки `bacregdb` обучающей части этой модели — самого белка и белков BacRegDB,
        у которых в `related` он стоит с долей ≥ 0,900. Места недопущенных к модели белков сюда не входят."""
        if self.field is None:
            return []
        if self._bacregdb_kin is None:
            self._bacregdb_kin = data.bacregdb_kin(self.config)
        kin = self._bacregdb_kin.get(protein, set())
        return [(r['genome'], r['start'], r['end']) for r in self.labels
                if r['lane'] == 'bacregdb' and r['protein'] in kin]

    def _kin_by_genome(self, protein):
        """Метки белка `protein` и его вариантов ≥90% по геномам, `{геном: [(начало, конец), …]}` — одни для правила
        «свой контекст» чужого окна (`own_context`, SPEC п. 4.5) и для правила пары подмены (`swap_blocked`, п. 4.6):
        опытные метки (`_kin_labels`) белка и белков его группы вариантов § 2.6 той же части разбиения (метки чужих
        частей запрета не создают) и, при поле `bacregdb`, места BacRegDB (`_kin_places`). Считается один раз на белок;
        списки ответа не меняются вызывающими."""
        found = self._kin_genomes.get(protein)
        if found is None:
            row = self.split_rows.get(protein)
            kin = [protein]
            if row is not None:
                part = row[self.column]
                kin = sorted(p for p, r in self.split_rows.items()
                             if r['variant_group'] == row['variant_group'] and r[self.column] == part)
                if protein not in kin:
                    kin.append(protein)
            table = self._kin_labels()
            found = {}
            for p in kin:
                for genome, a, b in table.get(p, ()):
                    found.setdefault(genome, []).append((a, b))
            for genome, a, b in self._kin_places(protein):
                found.setdefault(genome, []).append((a, b))
            self._kin_genomes[protein] = found
        return found

    def swap_blocked(self, positive, protein):
        """Правило пары подмены (SPEC п. 4.6, `CONTRACTS.md` § 4.5): `True`, если интервал `positive` (`{'genome',
        'start', 'end'}`) вместе с контекстом — `[start − context, end + context)` — задевает метку белка `protein` или его
        варианта ≥90% (метки — те же, что у правила «свой контекст», `_kin_by_genome`); у кольцевого репликона метки
        учитываются по кольцу. Тогда пары подмены нет: там X, может быть, и правда садится. Интервалы полуоткрытые:
        контекст, кончающийся ровно на начале метки, её не задевает."""
        genome = positive['genome']
        key = (protein, genome)
        found = self._swap_marks.get(key)
        if found is None:
            marks = self._kin_by_genome(protein).get(genome, [])
            if marks and data.topology(self.dirs, genome) == 'circular':
                n = len(data.genome_sequence(self.dirs, genome))
                marks = marks + [(a - n, b - n) for a, b in marks] + [(a + n, b + n) for a, b in marks]
            merged = data.merge_intervals(marks)
            found = (np.array([a for a, _ in merged], dtype=np.int64), np.array([b for _, b in merged], dtype=np.int64))
            self._swap_marks[key] = found
        starts, ends = found
        if starts.size == 0:
            return False
        low = int(positive['start']) - self.context
        high = int(positive['end']) + self.context
        # объединённые метки отсортированы и не пересекаются: задеть контекст может только первая метка, чей конец
        # правее его начала, — у следующих и начало правее
        k = int(np.searchsorted(ends, low, 'right'))
        return bool(k < ends.size and starts[k] < high)

    def own_context(self, protein):
        """Где окно чужого белка задело бы метку самого белка `protein` или его варианта ≥90% (SPEC п. 4.5,
        дописано 04.11.2026): «окна, контекст которых несёт метку самого белка Y … в пределах поля видимости
        ±70 оснований вокруг центра сравниваемого интервала». Свобода от меток требуется, как у неразмеченных
        окон, для **сравниваемого интервала вместе с контекстом** — `[c − (w − 1) // 2 − context,
        c − (w − 1) // 2 + w + context)` вокруг центра `c` строки пула при ширине `w` положительного примера.

        Ответ — строки пула, которые хоть при какой-то ширине до полезной части окна могут задеть метку:
        `(номера в foreign_rows, внутри метки, расстояние до конца метки слева, до начала метки справа)`. Окно
        ширины `w` задевает метку ровно тогда, когда центр внутри метки, или левое расстояние меньше
        `(w − 1) // 2 + context`, или правое меньше `w // 2 + 1 + context`. Варианты ≥90% — группа вариантов
        § 2.6 той же части разбиения (у вариантов она всегда та же: порог 90% строже порога групп родства);
        метки чужих частей разбиения запрета не создают (п. 4.5). У кольцевого репликона метки за концом
        координат учитываются по кольцу."""
        found = self._own_context.get(protein)
        if found is None:
            by_genome = self._kin_by_genome(protein)
            widest = self.useful
            left_need = (widest - 1) // 2 + self.context
            right_need = widest // 2 + 1 + self.context
            parts = []
            for genome in sorted(by_genome):
                rows = np.flatnonzero(self._foreign_genome == genome)
                if rows.size == 0:
                    continue
                marks = by_genome[genome]
                if data.topology(self.dirs, genome) == 'circular':
                    n = len(data.genome_sequence(self.dirs, genome))
                    marks = marks + [(a - n, b - n) for a, b in marks] + [(a + n, b + n) for a, b in marks]
                merged = data.merge_intervals(marks)
                a = np.array([x for x, _ in merged], dtype=np.int64)
                b = np.array([y for _, y in merged], dtype=np.int64)
                c = self._foreign_centre[rows]
                last = np.searchsorted(a, c, 'right') - 1          # последняя метка, начатая не правее центра
                has_left = last >= 0
                inside = np.zeros(c.size, dtype=bool)
                inside[has_left] = b[last[has_left]] > c[has_left]
                big = np.iinfo(np.int64).max // 4
                left = np.full(c.size, big, dtype=np.int64)
                left[has_left] = c[has_left] - b[last[has_left]]
                nxt = last + 1
                has_right = nxt < a.size
                right = np.full(c.size, big, dtype=np.int64)
                right[has_right] = a[nxt[has_right]] - c[has_right]
                near = inside | (left < left_need) | (right < right_need)
                if near.any():
                    parts.append((rows[near], inside[near], left[near], right[near]))
            if parts:
                found = tuple(np.concatenate([x[k] for x in parts]) for k in range(4))
            else:
                empty = np.zeros(0, dtype=np.int64)
                found = (empty, np.zeros(0, dtype=bool), empty, empty)
            self._own_context[protein] = found
        return found

    def foreign_candidates(self, positive):
        """Маска по `foreign_rows`: годные чужие отрицательные для положительного примера `positive` (строка
        `rows`) по SPEC п. 4.5 — правила до 04.11 (`foreign_base`), **доля A+T на поле видимости** вокруг центра
        сравниваемого интервала в пределах двух процентных пунктов от того же числа у положительного примера
        (тот же порог и тот же счёт, что у `matched_negatives`) и окно той же ширины вокруг центра строки,
        которое вместе с контекстом не задевает меток самого белка и его вариантов ≥90% (`own_context`)."""
        protein = positive['protein']
        found = self.foreign_base(protein).copy()
        if not found.any():
            return found
        width = int(positive['end']) - int(positive['start'])
        centre = int(positive['start']) + (width - 1) // 2
        share = self._at_share(positive['genome'], centre)
        found &= np.abs(self._foreign_at - share) <= AT_TOLERANCE + 1e-12
        rows, inside, left, right = self.own_context(protein)
        if rows.size:
            hit = inside | (left < (width - 1) // 2 + self.context) | (right < width // 2 + 1 + self.context)
            found[rows[hit]] = False
        return found

    def foreign(self, positive, rng):
        """Чужой отрицательный пример для белка положительного примера, или `None`, если годного нет.

        Годные — `foreign_candidates` (с 04.11.2026 — подбор по составу и запрет контекста с меткой самого
        белка, SPEC п. 4.5). Из годных белок выбирается по правилу п. 4.7 — вероятность ∝ min(число его годных
        интервалов, 200), — а интервал внутри белка равновероятно (п. 4.5, решение владельца 31.10); всё тем
        же генератором. Ни одного годного — **пара не строится**. Выбор точный, а не перебором наугад: при
        подборе по составу годна лишь часть пула, и отбрасывание наугад меняло бы веса п. 4.7.
        """
        found = self.foreign_candidates(positive)
        if not found.any():
            return None
        numbers = self.foreign_rows[found]
        owners = self._foreign_pidx[found]
        counts = np.bincount(owners, minlength=len(self._foreign_names))
        present = np.flatnonzero(counts)
        weights = protein_choice_weights({self._foreign_names[k]: int(counts[k]) for k in present})
        names = sorted(weights)
        chances = np.array([weights[k] for k in names], dtype=float)
        chosen = names[int(rng.choice(len(names), p=chances))]
        inside = numbers[owners == self._foreign_place[chosen]]
        return self.rows[int(inside[int(rng.integers(0, len(inside)))])]

    # -- отбор внутри дорожки (SPEC п. 4.7) ------------------------------------------------------
    def sample(self, rng):
        """Положительный пример: дорожка, затем белок с вероятностью, пропорциональной
        min(число его интервалов, 200), затем интервал внутри белка равновероятно (SPEC пп. 4.5, 4.7).

        Доля координатных дорожек — поле `coord_share` (по умолчанию половина, § 2.9); пустая дорожка свою долю
        отдаёт остальным (§ 4.5).
        """
        lanes = [lane for lane in ('coord', 'interval') if lane in self.lanes]
        if not lanes:
            return None
        if len(lanes) == 1:
            lane = lanes[0]
        else:
            lane = 'coord' if float(rng.random()) < self.coord_share else 'interval'
        keys, probabilities = self.lane_pick[lane]
        protein = keys[int(rng.choice(len(keys), p=probabilities))]
        inside = self.lanes[lane][protein]
        return int(inside[int(rng.integers(0, len(inside)))])

    # -- признаки белка (кэш § 2.8) --------------------------------------------------------------
    def features(self, protein):
        """Белковый вход — признаки и маска сбора — **только через** `features.load_protein` (`CONTRACTS.md`
        § 4.10): так маска домена (`protein_mask`) одна у шага обучения, проверки остановки и диагностики."""
        if protein in self._features:
            self._features.move_to_end(protein)
            return self._features[protein]
        pair = load_protein(self.config, protein, device=self.device)
        self._features[protein] = pair
        if len(self._features) > FEATURE_CACHE:
            self._features.popitem(last=False)
        return pair

    # -- пулы на диске (`pool_cache`, § 2.9; SPEC п. 4.8, правка 3 исправляющих кругов) -----------
    def pool_key(self):
        """Ключ файлов пула (`CONTRACTS.md` § 2.9, поле `pool_cache`): заход, столбец разбиения (`folds`),
        `prodoric_in_training`, поле видимости и sha256 таблиц `sites`, `peaks`, `split`, `proteins`; к нему — версия
        формата файлов и наименьшая ширина интервала, от которых пул зависит в коде. Пулы другого ключа не читаются:
        другой выбор или другой заход — другие запретные основания."""
        if self._pool_key is None:
            root = Path(self.config['data_root'])
            self._pool_key = OrderedDict([
                ('format', POOL_FORMAT), ('fold', None if self.fold is None else int(self.fold)),
                ('folds', self.column), ('prodoric_in_training', bool(self.choices.get('prodoric_in_training', True))),
                ('receptive_field', int(self.rf)), ('shortest_interval', SHORTEST_INTERVAL),
                ('tables', OrderedDict((name, _file_sha256(root / name))
                                       for name in ('sites.tsv', 'peaks.tsv', 'split.tsv', 'proteins.tsv')))])
            if self.field is not None:
                # поле `bacregdb` (§ 2.9): его значение и суммы его таблиц — другие запретные основания и другие
                # метки; при выключенном поле ключ прежний, и пулы, собранные раньше, читаются, как читались
                tables = Path(self.field['tables'])
                self._pool_key['bacregdb'] = OrderedDict([
                    ('field', self.field),
                    ('tables', OrderedDict((name, _file_sha256(tables / name) if (tables / name).exists() else None)
                                           for name in ('corrections.tsv', 'labels.tsv', 'proteins.tsv',
                                                        'genome_labels.tsv')))])
        return self._pool_key

    def is_record(self, genome):
        """Запись второй папки (§ 4.11): найдена не в `genomes_dir`, а в `bacregdb.genomes`."""
        found = self._records.get(genome)
        if found is None:
            found = self._records[genome] = data.is_record(self.dirs, genome)
        return found

    def stride(self, genome):
        """Шаг кандидатов пула генома: `RECORD_STRIDE` у записи второй папки (читается при каждом обращении), 1 у
        нашего генома."""
        return int(RECORD_STRIDE) if self.is_record(genome) else 1

    def pool_name(self, genome):
        """Имя папки пула генома в `pool_cache`: номер записи и первые 24 знака sha256 ключа `pool_key` вместе с
        номером записи и суммами файлов генома (`.fasta`, `.gbff`) — пул зависит и от последовательности, и от
        разметки генов. Полный ключ лежит в `meta.json` пула и сверяется при чтении."""
        cached = (genome, self.stride(genome) if self.field is not None else None)
        name = self._pool_names.get(cached)
        if name is None:
            name = '%s.%s' % (genome, hashlib.sha256(json.dumps(self.genome_key(genome)).encode()).hexdigest()[:24])
            self._pool_names[cached] = name
        return name

    def genome_key(self, genome):
        """Ключ пула генома: `pool_key`, номер записи и суммы её файлов — из той папки списка, где запись найдена
        (§ 2.9). При включённом поле — и шаг кандидатов `stride` (1 у наших геномов, `RECORD_STRIDE` у записей второй
        папки), поэтому пул другого шага не читается, а ключ нашего генома от `RECORD_STRIDE` не зависит."""
        files = self._genome_keys.get(genome)
        if files is None:
            _, folder = data.genome_folder(self.dirs, genome)
            files = OrderedDict()
            for suffix in ('.fasta', '.gbff'):
                path = Path(folder) / (genome + suffix)
                files[suffix] = _file_sha256(path) if path.exists() else None
            self._genome_keys[genome] = files
        found = OrderedDict([('pool', self.pool_key()), ('genome', genome), ('files', files)])
        if self.field is not None:
            found['stride'] = self.stride(genome)
        return found

    def ensure_pool(self, genome, blocking=True):
        """Пул генома на диске: `ready` — уже лежит, `built` — собран сейчас, `busy` — его собирает другой процесс
        (только при `blocking=False`). Собирает тот, кто взял замок файла; запись — во временную папку и затем
        одним переименованием, поэтому читающий никогда не видит недописанного пула (на машине круга обучения
        стартуют разом)."""
        name = self.pool_name(genome)
        final = self.pool_dir / name
        if (final / 'meta.json').exists():
            return 'ready'
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        handle = os.open(str(self.pool_dir / (name + '.lock')), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                return 'busy'
            if (final / 'meta.json').exists():
                return 'ready'
            begin = time.time()
            made = negative_pool(self.config, self.choices, self.fold, genome, labels=self.labels)
            self.pool_builds += 1
            self.pool_seconds += time.time() - begin
            temporary = Path(tempfile.mkdtemp(prefix='.tmp-%s-' % name, dir=str(self.pool_dir)))
            try:
                present = [key for key in POOL_ARRAYS if key in made]
                for key in present:
                    np.save(temporary / (key + '.npy'), np.ascontiguousarray(made[key]))
                boxes = made.get('boxes')
                if boxes is not None:
                    keys = list(boxes)
                    order = (np.concatenate([boxes[k] for k in keys]).astype(np.int32) if keys
                             else np.zeros(0, dtype=np.int32))
                    sizes = np.array([boxes[k].size for k in keys], dtype=np.int64)
                    ends = np.cumsum(sizes)
                    np.save(temporary / 'box_order.npy', order)
                    np.save(temporary / 'box_keys.npy', np.array([[int(g), int(a)] for g, a in keys],
                                                                 dtype=np.int64).reshape(-1, 2))
                    np.save(temporary / 'box_bounds.npy', np.stack([ends - sizes, ends], axis=1).reshape(-1, 2))
                meta = {'format': POOL_FORMAT, 'name': name, 'genome': genome, 'key': self.genome_key(genome),
                        'context': int(made['context']), 'size': int(made['centre'].size), 'arrays': present,
                        'boxes': boxes is not None, 'built_utc': _now()}
                (temporary / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
                try:
                    os.rename(temporary, final)
                except OSError:
                    if (final / 'meta.json').exists():          # кто-то обошёл замок и успел раньше
                        shutil.rmtree(temporary, ignore_errors=True)
                        return 'ready'
                    raise
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            return 'built'
        finally:
            os.close(handle)

    def _disk_pool(self, genome):
        """Пул генома с диска: массивы — `np.memmap` (одна копия в страничном кэше на все обучения машины),
        указатель по типу и доле A+T — срезы одного массива. Значения те же, что у `negative_pool`, поэтому и
        выбор `matched_negatives` тот же (тест 18 п. 4.11)."""
        self.ensure_pool(genome, blocking=True)
        final = self.pool_dir / self.pool_name(genome)
        meta = json.loads((final / 'meta.json').read_text())
        if meta.get('key') != json.loads(json.dumps(self.genome_key(genome))):
            raise RuntimeError('пул %s: ключ в meta.json не тот, что у этого обучения (CONTRACTS § 2.9)' % final)
        size = int(meta['size'])

        def load(key):
            path = final / (key + '.npy')
            return np.load(path, mmap_mode='r') if size > 0 else np.load(path)
        pool = {key: load(key) for key in meta['arrays']}
        pool['genome'] = np.broadcast_to(np.array(genome), (size,))
        pool['context'] = int(meta['context'])
        pool['one_genome'] = genome
        if meta['boxes']:
            order = np.load(final / 'box_order.npy', mmap_mode='r') if size > 0 else np.load(final / 'box_order.npy')
            keys = np.load(final / 'box_keys.npy')
            bounds = np.load(final / 'box_bounds.npy')
            pool['boxes'] = {(bool(g), int(a)): order[int(s):int(e)]
                             for (g, a), (s, e) in zip(keys.tolist(), bounds.tolist())}
        return pool

    def ensure_pools(self, workers=0):
        """Все пулы обучающей части на диске до первого шага (`pool_cache`). Сначала каждый геном, чей замок
        свободен, — сразу (при `workers` > 0 — в рабочих процессах параллельно), затем те, что собирал кто-то
        другой, — с ожиданием. Ответ — сколько пулов собрало это обучение (0 — все прочитаны готовыми)."""
        if self.pool_dir is None:
            return 0
        # только наши геномы: пул записи второй папки (поле `bacregdb`) строится или читается, когда он впервые
        # нужен, — при первом примере из этой записи (`pool`; `CONTRACTS.md` § 4.5)
        genomes = sorted({row['genome'] for row in self.rows if not self.is_record(row['genome'])})
        status = {}
        if workers > 0 and len(genomes) > 1:
            pool = _fork_pool(workers, (self, None, None))
            try:
                for genome, got, seconds in pool.imap(_ensure_task, genomes):
                    status[genome] = got
                    if got == 'built':
                        self.pool_builds += 1
                        self.pool_seconds += seconds
            finally:
                pool.close()
                pool.join()
                _release_work()
        else:
            for genome in genomes:
                status[genome] = self.ensure_pool(genome, blocking=False)
        for genome in genomes:
            if status[genome] == 'busy':
                status[genome] = self.ensure_pool(genome, blocking=True)
        return sum(1 for got in status.values() if got == 'built')

    def counters(self):
        return {name: getattr(self, name) for name in EXAMPLE_COUNTERS}

    def set_counters(self, values):
        for name, value in values.items():
            setattr(self, name, value)

    def add_counters(self, delta):
        for name, value in delta.items():
            setattr(self, name, getattr(self, name) + value)

    def preload(self):
        """Геномы, аннотации, опытные метки и годные чужие каждого белка — в память основного процесса до запуска
        рабочих: процессы создаются копированием (`fork`) и читают это общим, не копируя и не пересчитывая в
        каждом. Ответы от этого не меняются: это те же значения, что иначе считались бы по первому запросу.

        Записи второй папки (поле `bacregdb`) не читаются: их читают по надобности, через ограниченный кэш (§ 4.5)."""
        for genome in sorted({row['genome'] for row in self.rows if not self.is_record(row['genome'])}):
            sequence = data.genome_sequence(self.dirs, genome)
            data.gene_mask(self.dirs, genome, len(sequence))
            data.topology(self.dirs, genome)
        self._kin_labels()
        for protein in sorted({row['protein'] for row in self.rows}):
            self.foreign_base(protein)
            self.own_context(protein)

    def pool(self, genome):
        """Пул отрицательных окон генома; редко нужные вытесняются по `POOL_BUDGET` и при надобности
        собираются заново. Метки передаются готовыми: числа от этого не меняются, только время. При
        `pool_cache` пул читается с диска (`_disk_pool`) и не вытесняется: память держит страничный кэш.

        Пул записи второй папки (поле `bacregdb`) — в своей очереди не больше `RECORD_POOLS` пулов (`_record_pool`)."""
        if self.is_record(genome):
            return self._record_pool(genome)
        if genome in self._pools:
            self._pools.move_to_end(genome)
            return self._pools[genome]
        if self.pool_dir is not None:
            made = self._disk_pool(genome)
            self._pools[genome] = made
            return made
        begin = time.time()
        made = negative_pool(self.config, self.choices, self.fold, genome, labels=self.labels)
        self.pool_builds += 1
        self.pool_seconds += time.time() - begin
        self._pools[genome] = made
        self._held += int(made['centre'].size)
        while len(self._pools) > 1 and self._held > POOL_BUDGET:
            _, gone = self._pools.popitem(last=False)
            self._held -= int(gone['centre'].size)
        return made

    def _trim_record_pools(self):
        bound = RECORD_POOLS
        if bound is None:
            return
        while len(self._record_pools) > max(int(bound), 0):
            self._record_pools.popitem(last=False)

    def _record_pool(self, genome):
        """Пул записи второй папки (`CONTRACTS.md` § 4.5): строится (при `pool_cache` — на диск, под замком, как прочие)
        или читается с диска, когда он впервые нужен; держится в очереди последнего обращения, не больше
        `RECORD_POOLS` (читается при каждом обращении; `None` — без предела): сверх предела уходит давний и при
        надобности строится или читается заново, а вытесненный данные обучения больше нигде не держат. Шаг — в ключе
        очереди: пул другого `RECORD_STRIDE` не выдаётся."""
        key = (genome, self.stride(genome))
        made = self._record_pools.get(key)
        if made is not None:
            self._record_pools.move_to_end(key)
        else:
            if self.pool_dir is not None:
                made = self._disk_pool(genome)
            else:
                begin = time.time()
                made = negative_pool(self.config, self.choices, self.fold, genome, labels=self.labels)
                self.pool_builds += 1
                self.pool_seconds += time.time() - begin
            self._record_pools[key] = made
        self._trim_record_pools()
        return made

    def sticky_places(self):
        """Число «липких» мест — где садятся 20 и больше разных белков StrepTRN — **по обучающей
        части этой модели** (SPEC п. 4.7: «Число таких мест пишется в журнал обучения»; поля в
        контракте не было до 31.10).

        Считается по геномам, где у обучающей части есть интервалы: «липкое» место создают только
        они. Место — отрезок подряд идущих оснований маски, а не основание.
        """
        if self._sticky is None:
            total = 0
            for genome in sorted({r['genome'] for r in self.labels if r['lane'] == 'interval'}):
                length = len(data.genome_sequence(self.dirs, genome))
                mask = data.sticky_mask(self.labels, genome, length)
                total += int(mask[0]) + int((mask[1:] & ~mask[:-1]).sum())
            self._sticky = total
        return self._sticky

    def _at_share(self, genome, centre):
        sequence = data.genome_sequence(self.dirs, genome)
        lo, hi = centre - self.context, centre + self.context + 1
        if lo < 0 or hi > len(sequence):
            piece = ''.join(sequence[i % len(sequence)] for i in range(lo, hi))
        else:
            piece = sequence[lo:hi]
        return (piece.count('A') + piece.count('T')) / self.rf

    # -- построение одного примера ---------------------------------------------------------------
    def build_example(self, number, rng, plan):
        """Положительный пример со своими отрицательными (SPEC п. 4.5), или `None`, если подходящих
        неразмеченных окон нет вовсе — тогда пример в шаг не входит."""
        row = self.rows[number]
        genome = row['genome']
        sequence = data.genome_sequence(self.dirs, genome)
        circular = data.topology(self.dirs, genome) == 'circular'
        width = row['end'] - row['start']
        centre = row['start'] + (width - 1) // 2
        pool = self.pool(genome)
        positive = {'genome': genome, 'at': self._at_share(genome, centre),
                    'genic': bool(data.gene_mask(self.dirs, genome, len(sequence))[centre]),
                    'width': width}
        chosen = matched_negatives(positive, pool, rng, n=UNLABELLED_PER_EXAMPLE)
        if len(chosen) == 0:
            return None
        places = [(sequence, centre, circular)]
        for index in chosen:
            places.append((sequence, int(pool['centre'][int(index)]), circular))
        windows, spans = [], []
        for sequence_of, middle, circ in places:
            made = self._span(sequence_of, middle, circ, width)
            if isinstance(made, str):
                # у положительного примера и его неразмеченных окон выпадает весь пример: сравнивать
                # и объяснять нечего (SPEC п. 4.1)
                if made == 'short':
                    self.short_replicons += 1
                else:
                    self.unscorable += 1
                return None
            windows.append(made[0])
            spans.append(made[1])
        unlabelled = list(range(1, len(windows)))
        foreign_index, foreign_protein = None, None
        if plan['foreign_lane']:
            other = self.foreign(row, rng)
            if other is None:
                self.foreign_not_built += 1
                self.foreign_without_candidate += 1
            else:
                # «окно той же ширины вокруг центра сайта чужого белка» — ширину задаёт
                # положительный пример (SPEC п. 4.5, таблица геометрии пары)
                foreign_centre = other['start'] + (other['end'] - other['start'] - 1) // 2
                made = self._span(data.genome_sequence(self.dirs, other['genome']),
                                  foreign_centre,
                                  data.topology(self.dirs, other['genome']) == 'circular',
                                  width)
                if isinstance(made, str):
                    # **У края репликона не строится только чужая пара**, а не выбрасывается весь
                    # пример (SPEC п. 4.5: «пара не строится»; правка 31.10 по рецензии кода T09).
                    # Доля этой потери раздаётся остальным слагаемым (`example_loss`).
                    self.foreign_not_built += 1
                else:
                    windows.append(made[0])
                    spans.append(made[1])
                    foreign_index = len(windows) - 1
                    foreign_protein = other['protein']
        region = None
        if has_position_loss(row['lane'], row['label']):
            begin = spans[0][0] - self.context
            region = position_region(begin, begin + width, self.useful)
        # Пара подмены (SPEC п. 4.6, T515): X — белок чужой пары; нет чужой пары — нет и подмены; интервал вместе с
        # контекстом задевает метку X или его варианта ≥90% — пары нет (`swap_blocked`). План без ключа `loss_shares`
        # (так `build_example` зовёт тест 24) или без доли `swap` — без подмены. Генератор правило не трогает.
        swap = False
        if foreign_protein is not None and float((plan.get('loss_shares') or {}).get('swap', 0) or 0) > 0:
            swap = not self.swap_blocked({'genome': genome, 'start': row['start'], 'end': row['end']},
                                         foreign_protein)
        return {'protein': row['protein'], 'windows': windows, 'spans': spans,
                'unlabelled': unlabelled, 'foreign': foreign_index, 'region': region,
                'lane': LANE_OF[row['lane']], 'foreign_protein': foreign_protein, 'swap': swap}

    def _span(self, sequence_of, middle, circular, width):
        """Окно обучения и место сравниваемого интервала в нём, или строка-причина, по которой окна
        нет: `short` — репликон короче окна 512, `outside` — интервал не попал в полезную часть окна.

        Вне полезной части у оснований нет полного поля видимости, то есть нет и оценки (SPEC п. 4.1).
        У окна с центром в центре интервала это выполняется всегда (ширина не больше полезной части);
        нарушить может только окно, сдвинутое внутрь линейного репликона у самого его края.
        """
        made = _window(sequence_of, middle, circular)
        if made is None:
            return 'short'
        text, index = made
        start = index - (width - 1) // 2
        if start < self.context or start + width > self.context + self.useful:
            return 'outside'
        return text, (start, start + width)

    def precision_box(self, bf16):
        """Контекст точности прохода модели: автоприведение к BF16 по **типу устройства**, а не `cpu` жёстко
        (`CONTRACTS.md` § 4.5), или FP32. Каждый раз новый: проход примера, второй проход подмены и побочный пакет
        нормировки входят в него по отдельности."""
        return (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16) if bf16
                else nullcontext())

    def loss(self, model, built, plan, bf16):
        """Потеря примера: оценки его окон одним проходом, потом формула п. 4.6.

        Пара подмены (`built['swap']` при доле `swap` плана; T515): s(I | X) — **второй проход** положительного окна с
        вектором белка чужой пары `foreign_protein`, с градиентом через оба прохода (`CONTRACTS.md` § 4.5)."""
        batch = torch.cat([onehot(text) for text in built['windows']], dim=0).to(self.device)
        features, mask = self.features(built['protein'])
        shares = plan['loss_shares']
        with self.precision_box(bf16):
            scores = model(batch, features, mask)
        scores = scores.float()                 # потери в FP32 (SPEC п. 4.8)
        low, high = built['spans'][0]
        pos = interval_score(scores[0, low:high])
        unlabelled = [interval_score(scores[i, a:b])
                      for i, (a, b) in ((i, built['spans'][i]) for i in built['unlabelled'])]
        foreign = None
        if built['foreign'] is not None:
            a, b = built['spans'][built['foreign']]
            foreign = interval_score(scores[built['foreign'], a:b])
        position = None
        if built['region'] is not None:
            useful = scores[0, self.context:self.context + self.useful]
            position = position_loss(useful, built['region'])
        if not (built.get('swap') and float(shares.get('swap', 0) or 0) > 0):
            return example_loss(pos, unlabelled, foreign, position, shares)
        other, other_mask = self.features(built['foreign_protein'])
        with self.precision_box(bf16):
            swapped = model(batch[0:1], other, other_mask)
        swap = interval_score(swapped.float()[0, low:high])
        return example_loss(pos, unlabelled, foreign, position, shares, swap=swap)

    # -- правило остановки (SPEC п. 4.8) ---------------------------------------------------------
    def _control_sites(self):
        """Чистые точные сайты контрольной части и опытные метки каждого её белка вместе с его
        вариантами ≥90% (SPEC пп. 2.9, 3.4, 4.8). Считается один раз: таблицы читаются по разу,
        а не на каждую проверку правила остановки.

        При включённом поле `bacregdb` мера та же, что обязывающая мера оценки с полем (SPEC п. 4.8): сайты — в
        исправленных границах после сведения (`data.corrected_rows`), собственные метки белка — и места BacRegDB
        самого белка и белков BacRegDB, его вариантов ≥90% (`data.bacregdb_kin`, как `own_labels` оценки), а фон
        без обучающих меток других белков — и без меток BacRegDB этой модели (они в `self.labels`)."""
        if self._control is None:
            root = Path(self.config['data_root'])
            inside = {p for p, r in self.split_rows.items() if r[self.column] == 'control'}
            all_sites = data.corrected_rows(self.config, data.read_tsv(root / 'sites.tsv'))
            all_peaks = data.read_tsv(root / 'peaks.tsv')
            sites = [{'protein': r['protein'], 'genome': r['genome'], 'start': int(r['start']),
                      'end': int(r['end'])}
                     for r in all_sites if r['protein'] in inside and r['lane'] == 'exact']
            unclean = data.unclean_sites(sites, self.labels, self.dirs)
            by_protein = {}
            for number, site in enumerate(sites):
                if number not in unclean:
                    by_protein.setdefault(site['protein'], []).append(site)
            # «Что убирается из фона белка: окна, перекрывающие его опытные метки и метки его
            # вариантов ≥90%» (SPEC п. 3.4) — группа вариантов § 2.6 и есть порог 90%
            members = {}
            for protein, row in self.split_rows.items():
                members.setdefault(row['variant_group'], []).append(protein)
            own = {}
            for protein in by_protein:
                kin = set(members.get(self.split_rows[protein]['variant_group'], [protein]))
                marks = {}
                for row in all_sites:
                    if row['protein'] in kin:
                        marks.setdefault(row['genome'], []).append((int(row['start']), int(row['end'])))
                for row in all_peaks:
                    if row['protein'] in kin:
                        marks.setdefault(row['genome'], []).append((int(row['start']), int(row['end'])))
                own[protein] = marks
            if self.field is not None:
                kin_bac = data.bacregdb_kin(self.config)
                for row in data.bacregdb_places(self.config):
                    for protein in by_protein:
                        if row['protein'] in kin_bac.get(protein, ()):
                            own[protein].setdefault(row['genome'], []).append((row['start'], row['end']))
            self._control = (by_protein, own)
        return self._control

    def control_metric(self, model):
        """Средний по белкам AUROC на чистых сайтах контрольной части против межгенного фона без
        обучающих меток других белков (SPEC п. 4.8) — та же мера, что обязывающая в п. 3.6, чтобы
        остановка и итог не считались по-разному. `None`, если мерить нечем."""
        by_protein, own = self._control_sites()
        if not by_protein:
            return None
        was_training = model.training
        model.eval()
        values = []
        try:
            for protein in sorted(by_protein):
                got = self._protein_auroc(model, protein, by_protein[protein], own[protein])
                if got is not None:
                    values.append(got)
        finally:
            if was_training:
                model.train()
        return float(np.mean(values)) if values else None

    def _protein_auroc(self, model, protein, sites, own):
        """Мера остановки одного белка: оценки его чистых точных сайтов и межгенного фона **того же
        репликона**, сведённые `protein_auroc` (SPEC пп. 3.4, 4.8). `None`, если мерить нечем."""
        by_site, by_background = {}, {}
        by_genome = {}
        for site in sites:
            by_genome.setdefault(site['genome'], []).append(site)
        features = self.features(protein)
        for genome in sorted(by_genome):
            sequence = data.genome_sequence(self.dirs, genome)
            length = len(sequence)
            circular = data.topology(self.dirs, genome) == 'circular'
            track, mask = score_genome(model, features, sequence, circular)
            track = np.where(mask, track, np.float32(np.nan))
            here = by_genome[genome]
            widest = max(site['end'] - site['start'] for site in here)
            levels = _sparse_max(track, widest)
            genes = data.gene_mask(self.dirs, genome, length)
            intergenic = np.flatnonzero(~genes)           # «Фон — все межгенные позиции генома» (п. 3.4)
            blocked = np.zeros(length, dtype=bool)
            for row in self.labels:                       # обучающие метки других белков
                if row['genome'] == genome:
                    blocked[max(row['start'], 0):min(row['end'], length)] = True
            for a, b in own.get(genome, ()):              # опытные метки белка и его вариантов
                blocked[max(a, 0):min(b, length)] = True
            closed = np.concatenate(([0], np.cumsum(blocked)))
            positive, background = [], []
            for site in here:
                width = site['end'] - site['start']
                if width < 1 or width > length:
                    continue
                positive.append(_range_max(levels, np.array([site['start']]), width)[0])
                starts = intergenic - (width - 1) // 2
                starts = starts[(starts >= 0) & (starts + width <= length)]
                starts = starts[(closed[starts + width] - closed[starts]) == 0]
                if starts.size:
                    background.append(_range_max(levels, starts, width))
            if not positive or not background:
                continue
            pos = np.asarray(positive, dtype=np.float64)
            bg = np.concatenate(background)
            pos, bg = pos[~np.isnan(pos)], bg[~np.isnan(bg)]
            if pos.size and bg.size:
                by_site[genome], by_background[genome] = pos, bg
        if not by_site:
            return None
        return protein_auroc(by_site, by_background)


def _optimiser(model, config):
    """AdamW, скорость обучения п. 4.8; затухание весов — кроме смещений и нормировок (SPEC п. 4.8)."""
    if str(config['optimizer']).lower() != 'adamw':
        raise ValueError('оптимизатор закреплён п. 4.8: adamw, а не %r' % (config['optimizer'],))
    norms = {id(p) for module in model.modules() if isinstance(module, (_Norm, torch.nn.LayerNorm))
             for p in module.parameters(recurse=False)}
    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (plain if (name.endswith('bias') or id(parameter) in norms) else decayed).append(parameter)
    return torch.optim.AdamW(
        [{'params': decayed, 'weight_decay': float(config['weight_decay'])},
         {'params': plain, 'weight_decay': 0.0}], lr=float(config['lr']))


def _lr_at(step, config):
    """Скорость обучения 2·10⁻⁴ с разогревом 5% шагов и косинусным спадом до 2·10⁻⁵ (SPEC п. 4.8)."""
    total = int(config['max_updates'])
    warmup = max(1, int(round(float(config['warmup_frac']) * total)))
    first, last = float(config['lr']), float(config['lr_final'])
    if step < warmup:
        return first * (step + 1) / warmup
    part = (step - warmup) / max(1, total - warmup)
    return last + (first - last) * 0.5 * (1 + math.cos(math.pi * min(part, 1.0)))


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _stop_rule(config):
    """Правило остановки п. 4.8 (изменено 04.11.2026 решением владельца): проверка каждые `check_every`
    обновлений, прекращение после `patience_updates` обновлений без улучшения, но не раньше `min_updates` —
    раньше только по пределу шагов (`CONTRACTS.md` § 2.9). Ответ — `(patience_updates, min_updates)`.

    Поле `patience` (терпение в проверках) снято: при изменении `check_every` оно молча меняло правило.
    Если проверок не будет вовсе (`max_updates < check_every`, так обучается тест повторяемости), правилу
    нечего решать, и полей оно не требует; иначе их отсутствие или прежнее поле — отказ, а не догадка.
    """
    if int(config['max_updates']) < int(config['check_every']):
        return math.inf, 0
    if 'patience' in config:
        raise ValueError('поле patience снято 04.11.2026: терпение — patience_updates, в обновлениях '
                         '(CONTRACTS § 2.9, SPEC п. 4.8)')
    missing = [k for k in ('patience_updates', 'min_updates') if k not in config]
    if missing:
        raise ValueError('в конфигурации нет %s (CONTRACTS § 2.9, SPEC п. 4.8)' % ', '.join(missing))
    return int(config['patience_updates']), int(config['min_updates'])


def train(config_path, fold, seed, out_dir, choices, protein_input=True, device=None):
    """Одно обучение (`CONTRACTS.md` § 4.5): заход `fold` (`None` — итоговая модель п. 3.9), выбор
    пилота `choices` (§ 2.9), ProkaBind или модель без белкового входа п. 4.4.

    `device` — **необязательный** аргумент (§ 4.5): `None` значит «видеокарта, если она есть, иначе
    процессор». Выбор делается в одном месте (`scan.select_device`), модель, признаки белка и окна
    идут туда же, и выбранное устройство пишется в журнал. На процессоре байты обучения от этого не
    меняются — это проверяет тест 9 п. 4.11.

    Пишет веса в `out_dir/model.pt` и журнал в `out_dir/journal.json` (§ 3.0.2). Повторяемо при одном
    семени, побайтно по `model.pt` (тест 9 п. 4.11): семя ставится до постройки модели, все выборки
    идут из одного генератора, порядок обхода фиксирован, а проверки правила остановки генератор
    не трогают.

    Поля исправляющих кругов (§ 2.9; SPEC пп. 4.2, 4.5, 4.8, 6.6), каждое выключено по умолчанию:
    - `sampler` = `per_step` — генератор каждого примера свой, с семенем (семя, обновление, номер примера), и
      примеры могут собирать `example_workers` рабочих процессов: шаг от их числа не зависит (тест 22);
    - `pool_cache` — пулы отрицательных окон на диске, общие для обучений машины (тест 18);
    - `ema_decay` — скользящее среднее весов с разгоном `ema_beta`; проверка остановки, диагностика и зачётные веса —
      по нему, а его веса пишутся на каждой проверке в `out_dir/checkpoints/{обновление}.pt`;
    - `smooth_checks` — мера остановки усредняется по последним проверкам (на первых — по имеющимся); улучшение —
      сглаженная выше лучшей принятой сглаженной не меньше чем на `min_gain`, зачётная точка — последняя проверка
      лучшего окна, при равенстве — раннего (тест 19);
    - `protein_mask` — маска сбора белкового вектора (`load_protein`, § 4.10; тест 20); `coord_share` — доля
      координатных дорожек (тест 21); `diagnostics` — подвыборка § 3.0.4, по которой на каждой проверке пишется
      диагностика п. 6.6.

    Поля T515 (SPEC пп. 4.2, 4.6; тест 25), у модели без белкового входа ни одно не действует:
    - `swap_share` > 0 — слагаемое подмены в потере примера (`step_plan`, `build_example`, `loss`); журнал `swap`
      считает примеры шагов: с парой подмены, с чужой парой без подмены по правилу перекрытия, без чужой пары;
    - `protein_norm` — на каждом шаге побочный пакет из min(K, n) белков обучающих строк (`side_batch`, свой
      генератор), их сырые векторы с градиентом → `standardise` → `set_protein_norm` до потерь примеров; к средней по
      примерам один раз прибавляется `w_cov · covariance_penalty(z)`; после шага — скользящие статистики (без
      градиента, до скользящего среднего весов, которое усредняет их вместе с весами). Журнал `protein_norm_run`.
    """
    started = _now()
    clock = time.time()
    path = Path(config_path)
    config = json.loads(path.read_text())
    config_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    fields = round_fields(config)
    enforce_determinism()
    unit = select_device(device)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # копия файла конфигурации побайтно: по ней `load_trained` применяет модель с её полями (§ 4.10)
    (directory / 'config.json').write_bytes(path.read_bytes())
    choices = dict(choices)
    plan = step_plan(config, protein_input)

    torch.manual_seed(int(seed))
    model = build(dict(config, protein_paths=choices.get('protein_paths', 'both')),
                  protein_input=protein_input).to(unit)
    bundle = _TrainingData(config, choices, fold, device=unit)
    rng = np.random.default_rng(int(seed))
    optimiser = _optimiser(model, config)
    bf16 = str(config.get('precision', 'bf16')).lower() == 'bf16'

    # Поля T515 (SPEC пп. 4.2, 4.6): у модели без белкового входа нет ни подмены, ни нормировки при любом значении.
    swap_on = 'swap' in plan['loss_shares']
    norm = fields['protein_norm'] if protein_input else None
    side_pool, side_size, penalties = None, None, []
    if norm is not None:
        # побочный пакет — из отсортированных ключей белков обучающих строк захода (CONTRACTS § 2.9)
        side_pool = sorted({row['protein'] for row in bundle.rows})
        if len(side_pool) < 2:
            raise ValueError('protein_norm: у захода %d обучающих белков, побочному пакету нужно хотя бы два '
                             '(CONTRACTS § 2.9)' % len(side_pool))
        side_size = min(int(norm['side_proteins']), len(side_pool))

    max_updates = int(config['max_updates'])
    check_every = int(config['check_every'])
    min_gain = float(config['min_gain'])
    patience_updates, min_updates = _stop_rule(config)

    ema_decay, smooth_checks = fields['ema_decay'], fields['smooth_checks']
    per_step, workers = fields['sampler'] == 'per_step', fields['example_workers']
    # Скользящее среднее (SPEC п. 4.8): начинается с начальных весов; проверка, диагностика и зачётные веса — его.
    ema_model = None
    if ema_decay is not None:
        ema_model = copy.deepcopy(model)
        ema_pairs = list(zip(ema_model.state_dict().values(), model.state_dict().values()))
    target = model if ema_model is None else ema_model
    diagnostics = None if fields['diagnostics'] is None else _Diagnostics(fields['diagnostics'], bundle)
    if per_step and workers > 0:
        bundle.preload()
    pools_built = bundle.ensure_pools(workers if per_step else 0)
    setup_seconds = time.time() - clock

    samples, lane_samples, used_rows, foreign_samples, lane_draws = {}, {}, set(), {}, {}
    control, smoothed, diagnostic_records, checkpoints = [], [], [], []
    tally = {'foreign_pairs': 0, 'skipped': 0, 'short_lists': 0, 'bacregdb_draws': 0}
    swaps = OrderedDict([('pairs', 0), ('blocked', 0), ('without_foreign', 0)])
    best_value, best_update, best_state, best_raw = None, None, None, None
    reference, gained_at = None, None
    updates, attempts, empty_steps = 0, 0, 0
    stopped = 'max_updates'
    step_times, example_times, check_seconds, diagnostic_seconds = [], [], 0.0, 0.0
    interval_total = sum(1 for row in bundle.rows if row['lane'] == 'interval')
    shown = sorted({row['protein'] for row in bundle.rows})

    def drawn(number):
        lane = LANE_OF[bundle.rows[number]['lane']]
        lane_draws[lane] = lane_draws.get(lane, 0) + 1
        if bundle.rows[number]['lane'] == 'bacregdb':
            tally['bacregdb_draws'] += 1

    def record(number, built):
        row = bundle.rows[number]
        samples[row['protein']] = samples.get(row['protein'], 0) + 1
        lane_samples[built['lane']] = lane_samples.get(built['lane'], 0) + 1
        used_rows.add(number)
        if len(built['unlabelled']) < UNLABELLED_PER_EXAMPLE:
            tally['short_lists'] += 1
        if built['foreign'] is not None:
            tally['foreign_pairs'] += 1
            name = built['foreign_protein']
            foreign_samples[name] = foreign_samples.get(name, 0) + 1
        if swap_on:
            # примеры, вошедшие в шаги (CONTRACTS § 3.0.2): с парой подмены; с чужой парой, но без подмены по правилу
            # перекрытия; без чужой пары — в сумме все примеры шагов
            if built['foreign'] is None:
                swaps['without_foreign'] += 1
            elif built.get('swap'):
                swaps['pairs'] += 1
            else:
                swaps['blocked'] += 1

    def coverage():
        sampled = sum(1 for i in used_rows if bundle.rows[i]['lane'] == 'interval')
        shows = [samples.get(p, 0) for p in shown]
        return {'unsampled_peaks_share': (1.0 - sampled / interval_total) if interval_total else None,
                'lane_passes': {lane: (lane_samples.get(lane, 0) / bundle.lane_sizes[lane])
                                for lane in sorted(bundle.lane_sizes)},
                'median_shows': float(np.median(shows)) if shows else None,
                'never_shown': int(sum(1 for s in shows if s == 0)),
                'proteins_with_examples': len(shows)}

    guard = 4 * max_updates + 100
    examples = _Examples(bundle, seed, plan, workers, guard) if per_step else None
    model.train()
    try:
        while updates < max_updates and attempts < guard:
            attempts += 1
            begin = time.time()
            batch = []
            if examples is None:
                # прежний путь (`sampler` = `sequential`): один генератор на всё обучение, порядок тот же
                for _ in range(plan['positives']):
                    number = bundle.sample(rng)
                    if number is None:
                        break
                    drawn(number)
                    built = bundle.build_example(number, rng, plan)
                    if built is None:
                        tally['skipped'] += 1
                        continue
                    record(number, built)
                    batch.append((number, built))
            else:
                got = examples.get(attempts - 1)          # попытка с нуля (§ 2.9)
                for _, _, delta in got:
                    bundle.add_counters(delta)
                for number, built, _ in got:
                    if number is None:
                        break
                    drawn(number)
                    if built is None:
                        tally['skipped'] += 1
                        continue
                    record(number, built)
                    batch.append((number, built))
            if not batch:
                empty_steps += 1
                continue
            gathered = time.time() - begin
            for group in optimiser.param_groups:
                group['lr'] = _lr_at(updates, config)
            optimiser.zero_grad(set_to_none=True)
            penalty = None
            if norm is not None:
                # побочный пакет шага (SPEC п. 4.2; CONTRACTS § 2.9): сырые векторы белковой сети с градиентом, в той же
                # точности, что проход примеров; статистики и штраф — в FP32, как потери; статистики шага — до потерь
                with bundle.precision_box(bf16):
                    vectors = torch.cat([model.protein(*bundle.features(p))
                                         for p in side_batch(side_pool, side_size, seed, attempts - 1)])
                z, step_mean, step_var = standardise(vectors.float(), norm['eps'])
                penalty = covariance_penalty(z)
                model.set_protein_norm(step_mean, step_var)
            total = None
            for number, built in batch:
                term = bundle.weights[number] * bundle.loss(model, built, plan, bf16)
                total = term if total is None else total + term
            objective = total / len(batch)
            if penalty is not None:
                # штраф ковариаций — раз за шаг, с весом w_cov, к средней по примерам (SPEC п. 4.6, CONTRACTS § 2.9)
                objective = objective + norm['w_cov'] * penalty
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['grad_clip']))
            optimiser.step()
            if norm is not None:
                # скользящие статистики r ← (1 − m)·r + m·s без градиента — до скользящего среднего весов, которое
                # усредняет и их; статистики шага снимаются, чтобы режим обучения вне шага не взял чужие
                model.update_protein_norm(step_mean, step_var, norm['momentum'])
                model.set_protein_norm(None, None)
                penalties.append(float(penalty.detach()))
            if ema_model is not None:
                # w̄ ← β·w̄ + (1 − β)·w после обновления номер `updates` (с нуля), SPEC п. 4.8
                beta = ema_beta(updates, ema_decay)
                with torch.no_grad():
                    for average, weight in ema_pairs:
                        if average.is_floating_point():
                            average.mul_(beta).add_(weight, alpha=1.0 - beta)
                        else:
                            average.copy_(weight)
            updates += 1
            step_times.append(time.time() - begin)
            example_times.append(gathered)
            if updates % check_every == 0:
                begin = time.time()
                value = bundle.control_metric(target)
                check_seconds += time.time() - begin
                control.append([updates, value])
                window = [v for _, v in control[-smooth_checks:] if v is not None]
                level = (sum(window) / len(window)) if (value is not None and window) else None
                smoothed.append([updates, level])
                if diagnostics is not None:
                    begin = time.time()
                    made = diagnostics.measure(target, bundle)
                    made = dict(OrderedDict([('update', updates)]), **made, **coverage())
                    made['seconds'] = round(time.time() - begin, 1)
                    diagnostic_seconds += made['seconds']
                    diagnostic_records.append(made)
                # веса, которыми мерилась проверка (среднего при `ema_decay`, иначе сырые), — на каждой проверке (§ 3.0.2)
                folder = directory / 'checkpoints'
                folder.mkdir(exist_ok=True)
                torch.save(OrderedDict((k, v.detach().cpu().clone()) for k, v in target.state_dict().items()),
                           folder / ('%d.pt' % updates))
                checkpoints.append(updates)
                if level is not None:
                    # порог улучшения сравнивается с **лучшим принятым** значением, а не с предыдущим (п. 4.8)
                    if reference is None or level - reference >= min_gain:
                        reference, gained_at = level, updates
                    if best_value is None or level > best_value:
                        best_value, best_update, best_raw = level, updates, value
                        best_state = OrderedDict((k, v.detach().clone())
                                                 for k, v in target.state_dict().items())
                    # терпение — в обновлениях, и раньше `min_updates` правило обучение не прекращает (п. 4.8)
                    if updates - gained_at >= patience_updates and updates >= min_updates:
                        stopped = 'rule'
                        break
    finally:
        if examples is not None:
            examples.close()

    state = best_state if best_state is not None else target.state_dict()
    # веса пишутся на процессоре: иначе файл обучения на видеокарте нельзя прочесть без неё
    torch.save(OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items()),
               directory / 'model.pt')

    end = coverage()
    counted_update = int(best_update) if best_update is not None else int(updates)
    times = np.asarray(step_times, dtype=np.float64)
    gathering = np.asarray(example_times, dtype=np.float64)
    skip = min(len(times) // 5, 20)
    warm, warm_gathering = times[skip:], gathering[skip:]
    journal = {
        'config_sha256': config_sha256,
        'choices': choices,
        'fold': None if fold is None else int(fold),
        'seed': int(seed),
        'protein_input': bool(protein_input),
        'plan': plan,
        'training_proteins': list(bundle.training),
        'samples_by_protein': {k: samples[k] for k in sorted(samples)},
        'lane_passes': end['lane_passes'],
        'lane_draws': {lane: lane_draws[lane] for lane in sorted(lane_draws)},
        'unsampled_peaks_share': end['unsampled_peaks_share'],
        'empty_lanes': list(bundle.empty_lanes),
        'clipped_peaks': int(bundle.clipped_peaks),
        'sticky_places': int(bundle.sticky_places()),
        'foreign_samples_by_protein': {k: foreign_samples[k] for k in sorted(foreign_samples)},
        'foreign_pairs_not_built': int(bundle.foreign_not_built),
        'examples_outside_the_useful_part': int(bundle.unscorable),
        'device': device_name(unit),
        'control_metric': control,
        'updates': int(updates),
        'best_update': counted_update,
        'stopped_by': stopped,
        'started_utc': started,
        'finished_utc': _now(),
        # поля исправляющих кругов (§ 2.9, § 3.0.2; с 04.11.2026)
        'coord_share': fields['coord_share'],
        'ema_decay': ema_decay,
        'smooth_checks': smooth_checks,
        'protein_mask': fields['protein_mask'],
        'sampler': fields['sampler'],
        'example_workers': workers,
        'control_metric_smoothed': smoothed,
        'counted': {'kind': 'best' if ema_model is None else 'ema', 'update': counted_update},
        'pool_cache': {'dir': fields['pool_cache'], 'reused': bool(fields['pool_cache'] is not None
                                                                   and pools_built == 0 and bundle.pool_builds == 0)},
        'diagnostics': diagnostic_records,
        'checkpoints': checkpoints,
        'bacregdb': _bacregdb_journal(bundle, tally['bacregdb_draws'], lane_draws.get('coord', 0)),
        # поля T515 (§ 2.9, § 3.0.2): значения полей и что они сделали; у модели без белкового входа — null
        'swap_share': fields['swap_share'],
        'protein_norm': None if fields['protein_norm'] is None else dict(fields['protein_norm']),
        'swap': dict(swaps) if swap_on else None,
        'protein_norm_run': None if norm is None else OrderedDict([
            ('side', int(side_size)),
            ('penalty_mean', (sum(penalties) / len(penalties)) if penalties else None),
            ('penalty_last', penalties[-1] if penalties else None)]),
        # сверх контракта — чтобы разбор прогона не держался на догадках
        'examples_without_negatives': int(tally['skipped']),
        'steps_without_examples': int(empty_steps),
        'examples_with_fewer_than_four_negatives': int(tally['short_lists']),
        'foreign_pairs': int(tally['foreign_pairs']),
        'foreign_pairs_without_candidate': int(bundle.foreign_without_candidate),
        'sites_dropped_from_foreign_lane': int(bundle.unclean_dropped),
        'examples_on_short_replicons': int(bundle.short_replicons),
        'best_control_metric': best_raw if best_state is not None else None,
        'best_control_metric_smoothed': best_value,
        'negative_pool_builds': int(bundle.pool_builds),
        'negative_pool_seconds': round(bundle.pool_seconds, 1),
        'median_shows': end['median_shows'],
        'never_shown': end['never_shown'],
        'diagnostics_subsample': None if diagnostics is None else {'path': str(diagnostics.path),
                                                                    'sha256': diagnostics.sha256},
        'timing': {'setup_seconds': round(setup_seconds, 1), 'steps': int(times.size),
                   'step_seconds_mean': float(times.mean()) if times.size else None,
                   'step_seconds_median': float(np.median(times)) if times.size else None,
                   'step_seconds_after_warmup': float(warm.mean()) if warm.size else None,
                   'warmup_steps_skipped': int(skip),
                   'examples_seconds_after_warmup': float(warm_gathering.mean()) if warm_gathering.size else None,
                   'check_seconds': round(check_seconds, 1), 'diagnostic_seconds': round(diagnostic_seconds, 1),
                   'total_seconds': round(time.time() - clock, 1)},
    }
    (directory / 'journal.json').write_text(json.dumps(journal, ensure_ascii=False, indent=1) + '\n')
    return journal


def _bacregdb_journal(bundle, draws, coord_draws):
    """Поле `bacregdb` журнала (`CONTRACTS.md` § 3.0.2): `null` при выключенном поле; иначе значение поля, допущенные
    белки BacRegDB с местом, числа модели (`data.bacregdb_counts`), положительные примеры из строк BacRegDB и строки
    BacRegDB в пуле чужих отрицательных (их там нет по построению — число мерится по пулу, а не пишется)."""
    if bundle.field is None:
        return None
    counts = data.bacregdb_counts(bundle.config, bundle.choices, bundle.fold)
    out = OrderedDict([('field', bundle.field)])
    out.update(counts)
    out['draws'] = int(draws)
    out['foreign_rows'] = int(sum(1 for i in bundle.foreign_rows.tolist() if bundle.rows[i]['lane'] == 'bacregdb'))
    out['coord_shares'] = _coord_shares(bundle, draws, coord_draws)
    return out


def _coord_shares(bundle, draws, coord_draws):
    """Доли источников в координатной дорожке (T512, пункт 6, выбор 3; сверх перечня § 3.0.2): по правилу п. 4.7 —
    доля выборок, приходящихся на строки BacRegDB (белок ∝ min(число строк, 200), строка внутри белка
    равновероятно), и на белки только BacRegDB (их нет в `proteins.tsv` § 2.3); и доля строк BacRegDB среди
    выпавших координатных примеров этого обучения."""
    keys, chances = bundle.lane_pick.get('coord', ((), None))
    rows_share, only_share = 0.0, 0.0
    for protein, chance in zip(keys, [] if chances is None else chances.tolist()):
        numbers = bundle.lanes['coord'][protein]
        mine = sum(1 for i in numbers if bundle.rows[i]['lane'] == 'bacregdb')
        rows_share += chance * mine / len(numbers)
        if protein not in bundle.protein_rows:
            only_share += chance
    return OrderedDict([('bacregdb_rows_by_rule', rows_share), ('bacregdb_only_by_rule', only_share),
                        ('bacregdb_rows_drawn', (int(draws) / int(coord_draws)) if coord_draws else None)])


#: Поля § 2.9, которые журнал пишет в том же виде, что конфигурация (`pool_cache` и `diagnostics` журнал пишет иначе).
JOURNAL_FIELDS = ('coord_share', 'ema_decay', 'smooth_checks', 'protein_mask', 'sampler', 'example_workers')


def load_trained(out_dir, device=None):
    """Обученная модель с зачётными весами `{out_dir}/model.pt` в режиме оценки и словарь конфигурации обучения
    (`CONTRACTS.md` § 4.10): поля файла конфигурации, с которым оно шло (копия `{out_dir}/config.json`; у обучений до
    T34 копии нет — тогда `work/config/train.json` проекта, если его сумма та же, что в журнале), дополненные полями
    § 2.9 из журнала и полем `protein_paths` выбора. Модель, обученная с `protein_mask` = `dbd`, применяется только так:
    `features.load_protein(config, белок)` с этим словарём — маска та же, что при обучении.

    Модель, обученная с `protein_norm` (T515; § 4.1), строится по полю копии конфигурации с буферами
    `protein_norm_mean` и `protein_norm_var` и читает их из `model.pt` строго, вместе с весами (при `ema_decay` — уже
    усреднённые с весами); в режиме оценки вектор белка нормируется ими."""
    folder = Path(out_dir)
    journal = json.loads((folder / 'journal.json').read_text())
    copy_path = folder / 'config.json'
    if not copy_path.exists():
        copy_path = Path(__file__).resolve().parents[2] / 'work' / 'config' / 'train.json'
    raw = copy_path.read_bytes() if copy_path.exists() else b''
    if hashlib.sha256(raw).hexdigest() != journal.get('config_sha256'):
        raise ValueError('конфигурация обучения %s не найдена: нет копии config.json с суммой из журнала '
                         '(CONTRACTS § 4.10)' % folder)
    config = json.loads(raw.decode('utf-8'))
    for name in JOURNAL_FIELDS:
        if name in journal:
            config[name] = journal[name]
    config['protein_paths'] = (journal.get('choices') or {}).get('protein_paths', 'both')
    unit = select_device(device)
    model = build(config, protein_input=bool(journal.get('protein_input', True)))
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu'))
    return model.to(unit).eval(), config
