"""Модель ProkaBind: вход, выход, геометрия, оба белковых пути и модель без белкового входа
(SPEC пп. 4.1–4.4). Слои, формула числа параметров, нелинейности и порядок — `CONTRACTS.md` § 4.1.

SPEC называет части модели, но не слои, а другие естественные прочтения того же текста дают другое
число параметров — поэтому слои закреплены контрактом, и этот модуль исполняет его дословно.
Сумма при значениях SPEC — 994 785 параметров, поле видимости — 141 основание.

**Нормировка вектора белка по белкам** (SPEC п. 4.2, поле `protein_norm` `CONTRACTS.md` § 2.9; T515): при поле,
отличном от `null`, у модели с белковым входом вектор белка, который получает ствол, нормируется по каждой оси —
в режиме обучения статистиками побочного пакета шага (`set_protein_norm`, с градиентом), в режиме оценки скользящими
статистиками из буферов `protein_norm_mean` и `protein_norm_var`. При `null` и у модели без белкового входа нет ни
буферов, ни нормировки: `state_dict` и вычисления прежние.

**Перенос мотива внутри модели** (поле `retrieval`, разработка по решению владельца 20, 10.11.2026; по умолчанию `null` —
ничего не меняется). При поле `{"width": W, "hidden": H}` выход считает не линейный слой `head`, а малая сеть
`ret_head` от признаков ствола в позиции и трёх чисел той же позиции: оценки матрицы мотива родственника (лучшее
размещение с центром в позиции, обе цепи — то же число, что у соперника «перенос мотива» SPEC п. 3.3.1; приведена к z по
случайной ДНК, обрезана и поделена на 4), идентичности родственника и логарифма числа его сайтов. Матрицу, её среднее и
разброс, идентичность и число сайтов даёт вызывающий (`retrieved`); `None` или нулевая матрица — «родственника нет».
Матрица уже ширины поля `width`, нечётной; при ширине не больше поля видимости поле видимости прежнее (сам код
предела ширины не ставит). С ключом `alpha` («сложение»: слой `head` прежний, к оценке прибавляется `alpha · 4 · z`)
z считается к составу всего поданного окна, и тогда оценка основания зависит от длины и границ окна — см. описание
`retrieved_channels`.
"""
import math
import numbers
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Порядок каналов ДНК (`CONTRACTS.md` § 4.1).
BASES = 'ACGT'

#: Ключи поля `protein_norm` (`CONTRACTS.md` § 2.9) — ровно эти: лишний или недостающий ключ — отказ.
PROTEIN_NORM_KEYS = ('side_proteins', 'momentum', 'eps', 'w_cov')


def protein_norm_of(config):
    """Значение поля `protein_norm` (`CONTRACTS.md` § 2.9, SPEC п. 4.2): `None` — поля нет или `null`, нормировки нет;
    иначе словарь ровно с ключами `side_proteins` (K — целое, K ≥ 2), `momentum` (m в (0, 1]), `eps` (ε > 0) и `w_cov`
    (w ≥ 0), в этом порядке. Значение вне допустимого — отказ `ValueError`, а не догадка: не объект, лишний или
    недостающий ключ, не число (логическое значение числом не считается), бесконечность или NaN, K < 2 или дробное,
    m вне (0, 1], ε ≤ 0, w < 0."""
    value = config.get('protein_norm')
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('protein_norm — null или объект с ключами %s (CONTRACTS § 2.9), получено %r'
                         % (', '.join(PROTEIN_NORM_KEYS), value))
    missing = [k for k in PROTEIN_NORM_KEYS if k not in value]
    extra = sorted(str(k) for k in value if k not in PROTEIN_NORM_KEYS)
    if missing or extra:
        raise ValueError('protein_norm — ровно ключи %s (CONTRACTS § 2.9): недостаёт %s, лишние %s'
                         % (', '.join(PROTEIN_NORM_KEYS), missing, extra))

    def number(name):
        x = value[name]
        if isinstance(x, bool) or not isinstance(x, numbers.Real) or not math.isfinite(float(x)):
            raise ValueError('protein_norm.%s — конечное число (CONTRACTS § 2.9), получено %r' % (name, x))
        return x
    side = number('side_proteins')
    if int(side) != side or int(side) < 2:
        raise ValueError('protein_norm.side_proteins — целое K ≥ 2 (CONTRACTS § 2.9), получено %r' % (side,))
    momentum = float(number('momentum'))
    if not 0.0 < momentum <= 1.0:
        raise ValueError('protein_norm.momentum — m в (0, 1] (CONTRACTS § 2.9), получено %r' % (momentum,))
    eps = float(number('eps'))
    if not eps > 0.0:
        raise ValueError('protein_norm.eps — ε > 0 (CONTRACTS § 2.9), получено %r' % (eps,))
    weight = float(number('w_cov'))
    if not weight >= 0.0:
        raise ValueError('protein_norm.w_cov — w ≥ 0 (CONTRACTS § 2.9), получено %r' % (weight,))
    return OrderedDict([('side_proteins', int(side)), ('momentum', momentum), ('eps', eps), ('w_cov', weight)])


def retrieval_of(config):
    """Значение поля `retrieval`: `None` — поля нет или `null`; иначе `{"width": нечётное W ≥ 1, "hidden": H ≥ 1}` и, по
    желанию, `"alpha": a > 0` (CONTRACTS § 2.9). Верхнего предела ширины код не ставит."""
    value = config.get('retrieval')
    if value is None:
        return None
    if not isinstance(value, dict) or not {'width', 'hidden'} <= set(value) <= {'width', 'hidden', 'alpha'}:
        raise ValueError('retrieval — null или объект с ключами width, hidden и, по желанию, alpha, получено %r' % (value,))
    width, hidden = int(value['width']), int(value['hidden'])
    if width < 1 or width % 2 == 0 or hidden < 1:
        raise ValueError('retrieval: width — нечётное положительное, hidden ≥ 1, получено %r' % (value,))
    out = OrderedDict([('width', width), ('hidden', hidden)])
    if value.get('alpha') is not None:
        # «сложение»: оценка = выход обычного слоя head + alpha · z-оценка мотива родственника; alpha закреплён (не учится),
        # поэтому сеть учит остаток — то, чего перенос мотива не объясняет, — и не может мотивом пренебречь
        alpha = float(value['alpha'])
        if not alpha > 0:
            raise ValueError('retrieval.alpha — число > 0, получено %r' % (value['alpha'],))
        out['alpha'] = alpha
    return out


def standardise(vectors, eps):
    """Нормировка строк-векторов `[N, P]` (N ≥ 2) по каждой оси (`CONTRACTS.md` § 4.1, SPEC п. 4.2): `mean` — среднее
    по строкам, `var` — **смещённая** дисперсия по строкам (делитель N), `z = (vectors − mean) / √(var + eps)`. Ответ —
    `(z, mean, var)`; всё дифференцируемо, градиент идёт и через сами статистики."""
    if vectors.dim() != 2 or vectors.shape[0] < 2:
        raise ValueError('нормировка по белкам — строки [N, P] при N ≥ 2 (CONTRACTS § 4.1), получено %s'
                         % (tuple(vectors.shape),))
    mean = vectors.mean(dim=0)
    centred = vectors - mean
    var = (centred * centred).mean(dim=0)
    return centred / torch.sqrt(var + eps), mean, var


def covariance_penalty(z):
    """Штраф ковариаций (`CONTRACTS.md` § 4.1, SPEC п. 4.2): `c = zᵀz / N` по `z` `[N, P]`; штраф — сумма квадратов
    внедиагональных элементов `c`, делённая на P. Оси, которые движутся вместе, дают большие внедиагональные `c`."""
    if z.dim() != 2:
        raise ValueError('штраф ковариаций — по матрице [N, P], получено %s' % (tuple(z.shape),))
    rows, width = z.shape
    c = torch.matmul(z.transpose(0, 1), z) / rows
    off = c - torch.diag(torch.diagonal(c))
    return (off * off).sum() / width


def receptive_field(config):
    """Поле видимости модели (SPEC п. 4.1): `1 + (stem_kernel − 1) + 2 · Σ dilations`.

    При значениях SPEC — **141 основание** (по 70 с каждой стороны). Ширина порождаемых фильтров
    нечётная и равна ширине свёртки ствола (п. 4.3), поэтому поле не зависит от места, где отклик
    фильтров входит в ствол, и в формуле её нет; ширина больше ствола сделала бы формулу неверной,
    поэтому такая конфигурация отвергается, а не считается молча.
    """
    stem = int(config['stem_kernel'])
    width = int(config.get('filter_width', stem))
    if stem % 2 == 0 or width % 2 == 0:
        raise ValueError('ширина свёртки ствола и порождаемых фильтров нечётная (SPEC п. 4.3): %d, %d'
                         % (stem, width))
    if width > stem:
        raise ValueError('ширина фильтров %d больше ширины свёртки ствола %d: поле видимости вышло бы '
                         'шире формулы SPEC п. 4.1' % (width, stem))
    return 1 + (stem - 1) + 2 * sum(int(d) for d in config['dilations'])


def valid_positions(length, config):
    """Основания окна длины `length`, у которых поле видимости целиком внутри окна (`CONTRACTS.md`
    § 4.1). Окно короче поля видимости не даёт ни одной такой позиции (SPEC п. 4.1)."""
    rf = receptive_field(config)
    context = (rf - 1) // 2
    out = np.zeros(int(length), dtype=bool)
    if int(length) >= rf:
        out[context:int(length) - context] = True
    return out


class _Norm(nn.Module):
    """Нормировка по каналам **в каждой позиции отдельно** (`CONTRACTS.md` § 4.1), с обучаемыми γ и β.

    Нормировка через позиции сломала бы поле видимости: оценка основания стала бы зависеть от всего
    окна (тест 2 п. 4.11 ловит это точно, до нуля).
    """

    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):                          # [B, C, L] -> [B, C, L]
        y = F.layer_norm(x.transpose(1, 2), (x.shape[1],), self.weight, self.bias)
        return y.transpose(1, 2)


class _Block(nn.Module):
    """Остаточный блок ствола (`CONTRACTS.md` § 4.1): нормировка → FiLM этого блока, если он назван →
    GELU → свёртка окном 3 с расширением блока → GELU → свёртка 1×1 → прибавление входа блока."""

    def __init__(self, width, dilation):
        super().__init__()
        self.norm = _Norm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.proj = nn.Conv1d(width, width, 1)

    def forward(self, x, film):
        h = self.norm(x)
        if film is not None:
            gamma, beta = film
            h = h * (1 + gamma) + beta
        h = self.conv(F.gelu(h))
        h = self.proj(F.gelu(h))
        return x + h


class _ProteinNet(nn.Module):
    """Белковая часть (SPEC п. 4.2): из признаков ESM-2 на каждый остаток — один вектор белка ширины
    `protein_width`. Линейный слой, нормировка, остаточный блок, затем сбор по остаткам вниманием
    с `n_queries` обучаемыми запросами плюс среднее по остаткам. Заполняющие позиции в сборе не
    участвуют."""

    def __init__(self, config):
        super().__init__()
        esm = int(config['esm_dim'])
        width = int(config['protein_width'])
        queries = int(config['n_queries'])
        self.width = width
        self.inp = nn.Linear(esm, width)
        self.norm = nn.LayerNorm(width)
        self.res_up = nn.Linear(width, 2 * width)
        self.res_down = nn.Linear(2 * width, width)
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=width ** -0.5)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)
        self.out = nn.Linear((queries + 1) * width, width)

    def forward(self, features, mask):
        h = F.gelu(self.norm(self.inp(features)))              # [B, R, P]
        h = h + self.res_down(F.gelu(self.res_up(h)))
        keys, values = self.k_proj(h), self.v_proj(h)
        q = self.q_proj(self.queries).unsqueeze(0)             # [1, Q, P]
        att = torch.matmul(q, keys.transpose(1, 2)) / math.sqrt(self.width)
        att = att.masked_fill(~mask.unsqueeze(1), float('-inf'))
        answers = self.o_proj(torch.matmul(torch.softmax(att, dim=-1), values))   # [B, Q, P]
        weight = mask.unsqueeze(-1).to(h.dtype)
        average = (h * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1.0)      # [B, P]
        return self.out(torch.cat([answers.flatten(1), average], dim=-1))


class ProkaBind(nn.Module):
    """ДНК-часть и оба пути белка (SPEC пп. 4.1, 4.3, 4.4).

    `protein_input=False` — модель без белкового входа п. 4.4: вместо белковой сети обучаемый
    постоянный вектор той же ширины, и фильтры и FiLM получают его. Ответ от белка не зависит.
    `protein_paths='film_only'` — один FiLM без порождаемых фильтров (выбор пилота, п. 4.8).
    Поле `protein_norm` (п. 4.2) у модели с белковым входом добавляет буферы `protein_norm_mean` (нули) и
    `protein_norm_var` (единицы) ширины `protein_width` и нормировку вектора белка (`protein_vector`).
    """

    def __init__(self, config, protein_input=True):
        super().__init__()
        width = int(config['trunk_width'])
        stem = int(config['stem_kernel'])
        filters = int(config['n_filters'])
        fwidth = int(config['filter_width'])
        dilations = [int(d) for d in config['dilations']]
        protein_width = int(config['protein_width'])
        paths = str(config.get('protein_paths', 'both'))
        if paths not in ('both', 'film_only'):
            raise ValueError('protein_paths — `both` или `film_only` (SPEC п. 4.8), получено %r' % paths)
        self.config = dict(config)
        self.paths = paths
        self.protein_input = bool(protein_input)
        self.n_filters = filters
        self.filter_width = fwidth

        self.stem = nn.Conv1d(4, width, stem, padding=stem // 2)
        self.generate = nn.Linear(protein_width, filters * 4 * fwidth) if paths == 'both' else None
        self.mix = nn.Conv1d(width + (filters if paths == 'both' else 0), width, 1)
        self.blocks = nn.ModuleList([_Block(width, d) for d in dilations])
        # FiLM в блоках, названных `film_blocks` (с единицы, `CONTRACTS.md` § 2.9)
        self.film = nn.ModuleDict()
        for number in sorted(int(b) for b in config['film_blocks']):
            if not 1 <= number <= len(dilations):
                raise ValueError('film_blocks называет блок %d, а блоков %d' % (number, len(dilations)))
            self.film[str(number)] = nn.Linear(protein_width, 2 * width)
        self.head_norm = _Norm(width)
        self.retrieval = retrieval_of(config)
        if self.retrieval is None or 'alpha' in self.retrieval:
            self.head = nn.Linear(width, 1)
            self.ret_head = None
        else:
            # перенос мотива внутри модели: признаки ствола и три числа о мотиве родственника в той же позиции
            self.head = None
            hidden = self.retrieval['hidden']
            self.ret_head = nn.Sequential(nn.Linear(width + 3, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if protein_input:
            self.protein = _ProteinNet(config)
            self.constant = None
        else:
            self.protein = None
            self.constant = nn.Parameter(torch.zeros(protein_width))
        for layer in self.film.values():
            # «γ и β в начале нулевые — преобразование тождественно» (SPEC п. 4.3): и вес, и сдвиг
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Нормировка вектора белка по белкам (SPEC п. 4.2, `CONTRACTS.md` § 4.1): поле проверяется при любом входе, а
        # буферы и нормировка — только у модели с белковым входом. Буферы не трогают генератор случайных чисел, поэтому
        # начальные веса те же, что без поля.
        norm = protein_norm_of(config)
        self.protein_norm = norm if protein_input else None
        self._step_norm = None
        if self.protein_norm is not None:
            self.register_buffer('protein_norm_mean', torch.zeros(protein_width))
            self.register_buffer('protein_norm_var', torch.ones(protein_width))

    # ---- нормировка вектора белка (SPEC п. 4.2) ---------------------------------------------------
    def set_protein_norm(self, mean, var):
        """Статистики текущего шага обучения (`CONTRACTS.md` § 4.1): тензоры `[P]` с градиентом — среднее и смещённая
        дисперсия побочного пакета (`standardise`); ими **в режиме обучения** нормируется вектор белка. `None, None`
        снимает их: тогда проход в режиме обучения — отказ `RuntimeError`, а не нормировка чужими статистиками."""
        if self.protein_norm is None:
            raise RuntimeError('у этой модели нет нормировки вектора белка (protein_norm = null или нет белкового '
                               'входа, CONTRACTS § 4.1)')
        if mean is None or var is None:
            if mean is not None or var is not None:
                raise ValueError('статистики шага — обе или ни одной')
            self._step_norm = None
            return
        width = self.protein_norm_mean.shape[0]
        if tuple(mean.shape) != (width,) or tuple(var.shape) != (width,):
            raise ValueError('статистики шага — [P] при P = %d, получено %s и %s'
                             % (width, tuple(mean.shape), tuple(var.shape)))
        self._step_norm = (mean, var)

    def update_protein_norm(self, mean, var, momentum):
        """Скользящие статистики после шага (`CONTRACTS.md` § 2.9): `r ← (1 − m)·r + m·s`, без градиента и на месте —
        буферы остаются теми же тензорами, поэтому скользящее среднее весов (`ema_decay`) усредняет их вместе с весами."""
        m = float(momentum)
        with torch.no_grad():
            self.protein_norm_mean.mul_(1.0 - m).add_(mean.detach(), alpha=m)
            self.protein_norm_var.mul_(1.0 - m).add_(var.detach(), alpha=m)

    def _normalised(self, vec):
        """Вектор белка, нормированный по белкам (SPEC п. 4.2): `(v − mean) / √(var + eps)` — статистиками шага в
        режиме обучения, скользящими из буферов в режиме оценки. Без нормировки — сам вектор."""
        if self.protein_norm is None:
            return vec
        if self.training:
            if self._step_norm is None:
                raise RuntimeError('режим обучения: статистики шага нормировки вектора белка не заданы '
                                   '(set_protein_norm, CONTRACTS § 4.1)')
            mean, var = self._step_norm
        else:
            mean, var = self.protein_norm_mean, self.protein_norm_var
        return (vec - mean) / torch.sqrt(var + self.protein_norm['eps'])

    # ---- белковый вектор -------------------------------------------------------------------
    def protein_vector(self, protein, protein_mask, batch):
        """Вектор белка ширины `protein_width`, который получает ствол (после нормировки при `protein_norm`, п. 4.2),
        размноженный на `batch` окон ДНК одного белка. Сырой вектор белковой сети — `self.protein(...)`."""
        if self.protein is None:
            return self.constant.unsqueeze(0).expand(batch, -1)
        vec = self._normalised(self.protein(protein, protein_mask.to(torch.bool)))
        if vec.shape[0] == batch:
            return vec
        if vec.shape[0] == 1:
            return vec.expand(batch, -1)
        raise ValueError('белков %d, окон ДНК %d: окна одного примера подаются с одним белком'
                         % (vec.shape[0], batch))

    # ---- один проход -----------------------------------------------------------------------
    def pass_scores(self, dna, protein, protein_mask, reverse=False, retrieved=None):
        """Оценки **одного** прохода п. 4.1: прямого или обратно-комплементарного, уже развёрнутого
        к координатам прямой цепи. Из двух таких оценок складывается итоговая оценка места (`forward`);
        карта вклада берёт итоговую оценку, среднее двух проходов (п. 5.4, с 03.11.2026), а сравнение
        проходов в точке привязки нужно ориентации карт для логотипа (п. 5.4)."""
        x = dna.flip(1).flip(2) if reverse else dna
        out = self._trunk(x, self.protein_vector(protein, protein_mask, dna.shape[0]), retrieved)
        return out.flip(-1) if reverse else out

    def retrieved_channels(self, dna, retrieved):
        """Три числа о мотиве родственника в каждой позиции `[B, L, 3]` (поле `retrieval`): z-оценка лучшего размещения
        матрицы с центром в позиции по обеим цепям (обрезана до [−4, 12] и поделена на 4), идентичность родственника,
        log2(1 + число сайтов) / 6. `retrieved` — `(pwm [B, 4, W], mu [B], sd [B], identity [B], sites [B])`; строка с
        sd = 0 — родственника нет, все три числа нули. От цепи не зависит: на обратном проходе значение то же. С ключом
        `alpha` z считается к составу **всего поданного окна** (доли оснований окна), поэтому значение в позиции зависит
        от длины и границ окна — вариант разработки, в проход по геному с нарезкой не годится (ревизия 10.11)."""
        batch, _, length = dna.shape
        if retrieved is None:
            return dna.new_zeros(batch, length, 3)
        pwm, mu, sd, identity, sites = retrieved
        width = pwm.shape[2]
        flat = dna.reshape(1, batch * 4, length)
        has = (sd > 0).to(dna.dtype).unsqueeze(1)
        reverse = pwm.flip(1).flip(2)
        if self.retrieval is not None and 'alpha' in self.retrieval:
            # состав окна вместо равных долей: матрица банка — логарифм отношения к 0,25; здесь прямое и обратное
            # размещения приводятся к долям оснований самого окна (как соперник п. 3.3.1 считает к составу генома), и
            # каждое — к z своими средним и разбросом по этим долям; итог — большее из двух z, от цепи не зависит
            q = dna.mean(dim=2).clamp(min=0.02)
            q = q / q.sum(dim=1, keepdim=True)                                   # [B, 4]
            shift = torch.log(q / 0.25).unsqueeze(2)

            def placed(matrix):
                used = (matrix.abs().sum(dim=1, keepdim=True) > 0).to(matrix.dtype)   # столбцы самой матрицы, не поля
                m = (matrix - shift) * used
                mean_col = (q.unsqueeze(2) * m).sum(dim=1)
                var = ((q.unsqueeze(2) * m * m).sum(dim=1) - mean_col * mean_col).sum(dim=1)
                score = F.conv1d(flat, m, padding=width // 2, groups=batch).reshape(batch, length)
                return (score - mean_col.sum(dim=1, keepdim=True)) / torch.sqrt(var.clamp(min=1e-6)).unsqueeze(1)
            z = torch.maximum(placed(pwm), placed(reverse)).clamp(-4.0, 12.0) / 4.0 * has
        else:
            forward = F.conv1d(flat, pwm, padding=width // 2, groups=batch)
            backward = F.conv1d(flat, reverse, padding=width // 2, groups=batch)
            best = torch.maximum(forward, backward).reshape(batch, length)
            z = ((best - mu.unsqueeze(1)) / sd.clamp(min=1e-6).unsqueeze(1)).clamp(-4.0, 12.0) / 4.0 * has
        return torch.stack([z, (identity.unsqueeze(1) * has).expand(batch, length),
                            (torch.log2(1.0 + sites) / 6.0).unsqueeze(1).expand(batch, length) * has], dim=-1)

    def _trunk(self, dna, vec, retrieved=None):
        x = F.gelu(self.stem(dna))
        if self.generate is not None:
            batch, _, length = dna.shape
            # Порождаемые фильтры действуют на четыре канала ДНК (`CONTRACTS.md` § 4.1); у каждого
            # примера пакета свои фильтры, поэтому отклик считается групповой свёрткой.
            weight = self.generate(vec).reshape(batch * self.n_filters, 4, self.filter_width)
            response = F.conv1d(dna.reshape(1, batch * 4, length), weight,
                                padding=self.filter_width // 2, groups=batch)
            x = torch.cat([x, response.reshape(batch, self.n_filters, length)], dim=1)
        x = F.gelu(self.mix(x))
        for number, block in enumerate(self.blocks, start=1):
            film = None
            key = str(number)
            if key in self.film:
                gamma, beta = self.film[key](vec).chunk(2, dim=-1)
                film = (gamma.unsqueeze(-1), beta.unsqueeze(-1))
            x = block(x, film)
        if self.ret_head is None:
            out = self.head(self.head_norm(x).transpose(1, 2)).squeeze(-1)
            if self.retrieval is not None and retrieved is not None:
                # «сложение»: z-оценка мотива родственника (в единицах z, от −4 до 12) с закреплённым весом alpha
                out = out + self.retrieval['alpha'] * 4.0 * self.retrieved_channels(dna, retrieved)[..., 0]
            return out
        features = torch.cat([self.head_norm(x).transpose(1, 2), self.retrieved_channels(dna, retrieved)], dim=-1)
        return self.ret_head(features).squeeze(-1)

    # ---- две цепи --------------------------------------------------------------------------
    def strand_scores(self, dna, protein, protein_mask, retrieved=None):
        """Оценки двух проходов п. 4.1 по отдельности; `reverse` уже развёрнут к координатам прямой
        цепи (`CONTRACTS.md` § 4.1). `model(...)` равна их среднему."""
        return (self.pass_scores(dna, protein, protein_mask, reverse=False, retrieved=retrieved),
                self.pass_scores(dna, protein, protein_mask, reverse=True, retrieved=retrieved))

    def forward(self, dna, protein, protein_mask, retrieved=None):
        """Одна оценка на основание, общая для обеих цепей (SPEC п. 4.1): отрезок считается дважды,
        второй раз обратно-комплементарным, и оценки усредняются после разворота."""
        forward, reverse = self.strand_scores(dna, protein, protein_mask, retrieved)
        return (forward + reverse) / 2


def build(config, protein_input=True):
    """ProkaBind или, при `protein_input=False`, модель без белкового входа п. 4.4 (`CONTRACTS.md`
    § 4.1). Семя не ставится здесь: его ставит тот, кому нужна повторяемость (тест 9 п. 4.11)."""
    return ProkaBind(config, protein_input=protein_input)


def count_parameters(model):
    """Обучаемые параметры модели; у ProkaBind при значениях SPEC — 994 785 (`CONTRACTS.md` § 4.1)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
