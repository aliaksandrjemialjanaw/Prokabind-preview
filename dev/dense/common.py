"""Общие части плотного обучения (разработка по решению владельца 20, «BWH»; `dev/README.md`).

Модель, слагаемые потери и нормировка вектора белка — проекта (`src/prokabind`, код T515). Здесь — только подача:
окна собраны заранее, в шаге десятки сайтов разных белков, у каждого проходы (свой белок, сайт), (свой белок,
подобранное неразмеченное окно), (свой белок, сайт чужого белка), (чужой белок, тот же сайт)."""
import collections
import csv
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'lib'))
os.chdir(ROOT)
import torch                                                          # noqa: E402
from prokabind import train as T, data as D                           # noqa: E402

NIGHT = os.environ.get('PB_NIGHT', '/var/tmp/pb_night')
PACKS = os.path.join(NIGHT, 'packs')
RUNS = os.path.join(NIGHT, 'runs')
SPLIT_NUMBER = 4154109830
CHOICES = {'protein_paths': 'both', 'folds': 5, 'prodoric_in_training': True}
CFG_OURS = os.path.join(ROOT, 'configs/A.json')        # новый рецепт: подмена 0,5 и нормировка
CFG_BAC = os.path.join(ROOT, 'configs/B.json')         # он же с полем bacregdb
W, HALF, CTX = 161, 80, 70                                             # окно, половина, контекст поля видимости
SCORED = W - 2 * CTX                                                   # 21 оцениваемая позиция
LUT = np.full(256, 4, dtype=np.uint8)
for _k, _ch in enumerate('ACGT'):
    LUT[ord(_ch)] = _k
COMP = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def codes(text):
    return LUT[np.frombuffer(text.upper().encode('ascii', 'replace'), dtype=np.uint8)]


def revcomp_codes(a):
    return COMP[a[::-1]]


def group_mean(per_protein, groups):
    by = collections.defaultdict(list)
    for p, v in per_protein.items():
        by[groups.get(p, p)].append(v)
    return float(np.mean([np.mean(v) for v in by.values()])), len(by)


def boot(values, n=10000, seed=SPLIT_NUMBER):
    vals = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


EYE = torch.eye(5)[:, :4]                                              # код 4 (неизвестное основание) — нули
_EYES = {}


def onehot_codes(arr, device=None):
    idx = torch.from_numpy(np.ascontiguousarray(arr).astype(np.int64))
    if device is None or torch.device(device).type == 'cpu':
        return EYE[idx].permute(0, 2, 1).contiguous()
    eye = _EYES.get(str(device))
    if eye is None:
        eye = _EYES[str(device)] = EYE.to(device)
    return eye[idx.to(device)].permute(0, 2, 1).contiguous()


def window_scores(model, x, vec, retrieved=None):
    """Оценки 21 центральной позиции окон 161: среднее прямого и обратно-комплементарного проходов (SPEC п. 4.1).
    `retrieved` — мотив родственника для модели с полем `retrieval` (`model.retrieved_channels`), иначе `None`."""
    if retrieved is None:
        f = model._trunk(x, vec)
        r = model._trunk(x.flip(1).flip(2), vec).flip(-1)
    else:
        f = model._trunk(x, vec, retrieved)
        r = model._trunk(x.flip(1).flip(2), vec, retrieved).flip(-1)
    return ((f + r) / 2)[:, CTX:x.shape[2] - CTX]             # у окон 161 — 21 позиция; у окон пиков — их оцениваемая область


class Retrieval:
    """Банк матриц мотивов и родство (`prep_retrieval.py`): выбор родственника белку среди обучающих белков прогона."""

    def __init__(self, train_proteins, low=0.4, high=0.9):
        pack = pickle.load(open(os.path.join(PACKS, 'retrieval.pkl'), 'rb'))
        self.rel, self.bank, self.width = pack['rel'], pack['pwm'], pack['width']
        self.train = set(train_proteins)
        self.low, self.high = low, high
        self._pool = {}

    def pool(self, protein):
        got = self._pool.get(protein)
        if got is None:
            got = sorted(((v, q) for q, v in self.rel.get(protein, {}).items()
                          if q in self.train and q in self.bank and self.low <= v < self.high), reverse=True)
            self._pool[protein] = got
        return got

    def pick(self, protein, rng=None, none_share=0.0):
        """ближайший родственник (rng = None) или случайный из годных; с долей `none_share` — никого"""
        pool = self.pool(protein)
        if not pool or (rng is not None and rng.random() < none_share):
            return None
        return pool[0] if rng is None else pool[int(rng.integers(len(pool)))]

    def tensors(self, picks, device):
        """picks — список (идентичность, белок-родственник) или None по строкам пакета"""
        n = len(picks)
        pwm = np.zeros((n, 4, self.width), dtype=np.float32)
        mu = np.zeros(n, dtype=np.float32); sd = np.zeros(n, dtype=np.float32)
        ident = np.zeros(n, dtype=np.float32); sites = np.zeros(n, dtype=np.float32)
        for k, pk in enumerate(picks):
            if pk is None:
                continue
            b = self.bank[pk[1]]
            pwm[k] = b['pwm']; mu[k] = b['mu']; sd[k] = b['sd']; ident[k] = pk[0]; sites[k] = b['n']
        return tuple(torch.from_numpy(x).to(device) for x in (pwm, mu, sd, ident, sites))


def interval_scores(s, lo, hi):
    """Логарифм среднего экспонент по [lo, hi) каждой строки (SPEC п. 4.6); lo, hi — тензоры длин строк."""
    n = s.shape[1]
    pos = torch.arange(n, device=s.device).unsqueeze(0)
    mask = (pos >= lo.unsqueeze(1)) & (pos < hi.unsqueeze(1))
    masked = s.masked_fill(~mask, float('-inf'))
    return torch.logsumexp(masked, dim=1) - torch.log((hi - lo).float())


def span_of(width):
    """Место интервала ширины `width` с центром в центре окна, обрезанное до 21 оцениваемой позиции."""
    lo = HALF - (width - 1) // 2
    hi = lo + width
    return max(lo, CTX) - CTX, min(hi, W - CTX) - CTX


def feature_dirs(cfg):
    out = [os.path.join(ROOT, str(cfg['cache_dir']))]
    if cfg.get('bacregdb'):
        out.append(os.path.join(ROOT, str(cfg['bacregdb']['cache'])))
    out.append(os.path.join(ROOT, 'data/cache/esm2_library'))
    return out


class Features:
    """Признаки ESM-2 белков из кэшей проекта и кэша библиотеки; в памяти — float16."""

    def __init__(self, cfg):
        self.dirs = feature_dirs(cfg)
        self.mem = {}

    def path(self, protein):
        for d in self.dirs:
            p = os.path.join(d, protein + '.npy')
            if os.path.exists(p):
                return p
        return None

    def has(self, protein):
        return protein in self.mem or self.path(protein) is not None

    def get(self, protein):
        a = self.mem.get(protein)
        if a is None:
            a = np.load(self.path(protein))
            self.mem[protein] = a
        return torch.from_numpy(a.astype(np.float32))

    def on(self, protein, device):
        """признаки на устройстве счёта (float16 в памяти видеокарты; в float32 переводит вызывающий)"""
        key = (protein, str(device))
        t = self.mem.get(key)
        if t is None:
            t = self.mem[key] = torch.from_numpy(np.load(self.path(protein))).to(device)
        return t
