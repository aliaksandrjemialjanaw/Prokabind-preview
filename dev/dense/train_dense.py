"""Плотное обучение ProkaBind (разработка по решению владельца 20, «BWH»).

Модель и потери — проекта (код T515: оба белковых пути, подмена, нормировка вектора белка со штрафом ковариаций).
Подача: шаг — B сайтов разных белков; у сайта четыре прохода окна 161 основание:
  (свой белок, сайт), (свой белок, неразмеченное окно того же состава), (свой белок, сайт чужого белка), (чужой белок, сайт).
Дорожки:
  coord   — координатные строки захода 0 (точные сайты, PRODORIC; с `--coord bac` и места BacRegDB) с окружением из генома;
            чужой белок — по правилу проекта (`_allowed_foreign`: другая группа родства и другое семейство), сайт того же
            состава (доля A+T на поле видимости в пределах 0,03), без меток чужого у сайта и своих у чужого сайта;
  library — сайты библиотеки SigmoID без места (`dev/library/prep_library.py`): сайт вставляется в центр случайной
            межгенной «подложки» состава белка; чужой белок — белок библиотеки другого семейства HMM либо (доля
            `--same-family`) того же семейства с далёким CR-тегом (расстояние ≥ 4 или другая длина) и без родства ≥ 40%.
Отложено от обучения и измеряется на каждой проверке (доля пар «сайт выше»; 0,5 — случайность):
  CONTROL — контрольная часть (белки, которых не учит ни одна модель);
  INNER   — `--inner SEED` (по умолчанию 0 — выключено): доля групп наших обучающих белков убирается из обучения (и
            родственные им белки библиотеки и BacRegDB). **Пробы на убранных белках — только с `--inner-probes 1`, и это
            взгляд на проверочные данные заходов по SPEC п. 3.10** (белки обучающей части захода 0 — проверочные белки
            заходов 1–4): до фиксации — только по решению владельца. Ночью 09–10.11 пробы печатались в прогонах G…, R…,
            B1 (`dev/inner_held_proteins.tsv`, журнал PLAN 10.11); члены итоговых ансамблей учились с `--inner 0`;
  LIBHELD — доля групп вариантов белков библиотеки, по полосам идентичности с ближайшим обучающим белком.
Запуск: ./.venv/bin/python -I -B dev/dense/train_dense.py TAG [--coord ours|bac|none] [--library 0|1] ...
Выход: {NIGHT}/runs/TAG/: config.json, journal.json (их читает train.load_trained), checkpoints/N.pt (веса скользящего
среднего), model.pt (последняя точка), log.json."""
import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from prokabind.model import build, standardise, covariance_penalty

ap = argparse.ArgumentParser()
ap.add_argument('tag')
ap.add_argument('--coord', default='bac', choices=('ours', 'bac', 'bacpeaks', 'none'))
ap.add_argument('--library', type=int, default=1)
ap.add_argument('--share-library', type=float, default=0.5)
ap.add_argument('--same-family', type=float, default=0.5)
ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--threads', type=int, default=12)
ap.add_argument('--every', type=int, default=100)
ap.add_argument('--recipe', default='A', choices=('A', 'base'))
ap.add_argument('--inner', type=int, default=0)
ap.add_argument('--inner-probes', type=int, default=0, help='пробы на убранных обучающих белках: взгляд на проверочные данные заходов (SPEC п. 3.10) — только по решению владельца')
ap.add_argument('--inner-share', type=float, default=0.2)
ap.add_argument('--lib-held-share', type=float, default=0.08)
ap.add_argument('--cap', type=int, default=10)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--ema', type=float, default=0.999)
ap.add_argument('--feat-dropout', type=float, default=0.0)
ap.add_argument('--lib-weight', type=float, default=0.5)
ap.add_argument('--max-lib-proteins', type=int, default=0)
ap.add_argument('--paths', default='both', choices=('both', 'film_only'))
ap.add_argument('--device', default='cpu')
ap.add_argument('--retrieval', type=int, default=0)
ap.add_argument('--ret-none', type=float, default=0.3)
ap.add_argument('--ret-alpha', type=float, default=0.0)
ap.add_argument('--blind', type=int, default=0)
ap.add_argument('--negs', type=int, default=1)
ap.add_argument('--inner-by', default='variant', choices=('variant', 'related'))
ap.add_argument('--hold-file', default=None, help='JSON со списком «hold» (и «hold_variants»): белки, которых в обучении нет (набор разработки BacRegDB, prep_bacdev.py); расчётов на них обучение не делает')
ap.add_argument('--regprecise', type=float, default=0.0, help='доля пакета из дорожки RegPrecise — сайты на своих местах у своих белков (dev/regprecise/prep_regprecise.py)')
ap.add_argument('--rp-weight', type=float, default=1.0, help='вес примера дорожки RegPrecise')
ap.add_argument('--rp-pack', default='regprecise.pkl', help='пакет дорожки RegPrecise в {PACKS} (части скачивания: regprecise_q1.pkl, regprecise_h1.pkl)')
ap.add_argument('--drop-rows', default=None, help='FILE.json:КЛЮЧ[,КЛЮЧ] — номера строк координатного пакета, которые не учатся (например, bacregdb_origin.json:search_only)')
ap.add_argument('--peaks', type=float, default=0.0, help='доля пакета из дорожки интервалов — пиков DAP-seq (prep_peaks.py)')
ap.add_argument('--peak-mult', type=int, default=0, help='брать только пики, чью вершину накрывает не больше стольких пиков всех белков (0 — все): отсев «горячих» мест')
ap.add_argument('--peak-top', type=int, default=0, help='брать у белка только столько самых сильных пиков (0 — все)')
ap.add_argument('--peak-halfwidth', type=int, default=0, help='интервал пика — вершина ± столько оснований (0 — весь пик в оцениваемой области)')
ap.add_argument('--peak-cap', type=int, default=200, help='предел числа пиков белка в весе его выбора (SPEC п. 4.7)')
ap.add_argument('--drop-groups', type=float, default=0.0, help='доля групп родства наших обучающих белков, убираемая из обучения этого члена ансамбля (по семени); проб на убранных белках нет')
a = ap.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
dev = torch.device(a.device)
if dev.type == 'cuda':
    torch.backends.cudnn.benchmark = True
rng = np.random.default_rng(a.seed)

cfg = json.load(open(CFG_BAC if a.coord in ('bac', 'bacpeaks') else CFG_OURS))
PACK_FILE = {'ours': 'e2_data_ours.pkl', 'bac': 'e2_data_bac.pkl', 'bacpeaks': 'coord_bacpeaks.pkl'}
DROP_ROWS = set()
if a.drop_rows:
    _f, _keys = a.drop_rows.split(':')
    _d = json.load(open(_f if os.path.isabs(_f) else os.path.join(PACKS, _f)))
    for _k in _keys.split(','):
        DROP_ROWS |= set(_d[_k])
HOLD = set()
if a.hold_file:
    _h = json.load(open(a.hold_file if os.path.isabs(a.hold_file) else os.path.join(PACKS, a.hold_file)))
    HOLD = set(_h['hold']) | set(_h.get('hold_variants', ()))
cfg['protein_paths'] = a.paths
if a.recipe == 'base':
    cfg['protein_norm'] = None
    cfg['swap_share'] = 0.0
if a.retrieval:
    cfg['retrieval'] = {'width': 31, 'hidden': 64}
    if a.ret_alpha > 0:
        cfg['retrieval']['alpha'] = a.ret_alpha          # «сложение» с закреплённым весом (model.py)
SWAP = 0.0 if a.blind else float(cfg.get('swap_share') or 0.0)
run_dir = os.path.join(RUNS, a.tag)
os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
raw_cfg = json.dumps({k: v for k, v in cfg.items() if k != 'protein_paths'}, indent=1).encode()
open(os.path.join(run_dir, 'config.json'), 'wb').write(raw_cfg)
json.dump({'config_sha256': hashlib.sha256(raw_cfg).hexdigest(), 'choices': dict(CHOICES, protein_paths=a.paths), 'protein_input': not a.blind,
           'swap_share': cfg.get('swap_share'), 'protein_norm': cfg.get('protein_norm'), 'dense': vars(a)},
          open(os.path.join(run_dir, 'journal.json'), 'w'), indent=1)

feats = Features(cfg)
split_rows = {r['protein']: r for r in read_tsv('data/derived/split.tsv')}
our_group = {p: r['relatedness_group'] for p, r in split_rows.items()}
our_variant = {p: r['variant_group'] for p, r in split_rows.items()}


def held_hash(seed, key, share):
    return int(hashlib.sha256(('%d:%s' % (seed, key)).encode()).hexdigest()[:8], 16) / 16.0 ** 8 < share


# ---------------------------------------------------------------- координатная дорожка
class CoordSet:
    def __init__(self, pack, keep=None):
        n = len(pack['protein'])
        keep = list(range(n)) if keep is None else keep
        self.protein = [pack['protein'][i] for i in keep]
        self.genome = [pack['genome'][i] for i in keep]
        self.start = [pack['start'][i] for i in keep]
        self.end = [pack['end'][i] for i in keep]
        self.pos = pack['pos'][keep]
        self.neg = pack['neg'][keep]
        self.n_neg = [pack['n_neg'][i] for i in keep]
        self.lo = [pack['lo'][i] for i in keep]
        self.hi = [pack['hi'][i] for i in keep]
        self.at = np.asarray([pack['at'][i] for i in keep])
        self.weight = np.asarray([pack['weight'][i] for i in keep], dtype=np.float32) if 'weight' in pack else np.ones(len(keep), dtype=np.float32)
        self.marks = pack['marks']
        self._marks_np = {}
        self.rows_of = collections.defaultdict(list)
        for i, q in enumerate(self.protein):
            self.rows_of[q].append(i)
        self.proteins = sorted(self.rows_of)
        self.allowed = {q: set(pack['allowed'].get(q, ())) for q in self.proteins}
        self.order = np.argsort(self.at, kind='stable')
        self.at_sorted = self.at[self.order]
        self.n = len(keep)

    def touches(self, protein, genome, start, end):
        """задевает ли интервал с запасом 100 оснований метки белка в этом геноме (метки слиты и упорядочены)"""
        key = (protein, genome)
        arr = self._marks_np.get(key)
        if arr is None:
            m = self.marks.get(key, ())
            arr = self._marks_np[key] = (np.array([x for x, _ in m], dtype=np.int64), np.array([y for _, y in m], dtype=np.int64))
        starts, ends = arr
        if starts.size == 0:
            return False
        k = int(np.searchsorted(ends, start - 100, 'right'))        # первая метка, чей конец правее начала интервала
        return k < starts.size and starts[k] < end + 100

    def partner(self, i, rng):
        y = self.protein[i]
        if not self.allowed[y]:
            return None
        lo = np.searchsorted(self.at_sorted, self.at[i] - 0.03, 'left')
        hi = np.searchsorted(self.at_sorted, self.at[i] + 0.03, 'right')
        if hi <= lo:
            return None
        for _ in range(30):
            j = int(self.order[int(rng.integers(lo, hi))])
            x = self.protein[j]
            if x not in self.allowed[y]:
                continue
            if rng.random() > min(1.0, a.cap / len(self.rows_of[x])):
                continue
            if self.touches(x, self.genome[i], self.start[i], self.end[i]):
                continue
            if self.touches(y, self.genome[j], self.start[j], self.end[j]):
                continue
            return j
        return None

    def example(self, i, rng, j='draw', k=None):
        """окна примера: [сайт, неразмеченное, (сайт чужого, сайт)], белки, место интервала"""
        if j == 'draw':
            j = None if a.blind else self.partner(i, rng)
        y = self.protein[i]
        if k is None:
            ks = [int(t) for t in rng.permutation(self.n_neg[i])[:a.negs]]
            ks += [ks[-1]] * (a.negs - len(ks))             # подобранных окон меньше, чем нужно, — последнее повторяется
        else:
            ks = [k]
        wins, who = [self.pos[i]] + [self.neg[i][t] for t in ks], [y] * (1 + len(ks))
        if j is not None:
            wins += [self.pos[j], self.pos[i]]
            who += [y, self.protein[j]]
        return wins, who, (self.lo[i], self.hi[i]), float(self.weight[i])


# ---------------------------------------------------------------- дорожка библиотеки
def tag_distance(t1, t2):
    if not t1 or not t2 or 'crtag' in t1.lower() or 'crtag' in t2.lower() or ';' in t1 or ';' in t2:
        return None
    if len(t1) != len(t2):
        return 99
    return sum(1 for x, y in zip(t1, t2) if x != y)


class LibSet:
    def __init__(self, lib, proteins, hosts):
        self.meta, self.rel = lib['meta'], lib['rel']
        want = set(proteins)
        keep = [i for i, q in enumerate(lib['rows']['protein']) if q in want]
        self.protein = [lib['rows']['protein'][i] for i in keep]
        self.site = [lib['rows']['site'][i] for i in keep]
        self.width = [lib['rows']['width'][i] for i in keep]
        self.rows_of = collections.defaultdict(list)
        for i, q in enumerate(self.protein):
            self.rows_of[q].append(i)
        self.proteins = sorted(self.rows_of)
        self.by_family = collections.defaultdict(list)
        for q in self.proteins:
            self.by_family[self.meta[q]['tf_hmm']].append(q)
        self.host_win, self.host_at = hosts['win'], hosts['at']
        self.n = len(keep)

    def host(self, at, rng):
        lo = np.searchsorted(self.host_at, at - 0.02, 'left')
        hi = np.searchsorted(self.host_at, at + 0.02, 'right')
        if hi - lo < 20:
            lo, hi = max(0, lo - 200), min(len(self.host_at), hi + 200)
        return self.host_win[int(rng.integers(lo, hi))]

    def spliced(self, i, at, rng):
        w = self.width[i]
        site = self.site[i] if rng.random() < 0.5 else revcomp_codes(self.site[i])
        out = self.host(at, rng).copy()
        lo = HALF - (w - 1) // 2
        out[lo:lo + w] = site
        return out

    def partner(self, i, rng):
        y = self.protein[i]
        my, fam = self.meta[y], self.meta[y]['tf_hmm']
        related = self.rel.get(y, {})
        same = rng.random() < a.same_family
        for attempt in range(30):
            if same and attempt < 15 and len(self.by_family[fam]) > 1:
                x = self.by_family[fam][int(rng.integers(len(self.by_family[fam])))]
                d = tag_distance(my['crtag'], self.meta[x]['crtag'])
                if x == y or x in related or d is None or d < 4:
                    continue
            else:
                x = self.proteins[int(rng.integers(len(self.proteins)))]
                if x == y or self.meta[x]['tf_hmm'] == fam or x in related:
                    continue
            if rng.random() > min(1.0, a.cap / len(self.rows_of[x])):
                continue
            return self.rows_of[x][int(rng.integers(len(self.rows_of[x])))]
        return None

    def example(self, i, rng, j='draw', k=None):
        if j == 'draw':
            j = None if a.blind else self.partner(i, rng)
        y = self.protein[i]
        at = float(np.clip(self.meta[y]['at'] + rng.normal(0.0, 0.03), 0.15, 0.85))
        pos = self.spliced(i, at, rng)
        wins, who = [pos] + [self.host(at, rng) for _ in range(a.negs)], [y] * (1 + a.negs)
        if j is not None:
            wins += [self.spliced(j, at, rng), pos]
            who += [y, self.protein[j]]
        return wins, who, span_of(self.width[i]), a.lib_weight


# ---------------------------------------------------------------- дорожка интервалов: пики DAP-seq
class PeakSet:
    """Пик — интервал «сайт где-то здесь» (SPEC п. 4.5; `prep_peaks.py`). Пример: окно вокруг вершины пика; неразмеченное
    окно того же состава вне пиков белка и его вариантов; пик неродственного белка, не задевающий пики своего («чужой
    сайт»); тот же пик под белком, у которого здесь пика нет («подмена»). Оценка — логарифм средней экспоненты по
    части пика внутри оцениваемой области (`interval_scores`); границы — у всех окон примера одни."""

    def __init__(self, pack, part, proteins=None, top=0, halfwidth=0, mult=0):
        d = pack[part]
        keep = [i for i, q in enumerate(d['protein']) if proteins is None or q in proteins]
        if mult:
            # только места, где пиков разных белков немного. Кратность считается здесь, по пикам самого набора (для
            # обучающего — по обучающим пикам): поле `mult` пакета считано вместе с пиками контрольных белков, и отсев
            # по нему давал бы проверочным меткам менять обучение (SPEC п. 4.7; ревизия круга 10.11, находка Н2)
            _cov = np.zeros(int(max(d['end'])) + 2, dtype=np.int32)
            np.add.at(_cov, np.asarray(d['start']), 1); np.add.at(_cov, np.asarray(d['end']), -1)
            _depth = np.cumsum(_cov)
            keep = [i for i in keep if _depth[int(d['summit'][i])] <= mult]
        if top:                                            # только самые сильные пики белка
            by = collections.defaultdict(list)
            for i in keep:
                by[d['protein'][i]].append(i)
            keep = sorted(i for q, rows in by.items() for i in sorted(rows, key=lambda t: -d['strength'][t])[:top])
        self.seq, self.WP, self.RP = pack['seq'], pack['WP'], pack['RP']
        self.H = (self.WP - 1) // 2
        self.protein = [d['protein'][i] for i in keep]
        for key in ('start', 'end', 'summit', 'lo', 'hi'):
            setattr(self, key, np.asarray([d[key][i] for i in keep], dtype=np.int64))
        if halfwidth:                                      # интервал — вершина ± halfwidth (вершина стоит в позиции RP)
            self.lo = np.maximum(self.lo, self.RP - halfwidth)
            self.hi = np.minimum(self.hi, self.RP + halfwidth + 1)
        self.weight = np.asarray([d['weight'][i] for i in keep], dtype=np.float32)
        self.at = np.asarray([d['at'][i] for i in keep], dtype=np.float32)
        self.rows_of = collections.defaultdict(list)
        for i, q in enumerate(self.protein):
            self.rows_of[q].append(i)
        self.proteins = sorted(self.rows_of)
        have = set(self.proteins)
        self.marks = pack['marks']
        self.allowed = {q: set(pack['allowed'].get(q, ())) & have for q in self.proteins}
        self.bg_c, self.bg_at = pack['bg_centres'], pack['bg_at']
        self.order = np.argsort(self.at, kind='stable')
        self.at_sorted = self.at[self.order]
        self.n = len(keep)

    def window(self, c):
        return self.seq[c - self.H:c + self.H + 1]

    def touches(self, protein, start, end, margin=100):
        starts, ends = self.marks[protein]
        if starts.size == 0:
            return False
        k = int(np.searchsorted(ends, start - margin, 'right'))
        return k < starts.size and starts[k] < end + margin

    def negative(self, i, rng):
        y = self.protein[i]
        lo = int(np.searchsorted(self.bg_at, self.at[i] - 0.02, 'left'))
        hi = int(np.searchsorted(self.bg_at, self.at[i] + 0.02, 'right'))
        c = None
        for _ in range(40):
            c = int(self.bg_c[int(rng.integers(lo, hi))]) if hi > lo else int(self.bg_c[int(rng.integers(self.bg_c.size))])
            if not self.touches(y, c - self.RP, c + self.RP + 1):
                return c
        return c

    def partner(self, i, rng):
        y = self.protein[i]
        if not self.allowed[y]:
            return None
        lo = int(np.searchsorted(self.at_sorted, self.at[i] - 0.03, 'left'))
        hi = int(np.searchsorted(self.at_sorted, self.at[i] + 0.03, 'right'))
        if hi <= lo:
            return None
        for _ in range(30):
            j = int(self.order[int(rng.integers(lo, hi))])
            x = self.protein[j]
            if x not in self.allowed[y]:
                continue
            if rng.random() > min(1.0, a.peak_cap / len(self.rows_of[x])):
                continue
            if self.touches(x, int(self.start[i]), int(self.end[i])):     # у чужого белка здесь свой пик — подмена не годится
                continue
            if self.touches(y, int(self.start[j]), int(self.end[j])):     # у своего белка пик на месте чужого — он не чужой
                continue
            return j
        return None

    def example(self, i, rng, j='draw'):
        if j == 'draw':
            j = None if a.blind else self.partner(i, rng)
        y = self.protein[i]
        pos = self.window(int(self.summit[i]))
        wins, who = [pos, self.window(self.negative(i, rng))], [y, y]
        if j is not None:
            wins += [self.window(int(self.summit[j])), pos]
            who += [y, self.protein[j]]
        return wins, who, (int(self.lo[i]), int(self.hi[i])), float(self.weight[i])


# ---------------------------------------------------------------- наборы
C = pickle.load(open(os.path.join(PACKS, 'e2_control.pkl'), 'rb'))
C['weight'] = [1.0] * len(C['protein'])
CT = CoordSet(C)
held_ours = set()
TR = IN = None
if a.coord != 'none':
    P = pickle.load(open(os.path.join(PACKS, PACK_FILE[a.coord]), 'rb'))
    pack_proteins = set(P['protein'])
    if a.inner:
        # отложенные обучающие белки: по группам вариантов (как настоящее разбиение — у отложенного белка остаются
        # обучающие родственники 40–90%, полосы считаются ниже) или по группам родства (тогда родственников нет)
        key_of = our_variant if a.inner_by == 'variant' else our_group
        held_ours = {q for q in pack_proteins if q in key_of and held_hash(a.inner, key_of[q], a.inner_share)}
    drop_bac = set()
    if a.coord in ('bac', 'bacpeaks') and held_ours:
        for r in read_tsv('data/derived/sigmoid/proteins.tsv'):
            rel = {x.split(':')[0] for x in r['related'].split(';') if x}
            if r['protein'] not in split_rows and rel & held_ours:
                drop_bac.add(r['protein'])
    if a.drop_groups > 0:
        # член ансамбля без случайной части групп родства наших белков (и родственных им белков BacRegDB / библиотеки):
        # убранные белки просто не учатся — проб и оценок на них нет (SPEC п. 3.10)
        dropped = {q for q in our_group if held_hash(a.seed + 7000, our_group[q], a.drop_groups)}
        held_ours |= dropped
        for r in read_tsv('data/derived/sigmoid/proteins.tsv'):
            rel = {x.split(':')[0] for x in r['related'].split(';') if x}
            if r['protein'] not in split_rows and rel & dropped:
                drop_bac.add(r['protein'])
    keep_tr = [i for i, q in enumerate(P['protein']) if q not in held_ours and q not in drop_bac and q not in HOLD and feats.has(q)
               and i not in DROP_ROWS]
    TR = CoordSet(P, keep_tr)
    if held_ours and a.inner and a.inner_probes:
        IN = CoordSet(P, [i for i, q in enumerate(P['protein']) if q in held_ours])
LB = LH = None
lib_band = {}
if a.library:
    lib = pickle.load(open(os.path.join(PACKS, 'library.pkl'), 'rb'))
    hosts = np.load(os.path.join(PACKS, 'hosts.npz'))
    hosts = {'win': hosts['win'], 'at': hosts['at']}
    closed = {'fold0', 'control', 'holdout'}
    ok = [q for q, m in lib['meta'].items() if not (m['related_parts'] & closed) and feats.has(q)
          and not (set(m['related']) & held_ours) and q not in HOLD]
    ok.sort()
    if a.max_lib_proteins and len(ok) > a.max_lib_proteins:
        ok = sorted(ok, key=lambda q: hashlib.sha256(q.encode()).hexdigest())[:a.max_lib_proteins]
    lib_held = {q for q in ok if held_hash(a.seed + 1000, lib['meta'][q]['variant_group'], a.lib_held_share)
                and not lib['meta'][q]['identical']}
    lib_train = [q for q in ok if q not in lib_held]
    LB = LibSet(lib, lib_train, hosts)
    LH = LibSet(lib, sorted(lib_held), hosts) if lib_held else None
    train_all = set(lib_train) | (set(TR.proteins) if TR is not None else set())
    for q in lib_held:
        best = max([v for x, v in lib['rel'].get(q, {}).items() if x in train_all] +
                   [v for x, v in lib['meta'][q]['related'].items() if x in train_all] + [0.0])
        lib_band[q] = '>=70' if best >= 0.7 else '40-70' if best >= 0.4 else '<40'

assert not (HOLD and LB is not None), 'дорожка библиотеки с --hold-file: родственники белков разработки из библиотеки не снимаются (ревизия круга 10.11) — сначала выровнять правило с rp_ok'
RP = None
if a.regprecise > 0:
    # дорожка RegPrecise: регулятор не учится, если родственен (≥ 40%) белку закрытых частей (п. 2.7.3), убранному нашему
    # белку этого прогона или белку разработки BacRegDB (или совпадает с ним)
    rp = pickle.load(open(os.path.join(PACKS, a.rp_pack), 'rb'))
    _closed = {'fold0', 'control', 'holdout'}

    def rp_ok(q):
        m = rp['meta'][q]
        if m['related_parts'] & _closed or set(m['related']) & held_ours:
            return False
        if q in HOLD or set(m['related_bacregdb']) & HOLD:
            return False
        return feats.has(q)
    _ok = {q for q in rp['proteins'] if rp_ok(q)}
    rp['weight'] = [a.rp_weight] * len(rp['protein'])
    RP = CoordSet(rp, [i for i, q in enumerate(rp['protein']) if q in _ok])
    print('regprecise rows %d proteins %d (of %d in the pack)' % (RP.n, len(RP.proteins), len(rp['proteins'])), flush=True)
PK = PKC = None
pk_groups = {}
if a.peaks > 0:
    assert a.negs == 1 and not a.retrieval, 'дорожка пиков: одно неразмеченное окно, без поля retrieval'
    pk_pack = pickle.load(open(os.path.join(PACKS, 'peaks.pkl'), 'rb'))
    pk_groups = pk_pack['groups']
    PK = PeakSet(pk_pack, 'train', {q for q in set(pk_pack['train']['protein']) if q not in held_ours and q not in HOLD and feats.has(q)},
                 top=a.peak_top, halfwidth=a.peak_halfwidth, mult=a.peak_mult)
    PKC = PeakSet(pk_pack, 'control', {q for q in set(pk_pack['control']['protein']) if feats.has(q)}, halfwidth=a.peak_halfwidth,
                  mult=a.peak_mult)
print('coord train rows %s proteins %s | library train rows %s proteins %s | held: inner %s, library %s %s | control rows %d'
      % (TR.n if TR else 0, len(TR.proteins) if TR else 0, LB.n if LB else 0, len(LB.proteins) if LB else 0,
         len(IN.proteins) if IN else 0, len(LH.proteins) if LH else 0, dict(collections.Counter(lib_band.values())), CT.n), flush=True)
print('peaks train rows %s proteins %s | control peaks %s proteins %s | hold file proteins %d | dropped groups share %.2f (our proteins out: %d)'
      % (PK.n if PK else 0, len(PK.proteins) if PK else 0, PKC.n if PKC else 0, len(PKC.proteins) if PKC else 0, len(HOLD), a.drop_groups,
         len(held_ours)), flush=True)

# ---------------------------------------------------------------- модель
model = build(cfg, protein_input=not a.blind).to(dev)
train_proteins = sorted((set(TR.proteins) if TR is not None else set()) | (set(LB.proteins) if LB is not None else set())
                        | (set(PK.proteins) if PK is not None else set()) | (set(RP.proteins) if RP is not None else set()))
json.dump(train_proteins, open(os.path.join(run_dir, 'train_proteins.json'), 'w'))
RET = Retrieval(train_proteins) if a.retrieval else None
inner_band = {}
if IN is not None and os.path.exists(os.path.join(PACKS, 'retrieval.pkl')):
    _rel = (RET.rel if RET is not None else pickle.load(open(os.path.join(PACKS, 'retrieval.pkl'), 'rb'))['rel'])
    _tr = set(train_proteins)
    for q in IN.proteins:
        best = max([v for x, v in _rel.get(q, {}).items() if x in _tr] + [0.0])
        inner_band[q] = '>=70' if best >= 0.7 else '40-70' if best >= 0.4 else '<40'
    print('inner bands: %s' % dict(collections.Counter(inner_band.values())), flush=True)
norm = None if a.blind else model.protein_norm
ema = copy.deepcopy(model)
ema.eval()


def raw_vectors(net, plist, train_mode):
    if dev.type == 'cpu':
        # по одному белку, без заполнения: на процессоре заполнение до самого длинного белка шага втрое дороже
        out = []
        for p in plist:
            f = feats.get(p)
            if train_mode and a.feat_dropout > 0:
                f = f * (torch.rand(1, f.shape[1]) >= a.feat_dropout).float() / (1.0 - a.feat_dropout)
            out.append(net.protein(f.unsqueeze(0), torch.ones(1, f.shape[0], dtype=torch.bool)))
        return torch.cat(out)
    out = []
    for i in range(0, len(plist), 96):                    # на видеокарте — пакетами с заполнением и маской
        # признаки лежат в обычной памяти (float16) и уходят на видеокарту пакетом шага: на ней они заняли бы ~2,5 ГБ
        # у каждого обучения, и четыре обучения разом не помещались
        fs = [feats.on(p, 'cpu') for p in plist[i:i + 96]]
        longest = max(f.shape[0] for f in fs)
        xc = torch.zeros(len(fs), longest, fs[0].shape[1], dtype=torch.float16)
        mc = torch.zeros(len(fs), longest, dtype=torch.bool)
        for k, f in enumerate(fs):
            xc[k, :f.shape[0]] = f
            mc[k, :f.shape[0]] = True
        x, mask = xc.to(dev).float(), mc.to(dev)
        if train_mode and a.feat_dropout > 0:
            x = x * (torch.rand(len(fs), 1, x.shape[2], device=dev) >= a.feat_dropout).float() / (1.0 - a.feat_dropout)
        out.append(net.protein(x, mask))
    return torch.cat(out)


def vectors(net, plist, train_mode):
    if a.blind:
        return net.constant.unsqueeze(0).expand(len(plist), -1), None
    raw = raw_vectors(net, plist, train_mode)
    if norm is None:
        return raw, None
    if train_mode:
        z, mean, var = standardise(raw.float(), norm['eps'])
        net.update_protein_norm(mean, var, norm['momentum'])
        return z, covariance_penalty(z)
    return (raw - net.protein_norm_mean) / torch.sqrt(net.protein_norm_var + norm['eps']), None


def run_examples(net, examples, vec_of, ret_of=None):
    wins, who, lo, hi, has = [], [], [], [], []
    n_u = a.negs
    for w_, p_, (l_, h_), _ in examples:
        wins += w_; who += p_
        lo.append(l_); hi.append(h_); has.append(len(w_) == 3 + n_u)
    retrieved = RET.tensors([ret_of[q] for q in who], dev) if RET is not None else None
    s = window_scores(net, onehot_codes(np.stack(wins), dev), torch.stack([vec_of[q] for q in who]), retrieved)
    idx, at = [], 0
    for h_ in has:
        idx.append(at); at += (3 + n_u) if h_ else (1 + n_u)
    idx = torch.tensor(idx, device=dev); has_t = torch.tensor(has, device=dev)
    lo_t, hi_t = torch.tensor(lo, device=dev), torch.tensor(hi, device=dev)
    own = interval_scores(s[idx], lo_t, hi_t)
    unl = torch.stack([interval_scores(s[idx + 1 + t], lo_t, hi_t) for t in range(n_u)], dim=1)       # [примеров, n_u]
    j2 = torch.where(has_t, idx + 1 + n_u, idx); j3 = torch.where(has_t, idx + 2 + n_u, idx)
    foreign = interval_scores(s[j2], lo_t, hi_t)
    swap = interval_scores(s[j3], lo_t, hi_t)
    return own, unl, foreign, swap, has_t


# ---------------------------------------------------------------- закреплённые пробы
erng = np.random.default_rng(SPLIT_NUMBER + 7)


def fixed(S, per_protein, total):
    rows = []
    for q in S.proteins:
        r = S.rows_of[q]
        rows += [r[int(t)] for t in erng.permutation(len(r))[:per_protein]]
    if len(rows) > total:
        rows = [rows[int(t)] for t in erng.permutation(len(rows))[:total]]
    return [(S.protein[i], S.example(i, erng)) for i in rows]


PROBES = collections.OrderedDict()
if TR is not None:
    PROBES['train'] = fixed(TR, 2, 400)
PROBES['control'] = fixed(CT, 12, 10000)
CP = None
if os.path.exists(os.path.join(PACKS, 'control_peaks.pkl')):
    # сайты мотива MEME в пиках контрольных белков StrepTRN (prep_peaksites.py, prep_coord.py control_peaks)
    CPK = pickle.load(open(os.path.join(PACKS, 'control_peaks.pkl'), 'rb'))
    CP = CoordSet(CPK)
    PROBES['ctrlpeaks'] = fixed(CP, 20, 600)
if IN is not None and a.inner_probes:
    PROBES['inner'] = fixed(IN, 6, 10000)
if PKC is not None:
    PROBES['ctrlpk'] = fixed(PKC, 10, 600)             # пики белков контрольной части: окно пика, фон, чужой пик, подмена
    PROBES['trainpk'] = fixed(PK, 2, 600)              # пики обучающих белков: учится ли дорожка вообще
if LB is not None:
    PROBES['libtrain'] = fixed(LB, 1, 300)
    if LH is not None:
        PROBES['libheld'] = fixed(LH, 3, 900)
BANDS = {'control': C['band'], 'libheld': lib_band, 'inner': inner_band}
GROUPS = {'control': C['groups'], 'inner': our_group, 'ctrlpeaks': our_group, 'ctrlpk': pk_groups, 'trainpk': pk_groups, 'libheld': {q: lib['meta'][q]['variant_group'] for q in lib_band} if LB is not None else {}}
print('probes: ' + ', '.join('%s %d (%d with a partner)' % (k, len(v), sum(1 for _, e in v if len(e[0]) == 3 + a.negs)) for k, v in PROBES.items()), flush=True)


def evaluate(net, name):
    items = PROBES[name]
    if not items:
        return {'unl': None, 'foreign': None, 'swap': None}
    plist = sorted({q for _, e in items for q in e[1]})
    with torch.no_grad():
        v, _ = vectors(net, plist, False)
        vec_of = {q: v[k] for k, q in enumerate(plist)}
        ret_of = {q: RET.pick(q) for q in plist} if RET is not None else None
        parts = [run_examples(net, [e for _, e in items[i:i + 64]], vec_of, ret_of) for i in range(0, len(items), 64)]
        o, u, f, s, has = (torch.cat([p[k] for p in parts]).cpu().numpy() for k in range(5))
        u = u[:, 0]
    prot = np.array([q for q, _ in items])
    out = {}
    for key, other, sel in (('unl', u, np.ones(len(o), bool)), ('foreign', f, has), ('swap', s, has)):
        w_ = ((o[sel] > other[sel]) + 0.5 * (o[sel] == other[sel])).astype(float)
        per = collections.defaultdict(list)
        for q, val in zip(prot[sel], w_):
            per[q].append(val)
        per = {q: float(np.mean(val)) for q, val in per.items()}
        groups = GROUPS.get(name, {})
        out[key] = group_mean(per, groups)[0] if per else None
        for bname in ('>=70', '40-70', '<40'):
            if name in BANDS:
                sel_b = {q: val for q, val in per.items() if BANDS[name].get(q) == bname}
                out[key + ' ' + bname] = group_mean(sel_b, groups)[0] if sel_b else None
    return out


# ---------------------------------------------------------------- обучение
sp = torch.nn.functional.softplus
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
w_coord = None
if TR is not None:
    w_coord = np.array([min(len(TR.rows_of[q]), a.cap) for q in TR.proteins], dtype=float); w_coord /= w_coord.sum()
w_lib = None
if LB is not None:
    w_lib = np.array([min(len(LB.rows_of[q]), a.cap) for q in LB.proteins], dtype=float); w_lib /= w_lib.sum()
n_lib = int(round(a.batch * a.share_library)) if LB is not None else 0
if TR is None:
    n_lib = a.batch
w_rp = None
n_rp = 0
if RP is not None:
    w_rp = np.array([min(len(RP.rows_of[q]), a.cap) for q in RP.proteins], dtype=float); w_rp /= w_rp.sum()
    n_rp = min(int(round(a.batch * a.regprecise)), a.batch - n_lib)
w_pk = None
n_pk = 0
if PK is not None:
    w_pk = np.array([min(len(PK.rows_of[q]), a.peak_cap) for q in PK.proteins], dtype=float); w_pk /= w_pk.sum()
    n_pk = min(int(round(a.batch * a.peaks)), a.batch - n_lib - n_rp)
log = []
t0 = time.time()
f3 = lambda x: ' none' if x is None else '%.3f' % x
model.train()
for step in range(a.steps + 1):
    if step % a.every == 0 or step == a.steps:
        net = ema if step >= 200 else model
        net.eval()
        rec = {'step': step, 'seconds': round(time.time() - t0)}
        for name in PROBES:
            rec[name] = evaluate(net, name)
        model.train()
        log.append(rec)
        line = 'step %5d %5ds' % (step, rec['seconds'])
        for name in PROBES:
            r = rec[name]
            line += ' | %s unl %s for %s swap %s' % (name.upper(), f3(r['unl']), f3(r['foreign']), f3(r['swap']))
        print(line, flush=True)
        for name in ('control', 'inner', 'libheld'):
            if name in rec and BANDS.get(name):
                r = rec[name]
                print('            %s by band (>=70 / 40-70 / <40): unl %s %s %s | foreign %s %s %s | swap %s %s %s' % (
                    name.upper(), f3(r.get('unl >=70')), f3(r.get('unl 40-70')), f3(r.get('unl <40')),
                    f3(r.get('foreign >=70')), f3(r.get('foreign 40-70')), f3(r.get('foreign <40')),
                    f3(r.get('swap >=70')), f3(r.get('swap 40-70')), f3(r.get('swap <40'))), flush=True)
        state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save(state, os.path.join(run_dir, 'checkpoints', '%d.pt' % step))
        torch.save(state, os.path.join(run_dir, 'model.pt'))
        json.dump({'tag': a.tag, 'args': vars(a), 'log': log}, open(os.path.join(run_dir, 'log.json'), 'w'), indent=1)
        if step == a.steps:
            break
    lr = a.lr * min(1.0, (step + 1) / 100.0)
    if step >= 100:
        lr = 0.1 * a.lr + 0.9 * a.lr * 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, a.steps - 100)))
    for g in opt.param_groups:
        g['lr'] = lr
    examples = []
    for q in (rng.choice(len(LB.proteins), size=n_lib, p=w_lib) if n_lib else ()):
        y = LB.proteins[int(q)]
        examples.append(LB.example(LB.rows_of[y][int(rng.integers(len(LB.rows_of[y])))], rng))
    for q in (rng.choice(len(RP.proteins), size=n_rp, p=w_rp) if n_rp else ()):
        y = RP.proteins[int(q)]
        examples.append(RP.example(RP.rows_of[y][int(rng.integers(len(RP.rows_of[y])))], rng))
    n_co = a.batch - n_lib - n_pk - n_rp
    for q in (rng.choice(len(TR.proteins), size=n_co, p=w_coord) if n_co > 0 and TR is not None else ()):
        y = TR.proteins[int(q)]
        examples.append(TR.example(TR.rows_of[y][int(rng.integers(len(TR.rows_of[y])))], rng))
    examples_pk = []
    for q in (rng.choice(len(PK.proteins), size=n_pk, p=w_pk) if n_pk else ()):
        y = PK.proteins[int(q)]
        examples_pk.append(PK.example(PK.rows_of[y][int(rng.integers(len(PK.rows_of[y])))], rng))
    plist = sorted({q for e in examples + examples_pk for q in e[1]})
    v, penalty = vectors(model, plist, True)
    vec_of = {q: v[k] for k, q in enumerate(plist)}
    ret_of = {q: RET.pick(q, rng, a.ret_none) for q in plist} if RET is not None else None
    o, u, f, s, has = run_examples(model, examples, vec_of, ret_of)
    if examples_pk:                                     # окна пиков другой длины — отдельным проходом, потеря общая
        o2, u2, f2, s2, has2 = run_examples(model, examples_pk, vec_of, ret_of)
        o, u, f, s, has = torch.cat([o, o2]), torch.cat([u, u2]), torch.cat([f, f2]), torch.cat([s, s2]), torch.cat([has, has2])
    wts = torch.tensor([e[3] for e in examples + examples_pk], device=dev)
    hasf = has.float()
    l_u = sp(1 - (o.unsqueeze(1) - u)).mean(dim=1)          # неразмеченные пары входят средней (SPEC п. 4.6)
    l_f = sp(1 - (o - f)) * hasf
    l_s = sp(1 - (o - s)) * hasf
    present = 1.0 + hasf * (1.0 + SWAP)
    if a.blind:
        loss = (wts * l_u * 2.0).mean()            # модель без белкового входа: дорожка «чужой белок» выключена (SPEC п. 4.4)
    else:
        loss = (wts * (l_u + l_f + SWAP * l_s) * ((2.0 + SWAP) / present)).mean()
    if penalty is not None:
        loss = loss + norm['w_cov'] * penalty
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    beta = min(a.ema, (1.0 + step) / (10.0 + step))
    with torch.no_grad():
        for (k, e_), (_, m_) in zip(ema.state_dict().items(), model.state_dict().items()):
            if e_.is_floating_point():
                e_.mul_(beta).add_(m_, alpha=1.0 - beta)
            else:
                e_.copy_(m_)
print('done', a.tag, flush=True)
