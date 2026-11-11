"""Программа ProkaBind: запуск и показ (SPEC раздел 5; `CONTRACTS.md` § 4.9, § 3.0.1). Пишет задача прототипа T16.

Одна команда в терминале (п. 5.7), два режима:

* `run` — белок и геном → каталог результатов: дорожка оценок (TSV и bigWig), маска, список мест, карты вклада,
  логотип, веса сбора по остаткам, рисунки, сводка и **описание запуска** `run.json`;
* `show` — команда показа п. 5.8: считает то же, что `run`, — отдельной модели и отдельных правил у неё нет, — и
  сверх того рисует дорожку вокруг первых мест с подписанными координатами и картой вклада. До фиксации показ
  ограничен двумя наборами белков п. 5.8; команда сама проверяет допустимость белка **до** счёта и записывает
  каждый показ в журнал показов (`CONTRACTS.md` § 3.0.1).

Запуск из корня проекта: `PYTHONPATH=src ./.venv/bin/python -B -m prokabind show --protein … --genome … --out …`.

Все файлы каталога, **кроме описания запуска**, при одном входе на том же железе и той же сборке библиотек побайтно
одинаковы (п. 5.7): время, команда, пути и устройство живут только в `run.json`.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import figures as FIG
from . import inputs as IN
from . import output as OUT

ROOT = Path(__file__).resolve().parents[2]

#: По умолчанию печатаются первые 100 мест на геном — предел показа, а не утверждение, что сайтов сто (п. 5.3).
TOP = 100
#: Карта вклада — для верхних 50 мест, окно ±70 (п. 5.4).
MAPS = 50
HALF = 70
#: Пометка «рядом есть другое место»: другое место печатаемого списка ближе 50 оснований (п. 5.3).
NEAR = 50
#: «больший проход» — если разность проходов больше 0,1 стандартного отклонения дорожки (п. 5.3).
PASS_SD = 0.1
#: Окно дорожки на рисунке показа — ±250 оснований от точки привязки.
FIGURE_HALF = 250
FIGURE_PLACES = 10
#: Порог родства п. 5.8 и полосы п. 3.1; покрытие — более короткого белка (`CONTRACTS.md` § 2.6).
MIN_COVERAGE = 0.8
SHOW_IDENTITY = 40.0
VARIANT_IDENTITY = 90.0

DEFAULT_MODEL = 'work/tasks/T14/runs/film_only_no'
DEFAULT_CONFIG = 'work/config/train.json'
DEFAULT_ORIENTATION_REPORT = 'work/tasks/T14/report.json'
DEFAULT_DATA = 'data/derived'
DEFAULT_PROJECT_CACHE = 'data/cache/esm2'
DEFAULT_JOURNAL = 'work/tasks/T16/journal.tsv'
JOURNAL_COLUMNS = ['protein', 'genome', 'date', 'model', 'set', 'fasta', 'max_identity', 'nearest_outside']

CAVEAT_SCALE = ('Шкала оценок не калибрована и уровень её не задан: одно и то же число для двух разных белков не '
                'значит одного и того же; оценки разных белков между собой не сравниваются (SPEC пп. 5.2, 5.5).')
CAVEAT_LIST = ('Список — порядок предсказаний модели, а не перечень настоящих сайтов; первые %d мест — предел показа, '
               'а не утверждение, что сайтов столько (п. 5.3). «Мест выше на миллион оснований» — плотность, а не доля '
               'ошибок. Высокий верх списка бывает и у неактивного белка: состояние белка программа не определяет. '
               '«Ближайший ген» значит только расстояние, а не регуляцию (п. 5.5).')
CAVEAT_PASS = ('«Больший проход» — тот из двух проходов модели (прямой отрезок или его обратное дополнение), что дал в '
               'точке привязки большую оценку, если разность больше 0,1 стандартного отклонения дорожки, иначе «проходы '
               'близки». Это служебное поле, а не сторона, которой сидит белок (п. 5.3).')
CAVEAT_MAP = ('Карта вклада — вклад в оценку места: итоговая оценка места (среднее двух проходов) при каждой из четырёх '
              'букв в каждой позиции окна ±70 за вычетом среднего по четырём. Это объяснение оценки модели, а не '
              'измерение физического вклада основания в связывание (SPEC п. 5.4).')
CAVEAT_RESIDUES = ('Веса сбора по остаткам белка — отдельная иллюстрация, по профилю на каждый из четырёх обучаемых '
                   'запросов. Внимания по ДНК в модели нет вовсе, поэтому это не карта важности оснований; четыре '
                   'запроса не обязаны совпадать с биологией; высокий вес на остатке не значит, что этот остаток '
                   'определяет специфичность (SPEC п. 5.4).')
NOT_GIVEN = ('Чего программа не выдаёт ни в каком виде: вероятность связывания; сродство или константу связывания; '
             'утверждение «белок регулирует ген X»; предсказание оперонов и регулонов; число настоящих сайтов в геноме; '
             'сравнение двух белков между собой; оценку для белка длиннее 1 022 остатков или для участка короче поля '
             'видимости 141 (SPEC п. 5.5).')


# ---- общие мелочи --------------------------------------------------------------------------
def log(text):
    print(text, file=sys.stderr, flush=True)


def rel(path):
    """Путь от корня проекта, если файл внутри него; иначе — как есть."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def under_root(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def safe_name(name):
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def git_commit():
    try:
        head = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--', 'src'], capture_output=True,
                               text=True).stdout.strip()
        return {'commit': head or None, 'src_uncommitted': bool(dirty)}
    except OSError:
        return {'commit': None, 'src_uncommitted': None}


def cpu_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def versions():
    import importlib.metadata as md
    out = {'python': platform.python_version()}
    for name, dist in (('torch', 'torch'), ('numpy', 'numpy'), ('pyBigWig', 'pyBigWig'), ('fair-esm', 'fair-esm')):
        try:
            out[name] = md.version(dist)
        except md.PackageNotFoundError:
            out[name] = None
    mm = shutil.which('mmseqs')
    if mm:
        r = subprocess.run([mm, 'version'], capture_output=True, text=True)
        out['mmseqs'] = r.stdout.strip() or None
    else:
        out['mmseqs'] = None
    return out


def read_fasta_dict(path):
    out, name, buf = {}, None, []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    out[name] = ''.join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        out[name] = ''.join(buf)
    return out


def write_fasta(path, records):
    with open(path, 'w') as handle:
        for name, seq in records:
            handle.write('>%s\n%s\n' % (name, seq))


# ---- выравниватель белков (`CONTRACTS.md` § 2.6) ------------------------------------------------
def max_identity(query_fasta, target_fasta, workdir):
    """Наибольшая идентичность (в процентах) белка запросов с белком целей при покрытии более короткого ≥0,8 —
    MMseqs2 командой `CONTRACTS.md` § 2.6, в обе стороны, один поток, пара «белок с самим собой» не считается.
    Ответ — `(идентичность или None, ключ ближайшей цели или None)`; программы нет — `FileNotFoundError`."""
    mm = shutil.which('mmseqs')
    if mm is None:
        raise FileNotFoundError('mmseqs не установлен: идентичность белков посчитать нечем (CONTRACTS § 2.6)')
    best, who = None, None
    for i, (q, t) in enumerate(((query_fasta, target_fasta), (target_fasta, query_fasta))):
        out = os.path.join(workdir, 'hits%d.m8' % i)
        r = subprocess.run([mm, 'easy-search', str(q), str(t), out, os.path.join(workdir, 'tmp%d' % i),
                            '--exhaustive-search', '-e', '1e-3', '--format-output',
                            'query,target,fident,qcov,tcov,qlen,tlen,evalue', '--threads', '1', '-v', '1'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError('mmseqs easy-search: %s' % r.stderr[-300:])
        for line in open(out):
            f = line.rstrip('\n').split('\t')
            if len(f) < 8 or f[0] == f[1]:
                continue
            fident, qcov, tcov, qlen, tlen = (float(x) for x in f[2:7])
            cov = qcov if qlen <= tlen else tcov
            if cov >= MIN_COVERAGE and (best is None or fident * 100 > best):
                best, who = fident * 100, (f[1] if i == 0 else f[0])
    return best, who


def band_of(identity):
    """Полоса родства по правилу п. 3.1 и случай варианта п. 5.6."""
    if identity is not None and identity >= VARIANT_IDENTITY:
        return 'variant'
    if identity is not None and identity >= 70.0:
        return '>=70'
    if identity is not None and identity >= 40.0:
        return '40-70'
    return '<40'


BAND_WORDS = {
    'variant': 'вариант обучающего белка (идентичность ≥90% при покрытии ≥80%): для модели он не новый, полоса для '
               'него не определена (SPEC п. 5.6)',
    '>=70': 'полоса ≥70% (п. 3.1): близкий родственник обучающего белка; обещание этой полосы — не хуже допуска '
            'против переноса мотива (п. 1.5)',
    '40-70': 'полоса 40–70% (п. 3.1): та полоса, на которой проверяется главное обещание работы (п. 1.5)',
    '<40': 'полоса ниже 40% (п. 3.1): дальний белок; результат на таких белках показывается как есть, даже если он '
           'близок к случайному (п. 1.3)',
}


# ---- модель -------------------------------------------------------------------------------
def load_model(model_dir, config_path, kind, device):
    import torch
    from .model import build
    from .scan import enforce_determinism, select_device
    enforce_determinism()
    model_dir = under_root(model_dir)
    journal = json.load(open(model_dir / 'journal.json'))
    # Модель, обученная с маской домена, применяется с той же маской (CONTRACTS § 4.10), а домены нового белка
    # программа не размечает: без отказа она молча собрала бы вектор по всему белку.
    if (journal.get('protein_mask') or 'whole') != 'whole':
        raise SystemExit('модель %s обучена с protein_mask = %r; программа строит белковый вход по всему белку и '
                         'такую модель не применяет (CONTRACTS § 4.10)' % (model_dir, journal.get('protein_mask')))
    config = json.load(open(under_root(config_path)))
    config_sha = file_sha(under_root(config_path))
    choices = dict(journal.get('choices') or {})
    cfg = dict(config, protein_paths=choices.get('protein_paths', 'both'))
    net = build(cfg, protein_input=bool(journal.get('protein_input', True)))
    unit = select_device(device)
    net.load_state_dict(torch.load(model_dir / 'model.pt', map_location='cpu'))
    net = net.to(unit).eval()
    fold = journal.get('fold')
    fold_name = 'final' if fold is None else 'fold%d' % int(fold)
    if kind == 'pilot':
        what = ('pilot model (модель пилота T14), not the final model — не зачётная: %s, seed %s, protein paths %s, '
                'PRODORIC in training %s' % (fold_name, journal.get('seed'), choices.get('protein_paths'),
                                             'yes' if choices.get('prodoric_in_training') else 'no'))
    elif kind == 'fold':
        what = 'fold model (модель захода, не итоговая): %s, seed %s' % (fold_name, journal.get('seed'))
    else:
        what = 'final ProkaBind model (итоговая модель): seed %s' % journal.get('seed')
    info = {'dir': rel(model_dir), 'weights_sha256': file_sha(model_dir / 'model.pt'), 'kind': kind,
            'description': what, 'seed': journal.get('seed'), 'fold': fold_name, 'choices': choices,
            'protein_input': bool(journal.get('protein_input', True)), 'config': rel(under_root(config_path)),
            'config_sha256': config_sha, 'config_matches_journal': config_sha == journal.get('config_sha256'),
            'training_proteins': len(journal.get('training_proteins') or ())}
    return net, journal, unit, info


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strand_pair(net, protein, sequence, anchor, circular):
    """Оценки двух проходов в точке привязки на окне поля видимости этой точки (как в `work/eval/b14_explain.py`)."""
    import torch
    from .model import receptive_field
    from .scan import onehot
    rf = receptive_field(net.config)
    context = (rf - 1) // 2
    n = len(sequence)
    lo = anchor - context
    if circular:
        window = ''.join(sequence[(lo + i) % n] for i in range(rf))
    else:
        if lo < 0 or lo + rf > n:
            return float('nan'), float('nan')
        window = sequence[lo:lo + rf]
    unit = next(net.parameters()).device
    with torch.no_grad():
        fwd, rev = net.strand_scores(onehot(window).to(unit), protein[0].to(unit), protein[1].to(unit))
    return float(fwd[0, context]), float(rev[0, context])


def residue_weights(net, protein):
    """Веса сбора вниманием по остаткам белка (п. 5.4): `[Q, L]`, по профилю на обучаемый запрос.

    Ключи и запросы берутся из прохода самой белковой сети (перехватом выходов её слоёв), а не пересчитываются
    отдельно; ответ сверяется с её же выходом слоя ответов: веса, умноженные на значения, дают ровно то, что сеть
    передала дальше."""
    import torch
    pn = net.protein
    if pn is None:
        return None
    seen = {}
    hooks = [getattr(pn, name).register_forward_hook(lambda m, i, o, name=name: seen.__setitem__(name, (i[0], o)))
             for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
    features, mask = protein
    unit = next(net.parameters()).device
    try:
        with torch.no_grad():
            pn(features.to(unit), mask.to(unit))
    finally:
        for h in hooks:
            h.remove()
    with torch.no_grad():
        q, k, v = seen['q_proj'][1], seen['k_proj'][1], seen['v_proj'][1]
        q = q.unsqueeze(0) if q.dim() == 2 else q
        att = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(pn.width)
        att = att.masked_fill(~mask.to(unit).unsqueeze(1), float('-inf'))
        weights = torch.softmax(att, dim=-1)
        again = torch.matmul(weights, v)
        if not torch.allclose(again, seen['o_proj'][0], rtol=1e-4, atol=1e-5):
            raise RuntimeError('веса сбора не воспроизводят вход слоя ответов белковой сети')
    return weights[0].to('cpu', torch.float32).numpy()


# ---- допустимость показа (п. 5.8) ----------------------------------------------------------------
def check_show(protein, protein_path, data_root, workdir):
    """Белок допустим для показа до фиксации, только если он из контрольной части без пиков StrepTRN или вне
    собранных данных с идентичностью ниже 40% при покрытии ≥80% к каждому белку вне контрольной части (SPEC п. 5.8).
    Проверка идёт **до** счёта; недопустимый белок — отказ `not_allowed_before_fixation`."""
    from .demo import eligible
    root = under_root(data_root)
    need = [root / 'split.tsv', root / 'peaks.tsv', root / 'proteins.fasta']
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise IN.Refusal('show_unchecked', 'допустимость показа проверить нечем: нет %s (SPEC п. 5.8)' % ', '.join(missing))
    key = protein['key']
    part = {r['protein']: r['part'] for r in csv.DictReader(open(root / 'split.tsv'), delimiter='\t')}
    peaks = {r['protein'] for r in csv.DictReader(open(root / 'peaks.tsv'), delimiter='\t')}
    if not eligible(key, peaks):
        raise IN.Refusal('not_allowed_before_fixation', 'у белка есть пики StrepTRN — его данные входят во вход или фон '
                         'Б6 (SPEC п. 5.8, тест 12 п. 4.11)')
    if key in part and part[key] != 'control':
        raise IN.Refusal('not_allowed_before_fixation', 'белок входит в собранные данные (часть %s): до фиксации '
                         'показываются только белки контрольной части без пиков StrepTRN и белки вне собранных данных '
                         '(SPEC п. 5.8)' % part[key])
    set_name = 'control' if key in part else 'outside'
    seqs = read_fasta_dict(root / 'proteins.fasta')
    targets = [(k, s) for k, s in sorted(seqs.items()) if part.get(k, 'outside') != 'control']
    tf = os.path.join(workdir, 'outside_control.fasta')
    qf = os.path.join(workdir, 'shown.fasta')
    write_fasta(tf, targets)
    write_fasta(qf, [(key, protein['sequence'])])
    try:
        ident, who = max_identity(qf, tf, workdir)
    except FileNotFoundError as why:
        if set_name == 'outside':
            raise IN.Refusal('show_unchecked', str(why))
        ident, who = None, None
    if set_name == 'outside' and ident is not None and ident >= SHOW_IDENTITY:
        raise IN.Refusal('not_allowed_before_fixation', 'идентичность %.1f%% с белком %s вне контрольной части при '
                         'покрытии ≥80%%: ортолог проверочного белка — это его же ответ под другим именем '
                         '(SPEC п. 5.8)' % (ident, who[:12]))
    return {'set': set_name, 'max_identity': ident, 'nearest_outside': who, 'targets': len(targets),
            'fasta': rel(protein_path)}


def known_sites(data_root, key, names):
    """Известные сайты белка контрольной части в репликонах этого генома (`data/derived/sites.tsv`, § 2.2) — рисуются
    только у белков контрольной части (SPEC п. 5.8)."""
    path = under_root(data_root) / 'sites.tsv'
    out = {n: [] for n in names}
    if not path.exists():
        return out
    for r in csv.DictReader(open(path), delimiter='\t'):
        if r['protein'] == key and r['genome'] in out:
            out[r['genome']].append((int(r['start']), int(r['end'])))
    for n in out:
        out[n] = sorted(set(out[n]))
    return out


# ---- места ----------------------------------------------------------------------------------
def ring_distance(a, b, n, circular):
    d = abs(int(a) - int(b))
    return min(d, n - d) if circular else d


def gene_context(anchor, genes, n, circular):
    """Тип положения по основанию точки привязки (п. 3.4) и ближайший ген с расстоянием (0 — внутри гена)."""
    inside = any(s <= anchor < e for s, e, _, _, _ in genes)
    best = None
    for s, e, kind, name, rev in genes:
        if s <= anchor < e:
            d = 0
        else:
            d = min(ring_distance(anchor, s, n, circular), ring_distance(anchor, e - 1, n, circular))
        label = name or kind
        if best is None or (d, label) < best:
            best = (d, label)
    return ('genic' if inside else 'intergenic'), (best[1] if best else None), (best[0] if best else None)


# ---- главный ход -------------------------------------------------------------------------------
def execute(a, show):
    started = time.time()
    when = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        log('каталог результатов %s не пуст: готовое не переделывается молча — укажите пустой каталог' % out)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    desc = {'program': 'ProkaBind', 'mode': 'show' if show else 'run', 'command': [sys.executable] + sys.argv,
            'cwd': os.getcwd(), 'started_utc': when.isoformat().replace('+00:00', 'Z'), 'code': git_commit(),
            'libraries': versions(), 'notes': [], 'refusals': []}
    work = tempfile.mkdtemp(prefix='prokabind_')
    try:
        code = _execute(a, show, out, desc, work)
    except IN.Refusal as why:
        desc['refusal'] = {'code': why.code, 'reason': why.reason}
        log('отказ %s: %s' % (why.code, why.reason))
        code = 2
    except Exception as why:                                   # сбой программы — тоже в описание запуска
        desc['error'] = '%s: %s' % (type(why).__name__, why)
        with open(out / 'run.json', 'w', encoding='utf-8') as handle:
            handle.write(json.dumps(desc, ensure_ascii=False, indent=1) + '\n')
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    desc['seconds'] = round(time.time() - started, 1)
    desc['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace(
        '+00:00', 'Z')
    if code == 0:
        desc['outputs'] = [{'path': str(p.relative_to(out)), 'sha256': file_sha(p)}
                           for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run.json']
    with open(out / 'run.json', 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(desc, ensure_ascii=False, indent=1) + '\n')
    if code == 0 and show:
        append_journal(a, desc)
    return code


def append_journal(a, desc):
    path = under_root(a.journal)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    s = desc['show']
    row = {'protein': desc['protein']['key'], 'genome': ';'.join(r['name'] for r in desc['genome']['replicons']),
           'date': desc['started_utc'], 'model': desc['model']['description'], 'set': s['set'], 'fasta': s['fasta'],
           'max_identity': '' if s['max_identity'] is None else '%.1f' % s['max_identity'],
           'nearest_outside': s['nearest_outside'] or ''}
    with open(path, 'a', encoding='utf-8', newline='\n') as handle:
        if new:
            handle.write('\t'.join(JOURNAL_COLUMNS) + '\n')
        handle.write('\t'.join(str(row[c]).replace('\t', ' ') for c in JOURNAL_COLUMNS) + '\n')
    log('показ записан в журнал %s' % rel(path))


def _execute(a, show, out, desc, work):
    import torch
    from . import contrib, places as PL
    from .features import esm_weights_path, protein_features
    from .model import receptive_field
    from .scan import device_name, score_genome
    sys.path.insert(0, str(ROOT / 'work' / 'lib'))
    import eval_defs                                           # справочный уровень п. 5.3 — одна функция (п. 3.11)

    # 1. вход (п. 5.1): отказ по белку отменяет запуск
    protein = IN.read_protein(a.protein)
    desc['protein'] = {'file': rel(a.protein), 'file_sha256': file_sha(a.protein), 'name': protein['name'],
                       'key': protein['key'], 'sequence_sha256': protein['key'], 'length': protein['length'],
                       'notes': protein['notes']}
    replicons = IN.read_genome(a.genome)
    desc['genome'] = {'file': rel(a.genome), 'file_sha256': file_sha(a.genome)}
    annotation, ann_format = None, None
    if a.annotation:
        ann_format, annotation = IN.read_annotation(a.annotation)
        desc['annotation'] = {'file': rel(a.annotation), 'file_sha256': file_sha(a.annotation), 'format': ann_format}
    keys = {}
    for name in a.circular or ():
        keys[name] = 'circular'
    for name in a.linear or ():
        if keys.get(name) == 'circular':
            raise IN.Refusal('topology_conflict', 'репликон %s назван и кольцевым, и линейным' % name)
        keys[name] = 'linear'
    matched = IN.match_annotation(replicons, annotation, keys)
    desc['notes'] += ['protein: %s' % n for n in protein['notes']]
    desc['notes'] += ['%s: %s' % (r['name'], n) for r in replicons for n in r['notes']]
    if annotation is not None:
        desc['notes'] += ['%s: annotation_absent' % r['name'] for r in replicons if r['name'] not in annotation]
    if show:
        desc['show'] = check_show(protein, a.protein, a.data_root, work)
        log('показ допустим: набор %s, наибольшая идентичность с белком вне контрольной части %s'
            % (desc['show']['set'], 'нет пары при покрытии ≥80%' if desc['show']['max_identity'] is None
               else '%.1f%%' % desc['show']['max_identity']))

    # 2. модель и признаки белка — всегда через кэш (п. 5.7)
    net, journal, unit, minfo = load_model(a.model, a.config, a.model_kind, a.device)
    desc['model'] = minfo
    desc['device'] = {'name': device_name(unit) if unit.type == 'cuda' else cpu_name(), 'type': unit.type,
                      'torch_threads': torch.get_num_threads(),
                      'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    read_dirs = [str(under_root(d)) for d in (a.esm_cache_read or ())]
    feats, finfo = protein_features(protein['sequence'], protein['key'], read_dirs, str(under_root(a.esm_cache)))
    weights = esm_weights_path()
    finfo['file'] = rel(finfo['file'])
    desc['protein_network'] = {'package': 'fair-esm', 'loader': 'esm.pretrained.esm2_t33_650M_UR50D()',
                               'weights_file': weights, 'weights_sha256': file_sha(weights) if os.path.exists(weights)
                               else None, 'features': finfo}
    protein_pair = (torch.from_numpy(feats)[None], torch.ones((1, feats.shape[0]), dtype=torch.bool))
    rf = receptive_field(net.config)

    # 3. родство с обучающими белками модели (п. 5.6)
    desc['identity_to_training'] = training_identity(a, protein, journal, work)
    log('родство с обучающими белками: %s' % desc['identity_to_training'].get('band'))

    # 4. дорожка оценок, маска (п. 5.2)
    (out / 'track').mkdir()
    (out / 'mask').mkdir()
    reps = []
    for order, rep in enumerate(replicons):
        name, seq = rep['name'], rep['sequence']
        info = matched[name]
        circular = info['topology'] == 'circular'
        entry = {'name': name, 'length': len(seq), 'topology': info['topology'],
                 'topology_source': info['topology_source'], 'annotated': info['genes'] is not None,
                 'notes': rep['notes']}
        if len(seq) < rf:
            entry['skipped'] = 'insufficient_context'
            desc['refusals'].append({'replicon': name, 'code': 'insufficient_context',
                                     'reason': 'репликон короче поля видимости %d (SPEC п. 5.1)' % rf})
            reps.append(dict(entry, order=order, seq=seq, circular=circular, scores=None, mask=None, places=[],
                             genes=info['genes'] or []))
            continue
        t0 = time.time()
        scores, mask = score_genome(net, protein_pair, seq, circular)
        log('%s: проход %d оснований за %.0f с' % (name, len(seq), time.time() - t0))
        base = safe_name(name)
        OUT.write_track(out / 'track' / (base + '.tsv.gz'), name, scores, mask)
        OUT.write_bigwig(out / 'track' / (base + '.bw'), name, scores, mask)
        OUT.write_mask(out / 'mask' / (base + '.tsv.gz'), name, mask)
        entry['scored_bases'] = int(mask.sum())
        places = PL.build_places(scores, mask, circular)
        entry['n_places'] = len(places)
        reps.append(dict(entry, order=order, seq=seq, circular=circular, scores=scores, mask=mask, places=places,
                         genes=info['genes'] or []))
    desc['genome']['replicons'] = [{k: r[k] for k in ('name', 'length', 'topology', 'topology_source', 'annotated',
                                                      'notes', 'scored_bases', 'n_places', 'skipped') if k in r}
                                   for r in reps]
    scored = [r for r in reps if r['scores'] is not None and r['mask'].any()]
    if not scored:
        raise IN.Refusal('insufficient_context', 'ни одно основание генома не получило оценки (SPEC п. 5.1)')

    # 5. список мест (п. 5.3)
    all_scores = np.concatenate([r['scores'][r['mask']] for r in scored]).astype(np.float64)
    level = eval_defs.level_999(np.concatenate([r['scores'] for r in scored]),
                                np.concatenate([r['mask'] for r in scored]))
    sd = float(np.std(all_scores))
    merged = sorted(((p.score, r['order'], p.anchor, r, p) for r in scored for p in r['places']),
                    key=lambda t: (-t[0], t[1], t[2]))
    level_rank = sum(1 for t in merged if t[0] > level)
    counts = {}
    for t in merged:
        counts[t[0]] = counts.get(t[0], 0) + 1
    higher = {}
    for r in scored:
        values = np.sort(np.asarray([p.score for p in r['places']], dtype=np.float64))
        higher[r['name']] = values
    top = merged[:a.top]
    rows = []
    for rank, (score, order, anchor, r, p) in enumerate(top, start=1):
        n = len(r['seq'])
        above = int(higher[r['name']].size - np.searchsorted(higher[r['name']], score, side='right'))
        fwd, rev = strand_pair(net, protein_pair, r['seq'], anchor, r['circular'])
        if fwd - rev > PASS_SD * sd:
            larger = 'forward'
        elif rev - fwd > PASS_SD * sd:
            larger = 'reverse'
        else:
            larger = 'passes_close'
        row = {'rank': rank, 'replicon': r['name'], 'position': anchor + 1, 'score': np.float32(score),
               'places_higher_per_mb': above / float(r['scored_bases']) * 1e6, 'n_merged': p.n_merged,
               'edge': p.edge, 'tie': counts[score] > 1, 'near_other_place': False, 'larger_pass': larger,
               '_anchor': anchor, '_r': r, '_fwd': fwd, '_rev': rev, '_place': p}
        if r['annotated']:
            kind, gene, dist = gene_context(anchor, r['genes'], n, r['circular'])
            row.update(position_type=kind, nearest_gene=gene, gene_distance=dist)
        rows.append(row)
    for x in rows:
        for y in rows:
            if x is not y and x['replicon'] == y['replicon'] and ring_distance(
                    x['_anchor'], y['_anchor'], len(x['_r']['seq']), x['_r']['circular']) < NEAR:
                x['near_other_place'] = True
    place_cols = ['rank', 'replicon', 'position', 'score', 'places_higher_per_mb', 'n_merged', 'edge', 'tie',
                  'near_other_place', 'position_type', 'nearest_gene', 'gene_distance', 'larger_pass']
    OUT.write_table(out / 'places.tsv', place_cols, rows, comments=[
        'ProkaBind: первые %d мест генома; position — точка привязки места, от 1 (SPEC пп. 5.2, 5.3)' % len(rows),
        CAVEAT_SCALE, CAVEAT_LIST % a.top, CAVEAT_PASS])
    log('мест %d, напечатано %d; справочный уровень %.4g проходит после места %d' % (len(merged), len(rows), level,
                                                                                level_rank))
    desc['places'] = {'printed': len(rows), 'total': len(merged), 'level_999': level, 'level_passes_after_rank':
                      level_rank, 'track_sd': sd}

    # 6. карты вклада и логотип (п. 5.4)
    oriented, orient_source = orientation(a)
    maps, map_rows = [], []
    offsets = list(range(-HALF, HALF + 1))
    reversed_count = 0
    for row in rows[:a.maps]:
        r, anchor = row['_r'], row['_anchor']
        cmap = contrib.contribution_map(net, protein_pair, r['seq'], anchor, HALF, r['circular'])
        row['_map'] = cmap
        if not np.isfinite(cmap).all():
            continue
        reverse = row['_rev'] > row['_fwd']                  # при равенстве — прямой (как в b14_explain.py)
        row['_orient'] = 'reverse' if reverse else 'forward'
        n = len(r['seq'])
        for i, off in enumerate(offsets):
            pos = anchor + off
            pos = pos % n if r['circular'] else pos
            map_rows.append({'rank': row['rank'], 'replicon': r['name'], 'anchor': anchor + 1, 'offset': off,
                             'position': pos + 1, 'genome_base': r['seq'][pos], 'A': cmap[i, 0], 'C': cmap[i, 1],
                             'G': cmap[i, 2], 'T': cmap[i, 3], 'logo_orientation': row['_orient'] if oriented else ''})
        if oriented and reverse:
            maps.append(cmap[::-1, ::-1])
            reversed_count += 1
        else:
            maps.append(cmap)
    OUT.write_table(out / 'contribution_maps.tsv.gz',
                    ['rank', 'replicon', 'anchor', 'offset', 'position', 'genome_base', 'A', 'C', 'G', 'T',
                     'logo_orientation'], map_rows, comments=[
                        'Карты вклада верхних %d мест: вклад в оценку места (SPEC п. 5.4); offset — положение '
                        'относительно точки привязки, position — от 1' % len(maps), CAVEAT_MAP, CAVEAT_SCALE])
    log('карт вклада %d' % len(maps))
    logo = np.mean(np.stack(maps), axis=0) if maps else None
    if oriented:
        logo_note = ('логотип ориентирован: карты мест, где обратный проход дал в точке привязки большую оценку, '
                     'развёрнуты с дополнением (%d из %d); ориентацию подтвердил пилот (%s)'
                     % (reversed_count, len(maps), orient_source))
    else:
        logo_note = 'ориентация не подтверждена: логотип без ориентации (%s)' % orient_source
    if logo is not None:
        OUT.write_table(out / 'logo.tsv', ['offset', 'A', 'C', 'G', 'T'],
                        [{'offset': off, 'A': logo[i, 0], 'C': logo[i, 1], 'G': logo[i, 2], 'T': logo[i, 3]}
                         for i, off in enumerate(offsets)],
                        comments=['Логотип: средняя карта вклада верхних %d мест (SPEC п. 5.4)' % len(maps), logo_note,
                                  CAVEAT_MAP])
        FIG.logo_figure(out / 'logo.svg', logo, offsets,
                        'Логотип ProkaBind: %s — средняя карта вклада верхних %d мест' % (protein['name'], len(maps)),
                        [logo_note + '.', CAVEAT_MAP, 'Модель: ' + minfo['description'] + '.'])
    desc['logo'] = {'maps': len(maps), 'oriented': oriented, 'reversed': reversed_count if oriented else None,
                    'orientation_source': orient_source}

    # 7. веса сбора по остаткам (п. 5.4, отдельная иллюстрация)
    rw = residue_weights(net, protein_pair)
    if rw is not None:
        seq = protein['sequence']
        OUT.write_table(out / 'residue_weights.tsv', ['position', 'residue'] + ['query%d' % (k + 1) for k in range(rw.shape[0])],
                        [dict({'position': i + 1, 'residue': seq[i]},
                              **{'query%d' % (k + 1): np.float32(rw[k, i]) for k in range(rw.shape[0])})
                         for i in range(rw.shape[1])], comments=[CAVEAT_RESIDUES])
        FIG.residue_figure(out / 'residue_weights.svg', rw, seq,
                           'Веса сбора по остаткам белка %s (%d остатков)' % (protein['name'], len(seq)),
                           [CAVEAT_RESIDUES])

    # 8. показ: рисунок дорожки вокруг первых мест с картой вклада (п. 5.8)
    sites = {}
    recall = None
    if show and desc['show']['set'] == 'control':
        sites = known_sites(a.data_root, protein['key'], [r['name'] for r in scored])
        recall = control_recall(scored, sites, level)
        desc['show']['known_sites'] = recall
    figure_rows = rows[:a.figure_places]
    if show:
        panels = []
        for row in figure_rows:
            r, anchor = row['_r'], row['_anchor']
            n = len(r['seq'])
            lo, hi = max(0, anchor - FIGURE_HALF), min(n, anchor + FIGURE_HALF + 1)
            cmap = row.get('_map')
            mseq = ''.join(r['seq'][(anchor + o) % n] if r['circular'] else
                           (r['seq'][anchor + o] if 0 <= anchor + o < n else 'N') for o in offsets)
            panels.append({'rank': row['rank'], 'replicon': r['name'], 'anchor': anchor, 'score': float(row['score']),
                           'n_merged': row['n_merged'], 'larger_pass': {'forward': 'прямой', 'reverse': 'обратный',
                                                                        'passes_close': 'проходы близки'}[row['larger_pass']],
                           'lo': lo, 'hi': hi, 'track': r['scores'][lo:hi],
                           'places': [(x['rank'], x['_anchor']) for x in rows if x['_r'] is r],
                           'genes': [(s, e, nm, rv) for s, e, kind, nm, rv in r['genes'] if e > lo and s < hi
                                     and kind == 'gene'] or [(s, e, nm, rv) for s, e, kind, nm, rv in r['genes']
                                                              if e > lo and s < hi and kind == 'CDS'],
                           'sites': sites.get(r['name'], []), 'level': level,
                           'map': cmap if cmap is not None and np.isfinite(cmap).all() else None,
                           'map_offsets': offsets, 'seq': mseq})
        first = figure_rows[0]['_r'] if figure_rows else scored[0]
        nbin = max(1, int(math.ceil(len(first['seq']) / 1000.0)))
        vals = np.where(first['mask'], first['scores'], np.nan).astype(np.float64)
        bins = []
        for i in range(0, len(vals), nbin):
            piece = vals[i:i + nbin]
            bins.append(float(np.nanmax(piece)) if np.isfinite(piece).any() else float('nan'))
        overview = {'replicon': first['name'], 'length': len(first['seq']), 'bin': nbin, 'values': bins,
                    'places': [(x['rank'], x['_anchor']) for x in figure_rows if x['_r'] is first],
                    'sites': sites.get(first['name'], [])}
        notes = [CAVEAT_SCALE, CAVEAT_MAP, CAVEAT_PASS,
                 'Модель: ' + minfo['description'] + '.',
                 ('Белок контрольной части без пиков StrepTRN (набор 1 SPEC п. 5.8): оранжевым нарисованы его известные '
                  'сайты из собранных данных.') if desc['show']['set'] == 'control' else
                 ('Белок вне собранных данных (набор 2 SPEC п. 5.8): известных сайтов у него в данных нет, и ничего, '
                  'кроме предсказаний, не рисуется.'),
                 'Красная заливка карты — основание в этой позиции повышает оценку места, синяя — понижает; '
                 'жёлтая полоса на дорожке — окно карты (поле видимости точки привязки, 141 основание).']
        FIG.show_figure(out / 'show.svg',
                        'ProkaBind — показ: %s на %s' % (protein['name'], ', '.join(r['name'] for r in scored)),
                        'Первые %d мест; координаты — от 1; фиолетовый пунктир — справочный уровень (99,9-й перцентиль '
                        'оценок оснований этого белка в этом геноме)' % len(figure_rows), panels, notes, overview)

    # 9. сводка
    write_summary(out / 'summary.md', desc, protein, minfo, rows, level, level_rank, len(merged), maps, logo_note,
                  recall, show)
    return 0


def training_identity(a, protein, journal, work):
    """Идентичность с ближайшим обучающим белком этой модели и полоса родства по правилу п. 3.1 (п. 5.6)."""
    keys = list(journal.get('training_proteins') or ())
    if a.training_fasta:
        tf = str(under_root(a.training_fasta))
        source = rel(tf)
    else:
        path = under_root(a.data_root) / 'proteins.fasta'
        if not path.exists():
            return {'identity': None, 'band': None, 'note': 'последовательностей обучающих белков нет: %s' % rel(path)}
        seqs = read_fasta_dict(path)
        missing = [k for k in keys if k not in seqs]
        tf = os.path.join(work, 'training.fasta')
        write_fasta(tf, [(k, seqs[k]) for k in sorted(keys) if k in seqs])
        source = '%s, белки журнала обучения модели (%d, без последовательности %d)' % (rel(path), len(keys),
                                                                                       len(missing))
    qf = os.path.join(work, 'query_training.fasta')
    write_fasta(qf, [(protein['key'], protein['sequence'])])
    try:
        ident, who = max_identity(qf, tf, work)
    except FileNotFoundError as why:
        return {'identity': None, 'band': None, 'note': str(why), 'training_sequences': source}
    band = band_of(ident)
    return {'identity': ident, 'nearest_training_protein': who, 'band': band, 'words': BAND_WORDS[band],
            'coverage': 'покрытие более короткого ≥0,8, MMseqs2 по CONTRACTS § 2.6', 'training_sequences': source}


def orientation(a):
    """Подтверждена ли ориентация пилотом: интервал доли согласия (число 9 п. 4.9) не накрывает 0,5 (п. 5.4)."""
    if a.orientation in ('confirmed', 'unconfirmed'):
        return a.orientation == 'confirmed', 'ключ --orientation %s' % a.orientation
    path = under_root(a.orientation_report)
    if not path.exists():
        return False, 'нет %s — ориентация считается неподтверждённой' % rel(path)
    d = json.load(open(path))
    ci = d.get('orientation_agreement_ci95')
    if not (isinstance(ci, (list, tuple)) and len(ci) == 2):
        return False, '%s: нет интервала orientation_agreement_ci95 — ориентация считается неподтверждённой' % rel(path)
    lo, hi = float(ci[0]), float(ci[1])
    ok = not (lo <= 0.5 <= hi)
    return ok, 'доля согласия %.3f, интервал [%.3f; %.3f] %s 0,5 — %s, orientation_agreement_ci95' % (
        float(d.get('orientation_agreement', float('nan'))), lo, hi, 'не накрывает' if ok else 'накрывает', rel(path))


def control_recall(scored, sites, level):
    """Полнота у белка контрольной части: доля его известных сайтов, чья оценка (наибольшая по основаниям сайта,
    п. 3.4) выше справочного уровня — тем же определением уровня (пп. 5.3, 5.6). Служебная справка с пометкой о
    смещении: по контрольной части выбиралась контрольная точка (п. 4.8)."""
    total, defined, above, without = 0, 0, 0, 0
    for r in scored:
        for s, e in sites.get(r['name'], []):
            total += 1
            if s < 0 or e > len(r['seq']) or e <= s or not r['mask'][s:e].all():
                without += 1
                continue
            defined += 1
            if float(np.max(r['scores'][s:e])) > level:
                above += 1
    return {'sites': total, 'with_score': defined, 'without_score': without, 'above_level': above,
            'recall': (above / float(defined)) if defined else None,
            'mark': 'оптимистично: по этой части выбиралась контрольная точка (SPEC пп. 4.8, 5.6)'}


def write_summary(path, desc, protein, minfo, rows, level, level_rank, total, maps, logo_note, recall, show):
    L = []
    L.append('# ProkaBind: %s — сводка запуска' % protein['name'])
    L.append('')
    L.append('**Какая модель:** %s.' % minfo['description'])
    L.append('')
    L.append('Описание запуска (суммы файлов, топология, устройство, версии, команда, время) — `run.json` рядом.')
    L.append('')
    L.append('## Белок')
    L.append('')
    L.append('- запись `%s`, %d остатков, ключ (sha256 последовательности) `%s`' % (protein['name'], protein['length'],
                                                                                 protein['key']))
    if protein['notes']:
        L.append('- пометки входа: %s' % ', '.join('`%s`' % n for n in protein['notes']))
    t = desc.get('identity_to_training') or {}
    if t.get('band'):
        L.append('- идентичность с ближайшим обучающим белком модели: %s (`%s`)' % (
            'нет пары при покрытии ≥80%' if t['identity'] is None else '%.1f%%' % t['identity'],
            (t.get('nearest_training_protein') or '—')[:16]))
        L.append('- %s' % t['words'])
    else:
        L.append('- идентичность с обучающими белками не посчитана: %s' % t.get('note'))
    if show:
        s = desc['show']
        L.append('- набор показа SPEC п. 5.8: **%s** — %s; наибольшая идентичность с белком вне контрольной части: %s' % (
            s['set'], 'белок контрольной части без пиков StrepTRN' if s['set'] == 'control' else 'белок вне собранных данных',
            'нет пары при покрытии ≥80%' if s['max_identity'] is None else '%.1f%%' % s['max_identity']))
    L.append('')
    L.append('## Геном')
    L.append('')
    for r in desc['genome']['replicons']:
        topo = {'command_line': 'ключ командной строки', 'annotation': 'аннотация',
                'default_linear': 'не задана — линейный по умолчанию (SPEC п. 5.1)'}[r['topology_source']]
        line = '- `%s`: %d оснований, топология %s (%s)' % (r['name'], r['length'], r['topology'], topo)
        if 'skipped' in r:
            line += ' — пропущен: `%s`' % r['skipped']
        else:
            line += ', оценку получили %d оснований, мест %d' % (r['scored_bases'], r['n_places'])
        if not r['annotated']:
            line += '; аннотации нет — типа положения и ближайшего гена нет'
        L.append(line)
    L.append('')
    L.append('## Справочный уровень и полнота')
    L.append('')
    L.append('- справочный уровень — 99,9-й перцентиль оценок оснований этого белка в этом геноме: %s; он проходит '
             'после места %d списка (всего мест %d)' % (OUT.fmt9(level), level_rank, total))
    if recall is not None:
        L.append('- полнота у этого белка — доля его известных сайтов выше уровня: %s (%d из %d сайтов с оценкой; без '
                 'оценки %d) — **%s**' % ('—' if recall['recall'] is None else '%.3f' % recall['recall'],
                                           recall['above_level'], recall['with_score'], recall['without_score'],
                                           recall['mark']))
    L.append('- полнота модели по контрольной части: не измерено; по полосам родства — после результатов (SPEC п. 5.6)')
    L.append('')
    L.append('## Первые места')
    L.append('')
    L.append('> ' + CAVEAT_SCALE)
    L.append('> ' + (CAVEAT_LIST % len(rows)))
    L.append('> ' + CAVEAT_PASS)
    L.append('')
    L.append('| место | репликон | положение (от 1) | оценка | мест выше на Мб | склеено | тип | ближайший ген | '
             'расстояние | больший проход | пометки |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for row in rows[:20]:
        flags = [f for f, on in (('edge', row['edge']), ('tie', row['tie']), ('near', row['near_other_place'])) if on]
        L.append('| %d | %s | %d | %s | %.3g | %d | %s | %s | %s | %s | %s |' % (
            row['rank'], row['replicon'], row['position'], OUT.fmt9(row['score']), row['places_higher_per_mb'],
            row['n_merged'], row.get('position_type') or '—', row.get('nearest_gene') or '—',
            '—' if row.get('gene_distance') is None else row['gene_distance'], row['larger_pass'],
            ', '.join(flags) or ''))
    L.append('')
    L.append('Полный список (%d мест) — `places.tsv`.' % len(rows))
    L.append('')
    L.append('## Карты вклада и логотип')
    L.append('')
    L.append('- карт: %d (верхние места; `contribution_maps.tsv.gz`), логотип — `logo.tsv`, `logo.svg`' % len(maps))
    L.append('- ' + logo_note)
    L.append('- ' + CAVEAT_MAP)
    L.append('- веса сбора по остаткам — `residue_weights.tsv`, `residue_weights.svg`. ' + CAVEAT_RESIDUES)
    L.append('')
    L.append('## Чего здесь нет')
    L.append('')
    L.append(NOT_GIVEN)
    L.append('')
    L.append('## Файлы')
    L.append('')
    L.append('- `track/*.tsv.gz` — дорожка: репликон, положение от 1, оценка (девять значащих знаков; пусто — оценки '
             'нет, ноль — обычная оценка); `track/*.bw` — та же дорожка в bigWig (интервал основания `[i, i + 1)` с '
             'нуля, как принято в bigWig; просмотрщик показывает его от 1)')
    L.append('- `mask/*.tsv.gz` — отрезки оснований с оценкой и без неё, от 1, оба конца включительно')
    L.append('- `places.tsv`, `contribution_maps.tsv.gz`, `logo.tsv`, `logo.svg`, `residue_weights.*`%s'
             % (', `show.svg` — рисунок показа' if show else ''))
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write('\n'.join(L) + '\n')


def parser():
    p = argparse.ArgumentParser(prog='prokabind', description='ProkaBind: оценка каждого основания генома для белка '
                                '(SPEC раздел 5). Режимы: run — запуск; show — показ до фиксации (п. 5.8).')
    sub = p.add_subparsers(dest='mode', required=True)
    for mode in ('run', 'show'):
        s = sub.add_parser(mode)
        s.add_argument('--protein', required=True, help='FASTA одного белка, 1–1 022 остатка (п. 5.1)')
        s.add_argument('--genome', required=True, help='FASTA генома: один или несколько репликонов')
        s.add_argument('--annotation', help='GBFF или GFF3; без неё нет типа положения и ближайшего гена')
        s.add_argument('--circular', action='append', metavar='REPLICON', help='репликон кольцевой (ключ сильнее аннотации)')
        s.add_argument('--linear', action='append', metavar='REPLICON', help='репликон линейный')
        s.add_argument('--out', required=True, help='каталог результатов (пустой или новый)')
        s.add_argument('--model', default=DEFAULT_MODEL, help='папка модели: model.pt и journal.json (§ 3.0.2)')
        s.add_argument('--model-kind', default='pilot', choices=('pilot', 'fold', 'final'),
                       help='что это за модель — печатается в поле «какая модель» (п. 5.6)')
        s.add_argument('--config', default=DEFAULT_CONFIG)
        s.add_argument('--esm-cache-read', action='append', default=None, metavar='DIR',
                       help='кэш признаков ESM-2 только для чтения (по умолчанию — кэш проекта %s)' % DEFAULT_PROJECT_CACHE)
        s.add_argument('--esm-cache', default=os.path.join(os.path.expanduser('~'), '.cache', 'prokabind', 'esm2'),
                       help='кэш записи признаков ESM-2: признаки нового белка считаются, пишутся сюда и читаются отсюда')
        s.add_argument('--data-root', default=DEFAULT_DATA, help='данные проекта (последовательности обучающих белков; '
                       'у show — допустимость показа)')
        s.add_argument('--training-fasta', help='последовательности обучающих белков модели (по умолчанию — из '
                       'proteins.fasta по журналу обучения)')
        s.add_argument('--top', type=int, default=TOP)
        s.add_argument('--maps', type=int, default=MAPS)
        s.add_argument('--figure-places', type=int, default=FIGURE_PLACES)
        s.add_argument('--orientation', default='auto', choices=('auto', 'confirmed', 'unconfirmed'))
        s.add_argument('--orientation-report', default=DEFAULT_ORIENTATION_REPORT)
        s.add_argument('--device', default=None, help='cpu или cuda; по умолчанию — видеокарта, если есть')
        if mode == 'show':
            s.add_argument('--journal', default=DEFAULT_JOURNAL, help='журнал показов (CONTRACTS § 3.0.1)')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.esm_cache_read is None:
        a.esm_cache_read = [DEFAULT_PROJECT_CACHE] if (ROOT / DEFAULT_PROJECT_CACHE).is_dir() else []
    return execute(a, a.mode == 'show')


if __name__ == '__main__':
    sys.exit(main())
