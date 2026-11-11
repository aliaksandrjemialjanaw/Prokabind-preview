"""ProkaBind: белок + геном → список мест, где белок, скорее всего, связывается с ДНК.

Оценка места = z(средняя z-оценок моделей ансамбля со своим белком) + z(«перенос мотива» от ближайшего обучающего
родственника), если родственник есть: идентичность ≥ 40% при покрытии ≥ 80% более короткого белка (поиск — MMseqs2),
матрица — по сайтам родственника. Место — окно 21 основание; его оценка — наибольшая оценка основания в окне; z — по
всем местам генома. Без родственника работает один ансамбль.

Запуск (из корня репозитория):
  python predict.py --protein БЕЛОК.fasta|ПОСЛЕДОВАТЕЛЬНОСТЬ --genome ГЕНОМ.fasta [--circular] [--top 50]
                    [--out sites.tsv] [--threads 8] [--device cpu|cuda] [--no-motif] [--bundle models/dev_v3]
Выход — таблица мест по убыванию оценки: репликон; начало и конец места (с нуля, конец не включён); итоговая оценка;
вклад ансамбля; вклад мотива родственника; чем уточнён центр места (motif — лучшим размещением матрицы родственника,
ensemble — основанием с наибольшей оценкой ансамбля); последовательность места. `--tracks FILE.npz` пишет дорожки
оценок по всему геному. Признаки ESM-2 нового белка считаются один раз (при первом запуске пакет fair-esm скачивает
веса ESM-2 650M, около 2,5 ГБ) и кладутся в cache/esm2."""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description='ProkaBind: места связывания белка-регулятора в геноме')
ap.add_argument('--protein', required=True, help='файл FASTA с одним белком или сама последовательность')
ap.add_argument('--genome', required=True, help='файл FASTA с геномом (один или несколько репликонов)')
ap.add_argument('--bundle', default=os.path.join(ROOT, 'models', 'dev_v3'), help='папка набора модели')
ap.add_argument('--circular', action='store_true', help='считать репликоны кольцевыми (у бактерий обычно так); иначе у краёв по 70 оснований оценки нет')
ap.add_argument('--top', type=int, default=50, help='сколько мест выдать')
ap.add_argument('--out', default=None, help='файл таблицы мест (иначе — на экран)')
ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--device', default='cpu')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--tracks', default=None, help='записать дорожки оценок (npz): ансамбль, мотив, сумма, основания')
ap.add_argument('--no-motif', action='store_true', help='только ансамбль моделей, без переноса мотива')
a = ap.parse_args()
# пути пользователя — относительно папки, из которой он запустил программу (код ниже работает из корня репозитория)
for name in ('protein', 'genome', 'bundle'):
    value = getattr(a, name)
    if os.path.exists(value):
        setattr(a, name, os.path.abspath(value))
for name in ('out', 'tracks'):
    if getattr(a, name):
        setattr(a, name, os.path.abspath(getattr(a, name)))
if not os.path.isfile(a.genome):
    sys.exit('геном: нет файла %s (нужен файл FASTA)' % a.genome)
if not a.no_motif and shutil.which('mmseqs') is None:
    sys.exit('не найдена программа mmseqs (MMseqs2): она ищет изученного родственника белка. Установите её '
             '(например, conda install -c bioconda mmseqs2) или запустите с --no-motif — тогда работает один ансамбль')

sys.path.insert(0, os.path.join(ROOT, 'dev'))
from predictor import *                                                  # noqa: E402,F401,F403
from prokabind.features import protein_features                          # noqa: E402

torch.set_num_threads(a.threads)
dev = torch.device(a.device)

seq = read_fasta(a.protein)[0][1] if os.path.exists(a.protein) else a.protein
seq = ''.join(seq.split()).upper().rstrip('*')
if not seq or set(seq) - set('ACDEFGHIKLMNPQRSTVWYXBZUO'):
    sys.exit('белок: пустая последовательность или не аминокислоты (%s)' % ''.join(sorted(set(seq) - set('ACDEFGHIKLMNPQRSTVWYXBZUO'))))
if len(seq) > 1022:
    sys.exit('белок длиннее 1 022 остатков — предел ESM-2')
key = hashlib.sha256(seq.encode()).hexdigest()
replicons = [(n, s.upper(), bool(a.circular)) for n, s in read_fasta(a.genome)]
if not a.circular:
    print('репликоны считаются линейными (у краёв по 70 оснований оценки нет); для кольцевого генома — ключ --circular')
print('белок %s… (%d остатков); репликонов %d, оснований %d' % (key[:12], len(seq), len(replicons), sum(len(s) for _, s, _ in replicons)), flush=True)

# 1. признаки ESM-2: из папки cache/esm2 (там лежат признаки белка примера); нового белка — посчитать и положить туда же
cache = os.path.join(ROOT, 'cache', 'esm2')
feat, info = protein_features(seq, key, [cache], cache)
features = torch.from_numpy(feat).unsqueeze(0)
mask = torch.ones(1, feat.shape[0], dtype=torch.bool)
print('признаки ESM-2: %s' % {'read_cache': 'взяты из cache/esm2', 'computed_now': 'посчитаны сейчас'}[info['origin']], flush=True)

# 2. ближайший изученный родственник
bank = pickle.load(open(os.path.join(a.bundle, 'site_bank.pkl'), 'rb'))
relative = None if a.no_motif else find_relative(seq, a.bundle, bank)
if relative is not None:
    print('родственник: %s…, идентичность %.0f%%, сайтов в матрице %d' % (relative['protein'][:12], 100 * relative['identity'], len(relative['sites'])), flush=True)
elif not a.no_motif:
    print('изученного родственника (идентичность ≥ 40%) нет: оценка — только ансамбль моделей', flush=True)

# 3. ансамбль и места
members = load_members(a.bundle, dev)
print('моделей в ансамбле: %d' % len(members), flush=True)
tracks = score_genome_places(members, features, mask, replicons, relative['sites'] if relative else None, batch=a.batch, log=lambda s: print(s, flush=True))
rows = top_places(tracks, replicons, a.top)
left = (WIDTH - 1) // 2
lines = ['replicon\tstart\tend\tscore\tensemble_z\tmotif_z\tcentre_by\tsequence']
for score, k, i, c, by in rows:
    name, s, circ = replicons[k]
    n = len(s)
    lo, hi = c - left, c - left + WIDTH
    window = ''.join(s[j % n] for j in range(lo, hi)) if circ else s[max(0, lo):hi]
    motif = tracks[k]['motif']
    lines.append('%s\t%d\t%d\t%.3f\t%.3f\t%s\t%s\t%s' % (name, lo, hi, score, tracks[k]['ensemble'][i],
                                                          '' if motif is None else '%.3f' % motif[i],
                                                          {'motif_bases': 'motif', 'bases': 'ensemble', 'place': 'place'}[by], window))
if a.tracks:
    np.savez_compressed(a.tracks, **{'%s|%s' % (replicons[k][0], key): np.asarray(v, dtype=np.float32)
                                     for k, tr in enumerate(tracks) for key, v in tr.items() if v is not None})
    print('дорожки записаны: %s' % a.tracks)
text = '\n'.join(lines) + '\n'
if a.out:
    with open(a.out, 'w') as f:
        f.write(text)
    print('записано: %s (%d мест)' % (a.out, len(rows)))
else:
    print(text)
