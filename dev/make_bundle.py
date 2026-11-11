"""Собрать «набор модели» для предсказаний (разработка, решение владельца 20): ансамбль плотных моделей и всё, что нужно
переносу мотива, — в одну папку ВНЕ git (в ней последовательности сайтов и белков источников):

  {OUT}/members/{имя}/config.json, journal.json, model.pt      — члены ансамбля (точка обучения названа при сборке)
  {OUT}/train_proteins.fasta                                    — обучающие белки (для поиска родственника)
  {OUT}/site_bank.pkl                                           — сайты обучающих белков и их группы вариантов
                                                                  (объединение обучающих белков членов)
  {OUT}/bundle.json                                             — что это, из чего, суммы файлов
Запуск: ./.venv/bin/python -I -B dev/make_bundle.py OUT RUN_DIR@UPDATE [RUN_DIR@UPDATE ...]
        [--rp-pack regprecise_full.pkl --rp-fasta data/derived/regprecise_full/regulators.fasta] [--skip-lanes exact]"""
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dense'))
from common import *

out = sys.argv[1]
members = [x for i, x in enumerate(sys.argv[2:], 2) if not x.startswith('--') and not sys.argv[i - 1].startswith('--')]
os.makedirs(os.path.join(out, 'members'), exist_ok=True)
train = None
names = []
for spec in members:
    run, update = (spec.split('@') + [None])[:2]
    name = os.path.basename(run.rstrip('/'))
    d = os.path.join(out, 'members', name)
    os.makedirs(d, exist_ok=True)
    for f in ('config.json', 'journal.json'):
        shutil.copy(os.path.join(run, f), d)
    shutil.copy(os.path.join(run, 'model.pt') if update is None else os.path.join(run, 'checkpoints', update + '.pt'), os.path.join(d, 'model.pt'))
    # банк переноса мотива — объединение обучающих белков членов: «соперник видит те же данные, что модель» (SPEC
    # п. 3.3.1), а ансамбль видел всё, что видел хоть один его член. У прогона без списка (ранние прогоны ночи) список
    # не добавляется; хоть у одного члена он должен быть.
    tpath = os.path.join(run, 'train_proteins.json')
    listed = os.path.exists(tpath)
    if listed:
        tp = set(json.load(open(tpath)))
        train = tp if train is None else (train | tp)
    names.append({'name': name, 'run': run, 'update': update, 'train_proteins_listed': listed})
if train is None:
    sys.exit('ни у одного члена нет train_proteins.json: банк переноса мотива собрать не из чего')
# `--skip-lanes exact[,…]` — не класть в банк сайты названных дорожек координатного пакета (набор для показа вне проекта:
# точные сайты пришли из RegulonDB и CollecTF, чьи условия не дают раздавать копии данных; `DATA_POLICY.md`, раздел 11)
SKIP_LANES = set(sys.argv[sys.argv.index('--skip-lanes') + 1].split(',')) if '--skip-lanes' in sys.argv else set()
lib = pickle.load(open(os.path.join(PACKS, 'library.pkl'), 'rb'))
split = {r['protein']: r for r in read_tsv('data/derived/split.tsv')}
letters = 'ACGT'
sites = collections.defaultdict(set)
# координатные строки обучения: пакет, на котором учились члены (`--coord bac`), и тот же набор строк из общего пакета
# (без сайтов мотива MEME в пиках — члены набора на них не учились)
for pack in ('e2_data_bac.pkl', 'coord_bacpeaks.pkl'):
    P = pickle.load(open(os.path.join(PACKS, pack), 'rb'))
    for i, p in enumerate(P['protein']):
        if p not in train or P['lane'][i] == 'peaksite' or P['lane'][i] in SKIP_LANES:
            continue
        w = P['end'][i] - P['start'][i]
        if w > W:
            continue
        lo = HALF - (w - 1) // 2
        s = ''.join(letters[c] if c < 4 else 'N' for c in P['pos'][i][lo:lo + w])
        if 'N' not in s:
            sites[p].add(s)
# сайты библиотеки — только у белков, допущенных правилом утечки п. 2.7.3 (нет родства ≥ 40% с белками fold0, контрольной
# части и отложенного набора): то же условие, что у обучения (`train_dense.py`) и у честного соперника (`pwm_full.py`).
# Наш обучающий белок, совпадающий с белком библиотеки, но родственный закрытому белку, учится своими координатными
# строками, а его сайты библиотеки в банк не идут (ревизия 10.11, находка 2: без условия в банк попали 1 553 таких сайта
# у 33 белков, и набор не был чист для будущей оценки на белках fold0 и отложенного набора).
closed = {'fold0', 'control', 'holdout'}
lib_ok = {q for q, m in lib['meta'].items() if not (m['related_parts'] & closed)}
skipped_lib = collections.Counter()
for p, site in zip(lib['rows']['protein'], lib['rows']['site']):
    if p not in train:
        continue
    if p not in lib_ok:
        skipped_lib[p] += 1
        continue
    sites[p].add(''.join(letters[c] for c in site))


# дорожка RegPrecise (круг улучшений 10.11): сайты регуляторов, на которых учились члены (`--rp-pack PACK.pkl`), — в банк;
# допуск тот же, что у обучения: нет родства ≥ 40% с белками fold0, контрольной части и отложенного набора
rp_meta, rp_seqs = {}, {}
if '--rp-pack' in sys.argv:
    rp = pickle.load(open(os.path.join(PACKS, sys.argv[sys.argv.index('--rp-pack') + 1]), 'rb'))
    rp_meta = rp['meta']
    n_rp = 0
    for i, p in enumerate(rp['protein']):
        if p not in train or (rp_meta[p]['related_parts'] & closed):
            continue
        w = rp['end'][i] - rp['start'][i]
        lo = HALF - (w - 1) // 2
        s_ = ''.join(letters[c] if c < 4 else 'N' for c in rp['pos'][i][lo:lo + w])
        if 'N' not in s_:
            sites[p].add(s_); n_rp += 1
    name = None
    for line in open(os.path.join(ROOT, sys.argv[sys.argv.index('--rp-fasta') + 1])):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
        elif name:
            rp_seqs[name] = line
    print('RegPrecise: в банк добавлено %d сайтов' % n_rp)


def group_of(p):
    if p in rp_meta and p not in split and p not in lib['meta']:
        return 'rp:' + rp_meta[p]['variant_group']
    if p in split:
        return 'ours:' + split[p]['variant_group']
    if p in lib['meta']:
        return 'lib:' + lib['meta'][p]['variant_group']
    return 'bac:' + p


seqs = {}
for path in ('data/derived/proteins.fasta', 'data/derived/sigmoid/proteins.fasta', os.path.join(NIGHT, 'library/proteins.fasta')):
    name = None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:].split()[0]
            seqs.setdefault(name, '')
            keep = seqs[name] == ''
        elif name and keep:
            seqs[name] += line
for k_, v_ in rp_seqs.items():
    seqs.setdefault(k_, v_)
with open(os.path.join(out, 'train_proteins.fasta'), 'w') as f:
    for p in sorted(train):
        if p in sites and p in seqs:
            f.write('>%s\n%s\n' % (p, seqs[p]))
bank = {'sites': {p: sorted(v) for p, v in sites.items()}, 'group': {p: group_of(p) for p in sites}}
pickle.dump(bank, open(os.path.join(out, 'site_bank.pkl'), 'wb'), protocol=4)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


info = {'made_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'members': names,
        'training_proteins_with_sites': len(bank['sites']), 'sites_in_bank': sum(len(v) for v in bank['sites'].values()),
        'library_rows_left_out_by_leak_rule': {'proteins': len(skipped_lib), 'rows': sum(skipped_lib.values())},
        'note': 'разработка 10.11.2026; обучающая часть захода 0; белки fold0, control, holdout не учились, и сайтов библиотеки белков, родственных им на 40% и больше, в банке нет',
        'bank_lanes_left_out': sorted(SKIP_LANES),
        'sha256': {os.path.relpath(os.path.join(dp, f), out): sha(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs if f != 'bundle.json'}}
json.dump(info, open(os.path.join(out, 'bundle.json'), 'w'), indent=1, ensure_ascii=False)
print('набор: %d членов, %d обучающих белков с сайтами (%d сайтов); правило утечки сняло %d строк библиотеки у %d белков -> %s'
      % (len(names), len(bank['sites']), info['sites_in_bank'], sum(skipped_lib.values()), len(skipped_lib), out))
