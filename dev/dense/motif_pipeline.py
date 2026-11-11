"""Сводная таблица «перенос мотива для любого белка» против «системы с моделью» на контрольной части (круг улучшений,
решение владельца 21: «наибольший разрыв между переносом мотива и тем, что даёт только наша модель»).

Перенос мотива сам по себе умеет только то, что ему даёт родственник:
  * белок с обучающим родственником ≥ 40% — соперник SPEC п. 3.3.1 в полном виде (`pwm_full.py`, ключ `pwmfull`);
  * белок без такого родственника — мотив ближайшего по последовательности ДАЛЁКОГО родственника (идентичность 20–40% при
    покрытии ≥ 80%; таблица `control_relatives_low20.json` ночи 10.11, `find_relative(..., low=0.2)` по банку набора
    `dev_v2`) — правило соперника, продлённое вниз; п. 3.3.1 его там не определяет;
  * белок вовсе без родственника ≥ 20% — предсказать нечем: случайность (AUROC 0,5; в верху списка — 1% сайтов).
Система: сумма z «ансамбль + мотив родственника», если родственник ≥ 40% есть, иначе один ансамбль.

Меры — как в `analyze_round.py`: AUROC меры остановки и «верх списка» (доля сайтов выше 99% фона пары), среднее по парам
белка, по белкам группы родства, по группам; интервалы — бутстреп по группам. Только контрольная часть.

Запуск: ./.venv/bin/python -I -B dev/dense/motif_pipeline.py EV_DIR РУКА [РУКА…] [--out OUT.json]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pwm_windows import *

argv = sys.argv[1:]
out_path = argv[argv.index('--out') + 1] if '--out' in argv else None
if out_path:
    del argv[argv.index('--out'):argv.index('--out') + 2]
ev_dir, arms = argv[0], argv[1:]
WP = pickle.load(open(os.path.join(PACKS, 'control_win.pkl'), 'rb'))
items, band, groups = WP['items'], WP['band'], WP['groups']
WS = pickle.load(open(os.path.join(PACKS, 'pwmfull_window_scores.pkl'), 'rb'))
LOW = json.load(open(os.path.join(NIGHT, 'control_relatives_low20.json')))


def clean(sc):
    pos, bg = sc
    b = np.concatenate(bg).astype(float) if len(bg) else np.zeros(0)
    return np.asarray(pos, dtype=float), b


def auroc(sc):
    p, b = clean(sc)
    p, b = p[~np.isnan(p)], b[~np.isnan(b)]
    return float(T._auroc(p, b)) if p.size and b.size else None


def recall1(sc, q=0.99):
    p, b = clean(sc)
    p, b = p[~np.isnan(p)], b[~np.isnan(b)]
    return float(np.mean(p > np.quantile(b, q))) if p.size and b.size >= 100 else None


def zsum(a_, b_):
    """сумма z двух оценок на одних и тех же окнах; None, если окна не совпали"""
    (p1, b1), (p2, b2) = clean(a_), clean(b_)
    if b1.size != b2.size or p1.size != p2.size:
        return None
    ok, okp = ~np.isnan(b1) & ~np.isnan(b2), ~np.isnan(p1) & ~np.isnan(p2)
    if not ok.any() or not okp.any():
        return None
    z = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-9)
    return z(p1[okp], b1[ok]) + z(p2[okp], b2[ok]), [z(b1[ok], b1[ok]) + z(b2[ok], b2[ok])]


# мотив далёкого родственника на окнах пакета — один раз на пару
distant = {}
pwm_cache = {}
for it in items:
    p, g = it['protein'], it['genome']
    rel = LOW.get(p)
    if band[p] != '<40' or not rel or not rel.get('sites'):
        continue
    key = (rel['protein'], tuple(it['composition']))
    if key not in pwm_cache:
        pwm_cache[key] = pwm_for(sorted(rel['sites']), it['composition'])
    distant[(p, g)] = item_pwm_scores(pwm_cache[key], it)
has_distant = {p for (p, g) in distant}
no_relative = sorted({it['protein'] for it in items if band[it['protein']] == '<40'} - has_distant)
print('полоса <40%%: белков с далёким родственником (20–40%%) %d, без родственника %d' % (len(has_distant), len(no_relative)))

result = {}
for arm in arms:
    EV = pickle.load(open(os.path.join(ev_dir, 'ev_%s_control.pkl' % arm), 'rb'))[arm]['pairs']
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        p, g = it['protein'], it['genome']
        if (p, g) not in EV:
            continue
        model = EV[(p, g)]['own']
        if band[p] != '<40':
            ref = WS.get((p, g), {}).get('pwmfull')
            motif = ref
            system = zsum(model, ref) if ref is not None else None
            if ref is None or system is None:                 # родственник без меток — как у analyze_round: пары нет в сравнении
                continue
        else:
            motif = distant.get((p, g))
            system = model
        for name, sc in (('model', model), ('system', system), ('motif', motif)):
            if sc is None:
                per[name + '|auroc'][p].append(0.5); per[name + '|r1'][p].append(0.01)      # предсказать нечем
                continue
            a_, r_ = auroc(sc), recall1(sc)
            if a_ is not None:
                per[name + '|auroc'][p].append(a_)
            if r_ is not None:
                per[name + '|r1'][p].append(r_)
        if band[p] == '<40' and motif is not None:
            s2 = zsum(model, motif)
            if s2 is not None:
                per['model+distant|auroc'][p].append(auroc(s2)); per['model+distant|r1'][p].append(recall1(s2))
    mean = {k: {p: float(np.mean(v)) for p, v in d.items() if v} for k, d in per.items()}

    def table(key, base=None, sel=lambda p: True):
        ps = [p for p in mean.get(key, {}) if sel(p) and (base is None or p in mean.get(base, {}))]
        byg = collections.defaultdict(list)
        for p in ps:
            byg[groups[p]].append(mean[key][p] - (mean[base][p] if base else 0.0))
        v = [float(np.mean(x)) for x in byg.values()]
        if not v:
            return '-'
        if base is None:
            return '%.3f (%d)' % (np.mean(v), len(v))
        m, lo, hi = boot(v)
        return '%+.3f [%+.3f; %+.3f] %d/%d' % (m, lo, hi, sum(1 for x in v if x > 0), len(v))

    classes = [('все белки', lambda p: True), ('родственник ≥ 40%', lambda p: band[p] != '<40'),
               ('без родственника ≥ 40%', lambda p: band[p] == '<40'),
               ('  из них с далёким родственником', lambda p: band[p] == '<40' and p in has_distant),
               ('  из них вовсе без родственника', lambda p: band[p] == '<40' and p not in has_distant)]
    res = {}
    print('== %s' % arm)
    for label, sel in classes:
        row = {}
        for m_ in ('auroc', 'r1'):
            row[m_] = {'motif': table('motif|' + m_, None, sel), 'model': table('model|' + m_, None, sel),
                       'system': table('system|' + m_, None, sel), 'system-motif': table('system|' + m_, 'motif|' + m_, sel)}
        res[label.strip()] = row
        print('  %-34s AUROC: перенос мотива %s | модель %s | система %s | система − перенос %s' % (
            label, row['auroc']['motif'], row['auroc']['model'], row['auroc']['system'], row['auroc']['system-motif']))
        print('  %-34s верх списка: перенос мотива %s | модель %s | система %s | система − перенос %s' % (
            '', row['r1']['motif'], row['r1']['model'], row['r1']['system'], row['r1']['system-motif']))
    sel_d = lambda p: band[p] == '<40' and p in has_distant
    res['model+distant-model'] = {m_: table('model+distant|' + m_, 'model|' + m_, sel_d) for m_ in ('auroc', 'r1')}
    print('  далёкий мотив в сумме с моделью − модель: AUROC %s | верх списка %s' % (res['model+distant-model']['auroc'], res['model+distant-model']['r1']))
    result[arm] = {'table': res, 'per_protein': mean}
if out_path:
    json.dump({'arms': result, 'distant': sorted(has_distant), 'no_relative': no_relative}, open(out_path, 'w'), indent=1, ensure_ascii=False)
