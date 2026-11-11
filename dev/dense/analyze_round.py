"""Разбор оценок круга улучшений дома (решение владельца 21): по выходам `eval_windows.py` на контрольной части и на
белках разработки BacRegDB — AUROC по полосам, сумма с переносом мотива (контрольная часть, соперник `pwmfull`),
«свой − чужой», «свой − нулевой вектор» и парные разницы рук с опорной рукой.

Файлы: {DIR}/ev_{РУКА}_control.pkl и {DIR}/ev_{РУКА}_bacdev.pkl (в каждом — запись с именем руки).
Запуск: ./.venv/bin/python -I -B dev/dense/analyze_round.py DIR РУКА [РУКА ...] [--base РУКА] [--out FILE.json]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

argv = sys.argv[1:]
opt = lambda name, default=None: argv[argv.index(name) + 1] if name in argv else default
folder = argv[0]
arms = [x for x in argv[1:] if not x.startswith('--') and x != opt('--base') and x != opt('--out')]
base = opt('--base', arms[0])
if base not in arms:
    arms = [base] + arms
WS = pickle.load(open(os.path.join(PACKS, 'pwmfull_window_scores.pkl'), 'rb'))
CW = pickle.load(open(os.path.join(PACKS, 'control_win.pkl'), 'rb'))
BW = pickle.load(open(os.path.join(PACKS, 'bacdev_win.pkl'), 'rb'))
origin = json.load(open(os.path.join(PACKS, 'bacregdb_origin.json'))) if os.path.exists(os.path.join(PACKS, 'bacregdb_origin.json')) else {}
P = pickle.load(open(os.path.join(PACKS, 'e2_data_bac.pkl'), 'rb'))
row_class = {}
for k, v in origin.items():
    for i in v:
        row_class[(P['protein'][i], P['genome'][i], int(P['start'][i]), int(P['end'][i]))] = k
named = {}                      # (белок, геном) -> маска сайтов «названных по гену» (не выход поиска по профилю)
for it in BW['items']:
    named[(it['protein'], it['genome'])] = np.asarray([row_class.get((it['protein'], it['genome'], s, e)) in ('named', 'regprecise_only', 'search_and_regprecise')
                                                        for s, e in it['sites']])


# «чистые» сайты белков разработки: без общего отрезка ≥ 12 оснований с обучающими строками рук (ревизия круга, Н1;
# маску строит prep_bacdev_clean.py)
clean = {}
_cp = os.path.join(PACKS, 'bacdev_clean_sites.json')
if os.path.exists(_cp):
    for k, v in json.load(open(_cp))['mask'].items():
        clean[tuple(k.split('|'))] = np.asarray(v, dtype=bool)


def auroc(pos, bg):
    b = np.concatenate(bg).astype(float) if len(bg) else np.zeros(0)
    p, b = np.asarray(pos, dtype=float), b
    p, b = p[~np.isnan(p)], b[~np.isnan(b)]
    return float(T._auroc(p, b)) if p.size and b.size else None


def recall1(pos, bg, q=0.99):
    """доля сайтов с оценкой выше 99% фона пары — приближение «верха списка» (первые ~1% мест)"""
    b = np.concatenate(bg).astype(float) if len(bg) else np.zeros(0)
    p = np.asarray(pos, dtype=float)
    p, b = p[~np.isnan(p)], b[~np.isnan(b)]
    if not p.size or b.size < 100:
        return None
    return float(np.mean(p > np.quantile(b, q)))


def zparts(first, second):
    (p1, b1), (p2, b2) = first, second
    B1, B2 = np.concatenate(b1).astype(float), np.concatenate(b2).astype(float)
    if B1.size != B2.size:
        return None
    ok = ~np.isnan(B1) & ~np.isnan(B2); okp = ~np.isnan(p1) & ~np.isnan(p2)
    if not ok.any() or not okp.any():
        return None
    z = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-9)
    return z(p1[okp], B1[ok]) + z(p2[okp], B2[ok]), z(B1[ok], B1[ok]) + z(B2[ok], B2[ok])


def zsum(first, second):
    got = zparts(first, second)
    return None if got is None else float(T._auroc(got[0], got[1]))


def per_protein(pairs, key='own', mask=None, fn=None):
    fn = fn or auroc
    tmp = collections.defaultdict(list)
    for (p, g), rec in pairs.items():
        if key not in rec:
            continue
        pos, bg = rec[key]
        if mask is not None:
            m = mask.get((p, g))
            if m is None or not m.any():
                continue
            pos, bg = pos[m], [b for b, keep in zip(bg, m) if keep]
        v = fn(pos, bg)
        if v is not None:
            tmp[p].append(v)
    return {p: float(np.mean(v)) for p, v in tmp.items()}


def bands(per, band, groups, other=None, names=('all', '>=40', '40-70', '<40')):
    out = {}
    for b in names:
        sel = [p for p in per if (b == 'all' or band[p] == b or (b == '>=40' and band[p] != '<40')) and (other is None or p in other)]
        byg = collections.defaultdict(list)
        for p in sel:
            byg[groups[p]].append(per[p] - (other[p] if other is not None else 0.0))
        d = [float(np.mean(v)) for v in byg.values()]
        if not d:
            continue
        if other is None:
            out[b] = '%.3f (%d)' % (np.mean(d), len(d))
        elif len(d) >= 4:
            m, lo, hi = boot(d)
            out[b] = '%+.3f [%+.3f; %+.3f] %d/%d' % (m, lo, hi, sum(1 for x in d if x > 0), len(d))
    return out


show = lambda d: ' | '.join('%s %s' % kv for kv in d.items())
res = {}
for arm in arms:
    r = {}
    for part, WPK in (('control', CW), ('bacdev', BW)):
        path = os.path.join(folder, 'ev_%s_%s.pkl' % (arm, part))
        if not os.path.exists(path):
            continue
        pairs = pickle.load(open(path, 'rb'))[arm]['pairs']
        band, groups = WPK['band'], WPK['groups']
        d = {'own': per_protein(pairs), 'own_r1': per_protein(pairs, fn=recall1)}
        if part == 'control':
            hyb, pwm = collections.defaultdict(list), collections.defaultdict(list)
            hyb_r, pwm_r = collections.defaultdict(list), collections.defaultdict(list)
            for (p, g), rec in pairs.items():
                ref = WS.get((p, g), {}).get('pwmfull')
                if ref is None:
                    continue
                v1, v2 = auroc(*ref), zsum(ref, rec['own'])
                if v1 is not None and v2 is not None:
                    pwm[p].append(v1); hyb[p].append(v2)
                    zp = zparts(ref, rec['own'])
                    r1, r2 = recall1(*ref), recall1(zp[0], [zp[1]])
                    if r1 is not None and r2 is not None:
                        pwm_r[p].append(r1); hyb_r[p].append(r2)
            d['pwm'] = {p: float(np.mean(v)) for p, v in pwm.items()}
            d['hyb'] = {p: float(np.mean(v)) for p, v in hyb.items()}
            d['pwm_r1'] = {p: float(np.mean(v)) for p, v in pwm_r.items()}
            d['hyb_r1'] = {p: float(np.mean(v)) for p, v in hyb_r.items()}
        else:
            d['own_named'] = per_protein(pairs, mask=named)
            if clean:
                d['own_clean'] = per_protein(pairs, mask=clean)
                d['own_clean_r1'] = per_protein(pairs, mask=clean, fn=recall1)
        for key, thin in (('null', 'own_thin'), ('foreign', 'own_thin')):
            a_, b_ = collections.defaultdict(list), collections.defaultdict(list)
            for (p, g), rec in pairs.items():
                if key not in rec or thin not in rec:
                    continue
                v1 = auroc(*rec[thin])
                if key == 'null':
                    v0 = auroc(*rec['null'])
                else:
                    fv = [auroc(*x) for x in rec['foreign']]
                    fv = [x for x in fv if x is not None]
                    v0 = float(np.mean(fv)) if fv else None
                if v0 is not None and v1 is not None:
                    a_[p].append(v1); b_[p].append(v0)
            if a_:
                d['thin_' + key] = {p: float(np.mean(v)) for p, v in a_.items()}
                d[key] = {p: float(np.mean(v)) for p, v in b_.items()}
        r[part] = d
        print('%-8s %-7s own: %s' % (arm, part, show(bands(d['own'], band, groups))))
        if 'hyb' in d:
            print('%-8s %-7s   сумма с переносом мотива: %s ; сумма − соперник: %s ; модель − соперник: %s' % (
                '', '', show(bands(d['hyb'], band, groups, names=('>=40', '40-70'))), show(bands(d['hyb'], band, groups, d['pwm'], names=('>=40', '40-70'))),
                show(bands({p: d['own'][p] for p in d['pwm']}, band, groups, d['pwm'], names=('>=40',)))))
        print('%-8s %-7s   верх списка (доля сайтов выше 99%% фона): модель %s%s' % ('', '', show(bands(d['own_r1'], band, groups)),
              (' ; сумма %s ; соперник %s ; сумма − соперник %s' % (show(bands(d['hyb_r1'], band, groups, names=('>=40', '40-70'))),
                                                               show(bands(d['pwm_r1'], band, groups, names=('>=40', '40-70'))),
                                                               show(bands(d['hyb_r1'], band, groups, d['pwm_r1'], names=('>=40', '40-70'))))) if 'hyb_r1' in d else ''))
        if 'own_named' in d:
            print('%-8s %-7s   только сайты не из поиска по профилю: %s' % ('', '', show(bands(d['own_named'], band, groups))))
        if 'own_clean' in d:
            print('%-8s %-7s   только «чистые» сайты (без общих 12 оснований с обучающими строками): AUROC %s ; верх списка %s' % (
                '', '', show(bands(d['own_clean'], band, groups)), show(bands(d['own_clean_r1'], band, groups))))
        if 'null' in d:
            print('%-8s %-7s   свой − нулевой вектор: %s' % ('', '', show(bands(d['thin_null'], band, groups, d['null'], names=('all', '<40')))))
        if 'foreign' in d:
            print('%-8s %-7s   свой − чужой: %s' % ('', '', show(bands(d['thin_foreign'], band, groups, d['foreign'], names=('all', '<40')))))
    res[arm] = r
print()
for arm in arms:
    if arm == base:
        continue
    for part, WPK in (('control', CW), ('bacdev', BW)):
        if part not in res[arm] or part not in res[base]:
            continue
        band, groups = WPK['band'], WPK['groups']
        for key in ('own', 'hyb', 'own_named', 'own_r1', 'hyb_r1', 'own_clean', 'own_clean_r1'):
            if key in res[arm][part] and key in res[base][part]:
                print('%-8s − %-8s %-7s %-9s %s' % (arm, base, part, key, show(bands(res[arm][part][key], band, groups, res[base][part][key]))))
if opt('--out'):
    json.dump(res, open(opt('--out'), 'w'))
