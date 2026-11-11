"""Пакет окон для оценки на машине без геномов (круг улучшений, решение владельца 21).

Из пакета пар «белок + запись генома» (формат `e0_setup.pkl`: сайты, центры фона, запретные основания) вырезаются окна
ровно так, как их режет `eval_control.py` (`window_scores_model`): у каждого сайта — отрезок сайта с контекстом 70, у
фона — окно ширины самого широкого сайта пары с контекстом; для каждого сайта отмечено, какие центры фона годны (окно
его ширины внутри записи и не задевает запретных оснований). Оценка по такому пакету — `eval_windows.py`; на
контрольной части она обязана совпасть с `eval_control.py`.

Запуск: ./.venv/bin/python -I -B dev/dense/winpack.py SETUP.pkl CONTROL.pkl OUT.pkl
  контрольная часть: winpack.py e0_setup.pkl e2_control.pkl control_win.pkl
  белки разработки:  winpack.py bacdev_setup.pkl bacdev_control.pkl bacdev_win.pkl"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

setup, control, out = (x if os.path.isabs(x) else os.path.join(PACKS, x) for x in sys.argv[1:4])
S = pickle.load(open(setup, 'rb'))
C = pickle.load(open(control, 'rb'))
dirs = D.genome_dirs(json.load(open(CFG_BAC)))
_seq = {}


def genome_seq(g):
    if g not in _seq:
        if len(_seq) > 40:
            _seq.clear()
        _seq[g] = D.genome_sequence(dirs, g).upper()
    return _seq[g]


def cut(seq, s, e, circular):
    n = len(seq)
    if s >= 0 and e <= n:
        return seq[s:e]
    if not circular:
        return None
    return ''.join(seq[i % n] for i in range(s, e))


items = []
for it in S['items']:
    p, g = it['protein'], it['genome']
    if p not in C['band']:
        continue
    seq = genome_seq(g); n = it['length']; circ = it['circular']
    blocked = np.zeros(n, dtype=bool); blocked[it['blocked_idx']] = True
    closed = np.concatenate(([0], np.cumsum(blocked)))
    wins = []
    for (s0, e0) in it['sites']:
        w = e0 - s0
        if w < 1 or w > n:
            continue
        st = it['centres'] - (w - 1) // 2
        ok = (st >= 0) & (st + w <= n)
        okk = np.zeros(it['centres'].size, dtype=bool)
        okk[ok] = (closed[st[ok] + w] - closed[st[ok]]) == 0
        wins.append((s0, w, okk))
    if not wins:
        continue
    wmax = max(w for _, w, _ in wins)
    seg_start = it['centres'] - (wmax - 1) // 2
    segs, keep = [], []
    for k, s in enumerate(seg_start.tolist()):
        t = cut(seq, s - CTX, s + wmax + CTX, circ)
        if t is None or set(t) - set('ACGT'):
            continue
        segs.append(codes(t)); keep.append(k)
    site_segs = []
    for s0, w, _ in wins:
        t = cut(seq, s0 - CTX, s0 + w + CTX, circ)
        site_segs.append(None if (t is None or set(t) - set('ACGT')) else codes(t))
    comp = [seq.count(b) for b in 'ACGT']
    items.append({'protein': p, 'genome': g, 'sites': [(s0, s0 + w) for s0, w, _ in wins], 'w': [w for _, w, _ in wins], 'wmax': wmax,
                  'site_segs': site_segs, 'bg_segs': np.stack(segs) if segs else np.zeros((0, wmax + 2 * CTX), dtype=np.uint8),
                  'bg_keep': np.asarray(keep, dtype=np.int64), 'ok': np.stack([okk for _, _, okk in wins]),
                  'n_centres': int(it['centres'].size), 'composition': comp})
pickle.dump({'items': items, 'band': C['band'], 'groups': C['groups'], 'allowed': C['allowed'], 'source': [setup, control]},
            open(out, 'wb'), protocol=4)
print('пакет окон: пар %d, белков %d, сайтов %d, окон фона %d -> %s (%.1f МБ)' % (
    len(items), len({it['protein'] for it in items}), sum(len(it['w']) for it in items), sum(len(it['bg_keep']) for it in items),
    out, os.path.getsize(out) / 1e6))
