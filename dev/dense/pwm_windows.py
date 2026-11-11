"""Оценка мотива (матрицы `eval_defs.build_pwm`) на окнах пакета `winpack.py` — то же число, что даёт
`eval_defs.motif_transfer` в основании генома (лучшее размещение матрицы с центром в основании, обе цепи), но без
прохода по всему геному: окна пакета несут контекст 70 оснований с каждой стороны (круг улучшений, решение владельца 21).

`pwm_for(sites, composition)` — матрица по рецепту SPEC п. 3.3.1 при составе генома, записанном в пакете;
`item_pwm_scores(pwm, item)` — (оценки сайтов, оценки фона по сайтам) в том же виде, что у `eval_windows.item_scores`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import eval_defs as E


def pwm_for(sites, composition):
    """состав — числа оснований A, C, G, T генома; матрице нужны только их доли"""
    total = float(sum(composition))
    surrogate = ''.join(b * max(1, int(round(100000 * c / total))) for b, c in zip('ACGT', composition))
    return E.build_pwm(sites, surrogate)


def base_scores(pwm, segs):
    """[n, L − 2·CTX]: оценка мотива в каждом оцениваемом основании окон `segs` [n, L] (коды 0–3)"""
    k = pwm.shape[0]
    n, L = segs.shape
    out = np.full((n, L - 2 * CTX), np.nan)
    if k == 0 or k > L:
        return out
    starts = L - k + 1
    fwd = np.zeros((n, starts)); rev = np.zeros((n, starts))
    idx = segs.astype(np.int64)
    comp = pwm[::-1, ::-1]
    for j in range(k):
        col = idx[:, j:j + starts]
        fwd += pwm[j][col]
        rev += comp[j][col]
    best = np.maximum(fwd, rev)
    centres = np.arange(CTX, L - CTX)
    s = centres - (k - 1) // 2
    ok = (s >= 0) & (s < starts)
    out[:, ok] = best[:, s[ok]]
    return out


def item_pwm_scores(pwm, it, thin=False):
    keep, n_c, wmax = it['bg_keep'], it['n_centres'], it['wmax']
    bg_scores = base_scores(pwm, it['bg_segs']) if keep.size else np.zeros((0, wmax))
    pos, bg = [], []
    for k_, w in enumerate(it['w']):
        seg = it['site_segs'][k_]
        pos.append(np.nan if seg is None else float(np.nanmax(base_scores(pwm, seg[None, :])[0])))
        off = (wmax - 1) // 2 - (w - 1) // 2
        vals = np.full(n_c, np.nan)
        if keep.size:
            sub = bg_scores[:, off:off + w]
            vals[keep] = np.where(np.all(np.isnan(sub), axis=1), np.nan, np.nanmax(np.where(np.isnan(sub), -np.inf, sub), axis=1))
        sel = it['ok'][k_] if not thin else (it['ok'][k_] & (np.arange(n_c) % 3 == 0))
        bg.append(vals[sel])
    return np.asarray(pos, dtype=float), bg


if __name__ == '__main__':                      # сверка с eval_defs.motif_transfer на настоящем геноме и с оценками pwm_full.py
    WP = pickle.load(open(os.path.join(PACKS, 'control_win.pkl'), 'rb'))
    WS = pickle.load(open(os.path.join(PACKS, 'pwmfull_window_scores.pkl'), 'rb'))
    dirs = D.genome_dirs(json.load(open(CFG_BAC)))
    rng = np.random.default_rng(5)
    it = WP['items'][3]
    seq = D.genome_sequence(dirs, it['genome']).upper()
    sites = [''.join('ACGT'[c] for c in s[CTX:len(s) - CTX]) for s in it['site_segs'] if s is not None][:6] or ['ACGTTGACA']
    track = E.motif_transfer(sites, seq)
    pwm_true = E.build_pwm(sites, seq)
    pwm = pwm_for(sites, it['composition'])
    print('матрица: наибольшее расхождение с build_pwm на геноме %.2e' % np.abs(pwm - pwm_true).max())
    pos, bg = item_pwm_scores(pwm_true, it)
    # эталон: максимум дорожки motif_transfer по основаниям сайта и по окну фона
    s0, e0 = it['sites'][0]
    print('сайт: окна %.6f, дорожка %.6f' % (pos[0], np.nanmax(track[s0:e0])))
