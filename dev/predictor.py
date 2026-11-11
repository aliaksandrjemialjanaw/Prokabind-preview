"""Ядро предсказания (разработка, решение владельца 20): общий код `predict.py` и проверки `scan_check.py`.

Оценка места генома = z(средняя z-оценок моделей ансамбля) + z(оценка «переноса мотива» ближайшего обучающего
родственника), как в мере ночи на контрольной части (`dense/eval_control.py`, «hybrid»): оценка места — наибольшая оценка
основания в окне `WIDTH` оснований с центром в этом основании (мера остановки SPEC п. 4.8 берёт максимум по основаниям
сайта); z — по всем местам генома, где оценка есть. Нет родственника — второго слагаемого нет."""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dense'))
from common import *
import eval_defs as E
from prokabind.model import build, receptive_field
from prokabind.scan import onehot, valid_mask

WIDTH = 21          # ширина места: столько оцениваемых позиций у окна обучения (dense/common.py, SCORED)
SCAN_WINDOW = 4096  # как у прохода проекта (prokabind.scan.WINDOW)


def read_fasta(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            out.append([line[1:].split()[0] if len(line) > 1 else 'seq%d' % (len(out) + 1), []])
        elif line and out:
            out[-1][1].append(line)
    return [(n, ''.join(parts)) for n, parts in out]


def load_members(bundle, device):
    members = []
    for name in sorted(os.listdir(os.path.join(bundle, 'members'))):
        d = os.path.join(bundle, 'members', name)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        journal = json.load(open(os.path.join(d, 'journal.json')))
        cfg['protein_paths'] = (journal.get('choices') or {}).get('protein_paths', 'both')
        m = build(cfg, protein_input=bool(journal.get('protein_input', True)))
        m.load_state_dict(torch.load(os.path.join(d, 'model.pt'), map_location='cpu'))
        if m.retrieval is not None:
            raise ValueError('%s: модель с полем retrieval в набор не берётся — мотив родственника добавляется снаружи' % name)
        members.append((name, m.to(device).eval()))
    return members


def find_relative(sequence, bundle, bank, low=0.4, exclude=()):
    """Ближайший обучающий белок с сайтами: идентичность ≥ `low` при покрытии ≥ 80% более короткого — команда и правило
    CONTRACTS «Выравниватель белков» (поиск в обе стороны, наибольшая fident). Ответ — словарь или None."""
    fasta = os.path.join(bundle, 'train_proteins.fasta')
    best = {}
    with tempfile.TemporaryDirectory() as tmp:
        q = os.path.join(tmp, 'q.fa')
        with open(q, 'w') as f:
            f.write('>query\n%s\n' % sequence)
        for flip, (x, y) in enumerate(((q, fasta), (fasta, q))):
            out = os.path.join(tmp, 'hits%d.m8' % flip)
            subprocess.run(['mmseqs', 'easy-search', x, y, out, os.path.join(tmp, 't%d' % flip), '--exhaustive-search', '-e', '1e-3',
                            '--threads', '1', '--format-output', 'query,target,fident,qcov,tcov,qlen,tlen,evalue'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for line in open(out):
                f = line.rstrip('\n').split('\t')
                cov = float(f[3]) if int(f[5]) <= int(f[6]) else float(f[4])
                t = f[0] if flip else f[1]
                if cov >= 0.8:
                    best[t] = max(best.get(t, 0.0), float(f[2]))
    cand = sorted(((v, t) for t, v in best.items() if v >= low and t in bank['sites'] and t not in exclude), reverse=True)
    if not cand:
        return None
    ident, rel = cand[0]
    group = bank['group'][rel]
    pooled = sorted({s for p, g in bank['group'].items() if g == group for s in bank['sites'].get(p, ())})
    if len(pooled) > 60:                                 # выравнивание build_pwm квадратично: не больше 60 сайтов, отбор по sha256
        pooled = sorted(pooled, key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:60]
    return {'protein': rel, 'identity': ident, 'sites': pooled, 'candidates': len(cand)}


def model_track(model, features, mask, sequence, circular, batch=16):
    """Оценка каждого основания (среднее прямого и обратно-комплементарного проходов) — то же, что
    `prokabind.scan.score_genome`, но пакетами окон и с вектором белка, посчитанным один раз. NaN — оценки нет."""
    rf = receptive_field(model.config)
    context = (rf - 1) // 2
    n = len(sequence)
    scores = np.full(n, np.nan, dtype=np.float32)
    ok = valid_mask(sequence, rf, bool(circular))
    if n < rf:
        return scores
    if circular:
        extended, shift, first, last = sequence[-context:] + sequence + sequence[:context], context, 0, n
    else:
        extended, shift, first, last = sequence, 0, context, n - context
    unit = next(model.parameters()).device
    stride = SCAN_WINDOW - 2 * context
    spans = [(start, min(start + stride, last)) for start in range(first, last, stride)]
    with torch.no_grad():
        vec = model.protein_vector(features.to(unit), mask.to(unit), 1)
        full = [sp for sp in spans if sp[1] - sp[0] == stride]
        rest = [sp for sp in spans if sp[1] - sp[0] != stride]
        groups = [full[i:i + batch] for i in range(0, len(full), batch)] + [[sp] for sp in rest]
        for group in groups:
            x = torch.cat([onehot(extended[a + shift - context:b + shift + context]) for a, b in group], dim=0).to(unit)
            v = vec.expand(x.shape[0], -1)
            y = (model._trunk(x, v) + model._trunk(x.flip(1).flip(2), v).flip(-1)) / 2
            y = y.detach().to('cpu', torch.float32).numpy()
            for k, (a, b) in enumerate(group):
                scores[a:b] = y[k, context:context + (b - a)]
    scores[~ok] = np.nan
    return scores


def place_track(track, circular, width=WIDTH):
    """Оценка места: наибольшая оценка основания в окне `width` с центром в основании ([i − (width−1)//2, … + width),
    центр по CONTRACTS § 1); NaN — в окне нет ни одной оценки."""
    n = track.size
    left = (width - 1) // 2
    right = width - 1 - left
    t = np.where(np.isnan(track), -np.inf, track).astype(np.float32)
    if circular and n >= width:
        t = np.concatenate([t[n - left:], t, t[:right]])
    else:
        t = np.concatenate([np.full(left, -np.inf, dtype=np.float32), t, np.full(right, -np.inf, dtype=np.float32)])
    out = np.lib.stride_tricks.sliding_window_view(t, width).max(axis=1)
    return np.where(np.isinf(out), np.nan, out).astype(np.float64)


def standardise(tracks):
    """z по всем местам всех репликонов, где оценка есть: список дорожек -> список z-дорожек"""
    flat = np.concatenate([t[~np.isnan(t)] for t in tracks]) if tracks else np.zeros(0)
    if flat.size == 0:
        return tracks
    m, sd = flat.mean(), flat.std() + 1e-9
    return [(t - m) / sd for t in tracks]


def score_genome_places(members, features, mask, replicons, relative_sites=None, batch=16, width=WIDTH, log=None):
    """replicons — список (имя, последовательность, кольцевой). Ответ — по репликону словарь дорожек мест:
    'ensemble' (z ансамбля), 'motif' (z переноса мотива или None), 'total' (их сумма; без мотива — ансамбль),
    'bases' (средняя по моделям оценка основания до окна — для точного положения внутри места)."""
    per_member = []
    base_mean = [np.zeros(len(s), dtype=np.float64) for _, s, _ in replicons]
    for name, m in members:
        t0 = time.time()
        raw = [model_track(m, features, mask, s, c, batch) for _, s, c in replicons]
        places = standardise([place_track(t, c, width) for t, (_, _, c) in zip(raw, replicons)])
        per_member.append(places)
        # основания: z по основаниям у каждой модели, потом средняя
        for k, z in enumerate(standardise([t.astype(np.float64) for t in raw])):
            base_mean[k] += z / len(members)
        if log:
            log('  модель %s: %.0f с' % (name, time.time() - t0))
    ensemble = standardise([np.mean([pm[k] for pm in per_member], axis=0) for k in range(len(replicons))])
    motif, motif_raw = None, None
    if relative_sites:
        motif_raw = [E.motif_transfer(relative_sites, s) for _, s, _ in replicons]
        motif = standardise([place_track(t, c, width) for t, (_, _, c) in zip(motif_raw, replicons)])
    out = []
    for k in range(len(replicons)):
        total = ensemble[k] if motif is None else ensemble[k] + motif[k]
        out.append({'ensemble': ensemble[k], 'motif': None if motif is None else motif[k], 'total': total, 'bases': base_mean[k],
                    'motif_bases': None if motif_raw is None else motif_raw[k]})
    return out


def refine(track, replicon, i, width=WIDTH):
    """Точный центр места: основание с наибольшей оценкой мотива родственника в окне места (у мотива оценка основания —
    лучшее размещение матрицы с центром в нём), а без мотива — с наибольшей оценкой ансамбля. Оценка места — плато
    шириной в окно вокруг лучшего основания, поэтому центр с наибольшей оценкой сам положения сайта не даёт."""
    _, s, circ = replicon
    n = len(s)
    left = (width - 1) // 2
    span = np.arange(i - left, i - left + width)
    span = span % n if circ else span[(span >= 0) & (span < n)]
    for key in ('motif_bases', 'bases'):
        t = track.get(key)
        if t is None:
            continue
        vals = t[span]
        if not np.all(np.isnan(vals)):
            return int(span[int(np.nanargmax(vals))]), key
    return int(i), 'place'


def top_places(tracks, replicons, top, gap=WIDTH):
    """Лучшие места: по убыванию итоговой оценки; ответ — список (оценка, номер репликона, основание с этой оценкой,
    уточнённый центр места, чем уточнён). Оценка места — плато шириной в окно, поэтому кандидат сперва приводится к
    точному центру (`refine`), и запрет ставится вокруг него: центры двух мест репликона не ближе `gap` + 1 оснований,
    окна мест не перекрываются (ревизия 10.11: запрет вокруг кандидата, а не центра, пропускал соседние перекрытые места)."""
    cand = []
    for k, tr in enumerate(tracks):
        total = tr['total']
        ok = ~np.isnan(total)
        n = total.size
        take = min(n, max(20000, top * 400))
        idx = np.argpartition(-np.where(ok, total, -np.inf), take - 1)[:take]
        idx = idx[ok[idx]]
        cand.extend((float(total[i]), k, int(i)) for i in idx)
    cand.sort(reverse=True)
    blocked = [np.zeros(len(s), dtype=bool) for _, s, _ in replicons]

    def block(k, i):
        n = blocked[k].size
        if replicons[k][2]:
            blocked[k][np.arange(i - gap, i + gap + 1) % n] = True
        else:
            blocked[k][max(0, i - gap):i + gap + 1] = True

    rows = []
    for score, k, i in cand:
        if blocked[k][i]:
            continue
        c, by = refine(tracks[k], replicons[k], i)
        if blocked[k][c]:               # это плато уже выданного места: закрыть и идти дальше
            block(k, i)
            continue
        block(k, c)
        block(k, i)
        rows.append((score, k, i, c, by))
        if len(rows) >= top:
            break
    return rows
