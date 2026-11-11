"""Рисунки программы в SVG (SPEC пп. 5.4, 5.8). Пишет задача прототипа T16.

Библиотеки рисования в окружении нет (T16), поэтому SVG собирается здесь текстом. Без PyTorch. Рисунок —
**детерминированный текст**: ни времени, ни путей, числа печатаются с постоянным числом знаков, — иначе два
одинаковых запуска разошлись бы байтами (п. 5.7).

Координаты на рисунках — от 1, как в таблицах (п. 5.2).
"""
import numpy as np

FONT = 'DejaVu Sans, Arial, Helvetica, sans-serif'
MONO = 'DejaVu Sans Mono, Courier New, monospace'
#: Цвета оснований — привычные цвета логотипов мотивов.
BASE_COLOURS = {'A': '#1a9641', 'C': '#2b6cb0', 'G': '#e69b00', 'T': '#c62828'}
BASES = 'ACGT'
TRACK_COLOUR = '#2f3e46'
SITE_COLOUR = '#f28e2b'
GENE_COLOUR = '#9aa5ad'
LEVEL_COLOUR = '#8e44ad'


def _esc(text):
    return (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;'))


def _f(x):
    """Число на рисунке: два знака после точки, без «-0.00»."""
    v = round(float(x), 2)
    return '%.2f' % (0.0 if v == 0 else v)


class Svg:
    def __init__(self, width, height):
        self.width, self.height = int(width), int(height)
        self.parts = []

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, s, size=11, anchor='start', colour='#222', weight='normal', family=FONT, rotate=None):
        tr = ' transform="rotate(%s %s %s)"' % (_f(rotate), _f(x), _f(y)) if rotate is not None else ''
        self.add('<text x="%s" y="%s" font-family="%s" font-size="%s" text-anchor="%s" fill="%s" font-weight="%s"%s>%s'
                 '</text>' % (_f(x), _f(y), family, _f(size), anchor, colour, weight, tr, _esc(s)))

    def line(self, x1, y1, x2, y2, colour='#222', width=1.0, dash=None):
        d = ' stroke-dasharray="%s"' % dash if dash else ''
        self.add('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s/>'
                 % (_f(x1), _f(y1), _f(x2), _f(y2), colour, _f(width), d))

    def rect(self, x, y, w, h, fill, stroke='none', opacity=None):
        o = ' fill-opacity="%s"' % _f(opacity) if opacity is not None else ''
        self.add('<rect x="%s" y="%s" width="%s" height="%s" fill="%s" stroke="%s"%s/>'
                 % (_f(x), _f(y), _f(max(w, 0)), _f(max(h, 0)), fill, stroke, o))

    def polyline(self, points, colour=TRACK_COLOUR, width=1.0):
        if len(points) < 2:
            return
        self.add('<polyline fill="none" stroke="%s" stroke-width="%s" points="%s"/>'
                 % (colour, _f(width), ' '.join('%s,%s' % (_f(x), _f(y)) for x, y in points)))

    def letter(self, x, baseline, width, height, base, flip=False):
        """Буква основания высотой `height` пикселей, стоящая на `baseline` (при `flip` — висящая под ней).
        Высота заглавной буквы шрифта — около 0,72 кегля; буква растягивается по вертикали до нужной высоты."""
        if height <= 0.05:
            return
        size = 10.0
        sx = width / (0.62 * size)
        sy = height / (0.72 * size)
        if flip:
            y = baseline + height
        else:
            y = baseline
        self.add('<text transform="translate(%s %s) scale(%s %s)" font-family="%s" font-size="10" font-weight="bold" '
                 'fill="%s" text-anchor="middle">%s</text>'
                 % (_f(x + width / 2), _f(y), '%.4f' % sx, '%.4f' % sy, MONO, BASE_COLOURS[base], base))

    def render(self):
        head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n'
                '<rect x="0" y="0" width="%d" height="%d" fill="#ffffff"/>\n'
                % (self.width, self.height, self.width, self.height, self.width, self.height))
        return head + '\n'.join(self.parts) + '\n</svg>\n'

    def save(self, path):
        with open(path, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(self.render())


def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ''
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def _notes(svg, x, y, notes, width_chars=150, size=10.5):
    for note in notes:
        for line in _wrap(note, width_chars):
            svg.text(x, y, line, size=size, colour='#444')
            y += size * 1.45
    return y


def logo_block(svg, x0, y0, matrix, width, height, scale=None, offsets=None, label_every=10):
    """Логотип матрицы «позиция × основание» (за вычетом среднего): положительные значения — буквы над осью,
    отрицательные — под ней. Возвращает использованный масштаб (значение на пиксель)."""
    m = np.asarray(matrix, dtype=np.float64)
    k = m.shape[0]
    col = width / float(k)
    half = height / 2.0
    axis = y0 + half
    if scale is None:
        pos = np.nansum(np.where(m > 0, m, 0), axis=1)
        neg = np.nansum(np.where(m < 0, -m, 0), axis=1)
        top = max(float(np.nanmax(pos)) if pos.size else 0.0, float(np.nanmax(neg)) if neg.size else 0.0)
        scale = top / half if top > 0 else 1.0
    svg.line(x0, axis, x0 + width, axis, colour='#999', width=0.6)
    for i in range(k):
        row = m[i]
        if not np.isfinite(row).all():
            continue
        x = x0 + i * col
        up = axis
        for b in sorted(range(4), key=lambda j: row[j]):
            if row[b] > 0:
                h = row[b] / scale
                svg.letter(x, up, col, h, BASES[b])
                up -= h
        down = axis
        for b in sorted(range(4), key=lambda j: -row[j]):
            if row[b] < 0:
                h = -row[b] / scale
                svg.letter(x, down, col, h, BASES[b], flip=True)
                down += h
    if offsets is not None:
        for i, off in enumerate(offsets):
            if off % label_every == 0:
                svg.line(x0 + (i + 0.5) * col, y0 + height, x0 + (i + 0.5) * col, y0 + height + 3, colour='#666', width=0.6)
                svg.text(x0 + (i + 0.5) * col, y0 + height + 13, '%+d' % off if off else '0', size=9, anchor='middle',
                         colour='#555')
    return scale


def logo_figure(path, matrix, offsets, title, notes):
    """Логотип по верхним местам (п. 5.4): средняя карта вклада, ориентированная или нет."""
    k = len(offsets)
    width = max(900, 7 * k + 120)
    svg = Svg(width, 360 + 16 * len(notes) * 2)
    svg.text(20, 26, title, size=15, weight='bold')
    scale = logo_block(svg, 60, 50, matrix, width - 90, 220, offsets=offsets)
    svg.text(20, 50 + 110, 'вклад', size=10, colour='#555', rotate=-90)
    svg.text(60, 300, 'положение относительно точки привязки места, оснований; над осью — основания, повышающие '
             'оценку места, под осью — понижающие (одна высота буквы ≈ %s единиц оценки на 10 пикселей)'
             % ('%.3g' % (scale * 10)), size=10, colour='#444')
    _notes(svg, 20, 326, notes)
    svg.save(path)


def heat_block(svg, x0, y0, matrix, width, row_h, vmax):
    """Карта вклада четырьмя строками A, C, G, T: красное — основание в этой позиции повышает оценку места,
    синее — понижает; насыщенность — до `vmax`."""
    m = np.asarray(matrix, dtype=np.float64)
    k = m.shape[0]
    col = width / float(k)
    for b in range(4):
        svg.text(x0 - 4, y0 + (b + 0.75) * row_h, BASES[b], size=9, anchor='end', colour=BASE_COLOURS[BASES[b]],
                 weight='bold', family=MONO)
        for i in range(k):
            v = m[i, b]
            if not np.isfinite(v):
                fill = '#dddddd'
            else:
                t = max(-1.0, min(1.0, v / vmax)) if vmax > 0 else 0.0
                if t >= 0:
                    r, g, bl = 255, int(round(255 - 200 * t)), int(round(255 - 200 * t))
                else:
                    r, g, bl = int(round(255 + 200 * t)), int(round(255 + 120 * t)), 255
                fill = '#%02x%02x%02x' % (r, g, bl)
            svg.rect(x0 + i * col, y0 + b * row_h, col + 0.02, row_h, fill)


def show_figure(path, title, subtitle, panels, notes, overview=None):
    """Рисунок показа (п. 5.8): дорожка вокруг первых мест с подписанными координатами и картой вклада.

    `panels` — по месту: `{'rank', 'replicon', 'anchor' (с нуля), 'score', 'lo', 'hi' (окно дорожки, с нуля, конец
    не включён), 'track' (оценки окна, NaN — нет), 'places' ([(ранг, точка привязки)] в окне), 'genes'
    ([(start, end, имя, обратная цепь)]), 'sites' ([(start, end)] — известные сайты, только у белка контрольной
    части), 'level', 'map' ([141, 4] или None), 'map_offsets', 'seq' (основания окна карты), 'larger_pass'}`."""
    width = 1240
    left, right = 80, 30
    plot_w = width - left - right
    panel_h = 330
    over_h = 150 if overview is not None else 0
    notes_h = 20 + 16 * sum(len(_wrap(n, 165)) for n in notes)
    svg = Svg(width, 70 + over_h + panel_h * len(panels) + notes_h + 20)
    svg.text(20, 28, title, size=16, weight='bold')
    svg.text(20, 50, subtitle, size=11, colour='#444')
    y = 70
    if overview is not None:
        _overview(svg, left, y, plot_w, overview)
        y += over_h
    for p in panels:
        vmax = float(np.nanmax(np.abs(p['map']))) if p['map'] is not None and np.isfinite(p['map']).any() else 0.0
        _panel(svg, left, y, plot_w, p, vmax)
        y += panel_h
    _notes(svg, 20, y + 10, notes, width_chars=165)
    svg.save(path)


def _overview(svg, x0, y0, w, ov):
    """Обзор репликона: наибольшая оценка в каждой из полос, первые места и (у белка контрольной части) известные
    сайты."""
    h = 80
    svg.text(x0, y0 + 12, 'Репликон %s целиком: наибольшая оценка в полосе по %s оснований; треугольники — первые места; '
             '%s' % (ov['replicon'], '{:,}'.format(ov['bin']).replace(',', ' '),
                     'оранжевые штрихи — известные сайты белка' if ov.get('sites') else 'известные сайты не рисуются'),
             size=10.5, colour='#333')
    top = y0 + 22
    vals = np.asarray(ov['values'], dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    n = int(ov['length'])
    pts = []
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            svg.polyline(pts)
            pts = []
            continue
        pts.append((x0 + (i + 0.5) * w / len(vals), top + h - (v - lo) / (hi - lo) * h))
    svg.polyline(pts, width=0.8)
    svg.line(x0, top + h, x0 + w, top + h, colour='#888', width=0.6)
    for s, e in ov.get('sites', ()):
        xs = x0 + (s + e) / 2.0 / n * w
        svg.line(xs, top + h + 2, xs, top + h + 12, colour=SITE_COLOUR, width=1.2)
    for rank, anchor in ov['places']:
        xa = x0 + anchor / float(n) * w
        svg.add('<path d="M %s %s L %s %s L %s %s Z" fill="#c0392b"/>'
                % (_f(xa), _f(top - 1), _f(xa - 3.5), _f(top - 8), _f(xa + 3.5), _f(top - 8)))
    for k in range(0, 6):
        pos = int(round(k * (n - 1) / 5.0))
        xx = x0 + pos / float(n) * w
        svg.line(xx, top + h, xx, top + h + 4, colour='#666', width=0.6)
        svg.text(xx, top + h + 24, '{:,}'.format(pos + 1).replace(',', ' '), size=9.5, anchor='middle', colour='#555')


def _panel(svg, x0, y0, w, p, vmax):
    lo, hi = int(p['lo']), int(p['hi'])
    span = hi - lo
    track = np.asarray(p['track'], dtype=np.float64)
    title = 'Место %d · %s : %s · оценка %s · склеено максимумов %d · больший проход: %s' % (
        p['rank'], p['replicon'], '{:,}'.format(p['anchor'] + 1).replace(',', ' '), '%.4g' % p['score'],
        p.get('n_merged', 0), p.get('larger_pass', ''))
    svg.text(x0 - 60, y0 + 14, title, size=12, weight='bold')
    th = 110
    top = y0 + 26

    def xpos(i):                       # основание с нуля → x (центр основания)
        return x0 + (i - lo + 0.5) * w / float(span)

    finite = track[np.isfinite(track)]
    level = p.get('level')
    vals = list(finite) + ([level] if level is not None and np.isfinite(level) else [])
    vlo, vhi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    if vhi <= vlo:
        vhi = vlo + 1.0
    pad = (vhi - vlo) * 0.08
    vlo, vhi = vlo - pad, vhi + pad

    def ypos(v):
        return top + th - (v - vlo) / (vhi - vlo) * th

    svg.rect(x0, top, w, th, '#fafafa', stroke='#cccccc')
    # окно карты вклада — поле видимости точки привязки
    if p.get('map_offsets') is not None:
        a0 = p['anchor'] + p['map_offsets'][0]
        a1 = p['anchor'] + p['map_offsets'][-1] + 1
        svg.rect(xpos(a0) - 0.5 * w / span, top, (a1 - a0) * w / span, th, '#fff3cd', opacity=0.8)
    for s, e in p.get('sites', ()):
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > s2:
            svg.rect(xpos(s2) - 0.5 * w / span, top + th - 10, (e2 - s2) * w / span, 10, SITE_COLOUR, opacity=0.85)
    if level is not None and np.isfinite(level) and vlo <= level <= vhi:
        svg.line(x0, ypos(level), x0 + w, ypos(level), colour=LEVEL_COLOUR, width=0.8, dash='4 3')
        svg.text(x0 + w - 2, ypos(level) - 3, 'справочный уровень (99,9-й перцентиль оснований)', size=8.5,
                 anchor='end', colour=LEVEL_COLOUR)
    pts = []
    for k, v in enumerate(track):
        if not np.isfinite(v):
            svg.polyline(pts)
            pts = []
            continue
        pts.append((xpos(lo + k), ypos(v)))
    svg.polyline(pts, width=1.0)
    for rank, anchor in p.get('places', ()):
        if lo <= anchor < hi:
            svg.line(xpos(anchor), top, xpos(anchor), top + th, colour='#c0392b' if rank == p['rank'] else '#e6a1a1',
                     width=1.0, dash=None if rank == p['rank'] else '2 2')
            svg.text(xpos(anchor) + 2, top + 10, '#%d' % rank, size=8.5, colour='#c0392b')
    # ось оценок
    for v in (vlo + pad, (vlo + vhi) / 2.0, vhi - pad):
        svg.line(x0 - 4, ypos(v), x0, ypos(v), colour='#666', width=0.6)
        svg.text(x0 - 6, ypos(v) + 3, '%.3g' % v, size=8.5, anchor='end', colour='#555')
    svg.text(x0 - 48, top + th / 2.0 + 20, 'оценка', size=9.5, colour='#555', rotate=-90)
    # ось координат: от 1 (п. 5.2)
    step = 50 if span <= 600 else 100
    first = ((lo + 1 + step - 1) // step) * step
    for pos in range(first, hi + 1, step):
        xx = xpos(pos - 1)
        svg.line(xx, top + th, xx, top + th + 4, colour='#666', width=0.6)
        svg.text(xx, top + th + 14, '{:,}'.format(pos).replace(',', ' '), size=8.5, anchor='middle', colour='#555')
    # гены
    gy = top + th + 22
    for s, e, name, rev in p.get('genes', ()):
        s2, e2 = max(s, lo), min(e, hi)
        if e2 <= s2:
            continue
        xa, xb = xpos(s2) - 0.5 * w / span, xpos(e2 - 1) + 0.5 * w / span
        svg.rect(xa, gy, xb - xa, 9, GENE_COLOUR)
        tip = xa if rev else xb
        if (rev and s >= lo) or (not rev and e <= hi):
            d = -6 if rev else 6
            svg.add('<path d="M %s %s L %s %s L %s %s Z" fill="%s"/>'
                    % (_f(tip), _f(gy - 2), _f(tip + d), _f(gy + 4.5), _f(tip), _f(gy + 11), GENE_COLOUR))
        if name and xb - xa > 24:
            svg.text((xa + xb) / 2.0, gy + 20, name, size=8.5, anchor='middle', colour='#333', family=MONO)
    # карта вклада места
    if p['map'] is not None:
        offs = p['map_offsets']
        my = gy + 34
        svg.text(x0 - 60, my + 8, 'Карта вклада места %d — вклад в оценку места, ±70 оснований от точки привязки; '
                 'буквы — основание генома с его вкладом, клетки — все четыре основания (насыщенность до ±%s)'
                 % (p['rank'], '%.3g' % vmax), size=10, colour='#333')
        seq = p.get('seq') or ''
        mat = np.asarray(p['map'], dtype=np.float64)
        own = np.full(mat.shape[0], np.nan)
        for i, ch in enumerate(seq):
            if ch in BASES:
                own[i] = mat[i, BASES.index(ch)]
        own_m = np.zeros_like(mat)
        own_m[:] = np.nan
        for i, ch in enumerate(seq):
            if ch in BASES:
                own_m[i] = 0.0
                own_m[i, BASES.index(ch)] = own[i]
        svg.text(x0 - 6, my + 34, 'основание генома', size=8, anchor='end', colour='#555')
        logo_block(svg, x0, my + 12, own_m, w, 50)
        heat_block(svg, x0, my + 66, mat, w, 9, vmax)
        for i, off in enumerate(offs):
            if off % 10 == 0:
                xx = x0 + (i + 0.5) * w / len(offs)
                svg.text(xx, my + 66 + 36 + 10, '%+d' % off if off else '0', size=8, anchor='middle', colour='#555')


def residue_figure(path, weights, sequence, title, notes):
    """Веса сбора по остаткам белка (п. 5.4): четыре профиля по длине белка, по одному на обучаемый запрос —
    отдельная иллюстрация с обязательными оговорками, **не** карта важности оснований."""
    w = np.asarray(weights, dtype=np.float64)
    q, n = w.shape
    width, left = 1240, 70
    plot_w = width - left - 30
    row_h = 70
    notes_h = 20 + 16 * sum(len(_wrap(x, 165)) for x in notes)
    svg = Svg(width, 60 + q * (row_h + 14) + 40 + notes_h)
    svg.text(20, 28, title, size=15, weight='bold')
    colours = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    y = 50
    for k in range(q):
        row = w[k]
        top, bottom = float(row.max()), float(row.min())
        uniform = 1.0 / n
        hi_v, lo_v = max(top, uniform), min(bottom, uniform)
        if hi_v <= lo_v:
            hi_v = lo_v + 1.0

        def yv(v):
            return y + row_h - 3 - (v - lo_v) / (hi_v - lo_v) * (row_h - 6)

        svg.rect(left, y, plot_w, row_h, '#fafafa', stroke='#cccccc')
        svg.line(left, yv(uniform), left + plot_w, yv(uniform), colour='#999', width=0.6, dash='3 3')
        pts = [(left + (i + 0.5) * plot_w / n, yv(row[i])) for i in range(n)]
        svg.polyline(pts, colour=colours[k % len(colours)], width=1.0)
        svg.text(left - 6, y + 14, 'запрос %d' % (k + 1), size=9.5, anchor='end', colour=colours[k % len(colours)])
        svg.text(left - 6, y + 28, 'от %.3g' % bottom, size=8.5, anchor='end', colour='#555')
        svg.text(left - 6, y + 40, 'до %.3g' % top, size=8.5, anchor='end', colour='#555')
        best = int(np.argmax(row)) if row.size else 0
        svg.text(left + (best + 0.5) * plot_w / n, y + 10, '%s%d' % (sequence[best], best + 1), size=8.5,
                 anchor='middle', colour='#333', family=MONO)
        y += row_h + 14
    step = 50 if n <= 600 else 100
    for pos in range(step, n + 1, step):
        xx = left + (pos - 0.5) * plot_w / n
        svg.line(xx, y - 14, xx, y - 10, colour='#666', width=0.6)
        svg.text(xx, y, str(pos), size=8.5, anchor='middle', colour='#555')
    svg.text(left, y + 16, 'номер остатка белка, от 1; ось каждой строки — от наименьшего до наибольшего веса этого запроса; '
             'серый пунктир — равномерный вес 1/%d' % n, size=9.5, colour='#555')
    _notes(svg, 20, y + 40, notes, width_chars=165)
    svg.save(path)
