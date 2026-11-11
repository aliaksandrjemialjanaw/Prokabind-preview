"""Тесты ядра предсказания `dev/predictor.py` (разработка, решение владельца 20). Данных проекта не читают.
Запуск: ./.venv/bin/python -I -B -m pytest -q -p no:cacheprovider dev/tests

`test_top_places_*` — свойство правки 10.11 по первой ревизии ночи (находка Н10): центры двух мест одного репликона не
ближе WIDTH + 1 оснований, окна мест не перекрываются. На прежней версии (`5872846`) тест падает: запрет стоял вокруг
кандидата на краю плато, а не вокруг уточнённого центра (вторая ревизия, `r08_top_places.py`: 3 906 близких пар на 160
дорожках)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import predictor as P


def _tracks(n, rng, circular, with_motif):
    """Синтетические дорожки одного репликона: редкие сильные основания на шумном фоне; дорожки мест — как у предсказателя."""
    def bases():
        b = rng.normal(0.0, 1.0, n)
        hot = rng.choice(n, size=max(4, n // 150), replace=False)
        b[hot] += rng.uniform(4.0, 9.0, hot.size)
        return b
    ens = bases()
    ens_place = P.standardise([P.place_track(ens.astype(np.float32), circular)])[0]
    track = {'ensemble': ens_place, 'bases': ens, 'motif': None, 'motif_bases': None, 'total': ens_place}
    if with_motif:
        mot = bases()
        mot_place = P.standardise([P.place_track(mot.astype(np.float32), circular)])[0]
        track.update(motif=mot_place, motif_bases=mot, total=ens_place + mot_place)
    return track


def _min_gap(centres, n, circular):
    c = np.sort(np.asarray(centres))
    d = np.diff(c)
    if circular and c.size > 1:
        d = np.append(d, c[0] + n - c[-1])
    return int(d.min()) if d.size else n


def test_place_track_is_window_maximum():
    t = np.array([np.nan, 1, 5, 2, np.nan, np.nan, np.nan, 0, 3], dtype=np.float32)
    assert np.array_equal(P.place_track(t, False, 3), np.array([1, 5, 5, 5, 2, np.nan, 0, 3, 3]), equal_nan=True)
    assert np.array_equal(P.place_track(t, True, 3), np.array([3, 5, 5, 5, 2, np.nan, 0, 3, 3]), equal_nan=True)


def test_top_places_centres_do_not_overlap():
    rng = np.random.default_rng(20261110)
    for circular in (False, True):
        for with_motif in (False, True):
            for _ in range(10):
                n = int(rng.integers(3000, 9000))
                replicon = ('r', 'A' * n, circular)
                track = _tracks(n, rng, circular, with_motif)
                rows = P.top_places([track], [replicon], 60)
                assert len(rows) == 60
                centres = [c for _, _, _, c, _ in rows]
                assert _min_gap(centres, n, circular) >= P.WIDTH + 1
                scores = [s for s, *_ in rows]
                assert scores == sorted(scores, reverse=True)


def test_top_places_centre_is_refined_to_best_base():
    n = 600
    bases = np.zeros(n)
    bases[100], bases[118], bases[300] = 5.0, 4.9, 4.0          # второе основание — внутри окна первого места
    place = P.place_track(bases.astype(np.float32), False)
    track = {'ensemble': place, 'bases': bases, 'motif': None, 'motif_bases': None, 'total': place}
    rows = P.top_places([track], [('r', 'A' * n, False)], 2)
    assert [c for _, _, _, c, _ in rows] == [100, 300]
    assert [by for *_, by in rows] == ['bases', 'bases']
