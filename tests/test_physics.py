# -*- coding: utf-8 -*-
"""Физика синтеза НЗ (Э2 §2, §6.0): инварианты S1/S2, сферическая vs плоская геометрия,
муф3000-инвариант против ARTIST, O/X-соотношение, трассировка Бугера, синтез масок."""
import glob

import numpy as np
import pytest

from pyon import digi_formats as dfm
from pyon import ion_report as ir
from pyon import oblique_synth as obs

from conftest import ROOT, RSF_DIR, SBF_DIR


def _parabolic_trace(fo=8.0, h0=220.0, n=60):
    """Синтетический вертикальный след: h′(f) растёт к асимптоте foF2 (как реальный F2)."""
    f = np.linspace(2.0, fo - 0.05, n)
    return f, h0 + 40.0 / np.sqrt(1 - (f / fo) ** 2)


def test_s1_s2_invariants_by_construction():
    f, h = _parabolic_trace()
    for d in obs.D_SET:
        f1, p1 = obs.oblique_transform(f, h, d, hops=1)
        f2, p2 = obs.oblique_transform(f, h, d, hops=2)
        assert f2.max() < f1.max(), "S1: МПЧ(2F2) < МПЧ(1F2)"
        assert p2.min() > p1.min() and p2.mean() > p1.mean(), "S2: P′(2F2) > P′(1F2)"
        assert f1.max() > f.max(), "секанс: МПЧ трассы выше foF2"


def test_nose_and_two_rays():
    f, h = _parabolic_trace()
    fob, _ = obs.oblique_transform(f, h, 1500.0)
    k = int(np.argmax(fob))
    assert 0 < k < len(fob) - 1                     # «нос» внутри следа: нижний и верхний лучи
    assert np.all(np.diff(fob[:k]) > 0) and np.all(np.diff(fob[k:]) < 0)


def test_flat_geometry_overestimates_muf():
    f, h = _parabolic_trace()
    r300 = obs.muf(f, h, 300.0, geometry="flat") / obs.muf(f, h, 300.0)
    r1500 = obs.muf(f, h, 1500.0, geometry="flat") / obs.muf(f, h, 1500.0)
    assert 1.0 <= r300 < 1.03
    assert 1.10 < r1500 < 1.25                      # Э2 §2.4: ~17 % при D = 1500


def test_low_layer_vanishes_beyond_horizon():
    f = np.linspace(1.5, 3.0, 10); h = np.full(10, 110.0)           # E-слой
    assert len(obs.oblique_transform(f, h, 2000.0)[0]) > 0
    assert len(obs.oblique_transform(f, h, 2500.0)[0]) == 0         # tanΔ ≤ 0: 1E исчезает
    assert np.isnan(obs.muf(f, h, 2500.0))


def test_x_trace_exact_relation():
    fo = np.array([2.0, 5.0, 10.0]); fb = 1.3
    fx, hx = obs.x_trace_from_o(fo, np.array([200.0, 250.0, 300.0]), fb)
    assert np.allclose(fo ** 2, fx * (fx - fb), atol=1e-9)         # fo² = fx(fx − fB)
    assert np.all(fx > fo) and np.allclose(hx, [200, 250, 300])
    assert abs((fx[-1] - fo[-1]) - fb / 2) < 0.05                  # при fo ≫ fB сдвиг ≈ fB/2


def test_muf3000_invariant_against_artist():
    """Э3 E0-санити: наш МПЧ(3000)/ARTIST ≈ 0.90 (k_эмп = 1/0.898 ≈ 1.11), разброс узкий."""
    saos = sorted(glob.glob(str(RSF_DIR / "scaled/*.SAO"))) + sorted(glob.glob(str(SBF_DIR / "scaled/*.SAO")))
    if not saos:
        pytest.skip("нет SAO-образцов")
    ratios = []
    for p in saos:
        ours, art = obs.muf3000_check(dfm.read_sao(p))
        if np.isfinite(ours) and np.isfinite(art) and art > 0:
            ratios.append(ours / art)
    r = np.array(ratios)
    assert len(r) > 100
    k = 1.0 / np.median(r)
    assert 1.05 <= k <= 1.20, f"k_эмп = {k:.3f} вне [1.05, 1.2]"
    assert np.percentile(r, 75) - np.percentile(r, 25) < 0.02


def test_muf_secant_report_equals_synth(sao_ji):
    fq, vh = sao_ji["F2o_freq"], sao_ji["F2o_vh"]
    tab = ir.muf_secant(fq, vh, (300, 800, 1500, 3000))
    for d, v in tab.items():
        assert abs(v - obs.muf(fq, vh, float(d))) < 1e-9


def test_bouguer_trace_basic(sao_ji):
    ph, pf = sao_ji["profile_h"], sao_ji["profile_fp"]
    fo = sao_ji["scaled"]["foF2"]
    d1, p1 = obs.bouguer_trace(ph, pf, 0.6 * fo, 30.0)
    d2, p2 = obs.bouguer_trace(ph, pf, 0.6 * fo, 45.0)
    assert np.isfinite(d1) and np.isfinite(p1) and 0 < d1 < d2 and p1 < p2   # круче угол — дальше
    assert p1 > d1                                                   # групповой путь длиннее дальности
    assert np.isnan(obs.bouguer_trace(ph, pf, 3.0 * fo, 10.0)[0])    # луч насквозь


def test_oblique_masks_from_sao(sao_ji):
    for d in obs.D_SET:
        y, lab = obs.oblique_masks_from_sao(sao_ji, d, "O")
        assert y.shape == (obs.NP, obs.NF) and y.dtype == np.int8
        assert set(np.unique(y)) <= set(range(len(obs.OB_CLASSES)))
        assert (y == obs.OB_CLASSES.index("F2")).any() and (y == obs.OB_CLASSES.index("MH")).any()
        assert lab["muf_MH"] <= lab["muf_F2"] and lab["D_km"] == d       # S1 по построению
        assert lab["muf_F2"] == pytest.approx(obs.muf(sao_ji["F2o_freq"], sao_ji["F2o_vh"], d))
    yx, labx = obs.oblique_masks_from_sao(sao_ji, 800.0, "X")             # у ARTIST X-полилиний нет → X из O (fo²=fx(fx−fB), E3b)
    assert (yx == obs.OB_CLASSES.index("F2")).any() and (yx == obs.OB_CLASSES.index("MH")).any()
    _, lab800 = obs.oblique_masks_from_sao(sao_ji, 800.0, "O")
    assert 0.2 < labx["muf_F2"] - lab800["muf_F2"] < 1.5 and labx["muf_MH"] <= labx["muf_F2"]   # X выше O примерно на fB/2·sec; S1
    fx, hx = obs.x_trace_from_o(sao_ji["F2o_freq"], sao_ji["F2o_vh"], 0.8, sao_ji["profile_h"], sao_ji["profile_fp"])
    fx0, hx0 = obs.x_trace_from_o(sao_ji["F2o_freq"], sao_ji["F2o_vh"], 0.8)
    assert np.allclose(fx, fx0) and np.nanmax(np.abs(hx - hx0)) < 80 and np.nanmedian(np.abs(hx - hx0)) > 0   # E3b: поправка есть и разумна


def test_raster_polyline_thickness():
    g = obs.raster_polyline([2.0, 24.0], [300.0, 3200.0], obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX, 128, thick=1)
    assert g.any() and g[0, 0] and g[-1, -1]
    assert (g.sum(0) <= 4).all()                                     # ±1 по y → не толще 3–4
    assert not obs.raster_polyline([], [], 0, 1, 0, 1, 8).any()
