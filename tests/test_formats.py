# -*- coding: utf-8 -*-
"""Парсеры digi_formats на образцах Щирого (Э1 часть II §2–§4: числа сверены с pynasonde и PNG)."""
import datetime as dt
import time

import numpy as np
import pytest

from pyon import canon
from pyon import digi_formats as dfm

from conftest import ROOT


# ------------------------------------------------------------------ RSF / SBF (read_ionogram)
def test_rsf_preface_and_table(rsf_path):
    pf, df = dfm.read_ionogram(rsf_path)
    assert pf.format_name == "RSF" and pf.date == dt.datetime(2022, 1, 1, 0, 0, 0)
    assert pf.n_heights == 256 and pf.range_start_km == 80 and pf.range_inc_km == 5.0
    assert (pf.f_start_hz, pf.f_stop_hz, pf.f_coarse_step_hz) == (1.0e6, 12.0e6, 100e3)
    assert pf.both_polarizations
    assert len(df) == 54780                       # 220 групп × 249 бинов (сверено с pynasonde)
    assert df.freq_mhz.nunique() == 110 and df.freq_mhz.min() == 1.0 and df.freq_mhz.max() == 11.9
    assert df.height_km.min() == 80.0 and df.height_km.max() == 1320.0
    assert set(df.pol.unique()) == {"O", "X"}
    assert {"phase_deg", "azimuth_deg", "doppler", "mpa_db"} <= set(df.columns)
    assert df.amp_db.between(0, 93).all() and df.azimuth_deg.isin(range(0, 361, 60)).all()


def test_sbf_preface_and_table(sbf_path):
    pf, df = dfm.read_ionogram(sbf_path)
    assert pf.format_name == "SBF" and pf.date == dt.datetime(2022, 1, 1, 0, 0, 0)
    assert pf.n_heights == 256 and pf.range_inc_km == 2.5
    assert len(df) == 138240                      # 540 групп × 256 бинов
    assert df.freq_mhz.nunique() == 270 and df.freq_mhz.min() == 1.5 and df.freq_mhz.max() == 14.95
    assert df.height_km.max() == 717.5
    assert "phase_deg" not in df.columns          # SBF: 1 байт/бин, направлений нет


# ------------------------------------------------------------------ read_canon (быстрый декодер)
def test_read_canon_regression(rsf_path, sbf_path):
    """Бит-в-бит совпадение с прототипной реализацией (эталон tests/data/canon_ref.npz)."""
    ref = np.load(ROOT / "tests/data/canon_ref.npz")
    for key, path in (("rsf", rsf_path), ("sbf", sbf_path)):
        x = dfm.read_canon(path)
        assert x.shape == (2, canon.NH, canon.NF) and x.dtype == np.uint8
        assert np.array_equal(x, ref[key]), f"{key}: {(x != ref[key]).sum()} расхождений"


def test_read_canon_signal_matches_pandas_path(rsf_path):
    """Сигнал канонической матрицы согласован с медленным путём (порог медиана+6 дБ)."""
    pf, df = dfm.read_ionogram(rsf_path)
    x = dfm.read_canon(rsf_path)
    o = df[df.pol == "O"]
    thr = np.median(o.amp_db[o.amp_db > 0]) + 6
    strong = o[(o.amp_db > thr) & (o.height_km <= canon.H_MAX)]
    jf = canon.to_grid(strong.freq_mhz, canon.F_MIN, canon.F_MAX, canon.NF)
    jh = canon.to_grid(strong.height_km, canon.H_MIN, canon.H_MAX, canon.NH)
    assert (x[0, jh, jf] > 0).mean() > 0.99      # каждый сильный бин виден в матрице
    assert 0.3 < (x[0] > 0).sum() / len(strong) <= 1.05   # и почти нет лишних


def test_read_canon_is_fast(rsf_path, sbf_path):
    for path, limit_ms in ((rsf_path, 2.0), (sbf_path, 3.0)):
        dfm.read_canon(path)
        t0 = time.perf_counter()
        for _ in range(20):
            dfm.read_canon(path)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        assert ms < limit_ms, f"{path.name}: {ms:.2f} мс > {limit_ms} мс (лоадер — узкое место, Э3 §2.2)"


# ------------------------------------------------------------------ SAO
def test_sao_jicamarca(sao_ji):
    sc = sao_ji["scaled"]
    assert sc["foF2"] == 8.2 and sc["fxI"] == 8.6 and sc["hF"] == 223.0        # PNG Ion2PNG
    assert abs(sc["hmF2"] - 369.9) < 1e-6 and abs(sc["M3000F2"] - 2.676) < 1e-6
    assert np.isnan(sc["foE"]) and np.isnan(sc["foF1"])                       # ночь
    assert sao_ji["datetime"] == dt.datetime(2022, 1, 1, 0, 0, 0)
    assert len(sao_ji["F2o_freq"]) == 58 == len(sao_ji["F2o_vh"])
    assert len(sao_ji["profile_h"]) == 95
    assert int(sao_ji["analysis_flags"][9]) == 11                              # C-level
    assert "ARTIST" in sao_ji["system_desc"]


def test_sao_rome(sao_ro):
    sc = sao_ro["scaled"]
    assert sc["foF2"] == 3.75 and sc["fxI"] == 4.3 and sc["fmin"] == 2.2
    assert len(sao_ro["profile_h"]) == 162 and len(sao_ro["F2o_freq"]) == 32
    assert "qual_letters" in sao_ro and "desc_letters" in sao_ro                # ARTIST-5 пишет группы 54–55
    assert "Rome" in sao_ro["system_desc"]


def test_sao_scaled_missing_is_nan(sao_ji):
    assert not (sao_ji["scaled"] >= 9999).any()


# ------------------------------------------------------------------ EDP / DFT
def test_edp(edp_path):
    head, prof = dfm.read_edp(edp_path)
    assert abs(head["fof2"] - 8.66) < 1e-6 and abs(head["hmf2"] - 414.06) < 1e-6
    assert len(prof) == 308 and prof.height_km.is_monotonic_increasing
    assert not (prof == -99).any().any()                                        # -99 → NaN


def test_dft_header(dft_path):
    hdr, amp, phase = dfm.read_dft(dft_path)
    assert hdr["year"] == 2022 and hdr["doy"] == 1 and hdr["start_freq_khz"] == 3800.0
    assert amp.shape[1:] == (16, 128) and phase.shape == amp.shape


# ------------------------------------------------------------------ canon: маски и ридауты
def test_masks_from_sao_matches_trace(sao_ji):
    y = canon.masks_from_sao(sao_ji)
    assert y.shape == (canon.NH, canon.NF) and y.dtype == np.int8
    assert set(np.unique(y)) == {0, canon.CLASSES.index("F2")}                # ночь: только F2
    # эталон прототипа I.8.1 — цикл по точкам, ±1 бин по высоте
    ref = np.zeros((canon.NH, canon.NF), np.int8)
    for f0, h0 in zip(sao_ji["F2o_freq"], sao_ji["F2o_vh"]):
        if canon.F_MIN <= f0 <= canon.F_MAX and canon.H_MIN <= h0 <= canon.H_MAX:
            jf = int(round((f0 - canon.F_MIN) / (canon.F_MAX - canon.F_MIN) * (canon.NF - 1)))
            jh = int(round((h0 - canon.H_MIN) / (canon.H_MAX - canon.H_MIN) * (canon.NH - 1)))
            ref[max(jh - 1, 0):jh + 2, jf] = 1
    assert np.array_equal(y, ref)


def test_readouts_reproduce_artist(sao_ji, sao_ro):
    for sao in (sao_ji, sao_ro):
        y = canon.masks_from_sao(sao)
        fo = canon.fmax_readout(y, canon.CLASSES.index("F2"))
        hm = canon.hmin_readout(y, canon.CLASSES.index("F2"))
        assert abs(fo - sao["scaled"]["foF2"]) <= 0.12                         # шаг решётки 0.11 МГц
        assert abs(hm - sao["scaled"]["hF"]) <= 10.0                            # ±1 бин × 5 км
    assert np.isnan(canon.fmax_readout(np.zeros((8, 8), np.int8), 1))
