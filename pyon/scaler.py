# -*- coding: utf-8 -*-
"""
scaler.py — детерминированный измеритель характеристик по предсказанной разметке
(Э1 §I.8 уровень L4 «модель не измеритель»): из маски классов (+ профиля fp(h), + гирочастоты)
восстанавливаем таблицу «как у дигизонда» (Ion2PNG / SAO scaled): foF2, foF1, foE, foEs,
fmin, fxI, h′F, h′F2, h′E, h′Es, hmF2, hmF1, hmE, MUF(D) по дальностям, M(3000).

Соглашения (те же, что у прототипа и `canon`):
  критическая/предельная частота слоя — последняя колонка с ≥ min_pixels пикселями класса;
  h′ слоя — 5-й перцентиль высот класса (нижняя кромка); h′F2 — нижняя кромка F2, h′F — нижняя
  кромка F-области (min по F1/F2); fmin — первая колонка с любым следом;
  fxI — по магнитоионному соотношению fx² − fx·fB = fo² от foF2 (X-след не сегментируется; Э2 §2.6);
  MUF(D) — закон секанса на сферической Земле по полилинии F2 (`oblique_synth.muf`, k = 1 —
  систематически ~10 % ниже ARTIST MUF(3000), считаемой по профилю; Э2 §6.0);
  hmF2/hmF1/hmE — из профиля fp(h): высота максимума fp в пределах предсказанной валидности
  (hmE/hmF1 — локальные максимумы ниже соответствующих критических частот, если есть).
Все выходы — float, отсутствующее = NaN.
"""
from __future__ import annotations

import math

import numpy as np

from pyon import canon
from pyon import oblique_synth as obs

MUF_DISTANCES = (100, 200, 400, 600, 800, 1000, 1500, 3000)


def trace_from_mask(pm: np.ndarray, ci: int, f_axis=canon.f_axis, h_axis=canon.h_axis):
    """Полилиния следа класса ci: по каждой колонке с пикселями класса — медиана их высот."""
    cols = np.flatnonzero((pm == ci).any(0))
    if not len(cols):
        return np.array([]), np.array([])
    hs = np.array([np.median(h_axis[np.flatnonzero(pm[:, j] == ci)]) for j in cols])
    return f_axis[cols], hs


def fx_from_fo(fo: float, f_b: float) -> float:
    """fo² = fx(fx − fB) ⇒ fx = fB/2 + sqrt(fB²/4 + fo²)."""
    return float(f_b / 2 + math.sqrt(f_b * f_b / 4 + fo * fo)) if np.isfinite(fo) else float("nan")


def profile_peaks(fp: np.ndarray, h_axis=canon.h_axis):
    """Максимум профиля (hmF2, NmF2 → fp_max) и локальные максимумы ниже (E/F1) по валидным точкам."""
    ok = np.isfinite(fp)
    if ok.sum() < 3:
        return dict(hmF2=np.nan, fpmax=np.nan, peaks=[])
    idx = np.flatnonzero(ok)
    im = idx[np.nanargmax(fp[idx])]
    peaks = []
    v = fp[idx]
    for k in range(1, len(idx) - 1):        # локальные максимумы ниже главного
        if idx[k] < im and v[k] >= v[k - 1] and v[k] > v[k + 1]:
            peaks.append((float(h_axis[idx[k]]), float(v[k])))
    return dict(hmF2=float(h_axis[im]), fpmax=float(fp[im]), peaks=peaks)


def scale_vertical(pm: np.ndarray, prof: np.ndarray | None = None, f_b: float = 1.3,
                   distances=MUF_DISTANCES) -> dict:
    """Маска [NH, NF] классов canon.CLASSES (+ профиль fp(h) [NH], NaN вне валидности) → таблица."""
    C = canon.CLASSES.index
    r = {}
    for cls in ("F2", "F1", "E", "Es"):
        r[f"fo{cls}" if cls != "Es" else "foEs"] = canon.fmax_readout(pm, C(cls))
    r["hF2"] = canon.hmin_readout(pm, C("F2"))
    r["hF1"] = canon.hmin_readout(pm, C("F1"))
    r["hE"] = canon.hmin_readout(pm, C("E"))
    r["hEs"] = canon.hmin_readout(pm, C("Es"))
    r["hF"] = float(np.nanmin([r["hF2"], r["hF1"]])) if np.isfinite([r["hF2"], r["hF1"]]).any() else float("nan")
    any_cols = np.flatnonzero((pm > 0).sum(0) >= 2)
    r["fmin"] = float(canon.f_axis[any_cols[0]]) if len(any_cols) else float("nan")
    r["fxI"] = fx_from_fo(r["foF2"], f_b)
    fq, vh = trace_from_mask(pm, C("F2"))
    if len(fq) >= 2:
        for D in distances:
            r[f"MUF{D}"] = obs.muf(fq, vh, float(D), 1, "spherical")
        r["M3000F2"] = r["MUF3000"] / r["foF2"] if np.isfinite(r["foF2"]) and r["foF2"] > 0 else float("nan")
    else:
        for D in distances:
            r[f"MUF{D}"] = float("nan")
        r["M3000F2"] = float("nan")
    if prof is not None:
        pk = profile_peaks(np.asarray(prof, float))
        r["hmF2"] = pk["hmF2"]; r["NmF2_fp"] = pk["fpmax"]
        lower = [h for h, v in pk["peaks"]]
        r["hmF1"] = float(lower[-1]) if len(lower) >= 1 and np.isfinite(r["foF1"]) else float("nan")
        r["hmE"] = float(lower[0]) if len(lower) >= 1 and np.isfinite(r["foE"]) else float("nan")
    else:
        r["hmF2"] = r["hmF1"] = r["hmE"] = r["NmF2_fp"] = float("nan")
    return r


# соответствие «наше имя → колонка манифеста / ключ SAO scaled» для сравнения с ARTIST
ARTIST_KEYS = {"foF2": "foF2", "foF1": "foF1", "foE": "foE", "foEs": "foEs", "fmin": "fmin", "fxI": "fxI",
               "hF": "hF", "hF2": "hF2", "hE": "hE", "hEs": "hEs", "hmF2": "hmF2", "hmF1": "hmF1", "hmE": "zmE",
               "MUF3000": "MUF3000F2", "M3000F2": "M3000F2"}
REPORT_ROWS = ["foF2", "foF1", "foE", "foEs", "fmin", "fxI", "hF", "hF2", "hE", "hEs", "hmF2", "hmF1", "hmE",
               "MUF3000", "M3000F2"]
