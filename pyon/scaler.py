# -*- coding: utf-8 -*-
"""
scaler.py — детерминированный измеритель характеристик по предсказанной разметке
(Э1 §I.8 уровень L4 «модель не измеритель»): из маски классов (+ профиля fp(h), + гирочастоты)
восстанавливаем таблицу «как у дигизонда» (Ion2PNG / SAO scaled): foF2, foF1, foE, foEs,
fmin, fxI, h′F, h′F2, h′E, h′Es, hmF2, hmF1, hmE, MUF(D) по дальностям, M(3000).

Соглашения (те же, что у прототипа и `canon`):
  критическая/предельная частота слоя — последняя колонка с ≥ min_pixels пикселями класса;
  h′ слоя — минимум ЦЕНТРАЛЬНОЙ линии следа (по колонкам с ≥2 пикселями класса — медиана высот;
  `h_lower`): нижняя кромка маски (5-й перцентиль строк, `canon.hmin_readout`) на растре ±1 строка
  давала на самих масках ARTIST сдвиг −5 (h′E) … −8 км (h′Es); центральная линия — 0 … −2.6 км
  (калибровка 2026-09-05, 1500 val-масок; Э3 §4 санити измерителя);
  h′F2 — начало следа F2 выше foF1 (при наличии F1), h′F — min(h′F1, h′F2); fmin — первая колонка
  с любым следом; fxI — по магнитоионному соотношению fx² − fx·fB = fo² от foF2 (X-след не
  сегментируется; Э2 §2.6);
  MUF(D) — закон секанса на сферической Земле по полилинии F2 (`oblique_synth.muf`) × k = K_MUF =
  1.11 (эмпирический множитель Э2 §6.0: на масках ARTIST при k = 1 недобор −1.8 МГц к ARTIST
  MUF(3000), при 1.11 — −0.1 МГц, медиана |Δ| 0.08). Синтетический НЗ-мир остаётся при k = 1;
  множитель — только для таблицы «как у дигизонда»;
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
K_MUF = 1.11          # эмпирический множитель кривизны к ARTIST MUF(3000) (Э2 §6.0)


def trace_from_mask(pm: np.ndarray, ci: int, f_axis=canon.f_axis, h_axis=canon.h_axis, min_pixels: int = 2):
    """Полилиния (центральная линия) следа класса ci: по каждой колонке с ≥ min_pixels пикселями
    класса — медиана их высот (одиночные пиксели предсказания не образуют след)."""
    cols = np.flatnonzero((pm == ci).sum(0) >= min_pixels)
    if not len(cols):
        return np.array([]), np.array([])
    hs = np.array([np.median(h_axis[np.flatnonzero(pm[:, j] == ci)]) for j in cols])
    return f_axis[cols], hs


def h_lower(pm: np.ndarray, ci: int, f_axis=canon.f_axis, h_axis=canon.h_axis) -> float:
    """h′ слоя: минимум центральной линии следа класса ci (NaN, если следа нет)."""
    _, hs = trace_from_mask(pm, ci, f_axis, h_axis)
    return float(hs.min()) if len(hs) else float("nan")


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


def h_reach(fp: np.ndarray, f: float, h_axis=canon.h_axis) -> float:
    """Первая высота, где профиль fp(h) достигает частоты f (линейная интерполяция между узлами);
    NaN, если профиль не дотягивает до f."""
    ok = np.flatnonzero(np.isfinite(fp))
    if len(ok) < 2:
        return float("nan")
    v, h = fp[ok], h_axis[ok]
    above = np.flatnonzero(v >= f)
    if not len(above):
        return float("nan")
    j = above[0]
    if j == 0 or v[j] == v[j - 1]:
        return float(h[j])
    return float(h[j - 1] + (h[j] - h[j - 1]) * (f - v[j - 1]) / (v[j] - v[j - 1]))


def scale_vertical(pm: np.ndarray, prof: np.ndarray | None = None, f_b: float = 1.3,
                   distances=MUF_DISTANCES, k_muf: float = K_MUF, hm_f2: float = float("nan")) -> dict:
    """Маска [NH, NF] классов canon.CLASSES (+ профиль fp(h) [NH], NaN вне валидности; + hmF2 прямой
    регрессии головы, если есть — тогда hmF2 берётся из него, а не из пика профиля) → таблица."""
    C = canon.CLASSES.index
    r = {}
    for cls in ("F2", "F1", "E", "Es"):
        r[f"fo{cls}" if cls != "Es" else "foEs"] = canon.fmax_readout(pm, C(cls))
    # h′F2 по ARTIST — минимальная действующая высота следа F2 ВЫШЕ foF1 (начало следа F2 после
    # каспа F1); без F1 — нижняя кромка всей маски F2. Иначе при наличии F1 наша hF2 уходила на
    # 150+ км ниже ARTIST (панели prefull 2026-09-05).
    r["hF1"] = h_lower(pm, C("F1"))
    if np.isfinite(r["foF1"]):
        pm_f2 = pm.copy(); pm_f2[:, canon.f_axis <= r["foF1"]] = 0
        r["hF2"] = h_lower(pm_f2, C("F2"))
        if not np.isfinite(r["hF2"]):
            r["hF2"] = h_lower(pm, C("F2"))
    else:
        r["hF2"] = h_lower(pm, C("F2"))
    r["hE"] = h_lower(pm, C("E"))
    r["hEs"] = h_lower(pm, C("Es"))
    r["hF"] = float(np.nanmin([r["hF2"], r["hF1"]])) if np.isfinite([r["hF2"], r["hF1"]]).any() else float("nan")
    any_cols = np.flatnonzero((pm > 0).sum(0) >= 2)
    r["fmin"] = float(canon.f_axis[any_cols[0]]) if len(any_cols) else float("nan")
    r["fxI"] = fx_from_fo(r["foF2"], f_b)
    fq, vh = trace_from_mask(pm, C("F2"))
    if len(fq) >= 2:
        for D in distances:
            r[f"MUF{D}"] = k_muf * obs.muf(fq, vh, float(D), 1, "spherical")
        r["M3000F2"] = r["MUF3000"] / r["foF2"] if np.isfinite(r["foF2"]) and r["foF2"] > 0 else float("nan")
    else:
        for D in distances:
            r[f"MUF{D}"] = float("nan")
        r["M3000F2"] = float("nan")
    if prof is not None:
        fp = np.asarray(prof, float)
        pk = profile_peaks(fp)
        r["hmF2"] = float(hm_f2) if np.isfinite(hm_f2) else pk["hmF2"]; r["NmF2_fp"] = pk["fpmax"]
        # hmF1/hmE — высота, где fp(h) впервые достигает foF1/foE (F1 — уступ, а не пик профиля: поиск
        # локальных максимумов давал hmF1 ≈ hmE ≈ 100 км против 165–180 у ARTIST; панели E1 2026-09-06)
        r["hmF1"] = h_reach(fp, r["foF1"]) if np.isfinite(r["foF1"]) else float("nan")
        r["hmE"] = h_reach(fp, r["foE"]) if np.isfinite(r["foE"]) else float("nan")
    else:
        r["hmF2"] = r["hmF1"] = r["hmE"] = r["NmF2_fp"] = float("nan")
    return r


# соответствие «наше имя → колонка манифеста / ключ SAO scaled» для сравнения с ARTIST
ARTIST_KEYS = {"foF2": "foF2", "foF1": "foF1", "foE": "foE", "foEs": "foEs", "fmin": "fmin", "fxI": "fxI",
               "hF": "hF", "hF2": "hF2", "hE": "hE", "hEs": "hEs", "hmF2": "hmF2", "hmF1": "hmF1", "hmE": "zmE",
               "MUF3000": "MUF3000F2", "M3000F2": "M3000F2"}
REPORT_ROWS = ["foF2", "foF1", "foE", "foEs", "fmin", "fxI", "hF", "hF2", "hE", "hEs", "hmF2", "hmF1", "hmE",
               "MUF3000", "M3000F2"]
