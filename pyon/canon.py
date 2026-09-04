# -*- coding: utf-8 -*-
"""
canon.py — каноническая решётка ВЗ-ионограммы, растеризация разметки ARTIST и жёсткие
ридауты (Э1 §I.8 «Вход x», «L4 — модель не измеритель»; Э3 §2.2, §3.2).

Единственное место, где заданы оси ВЗ-тензоров: `digi_formats.read_canon`, `loader`,
`dataset_cache`, `logic`, `gates` берут константы отсюда (до ревизии 2026-09-04 они
дублировались в dataset_cache и в ноутбуках).

Решётка: NF × NH = 128 × 128; частота 1–15 МГц (шаг ≈ 0.11 МГц), действующая высота
80–720 км (шаг ≈ 5.04 км). Классы маски: 0 BG, 1 F2, 2 F1, 3 E, 4 Es — O-полилинии SAO
(X-следы — кандидат v0.4 / E3b). Растеризация: точка полилинии → ближайшая ячейка,
утолщение ±thick по высоте (прототип I.8.1: ±1 бин).

Жёсткие ридауты (детерминированный измеритель по маске):
  fmax_readout(pm, ci, axis)  — последняя частотная колонка с ≥ min_pixels пикселями класса:
                               foF2 для F2 на ВЗ (axis=f_axis); МПЧ моды на НЗ (axis=fob_axis)
  hmin_readout(pm, ci, axis)  — 5-й перцентиль строк класса: h′F на ВЗ; P′ носа на НЗ (p_axis)
"""
from __future__ import annotations

import numpy as np

NF = NH = 128
F_MIN, F_MAX = 1.0, 15.0        # МГц
H_MIN, H_MAX = 80.0, 720.0      # км
CLASSES = ["BG", "F2", "F1", "E", "Es"]
TRACE_KEYS = {"F2": "F2o", "F1": "F1o", "E": "Eo", "Es": "Es"}   # класс → префикс групп SAO
f_axis = np.linspace(F_MIN, F_MAX, NF)
h_axis = np.linspace(H_MIN, H_MAX, NH)


def to_grid(values, lo: float, hi: float, n: int) -> np.ndarray:
    """Физическая координата → индекс ячейки (round); вне [lo, hi] или NaN → −1."""
    v = np.asarray(values, float)
    ok = np.isfinite(v) & (v >= lo) & (v <= hi)
    j = np.full(v.shape, -1, int)
    j[ok] = np.round((v[ok] - lo) / (hi - lo) * (n - 1)).astype(int)
    return j


def masks_from_sao(sao: dict, thick: int = 1) -> np.ndarray:
    """Полилинии SAO (O-компонента) → маска int8 [NH, NF] классов CLASSES.

    Порядок записи F2 → F1 → E → Es: при коллизии ячейки побеждает записанный позже
    (семантика прототипа). Точки вне решётки отбрасываются.
    """
    y = np.zeros((NH, NF), np.int8)
    for cls, key in TRACE_KEYS.items():
        fq, vh = sao.get(f"{key}_freq"), sao.get(f"{key}_vh")
        if fq is None or vh is None or not len(fq):
            continue
        jf, jh = to_grid(fq, F_MIN, F_MAX, NF), to_grid(vh, H_MIN, H_MAX, NH)
        ok = (jf >= 0) & (jh >= 0)
        if not ok.any():
            continue
        ci = CLASSES.index(cls)
        for d in range(-thick, thick + 1):
            y[np.clip(jh[ok] + d, 0, NH - 1), jf[ok]] = ci
    return y


def fmax_readout(pm: np.ndarray, ci: int, axis: np.ndarray = f_axis, min_pixels: int = 2) -> float:
    """Максимальная частота присутствия класса ci: последняя колонка с ≥ min_pixels пикселями."""
    cols = np.flatnonzero((pm == ci).sum(0) >= min_pixels)
    return float(axis[cols[-1]]) if len(cols) else float("nan")


def hmin_readout(pm: np.ndarray, ci: int, axis: np.ndarray = h_axis, pct: float = 5.0) -> float:
    """Нижняя кромка класса ci: pct-й перцентиль строк, где класс присутствует."""
    rows = np.flatnonzero((pm == ci).any(1))
    return float(axis[int(np.percentile(rows, pct))]) if len(rows) else float("nan")
