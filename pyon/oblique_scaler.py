# -*- coding: utf-8 -*-
"""
oblique_scaler.py — измеритель характеристик НАКЛОННОЙ ионограммы по предсказанной разметке
(аналог `pyon.scaler` для ВЗ; запрос АХ 2026-09-07: «надо смотреть, что по нашей разметке мы сможем
вывести, а не просто пятна на картинке»).

Аналога ARTIST для НЗ не существует, поэтому состав таблицы задан нормативом и практикой ЛЧМ-зондирования
(РД 52.26.817–2023 §7.3; Щирый 2007, гл. 3):
  МПЧ(мода)   — максимальная наблюдаемая частота моды (последняя частотная колонка с ≥2 пикселями класса);
  НПЧ         — наименьшая наблюдаемая частота по всем модам (первая колонка с любым следом);
  P′min(мода) — минимальный групповой путь моды («нос»), км, и задержка t = P′/c, мс;
  ОРЧ         — оптимальная рабочая частота 0.85·МПЧ(1F2) (стандартный эксплуатационный запас);
  h′экв       — эквивалентная высота отражения 1F2 из P′min и длины трассы D по сферической геометрии;
  fv(1F2)     — эквивалентная вертикальная частота МПЧ/sec φ0 (обратный секанс: сверка с foF2 станции);
  2F2/1F2     — инвариант Пономарчука (медиана по корпусу 0.79);
  X−O         — магнитоионное расщепление по МПЧ (проверяется формой S7 онтологии).
Все выходы — float, отсутствующее = NaN. Полоса следа по частоте (Δf) даётся для контроля полноты.
"""
from __future__ import annotations

import numpy as np

from pyon import canon
from pyon import oblique_synth as obs

MODES = ("F2", "F1", "E", "Es", "MH", "X")
OB_REPORT_ROWS = ["МПЧ1F2", "МПЧ2F2", "МПЧ_X", "МПЧ_E", "МПЧ_Es", "НПЧ", "ОРЧ",
                  "P1F2", "t1F2", "P2F2", "h_экв", "h_МПЧ", "fv1F2", "2F2/1F2", "X-O", "Δf1F2"]


def _mode_stats(pm: np.ndarray, cls: str) -> tuple[float, float, float]:
    """(МПЧ, P′min, ширина по частоте) для класса; NaN, если следа нет."""
    if cls not in obs.OB_CLASSES:
        return np.nan, np.nan, np.nan
    ci = obs.OB_CLASSES.index(cls)
    cols = np.flatnonzero((pm == ci).sum(0) >= 2)
    if not len(cols):
        return np.nan, np.nan, np.nan
    rows = np.flatnonzero((pm == ci).any(1))
    return (float(obs.fob_axis[cols[-1]]), float(obs.p_axis[rows[0]]),
            float(obs.fob_axis[cols[-1]] - obs.fob_axis[cols[0]]))


def equivalent_height(p_km: float, d_km: float) -> float:
    """h′ по групповому пути и длине трассы (сферическая геометрия, один скачок):
    P′ = 2·|хорда|, хорда² = R² + (R+h)² − 2R(R+h)cos θ, θ = D/2R. Решаем относительно h."""
    if not (np.isfinite(p_km) and np.isfinite(d_km)) or p_km <= 0:
        return np.nan
    R, th = obs.R_E, (d_km / 2.0) / obs.R_E
    c = np.cos(th)
    disc = (R * c) ** 2 - (R ** 2 - (p_km / 2.0) ** 2)
    if disc < 0:
        return np.nan
    return float(R * c + np.sqrt(disc) - R)


def sec_phi0(h_km: float, d_km: float) -> float:
    """Секанс угла падения в вершине для восстановленной высоты (обратно к oblique_transform)."""
    if not (np.isfinite(h_km) and np.isfinite(d_km)) or h_km <= 0:
        return np.nan
    return 1.0 / max(np.cos(obs.phi0_at(h_km, d_km)), 1e-6)


def scale_oblique(pm: np.ndarray, d_km: float = np.nan) -> dict:
    """Маска [NP, NF] классов obs.OB_CLASSES (+ длина трассы D, км) → таблица характеристик."""
    r: dict = {}
    st = {c: _mode_stats(pm, c) for c in MODES}
    r["МПЧ1F2"], r["P1F2"], r["Δf1F2"] = st["F2"]
    r["МПЧ2F2"], r["P2F2"], _ = st["MH"]
    r["МПЧ_X"] = st["X"][0]
    r["МПЧ_E"], r["МПЧ_Es"] = st["E"][0], st["Es"][0]
    any_cols = np.flatnonzero((pm > 0).sum(0) >= 2)
    r["НПЧ"] = float(obs.fob_axis[any_cols[0]]) if len(any_cols) else np.nan
    r["ОРЧ"] = 0.85 * r["МПЧ1F2"] if np.isfinite(r["МПЧ1F2"]) else np.nan
    r["t1F2"] = r["P1F2"] / obs.C_KM_MS if np.isfinite(r["P1F2"]) else np.nan
    r["h_экв"] = equivalent_height(r["P1F2"], d_km)          # эквивалентная высота на НОСУ (аналог h′F)
    # эквивалентная вертикальная частота: обратный секанс берётся в точке МПЧ (там своя высота), а не
    # на носу — иначе секанс завышен и fv занижена вдвое (проверка 2026-09-07)
    ci = obs.OB_CLASSES.index("F2")
    cols = np.flatnonzero((pm == ci).sum(0) >= 2)
    if len(cols) and np.isfinite(d_km):
        rows_at_muf = np.flatnonzero(pm[:, cols[-1]] == ci)
        p_at_muf = float(np.median(obs.p_axis[rows_at_muf])) if len(rows_at_muf) else np.nan
        h_muf = equivalent_height(p_at_muf, d_km)
        sec = sec_phi0(h_muf, d_km)
        r["h_МПЧ"] = h_muf
        r["fv1F2"] = r["МПЧ1F2"] / sec if np.isfinite(r["МПЧ1F2"]) and np.isfinite(sec) else np.nan
    else:
        r["h_МПЧ"] = r["fv1F2"] = np.nan
    r["2F2/1F2"] = (r["МПЧ2F2"] / r["МПЧ1F2"]) if np.isfinite(r["МПЧ2F2"]) and np.isfinite(r["МПЧ1F2"]) and r["МПЧ1F2"] > 0 else np.nan
    r["X-O"] = (r["МПЧ_X"] - r["МПЧ1F2"]) if np.isfinite(r["МПЧ_X"]) and np.isfinite(r["МПЧ1F2"]) else np.nan
    return r


def table_from_labels(lab: dict | np.ndarray, names: list[str] | None = None, d_km: float = np.nan) -> dict:
    """Аналитические метки синтеза → та же таблица (для колонки «метка» в панелях)."""
    if isinstance(lab, dict):
        g = lambda k: float(lab.get(k, np.nan))
    else:
        idx = {n: i for i, n in enumerate(names or [])}
        g = lambda k: float(lab[idx[k]]) if k in idx else np.nan
    out = {k: np.nan for k in OB_REPORT_ROWS}
    out["МПЧ1F2"], out["МПЧ2F2"], out["МПЧ_X"] = g("muf_F2"), g("muf_MH"), g("muf_F2_x")
    out["МПЧ_E"], out["МПЧ_Es"] = g("muf_E"), g("muf_Es")
    out["ОРЧ"] = 0.85 * out["МПЧ1F2"] if np.isfinite(out["МПЧ1F2"]) else np.nan
    out["2F2/1F2"] = out["МПЧ2F2"] / out["МПЧ1F2"] if np.isfinite(out["МПЧ2F2"]) and out["МПЧ1F2"] > 0 else np.nan
    out["X-O"] = out["МПЧ_X"] - out["МПЧ1F2"] if np.isfinite(out["МПЧ_X"]) and np.isfinite(out["МПЧ1F2"]) else np.nan
    return out
