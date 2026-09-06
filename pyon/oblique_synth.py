# -*- coding: utf-8 -*-
"""
oblique_synth.py — аналитический синтез ионограмм наклонного зондирования (НЗ)
из вертикальных следов ARTIST (SAO).

Физика (Э2 §2): закон секанса f_нз = k·f_в·sec(φ0) + теорема эквивалентности Мартина
(групповой путь наклонного луча = путь по прямым в треугольнике «передатчик — вершина на
действующей высоте h′(f_в) эквивалентной вертикальной волны — приёмник»).
Геометрия ПО УМОЛЧАНИЮ — сферическая (Э2 §2.4): треугольник Брейта–Тьюва на сферической
Земле R = 6371 км, k = 1 (самосогласованный синтетический мир; эмпирический k ≈ 1.11
измерен сверкой с ARTIST M3000F2, Э2 §6.0). Плоская Земля (geometry="flat") оставлена
только для сравнения: завышает МПЧ на ~17 % при D = 1500 км и ~73 % при D = 3000 км.
Многоскачковые моды: скачок дальностью D/n, путь ×n; при уходе вершины за горизонт
(tan Δ ≤ 0) мода геометрически невозможна. O- и X-компоненты — раздельно
(x_trace_from_o: частоты точно по fo² = fx(fx − fB), высоты — первый порядок; точный h′x
по профилю NHPC — задача E3b).

Литература: Martyn (1935), Smith (1939, кривые передачи), Davies "Ionospheric
Radio" (1990, гл. 6), РД 52.26.817–2023 §7, Щирый (2007). Подробно —
research/oblique_synthesis.md.
"""
from __future__ import annotations

import numpy as np

# каноническая решётка НЗ-ионограммы
NF = NP = 128
FOB_MIN, FOB_MAX = 2.0, 24.0      # МГц
P_MIN, P_MAX = 300.0, 3200.0      # км (групповой путь)
C_KM_MS = 299.792458              # км/мс
D_SET = (300.0, 800.0, 1500.0)    # дальности трасс, км

fob_axis = np.linspace(FOB_MIN, FOB_MAX, NF)
p_axis = np.linspace(P_MIN, P_MAX, NP)

# классы наклонной маски
OB_CLASSES = ["BG", "F2", "F1", "E", "Es", "MH"]  # MH = кратник основного следа (2 скачка F2)


R_E = 6371.0  # км


def oblique_transform(fv, hv, d_km: float, hops: int = 1, geometry: str = "spherical"):
    """Вертикальный след (f_в, h′) → наклонный (f_нз, P′групп) для трассы d_km, hops скачков.

    Каждая точка вертикального следа даёт точку наклонного; немонотонность
    f_нз(f_в) автоматически порождает нижний/верхний лучи и «нос» МПЧ.

    geometry="flat": плоская Земля (sec φ0 = sqrt(1+(D/2n/h′)²)) — годится до D/n ≲ 500 км,
    дальше ЗАВЫШАЕТ секанс (и МПЧ) в разы.
    geometry="spherical" (по умолчанию): треугольник Брейта–Тьюва на сферической Земле —
    θ = (D/n)/2R (половинный центральный угол скачка),
    tanΔ = (cosθ − R/(R+h′))/sinθ (угол места), sinφ0 = R·cosΔ/(R+h′),
    P′ = 2n·sqrt(R² + (R+h′)² − 2R(R+h′)cosθ) (хорды передатчик—вершина—приёмник).
    Точки с Δ < 0 (вершина за горизонтом — скачок геометрически невозможен для столь
    низкого h′) отбрасываются: так 1E-мода сама исчезает при D ≳ 2000 км.
    Поправочный множитель k≈1.0–1.2 (кривизна слоя) НЕ вводится (k=1): синтетический мир
    самосогласован, а для сопоставления с ARTIST M3000F2 расхождение — часть проверки.
    """
    fv = np.asarray(fv, float); hv = np.asarray(hv, float)
    ok = np.isfinite(fv) & np.isfinite(hv) & (hv > 50.0) & (fv > 0.1)
    fv, hv = fv[ok], hv[ok]
    if geometry == "flat":
        half = d_km / hops / 2.0
        sec = np.sqrt(1.0 + (half / hv) ** 2)
        return fv * sec, 2.0 * hops * np.sqrt(half ** 2 + hv ** 2)
    theta = (d_km / hops) / (2.0 * R_E)
    rh = R_E + hv
    tan_delta = (np.cos(theta) - R_E / rh) / np.sin(theta)
    vis = tan_delta > 0.0                      # вершина видна над горизонтом
    delta = np.arctan(tan_delta[vis])
    sin_phi0 = R_E * np.cos(delta) / rh[vis]
    sec = 1.0 / np.sqrt(np.clip(1.0 - sin_phi0 ** 2, 1e-9, None))
    chord = np.sqrt(R_E ** 2 + rh[vis] ** 2 - 2 * R_E * rh[vis] * np.cos(theta))
    return fv[vis] * sec, 2.0 * hops * chord


def muf(fv, hv, d_km: float, hops: int = 1, geometry: str = "spherical") -> float:
    """МПЧ моды = максимум f_нз по следу (эквивалент касания кривой передачи Смита)."""
    f_ob, _ = oblique_transform(fv, hv, d_km, hops, geometry)
    return float(f_ob.max()) if len(f_ob) else np.nan


def muf3000_check(sao: dict) -> tuple[float, float]:
    """Перекрёстная проверка с ARTIST: наш МПЧ(3000, сфер.) против M3000F2·foF2.

    ARTIST считает M3000F2 методом кривых передачи Смита — независимая реализация
    той же физики. Возвращает (наш МПЧ, МПЧ ARTIST); nan при нехватке данных.
    """
    sc = sao.get("scaled")
    fq, vh = sao.get("F2o_freq"), sao.get("F2o_vh")
    if sc is None or fq is None or vh is None or not len(fq):
        return np.nan, np.nan
    m3, fof2 = float(sc.get("M3000F2", np.nan)), float(sc.get("foF2", np.nan))
    return muf(fq, vh, 3000.0, 1, "spherical"), m3 * fof2


def bouguer_trace(prof_h, prof_fp, f_mhz: float, phi0_deg: float, n_int: int = 4000):
    """«Точный вариант»: численная трассировка по ИСТИННОМУ профилю fp(h) (NHPC ARTIST)
    в сферически-слоистой изотропной ионосфере (инвариант Бугера).

    n(h)·(R+h)·sinφ(h) = R·sinφ0;  отражение при n(h_r) = R·sinφ0/(R+h_r);
    D = 2R∫ tanφ/r dr,  P′ = 2∫ dr/(n cosφ)  (групповой показатель 1/n).
    Возвращает (D_км, P′_км) или (nan, nan), если луч проникает слой.
    Квадратура по dr с полушагом у точки отражения (лог-особенность интегрируема).
    """
    prof_h = np.asarray(prof_h, float); prof_fp = np.asarray(prof_fp, float)
    ok = np.isfinite(prof_h) & np.isfinite(prof_fp)
    prof_h, prof_fp = prof_h[ok], prof_fp[ok]
    if len(prof_h) < 5:
        return np.nan, np.nan
    sin_phi0 = np.sin(np.radians(phi0_deg))
    h0 = prof_h[0]
    # точка отражения: n(h) (R+h) = R sinφ0, n² = 1 − (fp/f)²
    hs = np.linspace(h0, prof_h[-1], n_int)
    fp = np.interp(hs, prof_h, prof_fp)
    n2 = 1.0 - (fp / f_mhz) ** 2
    lhs = np.sqrt(np.clip(n2, 0, None)) * (R_E + hs)
    rhs = R_E * sin_phi0
    below = lhs <= rhs
    if not below.any():
        return np.nan, np.nan                     # луч прошёл насквозь
    ir = int(np.argmax(below))
    if ir == 0:
        return np.nan, np.nan
    h_r = hs[ir - 1] + (hs[ir] - hs[ir - 1]) * (lhs[ir - 1] - rhs) / (lhs[ir - 1] - lhs[ir] + 1e-12)
    # интегрирование от земли до h_r (свободное пространство до h0 — аналитически)
    hh = h_r - (h_r - h0) * np.linspace(1, 0, n_int)[1:] ** 2   # сгущение к точке отражения
    fp2 = np.interp(hh, prof_h, prof_fp)
    n = np.sqrt(np.clip(1.0 - (fp2 / f_mhz) ** 2, 1e-10, None))
    r = R_E + hh
    sin_phi = np.clip(rhs / (n * r), 0, 1 - 1e-12)
    cos_phi = np.sqrt(1 - sin_phi ** 2)
    dD = np.trapz(sin_phi / cos_phi / r, hh) * 2 * R_E
    dP = np.trapz(1.0 / (n * cos_phi), hh) * 2
    # участок 0..h0 (вакуум): φ = const из sinφ = R sinφ0/r
    rr = np.linspace(R_E, R_E + h0, 200)
    sph = np.clip(R_E * sin_phi0 / rr, 0, 1 - 1e-12); cph = np.sqrt(1 - sph ** 2)
    dD += np.trapz(sph / cph / rr, rr) * 2 * R_E
    dP += np.trapz(1.0 / cph, rr) * 2
    return float(dD), float(dP)


def group_index(fp: np.ndarray, f: float, f_b: float, mode: str) -> np.ndarray:
    """Групповой показатель μ′ = d(f·n)/df квазипродольной формулы Апплтона–Хартри без столкновений:
    n² = 1 − X/(1 − s·Y), X = fp²/f², Y = fB/f; s = 0 — O-мода (отражение при X = 1), s = 1 — X-мода
    (отражение при X = 1 − Y ⇔ fo² = fx(fx − fB)). Возвращает μ′ на узлах профиля до отражения (NaN выше)."""
    s = 1.0 if mode == "X" else 0.0
    def n_of(ff):
        return np.sqrt(np.clip(1.0 - (fp / ff) ** 2 / (1.0 - s * f_b / ff), 1e-9, None))
    n = n_of(f); d = f * 1e-3
    mu = n + f * (n_of(f + d) - n_of(f - d)) / (2 * d)
    refl = (fp / f) ** 2 >= (1.0 - s * f_b / f)
    j = np.flatnonzero(refl)
    mu = mu.astype(float)
    if len(j):
        mu[j[0] + 1:] = np.nan
    return np.clip(mu, 0.0, 50.0)


def hprime_from_profile(prof_h, prof_fp, f, f_b: float, mode: str):
    """Действующая высота h′(f) = h₀ + ∫ μ′ dh до высоты отражения по профилю (h, fp); NaN, если волна не
    отражается внутри профиля. f — скаляр или массив (векторизовано по частотам: матрица [n_f, n_h])."""
    h = np.asarray(prof_h, float); fp = np.asarray(prof_fp, float)
    ok = np.isfinite(h) & np.isfinite(fp) & (fp > 0)
    h, fp = h[ok], fp[ok]
    f = np.atleast_1d(np.asarray(f, float))
    if len(h) < 3:
        return np.full(f.shape, np.nan) if f.size > 1 else np.nan
    s_ = 1.0 if mode == "X" else 0.0
    ff = f[:, None]
    def n_of(q):
        return np.sqrt(np.clip(1.0 - (fp[None, :] / q) ** 2 / (1.0 - s_ * f_b / q), 1e-9, None))
    d = ff * 1e-3
    mu = np.clip(n_of(ff) + ff * (n_of(ff + d) - n_of(ff - d)) / (2 * d), 0.0, 50.0)         # [n_f, n_h]
    refl = (fp[None, :] / ff) ** 2 >= (1.0 - s_ * f_b / ff)                                   # отражение
    has = refl.any(1); j = np.where(has, refl.argmax(1), len(h) - 1)
    idx = np.arange(len(h))[None, :]
    mu = np.where(idx <= j[:, None], mu, 0.0)                                                  # интегрируем до отражения
    hp = h[0] + np.trapz(mu, h, axis=1)
    hp = np.where(has & (j > 0), hp, np.nan)
    return hp if f.size > 1 else float(hp[0])


def x_trace_from_o(fv_o, hv_o, f_b: float, prof_h=None, prof_fp=None):
    """X-след из O-следа вертикальной ионограммы (в корпусе ARTIST-5 X-полилиний
    в SAO нет — проверено 0/224 файлов, есть только скаляр fxI).

    Частоты: точное магнитоионное соотношение отражения от одного уровня Nₑ —
    fo² = fx(fx − fB)  ⇒  fx = fB/2 + sqrt((fB/2)² + fo²)  (НЕ приближение fo+fB/2).
    Высоты: h′ переносится с O-точки — приближение первого порядка (групповое
    замедление X-моды отличается; выше ~2fB ошибка мала, у fmin — заметна).
    «Точный вариант» (задача серверного эксперимента): h′x(f) интегрированием
    группового показателя X-моды (Апплтон–Хартри) по профилю NHPC.
    fB — гирочастота станции (МГц); для F-области умножать на ≈0.89 (см. онтологию,
    iono-observation:hasStationGyroFrequency).
    """
    fv_o = np.asarray(fv_o, float); hv = np.asarray(hv_o, float).copy()
    fx = f_b / 2.0 + np.sqrt((f_b / 2.0) ** 2 + fv_o ** 2)
    if prof_h is not None and prof_fp is not None and len(np.asarray(prof_h)) >= 3:
        # E3b (аудит 2026-09-06): h′x(fx) = h′o(fo) + [∫μ′x(fx) dh − ∫μ′o(fo) dh] по профилю NHPC;
        # без поправки ошибка медиана 11 км, у носа до 70 км. Поправка применяется там, где обе
        # волны отражаются внутри профиля; иначе — первый порядок (h′x = h′o).
        ho = hprime_from_profile(prof_h, prof_fp, fv_o, f_b, "O")
        hx = hprime_from_profile(prof_h, prof_fp, fx, f_b, "X")
        okc = np.isfinite(ho) & np.isfinite(hx)
        hv[okc] = hv[okc] + (hx[okc] - ho[okc])
    return fx, hv


def raster_polyline(xs, ys, x0, x1, y0, y1, n: int = 128, thick: int = 1) -> np.ndarray:
    """Ломаная (в физических координатах) → булева маска n×n [y, x] с утолщением ±thick по y."""
    g = np.zeros((n, n), bool)
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) < 1:
        return g
    for k in range(max(1, len(xs) - 1)):
        k2 = min(k + 1, len(xs) - 1)
        # число точек на сегмент — по его длине в ячейках (у носа 2F2 сегменты по P′ длиннее 8 ячеек
        # давали разрывы: аудит 2026-09-06, 6 % следов MH)
        span = max(abs(xs[k2] - xs[k]) / (x1 - x0), abs(ys[k2] - ys[k]) / (y1 - y0)) * (n - 1)
        m = int(max(8, np.ceil(span) + 2))
        xi = np.linspace(xs[k], xs[k2], m); yi = np.linspace(ys[k], ys[k2], m)
        jx = np.round((xi - x0) / (x1 - x0) * (n - 1)).astype(int)
        jy = np.round((yi - y0) / (y1 - y0) * (n - 1)).astype(int)
        okm = (jx >= 0) & (jx < n) & (jy >= 0) & (jy < n)
        g[jy[okm], jx[okm]] = True
    if thick:
        acc = g.copy()
        for s in range(1, thick + 1):
            acc[s:] |= g[:-s]; acc[:-s] |= g[s:]
        g = acc
    return g


# вертикальные следы SAO → классы наклонной маски (1 скачок; MH — кратник F2, 2 скачка).
# O- и X-компоненты пересчитываются РАЗДЕЛЬНО, каждая из СВОИХ полилиний SAO
# (F2o/F1o/Eo/Es против F2x/F1x/Ex): так в синтетику переносится наблюдённое
# магнитоионное расщепление, а не модельное. Оговорка (Davies 1990): для X-моды
# теоремы эквивалентности приближённы (сдвиг ~fB/2, продольно-поперечная
# зависимость) — «точный вариант» потребует магнитоионной трассировки
# (Джонс–Стивенсон; PHaRLAP/IONORT); принятая здесь точность — уровень данных SAO.
SAO_TRACES_BY_COMPONENT = {
    "O": {"F2": "F2o", "F1": "F1o", "E": "Eo", "Es": "Es"},
    "X": {"F2": "F2x", "F1": "F1x", "E": "Ex", "Es": "Esx"},   # Es в SAO 4.3 без X-полилинии — Esx строится из Es
}
SAO_TRACES = SAO_TRACES_BY_COMPONENT["O"]         # обратная совместимость


def oblique_masks_from_sao(sao: dict, d_km: float, component: str = "O") -> tuple[np.ndarray, dict]:
    """SAO → (маска int8 [NP, NF] классов OB_CLASSES, точные МПЧ-метки).

    component: "O" или "X" — какое семейство полилиний SAO пересчитывать
    (см. SAO_TRACES_BY_COMPONENT). Метки: muf_<класс> для 1 скачка и muf_MH
    (МПЧ кратника 2F2 — по S1 обязана быть ≤ muf_F2).
    """
    traces = SAO_TRACES_BY_COMPONENT[component]
    f2key = traces["F2"]
    y = np.zeros((NP, NF), np.int8)
    labels = {}
    mh_mufs = []
    sao = dict(sao)
    if component == "X" and not len(sao.get(f"{f2key}_freq", []) or []):
        # в корпусе ARTIST-5 X-полилиний нет (0/200 val, аудит 2026-09-06) → X-следы из O-следов:
        # частоты точно по fo² = fx(fx − fB) (fB станции: ×0.89 для F-области, ×0.95 для E — гирочастота
        # на высоте слоя), высоты — с поправкой E3b по профилю NHPC, если он есть
        gc = sao.get("geophys_const")
        fb0 = float(np.atleast_1d(np.asarray(gc, float))[0]) if gc is not None and np.size(gc) else 1.3
        ph, pf = sao.get("profile_h"), sao.get("profile_fp")
        for cls, okey in SAO_TRACES_BY_COMPONENT["O"].items():
            xkey = traces.get(cls)
            fq, vh = sao.get(f"{okey}_freq"), sao.get(f"{okey}_vh")
            if xkey is None or fq is None or vh is None or not len(fq):
                continue
            fb = fb0 * (0.89 if cls in ("F2", "F1") else 0.95)
            fx, hx = x_trace_from_o(fq, vh, fb, ph, pf)
            sao[f"{xkey}_freq"], sao[f"{xkey}_vh"] = fx, hx
    # сначала MH (2 скачка F2), потом 1-скачковые поверх — приоритет у основного следа.
    # MH сознательно ограничен кратником F2: доминирующая многоскачковая мода на реальных НЗ;
    # смешение слоёв в одном классе ломало бы семантику мод 2F2 при проверке форм S1/S2.
    fq, vh = sao.get(f"{f2key}_freq"), sao.get(f"{f2key}_vh")
    if fq is not None and vh is not None and len(fq):
        f2h, p2h = oblique_transform(fq, vh, d_km, hops=2)
        if len(f2h):
            y[raster_polyline(f2h, p2h, FOB_MIN, FOB_MAX, P_MIN, P_MAX, NP)] = OB_CLASSES.index("MH")
            mh_mufs.append(float(f2h.max()))
    for cls, key in traces.items():
        fq, vh = sao.get(f"{key}_freq"), sao.get(f"{key}_vh")
        if fq is None or vh is None or not len(fq):
            continue
        f1h, p1h = oblique_transform(fq, vh, d_km, hops=1)
        if len(f1h):
            y[raster_polyline(f1h, p1h, FOB_MIN, FOB_MAX, P_MIN, P_MAX, NP)] = OB_CLASSES.index(cls)
            labels[f"muf_{cls}"] = float(f1h.max())
    labels["muf_MH"] = max(mh_mufs) if mh_mufs else np.nan
    labels["D_km"] = d_km
    return y, labels
