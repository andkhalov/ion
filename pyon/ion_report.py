"""
ion_report.py — рендер ионограммы «как у дигизонда» (Ion2PNG) и сравнение с ARTIST.

  digisonde_report(...)   — ионограмма RSF/SBF + профиль NHPC (истинная высота) + следы ARTIST
                            + таблица характеристик + таблица MUF(D) (оценка по закону секанса)
  compare_sao_parsers(...) — SAO через digi_formats и через pynasonde + значения с официального PNG
  baseline_scale(...)     — простейший собственный автоскейлер по сырой ионограмме
  baseline_vs_artist(...) — прогон по всем ионограммам суток: наш baseline vs ARTIST

Все функции работают с парсерами digi_formats.py; pynasonde нужен только для compare_sao_parsers.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyon.digi_formats as dfm
from pyon import oblique_synth as obs

R_EARTH = obs.R_E

# порядок и подписи как на Ion2PNG (левая колонка отчёта дигизонда)
REPORT_ROWS = [
    ("foF2", "foF2"), ("foF1", "foF1"), ("foF1p", "foF1p"), ("foE", "foE"), ("foEp", "foEp"), ("fxI", "fxI"),
    ("foEs", "foEs"), ("fmin", "fmin"), None,
    ("MUF(D)", "MUF3000F2"), ("M(D)", "M3000F2"), ("D", "D"), None,
    ("h'F", "hF"), ("h'F2", "hF2"), ("h'E", "hE"), ("h'Es", "hEs"), None,
    ("hmF2", "hmF2"), ("hmF1", "hmF1"), ("hmE", "zmE"), ("yF2", "yF2"), ("yF1", "yF1"), ("yE", "yE"),
    ("B0", "B0"), ("B1", "B1"), None, ("TEC", "TEC"),
]
TRACES = [  # (ключ в SAO, подпись, цвет, стиль)
    ("F2o", "F2 (O)", "navy", "-"), ("F1o", "F1 (O)", "royalblue", "-"), ("Eo", "E (O)", "saddlebrown", "-"),
    ("Es", "Es (O)", "magenta", "-"), ("F2x", "F2 (X)", "darkgreen", "--"), ("F1x", "F1 (X)", "green", "--"),
    ("Ex", "E (X)", "olive", "--"),
]


def echo_mask(df: pd.DataFrame, margin_db: float = 6.0) -> pd.Series:
    return df.amp_db > df.mpa_db + margin_db


def c_level(sao: dict) -> str:
    """Уровень уверенности ARTIST: группа 5 «analysis flags», позиция 10 (две цифры, 11 — лучший, 55 — худший)."""
    a = sao.get("analysis_flags", [])
    return f"{int(a[9]):02d}" if len(a) >= 10 else "n/a"


def muf_secant(trace_f, trace_h, distances_km=(100, 200, 400, 600, 800, 1000, 1500, 3000)) -> dict:
    """MUF(D) по O-следу F2: зеркало на действующей высоте над сферической Землёй, закон секанса —
    та же физика, что `oblique_synth.muf` (ревизия 2026-09-04: собственная формула снята, обе
    давали тождественный результат; tests/test_physics.py). Обычно на ~10 % ниже ARTIST MUF(3000),
    считаемой по истинному профилю (k_эмп ≈ 1.11, Э2 §6.0)."""
    return {D: obs.muf(trace_f, trace_h, float(D), 1, "spherical") for D in distances_km}


def digisonde_report(ion_file: str, sao_file: str, png_file: str | None = None, fmax: float | None = None,
                     hmax: float | None = None, margin_db: float = 6.0, show_azimuth: bool = False,
                     figsize=(17, 7.5), title: str | None = None):
    """Рисует отчёт в стиле Ion2PNG. Возвращает (fig, dict с характеристиками и MUF-таблицей)."""
    pf, df = dfm.read_ionogram(ion_file)
    sao = dfm.read_sao(sao_file)
    sc = sao["scaled"]
    fmax = fmax or pf.f_stop_hz / 1e6
    hmax = hmax or min(df.height_km.max(), 1000)
    e = df[echo_mask(df, margin_db)]

    ncols = 3 if png_file else 2
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, ncols, width_ratios=[0.9, 3.2] + ([3.0] if png_file else []), wspace=0.08)
    ax_t = fig.add_subplot(gs[0, 0]); ax_t.axis("off")
    ax = fig.add_subplot(gs[0, 1])

    # --- эхо: O красным, X зелёным (как у Lowell); для RSF можно раскрасить по азимуту
    if show_azimuth and "azimuth_deg" in e:
        cols = {0: "#8B0000", 60: "#00BFFF", 120: "#1E90FF", 180: "#FFD700", 240: "#FFA500", 300: "#191970", 360: "#808080"}
        for az, c in cols.items():
            s = e[(e.azimuth_deg == az)]
            ax.scatter(s.freq_mhz, s.height_km, s=4, c=c, linewidths=0, label=f"az {az}°" if az < 360 else "n/a")
    else:
        for pol, c, lab in [("O", "#d62728", "O-эхо"), ("X", "#2ca02c", "X-эхо")]:
            s = e[e.pol == pol]
            ax.scatter(s.freq_mhz, s.height_km, s=4, c=c, linewidths=0, label=lab, alpha=0.8)

    # --- следы ARTIST из SAO (действующие высоты)
    for key, lab, c, ls in TRACES:
        fq, vh = sao.get(f"{key}_freq"), sao.get(f"{key}_vh")
        if fq is not None and len(fq):
            ax.plot(fq, vh, ls, color=c, lw=1.8, label=f"ARTIST {lab}")
    # --- профиль NHPC: плазменная частота ↔ ИСТИННАЯ высота (чёрная линия на отчёте дигизонда)
    if len(sao.get("profile_h", [])):
        ax.plot(sao["profile_fp"], sao["profile_h"], "-", color="black", lw=2.2, label="NHPC: fp(h) — истинная высота")
    # --- вертикальные метки критических частот
    for name, c in [("foF2", "#d62728"), ("fxI", "#2ca02c"), ("foF1", "dimgray"), ("foE", "saddlebrown"), ("foEs", "magenta"), ("fmin", "gray")]:
        v = sc.get(name, np.nan)
        if np.isfinite(v):
            ax.axvline(v, color=c, ls=":", lw=1); ax.text(v, hmax * 0.985, name, color=c, fontsize=8, ha="center", va="top")
    ax.set_xlim(df.freq_mhz.min(), fmax); ax.set_ylim(80, hmax)
    ax.set_xlabel("Частота, МГц"); ax.set_ylabel("Высота, км (h′ для эха и следов; истинная — для профиля)")
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_title(title or f"{Path(ion_file).name}: {pf.date} UT — {pf.format_name}, {len(df.freq_mhz.unique())} частот × {pf.n_heights} высот")

    # --- таблица характеристик (левая колонка)
    lines = []
    for row in REPORT_ROWS:
        if row is None:
            lines.append("─" * 16); continue
        lab, key = row
        v = sc.get(key, np.nan)
        lines.append(f"{lab:<7s}{'N/A' if not np.isfinite(v) else f'{v:8.3f}'.rstrip('0').rstrip('.'):>9s}")
    lines += ["─" * 16, f"C-level     {c_level(sao)}", f"{sao['system_desc'].split(',')[1].strip() if ',' in sao['system_desc'] else ''}"]
    ax_t.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top", ha="left", transform=ax_t.transAxes)

    # --- таблица MUF(D) по закону секанса + ARTIST MUF(3000)
    muf = muf_secant(sao.get("F2o_freq", []), sao.get("F2o_vh", []))
    dist = list(muf.keys())
    txt = "D   " + " ".join(f"{d:>5d}" for d in dist) + " [км]\nMUF " + " ".join(f"{muf[d]:5.1f}" for d in dist) + " [МГц]  (оценка: зеркало на h′, закон секанса)"
    if np.isfinite(sc.get("MUF3000F2", np.nan)):
        txt += f"\nARTIST MUF(3000) = {sc['MUF3000F2']:.2f} МГц, M(3000) = {sc['M3000F2']:.3f}  (по истинному профилю NHPC)"
    fig.text(0.13, 0.015, txt, family="monospace", fontsize=8.5, va="bottom")

    if png_file and os.path.exists(png_file):
        ax_p = fig.add_subplot(gs[0, 2]); ax_p.imshow(mpimg.imread(png_file)); ax_p.axis("off"); ax_p.set_title("официальный PNG (Ion2PNG / DIAS)")
    fig.subplots_adjust(bottom=0.14, top=0.93, left=0.02, right=0.99)
    return fig, dict(scaled=sc, muf_secant=muf, c_level=c_level(sao), preface=pf)


# ---------------------------------------------------------------------------- сравнение парсеров
PNG_VALUES_JI91J_000000 = {  # прочитано глазами с JI91J_2022001000000_IO.PNG (Ion2PNG 1.2.05)
    "foF2": 8.200, "foF1": np.nan, "foF1p": np.nan, "foE": np.nan, "foEp": 0.65, "fxI": 8.60, "foEs": np.nan, "fmin": 2.50,
    "MUF3000F2": 21.94, "M3000F2": 2.68, "D": 3000.0, "hF": 223.0, "hF2": np.nan, "hE": np.nan, "hEs": np.nan,
    "hmF2": 369.9, "hmF1": np.nan, "zmE": 110.0, "yF2": 114.4, "yF1": np.nan, "yE": 20.0, "B0": 136.0, "B1": 1.16,
}
PYN_TO_OURS = {  # имена колонок pynasonde.get_scaled_datasets() → наши (SAO 4.3)
    "foF2": "foF2", "foF1": "foF1", "foF1p": "foF1p", "foE": "foE", "foEp": "foEp", "fxI": "fxI", "foEs": "foEs", "fmin": "fmin",
    "MUF3000": "MUF3000F2", "M3000F": "M3000F2", "D": "D", "hF": "hF", "hF2": "hF2", "hE": "hE", "hEs": "hEs",
    "hmF2": "hmF2", "hmF1": "hmF1", "hmE": "zmE", "ymF2": "yF2", "ymF1": "yF1", "ymE": "yE", "B0": "B0", "B1": "B1",
    "TEC": "TEC", "h05NmF2": "zhalfNm", "foFp": "foF2p", "Ht": "scaleF2", "fMUF": "fMUF", "hMUF": "h_fMUF",
}


def compare_sao_parsers(sao_file: str, png_values: dict | None = None) -> pd.DataFrame:
    """Одна и та же SAO тремя способами: digi_formats, pynasonde, значения с официального PNG (если даны)."""
    ours = dfm.read_sao(sao_file)["scaled"]
    from pynasonde.digisonde.parsers.sao import SaoExtractor
    ex = SaoExtractor(sao_file, True, True); ex.extract()
    pyn_row = ex.get_scaled_datasets().iloc[0]
    pyn = {PYN_TO_OURS.get(k, k): v for k, v in pyn_row.items() if k in PYN_TO_OURS}
    keys = [k for k in ours.index if k in pyn or (png_values and k in png_values)]
    tab = pd.DataFrame({"digi_formats": [ours.get(k, np.nan) for k in keys],
                        "pynasonde": [pyn.get(k, np.nan) for k in keys]}, index=keys)
    if png_values:
        tab["PNG (Ion2PNG)"] = [png_values.get(k, np.nan) for k in keys]
    tab["Δ pyn−ours"] = tab["pynasonde"] - tab["digi_formats"]
    return tab


# ---------------------------------------------------------------------------- собственный baseline-автоскейлер
def baseline_scale(df: pd.DataFrame, margin_db: float = 6.0, h_lo: float = 150.0, h_hi: float = 700.0,
                   min_bins: int = 3, gap_max_mhz: float = 1.0, min_run: int = 3) -> dict:
    """Наивный автоскейлер по сырой ионограмме (для сравнения с ARTIST, не для использования!).

    fmin  — начало первого «пробега» из ≥ min_run подряд идущих частот, где есть ≥ min_bins эхо-бинов O-моды;
    h'F   — медиана минимальных высот O-эха вдоль следа (нижняя кромка следа F);
    foF2  — последняя частота O-следа, если считать след непрерывным при разрывах ≤ gap_max_mhz;
    fxI   — то же для X-моды.
    Ничего не знает о кратниках, Es, боковых эхо, помехах-столбцах и spread-F — это и видно в сравнении.
    """
    out = dict(fmin=np.nan, hF=np.nan, foF2=np.nan, fxI=np.nan)
    e = df[echo_mask(df, margin_db) & (df.height_km >= h_lo) & (df.height_km <= h_hi)]
    # шаг развёртки — медиана разностей (минимум сбивают «forced»-частоты со смещением на 10–30 кГц)
    step = float(np.median(np.diff(np.sort(df.freq_mhz.unique())))) if df.freq_mhz.nunique() > 1 else 0.1

    def trace_end(sub):
        cnt = sub.groupby("freq_mhz").size()
        good = cnt[cnt >= min_bins].index.values
        if len(good) == 0:
            return np.nan, np.nan
        # старт: первый пробег из min_run подряд идущих частот (шаг развёртки)
        f0 = np.nan
        for i in range(len(good) - min_run + 1):
            if good[i + min_run - 1] - good[i] <= (min_run - 1) * step * 1.01:
                f0 = good[i]; break
        if not np.isfinite(f0):
            return np.nan, np.nan
        last = f0
        for f in good[good > f0]:
            if f - last > gap_max_mhz:
                break
            last = f
        return f0, last

    fo0, fo_end = trace_end(e[e.pol == "O"])
    fx0, fx_end = trace_end(e[e.pol == "X"])
    out["fmin"], out["foF2"], out["fxI"] = fo0, fo_end, fx_end
    if np.isfinite(fo0):
        band = e[(e.pol == "O") & (e.freq_mhz >= fo0) & (e.freq_mhz <= fo_end)]
        if len(band):
            # h'F у ARTIST — минимальная действующая высота следа F; берём 10-й процентиль нижней кромки,
            # чтобы одиночные шумовые бины ниже следа не занижали оценку
            out["hF"] = float(np.percentile(band.groupby("freq_mhz").height_km.min(), 10))
    return out


def baseline_vs_artist(ion_dir: str, sao_dir: str, ext: str, **kw) -> pd.DataFrame:
    """Для всех ионограмм каталога: наш baseline vs ARTIST (SAO). Возвращает таблицу по времени."""
    rows = []
    for f in sorted(glob.glob(os.path.join(ion_dir, f"*.{ext}"))):
        stem = Path(f).stem
        sao_f = os.path.join(sao_dir, stem + ".SAO")
        pf, df = dfm.read_ionogram(f)
        b = baseline_scale(df, **kw)
        r = dict(time=pf.date, **{f"{k}_base": v for k, v in b.items()})
        if os.path.exists(sao_f):
            sc = dfm.read_sao(sao_f)["scaled"]
            for k in ["fmin", "hF", "foF2", "fxI"]:
                r[f"{k}_artist"] = sc.get(k, np.nan)
            r["c_level"] = c_level(dfm.read_sao(sao_f))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def plot_baseline_vs_artist(tab: pd.DataFrame, station: str):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    for j, (k, unit) in enumerate([("foF2", "МГц"), ("fxI", "МГц"), ("fmin", "МГц"), ("hF", "км")]):
        a, b = tab[f"{k}_artist"], tab[f"{k}_base"]
        ax = axes[0, j]
        ax.plot(tab.time, a, "o-", ms=3, label="ARTIST (SAO)"); ax.plot(tab.time, b, "x--", ms=4, label="наш baseline")
        ax.set_title(f"{station}: {k}"); ax.set_ylabel(unit); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H")); ax.set_xlabel("UT, ч")
        ax = axes[1, j]
        ok = a.notna() & b.notna()
        ax.scatter(a[ok], b[ok], s=14)
        lim = [np.nanmin([a[ok].min(), b[ok].min()]), np.nanmax([a[ok].max(), b[ok].max()])]
        ax.plot(lim, lim, "k:", lw=1)
        rmse = np.sqrt(np.mean((a[ok] - b[ok]) ** 2)); bias = np.mean(b[ok] - a[ok])
        n_art = a.notna().sum(); n_base = b.notna().sum()
        ax.set_title(f"n={ok.sum()}  RMSE={rmse:.2f} {unit}  bias={bias:+.2f}\nARTIST есть в {n_art}, baseline в {n_base} из {len(tab)}", fontsize=9)
        ax.set_xlabel(f"ARTIST {k}"); ax.set_ylabel(f"baseline {k}"); ax.grid(alpha=.3)
    return fig
