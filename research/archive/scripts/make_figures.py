"""Рисует обзорные картинки по образцам RSF / SBF / SAO / EDP / DFT (см. format_description.md)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # корень проекта — для пакета modules
import modules.digi_formats as dfm

HERE = Path(__file__).parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
RSF_DIR = HERE / "RSF-samples-w-img-n-sao-n-dft"
SBF_DIR = HERE / "SBF-samples-w-img-n-sao"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def echo_mask(df: pd.DataFrame, margin_db: float = 6.0) -> pd.Series:
    """Эхо = амплитуда выше «наиболее вероятной амплитуды» (шума) группы на margin_db."""
    return df.amp_db > df.mpa_db + margin_db


def grid(df: pd.DataFrame, col: str, pol: str):
    """long-таблица → 2-D матрица (height × freq) для pcolormesh."""
    sub = df[df.pol == pol]
    p = sub.pivot_table(index="height_km", columns="freq_mhz", values=col, aggfunc="first")
    return p.columns.values, p.index.values, p.values


def plot_ionogram(df, pf, sao, title, fname, fmax, hmax):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, pol in zip(axes[:2], ["O", "X"]):
        f, h, a = grid(df, "amp_db", pol)
        # вычитаем шумовой уровень (MPA) → SNR-подобная величина
        _, _, m = grid(df, "mpa_db", pol)
        pc = ax.pcolormesh(f, h, np.clip(a - m, 0, None), cmap="inferno", vmin=0, vmax=30, shading="nearest")
        ax.set_title(f"{pol}-мода: амплитуда − MPA, дБ")
        ax.set_xlabel("Частота, МГц"); ax.set_ylabel("Действующая высота h', км")
        ax.set_xlim(f.min(), min(f.max(), fmax)); ax.set_ylim(80, hmax)
        fig.colorbar(pc, ax=ax, shrink=0.8, label="дБ над шумом")
    # третья панель: эхо-точки, цвет = доплер (RSF/SBF) + трасса ARTIST из SAO
    ax = axes[2]
    e = df[echo_mask(df)]
    for pol, mk in [("O", "s"), ("X", "^")]:
        s = e[e.pol == pol]
        sc = ax.scatter(s.freq_mhz, s.height_km, c=s.doppler, cmap="coolwarm", vmin=0, vmax=7, s=4, marker=mk, label=f"{pol} эхо", linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="Doppler bin (0..7)")
    for key, lab, c in [("F2o", "ARTIST F2 (O)", "lime"), ("F1o", "ARTIST F1 (O)", "cyan"), ("Eo", "ARTIST E (O)", "yellow"), ("Es", "ARTIST Es", "magenta")]:
        if f"{key}_freq" in sao and len(sao[f"{key}_freq"]):
            ax.plot(sao[f"{key}_freq"], sao[f"{key}_vh"], "-", color=c, lw=2, label=lab)
    sc_ = sao["scaled"]
    for name, c in [("foF2", "lime"), ("foE", "yellow"), ("foEs", "magenta"), ("fmin", "gray")]:
        if not np.isnan(sc_.get(name, np.nan)):
            ax.axvline(sc_[name], color=c, ls="--", lw=1, label=f"{name}={sc_[name]:.2f} МГц")
    ax.set_xlim(df.freq_mhz.min(), fmax); ax.set_ylim(80, hmax)
    ax.set_xlabel("Частота, МГц"); ax.set_title("Эхо (>MPA+6 дБ) и трассы ARTIST из SAO")
    ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(title)
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_rsf_directions(df, title, fname):
    """Только RSF: азимут прихода и доплер для эхо-сигналов."""
    e = df[echo_mask(df)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {0: "#8B0000", 60: "#4169E1", 120: "#1E90FF", 180: "#FFA500", 240: "#FFD700", 300: "#191970", 360: "#808080"}
    labels = {0: "вертикаль (0°)", 60: "60° (NE)", 120: "120° (SE)", 180: "180° (S)", 240: "240° (SW)", 300: "300° (NW)", 360: "не определено"}
    ax = axes[0]
    for az, c in colors.items():
        s = e[(e.azimuth_deg == az) & (e.pol == "O")]
        ax.scatter(s.freq_mhz, s.height_km, s=5, c=c, label=labels[az], linewidths=0)
    ax.set_xlim(1, 12); ax.set_ylim(80, 1000); ax.legend(fontsize=7, loc="upper right")
    ax.set_title("O-мода: азимут прихода (3 бита, 60° сектора)"); ax.set_xlabel("Частота, МГц"); ax.set_ylabel("h', км")
    ax = axes[1]
    s = e[e.pol == "O"]
    sc = ax.scatter(s.freq_mhz, s.height_km, s=5, c=s.phase_deg, cmap="twilight", linewidths=0)
    fig.colorbar(sc, ax=ax, label="фаза, °"); ax.set_xlim(1, 12); ax.set_ylim(80, 1000)
    ax.set_title("O-мода: фаза (5 бит × 11.25°)"); ax.set_xlabel("Частота, МГц")
    ax = axes[2]
    e.doppler.value_counts().sort_index().plot.bar(ax=ax, color="steelblue")
    ax.set_title("Распределение доплеровских бинов эхо"); ax.set_xlabel("Doppler bin"); ax.set_ylabel("число бинов дальности")
    fig.suptitle(title); fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def load_sao_dir(d: Path) -> tuple[pd.DataFrame, list]:
    rows, saos = [], []
    for f in sorted(glob.glob(str(d / "scaled" / "*.SAO"))):
        s = dfm.read_sao(f)
        saos.append((s["datetime"], s))
        r = s["scaled"].to_dict(); r["datetime"] = s["datetime"]; r["file"] = os.path.basename(f)
        r["n_profile"] = len(s.get("profile_h", []))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("datetime"), saos


def plot_sao_daily(tab: pd.DataFrame, station: str, fname: str):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    t = tab.datetime
    ax = axes[0]
    for k, c in [("foF2", "C3"), ("foF1", "C1"), ("foE", "C2"), ("foEs", "C4"), ("fmin", "gray"), ("fxI", "C5")]:
        if k in tab: ax.plot(t, tab[k], ".-", color=c, label=k, ms=4)
    ax.set_ylabel("МГц"); ax.legend(ncol=6, fontsize=8); ax.set_title(f"{station}: критические частоты (SAO, ARTIST)")
    ax = axes[1]
    for k, c in [("hmF2", "C3"), ("hF", "C0"), ("hF2", "C9"), ("hmF1", "C1"), ("hE", "C2"), ("hEs", "C4"), ("zmE", "C8")]:
        if k in tab: ax.plot(t, tab[k], ".-", color=c, label=k, ms=4)
    ax.set_ylabel("км"); ax.legend(ncol=7, fontsize=8); ax.set_title("высоты: h' (действующие) и hm (истинные, NHPC)")
    ax = axes[2]
    for k, c in [("MUF3000F2", "C3"), ("M3000F2", "C0")]:
        if k in tab: ax.plot(t, tab[k], ".-", color=c, label=k, ms=4)
    ax.set_ylabel("МГц / б/р"); ax.legend(fontsize=8); ax.set_title("MUF(3000)F2 и M(3000)F2")
    ax = axes[3]
    for k, c in [("TEC", "C3"), ("B0", "C0"), ("yF2", "C2")]:
        if k in tab: ax.plot(t, tab[k], ".-", color=c, label=k, ms=4)
    ax.set_ylabel("TECU / км"); ax.legend(fontsize=8); ax.set_title("ионосферный TEC (из профиля NHPC), B0 (IRI), yF2")
    ax.set_xlabel("UT, 2022-01-01")
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_isodensity(saos: list, station: str, fname: str, hmax=1000):
    """Суточная карта плазменной частоты fp(h, t) из профилей NHPC в SAO."""
    times, prof = [], []
    hgrid = np.arange(90, hmax + 1, 5.0)
    for t, s in saos:
        if len(s.get("profile_h", [])) < 3:
            continue
        h, fp = np.asarray(s["profile_h"]), np.asarray(s["profile_fp"])
        o = np.argsort(h)
        times.append(t); prof.append(np.interp(hgrid, h[o], fp[o], left=np.nan, right=np.nan))
    if not times:
        return
    P = np.array(prof).T
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    pc = ax.pcolormesh(pd.to_datetime(times), hgrid, P, cmap="viridis", shading="nearest")
    cs = ax.contour(pd.to_datetime(times), hgrid, P, levels=np.arange(1, 15), colors="w", linewidths=0.5)
    ax.clabel(cs, fmt="%d", fontsize=7)
    fig.colorbar(pc, ax=ax, label="плазменная частота fp, МГц")
    ax.set_ylabel("истинная высота, км"); ax.set_xlabel("UT, 2022-01-01")
    ax.set_title(f"{station}: профиль Ne(h,t) из SAO (NHPC), изолинии fp — аналог Digisonde Isodensity plot")
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_profiles(fname):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (d, st) in zip(axes, [(RSF_DIR, "JI91J"), (SBF_DIR, "RO041")]):
        edp_f = sorted(glob.glob(str(d / "scaled" / "*000000.EDP")))[0]
        sao_f = edp_f.replace(".EDP", ".SAO")
        head, edp = dfm.read_edp(edp_f)
        sao = dfm.read_sao(sao_f)
        ok = edp.dropna(subset=["ne_m3"])
        ax.errorbar(ok.ne_m3, ok.height_km, xerr=ok.ne_conf, fmt="-", color="C0", ecolor="C0", alpha=0.6, label="EDP: Ne ± σ (qualscan)")
        if len(sao.get("profile_ne", [])):
            ax.plot(np.asarray(sao["profile_ne"]) * 1e6, sao["profile_h"], "--", color="C3", label="SAO: профиль NHPC (ARTIST)")
        ax.axhline(head["hmf2"], color="k", ls=":", label=f"EDP hmF2={head['hmf2']:.0f} км, foF2={head['fof2']:.2f} МГц")
        ax.set_xscale("log"); ax.set_xlabel("Ne, м⁻³"); ax.set_ylabel("высота, км"); ax.legend(fontsize=8)
        ax.set_title(f"{st} 2022-01-01 00:00 UT — профиль электронной концентрации")
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_dft(fname):
    files = sorted(glob.glob(str(RSF_DIR / "drift" / "*.DFT")))[:1]
    hdr, amp, ph = dfm.read_dft(files[0])
    nb = amp.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax = axes[0]
    A = amp.reshape(nb * 16, 128)
    pc = ax.pcolormesh(np.arange(128) - 64, np.arange(nb * 16), A, cmap="inferno", shading="nearest")
    fig.colorbar(pc, ax=ax, label="амплитуда, дБ (3/8 дБ/ед.)")
    ax.set_xlabel("доплеровская линия (бин − 64)"); ax.set_ylabel("под-случай (антенна × высота × частота)")
    ax.set_title(f"DFT {os.path.basename(files[0])}: {nb} блоков × 16 спектров")
    ax = axes[1]
    for i in range(0, 16, 4):
        ax.plot(np.arange(128) - 64, amp[0, i], label=f"спектр {i}")
    ax.set_xlabel("доплеровская линия"); ax.set_ylabel("дБ"); ax.legend(fontsize=8); ax.set_title("отдельные доплеровские спектры (блок 0)")
    fig.suptitle(f"заголовок: {hdr}", fontsize=7)
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


def plot_day_ionogram_stack(d: Path, ext: str, station: str, fname: str, fmax: float):
    """Суточная развёртка: для каждой ионограммы — профиль «есть эхо O-моды на частоте f» → карта f×t (аналог directogram)."""
    files = sorted(glob.glob(str(d / "ionogram" / f"*.{ext}")))
    ts, mats, fgrid = [], [], None
    for f in files:
        pf, df = dfm.read_ionogram(f)
        e = df[(df.pol == "O") & echo_mask(df)]
        # минимальная высота эхо на каждой частоте (нижняя огибающая трассы)
        g = e.groupby("freq_mhz").height_km.min()
        if fgrid is None:
            fgrid = np.sort(df.freq_mhz.unique())
        ts.append(pf.date); mats.append(g.reindex(fgrid).values)
    M = np.array(mats).T
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    pc = ax.pcolormesh(pd.to_datetime(ts), fgrid, M, cmap="turbo", shading="nearest")
    fig.colorbar(pc, ax=ax, label="h' нижней кромки O-эхо, км")
    ax.set_ylabel("частота, МГц"); ax.set_xlabel("UT"); ax.set_ylim(fgrid.min(), fmax)
    ax.set_title(f"{station}: сутки {ext}-ионограмм ({len(files)} файлов) — минимальная высота O-эха по частоте")
    fig.savefig(OUT / fname, dpi=130); plt.close(fig)


if __name__ == "__main__":
    # --- одиночные ионограммы 00:00 UT
    pf, rsf = dfm.read_ionogram(RSF_DIR / "ionogram" / "JI91J_2022001000000.RSF")
    sao_ji = dfm.read_sao(RSF_DIR / "scaled" / "JI91J_2022001000000.SAO")
    plot_ionogram(rsf, pf, sao_ji, f"RSF — Jicamarca (JI91J) {pf.date} UT: {pf.f_start_hz/1e6:g}–{pf.f_stop_hz/1e6:g} МГц шаг {pf.f_coarse_step_hz/1e3:g} кГц, {pf.n_heights} высот × {pf.range_inc_km} км, {pf.n_doppler_lines} доплер. линий", "rsf_ionogram_JI91J_000000.png", 12, 1000)
    plot_rsf_directions(rsf, "RSF JI91J 2022-01-01 00:00 UT — направления, фаза, доплер", "rsf_directions_JI91J_000000.png")

    pf2, sbf = dfm.read_ionogram(SBF_DIR / "ionogram" / "RO041_2022001000000.SBF")
    sao_ro = dfm.read_sao(SBF_DIR / "scaled" / "RO041_2022001000000.SAO")
    plot_ionogram(sbf, pf2, sao_ro, f"SBF — Rome (RO041) {pf2.date} UT: {pf2.f_start_hz/1e6:g}–{pf2.f_stop_hz/1e6:g} МГц шаг {pf2.f_coarse_step_hz/1e3:g} кГц, {pf2.n_heights} высот × {pf2.range_inc_km} км, {pf2.n_doppler_lines} доплер. линий", "sbf_ionogram_RO041_000000.png", 15, 700)

    # ещё одна дневная ионограмма для контраста
    pf3, rsf3 = dfm.read_ionogram(RSF_DIR / "ionogram" / "JI91J_2022001180000.RSF")
    plot_ionogram(rsf3, pf3, dfm.read_sao(RSF_DIR / "scaled" / "JI91J_2022001180000.SAO"), f"RSF — Jicamarca (JI91J) {pf3.date} UT (день, 13 LT)", "rsf_ionogram_JI91J_180000.png", 12, 1000)
    pf4, sbf4 = dfm.read_ionogram(SBF_DIR / "ionogram" / "RO041_2022001120000.SBF")
    plot_ionogram(sbf4, pf4, dfm.read_sao(SBF_DIR / "scaled" / "RO041_2022001120000.SAO"), f"SBF — Rome (RO041) {pf4.date} UT (день, 13 LT)", "sbf_ionogram_RO041_120000.png", 15, 700)

    # --- суточные ряды SAO
    tab_ji, saos_ji = load_sao_dir(RSF_DIR); tab_ji.to_csv(OUT / "sao_scaled_JI91J.csv", index=False)
    tab_ro, saos_ro = load_sao_dir(SBF_DIR); tab_ro.to_csv(OUT / "sao_scaled_RO041.csv", index=False)
    plot_sao_daily(tab_ji, "Jicamarca JI91J, 2022-01-01", "sao_daily_JI91J.png")
    plot_sao_daily(tab_ro, "Rome RO041, 2022-01-01", "sao_daily_RO041.png")
    plot_isodensity(saos_ji, "JI91J", "sao_isodensity_JI91J.png", 1000)
    plot_isodensity(saos_ro, "RO041", "sao_isodensity_RO041.png", 600)
    plot_profiles("edp_vs_sao_profiles.png")
    plot_dft("dft_spectra_JI91J.png")
    plot_day_ionogram_stack(RSF_DIR, "RSF", "JI91J", "rsf_day_stack_JI91J.png", 12)
    plot_day_ionogram_stack(SBF_DIR, "SBF", "RO041", "sbf_day_stack_RO041.png", 15)
    print(tab_ji[["datetime", "foF2", "hmF2", "foE", "foEs", "MUF3000F2", "TEC", "n_profile"]].describe().T)
    print(tab_ro[["datetime", "foF2", "hmF2", "foE", "foEs", "MUF3000F2", "TEC", "n_profile"]].describe().T)
    print("done")
