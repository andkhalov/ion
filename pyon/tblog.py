# -*- coding: utf-8 -*-
"""
tblog.py — логирование эксперимента в TensorBoard (CLAUDE.md §2.4; Э3 §3) + metrics.csv.

Один ран = один каталог `runs/<этап>/<ран>/` (events TensorBoard, metrics.csv, weights.pt,
config.json). Логируем: скаляры (все метрики Э3 §3, покомпонентный L_logic), панели
ионограмм `вход | цель | предсказание` на ФИКСИРОВАННОМ наборе (Э3 §3.5-а), суточные треки
foF2(t)/МПЧ(t) «модель vs ARTIST/метки» (Э3 §3.5-б), гистограммы (распределение L_logic,
инвариант Пономарчука). Сервер: `tensorboard --logdir runs --port 13133 --bind_all`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import ListedColormap        # noqa: E402
from torch.utils.tensorboard import SummaryWriter   # noqa: E402

MASK_COLORS = ["black", "cyan", "lime", "yellow", "magenta", "red"]   # BG F2 F1 E Es MH
CMAP_MASK = ListedColormap(MASK_COLORS)


class TBLog:
    """Тонкая обёртка SummaryWriter: скаляры + строка metrics.csv на эпоху, фигуры, гистограммы."""

    def __init__(self, logdir: str | Path, config: dict | None = None):
        self.dir = Path(logdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.w = SummaryWriter(str(self.dir))
        self.rows: list[dict] = []
        if config is not None:
            (self.dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1, default=str),
                                                  encoding="utf-8")
            self.w.add_text("config", "```\n" + json.dumps(config, ensure_ascii=False, indent=1, default=str) + "\n```")

    # ---------------------------------------------------------------- скаляры / csv
    def scalars(self, values: dict, step: int, prefix: str = ""):
        for k, v in values.items():
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            self.w.add_scalar(prefix + k, float(v), step)

    def row(self, values: dict, step: int):
        """Строка metrics.csv (перезаписывается целиком — файл мал)."""
        r = {"step": step}
        r.update({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in values.items()})
        self.rows.append(r)
        pd.DataFrame(self.rows).to_csv(self.dir / "metrics.csv", index=False)

    # ---------------------------------------------------------------- панели ионограмм
    def ionograms(self, tag: str, x, y, pm, step: int, extent=(1, 15, 80, 720), titles=None,
                  n_classes: int = 5, xlabel="МГц", ylabel="км", save=None):
        """K строк × 3 панели: вход (канал O) | цель | предсказание. x: [K,2,H,W] 0..1 или uint8;
        y, pm: [K,H,W] int. extent — физические оси (ВЗ: 1–15 МГц × 80–720 км). save — PNG-копия."""
        x = np.asarray(x, dtype=np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        y, pm = np.asarray(y), np.asarray(pm)
        K = len(x)
        fig, axes = plt.subplots(K, 3, figsize=(10.5, 2.6 * K), squeeze=False, constrained_layout=True)
        for r in range(K):
            panels = [(x[r, 0], "inferno", dict(vmin=0, vmax=1)),
                      (y[r], CMAP_MASK, dict(vmin=0, vmax=len(MASK_COLORS) - 1)),
                      (pm[r], CMAP_MASK, dict(vmin=0, vmax=len(MASK_COLORS) - 1))]
            for c, (img, cm, kw) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(img, origin="lower", cmap=cm, aspect="auto", extent=extent, **kw)
                if r == 0:
                    ax.set_title(["вход (O)", "цель", "предсказание"][c], fontsize=9)
                if c == 0:
                    ax.set_ylabel((titles[r] if titles else "") + f"\n{ylabel}", fontsize=7)
                if r == K - 1:
                    ax.set_xlabel(xlabel, fontsize=8)
                ax.tick_params(labelsize=7)
        if save is not None:
            fig.savefig(save, dpi=80)
        self.w.add_figure(tag, fig, step)
        plt.close(fig)

    # ---------------------------------------------------------------- суточные треки
    def track(self, tag: str, hours, series: dict, step: int, ylabel: str = "foF2, МГц", title: str = ""):
        """Суточный ход: hours — часы UT; series — {имя: значения} (ARTIST/метки — точками,
        модели — линиями). NaN пропускаются."""
        fig, ax = plt.subplots(figsize=(7.5, 3), constrained_layout=True)
        hours = np.asarray(hours, float)
        for name, vals in series.items():
            vals = np.asarray(vals, float)
            ok = np.isfinite(vals)
            if name.lower().startswith(("artist", "метк", "label")):
                ax.plot(hours[ok], vals[ok], "o", ms=3, color="black", label=name, zorder=3)
            else:
                ax.plot(hours[ok], vals[ok], "-", lw=1.4, label=name)
        ax.set(xlabel="UT, ч", ylabel=ylabel, title=title, xlim=(0, 24))
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        self.w.add_figure(tag, fig, step)
        plt.close(fig)

    # ---------------------------------------------------------------- панель «как у дигизонда»
    def digisonde(self, tag: str, samples: list, step: int, extent=(1, 15, 80, 720), save: str | Path | None = None):
        """Панели в стиле Ion2PNG (Э1 §I.8 L4; DIARY 2026-09-05): для каждого образца — эхо (O),
        следы ARTIST (чёрные точки), контуры нашей маски, профиль fp(h): ARTIST чёрной линией,
        наш — красной; слева таблица характеристик «ARTIST | наша | Δ».
        samples: [dict(x=[2,H,W], y=[H,W], pm=[H,W], p_true=[H], p_pred=[H], p_valid=[H] bool,
                      table=[(имя, artist, ours)], title=str)]."""
        import matplotlib.patheffects as pe
        K = len(samples)
        fig, axes = plt.subplots(K, 2, figsize=(11, 3.4 * K), squeeze=False, constrained_layout=True,
                                 gridspec_kw=dict(width_ratios=[1.0, 3.2]))
        f0, f1, h0, h1 = extent
        for r, s_ in enumerate(samples):
            x = np.asarray(s_["x"], np.float32); x = x / 255.0 if x.max() > 1.5 else x
            y, pm = np.asarray(s_["y"]), np.asarray(s_["pm"])
            H, W = y.shape
            fx = np.linspace(f0, f1, W); hy = np.linspace(h0, h1, H)
            axt, ax = axes[r]
            axt.axis("off")
            lines = [f"{'':7s}{'ARTIST':>8s}{'наша':>8s}{'Δ':>7s}"]
            for name, a, b in s_["table"]:
                fa = "  N/A " if not np.isfinite(a) else f"{a:6.2f}"
                fb = "  N/A " if not np.isfinite(b) else f"{b:6.2f}"
                fd = "" if not (np.isfinite(a) and np.isfinite(b)) else f"{b - a:+6.2f}"
                lines.append(f"{name:7s}{fa:>8s}{fb:>8s}{fd:>7s}")
            axt.text(0, 1, "\n".join(lines), family="monospace", fontsize=7.5, va="top", ha="left",
                     transform=axt.transAxes)
            ax.imshow(x[0], origin="lower", cmap="Greys", aspect="auto", extent=extent, vmin=0, vmax=1, alpha=0.9)
            for ci in range(1, len(MASK_COLORS) - 1):
                if (y == ci).any():
                    rr, cc = np.nonzero(y == ci)
                    ax.plot(fx[cc], hy[rr], ".", ms=1.5, color="black", alpha=0.6)
                if (pm == ci).any():
                    ax.contour((pm == ci).astype(float), levels=[0.5], colors=[MASK_COLORS[ci]], linewidths=1.2,
                               extent=extent, origin="lower")
            pt, pp = np.asarray(s_["p_true"], float), np.asarray(s_["p_pred"], float)
            okt = np.isfinite(pt)
            if okt.any():
                ax.plot(pt[okt], hy[okt], "-", color="black", lw=2.0, label="профиль ARTIST (NHPC)")
            pv = np.asarray(s_.get("p_valid", np.isfinite(pp)), bool)
            if pv.any():
                ax.plot(pp[pv], hy[pv], "-", color="red", lw=1.6, label="профиль наш",
                        path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
            ax.set(xlim=(f0, f1), ylim=(h0, h1), xlabel="МГц", ylabel="км", title=s_.get("title", ""))
            ax.tick_params(labelsize=7); ax.title.set_fontsize(8); ax.grid(alpha=.25)
            if r == 0:
                ax.legend(fontsize=7, loc="upper right")
        if save:
            fig.savefig(save, dpi=110)
        self.w.add_figure(tag, fig, step)
        plt.close(fig)

    def tracks_grid(self, tag: str, hours, chars: dict, step: int, title: str = "", save: str | Path | None = None):
        """Суточные треки нескольких характеристик + разброс (как figures/baseline_vs_artist_*.png):
        chars = {имя: (artist[], ours[], единица)}; верх — ход по UT, низ — ours vs ARTIST c RMSE/bias."""
        names = list(chars)
        n = len(names)
        fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.2), squeeze=False, constrained_layout=True)
        hours = np.asarray(hours, float)
        for j, name in enumerate(names):
            a, b, unit = chars[name]
            a, b = np.asarray(a, float), np.asarray(b, float)
            ax = axes[0, j]
            ax.plot(hours, a, "o-", ms=2.5, lw=1, color="tab:blue", label="ARTIST (SAO)")
            ax.plot(hours, b, "x--", ms=3, lw=1, color="tab:orange", label="наша модель")
            ax.set(title=f"{name}", ylabel=unit, xlabel="UT, ч", xlim=(0, 24)); ax.grid(alpha=.3)
            ax.tick_params(labelsize=7); ax.title.set_fontsize(9)
            if j == 0:
                ax.legend(fontsize=7)
            ax = axes[1, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 2:
                ax.scatter(a[ok], b[ok], s=10)
                lim = [min(a[ok].min(), b[ok].min()), max(a[ok].max(), b[ok].max())]
                ax.plot(lim, lim, "k:", lw=1)
                rmse = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))); bias = float(np.mean(b[ok] - a[ok]))
                ax.set_title(f"n={int(ok.sum())} RMSE={rmse:.2f} {unit} bias={bias:+.2f}\n"
                             f"ARTIST есть в {int(np.isfinite(a).sum())}, наша в {int(np.isfinite(b).sum())} из {len(a)}", fontsize=7.5)
            ax.set(xlabel=f"ARTIST {name}", ylabel=f"наша {name}"); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        fig.suptitle(title, fontsize=10)
        if save:
            fig.savefig(save, dpi=110)
        self.w.add_figure(tag, fig, step)
        plt.close(fig)

    # ---------------------------------------------------------------- рендерер: реальное vs реализации
    def renders(self, tag: str, x, y, s1, s2, step: int, extent=(1, 15, 80, 720), titles=None,
                save: str | Path | None = None):
        """K строк × 4: реальное (O) | маска | рендер 1 (O) | рендер 2 (O) — CLAUDE §2.4 «реальное vs 2 реализации»."""
        x = np.asarray(x, np.float32); x = x / 255.0 if x.max() > 1.5 else x
        K = len(x)
        fig, axes = plt.subplots(K, 4, figsize=(13, 2.6 * K), squeeze=False, constrained_layout=True)
        for r in range(K):
            panels = [(x[r, 0], "inferno", dict(vmin=0, vmax=1)), (np.asarray(y)[r], CMAP_MASK, dict(vmin=0, vmax=len(MASK_COLORS) - 1)),
                      (np.asarray(s1)[r, 0], "inferno", dict(vmin=0, vmax=1)), (np.asarray(s2)[r, 0], "inferno", dict(vmin=0, vmax=1))]
            for c, (img, cm, kw) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(img, origin="lower", cmap=cm, aspect="auto", extent=extent, **kw)
                if r == 0:
                    ax.set_title(["реальное (O)", "маска ARTIST", "рендер 1", "рендер 2"][c], fontsize=9)
                if c == 0:
                    ax.set_ylabel((titles[r] if titles else "") + "\nкм", fontsize=7)
                ax.tick_params(labelsize=7)
        if save:
            fig.savefig(save, dpi=100)
        self.w.add_figure(tag, fig, step)
        plt.close(fig)

    # ---------------------------------------------------------------- гистограммы
    def hist(self, tag: str, values, step: int):
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        if len(v):
            self.w.add_histogram(tag, v, step)

    def text(self, tag: str, s: str, step: int = 0):
        self.w.add_text(tag, s, step)

    def close(self):
        self.w.flush()
        self.w.close()
