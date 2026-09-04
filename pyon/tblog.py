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
                  n_classes: int = 5, xlabel="МГц", ylabel="км"):
        """K строк × 3 панели: вход (канал O) | цель | предсказание. x: [K,2,H,W] 0..1 или uint8;
        y, pm: [K,H,W] int. extent — физические оси (ВЗ: 1–15 МГц × 80–720 км)."""
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
