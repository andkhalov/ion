#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_pipeline.py — компактная схема предлагаемого порядка интерпретации ионограммы.

Четыре блока в строку и контур пересмотра. Реализован только последний блок
(слой знаний); остальные предлагаются и в настоящей работе не строились.

Запуск: local/venv/bin/python research/fig_pipeline.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.6})

EC = "#4a4a4a"
RED = "#b00020"
VIO = "#5b3f8c"

fig, ax = plt.subplots(figsize=(7.0, 1.42))
ax.set_xlim(0, 112)
ax.set_ylim(-1, 21)
ax.axis("off")


def box(x, w, text, fc, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, 8.0), w, 10.0,
                                boxstyle="round,pad=0.30,rounding_size=1.2",
                                fc=fc, ec=EC, lw=lw, zorder=2))
    ax.text(x + w / 2, 13.0, text, ha="center", va="center",
            fontsize=7.6, zorder=3, linespacing=1.3)


def arrow(p, q, color=EC, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=0.9, linestyle=ls, zorder=1,
                                 shrinkA=1.2, shrinkB=1.2))


box(0.5, 21.0, "Ионограмма\nи паспорт трассы", "#eef2f7")
box(25.0, 25.0, "Выделение треков,\nизмерение МНЧ и задержек", "#e9f3ea")
box(53.5, 26.0, "Гипотезы приписывания\nобозначений мод", "#e9f3ea")
box(83.0, 28.5, "Онтология, ризонер,\nформы S1–S6", "#f2edf8", lw=1.7)

arrow((21.5, 13.0), (25.0, 13.0))
arrow((50.0, 13.0), (53.5, 13.0))
arrow((79.5, 13.0), (83.0, 13.0))

# контур пересмотра: от слоя знаний обратно к порождению гипотез
ax.add_patch(FancyArrowPatch((84.5, 8.0), (84.5, 3.6), arrowstyle="-",
                             color=RED, lw=0.9, linestyle=(0, (3, 2)), zorder=1))
ax.add_patch(FancyArrowPatch((84.5, 3.6), (66.5, 3.6), arrowstyle="-",
                             color=RED, lw=0.9, linestyle=(0, (3, 2)), zorder=1))
arrow((66.5, 3.6), (66.5, 8.0), color=RED, ls=(0, (3, 2)))
ax.text(41.0, 3.6, "нарушения ограничений:\nпересмотр гипотез",
        fontsize=6.8, color=RED, ha="center", va="center", linespacing=1.25)
ax.text(97.2, 18.9, "реализовано в настоящей работе", fontsize=6.8, color=VIO,
        ha="center", va="bottom", style="italic")

fig.tight_layout(pad=0.10)
fig.savefig(FIG / "pipeline.pdf")
fig.savefig(FIG / "pipeline.png", dpi=220)
print("записано:", FIG / "pipeline.pdf")
