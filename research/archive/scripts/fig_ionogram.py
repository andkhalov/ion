#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_ionogram.py — схематическая ионограмма НЗ с двумя вариантами разметки.

Рисунок СХЕМАТИЧЕСКИЙ: треки построены параметрической моделью формы следа
(параболическая аппроксимация окрестности каустики, как в OIASA
[Ippolito et al., 2018]), а не по измеренным данным. Числовые значения МНЧ и
минимальных задержек совпадают со сценой A из research/onto/ground-A.ttl.

Смысл рисунка: панели (a) и (b) визуально идентичны и различаются только
приписыванием меток. Панель (b) физически недопустима и отсекается
SHACL-формами S1 и S2, тогда как чисто визуальный классификатор не имеет
признака, позволяющего их различить.

Запуск: local/venv/bin/python research/fig_ionogram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

# десятичный разделитель — запятая (§5 регламента)
COMMA = FuncFormatter(lambda v, _: ("%g" % v).replace(".", ","))

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def lower_branch(f_lo, f_mof, tau_min, tau_nose, k=4.0, n=260):
    """Нижний луч: задержка почти постоянна и резко растёт у каустики."""
    f = np.linspace(f_lo, f_mof, n)
    tau = tau_min + (tau_nose - tau_min) * ((f - f_lo) / (f_mof - f_lo)) ** k
    return f, tau


def upper_branch(f_mof, f_up, tau_nose, tau_max, m=0.55, n=200):
    """Верхний (педерсеновский) луч: круто уходит по задержке от каустики."""
    f = np.linspace(f_mof, f_up, n)
    tau = tau_nose + (tau_max - tau_nose) * ((f_mof - f) / (f_mof - f_up)) ** m
    return f, tau


def flat_trace(f_lo, f_mof, tau, n=120, slope=0.10):
    """След отражения от тонкого слоя Es: задержка почти не зависит от частоты."""
    f = np.linspace(f_lo, f_mof, n)
    t = tau + slope * ((f - f_lo) / (f_mof - f_lo)) ** 6
    return f, t


# --- геометрия сцены A (МНЧ и минимальные задержки — из ground-A.ttl) ---
TR = {
    "tr1": lower_branch(4.6, 10.1, 3.67, 4.02),          # МНЧ 10,1  τmin 3,67
    "tr2": upper_branch(10.1, 8.4, 4.10, 4.78),           # МНЧ 10,1  τmin 4,10
    "tr3": lower_branch(3.8, 7.2, 4.69, 5.05),           # МНЧ  7,2  τmin 4,69
    "tr4": flat_trace(2.3, 17.7, 3.34),                  # МНЧ 17,7  τmin 3,34
}
# точки постановки подписей: (x, y, выравнивание)
ANCHOR = {
    "tr1": (6.3, 3.62, "center", "bottom"),
    "tr2": (9.4, 4.62, "center", "center"),
    "tr3": (5.0, 4.64, "center", "bottom"),
    "tr4": (3.4, 3.29, "center", "bottom"),
}
LABEL_OK = {"tr1": "1F2", "tr2": "1F2п", "tr3": "2F2", "tr4": "1Es"}
LABEL_BAD = {"tr1": "2F2", "tr2": "1F2п", "tr3": "1F2", "tr4": "1Es"}


def draw(ax, labels, title, bad_keys=()):
    rng = np.random.default_rng(7)
    for key, (f, tau) in TR.items():
        # «размытие» следа: пучок элементарных лучей вокруг центральной кривой
        for _ in range(9):
            jitter = rng.normal(0.0, 0.028, size=tau.size)
            ax.plot(f, tau + jitter, lw=0.6, color="0.55", alpha=0.35, zorder=1)
        ax.plot(f, tau, lw=1.5, color="k", zorder=3)
        x, y, ha, va = ANCHOR[key]
        col = "#b00020" if key in bad_keys else "k"
        weight = "bold" if key in bad_keys else "normal"
        ax.text(x, y, labels[key], ha=ha, va=va, color=col, fontweight=weight,
                fontsize=9.5, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(1.8, 19)
    ax.set_ylim(2.95, 5.25)          # задержка растёт вверх (по замечанию А. О. Щирого)
    ax.set_xlabel("частота, МГц")
    ax.set_ylabel("групповая задержка, мс")
    ax.set_title(title, fontsize=9)
    ax.grid(True, lw=0.3, color="0.85")
    ax.xaxis.set_major_locator(MultipleLocator(2.5))
    ax.xaxis.set_major_formatter(COMMA)
    ax.yaxis.set_major_formatter(COMMA)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35), sharey=True)
draw(axes[0], LABEL_OK, "а) физически допустимая разметка")
draw(axes[1], LABEL_BAD, "б) обозначения 1F2 и 2F2 переставлены", bad_keys=("tr1", "tr3"))
for ax in axes:
    ax.plot([10.1], [4.06], marker="o", ms=3.2, mfc="white", mec="k", mew=0.9, zorder=5)
axes[0].annotate("МНЧ 1F2 = 10,1 МГц:\nлучи сходятся", xy=(10.1, 4.06), xytext=(14.2, 4.55),
                 fontsize=7.5, ha="center", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="0.35"))
axes[1].set_ylabel("")
fig.tight_layout(pad=0.6)
out = FIG / "ionogram-scene.pdf"
fig.savefig(out)
fig.savefig(FIG / "ionogram-scene.png", dpi=220)
print("записано:", out)
