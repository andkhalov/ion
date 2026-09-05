# -*- coding: utf-8 -*-
"""
models.py — нейросетевые модели эксперимента (Э1 §I.8 «Модели»; Э3 §2 E1/E5).

U-Net параметризован для предварительного исследования архитектуры (DIARY 2026-09-05):
глубина (число уровней энкодера), ширина (base), нормализация (batch | group | none),
dropout (после каждого блока), skip-соединения (вкл/выкл), координатные каналы.
**Значения по умолчанию воспроизводят прототип** `iono_study.ipynb` (ячейка 7) бит-в-бит по
структуре: depth=3, base=16, BatchNorm, без dropout, со skip — 117 605 параметров при
cin=2, cout=5. Используется:
  - ВЗ-сегментатор (E1):  UNet(cin=2 [O, X], cout=5 [BG, F2, F1, E, Es])
  - НЗ-сегментатор (E5):  UNet(cin=2, cout=6 [+MH])
  - рендерер шума (E4):   отдельный класс появится в pyon/renderer.py (Э2 §4.2)

Литература по архитектурам на ионограммах (lit/): NOIRE-Net (Kvammen et al. 2024) — VGG-подобная
CNN, BatchNorm после каждой свёртки, dropout только между полносвязными слоями, Adam
1e-3→1e-5, batch 64, 100 эпох, «гиперпараметры после нескольких проб, dropout не исследован»;
Castro et al. 2025 — ResNet-блоки + BN + dropout; Sherstyukov et al. 2024 — CNN-регрессия
параметров. Сегментационных U-Net на ионограммах в открытых работах с деталями — нет
(DIAS, Xiao 2020 — U-Net, статья не в lit/). Поэтому параметры подбираются на подмножестве.
"""
from __future__ import annotations

import torch
from torch import nn


def _norm(kind: str, ch: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"норма: {kind}")


class Block(nn.Module):
    """Два свёрточных слоя 3×3 с нормализацией и ReLU (+ dropout после блока при p > 0)."""

    def __init__(self, cin: int, cout: int, norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1), _norm(norm, cout), nn.ReLU(),
                  nn.Conv2d(cout, cout, 3, padding=1), _norm(norm, cout), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.n = nn.Sequential(*layers)

    def forward(self, x):
        return self.n(x)


class UNet(nn.Module):
    """Компактный U-Net: энкодер base·2^k на k = 0..depth−1, decoder с ConvTranspose и skip."""

    def __init__(self, cin: int, cout: int, base: int = 16, depth: int = 3, norm: str = "batch",
                 dropout: float = 0.0, skip: bool = True, coords: bool = False):
        super().__init__()
        assert depth >= 2
        self.coords, self.skip, self.depth = coords, skip, depth
        cin += 2 if coords else 0
        widths = [base * 2 ** k for k in range(depth)]
        self.down = nn.ModuleList([Block(cin if k == 0 else widths[k - 1], widths[k], norm, dropout)
                                   for k in range(depth)])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for k in range(depth - 1, 0, -1):            # widths[k] -> widths[k-1]
            self.up.append(nn.ConvTranspose2d(widths[k], widths[k - 1], 2, 2))
            self.dec.append(Block(widths[k - 1] * (2 if skip else 1), widths[k - 1], norm, dropout))
        self.head = nn.Conv2d(widths[0], cout, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.coords:
            B, _, H, W = x.shape
            hh = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            ww = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([x, hh, ww], 1)
        feats = []
        for k, blk in enumerate(self.down):
            x = blk(x if k == 0 else self.pool(x))
            feats.append(x)
        y = feats[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            s = feats[-2 - j]
            y = up(y)
            y = dec(torch.cat([y, s], 1) if self.skip else y)
        return self.head(y)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
