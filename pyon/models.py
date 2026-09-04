# -*- coding: utf-8 -*-
"""
models.py — нейросетевые модели эксперимента (Э1 §I.8 «Модели»; Э3 §2 E1/E5).

Перенос из `iono_study.ipynb` (ячейка 7, прототип I.8.1–I.8.3-финал) без изменений
архитектуры: компактный U-Net (3 уровня, base=16 → ~118k параметров при cin=2, cout=5)
на канонической решётке 128×128. Используется:
  - ВЗ-сегментатор (E1):  UNet(cin=2 [O, X], cout=5 [BG, F2, F1, E, Es], coords=False|True)
  - НЗ-сегментатор (E5):  UNet(cin=2, cout=6 [+MH])
  - рендерер шума (E4):   отдельный класс появится в pyon/renderer.py (Э2 §4.2)

`coords=True` — абляция «+coords»: два координатных канала (нормированные высота/частота)
конкатенируются ко входу (прототип: IoU растёт, foF2-ридаут — нет; Э1 §I.8.2).
"""
from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    """Два свёрточных слоя 3×3 с BatchNorm и ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU())

    def forward(self, x):
        return self.n(x)


class UNet(nn.Module):
    """Компактный U-Net: энкодер base→2base→4base, два апсемпла со skip-соединениями."""

    def __init__(self, cin: int, cout: int, base: int = 16, coords: bool = False):
        super().__init__()
        self.coords = coords
        cin += 2 if coords else 0
        self.d1, self.d2, self.d3 = Block(cin, base), Block(base, base * 2), Block(base * 2, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.b2 = Block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.b1 = Block(base * 2, base)
        self.head = nn.Conv2d(base, cout, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        if self.coords:
            B, _, H, W = x.shape
            hh = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            ww = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            x = torch.cat([x, hh, ww], 1)
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        y = self.b2(torch.cat([self.u2(c3), c2], 1))
        y = self.b1(torch.cat([self.u1(y), c1], 1))
        return self.head(y)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
