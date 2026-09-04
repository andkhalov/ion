# -*- coding: utf-8 -*-
"""
loader.py — потоковая подача данных в обучение (Э3 §2.2).

Принцип: НИКАКИХ полнокорпусных тензорных кэшей — они не масштабируются на целевой
корпус (100–300 тыс. ионограмм). Декодирование сырья происходит НА ЛЕТУ в воркерах
torch DataLoader по описи data/manifest.csv (см. pyon.manifest); префетч воркеров
перекрывает шаг GPU. Требование Э3: суммарная скорость воркеров ≥ 2× скорости шага
обучения — мерить в dry-прогоне (throughput()).

Датасеты:
  VerticalDataset  — (RSF|SBF)+SAO → X float32 [2,128,128] (амплитуда над медианой+6 дБ),
                     Y int64 [128,128] (BG/F2/F1/E/Es из полилиний ARTIST).
  ObliqueDataset   — SAO → синтетическая НЗ-маска int64 [128,128] (классы OB_CLASSES)
                     сферическим секансом; дальность D — случайная из d_set на каждый
                     показ (fixed_d — для валидации); component: "O" | "X".
                     «Сырьё» для НЗ порождает рендерер на GPU уже В батче — тут только маски.

Пример:
    df = pd.read_csv("data/manifest.csv")
    ds = VerticalDataset(df[df.split == "train"])
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True,
                                     num_workers=8, persistent_workers=True, prefetch_factor=4)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm                              # noqa: E402
from pyon import oblique_synth as obs                             # noqa: E402
from pyon.dataset_cache import masks_from_sao                     # noqa: E402


class VerticalDataset(Dataset):
    """ВЗ: декодирование пары (сырьё, SAO) на лету в воркере DataLoader."""

    def __init__(self, manifest: pd.DataFrame):
        self.paths = manifest["path"].to_numpy()
        self.saos = manifest["sao"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            x = dfm.read_canon(str(ROOT / self.paths[i])).astype(np.float32) / 255.0
            y = masks_from_sao(dfm.read_sao(str(ROOT / self.saos[i]))).astype(np.int64)
        except Exception:
            # битый файл не должен ронять эпоху: отдаём пустой образец (фон),
            # доля таких — метрика санити E0 (логировать в TensorBoard)
            x = np.zeros((2, 128, 128), np.float32)
            y = np.zeros((128, 128), np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


class ObliqueDataset(Dataset):
    """НЗ-синтетика: только SAO → маска; D случайна на показ (детерминизм — через fixed_d)."""

    def __init__(self, manifest: pd.DataFrame, component: str = "O",
                 d_set=obs.D_SET, fixed_d: float | None = None, seed: int = 0):
        self.saos = manifest["sao"].to_numpy()
        self.component = component
        self.d_set = tuple(d_set)
        self.fixed_d = fixed_d
        self.seed = seed

    def __len__(self):
        return len(self.saos)

    def __getitem__(self, i: int):
        if self.fixed_d is not None:
            d = float(self.fixed_d)
        else:  # воспроизводимо при том же seed и номере показа
            d = self.d_set[np.random.default_rng((self.seed, i)).integers(len(self.d_set))]
        try:
            sao = dfm.read_sao(str(ROOT / self.saos[i]))
            y, lab = obs.oblique_masks_from_sao(sao, d, self.component)
            muf_f2 = lab.get("muf_F2", np.nan)
            muf_mh = lab.get("muf_MH", np.nan)
        except Exception:
            y, muf_f2, muf_mh = np.zeros((obs.NP, obs.NF), np.int8), np.nan, np.nan
        return (torch.from_numpy(y.astype(np.int64)), torch.tensor(d),
                torch.tensor(float(muf_f2)), torch.tensor(float(muf_mh)))


def throughput(dl, n_batches: int = 20) -> float:
    """Образцов/с через DataLoader — dry-мера E0 (сравнивать с 2× скоростью шага GPU)."""
    it = iter(dl)
    next(it)                                   # прогрев воркеров вне замера
    t0, n = time.time(), 0
    for _ in range(n_batches):
        b = next(it)
        n += len(b[0])
    return n / (time.time() - t0)
