# -*- coding: utf-8 -*-
"""
loader.py — потоковая подача данных в обучение (Э3 §2.2).

Принцип: НИКАКИХ полнокорпусных ДИСКОВЫХ тензорных кэшей — они не масштабируются на целевой
корпус (100–300 тыс. ионограмм). Декодирование сырья происходит НА ЛЕТУ в воркерах
torch DataLoader по описи data/manifest.csv (см. pyon.manifest); префетч воркеров
перекрывает шаг GPU. Требование Э3: суммарная скорость воркеров ≥ 2× скорости шага
обучения — мерить в dry-прогоне (throughput()).
IO-предел (замер 2026-09-05: корпус 33+ ГБ на диске QEMU не влезает в page cache; случайное
чтение 200–450 обр/с против 2600 обр/с у GPU) закрывается по Э3 §2.2 ОГРАНИЧЕННЫМ RAM-кэшем
декодированных образцов в `pyon.training` (`--cache_gb`; заполняется в нулевой эпохе через
воркеры, далее батчи из памяти; дисковых артефактов нет) и дискодружественным порядком
чтения `BlockShuffleSampler` для нулевой эпохи.

Датасеты (тензоры отдаются КОМПАКТНЫМИ — uint8/int8: через IPC воркер→главный процесс
идёт в 5 раз меньше байт, чем float32/int64; в тренировочном цикле на GPU:
`x = x.to(dev).float().div_(255)`, `y = y.to(dev).long()` — замер E0 2026-09-04, 4 CPU):
  VerticalDataset  — (RSF|SBF)+SAO → X uint8 [2,128,128] (амплитуда над медианой+6 дБ,
                     0..24 дБ → 0..255), Y int8 [128,128] (BG/F2/F1/E/Es из полилиний ARTIST),
                     P float32 [128] — профиль NHPC fp(h) на h_axis (NaN вне нижней стороны).
  ObliqueDataset   — SAO → синтетическая НЗ-маска int8 [128,128] (классы OB_CLASSES)
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
from torch.utils.data import Dataset, Sampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm                              # noqa: E402
from pyon import oblique_synth as obs                             # noqa: E402
from pyon import canon                                            # noqa: E402


class VerticalDataset(Dataset):
    """ВЗ: декодирование пары (сырьё, SAO) на лету в воркере DataLoader."""

    def __init__(self, manifest: pd.DataFrame):
        self.paths = manifest["path"].to_numpy()
        self.saos = manifest["sao"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        """→ (X uint8 [2,NH,NF], Y int8 [NH,NF], P float32 [NH] — fp(h) профиля NHPC, NaN вне валидности)."""
        try:
            x = np.ascontiguousarray(dfm.read_canon(str(ROOT / self.paths[i])), dtype=np.uint8)
            sao = dfm.read_sao(str(ROOT / self.saos[i]))
            y = np.ascontiguousarray(canon.masks_from_sao(sao), dtype=np.int8)
            p = canon.profile_from_sao(sao)
        except Exception:
            # битый файл не должен ронять эпоху: отдаём пустой образец (фон),
            # доля таких — метрика санити E0 (логировать в TensorBoard)
            x = np.zeros((2, canon.NH, canon.NF), np.uint8)
            y = np.zeros((canon.NH, canon.NF), np.int8)
            p = np.full(canon.NH, np.nan, np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(p)


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
        return (torch.from_numpy(np.ascontiguousarray(y, dtype=np.int8)), torch.tensor(d),
                torch.tensor(float(muf_f2)), torch.tensor(float(muf_mh)))


class ObliqueShardDataset(Dataset):
    """Материализованный НЗ-датасет (`pyon.oblique_dataset`): шарды npz → (маска O, маска X, метки).
    Шард распаковывается целиком при первом обращении и держится в кэше воркера; вместе с
    `ShardSampler` (обход по шардам) каждый шард распаковывается ровно один раз за эпоху."""

    def __init__(self, root, split: str, cache: int = 1):
        self.files = sorted(Path(root).glob(f"{split}_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"нет шардов {split}_*.npz в {root}")
        self.sizes = []
        for f in self.files:
            with np.load(f) as z:
                self.sizes.append(int(len(z["idx"])))
        self.offsets = np.cumsum([0] + self.sizes)
        self.cache_size = max(1, cache)
        self._cache: dict = {}

    def __len__(self):
        return int(self.offsets[-1])

    def shard_of(self, i: int) -> int:
        return int(np.searchsorted(self.offsets, i, side="right") - 1)

    def _shard(self, k: int):
        if k not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            with np.load(self.files[k]) as z:
                self._cache[k] = (z["masks_o"], z["masks_x"], z["labels"], z["idx"])
        return self._cache[k]

    def __getitem__(self, i: int):
        k = self.shard_of(i)
        mo, mx, lab, _ = self._shard(k)
        j = int(i - self.offsets[k])
        return (torch.from_numpy(np.ascontiguousarray(mo[j])), torch.from_numpy(np.ascontiguousarray(mx[j])),
                torch.from_numpy(np.ascontiguousarray(lab[j])))


class ShardSampler(Sampler):
    """Порядок обхода шардового датасета: шарды в случайном порядке, внутри шарда — перемешивание
    (распаковка каждого шарда один раз за эпоху). `set_epoch(e)` меняет порядок."""

    def __init__(self, ds: ObliqueShardDataset, seed: int = 0, shuffle: bool = True):
        self.ds, self.seed, self.shuffle, self.epoch = ds, seed, shuffle, 0

    def set_epoch(self, e: int):
        self.epoch = e

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        order = rng.permutation(len(self.ds.files)) if self.shuffle else np.arange(len(self.ds.files))
        for k in order:
            idx = np.arange(self.ds.offsets[k], self.ds.offsets[k + 1])
            if self.shuffle:
                rng.shuffle(idx)
            yield from idx.tolist()


class WithIndex(Dataset):
    """Обёртка: __getitem__ возвращает (*item, idx) — для заполнения RAM-кэша в тренировочном цикле."""

    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i: int):
        return (*self.ds[i], i)


class BlockShuffleSampler(torch.utils.data.Sampler):
    """Перемешивание, дружелюбное к диску (корпус больше page cache, диск — HDD/QEMU):
    индексы делятся на блоки по `block` соседних строк манифеста (манифест отсортирован по
    станции и времени — соседние файлы лежат рядом на диске), блоки перемешиваются, внутри
    блока — тоже; `streams` блоков читаются вперемежку (round-robin), поэтому батч из 64
    образцов собирается из `streams` разных периодов/станций. Полностью случайный порядок
    = block >= n. Детерминирован по (seed, epoch): вызывать set_epoch(e)."""

    def __init__(self, n: int, block: int = 512, streams: int = 8, seed: int = 0):
        self.n, self.block, self.streams, self.seed, self.epoch = n, block, streams, seed, 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        blocks = [np.arange(i, min(i + self.block, self.n)) for i in range(0, self.n, self.block)]
        rng.shuffle(blocks)
        for b in blocks:
            rng.shuffle(b)
        out = []
        for k in range(0, len(blocks), self.streams):
            group = blocks[k:k + self.streams]
            L = max(len(b) for b in group)
            for j in range(L):
                for b in group:
                    if j < len(b):
                        out.append(int(b[j]))
        return iter(out)


def throughput(dl, n_batches: int = 20) -> float:
    """Образцов/с через DataLoader — dry-мера E0 (сравнивать с 2× скоростью шага GPU)."""
    it = iter(dl)
    next(it)                                   # прогрев воркеров вне замера
    t0, n = time.time(), 0
    for _ in range(n_batches):
        b = next(it)
        n += len(b[0])
    return n / (time.time() - t0)
