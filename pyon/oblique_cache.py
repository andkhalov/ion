# -*- coding: utf-8 -*-
"""
oblique_cache.py — smoke-сборка фиксированного синтетического НЗ-набора из SAO корпуса
(Э3 §2.2: только малые фиксированные наборы; в full-обучении НЗ-маски синтезируются на лету
в `loader.ObliqueDataset`).

На каждый SAO: маски классов OB_CLASSES (BG/F2/F1/E/Es/MH) на решётке НЗ
(2–24 МГц × 300–3200 км группового пути) для дальностей D_SET = (300, 800, 1500) км
+ точные аналитические МПЧ-метки каждой моды (родословная — след ARTIST).

Запуск:  python -m pyon.oblique_cache [--procs 8]
Выход:   data/oblique_cache/shard_XXXX.npz (Y: n×3×128×128 int8) + meta.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm            # noqa: E402
from pyon import oblique_synth as obs           # noqa: E402
from pyon.manifest import collect_files         # noqa: E402

SHARD = 2000


def process_one(f: str):
    try:
        sao_f = f.replace("/ionogram/", "/scaled/").rsplit(".", 1)[0] + ".SAO"
        if not os.path.exists(sao_f):
            return None
        sao = dfm.read_sao(sao_f)
        ys, meta = [], dict(stem=os.path.basename(f))
        for di, d in enumerate(obs.D_SET):
            y, lab = obs.oblique_masks_from_sao(sao, d)
            if not (y > 0).any():
                return None
            ys.append(y)
            for k, v in lab.items():
                if k != "D_km":
                    meta[f"{k}_D{int(d)}"] = round(float(v), 3) if np.isfinite(v) else np.nan
        return np.stack(ys), meta
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    # ЗАЩИТА: прогон с --limit (dry/smoke) пишет в отдельную директорию,
    # чтобы не затирать шарды и meta.csv полного кэша
    out_dir = ROOT / ("data/oblique_cache_smoke" if args.limit else "data/oblique_cache")
    out_dir.mkdir(exist_ok=True)
    files = collect_files()
    if args.limit:
        files = files[:args.limit]
    print(f"файлов: {len(files)}; процессов: {args.procs}", flush=True)
    t0 = time.time(); metas, ys, shard = [], [], 0
    with Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(process_one, files, chunksize=32)):
            if r is not None:
                y, m = r; ys.append(y); m["shard"] = shard; m["idx"] = len(ys) - 1
                metas.append(m)
            if len(ys) >= SHARD:
                np.savez_compressed(out_dir / f"shard_{shard:04d}.npz", Y=np.stack(ys))
                ys, shard = [], shard + 1
            if (k + 1) % 4000 == 0:
                print(f"  {k+1}/{len(files)} за {time.time()-t0:.0f} с, годных {len(metas)}", flush=True)
    if ys:
        np.savez_compressed(out_dir / f"shard_{shard:04d}.npz", Y=np.stack(ys))
    pd.DataFrame(metas).to_csv(out_dir / "meta.csv", index=False)
    print(f"ГОТОВО: {len(metas)} образцов, {shard+1} шардов, {time.time()-t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
