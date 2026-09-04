# -*- coding: utf-8 -*-
"""
dataset_cache.py — smoke-сборка фиксированного ВЗ-набора из корпуса (Э3 §2.2: ТОЛЬКО для
малых фиксированных наборов; в full-обучении корпус подаётся потоковым лоадером).

Каждая пара (RSF|SBF, SAO) → тензоры решётки `pyon.canon` (128×128, 1–15 МГц × 80–720 км):
X uint8 [2, NH, NF] через `digi_formats.read_canon` (тот же декодер, что в лоадере — набор
идентичен потоковым образцам) и маска Y int8 через `canon.masks_from_sao`. Метаданные — CSV
(станция, время, foF2/hF/foE/foEs/fxI ARTIST, C-level).

Запуск:  python -m pyon.dataset_cache [--limit N] [--procs 8]   (--limit → *_smoke)
Выход:   data/corpus_cache/shard_XXXX.npz + meta.csv
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
from pyon import canon                              # noqa: E402
from pyon import digi_formats as dfm                # noqa: E402
from pyon.manifest import collect_files, stem_time  # noqa: E402

SHARD = 2000


def process_one(f: str):
    try:
        sao_f = f.replace("/ionogram/", "/scaled/").rsplit(".", 1)[0] + ".SAO"
        if not os.path.exists(sao_f):
            return None
        st, t = stem_time(f)
        sao = dfm.read_sao(sao_f)
        sc = sao["scaled"]
        a = sao.get("analysis_flags", [])
        clevel = int(a[9]) if len(a) >= 10 else -1
        return (dfm.read_canon(f), canon.masks_from_sao(sao),
                dict(stem=os.path.basename(f), station=st,
                     time=str(t), foF2=float(sc.get("foF2", np.nan)),
                     hF=float(sc.get("hF", np.nan)), foE=float(sc.get("foE", np.nan)),
                     foEs=float(sc.get("foEs", np.nan)), fxI=float(sc.get("fxI", np.nan)),
                     c_level=clevel))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    # ЗАЩИТА: прогон с --limit (dry/smoke) пишет в отдельную директорию,
    # чтобы не затирать шарды и meta.csv полного кэша
    out_dir = ROOT / ("data/corpus_cache_smoke" if args.limit else "data/corpus_cache")
    out_dir.mkdir(exist_ok=True)
    files = collect_files()
    if args.limit:
        files = files[:args.limit]
    print(f"файлов: {len(files)}; процессов: {args.procs}", flush=True)
    t0 = time.time(); metas = []; xs, ys, shard = [], [], 0
    with Pool(args.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(process_one, files, chunksize=16)):
            if r is not None:
                x, y, m = r; xs.append(x); ys.append(y); m["shard"] = shard; m["idx"] = len(xs) - 1
                metas.append(m)
            if len(xs) >= SHARD:
                np.savez_compressed(out_dir / f"shard_{shard:04d}.npz",
                                    X=np.stack(xs), Y=np.stack(ys))
                xs, ys, shard = [], [], shard + 1
            if (k + 1) % 2000 == 0:
                print(f"  {k+1}/{len(files)} за {time.time()-t0:.0f} с "
                      f"({(k+1)/(time.time()-t0):.0f} ф/с), годных {len(metas)}", flush=True)
    if xs:
        np.savez_compressed(out_dir / f"shard_{shard:04d}.npz", X=np.stack(xs), Y=np.stack(ys))
    pd.DataFrame(metas).to_csv(out_dir / "meta.csv", index=False)
    print(f"ГОТОВО: {len(metas)} пар, {shard+1} шардов, {time.time()-t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
