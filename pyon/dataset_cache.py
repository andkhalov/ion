# -*- coding: utf-8 -*-
"""
dataset_cache.py — параллельная сборка обучающего кэша из корпуса NOAA.

Каждая пара (RSF|SBF, SAO) → канонические тензоры на решётке 128×128
(1–15 МГц × 80–720 км): X uint8 (2 канала O/X, амплитуда над шумом:
порог = медиана положительных амплитуд + 6 дБ — РОБАСТНО, т.к. MPA из
PREFACE у DPS-4D-станций (JR055/PQ052) в других единицах; 0..24 дБ → 0..255)
и маска классов Y int8 (0 BG, 1 F2, 2 F1, 3 E, 4 Es — растеризация SAO-полилиний
±1 бин). Метаданные — CSV (станция, время, foF2/hF ARTIST, C-level).

Запуск:  python -m pyon.dataset_cache [--limit N] [--procs 8]
Выход:   data/corpus_cache/shard_XXXX.npz + meta.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm  # noqa: E402

NF = NH = 128
F_MIN, F_MAX, H_MIN, H_MAX = 1.0, 15.0, 80.0, 720.0
CLASSES = ["BG", "F2", "F1", "E", "Es"]
TR_KEYS = {"F2": "F2o", "F1": "F1o", "E": "Eo", "Es": "Es"}
f_axis = np.linspace(F_MIN, F_MAX, NF)
h_axis = np.linspace(H_MIN, H_MAX, NH)
SHARD = 2000


def canon_matrix_u8(df) -> np.ndarray:
    out = np.zeros((2, NH, NF), np.uint8)
    for ci, pol in enumerate(["O", "X"]):
        s = df[df.pol == pol]
        if not len(s):
            continue
        fs = np.sort(s.freq_mhz.unique()); hs = np.sort(s.height_km.unique())
        a = s.amp_db.values
        noise = np.median(a[a > 0]) + 6 if (a > 0).any() else 0.0
        m = np.zeros((len(hs), len(fs)), np.float32)
        m[np.searchsorted(hs, s.height_km.values), np.searchsorted(fs, s.freq_mhz.values)] = \
            np.clip(a - noise, 0, 24)
        fj = np.clip(np.searchsorted(fs, f_axis), 0, len(fs) - 1)
        hj = np.clip(np.searchsorted(hs, h_axis), 0, len(hs) - 1)
        okf = (f_axis >= fs[0]) & (f_axis <= fs[-1]); okh = (h_axis >= hs[0]) & (h_axis <= hs[-1])
        out[ci][np.ix_(okh, okf)] = (m[np.ix_(hj[okh], fj[okf])] * 255 / 24).astype(np.uint8)
    return out


def masks_from_sao(sao) -> np.ndarray:
    y = np.zeros((NH, NF), np.int8)
    for cls, key in TR_KEYS.items():
        fq, vh = sao.get(f"{key}_freq"), sao.get(f"{key}_vh")
        if fq is None or not len(fq):
            continue
        ci = CLASSES.index(cls)
        for f0, h0 in zip(np.asarray(fq, float), np.asarray(vh, float)):
            if not (F_MIN <= f0 <= F_MAX and H_MIN <= h0 <= H_MAX):
                continue
            jf = int(round((f0 - F_MIN) / (F_MAX - F_MIN) * (NF - 1)))
            jh = int(round((h0 - H_MIN) / (H_MAX - H_MIN) * (NH - 1)))
            y[max(jh - 1, 0):jh + 2, jf] = ci
    return y


def process_one(f: str):
    try:
        sao_f = f.replace("/ionogram/", "/scaled/").rsplit(".", 1)[0] + ".SAO"
        if not os.path.exists(sao_f):
            return None
        pf, df = dfm.read_ionogram(f)
        sao = dfm.read_sao(sao_f)
        sc = sao["scaled"]
        a = sao.get("analysis_flags", [])
        clevel = int(a[9]) if len(a) >= 10 else -1
        return (canon_matrix_u8(df), masks_from_sao(sao),
                dict(stem=os.path.basename(f), station=os.path.basename(f)[:5],
                     time=str(pf.date), foF2=float(sc.get("foF2", np.nan)),
                     hF=float(sc.get("hF", np.nan)), foE=float(sc.get("foE", np.nan)),
                     foEs=float(sc.get("foEs", np.nan)), fxI=float(sc.get("fxI", np.nan)),
                     c_level=clevel))
    except Exception:
        return None


def collect_files() -> list[str]:
    pats = [str(ROOT / "data/corpus/*/*/*/ionogram/*.RSF"),
            str(ROOT / "data/corpus/*/*/*/ionogram/*.SBF"),
            str(ROOT / "data/RSF-samples-w-img-n-sao-n-dft/ionogram/*.RSF"),
            str(ROOT / "data/SBF-samples-w-img-n-sao/ionogram/*.SBF")]
    return sorted({f for p in pats for f in glob.glob(p)})


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
