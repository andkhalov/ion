# -*- coding: utf-8 -*-
"""
oblique_dataset.py — материализованный датасет НЗ (E3, решение АХ 2026-09-06: «посчитать датасет сырых
данных для обучения НЗ-модели вместе с шумом»): для каждой пары манифеста и каждой дальности D ∈ D_SET
— маска O (классы OB_CLASSES), маска X (из O по fo² = fx(fx − fB), высоты с поправкой E3b по профилю
NHPC), метки (МПЧ по классам, МПЧ кратника, D, fB), метаданные (станция, время, сплит, возмущённость).
Шум НЕ материализуется: он кладётся при обучении GAN-рендерером на GPU (новая реализация на каждый
показ = бесконечная аугментация); для визуального контроля сохраняется галерея готовых образцов
(маска O | маска X | рендер O+X | рендер O) — `gallery/`.

Формат: data/oblique/<split>_<NNN>.npz (сжато; masks_o, masks_x int8 [n,128,128], labels float32 [n,K],
idx int32 → строка meta), data/oblique/meta.csv, data/oblique/labels.json (имена колонок).
Чирп-зонд (Тромсё) не разделяет поляризаций: вход НЗ-модели — суммарная мощность O + X в одном
канале (X-след смещён на fB·sec φ/2), второй канал нулевой — так же, как в `pyon.tromso`.

Запуск: python -m pyon.oblique_dataset --manifest data/manifest.csv --procs 4 [--limit 0] [--gallery 24 --renderer runs/E4/gan/weights.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import digi_formats as dfm                       # noqa: E402
from pyon import oblique_synth as obs                       # noqa: E402

OUT = ROOT / "data" / "oblique"
LABELS = ["muf_F2", "muf_F1", "muf_E", "muf_Es", "muf_MH", "muf_F2_x", "muf_MH_x", "D_km", "fB"]
SHARD = 4096


def _one(args):
    i, sao_path = args
    try:
        sao = dfm.read_sao(str(ROOT / sao_path))
    except Exception:
        return i, None
    out = []
    gc = sao.get("geophys_const")
    fb = float(np.atleast_1d(np.asarray(gc, float))[0]) if gc is not None and np.size(gc) else 1.3
    if not np.isfinite(fb) or fb <= 0:
        fb = 1.3
    for d in obs.D_SET:
        try:
            yo, lo = obs.oblique_masks_from_sao(sao, d, "O")
            yx, lx = obs.oblique_masks_from_sao(sao, d, "X")
        except Exception:
            return i, None
        lab = np.array([lo.get("muf_F2", np.nan), lo.get("muf_F1", np.nan), lo.get("muf_E", np.nan), lo.get("muf_Es", np.nan),
                        lo.get("muf_MH", np.nan), lx.get("muf_F2", np.nan), lx.get("muf_MH", np.nan), d, fb], np.float32)
        out.append((yo, yx, lab))
    return i, out


def build(manifest: str, procs: int, limit: int = 0):
    df = pd.read_csv(ROOT / manifest, low_memory=False)
    if limit:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, limit).round().astype(int))]
    df = df.reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    meta_rows, buf, shard_id, n_bad = [], [], {"train": 0, "val": 0}, 0
    bufs = {"train": [], "val": []}
    t0 = time.time()

    def flush(split):
        b = bufs[split]
        if not b:
            return
        np.savez_compressed(OUT / f"{split}_{shard_id[split]:03d}.npz",
                            masks_o=np.stack([x[0] for x in b]), masks_x=np.stack([x[1] for x in b]),
                            labels=np.stack([x[2] for x in b]), idx=np.array([x[3] for x in b], np.int32))
        shard_id[split] += 1; bufs[split] = []

    with Pool(procs) as pool:
        for k, (i, res) in enumerate(pool.imap(_one, [(i, r.sao) for i, r in df.iterrows()], chunksize=32)):
            if res is None:
                n_bad += 1; continue
            r = df.iloc[i]
            for yo, yx, lab in res:
                mi = len(meta_rows)
                meta_rows.append(dict(idx=mi, row=i, station=r.station, time=r.time, split=r.split, D_km=float(lab[7]),
                                      disturbed=int(r.get("disturbed", 0)), c_level=r.get("c_level", np.nan), foF2=r.get("foF2", np.nan)))
                bufs[r.split].append((yo, yx, lab, mi))
                if len(bufs[r.split]) >= SHARD:
                    flush(r.split)
            if (k + 1) % 5000 == 0:
                print(f"  {k + 1}/{len(df)} SAO, {len(meta_rows)} масок, {time.time() - t0:.0f} с", flush=True)
    for split in bufs:
        flush(split)
    meta = pd.DataFrame(meta_rows); meta.to_csv(OUT / "meta.csv", index=False)
    json.dump(dict(labels=LABELS, shard=SHARD, classes=obs.OB_CLASSES, grid=dict(NF=obs.NF, NP=obs.NP, FOB=[obs.FOB_MIN, obs.FOB_MAX], P=[obs.P_MIN, obs.P_MAX]),
                   D_SET=list(obs.D_SET), manifest=manifest, n=len(meta), n_bad_sao=n_bad, built=time.strftime("%Y-%m-%d %H:%M:%S %Z")),
              open(OUT / "labels.json", "w"), indent=1, ensure_ascii=False)
    print(f"[oblique_dataset] {len(meta)} масок из {len(df)} SAO ({n_bad} битых) за {time.time() - t0:.0f} с → {OUT} "
          f"({sum(f.stat().st_size for f in OUT.glob('*.npz')) / 2**30:.2f} ГБ, шардов train {shard_id['train']} / val {shard_id['val']})", flush=True)
    return meta


def load_shard(path: Path):
    z = np.load(path)
    return z["masks_o"], z["masks_x"], z["labels"], z["idx"]


def gallery(renderer: str, n: int = 24, seed: int = 0):
    """Галерея готовых образцов: маска O | маска X | рендер (O + X, как видит чирп-зонд) | рендер O."""
    import torch, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from pyon import renderer as rnd, tblog
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ren = rnd.load_renderer(ROOT / renderer, dev); mh2f2 = rnd.MH2F2.to(dev)
    meta = pd.read_csv(OUT / "meta.csv")
    rng = np.random.default_rng(seed)
    pick = meta[meta.split == "val"].sample(n, random_state=seed)
    shards = sorted(OUT.glob("val_*.npz"))
    cache = {}
    def get(mi):
        for sp in shards:
            if sp not in cache:
                cache[sp] = load_shard(sp)
            mo, mx, lab, idx = cache[sp]
            j = np.flatnonzero(idx == mi)
            if len(j):
                return mo[j[0]], mx[j[0]], lab[j[0]]
        raise KeyError(mi)
    (OUT / "gallery").mkdir(exist_ok=True)
    cmap = ListedColormap(tblog.MASK_COLORS)
    ext = (obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX)
    fig, ax = plt.subplots(n, 4, figsize=(14, 2.6 * n), squeeze=False)
    torch.manual_seed(seed)
    for r, (_, m) in enumerate(pick.iterrows()):
        mo, mx, lab = get(int(m.idx))
        xo, xx = render_ox(ren, mh2f2, torch.from_numpy(mo)[None].to(dev).long(), torch.from_numpy(mx)[None].to(dev).long())
        panels = [(mo, cmap, dict(vmin=0, vmax=len(tblog.MASK_COLORS) - 1)), (mx, cmap, dict(vmin=0, vmax=len(tblog.MASK_COLORS) - 1)),
                  ((xo + xx).clip(0, 1)[0, 0].cpu().numpy(), "inferno", dict(vmin=0, vmax=1)), (xo[0, 0].cpu().numpy(), "inferno", dict(vmin=0, vmax=1))]
        for c, (img, cm, kw) in enumerate(panels):
            a = ax[r, c]; a.imshow(img, origin="lower", cmap=cm, aspect="auto", extent=ext, interpolation="nearest", **kw)
            if r == 0:
                a.set_title(["маска O", "маска X (E3b)", "рендер O+X (вход НЗ-модели)", "рендер O"][c], fontsize=9)
            a.tick_params(labelsize=6)
        ax[r, 0].set_ylabel(f"{m.station} {str(m.time)[:16]}\nD={m.D_km:.0f} МПЧ {lab[0]:.1f}/{lab[4]:.1f}", fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "gallery" / f"gallery_{Path(renderer).parent.name}.png", dpi=80); plt.close(fig)
    print("галерея →", OUT / "gallery" / f"gallery_{Path(renderer).parent.name}.png")


def render_ox(ren, mh2f2, mo, mx):
    """Рендер входа НЗ-модели: O-канал рендера маски O (фон + O-следы) + X-следы (пиксели расширенной
    маски X из рендера маски X). Возвращает (x_o [B,2,H,W], x_add [B,2,H,W]) — второй суммируется с первым."""
    import torch, torch.nn.functional as Fn
    xo = ren.sample(mh2f2[mo])
    xx = ren.sample(mh2f2[mx])
    reg = Fn.max_pool2d((mx > 0).float().unsqueeze(1), 5, stride=1, padding=2)
    return xo, xx * reg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gallery", type=int, default=0, help="число образцов галереи (0 — не строить)")
    ap.add_argument("--renderer", default="runs/E4/gan/weights.pt")
    ap.add_argument("--no-build", action="store_true", help="только галерея")
    a = ap.parse_args()
    if not a.no_build:
        build(a.manifest, a.procs, a.limit)
    if a.gallery:
        gallery(a.renderer, a.gallery)


if __name__ == "__main__":
    main()
