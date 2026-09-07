# -*- coding: utf-8 -*-
"""
showcase.py — показательные борды TensorBoard (запрос АХ 2026-09-07): вся цепочка на одной картинке.

Борд «chain» (5 колонок на пример):
  1. ВЗ, реальное сырьё дигизонда (канал O) + следы ARTIST точками — что видит ВЗ-модель;
  2. ВЗ, разметка ARTIST (классы F2/F1/E/Es) — цель ВЗ-модели;
  3. НЗ, аналитическая разметка (пересчёт следов секансом на сфере для дальности D, компоненты O и X,
     кратник MH) — цель НЗ-модели, «SAO, перенесённая на наклонную трассу»;
  4. НЗ, синтетический вход (та же разметка, «озвученная» GAN-рендерером) — что видит НЗ-модель;
  5. НЗ, предсказание НЗ-модели + измеренная МПЧ.
Борд «tromso»: реальный снимок Тромсё в нашей растеризации | предсказание | контуры поверх снимка,
подпись — МПЧ 1F2/2F2 и время UTC.

Запуск: python -m pyon.showcase --nz_weights runs/E5/baseline/weights.pt --renderer runs/E4/gan/weights.pt
        [--n 6 --d 430 --route sgo-tgo --out runs/showcase]
Артефакты: runs/<out>/ (события TensorBoard) + png/chain.png, png/tromso.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib                                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import ListedColormap                                 # noqa: E402

from pyon import canon, tblog, training as T                                 # noqa: E402
from pyon import digi_formats as dfm                                          # noqa: E402
from pyon import oblique_synth as obs                                         # noqa: E402
from pyon import oblique_train as OT                                          # noqa: E402
from pyon import renderer as rnd                                              # noqa: E402
from pyon import tromso as tg                                                 # noqa: E402
from pyon.models import UNet                                                  # noqa: E402

VS_EXT = (canon.F_MIN, canon.F_MAX, canon.H_MIN, canon.H_MAX)
OB_EXT = (obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX)


def load_nz(weights: str, dev):
    ck = torch.load(ROOT / weights, map_location=dev); c = ck["cfg"]
    net = UNet(2, len(obs.OB_CLASSES), base=c["base"], depth=c["depth"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, c


def chain_board(log: tblog.TBLog, rows: pd.DataFrame, nz_net, ren, dev, d_km: float, az: float, save: Path):
    cmap_vs = ListedColormap(tblog.MASK_COLORS[:len(canon.CLASSES)])
    cmap_ob = ListedColormap(tblog.MASK_COLORS[:len(obs.OB_CLASSES)])
    mh2f2 = rnd.MH2F2.to(dev)
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(21, 3.4 * n), squeeze=False)
    titles = ["1. ВЗ: реальное сырьё + следы ARTIST", "2. ВЗ: разметка ARTIST (цель ВЗ-модели)",
              f"3. НЗ: аналитическая разметка (D={d_km:.0f} км, азимут {az:.0f}°)",
              "4. НЗ: синтетический вход (рендер шума)", "5. НЗ: предсказание нашей модели"]
    for r, row in enumerate(rows.itertuples()):
        x = dfm.read_canon(str(ROOT / row.path)).astype(np.float32) / 255.0
        sao = dfm.read_sao(str(ROOT / row.sao))
        y_vs = canon.masks_from_sao(sao)
        yo, lab_o = obs.oblique_masks_from_sao(sao, d_km, "O")
        yx, _ = obs.oblique_masks_from_sao(sao, d_km, "X", az)
        y_ob = OT.compose_target(torch.from_numpy(yo)[None].long(), torch.from_numpy(yx)[None].long())[0].numpy()
        with torch.no_grad():
            torch.manual_seed(row.Index)
            xin = OT.render_input(ren, mh2f2, torch.from_numpy(yo)[None].to(dev).long(),
                                  torch.from_numpy(yx)[None].to(dev).long(), "ox", True, 0.0, (0.04, 0.10))
            pm = nz_net(xin).float().argmax(1)[0].cpu().numpy()
        f1, f2, pn = OT.muf_readouts(pm[None])
        ax[r, 0].imshow(x[0], origin="lower", extent=VS_EXT, aspect="auto", cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        for cls, key in (("F2", "F2o"), ("F1", "F1o"), ("E", "Eo"), ("Es", "Es")):
            fq, vh = sao.get(f"{key}_freq"), sao.get(f"{key}_vh")
            if fq is not None and len(fq):
                ax[r, 0].plot(fq, vh, ".", ms=1.6, color=tblog.MASK_COLORS[canon.CLASSES.index(cls)])
        ax[r, 1].imshow(y_vs, origin="lower", extent=VS_EXT, aspect="auto", cmap=cmap_vs, vmin=0, vmax=len(canon.CLASSES) - 1, interpolation="nearest")
        ax[r, 2].imshow(y_ob, origin="lower", extent=OB_EXT, aspect="auto", cmap=cmap_ob, vmin=0, vmax=len(obs.OB_CLASSES) - 1, interpolation="nearest")
        ax[r, 3].imshow(xin[0, 0].cpu().numpy(), origin="lower", extent=OB_EXT, aspect="auto", cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax[r, 4].imshow(pm, origin="lower", extent=OB_EXT, aspect="auto", cmap=cmap_ob, vmin=0, vmax=len(obs.OB_CLASSES) - 1, interpolation="nearest")
        ax[r, 0].set_ylabel(f"{row.station} {str(row.time)[:16]}\nfoF2 {row.foF2}  h′F {row.hF}", fontsize=7)
        ax[r, 2].set_xlabel(f"метка МПЧ {lab_o.get('muf_F2', np.nan):.2f} / кратник {lab_o.get('muf_MH', np.nan):.2f} МГц", fontsize=7)
        ax[r, 4].set_xlabel(f"наша МПЧ {f1[0]:.2f} / кратник {f2[0]:.2f} МГц", fontsize=7)
        for c in range(5):
            ax[r, c].tick_params(labelsize=6)
            if r == 0:
                ax[r, c].set_title(titles[c], fontsize=9)
    fig.suptitle("Цепочка: реальная ВЗ-ионограмма → разметка ARTIST → перенос на наклонную трассу → синтез шума → интерпретация НЗ-моделью", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(save, dpi=85); log.w.add_figure("chain/ВЗ→НЗ", fig, 0); plt.close(fig)
    print("→", save)


def tromso_board(log: tblog.TBLog, nz_net, dev, route: str, n: int, active: float, save: Path):
    cmap_ob = ListedColormap(tblog.MASK_COLORS[:len(obs.OB_CLASSES)])
    files = sorted((ROOT / "data" / "tromso" / route).glob("*.png"))
    if len(files) > n:
        files = [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]
    xs, cov, tt = [], [], []
    for fp in files:
        try:
            snr, f, r = tg.png_to_snr(fp, route)
            x, cv = tg.rasterize(snr, f, r, "zero", active, None)
            xs.append(x); cov.append(cv); tt.append(fp.stem)
        except Exception:
            continue
    X = torch.from_numpy(np.stack(xs)).float().div(255)
    pm, _ = OT.predict(nz_net, X, dev)
    pm = np.where(np.stack(cov), pm, 0)
    f1, f2, pn = OT.muf_readouts(pm)
    m = len(xs)
    fig, ax = plt.subplots(3, m, figsize=(3.4 * m, 10.2), squeeze=False)
    for k in range(m):
        ax[0, k].imshow(X[k, 0].numpy(), origin="lower", extent=OB_EXT, aspect="auto", cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
        ax[0, k].set_title(f"{tt[k][9:15]} UTC", fontsize=9)
        ax[1, k].imshow(pm[k], origin="lower", extent=OB_EXT, aspect="auto", cmap=cmap_ob, vmin=0, vmax=len(obs.OB_CLASSES) - 1, interpolation="nearest")
        ax[1, k].set_title(f"МПЧ {f1[k]:.2f} / {f2[k]:.2f} МГц" if np.isfinite(f1[k]) else "след не найден", fontsize=8)
        ax[2, k].imshow(X[k, 0].numpy(), origin="lower", extent=OB_EXT, aspect="auto", cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        for ci in range(1, len(obs.OB_CLASSES)):
            mask = pm[k] == ci
            if mask.any():
                ax[2, k].contour(np.linspace(*OB_EXT[:2], obs.NF), np.linspace(*OB_EXT[2:], obs.NP), mask.astype(float),
                                 levels=[0.5], colors=[tblog.MASK_COLORS[ci]], linewidths=0.9)
        for r_ in range(3):
            ax[r_, k].tick_params(labelsize=6)
    for r_, lab in enumerate(["реальный снимок Тромсё (наша растеризация)", "разметка нашей модели", "контуры поверх снимка"]):
        ax[r_, 0].set_ylabel(f"P′, км\n{lab}", fontsize=8)
    fig.suptitle(f"Реальные наклонные ионограммы Тромсё, трасса {route}: что видит модель и как размечает", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(save, dpi=85); log.w.add_figure(f"tromso/{route}", fig, 0); plt.close(fig)
    print("→", save)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nz_weights", default="runs/E5/baseline/weights.pt")
    ap.add_argument("--renderer", default="runs/E4/gan/weights.pt")
    ap.add_argument("--manifest", default="data/manifest_e1.csv")
    ap.add_argument("--out", default="runs/showcase")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--d", type=float, default=430.0)
    ap.add_argument("--az", type=float, default=315.0)
    ap.add_argument("--route", default="sgo-tgo")
    ap.add_argument("--tromso_n", type=int, default=6)
    ap.add_argument("--active", type=float, default=0.05)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rundir = ROOT / a.out; (rundir / "png").mkdir(parents=True, exist_ok=True)
    log = tblog.TBLog(rundir, dict(nz_weights=a.nz_weights, renderer=a.renderer, D_km=a.d, azimuth=a.az, route=a.route))
    log.readme(f"""## Показательные примеры системы (борды `chain` и `tromso`)
**chain** — вся цепочка на одной строке: реальное сырьё ВЗ с следами ARTIST → разметка ARTIST → её перенос на наклонную трассу
D = {a.d:.0f} км, азимут {a.az:.0f}° (сферический секанс, компоненты O и X, кратник) → синтетический вход НЗ-модели (GAN-рендерер `{a.renderer}`)
→ разметка НЗ-моделью `{a.nz_weights}` с измеренной МПЧ. Показывает, что цель НЗ-модели — та же разметка ARTIST, пересчитанная физикой.
**tromso** — РЕАЛЬНЫЕ наклонные ионограммы Тромсё (трасса {a.route}): наша растеризация снимка, разметка модели, контуры поверх снимка.
Меток для них не существует, поэтому это качественная проверка + метрики без меток (доля найденных следов, гладкость МПЧ во времени, гейт).""")
    net, c = load_nz(a.nz_weights, dev)
    ren = rnd.load_renderer(ROOT / a.renderer, dev)
    df = pd.read_csv(ROOT / a.manifest, low_memory=False)
    lset_p = ROOT / "runs" / "E1" / "logging_set.json"
    if lset_p.exists():
        lset = json.loads(lset_p.read_text(encoding="utf-8"))
        sel = df[df.path.isin([im["path"] for im in lset["images"]])]
        sel = sel[sel.foF2.notna()].groupby("station").head(2).head(a.n).reset_index(drop=True)
    else:
        sel = df[(df.split == "val") & df.foF2.notna()].groupby("station").head(2).head(a.n).reset_index(drop=True)
    chain_board(log, sel, net, ren, dev, a.d, a.az, rundir / "png" / "chain.png")
    tromso_board(log, net, dev, a.route, a.tromso_n, a.active, rundir / "png" / "tromso.png")
    log.close()


if __name__ == "__main__":
    main()
