# -*- coding: utf-8 -*-
"""
renderer.py — нейрорендерер шума: обратная задача «разметка → сырьё» (этап E4; Э2 §4; Э3 §2 E4, §3.4).

Модель (Э2 §4.2, перенос из iono_study.ipynb, ячейка 24): U-Net; вход — one-hot маска классов
(5 каналов ВЗ; класс MH наклонной маски подаётся в канал F2 — кратник выглядит как F2-эхо)
+ 2 координатных канала + 1 канал шума z ~ U(0,1); выход на каждую поляризацию (O, X): логит
вероятности активности пикселя p и амплитуда a. Лосс: Σ_pol [BCE(p, 1[X>0]) + |a − X|·1[X>0]];
сэмплирование X_синт = 1[u < p]·a, u ~ U(0,1) — источник спекл-текстуры (каждая реализация новая;
в обучении НЗ-модели работает как бесконечная аугментация, Tobin 2017).

Метрики качества синтетики (Э3 §3.4): KS-расстояние амплитудных распределений активных пикселей
(реальное vs рендер), L1-расстояние профилей плотности эха по частоте (RFI-полосы) и по высоте,
доля активных пикселей; главный судья — сим2реал-матрица 2×2 (`sim2real`): сегментатор,
обученный на рендере, читает реальное (цель ≥ 0.8 × контроля «реальное→реальное», Э3 §5 S3).
Прототип (Э2 §6.0): рендер→рендер IoU 0.83, рендер→реальное 0.17 против 0.37 контроля —
разрыв ОТКРЫТ; варианты закрытия (E4): адверсариальный член, трансплантация реального фона, FDA.

Запуск:  python -m pyon.renderer --stage E4 --run base --manifest data/manifest.csv --epochs 5
         python -m pyon.renderer --dry
Артефакты: runs/E4/<ран>/{events, metrics.csv, weights.pt, config.json, summary.json, png/}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from scipy.stats import ks_2samp
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, tblog, training                    # noqa: E402
from pyon import oblique_synth as obs                       # noqa: E402
from pyon.models import UNet, n_params                      # noqa: E402

N_CLASSES = len(canon.CLASSES)                              # 5: BG F2 F1 E Es
MH2F2 = torch.tensor([0, 1, 2, 3, 4, 1])                    # НЗ-классы (obs.OB_CLASSES) → ВЗ-каналы


class Renderer(nn.Module):
    """U-Net(n_classes + 3 → 4): [p_O, a_O, p_X, a_X]."""

    def __init__(self, base: int = 16, depth: int = 3, n_classes: int = N_CLASSES):
        super().__init__()
        self.n_classes = n_classes
        self.net = UNet(n_classes + 3, 4, base=base, depth=depth)

    def make_input(self, mask: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """mask [B,H,W] int (ВЗ-классы 0..4; НЗ-маски сначала пропустить через MH2F2)."""
        B, H, W = mask.shape
        dev = mask.device
        oh = Fn.one_hot(mask.long().clamp(0, self.n_classes - 1), self.n_classes).permute(0, 3, 1, 2).float()
        hh = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, 1, H, W)
        ww = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, 1, H, W)
        z = torch.rand(B, 1, H, W, device=dev) if noise is None else noise
        return torch.cat([oh, hh, ww, z], 1)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        return self.net(self.make_input(mask))

    @torch.no_grad()
    def sample(self, mask: torch.Tensor) -> torch.Tensor:
        """Маска → синтетическое сырьё [B,2,H,W] в 0..1 (новая реализация спекла при каждом вызове)."""
        out = self.forward(mask)
        xs = []
        for c in range(2):
            p = torch.sigmoid(out[:, 2 * c]); a = out[:, 2 * c + 1].clamp(0, 1)
            xs.append((torch.rand_like(p) < p).float() * a)
        return torch.stack(xs, 1)


def render_loss(out: torch.Tensor, x: torch.Tensor, pos_weight: float = 8.0) -> torch.Tensor:
    """x — реальное сырьё [B,2,H,W] в 0..1."""
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=x.device))
    loss = 0.0
    for c in range(2):
        act = (x[:, c] > 0).float()
        loss = loss + bce(out[:, 2 * c], act) + ((out[:, 2 * c + 1] - x[:, c]).abs() * act).sum() / (act.sum() + 1)
    return loss


def noise_stats(x_real: np.ndarray, x_synt: np.ndarray) -> dict:
    """Э3 §3.4: KS амплитуд активных пикселей; L1 профилей плотности эха по частоте/высоте; доля активных."""
    out = {}
    for c, pol in enumerate("OX"):
        r, s = x_real[:, c], x_synt[:, c]
        ar, as_ = r[r > 0], s[s > 0]
        out[f"ks_amp_{pol}"] = float(ks_2samp(ar[:200000], as_[:200000]).statistic) if len(ar) > 10 and len(as_) > 10 else np.nan
        out[f"active_real_{pol}"] = float((r > 0).mean()); out[f"active_synt_{pol}"] = float((s > 0).mean())
        dfr, dfs = (r > 0).mean((0, 1)), (s > 0).mean((0, 1))        # плотность по частоте (колонки)
        dhr, dhs = (r > 0).mean((0, 2)), (s > 0).mean((0, 2))        # по высоте (строки)
        out[f"dens_freq_l1_{pol}"] = float(np.abs(dfr - dfs).mean() / (dfr.mean() + 1e-6))
        out[f"dens_height_l1_{pol}"] = float(np.abs(dhr - dhs).mean() / (dhr.mean() + 1e-6))
    return out


@dataclass
class RenderConfig:
    run: str = "base"
    stage: str = "E4"
    manifest: str = "data/manifest.csv"
    epochs: int = 5
    batch: int = 64
    lr: float = 2e-3
    sched: str = "cosine"
    base: int = 16
    depth: int = 3
    pos_weight: float = 8.0
    amp: bool = True
    workers: int = 8
    seed: int = 0
    limit: int = 0
    train_size: int = 40000         # train-образцов в RAM (равномерно по train-сплиту)
    val_size: int = 2000
    log_images: int = 16
    images_every: int = 1
    max_steps: int = 0
    dry: bool = False
    device: str = "cuda"


def train_renderer(cfg: RenderConfig) -> dict:
    if cfg.dry:
        cfg.limit = cfg.limit or 48; cfg.epochs = 1; cfg.max_steps = 1; cfg.val_size = 8; cfg.train_size = 24
        cfg.log_images = 2; cfg.workers = min(cfg.workers, 2); cfg.stage = "dry"
    training.set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    rundir = ROOT / "runs" / cfg.stage / cfg.run
    prov = dict(device=str(dev), torch=torch.__version__, git_commit=training._git_commit(),
                manifest_md5=training._md5(ROOT / cfg.manifest), argv=" ".join(sys.argv),
                started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    log = tblog.TBLog(rundir, asdict(cfg) | prov)
    t_all = time.time()
    tc = training.TrainConfig(manifest=cfg.manifest, limit=cfg.limit, val_size=cfg.val_size)
    df, tr, va = training.load_split(tc)
    tr = tr.iloc[np.unique(np.linspace(0, len(tr) - 1, min(cfg.train_size, len(tr))).round().astype(int))].reset_index(drop=True)
    t0 = time.time()
    Xt, Yt, _ = training.decode(tr, cfg.workers)
    Xv, Yv, _ = training.decode(va, cfg.workers)
    print(f"[{cfg.stage}/{cfg.run}] train {len(Xt)} / val {len(Xv)} пар в RAM за {time.time() - t0:.0f} с; устройство {dev}", flush=True)
    lset = training.select_logging_set(df[df.split == "val"], ROOT / "runs" / cfg.stage / "logging_set.json")
    img_rows = df[df.path.isin([im["path"] for im in lset["images"]])].sort_values(["station", "time"]).head(cfg.log_images)
    Xi, Yi, _ = training.decode(img_rows.reset_index(drop=True), min(cfg.workers, 2))
    titles = [f"{r.station} {str(r.time)[:16]}" for r in img_rows.itertuples()]

    net = Renderer(cfg.base, cfg.depth).to(dev)
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr / 100) if cfg.sched == "cosine" else None
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    print(f"  рендерер {n_params(net)} параметров (base {cfg.base}, depth {cfg.depth}), pos_weight {cfg.pos_weight}", flush=True)
    best = dict(value=float("inf"), epoch=-1); hist = []
    for ep in range(cfg.epochs):
        net.train(); t_ep = time.time(); tot = n = 0
        perm = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(cfg.seed * 100 + ep))
        for step, j in enumerate(perm.split(cfg.batch)):
            if cfg.max_steps and step >= cfg.max_steps:
                break
            x = training.to_input(Xt[j], dev); y = Yt[j].to(dev).long()
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                out = net(y)
            loss = render_loss(out.float(), x, cfg.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(j); n += len(j)
        if sched is not None:
            sched.step()
        t_train = time.time() - t_ep
        net.eval(); vt = vn = 0; xs_r, xs_s = [], []
        with torch.no_grad():
            for k in range(0, len(Xv), 128):
                x = training.to_input(Xv[k:k + 128], dev); y = Yv[k:k + 128].to(dev).long()
                vt += render_loss(net(y).float(), x, cfg.pos_weight).item() * len(x); vn += len(x)
                if k < 1024:
                    xs_r.append(x.cpu().numpy()); xs_s.append(net.sample(y).cpu().numpy())
        m = {"epoch": ep, "train/loss": tot / max(n, 1), "val/loss": vt / max(vn, 1), "time/train_s": t_train,
             "time/train_samples_per_s": n / max(t_train, 1e-9), "train/lr": opt.param_groups[0]["lr"]}
        m.update({f"val/{k}": v for k, v in noise_stats(np.concatenate(xs_r), np.concatenate(xs_s)).items()})
        if len(Xi) and (ep % cfg.images_every == 0 or ep == cfg.epochs - 1):
            with torch.no_grad():
                yi = Yi.to(dev).long()
                s1, s2 = net.sample(yi).cpu().numpy(), net.sample(yi).cpu().numpy()
            (rundir / "png").mkdir(exist_ok=True)
            log.renders("renders/fixed_set", Xi.numpy(), Yi.numpy(), s1, s2, ep, titles=titles,
                        save=rundir / "png" / f"renders_ep{ep:02d}.png")
        log.scalars({k: v for k, v in m.items() if k != "epoch"}, ep); log.row(m, ep); hist.append(m)
        ckpt = {"state_dict": net.state_dict(), "cfg": asdict(cfg), "epoch": ep, "metrics": m}
        torch.save(ckpt, rundir / "weights_last.pt")
        if m["val/loss"] < best["value"]:
            best = dict(value=m["val/loss"], epoch=ep); torch.save(ckpt, rundir / "weights.pt")
        print(f"  ep{ep}: train {m['train/loss']:.3f} | val {m['val/loss']:.3f} KS_O {m['val/ks_amp_O']:.3f} "
              f"активн. real/synt O {m['val/active_real_O']:.3f}/{m['val/active_synt_O']:.3f} "
              f"плотн. L1 f/h {m['val/dens_freq_l1_O']:.2f}/{m['val/dens_height_l1_O']:.2f} | {t_train:.0f} с, "
              f"{m['time/train_samples_per_s']:.0f} обр/с", flush=True)
    summary = {"run": cfg.run, "stage": cfg.stage, "params": n_params(net), "n_train": int(len(Xt)), "n_val": int(len(Xv)),
               "best_epoch": best["epoch"], "best_value": best["value"], "best": hist[best["epoch"]] if best["epoch"] >= 0 else {},
               "last": hist[-1], "time_total_s": time.time() - t_all}
    (rundir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log.close()
    print(f"[{cfg.stage}/{cfg.run}] готово за {time.time() - t_all:.0f} с → {rundir}", flush=True)
    return summary


def load_renderer(path: str | Path, dev="cuda") -> Renderer:
    ck = torch.load(path, map_location=dev)
    c = ck["cfg"]
    net = Renderer(c.get("base", 16), c.get("depth", 3)).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(RenderConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train_renderer(RenderConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
