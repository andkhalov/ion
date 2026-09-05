# -*- coding: utf-8 -*-
"""
oblique_train.py — НЗ-модель на синтетике (этап E5; Э2 §3, §5; Э3 §2 E3/E5, §3.1–3.3, §3.5–3.6).

Данные (Э3 §2.2 E3): `loader.ObliqueDataset` синтезирует маску НЗ на лету из SAO — следы ARTIST
пересчитаны секансом на сферической Земле (Брейт–Тьюв, Мартин; Э2 §2.4), O-компонента (X — E3b),
дальность D случайна из D_SET на каждый показ, классы obs.OB_CLASSES (BG F2 F1 E Es MH — кратник
2F2), точные МПЧ-метки мод. Сырьё порождает нейрорендерер (E4, `pyon.renderer`) на GPU в батче:
новая реализация спекла на каждом показе (бесконечная аугментация).

Модель: U-Net(2 → 6) той же архитектуры, что E1. Раны: baseline (CE) | +lognorm / +hinge
(CE + λ·L_logic: S1 МПЧ(2F2) ≤ МПЧ(1F2), S2 P′(2F2) ≥ P′(1F2)+150 км, P2 связность — с детачем).

Метрики (Э3 §3): IoU по классам (в т.ч. MH); |ΔМПЧ| RMSE/медиана для 1F2 и 2F2 против аналитических
меток; P′ носа; инвариант Пономарчука МПЧ(2F2)/МПЧ(1F2) (медиана/IQR предсказаний vs меток +
гистограмма); SHACL-гейт НЗ-сцен (`gates.oblique_scene`) + референс меток (обязан быть 0 %);
стратификация по D; кросс-тест «ВЗ-модель zero-shot на НЗ» (`--vz_weights`); суточные треки
МПЧ(t) на фиксированных сутках E1 при фиксированной D.

Запуск:  python -m pyon.oblique_train --stage E5 --run lognorm --variant lognorm \\
             --renderer runs/E4/base/weights.pt --manifest data/manifest.csv --epochs 5
         python -m pyon.oblique_train --dry --renderer runs/dry/dry_render/weights.pt
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
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, gates, loader, logic, renderer as rnd, tblog, training   # noqa: E402
from pyon import digi_formats as dfm                                                # noqa: E402
from pyon import oblique_synth as obs                                               # noqa: E402
from pyon import validate as vd                                                     # noqa: E402
from pyon.models import UNet, n_params                                              # noqa: E402

VARIANTS = {"baseline": None, "lognorm": "lognorm", "hinge": "hinge"}
CE_WEIGHTS = [0.05, 1.0, 1.0, 1.0, 1.0, 1.0]
EXTENT = (obs.FOB_MIN, obs.FOB_MAX, obs.P_MIN, obs.P_MAX)
iF2, iMH, iEs = obs.OB_CLASSES.index("F2"), obs.OB_CLASSES.index("MH"), obs.OB_CLASSES.index("Es")


@dataclass
class ObliqueConfig:
    run: str = "lognorm"
    stage: str = "E5"
    manifest: str = "data/manifest.csv"
    renderer: str = "runs/E4/base/weights.pt"
    variant: str = "lognorm"        # baseline | lognorm | hinge
    lam: float = 0.5
    component: str = "O"
    epochs: int = 5
    steps_per_epoch: int = 0        # 0 = весь train-манифест
    batch: int = 64
    lr: float = 2e-3
    sched: str = "cosine"
    depth: int = 4
    base: int = 32
    amp: bool = True
    workers: int = 8
    seed: int = 0
    limit: int = 0
    val_size: int = 2400
    gate_n: int = 150
    gate_every: int = 2
    gate_procs: int = 4
    log_images: int = 16
    images_every: int = 1
    track_d: float = 800.0          # дальность для суточных треков МПЧ
    vz_weights: str = ""            # кросс-тест: ВЗ-модель zero-shot
    max_steps: int = 0
    dry: bool = False
    device: str = "cuda"


def decode_oblique(df: pd.DataFrame, component: str, workers: int, seed: int = 0, fixed_d=None, batch: int = 128):
    """ObliqueDataset → Y int8 [N,H,W], D [N], muf_F2 [N], muf_MH [N] (в RAM)."""
    if not len(df):
        z = torch.zeros(0)
        return torch.zeros((0, obs.NP, obs.NF), dtype=torch.int8), z, z, z
    ds = loader.ObliqueDataset(df, component=component, fixed_d=fixed_d, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    ys, ds_, m1, m2 = zip(*[b for b in dl])
    return torch.cat(ys), torch.cat(ds_), torch.cat(m1), torch.cat(m2)


def muf_readouts(pm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """МПЧ 1F2 (класс F2), МПЧ 2F2 (MH) и P′ носа F2 по маскам."""
    f1 = np.array([canon.fmax_readout(p, iF2, obs.fob_axis) for p in pm])
    f2 = np.array([canon.fmax_readout(p, iMH, obs.fob_axis) for p in pm])
    pn = np.array([canon.hmin_readout(p, iF2, obs.p_axis) for p in pm])
    return f1, f2, pn


@torch.no_grad()
def predict(net, X: torch.Tensor, dev, batch: int = 128):
    net.eval(); pms, comps = [], {}
    for k in range(0, len(X), batch):
        lg = net(X[k:k + batch].to(dev)).float()
        pms.append(lg.argmax(1).to(torch.int8).cpu())
        _, c = logic.oblique_logic(lg, "hinge")
        for name, v in c.items():
            comps.setdefault(name, []).append(v.detach().cpu())
    return torch.cat(pms).numpy(), {k: torch.cat(v).numpy() for k, v in comps.items()}


def evaluate(net, Xv, Yv, Dv, M1, M2, dev, cfg, vocab, epoch, log, ref_gate: dict, vz_net=None):
    m = {}
    pm, comps = predict(net, Xv, dev)
    Yn = Yv.numpy()
    for k, v in comps.items():
        m[f"val/logic_{k}"] = float(v.mean())
    tot = sum(comps.values()); m["val/logic_total"] = float(tot.mean()); log.hist("val/logic_total_hist", tot, epoch)
    for ci, cls in enumerate(obs.OB_CLASSES[1:], 1):
        inter = ((pm == ci) & (Yn == ci)).sum(); union = ((pm == ci) | (Yn == ci)).sum()
        m[f"val/IoU_{cls}"] = float(inter / union) if union else np.nan
    f1, f2, pn = muf_readouts(pm)
    lab1, lab2 = M1.numpy(), M2.numpy()
    for name, pred, lab in (("MUF1F2", f1, lab1), ("MUF2F2", f2, lab2)):
        st = training.err_stats(pred, lab)
        for k, v in st.items():
            m[f"val/{name}_{k}"] = v
    Dn = Dv.numpy()
    for d in sorted(set(Dn.tolist())):
        mk = Dn == d
        st = training.err_stats(f1[mk], lab1[mk]); m[f"strat/D{int(d)}/MUF1F2_rmse"] = st.get("rmse", np.nan)
        inter = ((pm[mk] == iF2) & (Yn[mk] == iF2)).sum(); union = ((pm[mk] == iF2) | (Yn[mk] == iF2)).sum()
        m[f"strat/D{int(d)}/IoU_F2"] = float(inter / union) if union else np.nan
    # инвариант Пономарчука: отношение МПЧ кратностей
    okl = np.isfinite(lab1) & np.isfinite(lab2); okp = np.isfinite(f1) & np.isfinite(f2)
    rl, rp = lab2[okl] / lab1[okl], f2[okp] / f1[okp]
    if len(rl):
        m["val/inv_ratio_label_med"] = float(np.median(rl)); m["val/inv_ratio_label_iqr"] = float(np.percentile(rl, 75) - np.percentile(rl, 25))
    if len(rp):
        m["val/inv_ratio_pred_med"] = float(np.median(rp)); m["val/inv_ratio_pred_iqr"] = float(np.percentile(rp, 75) - np.percentile(rp, 25))
        log.hist("val/inv_ratio_pred", rp, epoch); log.hist("val/inv_ratio_label", rl, epoch)
    n_g = min(cfg.gate_n, len(pm))
    if n_g and (epoch % cfg.gate_every == 0 or epoch == cfg.epochs - 1):
        t_g = time.time()
        rate, warn, _ = gates.gate_rate(pm[:n_g], gates.oblique_scene, vocab, prefix=f"e{epoch}_", procs=cfg.gate_procs, with_warnings=True)
        m["gate/violations"], m["gate/warnings"] = rate, warn
        if "labels" not in ref_gate:
            ref_gate["labels"], _, _ = gates.gate_rate(Yn[:n_g], gates.oblique_scene, vocab, prefix="lab_", procs=cfg.gate_procs, with_warnings=True)
        m["gate/labels_violations"] = ref_gate["labels"]; m["gate/n"] = n_g; m["time/gate_s"] = time.time() - t_g
    if vz_net is not None and "vz" not in ref_gate:      # кросс-тест ВЗ-модели zero-shot (один раз)
        vz_net.eval(); pmv = []
        with torch.no_grad():
            for k in range(0, len(Xv), 128):
                pmv.append(vz_net(Xv[k:k + 128].to(dev)).float().argmax(1).cpu())
        pmv = torch.cat(pmv).numpy()
        fv = np.array([canon.fmax_readout(p, 1, obs.fob_axis) for p in pmv])
        inter = ((pmv == 1) & (Yn == iF2)).sum(); union = ((pmv == 1) | (Yn == iF2)).sum()
        ref_gate["vz"] = dict(IoU_F2=float(inter / max(union, 1)), MUF1F2_rmse=training.err_stats(fv, lab1).get("rmse", np.nan))
    if "vz" in ref_gate:
        m["crosstest/vz_zero_shot_IoU_F2"] = ref_gate["vz"]["IoU_F2"]; m["crosstest/vz_zero_shot_MUF1F2_rmse"] = ref_gate["vz"]["MUF1F2_rmse"]
    return m, pm, pd.DataFrame(dict(D=Dn, MUF1F2_label=lab1, MUF1F2_pred=f1, MUF2F2_label=lab2, MUF2F2_pred=f2, Pnose_pred=pn, logic_total=tot))


def train_oblique(cfg: ObliqueConfig) -> dict:
    if cfg.dry:
        cfg.limit = cfg.limit or 48; cfg.epochs = 1; cfg.max_steps = 1; cfg.val_size = 8; cfg.gate_n = 2
        cfg.log_images = 2; cfg.workers = min(cfg.workers, 2); cfg.gate_procs = 2; cfg.stage = "dry"
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
    ren = rnd.load_renderer(ROOT / cfg.renderer, dev)
    mh2f2 = rnd.MH2F2.to(dev)
    t0 = time.time()
    Yv, Dv, M1, M2 = decode_oblique(va, cfg.component, cfg.workers, seed=cfg.seed + 7)
    torch.manual_seed(cfg.seed + 1)
    Xv = torch.cat([ren.sample(mh2f2[Yv[k:k + 128].to(dev).long()]).cpu() for k in range(0, len(Yv), 128)]) if len(Yv) else torch.zeros(0)
    print(f"[{cfg.stage}/{cfg.run}] train {len(tr)} SAO, val {len(Yv)} (D: {np.unique(Dv.numpy()).tolist()}), "
          f"рендер val за {time.time() - t0:.0f} с; устройство {dev}", flush=True)
    dl = DataLoader(loader.ObliqueDataset(tr, component=cfg.component, seed=cfg.seed), batch_size=cfg.batch, shuffle=True,
                    num_workers=cfg.workers, persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers > 0 else None)
    lset = training.select_logging_set(df[df.split == "val"], ROOT / "runs" / "E1" / "logging_set.json") \
        if (ROOT / "runs" / "E1" / "logging_set.json").exists() else \
        training.select_logging_set(df[df.split == "val"], ROOT / "runs" / cfg.stage / "logging_set.json")
    day_sets = []
    for d in lset["days"]:
        rows = df[(df.station == d["station"]) & (pd.to_datetime(df.time).dt.date.astype(str) == d["date"])].sort_values("time").reset_index(drop=True)
        Yd, _, m1d, m2d = decode_oblique(rows, cfg.component, min(cfg.workers, 2), fixed_d=cfg.track_d)
        day_sets.append((d, rows, Yd, m1d.numpy(), m2d.numpy()))
    Yi, Di = Yv[:cfg.log_images], Dv[:cfg.log_images]
    print(f"  панелей {len(Yi)}, суток-треков {len(day_sets)} при D={cfg.track_d:.0f}", flush=True)

    net = UNet(2, len(obs.OB_CLASSES), base=cfg.base, depth=cfg.depth).to(dev)
    vz_net = None
    if cfg.vz_weights:
        ck = torch.load(ROOT / cfg.vz_weights, map_location=dev); c = ck["cfg"]
        vz_net = UNet(2, len(canon.CLASSES), base=c["base"], depth=c["depth"], profile=c.get("profile", False)).to(dev)
        vz_net.load_state_dict(ck["state_dict"])
    opt = torch.optim.Adam(net.parameters(), cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs), eta_min=cfg.lr / 100) if cfg.sched == "cosine" else None
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(CE_WEIGHTS, device=dev))
    vocab = vd.load_vocabulary(); ref_gate: dict = {}
    variant = VARIANTS[cfg.variant]
    print(f"  U-Net {n_params(net)} параметров, вариант {cfg.variant}, рендерер {cfg.renderer}", flush=True)
    best = dict(value=float("inf"), epoch=-1); hist = []
    for ep in range(cfg.epochs):
        net.train(); t_ep = time.time(); n_seen = 0; sums = {"CE": 0.0, "logic": 0.0}
        for step, (y, d, _, _) in enumerate(dl):
            if (cfg.max_steps and step >= cfg.max_steps) or (cfg.steps_per_epoch and step >= cfg.steps_per_epoch):
                break
            y = y.to(dev).long()
            x = ren.sample(mh2f2[y])
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                lg = net(x); loss_ce = ce(lg, y)
            loss = loss_ce
            if variant:
                l_logic, _ = logic.oblique_logic(lg.float(), variant)
                loss = loss + cfg.lam * l_logic; sums["logic"] += l_logic.item() * len(y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sums["CE"] += loss_ce.item() * len(y); n_seen += len(y)
            if cfg.dry:
                print(f"  dry: x {tuple(x.shape)} y {tuple(y.shape)} logits {tuple(lg.shape)} CE {loss_ce.item():.3f} loss {loss.item():.3f}", flush=True)
        if sched is not None:
            sched.step()
        t_train = time.time() - t_ep
        m = {"epoch": ep, "train/CE": sums["CE"] / max(n_seen, 1), "time/train_s": t_train,
             "time/train_samples_per_s": n_seen / max(t_train, 1e-9), "train/lr": opt.param_groups[0]["lr"]}
        if variant:
            m["train/logic"] = sums["logic"] / max(n_seen, 1)
        t_ev = time.time()
        mv, pm, rt = evaluate(net, Xv, Yv, Dv, M1, M2, dev, cfg, vocab, ep, log, ref_gate, vz_net)
        m.update(mv)
        if len(Yi) and (ep % cfg.images_every == 0 or ep == cfg.epochs - 1):
            (rundir / "png").mkdir(exist_ok=True)
            log.ionograms("images/fixed_set", Xv[:len(Yi)].numpy(), Yi.numpy(), pm[:len(Yi)], ep, extent=EXTENT,
                          titles=[f"D={float(d):.0f}" for d in Di], xlabel="МГц", ylabel="P′, км")
        for d, rows, Yd, m1d, m2d in day_sets:
            if not len(Yd):
                continue
            with torch.no_grad():
                Xd = torch.cat([ren.sample(mh2f2[Yd[k:k + 128].to(dev).long()]).cpu() for k in range(0, len(Yd), 128)])
            pmd, _ = predict(net, Xd, dev)
            f1, f2, _ = muf_readouts(pmd)
            key = f"{d['station']}_{d['date']}_{d['kind']}"
            hours = pd.to_datetime(rows.time).dt.hour.values + pd.to_datetime(rows.time).dt.minute.values / 60
            log.tracks_grid(f"track/{key}", hours, {"MUF1F2": (m1d, f1, "МГц"), "MUF2F2": (m2d, f2, "МГц")}, ep,
                            title=f"{key} D={cfg.track_d:.0f} км", save=rundir / "png" / f"track_{key}_ep{ep:02d}.png")
            for name, pred, lab in (("MUF1F2", f1, m1d), ("MUF2F2", f2, m2d)):
                st = training.err_stats(pred, lab); m[f"track/{key}/{name}_rmse"] = st.get("rmse", np.nan)
        m["time/eval_s"] = time.time() - t_ev
        log.scalars({k: v for k, v in m.items() if k != "epoch"}, ep); log.row(m, ep); hist.append(m)
        ckpt = {"state_dict": net.state_dict(), "cfg": asdict(cfg), "epoch": ep, "metrics": m}
        torch.save(ckpt, rundir / "weights_last.pt")
        crit = m.get("val/MUF1F2_med", np.nan)
        if np.isfinite(crit) and crit < best["value"]:
            best = dict(value=float(crit), epoch=ep); torch.save(ckpt, rundir / "weights.pt")
        print(f"  ep{ep}: train CE {m['train/CE']:.3f}" + (f" logic {m['train/logic']:.4f}" if variant else "")
              + f" | L_hinge {m['val/logic_total']:.4f} IoU F2 {m.get('val/IoU_F2', np.nan):.3f} MH {m.get('val/IoU_MH', np.nan):.3f} "
              f"МПЧ1F2 RMSE {m.get('val/MUF1F2_rmse', np.nan):.2f} med {m.get('val/MUF1F2_med', np.nan):.2f} "
              f"МПЧ2F2 RMSE {m.get('val/MUF2F2_rmse', np.nan):.2f} inv {m.get('val/inv_ratio_pred_med', np.nan):.3f}/{m.get('val/inv_ratio_label_med', np.nan):.3f} "
              f"gate {m.get('gate/violations', np.nan):.0%} (метки {m.get('gate/labels_violations', np.nan):.0%}) | {t_train:.0f}+{m['time/eval_s']:.0f} с, "
              f"{m['time/train_samples_per_s']:.0f} обр/с", flush=True)
    rt.to_csv(rundir / "val_readouts.csv", index=False)
    summary = {"run": cfg.run, "stage": cfg.stage, "variant": cfg.variant, "params": n_params(net), "n_val": int(len(Yv)),
               "best_epoch": best["epoch"], "best_value": best["value"], "best": hist[best["epoch"]] if best["epoch"] >= 0 else {},
               "last": hist[-1], "time_total_s": time.time() - t_all}
    (rundir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log.close()
    print(f"[{cfg.stage}/{cfg.run}] готово за {time.time() - t_all:.0f} с → {rundir}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(ObliqueConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train_oblique(ObliqueConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
