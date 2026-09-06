# -*- coding: utf-8 -*-
"""
gan.py — рендерер шума v2: условный GAN «маска ARTIST → реальное сырьё» (этап E4, решение АХ
2026-09-06: попиксельный рендерер v1 (`renderer.py`) выучивал условное среднее — кляксы по всему полю,
визуально не похоже на реальные ионограммы; нужен дискриминатор, который отличает настоящую
ионограмму от сгенерированной, «всё по классике, архитектура современнее»).

Генератор G(mask, z): U-Net (one-hot маска + координаты) с ГЛОБАЛЬНЫМ латентом z → FiLM (масштаб/сдвиг
признаков декодера на каждом уровне) и ПОПИКСЕЛЬНОЙ инъекцией шума в декодере (как в StyleGAN) —
чтобы положение помеховых полос, мусора и кратников менялось от образца к образцу; выход 2 канала
(O, X) в 0..1, clamp(relu) — точные нули как у реального сырья. Дискриминатор D(mask, x): PatchGAN со
спектральной нормализацией (патч-логиты 8×8) + возврат промежуточных признаков.
Лоссы: hinge (D: relu(1−D(real)) + relu(1+D(fake)); G: −D(fake)); R1-штраф на реальных (γ, лениво раз в
r1_every шагов); feature matching по признакам D (λ_fm, стабилизация, pix2pixHD); L1 ТОЛЬКО внутри
расширенной маски следа (λ_l1: геометрия следа детерминирована маской; на фоне L1 запрещён — он
и давал условное среднее). EMA генератора для сэмплирования.
Интерфейс совместим с v1: `sample(mask)` → [B,2,H,W] 0..1; загрузка — `renderer.load_renderer` (kind="gan").

Запуск: python -m pyon.gan --stage E4 --run gan --manifest data/manifest.csv --steps 60000 --train_size 60000
        python -m pyon.gan --dry
Артефакты: runs/E4/<ран>/{events, metrics.csv, weights.pt (EMA), config.json, png/renders_*.png, png/turing_*.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from torch import nn
from torch.nn.utils import spectral_norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pyon import canon, tblog, training                    # noqa: E402
from pyon.renderer import noise_stats, MH2F2               # noqa: E402

N_CLASSES = len(canon.CLASSES)


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c), c)


class FiLM(nn.Module):
    def __init__(self, wdim: int, c: int):
        super().__init__()
        self.lin = nn.Linear(wdim, 2 * c)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, w):
        s, b = self.lin(w).chunk(2, 1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


class DecBlock(nn.Module):
    """Upsample → concat skip → conv → GN → FiLM(w) → +шум·scale → LeakyReLU → conv → GN → LeakyReLU."""

    def __init__(self, cin: int, cskip: int, cout: int, wdim: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin + cskip, cout, 3, padding=1); self.n1 = _gn(cout); self.film = FiLM(wdim, cout)
        self.noise_scale = nn.Parameter(torch.zeros(1, cout, 1, 1))
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1); self.n2 = _gn(cout)

    def forward(self, x, skip, w):
        x = Fn.interpolate(x, scale_factor=2, mode="nearest")
        x = self.c1(torch.cat([x, skip], 1)); x = self.film(self.n1(x), w)
        x = x + self.noise_scale * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = Fn.leaky_relu(x, 0.2)
        return Fn.leaky_relu(self.n2(self.c2(x)), 0.2)


class Generator(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, base: int = 48, depth: int = 4, zdim: int = 128, wdim: int = 256):
        super().__init__()
        self.n_classes, self.zdim = n_classes, zdim
        self.corr = (0, 0)                                  # совместимость с интерфейсом v1 (не используется)
        widths = [base * 2 ** k for k in range(depth)]
        self.mlp = nn.Sequential(nn.Linear(zdim, wdim), nn.LeakyReLU(0.2), nn.Linear(wdim, wdim), nn.LeakyReLU(0.2))
        self.enc = nn.ModuleList()
        cin = n_classes + 2
        for k, c in enumerate(widths):
            self.enc.append(nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), _gn(c), nn.LeakyReLU(0.2),
                                          nn.Conv2d(c, c, 3, padding=1), _gn(c), nn.LeakyReLU(0.2)))
            cin = c
        self.dec = nn.ModuleList([DecBlock(widths[k], widths[k - 1], widths[k - 1], wdim) for k in range(depth - 1, 0, -1)])
        self.head = nn.Conv2d(widths[0], 4, 1)        # на поляризацию: логит активности, логит амплитуды

    def make_input(self, mask: torch.Tensor) -> torch.Tensor:
        B, H, W = mask.shape; dev = mask.device
        oh = Fn.one_hot(mask.long().clamp(0, self.n_classes - 1), self.n_classes).permute(0, 3, 1, 2).float()
        hh = torch.linspace(0, 1, H, device=dev).view(1, 1, H, 1).expand(B, 1, H, W)
        ww = torch.linspace(0, 1, W, device=dev).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([oh, hh, ww], 1)

    def forward(self, mask: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        x = self.make_input(mask)
        if z is None:
            z = torch.randn(x.shape[0], self.zdim, device=x.device)
        w = self.mlp(z)
        skips = []
        for k, blk in enumerate(self.enc):
            x = blk(x if k == 0 else Fn.avg_pool2d(x, 2)); skips.append(x)
        y = skips[-1]
        for j, blk in enumerate(self.dec):
            y = blk(y, skips[-2 - j], w)
        o = self.head(y)
        # Сырьё разрежено (~85 % точных нулей после порога): выход = ворота активности × амплитуда.
        # Ворота — Бернулли со straight-through (вперёд — жёсткий 0/1, назад — градиент через
        # σ(логит)): точные нули как у реального, градиент не умирает (relu-выход выродился в нули,
        # запуск 2026-09-06 13:00). Амплитуда — сигмоида.
        logit, amp = o[:, 0::2], o[:, 1::2]
        p = torch.sigmoid(logit)
        hard = (torch.rand_like(p) < p).float()
        gate = hard + p - p.detach()
        return gate * torch.sigmoid(amp)

    @torch.no_grad()
    def sample(self, mask: torch.Tensor) -> torch.Tensor:
        return self.forward(mask)


class Discriminator(nn.Module):
    """PatchGAN со спектральной нормализацией: вход [one-hot маска, x] → патч-логиты 8×8 + признаки."""

    def __init__(self, n_classes: int = N_CLASSES, base: int = 64):
        super().__init__()
        self.n_classes = n_classes
        chans = [n_classes + 2, base, base * 2, base * 4, base * 8]
        self.blocks = nn.ModuleList([nn.Sequential(spectral_norm(nn.Conv2d(chans[k], chans[k + 1], 4, stride=2, padding=1)), nn.LeakyReLU(0.2))
                                     for k in range(4)])
        self.out = spectral_norm(nn.Conv2d(chans[-1], 1, 3, padding=1))

    def forward(self, mask: torch.Tensor, x: torch.Tensor):
        oh = Fn.one_hot(mask.long().clamp(0, self.n_classes - 1), self.n_classes).permute(0, 3, 1, 2).float()
        h = torch.cat([oh, x], 1); feats = []
        for b in self.blocks:
            h = b(h); feats.append(h)
        return self.out(h), feats


@dataclass
class GanConfig:
    run: str = "gan"
    stage: str = "E4"
    manifest: str = "data/manifest.csv"
    steps: int = 60000
    batch: int = 64
    lr: float = 2e-4
    base: int = 48
    depth: int = 4
    zdim: int = 128
    d_base: int = 64
    lam_fm: float = 2.0
    lam_l1: float = 5.0
    r1: float = 0.1                 # γ R1 (лениво раз в r1_every; эффективно γ·r1_every); 1.0 давил D в константу
    r1_every: int = 16
    ema: float = 0.999
    train_size: int = 60000
    cache: str = "local/gan_cache.npz"   # декодированные пары (X uint8, Y int8) — чтобы перезапуски не парсили корпус
    val_size: int = 1024
    eval_every: int = 2000
    log_images: int = 16
    workers: int = 3
    seed: int = 0
    mult: float = 0.0               # доля образцов с кратником во входе (0 — как есть; кратники реальные ионограммы содержат и так)
    dry: bool = False
    device: str = "cuda"


def trace_region(y: torch.Tensor, dilate: int = 2) -> torch.Tensor:
    k = 2 * dilate + 1
    return Fn.max_pool2d((y > 0).float().unsqueeze(1), k, stride=1, padding=dilate)


def train_gan(cfg: GanConfig) -> dict:
    if cfg.dry:
        cfg.steps, cfg.train_size, cfg.val_size, cfg.eval_every, cfg.log_images, cfg.stage = 6, 256, 64, 3, 4, "dry"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    rundir = ROOT / "runs" / cfg.stage / cfg.run; (rundir / "png").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / cfg.manifest, low_memory=False)
    tr_all, va_all = df[df.split == "train"], df[df.split == "val"]
    tr = tr_all.iloc[np.unique(np.linspace(0, len(tr_all) - 1, min(cfg.train_size, len(tr_all))).round().astype(int))].reset_index(drop=True)
    va = va_all.iloc[np.unique(np.linspace(0, len(va_all) - 1, min(cfg.val_size, len(va_all))).round().astype(int))].reset_index(drop=True)
    t0 = time.time()
    cache = ROOT / cfg.cache if cfg.cache and not cfg.dry else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        if int(z["n_train"]) == len(tr) and int(z["n_val"]) == len(va) and str(z["manifest"]) == cfg.manifest:
            Xt, Yt, Xv, Yv = (torch.from_numpy(z[k]) for k in ("Xt", "Yt", "Xv", "Yv"))
            print(f"  кэш {cache}: train {len(Xt)} / val {len(Xv)} загружены за {time.time() - t0:.0f} с", flush=True)
        else:
            cache = None
    if cache is None or not cache.exists():
        Xt, Yt, _ = training.decode(tr, cfg.workers); Xv, Yv, _ = training.decode(va, cfg.workers)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, Xt=Xt.numpy(), Yt=Yt.numpy(), Xv=Xv.numpy(), Yv=Yv.numpy(), n_train=len(tr), n_val=len(va), manifest=cfg.manifest)
    lset_path = ROOT / "runs" / "E1" / "logging_set.json"
    lset = training.select_logging_set(va_all, lset_path if lset_path.exists() else rundir / "logging_set.json")
    img_rows = df[df.path.isin([im["path"] for im in lset["images"]])].sort_values(["station", "time"]).head(cfg.log_images).reset_index(drop=True)
    Xi, Yi, _ = training.decode(img_rows, min(cfg.workers, 2))
    titles = [f"{r.station} {str(r.time)[:16]}" for r in img_rows.itertuples()]
    print(f"[{cfg.stage}/{cfg.run}] train {len(Xt)} / val {len(Xv)} пар в RAM за {time.time() - t0:.0f} с; {dev}", flush=True)
    G = Generator(base=cfg.base, depth=cfg.depth, zdim=cfg.zdim).to(dev); D = Discriminator(base=cfg.d_base).to(dev)
    G_ema = deepcopy(G).eval()
    for p_ in G_ema.parameters():
        p_.requires_grad_(False)
    optG = torch.optim.Adam(G.parameters(), cfg.lr, betas=(0.0, 0.99)); optD = torch.optim.Adam(D.parameters(), cfg.lr, betas=(0.0, 0.99))
    log = tblog.TBLog(rundir, asdict(cfg) | dict(params_G=sum(p.numel() for p in G.parameters()), params_D=sum(p.numel() for p in D.parameters())))
    log.readme(f"""## {cfg.stage}/{cfg.run} — рендерер шума v2: условный GAN (маска ARTIST → сырьё)
G(маска, z): U-Net + глобальный латент (FiLM) + попиксельный шум в декодере; D(маска, x): PatchGAN (SN). Лоссы: hinge, R1 (γ={cfg.r1}), feature matching (λ={cfg.lam_fm}), L1 внутри маски следа (λ={cfg.lam_l1}). Сэмплы — EMA-генератор.
**SCALARS** `1_train/*`: loss_D, loss_G (hinge; порядок ~1), loss_fm, loss_l1, D_real/D_fake — средние логиты (расходятся → D побеждает).
`2_val/*` (каждые {cfg.eval_every} шагов, {cfg.val_size} val-масок, EMA): ks_amp_O — KS амплитуд активных пикселей реальное vs рендер; active_real_O vs active_synt_O — доля активных;
dens_freq_l1_O/dens_height_l1_O — L1 профилей плотности эха по частоте/высоте; corr_f/corr_h — P(сосед активен|активен)/p у рендера (реальное: ~3.7 / ~5.6 фон).
**IMAGES** `renders` — реальное (O) | маска ARTIST | рендер 1 | рендер 2 (фиксированный набор {cfg.log_images} ионограмм) — главная визуальная проверка;
`turing` — 16 плиток вперемешку реальные/рендер БЕЗ подписей (ключ — в TEXT `turing_key`): можно ли отличить глазами.""")
    print(f"  G {sum(p.numel() for p in G.parameters()) / 1e6:.2f} M, D {sum(p.numel() for p in D.parameters()) / 1e6:.2f} M параметров", flush=True)
    n = len(Xt); step = 0; t_ep = time.time(); sums = {}
    best = dict(value=float("inf"), step=-1)
    while step < cfg.steps:
        perm = torch.randperm(n)
        for j in perm.split(cfg.batch):
            if step >= cfg.steps or len(j) < cfg.batch:
                break
            x = Xt[j].to(dev).float().div_(255); y = Yt[j].to(dev).long()
            if cfg.mult > 0:
                y_in = training.add_multiples(y, cfg.mult)
            else:
                y_in = y
            # --- D
            with torch.no_grad():
                fake = G(y_in)
            need_r1 = cfg.r1 > 0 and step % cfg.r1_every == 0
            x_real = x.requires_grad_(need_r1)
            d_real, _ = D(y_in, x_real); d_fake, _ = D(y_in, fake)
            loss_d = Fn.relu(1 - d_real).mean() + Fn.relu(1 + d_fake).mean()
            if need_r1:
                grad, = torch.autograd.grad(d_real.sum(), x_real, create_graph=True)
                loss_d = loss_d + 0.5 * cfg.r1 * cfg.r1_every * grad.pow(2).flatten(1).sum(1).mean()
            optD.zero_grad(set_to_none=True); loss_d.backward(); optD.step()
            x = x.detach()
            # --- G
            fake = G(y_in)
            d_fake, f_fake = D(y_in, fake)
            with torch.no_grad():
                _, f_real = D(y_in, x)
            loss_g = -d_fake.mean()
            loss_fm = sum((a - b).abs().mean() for a, b in zip(f_fake, f_real)) / len(f_fake)
            reg = trace_region(y)
            loss_l1 = ((fake - x).abs() * reg).sum() / (reg.sum() * 2 + 1)
            loss = loss_g + cfg.lam_fm * loss_fm + cfg.lam_l1 * loss_l1
            optG.zero_grad(set_to_none=True); loss.backward(); optG.step()
            with torch.no_grad():
                for pe, pg in zip(G_ema.parameters(), G.parameters()):
                    pe.lerp_(pg, 1 - cfg.ema)
            for k, v in dict(loss_D=loss_d.item(), loss_G=loss_g.item(), loss_fm=loss_fm.item(), loss_l1=loss_l1.item(),
                             D_real=d_real.mean().item(), D_fake=d_fake.mean().item()).items():
                sums[k] = sums.get(k, 0.0) + v
            step += 1
            if step % cfg.eval_every == 0 or step == cfg.steps:
                m = {f"train/{k}": v / cfg.eval_every for k, v in sums.items()}; sums = {}
                m["train/steps_per_s"] = cfg.eval_every / max(time.time() - t_ep, 1e-9); t_ep = time.time()
                G_ema.eval(); torch.manual_seed(cfg.seed + step)
                with torch.no_grad():
                    xs = torch.cat([G_ema.sample(Yv[k:k + 128].to(dev).long()).cpu() for k in range(0, len(Yv), 128)]).numpy()
                xr = Xv.float().div(255).numpy()
                m.update({f"val/{k}": v for k, v in noise_stats(xr, xs).items()})
                act = (xs[:, 0] > 0).astype(np.float32); p_ = act.mean()
                m["val/corr_f"] = float((act[:, :, 1:] * act[:, :, :-1]).mean() / max(p_ ** 2, 1e-9)); m["val/corr_h"] = float((act[:, 1:] * act[:, :-1]).mean() / max(p_ ** 2, 1e-9))
                score = m["val/ks_amp_O"] + abs(m["val/active_synt_O"] - m["val/active_real_O"]) * 5 + m["val/dens_freq_l1_O"] + m["val/dens_height_l1_O"]
                m["val/score"] = score
                ck = dict(kind="gan", state_dict=G_ema.state_dict(), cfg=asdict(cfg), step=step, metrics=m)
                torch.save(ck, rundir / "weights_last.pt")
                if score < best["value"]:
                    best = dict(value=score, step=step); torch.save(ck, rundir / "weights.pt")
                with torch.no_grad():
                    yi = Yi.to(dev).long(); s1 = G_ema.sample(yi).cpu().numpy(); s2 = G_ema.sample(yi).cpu().numpy()
                log.renders("renders", Xi.numpy(), Yi.numpy(), s1, s2, step, save=rundir / "png" / f"renders_{step:06d}.png")
                turing(log, Xi.numpy(), s1, step, rundir / "png" / f"turing_{step:06d}.png", cfg.seed + step)
                log.scalars(m, step); log.row(m, step)
                print(f"  шаг {step}: D {m['train/loss_D']:.3f} G {m['train/loss_G']:.3f} fm {m['train/loss_fm']:.3f} l1 {m['train/loss_l1']:.4f} | "
                      f"KS {m['val/ks_amp_O']:.3f} активн. {m['val/active_real_O']:.3f}/{m['val/active_synt_O']:.3f} плотн. {m['val/dens_freq_l1_O']:.2f}/{m['val/dens_height_l1_O']:.2f} "
                      f"corr {m['val/corr_f']:.2f}/{m['val/corr_h']:.2f} score {score:.3f} | {m['train/steps_per_s']:.1f} шаг/с", flush=True)
    json.dump(dict(best=best, steps=step, time_total_s=time.time() - t0), open(rundir / "summary.json", "w"), indent=1)
    print(f"[{cfg.stage}/{cfg.run}] готово: {step} шагов за {time.time() - t0:.0f} с; лучший score {best['value']:.3f} на шаге {best['step']} → {rundir}", flush=True)
    return best


def turing(log: tblog.TBLog, xr: np.ndarray, xf: np.ndarray, step: int, save: Path, seed: int):
    """16 плиток (канал O) вперемешку реальные/рендер без подписей; ключ — в TEXT."""
    import matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed); n = min(8, len(xr))
    xr = np.asarray(xr, np.float32); xr = xr / 255.0 if xr.max() > 1.5 else xr      # реальное — uint8 (ошибка 2026-09-06: плитки заливались)
    tiles = [("R", xr[i, 0]) for i in range(n)] + [("F", xf[i, 0]) for i in range(n)]
    order = rng.permutation(len(tiles))
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    for k, o in enumerate(order):
        a = ax[k // 4, k % 4]; a.imshow(tiles[o][1], origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        a.set_title(str(k + 1), fontsize=9); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(save, dpi=80); log.w.add_figure("turing", fig, step); plt.close(fig)
    key = ", ".join(f"{k + 1}:{'реальное' if tiles[o][0] == 'R' else 'рендер'}" for k, o in enumerate(order))
    log.text("turing_key", f"шаг {step}: {key}", step)
    (save.parent / f"turing_{step:06d}_key.txt").write_text(key + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(GanConfig):
        if f.type is bool or f.type == "bool":
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    train_gan(GanConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
