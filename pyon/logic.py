# -*- coding: utf-8 -*-
"""
logic.py — дифференцируемые релаксации аксиом онтологии («логическая голова», Э2 §4-bis).

Три слоя (Э2 §4-bis.1): заземление — логиты сети → мягкие ридауты (масса класса в столбце,
присутствие π, soft-argmax положения, logsumexp-максимум частоты присутствия); атомы —
нарушение v = (a − b + margin)/τ и штраф pen(v); связки — сумма штрафов (продукт-t-норма
в лог-пространстве), квантор «по столбцам/образцам» — взвешенное присутствиями среднее.

Релаксации (Э2 §4-bis.2): "hinge" = max(0, v) (семейство Лукасевича); "lognorm" =
softplus_β(v) = −(1/β)·log σ(−βv), β = 2 (продукт-t-норма). Литература: van Krieken et al.
2022, Badreddine et al. 2022 (LTN) — lit/tensor_logic/.

ПРАВИЛА ДЕТАЧА (Э2 §4-bis.3, обязательны): (1) гейты присутствия детачатся; (2) опорная
сторона неравенства детачится — градиент только в нарушителя (`order_penalty`). Прототип
(iono_study.ipynb, ячейки 7 и 29) применял правило (2) к P4/S1/S2, но НЕ к P1/P3;
здесь правило применено единообразно (`detach_support=True`), прототипное поведение —
`detach_support=False` (абляция E1). Решение ревизии 2026-09-04, см. DIARY.

Пропозиции (Э2 §4-bis.4; допуски — рабочие значения прототипа, подлежат калибровке):
  ВЗ, vertical_logic:  P1 порядок высот E<F1<F2 (форма S4; 20/10 км, τ=100 км),
                       P2 связность следа (РД §7.3.7; 60 км), P3 не-кратник: парная форма в столбце (V3),
                       P4 порядок частот foE<foF1<foF2 (Q1; Es исключён — ревью Ф-01)
  НЗ, oblique_logic:   S1 МПЧ(2F2) ≤ МПЧ(1F2) (форма S1; 0.02), S2 P′(2F2) ≥ P′(1F2)+150 км (S2),
                       P2 связность (3 строки)
Возвращают (total: скаляр, components: {имя: Tensor[B]}) — покомпонентный и пообразцовый
L_logic для TensorBoard (Э3 §3.3: скаляры по компонентам, гистограмма распределения).
"""
from __future__ import annotations

import torch
import torch.nn.functional as Fn

from pyon import canon
from pyon import oblique_synth as obs

MASS0 = 3.0            # бинов в столбце для присутствия π = 1
BETA_FMAX = 12.0       # жёсткость logsumexp-максимума частоты
P3_RATIO = 1.8         # P3: кратник — масса F2 выше 1.8 × высоты другой массы F2 в том же столбце
SOFTPLUS_BETA = 2.0    # β релаксации "lognorm"


def penalty(variant: str):
    """Штрафная функция релаксации по имени варианта."""
    if variant == "hinge":
        return Fn.relu
    if variant in ("lognorm", "softplus"):
        return lambda u: Fn.softplus(u, beta=SOFTPLUS_BETA)
    raise ValueError(f"неизвестная релаксация: {variant}")


def order_penalty(a, b_support, margin: float, pen, detach: bool = True):
    """Атом «a ≤ b − margin»: штраф pen(a − b + margin); опора b детачится (правило 2)."""
    b = b_support.detach() if detach else b_support
    return pen(a - b + margin)


def soft_readouts(logits: torch.Tensor, pos_axis):
    """Заземление. logits [B, C, H, F] → pres [B, C−1, F], pos_soft [B, C−1, F],
    pos_mean [B, C−1], fmax [B, C−1] (нормированная частота 0..1), present [B, C−1].
    pos_axis — координаты строк (км для ВЗ, 0..1 для НЗ); фон (класс 0) исключён."""
    p = logits.softmax(1)
    q = p[:, 1:]
    mass = q.sum(2)
    pres = (mass / MASS0).clamp(0, 1)
    pa = torch.as_tensor(pos_axis, dtype=logits.dtype, device=logits.device).view(1, 1, -1, 1)
    pos_soft = (q * pa).sum(2) / (mass + 1e-6)
    w = pres / (pres.sum(-1, keepdim=True) + 1e-6)
    pos_mean = (pos_soft * w).sum(-1)
    fn = torch.linspace(0, 1, pres.shape[-1], device=logits.device, dtype=logits.dtype)
    fmax = torch.logsumexp(torch.log(pres + 1e-6) + BETA_FMAX * fn, -1) / BETA_FMAX
    return pres, pos_soft, pos_mean, fmax, pres.mean(-1)


def pairwise_ratio_penalty(logits, ci: int, pos_axis, ratio: float, scale: float, pen,
                           detach_support: bool = True):
    """Штраф «в столбце нет пары масс класса ci на положениях h < h′ с h′ > ratio·h»:
    Σ_{h,h′} q(h)·q(h′)·pen((h′ − ratio·h)/scale) / (Σ_h q(h))² → [B, F] — средний штраф по
    паре пикселей класса в столбце (ограничен max pen; у необученной сети с размазанной массой
    не взрывается). Опора q(h) и знаменатель детачатся: градиент только вниз на верхнюю массу
    (нарушителя), «разбавить» знаменатель фантомной массой нельзя."""
    q = logits.softmax(1)[:, ci]                                     # [B, H, F]
    pa = torch.as_tensor(pos_axis, dtype=logits.dtype, device=logits.device)
    M = pen((pa.view(1, -1) - ratio * pa.view(-1, 1)) / scale)      # [H(h), H(h′)]
    q_sup = q.detach() if detach_support else q
    mass2 = (q_sup.sum(1) ** 2 + 1e-6)                               # [B, F], детачен вместе с q_sup
    return torch.einsum("bhf,hk,bkf->bf", q_sup, M, q) / mass2


def continuity(pres, pos_soft, gap: float, scale: float, pen):
    """P2: |pos(f_{j+1}) − pos(f_j)| ≤ gap между соседними столбцами; веса присутствия детачатся."""
    dh = (pos_soft[:, :, 1:] - pos_soft[:, :, :-1]).abs()
    wc = (pres[:, :, 1:] * pres[:, :, :-1]).detach()
    return (wc * pen((dh - gap) / scale)).sum((1, 2)) / (wc.sum((1, 2)) + 1e-6)


def vertical_logic(logits: torch.Tensor, variant: str = "hinge", detach_support: bool = True):
    """L_logic ВЗ: P1 + P2 + P3 + P4 (классы canon.CLASSES, высоты canon.h_axis в км)."""
    pen = penalty(variant)
    pres, h_soft, h_mean, fmax, present = soft_readouts(logits, canon.h_axis)
    i = {c: canon.CLASSES.index(c) - 1 for c in ("F2", "F1", "E")}

    def g(a, b):
        return (present[:, i[a]] * present[:, i[b]]).detach()

    def hp(lo, hi, margin_km):   # P1: слой lo ниже слоя hi
        return order_penalty(h_mean[:, i[lo]] / 100, h_mean[:, i[hi]] / 100, margin_km / 100, pen, detach_support)

    P1 = g("E", "F2") * hp("E", "F2", 20) + g("F1", "F2") * hp("F1", "F2", 10) + g("E", "F1") * hp("E", "F1", 10)
    P2 = continuity(pres, h_soft, gap=60.0, scale=100.0, pen=pen)
    # P3 (не-кратник, форма V3), парное заземление (ревизия 2026-09-04, Э2 §4-bis.4):
    # в столбце f нет двух масс F2 на высотах h < h′ с h′ > 1.8·h:
    #   P3(f) = Σ_{h,h′} q(h)·q(h′)·relu(h′ − 1.8·h)/100 / (Σq)²,  опора q(h) и знаменатель детачатся.
    # Почему не прототипный soft-argmax + min по столбцам: (а) на ВЗ кратник лежит в ТЕХ ЖЕ
    # столбцах, что основной след — среднее по столбцу уходило в середину и P3 его не видел;
    # (б) min по всем столбцам брал псевдовысоту остаточной массы softmax в пустых столбцах.
    # Парная форма линейна по q (без экспонент logsumexp — остаточная масса не усиливается,
    # подавляется квадратично), кратник виден в любом столбце, опора hmin не нужна.
    P3 = pairwise_ratio_penalty(logits, canon.CLASSES.index("F2"), canon.h_axis, P3_RATIO, 100.0, pen,
                                detach_support) * pres[:, i["F2"]].detach()
    P3 = P3.mean(-1)

    def fp(lo, hi):              # P4: fmax(lo) < fmax(hi)
        return order_penalty(fmax[:, i[lo]], fmax[:, i[hi]], 0.01, pen, True)

    P4 = g("E", "F2") * fp("E", "F2") + g("E", "F1") * fp("E", "F1") + g("F1", "F2") * fp("F1", "F2")
    comps = {"P1": P1, "P2": P2, "P3": P3, "P4": P4}                  # каждый — [B] (по образцам)
    return sum(c.mean() for c in comps.values()), comps


def oblique_logic(logits: torch.Tensor, variant: str = "lognorm", detach_support: bool = True):
    """L_logic НЗ: S1 + S2 + P2 (классы obs.OB_CLASSES; групповой путь нормирован 0..1)."""
    pen = penalty(variant)
    pos_axis = torch.linspace(0, 1, obs.NP)
    pres, d_soft, d_mean, fmax, present = soft_readouts(logits, pos_axis)
    iF2, iMH = obs.OB_CLASSES.index("F2") - 1, obs.OB_CLASSES.index("MH") - 1
    g = (present[:, iF2] * present[:, iMH]).detach()
    # S1: МПЧ(MH) ≤ МПЧ(F2) — нарушитель MH, опора F2
    S1 = g * order_penalty(fmax[:, iMH], fmax[:, iF2], 0.02, pen, detach_support)
    # S2: P′(MH) ≥ P′(F2) + 150 км  ⇔  −P′(MH) ≤ −P′(F2) − margin — нарушитель MH, опора F2
    S2 = g * order_penalty(-d_mean[:, iMH], -d_mean[:, iF2], 150.0 / (obs.P_MAX - obs.P_MIN), pen, detach_support)
    P2 = continuity(pres, d_soft, gap=3.0 / obs.NP, scale=1.0, pen=pen)
    comps = {"S1": S1, "S2": S2, "P2": P2}                            # каждый — [B]
    return sum(c.mean() for c in comps.values()), comps


def logits_from_mask(mask, n_classes: int, scale: float = 30.0) -> torch.Tensor:
    """Маска [B, H, F] (int) → «уверенные» логиты [B, C, H, F] (one-hot·scale): для тестов и
    для оценки L_logic самой разметки (учителя/меток).

    scale ≥ 30: при scale = 10 остаточная масса softmax в пустых столбцах (≈4.5e-5 на пиксель,
    ≈0.006 на столбец) через logsumexp-максимум с β = 12 сдвигает fmax к верхней границе —
    столбец на f = 1 с π = 0.002 весит как столбец следа на f = 0.5 с π = 1. Та же
    чувствительность есть у обученной сети (остаточная масса 1e-3…1e-2) — кандидат на
    переработку заземления fmax в E1 (порог присутствия / температура), см. DIARY 2026-09-04."""
    m = torch.as_tensor(mask).long()
    return Fn.one_hot(m, n_classes).permute(0, 3, 1, 2).float() * scale
