"""Synthetic gradients with a known mean, covariance and prompt effect (DIRECTIVE-VLM-VAD-GRAD-001-002 D7).

g_i = mu_class(i) + b_p + w_i with b_p ~ N(0, s_b^2 I) shared by the siblings of prompt p and w_i ~ N(0, s_w^2 I).
Per class: m = ||mu||^2, sigma_b^2 = d s_b^2, sigma_w^2 = d s_w^2, E2 = m + sigma_b^2 + sigma_w^2. Defaults: class C
m = 1, q = 0.01, ICC 0.8; class I m = 3. The log-probability gradient is h_i = r_i nu + e_i, so the prompt-centered sum
sum_i (r_i - mean r) h_i has the direction nu, and a_X = cos(-mu_X, nu) (R2 sign: -mu_X is the step direction). Flags are random and independent of the rest,
so a class x flag subset has the truth of its class. D is a random positive diagonal; the truth in the D metric uses
mu^T D mu and tr(D). The vectors go through accum.finish_prompt and accum.gram_file, as in moments.py.
"""

from __future__ import annotations

import math

import torch

import accum

PROMPTS, SIBLINGS = 375, 8


def basis(d: int, gen: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, 3, generator=gen, dtype=torch.float64))
    return q.T  # 3 orthonormal rows


def make_truth(d: int = 2000, m_c: float = 1.0, m_i: float = 3.0, q_c: float = 0.01, icc: float = 0.8,
               cos_ci: float = 0.3, a_c: float = 0.5, a_i: float = -0.2, seed: int = 1) -> dict:
    gen = torch.Generator().manual_seed(seed)
    e = basis(d, gen)
    mu_c = math.sqrt(m_c) * e[0]
    mu_i = math.sqrt(m_i) * (cos_ci * e[0] + math.sqrt(1 - cos_ci ** 2) * e[1])
    # nu with cos(mu_C, nu) = a_c and cos(mu_I, nu) = a_i, in the span of e0, e1, e2.
    x = a_c
    y = (a_i - cos_ci * x) / math.sqrt(1 - cos_ci ** 2)
    nu = x * e[0] + y * e[1] + math.sqrt(max(0.0, 1 - x * x - y * y)) * e[2]
    sigma2 = m_c / q_c - m_c
    dvec = 0.5 + 1.5 * torch.rand(d, generator=gen, dtype=torch.float64)
    return {"d": d, "mu": {"C": mu_c, "I": mu_i}, "nu": nu, "s_b2": icc * sigma2 / d, "s_w2": (1 - icc) * sigma2 / d,
            "D": dvec, "a": {"C": a_c, "I": a_i}, "cos_ci": cos_ci}


def true_values(truth: dict, cls: str, metric: str) -> dict:
    mu, dvec = truth["mu"][cls], truth["D"]
    if metric == "raw":
        m, tr = float(mu @ mu), float(truth["d"])
    else:
        m, tr = float(mu @ (dvec * mu)), float(dvec.sum())
    sb2, sw2 = truth["s_b2"] * tr, truth["s_w2"] * tr
    e2 = m + sb2 + sw2
    return {"m": m, "E2": e2, "sigma_b2": sb2, "sigma_w2": sw2, "sigma2": sb2 + sw2, "q": m / e2,
            "ICC": sb2 / (sb2 + sw2), "s_same": m + sb2}


def true_cross(truth: dict, metric: str) -> dict:
    w = torch.ones_like(truth["D"]) if metric == "raw" else truth["D"]
    mu_c, mu_i, nu = truth["mu"]["C"], truth["mu"]["I"], truth["nu"]

    def cos(a, b):
        return float(a @ (w * b)) / math.sqrt(float(a @ (w * a)) * float(b @ (w * b)))

    return {"cos_mu_C_mu_I|pooled": cos(mu_c, mu_i), "a|C": cos(-mu_c, nu), "a|I": cos(-mu_i, nu)}


def split(v: torch.Tensor) -> list[torch.Tensor]:
    """A vector as two parameter tensors (the code works on lists of tensors)."""
    n = v.numel() // 3
    return [v[:n].float().reshape(n, 1), v[n:].float()]


def draw(truth: dict, seed: int, prompts: int = PROMPTS) -> dict:
    """One synthetic Gram file."""
    gen = torch.Generator().manual_seed(1000 + seed)
    d = truth["d"]
    sb, sw = math.sqrt(truth["s_b2"]), math.sqrt(truth["s_w2"])
    dlist = split(truth["D"])
    store = accum.Store([t.shape for t in dlist])
    buf = [torch.zeros(t.shape) for t in dlist]
    a_buf = [torch.zeros(t.shape) for t in dlist]
    folds = accum.fold_of(list(range(prompts)))
    records = []
    for p in range(prompts):
        acc = float(torch.rand(1, generator=gen))  # prompt accuracy, uniform on (0, 1)
        labels = [int(x) for x in (torch.rand(SIBLINGS, generator=gen) < acc).tolist()]
        flags = [accum.FLAGS[int(i)] for i in torch.multinomial(torch.tensor([0.4, 0.4, 0.2]), SIBLINGS, True, generator=gen)]
        b = sb * torch.randn(d, generator=gen, dtype=torch.float64)
        grads, hs = [], []
        for r in labels:
            g = truth["mu"]["C" if r else "I"] + b + sw * torch.randn(d, generator=gen, dtype=torch.float64)
            h = r * truth["nu"] + 0.1 * torch.randn(d, generator=gen, dtype=torch.float64)
            grads.append(split(g))
            hs.append(split(h))
        weights = accum.a_weights(labels)
        a_p = accum.weighted_sum(hs, weights, a_buf) if any(weights) else None
        records.append(accum.finish_prompt(store, p, folds[p], [p * SIBLINGS + s for s in range(SIBLINGS)], labels,
                                           flags, grads, dlist, a_p, buf))
    return accum.gram_file(store, dlist, "cpu", records, f"synthetic-{seed}")
