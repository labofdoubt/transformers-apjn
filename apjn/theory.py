from __future__ import annotations

import math

import numpy as np

from .config import MeanFieldConfig

_EPS = 1e-12


def build_mean_field_cfg_for_vit_base(
    depth: int | None = None,
    *,
    attn_mult: float = 1.0,
    mlp_mult: float = 1.0,
):
    del depth
    scale = math.sqrt(768.0 / 1024.0)
    return MeanFieldConfig(
        sigma_w1=0.64 * scale * float(mlp_mult),
        sigma_w2=1.28 * scale * float(mlp_mult),
        sigma_o=0.64 * scale * float(attn_mult),
        sigma_v=0.64 * scale * float(attn_mult),
        sigma_a=0.64 * 0.64 * 768.0 / 1024.0,
    )


def kappa_relu_np(rho):
    rho = np.clip(rho, -1.0, 1.0)
    return (1.0 / (2.0 * np.pi)) * (
        np.sqrt(np.maximum(0.0, 1.0 - rho**2)) + rho * (np.pi - np.arccos(rho))
    )


def tilde_q_erf_np(q, alpha):
    x = (2.0 * alpha**2 * q) / (1.0 + 2.0 * alpha**2 * q)
    return (2.0 / np.pi) * np.arcsin(np.clip(x, -1.0, 1.0))


def tilde_p_erf_np(q, p, alpha):
    x = (2.0 * alpha**2 * p) / (1.0 + 2.0 * alpha**2 * q)
    return (2.0 / np.pi) * np.arcsin(np.clip(x, -1.0, 1.0))


def simulate_recursions_full(
    num_layers: int,
    p0: float,
    n_tokens: int,
    mode: str,
    *,
    alpha: float = 1.0,
    sigma_w1: float = 0.64,
    sigma_w2: float = 1.28,
    sigma_o: float = 0.64,
    sigma_v: float = 0.64,
    sigma_a: float = 0.64 * 0.64,
    q0: float = 1.0,
):
    depth = int(num_layers)
    q = np.zeros(depth + 1, dtype=float)
    p = np.zeros(depth + 1, dtype=float)
    q_hat = np.zeros(depth, dtype=float)
    p_hat = np.zeros(depth, dtype=float)
    q_hat_half = np.zeros(depth, dtype=float)
    p_hat_half = np.zeros(depth, dtype=float)
    kappa_prime = np.zeros(depth, dtype=float)
    j_forward = np.zeros(depth + 1, dtype=float)
    k_forward = np.zeros(depth + 1, dtype=float)

    q[0] = float(q0)
    p[0] = float(p0)
    j_forward[0] = 1.0
    att_scale = float(sigma_o) ** 2 * float(sigma_v) ** 2
    mlp_scale = float(sigma_w1) ** 2 * float(sigma_w2) ** 2
    n_tokens = float(n_tokens)

    for layer in range(depth):
        ql = float(q[layer])
        pl = float(p[layer])

        if mode.lower() == "erf":
            q_hat[layer] = (4.0 * alpha**2 / np.pi) / math.sqrt(max(1.0 + 4.0 * alpha**2 * ql, _EPS))
            p_hat[layer] = (4.0 * alpha**2 / np.pi) / math.sqrt(
                max((1.0 + 2.0 * alpha**2 * ql) ** 2 - 4.0 * alpha**4 * pl**2, _EPS)
            )
            up = float(tilde_p_erf_np(ql, pl, alpha))
            q_half = ql + att_scale * up
            p_half = pl + att_scale * up
            q_hat_half[layer] = (4.0 * alpha**2 / np.pi) / math.sqrt(max(1.0 + 4.0 * alpha**2 * q_half, _EPS))
            p_hat_half[layer] = (4.0 * alpha**2 / np.pi) / math.sqrt(
                max((1.0 + 2.0 * alpha**2 * q_half) ** 2 - 4.0 * alpha**4 * p_half**2, _EPS)
            )
            u_half = float(tilde_q_erf_np(q_half, alpha))
            v_half = float(tilde_p_erf_np(q_half, p_half, alpha))
            rho_half = float(np.clip(v_half / max(u_half, _EPS), -1.0, 1.0))
            q[layer + 1] = q_half + 0.5 * mlp_scale * u_half
            p[layer + 1] = p_half + mlp_scale * u_half * kappa_relu_np(rho_half)
        elif mode.lower() == "layernorm":
            rho = float(np.clip(pl / max(ql, _EPS), -1.0, 1.0))
            q_hat[layer] = 1.0 / max(ql, _EPS)
            p_hat[layer] = 1.0 / max(ql, _EPS)
            q_half = ql + att_scale * rho
            p_half = pl + att_scale * rho
            q_hat_half[layer] = 1.0 / max(q_half, _EPS)
            p_hat_half[layer] = 1.0 / max(q_half, _EPS)
            rho_half = float(np.clip(p_half / max(q_half, _EPS), -1.0, 1.0))
            q[layer + 1] = q_half + 0.5 * mlp_scale
            p[layer + 1] = p_half + mlp_scale * kappa_relu_np(rho_half)
        else:
            raise ValueError("mode must be 'erf' or 'layernorm'.")

        kappa_prime[layer] = 0.25 + (1.0 / (2.0 * np.pi)) * np.arcsin(np.clip(rho_half, -1.0, 1.0))
        j_half = (1.0 + att_scale * q_hat[layer] / n_tokens) * j_forward[layer] + att_scale * p_hat[layer] * k_forward[layer]
        k_half = (1.0 + att_scale * p_hat[layer]) * k_forward[layer] + (att_scale * q_hat[layer] / n_tokens) * j_forward[layer]
        k_forward[layer + 1] = (1.0 + mlp_scale * kappa_prime[layer] * p_hat_half[layer]) * k_half
        j_forward[layer + 1] = (1.0 + 0.5 * mlp_scale * q_hat_half[layer]) * j_half

    j = np.zeros(depth + 1, dtype=float)
    k = np.zeros(depth + 1, dtype=float)
    j[depth] = 1.0
    for layer in range(depth - 1, -1, -1):
        j_half = (1.0 + 0.5 * mlp_scale * q_hat_half[layer]) * j[layer + 1]
        k_half = (1.0 + mlp_scale * kappa_prime[layer] * p_hat_half[layer]) * k[layer + 1]
        j[layer] = (1.0 + att_scale * q_hat[layer] / n_tokens) * j_half + att_scale * q_hat[layer] * k_half
        k[layer] = (1.0 + att_scale * p_hat[layer]) * k_half + (att_scale * p_hat[layer] / n_tokens) * j_half

    return {
        "q": q,
        "p": p,
        "p_over_q": p / np.maximum(q, _EPS),
        "K": k,
        "K_forward": k_forward,
        "J": j,
        "J_forward": j_forward,
    }


def compute_theory_bundle(
    *,
    depth: int,
    n_tokens: int,
    q0: float,
    p0: float,
    alphas,
    mean_field_cfg: MeanFieldConfig,
):
    preln = simulate_recursions_full(
        num_layers=int(depth),
        q0=float(q0),
        p0=float(p0),
        n_tokens=int(n_tokens),
        mode="layernorm",
        sigma_w1=float(mean_field_cfg.sigma_w1),
        sigma_w2=float(mean_field_cfg.sigma_w2),
        sigma_o=float(mean_field_cfg.sigma_o),
        sigma_v=float(mean_field_cfg.sigma_v),
        sigma_a=float(mean_field_cfg.sigma_a),
    )
    derf = {}
    for alpha in np.asarray(alphas, dtype=float):
        derf[float(alpha)] = simulate_recursions_full(
            num_layers=int(depth),
            q0=float(q0),
            p0=float(p0),
            n_tokens=int(n_tokens),
            mode="erf",
            alpha=float(alpha),
            sigma_w1=float(mean_field_cfg.sigma_w1),
            sigma_w2=float(mean_field_cfg.sigma_w2),
            sigma_o=float(mean_field_cfg.sigma_o),
            sigma_v=float(mean_field_cfg.sigma_v),
            sigma_a=float(mean_field_cfg.sigma_a),
        )
    return {
        "depth": int(depth),
        "n_tokens": int(n_tokens),
        "layers": np.arange(int(depth) + 1, dtype=int),
        "preln": preln,
        "derf": derf,
    }
