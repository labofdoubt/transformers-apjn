from __future__ import annotations

import math

import torch
import torch.nn as nn

from dynamic_tanh import DynamicErf, convert_ln_to_derf
from timm.models import create_model

from .config import ModelConfig
from .runtime import DEVICE

_EPS = 1e-12


def replace_gelu_with_relu(module: nn.Module, *, inplace_relu: bool = False) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, nn.ReLU(inplace=inplace_relu))
        else:
            replace_gelu_with_relu(child, inplace_relu=inplace_relu)
    return module


def build_vit(model_cfg: ModelConfig, *, use_derf: bool):
    model = create_model(
        model_cfg.model_name,
        pretrained=False,
        num_classes=int(model_cfg.num_classes),
        global_pool="avg",
        drop_path_rate=0.0,
        depth=int(model_cfg.depth),
    )
    if model_cfg.replace_gelu_with_relu:
        replace_gelu_with_relu(model, inplace_relu=model_cfg.inplace_relu)
    if use_derf:
        model = convert_ln_to_derf(model)
    model.eval().to(DEVICE)
    return model


def scale_vit_mlp_and_value_attn_init_std(
    model: nn.Module,
    *,
    mlp_multiplier: float = 1.0,
    attn_multiplier: float = 1.0,
):
    mlp_multiplier = float(mlp_multiplier)
    attn_multiplier = float(attn_multiplier)
    if mlp_multiplier <= 0.0 or attn_multiplier <= 0.0:
        raise ValueError("Initialization multipliers must be positive.")

    with torch.no_grad():
        for block in model.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is not None and abs(mlp_multiplier - 1.0) > 1e-12:
                for attr in ("fc1", "fc2"):
                    layer = getattr(mlp, attr, None)
                    if isinstance(layer, nn.Linear):
                        layer.weight.mul_(mlp_multiplier)

            attn = getattr(block, "attn", None)
            if attn is None or abs(attn_multiplier - 1.0) <= 1e-12:
                continue
            qkv = getattr(attn, "qkv", None)
            proj = getattr(attn, "proj", None)
            if isinstance(qkv, nn.Linear):
                dim = qkv.weight.shape[0] // 3
                qkv.weight[2 * dim :].mul_(attn_multiplier)
                if qkv.bias is not None:
                    qkv.bias[2 * dim :].mul_(attn_multiplier)
            if isinstance(proj, nn.Linear):
                proj.weight.mul_(attn_multiplier)
                if proj.bias is not None:
                    proj.bias.mul_(attn_multiplier)
    return model


def set_all_derf_alpha_(model: nn.Module, alpha_value: float) -> int:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, DynamicErf):
                module.alpha_init_value = float(alpha_value)
                module.alpha.fill_(float(alpha_value))
                count += 1
    return count


def get_vit_seq_len_and_dim(model: nn.Module) -> tuple[int, int]:
    embed_dim = int(getattr(model, "embed_dim"))
    n_patches = int(getattr(model.patch_embed, "num_patches"))
    has_cls = hasattr(model, "cls_token") and model.cls_token is not None
    return n_patches + (1 if has_cls else 0), embed_dim


def _random_orthogonal(d: int, device, dtype, generator: torch.Generator):
    mat = torch.randn(d, d, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(mat, mode="reduced")
    diag = torch.sign(torch.diagonal(r, 0))
    diag[diag == 0] = 1.0
    return q * diag


def make_equiangular_tokens(
    N: int,
    d: int,
    q0: float,
    p0: float,
    *,
    device,
    dtype,
    seed: int,
    random_rotate: bool = True,
):
    if N < 2:
        raise ValueError("N must be at least 2.")
    rho = float(p0) / float(q0)
    rho_min = -1.0 / (int(N) - 1)
    if rho < rho_min - 1e-8 or rho >= 1.0:
        raise ValueError(f"Need p0 / q0 in [{rho_min}, 1). Got {rho}.")

    lam1 = d * (1.0 - rho)
    lam2 = d * (1.0 + (N - 1) * rho)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))

    u = torch.ones(N, 1, device=device, dtype=dtype) / math.sqrt(N)
    mat = torch.randn(N, N - 1, device=device, dtype=dtype, generator=gen)
    mat = mat - u @ (u.T @ mat)
    v, _ = torch.linalg.qr(mat, mode="reduced")

    parts = []
    if lam1 > _EPS:
        parts.append(math.sqrt(lam1) * v)
    if lam2 > _EPS:
        parts.append(math.sqrt(lam2) * u)
    basis = torch.cat(parts, dim=1)
    rank = int(basis.shape[1])
    if d < rank:
        raise ValueError(f"Need d >= rank={rank}, got d={d}.")

    x = torch.cat([basis, torch.zeros(N, d - rank, device=device, dtype=dtype)], dim=1)
    if random_rotate:
        x = x @ _random_orthogonal(d, device=device, dtype=dtype, generator=gen)
    return math.sqrt(float(q0)) * x


def make_apjn_equiangular_block0_batch(
    *,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    q0: float,
    p0: float,
    seed: int,
    random_rotate: bool,
):
    x = make_equiangular_tokens(
        N=int(seq_len),
        d=int(embed_dim),
        q0=float(q0),
        p0=float(p0),
        device=DEVICE,
        dtype=torch.float32,
        seed=int(seed),
        random_rotate=bool(random_rotate),
    ).detach()
    return x.unsqueeze(0).expand(int(batch_size), -1, -1).clone().detach()


def _extract_vit_pre_blocks_state_dict(model: nn.Module):
    prefixes = ("patch_embed.", "norm_pre.")
    exact_keys = {"cls_token", "reg_token", "pos_embed"}
    out = {}
    for key, value in model.state_dict().items():
        if key in exact_keys or any(key.startswith(prefix) for prefix in prefixes):
            out[key] = value.detach().cpu().clone()
    return out


def _load_partial_state_dict_(model: nn.Module, partial_state_dict):
    if not partial_state_dict:
        return model
    state = model.state_dict()
    state.update({
        key: value.to(device=state[key].device, dtype=state[key].dtype)
        for key, value in partial_state_dict.items()
    })
    model.load_state_dict(state, strict=True)
    return model


def capture_X_list_and_logits(
    model: nn.Module,
    images: torch.Tensor,
    *,
    block0_input_override: torch.Tensor | None = None,
):
    depth = len(model.blocks)
    x0_ref = {"x0": None}
    outputs = [None] * depth
    hooks = []

    def pre_hook_block0(_module, inputs):
        x_in = inputs[0]
        if block0_input_override is not None:
            x0 = block0_input_override.to(device=x_in.device, dtype=x_in.dtype)
            if tuple(x0.shape) != tuple(x_in.shape):
                raise ValueError(f"Expected override shape {tuple(x_in.shape)}, got {tuple(x0.shape)}.")
            x0 = x0.detach().clone().requires_grad_(True)
        else:
            x0 = x_in.detach().clone().requires_grad_(True)
        x0_ref["x0"] = x0
        return (x0,)

    def hook_block_i(index: int):
        def _hook(_module, _inputs, output):
            outputs[index] = output[0] if isinstance(output, (tuple, list)) else output
        return _hook

    hooks.append(model.blocks[0].register_forward_pre_hook(pre_hook_block0))
    for index in range(depth):
        hooks.append(model.blocks[index].register_forward_hook(hook_block_i(index)))

    try:
        logits = model(images.to(DEVICE, dtype=torch.float32, non_blocking=False))
        x0 = x0_ref["x0"]
        if x0 is None or any(x is None for x in outputs):
            raise RuntimeError("Failed to capture transformer block activations.")
        return [x0] + outputs, logits
    finally:
        for hook in hooks:
            hook.remove()
