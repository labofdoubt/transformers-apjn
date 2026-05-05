from __future__ import annotations

import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from datasets import build_dataset
from dynamic_tanh import DynamicErf, convert_ln_to_derf
from timm.models import create_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_CIFAR_DATASET_CACHE: dict[tuple[int, int], object] = {}
_CIFAR_STREAM_CACHE: dict[tuple[int, int, int], dict] = {}
_EPS = 1e-12


@dataclass
class ModelConfig:
    model_name: str = "vit_base_patch16_224"
    depth: int = 12
    num_classes: int = 100
    img_size: int = 224
    replace_gelu_with_relu: bool = True
    inplace_relu: bool = False
    seed: int = 0


@dataclass
class MeanFieldConfig:
    sigma_w1: float = 0.64
    sigma_w2: float = 1.28
    sigma_o: float = 0.64
    sigma_v: float = 0.64
    sigma_a: float = 0.64 * 0.64


@dataclass
class APJNCifarConfig:
    input_source: str = "cifar"
    batch_size: int = 1
    cifar_batch_seed: int = 0
    cifar_batch_draw_index: int = 0
    cifar_std_threshold: float | None = None
    cifar_max_epochs_to_search: int = 20
    equiangular_q0: float = 1.0
    equiangular_p0: float = 0.0
    equiangular_seed: int = 0
    equiangular_random_rotate: bool = True
    apjn_layers: tuple[int, ...] = (4, 8, 12)
    direct_layers: tuple[int, ...] = (4, 8, 12)
    direct_source_block: int = 0
    alphas: tuple[float, ...] = (0.5, 1.0)
    j_num_draws: int = 10
    j_normalize_by: str = "Y"
    num_model_inits: int = 1
    attn_mult: float = 1.0
    mlp_mult: float = 1.0


def cfg_to_dict(cfg):
    out = asdict(cfg)
    for key, value in list(out.items()):
        if isinstance(value, tuple):
            out[key] = list(value)
    return out


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clear_cifar_experiment_cache():
    _CIFAR_DATASET_CACHE.clear()
    _CIFAR_STREAM_CACHE.clear()


def _make_cifar_dataset_key(*, img_size: int, num_classes: int):
    return (int(img_size), int(num_classes))


def _get_cached_cifar_dataset(*, img_size: int, num_classes: int):
    key = _make_cifar_dataset_key(img_size=img_size, num_classes=num_classes)
    if key in _CIFAR_DATASET_CACHE:
        return _CIFAR_DATASET_CACHE[key]

    args = SimpleNamespace(
        data_set="CIFAR",
        data_path="/tmp/cifar100",
        eval_data_path=None,
        nb_classes=int(num_classes),
        input_size=int(img_size),
        imagenet_default_mean_and_std=True,
        color_jitter=0.4,
        aa="rand-m9-mstd0.5-inc1",
        train_interpolation="bicubic",
        reprob=0.25,
        remode="pixel",
        recount=1,
        crop_pct=None,
    )
    dataset, _ = build_dataset(is_train=True, args=args)
    _CIFAR_DATASET_CACHE[key] = dataset
    return dataset


def _make_cifar_loader(*, dataset, batch_size: int, loader_seed: int):
    gen = torch.Generator()
    gen.manual_seed(int(loader_seed))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )


def _get_or_create_cifar_stream(
    *,
    batch_size: int,
    img_size: int,
    num_classes: int,
    loader_seed: int,
):
    key = (int(batch_size), int(img_size), int(num_classes), int(loader_seed))
    if key in _CIFAR_STREAM_CACHE:
        return _CIFAR_STREAM_CACHE[key]

    dataset = _get_cached_cifar_dataset(img_size=img_size, num_classes=num_classes)
    loader = _make_cifar_loader(dataset=dataset, batch_size=batch_size, loader_seed=loader_seed)
    stream = {"loader": loader, "iterator": iter(loader), "accepted_batches": [], "epoch": 0}
    _CIFAR_STREAM_CACHE[key] = stream
    return stream


def get_cifar_batch_no_filter(
    batch_size: int,
    img_size: int,
    num_classes: int,
    *,
    loader_seed: int,
    draw_index: int,
):
    stream = _get_or_create_cifar_stream(
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        loader_seed=loader_seed,
    )
    accepted = stream["accepted_batches"]
    while len(accepted) <= int(draw_index):
        try:
            samples, targets = next(stream["iterator"])
        except StopIteration:
            stream["epoch"] += 1
            stream["iterator"] = iter(stream["loader"])
            samples, targets = next(stream["iterator"])
        accepted.append((samples.detach().cpu().clone(), targets.detach().cpu().clone()))

    samples, targets = accepted[int(draw_index)]
    return samples.clone(), targets.clone(), {
        "loader_seed": int(loader_seed),
        "draw_index": int(draw_index),
        "filtering": "none",
    }


def get_cifar_batch(
    batch_size: int,
    img_size: int,
    num_classes: int,
    *,
    loader_seed: int,
    draw_index: int,
    std_threshold: float,
    max_epochs_to_search: int,
):
    stream = _get_or_create_cifar_stream(
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        loader_seed=loader_seed,
    )
    accepted = stream["accepted_batches"]
    while len(accepted) <= int(draw_index):
        if int(stream["epoch"]) >= int(max_epochs_to_search):
            raise RuntimeError(
                f"Could not find CIFAR batch draw_index={draw_index} within {max_epochs_to_search} epochs."
            )
        try:
            samples, targets = next(stream["iterator"])
        except StopIteration:
            stream["epoch"] += 1
            stream["iterator"] = iter(stream["loader"])
            continue
        if float(samples.std()) <= float(std_threshold):
            continue
        accepted.append((samples.detach().cpu().clone(), targets.detach().cpu().clone()))

    samples, targets = accepted[int(draw_index)]
    return samples.clone(), targets.clone(), {
        "loader_seed": int(loader_seed),
        "draw_index": int(draw_index),
        "filtering": "std_threshold",
        "std_threshold": float(std_threshold),
    }


def get_synth_images_batch(batch_size: int, img_size: int, num_classes: int):
    samples = torch.randn(int(batch_size), 3, int(img_size), int(img_size), dtype=torch.float32)
    targets = torch.randint(low=0, high=int(num_classes), size=(int(batch_size),), dtype=torch.long)
    return samples, targets, {"kind": "synthetic_images"}


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


def _maybe_drop_cls(x: torch.Tensor, exclude_cls: bool):
    if exclude_cls and x.ndim == 3 and x.shape[1] > 1:
        return x[:, 1:, :]
    return x


def mean_token_sqnorm_over_d(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    return x.pow(2).sum(dim=-1).mean() / x.shape[-1]


def mean_all_pairs_token_dot_over_d(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    _, n_tokens, d = x.shape
    if n_tokens < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    gram = (x @ x.transpose(-1, -2)) / d
    mask = ~torch.eye(n_tokens, device=x.device, dtype=torch.bool).unsqueeze(0)
    return gram.masked_select(mask).mean()


def mean_token_sqnorm_over_d_per_sample(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    return x.pow(2).sum(dim=-1).mean(dim=1) / x.shape[-1]


def mean_all_pairs_token_dot_over_d_per_sample(x: torch.Tensor, exclude_cls: bool = True) -> torch.Tensor:
    x = _maybe_drop_cls(x, exclude_cls)
    batch_size, n_tokens, d = x.shape
    if n_tokens < 2:
        return torch.zeros(batch_size, device=x.device, dtype=x.dtype)
    gram = (x @ x.transpose(-1, -2)) / d
    mask = ~torch.eye(n_tokens, device=x.device, dtype=torch.bool).unsqueeze(0)
    return gram.masked_select(mask).reshape(batch_size, -1).mean(dim=1)


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


def _capture_block_outputs_for_apjn(
    model: nn.Module,
    images: torch.Tensor,
    *,
    block0_input_override: torch.Tensor | None = None,
):
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x_list, _ = capture_X_list_and_logits(
            model,
            images,
            block0_input_override=block0_input_override,
        )
    return x_list


def resolve_apjn_layers(apjn_layers, depth: int):
    depth = int(depth)
    out = sorted(set(int(layer) for layer in apjn_layers))
    for layer in out:
        if layer < 0 or layer > depth:
            raise ValueError(f"APJN layer {layer} is outside [0, {depth}].")
    return out


def resolve_direct_layers(direct_layers, depth: int, source_block: int):
    depth = int(depth)
    source_block = int(source_block)
    out = sorted(set(int(layer) for layer in direct_layers))
    for layer in out:
        if layer < source_block or layer > depth:
            raise ValueError(f"Direct layer {layer} must lie in [{source_block}, {depth}].")
    return out


def _estimate_inverse_J_from_Xlist_per_sample(
    x_list,
    *,
    l0_list,
    j_num_draws: int,
    j_normalize_by: str,
    draw_block_size: int = 1,
):
    if j_normalize_by not in ("Y", "X", "none"):
        raise ValueError("j_normalize_by must be 'Y', 'X', or 'none'.")

    last = x_list[-1]
    batch_size = int(last.shape[0])
    out = {}

    def denom_for(x_l0):
        if j_normalize_by == "none":
            return 1.0
        if j_normalize_by == "Y":
            return float(last[0].numel())
        return float(x_l0[0].numel())

    for layer in list(int(x) for x in l0_list):
        if layer == len(x_list) - 1:
            value = 1.0 if j_normalize_by != "none" else float(last[0].numel())
            out[int(layer)] = np.full(batch_size, value, dtype=float)
            continue

        x_l0 = x_list[int(layer)]
        denom = denom_for(x_l0)
        acc = np.zeros(batch_size, dtype=float)
        draws_done = 0
        while draws_done < int(j_num_draws):
            cur_block = min(int(draw_block_size), int(j_num_draws) - draws_done)
            for draw_offset in range(cur_block):
                rand = torch.randn_like(last)
                scalar = (last * rand).sum()
                retain_graph = not (
                    layer == list(int(x) for x in l0_list)[-1]
                    and draws_done + draw_offset + 1 == int(j_num_draws)
                )
                grad = torch.autograd.grad(
                    scalar,
                    x_l0,
                    retain_graph=retain_graph,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                acc += grad.pow(2).reshape(batch_size, -1).sum(dim=1).detach().cpu().numpy() / denom
            draws_done += cur_block
        out[int(layer)] = acc / float(j_num_draws)
    return out


def _estimate_direct_J_from_Xlist_per_sample(
    x_list,
    *,
    target_layers,
    source_block_index: int,
    j_num_draws: int,
    j_normalize_by: str,
    draw_block_size: int = 1,
):
    if j_normalize_by not in ("Y", "X", "none"):
        raise ValueError("j_normalize_by must be 'Y', 'X', or 'none'.")

    source = x_list[int(source_block_index)]
    batch_size = int(source.shape[0])
    out = {}

    def denom_for(x_target):
        if j_normalize_by == "none":
            return 1.0
        if j_normalize_by == "Y":
            return float(x_target[0].numel())
        return float(source[0].numel())

    ordered_targets = sorted(int(layer) for layer in target_layers)
    for idx, layer in enumerate(ordered_targets):
        x_target = x_list[int(layer)]
        denom = denom_for(x_target)
        acc = np.zeros(batch_size, dtype=float)
        draws_done = 0
        while draws_done < int(j_num_draws):
            cur_block = min(int(draw_block_size), int(j_num_draws) - draws_done)
            for draw_offset in range(cur_block):
                rand = torch.randn_like(x_target)
                scalar = (x_target * rand).sum()
                retain_graph = not (
                    idx == len(ordered_targets) - 1
                    and draws_done + draw_offset + 1 == int(j_num_draws)
                )
                grad = torch.autograd.grad(
                    scalar,
                    source,
                    retain_graph=retain_graph,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                acc += grad.pow(2).reshape(batch_size, -1).sum(dim=1).detach().cpu().numpy() / denom
            draws_done += cur_block
        out[int(layer)] = acc / float(j_num_draws)
    return out


def _summarize_activation_dot_stats_from_x_list_per_sample(x_list, *, layer_indices):
    out = {}
    for layer in sorted(set(int(layer) for layer in layer_indices)):
        x = x_list[int(layer)].detach().to(torch.float32)
        out[int(layer)] = {
            "q": np.asarray(mean_token_sqnorm_over_d_per_sample(x, exclude_cls=True).cpu().numpy(), dtype=float),
            "p": np.asarray(mean_all_pairs_token_dot_over_d_per_sample(x, exclude_cls=True).cpu().numpy(), dtype=float),
        }
    return out


def estimate_inverse_J_and_activation_stats_hutchinson_per_sample(
    model: nn.Module,
    images: torch.Tensor,
    *,
    l0_list,
    activation_layer_list=None,
    j_num_draws: int,
    j_normalize_by: str,
    draw_block_size: int = 1,
    block0_input_override: torch.Tensor | None = None,
):
    x_list = _capture_block_outputs_for_apjn(
        model,
        images,
        block0_input_override=block0_input_override,
    )
    inverse = _estimate_inverse_J_from_Xlist_per_sample(
        x_list,
        l0_list=list(int(x) for x in l0_list),
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        draw_block_size=draw_block_size,
    )
    layers = activation_layer_list if activation_layer_list is not None else l0_list
    activation_stats = _summarize_activation_dot_stats_from_x_list_per_sample(
        x_list,
        layer_indices=list(int(x) for x in layers),
    )
    return inverse, activation_stats


def estimate_direct_J_and_activation_stats_hutchinson_per_sample(
    model: nn.Module,
    images: torch.Tensor,
    *,
    direct_layers,
    direct_source_block: int,
    activation_layer_list=None,
    j_num_draws: int,
    j_normalize_by: str,
    draw_block_size: int = 1,
    block0_input_override: torch.Tensor | None = None,
):
    x_list = _capture_block_outputs_for_apjn(
        model,
        images,
        block0_input_override=block0_input_override,
    )
    forward = _estimate_direct_J_from_Xlist_per_sample(
        x_list,
        target_layers=list(int(x) for x in direct_layers),
        source_block_index=int(direct_source_block),
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        draw_block_size=draw_block_size,
    )
    layers = activation_layer_list if activation_layer_list is not None else direct_layers
    activation_stats = _summarize_activation_dot_stats_from_x_list_per_sample(
        x_list,
        layer_indices=list(int(x) for x in layers),
    )
    return forward, activation_stats


def collect_block0_input_stats(
    model: nn.Module,
    images: torch.Tensor,
    *,
    block0_input_override: torch.Tensor | None = None,
):
    x_list = _capture_block_outputs_for_apjn(
        model,
        images,
        block0_input_override=block0_input_override,
    )
    x0 = x_list[0]
    return {
        "q": float(mean_token_sqnorm_over_d(x0, exclude_cls=True).cpu().item()),
        "p": float(mean_all_pairs_token_dot_over_d(x0, exclude_cls=True).cpu().item()),
    }


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
    j_direct = np.zeros(depth + 1, dtype=float)
    k_direct = np.zeros(depth + 1, dtype=float)

    q[0] = float(q0)
    p[0] = float(p0)
    j_direct[0] = 1.0
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
            q_hat_half[layer] = (4.0 * alpha**2 / np.pi) / math.sqrt(
                max(1.0 + 4.0 * alpha**2 * q_half, _EPS)
            )
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
        j_half = (1.0 + att_scale * q_hat[layer] / n_tokens) * j_direct[layer] + att_scale * p_hat[layer] * k_direct[layer]
        k_half = (1.0 + att_scale * p_hat[layer]) * k_direct[layer] + (att_scale * q_hat[layer] / n_tokens) * j_direct[layer]
        k_direct[layer + 1] = (1.0 + mlp_scale * kappa_prime[layer] * p_hat_half[layer]) * k_half
        j_direct[layer + 1] = (1.0 + 0.5 * mlp_scale * q_hat_half[layer]) * j_half

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
        "K_direct": k_direct,
        "J": j,
        "J_direct": j_direct,
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


def _average_layer_value_dicts(dict_list):
    if not dict_list:
        return {}
    keys = sorted(int(k) for k in dict_list[0].keys())
    out = {}
    for key in keys:
        out[int(key)] = float(np.mean([float(d[int(key)]) for d in dict_list]))
    return out


def _average_activation_dicts(dict_list):
    if not dict_list:
        return {}
    keys = sorted(int(k) for k in dict_list[0].keys())
    out = {}
    for key in keys:
        q_vals = [float(d[int(key)]["q"]) for d in dict_list]
        p_vals = [float(d[int(key)]["p"]) for d in dict_list]
        out[int(key)] = {"q": float(np.mean(q_vals)), "p": float(np.mean(p_vals))}
    return out


def _folder_name_with_postfix(base_name: str, result_postfix: str | None):
    postfix = str(result_postfix or "").strip()
    return base_name if not postfix else f"{base_name}_{postfix}"


def _save_bundle_pickle(bundle, save_root, folder_name, filename):
    out_dir = Path(save_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return out_path


def load_saved_bundle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _build_input_batch(
    model_cfg: ModelConfig,
    apjn_cfg: APJNCifarConfig,
    *,
    seq_len: int | None = None,
    embed_dim: int | None = None,
):
    input_source = str(apjn_cfg.input_source).lower()
    if input_source == "cifar":
        if apjn_cfg.cifar_std_threshold is None:
            samples, targets, batch_meta = get_cifar_batch_no_filter(
                batch_size=apjn_cfg.batch_size,
                img_size=model_cfg.img_size,
                num_classes=model_cfg.num_classes,
                loader_seed=apjn_cfg.cifar_batch_seed,
                draw_index=apjn_cfg.cifar_batch_draw_index,
            )
        else:
            samples, targets, batch_meta = get_cifar_batch(
                batch_size=apjn_cfg.batch_size,
                img_size=model_cfg.img_size,
                num_classes=model_cfg.num_classes,
                loader_seed=apjn_cfg.cifar_batch_seed,
                draw_index=apjn_cfg.cifar_batch_draw_index,
                std_threshold=float(apjn_cfg.cifar_std_threshold),
                max_epochs_to_search=int(apjn_cfg.cifar_max_epochs_to_search),
            )
        return samples, targets, batch_meta, None

    if input_source != "equiangular":
        raise ValueError("input_source must be 'cifar' or 'equiangular'.")
    if seq_len is None or embed_dim is None:
        raise ValueError("seq_len and embed_dim are required for equiangular inputs.")

    samples, targets, synth_meta = get_synth_images_batch(
        apjn_cfg.batch_size,
        model_cfg.img_size,
        model_cfg.num_classes,
    )
    x0 = make_apjn_equiangular_block0_batch(
        batch_size=apjn_cfg.batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        q0=float(apjn_cfg.equiangular_q0),
        p0=float(apjn_cfg.equiangular_p0),
        seed=int(apjn_cfg.equiangular_seed),
        random_rotate=bool(apjn_cfg.equiangular_random_rotate),
    )
    meta = {
        "kind": "equiangular_block0_tokens",
        "q0": float(apjn_cfg.equiangular_q0),
        "p0": float(apjn_cfg.equiangular_p0),
        "seed": int(apjn_cfg.equiangular_seed),
        "random_rotate": bool(apjn_cfg.equiangular_random_rotate),
        "backing_inputs": synth_meta,
    }
    return samples, targets, meta, x0


def run_cifar_apjn_experiment(
    model_cfg: ModelConfig,
    apjn_cfg: APJNCifarConfig,
):
    alpha_list = [float(alpha) for alpha in np.asarray(apjn_cfg.alphas, dtype=float)]
    num_inits = max(1, int(apjn_cfg.num_model_inits))

    seed_all(model_cfg.seed)
    preview_model = build_vit(model_cfg, use_derf=False)
    scale_vit_mlp_and_value_attn_init_std(
        preview_model,
        mlp_multiplier=float(apjn_cfg.mlp_mult),
        attn_multiplier=float(apjn_cfg.attn_mult),
    )
    seq_len, embed_dim = get_vit_seq_len_and_dim(preview_model)
    depth = len(preview_model.blocks)
    apjn_layers = resolve_apjn_layers(apjn_cfg.apjn_layers, depth)
    direct_layers = resolve_direct_layers(apjn_cfg.direct_layers, depth, apjn_cfg.direct_source_block)
    samples, targets, batch_meta, x0 = _build_input_batch(
        model_cfg,
        apjn_cfg,
        seq_len=seq_len,
        embed_dim=embed_dim,
    )
    input_stats = collect_block0_input_stats(preview_model, samples, block0_input_override=x0)
    del preview_model
    cuda_cleanup()

    preln_inverse_runs = []
    preln_forward_runs = []
    preln_stats_runs = []
    derf_inverse_runs = {alpha: [] for alpha in alpha_list}
    derf_forward_runs = {alpha: [] for alpha in alpha_list}
    derf_stats_runs = {alpha: [] for alpha in alpha_list}
    preln_pre_blocks_state = None
    derf_pre_blocks_state = None

    for init_idx in range(num_inits):
        seed_all(int(model_cfg.seed) + int(init_idx))
        preln_model = build_vit(model_cfg, use_derf=False)
        scale_vit_mlp_and_value_attn_init_std(
            preln_model,
            mlp_multiplier=float(apjn_cfg.mlp_mult),
            attn_multiplier=float(apjn_cfg.attn_mult),
        )
        if preln_pre_blocks_state is None:
            preln_pre_blocks_state = _extract_vit_pre_blocks_state_dict(preln_model)

        inverse_pre, stats_pre = estimate_inverse_J_and_activation_stats_hutchinson_per_sample(
            preln_model,
            samples,
            l0_list=apjn_layers,
            activation_layer_list=sorted(set(apjn_layers + direct_layers + [int(apjn_cfg.direct_source_block)])),
            j_num_draws=int(apjn_cfg.j_num_draws),
            j_normalize_by=str(apjn_cfg.j_normalize_by),
            block0_input_override=x0,
        )
        forward_pre, _ = estimate_direct_J_and_activation_stats_hutchinson_per_sample(
            preln_model,
            samples,
            direct_layers=direct_layers,
            direct_source_block=int(apjn_cfg.direct_source_block),
            activation_layer_list=[],
            j_num_draws=int(apjn_cfg.j_num_draws),
            j_normalize_by=str(apjn_cfg.j_normalize_by),
            block0_input_override=x0,
        )
        preln_inverse_runs.append({k: float(np.mean(v)) for k, v in inverse_pre.items()})
        preln_forward_runs.append({k: float(np.mean(v)) for k, v in forward_pre.items()})
        preln_stats_runs.append({
            k: {"q": float(np.mean(v["q"])), "p": float(np.mean(v["p"]))}
            for k, v in stats_pre.items()
        })
        del preln_model
        cuda_cleanup()

        derf_model = build_vit(model_cfg, use_derf=True)
        if derf_pre_blocks_state is not None:
            _load_partial_state_dict_(derf_model, derf_pre_blocks_state)
        scale_vit_mlp_and_value_attn_init_std(
            derf_model,
            mlp_multiplier=float(apjn_cfg.mlp_mult),
            attn_multiplier=float(apjn_cfg.attn_mult),
        )
        if derf_pre_blocks_state is None:
            derf_pre_blocks_state = _extract_vit_pre_blocks_state_dict(derf_model)

        for alpha in alpha_list:
            set_all_derf_alpha_(derf_model, alpha)
            inverse_derf, stats_derf = estimate_inverse_J_and_activation_stats_hutchinson_per_sample(
                derf_model,
                samples,
                l0_list=apjn_layers,
                activation_layer_list=sorted(set(apjn_layers + direct_layers + [int(apjn_cfg.direct_source_block)])),
                j_num_draws=int(apjn_cfg.j_num_draws),
                j_normalize_by=str(apjn_cfg.j_normalize_by),
                block0_input_override=x0,
            )
            forward_derf, _ = estimate_direct_J_and_activation_stats_hutchinson_per_sample(
                derf_model,
                samples,
                direct_layers=direct_layers,
                direct_source_block=int(apjn_cfg.direct_source_block),
                activation_layer_list=[],
                j_num_draws=int(apjn_cfg.j_num_draws),
                j_normalize_by=str(apjn_cfg.j_normalize_by),
                block0_input_override=x0,
            )
            derf_inverse_runs[alpha].append({k: float(np.mean(v)) for k, v in inverse_derf.items()})
            derf_forward_runs[alpha].append({k: float(np.mean(v)) for k, v in forward_derf.items()})
            derf_stats_runs[alpha].append({
                k: {"q": float(np.mean(v["q"])), "p": float(np.mean(v["p"]))}
                for k, v in stats_derf.items()
            })
        del derf_model
        cuda_cleanup()

    return {
        "kind": "apjn_experiment",
        "model_cfg": cfg_to_dict(model_cfg),
        "apjn_cfg": cfg_to_dict(apjn_cfg),
        "input_source": str(apjn_cfg.input_source).lower(),
        "depth": int(depth),
        "seq_len": int(seq_len),
        "n_tokens_ex_cls": int(seq_len - 1),
        "embed_dim": int(embed_dim),
        "num_model_inits": int(num_inits),
        "batch_meta": batch_meta,
        "input_stats": input_stats,
        "inverse_apjn": {
            "layers": apjn_layers,
            "preln": _average_layer_value_dicts(preln_inverse_runs),
            "derf": {alpha: _average_layer_value_dicts(derf_inverse_runs[alpha]) for alpha in alpha_list},
        },
        "forward_apjn": {
            "source_block_index": int(apjn_cfg.direct_source_block),
            "layers": direct_layers,
            "preln": _average_layer_value_dicts(preln_forward_runs),
            "derf": {alpha: _average_layer_value_dicts(derf_forward_runs[alpha]) for alpha in alpha_list},
        },
        "activation_stats": {
            "preln": _average_activation_dicts(preln_stats_runs),
            "derf": {alpha: _average_activation_dicts(derf_stats_runs[alpha]) for alpha in alpha_list},
        },
    }


def _run_samplewise_apjn_with_activation_stats(
    model_cfg: ModelConfig,
    *,
    alphas,
    n_samples: int,
    batch_size: int,
    batch_seed: int,
    measurement_layers,
    activation_stat_layers,
    j_num_draws: int,
    hutchinson_block_size: int,
    num_model_inits: int,
    attn_mult: float,
    mlp_mult: float,
    keep_pre_blocks_init: bool,
    skip_preln: bool,
    direction: str,
    source_block_index: int = 0,
    deterministic: bool = False,
    save_every_n_samples: int = 25,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
):
    if deterministic:
        clear_cifar_experiment_cache()
    rng = np.random.default_rng(int(batch_seed) if deterministic else None)
    alpha_list = [float(alpha) for alpha in np.asarray(alphas, dtype=float)]
    num_inits = max(1, int(num_model_inits))
    depth = int(model_cfg.depth)
    measurement_layers = tuple(int(layer) for layer in measurement_layers)
    activation_stat_layers = tuple(int(layer) for layer in activation_stat_layers)

    bundle = {
        "kind": f"{direction}_apjn_activation_stats",
        "model_cfg": cfg_to_dict(model_cfg),
        "depth": int(depth),
        "n_tokens_ex_cls": None,
        "alphas": alpha_list,
        "layers": {
            "apjn": list(measurement_layers),
            "activation_stats": list(activation_stat_layers),
        },
        "config": {
            "n_samples": int(n_samples),
            "batch_size": int(batch_size),
            "batch_seed": int(batch_seed),
            "j_num_draws": int(j_num_draws),
            "hutchinson_block_size": int(hutchinson_block_size),
            "num_model_inits": int(num_inits),
            "attn_mult": float(attn_mult),
            "mlp_mult": float(mlp_mult),
            "keep_pre_blocks_init": bool(keep_pre_blocks_init),
            "skip_preln": bool(skip_preln),
            "deterministic": bool(deterministic),
        },
        "samples": [],
    }
    if direction == "forward":
        bundle["layers"]["source_block_index"] = int(source_block_index)

    preln_pre_blocks_state = None
    derf_pre_blocks_state = None
    folder_name = _folder_name_with_postfix(
        f"{direction}_apjn_activation_stats_depth{depth}_bs{int(batch_size)}",
        result_postfix,
    )

    for sample_index in range(int(n_samples)):
        draw_index = int(sample_index // int(batch_size))
        batch_position = int(sample_index % int(batch_size))
        if deterministic:
            seed_all(int(batch_seed) + int(draw_index))
        else:
            seed_all(int(rng.integers(0, 2**31 - 1)))

        samples, targets, batch_meta = get_cifar_batch_no_filter(
            batch_size=int(batch_size),
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=int(batch_seed),
            draw_index=int(draw_index),
        )

        if bundle["n_tokens_ex_cls"] is None:
            preview_model = build_vit(model_cfg, use_derf=False)
            bundle["n_tokens_ex_cls"] = int(get_vit_seq_len_and_dim(preview_model)[0] - 1)
            del preview_model
            cuda_cleanup()

        sample_record = {
            "sample_index": int(sample_index),
            "batch_draw_index": int(draw_index),
            "batch_position": int(batch_position),
            "batch_meta": {
                **batch_meta,
                "target": int(targets[int(batch_position)].item()),
            },
            f"{direction}_apjn": {"preln": {}, "derf": {}},
            "activation_stats": {"preln": {}, "derf": {}},
        }

        if not bool(skip_preln):
            per_init_values = []
            per_init_stats = []
            for init_idx in range(num_inits):
                seed_all(int(model_cfg.seed) + int(init_idx))
                model = build_vit(model_cfg, use_derf=False)
                if keep_pre_blocks_init and preln_pre_blocks_state is not None:
                    _load_partial_state_dict_(model, preln_pre_blocks_state)
                scale_vit_mlp_and_value_attn_init_std(
                    model,
                    mlp_multiplier=float(mlp_mult),
                    attn_multiplier=float(attn_mult),
                )
                if keep_pre_blocks_init and preln_pre_blocks_state is None:
                    preln_pre_blocks_state = _extract_vit_pre_blocks_state_dict(model)
                if direction == "inverse":
                    values, stats = estimate_inverse_J_and_activation_stats_hutchinson_per_sample(
                        model,
                        samples,
                        l0_list=measurement_layers,
                        activation_layer_list=activation_stat_layers,
                        j_num_draws=int(j_num_draws),
                        j_normalize_by="Y",
                        draw_block_size=int(hutchinson_block_size),
                    )
                else:
                    values, stats = estimate_direct_J_and_activation_stats_hutchinson_per_sample(
                        model,
                        samples,
                        direct_layers=measurement_layers,
                        direct_source_block=int(source_block_index),
                        activation_layer_list=activation_stat_layers,
                        j_num_draws=int(j_num_draws),
                        j_normalize_by="Y",
                        draw_block_size=int(hutchinson_block_size),
                    )
                per_init_values.append({k: float(v[batch_position]) for k, v in values.items()})
                per_init_stats.append({
                    k: {"q": float(v["q"][batch_position]), "p": float(v["p"][batch_position])}
                    for k, v in stats.items()
                })
                del model
                cuda_cleanup()
            sample_record[f"{direction}_apjn"]["preln"] = _average_layer_value_dicts(per_init_values)
            sample_record["activation_stats"]["preln"] = _average_activation_dicts(per_init_stats)

        for alpha in alpha_list:
            per_init_values = []
            per_init_stats = []
            for init_idx in range(num_inits):
                seed_all(int(model_cfg.seed) + int(init_idx))
                model = build_vit(model_cfg, use_derf=True)
                if keep_pre_blocks_init and derf_pre_blocks_state is not None:
                    _load_partial_state_dict_(model, derf_pre_blocks_state)
                scale_vit_mlp_and_value_attn_init_std(
                    model,
                    mlp_multiplier=float(mlp_mult),
                    attn_multiplier=float(attn_mult),
                )
                set_all_derf_alpha_(model, float(alpha))
                if keep_pre_blocks_init and derf_pre_blocks_state is None:
                    derf_pre_blocks_state = _extract_vit_pre_blocks_state_dict(model)
                if direction == "inverse":
                    values, stats = estimate_inverse_J_and_activation_stats_hutchinson_per_sample(
                        model,
                        samples,
                        l0_list=measurement_layers,
                        activation_layer_list=activation_stat_layers,
                        j_num_draws=int(j_num_draws),
                        j_normalize_by="Y",
                        draw_block_size=int(hutchinson_block_size),
                    )
                else:
                    values, stats = estimate_direct_J_and_activation_stats_hutchinson_per_sample(
                        model,
                        samples,
                        direct_layers=measurement_layers,
                        direct_source_block=int(source_block_index),
                        activation_layer_list=activation_stat_layers,
                        j_num_draws=int(j_num_draws),
                        j_normalize_by="Y",
                        draw_block_size=int(hutchinson_block_size),
                    )
                per_init_values.append({k: float(v[batch_position]) for k, v in values.items()})
                per_init_stats.append({
                    k: {"q": float(v["q"][batch_position]), "p": float(v["p"][batch_position])}
                    for k, v in stats.items()
                })
                del model
                cuda_cleanup()
            sample_record[f"{direction}_apjn"]["derf"][float(alpha)] = _average_layer_value_dicts(per_init_values)
            sample_record["activation_stats"]["derf"][float(alpha)] = _average_activation_dicts(per_init_stats)

        bundle["samples"].append(sample_record)
        if save_results and (sample_index + 1) % max(1, int(save_every_n_samples)) == 0:
            _save_bundle_pickle(bundle, save_root=save_root, folder_name=folder_name, filename="results.pkl")

    if save_results:
        saved_path = _save_bundle_pickle(bundle, save_root=save_root, folder_name=folder_name, filename="results.pkl")
        bundle["saved_path"] = str(saved_path)
    return bundle


def run_cifar_inverse_apjn_with_activation_stats(
    model_cfg: ModelConfig,
    *,
    alphas,
    n_samples: int = 16,
    layer_stride: int = 4,
    activation_stat_blocks=None,
    batch_size: int = 1,
    batch_seed: int = 0,
    j_num_draws: int = 10,
    hutchinson_block_size: int = 1,
    num_model_inits: int = 1,
    attn_mult: float = 1.0,
    mlp_mult: float = 1.0,
    keep_pre_blocks_init: bool = False,
    skip_preln: bool = False,
    deterministic: bool = False,
    save_every_n_samples: int = 25,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
):
    depth = int(model_cfg.depth)
    layers = tuple(layer for layer in range(0, depth + 1, int(layer_stride)) if 0 < layer < depth)
    stat_layers = tuple(
        resolve_apjn_layers(activation_stat_blocks if activation_stat_blocks is not None else layers, depth)
    )
    return _run_samplewise_apjn_with_activation_stats(
        model_cfg,
        alphas=alphas,
        n_samples=n_samples,
        batch_size=batch_size,
        batch_seed=batch_seed,
        measurement_layers=layers,
        activation_stat_layers=stat_layers,
        j_num_draws=j_num_draws,
        hutchinson_block_size=hutchinson_block_size,
        num_model_inits=num_model_inits,
        attn_mult=attn_mult,
        mlp_mult=mlp_mult,
        keep_pre_blocks_init=keep_pre_blocks_init,
        skip_preln=skip_preln,
        direction="inverse",
        deterministic=deterministic,
        save_every_n_samples=save_every_n_samples,
        save_results=save_results,
        save_root=save_root,
        result_postfix=result_postfix,
    )


def run_cifar_forward_apjn_with_activation_stats(
    model_cfg: ModelConfig,
    *,
    alphas,
    source_block_index: int = 0,
    n_samples: int = 16,
    layer_stride: int = 4,
    activation_stat_blocks=None,
    batch_size: int = 1,
    batch_seed: int = 0,
    j_num_draws: int = 10,
    hutchinson_block_size: int = 1,
    num_model_inits: int = 1,
    attn_mult: float = 1.0,
    mlp_mult: float = 1.0,
    keep_pre_blocks_init: bool = False,
    skip_preln: bool = False,
    deterministic: bool = False,
    save_every_n_samples: int = 25,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
):
    depth = int(model_cfg.depth)
    source_block_index = int(source_block_index)
    layers = tuple(layer for layer in range(source_block_index, depth + 1, int(layer_stride)) if source_block_index <= layer < depth)
    stat_layers = tuple(
        resolve_apjn_layers(activation_stat_blocks if activation_stat_blocks is not None else layers, depth)
    )
    return _run_samplewise_apjn_with_activation_stats(
        model_cfg,
        alphas=alphas,
        n_samples=n_samples,
        batch_size=batch_size,
        batch_seed=batch_seed,
        measurement_layers=layers,
        activation_stat_layers=stat_layers,
        j_num_draws=j_num_draws,
        hutchinson_block_size=hutchinson_block_size,
        num_model_inits=num_model_inits,
        attn_mult=attn_mult,
        mlp_mult=mlp_mult,
        keep_pre_blocks_init=keep_pre_blocks_init,
        skip_preln=skip_preln,
        direction="forward",
        source_block_index=source_block_index,
        deterministic=deterministic,
        save_every_n_samples=save_every_n_samples,
        save_results=save_results,
        save_root=save_root,
        result_postfix=result_postfix,
    )


def resolve_float_key(d, target, tol=1e-9):
    keys = [float(key) for key in d.keys()]
    if not keys:
        raise KeyError("Dictionary has no keys.")
    best = min(keys, key=lambda key: abs(key - float(target)))
    if abs(best - float(target)) > tol:
        raise KeyError(f"Could not match target={target} to available keys={keys}.")
    return best


def _get_activation_stats_for_variant(sample, *, model_variant: str, alpha: float | None):
    if model_variant == "preln":
        return sample["activation_stats"]["preln"], None
    if model_variant != "derf":
        raise ValueError("model_variant must be 'preln' or 'derf'.")
    key = resolve_float_key(sample["activation_stats"]["derf"], float(alpha))
    return sample["activation_stats"]["derf"][key], float(key)


def _get_measurements_for_variant(sample, *, direction: str, model_variant: str, alpha: float | None):
    if model_variant == "preln":
        return sample[f"{direction}_apjn"]["preln"], None
    if model_variant != "derf":
        raise ValueError("model_variant must be 'preln' or 'derf'.")
    key = resolve_float_key(sample[f"{direction}_apjn"]["derf"], float(alpha))
    return sample[f"{direction}_apjn"]["derf"][key], float(key)


def build_inverse_theory_comparison(
    bundle,
    *,
    sample_index: int,
    block_init_condition: int,
    model_variant: str,
    alpha: float | None = None,
):
    sample = bundle["samples"][int(sample_index)]
    stats_dict, alpha_used = _get_activation_stats_for_variant(
        sample,
        model_variant=model_variant,
        alpha=alpha,
    )
    measured_dict, _ = _get_measurements_for_variant(
        sample,
        direction="inverse",
        model_variant=model_variant,
        alpha=alpha,
    )
    if int(block_init_condition) not in stats_dict:
        raise ValueError(f"Missing activation stats for block {block_init_condition}.")

    mean_field_cfg = build_mean_field_cfg_for_vit_base(
        attn_mult=float(bundle["config"]["attn_mult"]),
        mlp_mult=float(bundle["config"]["mlp_mult"]),
    )
    theory = simulate_recursions_full(
        num_layers=int(bundle["depth"]) - int(block_init_condition),
        q0=float(stats_dict[int(block_init_condition)]["q"]),
        p0=float(stats_dict[int(block_init_condition)]["p"]),
        n_tokens=int(bundle["n_tokens_ex_cls"]),
        mode="layernorm" if model_variant == "preln" else "erf",
        alpha=1.0 if alpha_used is None else float(alpha_used),
        sigma_w1=float(mean_field_cfg.sigma_w1),
        sigma_w2=float(mean_field_cfg.sigma_w2),
        sigma_o=float(mean_field_cfg.sigma_o),
        sigma_v=float(mean_field_cfg.sigma_v),
        sigma_a=float(mean_field_cfg.sigma_a),
    )
    layers = sorted(int(layer) for layer in measured_dict.keys() if int(layer) >= int(block_init_condition))
    return {
        "direction": "inverse",
        "model_variant": model_variant,
        "alpha": alpha_used,
        "block_init_condition": int(block_init_condition),
        "layers": np.asarray(layers, dtype=int),
        "measured": np.asarray([float(measured_dict[layer]) for layer in layers], dtype=float),
        "theory": np.asarray([float(theory["J"][layer - int(block_init_condition)]) for layer in layers], dtype=float),
        "initial_conditions": dict(stats_dict[int(block_init_condition)]),
    }


def build_forward_theory_comparison(
    bundle,
    *,
    sample_index: int,
    block_init_condition: int,
    model_variant: str,
    alpha: float | None = None,
):
    source_block_index = int(bundle["layers"]["source_block_index"])
    sample = bundle["samples"][int(sample_index)]
    stats_dict, alpha_used = _get_activation_stats_for_variant(
        sample,
        model_variant=model_variant,
        alpha=alpha,
    )
    measured_dict, _ = _get_measurements_for_variant(
        sample,
        direction="forward",
        model_variant=model_variant,
        alpha=alpha,
    )
    if int(block_init_condition) not in stats_dict:
        raise ValueError(f"Missing activation stats for block {block_init_condition}.")
    if int(block_init_condition) > int(source_block_index):
        raise ValueError("block_init_condition must not exceed source_block_index for forward comparisons.")

    mean_field_cfg = build_mean_field_cfg_for_vit_base(
        attn_mult=float(bundle["config"]["attn_mult"]),
        mlp_mult=float(bundle["config"]["mlp_mult"]),
    )
    theory = simulate_recursions_full(
        num_layers=int(bundle["depth"]) - int(block_init_condition),
        q0=float(stats_dict[int(block_init_condition)]["q"]),
        p0=float(stats_dict[int(block_init_condition)]["p"]),
        n_tokens=int(bundle["n_tokens_ex_cls"]),
        mode="layernorm" if model_variant == "preln" else "erf",
        alpha=1.0 if alpha_used is None else float(alpha_used),
        sigma_w1=float(mean_field_cfg.sigma_w1),
        sigma_w2=float(mean_field_cfg.sigma_w2),
        sigma_o=float(mean_field_cfg.sigma_o),
        sigma_v=float(mean_field_cfg.sigma_v),
        sigma_a=float(mean_field_cfg.sigma_a),
    )
    source_rel = int(source_block_index) - int(block_init_condition)
    source_value = float(theory["J_direct"][source_rel])
    layers = sorted(int(layer) for layer in measured_dict.keys() if int(layer) >= int(source_block_index))
    return {
        "direction": "forward",
        "model_variant": model_variant,
        "alpha": alpha_used,
        "block_init_condition": int(block_init_condition),
        "source_block_index": int(source_block_index),
        "layers": np.asarray(layers, dtype=int),
        "measured": np.asarray([float(measured_dict[layer]) for layer in layers], dtype=float),
        "theory": np.asarray(
            [float(theory["J_direct"][layer - int(block_init_condition)] / max(source_value, _EPS)) for layer in layers],
            dtype=float,
        ),
        "initial_conditions": dict(stats_dict[int(block_init_condition)]),
    }


def build_equangular_theory_comparison(experiment_bundle):
    mean_field_cfg = build_mean_field_cfg_for_vit_base(
        attn_mult=float(experiment_bundle["apjn_cfg"]["attn_mult"]),
        mlp_mult=float(experiment_bundle["apjn_cfg"]["mlp_mult"]),
    )
    theory = compute_theory_bundle(
        depth=int(experiment_bundle["depth"]),
        n_tokens=int(experiment_bundle["n_tokens_ex_cls"]),
        q0=float(experiment_bundle["input_stats"]["q"]),
        p0=float(experiment_bundle["input_stats"]["p"]),
        alphas=experiment_bundle["apjn_cfg"]["alphas"],
        mean_field_cfg=mean_field_cfg,
    )
    source = int(experiment_bundle["forward_apjn"]["source_block_index"])
    source_pre = float(theory["preln"]["J_direct"][source])
    out = {
        "inverse": {
            "direction": "inverse",
            "layers": np.asarray(experiment_bundle["inverse_apjn"]["layers"], dtype=int),
            "preln": {
                "measured": np.asarray(
                    [experiment_bundle["inverse_apjn"]["preln"][int(layer)] for layer in experiment_bundle["inverse_apjn"]["layers"]],
                    dtype=float,
                ),
                "theory": np.asarray(
                    [theory["preln"]["J"][int(layer)] for layer in experiment_bundle["inverse_apjn"]["layers"]],
                    dtype=float,
                ),
            },
            "derf": {},
        },
        "forward": {
            "direction": "forward",
            "source_block_index": int(source),
            "layers": np.asarray(experiment_bundle["forward_apjn"]["layers"], dtype=int),
            "preln": {
                "measured": np.asarray(
                    [experiment_bundle["forward_apjn"]["preln"][int(layer)] for layer in experiment_bundle["forward_apjn"]["layers"]],
                    dtype=float,
                ),
                "theory": np.asarray(
                    [theory["preln"]["J_direct"][int(layer)] / max(source_pre, _EPS) for layer in experiment_bundle["forward_apjn"]["layers"]],
                    dtype=float,
                ),
            },
            "derf": {},
        },
        "theory_bundle": theory,
    }
    for alpha, derf_theory in theory["derf"].items():
        source_derf = float(derf_theory["J_direct"][source])
        out["inverse"]["derf"][float(alpha)] = {
            "measured": np.asarray(
                [experiment_bundle["inverse_apjn"]["derf"][float(alpha)][int(layer)] for layer in experiment_bundle["inverse_apjn"]["layers"]],
                dtype=float,
            ),
            "theory": np.asarray(
                [derf_theory["J"][int(layer)] for layer in experiment_bundle["inverse_apjn"]["layers"]],
                dtype=float,
            ),
        }
        out["forward"]["derf"][float(alpha)] = {
            "measured": np.asarray(
                [experiment_bundle["forward_apjn"]["derf"][float(alpha)][int(layer)] for layer in experiment_bundle["forward_apjn"]["layers"]],
                dtype=float,
            ),
            "theory": np.asarray(
                [derf_theory["J_direct"][int(layer)] / max(source_derf, _EPS) for layer in experiment_bundle["forward_apjn"]["layers"]],
                dtype=float,
            ),
        }
    return out


def compute_error_metrics(theory_vals, measured_vals, eps=1e-12):
    theory_vals = np.asarray(theory_vals, dtype=float)
    measured_vals = np.asarray(measured_vals, dtype=float)
    mape = 100.0 * np.mean(np.abs(theory_vals - measured_vals) / np.maximum(np.abs(measured_vals), eps))
    smape = 200.0 * np.mean(
        np.abs(theory_vals - measured_vals) / np.maximum(np.abs(theory_vals) + np.abs(measured_vals), eps)
    )
    typ_mult_err = float(
        np.exp(np.mean(np.abs(np.log(np.maximum(np.abs(measured_vals), eps) / np.maximum(np.abs(theory_vals), eps)))))
    )
    return {"mape": float(mape), "smape": float(smape), "typ_mult_err": float(typ_mult_err)}


def split_layers_into_three_regions(layers):
    layers = np.asarray(sorted(int(layer) for layer in layers), dtype=int)
    if layers.size < 3:
        raise ValueError("Need at least three layers to split into three regions.")
    names = ("Shallow", "Middle", "Deep")
    chunks = np.array_split(layers, 3)
    return {name: [int(x) for x in chunk.tolist()] for name, chunk in zip(names, chunks)}


def compute_depthwise_gmfe_records(
    bundle,
    *,
    direction: str,
    block_init_condition: int,
    block_start_comparing_apjn: int,
    alphas,
):
    out = {"preln": [], "derf": []}
    for sample_index, _sample in enumerate(bundle["samples"]):
        pre_cmp = (
            build_inverse_theory_comparison(
                bundle,
                sample_index=sample_index,
                block_init_condition=block_init_condition,
                model_variant="preln",
            )
            if direction == "inverse"
            else build_forward_theory_comparison(
                bundle,
                sample_index=sample_index,
                block_init_condition=block_init_condition,
                model_variant="preln",
            )
        )
        regions = split_layers_into_three_regions(pre_cmp["layers"])
        for region_name, region_layers in regions.items():
            mask = [int(layer) for layer in region_layers if int(layer) >= int(block_start_comparing_apjn)]
            if not mask:
                continue
            keep = np.isin(pre_cmp["layers"], np.asarray(mask, dtype=int))
            metrics = compute_error_metrics(pre_cmp["theory"][keep], pre_cmp["measured"][keep])
            out["preln"].append({
                "sample_index": int(sample_index),
                "region": region_name,
                **metrics,
            })

        for alpha in np.asarray(alphas, dtype=float):
            derf_cmp = (
                build_inverse_theory_comparison(
                    bundle,
                    sample_index=sample_index,
                    block_init_condition=block_init_condition,
                    model_variant="derf",
                    alpha=float(alpha),
                )
                if direction == "inverse"
                else build_forward_theory_comparison(
                    bundle,
                    sample_index=sample_index,
                    block_init_condition=block_init_condition,
                    model_variant="derf",
                    alpha=float(alpha),
                )
            )
            for region_name, region_layers in regions.items():
                mask = [int(layer) for layer in region_layers if int(layer) >= int(block_start_comparing_apjn)]
                if not mask:
                    continue
                keep = np.isin(derf_cmp["layers"], np.asarray(mask, dtype=int))
                metrics = compute_error_metrics(derf_cmp["theory"][keep], derf_cmp["measured"][keep])
                out["derf"].append({
                    "sample_index": int(sample_index),
                    "region": region_name,
                    "alpha": float(derf_cmp["alpha"]),
                    **metrics,
                })
    return out

