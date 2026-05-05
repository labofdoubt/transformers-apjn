from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .models import capture_X_list_and_logits


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


def resolve_backward_apjn_layers(backward_apjn_layers, depth: int):
    depth = int(depth)
    out = sorted(set(int(layer) for layer in backward_apjn_layers))
    for layer in out:
        if layer < 0 or layer > depth:
            raise ValueError(f"APJN layer {layer} is outside [0, {depth}].")
    return out


def resolve_forward_apjn_layers(forward_apjn_layers, depth: int, source_block: int):
    depth = int(depth)
    source_block = int(source_block)
    out = sorted(set(int(layer) for layer in forward_apjn_layers))
    for layer in out:
        if layer < source_block or layer > depth:
            raise ValueError(f"Forward APJN layer {layer} must lie in [{source_block}, {depth}].")
    return out


def _estimate_backward_J_from_x_list_batched(
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

    ordered_layers = list(int(x) for x in l0_list)
    for layer in ordered_layers:
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
                    layer == ordered_layers[-1]
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


def _estimate_forward_J_from_x_list_batched(
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


def _summarize_activation_dot_stats_from_x_list_batched(x_list, *, layer_indices):
    out = {}
    for layer in sorted(set(int(layer) for layer in layer_indices)):
        x = x_list[int(layer)].detach().to(torch.float32)
        out[int(layer)] = {
            "q": np.asarray(mean_token_sqnorm_over_d_per_sample(x, exclude_cls=True).cpu().numpy(), dtype=float),
            "p": np.asarray(mean_all_pairs_token_dot_over_d_per_sample(x, exclude_cls=True).cpu().numpy(), dtype=float),
        }
    return out


def estimate_backward_J_and_activation_stats_hutchinson_batched(
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
    backward = _estimate_backward_J_from_x_list_batched(
        x_list,
        l0_list=list(int(x) for x in l0_list),
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        draw_block_size=draw_block_size,
    )
    layers = activation_layer_list if activation_layer_list is not None else l0_list
    activation_stats = _summarize_activation_dot_stats_from_x_list_batched(
        x_list,
        layer_indices=list(int(x) for x in layers),
    )
    return backward, activation_stats


def estimate_forward_J_and_activation_stats_hutchinson_batched(
    model: nn.Module,
    images: torch.Tensor,
    *,
    forward_layers,
    forward_source_block: int,
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
    forward = _estimate_forward_J_from_x_list_batched(
        x_list,
        target_layers=list(int(x) for x in forward_layers),
        source_block_index=int(forward_source_block),
        j_num_draws=j_num_draws,
        j_normalize_by=j_normalize_by,
        draw_block_size=draw_block_size,
    )
    layers = activation_layer_list if activation_layer_list is not None else forward_layers
    activation_stats = _summarize_activation_dot_stats_from_x_list_batched(
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
