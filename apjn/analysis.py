from __future__ import annotations

import numpy as np

from .io_utils import compute_error_metrics, resolve_float_key, split_layers_into_three_regions
from .theory import build_mean_field_cfg_for_vit_base, compute_theory_bundle, simulate_recursions_full

_EPS = 1e-12


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


def build_backward_theory_comparison(
    bundle,
    *,
    sample_index: int,
    block_init_condition: int,
    model_variant: str,
    alpha: float | None = None,
):
    sample = bundle["samples"][int(sample_index)]
    stats_dict, alpha_used = _get_activation_stats_for_variant(sample, model_variant=model_variant, alpha=alpha)
    measured_dict, _ = _get_measurements_for_variant(sample, direction="backward", model_variant=model_variant, alpha=alpha)
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
        "direction": "backward",
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
    stats_dict, alpha_used = _get_activation_stats_for_variant(sample, model_variant=model_variant, alpha=alpha)
    measured_dict, _ = _get_measurements_for_variant(sample, direction="forward", model_variant=model_variant, alpha=alpha)
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
    source_value = float(theory["J_forward"][source_rel])
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
            [float(theory["J_forward"][layer - int(block_init_condition)] / max(source_value, _EPS)) for layer in layers],
            dtype=float,
        ),
        "initial_conditions": dict(stats_dict[int(block_init_condition)]),
    }


def build_perm_inv_input_exp_theory_comparison(experiment_bundle):
    mean_field_cfg = build_mean_field_cfg_for_vit_base(
        attn_mult=float(experiment_bundle["exp_cfg"]["attn_mult"]),
        mlp_mult=float(experiment_bundle["exp_cfg"]["mlp_mult"]),
    )
    theory = compute_theory_bundle(
        depth=int(experiment_bundle["depth"]),
        n_tokens=int(experiment_bundle["n_tokens_ex_cls"]),
        q0=float(experiment_bundle["input_stats"]["q"]),
        p0=float(experiment_bundle["input_stats"]["p"]),
        alphas=experiment_bundle["exp_cfg"]["alphas"],
        mean_field_cfg=mean_field_cfg,
    )
    source = int(experiment_bundle["forward_apjn"]["source_block_index"])
    source_pre = float(theory["preln"]["J_forward"][source])
    out = {
        "backward": {
            "direction": "backward",
            "layers": np.asarray(experiment_bundle["backward_apjn"]["layers"], dtype=int),
            "preln": {
                "measured": np.asarray(
                    [experiment_bundle["backward_apjn"]["preln"][int(layer)] for layer in experiment_bundle["backward_apjn"]["layers"]],
                    dtype=float,
                ),
                "theory": np.asarray(
                    [theory["preln"]["J"][int(layer)] for layer in experiment_bundle["backward_apjn"]["layers"]],
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
                    [theory["preln"]["J_forward"][int(layer)] / max(source_pre, _EPS) for layer in experiment_bundle["forward_apjn"]["layers"]],
                    dtype=float,
                ),
            },
            "derf": {},
        },
        "theory_bundle": theory,
    }
    for alpha, derf_theory in theory["derf"].items():
        source_derf = float(derf_theory["J_forward"][source])
        out["backward"]["derf"][float(alpha)] = {
            "measured": np.asarray(
                [experiment_bundle["backward_apjn"]["derf"][float(alpha)][int(layer)] for layer in experiment_bundle["backward_apjn"]["layers"]],
                dtype=float,
            ),
            "theory": np.asarray(
                [derf_theory["J"][int(layer)] for layer in experiment_bundle["backward_apjn"]["layers"]],
                dtype=float,
            ),
        }
        out["forward"]["derf"][float(alpha)] = {
            "measured": np.asarray(
                [experiment_bundle["forward_apjn"]["derf"][float(alpha)][int(layer)] for layer in experiment_bundle["forward_apjn"]["layers"]],
                dtype=float,
            ),
            "theory": np.asarray(
                [derf_theory["J_forward"][int(layer)] / max(source_derf, _EPS) for layer in experiment_bundle["forward_apjn"]["layers"]],
                dtype=float,
            ),
        }
    return out


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
            build_backward_theory_comparison(
                bundle,
                sample_index=sample_index,
                block_init_condition=block_init_condition,
                model_variant="preln",
            )
            if direction == "backward"
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
                build_backward_theory_comparison(
                    bundle,
                    sample_index=sample_index,
                    block_init_condition=block_init_condition,
                    model_variant="derf",
                    alpha=float(alpha),
                )
                if direction == "backward"
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
