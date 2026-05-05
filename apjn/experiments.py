from __future__ import annotations

import math

import numpy as np

from .config import ModelConfig, PermSymInputExpConfig, cfg_to_dict
from .data import clear_cifar_experiment_cache, get_cifar_batch, get_synth_images_batch
from .io_utils import _average_activation_dicts, _average_layer_value_dicts, _folder_name_with_postfix, _save_bundle_pickle
from .measurements import (
    collect_block0_input_stats,
    estimate_backward_J_and_activation_stats_hutchinson_batched,
    estimate_forward_J_and_activation_stats_hutchinson_batched,
    resolve_backward_apjn_layers,
    resolve_forward_apjn_layers,
)
from .models import (
    _extract_vit_pre_blocks_state_dict,
    _load_partial_state_dict_,
    build_vit,
    get_vit_seq_len_and_dim,
    make_apjn_equiangular_block0_batch,
    scale_vit_mlp_and_value_attn_init_std,
    set_all_derf_alpha_,
)
from .runtime import cuda_cleanup, seed_all, tqdm


def _build_perm_sym_input_batch(
    model_cfg: ModelConfig,
    exp_cfg: PermSymInputExpConfig,
    *,
    seq_len: int | None = None,
    embed_dim: int | None = None,
):
    if seq_len is None or embed_dim is None:
        raise ValueError("seq_len and embed_dim are required for equiangular inputs.")

    samples, synth_meta = get_synth_images_batch(
        exp_cfg.batch_size,
        model_cfg.img_size,
    )
    x0 = make_apjn_equiangular_block0_batch(
        batch_size=exp_cfg.batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        q0=float(exp_cfg.equiangular_q0),
        p0=float(exp_cfg.equiangular_p0),
        seed=int(exp_cfg.equiangular_seed),
        random_rotate=bool(exp_cfg.equiangular_random_rotate),
    )
    meta = {
        "kind": "perm_symmetric_block0_tokens",
        "q0": float(exp_cfg.equiangular_q0),
        "p0": float(exp_cfg.equiangular_p0),
        "seed": int(exp_cfg.equiangular_seed),
        "random_rotate": bool(exp_cfg.equiangular_random_rotate),
        "backing_inputs": synth_meta,
    }
    return samples, meta, x0


def run_perm_inv_input_apjn_exp(
    model_cfg: ModelConfig,
    exp_cfg: PermSymInputExpConfig,
):
    alpha_list = [float(alpha) for alpha in np.asarray(exp_cfg.alphas, dtype=float)]
    num_inits = max(1, int(exp_cfg.num_model_inits))

    seed_all(model_cfg.seed)
    preview_model = build_vit(model_cfg, use_derf=False)
    scale_vit_mlp_and_value_attn_init_std(
        preview_model,
        mlp_multiplier=float(exp_cfg.mlp_mult),
        attn_multiplier=float(exp_cfg.attn_mult),
    )
    seq_len, embed_dim = get_vit_seq_len_and_dim(preview_model)
    depth = len(preview_model.blocks)
    backward_apjn_layers = resolve_backward_apjn_layers(exp_cfg.backward_apjn_layers, depth)
    forward_apjn_layers = resolve_forward_apjn_layers(
        exp_cfg.forward_apjn_layers,
        depth,
        exp_cfg.forward_source_block,
    )
    samples, batch_meta, x0 = _build_perm_sym_input_batch(
        model_cfg,
        exp_cfg,
        seq_len=seq_len,
        embed_dim=embed_dim,
    )
    input_stats = collect_block0_input_stats(preview_model, samples, block0_input_override=x0)
    del preview_model
    cuda_cleanup()

    preln_backward_runs = []
    preln_forward_runs = []
    preln_stats_runs = []
    derf_backward_runs = {alpha: [] for alpha in alpha_list}
    derf_forward_runs = {alpha: [] for alpha in alpha_list}
    derf_stats_runs = {alpha: [] for alpha in alpha_list}
    preln_pre_blocks_state = None
    derf_pre_blocks_state = None

    for init_idx in range(num_inits):
        seed_all(int(model_cfg.seed) + int(init_idx))
        preln_model = build_vit(model_cfg, use_derf=False)
        scale_vit_mlp_and_value_attn_init_std(
            preln_model,
            mlp_multiplier=float(exp_cfg.mlp_mult),
            attn_multiplier=float(exp_cfg.attn_mult),
        )
        if preln_pre_blocks_state is None:
            preln_pre_blocks_state = _extract_vit_pre_blocks_state_dict(preln_model)

        backward_pre, stats_pre = estimate_backward_J_and_activation_stats_hutchinson_batched(
            preln_model,
            samples,
            l0_list=backward_apjn_layers,
            activation_layer_list=sorted(set(backward_apjn_layers + forward_apjn_layers + [int(exp_cfg.forward_source_block)])),
            j_num_draws=int(exp_cfg.j_num_draws),
            j_normalize_by=str(exp_cfg.j_normalize_by),
            block0_input_override=x0,
        )
        forward_pre, _ = estimate_forward_J_and_activation_stats_hutchinson_batched(
            preln_model,
            samples,
            forward_layers=forward_apjn_layers,
            forward_source_block=int(exp_cfg.forward_source_block),
            activation_layer_list=[],
            j_num_draws=int(exp_cfg.j_num_draws),
            j_normalize_by=str(exp_cfg.j_normalize_by),
            block0_input_override=x0,
        )
        preln_backward_runs.append({k: float(np.mean(v)) for k, v in backward_pre.items()})
        preln_forward_runs.append({k: float(np.mean(v)) for k, v in forward_pre.items()})
        preln_stats_runs.append({k: {"q": float(np.mean(v["q"])), "p": float(np.mean(v["p"]))} for k, v in stats_pre.items()})
        del preln_model
        cuda_cleanup()

        derf_model = build_vit(model_cfg, use_derf=True)
        if derf_pre_blocks_state is not None:
            _load_partial_state_dict_(derf_model, derf_pre_blocks_state)
        scale_vit_mlp_and_value_attn_init_std(
            derf_model,
            mlp_multiplier=float(exp_cfg.mlp_mult),
            attn_multiplier=float(exp_cfg.attn_mult),
        )
        if derf_pre_blocks_state is None:
            derf_pre_blocks_state = _extract_vit_pre_blocks_state_dict(derf_model)

        for alpha in alpha_list:
            set_all_derf_alpha_(derf_model, alpha)
            backward_derf, stats_derf = estimate_backward_J_and_activation_stats_hutchinson_batched(
                derf_model,
                samples,
                l0_list=backward_apjn_layers,
                activation_layer_list=sorted(set(backward_apjn_layers + forward_apjn_layers + [int(exp_cfg.forward_source_block)])),
                j_num_draws=int(exp_cfg.j_num_draws),
                j_normalize_by=str(exp_cfg.j_normalize_by),
                block0_input_override=x0,
            )
            forward_derf, _ = estimate_forward_J_and_activation_stats_hutchinson_batched(
                derf_model,
                samples,
                forward_layers=forward_apjn_layers,
                forward_source_block=int(exp_cfg.forward_source_block),
                activation_layer_list=[],
                j_num_draws=int(exp_cfg.j_num_draws),
                j_normalize_by=str(exp_cfg.j_normalize_by),
                block0_input_override=x0,
            )
            derf_backward_runs[alpha].append({k: float(np.mean(v)) for k, v in backward_derf.items()})
            derf_forward_runs[alpha].append({k: float(np.mean(v)) for k, v in forward_derf.items()})
            derf_stats_runs[alpha].append({k: {"q": float(np.mean(v["q"])), "p": float(np.mean(v["p"]))} for k, v in stats_derf.items()})
        del derf_model
        cuda_cleanup()

    return {
        "kind": "perm_sym_input_apjn_experiment",
        "model_cfg": cfg_to_dict(model_cfg),
        "exp_cfg": cfg_to_dict(exp_cfg),
        "depth": int(depth),
        "seq_len": int(seq_len),
        "n_tokens_ex_cls": int(seq_len - 1),
        "embed_dim": int(embed_dim),
        "num_model_inits": int(num_inits),
        "batch_meta": batch_meta,
        "input_stats": input_stats,
        "backward_apjn": {
            "layers": backward_apjn_layers,
            "preln": _average_layer_value_dicts(preln_backward_runs),
            "derf": {alpha: _average_layer_value_dicts(derf_backward_runs[alpha]) for alpha in alpha_list},
        },
        "forward_apjn": {
            "source_block_index": int(exp_cfg.forward_source_block),
            "layers": forward_apjn_layers,
            "preln": _average_layer_value_dicts(preln_forward_runs),
            "derf": {alpha: _average_layer_value_dicts(derf_forward_runs[alpha]) for alpha in alpha_list},
        },
        "activation_stats": {
            "preln": _average_activation_dicts(preln_stats_runs),
            "derf": {alpha: _average_activation_dicts(derf_stats_runs[alpha]) for alpha in alpha_list},
        },
    }


def _run_batchwise_apjn_with_activation_stats(
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
    keep_pre_transformer_init: bool,
    skip_preln: bool,
    direction: str,
    source_block_index: int = 0,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
    deterministic: bool = False,
):
    rng = np.random.default_rng(int(batch_seed) if deterministic else None)
    if deterministic:
        clear_cifar_experiment_cache()
        loader_seed_random = int(batch_seed)
    else:
        loader_seed_random = int(rng.integers(0, 2**31 - 1))
    alpha_list = [float(alpha) for alpha in np.asarray(alphas, dtype=float)]
    num_inits = max(1, int(num_model_inits))
    depth = int(model_cfg.depth)
    batch_size = max(1, int(batch_size))
    measurement_layers = tuple(int(layer) for layer in measurement_layers)
    activation_stat_layers = tuple(int(layer) for layer in activation_stat_layers)
    num_batches = int(math.ceil(int(n_samples) / float(batch_size)))

    bundle = {
        "kind": f"{direction}_apjn_activation_stats",
        "model_cfg": cfg_to_dict(model_cfg),
        "depth": int(depth),
        "n_tokens_ex_cls": None,
        "alphas": alpha_list,
        "layers": {
            f"{direction}_apjn": list(measurement_layers),
            "activation_stats": list(activation_stat_layers),
        },
        "config": {
            "n_samples": int(n_samples),
            "batch_size": int(batch_size),
            "batch_seed": int(batch_seed),
            "loader_seed_random": int(loader_seed_random),
            "j_num_draws": int(j_num_draws),
            "hutchinson_block_size": int(hutchinson_block_size),
            "num_model_inits": int(num_inits),
            "attn_mult": float(attn_mult),
            "mlp_mult": float(mlp_mult),
            "keep_pre_transformer_init": bool(keep_pre_transformer_init),
            "skip_preln": bool(skip_preln),
            "deterministic": bool(deterministic),
            "randomized_sampling": not bool(deterministic),
            "batch_filtering": "none",
            "per_sample_from_batched_passes": True,
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
    batch_records = []
    for draw_index in range(num_batches):
        seed_all(int(loader_seed_random) + int(draw_index))
        samples, targets, batch_meta = get_cifar_batch(
            batch_size=int(batch_size),
            img_size=model_cfg.img_size,
            num_classes=model_cfg.num_classes,
            loader_seed=int(loader_seed_random),
            draw_index=int(draw_index),
        )
        batch_n = int(samples.shape[0])
        batch_records.append({
            "draw_index": int(draw_index),
            "samples": samples,
            "targets": targets,
            "batch_meta": dict(batch_meta),
            "batch_n": int(batch_n),
            "preln_values_sum": ({int(layer): np.zeros(batch_n, dtype=float) for layer in measurement_layers} if not bool(skip_preln) else {}),
            "preln_stats_sum": (
                {int(layer): {"q": np.zeros(batch_n, dtype=float), "p": np.zeros(batch_n, dtype=float)} for layer in activation_stat_layers}
                if not bool(skip_preln) else {}
            ),
            "derf_values_sum": {float(alpha): {int(layer): np.zeros(batch_n, dtype=float) for layer in measurement_layers} for alpha in alpha_list},
            "derf_stats_sum": {
                float(alpha): {int(layer): {"q": np.zeros(batch_n, dtype=float), "p": np.zeros(batch_n, dtype=float)} for layer in activation_stat_layers}
                for alpha in alpha_list
            },
        })

    if bundle["n_tokens_ex_cls"] is None:
        preview_model = build_vit(model_cfg, use_derf=False)
        bundle["n_tokens_ex_cls"] = int(get_vit_seq_len_and_dim(preview_model)[0] - 1)
        del preview_model
        cuda_cleanup()

    total_stages = (0 if bool(skip_preln) else 1) + len(alpha_list)
    total_batch_passes = len(batch_records) * int(num_inits) * int(total_stages)
    progress = tqdm(total=total_batch_passes, desc=f"run_cifar_{direction}_apjn_with_activation_stats", leave=False)

    try:
        if not bool(skip_preln):
            for init_idx in range(int(num_inits)):
                progress.set_description(f"{direction} preln init {int(init_idx) + 1}/{int(num_inits)}")
                seed_all(int(model_cfg.seed) + int(init_idx))
                model = build_vit(model_cfg, use_derf=False)
                if keep_pre_transformer_init and preln_pre_blocks_state is not None:
                    _load_partial_state_dict_(model, preln_pre_blocks_state)
                scale_vit_mlp_and_value_attn_init_std(model, mlp_multiplier=float(mlp_mult), attn_multiplier=float(attn_mult))
                if keep_pre_transformer_init and preln_pre_blocks_state is None:
                    preln_pre_blocks_state = _extract_vit_pre_blocks_state_dict(model)
                try:
                    for batch_idx, rec in enumerate(batch_records, start=1):
                        progress.set_postfix_str(f"batch {int(batch_idx)}/{len(batch_records)}")
                        if direction == "backward":
                            values, stats = estimate_backward_J_and_activation_stats_hutchinson_batched(
                                model,
                                rec["samples"],
                                l0_list=measurement_layers,
                                activation_layer_list=activation_stat_layers,
                                j_num_draws=int(j_num_draws),
                                j_normalize_by="Y",
                                draw_block_size=int(hutchinson_block_size),
                            )
                        else:
                            values, stats = estimate_forward_J_and_activation_stats_hutchinson_batched(
                                model,
                                rec["samples"],
                                forward_layers=measurement_layers,
                                forward_source_block=int(source_block_index),
                                activation_layer_list=activation_stat_layers,
                                j_num_draws=int(j_num_draws),
                                j_normalize_by="Y",
                                draw_block_size=int(hutchinson_block_size),
                            )
                        for layer in measurement_layers:
                            rec["preln_values_sum"][int(layer)] += np.asarray(values[int(layer)], dtype=float)
                        for layer in activation_stat_layers:
                            rec["preln_stats_sum"][int(layer)]["q"] += np.asarray(stats[int(layer)]["q"], dtype=float)
                            rec["preln_stats_sum"][int(layer)]["p"] += np.asarray(stats[int(layer)]["p"], dtype=float)
                        progress.update(1)
                finally:
                    del model
                    cuda_cleanup()

        for alpha in alpha_list:
            for init_idx in range(int(num_inits)):
                progress.set_description(f"{direction} derf alpha={float(alpha):g} init {int(init_idx) + 1}/{int(num_inits)}")
                seed_all(int(model_cfg.seed) + int(init_idx))
                model = build_vit(model_cfg, use_derf=True)
                if keep_pre_transformer_init and derf_pre_blocks_state is not None:
                    _load_partial_state_dict_(model, derf_pre_blocks_state)
                scale_vit_mlp_and_value_attn_init_std(model, mlp_multiplier=float(mlp_mult), attn_multiplier=float(attn_mult))
                set_all_derf_alpha_(model, float(alpha))
                if keep_pre_transformer_init and derf_pre_blocks_state is None:
                    derf_pre_blocks_state = _extract_vit_pre_blocks_state_dict(model)
                try:
                    for batch_idx, rec in enumerate(batch_records, start=1):
                        progress.set_postfix_str(f"batch {int(batch_idx)}/{len(batch_records)}")
                        if direction == "backward":
                            values, stats = estimate_backward_J_and_activation_stats_hutchinson_batched(
                                model,
                                rec["samples"],
                                l0_list=measurement_layers,
                                activation_layer_list=activation_stat_layers,
                                j_num_draws=int(j_num_draws),
                                j_normalize_by="Y",
                                draw_block_size=int(hutchinson_block_size),
                            )
                        else:
                            values, stats = estimate_forward_J_and_activation_stats_hutchinson_batched(
                                model,
                                rec["samples"],
                                forward_layers=measurement_layers,
                                forward_source_block=int(source_block_index),
                                activation_layer_list=activation_stat_layers,
                                j_num_draws=int(j_num_draws),
                                j_normalize_by="Y",
                                draw_block_size=int(hutchinson_block_size),
                            )
                        for layer in measurement_layers:
                            rec["derf_values_sum"][float(alpha)][int(layer)] += np.asarray(values[int(layer)], dtype=float)
                        for layer in activation_stat_layers:
                            rec["derf_stats_sum"][float(alpha)][int(layer)]["q"] += np.asarray(stats[int(layer)]["q"], dtype=float)
                            rec["derf_stats_sum"][float(alpha)][int(layer)]["p"] += np.asarray(stats[int(layer)]["p"], dtype=float)
                        progress.update(1)
                finally:
                    del model
                    cuda_cleanup()
    finally:
        progress.close()

    for rec in batch_records:
        for batch_position in range(int(rec["batch_n"])):
            if len(bundle["samples"]) >= int(n_samples):
                break
            sample_record = {
                "sample_index": int(len(bundle["samples"])),
                "batch_draw_index": int(rec["draw_index"]),
                "batch_position": int(batch_position),
                "batch_meta": {
                    **dict(rec["batch_meta"]),
                    "batch_draw_index": int(rec["draw_index"]),
                    "batch_position": int(batch_position),
                    "target": int(rec["targets"][int(batch_position)].item()),
                },
                f"{direction}_apjn": {
                    "preln": (
                        {}
                        if bool(skip_preln) else {
                            int(layer): float(rec["preln_values_sum"][int(layer)][int(batch_position)] / float(num_inits))
                            for layer in measurement_layers
                        }
                    ),
                    "derf": {
                        float(alpha): {
                            int(layer): float(rec["derf_values_sum"][float(alpha)][int(layer)][int(batch_position)] / float(num_inits))
                            for layer in measurement_layers
                        }
                        for alpha in alpha_list
                    },
                },
                "activation_stats": {
                    "preln": (
                        {}
                        if bool(skip_preln) else {
                            int(layer): {
                                "q": float(rec["preln_stats_sum"][int(layer)]["q"][int(batch_position)] / float(num_inits)),
                                "p": float(rec["preln_stats_sum"][int(layer)]["p"][int(batch_position)] / float(num_inits)),
                            }
                            for layer in activation_stat_layers
                        }
                    ),
                    "derf": {
                        float(alpha): {
                            int(layer): {
                                "q": float(rec["derf_stats_sum"][float(alpha)][int(layer)]["q"][int(batch_position)] / float(num_inits)),
                                "p": float(rec["derf_stats_sum"][float(alpha)][int(layer)]["p"][int(batch_position)] / float(num_inits)),
                            }
                            for layer in activation_stat_layers
                        }
                        for alpha in alpha_list
                    },
                },
            }
            bundle["samples"].append(sample_record)
        if len(bundle["samples"]) >= int(n_samples):
            break

    if save_results:
        saved_path = _save_bundle_pickle(bundle, save_root=save_root, folder_name=folder_name, filename="results.pkl")
        bundle["saved_path"] = str(saved_path)
    return bundle


def run_cifar_backward_apjn_with_activation_stats(
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
    keep_pre_transformer_init: bool = False,
    skip_preln: bool = False,
    deterministic: bool = False,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
):
    depth = int(model_cfg.depth)
    layers = tuple(layer for layer in range(0, depth + 1, int(layer_stride)) if 0 < layer < depth)
    stat_layers = tuple(resolve_backward_apjn_layers(activation_stat_blocks if activation_stat_blocks is not None else layers, depth))
    return _run_batchwise_apjn_with_activation_stats(
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
        keep_pre_transformer_init=keep_pre_transformer_init,
        skip_preln=skip_preln,
        direction="backward",
        deterministic=deterministic,
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
    keep_pre_transformer_init: bool = False,
    skip_preln: bool = False,
    deterministic: bool = False,
    save_results: bool = False,
    save_root: str = "/tmp/apjn_results",
    result_postfix: str = "",
):
    depth = int(model_cfg.depth)
    source_block_index = int(source_block_index)
    layers = tuple(layer for layer in range(source_block_index, depth + 1, int(layer_stride)) if source_block_index <= layer < depth)
    stat_layers = tuple(resolve_backward_apjn_layers(activation_stat_blocks if activation_stat_blocks is not None else layers, depth))
    return _run_batchwise_apjn_with_activation_stats(
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
        keep_pre_transformer_init=keep_pre_transformer_init,
        skip_preln=skip_preln,
        direction="forward",
        source_block_index=source_block_index,
        deterministic=deterministic,
        save_results=save_results,
        save_root=save_root,
        result_postfix=result_postfix,
    )
