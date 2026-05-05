from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


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


def resolve_float_key(d, target, tol=1e-9):
    keys = [float(key) for key in d.keys()]
    if not keys:
        raise KeyError("Dictionary has no keys.")
    best = min(keys, key=lambda key: abs(key - float(target)))
    if abs(best - float(target)) > tol:
        raise KeyError(f"Could not match target={target} to available keys={keys}.")
    return best


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
