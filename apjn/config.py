from __future__ import annotations

from dataclasses import asdict, dataclass


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
class PermSymInputExpConfig:
    batch_size: int = 1
    equiangular_q0: float = 1.0
    equiangular_p0: float = 0.0
    equiangular_seed: int = 0
    equiangular_random_rotate: bool = True
    backward_apjn_layers: tuple[int, ...] = (4, 8, 12)
    forward_apjn_layers: tuple[int, ...] = (4, 8, 12)
    forward_source_block: int = 0
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
