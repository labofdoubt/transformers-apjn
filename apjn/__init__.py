from .analysis import (
    build_backward_theory_comparison,
    build_forward_theory_comparison,
    build_perm_inv_input_exp_theory_comparison,
    compute_depthwise_gmfe_records,
)
from .config import MeanFieldConfig, ModelConfig, PermSymInputExpConfig, cfg_to_dict
from .data import clear_cifar_experiment_cache, get_cifar_batch, get_synth_images_batch
from .experiments import (
    run_cifar_backward_apjn_with_activation_stats,
    run_cifar_forward_apjn_with_activation_stats,
    run_perm_inv_input_apjn_exp,
)
from .io_utils import (
    compute_error_metrics,
    load_saved_bundle,
    resolve_float_key,
    split_layers_into_three_regions,
)
from .measurements import (
    collect_block0_input_stats,
    estimate_backward_J_and_activation_stats_hutchinson_batched,
    estimate_forward_J_and_activation_stats_hutchinson_batched,
    mean_all_pairs_token_dot_over_d,
    mean_all_pairs_token_dot_over_d_per_sample,
    mean_token_sqnorm_over_d,
    mean_token_sqnorm_over_d_per_sample,
    resolve_backward_apjn_layers,
    resolve_forward_apjn_layers,
)
from .models import (
    build_vit,
    capture_X_list_and_logits,
    get_vit_seq_len_and_dim,
    make_apjn_equiangular_block0_batch,
    make_equiangular_tokens,
    scale_vit_mlp_and_value_attn_init_std,
    set_all_derf_alpha_,
)
from .runtime import DEVICE, cuda_cleanup, seed_all, tqdm
from .theory import (
    build_mean_field_cfg_for_vit_base,
    compute_theory_bundle,
    kappa_relu_np,
    simulate_recursions_full,
    tilde_p_erf_np,
    tilde_q_erf_np,
)
