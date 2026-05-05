# Norm-Free Transformer Experiments

This repo has two main parts:

- APJN utilities for comparing empirical ViT Jacobian norms to mean-field theory.
- ViT training code for pre-LN and Derf models.

## Start Here: APJN Colab Notebook

If your goal is to run, inspect, or extend the APJN experiments, start with [notebooks/apjn_colab_demo.ipynb](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/notebooks/apjn_colab_demo.ipynb).

It is the main entry point for the APJN part of this repo and shows the intended end-to-end workflow. It:

- clones the repo and installs the minimal dependencies,
- runs permutation-symmetric input APJN experiments on ViT models,
- runs backward APJN measurements on CIFAR-100,
- runs forward APJN measurements on CIFAR-100,
- builds theory curves from measured initial conditions,
- plots theory vs. experiment,
- plots depth-wise GMFEs.

If you want to understand how the APJN pipeline is meant to be used end-to-end, start with that notebook.

## APJN Tools

The notebook-facing APJN code lives in:

- [apjn/](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/apjn): computation, theory, experiment runners, save/load helpers.
- [vit_apjn_plots.py](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/vit_apjn_plots.py): plotting utilities.

The main APJN entry points are:

- `run_perm_inv_input_apjn_exp`
  Runs permutation-symmetric input APJN experiments.
- `run_cifar_backward_apjn_with_activation_stats`
  Measures backward APJNs on CIFAR-100 and saves per-sample activation statistics used later for theory initialization.
- `run_cifar_forward_apjn_with_activation_stats`
  Measures forward APJNs on CIFAR-100 and saves per-sample activation statistics.
- `simulate_recursions_full`
  Runs the mean-field recurrence used to produce theoretical `q`, `p`, backward APJN, and forward APJN curves.
- `build_backward_theory_comparison`
  Builds theory-vs-experiment comparisons for backward APJNs from saved CIFAR runs.
- `build_forward_theory_comparison`
  Builds theory-vs-experiment comparisons for forward APJNs from saved CIFAR runs.
- `build_perm_inv_input_exp_theory_comparison`
  Builds theory-vs-experiment comparisons for the permutation-symmetric input experiment.

## Setup

Install the core dependency first:

```bash
pip install timm
```

For CIFAR runs, `--data_set CIFAR --data_path /tmp/cifar100` is enough; the dataset will be downloaded there if needed.

## Training ViTs

The training loop already lives in [main.py](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/main.py). A convenient way to launch runs from a notebook is to build the CLI args with `get_args_parser()` and then call `main.main(args)`.

```python
from pathlib import Path

import main as dyt_main

MODEL_NAME = "vit_base_patch16_224"
EPOCHS = 20
BATCH_SIZE = 128
LR = 3e-4
DEPTH = 36

OUTPUT_DIR = Path("/tmp/vit_run")
LOG_DIR = OUTPUT_DIR / "tb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

parser = dyt_main.get_args_parser()
args = parser.parse_args([
    "--model", MODEL_NAME,
    "--epochs", str(EPOCHS),
    "--batch_size", str(BATCH_SIZE),
    "--lr", str(LR),
    "--min_lr", str(LR),
    "--weight_decay", "0.05",
    "--warmup_epochs", "3",
    "--warmup_steps", "-1",
    "--opt_betas", "0.9", "0.999",
    "--data_set", "CIFAR",
    "--data_path", "/tmp/cifar100",
    "--nb_classes", "100",
    "--input_size", "224",
    "--output_dir", str(OUTPUT_DIR),
    "--log_dir", str(LOG_DIR),
    "--model_depth", str(DEPTH),
    "--save_ckpt", "true",
    "--save_init_ckpt", "true",
    "--save_best_ckpt", "false",
    "--save_best_ema_ckpt", "false",
    "--save_ckpt_freq", "5",
    "--save_ckpt_num", "8",
    "--enable_wandb", "false",
])

dyt_main.main(args)
```

### Important Training Flags

- `--model_depth`: sets the number of transformer blocks passed to `timm.create_model(...)`.
- `--use_mlp_init_std_multiplier true` with `--mlp_init_std_multiplier X`: multiplies the initial MLP `fc1` and `fc2` weights in each ViT block by `X`.
- `--use_attn_value_init_std_multiplier true` with `--attn_value_init_std_multiplier X`: multiplies the attention value projection and attention output projection weights by `X`.
- `--dynamic_erf true`: replaces block-local LayerNorms with `DynamicErf`.
- `--derf_alpha_init_value A`: initializes every `DynamicErf` layer with `alpha=A`.

## Concrete Training Patterns

### Derf Depth Sweep

```python
from pathlib import Path

import main as dyt_main

MODEL_NAME = "vit_base_patch16_224"
EPOCHS = 20
SAVE_FREQ = 5
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

DATA_SET = "CIFAR"
DATA_PATH = "/tmp/cifar100"
NB_CLASSES = 100
INPUT_SIZE = 224

DYNAMIC_ERF = True
DERF_FREEZE_ALPHA = False

BATCH_SIZE = 128
LR = 3e-4
MIN_LR = LR
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 3
WARMUP_STEPS = -1
BETA2 = 0.999

for DERF_ALPHA_INIT_VALUE in [0.7, 0.8]:
    for DEPTH in [30, 36]:
        OUTPUT_DIR = Path(
            f"/content/drive/MyDrive/ml_projects/norm_free_transformer_depth/"
            f"derf_depth_{DEPTH}_epochs_{EPOCHS}_warmup_{WARMUP_EPOCHS}_"
            f"alpha_{DERF_ALPHA_INIT_VALUE}_lr_{LR}_no_lr_decay"
        )
        LOG_DIR = OUTPUT_DIR / "tb"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        parser = dyt_main.get_args_parser()
        args = parser.parse_args([
            "--model", MODEL_NAME,
            "--epochs", str(EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--lr", str(LR),
            "--min_lr", str(MIN_LR),
            "--weight_decay", str(WEIGHT_DECAY),
            "--warmup_epochs", str(WARMUP_EPOCHS),
            "--warmup_steps", str(WARMUP_STEPS),
            "--early_stopping", "true" if EARLY_STOPPING else "false",
            "--early_stopping_patience", str(EARLY_STOPPING_PATIENCE),
            "--opt_betas", "0.9", str(BETA2),
            "--data_set", DATA_SET,
            "--data_path", DATA_PATH,
            "--nb_classes", str(NB_CLASSES),
            "--input_size", str(INPUT_SIZE),
            "--output_dir", str(OUTPUT_DIR),
            "--log_dir", str(LOG_DIR),
            "--model_depth", str(DEPTH),
            "--dynamic_erf", "true" if DYNAMIC_ERF else "false",
            "--derf_alpha_init_value", str(DERF_ALPHA_INIT_VALUE),
            "--derf_freeze_alpha", "true" if DERF_FREEZE_ALPHA else "false",
            "--save_ckpt", "true",
            "--save_init_ckpt", "true",
            "--save_best_ckpt", "false",
            "--save_best_ema_ckpt", "false",
            "--save_ckpt_freq", str(SAVE_FREQ),
            "--save_ckpt_num", "8",
            "--enable_wandb", "false",
        ])
        dyt_main.main(args)
```

### Pre-LN Depth Sweep

This is the same pattern, except `--dynamic_erf false`. You still control depth with `--model_depth`, but the model stays in its standard pre-LN form.

```python
from pathlib import Path

import main as dyt_main

MODEL_NAME = "vit_base_patch16_224"
EPOCHS = 20
DEPTH = 48

parser = dyt_main.get_args_parser()
args = parser.parse_args([
    "--model", MODEL_NAME,
    "--epochs", str(EPOCHS),
    "--batch_size", "128",
    "--lr", "1e-4",
    "--min_lr", "1e-4",
    "--weight_decay", "0.05",
    "--warmup_epochs", "3",
    "--warmup_steps", "-1",
    "--early_stopping", "true",
    "--early_stopping_patience", "10",
    "--opt_betas", "0.9", "0.999",
    "--data_set", "CIFAR",
    "--data_path", "/tmp/cifar100",
    "--nb_classes", "100",
    "--input_size", "224",
    "--output_dir", "/tmp/preln_depth_48",
    "--log_dir", "/tmp/preln_depth_48/tb",
    "--model_depth", str(DEPTH),
    "--dynamic_erf", "false",
    "--derf_alpha_init_value", "0.5",
    "--derf_freeze_alpha", "false",
    "--save_ckpt", "true",
    "--save_init_ckpt", "true",
    "--save_best_ckpt", "false",
    "--save_best_ema_ckpt", "false",
    "--save_ckpt_freq", "5",
    "--save_ckpt_num", "8",
    "--enable_wandb", "false",
])
dyt_main.main(args)
```

### Scaling Both MLP and Attention-Value Initialization

```python
from pathlib import Path

import main as dyt_main

MODEL_NAME = "vit_base_patch16_224"
EPOCHS = 15
DEPTH = 12
DERF_ALPHA_INIT_VALUE = 0.5

for mult in [4.0]:
    parser = dyt_main.get_args_parser()
    args = parser.parse_args([
        "--model", MODEL_NAME,
        "--epochs", str(EPOCHS),
        "--batch_size", "240",
        "--lr", "3e-4",
        "--min_lr", "3e-4",
        "--weight_decay", "0.05",
        "--warmup_epochs", "3",
        "--warmup_steps", "-1",
        "--data_set", "CIFAR",
        "--data_path", "/tmp/cifar100",
        "--nb_classes", "100",
        "--input_size", "224",
        "--output_dir", f"/tmp/derf_mult_{mult}",
        "--log_dir", f"/tmp/derf_mult_{mult}/tb",
        "--model_depth", str(DEPTH),
        "--use_mlp_init_std_multiplier", "true",
        "--mlp_init_std_multiplier", str(mult),
        "--use_attn_value_init_std_multiplier", "true",
        "--attn_value_init_std_multiplier", str(mult),
        "--dynamic_erf", "true",
        "--derf_alpha_init_value", str(DERF_ALPHA_INIT_VALUE),
        "--derf_freeze_alpha", "false",
        "--save_ckpt", "true",
        "--save_init_ckpt", "true",
        "--save_best_ckpt", "false",
        "--save_best_ema_ckpt", "false",
        "--save_ckpt_freq", "5",
        "--save_ckpt_num", "8",
        "--enable_wandb", "false",
    ])
    dyt_main.main(args)
```
