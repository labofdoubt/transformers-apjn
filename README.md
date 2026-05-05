# Norm-Free Transformer Experiments

This repo contains ViT training code plus APJN utilities for comparing empirical Jacobian norms to mean-field theory.

## Setup

Install the core dependency first:

```bash
pip install timm
```

For CIFAR runs, `--data_set CIFAR --data_path /tmp/cifar100` is enough; the dataset will be downloaded there if needed.

## Training ViTs from Python

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

### Important APJN-related training flags

- `--model_depth`: overrides the number of transformer blocks passed to `timm.create_model(...)`.
- `--use_mlp_init_std_multiplier true` with `--mlp_init_std_multiplier X`: multiplies the initial MLP `fc1` and `fc2` weights in every ViT block by `X`.
- `--use_attn_value_init_std_multiplier true` with `--attn_value_init_std_multiplier X`: multiplies the attention value projection and attention output projection weights by `X`.
- `--dynamic_erf true`: replaces block-local LayerNorms with `DynamicErf`.
- `--derf_alpha_init_value A`: initializes every `DynamicErf` layer with `alpha=A`.
- `--derf_freeze_alpha true|false`: if `true`, keeps `alpha` fixed during training; if `false`, `alpha` is learned.

## Concrete training patterns

### Derf depth sweep

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
            "--dynamic_erf", "true",
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

### Pre-LN depth sweep

This is the same pattern, except `--dynamic_erf false`. In other words, you still control the depth with `--model_depth`, but you leave the model in its standard pre-LN form.

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

### Scaling both MLP and attention-value initialization

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

## APJN utilities

The refactored notebook-facing utilities now live in:

- [vit_apjn_notebook_helpers.py](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/vit_apjn_notebook_helpers.py): computation, theory, save/load, CIFAR and equiangular APJN runners.
- [vit_apjn_plots.py](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/vit_apjn_plots.py): plotting only.

The main entry points are:

- `simulate_recursions_full`
- `run_cifar_inverse_apjn_with_activation_stats`
- `run_cifar_forward_apjn_with_activation_stats`
- `run_cifar_apjn_experiment` with `input_source="equiangular"`

There is also a Colab notebook in [notebooks/apjn_colab_demo.ipynb](/Users/sergeyalekseev/Desktop/ML_projects/norm_free_transformer/subcritical_signal_prop/notebooks/apjn_colab_demo.ipynb) that clones the repo, runs the APJN workflows, computes theory curves, and plots theory-vs-experiment plus GMFEs.
