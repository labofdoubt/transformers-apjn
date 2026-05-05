# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

from dynamic_tanh import (
    DynamicErf,
    DynamicTanh,
    UnboundedAct,
    convert_ln_to_derf,
    convert_ln_to_dyt,
    convert_ln_to_unbounded_act,
)


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ResidualBranchScale(nn.Module):
    """
    Simple non-trainable multiplier used to rescale ViT residual branches.
    """

    def __init__(self, scale: float):
        super().__init__()
        self.register_buffer("_scale", torch.tensor(float(scale)), persistent=False)

    def forward(self, x):
        return x * self._scale


class LearnableResidualBranchScale(nn.Module):
    """
    Learnable scalar multiplier used to rescale a single ViT residual branch.
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class LearnablePostNormScale(nn.Module):
    """
    Learnable scalar multiplier applied immediately after a normalization module.
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


def apply_residual_branch_scale_vit(model: nn.Module, scale: float):
    """
    Attach a ResidualBranchScale module after each ViT block's drop_path so that the
    residual contribution is multiplied by `scale` before being added.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Residual branch scaling currently supports ViT models with a `blocks` attribute.")

    for block in model.blocks:
        for attr in ("drop_path1", "drop_path2"):
            module = getattr(block, attr, None)
            if module is None:
                continue
            if any(isinstance(child, ResidualBranchScale) for child in module.modules()):
                continue  # avoid double-wrapping
            wrapped = nn.Sequential(module, ResidualBranchScale(scale))
            setattr(block, attr, wrapped)


def apply_learnable_residual_branch_scale_vit(model: nn.Module, init_value: float):
    """
    Attach a learnable scalar after each ViT block's drop_path so each residual
    branch gets its own trainable multiplier before being added back.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Learnable residual branch scaling currently supports ViT models with a `blocks` attribute.")

    for block in model.blocks:
        for attr in ("drop_path1", "drop_path2"):
            module = getattr(block, attr, None)
            if module is None:
                continue
            if any(isinstance(child, (ResidualBranchScale, LearnableResidualBranchScale)) for child in module.modules()):
                continue
            wrapped = nn.Sequential(module, LearnableResidualBranchScale(init_value))
            setattr(block, attr, wrapped)


def apply_learnable_post_norm_scale_vit(model: nn.Module, init_value: float):
    """
    Wrap each ViT normalization module so its output is multiplied by a learnable scalar.
    """
    norm_types = (nn.LayerNorm, DynamicTanh, DynamicErf, UnboundedAct)

    def _wrap(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Sequential) and any(isinstance(grandchild, LearnablePostNormScale) for grandchild in child.children()):
                continue
            if isinstance(child, norm_types):
                module.add_module(name, nn.Sequential(child, LearnablePostNormScale(init_value)))
                continue
            _wrap(child)

    _wrap(model)


def apply_learnable_post_fc_norm_scale_vit(model: nn.Module, init_value: float):
    """
    Wrap only ViT `fc_norm` so its output is multiplied by a learnable scalar.
    """
    if not hasattr(model, "fc_norm"):
        raise ValueError("Post-fc_norm learnable scaling currently supports ViT models with an `fc_norm` attribute.")
    if isinstance(model.fc_norm, nn.Sequential) and any(isinstance(child, LearnablePostNormScale) for child in model.fc_norm.children()):
        return

    model.fc_norm = nn.Sequential(model.fc_norm, LearnablePostNormScale(init_value))


def scale_vit_mlp_init_std(model: nn.Module, multiplier: float):
    """
    Multiply the weights of each ViT block's MLP fc1/fc2 layers by `multiplier`.
    """
    if multiplier <= 0:
        raise ValueError("MLP init std multiplier must be positive.")
    if not hasattr(model, "blocks"):
        raise ValueError("MLP init std scaling currently supports ViT models with a `blocks` attribute.")

    with torch.no_grad():
        for block in model.blocks:
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                continue
            for attr in ("fc1", "fc2"):
                layer = getattr(mlp, attr, None)
                if isinstance(layer, nn.Linear):
                    layer.weight.mul_(multiplier)


def scale_vit_attn_value_and_proj(model: nn.Module, multiplier: float):
    """
    Multiply the attention value projection (within qkv) and output projection weights/biases by `multiplier`.
    """
    if multiplier <= 0:
        raise ValueError("Attention init std multiplier must be positive.")
    if not hasattr(model, "blocks"):
        raise ValueError("Attention init std scaling currently supports ViT models with a `blocks` attribute.")

    with torch.no_grad():
        for block in model.blocks:
            attn = getattr(block, "attn", None)
            if attn is None:
                continue
            qkv = getattr(attn, "qkv", None)
            proj = getattr(attn, "proj", None)
            if isinstance(qkv, nn.Linear):
                dim = qkv.weight.shape[0] // 3
                qkv.weight[2 * dim:] *= multiplier
                if qkv.bias is not None:
                    qkv.bias[2 * dim:] *= multiplier
            if isinstance(proj, nn.Linear):
                proj.weight.mul_(multiplier)
                if proj.bias is not None:
                    proj.bias.mul_(multiplier)


def add_vit_pre_layernorm(model: nn.Module):
    """
    Replace ViT `norm_pre` with a LayerNorm so tokens are normalized before block 0.
    """
    if not hasattr(model, "norm_pre"):
        raise ValueError("Pre-block LayerNorm currently supports ViT models with a `norm_pre` attribute.")
    if not hasattr(model, "num_features"):
        raise ValueError("Unable to infer embedding dimension for ViT pre-block LayerNorm.")

    model.norm_pre = nn.LayerNorm(model.num_features)


def set_vit_block_norm_gamma_init(model: nn.Module, value: float):
    """
    Set the affine weight for normalization modules inside transformer blocks only.
    This excludes `norm_pre` and any final head/fc normalization layers because the
    traversal is restricted to `model.blocks`.
    """
    if not hasattr(model, "blocks"):
        raise ValueError("Block norm gamma initialization currently supports ViT models with a `blocks` attribute.")

    norm_types = (nn.LayerNorm, DynamicTanh, DynamicErf, UnboundedAct)
    with torch.no_grad():
        for block in model.blocks:
            for module in block.modules():
                if isinstance(module, norm_types) and getattr(module, "weight", None) is not None:
                    module.weight.fill_(value)


def set_vit_final_norm_gamma_init(model: nn.Module, value: float):
    """
    Set the affine weight for the final ViT normalization layer only.
    Prefers `fc_norm` when present, otherwise falls back to `norm`.
    """
    norm = None
    if hasattr(model, "fc_norm") and getattr(model.fc_norm, "weight", None) is not None:
        norm = model.fc_norm
    elif hasattr(model, "norm") and getattr(model.norm, "weight", None) is not None:
        norm = model.norm

    if norm is None:
        raise ValueError("Final norm gamma initialization currently supports ViT models with a writable `fc_norm` or `norm` weight.")

    with torch.no_grad():
        norm.weight.fill_(value)


def replace_vit_final_fc_norm_with_derf(model: nn.Module, alpha_init_value: float):
    """
    Replace only ViT `fc_norm` with DynamicErf, leaving block-local norms unchanged.
    """
    if not hasattr(model, "fc_norm"):
        raise ValueError("Final fc_norm replacement currently supports ViT models with an `fc_norm` attribute.")
    if not isinstance(model.fc_norm, nn.LayerNorm):
        raise ValueError("Expected model.fc_norm to be an nn.LayerNorm before DynamicErf replacement.")

    model.fc_norm = DynamicErf(
        model.fc_norm.normalized_shape,
        channels_last=True,
        alpha_init_value=alpha_init_value,
    )

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--model_depth', default=None, type=int,
                        help='Optional override for model depth (number of transformer blocks) when supported by the chosen model. '
                             'For ViT architectures this maps to the depth argument in timm.create_model.')
    parser.add_argument('--use_residual_branch_scale', type=str2bool, default=False,
                        help='If true, multiply each ViT residual branch (attention and MLP path) by a fixed coefficient.')
    parser.add_argument('--residual_branch_scale_coef', type=float, default=None,
                        help='Non-trainable multiplier applied to every ViT residual branch when --use_residual_branch_scale is true.')
    parser.add_argument('--use_learnable_residual_branch_scale', type=str2bool, default=False,
                        help='If true, multiply each ViT residual branch by its own learnable scalar parameter.')
    parser.add_argument('--learnable_residual_branch_scale_init_value', type=float, default=1.0,
                        help='Initialization value for each learnable ViT residual-branch scale.')
    parser.add_argument('--use_learnable_post_layernorm_scale', type=str2bool, default=False,
                        help='If true, multiply the output of each ViT normalization module by a learnable scalar parameter.')
    parser.add_argument('--learnable_post_layernorm_scale_init_value', type=float, default=1.0,
                        help='Initialization value for each learnable post-layernorm scale.')
    parser.add_argument('--use_learnable_post_fc_norm_scale', type=str2bool, default=False,
                        help='If true, multiply only the output of ViT fc_norm by a learnable scalar parameter.')
    parser.add_argument('--learnable_post_fc_norm_scale_init_value', type=float, default=1.0,
                        help='Initialization value for the learnable post-fc_norm scale.')
    parser.add_argument('--use_mlp_init_std_multiplier', type=str2bool, default=False,
                        help='If true, multiply ViT MLP fc1/fc2 weight std by a fixed coefficient right after initialization.')
    parser.add_argument('--mlp_init_std_multiplier', type=float, default=None,
                        help='Scaling coefficient applied to ViT MLP fc1/fc2 weights when --use_mlp_init_std_multiplier is true.')
    parser.add_argument('--use_attn_value_init_std_multiplier', type=str2bool, default=False,
                        help='If true, rescale the attention value/projection weights at initialization.')
    parser.add_argument('--attn_value_init_std_multiplier', type=float, default=None,
                        help='Scaling coefficient applied to attention value and output projection weights when --use_attn_value_init_std_multiplier is true.')
    parser.add_argument('--add_pre_block_layernorm', type=str2bool, default=False,
                        help='If true, replace ViT norm_pre with a LayerNorm so the input is normalized before transformer block 0.')
    parser.add_argument('--block_layernorm_gamma_init_value', type=float, default=None,
                        help='If set, initialize the affine weight of normalization layers inside ViT transformer blocks to this value. Excludes norm_pre and fc_norm.')
    parser.add_argument('--final_layernorm_gamma_init_value', type=float, default=None,
                        help='If set, initialize the affine weight of the final ViT normalization layer (`fc_norm` if present, otherwise `norm`). Excludes `norm_pre`.')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', type=str2bool, default=True)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=3, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")
    
    parser.add_argument('--dynamic_tanh', type=str2bool, default=False)
    parser.add_argument('--dyt_alpha_init_value', type=float, default=0.5,
                        help='Initial value for DyT alpha (only used when --dynamic_tanh true).')
    parser.add_argument('--dynamic_erf', type=str2bool, default=False)
    parser.add_argument('--derf_alpha_init_value', type=float, default=0.5,
                        help='Initial value for Derf alpha (only used when --dynamic_erf true).')
    parser.add_argument('--derf_freeze_alpha', type=str2bool, default=False,
                        help='If true, keep DynamicErf alpha fixed at its initialization value.')
    parser.add_argument('--final_fc_norm_dynamic_erf', type=str2bool, default=False,
                        help='If true, replace only ViT fc_norm with DynamicErf and keep block-local LayerNorms unchanged.')
    parser.add_argument('--final_fc_norm_derf_alpha_init_value', type=float, default=0.5,
                        help='Initial value for DynamicErf alpha when --final_fc_norm_dynamic_erf is true.')
    parser.add_argument('--final_ln_not_replaced', type=str2bool, default=False,
                        help='If true, keep the final ViT LayerNorm (`norm` or `fc_norm`) as LayerNorm when using --dynamic_tanh or --dynamic_erf.')
    parser.add_argument('--unbounded_act', type=str2bool, default=False,
                        help='If true, replace LayerNorm layers with UnboundedAct.')
    parser.add_argument('--unbounded_act_alpha', type=float, default=0.25,
                        help='Alpha for UnboundedAct (only used when --unbounded_act true). Must be in (0, 0.5).')
    parser.add_argument('--save_init_ckpt', type=str2bool, default=False,
                        help='If true, save a checkpoint right after initialization (before training).')
    parser.add_argument('--save_best_ckpt', type=str2bool, default=True,
                        help='If true, save checkpoint-best when validation acc improves.')
    parser.add_argument('--save_best_ema_ckpt', type=str2bool, default=True,
                        help='If true, save checkpoint-best-ema when EMA validation acc improves.')
    parser.add_argument('--early_stopping', type=str2bool, default=False,
                        help='If true, stop training when top-1 validation accuracy does not improve for a patience window.')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of consecutive epochs without top-1 validation accuracy improvements before early stopping. Only used when --early_stopping true.')

    return parser

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if args.early_stopping:
        if args.early_stopping_patience < 1:
            raise ValueError("--early_stopping_patience must be >= 1")
        if data_loader_val is None:
            raise ValueError("Early stopping requires validation data; disable --early_stopping or enable evaluation.")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    if "convnext" in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            ls_init_value=args.layer_scale_init_value,
            head_init_scale=args.head_init_scale,
        )
    elif "vit" in args.model:
        extra_model_kwargs = {}
        if args.model_depth is not None:
            extra_model_kwargs['depth'] = args.model_depth
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
            drop_path_rate=args.drop_path,
            **extra_model_kwargs,
        )
    else:
        raise ValueError(f"Unrecognized model: {args.model}")

    if args.add_pre_block_layernorm:
        if "vit" not in args.model:
            raise ValueError("--add_pre_block_layernorm currently supports ViT models only.")
        add_vit_pre_layernorm(model)

    activation_overrides = [
        args.dynamic_tanh,
        args.dynamic_erf,
        args.final_fc_norm_dynamic_erf,
        args.unbounded_act,
    ]
    if sum(activation_overrides) > 1:
        raise ValueError("Choose only one of --dynamic_tanh, --dynamic_erf, --final_fc_norm_dynamic_erf, or --unbounded_act")
    if args.dynamic_tanh:
        model = convert_ln_to_dyt(
            model,
            alpha_init_value=args.dyt_alpha_init_value,
            final_ln_not_replaced=args.final_ln_not_replaced,
        )
    if args.dynamic_erf:
        model = convert_ln_to_derf(
            model,
            alpha_init_value=args.derf_alpha_init_value,
            freeze_alpha=args.derf_freeze_alpha,
            final_ln_not_replaced=args.final_ln_not_replaced,
        )
    if args.final_fc_norm_dynamic_erf:
        if "vit" not in args.model:
            raise ValueError("--final_fc_norm_dynamic_erf currently supports ViT models only.")
        replace_vit_final_fc_norm_with_derf(
            model,
            alpha_init_value=args.final_fc_norm_derf_alpha_init_value,
        )
    if args.unbounded_act:
        model = convert_ln_to_unbounded_act(model, alpha=args.unbounded_act_alpha)
    if args.block_layernorm_gamma_init_value is not None:
        if "vit" not in args.model:
            raise ValueError("--block_layernorm_gamma_init_value currently supports ViT models only.")
        set_vit_block_norm_gamma_init(model, value=args.block_layernorm_gamma_init_value)
    if args.final_layernorm_gamma_init_value is not None:
        if "vit" not in args.model:
            raise ValueError("--final_layernorm_gamma_init_value currently supports ViT models only.")
        set_vit_final_norm_gamma_init(model, value=args.final_layernorm_gamma_init_value)
    if args.use_learnable_post_layernorm_scale and args.use_learnable_post_fc_norm_scale:
        raise ValueError("Choose only one of --use_learnable_post_layernorm_scale or --use_learnable_post_fc_norm_scale")
    if args.use_learnable_post_layernorm_scale:
        if "vit" not in args.model:
            raise ValueError("--use_learnable_post_layernorm_scale currently supports ViT models only.")
        apply_learnable_post_norm_scale_vit(
            model,
            init_value=args.learnable_post_layernorm_scale_init_value,
        )
    if args.use_learnable_post_fc_norm_scale:
        if "vit" not in args.model:
            raise ValueError("--use_learnable_post_fc_norm_scale currently supports ViT models only.")
        apply_learnable_post_fc_norm_scale_vit(
            model,
            init_value=args.learnable_post_fc_norm_scale_init_value,
        )
    if args.use_residual_branch_scale and args.use_learnable_residual_branch_scale:
        raise ValueError("Choose only one of --use_residual_branch_scale or --use_learnable_residual_branch_scale")
    if args.use_residual_branch_scale:
        if args.residual_branch_scale_coef is None:
            raise ValueError("--residual_branch_scale_coef must be set when --use_residual_branch_scale is true.")
        if "vit" not in args.model:
            raise ValueError("--use_residual_branch_scale currently supports ViT models only.")
        apply_residual_branch_scale_vit(model, scale=args.residual_branch_scale_coef)
    if args.use_learnable_residual_branch_scale:
        if "vit" not in args.model:
            raise ValueError("--use_learnable_residual_branch_scale currently supports ViT models only.")
        apply_learnable_residual_branch_scale_vit(
            model,
            init_value=args.learnable_residual_branch_scale_init_value,
        )
    if args.use_mlp_init_std_multiplier:
        if args.mlp_init_std_multiplier is None:
            raise ValueError("--mlp_init_std_multiplier must be set when --use_mlp_init_std_multiplier is true.")
        if "vit" not in args.model:
            raise ValueError("--use_mlp_init_std_multiplier currently supports ViT models only.")
        scale_vit_mlp_init_std(model, multiplier=args.mlp_init_std_multiplier)
    if args.use_attn_value_init_std_multiplier:
        if args.attn_value_init_std_multiplier is None:
            raise ValueError("--attn_value_init_std_multiplier must be set when --use_attn_value_init_std_multiplier is true.")
        if "vit" not in args.model:
            raise ValueError("--use_attn_value_init_std_multiplier currently supports ViT models only.")
        scale_vit_attn_value_and_proj(model, multiplier=args.attn_value_init_std_multiplier)

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    # Optional: save an init checkpoint (model + optimizer + scaler) before training
    if args.output_dir and args.save_ckpt and args.save_init_ckpt:
        utils.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch="init", model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        return

    max_accuracy = 0.0
    best_accuracy_epoch = args.start_epoch - 1
    epochs_since_improvement = 0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        improved_this_epoch = False
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                best_accuracy_epoch = epoch
                improved_this_epoch = True
                if args.output_dir and args.save_ckpt and args.save_best_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt and args.save_best_ema_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)

        if args.early_stopping and data_loader_val is not None:
            if improved_this_epoch:
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            if epochs_since_improvement >= args.early_stopping_patience:
                best_epoch_display = best_accuracy_epoch if best_accuracy_epoch >= 0 else "N/A"
                print(
                    f"Early stopping triggered: no validation top-1 accuracy improvement for "
                    f"{args.early_stopping_patience} consecutive epochs. "
                    f"Best accuracy {max_accuracy:.2f}% at epoch {best_epoch_display}."
                )
                break

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
