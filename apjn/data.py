from __future__ import annotations

from types import SimpleNamespace

import torch

from datasets import build_dataset

_CIFAR_DATASET_CACHE: dict[tuple[int, int], object] = {}
_CIFAR_STREAM_CACHE: dict[tuple[int, int, int], dict] = {}


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


def get_cifar_batch(
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


def get_synth_images_batch(batch_size: int, img_size: int):
    samples = torch.randn(int(batch_size), 3, int(img_size), int(img_size), dtype=torch.float32)
    return samples, {"kind": "synthetic_images"}
