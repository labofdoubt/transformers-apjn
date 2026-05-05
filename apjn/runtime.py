from __future__ import annotations

import random

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class _TqdmFallback:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable) if self.iterable is not None else iter(())

        def update(self, n=1):
            return None

        def set_description(self, desc=None):
            return None

        def set_postfix_str(self, s=None):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable=iterable, **kwargs)


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
