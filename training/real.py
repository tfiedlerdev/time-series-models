"""Windows of real series, for post-training next to the synthetic tasks.

The sources and the reasons some are left out are in `training/real_sources.py`. Every
twentieth series of each dataset is held out of training and only used by the eval.
"""

import functools
from pathlib import Path

import numpy as np
import torch

REAL_ROOT = Path(__file__).resolve().parents[1] / "data" / "gift-pretrain"
HELDOUT_EVERY = 20


@functools.cache
def load_dataset(name: str, max_series: int, min_length: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """`(train, heldout)` univariate float32 rows. A multivariate series gives one row per variate."""
    import datasets

    table = datasets.load_from_disk(str(REAL_ROOT / name)).with_format("numpy", columns=["target"])
    order = np.random.default_rng(0).permutation(len(table))
    train: list[np.ndarray] = []
    heldout: list[np.ndarray] = []
    for position, index in enumerate(order):
        if len(train) >= max_series and len(heldout) >= max(1, max_series // HELDOUT_EVERY):
            break
        target = np.asarray(table[int(index)]["target"], dtype=np.float32)
        rows = target[None, :] if target.ndim == 1 else target
        for row in rows:
            if int(np.isfinite(row).sum()) < min_length:
                continue
            (heldout if position % HELDOUT_EVERY == 0 else train).append(row)
    return tuple(train[:max_series]), tuple(heldout)


def draw_window(
    rng: np.random.Generator,
    rows: tuple[np.ndarray, ...],
    context_len: int,
    horizons: list[int],
    min_context: int,
    tries: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    """A context and a fully observed future from one row, or None when no try fits."""
    for _ in range(tries):
        row = rows[int(rng.integers(0, len(rows)))]
        fitting = [horizon for horizon in horizons if horizon <= (row.shape[0] - min_context)]
        if not fitting:
            continue
        horizon = int(rng.choice(fitting))
        length = min(context_len, row.shape[0] - horizon)
        start = int(rng.integers(0, row.shape[0] - horizon - length + 1))
        context = row[start : start + length]
        future = row[start + length : start + length + horizon]
        seen = np.isfinite(context)
        if not np.isfinite(future).all() or seen.mean() < 0.5 or not seen[-1]:
            continue
        if float(np.std(context[seen])) <= 0:
            continue
        return context, future
    return None


def pack_window(context: np.ndarray, future: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(context, dtype=torch.float32, device=device)[None, None, :],
        torch.tensor(future, dtype=torch.float32, device=device)[None, None, :],
    )
