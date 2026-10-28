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


@functools.cache
def load_multivariate(name: str, max_series: int, min_length: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """`(train, heldout)` float32 series with two or more variates, split like `load_dataset`."""
    import datasets

    table = datasets.load_from_disk(str(REAL_ROOT / name)).with_format("numpy", columns=["target"])
    order = np.random.default_rng(0).permutation(len(table))
    train: list[np.ndarray] = []
    heldout: list[np.ndarray] = []
    for position, index in enumerate(order[: max_series + max_series // HELDOUT_EVERY + 1]):
        target = np.asarray(table[int(index)]["target"], dtype=np.float32)
        if target.ndim != 2 or target.shape[0] < 2 or target.shape[1] < min_length:
            continue
        (heldout if position % HELDOUT_EVERY == 0 else train).append(target)
    return tuple(train[:max_series]), tuple(heldout)


def draw_covariate_window(
    rng: np.random.Generator,
    series: tuple[np.ndarray, ...],
    context_len: int,
    horizons: list[int],
    min_context: int,
    max_covariates: int,
    tries: int = 20,
    horizon_weights: list[float] | None = None,
    finite_future: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """`(context, future, covariate context, covariate horizon)` from one multivariate series.

    One variate is the target. Up to `max_covariates` of the others are covariates over the
    context and the horizon; the caller decides which of them stop at the origin. Teacher
    labels never read the future, so `finite_future=False` keeps windows with gaps there,
    which would otherwise leave mostly short horizons on gappy series.
    """
    for _ in range(tries):
        array = series[int(rng.integers(0, len(series)))]
        horizon = _fitting_horizon(rng, horizons, horizon_weights, array.shape[1] - min_context)
        if horizon is None:
            continue
        length = min(context_len, array.shape[1] - horizon)
        start = int(rng.integers(0, array.shape[1] - horizon - length + 1))
        order = rng.permutation(array.shape[0])
        target, others = order[0], order[1 : 1 + max_covariates]
        window = array[:, start : start + length + horizon]
        context, future = window[target, :length], window[target, length:]
        seen = np.isfinite(context)
        if (finite_future and not np.isfinite(future).all()) or seen.mean() < 0.5 or not seen[-1]:
            continue
        if float(np.std(context[seen])) <= 0:
            continue
        rows = window[others]
        if (np.isfinite(rows).mean(axis=1) < 0.5).any():
            continue
        return context, future, rows[:, :length], rows[:, length:]
    return None


def draw_window(
    rng: np.random.Generator,
    rows: tuple[np.ndarray, ...],
    context_len: int,
    horizons: list[int],
    min_context: int,
    tries: int = 20,
    horizon_weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """A context and a fully observed future from one row, or None when no try fits."""
    for _ in range(tries):
        row = rows[int(rng.integers(0, len(rows)))]
        horizon = _fitting_horizon(rng, horizons, horizon_weights, row.shape[0] - min_context)
        if horizon is None:
            continue
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


def _fitting_horizon(
    rng: np.random.Generator, horizons: list[int], weights: list[float] | None, limit: int
) -> int | None:
    """A horizon no longer than `limit`, drawn by `weights` among the ones that fit."""
    weights = weights or [1.0] * len(horizons)
    fitting = [(horizon, weight) for horizon, weight in zip(horizons, weights) if horizon <= limit]
    if not fitting:
        return None
    chances = np.array([weight for _, weight in fitting], dtype=np.float64)
    return int(fitting[int(rng.choice(len(fitting), p=chances / chances.sum()))][0])


def pack_window(context: np.ndarray, future: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(context, dtype=torch.float32, device=device)[None, None, :],
        torch.tensor(future, dtype=torch.float32, device=device)[None, None, :],
    )
