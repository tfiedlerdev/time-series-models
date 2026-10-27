"""Synthetic series used by every training task."""

import numpy as np
import torch


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    out[steps:] = values[:-steps]
    return out


def observe(values: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise whose scale is `noise_frac` of this series' own spread."""
    out = values.astype(float).copy()
    seen = np.isfinite(out)
    out[seen] += rng.normal(0, noise_frac * float(np.std(out[seen])), int(seen.sum()))
    return out


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** power


def walk(rng: np.random.Generator, length: int, scale: float) -> np.ndarray:
    innov = rng.normal(0.0, scale, length)
    out = np.empty(length)
    out[0] = innov[0]
    for index in range(1, length):
        out[index] = 0.97 * out[index - 1] + innov[index]
    return out


def seasonal_parts(length: int, daily_phase: float, weekly_phase: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(length)
    daily = np.sin(2 * np.pi * time / 24.0 + daily_phase)
    weekly = 0.6 * np.sin(2 * np.pi * time / 168.0 + weekly_phase)
    return daily, weekly


def calendar(length: int, daily_phase: float, weekly_phase: float) -> np.ndarray:
    daily, weekly = seasonal_parts(length, daily_phase, weekly_phase)
    return daily + weekly


def make_signal(rng: np.random.Generator, length: int) -> np.ndarray:
    """Seasonal swing plus a mean-reverting walk, the signal measured in view_curve.py."""
    innov = rng.normal(0, 0.35, length)
    signal = np.empty(length)
    signal[0] = innov[0]
    for i in range(1, length):
        signal[i] = 0.98 * signal[i - 1] + innov[i]
    signal += 1.5 * np.sin(2 * np.pi * np.arange(length) / 64)
    return signal


def make_series(rng: np.random.Generator, length: int, kind: str) -> np.ndarray:
    """A series whose own future is the forecast target."""
    if kind == "train":
        return make_signal(rng, length)
    if kind == "walk":
        return np.cumsum(rng.normal(0, 0.4, length))
    if kind == "season":
        time = np.arange(length)
        return (
            2.0 * np.sin(2 * np.pi * time / 24)
            + 0.8 * np.sin(2 * np.pi * time / 168)
            + rng.normal(0, 0.3, length)
        )
    if kind == "trend":
        time = np.arange(length)
        return float(rng.normal()) + float(rng.normal(0, 0.01)) * time + rng.normal(0, 0.4, length)
    raise ValueError(kind)


def make_covariate(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy nonlinear view. About half are delayed."""
    lag_k = 0 if rng.random() < 0.5 else int(rng.integers(1, 41))
    src = signal if lag_k == 0 else lag(signal, lag_k)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        values = signed_power(src, float(rng.uniform(0.4, 1.8)))
    elif kind == 1:
        values = np.tanh(src / float(rng.uniform(0.5, 3.0)))
    else:
        values = src / (1 + float(rng.uniform(0.1, 0.8)) * src**2)
    return observe(values, float(rng.uniform(0.1, 0.7)), rng)


def mask_stretches(
    rows: list[np.ndarray],
    rng: np.random.Generator,
    n_stretches: int,
    gap_probability: float,
    switch_from: int,
    switch_to: int,
) -> None:
    """Hide readings in place, one stretch at a time, so a different one is present in each.

    Each stretch keeps one reading and sets the others to NaN; the kept reading changes
    from one stretch to the next. With a single reading, its stretches alternate between
    present and missing. A stretch is all missing with `gap_probability`. One boundary
    always falls in `[switch_from, switch_to)`, so the switch can land near the origin or
    on the horizon.
    """
    length = rows[0].shape[0]
    grid = np.arange(16, length - 15, 16)
    first = int(rng.integers(switch_from, switch_to))
    others = rng.choice(grid[np.abs(grid - first) >= 16], max(0, n_stretches - 2), replace=False)
    bounds = [0, *sorted({first, *others.tolist()}), length]
    choices = len(rows) if len(rows) > 1 else 2
    keep = int(rng.integers(choices))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        present = None if rng.random() < gap_probability else keep
        for index, row in enumerate(rows):
            if index != present:
                row[start:stop] = np.nan
        keep = (keep + 1 + int(rng.integers(choices - 1))) % choices


def pack(rows: list[np.ndarray], variates: bool, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(np.stack(rows), dtype=torch.float32, device=device)
    return tensor if variates else tensor[:, None, :]
