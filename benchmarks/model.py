"""Load a Timestral size and forecast a batch of series."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

SIZES = {
    "5M": "mistralai/Timestral-5M",
    "28M": "mistralai/Timestral-28M",
    "123M": "mistralai/Timestral-123M",
    "209M": "mistralai/Timestral-209M",
}


@dataclass
class SeriesInput:
    """One series. `known` and `categorical_known` include the horizon."""

    target: np.ndarray
    past: np.ndarray | None = None
    known: np.ndarray | None = None
    categorical_past: np.ndarray | None = None
    categorical_known: np.ndarray | None = None


def load_forecaster(size: str, device: str) -> TimestralForecaster:
    if device == "mps":
        raise SystemExit("Timestral's scaler uses float64, which MPS cannot run. Use cpu or cuda.")
    if size not in SIZES:
        raise SystemExit(f"Unknown model size {size}. Choose one of {', '.join(SIZES)}.")
    return TimestralForecaster.from_pretrained(SIZES[size], device=device)


def apply_checkpoint(forecaster: TimestralForecaster, path: Path) -> None:
    try:
        blob = torch.load(path, map_location=forecaster.device, weights_only=True)
    except Exception:
        blob = torch.load(path, map_location=forecaster.device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    forecaster.model.load_state_dict(state)


def checkpoint_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def forecast_quantiles(
    forecaster: TimestralForecaster,
    series: list[SeriesInput],
    horizon: int,
    batch_size: int,
) -> tuple[np.ndarray, tuple[float, ...], int]:
    """Forecast every series. Returns `(N, V, H, Q)`, the quantile levels, and a repair count."""
    if not series:
        raise ValueError("no series to forecast")
    prepared = [_clip(item, horizon, forecaster.max_context) for item in series]
    variates = prepared[0].target.shape[0]
    if any(item.target.shape[0] != variates for item in prepared):
        raise ValueError("a batch mixes series with different numbers of variates")

    levels = tuple(float(level) for level in forecaster.quantile_levels)
    output = np.empty((len(prepared), variates, horizon, len(levels)), dtype=np.float64)
    repairs = 0
    for group in _groups(prepared):
        ordered = sorted(group, key=lambda index: prepared[index].target.shape[-1])
        for start in range(0, len(ordered), batch_size):
            indexes = ordered[start : start + batch_size]
            quantiles = _forecast_group(forecaster, [prepared[index] for index in indexes], horizon)
            for local, index in enumerate(indexes):
                repaired, count = _repair(quantiles[local], prepared[index].target)
                output[index] = repaired
                repairs += count
    return output, levels, repairs


def quantile_index(levels: tuple[float, ...], requested: float) -> int:
    index = min(range(len(levels)), key=lambda item: abs(levels[item] - requested))
    if abs(levels[index] - requested) > 1e-6:
        raise ValueError(f"quantile {requested} is not one of {levels}")
    return index


def _clip(series: SeriesInput, horizon: int, max_context: int) -> SeriesInput:
    target = np.asarray(series.target, dtype=np.float64)
    length = target.shape[-1]
    _check_length("past", series.past, length)
    _check_length("known", series.known, length + horizon)
    _check_length("categorical past", series.categorical_past, length)
    _check_length("categorical known", series.categorical_known, length + horizon)
    if length <= max_context:
        return SeriesInput(target, series.past, series.known, series.categorical_past, series.categorical_known)
    start = length - max_context

    def past(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else array[..., start:]

    def known(array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        return np.concatenate([array[..., start:length], array[..., length:]], axis=-1)

    return SeriesInput(
        target[..., start:],
        past(series.past),
        known(series.known),
        past(series.categorical_past),
        known(series.categorical_known),
    )


def _check_length(name: str, array: np.ndarray | None, expected: int) -> None:
    if array is not None and array.shape[-1] != expected:
        raise ValueError(f"{name} covariate length is {array.shape[-1]}, expected {expected}")


def _groups(series: list[SeriesInput]) -> list[list[int]]:
    groups: dict[tuple, list[int]] = {}
    for index, item in enumerate(series):
        key = (
            None if item.past is None else item.past.shape[0],
            None if item.known is None else item.known.shape[0],
            None if item.categorical_past is None else item.categorical_past.shape[0],
            None if item.categorical_known is None else item.categorical_known.shape[0],
        )
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _forecast_group(forecaster: TimestralForecaster, series: list[SeriesInput], horizon: int) -> np.ndarray:
    context = _pad_context([item.target for item in series])
    past = _pad_optional([item.past for item in series])
    categorical_past = _pad_optional([item.categorical_past for item in series])
    known = _pad_known([item.known for item in series], horizon)
    categorical_known = _pad_known([item.categorical_known for item in series], horizon)
    if known is not None and not forecaster.supports_known_covariates:
        raise RuntimeError("This Timestral checkpoint cannot use known-future covariates.")
    forecast = forecaster.forecast(
        context,
        horizon,
        past_covariates=past,
        known_covariates=known,
        categorical_past_covariates=categorical_past,
        categorical_known_covariates=categorical_known,
    )
    return forecast.quantiles.detach().cpu().numpy()


def _pad_context(rows: list[np.ndarray]) -> np.ndarray:
    channels = rows[0].shape[0]
    length = max(row.shape[-1] for row in rows)
    batch = np.full((len(rows), channels, length), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        batch[index, :, -row.shape[-1] :] = row
    return batch


def _pad_optional(rows: list[np.ndarray | None]) -> np.ndarray | None:
    if rows[0] is None:
        return None
    return _pad_context(rows)


def _pad_known(rows: list[np.ndarray | None], horizon: int) -> np.ndarray | None:
    if rows[0] is None:
        return None
    contexts = [row[..., :-horizon] for row in rows]
    futures = np.stack([row[..., -horizon:] for row in rows], axis=0)
    return np.concatenate([_pad_context(contexts), futures], axis=-1)


def _repair(quantiles: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace a non-finite step with the last observed context value."""
    finite = np.isfinite(quantiles).all(axis=-1)
    if bool(finite.all()):
        return quantiles, 0
    repaired = quantiles.copy()
    count = 0
    for variate in range(quantiles.shape[0]):
        observed = context[variate][np.isfinite(context[variate])]
        fill = float(observed[-1]) if len(observed) else 0.0
        missing = ~finite[variate]
        count += int(missing.sum())
        repaired[variate, missing, :] = fill
    return repaired, count
