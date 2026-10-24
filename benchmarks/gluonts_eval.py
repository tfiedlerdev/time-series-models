"""Score GluonTS test windows the way GIFT-Eval does."""

import numpy as np
import pandas as pd
from gluonts.dataset.util import forecast_start
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import Predictor, evaluate_model
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from benchmarks.model import SeriesInput, forecast_quantiles, quantile_index

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PRIMARY = "MASE[0.5]"
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILES)),
]


class TimestralPredictor(Predictor):
    def __init__(self, forecaster, prediction_length: int, batch_size: int):
        super().__init__(prediction_length=prediction_length)
        self.forecaster = forecaster
        self.batch_size = batch_size

    def predict(self, dataset, **kwargs):
        entries = list(dataset)
        series = [_series_from_entry(entry, self.prediction_length) for entry in entries]
        quantiles, levels, repairs = forecast_quantiles(self.forecaster, series, self.prediction_length, self.batch_size)
        if repairs:
            print(f"  filled {repairs} non-finite forecast steps from the last observation", flush=True)
        for entry, forecast in zip(entries, quantiles):
            yield _quantile_forecast(entry, forecast, levels)


def score_windows(forecaster, test_data, prediction_length: int, batch_size: int, freq: str) -> dict[str, float]:
    try:
        seasonality = int(get_seasonality(freq))
    except Exception:
        seasonality = 1
    frame = evaluate_model(
        TimestralPredictor(forecaster, prediction_length, batch_size),
        test_data=test_data,
        metrics=METRICS,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=seasonality,
    )
    row = frame.iloc[0]
    metrics = {}
    for key in frame.columns:
        value = row[key]
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        if hasattr(value, "item"):
            value = value.item()
        metrics[str(key)] = float(value)
    return metrics


def _series_from_entry(entry: dict, horizon: int) -> SeriesInput:
    target = np.asarray(entry["target"], dtype=np.float64)
    if target.ndim == 1:
        target = target[None, :]
    context_len = target.shape[-1]
    past_rows = []
    known_rows = []
    categorical_past = []
    categorical_known = []

    for field, categorical in (
        ("feat_dynamic_real", False),
        ("feat_dynamic_cat", True),
        ("past_feat_dynamic_real", False),
    ):
        if field not in entry or entry[field] is None:
            continue
        matrix = _dynamic_matrix(entry[field])
        if matrix is None:
            continue
        kind, values = _place_dynamic(matrix, context_len, horizon)
        bucket = categorical_known if categorical and kind == "known" else None
        if categorical and kind == "past":
            bucket = categorical_past
        if not categorical and kind == "known":
            bucket = known_rows
        if not categorical and kind == "past":
            bucket = past_rows
        bucket.append(values)

    for field, categorical in (("feat_static_real", False), ("feat_static_cat", True)):
        if field not in entry or entry[field] is None:
            continue
        static = np.asarray(entry[field], dtype=np.float64).reshape(-1)
        if static.size == 0:
            continue
        repeated = np.repeat(static[:, None], context_len + horizon, axis=1)
        (categorical_known if categorical else known_rows).append(repeated)

    return SeriesInput(
        target=target,
        past=_concat(past_rows),
        known=_concat(known_rows),
        categorical_past=_concat(categorical_past),
        categorical_known=_concat(categorical_known),
    )


def _dynamic_matrix(value) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] == 0:
        return None
    return array


def _place_dynamic(array: np.ndarray, context_len: int, horizon: int) -> tuple[str, np.ndarray]:
    length = array.shape[-1]
    if length == context_len:
        return "past", array
    if length == context_len + horizon:
        return "known", array
    if length > context_len + horizon:
        return "known", array[..., : context_len + horizon]
    if length > context_len:
        missing = horizon - (length - context_len)
        pad = np.full((array.shape[0], missing), np.nan, dtype=np.float64)
        return "known", np.concatenate([array, pad], axis=-1)
    past = np.full((array.shape[0], context_len), np.nan, dtype=np.float64)
    past[:, -length:] = array
    return "past", past


def _concat(rows: list[np.ndarray]) -> np.ndarray | None:
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def _quantile_forecast(entry: dict, quantiles: np.ndarray, levels: tuple[float, ...]) -> QuantileForecast:
    """`quantiles` is `(V, H, Q)`. GluonTS wants `(K, H)` or `(K, H, V)`."""
    median = quantile_index(levels, 0.5)
    indexes = [quantile_index(levels, level) for level in QUANTILES]
    if quantiles.shape[0] == 1:
        rows = [quantiles[0, :, median]]
        keys = ["mean"]
        for level, index in zip(QUANTILES, indexes):
            rows.append(quantiles[0, :, index])
            keys.append(str(level))
        arrays = np.stack(rows, axis=0)
    else:
        rows = [np.transpose(quantiles[:, :, median], (1, 0))]
        keys = ["mean"]
        for level, index in zip(QUANTILES, indexes):
            rows.append(np.transpose(quantiles[:, :, index], (1, 0)))
            keys.append(str(level))
        arrays = np.stack(rows, axis=0)
    return QuantileForecast(
        forecast_arrays=arrays,
        start_date=forecast_start(entry),
        forecast_keys=keys,
        item_id=entry.get("item_id"),
    )
