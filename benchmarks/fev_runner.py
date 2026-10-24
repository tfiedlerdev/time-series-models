"""fev-bench adapter. The task list is the official benchmark YAML."""

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fev import Benchmark, Task
from fev.metrics import get_metric

from benchmarks.model import SeriesInput, forecast_quantiles, quantile_index

FEV_BENCH_URL = "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml"


@dataclass(frozen=True)
class FevTask:
    name: str
    task: Task

    @property
    def horizon(self) -> int:
        return self.task.horizon


def load_tasks() -> list[FevTask]:
    try:
        benchmark = Benchmark.from_yaml(FEV_BENCH_URL)
    except ValueError as error:
        raise SystemExit(f"Could not download the fev-bench task list from {FEV_BENCH_URL}.") from error
    return [FevTask(task.task_name, task) for task in benchmark.tasks]


def evaluate(forecaster, task: FevTask, batch_size: int, model_name: str) -> tuple[dict[str, float], str]:
    windows = []
    repairs = 0
    for index, window in enumerate(task.task.iter_windows()):
        print(f"  window {index + 1}/{task.task.num_windows}", flush=True)
        predictions, repaired = _predict_window(forecaster, task.task, window, batch_size)
        windows.append(predictions)
        repairs += repaired
    if repairs:
        print(f"  filled {repairs} non-finite forecast steps from the last observation", flush=True)
    summary = task.task.evaluation_summary(windows, model_name=model_name, trained_on_this_dataset=False)
    names = [get_metric(item).name for item in [task.task.eval_metric, *task.task.extra_metrics]]
    metrics = {}
    for key in ["test_error", *names]:
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[key] = float(value)
    return metrics, names[0]


def _predict_window(forecaster, task: Task, window, batch_size: int):
    past, future = window.get_input_data()
    series = _series_from_window(task, past, future)
    quantiles, levels, repairs = forecast_quantiles(forecaster, series, task.horizon, batch_size)
    columns = task.target_columns
    rows = [{column: [] for column in columns} for _ in range(len(series))]
    median_index = quantile_index(levels, 0.5)
    requested = [(level, str(level), quantile_index(levels, level)) for level in task.quantile_levels]
    for series_index in range(len(series)):
        for variate, column in enumerate(columns):
            record = {"predictions": quantiles[series_index, variate, :, median_index].tolist()}
            for _level, key, index in requested:
                record[key] = quantiles[series_index, variate, :, index].tolist()
            rows[series_index][column] = record
    if len(columns) == 1:
        return [row[columns[0]] for row in rows], repairs
    return {column: [row[column] for row in rows] for column in columns}, repairs


def _series_from_window(task: Task, past, future) -> list[SeriesInput]:
    past_table = past.data.table
    future_table = future.data.table
    targets = [_numeric_sequences(past_table, column) for column in task.target_columns]
    count = len(targets[0])
    known_numeric, known_codes = _dynamic_pair(task.known_dynamic_columns, past_table, future_table, task.horizon)
    past_numeric, past_codes = _dynamic_pair(task.past_dynamic_columns, past_table, None, task.horizon)
    static_numeric, static_codes = _static_columns(task.static_columns, past_table)

    series = []
    for index in range(count):
        target = np.stack([column[index] for column in targets], axis=0)
        context_len = target.shape[-1]
        known = _stack_known(known_numeric, static_numeric, index, context_len, task.horizon)
        categorical_known = _stack_known(known_codes, static_codes, index, context_len, task.horizon)
        series.append(
            SeriesInput(
                target=target,
                past=_row(past_numeric, index),
                known=known,
                categorical_past=_row(past_codes, index),
                categorical_known=categorical_known,
            )
        )
    return series


def _dynamic_pair(columns: list[str], past_table, future_table, horizon: int):
    numeric: list[list[np.ndarray]] = []
    categorical: list[list[np.ndarray]] = []
    for column in columns:
        past_kind, past_values = _read_column(past_table, column)
        if future_table is None:
            future_kind, future_values = None, None
        else:
            future_kind, future_values = _read_column(future_table, column)
        if past_kind == "numeric" and future_kind in (None, "numeric"):
            numeric.append(_align_dynamic(past_values, future_values, horizon))
        else:
            encoded_past, encoded_future = _encode_pair(past_values, future_values)
            categorical.append(_align_dynamic(encoded_past, encoded_future, horizon))
    return numeric, categorical


def _align_dynamic(past_values, future_values, horizon: int) -> list[np.ndarray]:
    aligned = []
    for index, past in enumerate(past_values):
        past = np.asarray(past, dtype=np.float64)
        if future_values is None:
            aligned.append(past)
            continue
        future = _fit_horizon(np.asarray(future_values[index], dtype=np.float64), horizon)
        aligned.append(np.concatenate([past, future]))
    return aligned


def _fit_horizon(values: np.ndarray, horizon: int) -> np.ndarray:
    if values.shape[-1] == horizon:
        return values
    if values.shape[-1] > horizon:
        return values[..., :horizon]
    pad = np.full(horizon - values.shape[-1], np.nan, dtype=np.float64)
    return np.concatenate([values, pad])


def _static_columns(columns: list[str], table):
    numeric: list[list[np.ndarray]] = []
    categorical: list[list[np.ndarray]] = []
    for column in columns:
        kind, values = _read_column(table, column)
        if kind == "numeric":
            numeric.append([_as_static_code(item) for item in values])
        else:
            categorical.append(_encode_static(values))
    return numeric, categorical


def _as_static_code(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return np.array([np.nan], dtype=np.float64)
    return array


def _as_matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"covariate has shape {array.shape}")
    return array


def _stack_known(dynamic_columns, static_columns, index: int, context_len: int, horizon: int) -> np.ndarray | None:
    rows = [_as_matrix(column[index]) for column in dynamic_columns]
    width = context_len + horizon
    for column in static_columns:
        codes = np.asarray(column[index], dtype=np.float64).reshape(-1, 1)
        rows.append(np.repeat(codes, width, axis=1))
    if not rows:
        return None
    return np.concatenate(rows, axis=0)


def _row(columns: list[list[np.ndarray]], index: int) -> np.ndarray | None:
    if not columns:
        return None
    return np.concatenate([_as_matrix(column[index]) for column in columns], axis=0)


def _encode_pair(past_values, future_values):
    vocab: dict[str, float] = {}

    def encode(values):
        if values is None:
            return None
        encoded = []
        for sequence in values:
            encoded.append(_encode_sequence(sequence, vocab))
        return encoded

    return encode(past_values), encode(future_values)


def _encode_static(values) -> list[np.ndarray]:
    vocab: dict[str, float] = {}
    encoded = []
    for value in values:
        if value is None:
            encoded.append(np.array([np.nan]))
            continue
        array = np.asarray(value)
        if array.ndim == 0 or array.shape == () or (array.size == 1 and array.dtype != object):
            token = None if value is None else str(value)
            if token is None:
                encoded.append(np.array([np.nan]))
            else:
                encoded.append(np.array([_token(vocab, token)]))
            continue
        encoded.append(_encode_sequence(value, vocab))
    return encoded


def _encode_sequence(sequence, vocab: dict[str, float]) -> np.ndarray:
    array = np.empty(len(sequence), dtype=np.float64)
    for index, value in enumerate(sequence):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            array[index] = np.nan
        else:
            array[index] = _token(vocab, str(value))
    return array


def _token(vocab: dict[str, float], token: str) -> float:
    if token not in vocab:
        vocab[token] = float(len(vocab))
    return vocab[token]


def _numeric_sequences(table, column: str) -> list[np.ndarray]:
    kind, values = _read_column(table, column)
    if kind != "numeric":
        raise ValueError(f"target column {column} is not numeric")
    return [np.asarray(item, dtype=np.float64) for item in values]


def _read_column(table, column: str):
    field = table.column(column)
    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
        lengths = pc.list_value_length(field).to_numpy()
        flat = pc.list_flatten(field)
        value_type = field.type.value_type
        if pa.types.is_floating(value_type) or pa.types.is_integer(value_type):
            numbers = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float64)
            return "numeric", _split(numbers, lengths)
        return "categorical", _split(flat.to_pylist(), lengths)
    if pa.types.is_floating(field.type) or pa.types.is_integer(field.type):
        return "numeric", np.asarray(field.to_numpy(zero_copy_only=False), dtype=np.float64)
    return "categorical", field.to_pylist()


def _split(flat, lengths: np.ndarray) -> list:
    rows = []
    offset = 0
    for length in lengths:
        rows.append(flat[offset : offset + int(length)])
        offset += int(length)
    return rows
