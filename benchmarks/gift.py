"""GIFT-Eval tasks. The split matches the official dataset loader.

The benchmark data lives at data/gift-eval in this repo:

    hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gift-eval
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from pandas.tseries.frequencies import to_offset
from toolz import compose

from benchmarks.gluonts_eval import PRIMARY, score_windows

TEST_FRACTION = 0.1
MAX_WINDOWS = 20
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
M4_HORIZON = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
HORIZON = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
OLD_FREQ = {"Y": "A", "YE": "A", "QE": "Q", "ME": "M", "h": "H", "min": "T", "s": "S", "us": "U"}
PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
PROPERTIES = {
    "m4_yearly": {"domain": "Econ/Fin", "frequency": "A", "num_variates": 1},
    "m4_quarterly": {"domain": "Econ/Fin", "frequency": "Q", "num_variates": 1},
    "m4_monthly": {"domain": "Econ/Fin", "frequency": "M", "num_variates": 1},
    "m4_weekly": {"domain": "Econ/Fin", "frequency": "W", "num_variates": 1},
    "m4_daily": {"domain": "Econ/Fin", "frequency": "D", "num_variates": 1},
    "m4_hourly": {"domain": "Econ/Fin", "frequency": "H", "num_variates": 1},
    "electricity": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "ett1": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "ett2": {"domain": "Energy", "frequency": "W", "num_variates": 7},
    "solar": {"domain": "Energy", "frequency": "W", "num_variates": 1},
    "hospital": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
    "covid_deaths": {"domain": "Healthcare", "frequency": "D", "num_variates": 1},
    "us_births": {"domain": "Healthcare", "frequency": "M", "num_variates": 1},
    "saugeen": {"domain": "Nature", "frequency": "M", "num_variates": 1},
    "temperature_rain": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "kdd_cup_2018": {"domain": "Nature", "frequency": "D", "num_variates": 1},
    "jena_weather": {"domain": "Nature", "frequency": "D", "num_variates": 21},
    "car_parts": {"domain": "Sales", "frequency": "M", "num_variates": 1},
    "restaurant": {"domain": "Sales", "frequency": "D", "num_variates": 1},
    "hierarchical_sales": {"domain": "Sales", "frequency": "W-WED", "num_variates": 1},
    "loop_seattle": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "sz_taxi": {"domain": "Transport", "frequency": "H", "num_variates": 1},
    "m_dense": {"domain": "Transport", "frequency": "D", "num_variates": 1},
    "bitbrains_fast_storage": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bitbrains_rnd": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 2},
    "bizitobs_application": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "bizitobs_service": {"domain": "Web/CloudOps", "frequency": "10S", "num_variates": 2},
    "bizitobs_l2c": {"domain": "Web/CloudOps", "frequency": "H", "num_variates": 7},
}
SHORT = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H "
    "electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D "
    "us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D "
    "ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T "
    "bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)
MED_LONG = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T "
    "LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T "
    "jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)


@dataclass(frozen=True)
class GiftTask:
    name: str
    storage_name: str
    term: str
    domain: str
    num_variates: int


def data_directory(override: Path | None) -> Path | None:
    from benchmarks.paths import data_directory as directory

    return directory(override, "GIFT_EVAL")


def load_tasks() -> list[GiftTask]:
    longer = set(MED_LONG.split())
    tasks = []
    for storage_name in SHORT.split():
        terms = ("short", "medium", "long") if storage_name in longer else ("short",)
        for term in terms:
            name, info = _config_id(storage_name, term)
            tasks.append(GiftTask(name, storage_name, term, info["domain"], info["num_variates"]))
    return tasks


def evaluate(forecaster, task: GiftTask, batch_size: int, storage: Path) -> tuple[dict[str, float], str]:
    dataset = _GiftDataset(task.storage_name, task.term, storage)
    print(
        f"  horizon {dataset.prediction_length}, windows {dataset.windows}, variates {dataset.target_dim}",
        flush=True,
    )
    metrics = score_windows(
        forecaster,
        dataset.test_data,
        dataset.prediction_length,
        batch_size,
        dataset.freq,
    )
    return metrics, PRIMARY


def _config_id(storage_name: str, term: str) -> tuple[str, dict]:
    if "/" in storage_name:
        key, freq = storage_name.split("/", 1)
    else:
        key, freq = storage_name, None
    pretty = PRETTY.get(key.lower(), key.lower())
    info = PROPERTIES[pretty]
    if freq is None:
        freq = info["frequency"]
    return f"{pretty}/{freq}/{term}", info


class _GiftDataset:
    """Same train/test cut as `gift_eval.data.Dataset`, forecasting variates jointly."""

    def __init__(self, name: str, term: str, storage: Path):
        import datasets

        path = storage / name
        if not path.exists():
            from benchmarks.paths import download_command

            raise FileNotFoundError(
                f"GIFT-Eval dataset {name} is not at {path}.\n  {download_command('GIFT_EVAL')}"
            )
        self.name = name
        self.term = term
        self.hf_dataset = datasets.load_from_disk(str(path)).with_format("numpy")
        process = ProcessDataEntry(self.freq, one_dim_target=self.target_dim == 1)
        self.gluonts_dataset = Map(compose(process, _itemize_start), self.hf_dataset)

    @property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @property
    def target_dim(self) -> int:
        target = np.asarray(self.hf_dataset[0]["target"])
        return target.shape[0] if target.ndim > 1 else 1

    @property
    def prediction_length(self) -> int:
        token = norm_freq_str(to_offset(self.freq).name)
        token = OLD_FREQ.get(token, token)
        table = M4_HORIZON if "m4" in self.name else HORIZON
        return TERM_MULTIPLIER[self.term] * table[token]

    @property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        count = math.ceil(TEST_FRACTION * self._min_length / self.prediction_length)
        return min(max(1, count), MAX_WINDOWS)

    @property
    def _min_length(self) -> int:
        target = self.hf_dataset.data.column("target")
        sample = np.asarray(self.hf_dataset[0]["target"])
        if sample.ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(target, 0, 1)))
        else:
            lengths = pc.list_value_length(target)
        return int(min(lengths.to_numpy()))

    @property
    def test_data(self):
        _, template = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        return template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )


def _itemize_start(entry: dict) -> dict:
    start = entry["start"]
    if hasattr(start, "item"):
        entry["start"] = start.item()
    return entry
