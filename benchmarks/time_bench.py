"""TIME tasks. Windows follow the benchmark's dataset class.

The data lives at data/time in this repo:

    hf download Real-TSF/TIME --repo-type dataset --local-dir data/time
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import split
from gluonts.itertools import Map
from toolz import compose

from benchmarks.gift import _itemize_start
from benchmarks.gluonts_eval import PRIMARY, score_windows

CONFIG = Path(__file__).resolve().parent / "tasks" / "time_datasets.yaml"


@dataclass(frozen=True)
class TimeTask:
    name: str
    storage_name: str
    term: str
    prediction_length: int
    test_length: int


def data_directory(override: Path | None) -> Path | None:
    from benchmarks.paths import data_directory as directory

    return directory(override, "TIME_DATASET")


def load_tasks() -> list[TimeTask]:
    document = yaml.safe_load(CONFIG.read_text())
    tasks = []
    for storage_name, spec in document["datasets"].items():
        for term in ("short", "medium", "long"):
            if term not in spec:
                continue
            tasks.append(
                TimeTask(
                    name=f"{storage_name}/{term}",
                    storage_name=storage_name,
                    term=term,
                    prediction_length=int(spec[term]["prediction_length"]),
                    test_length=int(spec["test_length"]),
                )
            )
    return tasks


def evaluate(forecaster, task: TimeTask, batch_size: int, storage: Path) -> tuple[dict[str, float], str]:
    dataset = _TimeDataset(task, storage)
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


class _TimeDataset:
    def __init__(self, task: TimeTask, storage: Path):
        import datasets

        path = storage / task.storage_name
        if not path.exists():
            from benchmarks.paths import download_command

            raise FileNotFoundError(
                f"TIME dataset {task.storage_name} is not at {path}.\n  {download_command('TIME_DATASET')}"
            )
        self.task = task
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
        return self.task.prediction_length

    @property
    def windows(self) -> int:
        return max(1, self.task.test_length // self.prediction_length)

    @property
    def test_data(self):
        _, template = split(self.gluonts_dataset, offset=-self.task.test_length)
        return template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
