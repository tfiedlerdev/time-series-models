"""Training-data mixture. Each task is optional; a null task is left out.

A task's share of the draws is its weight divided by the sum of the weights of the
tasks that are set. The checked-in file is `configs/data.yaml`.
"""

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_yaml import parse_yaml_file_as

DEFAULT_DATA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
SERIES_KINDS = frozenset({"walk", "season", "trend", "kernel", "composite", "intermittent"})
DISTILL_LABELS = frozenset({"teacher", "truth"})
DECOY_KINDS = frozenset({"both", "daily", "trend"})


def _known(kinds: list[str], allowed: frozenset[str], label: str) -> list[str]:
    unknown = [kind for kind in kinds if kind not in allowed]
    if unknown:
        raise ValueError(f"unknown {label} {unknown}; allowed values are {sorted(allowed)}")
    return kinds


class Weighted(BaseModel):
    """One discrete draw. `weights` defaults to equal weight on every value."""

    model_config = ConfigDict(extra="forbid")

    values: list[int] = Field(min_length=1)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _weights_match_values(self) -> "Weighted":
        if self.weights is None:
            self.weights = [1.0] * len(self.values)
        elif len(self.weights) != len(self.values):
            raise ValueError(f"weights has length {len(self.weights)} and values has length {len(self.values)}")
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("weights must be positive")
        return self

    def draw(self, rng: np.random.Generator) -> int:
        weights = np.asarray(self.weights, dtype=np.float64)
        return int(rng.choice(np.asarray(self.values), p=weights / weights.sum()))


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight: float = Field(gt=0)


class PoolConfig(TaskConfig):
    """Noisy views of one clean signal. Some stay known over the horizon."""

    counts: Weighted = Field(
        default_factory=lambda: Weighted(
            values=[0, 1, 2, 4, 8, 16, 32, 64],
            weights=[0.12, 0.16, 0.16, 0.16, 0.16, 0.16, 0.05, 0.03],
        )
    )
    past_view_probability: float = Field(default=0.5, ge=0, le=1)
    past_view_below: int = Field(default=64, ge=1)
    narrow_width: int = Field(default=20, ge=1)
    wide_width: int = Field(default=64, ge=1)
    wide_from: int = Field(default=32, ge=1)
    distractor_probability: float = Field(default=0.5, ge=0, le=1)
    distractor_min: int = Field(default=1, ge=0)
    distractor_max: int = Field(default=8, ge=0)
    distractor_kinds: list[str] = Field(default_factory=lambda: ["walk", "season", "trend"], min_length=1)

    @model_validator(mode="after")
    def _distractor_range(self) -> "PoolConfig":
        if self.distractor_max < self.distractor_min:
            raise ValueError("distractor_max is below distractor_min")
        if self.wide_width < self.narrow_width:
            raise ValueError("wide_width is below narrow_width")
        _known(self.distractor_kinds, SERIES_KINDS, "distractor kinds")
        return self


class HistoryConfig(TaskConfig):
    """Forecast a series from its own past. No covariates."""

    kinds: list[str] = Field(default_factory=lambda: ["walk", "season", "trend"], min_length=1)
    long_horizon_probability: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def _kinds(self) -> "HistoryConfig":
        _known(self.kinds, SERIES_KINDS, "history kinds")
        return self


class ExogenousConfig(TaskConfig):
    """A target driven by a latent walk, a calendar, and sometimes a slope.

    Decoys share one piece of that target. `known_counts` and `decoy_counts` are the
    draws used when a batch still has a true reading.
    """

    no_driver_probability: float = Field(default=0.25, ge=0, le=1)
    no_driver_decoy_counts: list[int] = Field(default_factory=lambda: [1, 2], min_length=1)
    known_counts: Weighted = Field(
        default_factory=lambda: Weighted(values=[1, 2, 3], weights=[2, 1, 1])
    )
    decoy_counts: Weighted = Field(default_factory=lambda: Weighted(values=[1, 2], weights=[2, 1]))
    decoy_kinds: list[str] = Field(default_factory=lambda: ["both", "daily", "trend"], min_length=1)
    long_horizon_probability: float = Field(default=0.5, ge=0, le=1)
    trend_probability: float = Field(default=0.5, ge=0, le=1)
    nonnegative_probability: float = Field(default=0.5, ge=0, le=1)
    gate_probability: float = Field(default=0.5, ge=0, le=1)
    static_probability: float = Field(default=0.5, ge=0, le=1)
    lag_min: int = Field(default=0, ge=0)
    lag_max: int = Field(default=5, ge=0)
    noise_min: float = Field(default=0.05, ge=0)
    noise_max: float = Field(default=0.45, gt=0)

    @model_validator(mode="after")
    def _ranges(self) -> "ExogenousConfig":
        if self.lag_max < self.lag_min:
            raise ValueError("lag_max is below lag_min")
        if self.noise_max < self.noise_min:
            raise ValueError("noise_max is below noise_min")
        if any(count < 0 for count in self.no_driver_decoy_counts):
            raise ValueError("no_driver_decoy_counts must be non-negative")
        _known(self.decoy_kinds, DECOY_KINDS, "decoy kinds")
        return self


class PastConfig(TaskConfig):
    """Lagged measurements that stop at the forecast origin."""

    counts: Weighted = Field(default_factory=lambda: Weighted(values=[1, 2, 3]))
    kinds: list[str] = Field(default_factory=lambda: ["walk", "season", "trend"], min_length=1)
    lag_min: int = Field(default=1, ge=1)
    lag_max: int = Field(default=12, ge=1)
    noise_min: float = Field(default=0.05, ge=0)
    noise_max: float = Field(default=0.3, gt=0)

    @model_validator(mode="after")
    def _ranges(self) -> "PastConfig":
        if self.lag_max < self.lag_min:
            raise ValueError("lag_max is below lag_min")
        if self.noise_max < self.noise_min:
            raise ValueError("noise_max is below noise_min")
        _known(self.kinds, SERIES_KINDS, "past kinds")
        return self


class LeadConfig(TaskConfig):
    """A target that follows an earlier measurement, which is known only up to the origin.

    The target is a nonlinear function of the measurement `delay` steps before, plus a walk
    of its own and sometimes a calendar. The measurement's recent values fix the first
    `delay` steps of the horizon and nothing after. Decoy rows are lagged readings of the
    target itself, which follow it instead of leading it.
    """

    delay_min: int = Field(default=4, ge=1)
    delay_max: int = Field(default=48, ge=1)
    noise_min: float = Field(default=0.05, ge=0)
    noise_max: float = Field(default=0.4, gt=0)
    own_scale: float = Field(default=0.12, ge=0)
    calendar_probability: float = Field(default=0.5, ge=0, le=1)
    decoy_counts: Weighted = Field(default_factory=lambda: Weighted(values=[0, 1, 2], weights=[2, 1, 1]))
    decoy_lag_min: int = Field(default=1, ge=1)
    decoy_lag_max: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _ranges(self) -> "LeadConfig":
        if self.delay_max < self.delay_min:
            raise ValueError("delay_max is below delay_min")
        if self.noise_max < self.noise_min:
            raise ValueError("noise_max is below noise_min")
        if self.decoy_lag_max < self.decoy_lag_min:
            raise ValueError("decoy_lag_max is below decoy_lag_min")
        return self


class MultiConfig(TaskConfig):
    """Several series that share one latent factor, forecast jointly."""

    variate_min: int = Field(default=2, ge=2)
    variate_max: int = Field(default=4, ge=2)
    coupling_min: float = Field(default=0.4, gt=0)
    coupling_max: float = Field(default=1.2, gt=0)
    innovation_scale: float = Field(default=0.25, gt=0)

    @model_validator(mode="after")
    def _ranges(self) -> "MultiConfig":
        if self.variate_max < self.variate_min:
            raise ValueError("variate_max is below variate_min")
        if self.coupling_max < self.coupling_min:
            raise ValueError("coupling_max is below coupling_min")
        return self


class LongConfig(TaskConfig):
    """History-only forecast with a long context. `train` is the pooling signal."""

    kinds: list[str] = Field(default_factory=lambda: ["train", "walk", "season", "trend"], min_length=1)

    @model_validator(mode="after")
    def _kinds(self) -> "LongConfig":
        _known(self.kinds, SERIES_KINDS | {"train"}, "long kinds")
        return self


class CategoricalConfig(TaskConfig):
    """One known categorical covariate, target-encoded the way inference does it."""

    bins_probability: float = Field(default=0.5, ge=0, le=1)
    n_codes: int = Field(default=4, ge=2)
    quantile_edges: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75], min_length=1)

    @model_validator(mode="after")
    def _edges(self) -> "CategoricalConfig":
        if any(edge <= 0 or edge >= 1 for edge in self.quantile_edges):
            raise ValueError("quantile_edges must lie strictly between 0 and 1")
        if list(self.quantile_edges) != sorted(self.quantile_edges):
            raise ValueError("quantile_edges must be increasing")
        return self


class DistillConfig(TaskConfig):
    """History-only replay on broad generic series, at varied context lengths and horizons.

    With `label: teacher` the loss is the distance between the fine-tune's quantiles and
    those of the frozen pretrained model on the same input, so the update pulls the
    forecast back toward the pretrained one wherever no other task asks for a change.
    With `label: truth` it is the ordinary pinball loss against the series itself.
    Covariate rows, when drawn, are unrelated generic series.
    """

    kinds: list[str] = Field(default_factory=lambda: ["kernel", "composite", "intermittent"], min_length=1)
    label: str = "teacher"
    context_lens: Weighted = Field(
        default_factory=lambda: Weighted(values=[256, 512, 1024, 2048, 4096], weights=[1, 2, 2, 2, 1])
    )
    horizons: Weighted = Field(
        default_factory=lambda: Weighted(values=[16, 48, 96, 168, 336], weights=[1, 2, 2, 2, 1])
    )
    past_probability: float = Field(default=0.0, ge=0, le=1)
    known_probability: float = Field(default=0.0, ge=0, le=1)
    covariate_max: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _kinds(self) -> "DistillConfig":
        _known(self.kinds, SERIES_KINDS, "distill kinds")
        if self.label not in DISTILL_LABELS:
            raise ValueError(f"unknown distill label {self.label}; allowed values are {sorted(DISTILL_LABELS)}")
        return self


class DataConfig(BaseModel):
    """Lengths shared by the tasks, and one optional subconfig per task.

    A missing or null task is not drawn. Shares are `weight / sum(weights)`.
    """

    model_config = ConfigDict(extra="forbid")

    context_len: int = Field(default=512, ge=1)
    horizon: int = Field(default=96, ge=1)
    long_context: int = Field(default=2048, ge=1)
    exo_horizon: int = Field(default=168, ge=1)
    pool: PoolConfig | None = None
    history: HistoryConfig | None = None
    exogenous: ExogenousConfig | None = None
    past: PastConfig | None = None
    multi: MultiConfig | None = None
    long: LongConfig | None = None
    categorical: CategoricalConfig | None = None
    lead: LeadConfig | None = None
    distill: DistillConfig | None = None

    @classmethod
    def load(cls, path: Path | str = DEFAULT_DATA_CONFIG) -> "DataConfig":
        return parse_yaml_file_as(cls, path)

    def tasks(self) -> list[tuple[str, TaskConfig]]:
        names = ("pool", "history", "exogenous", "past", "multi", "long", "categorical", "lead", "distill")
        return [(name, getattr(self, name)) for name in names if getattr(self, name) is not None]

    @model_validator(mode="after")
    def _at_least_one_task(self) -> "DataConfig":
        if not self.tasks():
            raise ValueError("at least one data task must be set; a null task is left out of training")
        return self

    def shares(self) -> dict[str, float]:
        active = self.tasks()
        total = sum(config.weight for _, config in active)
        return {name: config.weight / total for name, config in active}

    def draw_task(self, rng: np.random.Generator) -> str:
        shares = self.shares()
        names = list(shares)
        probabilities = np.array([shares[name] for name in names], dtype=np.float64)
        return str(rng.choice(names, p=probabilities))
