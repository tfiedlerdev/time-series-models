"""Pinball loss and the fixed before/after evals."""

import numpy as np
import torch
from timestral import TimestralForecaster

from training.batches import (
    make_categorical_batch,
    make_exogenous_batch,
    make_history_batch,
    make_lead_batch,
    make_multi_batch,
    make_past_batch,
    make_pool_batch,
)
from training.schedule import CONTEXT_LEN, EVAL_TASKS, HISTORY_KINDS, HORIZON, LONG_CONTEXT
from training.series import make_covariate, make_signal

# The view_curve.py figure, logged at a few counts. That script sweeps every count.
VIEW_COUNTS = (0, 1, 8, 32, 64)
VIEW_POOL = 64
LEAD_DELAY = 24


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor, levels: tuple[float, ...]) -> torch.Tensor:
    """Mean pinball loss. `quantiles` is `(B, V, H, Q)` and `target` is `(B, V, H)`."""
    err = target.unsqueeze(-1) - quantiles
    tau = quantiles.new_tensor(levels)
    return torch.maximum(tau * err, (tau - 1) * err).mean()


def _forecast_mae(
    forecaster: TimestralForecaster,
    context: torch.Tensor,
    known: torch.Tensor | None,
    past: torch.Tensor | None,
    future: torch.Tensor,
) -> float:
    quantiles = forecaster._forecast_raw(
        context,
        torch.ones_like(context, dtype=torch.bool),
        future.shape[-1],
        past_covariates=past,
        known_covariates=known,
    )
    median = quantiles[..., forecaster.median_index]
    return float((median - future).abs().mean())


def _median_mae(forecaster: TimestralForecaster, batches: list[tuple]) -> float:
    errors = [_forecast_mae(forecaster, context, known, past, future) for context, known, past, future, _tag in batches]
    return float(np.median(errors))


@torch.no_grad()
def evaluate(forecaster: TimestralForecaster, n_series: int, seed: int) -> dict[str, float]:
    """Median absolute error for every trained task. Pool counts are drawn first."""
    rng = np.random.default_rng(seed)
    device = forecaster.device
    scores: dict[str, float] = {}
    for n_known in (1, 5, 20, 64, 0):
        batches = [
            make_pool_batch(rng, 1, device, n_known=n_known, distractors=False, past_views=False)
            for _ in range(n_series)
        ]
        scores[f"pool/{n_known}"] = _median_mae(forecaster, batches)
    for kind in HISTORY_KINDS:
        batches = [make_history_batch(rng, 1, kind, CONTEXT_LEN, device, horizon=HORIZON) for _ in range(n_series)]
        scores[f"history/{kind}"] = _median_mae(forecaster, batches)
    for n_known in (1, 3):
        batches = [
            make_exogenous_batch(
                rng,
                1,
                device,
                n_known=n_known,
                n_decoy=0,
                static=False,
                nonnegative=False,
                gate=False,
                horizon=HORIZON,
                trend=False,
            )
            for _ in range(n_series)
        ]
        scores[f"exogenous/{n_known}"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng,
            1,
            device,
            n_known=1,
            n_decoy=1,
            static=False,
            nonnegative=False,
            gate=False,
            horizon=HORIZON,
            decoy_kind="both",
            trend=False,
        )
        for _ in range(n_series)
    ]
    scores["exogenous/select"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng,
            1,
            device,
            n_known=0,
            n_decoy=1,
            static=False,
            nonnegative=False,
            gate=False,
            horizon=HORIZON,
            decoy_kind="both",
            trend=False,
        )
        for _ in range(n_series)
    ]
    scores["exogenous/decoy"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng,
            1,
            device,
            n_known=1,
            n_decoy=1,
            static=False,
            nonnegative=False,
            gate=False,
            horizon=HORIZON,
            decoy_kind="daily",
            trend=False,
        )
        for _ in range(n_series)
    ]
    scores["exogenous/daily"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng,
            1,
            device,
            n_known=1,
            n_decoy=1,
            static=False,
            nonnegative=False,
            gate=False,
            horizon=HORIZON,
            decoy_kind="trend",
            trend=True,
        )
        for _ in range(n_series)
    ]
    scores["exogenous/trend"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng,
            1,
            device,
            n_known=1,
            n_decoy=0,
            static=True,
            nonnegative=True,
            gate=True,
            horizon=HORIZON,
            trend=False,
        )
        for _ in range(n_series)
    ]
    scores["exogenous/retail"] = _median_mae(forecaster, batches)
    batches = [make_past_batch(rng, 1, device, n_past=1) for _ in range(n_series)]
    scores["past/1"] = _median_mae(forecaster, batches)
    batches = [make_multi_batch(rng, 1, device, n_variates=3) for _ in range(n_series)]
    scores["multi/3"] = _median_mae(forecaster, batches)
    for kind in ("train", "walk"):
        batches = [make_history_batch(rng, 1, kind, LONG_CONTEXT, device, horizon=HORIZON) for _ in range(n_series)]
        scores[f"long/{kind}"] = _median_mae(forecaster, batches)
    for mode in ("bins", "random"):
        batches = [make_categorical_batch(rng, 1, device, mode=mode) for _ in range(n_series)]
        scores[f"categorical/{mode}"] = _median_mae(forecaster, batches)
    scores.update(view_curve(forecaster, n_series, seed + 10_000))
    scores.update(lead(forecaster, n_series, seed + 50_000))
    return scores


def lead(forecaster: TimestralForecaster, n_series: int, seed: int, delay: int = LEAD_DELAY) -> dict[str, float]:
    """Error before and after the delay, with the leading measurement and with only the lagging decoy."""
    rng = np.random.default_rng(seed)
    cells: dict[str, list[float]] = {name: [] for name in ("lead/early", "lead/late", "lead/early-none", "lead/late-none")}
    for _ in range(n_series):
        context, _known, past, future, _tag = make_lead_batch(
            rng, 1, forecaster.device, delay=delay, n_decoy=1, calendar=False
        )
        for suffix, rows in (("", past), ("-none", past[:, 1:])):
            quantiles = forecaster._forecast_raw(
                context, torch.ones_like(context, dtype=torch.bool), future.shape[-1], past_covariates=rows
            )
            error = (quantiles[..., forecaster.median_index] - future).abs()[0, 0]
            cells[f"lead/early{suffix}"].append(float(error[:delay].mean()))
            cells[f"lead/late{suffix}"].append(float(error[delay:].mean()))
    return {name: float(np.median(values)) for name, values in cells.items()}


def view_pool(rng: np.random.Generator, n_views: int) -> tuple[np.ndarray, np.ndarray]:
    """One clean signal and `n_views` noisy nonlinear views of it, context and horizon."""
    signal = make_signal(rng, CONTEXT_LEN + HORIZON)
    covariates = np.stack([make_covariate(signal, rng) for _ in range(n_views)])
    return signal, covariates


def split_mae(forecaster: TimestralForecaster, signal: np.ndarray, covariates: np.ndarray, chosen: np.ndarray) -> float:
    """MAE when `chosen` views are known over the horizon and the rest stop at the origin.

    Uses the public forecast, including symmetric averaging, with no non-negative clip.
    That is the view_curve.py measurement.
    """
    rest = np.setdiff1d(np.arange(covariates.shape[0]), chosen)
    kwargs = {"clip_nonnegative": False}
    if chosen.size:
        kwargs["known_covariates"] = covariates[chosen][None, :, :]
    if rest.size:
        kwargs["past_covariates"] = covariates[rest][None, :, :CONTEXT_LEN]
    context = signal[:CONTEXT_LEN][None, None, :]
    median = forecaster.forecast(context, HORIZON, **kwargs).median.detach().cpu().numpy()[0, 0]
    return float(np.mean(np.abs(median - signal[CONTEXT_LEN:])))


def view_curve(
    forecaster: TimestralForecaster,
    n_draws: int,
    seed: int,
    n_views: int = VIEW_POOL,
    counts: tuple[int, ...] = VIEW_COUNTS,
) -> dict[str, float]:
    """Median MAE of the view_curve.py split at a few counts. One new series per draw."""
    rng = np.random.default_rng(seed)
    columns: dict[int, list[float]] = {count: [] for count in counts}
    for _ in range(n_draws):
        signal, covariates = view_pool(rng, n_views)
        for n_known in counts:
            if n_known == 0:
                chosen = np.empty(0, dtype=int)
            elif n_known == n_views:
                chosen = np.arange(n_views)
            else:
                chosen = rng.choice(n_views, n_known, replace=False)
            columns[n_known].append(split_mae(forecaster, signal, covariates, chosen))
    return {f"views/{count}": float(np.median(errors)) for count, errors in columns.items()}


def print_scores(split: str, scores: dict[str, float]) -> None:
    ordered = [name for name in EVAL_TASKS if name in scores]
    print(f"{split:<6} " + "  ".join(f"{name} {scores[name]:.3f}" for name in ordered), flush=True)
