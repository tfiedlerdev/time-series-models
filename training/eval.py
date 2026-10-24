"""Pinball loss and the fixed before/after evals."""

import numpy as np
import torch
from timestral import TimestralForecaster

from training.batches import (
    make_categorical_batch,
    make_exogenous_batch,
    make_history_batch,
    make_multi_batch,
    make_past_batch,
    make_pool_batch,
)
from training.schedule import CONTEXT_LEN, EVAL_TASKS, HISTORY_KINDS, HORIZON, LONG_CONTEXT


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
        batches = [make_pool_batch(rng, 1, device, n_known=n_known, distractors=False) for _ in range(n_series)]
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
            )
            for _ in range(n_series)
        ]
        scores[f"exogenous/{n_known}"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng, 1, device, n_known=1, n_decoy=1, static=False, nonnegative=False, gate=False, horizon=HORIZON
        )
        for _ in range(n_series)
    ]
    scores["exogenous/select"] = _median_mae(forecaster, batches)
    batches = [
        make_exogenous_batch(
            rng, 1, device, n_known=1, n_decoy=0, static=True, nonnegative=True, gate=True, horizon=HORIZON
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
    return scores


def print_scores(split: str, scores: dict[str, float]) -> None:
    ordered = [name for name in EVAL_TASKS if name in scores]
    print(f"{split:<6} " + "  ".join(f"{name} {scores[name]:.3f}" for name in ordered), flush=True)
