"""Target error as decoys are added, on a fixed series.

Each series is drawn once: the target, its true covariates, and a pool of decoys.
A sweep point keeps that target and those true covariates, and takes the first N
rows of the pool. The `true=0` rows drop the true covariates and keep the same
decoys. Counts default to every integer from 0 through 16.

`easy` decoys are a random walk, a trend, or a different season. `calendar`
decoys are the exogenous rows: both cycles, the daily cycle with the weekly
cycle flipped, or the slope, each plus its own walk.

The fine-tune is the W&B artifact `timestral-5m-pool-mix:latest`, unless
`--artifact` or `--checkpoint` says otherwise. `--pretrained` scores the base
model only.

    uv run python decoy_sweep.py
    uv run python decoy_sweep.py --checkpoint checkpoints/pool-mix-5m.pt
    uv run python decoy_sweep.py --artifact timestral-5m-pool-mix:v2

Selection holds if the `true=1` error stays near the zero-decoy row as decoys
are added, and stays below the `true=0` row.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from timestral import TimestralForecaster

from benchmarks.model import apply_checkpoint
from training.batches import make_exogenous_batch
from training.schedule import CONTEXT_LEN, HISTORY_KINDS, HORIZON
from training.series import make_covariate, make_series, make_signal, pack

DEFAULT_ARTIFACT = "timestral-5m-pool-mix:latest"
COUNTS = tuple(range(17))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="W&B artifact name or alias.")
    parser.add_argument("--checkpoint", type=Path, help="Local weights. Skips the artifact download.")
    parser.add_argument("--pretrained", action="store_true", help="Score the base model only.")
    parser.add_argument("--true", type=int, default=1, help="Dependent covariates held fixed.")
    parser.add_argument("--counts", default=",".join(str(n) for n in COUNTS), help="Decoy counts, comma-separated.")
    parser.add_argument("--series", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    args = parser.parse_args()
    if args.true < 0:
        raise SystemExit("--true must be at least 0")
    counts = _counts(args.counts)
    device = _device(args.device)
    label, path = (None, None) if args.pretrained else _weights(args)
    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
    print(f"device {device}", flush=True)
    rows = _cases(args.true, counts, args.series, args.seed, device)
    with torch.no_grad():
        print("pretrained", flush=True)
        base = _score(forecaster, rows)
        tuned = None
        if path is not None:
            apply_checkpoint(forecaster, path)
            print(label, flush=True)
            tuned = _score(forecaster, rows)
    _print(rows, base, tuned)


def _score(forecaster: TimestralForecaster, rows) -> list[float]:
    scores = []
    for index, (_kind, _true, _decoys, batches) in enumerate(rows, start=1):
        scores.append(_median_mae_batch(forecaster, batches))
        print(f"  {index}/{len(rows)}", flush=True)
    return scores


def _median_mae_batch(forecaster: TimestralForecaster, batches: list[tuple]) -> float:
    """Median of the per-series mean absolute errors. One forward for the cell."""
    context = torch.cat([item[0] for item in batches], dim=0)
    future = torch.cat([item[3] for item in batches], dim=0)
    known = _cat_optional([item[1] for item in batches])
    past = _cat_optional([item[2] for item in batches])
    quantiles = forecaster._forecast_raw(
        context,
        torch.ones_like(context, dtype=torch.bool),
        future.shape[-1],
        past_covariates=past,
        known_covariates=known,
    )
    median = quantiles[..., forecaster.median_index]
    errors = (median - future).abs().flatten(start_dim=1).mean(dim=1)
    return float(errors.median())


def _cat_optional(rows: list[torch.Tensor | None]) -> torch.Tensor | None:
    if rows[0] is None:
        return None
    return torch.cat(rows, dim=0)


def _cases(n_true: int, counts: list[int], n_series: int, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    pool = max(counts)
    rows = []
    trues = (n_true,) if n_true == 0 else (n_true, 0)
    for kind in ("easy", "calendar"):
        drawn = [_draw(kind, n_true, pool, rng, device) for _ in range(n_series)]
        for true in trues:
            for n_decoy in counts:
                batches = [_take(item, true, n_decoy) for item in drawn]
                rows.append((kind, true, n_decoy, batches))
    return rows


def _draw(kind: str, n_true: int, n_pool: int, rng: np.random.Generator, device: torch.device):
    """One target, its true covariates, and `n_pool` decoys, in that row order."""
    if kind == "calendar":
        _context, known, past, future, _tag = make_exogenous_batch(
            rng,
            1,
            device,
            n_known=n_true,
            n_decoy=n_pool,
            static=False,
            nonnegative=False,
            gate=False,
            horizon=HORIZON,
        )
        context = _context
        true_rows = known[:, :n_true] if n_true else None
        decoys = known[:, n_true:] if n_pool else None
        return context, true_rows, decoys, past, future
    length = CONTEXT_LEN + HORIZON
    signal = make_signal(rng, length)
    views = [make_covariate(signal, rng) for _ in range(n_true)]
    extras = [make_series(rng, length, str(rng.choice(HISTORY_KINDS))) for _ in range(n_pool)]
    true_rows = torch.tensor(np.stack(views), dtype=torch.float32, device=device)[None] if views else None
    decoys = torch.tensor(np.stack(extras), dtype=torch.float32, device=device)[None] if extras else None
    context = pack([signal[:CONTEXT_LEN]], False, device)
    future = pack([signal[CONTEXT_LEN:]], False, device)
    return context, true_rows, decoys, None, future


def _take(item, n_true: int, n_decoy: int):
    context, true_rows, decoys, past, future = item
    parts = []
    if n_true and true_rows is not None:
        parts.append(true_rows[:, :n_true])
    if n_decoy and decoys is not None:
        parts.append(decoys[:, :n_decoy])
    known = torch.cat(parts, dim=1) if parts else None
    return context, known, past, future, f"{n_true}+{n_decoy}"


def _print(rows, base: list[float], tuned: list[float] | None) -> None:
    if tuned is None:
        print(f"{'kind':<10} {'true':>5} {'decoys':>7} {'pretrained':>12}")
        for (kind, n_true, n_decoy, _batches), score in zip(rows, base):
            print(f"{kind:<10} {n_true:5d} {n_decoy:7d} {score:12.3f}", flush=True)
        return
    print(f"{'kind':<10} {'true':>5} {'decoys':>7} {'pretrained':>12} {'fine-tune':>12} {'delta':>8}")
    for (kind, n_true, n_decoy, _batches), before, after in zip(rows, base, tuned):
        print(f"{kind:<10} {n_true:5d} {n_decoy:7d} {before:12.3f} {after:12.3f} {after - before:+8.3f}", flush=True)


def _weights(args) -> tuple[str, Path]:
    if args.checkpoint is not None:
        if not args.checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
        return args.checkpoint.name, args.checkpoint
    api = wandb.Api()
    name = args.artifact if "/" in args.artifact else f"{api.default_entity}/time-series-models/{args.artifact}"
    try:
        artifact = api.artifact(name)
    except wandb.Error as error:
        raise SystemExit(f"Could not read W&B artifact {name}: {error}") from error
    directory = Path(artifact.download())
    weights = sorted(directory.glob("*.pt"))
    if len(weights) != 1:
        found = [path.name for path in weights]
        raise SystemExit(f"Expected one .pt file in {name}, found {found}.")
    print(f"using {artifact.name}", flush=True)
    return str(artifact.name), weights[0]


def _counts(text: str) -> list[int]:
    try:
        counts = [int(item) for item in text.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"--counts must be integers, got {text}") from error
    if not counts or any(item < 0 for item in counts):
        raise SystemExit("--counts must be a comma-separated list of non-negative integers")
    return counts


def _device(name: str) -> torch.device:
    if name == "mps":
        raise SystemExit("MPS cannot run Timestral: its scaler uses float64. Use cpu or cuda.")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


if __name__ == "__main__":
    main()
