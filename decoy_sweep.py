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
    parser.add_argument(
        "--artifact",
        default=DEFAULT_ARTIFACT,
        help="W&B artifact name or alias. Several, comma-separated, are scored side by side.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Local weights. Skips the artifact download.")
    parser.add_argument("--pretrained", action="store_true", help="Score the base model only.")
    parser.add_argument("--true", type=int, default=1, help="Dependent covariates held fixed.")
    parser.add_argument("--counts", default=",".join(str(n) for n in COUNTS), help="Decoy counts, comma-separated.")
    parser.add_argument("--series", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--plot", type=Path, help="Save the sweep as a figure, one panel per decoy family.")
    args = parser.parse_args()
    if args.true < 0:
        raise SystemExit("--true must be at least 0")
    counts = _counts(args.counts)
    device = _device(args.device)
    if args.pretrained:
        tuned_models = []
    elif args.checkpoint is not None or "," not in args.artifact:
        tuned_models = [_weights(args)]
    else:
        tuned_models = [_weights(argparse.Namespace(checkpoint=None, artifact=name.strip())) for name in args.artifact.split(",")]
    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
    print(f"device {device}", flush=True)
    rows = _cases(args.true, counts, args.series, args.seed, device)
    columns: dict[str, list[float]] = {}
    with torch.no_grad():
        print("pretrained", flush=True)
        columns["pretrained"] = _score(forecaster, rows)
        for label, path in tuned_models:
            apply_checkpoint(forecaster, path)
            print(label, flush=True)
            columns[label] = _score(forecaster, rows)
    if len(columns) <= 2:
        tuned = None if len(columns) == 1 else list(columns.values())[1]
        _print(rows, columns["pretrained"], tuned)
    else:
        _print_columns(rows, columns)
    if args.plot is not None:
        _plot(rows, columns, args.plot)


def _print_columns(rows, columns: dict[str, list[float]]) -> None:
    labels = list(columns)
    print(f"{'kind':<10} {'true':>5} {'decoys':>7}" + "".join(f" {label[-24:]:>26}" for label in labels))
    for index, (kind, n_true, n_decoy, _batches) in enumerate(rows):
        cells = "".join(f" {columns[label][index]:26.3f}" for label in labels)
        print(f"{kind:<10} {n_true:5d} {n_decoy:7d}{cells}", flush=True)


def _plot(rows, columns: dict[str, list[float]], output: Path) -> None:
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(kind for kind, *_ in rows))
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5), layout="constrained", squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for color, (label, scores) in enumerate(columns.items()):
            for n_true in sorted({row[1] for row in rows}, reverse=True):
                points = [(row[2], score) for row, score in zip(rows, scores) if row[0] == kind and row[1] == n_true]
                xs, ys = zip(*points)
                style = "-" if n_true else "--"
                name = f"{label}, {n_true} true" if n_true else f"{label}, none"
                ax.plot(xs, ys, style, color=f"C{color}", lw=1.5, label=name)
        ax.set_title(f"{kind} decoys")
        ax.set_xlabel("decoys beside the target")
        ax.set_ylabel("median MAE")
    axes[0][0].legend(fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"saved {output}", flush=True)


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
    return args.artifact.split("/")[-1].split(":")[-1], weights[0]


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
