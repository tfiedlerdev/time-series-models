"""Error against how many noisy views are known ahead.

One clean signal, 100 noisy views. A count chooses which views extend over the horizon;
the rest stop at the origin. The same draws are scored with the pretrained 5M model and
with each fine-tune, by default checkpoints/pool-mix-5m.pt. This is the measurement behind
findings/view-curve.md.

    uv run python view_curve.py
    uv run python view_curve.py --checkpoint teacher=runs/a.pt --checkpoint heavy=runs/b.pt --output artifacts/curve.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from timestral import TimestralForecaster

from training.eval import split_mae, view_pool

CHECKPOINT = Path("checkpoints/pool-mix-5m.pt")
N_VIEWS = 100
N_DRAWS = 8
COUNTS = np.arange(N_VIEWS + 1)


def sweep(forecaster: TimestralForecaster, signal: np.ndarray, covariates: np.ndarray, choices: list[list[np.ndarray]]) -> np.ndarray:
    maes = np.empty((N_VIEWS + 1, N_DRAWS))
    for n_known, draws in enumerate(choices):
        for draw, chosen in enumerate(draws):
            maes[n_known, draw] = split_mae(forecaster, signal, covariates, chosen)
        if len(draws) == 1:
            maes[n_known, :] = maes[n_known, 0]
        if n_known % 10 == 0:
            print(f"n={n_known:3d}  median MAE {np.median(maes[n_known]):.3f}", flush=True)
    return maes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help=f"A fine-tune to score, repeatable. Defaults to fine-tune={CHECKPOINT}.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/view-curve.png"))
    args = parser.parse_args()
    checkpoints = [item.split("=", 1) for item in args.checkpoint or [f"fine-tune={CHECKPOINT}"]]

    rng = np.random.default_rng(0)
    signal, covariates = view_pool(rng, N_VIEWS)
    draw_rng = np.random.default_rng(1)
    choices: list[list[np.ndarray]] = []
    for n_known in COUNTS:
        repeats = 1 if n_known in (0, N_VIEWS) else N_DRAWS
        if n_known == 0:
            draws = [np.empty(0, dtype=int)]
        elif n_known == N_VIEWS:
            draws = [np.arange(N_VIEWS)]
        else:
            draws = [draw_rng.choice(N_VIEWS, n_known, replace=False) for _ in range(repeats)]
        choices.append(draws)

    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=args.device)
    print("pretrained", flush=True)
    curves = {"pretrained": sweep(forecaster, signal, covariates, choices)}
    for name, path in checkpoints:
        blob = torch.load(path, map_location=args.device, weights_only=True)
        forecaster.model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
        print(name, flush=True)
        curves[name] = sweep(forecaster, signal, covariates, choices)

    print(f"{'n':>4}" + "".join(f" {name:>12}" for name in curves))
    for n_known in (0, 1, 4, 8, 16, 32, 64, 100):
        print(f"{n_known:4d}" + "".join(f" {float(np.median(maes[n_known])):12.3f}" for maes in curves.values()))

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for index, (label, maes) in enumerate(curves.items()):
        color = f"C{index}"
        median = np.median(maes, axis=1)
        q25, q75 = np.percentile(maes, [25, 75], axis=1)
        ax.fill_between(COUNTS, q25, q75, color=color, alpha=0.2)
        ax.plot(COUNTS, median, color=color, lw=1.5, label=label)
    ax.set_xlim(0, N_VIEWS)
    ax.set_xlabel("covariates known at forecast time")
    ax.set_ylabel("MAE of the clean target")
    ax.set_title("Error against how many noisy views are known ahead")
    ax.legend(loc="upper right")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
