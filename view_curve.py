"""Error against how many noisy views are known ahead.

One clean signal, 100 noisy views. A count chooses which views extend over the horizon;
the rest stop at the origin. The same draws are scored with the pretrained 5M model and
with the latest fine-tune, checkpoints/pool-mix-5m.pt. This is the measurement behind
findings/view-curve.md.

    uv run python view_curve.py
"""

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
    print("pretrained", flush=True)
    pretrained = sweep(forecaster, signal, covariates, choices)
    blob = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    forecaster.model.load_state_dict(blob["model"] if isinstance(blob, dict) and "model" in blob else blob)
    print("fine-tune", flush=True)
    finetuned = sweep(forecaster, signal, covariates, choices)

    print(f"{'n':>4} {'pretrained':>12} {'fine-tune':>12} {'delta':>8}")
    for n_known in (0, 1, 4, 8, 16, 32, 64, 100):
        before = float(np.median(pretrained[n_known]))
        after = float(np.median(finetuned[n_known]))
        print(f"{n_known:4d} {before:12.3f} {after:12.3f} {after - before:+8.3f}")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for maes, color, label in ((pretrained, "C0", "pretrained"), (finetuned, "C1", "fine-tune")):
        median = np.median(maes, axis=1)
        q25, q75 = np.percentile(maes, [25, 75], axis=1)
        ax.fill_between(COUNTS, q25, q75, color=color, alpha=0.2)
        ax.plot(COUNTS, median, color=color, lw=1.5, label=label)
    ax.set_xlim(0, N_VIEWS)
    ax.set_xlabel("covariates known at forecast time")
    ax.set_ylabel("MAE of the clean target")
    ax.set_title("Error against how many noisy views are known ahead")
    ax.legend(loc="upper right")
    output = Path("artifacts/view-curve.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
