"""Fine-tune Timestral-5M to pool noisy views without dropping its other forecasts.

Pooling is 30% of the steps: a clean target, nonlinear noisy views known over
the horizon, and a covariate count drawn from 0 through 64. Zero covariates is an
ordinary forecast of that same series. Batches that do have views also get unrelated
series mixed in, so the loss only falls when those are ignored.

The exogenous task is 25% of the steps. The target is driven by a latent process.
Some known rows are noisy readings of that process. Others share its daily and weekly
cycle but have independent innovations, and that mismatch is already visible in the
context. Some of those batches are nonnegative, gated by a known on/off pattern, or
shifted by a static category. A past-only task carries measurements that stop at the
forecast origin.

The remaining steps replay other forecasts. History-only batches use a random walk,
a different seasonal pair, or a trend. A few batches forecast several coupled series
jointly, a few read a context of 2048 steps, and a few use a target-encoded categorical
covariate.

The model's scaler uses float64, which Apple MPS cannot run. On a Mac this script
uses the CPU. A CUDA machine, including a DGX Spark, is used when one is present.

    uv run python -m training
"""

import argparse

from training.fit import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Peak AdamW learning rate. Warmup over 5%% of steps, then cosine decay to a tenth of this.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda. MPS is not supported.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--eval-series", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", default="checkpoints/pool-mix-5m.pt")
    parser.add_argument("--wandb-project", default="time-series-models")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or user. Defaults to the logged-in entity.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip the W&B run and artifact upload.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
