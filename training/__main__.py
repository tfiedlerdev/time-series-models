"""Fine-tune Timestral-5M to pool noisy views without dropping its other forecasts.

Pooling is 15% of the steps: a clean target and noisy views of it. Half of those
batches reveal only some of a pool over the horizon and keep the rest as
past-only rows. Most of those pools are 20 views. A thin tail still uses 64. The other half are known views only, and some of those mix in
unrelated series. Zero known views is an ordinary forecast of that same series.

The exogenous task is 40% of the steps. The target is driven by a latent process.
Some known rows are noisy readings of that process. Decoy rows share one piece of it
and differ on the rest: both cycles, only the daily cycle, or only the slope. That
mismatch is already visible in the context. A quarter of the batches have no true
reading at all. Some are nonnegative,
gated by a known on/off pattern, or shifted by a static category. A past-only task
carries measurements that stop at the forecast origin.

The remaining steps replay other forecasts. History-only batches use a random walk,
a different seasonal pair, or a trend. A few batches forecast several coupled series
jointly, a few read a context of 2048 steps, and a few use a target-encoded categorical
covariate.

The model's scaler uses float64, which Apple MPS cannot run. On a Mac this script
uses the CPU. A CUDA machine, including a DGX Spark, is used when one is present.

The mixture is `configs/data.yaml`. A null task is left out, and each task's share
is its weight divided by the sum of the weights.

    uv run python -m training
    uv run python -m training --data-config configs/data.yaml
"""

import argparse

from training.data_config import DEFAULT_DATA_CONFIG, DataConfig
from training.fit import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-config",
        type=str,
        default=str(DEFAULT_DATA_CONFIG),
        help="YAML mixture. A null task is left out. Shares are weight / sum(weights).",
    )
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
    parser.add_argument(
        "--run-name",
        default="pool-mix-5m",
        help="W&B run name. Also an alias on the model artifact, so a benchmark can ask for it by name.",
    )
    parser.add_argument("--tags", default="", help="Extra W&B tags, comma-separated.")
    args = parser.parse_args()
    args.data = DataConfig.load(args.data_config)
    train(args)


if __name__ == "__main__":
    main()
