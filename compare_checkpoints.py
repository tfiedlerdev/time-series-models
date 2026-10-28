"""Score the pretrained model and several fine-tunes on the training eval suite.

Every model sees the same draws, so the columns are comparable task by task. The
fine-tunes are W&B artifacts, by version or by the run-name alias the training run
gave them, or local checkpoints. Each artifact is recorded as an input of this run.

    uv run python compare_checkpoints.py --artifacts timestral-5m-pool-mix:ref-main
    uv run python compare_checkpoints.py --artifacts a:v1,b:v2 --series 32
    uv run python compare_checkpoints.py --checkpoints checkpoints/pool-mix-5m.pt --no-wandb
"""

import argparse
from pathlib import Path

import torch
import wandb
from timestral import TimestralForecaster

from benchmarks.model import apply_checkpoint
from training.eval import evaluate
from training.fit import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="", help="Comma-separated W&B model artifacts, e.g. timestral-5m-pool-mix:v5.")
    parser.add_argument("--checkpoints", default="", help="Comma-separated local .pt files.")
    parser.add_argument("--series", type=int, default=32, help="Series per eval task.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="compare-checkpoints")
    parser.add_argument("--wandb-project", default="time-series-models")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if args.artifacts and args.no_wandb:
        raise SystemExit("--artifacts needs W&B. Pass --checkpoints instead, or drop --no-wandb.")

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            job_type="compare",
            tags=["timestral-5m", "compare"],
            config={"artifacts": args.artifacts, "checkpoints": args.checkpoints, "series": args.series, "seed": args.seed},
        )
        print(f"wandb {run.url}", flush=True)

    models: list[tuple[str, Path | None]] = [("pretrained", None)]
    for name in _split(args.artifacts):
        models.append((name, _download(run, name)))
    for name in _split(args.checkpoints):
        models.append((Path(name).stem, Path(name)))

    device = resolve_device(args.device)
    scores: dict[str, dict[str, float]] = {}
    for label, path in models:
        forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
        if path is not None:
            apply_checkpoint(forecaster, path)
        print(label, flush=True)
        with torch.no_grad():
            scores[label] = evaluate(forecaster, args.series, args.seed)

    tasks = list(scores["pretrained"])
    labels = [label for label, _ in models]
    print(f"{'task':<24}" + "".join(f"{label[-22:]:>24}" for label in labels))
    for task in tasks:
        print(f"{task:<24}" + "".join(f"{scores[label][task]:>24.3f}" for label in labels))
    if run is not None:
        rows = [[task, *[scores[label][task] for label in labels]] for task in tasks]
        run.log({"eval/mae": wandb.Table(data=rows, columns=["task", *labels])})
        for label in labels:
            for task in tasks:
                run.summary[f"{label}/{task}"] = scores[label][task]
        run.finish()


def _split(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _download(run: wandb.Run, name: str) -> Path:
    artifact = run.use_artifact(name)
    weights = sorted(Path(artifact.download()).glob("*.pt"))
    if len(weights) != 1:
        raise SystemExit(f"Expected one .pt file in {name}, found {[path.name for path in weights]}.")
    return weights[0]


if __name__ == "__main__":
    main()
