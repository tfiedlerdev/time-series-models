"""A benchmark run consumes the training artifact and logs the scores.

`run.use_artifact` is what records the lineage: this run is a downstream
consumer of the model the training run logged.
"""

from pathlib import Path

import wandb

from benchmarks.model import SIZES
from benchmarks.report import mase_key, mean_change, prob_key

DEFAULT_ARTIFACT = "timestral-5m-pool-mix:latest"


def open_run(args) -> wandb.Run | None:
    if args.no_wandb:
        return None
    if not wandb.login(prompt=False):
        raise SystemExit("wandb is not logged in. Run `wandb login`, then start this script again.")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"benchmark-{args.size}",
        group="benchmark",
        job_type="benchmark",
        tags=[f"timestral-{args.size.lower()}", "benchmark"],
        save_code=True,
        config={
            "model": SIZES[args.size],
            "size": args.size,
            "benchmarks": args.benchmarks,
            "tasks": args.tasks,
            "limit": args.limit,
            "batch_size": args.batch_size,
            "device": args.device,
            "artifact": None if args.checkpoint else args.artifact,
        },
    )
    print(f"wandb {run.url}", flush=True)
    return run


def use_model(run: wandb.Run, artifact_name: str, size: str) -> tuple[Path, str]:
    """Download the training artifact and record it as an input of this run."""
    try:
        artifact = run.use_artifact(artifact_name)
    except wandb.Error as error:
        raise SystemExit(f"Could not use W&B artifact {artifact_name}: {error}") from error
    base = (artifact.metadata or {}).get("base_model")
    if base and base != SIZES[size]:
        raise SystemExit(f"Artifact {artifact_name} was trained from {base}, which is not Timestral-{size}.")
    directory = Path(artifact.download())
    weights = sorted(directory.glob("*.pt"))
    if len(weights) != 1:
        names = [path.name for path in weights]
        raise SystemExit(f"Expected one .pt file in {artifact_name}, found {names}.")
    version = artifact.name
    print(f"using {version} ({weights[0].name})", flush=True)
    return weights[0], version


def log_scores(run: wandb.Run | None, benchmark: str, rows: list[dict], compared: bool) -> None:
    if run is None or not rows:
        return
    table = []
    for row in rows:
        current = row.get("current") or {}
        for metric, baseline in row["baseline"].items():
            value = current.get(metric) if compared else None
            delta = None if value is None else value - baseline
            prefix = f"{benchmark}/{row['task']}/{metric}"
            run.summary[f"{prefix}/baseline"] = baseline
            if value is not None:
                run.summary[f"{prefix}/current"] = value
                run.summary[f"{prefix}/delta"] = delta
            table.append([row["task"], metric, baseline, _cell(value), _cell(delta)])
    _log_medians(run, benchmark, rows, compared)
    run.log(
        {
            f"{benchmark}/scores": wandb.Table(
                data=table,
                columns=["task", "metric", "baseline", "current", "delta"],
            )
        }
    )


def _log_medians(run: wandb.Run, benchmark: str, rows: list[dict], compared: bool) -> None:
    metrics = list(rows[0]["baseline"])
    primary = rows[0]["primary"]
    highlighted = [primary]
    for key in (mase_key(rows[0]["baseline"]), prob_key(rows[0]["baseline"])):
        if key and key not in highlighted:
            highlighted.append(key)
    for metric in highlighted:
        if metric not in metrics:
            continue
        baseline = _median([row["baseline"][metric] for row in rows if metric in row["baseline"]])
        run.summary[f"{benchmark}/median/{metric}/baseline"] = baseline
        if not compared:
            continue
        current_values = [
            row["current"][metric] for row in rows if row.get("current") and metric in row["current"]
        ]
        if len(current_values) != len(rows):
            continue
        current = _median(current_values)
        run.summary[f"{benchmark}/median/{metric}/current"] = current
        run.summary[f"{benchmark}/median/{metric}/delta"] = current - baseline
        change = mean_change(rows, metric)
        if change is None:
            continue
        run.summary[f"{benchmark}/mean/{metric}/baseline"] = change["baseline"]
        run.summary[f"{benchmark}/mean/{metric}/current"] = change["current"]
        run.summary[f"{benchmark}/mean/{metric}/delta"] = change["delta"]
        run.summary[f"{benchmark}/mean/{metric}/improvement"] = change["improvement"]
        run.summary[f"{benchmark}/mean/{metric}/tasks"] = change["tasks"]


def _cell(value: float | None) -> float:
    return float("nan") if value is None else value


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
