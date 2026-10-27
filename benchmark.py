"""Compare a Timestral checkpoint with the pretrained model of the same size.

The candidate weights come from the W&B artifact the training run logged.
`run.use_artifact` records this benchmark as a downstream consumer of that
artifact. The pretrained scores are cached under
benchmarks/cache/<size>/<benchmark>/baseline and reused.

GIFT-Eval and TIME read data/<name> in this repo. fev-bench downloads its tasks.

Examples:
  uv run python benchmark.py --size 5M --benchmarks fev --tasks world_tourism
  uv run python benchmark.py --size 5M --benchmarks fev --only-cached-baseline
  hf download Salesforce/GiftEval --repo-type dataset --local-dir data/gift-eval
  hf download Real-TSF/TIME --repo-type dataset --local-dir data/time
"""

import argparse
import time
from pathlib import Path

from benchmarks.cache import PROTOCOL, cache_file, read_result, write_result
from benchmarks.fev_runner import evaluate as evaluate_fev
from benchmarks.fev_runner import load_tasks as load_fev_tasks
from benchmarks.model import SIZES, apply_checkpoint, checkpoint_fingerprint, load_forecaster
from benchmarks.paths import download_command
from benchmarks.report import print_report
from benchmarks.tracking import DEFAULT_ARTIFACT, log_scores, open_run, use_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", default="5M", choices=list(SIZES))
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Training artifact to consume with run.use_artifact.")
    parser.add_argument("--checkpoint", type=Path, help="Local weights instead of the W&B artifact. Skips lineage.")
    parser.add_argument("--benchmarks", default="fev,gift,time", help="Comma-separated: fev, gift, time.")
    parser.add_argument("--tasks", help="Comma-separated task names. A task matches when its name contains one of them.")
    parser.add_argument(
        "--only-cached-baseline",
        action="store_true",
        help="Keep tasks that already have a pretrained baseline, then score the checkpoint only where its own cache is missing.",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N tasks of each benchmark, after filtering.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu or cuda. MPS cannot run the float64 scaler.")
    parser.add_argument("--gift-data", type=Path, help="GiftEval directory. Defaults to data/gift-eval in this repo.")
    parser.add_argument("--time-data", type=Path, help="TIME directory. Defaults to data/time in this repo.")
    parser.add_argument("--wandb-project", default="time-series-models")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or user. Defaults to the logged-in entity.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip the W&B run. With no --checkpoint, score the pretrained model only.")
    parser.add_argument("--run-name", help="W&B run name. Defaults to benchmark-<size>.")
    parser.add_argument("--tags", default="", help="Extra W&B tags, comma-separated.")
    args = parser.parse_args()
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    requested = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    unknown = [item for item in requested if item not in {"fev", "gift", "time"}]
    if unknown:
        raise SystemExit(f"Unknown benchmark {', '.join(unknown)}. Choose from fev, gift, time.")
    filters = [item.strip() for item in args.tasks.split(",")] if args.tasks else None

    jobs = []
    for name in requested:
        tasks, storage, reason = _prepare(name, args, filters)
        if reason:
            print(reason)
            continue
        jobs.append((name, tasks, storage))
    if not jobs:
        raise SystemExit(1)
    if args.only_cached_baseline:
        jobs = _with_cached_baselines(args, jobs)
        if not jobs:
            raise SystemExit("No selected tasks have a cached baseline.")

    run = open_run(args)
    try:
        checkpoint = args.checkpoint
        artifact_version = None
        if checkpoint is None and run is not None:
            checkpoint, artifact_version = use_model(run, args.artifact, args.size)
        _score(args, jobs, run, checkpoint, artifact_version)
    finally:
        if run is not None:
            run.finish()


def _score(args, jobs, run, checkpoint: Path | None, artifact_version: str | None) -> None:
    fingerprint = checkpoint_fingerprint(checkpoint) if checkpoint else None
    forecaster = None
    for name, tasks, storage in jobs:
        missing_baseline = [task for task in tasks if read_result(_path(args.size, name, "baseline", task.name)) is None]
        missing_current = []
        if checkpoint is not None:
            missing_current = [
                task for task in tasks if read_result(_path(args.size, name, fingerprint, task.name)) is None
            ]
        print(
            f"{name}: {len(tasks)} tasks, {len(tasks) - len(missing_baseline)} baseline cached, "
            f"{len(missing_baseline)} baseline to run",
            flush=True,
        )
        if checkpoint is not None:
            print(
                f"{name}: {len(tasks) - len(missing_current)} checkpoint cached, {len(missing_current)} to run",
                flush=True,
            )

        if missing_baseline or missing_current:
            if forecaster is None:
                print(f"loading {SIZES[args.size]} on {args.device}", flush=True)
                forecaster = load_forecaster(args.size, args.device)
            if missing_baseline:
                _run_missing(
                    forecaster, name, missing_baseline, storage, args, checkpoint=None, fingerprint=None, artifact=None
                )
            if missing_current:
                apply_checkpoint(forecaster, checkpoint)
                _run_missing(
                    forecaster,
                    name,
                    missing_current,
                    storage,
                    args,
                    checkpoint=checkpoint,
                    fingerprint=fingerprint,
                    artifact=artifact_version,
                )
                forecaster = None

        baseline = [_load(args.size, name, "baseline", task.name) for task in tasks]
        current = [_load(args.size, name, fingerprint, task.name) for task in tasks] if fingerprint else None
        rows = []
        for task, base in zip(tasks, baseline):
            row = {"task": task.name, "primary": base["primary"], "baseline": base["metrics"]}
            if current is not None:
                match = next(item for item in current if item["task"] == task.name)
                row["current"] = match["metrics"]
            rows.append(row)
        print_report(name, args.size, rows, compared=current is not None, source=artifact_version or "checkpoint")
        log_scores(run, name, rows, compared=current is not None)


def _prepare(name: str, args, filters):
    if name == "fev":
        tasks = load_fev_tasks()
        storage = None
    elif name == "gift":
        from benchmarks.paths import data_directory

        storage = data_directory(args.gift_data, "GIFT_EVAL")
        if not storage.exists():
            return None, None, f"gift: {storage} is not there yet.\n  {download_command('GIFT_EVAL')}"
        from benchmarks.gift import load_tasks as load_gift_tasks

        tasks = load_gift_tasks()
    else:
        from benchmarks.paths import data_directory

        storage = data_directory(args.time_data, "TIME_DATASET")
        if not storage.exists():
            return None, None, f"time: {storage} is not there yet.\n  {download_command('TIME_DATASET')}"
        from benchmarks.time_bench import load_tasks as load_time_tasks

        tasks = load_time_tasks()
    selected = _select(tasks, filters, args.limit)
    if not selected:
        return None, None, f"{name}: no tasks matched."
    return selected, storage, None


def _with_cached_baselines(args, jobs):
    kept = []
    for name, tasks, storage in jobs:
        ready = [task for task in tasks if read_result(_path(args.size, name, "baseline", task.name)) is not None]
        print(f"{name}: {len(ready)} of {len(tasks)} tasks have a cached baseline", flush=True)
        if ready:
            kept.append((name, ready, storage))
    return kept


def _select(tasks, filters, limit):
    if filters:
        tasks = [task for task in tasks if any(item == task.name or item in task.name for item in filters)]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def _run_missing(
    forecaster, name: str, tasks, storage, args, checkpoint: Path | None, fingerprint: str | None, artifact: str | None
) -> None:
    kind = "baseline" if fingerprint is None else fingerprint
    model_name = f"timestral-{args.size}" if checkpoint is None else f"timestral-{args.size}-{checkpoint.stem}"
    for task in tasks:
        print(f"{name} {task.name}", flush=True)
        started = time.perf_counter()
        metrics, primary = _evaluate(forecaster, name, task, args.batch_size, model_name, storage)
        record = {
            "protocol": PROTOCOL,
            "size": args.size,
            "benchmark": name,
            "task": task.name,
            "model_id": SIZES[args.size],
            "checkpoint": artifact or (None if checkpoint is None else str(checkpoint)),
            "fingerprint": fingerprint,
            "primary": primary,
            "metrics": metrics,
            "seconds": round(time.perf_counter() - started, 3),
        }
        write_result(_path(args.size, name, kind, task.name), record)
        print(f"  {primary} {metrics.get(primary, float('nan')):.6g}  ({record['seconds']:.1f}s)", flush=True)


def _evaluate(forecaster, name: str, task, batch_size: int, model_name: str, storage):
    if name == "fev":
        return evaluate_fev(forecaster, task, batch_size, model_name)
    if name == "gift":
        from benchmarks.gift import evaluate as evaluate_gift

        return evaluate_gift(forecaster, task, batch_size, storage)
    from benchmarks.time_bench import evaluate as evaluate_time

    return evaluate_time(forecaster, task, batch_size, storage)


def _path(size: str, benchmark: str, kind: str, task: str) -> Path:
    return cache_file(size, benchmark, kind, task)


def _load(size: str, benchmark: str, kind: str, task: str) -> dict:
    record = read_result(_path(size, benchmark, kind, task))
    if record is None:
        raise SystemExit(f"Missing cached result for {benchmark} {task}.")
    return record


if __name__ == "__main__":
    main()
