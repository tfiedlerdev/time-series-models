"""Average improvement per benchmark from a finished W&B benchmark run.

The live benchmark now writes these means itself. This reads a run that
already finished, including one from before that, using the per-task summary
entries.

    uv run python benchmark_summary.py tfiedlerdev/time-series-models/ryg6jqbp
    uv run python benchmark_summary.py https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/ryg6jqbp

Improvement is the unweighted mean of (baseline - current) / baseline.
Positive means the fine-tune lowered the error.
"""

import argparse
from collections import defaultdict
from urllib.parse import urlparse

import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="entity/project/run_id, or the run URL.")
    args = parser.parse_args()
    run = wandb.Api().run(_run_path(args.run))
    groups = _pairs(run.summary)
    if not groups:
        raise SystemExit(f"{run.name} ({run.id}) has no per-task benchmark scores yet. State: {run.state}.")
    print(f"{run.name}  {run.id}  {run.state}")
    print(f"{'benchmark':<12} {'metric':<28} {'tasks':>6} {'baseline':>12} {'current':>12} {'delta':>12} {'improvement':>12}")
    for benchmark, metric in sorted(groups):
        pairs = groups[(benchmark, metric)]
        baseline = sum(item[0] for item in pairs) / len(pairs)
        current = sum(item[1] for item in pairs) / len(pairs)
        relative = [(base - value) / abs(base) for base, value in pairs if base != 0]
        improvement = sum(relative) / len(relative) if relative else float("nan")
        print(
            f"{benchmark:<12} {metric:<28} {len(pairs):>6} {baseline:12.4g} {current:12.4g} "
            f"{current - baseline:+12.4g} {improvement:+12.1%}"
        )


def _pairs(summary) -> dict[tuple[str, str], list[tuple[float, float]]]:
    collected: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for key, value in summary.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        parts = key.split("/")
        if len(parts) < 4 or parts[-1] not in {"baseline", "current"}:
            continue
        benchmark, metric = parts[0], parts[-2]
        task = "/".join(parts[1:-2])
        if task in {"", "median", "mean"} or metric == "test_error":
            continue
        collected[(benchmark, task, metric)][parts[-1]] = float(value)
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (benchmark, _task, metric), scores in collected.items():
        if "baseline" in scores and "current" in scores:
            groups[(benchmark, metric)].append((scores["baseline"], scores["current"]))
    return groups


def _run_path(text: str) -> str:
    path = urlparse(text).path.strip("/") if "://" in text else text.strip().strip("/")
    parts = [part for part in path.split("/") if part and part != "wandb"]
    if "runs" in parts:
        index = parts.index("runs")
        entity, project, run_id = parts[index - 2], parts[index - 1], parts[index + 1]
        return f"{entity}/{project}/{run_id}"
    if len(parts) == 3:
        return "/".join(parts)
    raise SystemExit(f"Could not read a run id from {text!r}. Pass entity/project/run_id or the run URL.")


if __name__ == "__main__":
    main()
