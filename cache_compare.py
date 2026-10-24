"""Compare cached checkpoint scores with the pretrained baseline.

Reads the fine-tune files under benchmarks/cache and the baseline file for the
same size, benchmark, and task. A task appears once its checkpoint file exists.

    uv run python cache_compare.py
    uv run python cache_compare.py benchmarks/cache/5M/fev/616099f4712a784b

Improvement is (baseline - current) / baseline. Positive means a lower error.
"""

import argparse
from collections import defaultdict
from pathlib import Path

from benchmarks.cache import CACHE_ROOT, cache_file, read_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", type=Path, help="Candidate cache directory. Default: every cached checkpoint.")
    args = parser.parse_args()
    directories = [args.directory] if args.directory else _candidate_dirs()
    if not directories:
        raise SystemExit(f"No checkpoint cache under {CACHE_ROOT}.")

    compared: list[tuple[dict, dict]] = []
    missing: list[str] = []
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            current = read_result(path)
            if current is None:
                continue
            baseline = read_result(cache_file(current["size"], current["benchmark"], "baseline", current["task"]))
            if baseline is None:
                missing.append(f"{current['benchmark']}/{current['task']}")
                continue
            compared.append((baseline, current))
    if not compared:
        raise SystemExit("No checkpoint file has a baseline next to it yet.")

    print(f"{'task':<28} {'metric':<12} {'baseline':>10} {'current':>10} {'delta':>10} {'improvement':>12}")
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for baseline, current in sorted(compared, key=lambda item: (item[1]["benchmark"], item[1]["task"])):
        for metric in _metrics(baseline, current):
            before = baseline["metrics"][metric]
            after = current["metrics"][metric]
            groups[(current["benchmark"], metric)].append((before, after))
            if metric not in {current["primary"], _mase(current["metrics"])}:
                continue
            gain = (before - after) / abs(before) if before else float("nan")
            print(
                f"{current['task']:<28} {metric:<12} {before:10.4g} {after:10.4g} "
                f"{after - before:+10.4g} {gain:+12.1%}"
            )

    print()
    print(f"{'benchmark':<12} {'metric':<12} {'tasks':>6} {'baseline':>10} {'current':>10} {'delta':>10} {'improvement':>12}")
    for benchmark, metric in sorted(groups):
        pairs = groups[(benchmark, metric)]
        before = sum(item[0] for item in pairs) / len(pairs)
        after = sum(item[1] for item in pairs) / len(pairs)
        relative = [(base - value) / abs(base) for base, value in pairs if base]
        gain = sum(relative) / len(relative) if relative else float("nan")
        print(
            f"{benchmark:<12} {metric:<12} {len(pairs):>6} {before:10.4g} {after:10.4g} "
            f"{after - before:+10.4g} {gain:+12.1%}"
        )
    for name in missing:
        print(f"missing baseline: {name}")


def _candidate_dirs() -> list[Path]:
    parents = {path.parent for path in CACHE_ROOT.glob("*/*/*/*.json") if path.parent.name != "baseline"}
    return sorted(parents)


def _metrics(baseline: dict, current: dict) -> list[str]:
    shared = [key for key in baseline["metrics"] if key in current["metrics"] and key != "test_error"]
    return shared


def _mase(metrics: dict) -> str | None:
    for key in ("MASE", "MASE[0.5]"):
        if key in metrics:
            return key
    return None


if __name__ == "__main__":
    main()
