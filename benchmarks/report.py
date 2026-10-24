"""Print a baseline-versus-current table. Lower error is the better forecast."""

import math


def mase_key(metrics: dict) -> str | None:
    for key in ("MASE", "MASE[0.5]"):
        if key in metrics:
            return key
    for key in metrics:
        if key.startswith("MASE"):
            return key
    return None


def prob_key(metrics: dict) -> str | None:
    for key in ("SQL", "WQL", "mean_weighted_sum_quantile_loss"):
        if key in metrics:
            return key
    return None


def print_report(benchmark: str, size: str, rows: list[dict], compared: bool, source: str = "checkpoint") -> None:
    title = f"{benchmark}  {size}"
    title += f"  {source} against the pretrained baseline" if compared else "  pretrained baseline"
    print(f"\n{title}")
    if not rows:
        print("  no tasks")
        return
    primary = rows[0]["primary"]
    header = f"{'task':<42} {'metric':<28} {'baseline':>12}"
    if compared:
        header += f" {'current':>12} {'delta':>12}"
    print(header)
    for row in rows:
        _print_row(row["task"], row["primary"], row["baseline"], row.get("current"), compared)
    print()
    _print_summary(rows, primary, "median " + primary, compared)
    extra = mase_key(rows[0]["baseline"])
    if extra and extra != primary:
        _print_summary(rows, extra, "median " + extra, compared)
    probabilistic = prob_key(rows[0]["baseline"])
    if probabilistic and probabilistic not in (primary, extra):
        _print_summary(rows, probabilistic, "median " + probabilistic, compared)
    if compared:
        _print_mean(rows, primary)
        if extra and extra != primary:
            _print_mean(rows, extra)
        print("delta = current - baseline. A negative delta is a lower error.")
        print("improvement is the average of (baseline - current) / baseline. Positive means less error.")


def _print_row(task: str, metric: str, baseline: dict, current: dict | None, compared: bool) -> None:
    base = baseline.get(metric)
    if base is None:
        return
    line = f"{task:<42} {metric:<28} {_fmt(base):>12}"
    if compared and current is not None and metric in current:
        value = current[metric]
        line += f" {_fmt(value):>12} {_fmt(value - base, signed=True):>12}"
    print(line)


def _print_summary(rows: list[dict], metric: str, label: str, compared: bool) -> None:
    base_values = [row["baseline"][metric] for row in rows if metric in row["baseline"]]
    if not base_values:
        return
    base = _median(base_values)
    line = f"{label:<42} {metric:<28} {_fmt(base):>12}"
    if compared:
        current_values = [row["current"][metric] for row in rows if row.get("current") and metric in row["current"]]
        if len(current_values) == len(base_values):
            current = _median(current_values)
            line += f" {_fmt(current):>12} {_fmt(current - base, signed=True):>12}"
    print(line)


def mean_change(rows: list[dict], metric: str) -> dict[str, float] | None:
    """Unweighted mean across tasks. `improvement` is the relative drop in error."""
    pairs = []
    for row in rows:
        baseline = row["baseline"].get(metric)
        current = (row.get("current") or {}).get(metric)
        if baseline is None or current is None:
            continue
        pairs.append((baseline, current))
    if not pairs:
        return None
    baseline = _mean([item[0] for item in pairs])
    current = _mean([item[1] for item in pairs])
    relative = [(base - value) / abs(base) for base, value in pairs if base != 0]
    return {
        "tasks": float(len(pairs)),
        "baseline": baseline,
        "current": current,
        "delta": current - baseline,
        "improvement": _mean(relative) if relative else float("nan"),
    }


def _print_mean(rows: list[dict], metric: str) -> None:
    change = mean_change(rows, metric)
    if change is None:
        return
    print(
        f"{'mean improvement':<42} {metric:<28} {_fmt(change['baseline']):>12} "
        f"{_fmt(change['current']):>12} {_fmt(change['delta'], signed=True):>12}  "
        f"({change['improvement']:+.1%})"
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _fmt(value: float, signed: bool = False) -> str:
    if not math.isfinite(value):
        return "nan"
    if signed:
        return f"{value:+.4g}"
    return f"{value:.4g}"
