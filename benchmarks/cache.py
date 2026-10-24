"""One pretrained baseline per model size, reused by later runs."""

import json
import math
from pathlib import Path

PROTOCOL = 1
ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "benchmarks" / "cache"


def cache_file(size: str, benchmark: str, kind: str, task: str) -> Path:
    """`kind` is `baseline` or a candidate fingerprint."""
    safe = task.replace("/", "__")
    return CACHE_ROOT / size / benchmark / kind / f"{safe}.json"


def read_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("protocol") != PROTOCOL:
        return None
    return record


def write_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def finite_metrics(metrics: dict) -> dict[str, float]:
    kept = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isfinite(number):
            kept[key] = number
    return kept
