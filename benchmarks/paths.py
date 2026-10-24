"""Where GIFT-Eval and TIME live in this repo."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "GIFT_EVAL": (ROOT / "data" / "gift-eval", "Salesforce/GiftEval"),
    "TIME_DATASET": (ROOT / "data" / "time", "Real-TSF/TIME"),
}


def data_directory(override: Path | None, env_name: str) -> Path:
    if override is not None:
        return override
    value = os.environ.get(env_name) or _env_file(env_name)
    if value:
        return Path(value)
    return DATASETS[env_name][0]


def download_command(env_name: str) -> str:
    directory, repo = DATASETS[env_name]
    return f"hf download {repo} --repo-type dataset --local-dir {directory}"


def _env_file(name: str) -> str | None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None
