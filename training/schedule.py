"""Lengths used by the fixed evals, and the learning-rate schedule.

The training mixture lives in configs/data.yaml.
"""

import math

CONTEXT_LEN = 512
LONG_CONTEXT = 2048
HORIZON = 96
EXO_HORIZON = 168
HISTORY_KINDS = ("walk", "season", "trend")
# Fixed evals, one row per trained task. Pool counts stay in the order of the first run's draws.
EVAL_TASKS = (
    "pool/1",
    "pool/5",
    "pool/20",
    "pool/64",
    "pool/0",
    "history/walk",
    "history/season",
    "history/trend",
    "exogenous/1",
    "exogenous/3",
    "exogenous/select",
    "exogenous/decoy",
    "exogenous/daily",
    "exogenous/trend",
    "exogenous/retail",
    "views/0",
    "views/1",
    "views/8",
    "views/32",
    "views/64",
    "past/1",
    "multi/3",
    "long/train",
    "long/walk",
    "categorical/bins",
    "categorical/random",
    "masked/1",
    "masked/1-full",
    "masked/1-none",
    "masked/2",
    "masked/2-full",
    "masked/2-none",
)


def warmup_steps(total: int) -> int:
    return max(1, total // 20)


def learning_rate(step: int, total: int, peak: float) -> float:
    """Linear warmup over 5% of steps, then cosine decay to a tenth of the peak."""
    warmup = warmup_steps(total)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    floor = 0.1 * peak
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
