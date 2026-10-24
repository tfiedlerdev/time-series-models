"""Lengths, task mixture, and the learning-rate schedule."""

import math

CONTEXT_LEN = 512
LONG_CONTEXT = 2048
HORIZON = 96
EXO_HORIZON = 168
COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
# At 6000 steps, pooling gets about 1800 updates and the exogenous task about 1500,
# so the two covariate lessons stay near each other.
TASKS = ("pool", "history", "exogenous", "past", "multi", "long", "categorical")
TASK_P = (0.30, 0.20, 0.25, 0.10, 0.06, 0.05, 0.04)
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
    "exogenous/retail",
    "past/1",
    "multi/3",
    "long/train",
    "long/walk",
    "categorical/bins",
    "categorical/random",
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
