"""One synthetic batch per forecasting task."""

import numpy as np
import torch
from timestral.covariate_encoding import target_encode

from training.schedule import CONTEXT_LEN, COUNTS, EXO_HORIZON, HISTORY_KINDS, HORIZON, LONG_CONTEXT, TASK_P, TASKS
from training.series import calendar, lag, make_covariate, make_series, make_signal, observe, pack, walk


def make_pool_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_known: int | None = None,
    distractors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, None, torch.Tensor, str]:
    """Clean target. `n_known` views, plus unrelated series on some training batches."""
    if n_known is None:
        n_known = int(rng.choice(COUNTS))
    n_distractors = int(rng.integers(1, 9)) if distractors and n_known > 0 and rng.random() < 0.5 else 0
    length = CONTEXT_LEN + HORIZON
    contexts, knowns, futures = [], [], []
    for _ in range(batch):
        signal = make_signal(rng, length)
        views = [make_covariate(signal, rng) for _ in range(n_known)]
        extras = [make_series(rng, length, str(rng.choice(HISTORY_KINDS))) for _ in range(n_distractors)]
        contexts.append(signal[:CONTEXT_LEN])
        futures.append(signal[CONTEXT_LEN:])
        if views or extras:
            knowns.append(np.stack(views + extras))
    known = torch.tensor(np.stack(knowns), dtype=torch.float32, device=device) if knowns else None
    tag = f"pool {n_known}+{n_distractors}"
    return pack(contexts, False, device), known, None, pack(futures, False, device), tag


def make_history_batch(
    rng: np.random.Generator,
    batch: int,
    kind: str,
    context_len: int,
    device: torch.device,
    horizon: int | None = None,
) -> tuple[torch.Tensor, None, None, torch.Tensor, str]:
    """History-only forecast. The label is the series, noise included."""
    if horizon is None:
        horizon = EXO_HORIZON if rng.random() < 0.25 else HORIZON
    length = context_len + horizon
    series = [make_series(rng, length, kind) for _ in range(batch)]
    context = pack([row[:context_len] for row in series], False, device)
    future = pack([row[context_len:] for row in series], False, device)
    return context, None, None, future, f"history {kind} {context_len} h{horizon}"


def make_multi_batch(
    rng: np.random.Generator, batch: int, device: torch.device, n_variates: int | None = None
) -> tuple[torch.Tensor, None, None, torch.Tensor, str]:
    """Several coupled series, each forecast from the shared history."""
    if n_variates is None:
        n_variates = int(rng.integers(2, 5))
    length = CONTEXT_LEN + HORIZON
    contexts, futures = [], []
    for _ in range(batch):
        shared = make_signal(rng, length)
        variates = []
        for _variate in range(n_variates):
            weight = float(rng.uniform(0.4, 1.2)) * float(rng.choice((-1.0, 1.0)))
            variates.append(weight * shared + np.cumsum(rng.normal(0, 0.25, length)))
        series = np.stack(variates)
        contexts.append(series[:, :CONTEXT_LEN])
        futures.append(series[:, CONTEXT_LEN:])
    tag = f"multi {n_variates}"
    return pack(contexts, True, device), None, None, pack(futures, True, device), tag


def make_categorical_batch(
    rng: np.random.Generator, batch: int, device: torch.device, mode: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, str]:
    """One known categorical covariate, target-encoded the way inference does it.

    `mode` is `bins`, `random`, or None. None flips a coin per series.
    """
    length = CONTEXT_LEN + HORIZON
    contexts, knowns, futures = [], [], []
    informative = 0
    for _ in range(batch):
        signal = make_signal(rng, length)
        use_bins = (rng.random() < 0.5) if mode is None else mode == "bins"
        if use_bins:
            edges = np.quantile(signal[:CONTEXT_LEN], (0.25, 0.5, 0.75))
            codes = np.digitize(signal, edges).astype(np.float64)
            informative += 1
        else:
            codes = rng.integers(0, 4, length).astype(np.float64)
        past, future_codes = target_encode(signal[:CONTEXT_LEN], codes[:CONTEXT_LEN], codes[CONTEXT_LEN:])
        knowns.append(np.concatenate([past, future_codes]))
        contexts.append(signal[:CONTEXT_LEN])
        futures.append(signal[CONTEXT_LEN:])
    known = pack(knowns, False, device)
    tag = f"categorical {informative}/{batch}"
    return pack(contexts, False, device), known, None, pack(futures, False, device), tag


def make_exogenous_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_known: int | None = None,
    n_decoy: int | None = None,
    static: bool | None = None,
    nonnegative: bool | None = None,
    gate: bool | None = None,
    horizon: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, str]:
    """Target driven by one latent walk. Some known rows read it; decoys only share the calendar.

    A decoy has the same daily and weekly periods as the target and an independent walk, over the
    context as well as the horizon. A gate is an on/off pattern known over the horizon that
    multiplies the target. A static code is constant per series and target-encoded.
    """
    if n_known is None:
        n_known = int(rng.choice((1, 1, 2, 3)))
    if n_decoy is None:
        n_decoy = int(rng.choice((1, 1, 2)))
    if horizon is None:
        horizon = EXO_HORIZON if rng.random() < 0.5 else HORIZON
    if nonnegative is None:
        nonnegative = bool(rng.random() < 0.5)
    if gate is None:
        gate = bool(rng.random() < 0.5)
    if static is None:
        static = bool(rng.random() < 0.5)
    length = CONTEXT_LEN + horizon
    time = np.arange(length)
    contexts, knowns, futures = [], [], []
    for _ in range(batch):
        open_hours = int(rng.integers(10, 20))
        shared = calendar(length, float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi)))
        driver = walk(rng, length, 0.25)
        own = walk(rng, length, 0.12)
        target = np.tanh(driver) + shared + own
        if static:
            code = float(rng.integers(0, 6))
            target = target + (code - 2.5) * 0.4
        if nonnegative:
            target = np.maximum(target, 0.0)
        rows = []
        for _view in range(n_known):
            lag_k = int(rng.integers(0, 6))
            src = driver if lag_k == 0 else lag(driver, lag_k)
            rows.append(observe(src + shared, float(rng.uniform(0.05, 0.45)), rng))
        for _view in range(n_decoy):
            rows.append(shared + 0.35 * walk(rng, length, 0.25))
        if gate:
            open_ = ((time % 24) < open_hours).astype(np.float64)
            target = target * open_
            rows.append(open_)
        if static:
            codes = np.full(length, code, dtype=np.float64)
            encoded_past, encoded_future = target_encode(target[:CONTEXT_LEN], codes[:CONTEXT_LEN], codes[CONTEXT_LEN:])
            rows.append(np.concatenate([encoded_past, encoded_future]))
        contexts.append(target[:CONTEXT_LEN])
        futures.append(target[CONTEXT_LEN:])
        knowns.append(np.stack(rows))
    known = torch.tensor(np.stack(knowns), dtype=torch.float32, device=device)
    marks = "".join(part for part, enabled in ((" gate", gate), (" static", static), (" rect", nonnegative)) if enabled)
    tag = f"exogenous {n_known}+{n_decoy} h{horizon}{marks}"
    return pack(contexts, False, device), known, None, pack(futures, False, device), tag


def make_past_batch(
    rng: np.random.Generator, batch: int, device: torch.device, n_past: int | None = None
) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, str]:
    """Lagged measurements that stop at the forecast origin."""
    if n_past is None:
        n_past = int(rng.choice((1, 2, 3)))
    length = CONTEXT_LEN + HORIZON
    contexts, pasts, futures = [], [], []
    for _ in range(batch):
        series = make_series(rng, length, str(rng.choice(HISTORY_KINDS)))
        rows = []
        for _channel in range(n_past):
            lagged = lag(series, int(rng.integers(1, 13)))
            rows.append(observe(lagged, float(rng.uniform(0.05, 0.3)), rng)[:CONTEXT_LEN])
        contexts.append(series[:CONTEXT_LEN])
        futures.append(series[CONTEXT_LEN:])
        pasts.append(np.stack(rows))
    past = torch.tensor(np.stack(pasts), dtype=torch.float32, device=device)
    tag = f"past {n_past}"
    return pack(contexts, False, device), None, past, pack(futures, False, device), tag


def next_batch(
    rng: np.random.Generator, batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, str, str]:
    """`(context, known, past, future, task, tag)`."""
    task = str(rng.choice(TASKS, p=TASK_P))
    if task == "pool":
        context, known, past, future, tag = make_pool_batch(rng, batch, device)
    elif task == "history":
        context, known, past, future, tag = make_history_batch(
            rng, batch, str(rng.choice(HISTORY_KINDS)), CONTEXT_LEN, device
        )
    elif task == "exogenous":
        context, known, past, future, tag = make_exogenous_batch(rng, batch, device)
    elif task == "past":
        context, known, past, future, tag = make_past_batch(rng, batch, device)
    elif task == "multi":
        context, known, past, future, tag = make_multi_batch(rng, batch, device)
    elif task == "long":
        kind = str(rng.choice(("train",) + HISTORY_KINDS))
        context, known, past, future, tag = make_history_batch(rng, batch, kind, LONG_CONTEXT, device, horizon=HORIZON)
        tag = "long" + tag.removeprefix("history")
    else:
        context, known, past, future, tag = make_categorical_batch(rng, batch, device)
    return context, known, past, future, task, tag
