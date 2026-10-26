"""One synthetic batch per forecasting task."""

import numpy as np
import torch
from timestral.covariate_encoding import target_encode

from training.data_config import (
    CategoricalConfig,
    DataConfig,
    ExogenousConfig,
    HistoryConfig,
    LongConfig,
    MultiConfig,
    PastConfig,
    PoolConfig,
)
from training.schedule import CONTEXT_LEN, EXO_HORIZON, HORIZON
from training.series import lag, make_covariate, make_series, make_signal, observe, pack, seasonal_parts, walk


def make_pool_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_known: int | None = None,
    distractors: bool = True,
    past_views: bool | None = None,
    config: PoolConfig | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, str]:
    """Clean target. `n_known` views extend over the horizon.

    Half of training batches hold a pool of views and reveal only `n_known` of them over the
    horizon. The others stop at the origin, which is the split in view_curve.py. That pool is 20 views
    for a count of 16 or below, and 64 for a count of 32 or 64. The remaining batches are known
    views only, and half of those append unrelated series.
    """
    cfg = config or PoolConfig(weight=1)
    if n_known is None:
        n_known = cfg.counts.draw(rng)
    if past_views is None:
        past_views = n_known < cfg.past_view_below and rng.random() < cfg.past_view_probability
    width = cfg.wide_width if n_known >= cfg.wide_from else cfg.narrow_width
    n_past = max(0, width - n_known) if past_views else 0
    if past_views or not distractors or n_known == 0 or rng.random() >= cfg.distractor_probability:
        n_distractors = 0
    else:
        n_distractors = int(rng.integers(cfg.distractor_min, cfg.distractor_max + 1))
    length = context_len + horizon
    contexts, knowns, pasts, futures = [], [], [], []
    for _ in range(batch):
        signal = make_signal(rng, length)
        views = [make_covariate(signal, rng) for _ in range(n_known)]
        hidden = [make_covariate(signal, rng)[:context_len] for _ in range(n_past)]
        extras = [make_series(rng, length, str(rng.choice(cfg.distractor_kinds))) for _ in range(n_distractors)]
        contexts.append(signal[:context_len])
        futures.append(signal[context_len:])
        if views or extras:
            knowns.append(np.stack(views + extras))
        if hidden:
            pasts.append(np.stack(hidden))
    known = torch.tensor(np.stack(knowns), dtype=torch.float32, device=device) if knowns else None
    past = torch.tensor(np.stack(pasts), dtype=torch.float32, device=device) if pasts else None
    tag = f"pool {n_known}+{n_distractors} past{n_past}"
    return pack(contexts, False, device), known, past, pack(futures, False, device), tag


def make_history_batch(
    rng: np.random.Generator,
    batch: int,
    kind: str,
    context_len: int,
    device: torch.device,
    horizon: int | None = None,
    config: HistoryConfig | None = None,
    short_horizon: int = HORIZON,
    long_horizon: int = EXO_HORIZON,
) -> tuple[torch.Tensor, None, None, torch.Tensor, str]:
    """History-only forecast. The label is the series, noise included."""
    probability = 0.25 if config is None else config.long_horizon_probability
    if horizon is None:
        horizon = long_horizon if rng.random() < probability else short_horizon
    length = context_len + horizon
    series = [make_series(rng, length, kind) for _ in range(batch)]
    context = pack([row[:context_len] for row in series], False, device)
    future = pack([row[context_len:] for row in series], False, device)
    return context, None, None, future, f"history {kind} {context_len} h{horizon}"


def make_multi_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_variates: int | None = None,
    config: MultiConfig | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> tuple[torch.Tensor, None, None, torch.Tensor, str]:
    """Several coupled series, each forecast from the shared history."""
    cfg = config or MultiConfig(weight=1)
    if n_variates is None:
        n_variates = int(rng.integers(cfg.variate_min, cfg.variate_max + 1))
    length = context_len + horizon
    contexts, futures = [], []
    for _ in range(batch):
        shared = make_signal(rng, length)
        variates = []
        for _variate in range(n_variates):
            weight = float(rng.uniform(cfg.coupling_min, cfg.coupling_max)) * float(rng.choice((-1.0, 1.0)))
            variates.append(weight * shared + np.cumsum(rng.normal(0, cfg.innovation_scale, length)))
        series = np.stack(variates)
        contexts.append(series[:, :context_len])
        futures.append(series[:, context_len:])
    tag = f"multi {n_variates}"
    return pack(contexts, True, device), None, None, pack(futures, True, device), tag


def make_categorical_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    mode: str | None = None,
    config: CategoricalConfig | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, str]:
    """One known categorical covariate, target-encoded the way inference does it.

    `mode` is `bins`, `random`, or None. None flips a coin per series.
    """
    cfg = config or CategoricalConfig(weight=1)
    length = context_len + horizon
    contexts, knowns, futures = [], [], []
    informative = 0
    for _ in range(batch):
        signal = make_signal(rng, length)
        use_bins = (rng.random() < cfg.bins_probability) if mode is None else mode == "bins"
        if use_bins:
            edges = np.quantile(signal[:context_len], cfg.quantile_edges)
            codes = np.digitize(signal, edges).astype(np.float64)
            informative += 1
        else:
            codes = rng.integers(0, cfg.n_codes, length).astype(np.float64)
        past, future_codes = target_encode(signal[:context_len], codes[:context_len], codes[context_len:])
        knowns.append(np.concatenate([past, future_codes]))
        contexts.append(signal[:context_len])
        futures.append(signal[context_len:])
    known = pack(knowns, False, device)
    tag = f"categorical {informative}/{batch}"
    return pack(contexts, False, device), known, None, pack(futures, False, device), tag


def _decoy_row(
    kind: str,
    daily: np.ndarray,
    weekly: np.ndarray,
    weekly_phase: float,
    trend_line: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """A known row that shares one piece of the target and differs on the rest.

    `both` shares the daily and weekly cycles. `daily` shares only the daily cycle and uses the
    opposite weekly phase. `trend` shares the slope and none of the cycles. Every kind carries
    its own walk, over the context and the horizon.
    """
    residual = 0.35 * walk(rng, daily.shape[0], 0.25)
    if kind == "both":
        return daily + weekly + residual
    if kind == "daily":
        other_weekly = 0.6 * np.sin(2 * np.pi * np.arange(daily.shape[0]) / 168.0 + weekly_phase + np.pi)
        return daily + other_weekly + residual
    if kind == "trend":
        return trend_line + residual
    raise ValueError(kind)


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
    decoy_kind: str | None = None,
    trend: bool | None = None,
    config: ExogenousConfig | None = None,
    context_len: int = CONTEXT_LEN,
    short_horizon: int = HORIZON,
    long_horizon: int = EXO_HORIZON,
) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor, str]:
    """Target driven by one latent walk, a calendar, and sometimes a slope.

    Some known rows are noisy readings of the walk plus the calendar. Decoys share one piece of
    the target: both cycles, the daily cycle only, or the slope only. A gate is an on/off pattern
    known over the horizon that multiplies the target. A static code is constant per series and
    target-encoded.
    """
    cfg = config or ExogenousConfig(weight=1)
    # A quarter of training batches have no true driver. Every known row is a decoy.
    if n_known is None and n_decoy is None and rng.random() < cfg.no_driver_probability:
        n_known = 0
        n_decoy = int(rng.choice(cfg.no_driver_decoy_counts))
    if n_known is None:
        n_known = cfg.known_counts.draw(rng)
    if n_decoy is None:
        n_decoy = cfg.decoy_counts.draw(rng)
    if horizon is None:
        horizon = long_horizon if rng.random() < cfg.long_horizon_probability else short_horizon
    if nonnegative is None:
        nonnegative = bool(rng.random() < cfg.nonnegative_probability)
    if gate is None:
        gate = bool(rng.random() < cfg.gate_probability)
    if static is None:
        static = bool(rng.random() < cfg.static_probability)
    length = context_len + horizon
    time = np.arange(length)
    contexts, knowns, futures = [], [], []
    for _ in range(batch):
        open_hours = int(rng.integers(10, 20))
        weekly_phase = float(rng.uniform(0, 2 * np.pi))
        daily, weekly = seasonal_parts(length, float(rng.uniform(0, 2 * np.pi)), weekly_phase)
        shared = daily + weekly
        series_trend = trend if trend is not None else bool(rng.random() < cfg.trend_probability)
        trend_line = float(rng.normal(0, 0.0015)) * time if series_trend else np.zeros(length)
        driver = walk(rng, length, 0.25)
        own = walk(rng, length, 0.12)
        target = np.tanh(driver) + shared + trend_line + own
        if static:
            code = float(rng.integers(0, 6))
            target = target + (code - 2.5) * 0.4
        if nonnegative:
            target = np.maximum(target, 0.0)
        rows = []
        for _view in range(n_known):
            lag_k = int(rng.integers(cfg.lag_min, cfg.lag_max + 1))
            src = driver if lag_k == 0 else lag(driver, lag_k)
            rows.append(observe(src + shared, float(rng.uniform(cfg.noise_min, cfg.noise_max)), rng))
        kinds = tuple(cfg.decoy_kinds) if series_trend else tuple(kind for kind in cfg.decoy_kinds if kind != "trend")
        if not kinds:
            kinds = ("both",)
        for _view in range(n_decoy):
            kind = decoy_kind if decoy_kind is not None else str(rng.choice(kinds))
            if kind == "trend" and not series_trend:
                kind = "both"
            rows.append(_decoy_row(kind, daily, weekly, weekly_phase, trend_line, rng))
        if gate:
            open_ = ((time % 24) < open_hours).astype(np.float64)
            target = target * open_
            rows.append(open_)
        if static:
            codes = np.full(length, code, dtype=np.float64)
            encoded_past, encoded_future = target_encode(target[:CONTEXT_LEN], codes[:CONTEXT_LEN], codes[CONTEXT_LEN:])
            rows.append(np.concatenate([encoded_past, encoded_future]))
        contexts.append(target[:context_len])
        futures.append(target[context_len:])
        knowns.append(np.stack(rows))
    known = torch.tensor(np.stack(knowns), dtype=torch.float32, device=device)
    marks = "".join(part for part, enabled in ((" gate", gate), (" static", static), (" rect", nonnegative)) if enabled)
    label = decoy_kind or "mixed"
    tag = f"exogenous {n_known}+{n_decoy} {label} h{horizon}{marks}"
    return pack(contexts, False, device), known, None, pack(futures, False, device), tag


def make_past_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_past: int | None = None,
    config: PastConfig | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, str]:
    """Lagged measurements that stop at the forecast origin."""
    cfg = config or PastConfig(weight=1)
    if n_past is None:
        n_past = cfg.counts.draw(rng)
    length = context_len + horizon
    contexts, pasts, futures = [], [], []
    for _ in range(batch):
        series = make_series(rng, length, str(rng.choice(cfg.kinds)))
        rows = []
        for _channel in range(n_past):
            lagged = lag(series, int(rng.integers(cfg.lag_min, cfg.lag_max + 1)))
            rows.append(observe(lagged, float(rng.uniform(cfg.noise_min, cfg.noise_max)), rng)[:context_len])
        contexts.append(series[:context_len])
        futures.append(series[context_len:])
        pasts.append(np.stack(rows))
    past = torch.tensor(np.stack(pasts), dtype=torch.float32, device=device)
    tag = f"past {n_past}"
    return pack(contexts, False, device), None, past, pack(futures, False, device), tag


def next_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    data: DataConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, str, str]:
    """`(context, known, past, future, task, tag)`."""
    cfg = data or DataConfig.load()
    task = cfg.draw_task(rng)
    if task == "pool":
        context, known, past, future, tag = make_pool_batch(
            rng, batch, device, config=cfg.pool, context_len=cfg.context_len, horizon=cfg.horizon
        )
    elif task == "history":
        context, known, past, future, tag = make_history_batch(
            rng,
            batch,
            str(rng.choice(cfg.history.kinds)),
            cfg.context_len,
            device,
            config=cfg.history,
            short_horizon=cfg.horizon,
            long_horizon=cfg.exo_horizon,
        )
    elif task == "exogenous":
        context, known, past, future, tag = make_exogenous_batch(
            rng,
            batch,
            device,
            config=cfg.exogenous,
            context_len=cfg.context_len,
            short_horizon=cfg.horizon,
            long_horizon=cfg.exo_horizon,
        )
    elif task == "past":
        context, known, past, future, tag = make_past_batch(
            rng, batch, device, config=cfg.past, context_len=cfg.context_len, horizon=cfg.horizon
        )
    elif task == "multi":
        context, known, past, future, tag = make_multi_batch(
            rng, batch, device, config=cfg.multi, context_len=cfg.context_len, horizon=cfg.horizon
        )
    elif task == "long":
        long_cfg: LongConfig = cfg.long
        kind = str(rng.choice(long_cfg.kinds))
        context, known, past, future, tag = make_history_batch(
            rng, batch, kind, cfg.long_context, device, horizon=cfg.horizon
        )
        tag = "long" + tag.removeprefix("history")
    else:
        context, known, past, future, tag = make_categorical_batch(
            rng, batch, device, config=cfg.categorical, context_len=cfg.context_len, horizon=cfg.horizon
        )
    return context, known, past, future, task, tag
