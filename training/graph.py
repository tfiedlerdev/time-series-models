"""A target composed of covariates that depend on each other.

Each covariate is a nonlinear function of its parents, each read with its own delay, plus
noise of its own scale. Roots are walks, the pooling signal, or a calendar plus a walk.
The target is a nonlinear composition of a few covariates. After the draw, each covariate
is removed, kept only through the context, masked on stretches of the context and known
elsewhere, or kept whole.

A draw is kept when a ridge forecast from the kept known covariates beats a last-value
forecast on the horizon by `min_gain`, or when a context-only covariate is a direct cause
of the target with a delay of at least `lead_min` steps. Otherwise the draw is either noise
or carries nothing the history does not, and it is drawn again.
"""

from dataclasses import dataclass

import numpy as np
import torch

from training.data_config import GraphConfig
from training.schedule import CONTEXT_LEN, HORIZON
from training.series import make_signal, pack, seasonal_parts, signed_power, walk

MODES = ("removed", "context", "masked", "full")


@dataclass
class GraphDraw:
    target: np.ndarray
    nodes: np.ndarray
    shown: np.ndarray
    modes: list[str]
    target_parents: dict[int, int]


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    if steps == 0:
        return values
    out = np.empty_like(values)
    out[steps:] = values[:-steps]
    out[:steps] = values[0]
    return out


def _response(rng: np.random.Generator, values: np.ndarray) -> np.ndarray:
    spread = float(np.std(values)) or 1.0
    scaled = values / spread
    kind = int(rng.integers(0, 4))
    if kind == 0:
        out = np.tanh(scaled / float(rng.uniform(0.5, 2.0)))
    elif kind == 1:
        out = signed_power(scaled, float(rng.uniform(0.5, 1.8)))
    elif kind == 2:
        out = scaled / (1 + float(rng.uniform(0.1, 0.5)) * scaled**2)
    else:
        out = scaled
    return float(rng.uniform(0.5, 1.5)) * float(rng.choice((-1.0, 1.0))) * out


def _root(rng: np.random.Generator, length: int) -> np.ndarray:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return walk(rng, length, 0.25)
    if kind == 1:
        return make_signal(rng, length) / 2
    daily, weekly = seasonal_parts(length, float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, 2 * np.pi)))
    return daily + weekly + walk(rng, length, 0.15)


def _delay(rng: np.random.Generator, cfg: GraphConfig) -> int:
    return 0 if rng.random() < 0.5 else int(rng.integers(1, cfg.delay_max + 1))


def _draw_values(rng: np.random.Generator, cfg: GraphConfig, length: int) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    n_nodes = cfg.node_counts.draw(rng)
    pad = cfg.delay_max * (n_nodes + 1)
    total = pad + length
    nodes: list[np.ndarray] = []
    for index in range(n_nodes):
        parents = [j for j in range(index) if rng.random() < 0.5]
        parents = list(rng.permutation(parents)[: cfg.max_parents])
        if parents:
            base = sum(_response(rng, _shift(nodes[j], _delay(rng, cfg))) for j in parents)
        else:
            base = _root(rng, total)
        noise = float(rng.uniform(cfg.noise_min, cfg.noise_max)) * (float(np.std(base)) or 1.0)
        nodes.append(base + rng.normal(0, noise, total))
    chosen = rng.choice(n_nodes, min(n_nodes, cfg.target_parent_counts.draw(rng)), replace=False)
    delays = {int(j): _delay(rng, cfg) for j in chosen}
    target = sum(_response(rng, _shift(nodes[j], delays[j])) for j in delays) + walk(rng, total, cfg.own_scale)
    return target[pad:], np.stack(nodes)[:, pad:], delays


def _ridge_gain(target: np.ndarray, rows: np.ndarray, context_len: int) -> float:
    """Horizon MAE of a ridge fit on the kept known rows, over that of the last value."""
    future = target[context_len:]
    naive = float(np.mean(np.abs(future - target[context_len - 1])))
    features = np.concatenate([rows, rows**2, np.ones((1, rows.shape[1]))]).T
    seen = np.isfinite(features[:context_len]).all(axis=1)
    if seen.sum() < 64 or not np.isfinite(features[context_len:]).all():
        return np.inf
    fit_x, fit_y = features[:context_len][seen], target[:context_len][seen]
    weights = np.linalg.solve(fit_x.T @ fit_x + 1e-2 * np.eye(fit_x.shape[1]), fit_x.T @ fit_y)
    ridge = float(np.mean(np.abs(features[context_len:] @ weights - future)))
    return ridge / max(naive, 1e-6)


def _mask_context(rng: np.random.Generator, row: np.ndarray, context_len: int) -> np.ndarray:
    out = row.copy()
    for _ in range(int(rng.integers(1, 4))):
        width = int(rng.integers(32, 129))
        start = int(rng.integers(0, max(1, context_len - width)))
        out[start : start + width] = np.nan
    return out


def _useful(draw: GraphDraw, cfg: GraphConfig, context_len: int) -> bool:
    known = [draw.shown[i] for i, mode in enumerate(draw.modes) if mode in ("full", "masked")]
    if known and _ridge_gain(draw.target, np.stack(known), context_len) < 1 - cfg.min_gain:
        return True
    leads = [i for i, mode in enumerate(draw.modes) if mode == "context" and draw.target_parents.get(i, 0) >= cfg.lead_min]
    return bool(leads)


def draw_graph(rng: np.random.Generator, cfg: GraphConfig, context_len: int, horizon: int) -> GraphDraw:
    """One kept draw. After `max_tries` rejections the last draw keeps every covariate whole."""
    probabilities = np.array(
        [cfg.removed_probability, cfg.context_probability, cfg.masked_probability, 0.0], dtype=np.float64
    )
    probabilities[3] = max(0.0, 1.0 - probabilities[:3].sum())
    probabilities /= probabilities.sum()
    draw = None
    for _ in range(cfg.max_tries):
        target, nodes, parents = _draw_values(rng, cfg, context_len + horizon)
        modes = [str(rng.choice(MODES, p=probabilities)) for _ in range(nodes.shape[0])]
        if all(mode == "removed" for mode in modes):
            modes[int(rng.integers(len(modes)))] = "full"
        shown = nodes.copy()
        for index, mode in enumerate(modes):
            if mode == "masked":
                shown[index] = _mask_context(rng, nodes[index], context_len)
        draw = GraphDraw(target, nodes, shown, modes, parents)
        if _useful(draw, cfg, context_len):
            return draw
    draw.modes = ["full"] * len(draw.modes)
    draw.shown = draw.nodes.copy()
    return draw


def graph_rows(draw: GraphDraw, context_len: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`(known, past)` rows for the forecaster, each `(C, T)` or None."""
    known = [draw.shown[i] for i, mode in enumerate(draw.modes) if mode in ("full", "masked")]
    past = [draw.shown[i][:context_len] for i, mode in enumerate(draw.modes) if mode == "context"]
    return (np.stack(known) if known else None, np.stack(past) if past else None)


def make_graph_batch(
    rng: np.random.Generator,
    device: torch.device,
    config: GraphConfig | None = None,
    context_len: int = CONTEXT_LEN,
    horizon: int = HORIZON,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, str]:
    """One series. Covariate counts differ between draws, so a batch holds one."""
    cfg = config or GraphConfig(weight=1)
    draw = draw_graph(rng, cfg, context_len, horizon)
    known, past = graph_rows(draw, context_len)
    counts = {mode: draw.modes.count(mode) for mode in MODES}
    tag = "graph " + " ".join(f"{mode}{count}" for mode, count in counts.items())
    return (
        pack([draw.target[:context_len]], False, device),
        None if known is None else torch.tensor(known[None], dtype=torch.float32, device=device),
        None if past is None else torch.tensor(past[None], dtype=torch.float32, device=device),
        pack([draw.target[context_len:]], False, device),
        tag,
    )
