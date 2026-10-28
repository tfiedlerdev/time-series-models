"""Simple forecasts on the graph eval's own draws, as a reference for the model's errors.

The draws are the ones `eval.graph` scores, from the same seed. A ridge fit on the
context, from each covariate's value and square, forecasts the horizon. It uses either
every node known whole, or the known rows as kept. Last value is the reference with no
covariate. Each column is the median horizon MAE over the draws.
"""

import argparse

import numpy as np

from training.data_config import GraphConfig
from training.graph import draw_graph, graph_rows
from training.schedule import CONTEXT_LEN, HORIZON


def ridge_mae(target: np.ndarray, rows: np.ndarray | None) -> float:
    future = target[CONTEXT_LEN:]
    if rows is None:
        return float(np.mean(np.abs(future - target[CONTEXT_LEN - 1])))
    features = np.concatenate([rows, rows**2, np.ones((1, rows.shape[1]))]).T
    seen = np.isfinite(features[:CONTEXT_LEN]).all(axis=1)
    if seen.sum() < 64 or not np.isfinite(features[CONTEXT_LEN:]).all():
        return float(np.mean(np.abs(future - target[CONTEXT_LEN - 1])))
    fit_x, fit_y = features[:CONTEXT_LEN][seen], target[:CONTEXT_LEN][seen]
    weights = np.linalg.solve(fit_x.T @ fit_x + 1e-2 * np.eye(fit_x.shape[1]), fit_x.T @ fit_y)
    return float(np.mean(np.abs(features[CONTEXT_LEN:] @ weights - future)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123 + 60_000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = GraphConfig(weight=1)
    cells: dict[str, list[float]] = {"ridge, every node": [], "ridge, kept known rows": [], "last value": []}
    for _ in range(args.series):
        draw = draw_graph(rng, cfg, CONTEXT_LEN, HORIZON)
        known, _past = graph_rows(draw, CONTEXT_LEN)
        cells["ridge, every node"].append(ridge_mae(draw.target, draw.nodes))
        cells["ridge, kept known rows"].append(ridge_mae(draw.target, known))
        cells["last value"].append(ridge_mae(draw.target, None))
    print(f"horizon MAE, median over {args.series} draws")
    for name, values in cells.items():
        print(f"{name:<24}{np.median(values):.3f}")


if __name__ == "__main__":
    main()
