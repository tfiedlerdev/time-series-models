# %%
"""Fine-tune Timestral-5M to pool noisy views without dropping its other forecasts.

Half the steps are the pooling task: a clean target, nonlinear noisy views known over
the horizon, and a covariate count drawn from 0 through 64. Zero covariates is an
ordinary forecast of that same series. Batches that do have views also get unrelated
series mixed in, so the loss only falls when those are ignored.

The remaining steps replay other forecasts. History-only batches use a random walk,
a different seasonal pair, or a trend, and the label is the series itself. A few
batches forecast several coupled series jointly, a few read a context of 2048 steps,
and a few use a target-encoded categorical covariate.

The model's scaler uses float64, which Apple MPS cannot run. On a Mac this script
uses the CPU. A CUDA machine, including a DGX Spark, is used when one is present.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from timestral import TimestralForecaster
from timestral.covariate_encoding import target_encode

CONTEXT_LEN = 512
LONG_CONTEXT = 2048
HORIZON = 96
COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
# One task per step. The pooling share keeps about as many pooling updates in
# 4000 steps as the pooling-only run had in 2000.
TASKS = ("pool", "history", "multi", "long", "categorical")
TASK_P = (0.50, 0.30, 0.10, 0.05, 0.05)
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
    "multi/3",
    "long/train",
    "long/walk",
    "categorical/bins",
    "categorical/random",
)


def lag(values: np.ndarray, steps: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    out[steps:] = values[:-steps]
    return out


def observe(values: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise whose scale is `noise_frac` of this series' own spread."""
    out = values.astype(float).copy()
    seen = np.isfinite(out)
    out[seen] += rng.normal(0, noise_frac * float(np.std(out[seen])), int(seen.sum()))
    return out


def signed_power(values: np.ndarray, power: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** power


def make_signal(rng: np.random.Generator, length: int) -> np.ndarray:
    """Seasonal swing plus a mean-reverting walk, the same process as main.py."""
    innov = rng.normal(0, 0.35, length)
    signal = np.empty(length)
    signal[0] = innov[0]
    for i in range(1, length):
        signal[i] = 0.98 * signal[i - 1] + innov[i]
    signal += 1.5 * np.sin(2 * np.pi * np.arange(length) / 64)
    return signal


def make_series(rng: np.random.Generator, length: int, kind: str) -> np.ndarray:
    """A series whose own future is the forecast target."""
    if kind == "train":
        return make_signal(rng, length)
    if kind == "walk":
        return np.cumsum(rng.normal(0, 0.4, length))
    if kind == "season":
        time = np.arange(length)
        return (
            2.0 * np.sin(2 * np.pi * time / 24)
            + 0.8 * np.sin(2 * np.pi * time / 168)
            + rng.normal(0, 0.3, length)
        )
    if kind == "trend":
        time = np.arange(length)
        return float(rng.normal()) + float(rng.normal(0, 0.01)) * time + rng.normal(0, 0.4, length)
    raise ValueError(kind)


def make_covariate(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy nonlinear view. About half are delayed."""
    lag_k = 0 if rng.random() < 0.5 else int(rng.integers(1, 41))
    src = signal if lag_k == 0 else lag(signal, lag_k)
    kind = int(rng.integers(0, 3))
    if kind == 0:
        values = signed_power(src, float(rng.uniform(0.4, 1.8)))
    elif kind == 1:
        values = np.tanh(src / float(rng.uniform(0.5, 3.0)))
    else:
        values = src / (1 + float(rng.uniform(0.1, 0.8)) * src**2)
    return observe(values, float(rng.uniform(0.1, 0.7)), rng)


def _pack(rows: list[np.ndarray], variates: bool, device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(np.stack(rows), dtype=torch.float32, device=device)
    return tensor if variates else tensor[:, None, :]


def make_pool_batch(
    rng: np.random.Generator,
    batch: int,
    device: torch.device,
    n_known: int | None = None,
    distractors: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, str]:
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
    return _pack(contexts, False, device), known, _pack(futures, False, device), tag


def make_history_batch(
    rng: np.random.Generator, batch: int, kind: str, context_len: int, device: torch.device
) -> tuple[torch.Tensor, None, torch.Tensor, str]:
    """History-only forecast. The label is the series, noise included."""
    length = context_len + HORIZON
    series = [make_series(rng, length, kind) for _ in range(batch)]
    context = _pack([row[:context_len] for row in series], False, device)
    future = _pack([row[context_len:] for row in series], False, device)
    return context, None, future, f"history {kind} {context_len}"


def make_multi_batch(
    rng: np.random.Generator, batch: int, device: torch.device, n_variates: int | None = None
) -> tuple[torch.Tensor, None, torch.Tensor, str]:
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
    return _pack(contexts, True, device), None, _pack(futures, True, device), tag


def make_categorical_batch(
    rng: np.random.Generator, batch: int, device: torch.device, mode: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
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
    known = _pack(knowns, False, device)
    tag = f"categorical {informative}/{batch}"
    return _pack(contexts, False, device), known, _pack(futures, False, device), tag


def next_batch(
    rng: np.random.Generator, batch: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, str, str]:
    """`(context, known, future, task, tag)`."""
    task = str(rng.choice(TASKS, p=TASK_P))
    if task == "pool":
        context, known, future, tag = make_pool_batch(rng, batch, device)
    elif task == "history":
        context, known, future, tag = make_history_batch(
            rng, batch, str(rng.choice(HISTORY_KINDS)), CONTEXT_LEN, device
        )
    elif task == "multi":
        context, known, future, tag = make_multi_batch(rng, batch, device)
    elif task == "long":
        kind = str(rng.choice(("train",) + HISTORY_KINDS))
        context, known, future, tag = make_history_batch(rng, batch, kind, LONG_CONTEXT, device)
        tag = "long" + tag.removeprefix("history")
    else:
        context, known, future, tag = make_categorical_batch(rng, batch, device)
    return context, known, future, task, tag


def pinball_loss(quantiles: torch.Tensor, target: torch.Tensor, levels: tuple[float, ...]) -> torch.Tensor:
    """Mean pinball loss. `quantiles` is `(B, V, H, Q)` and `target` is `(B, V, H)`."""
    err = target.unsqueeze(-1) - quantiles
    tau = quantiles.new_tensor(levels)
    return torch.maximum(tau * err, (tau - 1) * err).mean()


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        raise SystemExit(
            "MPS cannot run Timestral: its scaler uses float64, which MPS does not support. "
            "Use --device cpu on a Mac, or cuda on a DGX Spark."
        )
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _forecast_mae(forecaster: TimestralForecaster, context: torch.Tensor, known: torch.Tensor | None, future: torch.Tensor) -> float:
    quantiles = forecaster._forecast_raw(
        context, torch.ones_like(context, dtype=torch.bool), HORIZON, known_covariates=known
    )
    median = quantiles[..., forecaster.median_index]
    return float((median - future).abs().mean())


def _median_mae(forecaster: TimestralForecaster, batches: list[tuple]) -> float:
    errors = [_forecast_mae(forecaster, context, known, future) for context, known, future, _tag in batches]
    return float(np.median(errors))


@torch.no_grad()
def evaluate(forecaster: TimestralForecaster, n_series: int, seed: int) -> dict[str, float]:
    """Median absolute error for every trained task. Pool counts are drawn first."""
    rng = np.random.default_rng(seed)
    device = forecaster.device
    scores: dict[str, float] = {}
    for n_known in (1, 5, 20, 64, 0):
        batches = [make_pool_batch(rng, 1, device, n_known=n_known, distractors=False) for _ in range(n_series)]
        scores[f"pool/{n_known}"] = _median_mae(forecaster, batches)
    for kind in HISTORY_KINDS:
        batches = [make_history_batch(rng, 1, kind, CONTEXT_LEN, device) for _ in range(n_series)]
        scores[f"history/{kind}"] = _median_mae(forecaster, batches)
    batches = [make_multi_batch(rng, 1, device, n_variates=3) for _ in range(n_series)]
    scores["multi/3"] = _median_mae(forecaster, batches)
    for kind in ("train", "walk"):
        batches = [make_history_batch(rng, 1, kind, LONG_CONTEXT, device) for _ in range(n_series)]
        scores[f"long/{kind}"] = _median_mae(forecaster, batches)
    for mode in ("bins", "random"):
        batches = [make_categorical_batch(rng, 1, device, mode=mode) for _ in range(n_series)]
        scores[f"categorical/{mode}"] = _median_mae(forecaster, batches)
    return scores


def _print_scores(split: str, scores: dict[str, float]) -> None:
    ordered = [name for name in EVAL_TASKS if name in scores]
    print(f"{split:<6} " + "  ".join(f"{name} {scores[name]:.3f}" for name in ordered), flush=True)


def _rounded(scores: dict) -> dict:
    return {key: round(value, 3) for key, value in scores.items()}


def run_notes(args: argparse.Namespace) -> str:
    """What distinguishes this fine-tune from the pooling-only run."""
    shares = ", ".join(f"{name} {weight:.0%}" for name, weight in zip(TASKS, TASK_P))
    lr = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return (
        "Fine-tune of mistralai/Timestral-5M that pools many noisy known-future views of one "
        "clean series, with the other forecast tasks mixed back into the updates.\n\n"
        "The earlier run was 2000 steps of pooling only. The covariate count was drawn from "
        "1, 2, 4, 8, 16, 32, and 64, and the target was a clean AR(1) plus one season. "
        "Error at 64 views fell from 0.37 to 0.25. History-only error on a random walk rose "
        "from about 1.68 to 2.06, because that process never appeared and every batch rewarded "
        "trusting the covariates.\n\n"
        f"This run is {args.steps} steps, so the pooling task still receives about half of the "
        f"updates. Batch size {args.batch_size}, AdamW learning rate {lr}, "
        f"warmup {min(100, args.steps)} steps, weight decay 0, full model. "
        f"One task is drawn per step: {shares}. "
        "The pooling count is drawn from 0, 1, 2, 4, 8, 16, 32, and 64. "
        "Zero covariates is an ordinary forecast of the same series. "
        "Above zero, half the batches also append 1–8 unrelated known-future series, and the "
        "label stays the clean signal. "
        "History-only batches use a random walk, a two-period seasonal series, or a trend plus "
        "noise, and the label is that series. "
        "Multi batches forecast 2–4 series that share a latent factor. "
        "Long batches are history-only with a 2048-step context. "
        "Categorical batches use one target-encoded known covariate, either quantile bins of "
        "the clean signal or a random code. "
        "The loss is pinball on the model's 21 quantile levels. "
        f"Context {CONTEXT_LEN}, horizon {HORIZON}, except the long-context task."
    )


class Tracker:
    """Per-step W&B curves. Each task keeps its own smoothed loss, since the scales differ."""

    def __init__(self, run: wandb.Run | None) -> None:
        self.run = run
        self._smooth: dict[str, float] = {}
        self._loss_smooth: float | None = None

    def log_loss(self, step: int, task: str, loss: float, lr: float) -> None:
        if self.run is None:
            return
        previous = self._smooth.get(task)
        smooth = loss if previous is None else 0.9 * previous + 0.1 * loss
        self._smooth[task] = smooth
        self._loss_smooth = loss if self._loss_smooth is None else 0.9 * self._loss_smooth + 0.1 * loss
        self.run.log(
            {
                "loss": loss,
                "loss_smooth": self._loss_smooth,
                "lr": lr,
                f"{task}/loss": loss,
                f"{task}/loss_smooth": smooth,
            },
            step=step,
        )

    def log_eval(self, split: str, scores: dict[str, float]) -> None:
        if self.run is None:
            return
        for name, score in scores.items():
            self.run.summary[f"{name}/mae_{split}"] = score

    def log_improvement(self, before: dict[str, float], after: dict[str, float], step: int) -> None:
        """Before, after, and after-minus-before for every eval task. A negative delta is a lower error."""
        if self.run is None:
            return
        rows = []
        for name in EVAL_TASKS:
            delta = after[name] - before[name]
            self.run.summary[f"{name}/mae_delta"] = delta
            rows.append([name, before[name], after[name], delta])
        self.run.log(
            {"eval/mae": wandb.Table(data=rows, columns=["task", "before", "after", "delta"])},
            step=step,
        )


def open_run(args: argparse.Namespace, device: torch.device) -> wandb.Run | None:
    if args.no_wandb:
        return None
    if not wandb.login(prompt=False):
        raise SystemExit("wandb is not logged in. Run `wandb login`, then start this script again.")
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="pool-mix-5m",
        group="pool-mix",
        job_type="train",
        notes=run_notes(args),
        tags=["timestral-5m", "pool-mix", "replay"],
        save_code=True,
        config={
            "model": "mistralai/Timestral-5M",
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": 0.0,
            "warmup_steps": min(100, args.steps),
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "eval_series": args.eval_series,
            "context_len": CONTEXT_LEN,
            "long_context": LONG_CONTEXT,
            "horizon": HORIZON,
            "counts": list(COUNTS),
            "tasks": dict(zip(TASKS, TASK_P)),
            "device": str(device),
        },
    )
    print(f"wandb {run.url}", flush=True)
    return run


def log_checkpoint(run: wandb.Run | None, path: Path, steps: int, before: dict[str, float], after: dict[str, float]) -> None:
    if run is None:
        return
    artifact = wandb.Artifact(
        "timestral-5m-pool-mix",
        type="model",
        description="Timestral-5M after the mixed pooling and replay fine-tune.",
        metadata={
            "steps": steps,
            "base_model": "mistralai/Timestral-5M",
            "scores": {
                name: {
                    "mae_before": round(before[name], 3),
                    "mae_after": round(after[name], 3),
                    "mae_delta": round(after[name] - before[name], 3),
                }
                for name in EVAL_TASKS
            },
        },
    )
    artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=["latest", "pool-mix"])


def train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    run = open_run(args, device)
    tracker = Tracker(run)
    try:
        forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
        model = forecaster.model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        rng = np.random.default_rng(args.seed)
        print(f"device {device}", flush=True)
        before = evaluate(forecaster, args.eval_series, args.eval_seed)
        _print_scores("before", before)
        tracker.log_eval("before", before)

        for step in range(args.steps):
            warmup = min(100, args.steps)
            lr = args.lr * (step + 1) / warmup if step < warmup else args.lr
            for group in optimizer.param_groups:
                group["lr"] = lr
            context, known, future, task, tag = next_batch(rng, args.batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            quantiles = forecaster._forecast_raw(
                context, torch.ones_like(context, dtype=torch.bool), HORIZON, known_covariates=known
            )
            loss = pinball_loss(quantiles, future, forecaster.quantile_levels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            value = float(loss.detach())
            tracker.log_loss(step, task, value, lr)
            if step % args.log_every == 0 or step + 1 == args.steps:
                print(f"step {step:5d}  {tag:<22}  loss {value:.4f}", flush=True)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "steps": args.steps}, output)
        after = evaluate(forecaster, args.eval_series, args.eval_seed)
        _print_scores("after", after)
        print(f"saved {output}", flush=True)
        tracker.log_eval("after", after)
        tracker.log_improvement(before, after, args.steps)
        log_checkpoint(run, output, args.steps, before, after)
    finally:
        if run is not None:
            run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda. MPS is not supported.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=123)
    parser.add_argument("--eval-series", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output", default="checkpoints/pool-mix-5m.pt")
    parser.add_argument("--wandb-project", default="time-series-models")
    parser.add_argument("--wandb-entity", default=None, help="W&B team or user. Defaults to the logged-in entity.")
    parser.add_argument("--no-wandb", action="store_true", help="Skip the W&B run and artifact upload.")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
