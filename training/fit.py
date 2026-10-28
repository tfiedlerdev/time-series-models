"""The fine-tuning loop. Each update averages a few independently drawn tasks."""

import argparse
from pathlib import Path

import numpy as np
import torch
from timestral import TimestralForecaster

from training.batches import next_batch
from training.data_config import DEFAULT_DATA_CONFIG, DataConfig
from training.eval import evaluate, pinball_loss, print_scores
from training.schedule import learning_rate
from training.tracking import Tracker, log_checkpoint, open_run


def resolve_device(name: str) -> torch.device:
    if name == "mps":
        raise SystemExit(
            "MPS cannot run Timestral: its scaler uses float64, which MPS does not support. "
            "Use --device cpu on a Mac, or cuda on a DGX Spark."
        )
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _batch_key(sample: tuple) -> tuple:
    context, known, past, future = sample[:4]
    return (
        context.shape[1:],
        None if known is None else known.shape[1:],
        None if past is None else past.shape[1:],
        future.shape[1:],
        sample[4] == "distill",
    )


def _load_teacher(data: DataConfig, device: torch.device) -> TimestralForecaster | None:
    """The frozen pretrained model, when the distill task takes its labels from it."""
    if data.distill is None or data.distill.label != "teacher":
        return None
    teacher = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
    teacher.model.requires_grad_(False)
    return teacher


def _observed_std(context: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Standard deviation over the observed steps of each row, shaped `(B, V, 1)`."""
    count = observed.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (context * observed).sum(dim=-1, keepdim=True) / count
    return ((((context - mean) * observed) ** 2).sum(dim=-1, keepdim=True) / count).sqrt()


def _cat_samples(group: list[tuple]) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    context = torch.cat([sample[0] for sample in group], dim=0)
    known = None if group[0][1] is None else torch.cat([sample[1] for sample in group], dim=0)
    past = None if group[0][2] is None else torch.cat([sample[2] for sample in group], dim=0)
    future = torch.cat([sample[3] for sample in group], dim=0)
    return context, known, past, future


def train(args: argparse.Namespace) -> None:
    if not getattr(args, "data_config", None):
        args.data_config = str(DEFAULT_DATA_CONFIG)
    if getattr(args, "data", None) is None:
        args.data = DataConfig.load(args.data_config)
    device = resolve_device(args.device)
    run = open_run(args, device)
    tracker = Tracker(run)
    completed = 0
    committed = False
    stopped = False
    exit_code = 0
    try:
        forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", device=device)
        model = forecaster.model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        teacher = _load_teacher(args.data, device)
        rng = np.random.default_rng(args.seed)
        print(f"device {device}", flush=True)
        before = evaluate(forecaster, args.eval_series, args.eval_seed)
        print_scores("before", before)
        tracker.log_eval("before", before)

        def commit(steps: int) -> None:
            nonlocal committed
            if committed:
                return
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "steps": steps}, output)
            after = evaluate(forecaster, args.eval_series, args.eval_seed)
            print_scores("after", after)
            print(f"saved {output}", flush=True)
            tracker.log_eval("after", after)
            tracker.log_improvement(before, after, steps)
            log_checkpoint(run, output, steps, before, after, interrupted=steps < args.steps)
            committed = True

        for step in range(args.steps):
            lr = learning_rate(step, args.steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            samples = [next_batch(rng, 1, device, args.data) for _ in range(args.batch_size)]
            groups: dict[tuple, list] = {}
            for sample in samples:
                groups.setdefault(_batch_key(sample), []).append(sample)
            optimizer.zero_grad(set_to_none=True)
            weighed: dict[str, list[tuple[float, int]]] = {}
            overall = 0.0
            for group in groups.values():
                context, known, past, future = _cat_samples(group)
                observed = torch.isfinite(context)
                context = torch.where(observed, context, torch.zeros_like(context))
                quantiles = forecaster._forecast_raw(
                    context,
                    observed,
                    future.shape[-1],
                    past_covariates=past,
                    known_covariates=known,
                )
                if group[0][4] == "distill":
                    scale = _observed_std(context, observed).clamp_min(0.1).unsqueeze(-1)
                    if teacher is None:
                        loss = pinball_loss(quantiles / scale, future / scale[..., 0], forecaster.quantile_levels)
                    else:
                        with torch.no_grad():
                            labels = teacher._forecast_raw(
                                context,
                                observed,
                                future.shape[-1],
                                past_covariates=past,
                                known_covariates=known,
                            )
                        loss = ((quantiles - labels) / scale).abs().mean()
                else:
                    loss = pinball_loss(quantiles, future, forecaster.quantile_levels)
                (loss * (len(group) / args.batch_size)).backward()
                value = float(loss.detach())
                overall += value * len(group) / args.batch_size
                weighed.setdefault(group[0][4], []).append((value, len(group)))
            parts = [
                (task, sum(value * count for value, count in items) / sum(count for _, count in items))
                for task, items in weighed.items()
            ]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed = step + 1
            tracker.log_loss(step, parts, overall, lr)
            if step % args.log_every == 0 or step + 1 == args.steps:
                tags = ",".join(sorted({sample[4] for sample in samples}))
                print(f"step {step:5d}  {tags:<22}  loss {overall:.4f}", flush=True)

        commit(completed)
    except KeyboardInterrupt:
        stopped = True
        print(f"\ninterrupted after step {completed} of {args.steps}", flush=True)
        exit_code = 130
        if completed and not committed:
            print("Saving the checkpoint and running the eval. Interrupt again to skip that.", flush=True)
            try:
                commit(completed)
                exit_code = 0
            except KeyboardInterrupt:
                print("Skipped the remaining save.", flush=True)
    finally:
        if run is not None:
            run.summary["completed_steps"] = completed
            if stopped and completed < args.steps:
                run.summary["interrupted"] = True
            run.finish(exit_code)
