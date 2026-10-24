"""W&B run, per-task loss curves, and the model artifact."""

import argparse
from pathlib import Path

import torch
import wandb

from training.schedule import CONTEXT_LEN, COUNTS, EVAL_TASKS, HORIZON, LONG_CONTEXT, TASK_P, TASKS, warmup_steps


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
        f"This run is {args.steps} steps, so pooling receives about {args.steps * TASK_P[0]:.0f} updates "
        f"and the exogenous task about {args.steps * TASK_P[2]:.0f}. "
        f"Each update draws {args.batch_size} series from independent tasks and averages their gradients. "
        f"Peak AdamW learning rate {lr}, linear warmup over the first {warmup_steps(args.steps)} steps, "
        "then cosine decay to a tenth of the peak, weight decay 0, full model. "
        f"One task is drawn per series: {shares}. "
        "The pooling count is drawn from 0, 1, 2, 4, 8, 16, 32, and 64. "
        "Zero covariates is an ordinary forecast of the same series. "
        "Above zero, half the batches also append 1–8 unrelated known-future series, and the "
        "label stays the clean signal. "
        "Exogenous batches forecast a series driven by one latent walk plus a shared calendar. "
        "Some known rows are noisy readings of that walk. One or two others share the daily and "
        "weekly cycle and carry an independent walk, already in the context. Half also use a "
        "horizon of 168. Some are nonnegative, some are multiplied by a known on/off pattern, "
        "and some carry a constant target-encoded category. "
        "Past batches add lagged measurements that end at the forecast origin. "
        "History-only batches use a random walk, a two-period seasonal series, or a trend plus "
        "noise, and the label is that series. A quarter of them forecast 168 steps. "
        "Multi batches forecast 2–4 series that share a latent factor. "
        "Long batches are history-only with a 2048-step context. "
        "Categorical batches use one target-encoded known covariate, either quantile bins of "
        "the clean signal or a random code. "
        "The loss is pinball on the model's 21 quantile levels. "
        f"Context {CONTEXT_LEN}, horizon {HORIZON}, except where a task says otherwise."
    )


class Tracker:
    """Per-step W&B curves. Each task keeps its own smoothed loss, since the scales differ."""

    def __init__(self, run: wandb.Run | None) -> None:
        self.run = run
        self._smooth: dict[str, float] = {}
        self._loss_smooth: float | None = None

    def log_loss(self, step: int, parts: list[tuple[str, float]], overall: float, lr: float) -> None:
        if self.run is None:
            return
        payload: dict[str, float] = {"lr": lr, "loss": overall}
        self._loss_smooth = overall if self._loss_smooth is None else 0.9 * self._loss_smooth + 0.1 * overall
        payload["loss_smooth"] = self._loss_smooth
        for task, loss in parts:
            previous = self._smooth.get(task)
            smooth = loss if previous is None else 0.9 * previous + 0.1 * loss
            self._smooth[task] = smooth
            payload[f"{task}/loss"] = loss
            payload[f"{task}/loss_smooth"] = smooth
        self.run.log(payload, step=step)

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
            "warmup_steps": warmup_steps(args.steps),
            "lr_schedule": "warmup-cosine",
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
