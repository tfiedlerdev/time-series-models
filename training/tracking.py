"""W&B run, per-task loss curves, and the model artifact."""

import argparse
from pathlib import Path

import torch
import wandb

from training.data_config import DataConfig
from training.schedule import EVAL_TASKS, warmup_steps


def run_notes(args: argparse.Namespace) -> str:
    """What distinguishes this fine-tune from the pooling-only run."""
    data: DataConfig = args.data
    shares = data.shares()
    share_text = ", ".join(f"{name} {share:.0%}" for name, share in shares.items())
    lr = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    return (
        "Fine-tune of mistralai/Timestral-5M that pools many noisy known-future views of one "
        "clean series, with the other forecast tasks mixed back into the updates.\n\n"
        "The earlier run was 2000 steps of pooling only. The covariate count was drawn from "
        "1, 2, 4, 8, 16, 32, and 64, and the target was a clean AR(1) plus one season. "
        "Error at 64 views fell from 0.37 to 0.25. History-only error on a random walk rose "
        "from about 1.68 to 2.06, because that process never appeared and every batch rewarded "
        "trusting the covariates.\n\n"
        f"This run is {args.steps} steps, so pooling receives about {args.steps * shares.get('pool', 0):.0f} updates "
        f"and the exogenous task about {args.steps * shares.get('exogenous', 0):.0f}. "
        f"Each update draws {args.batch_size} series from independent tasks and averages their gradients. "
        f"Peak AdamW learning rate {lr}, linear warmup over the first {warmup_steps(args.steps)} steps, "
        "then cosine decay to a tenth of the peak, weight decay 0, full model. "
        f"Data config {args.data_config}. One task is drawn per series: {share_text}. "
        "The pooling count is drawn from 0, 1, 2, 4, 8, 16, 32, and 64, with most of the mass "
        "on 1 through 16 and a thin tail on 32 and 64. "
        "Zero known views is an ordinary forecast of the same series. "
        "Half of the pooling batches hold a pool of 20 views when the count is 16 or below, and 64 "
        "when it is 32 or 64, and reveal only the drawn count over the horizon; "
        "the other views stop at the origin. "
        "On the other half, above zero, half the batches also append 1–8 unrelated known-future series, "
        "and the label stays the clean signal. "
        "Exogenous batches forecast a series driven by one latent walk plus a shared calendar, "
        "and half of them also carry a slope. "
        "Some known rows are noisy readings of that walk. Decoy rows share one piece and differ "
        "on the rest: both cycles, the daily cycle with a different weekly cycle, or the slope "
        "with a different residual. The mismatch is already in the context. A quarter of the "
        "batches have no true reading, only those decoys. Half also use a horizon of 168. "
        "Some are nonnegative, some are multiplied by a known on/off pattern, "
        "and some carry a constant target-encoded category. "
        "Past batches add lagged measurements that end at the forecast origin. "
        "History-only batches use a random walk, a two-period seasonal series, or a trend plus "
        "noise, and the label is that series. A quarter of them forecast 168 steps. "
        "Multi batches forecast 2–4 series that share a latent factor. "
        "Long batches are history-only with a 2048-step context. "
        "Categorical batches use one target-encoded known covariate, either quantile bins of "
        "the clean signal or a random code. "
        "The loss is pinball on the model's 21 quantile levels. "
        f"Context {data.context_len}, long context {data.long_context}, "
        f"horizon {data.horizon}, exogenous horizon {data.exo_horizon}."
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
        name=args.run_name,
        group="pool-mix",
        job_type="train",
        notes=run_notes(args),
        tags=["timestral-5m", "pool-mix", "replay", *_extra_tags(args)],
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
            "data_config": str(args.data_config),
            "data": args.data.model_dump(mode="json"),
            "data_shares": args.data.shares(),
            "device": str(device),
        },
    )
    print(f"wandb {run.url}", flush=True)
    return run


def _extra_tags(args: argparse.Namespace) -> list[str]:
    return [tag.strip() for tag in getattr(args, "tags", "").split(",") if tag.strip()]


def log_checkpoint(
    run: wandb.Run | None,
    path: Path,
    steps: int,
    before: dict[str, float],
    after: dict[str, float],
    *,
    interrupted: bool = False,
) -> None:
    if run is None:
        return
    # A partial run keeps its own alias. Promoting it to latest would make the next
    # benchmark score a model that stopped early.
    aliases = ["interrupted"] if interrupted else ["latest", "pool-mix", run.name]
    artifact = wandb.Artifact(
        "timestral-5m-pool-mix",
        type="model",
        description="Timestral-5M after the mixed pooling and replay fine-tune.",
        metadata={
            "steps": steps,
            "interrupted": interrupted,
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
    run.log_artifact(artifact, aliases=aliases)
