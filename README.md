Side project, in progress. Fine-tunes of [Timestral](https://github.com/mistralai/timestral) (for now 5M, later larger sizes), and the benchmarks that score them. The `timestral/` directory is Mistral's inference code, Apache 2.0, included as a submodule. It is not part of this project.

## Noisy views of one signal

Pretrained Timestral-5M stops gaining once about ten noisy nonlinear views of a signal are known over the horizon. The error falls from about 1.16 with none known to about 0.43 by ten, and stays there through one hundred. A 6000-step fine-tune reaches about 0.25 by sixteen views and then flattens, still above an oracle that inverts the usable views.

![Error against how many noisy views are known ahead. Blue is pretrained Timestral-5M. Orange is the short fine-tune.](findings/view-curve.png)

The setup, the table, and why the error does not go to zero are in [findings/view-curve.md](findings/view-curve.md).

The aim is a better synthetic data mix: raise that capability without giving up performance on the model's other tasks. Replaying broad generic series against the frozen pretrained forecast is in [findings/distill-replay.md](findings/distill-replay.md).

A true covariate among decoys is in [findings/decoy-sweep.md](findings/decoy-sweep.md). A target that follows a measurement known only up to the origin is in [findings/leading-measurement.md](findings/leading-measurement.md).

## Research idea queue

One shape is missing from the synthetic mix. It is a capability the pretrained model has, and a fine-tune that never sees it can drop it.

**Masked dependent covariates.** Timestral treats NaN as missing, and the fine-tune never asks it to. Every training series is fully observed. A useful task is the exogenous series with its true readings masked on different stretches: at one time the first reading is present and the second is missing, and later the reverse. The forecast should follow whichever reading is actually there.

The [decoy sweep](findings/decoy-sweep.md) turned up a different hole. This one is a blend the fine-tune learned.

**A true reading beside a stack of partial matches.** The [decoy sweep](findings/decoy-sweep.md) shows the fine-tune using a true driver row at one or two calendar decoys, then blending those decoys in until, past about six, the driver row is worth nothing. Training only places one or two decoys next to that row, which is the region where the checkpoint still keeps it. The missing task is the exogenous series with its true reading kept, and a stack of partial-match decoys beside it: both cycles, the daily cycle with the weekly cycle flipped, or the slope, each with its own walk. The count should run through the range where the blend appears, up to 16, with most of the mass above two. The forecast should stay near the no-decoy error as the stack grows, and stay below the same series with the true reading removed. Some batches should keep the current all-decoy case, so a shared cycle is still used when that is all there is.

**A target composed of covariates that depend on each other.** The covariates in the current mix are each a view of one signal, or a decoy that shares a piece of it. What is missing is a small graph: each covariate is a nonlinear function of its parents plus its own noise, the noise scales differ, and the target is a nonlinear composition of a subset of them. After the series are drawn, hide some covariates in one of three ways. Remove one for the whole series. Keep one only through the context and drop it on the horizon. Mask one on stretches of the context and leave it known elsewhere, including the horizon. The forecast should follow whichever causes are actually present. A clean parent should be preferred to a noisy child of the same cause. A covariate that stops at the origin can still set the start of the horizon from its recent values. A covariate that was removed entirely should be ignored, and the target recovered from the ones that remain. Keep a draw only when a forecast that sees the kept covariates beats a history-only forecast, and when removing those covariates makes that forecast worse. Otherwise the batch is either pure noise or fully redundant, and the update has nothing to teach.
