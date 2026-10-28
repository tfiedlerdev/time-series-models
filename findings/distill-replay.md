# Replaying broad series against the pretrained forecast

Whether a slice of each update spent on generic series, labelled by the frozen pretrained model, keeps the fine-tune from giving up the forecasts it already had.

## Setup

The checked-in mixture is unchanged. On top of it, a fifth of the draws are history-only series from three generators: a random sum and product of kernels in the KernelSynth bank, drawn with random Fourier features; a composite of seasons, a trend, an AR process, level shifts and spikes; and intermittent counts whose rate follows a season and a slow walk. Context length is drawn from 256, 512, 1024, 2048 and 4096, and the horizon from 16, 48, 96, 168 and 336.

`teacher` sets the loss to the absolute distance between the fine-tune's quantiles and the quantiles of the frozen pretrained 5M model on the same input, scaled by the context's own spread. `truth` is ordinary pinball against the series, scaled the same way. Both runs are 6000 steps at batch 16, seed 0.

- `ref-main`, the checked-in mixture alone. W&B run [`06cmyv72`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/06cmyv72), artifact `timestral-5m-pool-mix:ref-main`.
- `distill-teacher-w25`. W&B run [`uwkp74q8`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/uwkp74q8), artifact `timestral-5m-pool-mix:distill-teacher-w25`.
- `distill-truth-w25`. W&B run [`a4jbw3ry`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/a4jbw3ry), artifact `timestral-5m-pool-mix:distill-truth-w25`.

The three models and the pretrained baseline share one training-eval draw of 32 series, W&B run [`nks2usj1`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/nks2usj1).

| task | pretrained | ref-main | teacher | truth |
|---|---:|---:|---:|---:|
| generic/kernel | 0.268 | 0.365 | 0.274 | 0.288 |
| generic/composite | 0.353 | 0.375 | 0.352 | 0.346 |
| generic/intermittent | 0.526 | 0.540 | 0.527 | 0.511 |
| pool/1 | 0.816 | 0.557 | 0.639 | 0.582 |
| pool/64 | 0.380 | 0.253 | 0.266 | 0.256 |
| views/64 | 0.351 | 0.258 | 0.264 | 0.255 |
| views/0 | 1.384 | 1.239 | 1.199 | 1.261 |
| long/walk | 1.670 | 1.844 | 1.745 | 1.725 |

`generic/*` is horizon MAE divided by the context's spread, on series the training mixture never builds except inside this task. The reference sees them only at eval.

## What it shows

The reference fine-tune pays for its pooling gain on series shaped like the benchmarks. On kernel draws its scaled error goes from 0.268 to 0.365. A long random-walk context goes from 1.670 to 1.844. On TIME, all 98 tasks, the mean relative change against the pretrained baseline is -3.5%, W&B run [`c1zb66vp`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/c1zb66vp). 83 of the 98 tasks get worse. The largest are high-frequency physical series: SG_Carpark/15T/medium -24%, Coastal_T_S/20T/short -18%, current_velocity/10T/long -14%.

Teacher labels put the kernel error back to 0.274, next to the pretrained 0.268, and the long walk back to 1.745. Pooling at 64 views stays at 0.266 against the reference's 0.253 and the pretrained 0.380. The single-view pooling task gives some of the gain back, 0.639 against the reference's 0.557.

Truth labels keep more of that single-view gain, 0.582, and they beat the pretrained model on intermittent counts, 0.511 against 0.526. The kernel error only comes back to 0.288.

TIME, all 98 tasks, is [`w6a0er5y`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/w6a0er5y) for the teacher run and [`h2bu1puf`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/h2bu1puf) for the truth run.

| model | TIME MASE, 98 tasks | TIME quantile loss | fev SQL, 26 covariate tasks |
|---|---:|---:|---:|
| ref-main | -3.5% | -2.8% | -8.0% on the 20 tasks it finished |
| distill-truth-w25 | -1.1% | -0.9% | -4.3% |
| distill-teacher-w25 | -0.1% | -0.0% | -2.3% |

The fev runs are [`qv48b031`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/qv48b031) and [`2ef5ouxx`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/2ef5ouxx). On the 20 fev tasks the reference finished, the teacher run is at -3.0% SQL against the reference's -8.0%.

Teacher is better than the pretrained model on 32 of the 98 TIME tasks and worse on 66. The losses are small: worse by more than one percent on 4 tasks, and the only one past two percent is Coastal_T_S/20T/medium at -2.6%, where the reference is at -16.1%. It beats the reference by more than two percentage points on 54 of the 98. On fev it is better on 5 of 26 tasks.

## What follows

A fifth of the updates, labelled by the frozen pretrained model on broad synthetic series, brings the TIME mean back to the pretrained model and keeps the many-view pooling gain. It does that by shrinking the fine-tune's regressions, not by beating the pretrained model on a block of tasks. Labelling those series with the truth keeps more of the single-view pooling gain and leaves about a percent of TIME, and 4.3% of fev SQL, on the table. The teacher run is the one that still pools and stops costing the forecasts the pretrained model already had.
