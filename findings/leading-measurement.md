# A leading measurement that stops at the origin

Whether a task where the target follows an earlier measurement, known only up to the forecast origin, teaches the model to read the first part of the horizon off that measurement.

## Setup

The checked-in mixture, plus a lead task at 7.4% of the draws. The measurement is a walk. The target is a nonlinear function of it from 4 to 48 steps earlier (a tanh, a signed power, or a signed scale), plus a smaller walk of its own and, in half the batches, a daily and weekly cycle. The measurement is a past-only row with noise at 5–40% of its spread, so its last `delay` values set the first `delay` steps of the horizon and say nothing after. Half the batches add one or two decoy rows: lagged readings of the target, which follow it instead of leading it. The run is 6000 steps at batch 16, seed 0. W&B run [`2aq5pxkp`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/2aq5pxkp), artifact `timestral-5m-pool-mix:lead-w08`.

The eval draws the task with a delay of 24, one decoy, and no cycle, and splits the horizon at the delay. `early` is the first 24 steps and `late` the rest. `-none` is the same series with the measurement row taken out and only the decoy left. 32 series, shared with the reference fine-tune and the pretrained model. W&B run [`4zvm4rtd`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/4zvm4rtd). The reference is [`06cmyv72`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/06cmyv72).

| task | pretrained | ref-main | lead-w08 |
|---|---:|---:|---:|
| lead/early | 0.676 | 0.728 | 0.630 |
| lead/early-none | 0.711 | 0.753 | 0.721 |
| lead/late | 1.050 | 0.990 | 0.921 |
| lead/late-none | 1.085 | 1.031 | 0.948 |
| pool/64 | 0.380 | 0.253 | 0.259 |
| views/64 | 0.351 | 0.258 | 0.256 |
| views/0 | 1.384 | 1.239 | 1.352 |

`lead_headroom.py` draws the same 32 series again and forecasts the early horizon with each series' true response applied to the noisy reading. With the target's own walk at the origin read off the context, the median early error is 0.292. Last value, with no measurement, is 0.769.

## What it shows

The lead run uses the measurement more than either other model. Removing the measurement raises its early error from 0.630 to 0.721, a gap of 0.091. The pretrained model's gap is 0.035, and the reference's 0.025. The late steps improve with or without the measurement, 0.921 and 0.948, so part of the gain is the run getting better at this kind of target overall.

It is still far from what the measurement allows. A forecast that knows the response and the delay reaches 0.292 on the early steps, and the run is at 0.630. The model has to infer both from 512 context steps per series, with a different nonlinearity each time. 6000 steps at 7.4% of the draws is about 7,000 such series, and that moved the error by about 0.05 relative to the pretrained model.

Pooling is unchanged. At 64 views the lead run is at 0.259 against the reference's 0.253. The forecast with no views known, `views/0`, is worse than the reference, 1.352 against 1.239.

TIME, all 98 tasks, is W&B run [`mq765m5p`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/mq765m5p). The fev covariate subset, 26 tasks, is [`gjwufp1y`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/gjwufp1y). Changes are in the error against the pretrained model, so a negative number is a lower error.

| model | TIME MASE | TIME quantile loss | fev SQL, 26 tasks | fev SQL, the 20 tasks ref-main finished |
|---|---:|---:|---:|---:|
| ref-main | +3.5% | +2.8% | | +8.0% |
| lead-w08 | +3.4% | +3.1% | +5.5% | +7.2% |
| distill-teacher-w25 | +0.1% | +0.05% | +2.3% | +3.0% |

The lead run is worse than the pretrained model on 85 of the 98 TIME tasks, the same pattern as the reference. The new task neither costs nor recovers general accuracy.

## What follows

A leading measurement is a skill the model can be taught a little of, and the benchmarks do not register it. The gap to the bound says the task needs either much more of the budget or an easier ramp: a fixed delay before a drawn one, or a linear response before a nonlinear one. It should go on top of the teacher replay, which is the only mixture that holds TIME. On its own it carries the reference fine-tune's +3.4% TIME error with it.
