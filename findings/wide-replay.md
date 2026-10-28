# A wider replay set

Whether replaying more kinds of generic series against the frozen pretrained forecast removes the small TIME and fev cost that the [teacher replay](distill-replay.md) still leaves.

## Setup

The teacher replay mixture, with the same 20% share of draws. Its three generators (kernel draws, composites, intermittent counts) are joined by four more:

- `regime`: an AR process that switches between two or three regimes, each with its own mean, memory and noise.
- `events`: a seasonal AR baseline with localised plateaus, ramps, decays and bumps.
- `piecewise`: a trend with one to five changepoints, a season, and AR noise, sometimes multiplicative.
- `profile`: a sawtooth, square, clipped-arch or narrow-peak cycle whose amplitude drifts, with noise that grows with the level. Arched profiles stay at zero between cycles, like solar output.

A fifth of the replay series have one to four NaN stretches in the context, and 15% are rounded onto a coarse grid. The training loop now passes the observed mask, so a NaN in the target context is read as missing. Labels are still the frozen 5M model's quantiles. The run is 6000 steps at batch 16, seed 0. W&B run [`1pgsulp5`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/1pgsulp5), artifact `timestral-5m-pool-mix:wide-replay`.

The training eval adds a history-only `generic/*` row for each new kind. 32 series, the same draws for every model. W&B runs [`ehriowa1`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/ehriowa1) (pretrained, ref-main, teacher) and [`o93suovx`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/o93suovx) (wide replay).

| task | pretrained | ref-main | teacher | wide replay |
|---|---:|---:|---:|---:|
| generic/kernel | 0.219 | 0.276 | 0.223 | 0.210 |
| generic/composite | 0.234 | 0.266 | 0.233 | 0.230 |
| generic/intermittent | 0.561 | 0.565 | 0.563 | 0.563 |
| generic/regime | 0.640 | 0.605 | 0.625 | 0.636 |
| generic/events | 0.326 | 0.323 | 0.325 | 0.326 |
| generic/piecewise | 0.297 | 0.374 | 0.312 | 0.311 |
| generic/profile | 0.173 | 0.194 | 0.174 | 0.173 |
| pool/64 | 0.380 | 0.253 | 0.266 | 0.270 |
| views/64 | 0.351 | 0.258 | 0.264 | 0.261 |
| long/walk | 1.670 | 1.844 | 1.745 | 1.771 |

## What it shows

The narrow replay already holds the new kinds. Changepoint trends are where the reference fine-tune loses most, 0.297 to 0.374, and the teacher replay is back at 0.312 without having seen one. The wide replay, which trains on them, is at 0.311. On profiles, events and intermittent counts every replay model sits on the pretrained error.

The benchmarks match the narrow replay. TIME, all 98 tasks, is W&B run [`0is8kgc1`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/0is8kgc1). The fev covariate subset, 26 tasks, is [`y2nwbj86`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/y2nwbj86). Changes are in the error against the pretrained model, so a negative number is a lower error.

| model | TIME MASE | TIME tasks worse | fev SQL | fev MASE |
|---|---:|---:|---:|---:|
| ref-main | +3.5% | 83 of 98 | +8.0% on 20 tasks | +8.5% on 20 tasks |
| teacher | +0.11% | 66 of 98 | +2.3% | +2.5% |
| wide replay | +0.10% | 67 of 98 | +2.2% | +2.4% |

Pooling is kept. At 64 views the wide replay is at 0.270 against the teacher run's 0.266.

## What follows

What the replay contains matters little once its labels come from the frozen model. Three generators and seven give the same TIME mean, the same fev cost, and the same pooling. Updates on kinds the replay never sees are held about as well as updates on kinds it does. The cost that is left, about 2% fev SQL, sits on the covariate tasks, and a history-only replay of any shape does not reach it.
