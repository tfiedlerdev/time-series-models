# Masked dependent covariates

Whether hiding the true exogenous readings on alternating stretches, including stretches that cross the forecast origin, teaches the model to follow whichever reading is present.

## Setup

The checked-in mixture, with 40% of the exogenous batches masking their true readings. Each stretch keeps one reading and sets the others to NaN, and the reading that is kept changes from one stretch to the next. One boundary falls between 128 steps before the origin and the end of the horizon. The run is 6000 steps at batch 16, seed 0. W&B run [`hn2d1cxu`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/hn2d1cxu), artifact `timestral-5m-pool-mix:masked-p40`.

The eval draws the exogenous series with the mask off, then hides the readings with four stretches and scores three views of the same series: the masked rows, every reading present, and no covariate rows. 32 series, shared with the reference fine-tune and the pretrained model. W&B run [`nglcfiox`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/nglcfiox). The reference is [`06cmyv72`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/06cmyv72).

| task | pretrained | ref-main | masked-p40 |
|---|---:|---:|---:|
| masked/1 | 0.544 | 0.502 | 0.488 |
| masked/1-full | 0.396 | 0.368 | 0.371 |
| masked/1-none | 0.619 | 0.618 | 0.617 |
| masked/2 | 0.527 | 0.501 | 0.478 |
| masked/2-full | 0.516 | 0.479 | 0.478 |
| masked/2-none | 0.713 | 0.615 | 0.574 |
| pool/64 | 0.380 | 0.253 | 0.257 |
| views/64 | 0.351 | 0.258 | 0.258 |

## What it shows

With one reading, the masked run is at 0.488 against the reference's 0.502 and a fully observed 0.371. Both fine-tunes already use the visible parts: the no-covariate error stays at 0.62. The new task moves the masked error by about 0.014.

With two readings the masked run lands on its own fully observed error, 0.478 against 0.478. The reference sits a little higher, 0.501 against a fully observed 0.479. The no-covariate error falls too, from 0.615 to 0.574, so part of the gain is the fine-tune getting better at this series, and part is the masked forecast catching the fully observed one.

Pooling is unchanged. At 64 views the masked run is at 0.257 against the reference's 0.253.

TIME, all 98 tasks, is W&B run [`b09lxsuf`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/b09lxsuf). The mean relative change in MASE is -3.5%, the same as the reference's finished TIME run, [`c1zb66vp`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/c1zb66vp). The fev covariate subset is [`lj0ayrdf`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/lj0ayrdf): SQL -6.3% on 26 tasks.

## What follows

The pretrained model and the reference fine-tune already follow a reading that is missing on some stretches. Training on the alternation pulls the two-reading case onto the fully observed error and leaves TIME where the reference fine-tune has it. The gain on one reading is a hundredth of MAE. The checked-in mixture already covers this shape well enough that the extra task is a small one.
