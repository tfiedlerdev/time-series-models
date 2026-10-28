# A shorter post-train on the teacher mixture

Whether fewer steps, or a lower learning rate, on the [teacher replay](distill-replay.md) mixture removes the fev cost it still leaves. Timestral's own post-training runs far fewer steps than its pretraining, so the 6000-step fine-tune may simply move the weights further than the new tasks need.

## Setup

The mixture is `configs/distill-teacher.yaml`, unchanged: a fifth of the draws are generic series labelled by the frozen 5M model. Batch 16, seed 0, warmup over 5% of the steps and cosine decay to a tenth of the peak. Two variations on the 6000-step run at peak 3e-5:

- `teacher-steps1500`: 1500 steps at peak 3e-5. W&B run [`0eqgwi8b`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/0eqgwi8b).
- `teacher-lr1e-5`: 6000 steps at peak 1e-5. W&B run [`1jcrl45e`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/1jcrl45e).

Both are plain command-line flags, `--steps 1500` and `--lr 1e-5`, so this branch has no code change.

## Benchmarks

TIME is all 98 tasks: W&B runs [`7nhonhmu`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/7nhonhmu) (1500 steps) and [`w3sdv36o`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/w3sdv36o) (lr 1e-5). fev is the 26-task covariate subset: [`emqt5seb`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/emqt5seb) and [`6fveic5s`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/6fveic5s). Changes are in the error against the pretrained model, so a negative number is a lower error.

| model | TIME MASE | TIME QL | TIME tasks worse | fev SQL | fev MASE | fev tasks worse | fev tasks >+1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref-main | +3.52% | +2.82% | 83 of 98 | +8.0% on 20 tasks | +8.5% on 20 tasks | | |
| teacher, 6000 steps at 3e-5 | +0.11% | +0.05% | 66 | +2.27% | +2.46% | 21 of 26 | 14 |
| 1500 steps at 3e-5 | +0.06% | +0.05% | 64 | +0.83% | +1.01% | 15 of 26 | 7 |
| 6000 steps at 1e-5 | +0.07% | +0.03% | 66 | +1.16% | +1.32% | 17 of 26 | 9 |

The median fev SQL change is +0.13% for 1500 steps and +0.44% at the lower rate, against +1.27% for the full run.

## What the fine-tune keeps

The training eval, 32 series and the same draws for every model. W&B runs [`oh8ud8g3`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/oh8ud8g3) (pretrained, ref-main, teacher) and [`ek85zo39`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/ek85zo39) (pretrained, 1500 steps, lr 1e-5). The pretrained column is the same in both.

| task | pretrained | ref-main | teacher | 1500 steps | lr 1e-5 |
|---|---:|---:|---:|---:|---:|
| pool/1 | 0.816 | 0.557 | 0.639 | 0.742 | 0.738 |
| pool/64 | 0.380 | 0.253 | 0.266 | 0.291 | 0.286 |
| views/8 | 0.436 | 0.348 | 0.361 | 0.377 | 0.377 |
| views/64 | 0.351 | 0.258 | 0.264 | 0.286 | 0.280 |
| categorical/random | 1.329 | 1.025 | 1.137 | 1.192 | 1.166 |
| exogenous/decoy | 0.625 | 0.602 | 0.589 | 0.606 | 0.601 |
| generic/kernel | 0.219 | 0.276 | 0.223 | 0.234 | 0.222 |
| generic/piecewise | 0.297 | 0.374 | 0.312 | 0.310 | 0.305 |
| long/walk | 1.670 | 1.844 | 1.745 | 1.747 | 1.766 |
| realcov/heldout | 0.341 | 0.324 | 0.318 | 0.324 | 0.326 |

## What follows

Most of the fev cost comes from how far the weights travel, not from the mixture. A quarter of the steps, or a third of the rate, cuts the fev SQL cost from +2.3% to about +1%, and halves the number of covariate tasks that get more than 1% worse. TIME was already flat with the teacher replay and stays flat.

The trade is pooling. Most of the gain on the new tasks is kept: at 64 views the error is 0.291 against 0.380 pretrained, about 80% of the way to the full run's 0.266. The single-view and categorical rows give up more, since those are the slowest to learn.

Of the two, the shorter run is cheaper on fev for nearly the same pooling and a quarter of the compute. It is the better default to carry into further mixture experiments, which should then be compared at 1500 steps.
