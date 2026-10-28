# Real series in the post-train

Whether a share of real series removes the fev cost the synthetic mixture leaves behind. Timestral's own post-training mixes real and synthetic data, and every mixture here so far has been synthetic only.

## The corpus

`Salesforce/GiftEvalPretrain` is GIFT-Eval's pretraining split, built without GIFT-Eval's test series. fev-bench and TIME were not part of that check, so `training/real_sources.py` keeps a dataset only when its source does not appear in either benchmark. Left out for overlap:

- fev-bench: favorita, m5, the proenfo loads (gfc12/14/17), kdd2022, fred_md, spain (entsoe), and the three tourism sets.
- TIME: the Australian electricity, solar and wind sets, azure_vm_traces_2017, bitcoin_with_missing, the two CDC flu sets, weather and oikolab_weather.

The climate grids and the multi-gigabyte traces are left out for size. What remains is 54 datasets, about 2 GB, in `data/gift-pretrain`. They cover traffic and transport, building energy and smart meters, air quality and subseasonal weather, web traffic, disease counts, ATM withdrawals, and the M1, M3 and CIF competitions. Every 20th series of each dataset is held out from training and scored in the eval as `real/heldout`: a scaled MAE on 128 windows with a 96-step horizon.

The `real` task draws a dataset, one of its training series, a context of 128 to 4096 steps and a horizon of 8 to 336. A multivariate dataset is split into one series per variate. The label is either the true future (scaled pinball, `label: truth`) or a frozen teacher's quantiles (`label: teacher`).

One caveat: in this branch a window's horizon is drawn uniformly among the horizons that fit, and its future must be fully observed. Short and gappy series therefore skew the draws toward short horizons. `exp/combined` weights the horizon draw and drops the finite-future check for teacher labels. The runs below predate that fix.

## Runs

All three runs are the [wide replay](wide-replay.md) mixture plus the `real` task at weight 0.4, a 22% share of the draws. Batch 16, peak lr 3e-5, seed 0.

| run | real labels | steps | training run |
|---|---|---:|---|
| `real-truth` | the true future | 6000 | [`koy0p281`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/koy0p281) |
| `real-teacher-209m` | frozen Timestral-209M | 6000 | [`cntqp9fi`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/cntqp9fi) |
| `real-truth-s1500` | the true future | 1500 | [`osu9c9qq`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/osu9c9qq) |

## Benchmarks

Changes are in the error against pretrained Timestral-5M, so a negative number is a lower error. TIME is all 98 tasks: [`669oa747`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/669oa747), [`ou4h3omq`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/ou4h3omq), [`k2npd3ts`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/k2npd3ts). fev is the 26-task covariate subset: [`9nj3xk1h`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/9nj3xk1h), [`no52m3v2`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/no52m3v2), [`rijzg3e9`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/rijzg3e9). The last row is the short post-train from [PR #7](https://github.com/tfiedlerdev/time-series-models/pull/7), without real series, for comparison at 1500 steps.

| run | TIME MASE | TIME tasks lower | TIME tasks >+1% | fev SQL | fev SQL median | fev tasks >+1% |
|---|---:|---:|---:|---:|---:|---:|
| wide replay | +0.10% | 31 of 98 | 2 | +2.15% | +1.06% | 14 of 26 |
| real, truth | +0.09% | 40 | 4 | +1.83% | +0.85% | 11 |
| real, 209M labels | +0.14% | 33 | 6 | +1.73% | +0.47% | 11 |
| real, truth, 1500 steps | +0.09% | 39 | 4 | +0.73% | +0.03% | 8 |
| teacher replay, 1500 steps, no real share | +0.06% | 34 | 4 | +0.83% | +0.13% | 7 |

## The training eval

32 series, the same draws for every model, from the `exp/combined` code. W&B runs [`e7kivcbo`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/e7kivcbo) (pretrained, wide replay, real truth), [`ek85zo39`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/ek85zo39) (real 209M) and [`n1zoxan2`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/n1zoxan2) (real truth, 1500 steps). The pretrained column is the same in all three.

| task | pretrained | wide replay | real, truth | real, 209M | real, truth, 1500 |
|---|---:|---:|---:|---:|---:|
| real/heldout | 0.502 | 0.507 | 0.510 | 0.495 | 0.504 |
| realcov/heldout | 0.341 | 0.327 | 0.325 | 0.327 | 0.330 |
| pool/64 | 0.380 | 0.270 | 0.266 | 0.269 | 0.295 |
| views/64 | 0.351 | 0.261 | 0.263 | 0.265 | 0.293 |
| exogenous/decoy | 0.625 | 0.589 | 0.585 | 0.585 | 0.611 |
| generic/kernel | 0.219 | 0.210 | 0.201 | 0.219 | 0.179 |
| long/walk | 1.670 | 1.771 | 1.723 | 1.732 | 1.685 |

## What follows

Real series do not move the benchmarks much. With true labels, TIME holds and the fev cost falls from +2.2% to +1.8%. Labels from the 209M teacher give about the same. At 1500 steps the real share is at +0.73% against +0.83% without it, a gap of a tenth of a point. The held-out real series do not improve either. The student trained on the true futures is at 0.510 against 0.502 pretrained. The 209M labels bring it to 0.495, about 1.4% below pretrained, and the only real-data run below the pretrained model there.

The truth-labelled runs are below the pretrained error on more TIME tasks than any synthetic-only mixture: 40 and 39 of 98, against 29 to 34. The mean does not move because about as much is lost on the others.

The fev cost sits on covariate tasks, and a univariate real series does not carry that structure. `exp/combined` takes the multivariate datasets and turns the other variates into covariates for the target. That is the next lever, ahead of a larger real share.
