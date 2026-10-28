# A target composed of covariates that depend on each other

Whether drawing the covariates from a small causal graph, then hiding some of them, teaches the model to follow whichever causes are actually present.

## Setup

The checked-in mixture, plus a graph task at 10.7% of the draws. Each draw has three to six nodes. A root is a walk, the pooling signal, or a calendar with a walk. Every later node is a nonlinear function of up to two earlier nodes, each delayed by 0 or by 1 to 24 steps, plus noise at 5–100% of its parents' spread. The target is a nonlinear composition of one to three nodes plus a small walk of its own. Then each covariate is hidden one way: removed for the whole series (20%), kept only through the context (25%), masked on one to three stretches of 32–128 context steps and known elsewhere (15%), or left whole. A draw is kept when a ridge forecast from the kept known rows beats the last value by 10%, or when a context-only covariate is a direct cause with a delay of at least 8. Otherwise it is drawn again. The run is 6000 steps at batch 16, seed 0. W&B run [`990aokr8`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/990aokr8), artifact `timestral-5m-pool-mix:graph-w12`.

The eval scores each graph three ways: the covariates as kept, every node known over the whole window, and none. 32 draws, shared with the reference fine-tune and the pretrained model. W&B run [`zwviooqk`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/zwviooqk). The reference is [`06cmyv72`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/06cmyv72).

| task | pretrained | ref-main | graph-w12 |
|---|---:|---:|---:|
| graph/kept | 0.821 | 0.766 | 0.770 |
| graph/all | 0.814 | 0.743 | 0.744 |
| graph/none | 0.883 | 0.853 | 0.820 |
| pool/64 | 0.380 | 0.253 | 0.257 |
| views/64 | 0.351 | 0.258 | 0.260 |
| views/0 | 1.384 | 1.239 | 1.230 |

`graph_headroom.py` scores simple forecasts on the same 32 draws. A ridge fit on the context, from each covariate's value and square, has a median horizon error of 0.657 with every node known and 0.760 with the kept known rows. Last value is 1.201.

## What it shows

The graph task did not change how the model uses covariates. With the covariates as kept, the run is at 0.770 against the reference's 0.766. With every node known it is at 0.744 against 0.743. The checked-in mixture already gets the model to that point.

What moved is the history-only forecast of these targets: 0.853 to 0.820. The covariates are worth less to the graph run than to the reference, 0.050 against 0.087, because its no-covariate forecast caught up.

The model leaves the full graph's information unused. With every node known, a ridge on values and squares is at 0.657, and every model is above 0.74. With only the kept rows, the model is about where that ridge is.

Pooling is unchanged. At 64 views the graph run is at 0.257 against the reference's 0.253.

TIME, all 98 tasks, is W&B run [`r3tfe8pr`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/r3tfe8pr). The fev covariate subset, 26 tasks, is [`f3bkv36d`](https://forge.coreweave.com/wandb/tfiedlerdev/time-series-models/runs/f3bkv36d). Changes are in the error against the pretrained model, so a negative number is a lower error.

| model | TIME MASE | TIME quantile loss | fev SQL, 26 tasks | fev SQL, the 20 tasks ref-main finished |
|---|---:|---:|---:|---:|
| ref-main | +3.5% | +2.8% | | +8.0% |
| graph-w12 | +3.4% | +2.9% | +5.5% | +7.2% |
| distill-teacher-w25 | +0.1% | +0.05% | +2.3% | +3.0% |

The graph run is worse than the pretrained model on 82 of the 98 TIME tasks, the same pattern as the reference.

## What follows

A tenth of the budget on dependent covariates, with some hidden, does not teach the model to use them better than the checked-in exogenous task already does. The gap to a ridge with every node known says there is something left to learn, and this task did not reach it in 6000 steps. The benchmarks carry the reference fine-tune's cost. If the task is tried again, it belongs on top of the teacher replay, with a larger share and a simpler graph first: one parent and one delay, before chains and masks.
