# Agent notes

## Reporting error changes

A change in an error keeps the sign of the error. MASE, quantile loss, and MAE are smaller when the forecast is better, so a negative change means the error fell. Write MASE −1.05% for a fine-tune whose MASE is 1.05% below the pretrained model's, and MASE +4.34% when the error rose by that much.

The relative-improvement form `(baseline − current) / |baseline|` flips this sign. Leave it out of tables and write-ups.

## Research ideas and findings

An idea for a later experiment goes in the README under **Research ideas**. Leave it there until it has been run.

Once it has been investigated, remove it from the README and write a report in `findings/`. The report should say what was measured, what the result was, and what follows from it. Include the figure when there is one. `findings/view-curve.md` is the pattern.
