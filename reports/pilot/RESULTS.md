# Experiment results

Errors are relative log-response units (dB), not calibrated SPL.

| Model | Seed | Split | MAE | Constant MAE | Wall MAE | Placement regret |
|---|---:|---|---:|---:|---:|---:|
| baseline | 11 | test | 6.410 | 6.794 | 6.173 | 1.602 |
| baseline | 11 | ood | 7.293 | 7.456 | 7.110 | 1.546 |
| baseline | 22 | test | 6.549 | 6.794 | 6.348 | 1.100 |
| baseline | 22 | ood | 7.043 | 7.456 | 6.952 | 1.238 |
| baseline | 33 | test | 6.365 | 6.794 | 6.236 | 1.454 |
| baseline | 33 | ood | 7.187 | 7.456 | 7.121 | 1.321 |
| boundary | 11 | test | 6.366 | 6.794 | 6.182 | 2.630 |
| boundary | 11 | ood | 7.348 | 7.456 | 7.186 | 1.276 |
| boundary | 22 | test | 6.489 | 6.794 | 6.247 | 1.529 |
| boundary | 22 | ood | 7.184 | 7.456 | 7.084 | 1.600 |
| boundary | 33 | test | 6.297 | 6.794 | 6.078 | 1.551 |
| boundary | 33 | ood | 7.252 | 7.456 | 7.129 | 1.521 |

test: paired boundary − baseline MAE = -0.057; room-bootstrap 95% CI [-0.128, 0.012]. Negative values favour boundary distance. CI averages over the available seeds; per-seed differences are recorded separately.

ood: paired boundary − baseline MAE = 0.087; room-bootstrap 95% CI [-0.026, 0.172]. Negative values favour boundary distance. CI averages over the available seeds; per-seed differences are recorded separately.

Placement regret compares the recommendation with the best of the same finite candidate set.
Timing includes preprocessing; solver timings separate cold assembly/factorisation and warm reuse.
Training and data generation costs are separate from inference. Inspect metrics.json for details.
