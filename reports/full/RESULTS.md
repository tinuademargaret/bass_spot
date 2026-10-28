# Experiment results

Errors are relative log-response units (dB), not calibrated SPL.

| Model | Seed | Split | MAE | Constant MAE | Wall MAE | Placement regret |
|---|---:|---|---:|---:|---:|---:|
| baseline | 11 | test | 5.039 | 6.706 | 5.059 | 1.220 |
| baseline | 11 | ood | 6.278 | 7.316 | 6.170 | 1.078 |
| baseline | 22 | test | 5.072 | 6.706 | 5.044 | 1.537 |
| baseline | 22 | ood | 6.383 | 7.316 | 6.241 | 1.107 |
| baseline | 33 | test | 5.231 | 6.706 | 5.263 | 1.213 |
| baseline | 33 | ood | 6.610 | 7.316 | 6.455 | 1.161 |
| boundary | 11 | test | 5.154 | 6.706 | 5.209 | 1.248 |
| boundary | 11 | ood | 6.496 | 7.316 | 6.386 | 1.100 |
| boundary | 22 | test | 5.104 | 6.706 | 5.133 | 1.067 |
| boundary | 22 | ood | 6.500 | 7.316 | 6.383 | 1.227 |
| boundary | 33 | test | 5.188 | 6.706 | 5.225 | 1.414 |
| boundary | 33 | ood | 6.693 | 7.316 | 6.532 | 1.064 |

test: paired boundary − baseline MAE = 0.035; room-bootstrap 95% CI [-0.022, 0.090]. Negative values favour boundary distance. CI averages over the available seeds; per-seed differences are recorded separately.

ood: paired boundary − baseline MAE = 0.139; room-bootstrap 95% CI [0.038, 0.241]. Negative values favour boundary distance. CI averages over the available seeds; per-seed differences are recorded separately.

Placement regret compares the recommendation with the best of the same finite candidate set.
Timing includes preprocessing; solver timings separate cold assembly/factorisation and warm reuse.
Training and data generation costs are separate from inference. Inspect metrics.json for details.
