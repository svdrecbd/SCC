# Stronger-attacker experiment registry — 2026-09-10

This registry includes all 27 completed training run directories in the stronger-attacker campaign. The [machine audit](../../../artifacts/strong-attack/audit.json) contains full configurations, result hashes, source and parent verification, and compute counters. The [results report](STRONG_ATTACK_RESULTS.md) interprets these exploratory comparisons. Earlier experiments remain in their own registries.

“No escape observed” describes evaluated checkpoints within that particular procedure. It is not a proof or a lower bound. Recovery step counts below exclude the preceding 300 removal updates; a recovery success at step 1,000 costs 1,300 total updates. Defense steps count outer updates; each non-control defense also runs 20,800 inner updates.

| Run directory | Completed updates | First observed outcome | Training timer (s) |
|---|---:|---|---:|
| [strong-defense17-control](../../../runs/strong-defense17-control) | 256 | Clean qualification passed | 8.60 |
| [strong-defense17-escape](../../../runs/strong-defense17-escape) | 256 | Clean qualification passed | 890.41 |
| [strong-defense17-escape_weight10](../../../runs/strong-defense17-escape_weight10) | 256 | Clean qualification passed | 866.88 |
| [strong-defense17-refusal](../../../runs/strong-defense17-refusal) | 256 | Clean qualification passed | 886.53 |
| [strong-matrix17-control/control/benign](../../../runs/strong-matrix17-control/control/benign) | 1,000 | Benign success at 1000 | 52.04 |
| [strong-matrix17-control/control/last_block](../../../runs/strong-matrix17-control/control/last_block) | 1,000 | No escape observed | 35.59 |
| [strong-matrix17-control/control/recovery](../../../runs/strong-matrix17-control/control/recovery) | 1,000 | Escape at 1000 (1,300 total) | 47.48 |
| [strong-matrix17-control/control/removal](../../../runs/strong-matrix17-control/control/removal) | 300 | No escape observed | 14.48 |
| [strong-matrix17-escape/escape/benign](../../../runs/strong-matrix17-escape/escape/benign) | 1,000 | Benign success at 1000 | 55.86 |
| [strong-matrix17-escape/escape/last_block](../../../runs/strong-matrix17-escape/escape/last_block) | 1,000 | No escape observed | 41.93 |
| [strong-matrix17-escape/escape/recovery](../../../runs/strong-matrix17-escape/escape/recovery) | 1,000 | No escape observed | 48.98 |
| [strong-matrix17-escape/escape/removal](../../../runs/strong-matrix17-escape/escape/removal) | 300 | No escape observed | 14.62 |
| [strong-matrix17-escape_weight10/escape_weight10/benign](../../../runs/strong-matrix17-escape_weight10/escape_weight10/benign) | 1,000 | Benign success at 1000 | 30.06 |
| [strong-matrix17-escape_weight10/escape_weight10/last_block](../../../runs/strong-matrix17-escape_weight10/escape_weight10/last_block) | 1,000 | No escape observed | 22.17 |
| [strong-matrix17-escape_weight10/escape_weight10/recovery](../../../runs/strong-matrix17-escape_weight10/escape_weight10/recovery) | 1,000 | No escape observed | 33.52 |
| [strong-matrix17-escape_weight10/escape_weight10/removal](../../../runs/strong-matrix17-escape_weight10/escape_weight10/removal) | 300 | No escape observed | 11.06 |
| [strong-matrix17-refusal/refusal/benign](../../../runs/strong-matrix17-refusal/refusal/benign) | 1,000 | Benign success at 1000 | 54.97 |
| [strong-matrix17-refusal/refusal/last_block](../../../runs/strong-matrix17-refusal/refusal/last_block) | 1,000 | No escape observed | 42.27 |
| [strong-matrix17-refusal/refusal/recovery](../../../runs/strong-matrix17-refusal/refusal/recovery) | 1,000 | Escape at 1000 (1,300 total) | 49.79 |
| [strong-matrix17-refusal/refusal/removal](../../../runs/strong-matrix17-refusal/refusal/removal) | 300 | No escape observed | 14.46 |
| [strong-short-adamw-lr00003](../../../runs/strong-short-adamw-lr00003) | 1,000 | No escape observed | 47.15 |
| [strong-short-adamw-lr0001](../../../runs/strong-short-adamw-lr0001) | 1,000 | No escape observed | 46.64 |
| [strong-short-adamw-lr0003](../../../runs/strong-short-adamw-lr0003) | 1,000 | Escape at 300 | 48.04 |
| [strong-short-defender17-control](../../../runs/strong-short-defender17-control) | 1,000 | Escape at 300 | 66.96 |
| [strong-short-defender17-escape](../../../runs/strong-short-defender17-escape) | 1,000 | Escape at 300 | 37.92 |
| [strong-short-defender17-escape_weight10](../../../runs/strong-short-defender17-escape_weight10) | 1,000 | Escape at 300 | 36.63 |
| [strong-short-defender17-refusal](../../../runs/strong-short-defender17-refusal) | 1,000 | Escape at 300 | 52.81 |

The recorded training timers sum to 3,557.86 seconds. Runs contended for local CPU resources; this sum is neither elapsed project time nor isolated throughput. Calibration, profiling, numerical work, evaluation, and checkpoint I/O are separate. Cloud spend is $0.

## Additional executed calibration and verification

| Artifact | Executed work and outcome |
|---|---|
| [Long attack, seed 10404](../../../artifacts/strong-attack/calibration17/result.json) | 1,300 updates; escape; 38.21 s training timer |
| [Long attack, seed 10405](../../../artifacts/strong-attack/calibration17/result.json) | 1,300 updates; escape; 36.93 s training timer |
| [Long attack, seed 10406](../../../artifacts/strong-attack/calibration17/result.json) | 1,300 updates; escape; 37.19 s training timer |
| [Short attack, seed 10404](../../../artifacts/strong-attack/calibration-short17/result.json) | 300 updates; no escape; 14.09 s training timer |
| [Short attack, seed 10405](../../../artifacts/strong-attack/calibration-short17/result.json) | 300 updates; escape; 12.14 s training timer |
| [Short attack, seed 10406](../../../artifacts/strong-attack/calibration-short17/result.json) | 300 updates; escape; 12.95 s training timer |
| [thread-profile.json](../../../artifacts/strong-attack/thread-profile.json) | CPU profiling: 100 native updates, two repeats each at 1/2/4 threads; four threads retained. Same batches, rounding differences between thread counts. |
| [gradient-check.json](../../../artifacts/strong-attack/gradient-check.json) | Actual-model frozen-displacement directional checks passed; validates the stated first-order surrogate. |
| [tests.txt](../../../artifacts/strong-attack/tests.txt) | 46 passing implementation tests, including native optimizer agreement and exact resumed training. |
| [exact-adam-32.json](../../../artifacts/strong-attack/exact-adam-32.json) | 32-update float64 differentiable-optimizer probe; finite gradient, no escape at this shorter budget. |
| [exact-adam-300.json](../../../artifacts/strong-attack/exact-adam-300.json) | 300-update float64 probe escapes; approximately 4.90 GB peak process RSS and gradient norm 92.35 million. |
| [long-gradient-check/result.json](../../../artifacts/strong-attack/long-gradient-check/result.json) | Original full-trajectory checks at perturbations 1e-10, 1e-11, 1e-12 fail the 1% agreement criterion. Endpoint recomputation matches exactly. Failures preserved. |
| [long-gradient-smaller-steps.json](../../../artifacts/strong-attack/long-gradient-smaller-steps.json) | Continued same saved gradient check: 2.50% relative error at 1e-13 and 0.42% at 1e-14. |
| [long-gradient-representable-directions.json](../../../artifacts/strong-attack/long-gradient-representable-directions.json) | Accounting for actual float64 perturbations gives 0.087% relative error at 1e-14; extreme local conditioning remains. |
| [exact-adam-300-eps1e4.json](../../../artifacts/strong-attack/exact-adam-300-eps1e4.json) | Adam epsilon 1e-4 probe: still escapes; gradient norm increases to about 25 billion. Full finite-difference validation not performed for this variant. |
| [additional-evaluation-initial/result.json](../../../artifacts/strong-attack/additional-evaluation-initial/result.json) | First three arms: 128 new tables, original and reordered, all 12,689 text validation blocks. Clean and benign endpoints and both short-attack endpoints pass; long recovery fails on coupling. |
| [additional-evaluation-high/result.json](../../../artifacts/strong-attack/additional-evaluation-high/result.json) | Weight-10 arm under the same broader evaluation; clean, benign and both short endpoints pass; long recovery fails. |
| [additional-evaluation/result.json](../../../artifacts/strong-attack/additional-evaluation/result.json) | Merge verifies identical evaluation contracts and parent language values, checkpoint hashes, and independent task-prediction rescoring. |
| [benign-predictions/result.json](../../../artifacts/strong-attack/benign-predictions/result.json) | All four arms, both table orders: separately retained and scored uppercase outputs match the broader evaluation. |
| [plots-final/plot-data.json](../../../artifacts/strong-attack/plots-final/plot-data.json) | Plot values for both external attack procedures; final rendered PNG and SVG alongside the data. |
| [audit.json](../../../artifacts/strong-attack/audit.json) | All 27 runs: source/parent/evaluation hash verification, matched ordinary batches and meta streams, calibration and training-query rescoring. |

Small differentiable-Adam tests and source copies are retained under [numerical-sources-final](../../../artifacts/strong-attack/numerical-sources-final) and [exact-adam-source](../../../artifacts/strong-attack/exact-adam-source). The original full-trajectory check also retains its own scripts and gradient checkpoint. The earlier [main audit](../../../artifacts/strong-attack/audit-main.json) and interim plot directories remain available; the final artifacts supplement rather than replace those records.

The final test split was unused throughout. The larger validation checks establish behavior for these saved endpoints; they are not independent model replications or unseen attack-family tests.
