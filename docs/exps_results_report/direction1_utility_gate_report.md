# Direction 1 Utility Gate Report

Scope: CUB-200, seed 1993, cached frozen ViT-B/16 features, classifier-only CPU path. Growth and additive edge-aware scoring remain disabled. Thresholds/rates are selected on calibration only; test labels are used here only for post-hoc evaluation.

## Executive conclusion

The utility-learning direction is partially validated as a diagnostic but not strong enough for a main paper claim. The best deployable variant is `utility + pair/node shrinkage`, with A_Last 82.48 versus the same-run compact baseline 82.41, a 0.07pp gain. Its test net gain is 0.07pp, but A_Avg drops 0.36pp from the compact selected baseline. This misses the minimum success rule of selected-relative A_Last >= +0.2pp and A_Avg drop <= 0.1pp.

The core failure mode is still calibration-to-test transfer: calibration precision is high, but test benefit/harm separation weakens sharply. The result should be kept as a negative/diagnostic Direction 1 outcome, while the oracle count-r3 upper bound remains useful evidence that compact errors are recoverable in principle.

## Trace Fields

The current trace includes the required compact/fallback correctness fields plus the new deployable signals:

- raw-side confidence: `raw_top1_score`, `raw_top2_score`, `raw_margin`, `raw_distance_to_top1`, `raw_distance_to_top2`
- cross-rank/disagreement: `raw_top1_class`, `compact_rank_of_raw_top1`, `raw_rank_of_compact_top1`, `compact_score_on_raw_top1`, `raw_score_on_compact_top1`, `score_cross_gap`
- node/class context: `selected_class`, `selected_node_id` via `nearest_compact_node`, `task_relation`, `selected_class_age`, `raw_class_age`, `fallback_available`

## Four-Quadrant Statistics

| Split | N | compact correct / raw correct | compact wrong / raw correct | compact correct / raw wrong | compact wrong / raw wrong | compact acc | raw fallback acc | oracle gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | 1000 | 665 (66.50%) | 37 (3.70%) | 137 (13.70%) | 161 (16.10%) | 80.20% | 70.20% | 3.70pp |
| test | 5794 | 3882 (67.00%) | 230 (3.97%) | 893 (15.41%) | 789 (13.62%) | 82.41% | 70.97% | 3.97pp |

Interpretation: benefit is much smaller than harm on both calibration and test. Any deployable gate must therefore optimize net utility, not raw correctness.

## Candidate Rule Search

| Rule | Candidates | Candidate rate | Benefit within | Harm within | Overall net gain |
| --- | --- | --- | --- | --- | --- |
| all_available | 5794 / 5794 | 100.00% | 3.97% | 15.41% | -11.44pp |
| disagree | 1443 / 5794 | 24.91% | 15.94% | 61.88% | -11.44pp |
| margin_raw_q30_q70 | 21 / 5794 | 0.36% | 47.62% | 38.10% | 0.03pp |
| rank_rule | 1103 / 5794 | 19.04% | 19.22% | 63.10% | -8.35pp |
| residual_cross_gap | 437 / 5794 | 7.54% | 24.03% | 48.97% | -1.88pp |
| enhanced | 370 / 5794 | 6.39% | 27.57% | 50.00% | -1.43pp |

The simple `disagree` pool raises benefit density relative to all samples but still has negative net gain. The stricter fixed rules reduce candidate count but do not produce a reliable positive-net candidate pool on test.

## Reference Controls

| Method | A_Avg | A_Last | fallback rate | compact acc | final acc | test net | oracle-improvable | compact MB | actual MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static fallback count-r3 | 81.8230 | 71.45 | 100.00% | 82.46% | 71.45% | -11.01pp | 4.16% | 4.8882 | 21.2368 |
| random fallback 10% control | 88.0960 | 81.36 | 10.20% | 82.46% | 81.36% | -1.10pp | 4.16% | 4.8882 | 21.2368 |
| random fallback 1% control | 88.8600 | 82.41 | 0.85% | 82.46% | 82.41% | -0.05pp | 4.16% | 4.8882 | 21.2368 |
| random fallback 2% control | 88.7885 | 82.29 | 1.90% | 82.46% | 82.29% | -0.17pp | 4.16% | 4.8882 | 21.2368 |
| current ridge gate | 88.7415 | 82.40 | 0.12% | 82.41% | 82.40% | -0.01pp | 3.97% | 4.5857 | 19.0059 |
| GBDT diagnostic max1 | 88.7830 | 82.45 | 0.52% | 82.41% | 82.45% | 0.04pp | 3.97% | 4.7011 | 19.1213 |
| GBDT diagnostic max2 | 88.7820 | 82.53 | 0.98% | 82.41% | 82.53% | 0.12pp | 3.97% | 4.7011 | 19.1213 |
| oracle count-r3 | 91.5190 | 86.62 | 4.16% | 82.46% | 86.62% | 4.16pp | 4.16% | 4.8882 | 21.2368 |

Notes: the 1% and 2% random controls are extra low-budget sanity checks run from the non-calibration random config; they are useful directionally but not a strict same-topology comparator for the learned utility gates. The strict utility-gate comparator remains the candidate-only compact baseline in the next table.

## Utility Gate Results

| Method | A_Avg | A_Last | selected-rel A_Last | selected-rel A_Avg | raw-rel A_Last | Old-New HM | test fallback | benefit sel | harm sel | test net | calib fallback | calib net | compact MB | actual MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| candidate-only rule gate | 88.7880 | 82.41 | 0.00pp | 0.00pp | -0.50pp | 80.62 | 0.00% | 0 | 0 | 0.00pp | 0.00% | 0.00pp | 4.5860 | 19.0062 |
| delta regression ridge | 88.3170 | 81.77 | -0.64pp | -0.47pp | -1.14pp | 79.79 | 2.97% | 41 | 78 | -0.64pp | 4.00% | 0.90pp | 4.5860 | 19.0062 |
| benefit-vs-harm logistic | 88.4305 | 82.33 | -0.08pp | -0.36pp | -0.58pp | 80.07 | 1.35% | 28 | 33 | -0.09pp | 2.00% | 0.80pp | 4.5870 | 19.0071 |
| utility + pair/node shrinkage | 88.4245 | 82.48 | 0.07pp | -0.36pp | -0.43pp | 80.48 | 0.81% | 21 | 17 | 0.07pp | 2.00% | 1.10pp | 4.5965 | 19.0166 |
| class_pair_only | 88.4390 | 82.40 | -0.01pp | -0.35pp | -0.51pp | 80.44 | 0.86% | 19 | 20 | -0.02pp | 2.00% | 1.00pp | 4.5965 | 19.0166 |
| node_pair_only | 88.2740 | 81.83 | -0.58pp | -0.51pp | -1.08pp | 79.82 | 3.52% | 52 | 86 | -0.59pp | 4.00% | 1.10pp | 4.5965 | 19.0166 |

Selected-relative uses the same calibration setting compact baseline: A_Avg 88.7880, A_Last 82.41. Raw-relative uses same-feature raw HC-SOINN A_Last 82.91.

## Pair/Node Shrinkage Ablation

| Variant | A_Last | selected-rel A_Last | test net | fallback rate | benefit sel | harm sel | utility precision | gate MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| delta regression ridge | 81.77 | -0.64pp | -0.64pp | 2.97% | 41 | 78 | 34.45% | 0.0004 |
| class_pair_only | 82.40 | -0.01pp | -0.02pp | 0.86% | 19 | 20 | 48.72% | 0.0109 |
| node_pair_only | 81.83 | -0.58pp | -0.59pp | 3.52% | 52 | 86 | 37.68% | 0.0109 |
| utility + pair/node shrinkage | 82.48 | 0.07pp | 0.07pp | 0.81% | 21 | 17 | 55.26% | 0.0109 |

`full` is the only tested utility-table variant with positive test net gain. `class_pair_only` is nearly neutral, while `node_pair_only` over-triggers and harms accuracy.

## Memory Ledger Audit

| Method | compact MB | actual MB | raw fallback MB | gate/table MB | compact reduction vs raw | actual reduction vs raw |
| --- | --- | --- | --- | --- | --- | --- |
| candidate-only rule gate | 4.5860 | 19.0062 | 1.7778 | 0.0004 | 40.13% | 16.63% |
| utility + pair/node shrinkage | 4.5965 | 19.0166 | 1.7778 | 0.0109 | 39.99% | 16.58% |

Raw fallback cache is counted in compact deployable memory. Even with count-r3 fallback and the pair/node table, compact memory remains below raw HC-SOINN by 39.99%.

## Success Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| net_gain > 0 | partial pass | full pair/node has test net 0.07pp; most other deployable gates are <= 0 |
| selected-relative A_Last >= +0.2pp | fail | best is 0.07pp |
| A_Avg drop <= 0.1pp | fail | full pair/node A_Avg delta is -0.36pp |
| raw-relative A_Last >= 0 | fail | full pair/node is -0.43pp vs same-feature raw HC-SOINN |
| compact memory reduction vs raw >= 30% | pass | 39.99% reduction |

## Decision

Do not promote the deployable reliability gate to the paper's main claim yet. The correct claim is narrower: oracle raw-node fallback proves recoverable compact-topology errors, and the utility/pair objective gives the first positive deployable net-gain point, but its effect is too small and too unstable for the planned success criteria. The next branch should be Direction 2 atom-conflict gate, or a deeper calibration distribution-shift study before spending on 3-seed Direction 1 runs.

## Artifacts

- `logs/direction1_utility_runs/utility_gate_results_raw.csv`
- `logs/direction1_utility_runs/utility_gate_results_derived.csv`
- `logs/direction1_utility_runs/four_quadrant_stats.csv`
- `logs/direction1_utility_runs/candidate_rule_search.csv`
- `logs/direction1_utility_runs/pair_node_ablation.csv`
- `logs/direction1_utility_runs/reference_controls_raw.csv`
- `logs/direction1_utility_runs/random_low_budget_controls_stdout.csv`
