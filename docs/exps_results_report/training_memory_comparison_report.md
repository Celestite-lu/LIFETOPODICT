# HC-SOINN vs LifeTopoDict Training Memory Comparison

Date: 2026-05-22

## Scope

This report measures CPU memory for classifier-only feature-cache runs. The backbone is not initialized; cached ViT-B/16-IN1K features are read from disk and only the classifier is trained.

Primary metric: process-tree PSS. RSS is also reported but is not the main metric because HC-SOINN uses multiprocessing and RSS double-counts shared forked pages. PSS divides shared pages proportionally and is closer to real physical memory pressure.

Dataset/protocol: CUB-200, 20 tasks, 10 base classes, 10 classes per increment, 3 seeds.

## Train-only Memory

These runs use `--training_time_profile_last_task true --skip_eval_for_training_time_profile true`, so per-task evaluation is skipped.

| Run | Peak PSS | Mean PSS | Peak USS | Mean USS | Peak RSS | Mean RSS | Peak process count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 812.48 MiB | 514.27 MiB | 810.56 MiB | 474.34 MiB | 7545.93 MiB | 2033.69 MiB | 81 |
| LifeTopoDict selected | 782.47 MiB | 515.26 MiB | 780.59 MiB | 471.09 MiB | 9276.43 MiB | 1898.30 MiB | 81 |

LifeTopoDict vs raw:

- Peak PSS: -30.01 MiB, -3.69%.
- Mean PSS: +0.99 MiB, +0.19%.
- Peak USS: -29.97 MiB, -3.70%.
- Mean USS: -3.26 MiB, -0.69%.
- Peak RSS: +1730.50 MiB, +22.93%, but RSS is inflated by shared pages in forked workers and should not be treated as physical memory usage.

## Full Train+Eval Memory

For reference, the normal classifier-only experiment loop includes evaluation after each task.

| Run | Peak PSS | Mean PSS | Peak USS | Mean USS | Peak RSS | Mean RSS | Peak process count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 883.20 MiB | 503.46 MiB | 881.29 MiB | 416.14 MiB | 8072.28 MiB | 2021.47 MiB | 81 |
| LifeTopoDict selected | 936.24 MiB | 533.87 MiB | 839.80 MiB | 446.72 MiB | 12564.64 MiB | 1949.17 MiB | 81 |

LifeTopoDict vs raw:

- Peak PSS: +53.04 MiB, +6.01%.
- Mean PSS: +30.41 MiB, +6.04%.
- Peak USS: -41.49 MiB, -4.71%.
- Mean USS: +30.58 MiB, +7.35%.

## Classifier-state Memory

Train-only final state reported by the classifier:

| Run | Compact deployable | Actual implementation |
|---|---:|---:|
| Raw HC-SOINN | 7.6598 MB | 15.2008 MB |
| LifeTopoDict selected | 3.1826 MB | 11.3271 MB |

LifeTopoDict reductions:

- Compact deployable memory: -58.45%.
- Actual implementation memory: -25.48%.

## Interpretation

LifeTopoDict reduces classifier-state memory clearly, but the end-to-end training RSS/PSS is dominated by runtime overhead: Python, PyTorch import state, NumPy/SciPy, feature-cache loaders, temporary clustering arrays, and multiprocessing workers. Therefore state compression only weakly affects measured training-time process memory.

For paper claims, the strongest memory evidence remains compact deployable memory and audited classifier implementation memory. Training-time CPU memory should be reported as an engineering/system metric: LifeTopoDict is roughly neutral in train-only PSS on CUB-200 and does not create a large RAM regression.

## Artifacts

- `scripts/run_with_cpu_memory_monitor.py`
- `logs/cpu_memory_training_comparison/raw_hc_soinn_cub_train_only_cpu_pss_summary.json`
- `logs/cpu_memory_training_comparison/ltd_selected_cub_train_only_cpu_pss_summary.json`
- `logs/cpu_memory_training_comparison/raw_hc_soinn_cub_classifier_only_cpu_pss_summary.json`
- `logs/cpu_memory_training_comparison/ltd_selected_cub_classifier_only_cpu_pss_summary.json`
