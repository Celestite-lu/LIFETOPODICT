# LifeTopoDict MVP 实验研究报告

日期：2026-05-22  
仓库：`/home/lyw/LifeTopoDict`

## 摘要

本轮实验围绕 LifeTopoDict MVP 的核心假设展开：在 HC-SOINN class-local topology 的基础上，用共享字典稀疏编码替代长期 prototype center 存储，并引入 lifecycle 与 edge-aware 机制，以在尽量不损失精度的前提下降低可部署内存。

当前最可靠的结论是：

1. 在 CUB-200 same-feature setting 下，LifeTopoDict selected 配置可以在 A_Last 仅下降 0.38pp 的情况下，将 compact deployable memory 从 7.6598 MB 降至 3.1827 MB，降幅 58.45%。
2. 修复运行时冗余存储后，actual implementation memory 也低于 raw HC-SOINN：17.7632 MB vs 22.7962 MB，降幅 22.08%。
3. lifecycle 在 selected 配置下能实际过滤 stale nodes：filtered inactive nodes 从 no_lifecycle 的 0.0000% 提升到 17.2433%，同时进一步降低 actual memory。
4. dictionary-coding 的内存贡献成立；lifecycle 的 stale suppression 贡献成立；但 growth 和 edge-aware scoring 的独立机制贡献尚未成立。
5. 2026-05-22 决策：residual dictionary growth 与 additive edge-aware scoring 已关闭，后续不再纳入实验设置；固定 `use_dictionary_growth=false`、`dict_max_growth_per_task=0`、`use_edge_aware_scoring=false`。
6. strict feature-cache 路径已改为 classifier-only：默认跳过 backbone 初始化，直接从磁盘读取缓存特征训练/评估 classifier。该工程修复把本进程 GPU 显存增量从 708 MiB 降至 0 MiB，但这是 raw HC-SOINN 与 LifeTopoDict 共享的系统优化，不是 LifeTopoDict 相对 raw 的机制优势。
7. 训练期 CPU 内存复测显示 LifeTopoDict train-only process-tree PSS 与 raw 基本持平：Peak PSS 782.47 MiB vs 812.48 MiB，Mean PSS 515.26 MiB vs 514.27 MiB。训练期进程级内存主要由 Python/NumPy/SciPy、feature loader、临时聚类数组和 multiprocessing worker 主导，不能等同于可部署 classifier state。
8. 因 growth 无可测下游收益、edge-aware default 严重退化，本轮结果不支持宣布“完整三机制 LifeTopoDict MVP 已成立”。更准确的结论是：当前代码支持一个更窄的 MVP 子命题，即 dictionary memory compression + lifecycle stale suppression 在 CUB-200 same-feature setting 下可行。

## 实验范围与公平性说明

本轮最终有效结果集中在 CUB-200：

- 数据集：CUB-200
- 协议：20 tasks，base 10 classes，之后每 task 10 classes
- Backbone：`pretrained_vit_b16_224`
- Feature：使用 frozen feature cache，避免重复 backbone 前向；strict cache 下默认跳过 backbone 初始化，仅训练/评估 classifier
- Classifier device：strict feature-cache 默认 CPU；如需把 `predict_topk()` 距离计算放到 GPU，可显式设置 `--feature_cache_classifier_device cuda`
- Seeds：1993 / 1994 / 1995
- Comparator：same-feature raw HC-SOINN，而不是论文 Table 2 的 CODA-Prompt 数值

重要说明：Yi2026BeyondPN Table 2 的 CUB-200 HC-SOINN A_Last 为 85.75，但本轮 raw HC-SOINN same-feature baseline 为 82.91。由于 backbone/feature setting 与论文 CODA-Prompt 行并非完全相同，本报告把 same-feature raw HC-SOINN 作为实际比较基线。后续若要做论文级复现结论，需要重新对齐 CODA-Prompt setting 或明确报告这一差异。

## 实验配置

### Raw HC-SOINN baseline

来源日志：

- `logs/phase1_baseline_after_memory/cub_raw_hc_soinn_memory.log`
- `logs/phase1_baseline_after_memory/phase1_baseline_after_memory_results.csv`

核心结果：

| Method | A_Avg | A_Last | Old | New | Old-New HM | Compact MB | Actual MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 88.4417 +/- 0.4137 | 82.9100 +/- 0.0000 | 82.7300 | 86.3700 | 84.5108 | 7.6598 | 22.7962 |

### LifeTopoDict selected 配置

来源日志：

- `logs/phase2_lifecycle_selected_fixed/cub_k12_cap10_ridge001_lifecycle_support07_node020_fixed.log`
- `logs/phase2_lifecycle_selected_fixed/phase2_lifecycle_selected_fixed_results.csv`

关键参数：

| 参数 | 值 |
|---|---:|
| `dict_sparse_k` | 12 |
| `dict_ridge_lambda` | 0.01 |
| `dict_max_growth_per_task` | 0（后续固定关闭；历史 selected 曾使用 10） |
| `use_dictionary_growth` | false |
| `lifecycle_theta_support` | 0.7 |
| `lifecycle_protect_old_topk` | false |
| `lifecycle_min_alive_ratio` | 0 |
| `lifecycle_theta_usage` | 0.05 |
| `lifecycle_T_inactive` | 1 |
| `lifecycle_node_inactive_threshold` | 0.20 |
| `use_edge_aware_scoring` | false |

该配置是 CUB tuned candidate，不是原始默认配置。默认 lifecycle 基本没有 inactive-node 行为面，因此不能作为 lifecycle 有效性的证据。

## Phase 2 主结果

| Config | A_Avg | A_Last | Old | New | Old-New HM | Compact MB | Actual MB | Filtered inactive nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 88.4417 +/- 0.4137 | 82.9100 +/- 0.0000 | 82.7300 | 86.3700 | 84.5108 | 7.6598 | 22.7962 | 0.0000% |
| LifeTopoDict selected | 88.4323 +/- 0.4642 | 82.5300 +/- 0.1044 | 82.3633 | 85.8033 | 84.0481 | 3.1827 | 17.7632 | 17.2433% |
| Matched no-lifecycle | 88.5928 +/- 0.4797 | 82.8167 +/- 0.0551 | 82.6300 | 86.3800 | 84.4634 | 3.1671 | 18.9040 | 0.0000% |

### Phase 2 成功标准检查

| 成功标准 | 结果 | 证据 |
|---|---|---|
| CUB 或 ImageNet-R 至少一个数据集 accuracy drop <= 0.5pp | 通过 | CUB A_Last 82.5300 vs raw 82.9100，drop 0.3800pp |
| compact deployable memory 降低 >= 30% | 通过 | 3.1827 MB vs 7.6598 MB，降低 58.45% |
| actual implementation memory 不高于 raw HC-SOINN | 通过 | 17.7632 MB vs 22.7962 MB，降低 22.08% |
| memory-Pareto 至少一个点不被 raw HC-SOINN 支配 | 通过 | selected 点以 0.38pp A_Last 代价换取大幅 memory 降低 |
| lifecycle 至少改善一个 stability/stale 指标 | 通过但较窄 | filtered inactive nodes 17.2433% vs no_lifecycle 0.0000%；Old-New HM 未提升 |

Phase 2 在 same-feature CUB comparator 下通过，但证据范围应限定为“CUB tuned selected 配置”。

## Phase 3 消融结果

来源日志：

- Full selected：`logs/phase2_lifecycle_selected_fixed/cub_k12_cap10_ridge001_lifecycle_support07_node020_fixed.log`
- No lifecycle：`logs/phase2_lifecycle_selected_fixed/cub_k12_cap10_ridge001_no_lifecycle_match_fixed.log`
- No growth：`logs/phase3_selected_ablation_fixed/cub_selected_no_growth_fixed.log`
- No dict：`logs/phase3_selected_ablation_fixed2/cub_selected_no_dict_fixed2.log`
- Edge-aware default：`logs/phase3_selected_ablation_fixed/cub_selected_edge_aware_default_fixed.log`

| Variant | A_Avg | A_Last | Old | New | Old-New HM | Compact MB | Actual MB | Filtered inactive nodes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw HC-SOINN / no_dict | 88.4417 +/- 0.4137 | 82.9100 +/- 0.0000 | 82.7300 | 86.3700 | 84.5108 | 7.6598 | 22.7962 | 0.0000% |
| full selected | 88.4323 +/- 0.4642 | 82.5300 +/- 0.1044 | 82.3633 | 85.8033 | 84.0481 | 3.1827 | 17.7632 | 17.2433% |
| no_lifecycle | 88.5928 +/- 0.4797 | 82.8167 +/- 0.0551 | 82.6300 | 86.3800 | 84.4634 | 3.1671 | 18.9040 | 0.0000% |
| no_growth | 88.4355 +/- 0.4611 | 82.5300 +/- 0.1044 | 82.3633 | 85.8033 | 84.0481 | 3.1106 | 17.6755 | 17.2133% |
| edge_aware default | 82.7378 +/- 1.0588 | 74.1967 +/- 0.2750 | 73.8133 | 81.4867 | 77.4604 | 3.1827 | 17.7632 | 17.2433% |

### 机制贡献判断

| 机制 | 判断 | 依据 |
|---|---|---|
| Dictionary coding | 成立，但主要是内存贡献 | A_Last 仅低于 raw 0.38pp，同时 compact memory 降 58.45%，actual memory 降 22.08% |
| Lifecycle | 部分成立 | stale-node suppression 成立，filtered inactive nodes 达 17.2433%；但 A_Last 和 Old-New HM 低于 no_lifecycle |
| Growth | 未成立 | full selected 比 no_growth 多 6 个 atoms，但 A_Last、Old/New、HM 基本相同，且 memory 更高 |
| Edge-aware scoring | 未成立且当前有害 | default edge-aware A_Last 74.1967，比 full selected 低 8.3333pp |

Phase 3 的关键结论是：完整机制链没有通过。LifeTopoDict 当前最强证据来自 dictionary-coding 和 lifecycle stale suppression，而不是 growth 或 edge。

## Edge / Growth 诊断

为判断 edge 与 growth 是“实现路径错误”还是“默认参数/尺度问题”，额外做了 CUB seed 1993 单 seed 诊断。

来源：

- `logs/diagnostics_edge_growth/diagnostics_edge_growth_results.csv`
- `logs/diagnostics_edge_growth/edge_growth_diagnostic_review.md`

| Variant | A_Avg | A_Last | Old | New | Compact MB | Actual MB | Atoms | Edge adjustment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full selected | 88.9175 | 82.46 | 82.65 | 79.05 | 3.1704 | 17.7529 | 205 | n/a |
| Edge scaled gamma=0.02 eta=0.01 | 88.9020 | 82.17 | 82.34 | 79.05 | 3.1704 | 17.7529 | 205 | +0.0031 |
| Edge support-only gamma=0.02 eta=0.00 | 88.8810 | 82.26 | 82.45 | 78.72 | 3.1704 | 17.7529 | 205 | -0.0028 |
| Growth theta=0.10 cap=10 | 88.9350 | 82.48 | 82.67 | 79.05 | 5.5131 | 20.6580 | 400 | n/a |

诊断结论：

1. edge default 崩溃主要是 score adjustment 的尺度/校准问题。缩小 gamma/eta 后不再灾难性退化，但仍低于 full selected。
2. support-only edge 也没有提升，说明问题不只是 age/risk penalty 的符号。
3. growth 阈值从 0.3 放宽到 0.1 后，atoms 从 205 增至 400，EffRank 明显上升，但 A_Last 只从 82.46 到 82.48，几乎无收益，且 memory 显著增加。

## 新增工程修改与资源占用复测

### Classifier-only feature-cache 路径

用户确认 backbone 输出特征已经缓存后，实验路径已更新为真正的 classifier-only：

- `utils/feature_cache.py` 增加 `feature_cache_classifier_only()` 与 `feature_cache_classifier_torch_device()`。
- `models/simplecil_hc_soinn.py` 和 `models/life_topo_dict.py` 在 strict feature-cache 下默认不构造 `SimpleVitNetKNN`，`self._network=None`。
- 训练阶段直接从 cached train features 构建/更新 HC-SOINN 或 LifeTopoDict classifier。
- 评估阶段直接从 cached test features 调用 `hc_soinn.predict_topk()`。
- `trainer.py` 在 backbone 跳过时报告 `All params: 0` / `Trainable params: 0`。
- `main.py` 新增 `--feature_cache_skip_backbone` 与 `--feature_cache_classifier_device`。前者在 strict cache 下默认 true；后者默认 CPU。
- raw HC-SOINN 的 t-SNE 可视化辅助路径增加缓存特征保护，避免 no-backbone 时触发空指针。

验证日志：

- `logs/gpu_memory_comparison_classifier_only/raw_hc_soinn_cub_classifier_only_gpu2.log`
- `logs/gpu_memory_comparison_classifier_only/ltd_selected_cub_classifier_only_gpu2.log`

两组日志均包含 `Classifier-only mode enabled` 与 `All params: 0`，并以 `EXIT_STATUS=0` 结束。

### GPU 显存复测

来源：

- `logs/gpu_memory_comparison_classifier_only/gpu_memory_classifier_only_report.md`
- `logs/gpu_memory_comparison_classifier_only/*_summary.json`

| Run | Previous strict-cache peak delta | Classifier-only peak delta | Mean GPU delta | 说明 |
|---|---:|---:|---:|---|
| Raw HC-SOINN | 708 MiB | 0 MiB | 0 MiB | backbone 不再初始化 |
| LifeTopoDict selected | 708 MiB | 0 MiB | 0 MiB | backbone 不再初始化 |

解释：之前显存没有下降，是因为 strict feature-cache 虽然避免了重复 backbone 前向，但仍初始化 frozen ViT backbone，GPU 常驻峰值由 PyTorch CUDA context / ViT 权重 / batch tensor 等共享开销主导。现在跳过 backbone 后，两者 GPU 增量都为 0 MiB。因此 GPU 显存下降应作为“缓存特征 + 跳过 backbone 初始化”的工程优化报告，而不是 LifeTopoDict 相对 HC-SOINN 的论文机制强证据。

### 训练期 CPU 内存复测

新增脚本：

- `scripts/run_with_cpu_memory_monitor.py`

该脚本采样 launched process + child process tree 的 RSS/PSS/USS。主指标采用 PSS，因为 HC-SOINN 压缩使用 multiprocessing，RSS 会重复计算 fork worker 共享页。

Train-only 口径使用：

```bash
--training_time_profile_last_task true --skip_eval_for_training_time_profile true
```

来源：

- `logs/cpu_memory_training_comparison/training_memory_comparison_report.md`
- `logs/cpu_memory_training_comparison/raw_hc_soinn_cub_train_only_cpu_pss_summary.json`
- `logs/cpu_memory_training_comparison/ltd_selected_cub_train_only_cpu_pss_summary.json`

| Run | Peak PSS | Mean PSS | Peak USS | Mean USS | Peak process count |
|---|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 812.48 MiB | 514.27 MiB | 810.56 MiB | 474.34 MiB | 81 |
| LifeTopoDict selected | 782.47 MiB | 515.26 MiB | 780.59 MiB | 471.09 MiB | 81 |

LifeTopoDict vs raw：

- Peak PSS：下降 30.01 MiB，约 3.69%。
- Mean PSS：上升 0.99 MiB，约 0.19%，基本持平。
- Peak USS：下降 29.97 MiB，约 3.70%。
- Mean USS：下降 3.26 MiB，约 0.69%。

同一 train-only 口径下的 final classifier state：

| Run | Compact deployable | Actual implementation |
|---|---:|---:|
| Raw HC-SOINN | 7.6598 MB | 15.2008 MB |
| LifeTopoDict selected | 3.1826 MB | 11.3271 MB |

LifeTopoDict state 降幅：

- Compact deployable memory：58.45%。
- Actual implementation memory：25.48%。

解释：训练期进程级内存没有随 classifier state 等比例下降，主要因为 runtime 占用来自 Python/PyTorch import state、NumPy/SciPy、feature-cache loader、临时 clustering arrays 和 multiprocessing workers。论文中应把 compact deployable / actual classifier implementation memory 作为主内存证据；训练期 PSS/RSS 只能作为工程系统指标，结论是 LifeTopoDict 不造成明显训练 RAM 回归。

## 已修复问题

本轮实验中修复了以下工程与诊断问题：

1. Feature cache 已可用于 CUB / CIFAR / ImageNet-R，实验命令使用 `--use_feature_cache true --feature_cache_strict true`，避免重复 backbone 前向。
2. raw HC-SOINN memory logging 已补齐，可报告 compact / actual memory breakdown。
3. LifeTopoDict memory slimming 已完成：默认 `drop_node_raw_after_dict_materialize=True`，materialize 后丢弃冗余 `node.center_raw` / `node.dict_recon_raw`，使 actual memory 低于 raw HC-SOINN。
4. 增加 `lifecycle_node_inactive_threshold`，使 node-level inactive filtering 可被诊断和控制。
5. 增加 lifecycle node summary 日志：protected / plastic / inactive / inactive_ratio。
6. `scripts/collect_results.py` 已支持 inactive node 与 filtered inactive node 指标解析。
7. `compute_GTE()` 无 growth record 时改为返回 0.0，避免日志出现 `nan`。
8. no-dict 场景下 PAD / EffRank / coeff_eff_rank 改为 0.0，避免无字典指标输出 `nan`。
9. `main.py` 增加 `--use_dict_coding` 与 `--seed` CLI 覆盖，便于消融与单 seed 诊断。
10. strict feature-cache 路径默认跳过 backbone 初始化，避免已缓存特征实验仍加载 frozen ViT backbone。
11. raw HC-SOINN 与 LifeTopoDict 均支持 no-backbone cached-feature 训练/评估；`predict_topk()` 默认在 CPU 上执行，必要时可通过 `--feature_cache_classifier_device cuda` 改为 GPU。
12. `trainer.py` 已支持 no-backbone 模型的参数量日志，避免调用 `count_parameters(None)`。
13. 新增 `scripts/run_with_cpu_memory_monitor.py`，支持 process-tree RSS/PSS/USS 采样，用于训练期 CPU 内存对比。
14. `main.py` 增加 `--training_time_profile_last_task` 与 `--skip_eval_for_training_time_profile` CLI 覆盖，便于 train-only 内存/时间复测。

## 风险与局限

1. 本报告有效结论主要来自 CUB-200 same-feature setting，尚未完成 CIFAR-100 / ImageNet-R 的同等流程。
2. 本轮 comparator 是 same-feature raw HC-SOINN，而不是 Yi2026BeyondPN Table 2 的 CODA-Prompt HC-SOINN 数值。
3. selected lifecycle 配置是 tuned candidate，不是默认 tuning-free 配置。跨数据集统一参数是否成立尚未验证。
4. lifecycle 的成功证据是 stale suppression / memory，而不是 Old-New HM 或 A_Last 提升。
5. growth 在 CUB selected 设置下没有独立 downstream 贡献。
6. edge-aware scoring 当前默认配置严重有害，不能作为 MVP 机制贡献。
7. GPU 显存复测证明 backbone 跳过后两种 classifier 都是 0 MiB 增量，因此 GPU memory 不是 LifeTopoDict 相对 raw HC-SOINN 的强机制支撑点。
8. 训练期 CPU PSS 与 USS 基本持平或轻微波动，说明 process-level training memory 被 runtime 与 multiprocessing 主导。不能用训练期 RSS/PSS 来主张 LifeTopoDict 大幅节省训练内存。
9. 由于 Phase 3 机制链未通过，本轮不应进入 Phase 4 大规模参数搜索，也不应宣布完整 LifeTopoDict MVP 成立。

## 结论

当前 LifeTopoDict 研究结果支持以下较窄结论：

> 在 CUB-200 same-feature HC-SOINN setting 下，LifeTopoDict 的 dictionary-coding 能显著降低 compact deployable memory，并在 memory slimming 后降低 actual implementation memory；配合 tuned lifecycle node suppression，可在 A_Last drop <= 0.5pp 的约束内实现 stale-node 过滤。

新增资源复测进一步限定了内存结论：LifeTopoDict 的强证据是 classifier state memory，而不是 GPU 显存或训练期 process-level RAM。strict feature-cache 下跳过 backbone 初始化可以把 raw HC-SOINN 与 LifeTopoDict 的 GPU 增量都降至 0 MiB；train-only CPU PSS 约 0.8 GB 且两者基本持平。

当前结果不支持以下更强结论：

> dictionary-coding、lifecycle、growth、edge 四个机制都已被独立验证，完整 LifeTopoDict MVP 已成立。

原因是：

- growth 没有可测下游收益；
- edge-aware default 导致严重精度崩溃；
- lifecycle 只在 stale suppression 指标上成立，没有改善 Old-New HM；
- 结果尚未扩展到 ImageNet-R 或 CIFAR-100。

## 建议下一步

1. 后续实验固定关闭 growth 与 additive edge-aware scoring；不要再调度 `ablation_no_growth.json` / `ablation_edge_aware.json`，也不要搜索 `dict_theta_residual`、`dict_max_growth_per_task`、`edge_score_gamma`、`edge_score_eta`。
2. 将下一阶段重点转向 `design/LifeTopoDict新方向详细方案.md` 中的 reliability-calibrated fallback gate 与 atom-class conflict gate。
3. 在 ImageNet-R 上复验保留配置（dictionary coding + lifecycle，growth/edge additive 关闭），确认 memory 与 stale suppression 是否跨数据集成立。
4. 如果目标是论文主表，重新建立与 Yi2026BeyondPN Table 2 完全对齐的 CODA-Prompt baseline；如果目标是工程 MVP，则明确采用 same-feature raw HC-SOINN 作为 comparator。
5. 报告内存时拆成三类：deployable classifier state、actual classifier implementation state、system/runtime memory。避免把 GPU 显存或训练期 RSS/PSS 误写成 LifeTopoDict 机制收益。
6. 暂缓 Phase 4 大规模参数搜索，直到新方向的 gate / fallback 机制先在小诊断中显示正贡献。
