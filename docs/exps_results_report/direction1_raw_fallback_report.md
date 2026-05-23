# Direction 1 Raw-Node Fallback Gate 研究报告

日期：2026-05-22  
范围：CUB-200 same-feature classifier-only，seed 1993，cached ViT-B/16 features，growth 与 additive edge-aware scoring 固定关闭。

## 1. 结论

Direction 1 的工程功能已经完成并可运行：raw fallback cache、prediction trace、memory ledger、ridge learned gate、held-out calibration split、跨任务累积 calibration、正例/增益不足保护，以及 static/random/oracle controls 均已接入。

但首轮 CUB seed 1993 结果显示：**oracle 上界强，deployable learned ridge gate 尚未有效**。

- `oracle count-r3`：A_Last 86.62，相对 compact 82.46 提升 +4.16pp，证明 compact/raw 有明显互补性。
- `learned ridge count-r3 val5 cumulative`：A_Last 82.40，低于 compact baseline 82.46，未达成 selected-relative +0.2pp 的最低成功标准。
- `static` 与 `random` 同预算对照均有害，说明收益不是简单来自额外 raw cache，而必须依赖可靠 gate。

当前应把 Direction 1 定义为：**上界成立、可部署 gate 未成立**。后续若继续，需要改 gate 特征和校准目标，而不是扩大 fallback budget。

## 2. 实现改动

主要代码：

- `utils/hc_soinn_classifier.py`
  - 新增 raw fallback cache 与推理 cache。
  - 新增 prediction trace 字段：compact margin、selected/fallback distance、node residual、count、node state、compact/fallback correctness。
  - 新增 learned ridge/logistic gate 拟合接口 `fit_raw_fallback_gate_from_trace()`。
  - 新增 gate model memory accounting，加入 compact/actual memory breakdown。
  - 新增 gate 保护：`fallback_gate_min_positives`、`fallback_gate_min_calibration_gain`。

- `models/life_topo_dict.py`
  - 新增 cached-feature train/calibration split。
  - learned gate 使用 cumulative held-out calibration set，不使用 test 调参。
  - 训练时序为：提取 cached train features -> compress/materialize/cache fallback -> calibration trace -> fit gate。

- `utils/feature_cache.py`
  - 新增 `get_cached_feature_dataset_split()`。
  - `ConcatDataset` 可被识别为 cached-feature loader，支持 cumulative calibration。

- `main.py`
  - 新增 fallback calibration / gate 保护 CLI 覆盖参数。

- `scripts/collect_results.py`
  - 新增 raw fallback 与 gate fit 统计解析。
  - 修正 raw fallback 行解析，避免被下一行 gate fit 覆盖同名字段。

配置：

- `exps/life_topo_dict/life_topo_dict_cub_fallback_ridge_count_r3.json`
- `exps/life_topo_dict/life_topo_dict_cub_fallback_static_count_r3.json`
- `exps/life_topo_dict/life_topo_dict_cub_fallback_random_count_r3.json`
- `exps/life_topo_dict/life_topo_dict_cub_fallback_oracle_count_r3.json`

所有新增配置均保持：

```text
use_dictionary_growth=false
dict_max_growth_per_task=0
use_edge_aware_scoring=false
use_feature_cache=true
feature_cache_skip_backbone=true
feature_cache_classifier_device=cpu
```

## 3. 验证与实验

语法检查通过：

```bash
python -m py_compile main.py models/life_topo_dict.py utils/hc_soinn_classifier.py utils/feature_cache.py scripts/collect_results.py
```

Smoke test：

```bash
python main.py --config exps/life_topo_dict/life_topo_dict_cub_fallback_ridge_count_r3.json --device 0 --seed 1993 --max_tasks 2
```

结果：通过。第 2 个任务 cumulative calibration samples=100，classifier-only/cache 路径正常；正例不足时 gate 自动关闭。

完整实验日志：

| Method | Log |
|---|---|
| learned ridge count-r3 val5 cumulative | `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_ridge_count_r3_val5_1993_pretrained_vit_b16_224_20260522_202928.log` |
| static count-r3 | `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_static_count_r3_1993_pretrained_vit_b16_224_20260522_203044.log` |
| random count-r3 | `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_random_count_r3_1993_pretrained_vit_b16_224_20260522_203125.log` |
| oracle count-r3 | `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_oracle_count_r3_1993_pretrained_vit_b16_224_20260522_203207.log` |

CSV 汇总：

`logs/direction1_raw_fallback_count_r3_results.csv`

## 4. 结果表

参考 same-feature baseline：

| Method | A_Avg | A_Last | Compact MB | Actual MB |
|---|---:|---:|---:|---:|
| Raw HC-SOINN mean, seeds 1993/1994/1995 | 88.44 | 82.91 | 7.6598 | 22.7962 |
| LifeTopoDict selected, seed 1993 | 88.9185 | 82.46 | 3.1104 | 17.6782 |

Direction 1 count-r3 单 seed：

| Method | A_Avg | A_Last | Compact MB | Actual MB | Fallback rate | Compact acc | Final acc | Oracle improvable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned ridge val5 cumulative | 88.7415 | 82.40 | 4.5857 | 19.0059 | 0.12% | 82.41 | 82.40 | 3.97% |
| static | 81.8230 | 71.45 | 4.8882 | 21.2368 | 100.00% | 82.46 | 71.45 | 4.16% |
| random | 88.0960 | 81.36 | 4.8882 | 21.2368 | 10.20% | 82.46 | 81.36 | 4.16% |
| oracle | 91.5190 | 86.62 | 4.8882 | 21.2368 | 4.16% | 82.46 | 86.62 | 4.16% |

## 5. 分析

1. **raw fallback cache 有可用信息。** Oracle 只在 4.16% 样本上切换，就把 final acc 从 82.46 提到 86.62。这说明 compact topology 的错误不是完全不可修复，且 count-r3 raw cache 覆盖了相当一部分 compact 错误。

2. **raw fallback 本身更弱。** Static fallback 的 fallback_acc 只有 71.45，远低于 compact 82.46；random 以 10% rate 切换也降到 81.36。因此不能做 static fusion 或高比例随机 fallback。

3. **learned ridge gate 太保守且泛化不足。** 最后一轮 calibration 有 1000 个样本、37 个正例，calibration gain=+0.20pp，但 test fallback rate 只有 0.12%，final acc 82.40，略低于 compact_acc 82.41。当前 reliability features/linear threshold 不能稳定识别 oracle 可修复样本。

4. **memory tradeoff 仍需谨慎。** count-r3 fallback cache 增加约 1.7778 MB compact memory。相对 raw HC-SOINN compact 7.6598 MB，oracle/controls 的 compact 4.8882 MB 仍降低约 36.2%；learned ridge 因 calibration holdout 导致节点略少，compact 4.5857 MB，降低约 40.1%。Actual memory 仍低于此前正式 raw baseline 的 22.7962 MB，但 fallback cache 明显吃掉了 selected compact 的一部分压缩收益，因此 Direction 1 的强支撑仍应放在 compact memory 与 accuracy tradeoff，而不是 actual memory。

## 6. 判定

Direction 1 首轮最低成功标准未通过：

| 标准 | 结果 |
|---|---|
| selected-relative A_Last +0.2pp | 未通过，learned ridge 82.40 vs selected 82.46 |
| raw-relative A_Last 追平 raw HC-SOINN | 未通过，82.40 vs 82.91 |
| A_Avg 下降不超过 0.1pp | 未通过，88.7415 vs selected 88.9185，下降 0.177pp |
| reliability gate 优于 random/static | A_Last 优于 random/static，但不是正收益 |
| oracle upper bound | 通过，86.62 显著高于 compact |

建议：保留 Direction 1 的 trace/cache/memory ledger 作为研究基础，但不要在论文中 claim learned fallback gate 已成功。下一步优先做 risk feature 诊断和 gate objective 改进，或者转向 Direction 2 atom-conflict gate。
