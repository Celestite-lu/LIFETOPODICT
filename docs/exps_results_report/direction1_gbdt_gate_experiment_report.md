# Direction 1 GBDT Raw-Fallback Gate 实现与验证报告

日期：2026-05-22

## 1. 本轮问题

上一轮 trace 诊断报告 `logs/direction1_trace_diagnostics/trace_diagnostic_report.md` 的核心结论是：

- `oracle count-r3` 说明 compact topology 与 raw HC-SOINN node fallback 之间存在明显互补性，CUB final task 上 oracle-positive 约 4.16%。
- 单特征或线性 gate 不能稳定区分 oracle-positive 与 fallback-harm。
- 下一步应实现轻量非线性 deployable gate，优先 GBDT / shallow RF，并在独立 calibration split 上训练，用低 fallback budget 下的 validation final accuracy 选择阈值。

本轮目标是把这个建议落实到代码，并用 CUB-200 cached-feature classifier-only 实验验证。

## 2. 已实现改动

### 2.1 CLI 与配置

新增参数：

- `fallback_gate_max_rate`
- `fallback_gate_tree_estimators`
- `fallback_gate_tree_max_depth`
- `fallback_gate_tree_min_samples_leaf`

`fallback_gate_type` 支持新增：

- `gbdt`
- `random_forest`

相关文件：

- `main.py`
- `models/life_topo_dict.py`
- `utils/hc_soinn_classifier.py`

新增实验配置：

| 配置 | 用途 |
|---|---|
| `exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3.json` | GBDT depth=2，calibration fallback budget ≤ 4% |
| `exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3_max2.json` | GBDT depth=2，calibration fallback budget ≤ 2% |
| `exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3_max1.json` | GBDT depth=2，calibration fallback budget ≤ 1% |
| `exps/life_topo_dict/life_topo_dict_cub_fallback_rf_count_r3.json` | random forest depth=4 对照 |

所有新增配置均保持：

- `use_feature_cache=true`
- `feature_cache_skip_backbone=true`
- `feature_cache_classifier_device=cpu`
- `use_dictionary_growth=false`
- `dict_max_growth_per_task=0`
- `use_edge_aware_scoring=false`
- `raw_fallback_per_class=3`
- `raw_fallback_select_by=count`

### 2.2 Gate 特征

在原有 reliability trace 基础上，GBDT/RF gate 使用以下特征族：

- compact score / margin / nearest distance
- raw fallback score / margin / distance
- compact 与 fallback 的 score/distance gap
- fallback margin ratio
- selected node residual / fallback residual / residual gap
- count log gap
- selected atom inactive/support/usage/entropy 摘要
- selected node lifecycle state one-hot

### 2.3 阈值目标

新增 `_select_raw_fallback_gate_threshold()`：

- 在 calibration trace 上枚举 score threshold。
- 只允许 fallback rate 不超过 `fallback_gate_max_rate`。
- 目标直接最大化 `mean(where(gate, fallback_correct, compact_correct))`。
- accuracy 相同则选择更低 fallback rate，避免无收益过触发。

这和 trace 报告建议一致：阈值不优化 oracle-positive F1，而优化最终 accuracy。

### 2.4 Memory 记账

`compact deployable memory` 已把 learned gate 模型纳入：

- 线性 gate：均值、方差、权重、bias、threshold。
- tree gate：额外按 pickle 后模型体积估算。

GBDT gate 约 0.116 MB；RF gate 约 0.541 MB。

## 3. 验证命令

语法检查：

```bash
python -m py_compile main.py models/life_topo_dict.py utils/hc_soinn_classifier.py
```

JSON/关键开关检查：

```bash
python - <<'PY'
import json
from pathlib import Path
for p in [
    'exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3.json',
    'exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3_max2.json',
    'exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3_max1.json',
    'exps/life_topo_dict/life_topo_dict_cub_fallback_rf_count_r3.json',
]:
    data = json.loads(Path(p).read_text())
    assert data['use_feature_cache'] and data['feature_cache_skip_backbone']
    assert data['feature_cache_classifier_device'] == 'cpu'
    assert data['use_dictionary_growth'] is False
    assert data['dict_max_growth_per_task'] == 0
    assert data['use_edge_aware_scoring'] is False
PY
```

主验证实验：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python main.py --config exps/life_topo_dict/life_topo_dict_cub_fallback_gbdt_count_r3_max1.json --device 0 --seed 1993
```

同样命令分别运行 seed 1994 / 1995。max2 也完成了 3 seed 确认。

## 4. Deployable Gate 结果

### 4.1 GBDT max1：更保守预算，3 seed

| seed | A_Avg | A_Last | compact acc | final acc | delta vs compact | test fallback rate | oracle-improvable | compact MB | actual MB | gate MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1993 | 88.7830 | 82.45 | 82.41 | 82.45 | +0.04pp | 0.52% | 3.97% | 4.7011 | 19.1213 | 0.1155 |
| 1994 | 87.9155 | 81.71 | 81.65 | 81.71 | +0.06pp | 0.28% | 4.57% | 4.6716 | 18.8860 | 0.1177 |
| 1995 | 87.7200 | 82.14 | 82.17 | 82.14 | -0.03pp | 0.31% | 3.90% | 4.6766 | 18.5483 | 0.1161 |
| mean ± std | 88.1395 ± 0.5658 | 82.10 ± 0.37 | 82.08 ± 0.39 | 82.10 ± 0.37 | +0.023 ± 0.047pp | 0.37 ± 0.13% | 4.15 ± 0.37% | 4.6831 ± 0.0158 | 18.8519 ± 0.2880 | 0.1164 ± 0.0011 |

日志：

- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max1_1993_pretrained_vit_b16_224_20260522_234921.log`
- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max1_1994_pretrained_vit_b16_224_20260522_234921.log`
- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max1_1995_pretrained_vit_b16_224_20260522_234921.log`

### 4.2 GBDT max2：2% calibration budget，3 seed

| seed | A_Avg | A_Last | compact acc | final acc | delta vs compact | test fallback rate | oracle-improvable | compact MB | actual MB | gate MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1993 | 88.7820 | 82.53 | 82.41 | 82.53 | +0.12pp | 0.98% | 3.97% | 4.7011 | 19.1213 | 0.1155 |
| 1994 | 87.9095 | 81.76 | 81.65 | 81.76 | +0.11pp | 0.60% | 4.57% | 4.6716 | 18.8860 | 0.1177 |
| 1995 | 87.7075 | 82.03 | 82.17 | 82.03 | -0.14pp | 1.26% | 3.90% | 4.6766 | 18.5483 | 0.1161 |
| mean ± std | 88.1330 ± 0.5711 | 82.11 ± 0.39 | 82.08 ± 0.39 | 82.11 ± 0.39 | +0.030 ± 0.147pp | 0.95 ± 0.33% | 4.15 ± 0.37% | 4.6831 ± 0.0158 | 18.8519 ± 0.2880 | 0.1164 ± 0.0011 |

日志：

- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max2_1993_pretrained_vit_b16_224_20260522_234140.log`
- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max2_1994_pretrained_vit_b16_224_20260522_234140.log`
- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max2_1995_pretrained_vit_b16_224_20260522_234140.log`

### 4.3 其他单 seed 对照

| 方法 | seed | A_Avg | A_Last | compact acc | final acc | delta vs compact | test fallback rate | gate MB | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GBDT max4 | 1993 | 88.7745 | 82.38 | 82.41 | 82.38 | -0.03pp | 2.83% | 0.1155 | 4% budget 过触发 |
| RF max4 | 1993 | 88.6640 | 82.29 | 82.41 | 82.29 | -0.12pp | 1.45% | 0.5413 | 精度更差且 gate 更大 |

相关日志：

- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_gbdt_count_r3_val5_max4_1993_pretrained_vit_b16_224_20260522_233413.log`
- `logs/life_topo_dict/cub/0/10/cub_direction1_fallback_rf_count_r3_val5_max4_1993_pretrained_vit_b16_224_20260522_233640.log`

## 5. 与上一轮 oracle / ridge 结果对照

来自 `logs/direction1_raw_fallback_report.md`：

| 方法 | A_Avg | A_Last | fallback rate | compact acc | final acc | oracle-improvable |
|---|---:|---:|---:|---:|---:|---:|
| learned ridge count-r3 val5 cumulative | 88.7415 | 82.40 | 0.12% | 82.41 | 82.40 | 3.97% |
| static count-r3 | 81.8230 | 71.45 | 100.00% | 82.46 | 71.45 | 4.16% |
| random count-r3 | 88.0960 | 81.36 | 10.20% | 82.46 | 81.36 | 4.16% |
| oracle count-r3 | 91.5190 | 86.62 | 4.16% | 82.46 | 86.62 | 4.16% |

GBDT 相对 ridge 的改进是：不再几乎完全不触发，能在 0.3%-1.0% test fallback rate 下带来很小的正均值。但它还没有接近 oracle 上界。

## 6. 结论

1. **功能实现有效。** GBDT/RF learned raw fallback gate 已接入训练、评估、CLI、配置和 memory ledger；strict feature cache + classifier-only + CPU classifier 路径可完整跑完 CUB-200。

2. **GBDT 是当前最合理的 deployable gate 候选。** RF gate 更大且精度更差；4% budget 过触发；1%-2% calibration budget 更符合 trace 诊断结论。

3. **当前 deployable gate 还不能作为强论文结论。** GBDT max1 的 3 seed 平均提升只有 +0.023pp，max2 为 +0.030pp；二者都没有达到可稳健宣称的 +0.2pp 级别，而且 max2 的 seed 1995 为 -0.14pp。

4. **max1 比 max2 更适合作为当前默认安全设置。** max2 均值略高，但方差明显更大；max1 的 test fallback rate 只有 0.37%，delta std 也更低。若后续必须保留 deployable fallback 分支，建议默认使用 GBDT max1，而不是 max2/max4/RF。

5. **真正瓶颈是 calibration-to-test transfer。** calibration 上每个 seed 都有约 +0.9pp 到 +2.0pp gain，但测试集只剩约 +0.02pp 到 +0.03pp。说明模型能在校准集学到 oracle-positive 形态，但 5 samples/class cumulative calibration 仍不足以稳定估计 fallback-harm 风险。

## 7. 后续建议

下一步不应继续盲目加大模型。更值得做的是 calibration 协议改进：

- 在 calibration 内再拆分 model-fit / threshold-selection，降低阈值过拟合。
- 尝试 leave-task-out 或 recent-task holdout，检查阈值是否对 task distribution shift 敏感。
- 把 objective 改成高精度优先：例如要求 selected fallback precision 在 calibration 上高于 0.55，并同时满足最小 expected gain。
- 报告 gate 的 selected precision/harm rate，而不只报告 fallback rate。
- 若 CUB 上仍无法稳定超过 +0.2pp，应转向 Direction 2 atom-conflict gate，而不是把 raw fallback gate 作为主贡献。

当前可写入论文的谨慎表述是：**oracle raw-node fallback 证明 compact topology 存在可恢复误差；deployable GBDT reliability gate 初步可用但收益很小，尚不能作为主支撑点。**
