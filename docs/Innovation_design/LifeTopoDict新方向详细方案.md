# LifeTopoDict 新方向详细方案

版本：2026-05-22

## 0. 结论摘要

本轮不再把 LifeTopoDict 定位为“完整新拓扑分类器”，而是定位为：

> 面向 HC-SOINN-style topology memory 的压缩、可靠性诊断与选择性修复框架。

已有结果表明，LifeTopoDict selected 在 CUB same-feature setting 下可将 compact memory 从 7.6598 MB 降至 3.1827 MB，降低 58.45%，但 A_Last 从 raw HC-SOINN 的 82.91 降至 82.53，下降 0.38pp。下一步目标不是继续修补 growth 或 additive edge scoring，而是利用 LifeTopoDict 已有的 dictionary-coded node、residual、atom usage、lifecycle state、memory ledger 等信号，寻找能超过 raw HC-SOINN 的准确率增益。

最终保留两个主方向：

1. **Reliability-Calibrated Compact Topology / Raw-Node Fallback Gate**
2. **ATD-Aware Shared-Atom Conflict Detection + Atom-Class Gate**

Risk-Adaptive Budget 只作为支撑组件，用于控制 fallback/repair budget，不作为独立研究方向。

目标承诺应保持克制：在保持约 50% 以上 compact memory reduction 的前提下，争取相对 raw HC-SOINN 达到 A_Last +0.1 到 +0.6pp、A_Avg +0.0 到 +0.3pp、Old-New HM +0.3 到 +0.9pp。

## 1. 共同前提与设计边界

### 1.1 已知实验事实

| Method | A_Avg | A_Last | Old-New HM | Compact MB | Actual MB |
|---|---:|---:|---:|---:|---:|
| Raw HC-SOINN | 88.4417 | 82.9100 | 84.5108 | 7.6598 | 22.7962 |
| LifeTopoDict selected | 88.4323 | 82.5300 | 84.0481 | 3.1827 | 17.7632 |
| Matched no-lifecycle | 88.5928 | 82.8167 | 84.4634 | 3.1671 | 18.9040 |

结论：

- Dictionary coding 的 memory compression 成立。
- Lifecycle hard inactive filtering 能过滤 stale nodes，但没有提升 Old-New HM。
- Growth 扩容无下游收益。
- Additive edge-aware scoring 默认有害，不作为主方向。
- 实验设置决策：从 2026-05-22 起，后续实验固定 `use_dictionary_growth=false`、`dict_max_growth_per_task=0`、`use_edge_aware_scoring=false`；不再调度 no-growth / edge-aware additive scoring 消融，也不搜索 growth/edge additive 参数。

### 1.2 本轮设计边界

本轮首要目标是提高 A_Last / A_Avg / Old-New HM。所有新机制都必须回答一个问题：

> 它如何让 LifeTopoDict 超过 raw HC-SOINN，而不是只让 selected LifeTopoDict 恢复压缩损失？

因此：

- 不把 growth 扩容作为主线。
- 不把 additive edge-aware scoring 作为主线。
- 不把 growth / additive edge-aware scoring 纳入后续实验配置；相关代码仅保留用于历史复现。
- 不把 lifecycle hard pruning 作为主线。
- 不在首轮实验中引入 FC fusion，以免污染 same-feature classifier-only 结论。
- 所有 raw fallback、atom repair、clone、额外缓存都必须进入 memory ledger。
- gate、threshold、budget policy 必须使用 validation split 或固定协议确定，不能用 test set 调参。

## 2. 方向一：Reliability-Calibrated Compact Topology / Raw-Node Fallback Gate

### 2.1 核心研究问题

LifeTopoDict 的 dictionary coding 已经证明可以用较小 memory 近似 raw HC-SOINN node center，但压缩误差不一定均匀分布。关键问题是：

> compact topology 在哪些样本、类别、节点上不可靠？能否只对这些高风险区域启用小额 raw-node fallback，从而在保持大幅 memory reduction 的同时追平或超过 raw HC-SOINN？

这个方向的重点不是“把 raw node 加回来”，而是证明：

1. 压缩损失集中在可识别的高风险区域；
2. LifeTopoDict 的 residual、margin、lifecycle、atom usage 等信号能识别这些区域；
3. targeted fallback 比随机保留 raw nodes 或等内存 raw baseline 更有效。

### 2.2 关键假设

H1：selected LifeTopoDict 相对 raw HC-SOINN 的 0.38pp A_Last 损失，不是均匀发生在所有样本上，而是集中在 low-margin / high-residual / compact-raw disagreement 样本。

H2：node residual、topology margin、nearest selected/raw distance gap、node lifecycle state、atom usage、old support 能预测 compact topology 的错误风险。

H3：在相同额外 memory budget 下，按 reliability 触发 raw fallback 优于随机 fallback、uniform fallback 和 static fusion。

H4：若 oracle fusion 显示 compact 与 raw predictions 具有互补性，则 learned gate 有进一步优化空间；若 oracle 上界也很低，则该方向应及时降级。

### 2.3 方法概述

方向一由三部分组成。

第一，构建 compact topology prediction trace。对每个测试样本记录：

- compact topology top-1 / top-2 class；
- top-1 / top-2 score margin；
- nearest compact node id、class、distance；
- selected node residual；
- selected node lifecycle state；
- selected node 使用的 atom top-k；
- selected node 的 inactive/protected atom ratio；
- compact prediction correctness；
- 若 fallback cache 存在，则记录 nearest raw fallback node、distance、raw fallback prediction correctness。

第二，构建受控 raw fallback cache。由于当前 LifeTopoDict 默认 `drop_node_raw_after_dict_materialize=True`，materialize 后会丢弃 `center_raw` / `dict_recon_raw`，因此不能无成本使用 raw node。需要新增一个明确计入 memory 的 fallback cache。

推荐首轮策略：

- 每类最多保留 `r=1/2/3` 个 raw snapshots；
- 优先保留 high residual nodes；
- 若 validation trace 可用，再加入 low validation margin、old support 高、class confusion risk 高等排序信号；
- cache 内容至少包括 raw center vector、class id、node id、count、residual、node state、source task；
- cache bytes 必须进入 compact deployable memory 与 actual implementation memory。

第三，训练或选择 reliability gate。首轮建议先做低复杂度 gate：

- Rule gate：满足 high residual + low margin 时 fallback；
- Logistic / ridge gate：输入 reliability features，输出 fallback probability；
- Hard switch：当 `g(x)=1` 时使用 raw fallback logits，否则使用 compact topology logits；
- Soft fusion：`score = (1 - g(x)) * score_compact + g(x) * score_raw`。

首轮推荐优先 hard switch 或单参数 soft gate，避免因复杂模型导致 validation overfitting。

### 2.4 Reliability Features

首轮可用或应补齐的特征：

| Feature | 来源 | 作用 |
|---|---|---|
| compact margin | predict logits / top-k scores | 低 margin 表示边界样本 |
| nearest compact distance | compact topology readout | 距离大表示 prototype 覆盖不足 |
| selected node residual | dictionary encoding | 高 residual 表示压缩重构不可靠 |
| compact-raw distance gap | fallback cache | 判断 compact 与 raw 几何是否偏移 |
| node lifecycle state | existing node_states | inactive/protected/plastic 作为可靠性提示 |
| atom usage EMA | existing atom_usage_ema | 低 usage atom 可能不稳定 |
| old support | existing computation | 高 old support 节点对旧类稳定性更敏感 |
| inactive/protected atom ratio | coeff + atom state | 衡量节点由哪些状态 atom 支撑 |
| class age / task id | class id to task mapping | 区分 old/new bias |
| edge reliability | existing metadata | 只作为诊断特征，不做 additive scoring |

### 2.5 实现改动

主要涉及：

- `LifeTopoDict/utils/hc_soinn_classifier.py`
  - 增加 prediction trace 输出接口；
  - 增加 raw fallback cache 数据结构；
  - 增加 fallback logits / fallback prediction；
  - 增加 memory breakdown 中的 fallback bytes；
  - 增加 gate inference 路径。

- `LifeTopoDict/models/life_topo_dict.py`
  - 增加 eval-time trace collection；
  - 增加 validation split calibration；
  - 增加 config / CLI 参数透传。

- 结果收集脚本
  - 增加 fallback rate、fallback correctness、compact/raw disagreement、oracle fusion upper bound；
  - 增加 raw-relative 与 selected-relative gain；
  - 增加 same-memory controls 的 memory ledger。

推荐配置参数：

```text
use_raw_fallback_gate: bool
raw_fallback_per_class: int
raw_fallback_select_by: residual | residual_margin | risk
fallback_gate_type: rule | logistic | ridge | oracle | random
fallback_gate_margin_threshold: float
fallback_gate_residual_threshold: float
fallback_memory_accounting: bool
enable_prediction_trace: bool
trace_split: val | test
```

### 2.6 首轮最小实验

首轮只做 CUB same-feature classifier-only，不引入 FC fusion、不引入 atom repair。

实验组：

| Group | 目的 |
|---|---|
| Raw HC-SOINN | raw baseline |
| LifeTopoDict selected | compact baseline |
| selected + no-raw reliability calibration | 检查 gate 特征本身是否有用 |
| selected + static raw fallback | 固定比例 fallback |
| selected + random raw fallback | 相同 fallback rate 随机触发 |
| selected + same-memory raw fallback | 公平内存对照 |
| selected + learned/rule reliability gate | 主方法 |
| oracle compact/raw fusion | 上界，不参与调参 |

首轮流程：

1. CUB single seed smoke test，确认 trace 与 memory ledger 正确。
2. 扫描 `raw_fallback_per_class = {1, 2, 3}`。
3. 在 validation split 上选择 gate threshold 或 logistic/ridge 参数。
4. 固定 gate 后跑 test。
5. 扩展到 CUB 3 seeds。

### 2.7 成功标准

最低成功标准：

- selected-relative A_Last 至少 +0.2pp；
- raw-relative A_Last 至少追平 raw HC-SOINN；
- A_Avg 下降不超过 0.1pp；
- Old-New HM 不低于 selected；
- compact memory reduction 仍保持 50% 以上，或明确报告 fallback 后的新 reduction；
- reliability gate 优于 random gate 和 static fallback；
- same-memory raw fallback 不能解释全部收益。

理想成功标准：

- selected-relative A_Last +0.4 到 +0.6pp；
- raw-relative A_Last 0 到 +0.2pp；
- Old-New HM +0.2pp 以上；
- oracle fusion 和 learned gate 之间仍有 gap，说明后续还有优化空间。

### 2.8 风险与回退

| 风险 | 判断方式 | 回退 |
|---|---|---|
| high residual / low margin 不能预测错误 | trace 中风险分组错误率无差异 | 方向一降级为 negative diagnostic |
| raw fallback 只是在加 memory | same-memory raw fallback 同样有效 | 论文只保留 memory-Pareto，不 claim reliability gate |
| learned gate 过拟合 | validation 好、test 差 | 改用固定 rule gate 或只报告 oracle upper bound |
| fallback memory 吃掉压缩收益 | compact reduction 低于 30% | 限制 fallback per class 或放弃 raw fallback |
| raw-relative 仍不超过 raw | A_Last 未追平 82.91 | 主 claim 改为 recovery / bounded-loss compression |

### 2.9 实现与首轮验证状态（2026-05-22）

已完成 Direction 1 的可运行实现：

- `utils/hc_soinn_classifier.py` 已加入 raw fallback cache、prediction trace、fallback logits、rule/random/static/oracle/ridge gate 推理、fallback memory ledger、learned gate model memory 统计。
- `models/life_topo_dict.py` 已加入 cached-feature train/calibration split。learned gate 使用 per-class held-out calibration，不使用 test 调参；新增跨任务累积 calibration set，避免每个任务只有 50 个样本导致正例过少。
- learned ridge gate 增加保守保护：若 calibration positives 少于阈值，或校准集上相对 compact 的增益低于阈值，则自动关闭 gate，避免少样本过拟合造成测试崩溃。
- `scripts/collect_results.py` 已能收集 raw fallback 与 raw fallback gate fit 统计。
- 新增 CUB count-r3 实验配置：`life_topo_dict_cub_fallback_ridge_count_r3.json`、`life_topo_dict_cub_fallback_static_count_r3.json`、`life_topo_dict_cub_fallback_random_count_r3.json`、`life_topo_dict_cub_fallback_oracle_count_r3.json`。这些配置继续固定关闭 residual growth 与 additive edge-aware scoring。

CUB-200 same-feature classifier-only、seed 1993 的首轮 count-r3 结果：

| Method | A_Avg | A_Last | Compact MB | Fallback rate | Compact acc | Final acc | Oracle improvable |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected compact baseline | 88.9185 | 82.46 | 3.1104 | 0.00% | 82.46 | 82.46 | - |
| learned ridge, val5 cumulative | 88.7415 | 82.40 | 4.5857 | 0.12% | 82.41 | 82.40 | 3.97% |
| static fallback, count-r3 | 81.8230 | 71.45 | 4.8882 | 100.00% | 82.46 | 71.45 | 4.16% |
| random fallback, count-r3 | 88.0960 | 81.36 | 4.8882 | 10.20% | 82.46 | 81.36 | 4.16% |
| oracle fallback, count-r3 | 91.5190 | 86.62 | 4.8882 | 4.16% | 82.46 | 86.62 | 4.16% |

结论：

- compact/raw complementarity 成立：oracle fallback 从 82.46 提升到 86.62（+4.16pp），说明小额 raw cache 中确实包含可修复的压缩错误。
- 当前可部署 learned ridge gate 未达成成功标准：最终 A_Last 82.40，低于 selected baseline 82.46，也低于 raw HC-SOINN same-feature baseline 82.91。
- static 与 random controls 均有害，说明“加 raw cache”本身不是收益来源；收益需要可靠 gate。
- Direction 1 暂时应定义为“上界成立、可部署 gate 未成立”。后续若继续推进，应优先改进 reliability features / calibration objective，而不是扩大 fallback budget。

## 3. 方向二：ATD-Aware Shared-Atom Conflict Detection + Atom-Class Gate

### 3.1 核心研究问题

Dictionary coding 的优势是跨类共享 atom，从而压缩 topology memory。但跨类共享也可能产生新的 CIL 风险：某些 atom 对旧类有支持，同时又被新类频繁激活，最终成为 old-new confusion 的来源。

核心问题是：

> 能否识别 shared-but-harmful atoms，并通过低风险的 atom-class gate / downweight 改善跨任务判别和 Old-New HM？

这个方向的重点不是进一步压缩，也不是扩容 atom，而是利用 LifeTopoDict 的 atom decomposition 做细粒度错误归因。

### 3.2 关键假设

H1：old-new confusion 中存在 atom-level concentration，即少量共享 atom 与较多错误预测正相关。

H2：有害共享 atom 具有可检测模式：高 old support、高 new usage、跨任务 class-pair confusion 中频繁出现、error PMI 为正。

H3：对这些 atom 做 class-conditional gate/downweight，可以提升 Old-New HM，同时不显著损害 A_Last。

H4：如果 random atom downweight 或 high-usage-only downweight 与 error-PMI gate 效果相同，则 atom conflict 方向不成立，应降级为诊断。

### 3.3 诊断指标

需要新增或系统化记录以下统计。

#### 3.3.1 Atom-Class Binding

对每个 atom `m` 与 class `c`，统计：

```text
support(m, c) = sum over nodes i in class c of |coeff_i,m| * node_count_i
binding(m, c) = support(m, c) / sum_c support(m, c)
```

解释：

- binding 集中在少数类：atom 偏 private；
- binding 分散在多类：atom 偏 shared；
- 若 shared atom 同时参与 old-new confusion，需要进一步判断是否 harmful。

#### 3.3.2 Old Support / New Usage

对当前 task `t`：

```text
old_support(m) = normalized support of atom m on classes before task t
new_usage(m) = normalized support of atom m on current/new classes
```

高 `old_support` + 高 `new_usage` 表示 atom 正在跨任务复用。复用本身不一定有害，需要结合错误统计。

#### 3.3.3 Atom Error PMI

记录样本预测错误时激活的 atom，计算 atom 与错误事件的关联。

```text
PMI(m, error) = log P(m, error) / (P(m) * P(error))
```

更细粒度地，可以对 class-pair error 计算：

```text
PMI(m, y_true -> y_pred)
```

其中 `y_true` 是旧类、`y_pred` 是新类时，尤其关注 old-to-new confusion。

#### 3.3.4 Atom Conflict Score

首轮可以定义一个可解释的 conflict score：

```text
conflict(m) =
    old_support(m)
  * new_usage(m)
  * relu(PMI(m, old_new_error))
  * sharedness(m)
```

其中：

```text
sharedness(m) = entropy(binding(m, :)) / log(num_classes_with_support)
```

也可以做 class-conditional 版本：

```text
conflict(m, c) =
    binding(m, c)
  * relu(PMI(m, errors involving c))
  * cross_task_usage(m)
```

首轮不需要复杂学习器，先用可解释 score 排序。

### 3.4 干预机制

首轮只做低风险干预，不做激进 clone/quarantine。

#### 3.4.1 Atom-Class Gate

为每个 class `c` 和 atom `m` 维护 gate：

```text
g_{m,c} in [0, 1]
```

当计算 class `c` 的 node reconstruction 或 node score 时，对该 class 下的 atom coefficient 加权：

```text
coeff'_{i,m,c} = g_{m,c} * coeff_{i,m}
```

如果 atom `m` 对 class `c` 的 conflict score 高，则降低 `g_{m,c}`。

#### 3.4.2 Inference-Time Downweight

为了降低实现风险，首轮可以不改变 dictionary encoding，只在 inference score 中对受冲突 atom 支撑的 node 做降权。

示意：

```text
node_reliability(i, c) = 1 - mean_m(|coeff_i,m| * conflict(m, c))
score'(x, node_i) = score(x, node_i) * node_reliability(i, c)
```

注意：这不是 additive edge scoring。它是 atom conflict reliability calibration，必须通过 random atom downweight 对照验证。

#### 3.4.3 Class-Conditional Atom Mask

对高冲突 atom，可在特定 class-pair 中禁用或弱化：

```text
if y_candidate in affected_classes(m):
    apply downweight to atom m contribution
```

首轮只建议对 top-K conflict atoms 使用，避免大范围扰动。

### 3.5 实现改动

主要涉及：

- atom activation trace：
  - 对每个 eval sample 记录 selected node 的 coeff top-k；
  - 记录 true class、pred class、old/new 标记、correctness；
  - 聚合为 per-atom counters，不长期保存全部 sample trace。

- atom-class binding table：
  - 由当前 class_clusters 中 node coeff 和 node count 计算；
  - 每个 task 后保存 compact CSV / JSON。

- conflict score：
  - 计算 per-atom old_support、new_usage、sharedness、error PMI；
  - 输出 top conflict atoms 及相关 class pairs。

- atom-class gate：
  - 新增 `atom_class_gate[m, c]` 或稀疏 map；
  - inference 时根据 candidate class 应用；
  - gate metadata 计入 memory ledger。

推荐配置参数：

```text
use_atom_conflict_gate: bool
atom_conflict_topk: int
atom_conflict_metric: pmi | old_new_pmi | conflict_score
atom_gate_strength: float
atom_gate_scope: global | class_conditional | class_pair
atom_gate_calibration_split: val
enable_atom_trace: bool
```

### 3.6 实验设计

方向二不应作为首轮第一实验。建议在方向一完成后进入。

实验组：

| Group | 目的 |
|---|---|
| selected LifeTopoDict | compact baseline |
| selected + atom conflict trace only | 验证冲突是否集中 |
| selected + random atom downweight | 随机对照 |
| selected + high-usage-only downweight | 排除 usage 简单解释 |
| selected + old-support-only downweight | 排除 old support 简单解释 |
| selected + error-PMI atom gate | 主方法 |
| selected + direction-one fallback gate + atom gate | 组合方法 |

必要分析：

- top conflict atoms 覆盖多少 old-new errors；
- conflict score 分桶后的 error rate；
- gated atoms 是否集中在 old-new confusion class pairs；
- HM 改善来自 old accuracy、新 accuracy，还是二者平衡；
- atom gate 是否牺牲新类 plasticity；
- random / high-usage-only 是否无法复现收益。

### 3.7 成功标准

最低成功标准：

- raw-relative Old-New HM +0.2pp 以上；
- A_Last 不下降，或下降不超过 0.1pp；
- error-PMI gate 优于 random atom downweight；
- top conflict atoms 的 error concentration 明显高于平均 atom；
- gate metadata memory 开销可忽略，且计入 ledger。

理想成功标准：

- raw-relative Old-New HM +0.5 到 +0.7pp；
- raw-relative A_Last +0.2 到 +0.4pp；
- 与方向一组合后达到 combined raw-relative A_Last +0.1 到 +0.6pp、HM +0.3 到 +0.9pp；
- per-class breakdown 显示 old-new balance 改善，而不是只提高某几个新类。

### 3.8 风险与回退

| 风险 | 判断方式 | 回退 |
|---|---|---|
| shared atom 不导致错误集中 | conflict score 与 error 无相关 | atom gate 降级为 diagnostic |
| gate 损害新类 | New accuracy 明显下降 | 改为只对 old-new class-pair 生效 |
| random downweight 同样有效 | random control 接近主方法 | 放弃 repair claim |
| trace 体量太大 | eval overhead 过高 | 只保存 aggregated counters |
| 与 fallback gate 冲突 | 组合后低于单独方法 | 两方向分开报告，atom gate 只作为 HM appendix |

## 4. 两个方向的关系

两个方向不是并列堆叠，而是同一 narrative 的两个层次。

| 层次 | 方向 | 解决的问题 |
|---|---|---|
| sample/node reliability | Direction 1 | 哪些 compact topology prediction 不可信，需要 raw fallback |
| atom/class conflict | Direction 2 | 哪些 shared atoms 造成 old-new confusion，需要 class-conditional repair |
| memory budget | Support component | fallback 和 repair 花多少 memory，花在哪里 |

推荐论文主线：

1. LifeTopoDict 用 dictionary coding 压缩 HC-SOINN topology memory。
2. 压缩后不是所有节点都同等可靠，错误集中在高风险样本/节点。
3. 用 reliability gate 分配小额 raw fallback budget，修复 sample-level compression risk。
4. 进一步发现 shared atoms 会诱发 old-new conflict。
5. 用 atom-class gate 修复 representation-level conflict。
6. 所有 fallback/repair 都进入 audited memory ledger，形成 memory-accuracy Pareto。

## 5. 总实验路线

### 5.1 第一阶段：Direction 1 MVP

范围：

- CUB same-feature classifier-only；
- compact topology logits + limited raw-node fallback logits；
- 不加 FC；
- 不加 atom repair；
- 不碰 growth / additive edge scoring。

目标：

- 证明高风险 compact predictions 可被识别；
- 证明 targeted fallback 优于 random/static/same-memory fallback；
- 将 selected A_Last 从 82.53 拉近 raw 82.91，最好追平或小幅超过。

预期时间：

- trace + fallback cache + memory ledger：2 到 3 天；
- CUB single seed smoke：1 天；
- controls + 3 seeds：3 到 5 天。

### 5.2 第二阶段：Direction 2 Diagnostics

范围：

- 在 Direction 1 trace 基础上增加 atom activation / error counters；
- 先不做 gate，只验证 shared atom conflict 是否存在。

目标：

- top conflict atoms 是否覆盖显著比例 old-new errors；
- conflict score 是否能区分高错误 atom；
- 若诊断不成立，停止 repair 实验。

预期时间：

- atom trace + aggregation：2 到 3 天；
- diagnostics + plots：1 到 2 天。

### 5.3 第三阶段：Atom-Class Gate

范围：

- 只做 low-risk atom-class gate / downweight；
- clone/quarantine 只作为 appendix exploratory。

目标：

- Old-New HM raw-relative +0.2pp 以上；
- 不牺牲 A_Last；
- random/high-usage-only controls 不能解释收益。

预期时间：

- gate implementation：2 到 4 天；
- CUB controls：3 到 5 天；
- ImageNet-R / CIFAR-100 sanity：视资源追加。

## 6. 必须保留的 Negative Controls

方向一：

- static fusion；
- oracle fusion；
- random gate；
- same-memory raw fallback；
- no-raw fusion；
- validation-overfitting check；
- per-task / per-class / high-residual breakdown；
- full memory ledger audit。

方向二：

- random atom downweight；
- high-usage-only downweight；
- old-support-only downweight；
- uniform budget；
- no-gate trace-only；
- class-pair shuffled PMI；
- memory ledger audit。

## 7. 最终风险矩阵

| 场景 | 影响 | 回退方案 |
|---|---|---|
| Direction 1 不能追平 raw | accuracy 主线不足 | 回退为 58% memory reduction with <=0.5pp loss |
| Direction 1 只对 CUB 有效 | 泛化不足 | 主文聚焦 fine-grained CUB，ImageNet-R/CIFAR 作为边界条件 |
| Direction 2 无 HM 收益 | atom conflict repair 不成立 | 保留 atom conflict diagnostics，不 claim repair |
| fallback memory 过大 | memory-Pareto 弱化 | 限制 fallback budget，报告 Pareto 曲线而非单点 |
| CODA-Prompt 对齐后差距大 | 论文级 baseline 风险 | same-feature classifier-only 做主线，CODA compatibility 放 appendix |
| 所有修复失败 | 无 accuracy gain | 回退为 deployable prototype-memory compression + diagnostic framework |

## 8. 推荐写法

不推荐标题方向：

- LifeTopoDict: A New Topology Classifier for CIL
- Improving HC-SOINN with Dictionary Growth and Edge-Aware Scoring
- Accuracy-Improving Dictionary Learning for CIL

推荐标题方向：

- LifeTopoDict: Audited Compression and Selective Repair for Topology Memory in Class-Incremental Learning
- Compression-Aware Reliability Calibration for HC-SOINN Topology Memory
- Diagnosing and Repairing Shared-Atom Conflict in Compressed Topology Classifiers

推荐一句话 claim：

> LifeTopoDict compresses HC-SOINN topology memory by about 58%, then uses reliability and atom-conflict diagnostics to spend a small audited repair budget on risky regions, targeting raw-HC-SOINN-level or slightly better A_Last with improved Old-New balance.
