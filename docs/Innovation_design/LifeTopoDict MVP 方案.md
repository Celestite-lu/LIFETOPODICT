**最终版：LifeTopoDict MVP 方案**

**快速查阅卡**

> 2026-05-22 实验决策更新：residual dictionary growth 与 additive edge-aware scoring 已关闭，后续不再纳入实验设置。当前保留主线为 `dictionary-coded nodes + lifecycle`；edge age/reliability 只作为诊断元数据保留，不再以 additive scoring 方式进入推理。

- **核心假设**：HC-SOINN 的 class-local topology nodes 应保留节点身份，但节点中心可由共享字典 atoms 稀疏编码，从而获得跨类共享、生命周期审计和 memory-Pareto 潜力。
- **方法名**：LifeTopoDict, Lifelong Topological Dictionary Prototypes。
- **三个推进点**：
  1. `dictionary-coded nodes`：`v_hat_{c,i}=norm(D a_{c,i})`，保留 HC-SOINN 节点身份，不让 atom 直接替代 node。
  2. `minimal lifecycle`：P0 只做 `protected/plastic/inactive`，管理 atom/node 是否可更新、读出、删除。
  3. `auditable edges`：P0 保存 edge age/reliability 作为诊断；additive edge-aware scoring 已弃用。
- **MVP 消融链（更新后）**：`HC-SOINN → +dictionary-coding → +minimal lifecycle`
- **完整验证矩阵**：额外加入 `HC-SOINN+STAR raw`、`dict-coded+STAR-gated`
- **MVP 成功标准**：accuracy drop `≤0.5pp`，compact memory 降低 `≥30%`，memory-Pareto 至少一个点不被 raw HC-SOINN 支配，implementation memory 单独报告。
- **Top-3 风险**：字典压缩损失局部拓扑；共享 atoms 造成负迁移；生命周期阈值被质疑为 heuristic。
- **关键裁决**：MVP 不承诺强过 `HC-SOINN+STAR`；完整 feature-adapted 验证阶段再追求该目标。

---

## 1. 融合定位与核心 Narrative

**方法名**：LifeTopoDict, Lifelong Topological Dictionary Prototypes。

**核心 narrative**：

> From topology-aware prototypes to lifelong topological dictionary prototypes.

更精确地说：

> LifeTopoDict first turns HC-SOINN’s class-local raw nodes into dictionary-coded local topology nodes, then attaches a minimal lifecycle layer that controls which atoms/nodes may be reused, updated, retired, or trusted. P0 validates compression and auditability; P1/P2 validate drift control and edge-aware inference.

LifeTopoDict 解决 Yi2026BeyondPN 未完全解决的四个问题：节点冗余、动态拓扑不可审计、STAR anchor 噪声/stale binding 风险、边 `E_c` 未进入推理。

与单独方法的本质区别：

| 方法 | 核心对象 | 局限 | LifeTopoDict 改变 |
|---|---|---|---|
| LifeGAD | 共享字典 + 单 prototype | 类内多峰/非凸不足 | 引入 HC-SOINN class-local 多节点 |
| HC-SOINN | 每类 raw nodes + edges | 节点独立、冗余、生命周期弱 | 节点中心由共享 atoms 编码 |
| STAR | anchor-driven drift adaptation | anchor memory 高、噪声绑定 | P1 作为 lifecycle-gated drift module |

**边界 caveat**：`summary_Yi2026BeyondPN.md` 没有“第 8.3 节 LifeGAD 对比表”；LifeGAD/HC-SOINN 对比来自 `LifeGAD_PTM_CIL_design_zh_revised.md` 第 8.3 节。  
来源：初步方案 L10-L22、L45-L53；LifeGAD revised §8.3 L1572-L1585；Agent C/E/H。

---

## 2. 三个推进点的技术方案

### 2.1 压缩与共享 HC-SOINN 节点

**机制设计**

保留 HC-SOINN 的 class-local node 身份。每个节点 `v_{c,i}` 由共享字典重构：

```text
D = [d_1, ..., d_M]          # global normalized atoms
a_{c,i}                      # node-level sparse coefficient
v_hat_{c,i}=norm(D a_{c,i})  # dictionary-coded local node
r_{c,i}=1-cos(v_{c,i}, v_hat_{c,i})
```

关键数据结构：

- `D`：共享 atoms，带 atom state。
- `node_coeff[c][i]`：节点级 sparse coefficients。
- `node_residual[c][i]`：重构残差，用于 grow 和失效诊断。
- `node_state[c][i]`：class-local node 状态。
- `materialized_center[c][i]`：推理缓存中的临时 `v_hat`。

**对接方式**

保留 HC-SOINN 的层次聚类、SOINN 精炼、class-local graph 和 global-local readout。替换的是长期 raw node center 存储，不替换节点身份。

P0 推理先 materialize `v_hat_{c,i}`，再复用原 `predict_topk()`；推理复杂度仍为 `O(N*C*D + N*M_nodes*D)`，额外重构成本在 task boundary/cache dirty 时发生。

来源：初步方案 L15-L16；Agent E；`hc_soinn_classifier.py::_Cluster:28-36`, `_ensure_predict_cache:436-500`, `predict_topk:722-918`；LifeGAD revised §2.2 L260-L308。

### 2.2 生命周期控制拓扑漂移

**机制设计**

P0 只实现三态，避免过度堆状态机：

| 状态 | 对象 | 行为 |
|---|---|---|
| `protected` | atom/node | 禁止普通更新；禁止低度删除；参与推理 |
| `plastic` | atom/node | 允许更新、重构、grow |
| `inactive` | atom/node | 不参与编码、读出、更新；物理 prune 暂缓 |

P1 再加入 `quarantined`、STAR anchor gate、stale-binding refresh。P2 才考虑 clone、rumination、feature-space transport。

**触发策略**

- `protected`：old support 高，且 AEPMI/PMI proxy 有效并低风险。
- `inactive`：usage 连续低，且 old support 低。
- PMI 少样本无效时不做硬迁移，只记录 `PMIInvalidRate` 或 soft risk。

**对接方式**

- 保留 HC-SOINN 节点生成和 global-local scoring。
- 门控节点删除和更新；源码中 `degree <= max_degree_for_removal` 删除规则必须受 protected/high-support gate 约束。
- Frozen-feature MVP 不启用 STAR；feature-adapted 完整验证中，STAR 只作为受 lifecycle gate 约束的模块。

来源：初步方案 L18-L19、L48；Agent E/F/H；LifeGAD revised §2.1 L197-L258、§6.1-6.7 L1216-L1485；`_simplified_soinn_on_clusters:202-230`；`STARAligner.select_anchors_for_current_task:93-136`。

### 2.3 真正利用边连接信息

**机制设计**

边仍定义在 class-local node graph 上：

```text
E_c: (i,j) connects v_hat_{c,i}, v_hat_{c,j}
```

不把边迁移为 atom graph，因为 HC-SOINN 边是类内流形邻接，而 LifeGAD atom 是跨类共享坐标。

P0：

- 持久化 edge age。
- 保存 local node index。
- 保存 edge reliability。
- 不进入主 score。

P1 选择一种进入主消融：

```text
local_score = nearest_node_score
            + gamma * edge_support
            - eta * edge_risk
```

或采用一跳 risk propagation。必须报告：

- `edge_use_rate`
- 正确/错误 prediction change
- `edge_margin_contribution`
- `edge_ablation_delta`
- `risk_suppression_rate`

还需新增边一致性验证：raw nodes 与 `v_hat` nodes 的 edge endpoint cosine shift、neighbor-rank preservation、local margin drop。

来源：初步方案 L21-L22、L36-L38；Agent E/H/J；`_simplified_soinn_on_clusters:137-245`, `compress:699-701`, `predict_topk:834-897`。

---

## 3. MVP 实现清单

**P0 必须实现**

1. Dictionary-coded node center  
   修改：`_Cluster:28-36`, `HCSOINNClassifier.__init__:329-350`, `_ensure_predict_cache:436-500`, `predict_topk:722-918`

2. Edge age 持久化  
   当前 `compress()` 后只保存 adjacency set，age 丢失。  
   修改：`_simplified_soinn_on_clusters:137-245`, `compress:699-701`

3. Minimal lifecycle  
   三态：`protected/plastic/inactive`。  
   修改：`_simplified_soinn_on_clusters:202-230`, `compress:643-718`, `_ensure_predict_cache:436-500`

4. Memory ledger  
   分开报告 `compact deployable memory` 与 `actual implementation memory`。  
   统计 atoms、coefficients、node metadata、edges、states、STAR image/feature anchors、implementation cache。

5. 协议调用修正  
   明确 task boundary 的 `compress()` 调用时机。  
   关注：`models/simplecil_hc_soinn.py::_train:130-132`, `after_task:45-50`, `trainer.py::_train:143-150`

**P1 主文消融后决定**

- `quarantined` soft risk reweighting
- STAR anchor confidence gate
- state-aware STAR delta scaling
- edge-aware local score / one-hop risk propagation
- sparse storage，不长期保存 reconstructed raw centers

**P2 Appendix/后续**

- clone
- selective anchor rumination
- feature-space transport
- full graph diffusion
- atom graph
- GPU/Torch 化重写
- STAR+replay

来源：Agent E P0/P1/P2；Agent F/I/J。

---

## 4. 消融实验设计

**MVP 降级链**

```text
HC-SOINN
→ +dictionary-coding
→ +minimal lifecycle
→ +edge-aware scoring
```

STAR 不进 P0，因为 frozen-feature classifier-head setting 下 STAR 的 drift 价值有限，且源码中 `simplecil_hc_soinn` 不调用 STAR；STAR memory 又被 image anchor 主导。

**完整验证改为可归因矩阵，而非单线递增**

| 方法 | 目的 |
|---|---|
| HC-SOINN raw | 原始拓扑 baseline |
| HC-SOINN+STAR raw | 原始 drift baseline |
| HC-SOINN same-memory raw | 控制节点预算公平性 |
| dict-coded | 验证字典编码是否保持局部判别并降存 |
| dict-coded+lifecycle | 验证状态控制是否改善稳定性 |
| dict-coded+edge | 验证边是否真实影响推理 |
| dict-coded+STAR-gated | 验证 lifecycle 是否能约束 STAR anchor/drift |

每个模块绑定不同证据：

- dictionary-coding：memory-Pareto、node residual、local margin drop
- lifecycle：worst-order、Old-New HM、stale-risk、high-risk suppression
- edge：edge-use、margin contribution、edge ablation

来源：初步方案 L31-L36；Agent F/I/J。

---

## 5. 实验方案

**数据集与协议**

- CUB-200：采用 Yi 主表 `20 tasks × 10 classes`
- ImageNet-R：`40 tasks × 5 classes`
- CIFAR-100：sanity benchmark，不承载主 topology claim
- 3 个 class-order seeds
- 统一 frozen ViT feature cache
- 所有 baseline 同协议重跑

Caveat：Yi 原文主表与附录对 CUB split 存在不一致，必须在实验记录中固定一个 split，禁止混表。  
来源：Agent C/F。

**Baselines**

拓扑分类器层：

- NCM / SimpleCIL
- HC-SOINN raw
- HC-SOINN same-memory raw
- LifeTopoDict P0
- LifeTopoDict + lifecycle
- LifeTopoDict + edge
- HC-SOINN+STAR raw
- LifeTopoDict+STAR-gated

Full PTM-CIL 系统层：

- LifeGAD single prototype
- SimpleCIL-HC-SOINN
- CODA/DualPrompt + HC-SOINN/STAR，仅完整验证阶段加入

**指标**

主指标：

- Average Accuracy
- Final Accuracy
- Forgetting
- Old/New Acc
- Old-New HM
- memory-Pareto AUC

Memory：

- compact deployable memory
- actual implementation memory
- atoms / coefficients / node metadata / edge / state
- STAR feature memory
- STAR image-anchor memory
- implementation cache

诊断：

- node reconstruction residual
- local margin drop
- shared atom reuse rate
- shared-atom conflict
- per-atom class entropy
- old support
- AEPMI / PMIInvalidRate
- EffRank / GTE_local
- edge-use rate
- edge-caused prediction change
- edge ablation delta

**HC-SOINN+STAR head-to-head**

分两套协议：

- Frozen-feature MVP：不把 STAR 作为 P0 目标，只报告与 STAR 的 gap 和 memory 差异。
- Feature-adapted 完整验证：比较 `HC-SOINN+STAR raw` vs `LifeTopoDict+STAR-gated`。

来源：Agent D/F/I。

---

## 6. 风险清单与预案

| 风险 | 概率 | 影响 | 预案 |
|---|---:|---:|---|
| HC-SOINN+STAR 已很强 | 高 | 高 | MVP 只承诺 memory-Pareto/非劣；完整验证再追求超过 STAR |
| 字典压缩损失局部拓扑 | 中高 | 高 | 报告 residual、edge consistency、local margin drop；高残差触发 grow 或回退 raw node |
| 共享 atoms 负迁移 | 中 | 高 | 报告 shared-atom conflict、class entropy、old support；高 residual 不强行共享 |
| PMI 少样本不可靠 | 高 | 高 | `n_min` 不足时禁止硬迁移；只做 soft risk；报告 PMIInvalidRate |
| lifecycle 阈值敏感 | 高 | 中高 | P0 只三态；阈值由 base/cal split 固定；做 sensitivity、random/usage/error-rate trigger 对照 |
| 中等任务相似性误触发 stale-binding | 中高 | 高 | stale 先入候选队列，不直接 quarantine |
| edge 名义使用 | 中高 | 中 | 必报 edge-use、prediction change、edge ablation；无贡献则降 appendix |
| STAR 口径混乱 | 中 | 高 | frozen 与 feature-adapted 分开 |
| edge age 源码丢失 | 高 | 中 | P0 修复 edge age persistence |
| 机制堆叠审稿风险 | 高 | 高 | 每模块绑定不同证据，不只看 Avg Acc |

**放弃融合条件**

- `+dictionary-coding` 在 CUB 和 ImageNet-R 均低于 raw HC-SOINN 超过 `1.0pp`，且 compact memory 降低不足 `30%`。
- memory-Pareto 全预算点被 raw HC-SOINN 支配。
- lifecycle 不改善任何 stability 指标，且 Old-New HM 下降超过 `0.5pp`。
- edge-use `<1%`，且 edge ablation 对 margin/HM/worst-order 无稳定正效应。
- shared atom conflict 导致旧类 accuracy 下降超过 `1.0pp`，protected/inactive gate 无法恢复。

来源：Agent F/H/I/J。

---

## 7. 时间线

**最简可行性验证：约 4-6 周**

1. 协议与源码对齐：1 周  
   split 固定、compress 时序、memory ledger、edge age 持久化。

2. Dictionary-coded nodes：1-2 周  
   实现 `D`、`a_{c,i}`、`v_hat`，跑 CUB 单 seed，报告 residual/memory/accuracy。

3. P0 lifecycle：1 周  
   三态、protected deletion gate、inactive readout mask。

4. CUB 3 seeds + ImageNet-R 初验：1-2 周  
   画 memory-Pareto，决定是否进入 P1 edge。

**完整融合验证：约 5-7 周**

1. Edge-aware scoring：1-2 周  
2. Feature-adapted STAR head-to-head：2-3 周  
3. 完整矩阵、sensitivity、failure cases：2 周

来源：Agent D 代码改动范围；Agent I 时间线审查。

---

## 8. 成功标准

**MVP 成立必须同时满足**

1. CUB-200 或 ImageNet-R 至少一个数据集上，accuracy drop `≤0.5pp`。
2. compact deployable memory 降低 `≥30%`。
3. actual implementation memory 不高于 raw HC-SOINN，或明确标为“非部署版 / compact potential only”。
4. memory-Pareto 至少一个点不被 raw HC-SOINN 支配。
5. minimal lifecycle 至少改善一个 stability 指标：worst-order、Old-New HM、stale candidate precision、high-risk node suppression。

**Edge 进入主文还需满足**

- `edge_use_rate ≥1%`
- edge ablation 对 margin、Old-New HM 或 worst-order 至少一个指标有稳定正向影响

**完整验证成立**

- 同协议、同 backbone、同 seeds 下，LifeTopoDict 与 `HC-SOINN+STAR` gap `≤0.5pp` 且 memory 更低，或 Avg/Final/Old-New HM 统计上超过 `HC-SOINN+STAR`。
- `dictionary-coding/lifecycle/edge` 三者分别在 memory/stability/edge-use 上有独立证据。

**失败判定**

若最终只能写成 “HC-SOINN with dictionary bottleneck”，且 lifecycle/edge 没有独立诊断贡献，应停止把 LifeTopoDict 作为主线。

来源：Agent F/I/J 最终审查裁决。
