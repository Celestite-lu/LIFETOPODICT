# LifeTopoDict Fusion MVP Plan: Memory-Efficient + Auditable Topological Dictionary Prototypes

> 文档版本：v1.0
> 生成日期：2026-05-21
> 基于 Agent A-F 全部发现 + 总指挥裁决

---

## 1. 融合定位与核心 Narrative

### 1.1 方法命名

**LifeTopoDict** — Lifelong Topological Dictionary Prototypes for Class-Incremental Learning。

### 1.2 核心假设（一句话）

> HC-SOINN 的 class-local topology nodes 应保留节点身份，但节点中心可由共享字典 atoms 稀疏编码，从而获得跨类共享、可审计的拓扑生命周期管理，以及 memory-Pareto 优势。

### 1.3 核心叙事

> LifeTopoDict first turns HC-SOINN's class-local raw nodes into dictionary-coded local topology nodes, then attaches a minimal lifecycle layer that controls which atoms/nodes may be reused, updated, retired, or trusted. The primary contribution is memory efficiency with auditable compression; lifecycle is the mechanism that prevents topological quality degradation after compression; edge utilization is an optional enhancement for exploiting the otherwise-unused class-local graph structure.

叙事重心依据总指挥裁决：字典压缩带来的 **memory-Pareto** 是最直接的贡献，lifecycle 是保证压缩后拓扑质量不退化的必要机制，边利用是锦上添花。

来源：总指挥裁决 1；Agent F 建议的叙事切换（从 "lifecycle" 转为 "Memory-Efficient + Auditable Topology"）；初步方案原文 L1-L16。

### 1.4 与单独方法的本质区别

| 方法 | 核心对象 | 核心局限 | LifeTopoDict 改变 |
|---|---|---|---|
| **LifeGAD** (单独) | 共享字典 + 单 prototype per class | 类内多峰/非凸分布表达不足；无 class-local topology | 引入 HC-SOINN 的 class-local 多节点，每个节点由稀疏字典编码 |
| **HC-SOINN** (单独) | 每类 raw nodes (~18-21 个) + edges | 节点间无跨类共享导致冗余；center_raw 存储开销大；边 E_c 未进入推理；无生命周期管理 | 节点中心由共享 atoms 稀疏重构，降低存储；加入 lifecycle 控制更新/删除；修复 edge age 持久化并引入 edge-aware scoring |
| **STAR** | anchor-driven drift adaptation | anchor 存储每类约 36MB（image anchors）；与字典编码有结构冲突（共享 atom 导致 drift 不独立） | P0 不启用 STAR；P1 探索 STAR drift 作用于 sparse coefficient a 而非 atom d_m |

来源：Agent A 初步方案 L5-L6；Agent C 论文提取（HC-SOINN 每类 ~18-21 子原型）；Agent D 源码分析（STAR anchor 约 36MB）；Agent E 可行性审查（STAR 与字典编码结构冲突）。

### 1.5 解决 Yi2026BeyondPN 的未解决问题

Yi2026BeyondPN (HC-SOINN) 明确存在的四个局限：

1. **节点冗余**：每类 ~18-21 个独立子原型，无跨类共享机制，总存储随类数线性增长。来源：Agent C 论文提取；Agent E memory 估算 36.4MB。
2. **动态拓扑不可审计**：节点生成/删除无状态追踪，无法判断哪些节点应该被保护/淘汰。来源：Agent D 源码分析，`_simplified_soinn_on_clusters` L202-230 的删除规则仅基于 degree，无 protected gate。
3. **边 E_c 推理时完全未用**：论文明确承认这是局限。来源：Agent C 论文提取；源码 `compress()` L699-701 仅保存 adjacency set，age 信息丢失。
4. **STAR anchor 噪声/stale binding 风险**：STAR 存储原始图像 anchors，每类约 36MB；且 `simplecil_hc_soinn.py` 中 STAR 未被调用。来源：Agent D 源码发现；Agent F 风险审查。

来源：Agent C 论文提取（双视角推理 alpha=0.5）；Agent D 源码分析（死代码 `tau_merge/tau_reject`，STAR 未调用）；Agent E 可行性审查（edge age 丢失）。

---

## 2. 三个推进点的技术方案

### 2.1 推进点一：Dictionary-Coded Nodes — 压缩与共享 HC-SOINN 节点

#### 2.1.1 机制设计

保留 HC-SOINN 的 class-local node 身份。每个节点 `v_{c,i}` 由全局共享字典 D 稀疏重构：

```text
D = [d_1, ..., d_M]               # 全局归一化 atoms, d_m in R^D, ||d_m||=1
a_{c,i} in R^M                    # 节点级 sparse coefficient (top-k 非零)
v_hat_{c,i} = norm(D @ a_{c,i})   # 字典编码后的 local node (归一化)
r_{c,i} = 1 - cos(v_{c,i}, v_hat_{c,i})   # 重构残差
```

**关键数据结构**（新增）：

| 数据结构 | 类型 | 位置 | 说明 |
|---|---|---|---|
| `D_t` | `np.ndarray [M, D]` | `HCSOINNClassifier` | 全局共享字典，M 个归一化 atoms |
| `node_coeff` | `Dict[int, List[np.ndarray]]` | `HCSOINNClassifier` | `node_coeff[c][i]` = 节点 (c,i) 的 sparse coefficient |
| `node_residual` | `Dict[int, List[float]]` | `HCSOINNClassifier` | `node_residual[c][i]` = 重构残差 r_{c,i} |
| `atom_state` | `List[str]` | `HCSOINNClassifier` | 每个 atom 的状态：plastic/protected/inactive |

**保留的数据结构**（不改变）：

| 数据结构 | 位置 | 说明 |
|---|---|---|
| `_Cluster.center` | `_Cluster` L31 | 归一化节点中心，字典编码后存储 `v_hat` |
| `_Cluster.center_raw` | `_Cluster` L34-36 | 未归一化中心，字典编码后存储 `D @ a_{c,i}`（未归一化的重构值） |
| `_Cluster.count` | `_Cluster` L32 | 节点样本计数，不变 |
| `class_clusters` | `HCSOINNClassifier` L333 | `{cls: List[_Cluster]}`，不变 |
| `class_mu` / `class_mu_raw` | L329-330 | NCM 中心，不变 |

**关键操作**：

1. **字典初始化**（base task compress 后）：
   - 输入：base task 所有类的 `_Cluster.center_raw`
   - 操作：对全部 base task 节点中心执行 K-Means 或 class-means 初始化，生成 `D_0` [M, D]
   - 来源：Phase3A X3/B13/B14（K-Means 初始化远优于随机初始化，Stability 0.927 vs 0.542）

2. **稀疏编码**（每个节点独立）：
   - 输入：节点 `v_{c,i}`（归一化 center），字典 `D_t`
   - 操作：Top-K Ridge 投影
     ```python
     # 伪代码
     sim = D_t.T @ v_{c,i}                      # [M] atom-node 相似度
     topk_indices = argsort(sim)[:k]             # 选 k 个最相关 atom
     D_sub = D_t[topk_indices]                   # [k, D]
     a_sub = ridge_solve(D_sub, v_{c,i}, lambda_2)  # [k] 带正则化解
     a_full = zeros(M); a_full[topk_indices] = a_sub
     ```
   - 残差计算：`r_{c,i} = 1 - cos(v_{c,i}, norm(D @ a_full))`
   - 来源：LifeGAD revised 2.2；Phase3A X4/B7（L2 正则 lambda 对闭式解稳定性至关重要）

3. **节点 Materialization**（推理时）：
   - 推理前缓存：`v_hat_{c,i} = norm(D @ a_{c,i})`
   - 写入 `_Cluster.center`（归一化）和 `_Cluster.center_raw`（未归一化重构值）
   - 推理复用原 `predict_topk()` 逻辑，无额外推理复杂度
   - 来源：Agent E 可行性审查（API 对接干净，推理复杂度不增加）

4. **字典更新**（incremental task compress 后）：
   - 新 task 节点加入后，重新执行稀疏编码
   - Plastic atoms 可通过最小化全体节点残差和更新方向
   - Protected atoms 冻结不更新

#### 2.1.2 与原始方法的对接方式

| HC-SOINN 机制 | 处理 |
|---|---|
| 层次聚类 `_hierarchical_cluster` | **保留**。仍然先聚类生成原始节点 |
| SOINN 精炼 `_simplified_soinn_on_clusters` | **保留**。精炼后的节点作为字典编码的输入 |
| `class_mu` / `class_mu_raw` | **保留**。NCM 中心不编码，用于 global score |
| `_Cluster.center` 存储 | **替换**。从存储 raw center 改为存储 `v_hat` |
| `_Cluster.center_raw` 存储 | **替换语义**。从存储原始未归一化中心改为存储 `D @ a`（未归一化重构值） |
| `predict_topk` 推理路径 | **不改变**。推理前 materialize `v_hat` 写入 `_Cluster.center` |
| `_ensure_predict_cache` | **不改变**。缓存基于 `_Cluster.center`，materialization 发生在 cache 构建前 |

**对接时机**：字典编码步骤插入 `compress()` 函数末尾（L718 之前），在 SOINN 精炼完成、`class_clusters` 更新之后，`invalidate_cache()` 之前。

来源：Agent E 可行性审查（"API 对接干净，compress 后插入字典编码步骤"）；`hc_soinn_classifier.py` `compress()` L643-718；`_Cluster` L28-36。

#### 2.1.3 代码级修改标注

| 修改位置 | 文件 | 行号范围 | 修改内容 |
|---|---|---|---|
| `_Cluster` 类 | `utils/hc_soinn_classifier.py` | L28-36 | 新增 `coeff: Optional[np.ndarray]` 字段和 `residual: float` 字段 |
| `HCSOINNClassifier.__init__` | `utils/hc_soinn_classifier.py` | L287-364 | 新增 `self.dict_atoms: Optional[np.ndarray]`、`self.atom_states: List[str]`、`self.dict_M: int`、`self.dict_k: int`、`self.dict_lambda2: float` 参数 |
| `compress()` | `utils/hc_soinn_classifier.py` | L643-718 | 在 L701（`class_edges` 更新后）插入字典编码调用：`self._encode_nodes_with_dict()` |
| 新增 `_encode_nodes_with_dict()` | `utils/hc_soinn_classifier.py` | 新方法 | 遍历 `class_clusters`，对每个节点执行 Top-K Ridge 编码，更新 `_Cluster.center` 和 `_Cluster.center_raw` |
| 新增 `_materialize_nodes()` | `utils/hc_soinn_classifier.py` | 新方法 | 从 `D` 和 `a` 重构 `v_hat`，写入 `_Cluster.center`，供 `_ensure_predict_cache` 使用 |
| `_ensure_predict_cache` | `utils/hc_soinn_classifier.py` | L436-500 | 在 cache 构建前调用 `_materialize_nodes()`（若 dirty flag 置位） |

---

### 2.2 推进点二：Minimal Lifecycle — 生命周期控制拓扑漂移

#### 2.2.1 机制设计

P0 只实现三态状态机，避免过度堆叠：

| 状态 | 适用对象 | 行为 | 触发条件 |
|---|---|---|---|
| **plastic** | atom / node | 允许编码更新、字典更新、重构 | 默认状态；或从 inactive 唤醒 |
| **protected** | atom / node | 禁止普通更新；禁止 degree-based 删除；参与推理 | old support 高（atom 被多个旧类节点依赖）|
| **inactive** | atom / node | 不参与编码/读出/更新；物理 prune 暂缓 | usage 连续低 + old support 低 |

**Atom-level 生命周期**：

```
state_m 的触发条件:
  → protected:  old_support_m >= theta_support AND task_age_m >= 2
  → inactive:   usage_m < theta_usage FOR consecutive T_inactive tasks
  → plastic:    (default) OR inactive node 的 residual 触发 grow 时重新激活
```

其中：
- `old_support_m` = 使用 atom m 的旧类节点数量 / 使用 atom m 的总节点数量
- `usage_m` = 最近一个 task 中 atom m 被选入 top-k 的次数
- `theta_support`, `theta_usage`, `T_inactive` 为超参数，由 base/cal split 固定

**Node-level 生命周期**：

```
node_state_{c,i} 的触发条件:
  → protected:  该节点所属类 cls 的 task_age >= 2
  → inactive:   residual r_{c,i} > theta_r_inactive AND count_{c,i} <= 1
  → plastic:    (default)
```

- protected 节点在 `_simplified_soinn_on_clusters` 的 degree-based 删除（L202-230）中豁免
- inactive 节点不参与 `_ensure_predict_cache` 的 materialization，不进入推理

**与源码现有 protected 机制的对接**：

源码已存在 `freeze_nodes()` 方法（L505-517），通过 `frozen_clusters` 存储深拷贝。LifeTopoDict 的 protected 机制与此不同：
- `freeze_nodes()` 是 STAR 的备份/恢复机制，全类冻结
- LifeTopoDict 的 protected 是细粒度状态机，per-atom/per-node 粒度

P0 不替换 `freeze_nodes()`，而是在其上层增加状态查询接口。

来源：初步方案 L18-L19；Agent E 可行性审查（"`freeze_nodes()` 已有 protected 原型，第 505-517 行"）；Agent F 风险审查（"P0 只三态，阈值由 base/cal split 固定"）；`hc_soinn_classifier.py` `_simplified_soinn_on_clusters` L202-230（degree-based 删除规则）。

#### 2.2.2 与原始方法的对接方式

| HC-SOINN 机制 | 处理 |
|---|---|
| 节点生成 `_hierarchical_cluster` | **保留**。新节点默认 plastic |
| SOINN 精炼 `_simplified_soinn_on_clusters` | **增强**。degree-based 删除（L202-230）需检查 `node_state != protected` |
| `freeze_nodes()` | **保留不替换**。STAR 备份机制独立于 lifecycle 状态机 |
| `add_features` / `compress` | **不改变**。生命周期检查在 compress 后执行 |
| 字典编码 `_encode_nodes_with_dict` | **增强**。protected atoms 不参与字典更新 |

#### 2.2.3 代码级修改标注

| 修改位置 | 文件 | 行号范围 | 修改内容 |
|---|---|---|---|
| `_simplified_soinn_on_clusters` | `utils/hc_soinn_classifier.py` | L202-203 | 删除判断增加条件：`and node_state[i] != 'protected'` |
| `compress()` | `utils/hc_soinn_classifier.py` | L699-718 | 在字典编码后调用 `_update_lifecycle_states()` |
| 新增 `_update_lifecycle_states()` | `utils/hc_soinn_classifier.py` | 新方法 | 计算每个 atom 的 `old_support` 和 `usage`，更新 `atom_states` 和 `node_states` |
| `_encode_nodes_with_dict()` | `utils/hc_soinn_classifier.py` | 新方法 | protected atoms 的行在字典更新中被冻结（行 mask 为 0） |
| `_materialize_nodes()` | `utils/hc_soinn_classifier.py` | 新方法 | inactive 节点跳过 materialization |
| `_ensure_predict_cache` | `utils/hc_soinn_classifier.py` | L462-476 | 只 materialize 非 inactive 节点 |

---

### 2.3 推进点三：Auditable Edges — 真正利用边连接信息

#### 2.3.1 机制设计

边定义在 class-local node graph 上，不迁移为 atom graph（因为 HC-SOINN 边是类内流形邻接，atoms 是跨类共享坐标）。

**P0（必须实现）：Edge Age 持久化修复**

当前源码中 `_simplified_soinn_on_clusters` 返回 `final_edges_map: Dict[int, Set[int]]`（L233-244），类型是 `Dict[int, Set[int]]`——只存邻接关系，age 信息在 `edges[i][j]` 的整数值中但被类型声明丢弃。

修复方案：将 `_simplified_soinn_on_clusters` 的返回类型改为 `Dict[int, Dict[int, int]]`，保留 age 值。

```python
# 修改前 (L240-243):
final_edges_map[new_idx] = set()
for nbr in edges[old_idx]:
    if nbr in final_indices:
        final_edges_map[new_idx].add(final_indices.index(nbr))

# 修改后:
final_edges_map[new_idx] = {}
for nbr in edges[old_idx]:
    if nbr in final_indices:
        final_edges_map[new_idx][final_indices.index(nbr)] = edges[old_idx][nbr]
```

同步修改 `compress()` 中 `class_edges` 的类型声明（L699）：
```python
# 修改前:
self.class_edges[cls] = soinn_edges   # Dict[int, Set[int]]

# 修改后:
self.class_edges[cls] = soinn_edges   # Dict[int, Dict[int, int]]
```

**P1（主文消融后决定）：Edge-Aware Local Score**

选择 **edge-aware local score** 方案（而非一跳 risk propagation），因为更简单且诊断性更强：

```text
S_local(x, c) = max_i cos(x, v_hat_{c,i})     # 原始 local score
S_edge(x, c) = S_local(x, c)
             + gamma * avg_{j in N(i*)} cos(x, v_hat_{c,j})   # 邻居支持
             - eta * edge_age(i*) / max_age                     # 老 edge 降权
```

其中 `i*` = 最相似节点，`N(i*)` = `i*` 的邻居集合。

必须报告的诊断指标：
- `edge_use_rate`：至少有一个 task 中 edge 对 prediction 产生影响的样本比例
- `edge_prediction_change`：edge score 导致 top-1 prediction 改变的样本数
- `edge_margin_contribution`：`gamma * avg_neighbor_sim - eta * age_penalty` 的分布
- `edge_ablation_delta`：关闭 edge score 后 accuracy 变化

来源：Agent E 可行性审查（"edge age 在 compress 后丢失需修复，选 edge-aware local score 方案"，"改动约 5 行"）；`hc_soinn_classifier.py` `_simplified_soinn_on_clusters` L233-244；`compress` L699-701。

#### 2.3.2 与原始方法的对接方式

| HC-SOINN 机制 | 处理 |
|---|---|
| `_simplified_soinn_on_clusters` 返回 edges | **修改返回类型**。`Dict[int, Set[int]]` → `Dict[int, Dict[int, int]]` |
| `class_edges` 存储 | **修改类型**。同步改为带 age 的 dict |
| `predict_topk` 推理 | P0 不改变；P1 在 L899-901 的 `final_scores` 计算中加入 edge 修正项 |

#### 2.3.3 代码级修改标注

| 修改位置 | 文件 | 行号范围 | 修改内容 |
|---|---|---|---|
| `_simplified_soinn_on_clusters` 返回类型 | `utils/hc_soinn_classifier.py` | L233-244 | `set()` → `dict()`，保存 age 值 |
| `compress` 中 `class_edges` 类型声明 | `utils/hc_soinn_classifier.py` | L699 | 类型标注改为 `Dict[int, Dict[int, int]]` |
| `_simplified_soinn_on_clusters` 内部 age 维护 | `utils/hc_soinn_classifier.py` | L176-185 | age 递增逻辑已存在（`edges[s1][nbr] += 1`），无需修改 |

P1 额外修改：

| 修改位置 | 文件 | 行号范围 | 修改内容 |
|---|---|---|---|
| `predict_topk` | `utils/hc_soinn_classifier.py` | L899-901 | 在 `final_scores` 中加入 edge-aware 修正 |
| `_ensure_predict_cache` | `utils/hc_soinn_classifier.py` | L436-500 | 缓存中额外存储 edge 信息（邻接 + age）|

---

### 2.4 三个推进点之间的依赖关系

```
推进点一 (Dictionary-Coded Nodes)
    ↓ [必须先完成，因为 lifecycle 作用于字典 atom 状态]
推进点二 (Minimal Lifecycle)
    ↓ [依赖 edge age 持久化（推进点三 P0），但 edge-aware scoring（推进点三 P1）可选]
推进点三 P0 (Edge Age 持久化)  ← 可与推进点一并行
推进点三 P1 (Edge-Aware Scoring)  ← 依赖推进点一 + 推进点二
```

- 推进点一是核心：字典编码定义了节点中心的新存储方式，后续 lifecycle 和 edge 都作用于编码后的节点
- 推进点三 P0（edge age 修复）与推进点一可并行开发，因为它们修改不同函数
- 推进点二依赖推进点一：atom 状态（protected/plastic/inactive）的语义建立在字典编码之后
- 推进点三 P1（edge-aware scoring）依赖推进点一和推进点二：edge 修正作用于 materialized 节点

来源：Agent E 可行性审查（"三个推进点均为'需调整'"）；总指挥裁决（优先级不变）。

---

## 3. MVP 实现清单

### 3.1 P0 — 必须实现

| 序号 | 模块 | 修改内容 | 预计改动行数 | 源码位置 |
|---|---|---|---|---|
| P0-1 | Dictionary-coded node center | 新增 `dict_atoms`, `node_coeff`, `node_residual` 数据结构；实现 `_encode_nodes_with_dict()` 和 `_materialize_nodes()` | ~60 行 | `utils/hc_soinn_classifier.py` `_Cluster` L28-36, `__init__` L287-364, `compress` L699-718 |
| P0-2 | Edge age 持久化 | 修改 `_simplified_soinn_on_clusters` 返回类型和 `class_edges` 类型 | ~5 行 | `utils/hc_soinn_classifier.py` L233-244, L699 |
| P0-3 | Minimal lifecycle (三态) | 实现 `_update_lifecycle_states()`；修改 degree-based 删除逻辑增加 protected gate；inactive 节点跳过 materialization | ~35 行 | `utils/hc_soinn_classifier.py` L202-230, `compress` L699-718 |
| P0-4 | Memory ledger | 分开报告 compact deployable memory 与 actual implementation memory | ~20 行 | 新增辅助函数 |
| P0-5 | 协议调用修正 | 明确 task boundary 的 `compress()` 调用时机；确认 `after_task()` 中字典编码的执行点 | ~5 行 | `models/simplecil_hc_soinn.py` L45-50, L130-132 |

**P0 总改动量**：约 125 行（与 Agent E 估算的 "约 100 行" 一致）。

来源：Agent E 可行性审查（"改动范围约 100 行"）；Agent D 源码分析；总指挥裁决。

### 3.2 P1 — 主文消融后决定

| 序号 | 模块 | 修改内容 | 条件 |
|---|---|---|---|
| P1-1 | Edge-aware local score | 在 `predict_topk` 的 `final_scores` 中加入 edge 修正项 | edge age P0 修复完成 + edge_use_rate >= 1% |
| P1-2 | `quarantined` soft risk reweighting | 四态扩展；stale atom 进入候选队列 | PMI 在 P0 验证中证明可靠（PMIInvalidRate < 30%）|
| P1-3 | STAR anchor confidence gate | STAR 作为受 lifecycle gate 约束的模块 | feature-adapted 完整验证阶段 |
| P1-4 | State-aware STAR delta scaling | STAR drift 作用于 sparse coefficient a 而非 atom d_m | P1-3 完成后 |
| P1-5 | Sparse storage | 不长期保存 reconstructed raw centers | P0 验证 residual 稳定 |

来源：初步方案 L169-176；Agent E P0/P1/P2 分类。

### 3.3 P2 — Appendix/后续工作

| 序号 | 模块 | 说明 |
|---|---|---|
| P2-1 | Clone | 高余弦冲突 atom 克隆为独立副本 |
| P2-2 | Selective anchor rumination | 旧类伪特征补强 |
| P2-3 | Feature-space transport | 仅 feature-adapted setting |
| P2-4 | Full graph diffusion | 图扩散推理 |
| P2-5 | Atom graph | 跨类 atom-level 邻接图 |
| P2-6 | GPU/Torch 化重写 | 全量 GPU 推理 |
| P2-7 | STAR + replay | STAR 与 rehearsal 结合 |

来源：初步方案 L179-185。

---

## 4. 消融实验设计

### 4.1 消融链

```text
HC-SOINN (raw)
→ +dictionary-coding
→ +minimal lifecycle
→ +edge-aware scoring
```

每层的目的、预期结论和诊断指标：

| 层级 | 变体 | 目的 | 预期结论 | 核心诊断指标 |
|---|---|---|---|---|
| 0 | HC-SOINN raw | 原始拓扑 baseline | — | Avg Acc, Final Acc, memory |
| 1 | +dictionary-coding | 验证字典编码是否保持局部判别并降存储 | accuracy drop <= 0.5pp, memory 降低 >= 30% | residual distribution, local margin drop, memory-Pareto |
| 2 | +minimal lifecycle | 验证状态控制是否改善稳定性 | worst-order 改善, Old-New HM 不降 | worst-order accuracy, Old-New HM, stale-risk rate, protected deletion rate |
| 3 | +edge-aware scoring | 验证边是否真实影响推理 | edge_use_rate >= 1%, 至少一个稳定性指标改善 | edge_use_rate, edge_prediction_change, edge_ablation_delta |

### 4.2 STAR 处理

STAR 不进 P0 消融链，原因：
1. Frozen-feature classifier-head setting 下 STAR 的 drift 价值有限
2. 源码中 `simplecil_hc_soinn.py` 不调用 STAR（Agent D 发现）
3. STAR memory 被 image anchor 主导（每类约 36MB，Agent D 源码分析）
4. STAR 与字典编码有结构冲突——共享 atom 导致 drift 不独立（Agent E 可行性审查）

总指挥裁决：P0 不启用 STAR；P1 探索 STAR drift 作用于 sparse coefficient a 而非 atom d_m。

### 4.3 完整验证矩阵

STAR 相关变体在 feature-adapted 完整验证阶段单独测试：

| 方法 | 目的 |
|---|---|
| HC-SOINN raw | 原始拓扑 baseline |
| HC-SOINN+STAR raw | 原始 drift baseline |
| HC-SOINN same-memory raw | 控制节点预算公平性（限制 max_prototypes_per_class 使总 memory 与 LifeTopoDict 相当）|
| dict-coded | 验证字典编码独立效果 |
| dict-coded+lifecycle | 验证 lifecycle 独立效果 |
| dict-coded+edge | 验证 edge 独立效果 |
| dict-coded+STAR-gated | 验证 lifecycle 是否能约束 STAR anchor/drift |

每个模块绑定不同证据维度，避免"机制堆叠"审稿风险：

- **dictionary-coding**：memory-Pareto、node residual、local margin drop
- **lifecycle**：worst-order、Old-New HM、stale-risk、protected deletion rate
- **edge**：edge-use、margin contribution、edge ablation delta

### 4.4 对照消融

| 对照 | 目的 |
|---|---|
| random coding (用随机 D 编码) | 排除"降维即有效" |
| fixed growth + random directions | 分离"发生 growth"与"残差方向信息" |
| lifecycle random trigger | PMI vs error-rate-only / usage-only / random trigger，验证 atom-error association proxy 独立价值 |
| full dictionary update (无 protected) | 验证 protected freeze 的独立贡献 |

来源：初步方案 L31-L36；Phase3A X9/C13-C14（residual growth 局部改进保证仅线性代数解释，需消融验证方向信息价值）；Phase3A F8（"消融必须包含 random/ablation baselines 排除'做点什么总比不做好'"）。

---

## 5. 实验方案

### 5.1 门控实验（Agent F 提出）

在正式消融之前，必须先通过三个 Gate。**如果 Gate 1 未通过，应停止整个融合方案。**

| Gate | 验证内容 | 通过标准 | 失败处理 |
|---|---|---|---|
| **Gate 1** | HC-SOINN 在 frozen feature 下相比 NCM/SimpleCIL 的增益 | HC-SOINN 增益 > 2pp (CUB-200 或 ImageNet-R) | **停止**。如果 HC-SOINN 本身无增益，字典编码的融合无意义 |
| **Gate 2** | 字典编码重构质量 | residual < 0.3 的节点比例 > 80% | 调整 M（字典规模）或 k（稀疏度），重新验证 |
| **Gate 3** | Memory-Pareto 可行性 | compact memory < 50% HC-SOINN raw memory | 调整 M 或采用 sparse storage；若仍不通过则降低 memory claim |

来源：Agent F 风险审查（三个门控实验）。

### 5.2 数据集与协议

| 数据集 | 任务划分 | 用途 | 说明 |
|---|---|---|---|
| **CUB-200** | 20 tasks x 10 classes (Yi 主表) | 主实验数据集 | 细粒度场景，HC-SOINN 最大增益场景 (+21pp) |
| **ImageNet-R** | 40 tasks x 5 classes | 主实验数据集 | 域偏移场景，验证字典鲁棒性 |
| **CIFAR-100** | 10 tasks x 10 classes | Sanity benchmark | 不承载主 topology claim，仅用于快速消融 |

协议细节：
- 3 个 class-order seeds
- 统一 frozen ViT-B/16-IN21K feature cache
- 所有 baseline 同协议重跑，不混用不同论文的公开数值
- **Caveat**：Yi 原文主表与附录对 CUB split 存在不一致（Agent C 发现）。必须在实验记录中固定一个 split，禁止混表。

来源：Agent C 论文提取（CUB-200 划分矛盾）；Phase3A A4（任务顺序显著影响排名）；Phase3A A17（不同论文公开协议不可直接混表）。

### 5.3 Baselines

**拓扑分类器层（P0 MVP）**：

| Baseline | 说明 |
|---|---|
| NCM / SimpleCIL | 无拓扑分类器 baseline |
| HC-SOINN raw | 原始 HC-SOINN，无字典编码 |
| HC-SOINN same-memory raw | 限制 max_prototypes_per_class 使总 memory 与 LifeTopoDict 相当 |
| dict-coded | LifeTopoDict P0（仅字典编码） |
| dict-coded+lifecycle | LifeTopoDict P0 + lifecycle |
| dict-coded+edge | LifeTopoDict P0 + edge-aware scoring |

**Full PTM-CIL 系统层（完整验证阶段）**：

| Baseline | 说明 |
|---|---|
| HC-SOINN+STAR raw | 原始 drift baseline |
| LifeTopoDict+STAR-gated | lifecycle 约束 STAR |
| LifeGAD single prototype | 单 prototype 对比 |
| SimpleCIL-HC-SOINN | 当前源码集成 |
| CODA/DualPrompt + HC-SOINN/STAR | 仅完整验证阶段加入 |

来源：Agent D 源码分析（`simplecil_hc_soinn.py` 为当前集成方式）；Phase3A A17。

### 5.4 评估指标

**主指标**：

| 指标 | 说明 |
|---|---|
| Average Accuracy | 全任务平均准确率 |
| Final Accuracy | 最终任务准确率 |
| Forgetting | 标准遗忘率 |
| Old/New Acc | 旧类/新类分别准确率 |
| Old-New HM | 旧类新类调和平均（主指标之一，暴露 old-new 平衡） |
| memory-Pareto AUC | memory-accuracy Pareto 前沿面积 |

**Memory 指标（分开报告）**：

| 指标 | 说明 |
|---|---|
| compact deployable memory | 仅 atoms + coefficients + node metadata + edges + states |
| actual implementation memory | 包含 implementation cache, STAR image/feature anchors 等 |
| atoms / coefficients / node metadata / edge / state 明细 | 各组件分别报告 |

**诊断指标**：

| 指标 | 关联推进点 | 说明 |
|---|---|---|
| node reconstruction residual | dictionary-coding | 字典编码残差分布 |
| local margin drop | dictionary-coding | 编码前后类间 margin 变化 |
| shared atom reuse rate | dictionary-coding | 跨类共享 atom 使用率 |
| shared-atom conflict | dictionary-coding | 共享 atom 被不同类使用的冲突率 |
| per-atom class entropy | dictionary-coding | 每个 atom 的类分布熵 |
| EffRank | dictionary-coding | 字典有效秩，监控增长是否真正增加表达能力 |
| old support | lifecycle | 每个 atom 被旧类依赖的程度 |
| PMIInvalidRate | lifecycle | PMI 统计不可靠的比例 |
| worst-order accuracy | lifecycle | 最差任务顺序下的准确率 |
| stale-risk rate | lifecycle | stale binding 检出率 |
| protected deletion rate | lifecycle | protected 节点成功豁免删除的比例 |
| edge-use rate | edge | 至少有一个 task 中 edge 对 prediction 产生影响的样本比例 |
| edge-caused prediction change | edge | edge score 导致 top-1 prediction 改变的样本数 |
| edge ablation delta | edge | 关闭 edge score 后 accuracy 变化 |
| edge margin contribution | edge | edge 修正项的数值分布 |

来源：初步方案 L259-290；Phase3A A1/A7/A8/A11（指标设计原则）。

### 5.5 HC-SOINN+STAR Head-to-Head 对比策略

分两套协议：

**Frozen-feature MVP（P0）**：
- 不把 STAR 作为 P0 目标
- 只报告 LifeTopoDict P0 与 HC-SOINN+STAR 的 gap 和 memory 差异
- LifeTopoDict P0 不承诺超过 HC-SOINN+STAR

**Feature-adapted 完整验证（P1+）**：
- 比较 `HC-SOINN+STAR raw` vs `LifeTopoDict+STAR-gated`
- STAR drift 作用于 sparse coefficient a 而非 atom d_m
- 目标：同协议下 gap <= 0.5pp 且 memory 更低

来源：总指挥裁决 2（P0 不启用 STAR，P1 探索 STAR drift 作用于 coefficient）；初步方案 L294-298。

---

## 6. 风险清单与预案

### 6.1 技术风险

| 编号 | 风险 | 概率 | 影响 | 预案 | 来源 |
|---|---|---:|---:|---|---|
| R1 | HC-SOINN+STAR 已很强，融合方案难以超过 | 高 | 高 | MVP 只承诺 memory-Pareto/非劣；完整验证再追求超过 STAR | Agent F |
| R2 | 字典压缩损失局部拓扑质量 | 中高 | 高 | 报告 residual、edge consistency、local margin drop；高残差 (>0.3) 触发 grow 或回退 raw node | Phase3A X3 |
| R3 | 共享 atoms 负迁移（不同类争抢同一 atom） | 中 | 高 | 报告 shared-atom conflict、class entropy、old support；高 residual 不强行共享；ICFL 高余弦重置 | Phase3A X1/X5 |
| R4 | PMI 少样本不可靠（CUB-200 20-task 下几乎必然不可靠） | 高 | 高 | `n_min` 不足时禁止硬迁移；只做 soft risk；报告 PMIInvalidRate；lifecycle 决策降级到 residual+usage+old support 驱动 | Phase3A X2 |
| R5 | 字典初始化质量决定 base 成败 | 高 | 高 | 使用 K-Means/class-means 初始化而非随机初始化；base task 后验证 residual 分布；Gate 2 检查 | Phase3A X3/B13/B14 |
| R6 | Lifecycle 阈值被质疑为 heuristic | 高 | 中高 | P0 只三态；阈值由 base/cal split 固定；做 sensitivity、random/usage/error-rate trigger 对照 | Phase3A E9 |
| R7 | 中等任务相似性误触发 stale-binding | 中高 | 高 | stale 先入候选队列不直接 quarantine；PMI 触发需要 n_min 保底 | Phase3A E10 |
| R8 | Edge 名义使用（对推理无实际贡献） | 中高 | 中 | 必报 edge-use、prediction change、edge ablation；无贡献则降 appendix | Agent F |
| R9 | 增益不均匀（某些 backbone 方法从多节点分类器不受益） | 中 | 中 | 报告 per-backbone 增益；若 SimpleCIL+HC-SOINN 增益 < 0.5pp 则不声称通用性 | Phase3A D20 |
| R10 | Edge age 源码丢失导致不可复现 | 已修复 | 中 | P0 修复 edge age persistence（5 行改动） | Agent E |

### 6.2 实验风险

| 编号 | 风险 | 概率 | 影响 | 预案 |
|---|---|---:|---:|---|
| R11 | CUB-200 split 不一致导致实验不可比 | 高 | 高 | 固定一个 split，禁止混表；在实验记录中明确标注 |
| R12 | 机制堆叠审稿风险（三个推进点叠加） | 高 | 高 | 每模块绑定不同证据维度，不只看 Avg Acc；消融链逐层叠加 |

### 6.3 放弃融合条件

以下任一条件满足应停止把 LifeTopoDict 作为主线：

1. `+dictionary-coding` 在 CUB 和 ImageNet-R **均低于** raw HC-SOINN 超过 1.0pp，且 compact memory 降低不足 30%
2. memory-Pareto **全预算点**被 raw HC-SOINN 支配
3. lifecycle 不改善任何 stability 指标，且 Old-New HM 下降超过 0.5pp
4. edge-use < 1%，且 edge ablation 对 margin/HM/worst-order 无稳定正效应
5. shared atom conflict 导致旧类 accuracy 下降超过 1.0pp，protected/inactive gate 无法恢复
6. **Gate 1 未通过**（HC-SOINN 增益 <= 2pp）

若最终只能写成 "HC-SOINN with dictionary bottleneck"，且 lifecycle/edge 没有独立诊断贡献，应停止把 LifeTopoDict 作为主线。

来源：Agent F 风险审查（三个门控）；初步方案 L319-327；Phase3A X2/X3。

---

## 7. 时间线

### 7.1 最简可行性验证：约 4-6 周

| 阶段 | 时间 | 内容 | 交付物 |
|---|---|---|---|
| 协议与源码对齐 | 1 周 | split 固定；compress 时序确认；memory ledger 设计；edge age 持久化修复 (P0-2) | 可复现的 baseline pipeline |
| Gate 1 验证 | 0.5 周 | CUB-200 单 seed 跑 HC-SOINN raw vs NCM/SimpleCIL | Gate 1 通过/不通过决定 |
| Dictionary-coded nodes (P0-1) | 1-1.5 周 | 实现 D、a、v_hat；跑 CUB 单 seed | Gate 2/3 结果 |
| Minimal lifecycle (P0-3) | 1 周 | 三态实现；protected deletion gate；inactive readout mask | CUB lifecycle 消融 |
| CUB 3 seeds + ImageNet-R 初验 | 1-2 周 | 完整 P0 消融链；画 memory-Pareto | 决定是否进入 P1 |

**关键决策点**：Gate 1 未通过 → 停止。Gate 2/3 未通过 → 调整 M/k 后重试一次。CUB 3 seeds P0 消融不满足成功标准 → 停止。

### 7.2 完整融合验证：约 5-7 周

| 阶段 | 时间 | 内容 |
|---|---|---|
| Edge-aware scoring (P1-1) | 1-2 周 | 实现 edge-aware local score；CUB+INR 消融 |
| Feature-adapted STAR head-to-head (P1-3/4) | 2-3 周 | STAR-gated 集成；drift 作用于 coefficient |
| 完整矩阵 + sensitivity + failure cases | 2 周 | 全消融矩阵；阈值敏感性；per-dataset failure 分析 |

来源：Agent E（代码改动范围约 100 行）；初步方案 L335-353。

---

## 8. 成功标准

### 8.1 MVP 成立条件（P0 必须同时满足）

| 编号 | 标准 | 量化阈值 |
|---|---|---|
| S1 | Accuracy 不显著退化 | CUB-200 或 ImageNet-R 至少一个数据集上，accuracy drop <= 0.5pp vs HC-SOINN raw |
| S2 | Memory 显著降低 | compact deployable memory 降低 >= 30% |
| S3 | Implementation memory 不膨胀 | actual implementation memory 不高于 raw HC-SOINN，或明确标为"非部署版 / compact potential only" |
| S4 | Memory-Pareto 成立 | 至少一个预算点不被 raw HC-SOINN 支配 |
| S5 | Lifecycle 有独立贡献 | 至少改善一个 stability 指标：worst-order、Old-New HM、stale candidate precision、high-risk node suppression |

### 8.2 Edge 进入主文的附加条件

| 编号 | 标准 | 量化阈值 |
|---|---|---|
| S6 | Edge 有实际使用 | edge_use_rate >= 1% |
| S7 | Edge 有独立正向影响 | edge ablation 对 margin、Old-New HM 或 worst-order 至少一个指标有稳定正向影响 |

### 8.3 完整验证成立条件

| 编号 | 标准 | 量化阈值 |
|---|---|---|
| S8 | 与 STAR 可比 | 同协议、同 backbone、同 seeds 下，LifeTopoDict 与 HC-SOINN+STAR gap <= 0.5pp 且 memory 更低，或 Avg/Final/Old-New HM 统计上超过 HC-SOINN+STAR |
| S9 | 三模块独立证据 | dictionary-coding/lifecycle/edge 三者分别在 memory/stability/edge-use 上有独立证据 |

### 8.4 失败判定

若最终只能写成 "HC-SOINN with dictionary bottleneck"，且 lifecycle/edge 没有独立诊断贡献，应停止把 LifeTopoDict 作为主线。

---

## 附录 A：源码关键位置速查表

| 功能 | 文件 | 函数/类 | 行号 |
|---|---|---|---|
| 节点数据结构 | `utils/hc_soinn_classifier.py` | `_Cluster.__init__` | L28-36 |
| 层次聚类 | `utils/hc_soinn_classifier.py` | `_hierarchical_cluster` | L38-67 |
| SOINN 精炼 | `utils/hc_soinn_classifier.py` | `_simplified_soinn_on_clusters` | L119-245 |
| Degree-based 删除 | `utils/hc_soinn_classifier.py` | `_simplified_soinn_on_clusters` | L202-230 |
| Edge age 丢失点 | `utils/hc_soinn_classifier.py` | `_simplified_soinn_on_clusters` | L240-243 |
| 并行 compress worker | `utils/hc_soinn_classifier.py` | `_compress_class_worker` | L247-285 |
| 分类器初始化 | `utils/hc_soinn_classifier.py` | `HCSOINNClassifier.__init__` | L287-364 |
| Freeze nodes | `utils/hc_soinn_classifier.py` | `freeze_nodes` | L505-517 |
| Add features | `utils/hc_soinn_classifier.py` | `add_features` | L605-641 |
| Compress | `utils/hc_soinn_classifier.py` | `compress` | L643-718 |
| class_edges 类型声明 | `utils/hc_soinn_classifier.py` | `compress` | L699-701 |
| Predict cache | `utils/hc_soinn_classifier.py` | `_ensure_predict_cache` | L436-500 |
| Predict topk | `utils/hc_soinn_classifier.py` | `predict_topk` | L722-918 |
| Score fusion | `utils/hc_soinn_classifier.py` | `predict_topk` | L899-901 |
| STAR 初始化 | `utils/STAR.py` | `STARAligner.__init__` | L23-42 |
| STAR anchor 存储 | `utils/STAR.py` | `select_anchors_for_current_task` | L44-136 |
| STAR drift 应用 | `utils/STAR.py` | `align_old_classes` | L160-282 |
| 模型集成 | `models/simplecil_hc_soinn.py` | `Learner` | L22-415 |
| after_task compress 调用 | `models/simplecil_hc_soinn.py` | `after_task` | L45-50 |
| _train 特征提取 | `models/simplecil_hc_soinn.py` | `_train` | L130-132 |

---

## 附录 B：死代码与源码发现

Agent D 源码分析发现的以下死代码/设计特征影响融合方案：

| 发现 | 位置 | 影响 |
|---|---|---|
| `tau_merge` / `tau_reject` / `soinn_lam` / `soinn_threshold_scale` 为死代码 | `__init__` L295-296, L300-301 | 不影响融合，但说明原始论文部分机制未在源码中实现 |
| SOINN 精炼输入是聚类质心而非原始样本 | `_compress_class_worker` L260-265 | 字典编码在质心层面执行，样本量更少，编码更快 |
| STAR 在 `simplecil_hc_soinn.py` 中没被调用 | `models/simplecil_hc_soinn.py` 全文 | 验证了 P0 不启用 STAR 的决策 |
| 推理复杂度 O(N x C x 61 x D) | `predict_topk` 全流程 | 字典编码 materialization 在 cache 构建时一次性执行，不增加推理复杂度 |
| `_Cluster.center` 存储归一化值，`center_raw` 存储未归一化值 | `_Cluster` L28-36 | 字典编码替换 `center_raw` 的存储方式，`center` 改为存储 `v_hat` |
| `class_edges` 类型 `Dict[int, Set[int]]` 丢失 age | `compress` L699 | P0 修复为 `Dict[int, Dict[int, int]]` |

---

## 附录 C：字典规模 M 预算分析

### C.1 Memory 估算

| 组件 | HC-SOINN Raw | LifeTopoDict (M=200) | 说明 |
|---|---|---|---|
| 节点 center_raw (200 类 x 20 节点) | 200 x 20 x 768 x 4B = 11.7 MB | — | 不再存储 |
| 节点 center (normalized) | 11.7 MB | 11.7 MB | 保留 |
| 字典 D | — | 200 x 768 x 4B = 0.6 MB | 新增 |
| 稀疏系数 a (k=5) | — | 200 x 20 x 5 x 4B = 0.08 MB | 新增 |
| STAR image anchors | 200 x 20 x 224x224x3 = 36 MB | — | P0 不启用 |
| **总计** | **~36.4 MB** | **~3.0 MB** | **91.8% 压缩** |

来源：Agent F 风险审查（"Memory 优势：3.0MB vs 36.4MB（91.8% 压缩）"）。

### C.2 字典规模 M 的选择

| 场景 | 推荐 M | 依据 |
|---|---|---|
| CUB-200 (200 类, D=768) | M = 100-200 | 每类约 1 个 atom，冗余度 ~1x |
| ImageNet-R (200 类, D=768) | M = 200-400 | 域偏移需要更多方向 |
| CIFAR-100 (100 类, D=768) | M = 50-100 | 较少类，较简单特征 |

消融应扫描 M in {100, 200, 400, 800}，k in {3, 5, 8}。

来源：Phase3A X9（字典容量与任务序列长度的关系）；Phase3A A16/F1。

---

*文档生成日期：2026-05-21*
*基于 Agent A-F 全部发现 + 总指挥裁决*
*所有设计决策可追溯至具体来源标注*
