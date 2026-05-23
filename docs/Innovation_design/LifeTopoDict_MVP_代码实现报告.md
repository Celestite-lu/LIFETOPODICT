# LifeTopoDict MVP 代码实现报告

本文档说明当前仓库中 LifeTopoDict MVP 的代码实现。报告以代码真实路径为准，覆盖入口、feature cache / classifier-only 运行路径、训练/评估时序、HC-SOINN 压缩、字典编码、归一化语义、生命周期、边元数据、memory ledger、运行时资源监控、配置与验证方式。Residual dictionary growth 与 additive edge-aware scoring 的代码保留作历史复现，但已关闭，后续不再纳入实验设置。

## 1. 实现范围

当前实现的核心目标是：在保留 HC-SOINN 的 class-local topology node 身份和 class-local edge graph 的前提下，将长期 prototype center 存储改为共享字典 atom 的稀疏编码表示，并增加可审计的 lifecycle 与 memory report。

主要代码文件：

| 文件 | 责任 |
|---|---|
| `models/life_topo_dict.py` | LifeTopoDict learner；接入 frozen ViT feature 提取、task-boundary 压缩、诊断输出 |
| `models/simplecil_hc_soinn.py` | raw HC-SOINN baseline learner；与 LifeTopoDict 共享评估前压缩、feature-cache classifier-only 路径 |
| `utils/hc_soinn_classifier.py` | 核心实现；HC-SOINN 节点压缩、共享字典、稀疏编码、materialization、lifecycle、edge age/reliability、推理；growth/edge additive scoring 仅保留为历史路径 |
| `utils/feature_cache.py` | frozen backbone feature cache；strict cache 下支持跳过 backbone 初始化、构造 cached-feature dataset |
| `main.py` | JSON 配置加载和 LifeTopoDict CLI 覆盖参数 |
| `trainer.py` | 通用增量学习时序；支持 no-backbone 模型参数量日志与 train-only profiling 开关 |
| `utils/ltd_metrics.py` | 独立诊断函数：DGR、AEPMI、PAD、memory report、edge consistency、same-memory 对比 |
| `scripts/collect_results.py` | 从日志收集 accuracy、diagnostics、edge scoring 指标 |
| `scripts/run_with_gpu_memory_monitor.py` | 用 `nvidia-smi` 采样 GPU memory.used，输出 peak/mean delta |
| `scripts/run_with_cpu_memory_monitor.py` | 采样 launched process tree 的 RSS/PSS/USS，覆盖 multiprocessing worker |
| `exps/life_topo_dict/*.json` | CIFAR/CUB/ImageNet-R 与消融配置 |

## 2. 总体数据流

### 2.1 运行入口

命令行入口在 `main.py`。

1. `main.py` 读取 `--config` 指定的 JSON。
2. JSON 参数进入 `args_dict`。
3. 命令行中显式传入的 LifeTopoDict 参数会覆盖 JSON。
4. `trainer.train(args_dict)` 启动增量学习流程。
5. `utils/factory.py` 根据 `model_name="life_topo_dict"` 加载 `models/life_topo_dict.py::Learner`。

CLI 可覆盖的 LifeTopoDict 参数包括：

| 参数 | 类型 | 默认语义 |
|---|---:|---|
| `dict_sparse_k` | int | Top-K Ridge support size |
| `dict_ridge_lambda` | float | Ridge 正则 |
| `lifecycle_theta_support` | float | old support 保护阈值 |
| `lifecycle_theta_usage` | float | usage 低使用阈值 |
| `lifecycle_T_inactive` | int | 连续低 usage 后 inactive 的 task 数 |
| `dict_theta_residual` | float | 历史 residual-triggered growth 阈值；后续实验不搜索 |
| `dict_max_growth_per_task` | int | 历史 growth cap；后续实验固定 0 |
| `use_dict_coding` | bool | 是否启用共享字典编码；关闭时退化为 raw/no-dict |
| `use_dictionary_growth` | bool | residual growth 历史开关；后续实验固定 false |
| `use_lifecycle` | bool | 是否启用 lifecycle 三态转换 |
| `use_protected_gate` | bool | 是否启用 protected node SOINN gate |
| `lifecycle_node_inactive_threshold` | float | node top-k inactive atom 比例阈值 |
| `use_edge_age_persistence` | bool | 是否保存 edge age |
| `use_edge_aware_scoring` | bool | additive edge-aware scoring 历史开关；后续实验固定 false |
| `edge_score_gamma` | float | 历史 edge scoring 参数；后续实验不搜索 |
| `edge_score_eta` | float | 历史 edge scoring 参数；后续实验不搜索 |
| `use_raw_fallback_gate` | bool | Direction 1 raw-node fallback gate 开关 |
| `raw_fallback_per_class` | int | 每类最多保留多少 raw fallback node snapshots |
| `raw_fallback_select_by` | str | raw fallback cache 排序策略，当前常用 `count` / `residual` |
| `fallback_gate_type` | str | `rule` / `random` / `static` / `oracle` / `logistic` / `ridge` |
| `fallback_calibration_samples_per_class` | int | learned gate 的每类 held-out calibration 样本数 |
| `fallback_calibration_cumulative` | bool | learned gate 是否使用跨任务累积 held-out calibration set |
| `fallback_gate_min_positives` | int | calibration 正例不足时关闭 learned gate 的保护阈值 |
| `fallback_gate_min_calibration_gain` | float | calibration 增益不足时关闭 learned gate 的保护阈值 |
| `use_feature_cache` | bool | 是否使用磁盘上的 frozen-backbone feature cache |
| `feature_cache_dir` | str | feature cache 根目录 |
| `feature_cache_strict` | bool | cache 缺失或不匹配时是否直接报错 |
| `feature_cache_dtype` | str | cache dtype 记录/覆盖，当前支持 `float32` / `float16` |
| `feature_cache_skip_backbone` | bool | strict cache 下是否跳过 backbone 初始化；默认 true |
| `feature_cache_classifier_device` | str | cached-feature 推理时 classifier 距离计算设备；默认 `cpu` |
| `seed` | int/list | 覆盖 JSON seed list |
| `max_tasks` | int | smoke run 的 task 数限制 |
| `training_time_profile_last_task` | bool | 启用 trainer 中的 train-only profiling 路径 |
| `skip_eval_for_training_time_profile` | bool | profiling 时跳过 eval，用于训练期资源测量 |

### 2.2 Feature cache 与 classifier-only 路径

`utils/feature_cache.py` 负责 cached feature 的 key、metadata、dataset wrapper 与 loader 日志。当前实验使用的主路径是 strict feature-cache：

```bash
--use_feature_cache true \
--feature_cache_strict true \
--feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict
```

在 strict feature-cache 下，`feature_cache_classifier_only(args)` 默认返回 true：

```text
use_feature_cache == true
feature_cache_strict == true
feature_cache_skip_backbone 默认为 true
```

此时 `models/life_topo_dict.py` 与 `models/simplecil_hc_soinn.py` 都不会构造 `SimpleVitNetKNN`：

```text
self._network = None
```

训练与评估均直接消费 cached features：

```text
train_loader_for_hc -> cached train features -> hc_soinn.add_features()
test_loader         -> cached test features  -> hc_soinn.predict_topk()
```

如果 cached feature 缺失且 strict=true，cache dataset 构建会报错，避免意外回退到不公平的 backbone 前向路径。如果需要保留旧的 profiling 行为，可以显式设置：

```bash
--feature_cache_skip_backbone false
```

classifier-only 默认把 `predict_topk()` 的 torch 距离计算放在 CPU：

```bash
--feature_cache_classifier_device cpu
```

如果确实需要 GPU 距离计算，可显式设置：

```bash
--feature_cache_classifier_device cuda
```

注意：即使 `predict_topk()` 使用 GPU，HC-SOINN 压缩、层次聚类、SOINN refinement、字典稀疏编码、lifecycle 更新仍主要是 NumPy/SciPy/CPU 路径。

`get_cached_feature_dataset_split()` 额外支持 per-class train/calibration split。该函数直接在 cached feature mmap 上切分 row positions，不重新前向 backbone；learned raw fallback gate 使用这个 split 构造 held-out calibration loader。`is_cached_feature_dataset()` 已支持 `ConcatDataset`，因此跨任务累积 calibration set 在 classifier-only 模式下仍会被识别为 cached-feature loader。

### 2.3 Direction 1 raw-node fallback gate

Direction 1 的实现目标是：在 compact dictionary-coded topology 之外，保留少量 raw node snapshots，并用 reliability gate 只在 compact prediction 不可靠时切换到 raw fallback prediction。相关代码主要在 `utils/hc_soinn_classifier.py` 与 `models/life_topo_dict.py`。

`HCSOINNClassifier` 新增状态：

| 字段 | 含义 |
|---|---|
| `raw_fallback_cache` | `{class_id: [raw snapshot]}`，每个 snapshot 记录 raw center、class、node id、count、residual、node state |
| `_raw_fallback_predict_cache` | 推理时使用的 tensor/array cache，包含 raw fallback centers、labels、counts、residuals、states |
| `_prediction_trace_records` | compact/fallback 对比 trace，用于 calibration 与诊断 |
| `_raw_fallback_gate_model` | learned ridge/logistic gate 的 feature mean/std、weights、bias、threshold |
| `_raw_fallback_gate_fit_stats` | calibration 样本数、正例数、compact/fallback/oracle/final accuracy、fallback rate、是否禁用 |

当前 learned gate 使用 ridge-style linear scorer。输入 reliability features 包括 compact margin、selected node residual、selected/fallback distance、distance gap、selected/fallback count、fallback residual、selected node lifecycle state one-hot。训练目标是 `compact_wrong && fallback_correct`，threshold 在 held-out calibration trace 上选择，并带两层保护：

1. `positives < fallback_gate_min_positives` 时禁用 gate；
2. calibration final accuracy 相对 compact 的增益低于 `fallback_gate_min_calibration_gain` 时禁用 gate。

`models/life_topo_dict.py` 的训练顺序为：

1. 从 cached train features 中为当前新类切出 classifier train 和 held-out calibration。
2. 用 classifier train 更新 HC-SOINN / LifeTopoDict nodes。
3. `compress()` 后构建 raw fallback cache 与 compact materialized nodes。
4. 若使用 learned gate，将当前 calibration dataset 追加进跨任务 cumulative calibration list。
5. 在 cumulative calibration loader 上跑 `_eval_cnn()` 收集 trace。
6. 调用 `fit_raw_fallback_gate_from_trace()` 拟合 gate。

该流程不使用 test set 调 gate。`oracle` gate 只用于上界实验，不可作为可部署结果。

### 2.4 Task 时序

仓库的通用 `trainer.py` 时序是：

```text
model.incremental_train(data_manager)
eval_results = model.eval_task()
model.after_task()
```

为了保证当前 task 的评估使用已经压缩、已经字典编码的节点，LifeTopoDict learner 在 `models/life_topo_dict.py::_train()` 内完成压缩：

```text
_train()
  -> _extract_class_features(train_loader_for_hc)
  -> hc_soinn.add_features(...)
  -> _compress_task_boundary()
       -> hc_soinn.set_task_boundary(self._known_classes)
       -> hc_soinn.compress()
```

因此 `eval_task()` 执行时，`class_clusters`、`dict_atoms_hat`、node coefficients、materialized centers、lifecycle states 都已经更新。`after_task()` 只做：

1. 如果 `_train()` 未完成压缩，则兜底压缩。
2. 输出 diagnostics 和 memory report。
3. 将 `_known_classes` 更新为 `_total_classes`。
4. 重置 `_hc_soinn_compressed_for_task=False`，等待下一 task。

raw HC-SOINN baseline `models/simplecil_hc_soinn.py` 也采用同样的“评估前压缩”时序，避免 raw baseline 与 LifeTopoDict 使用不同 task boundary。

在 classifier-only feature-cache 路径下，`trainer.py` 会检测 `model._network is None`，并输出：

```text
All params: 0 (backbone skipped; classifier-only feature-cache mode)
Trainable params: 0 (backbone skipped; classifier-only feature-cache mode)
```

这只是说明 backbone learner 已跳过；classifier state 仍由 `HCSOINNClassifier` 持有并在每个 task 更新。

## 3. 核心数据结构

### 3.1 `_Cluster`

`utils/hc_soinn_classifier.py::_Cluster` 表示 HC-SOINN class-local topology node。

字段：

| 字段 | 类型 | 语义 |
|---|---|---|
| `center` | `np.ndarray[d]` | L2-normalized node center，推理主用 |
| `center_raw` | `np.ndarray[d]` | 原始/兼容 center；在 LifeTopoDict materialize 后是 `D_hat @ a`，不再是真实 raw mean |
| `count` | int | 聚类/节点样本计数或 SOINN win count |
| `coeff` | `Optional[np.ndarray[1, M]]` | node 对共享字典的稀疏系数 |
| `residual` | float | `1 - cos(v, D_hat @ a)` |
| `node_state` | str | `protected` / `plastic` / `inactive` |

重要归一化约定：

```python
self.center = _normalize(center)
```

也就是说，任何进入 `_Cluster(center=...)` 的 center 都会被 `_normalize()` 处理。`center_raw` 不会在构造函数中归一化，只做 copy。

### 3.2 `HCSOINNClassifier` 的主要状态

| 字段 | 语义 |
|---|---|
| `class_mu` | 每类 normalized NCM center |
| `class_mu_raw` | 每类 raw mean center |
| `class_count` | 每类累计样本数 |
| `class_clusters` | `{class_id: List[_Cluster]}`，class-local topology nodes |
| `class_edges` | `{class_id: {node_i: {node_j: age}}}`，class-local edge age |
| `class_edge_reliability` | `{class_id: {node_i: {node_j: reliability}}}` |
| `buffers` | 当前未压缩 feature buffer |
| `dict_atoms_hat` | `[M, d]`，row L2-normalized shared dictionary，唯一真值 |
| `dict_atoms` | `[M, d]`，当前等于 `dict_atoms_hat`，保留兼容字段 |
| `atom_states` | 每个 atom 的 lifecycle state |
| `atom_origins` | `base` 或 `grown` |
| `atom_usage` | 累计 usage 诊断计数 |
| `atom_usage_ema` | lifecycle 使用的 EMA usage |
| `node_states` | `{class_id: List[str]}`，与 `_Cluster.node_state` 同步 |
| `old_support` | 预留字段；当前 old support 在 `_compute_old_support()` 中实时计算 |
| `inactive_mask` | atom active mask，`True` 表示非 inactive |
| `_low_usage_streak` | 每个 atom 连续低 usage task 数 |
| `_predict_cache` | 推理缓存，存放 tensor 化后的 NCM/prototype/label/index |
| `known_classes_before_task` | 当前 task 之前已知类别边界，用于 old support 过滤 |

## 4. 向量与归一化语义

### 4.1 基础归一化函数

代码中统一使用 `_normalize(v)`：

```python
norm = np.linalg.norm(v)
if norm < 1e-8:
    return v.astype(np.float32, copy=True)
return (v / norm).astype(np.float32, copy=True)
```

要点：

1. 小范数向量不会除以接近 0 的数，而是原样转为 `float32`。
2. 正常向量返回 L2 normalized `float32` copy。
3. 函数不修改输入数组。

cosine distance 使用 `_cosine_distance(a, b)`：

```text
a_norm = a / (||a|| + 1e-8)
b_norm = b / (||b|| + 1e-8)
distance = 1 - dot(a_norm, b_norm)
```

### 4.2 类均值归一化

`add_features(features, labels)` 负责把 backbone feature 加入 classifier。

输入处理：

```python
features = np.asarray(features, dtype=np.float32)
labels = np.asarray(labels, dtype=np.int64)
```

对每个 class：

```text
class_sum_new = class_mu_raw_old * old_count + sum(new_features)
class_count_new = old_count + n_new
class_mu_raw = class_sum_new / class_count_new
class_mu = _normalize(class_mu_raw)
```

因此：

| 字段 | 是否 L2 normalized |
|---|---|
| `class_mu_raw[c]` | 否，真实 raw mean |
| `class_mu[c]` | 是 |

### 4.3 HC-SOINN 聚类中的归一化

`_compress_class_worker()` 首先构造：

```text
feats_raw = feats.astype(float32)
feats_norm = feats_raw / (||feats_raw||_row + 1e-8)
```

层次聚类 `_hierarchical_cluster(feats_norm, feats_raw, ...)` 使用 `feats_norm` 做 clustering。每个 cluster：

```text
center = mean(normalized member features)
center_raw = mean(raw member features)
_Cluster(center, count, center_raw=center_raw)
```

由于 `_Cluster` 构造函数会 `_normalize(center)`，所以最终 `cluster.center` 是 normalized mean direction；`cluster.center_raw` 是未归一化 raw mean。

SOINN refinement `_simplified_soinn_on_clusters()`：

1. 初始 `nodes = [_normalize(c.copy()) for c in cluster_centers]`。
2. 输入信号 `x` 每次处理时做 `x_norm = _normalize(x)`。
3. winner `s1` 和 neighbor 的 normalized center 通过 `_spherical_interpolate()` 更新。
4. `_spherical_interpolate()` 的输入输出都保证 normalized。
5. `nodes_raw` 用线性增量更新，不归一化：

```text
nodes_raw[s1] = nodes_raw[s1] + eta1 * (x_raw - nodes_raw[s1])
nodes_raw[nbr] = nodes_raw[nbr] + eta2 * (x_raw - nodes_raw[nbr])
```

最终返回 `_Cluster(nodes[old_idx], ..., center_raw=nodes_raw[old_idx])`，所以 `center` 继续 normalized，`center_raw` 继续保留 raw-like vector。

### 4.4 字典 atom 归一化

当前 P0 约定：

```text
dict_atoms_hat: [M, d]，每一行 L2 normalized，唯一真值
dict_atoms:     [M, d]，当前等于 dict_atoms_hat，用于兼容旧字段
```

初始化 `dict_init()`：

```text
atoms = stack(class_mu_raw)
atoms_hat = atoms / clip(||atoms||_row, min=1e-8)
dict_atoms = atoms_hat.copy()
dict_atoms_hat = atoms_hat
```

新类注册 `dict_grow_for_new_classes()` 也对 `class_mu_raw` 做同样 row L2 normalize。

历史 residual growth `dict_grow()` 增长的 atom 是 normalized residual direction。该逻辑保留用于复现实验，但后续实验默认 `use_dictionary_growth=false` 且 `dict_max_growth_per_task=0`：

```text
v_norm = center_raw / (||center_raw|| + 1e-8)
v_hat = a @ D_hat
v_hat_norm = v_hat / (||v_hat|| + 1e-8)
r_vec = v_norm - v_hat_norm
new_atom = r_vec / ||r_vec||
```

因此 base atom 和 grown atom 的归一化语义一致。

### 4.5 稀疏编码与重构归一化

`sparse_encode(v)` 支持单向量 `[d]` 或 batch `[N, d]`。

步骤：

1. 取 `D_hat = self.dict_atoms_hat`，shape `[M, d]`。
2. 排除 inactive atoms：

```text
active_mask[m] = atom_states[m] != 'inactive'
D_active = D_hat[active_indices]
```

3. 对输入目标向量做 L2 normalize，仅用于 Top-K 选择和 residual：

```text
v_norm = v / (||v||_row + 1e-8)
cos_sim = v_norm @ D_active.T
```

4. 选 Top-K active atoms：

```text
top_k_indices = argpartition(-cos_sim, k)
```

如果 active atom 数小于等于 k，则使用所有 active atoms。

5. 对每个样本独立做 Ridge：

```text
D_S: [k, d]
gram = D_S @ D_S.T + lambda * I
a_support = solve(gram, D_S @ v_i)
```

注意：Top-K 用的是 normalized target `v_norm`，但 Ridge RHS 使用原始 `v_i`。这使系数可以吸收目标向量的幅度；最终 residual 仍在 cosine 空间计算。

6. 把 active coefficient 映射回完整 M 维：

```text
coeff_out[:, active_indices] = coeff_full
```

7. 重构和 residual：

```text
recon = coeff_out @ D_hat
recon_norm = recon / (||recon||_row + 1e-8)
residual = 1 - sum(v_norm * recon_norm)
```

### 4.6 Node materialization

`_materialize_nodes()` 把 sparse coefficient 变成推理用 node center：

```text
a = node.coeff.ravel()
z = a @ D_hat
if ||z|| > 1e-8:
    node.center = z / ||z||
node.center_raw = z
```

关键语义：

| 字段 | materialize 后含义 |
|---|---|
| `node.center` | `normalize(D_hat @ a)`，推理主用 normalized center |
| `node.center_raw` | `D_hat @ a`，未归一化重构向量；不是原始 raw mean |

如果 `z` 近零，代码保留旧的 `node.center`，但仍把 `node.center_raw` 设为 `z`。

## 5. HC-SOINN 压缩实现

### 5.1 Feature buffer

`add_features()` 将当前 task 的 backbone feature 按 class 放入 `buffers[class_id]`。同一 class 多次加入时，buffer 是 list，每次 append 一个 `[N_c, d]` array。

### 5.2 压缩输入

`compress()` 对每个有 buffer 的 class 执行：

```text
feats = concat(buffer chunks)
if class already has clusters:
    old_centers_raw = stack(previous cluster.center_raw)
    feats = concat(feats, old_centers_raw)
```

这意味着旧节点的 raw/compat center 会作为额外信号参与新一轮压缩，避免 class-local topology 完全由当前 task buffer 覆盖。

### 5.3 hierarchical clustering

每个 class 的 feature 进入 `_compress_class_worker()`，worker 内：

1. `feats_raw = feats.astype(np.float32)`
2. `feats_norm = feats_raw / (row_norm + 1e-8)`
3. `_hierarchical_cluster(feats_norm, feats_raw, target_k, linkage_method, distance_metric)`

`target_k`：

```text
target_k = feats.shape[0] if max_prototypes_per_class is None
           else min(max_prototypes_per_class, feats.shape[0])
```

默认配置中 `hcsoinn_max_proto_per_class=60`。

### 5.4 simplified SOINN refinement

如果 `use_soinn_refinement=True` 且 cluster 数大于 1，会执行 `_simplified_soinn_on_clusters()`。

初始 edge：

```text
for each node i:
    nearest_idx = argmin cosine_distance(i, j)
    edges[i][nearest_idx] = 0
    edges[nearest_idx][i] = 0
```

每次输入信号：

1. 找距离最近的两个节点 `s1, s2`。
2. 如果没有边，创建 `s1-s2` age 0。
3. `s1` 的所有边 age +1。
4. 如果 age > `soinn_ad`，删除该边。
5. 用递减学习率更新 winner 和 neighbor：

```text
eta1 = 1 / (t + iteration * n_signals + 1)
eta2 = 1 / (100 * (t + iteration * n_signals + 1))
```

6. normalized center 用 spherical interpolation；raw center 用线性插值。
7. 低度节点删除：

```text
to_remove = nodes with degree <= max_degree_for_removal
```

但 protected nodes 不进入删除集合。

### 5.5 Protected gate

protected gate 分两层：

1. worker 前：`compress()` 从旧 node state 中收集 protected old centers。
2. worker 内：用 `_match_protected_indices(protected_centers, clusters, threshold=0.7)` 把旧 protected center 映射到新 hierarchical clusters。

匹配逻辑：

```text
old_center normalized
sim = new_centers @ old_center
greedy one-to-one matching
only accept sim > 0.7
```

SOINN refinement 中 protected node：

1. 不做普通 winner center update。
2. 不做 neighbor center update。
3. 不参与低度删除。
4. 如果其他节点被删除，会同步 remap protected index。

worker 完成后，`compress()` 还会再次 old-to-new matching，把 protected state 写回 `node_states[class_id][idx]` 和 `_Cluster.node_state`。

## 6. 字典编码流程

`compress()` 在 class-local cluster 更新后进入 LifeTopoDict 字典流程：

```text
if use_dict_coding:
    if dictionary not initialized:
        dict_init()
    else:
        dict_grow_for_new_classes()

    _encode_nodes_with_dict()
    dict_grow()
    _materialize_nodes()
    _update_lifecycle_states()
    compute_diagnostics()
    invalidate_cache()
```

### 6.1 Dictionary initialization

首次 `compress()` 后调用 `dict_init()`：

1. 按 class id 排序收集所有 `class_mu_raw`。
2. stack 成 `[M0, d]`。
3. row L2 normalize。
4. 写入 `dict_atoms` 和 `dict_atoms_hat`。
5. 初始化：

```text
atom_states = ['plastic'] * M0
atom_origins = ['base'] * M0
atom_usage[m] = 0
atom_usage_ema[m] = 0
```

### 6.2 New-class atom registration

后续 task 中，`dict_grow_for_new_classes()` 查找没有 encoded node 的 class：

```text
encoded_classes = classes with at least one node.coeff is not None
new_classes = class_mu_raw.keys() - encoded_classes
```

对这些新类：

1. 用 `class_mu_raw` 创建新 atom。
2. row L2 normalize。
3. append 到 `dict_atoms` 和 `dict_atoms_hat`。
4. metadata 初始化为 plastic/base。
5. 所有已有 node coefficient 从 `[1, M_old]` zero-pad 到 `[1, M_new]`。
6. 只对新类 nodes 做 sparse_encode。

### 6.3 Node encoding

`_encode_nodes_with_dict()` 遍历所有 `class_clusters`：

1. 跳过 `node_state == 'inactive'`。
2. 用 `node.center_raw` 作为 target 调用 `sparse_encode()`。
3. 写入：

```text
node.coeff = coeff.reshape(1, -1)
node.residual = residual
```

4. 统计 per-task usage：

```text
support_indices = nonzero(abs(coeff) > 1e-6)
atom_usage_task[i] += 1
atom_usage[i] += 1
```

5. EMA usage：

```text
task_rate = atom_usage_task[m] / total_encoded
atom_usage_ema[m] = 0.7 * old + 0.3 * task_rate
```

### 6.4 Residual-triggered growth（历史路径，后续关闭）

`dict_grow()` 只在 `use_dictionary_growth=True` 时生效。根据 2026-05-22 实验决策，growth 无可测下游收益且增加 memory，后续实验固定：

```text
use_dictionary_growth = false
dict_max_growth_per_task = 0
```

本节仅说明保留代码的历史行为。

触发条件：

```text
node.residual > theta_residual
node.coeff is not None
node_state != inactive
```

高 residual nodes 按 residual 降序排序，最多取 `max_growth_per_task` 个。

对每个高 residual node：

```text
v_hat = a @ D_hat
v_norm = normalize(center_raw)
v_hat_norm = normalize(v_hat)
r_vec = v_norm - v_hat_norm
new_atom = normalize(r_vec)
```

新 atom append 后：

1. 所有 coefficient zero-pad。
2. 只 re-encode 触发 growth 的 nodes。
3. 记录 `residual_before` 和 `residual_after` 到 `_gte_records`。

GTE 计算：

```text
GTE = mean((residual_before - residual_after) / residual_before)
```

## 7. Lifecycle 实现

### 7.1 Atom state

当前 P0 三态：

| 状态 | 含义 |
|---|---|
| `protected` | old support 高；参与推理；不被普通 SOINN 更新/低度删除 |
| `plastic` | 默认状态；可编码、可参与 grow、可更新 |
| `inactive` | 不参与编码、读出、更新；物理 prune 暂缓 |

### 7.2 Usage

`_compute_atom_usage()` 优先返回 `atom_usage_ema`。如果 EMA 尚未建立，则 fallback：

1. 遍历所有 node coefficient。
2. 每个 node 取 abs coefficient 的 Top-K。
3. 统计 atom hit count。
4. 除以 total selections 得到 usage rate。
5. 用 fallback usage 初始化 EMA。

### 7.3 Old support

`_compute_old_support()` 使用 `known_classes_before_task` 区分旧类：

```text
old class: class_id < known_classes_before_task
current/new class: class_id >= known_classes_before_task
```

只统计旧类 nodes 的 coefficient：

```text
atom_abs_sum += abs(coeff)
node_count += 1
mean_support = atom_abs_sum / node_count
old_support = mean_support / max(mean_support)
```

如果没有 old node，所有 support 为 0。

### 7.4 State transition

`_update_lifecycle_states()`：

如果 `use_lifecycle=False`：

```text
all atoms -> plastic
all nodes -> plastic
inactive_mask = all True
```

如果启用 lifecycle：

```text
if usage[m] < theta_usage:
    low_usage_streak[m] += 1
else:
    low_usage_streak[m] = 0

if old_support[m] > theta_support:
    atom_state[m] = protected
elif low_usage_streak[m] >= T_inactive and old_support[m] < theta_support:
    atom_state[m] = inactive
else:
    atom_state[m] = plastic
```

`inactive_mask[m] = atom_state[m] != 'inactive'`。

### 7.5 Node state derivation

node state 从 node coefficient 的 Top-K atom states 派生：

```text
topk_idx = argsort(abs(coeff))[::-1][:k]
topk_states = atom_states[topk_idx]

if any protected:
    node_state = protected
elif all inactive:
    node_state = inactive
else:
    node_state = plastic
```

推理缓存构建时，LifeTopoDict 会过滤 inactive nodes；如果某个 class 全部 nodes inactive，则 fallback 到该 class 的 NCM center。

## 8. Edge age、reliability 与 edge-aware scoring

### 8.1 Edge age persistence

`class_edges[class_id]` 保存：

```text
{
  node_i: {
    node_j: age
  }
}
```

SOINN refinement 中：

1. 新边 age=0。
2. winner 的邻边每次被访问 age +1。
3. age > `soinn_ad` 的边被删除。
4. 节点删除/重编号时同步 remap edge endpoint。

如果 `use_edge_age_persistence=False`，`compress()` 会把所有 edge age 写成 0，但仍保留 topology adjacency。

### 8.2 Edge reliability

`_compute_edge_reliability(clusters, edges)` 在压缩后计算：

```text
endpoint_sim = max(0, dot(center_i, center_j))
age_factor = 1 / (1 + age)
reliability = endpoint_sim * age_factor
```

因为 `center_i` 和 `center_j` 都是 normalized center，所以 dot product 是 cosine similarity。age 越大，reliability 越低。

### 8.3 Edge-aware scoring（历史路径，后续关闭）

P1 开关 `use_edge_aware_scoring` 默认关闭。根据 2026-05-22 实验决策，additive edge-aware scoring 默认配置有害，后续不再纳入实验设置。本节仅说明保留代码的历史行为。

开启后，推理时 coarse filtering 被禁用，以便获得完整 `dist_proto_all`。对每个 sample、每个 class：

1. 找该 class 最近的 prototype/node。
2. 读取该 node 的 class-local neighbors。
3. 对每个 neighbor：

```text
neighbor_sim = max(0, 1 - distance_to_neighbor_proto)
support_term = neighbor_sim * edge_reliability
risk_term = age / (age + 1)
```

4. 聚合：

```text
edge_support = mean(support_terms)
edge_risk = mean(risk_terms)
delta = edge_score_eta * edge_risk - edge_score_gamma * edge_support
```

5. 因为 classifier 的 distance 越小越好，最终：

```text
final_scores = alpha * dist_ncm + (1 - alpha) * dist_sub + delta
```

若 support 强，`delta` 可能为负，降低该 class distance；若 edge age/risk 高，`delta` 为正，增加该 class distance。

诊断指标：

| 指标 | 含义 |
|---|---|
| `edge_use_rate` | 至少一个 class 使用 edge adjustment 的样本比例 |
| `edge_class_use_rate` | sample-class 粒度的 edge 使用比例 |
| `edge_margin_contribution` | 平均 edge support |
| `edge_risk_penalty` | 平均 edge risk |
| `edge_score_adjustment` | 平均 score delta |

## 9. 推理实现

### 9.1 Predict cache

`_ensure_predict_cache(device, query_dim, valid_classes)` 构建缓存：

| cache key | shape | 说明 |
|---|---:|---|
| `ncm_centers_t` | `[C, d]` | normalized class means |
| `all_protos_t` | `[P, d]` | active node centers；若 class 没有 active node，fallback NCM |
| `proto_labels_t` | `[P]` | prototype 对应原始 class id |
| `proto_class_index_t` | `[P]` | prototype 对应 `valid_classes` 内部 index |
| `proto_node_index_t` | `[P]` | class-local node index；fallback NCM 为 -1 |
| `proto_lookup` | dict | `(class_id, node_idx) -> proto position` |

cache key 由 `device`、`query_dim`、`valid_classes` 组成。`compress()`、`_materialize_nodes()` 等修改 prototype 的操作会调用 `invalidate_cache()`。

在 strict feature-cache classifier-only 路径下，`device` 默认来自：

```text
feature_cache_classifier_torch_device(args, fallback_device)
```

默认值是 CPU，因此 `predict_topk()` 不会触发新的 GPU resident state。若命令行指定 `--feature_cache_classifier_device cuda`，则只把推理中的 torch tensor distance 计算放到 GPU；训练阶段的压缩和字典更新仍在 CPU。

### 9.2 Query normalization

`predict_topk()`：

```python
query_t = torch.from_numpy(query_features).to(device, float32)
q_norm = torch.nn.functional.normalize(query_t, p=2, dim=1)
```

NCM center 和 prototype center 在 cache 中假设已经 normalized，因此距离计算：

```text
sim = q_norm @ protos_t.T
distance = 1 - sim
```

### 9.3 Global-local score

两类距离：

```text
dist_ncm: [N, C]，query 到 class NCM center
dist_sub: [N, C]，query 到该 class 最近 active topology node
```

融合：

```text
final_scores = alpha * dist_ncm + (1 - alpha) * dist_sub
```

默认 `alpha=0.5`。最终对 `final_scores` 取最小的 top-k classes。

### 9.4 Coarse top-k

如果 `coarse_topk` 设置且小于 class 数，且没有 EASE reweighting、没有 edge-aware scoring，则：

1. 先用 `dist_ncm` 找 coarse candidate classes。
2. 只在候选 class 的 prototypes 上算 subcluster distance。

edge-aware scoring 开启时禁用 coarse top-k，因为 edge adjustment 需要完整 prototype distance。后续实验保持 `use_edge_aware_scoring=false`，因此正常使用 coarse top-k 条件不受 edge scoring 影响。

### 9.5 `predict_class_logits`

`predict_class_logits()` 返回 `[N, total_classes]` logits，用于需要 logits 的接口：

```text
logit[class] = -final_distance
unknown class = -inf
```

该函数当前不包含 edge-aware scoring adjustment，只实现 NCM/subcluster fused distance。

## 10. Memory ledger

`_compute_memory_breakdown()` 输出 bytes 和 MB 两套字段。

### 10.1 Compact deployable memory

如果 `use_dict_coding=True` 且已有 atom：

```text
compact =
    one copy of atoms
  + coefficients
  + atom metadata
  + node metadata
  + edges
  + edge reliability
  + normalized class_mu
```

注意：compact 只计单份 atom matrix。虽然实现里同时保留 `dict_atoms` 和 `dict_atoms_hat`，但 deployable 只需要 `dict_atoms_hat`。

如果是 raw/no-dict baseline：

```text
compact =
    normalized class_mu
  + normalized node centers
  + node metadata
  + edges
  + edge reliability
```

这样 raw baseline 不会因“不计 prototype centers”被低估。

### 10.2 Actual implementation memory

actual 包含真实实现中的额外字段：

```text
actual =
    actual atom storage
  + coefficients
  + atom metadata
  + node metadata
  + edges
  + edge reliability
  + class_mu
  + class_mu_raw
  + node.center and node.center_raw
  + predict cache
  + feature buffers
  + frozen clusters
  + STAR feature/image anchor placeholders if present
```

numpy arrays 使用 `.nbytes` 精确统计；字符串、dict key/value metadata 使用近似字节数。

### 10.3 主要 memory 字段

| 字段 | 含义 |
|---|---|
| `atoms_bytes/mb` | deployable 单份 atom matrix |
| `atoms_actual_bytes/mb` | 实际实现中 atom 相关数组总和 |
| `atoms_duplicate_bytes/mb` | actual 与 deployable atom 的差 |
| `coefficients_bytes/mb` | 所有 node sparse coefficients |
| `atom_metadata_bytes/mb` | atom states/origins/usage 近似 |
| `node_metadata_bytes/mb` | count/residual/state 近似 |
| `edges_bytes/mb` | edge endpoint + age |
| `edge_reliability_bytes/mb` | edge endpoint + reliability |
| `class_mu_bytes/mb` | normalized class centers |
| `class_mu_raw_bytes/mb` | raw class centers |
| `node_centers_bytes/mb` | actual node center + center_raw |
| `caches_bytes/mb` | predict cache tensor/array |
| `buffers_bytes/mb` | feature buffers |
| `frozen_bytes/mb` | frozen cluster backup |
| `compact_deployable_bytes/mb` | deployable estimate |
| `actual_implementation_bytes/mb` | implementation memory |

### 10.4 运行时资源监控与 memory ledger 的区别

memory ledger 是 classifier state 口径，只统计 HC-SOINN / LifeTopoDict 持有的可部署状态和当前 Python 实现中可审计的 classifier 字段。它不等于进程运行时内存，也不等于 GPU 显存。

新增的资源监控脚本用于补充系统口径：

| 脚本 | 口径 | 用途 |
|---|---|---|
| `scripts/run_with_gpu_memory_monitor.py` | `nvidia-smi memory.used` 的 pre-run baseline、peak delta、mean delta | 判断实验进程是否新增 GPU resident memory |
| `scripts/run_with_cpu_memory_monitor.py` | launched process + child process tree 的 RSS/PSS/USS | 判断 classifier-only 训练过程的 CPU RAM 压力 |

CPU 监控主指标建议使用 PSS：

```text
PSS = proportional set size
```

原因是 HC-SOINN 压缩使用 multiprocessing，RSS 会把 fork worker 共享页重复计入，可能显著高估真实物理内存压力。USS 可作为独占内存参考。

当前 CUB classifier-only 复测结论：

| 口径 | Raw HC-SOINN | LifeTopoDict selected | 解释 |
|---|---:|---:|---|
| GPU peak delta | 0 MiB | 0 MiB | strict cache 下跳过 backbone 后，两者都不新增 GPU 显存 |
| Train-only peak PSS | 812.48 MiB | 782.47 MiB | process-level 训练 RAM 基本持平 |
| Train-only actual classifier state | 15.2008 MB | 11.3271 MB | LifeTopoDict classifier state 更小 |

因此，论文或报告中应区分三类 memory：

1. `compact_deployable_*`：部署时理论可携带的 classifier state。
2. `actual_implementation_*`：当前 Python 实现中 classifier 对象持有的可审计状态。
3. runtime/process/GPU memory：Python、PyTorch、NumPy/SciPy、feature loader、临时聚类数组、multiprocessing worker、CUDA context 等系统开销。

LifeTopoDict 的强支撑点是前两类 classifier state memory；GPU 显存下降来自 feature cache + backbone skip 的共同工程优化，不是 LifeTopoDict 相对 raw HC-SOINN 的机制优势。

## 11. Diagnostics

### 11.1 Built-in diagnostics

`compute_diagnostics()` 返回：

| 指标 | 计算 |
|---|---|
| `PAD` | class prototype mean 与 `class_mu` 的 cosine |
| `EffRank` | mean `(sum abs(a))^2 / sum(a^2)` |
| `GTE` | 历史 growth residual relative improvement；growth 关闭时应为 0 |
| `atom_count` | `len(atom_states)` |
| `protected_ratio` | protected atom 数 / atom 总数 |
| `inactive_ratio` | inactive atom 数 / atom 总数 |
| `avg_residual` | active encoded node residual 均值 |
| `memory` | memory ledger |
| `edge_score_stats` | 历史 edge-aware scoring 最近一次推理统计；后续实验不启用 |

### 11.2 `utils/ltd_metrics.py`

额外诊断：

| 函数 | 说明 |
|---|---|
| `compute_DGR()` | 比较 materialized prototypes 与 `center_raw` prototypes 的 nearest-prototype accuracy |
| `compute_signed_AEPMI()` | 估算 atom support 与错误预测的 signed PMI |
| `compute_memory_report()` | 包装 classifier memory breakdown |
| `compute_edge_consistency()` | 检查 edge endpoint 是否越界 |
| `same_memory_comparison()` | LifeTopoDict compact vs baseline memory |
| `run_all_diagnostics()` | 聚合以上诊断 |

注意：`compute_DGR()` 中 `center_raw` 在 LifeTopoDict materialize 后是 `D_hat @ a`，不是真实原始 raw mean。因此该指标更适合在保留 raw prototype 快照时解释；当前实现中应谨慎使用 DGR 作为“原始 raw center”对照。

## 12. 配置与消融

当前 `exps/life_topo_dict/` 包含 10 个 JSON：

| 配置 | 作用 |
|---|---|
| `life_topo_dict_cifar.json` | CIFAR-100 sanity，3 seeds |
| `life_topo_dict_cub.json` | CUB-200，`20 tasks x 10 classes`，3 seeds |
| `life_topo_dict_inr.json` | ImageNet-R，`40 tasks x 5 classes`，3 seeds |
| `ablation_no_growth.json` | 历史配置；growth 已全局关闭，后续不再作为独立实验 |
| `ablation_no_lifecycle.json` | 所有 atom/node 保持 plastic |
| `ablation_no_dict.json` | 同一 LifeTopoDict learner，但 `use_dict_coding=false` |
| `ablation_edge_aware.json` | 历史配置；文件内已禁用 additive edge-aware scoring，后续不再作为独立实验 |
| `ablation_raw_hc_soinn.json` | CIFAR raw HC-SOINN baseline |
| `ablation_raw_hc_soinn_cub.json` | CUB raw HC-SOINN baseline |
| `ablation_raw_hc_soinn_inr.json` | ImageNet-R raw HC-SOINN baseline |

默认 LifeTopoDict P0 配置：

```json
{
  "dict_sparse_k": 5,
  "dict_ridge_lambda": 0.1,
  "lifecycle_theta_support": 0.5,
  "lifecycle_theta_usage": 0.01,
  "lifecycle_T_inactive": 3,
  "dict_theta_residual": 0.3,
  "dict_max_growth_per_task": 0,
  "use_dictionary_growth": false,
  "use_lifecycle": true,
  "use_protected_gate": true,
  "use_edge_age_persistence": true,
  "use_edge_aware_scoring": false,
  "edge_score_gamma": 0.1,
  "edge_score_eta": 0.05
}
```

`edge_score_gamma` / `edge_score_eta` 保留在配置说明中仅用于历史代码解释；后续实验不搜索、不启用 additive edge-aware scoring。

## 13. 日志与结果收集

LifeTopoDict 在 `after_task()` 输出：

```text
[LifeTopoDict] Diagnostics: PAD=..., EffRank=..., GTE=..., atoms=..., protected=..., inactive=..., avg_residual=...
[LifeTopoDict] Memory: compact=..., actual=..., atoms=..., coeffs=..., edges=..., edge_rel=..., caches=..., buffers=..., frozen=...
[LifeTopoDict] Edge scoring: use_rate=..., class_use_rate=..., margin_contribution=..., risk_penalty=..., avg_adjustment=...
```

`scripts/collect_results.py` 会收集：

1. HC-SOINN top1/top5 curve。
2. Average Accuracy、Final Accuracy、Forgetting。
3. PAD、EffRank、GTE、atom_count、protected/inactive、avg_residual。
4. edge_use_rate、edge_class_use_rate、edge_margin_contribution、edge_risk_penalty、edge_score_adjustment。

示例：

```bash
python scripts/collect_results.py logs/life_topo_dict/ -o results.csv --recursive
```

feature-cache classifier-only 路径会额外输出：

```text
[FeatureCache] Classifier-only mode enabled: skipping backbone initialization ...
[FeatureCache] train_loader_for_hc is backed by cached features: ...
[FeatureCache] test_loader is backed by cached features: ...
All params: 0 (backbone skipped; classifier-only feature-cache mode)
```

资源监控脚本输出：

| 输出 | 说明 |
|---|---|
| `*_samples.csv` | 每个采样点的 memory 数值 |
| `*_summary.json` | return code、peak/mean/min memory、sample count |
| `*.log` | 被监控命令完整 stdout/stderr，末尾包含 `EXIT_STATUS` |

已生成的相关报告：

- `logs/gpu_memory_comparison_classifier_only/gpu_memory_classifier_only_report.md`
- `logs/cpu_memory_training_comparison/training_memory_comparison_report.md`

## 14. 当前实现的边界与注意事项

1. `center_raw` 在 LifeTopoDict materialize 后是 `D_hat @ a`，不是原始 feature raw mean。STAR raw-space transport 如需启用，必须重新定义或保留 raw snapshot。
2. `dict_atoms` 和 `dict_atoms_hat` 在实际实现中都存在；memory compact 只计一份，actual 会计入真实存在的数组。
3. `sparse_encode()` Top-K 使用 normalized target，但 Ridge RHS 使用原始 target `v_i`。这是当前代码行为，文档中所有 residual/重构解释均基于此。
4. additive edge-aware scoring 是历史 P1 消融开关，默认关闭且后续不再纳入实验设置。当前保留 edge age/reliability 作为诊断元数据，不让 edge 以 additive score 形式影响主 score。
5. `predict_class_logits()` 未接入 edge-aware scoring，只返回 NCM/subcluster fused logits；这与后续关闭 additive edge-aware scoring 的实验设置一致。
6. `utils/ltd_metrics.compute_DGR()` 的 raw 对照在当前 `center_raw` 语义下需要谨慎解释。
7. same-memory raw baseline 的严格节点预算仍需要根据实际 compact memory 结果手动选择 `hcsoinn_max_proto_per_class`。
8. strict feature-cache classifier-only 路径默认跳过 backbone；如果要复现旧的“cache 但仍初始化 backbone”的显存 profile，必须显式设置 `--feature_cache_skip_backbone false`。
9. residual growth 代码保留但后续实验固定关闭；不要再把 `dict_theta_residual` / `dict_max_growth_per_task` 纳入参数搜索。
10. classifier-only 默认 CPU 推理；`--feature_cache_classifier_device cuda` 只影响 `predict_topk()` 的 torch distance 计算，不会把 HC-SOINN 压缩/字典训练搬到 GPU。
11. GPU 显存不是 LifeTopoDict 相对 raw HC-SOINN 的强机制指标。跳过 backbone 后两者 GPU 增量都可以为 0 MiB。
12. 训练期 process-level PSS/RSS 主要由 runtime 与 multiprocessing 主导，不应替代 compact/actual classifier state memory。
13. 当前报告描述的是代码实现，不等价于已证明满足 accuracy drop、memory-Pareto、Old-New HM 等实验成功标准。

## 15. 建议验证命令

语法检查：

```bash
python -m py_compile main.py trainer.py \
  models/life_topo_dict.py models/simplecil_hc_soinn.py \
  utils/hc_soinn_classifier.py utils/ltd_metrics.py utils/feature_cache.py \
  scripts/collect_results.py scripts/run_with_gpu_memory_monitor.py \
  scripts/run_with_cpu_memory_monitor.py
```

JSON 检查：

```bash
python - <<'PY'
import json, glob
for path in sorted(glob.glob('exps/life_topo_dict/*.json')):
    with open(path) as f:
        json.load(f)
print('json_ok')
PY
```

主实验示例：

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_cub.json \
  --use_feature_cache true \
  --feature_cache_strict true \
  --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict

python main.py --config=exps/life_topo_dict/life_topo_dict_inr.json \
  --use_feature_cache true \
  --feature_cache_strict true \
  --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict
```

关键消融示例：

```bash
python main.py --config=exps/life_topo_dict/ablation_raw_hc_soinn_cub.json \
  --use_feature_cache true --feature_cache_strict true \
  --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict

python main.py --config=exps/life_topo_dict/ablation_no_dict.json \
  --use_feature_cache true --feature_cache_strict true \
  --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict

python main.py --config=exps/life_topo_dict/ablation_no_lifecycle.json \
  --use_feature_cache true --feature_cache_strict true \
  --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict

```

`ablation_no_growth.json` 与 `ablation_edge_aware.json` 仅作为历史配置保留，后续实验不再调度。

命令行覆盖示例：

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_cifar.json \
  --dict_sparse_k 10 \
  --dict_ridge_lambda 0.05 \
  --lifecycle_theta_support 0.3 \
  --use_lifecycle false
```

GPU 显存监控示例：

```bash
python scripts/run_with_gpu_memory_monitor.py \
  --gpu 2 \
  --interval 0.2 \
  --csv logs/gpu_memory_comparison_classifier_only/raw_samples.csv \
  --summary logs/gpu_memory_comparison_classifier_only/raw_summary.json \
  --log logs/gpu_memory_comparison_classifier_only/raw.log \
  -- env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python main.py \
    --config exps/life_topo_dict/ablation_raw_hc_soinn_cub.json \
    --device 0 \
    --use_feature_cache true \
    --feature_cache_strict true \
    --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict
```

训练期 CPU 内存监控示例：

```bash
python scripts/run_with_cpu_memory_monitor.py \
  --interval 0.1 \
  --csv logs/cpu_memory_training_comparison/raw_train_only_samples.csv \
  --summary logs/cpu_memory_training_comparison/raw_train_only_summary.json \
  --log logs/cpu_memory_training_comparison/raw_train_only.log \
  -- env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python main.py \
    --config exps/life_topo_dict/ablation_raw_hc_soinn_cub.json \
    --device 0 \
    --training_time_profile_last_task true \
    --skip_eval_for_training_time_profile true \
    --use_feature_cache true \
    --feature_cache_strict true \
    --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict
```

## 16. 实现状态总结

当前代码已经具备 LifeTopoDict MVP 的主要工程路径：

1. 评估前 task-boundary 压缩。
2. class-local HC-SOINN node 身份保留。
3. shared dictionary atoms。
4. Top-K Ridge sparse coding。
5. dictionary-coded node materialization。
6. residual-triggered atom growth 历史代码路径保留，但后续实验关闭。
7. protected/plastic/inactive lifecycle。
8. protected SOINN update/deletion gate。
9. inactive readout filtering。
10. edge age persistence。
11. edge reliability 元数据。
12. additive edge-aware scoring 历史代码路径保留，但后续实验关闭。
13. compact vs actual memory ledger。
14. CIFAR/CUB/ImageNet-R 与关键消融配置。
15. strict feature-cache classifier-only 路径：已缓存特征实验默认不初始化 backbone。
16. raw HC-SOINN 与 LifeTopoDict 共用 cached-feature 训练/评估路径，保证 same-feature 公平比较。
17. classifier-only 默认 CPU 距离计算，可选 GPU `predict_topk()`。
18. GPU memory 与 CPU process-tree memory 监控脚本。
19. train-only profiling CLI 覆盖，用于跳过 eval 的训练期资源测量。
20. Direction 1 raw-node fallback cache、prediction trace、rule/random/static/oracle/ridge gate 推理路径。
21. learned fallback gate 的 held-out calibration split、跨任务累积 calibration、正例/增益不足保护。
22. raw fallback 与 fallback gate memory / diagnostics / result collection。

仍需通过完整实验确认：

1. CUB 或 ImageNet-R 上 accuracy drop 是否 `<= 0.5pp`。
2. compact memory 是否降低 `>= 30%`。
3. memory-Pareto 是否至少一个点不被 raw HC-SOINN 支配。
4. lifecycle 是否改善 worst-order、Old-New HM 或其他 stability 指标。
5. 后续若重新设计非 additive edge 使用方式，需另立方案和配置；当前 additive edge-aware scoring 不再作为待确认项。
6. Direction 1 当前仅验证了 oracle 上界与工程可运行；learned deployable gate 在 CUB seed 1993 上尚未带来正向 A_Last 增益。
