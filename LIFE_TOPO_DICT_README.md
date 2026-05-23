# LifeTopoDict MVP

LifeTopoDict 将 HC-SOINN 的 class-local 拓扑节点替换为共享字典 atom 的稀疏编码表示，在保持分类精度的同时降低内存占用。

## 环境配置

```bash
conda env create -f environment_hc_soinn.yaml
conda activate hc_soinn
```

额外依赖（评估和可视化）：
```bash
pip install matplotlib  # 可选，用于 atom 可视化
```

## 运行方式

### 特征缓存

LifeTopoDict MVP 的 `life_topo_dict` 和 `simplecil_hc_soinn` 支持离线 frozen-backbone feature cache。先用完整数据集按 DataManager 原始 train/test split 顺序提取一次 `mode="test"` transform 下的特征，后续实验通过缓存直接喂给 HC-SOINN 和评估路径，避免每个 task 反复做 backbone 前向。

默认缓存根目录：

```bash
/home/lyw/data/FeatureCache/LifeTopoDict
```

构建 CIFAR-100 cache：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2 \
python scripts/build_feature_cache.py \
    --config exps/life_topo_dict/life_topo_dict_cifar.json \
    --device 0 \
    --cache-dir /home/lyw/data/FeatureCache/LifeTopoDict \
    --batch-size 256 \
    --num-workers 8 \
    --dtype float32 \
    --splits train test
```

使用缓存运行实验（建议 `feature_cache_strict=true`，避免缓存缺失时静默回退到 backbone 前向）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2 \
python main.py \
    --config exps/life_topo_dict/life_topo_dict_cifar.json \
    --device 0 \
    --use_feature_cache true \
    --feature_cache_strict true \
    --feature_cache_dir /home/lyw/data/FeatureCache/LifeTopoDict
```

缓存 key 包含 dataset、backbone_type、`test` transform 和 `frozen_feature` 提取模式，不包含 seed/class order；同一 dataset/backbone 的缓存可跨不同 seed 和 class order 复用。若需要重建同一路径缓存，给构建脚本加 `--overwrite`。

### CIFAR-100 实验

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_cifar.json
```

### CUB-200 实验

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_cub.json
```

### ImageNet-R 实验

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_inr.json
```

### 消融实验

```bash
# 无生命周期管理（所有 atom 保持 plastic，无 protected/inactive 状态转换）
python main.py --config=exps/life_topo_dict/ablation_no_lifecycle.json

# 无字典编码（同一模型类，use_dict_coding=false，回退 raw nodes）
python main.py --config=exps/life_topo_dict/ablation_no_dict.json

# 原始 HC-SOINN baseline（独立 baseline，不使用 LifeTopoDict）
python main.py --config=exps/life_topo_dict/ablation_raw_hc_soinn.json

# CUB-200 原始 HC-SOINN baseline
python main.py --config=exps/life_topo_dict/ablation_raw_hc_soinn_cub.json

# ImageNet-R 原始 HC-SOINN baseline
python main.py --config=exps/life_topo_dict/ablation_raw_hc_soinn_inr.json
```

### 命令行参数覆盖

所有 LifeTopoDict 参数可通过命令行覆盖：

```bash
python main.py --config=exps/life_topo_dict/life_topo_dict_cifar.json \
    --dict_sparse_k 10 \
    --dict_ridge_lambda 0.05 \
    --lifecycle_theta_support 0.3 \
    --use_lifecycle false
```

## 关键参数

### 字典编码参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dict_sparse_k` | 5 | Top-K Ridge 的 k 值 |
| `dict_ridge_lambda` | 0.1 | Ridge 正则化系数 |
| `dict_theta_residual` | 0.3 | 历史 growth 参数；后续实验不再使用 |
| `dict_max_growth_per_task` | 0 | 字典增长已关闭；后续实验保持 0 |

### 生命周期参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lifecycle_theta_support` | 0.5 | old support 保护阈值（高于此 → protected） |
| `lifecycle_theta_usage` | 0.01 | EMA usage 低于此值开始计 low-usage streak |
| `lifecycle_T_inactive` | 3 | 连续低 usage 任务数阈值（达到后 → inactive） |

### 消融开关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_dict_coding` | true | 总开关：启用字典编码 |
| `use_dictionary_growth` | false | residual-triggered 字典增长已弃用，后续实验保持关闭 |
| `use_lifecycle` | true | 启用生命周期状态管理（protected/plastic/inactive） |
| `use_protected_gate` | true | 启用 protected 节点在 SOINN 精炼中的删除保护 |
| `use_edge_age_persistence` | true | 持久化 edge age 并保存 edge reliability 元数据 |
| `use_edge_aware_scoring` | false | additive edge-aware scoring 已弃用，后续实验保持关闭 |
| `edge_score_gamma` | 0.1 | 历史 edge scoring 参数；后续实验不再搜索 |
| `edge_score_eta` | 0.05 | 历史 edge scoring 参数；后续实验不再搜索 |

## 向量语义约定（P0-1）

所有 atom 编码和 node materialization 遵守统一的向量语义：

```
dict_atoms_hat: [M, d] — 行 L2 归一化，唯一真值来源
dict_atoms:     [M, d] — 等于 dict_atoms_hat（保持向后兼容）

稀疏编码：在 D_hat 上做 Top-K Ridge
重构：     z = D_hat @ a（未归一化）
           v_hat = normalize(z) → cluster.center
           z                  → cluster.center_raw（兼容字段，非真实 raw mean）
```

- Base atoms 使用统一归一化语义；grown atoms 逻辑保留在代码中仅用于历史复现，后续实验关闭 growth
- `center_raw` 在 P0 阶段仅作为兼容字段，不参与 STAR transport

## 预期输出

训练过程中每个 task boundary 会输出：

**诊断指标：**
- 字典 atom 数量
- PAD（Prototype Alignment Degree，越高越好，max=1.0）
- EffRank（有效字典秩）
- GTE（历史 growth 诊断；growth 关闭时应为 0）
- avg_residual（平均重构残差）
- protected/inactive atom 比例
- edge scoring use_rate / class_use_rate / margin_contribution / risk_penalty（历史 edge-aware scoring 诊断；后续实验不启用）

**内存报告：**
- compact deployable memory（单份 atoms + coefficients + class means + metadata + edges）
- actual implementation memory（+ caches + buffers + frozen clusters）
- 分项：atoms、coefficients、edges、edge reliability、caches、buffers、frozen

## 文件结构

### 修改的文件

| 文件 | 修改范围 | 说明 |
|------|----------|------|
| `utils/hc_soinn_classifier.py` | 新增约 1200 行 | 核心：字典存储、稀疏编码、生命周期、诊断、memory ledger |
| `main.py` | 新增 14 个 CLI 参数 | LifeTopoDict 参数 + 消融开关支持 |
| `utils/factory.py` | 新增 1 行 | 模型注册 |

### 新增的文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `models/life_topo_dict.py` | ~210 | LifeTopoDict 模型（含 memory 报告输出） |
| `utils/ltd_metrics.py` | ~550 | 评估指标（DGR、AEPMI、PAD、内存报告） |
| `utils/ltd_visualize.py` | ~290 | 可视化工具（状态分布、残差直方图、使用热力图） |
| `scripts/collect_results.py` | ~380 | 实验结果收集脚本 |
| `exps/life_topo_dict/*.json` | 10 个文件 | CIFAR、CUB、ImageNet-R、消融与 raw baseline 配置 |

## 实现偏差说明

与设计方案的差异：

1. **Old support 使用系数幅度代替 PMI**：设计方案提到 AEPMI/PMI proxy，但实现使用 atom 在旧类节点中系数绝对值的归一化均值。这是合理简化——PMI 在少样本场景不可靠。

2. **节点状态从 atom 状态派生**：设计方案未明确 node state 是否独立管理。实现中 node_state 从 top-K atom 状态推导（any protected → protected; all inactive → inactive），减少状态不一致风险。

3. **P1 组件无接口预留**：quarantined 状态、edge-aware scoring、STAR-gated、selective rumination 均无显式接口占位。字符串类型的状态字段可自然扩展。

## 已修复的 P0 正确性问题

以下问题已在 P0 阶段修复，均通过审查验证：

1. **向量语义统一**（P0-1）：`dict_atoms` 和 `dict_atoms_hat` 现在都存储 L2 归一化 atoms；base atoms 和 grown atoms 使用完全相同的归一化语义；`_materialize_nodes()` 只使用 `D_hat`。

2. **Protected indices 映射**（P0-2）：实现了 old-to-new center matching：保存旧 protected node 的 center，`compress()` 后用 cosine similarity greedy matching（threshold=0.7）迁移 protected 状态，未匹配成功的 protected node 不强制迁移。

3. **Usage 统计 EMA**（P0-3）：引入 `atom_usage_ema`（EMA 衰减 factor=0.7）替代永不过期的累积计数器。`_compute_atom_usage()` 优先返回 EMA 值。新增 atoms 初始 EMA=0，后续 task 有机会获得 usage。

4. **消融开关**（P0-4）：新增显式 boolean 开关（`use_dictionary_growth`、`use_lifecycle`、`use_protected_gate`、`use_edge_age_persistence`）。其中 `use_dictionary_growth` 已按最新实验决策关闭，不再纳入后续实验设置。消融配置使用同一模型类 + 开关控制，不再依赖阈值 hack。新增 raw HC-SOINN baseline 配置。

5. **Memory ledger**（P0-5）：每个 task boundary 自动输出分项 memory 报告（compact deployable / actual implementation / atoms / coefficients / edges / edge reliability / caches / buffers / frozen），基于精确的 `.nbytes` 计算。compact deployable 只计单份 atom matrix；raw/no-dict baseline 会计入 prototype centers，避免 memory 对比低估。

6. **协议调用时序**（P0-6）：LifeTopoDict 和 raw HC-SOINN learner 现在在 `_train()` 抽取特征后立即 `compress()`，因此 `eval_task()` 使用当前 task 已压缩、已字典编码的节点；`after_task()` 只做兜底压缩、诊断和 `_known_classes` 更新。

7. **Protected gate 生效范围**（P0-7）：protected old nodes 会在 worker 内映射到新 hierarchical clusters，SOINN 精炼时禁止普通中心更新并跳过低度删除；压缩后仍用 old-to-new matching 迁移 protected 状态。

8. **Edge-aware scoring 开关**（P1-1，历史实现）: `use_edge_aware_scoring` 保留用于历史复现，但 additive edge-aware scoring 已因默认配置有害而弃用，后续实验不再启用或搜索相关参数。

## 已知问题清单

### MVP 范围外（P1 待实现）

- Quarantined 状态和 soft risk reweighting
- STAR anchor confidence gate
- State-aware STAR delta scaling
- Selective rumination

### 待后续优化

1. **`sparse_encode` 逐样本循环**：k 个样本分别做矩阵求解，大规模时会成为性能瓶颈。可改为批处理 Cholesky 或 vectorized 求解。

2. **`center_raw` 语义**：当前存储 `D_hat @ a`（归一化 atoms 的未归一化重构），不是真实的 raw mean。启用 STAR 时需要重新定义 raw-space transport。

3. **same-memory raw baseline**：已提供 raw baseline 配置和内存对比工具；严格 same-memory 节点预算仍需根据实际 LifeTopoDict compact memory 结果选择 `hcsoinn_max_proto_per_class`。
