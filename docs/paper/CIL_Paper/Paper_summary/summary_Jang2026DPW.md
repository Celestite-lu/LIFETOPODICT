# 论文详细总结：Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting

## 基本信息

- **标题**：Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting
- **作者**：Hyeonseo Jang, Hyuk Kwon, Kibok Lee
- **单位**：Yonsei University
- **发表年份/会议/期刊**：论文正文未明确标注发表会议/期刊，疑似 2026 年预印本/投稿论文
- **代码**：https://github.com/YonseiML/dpw
- **主干模型**：CLIP ViT-B/16

## 研究动机与问题定义

### 核心痛点

该论文聚焦于视觉-语言模型在域-类增量学习场景下的持续学习，针对现有 PEFT 方法的两个关键问题：

1. 缺乏 token 级别的细粒度权重分配。现有方法（DIKI、MoE-Adapters）在样本级别操作，对所有 token 分配相同权重的任务特定信息。但不同 token 的任务关联度不同，应有不同程度的调整。

2. 预训练注意力机制的固有缺陷。传统 prefix-tuning 通过 QK 点积计算 prefix score，但预训练的 Q/K 投影缺乏下游任务知识。实验表明，Q 投影后 task-relevant 与 task-irrelevant token 的余弦相似度从 0.02 升至 0.47；UMAP 显示不同任务在投影后坍缩至重叠区域。

### 问题形式化

- T 个任务顺序学习，目标是在下游任务上获得良好性能并保持零样本泛化能力
- 评估基准：MTIL（task-incremental）和 ODCL-CIL（class-incremental），11 个数据集，1201 类

## 方法/框架

提出 Dynamic Prefix Weighting (DPW)，包含三个核心组件：

### 1. RePA（Reparameterized Prefix Attention，重新参数化前缀注意力）

传统 prefix score：S_XP^(i) = Q_i(K_Pi)^T = (XW_i^Q + 1_m b_Q^T)(P_K W_i^K + 1_L b_K^T)^T

将其重新参数化为单一仿射变换（Eq. 5）：
- S_XP^(i) = X W_i^G + B_i^G
- W_i^G 合并了 W_i^Q、P_K、W_i^K 和偏置项
- 冻结原始 W_i^Q、W_i^K，仅训练 W_i^G 和 B_i^G
- 将二次复杂度的 QK 点积降为轻量仿射变换

验证实验（Table 4）：直接训练 W^Q/W^K 需 228M 参数但收益极小；RePA 仅需 30.8M 参数即大幅提升。

### 2. CondAct（Conditional Activation，条件激活）

替代 softmax 的两步操作：

**Step 1: 条件归一化（Eq. 8a）**
- 若 sum_k σ(s_ijk) >= 1，归一化处理
- 否则保留原始 sigmoid 值，不做归一化
- 核心设计：prefix 总权重上限为 1，但不强制等于 1

**Step 2: 条件过滤（Eq. 8b）**
- g~_ijk = g_ijk · I(g_ijk >= cutoff)
- cutoff 基于 prefix-[CLS] 注意力分数的高斯分布假设
- cutoff_ijk = 1 - σ(log φ(s_i,cls,k; μ_t, σ_t^2))
- 似然分数高时 cutoff=0（prefix 完全贡献），低时过滤

理论证明（App. D）：CondAct 将最坏情况 prefix 表示漂移从 L·c_V 降至 c_V。

### 3. RWM（Residual Weighting Mechanism，残差加权机制）

Adapter 设计：
- HydraLoRA 架构：共享冻结的 D 矩阵（SVD 初始化自 W_i^V 的 top-k 左奇异向量）+ 任务特定 U_t
- 实验验证（Table A.2）：持续更新 D 导致灾难性遗忘（Last 25.7）；随机初始化 D 无提升；SVD 初始化+冻结 D 最优

RWM 公式（Eq. 11）：
- O_adapter_i = Δ_i ⊙ E_i^t(X)
- Δ_ij = max(0, sum_k σ(s_ijk) - 1)
- 仅当 prefix sigmoid 之和 > 1 时 adapter 生效
- 使用原始 sigmoid 分数的残差（非归一化后权重）

最终输出（Eq. 12）：O_R_i = O_prefix_i + O_adapter_i

### 自适应头数（App. B.2）

基于任务难度动态确定头数 h'：
- easiness score = 类间方差 / (类内方差 + α/n_cls)，α=10
- F = easiness / zs_accuracy
- h' = 2^round(log2(h × F))
- 困难任务获得更多容量

### 训练设置

- 损失：交叉熵；LR=1.25，batch size=32，10 epoch，余弦调度
- Prefix 长度 L=8；LoRA rank：默认 64（Ours），4（Ours+）
- 插入：全部 12 层视觉+文本编码器
- 初始化：W_i^G 和 P_V 跨任务正交；B_i^G 视觉/文本分支各初始化为特定负值
- 硬件：单 NVIDIA 4090 GPU
- Ours+ 高效变体：每头仅用 d/h' 子维度

## 实验设置

### 基准
- MTIL：task-incremental（提供 task ID）
- ODCL-CIL：class-incremental（无 task ID，在所有已见类别中分类）
- 11 数据集：Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, SUN397
- 两种顺序：Order I（字母）、Order II（随机）

### 指标
- Transfer：零样本泛化/前向遗忘
- Avg.：所有阶段平均准确率
- Last：最终平均性能
- Mean：三者的均值

### Baselines
全量微调：ZSCL, GIFT；Prompt：CoLeCLIP, DIKI, DPeCLIP；Adapter：MoE-Adapters；传统 CL：LwF, iCaRL, WiSE-FT

### 实现
- CLIP ViT-B/16；与 DIKI 相同的手工 prompt、数据划分、验证策略
- ODCL-CIL 任务识别：沿用 DIKI 的似然估计方法，同时评估 batch size=256 和 batch size=1，取较低分

## 核心结果与发现

### MTIL Order I（Table 1）
| 方法 | Transfer | Avg. | Last | Mean | Params |
|------|----------|------|------|------|--------|
| Ours | 70.4 | 79.3 | 88.3 | 79.3 | 30.8M |
| Ours+ | 70.0 | 78.6 | 87.6 | 78.7 | 4.6M |
| GIFT | 69.3 | 77.3 | 86.0 | 77.5 | 149.6M |
| MoE-Adapters | 68.9 | 76.7 | 85.0 | 76.9 | 59.6M |
| DIKI | 68.7 | 76.3 | 85.1 | 76.7 | 1.8M |
| ZSCL | 68.1 | 75.4 | 83.6 | 75.7 | 149.6M |

### ODCL-CIL（Table 2）
| 方法 | Transfer | Avg. | Last | Mean |
|------|----------|------|------|------|
| Ours | 70.4 | 78.6 | 86.6 | 78.5 |
| Ours+ | 70.0 | 77.9 | 85.8 | 77.9 |
| DPeCLIP | 69.1 | 76.1 | 84.6 | 76.6 |
| CoLeCLIP | 68.8 | 73.7 | 79.7 | 74.1 |
| MoE-Adapters | 69.1 | 66.2 | 66.9 | 67.4 |

### MTIL Order II（Table E.2）
Ours Mean 76.7，比 GIFT（75.6）高 1.1 个百分点

### Few-Shot MTIL（Table C.1/E.1）
Ours Mean 76.4，优于 MoE-Adapters（75.6）和 DIKI（73.3）

### 消融实验

组件消融（Table 3）：
| 组合 | Transfer | Avg. | Last |
|------|----------|------|------|
| Baseline | 68.1 | 76.6 | 85.9 |
| +RePA only | 68.0 | 76.8 | 86.4 |
| +CondAct only | 69.5 | 77.8 | 86.8 |
| RePA+CondAct | 69.9 | 78.9 | 87.9 |
| 全部 | 70.4 | 79.3 | 88.3 |

CondAct 内部分解（Table 5）：
| 配置 | Transfer | Avg. | Last |
|------|----------|------|------|
| RePA baseline | 68.0 | 76.8 | 86.4 |
| +Sigmoid | 68.6 | 78.3 | 88.2 |
| +CondNorm | 69.9 | 79.0 | 88.2 |
| +Filtering | 70.4 | 79.3 | 88.3 |

RePA 设计对比（Table 4）：直接训练 W^Q/W^K 仅 Transfer 68.9（228M 参数），Ours 70.4（30.8M）

RWM 对比（Table 6）：加倍 prefix 长度反而退化（Transfer 69.5），可学习 router 边际提升（Last 88.1），RWM 最优（88.3）

Adapter 架构消融（Table A.2）：持续更新 D → Last 25.7（灾难性遗忘）；SVD 初始化+冻结 D → 最优

### 关键发现

1. 预训练 Q 投影破坏 token 区分性：相似度 0.02 -> 0.47；UMAP 显示任务坍缩
2. Prefix 与 Adapter 自适应分工：已训练任务 prefix 占 80-90%，未见任务 adapter 占 80%
3. RWM 促进正交互补：adapter-prefix 余弦相似度接近 0（均匀加权时 0.10-0.35）
4. 计算效率：Ours 3m16s < DIKI 3m46s << MoE-Adapters 35m47s
5. 参数效率：1M-5M 各量级 Ours+ 始终优于 DIKI
6. 动态 cutoff 优于固定阈值

## 主要贡献与局限性

### 贡献
1. 提出 DPW，首次在 prefix-tuning CL 中实现 token 级别权重分配
2. RePA：QK 点积简化为仿射变换，保留预训练知识并降低复杂度
3. CondAct：条件归一化+过滤替代 softmax，理论证明缓解前向遗忘
4. RWM：残差驱动 adapter 参与，实现 prefix-adapter 正交互补
5. 两个基准、两种顺序、few-shot 下均 SOTA

### 局限性
1. 依赖手工 prompt 模板
2. Prefix 长度 L=8 全局固定
3. ODCL-CIL 任务识别策略继承自 DIKI，可能引入误差
4. 仅验证 CLIP ViT-B/16
5. 高斯假设仅在两层的单数据集上验证
6. Adapter D 矩阵冻结牺牲灵活性

## 后续研究启发

1. Token 级权重思想推广至其他 PEFT 方法
2. 非参数化/可学习 cutoff
3. Prefix 长度+头数联合自适应
4. 扩展到更多 VLMs 和更大规模
5. Adapter-Prefix 协同的形式化理论
6. 与 memory-based 方法结合
7. 扩展到目标检测、分割等任务
