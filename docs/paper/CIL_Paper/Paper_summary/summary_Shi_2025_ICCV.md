# Lark: Low-Rank updates after knowledge localization for Few-shot Class-Incremental Learning —— 论文详细总结

---

## 基本信息

- **标题**：Lark: Low-Rank updates after knowledge localization for Few-shot Class-Incremental Learning
- **作者**：Jinxin Shi (华东师范大学), Jiabao Zhao (东华大学, 通讯), Yifan Yang (星环科技), Xingjiao Wu (华东师范大学), Jiawen Li (华东师范大学), Liang He (华东师范大学, 通讯)
- **发表会议/年份**：ICCV 2025
- **关键词**：Few-Shot Class-Incremental Learning (FSCIL)、Knowledge Localization、Low-Rank Update、Fisher Information Matrix、Vision Transformer (ViT)、Model Editing

---

## 研究动机与问题定义

### 痛点
1. **小样本类增量学习 (FSCIL) 的核心矛盾**：稳定性-可塑性困境 (stability-plasticity dilemma)——模型在学习新知识（可塑性）时会灾难性遗忘旧知识；过度强调保留旧知识（稳定性）则会限制新知识学习能力。
2. **ViT 引入 FSCIL 后问题加剧**：分类预测依赖 CLS token 与所有 patch tokens 的全局注意力交互，仅优化分类器会导致 backbone 与 classifier 之间的认知一致性破坏。ViT 参数量巨大且缺少 CNN 的归纳偏置，仅用少量增量样本直接微调极易过拟合和产生表征漂移 (representation drift)。
3. **现有方法的局限**：冻结 backbone（如 OrCo、CLOSER）导致静态 backbone 与持续演化的分类器之间错位。全局微调 backbone 又会遗忘已有知识。EWC/WaRP 等方法计算全参数梯度，在大模型中计算代价过高。
4. **本工作的核心洞察**：FSCIL 在 ViT 上需要两件事——(a) 精确定位最适合学习新知识的参数子集；(b) 以最小扰动的方式更新这些参数。

### 问题定义
FSCIL 训练过程分为 T+1 个 session：{D_base, D_1, ..., D_T}，对应互不相交的标签集 {C_base, C_1, ..., C_T}。D_base 为大规模训练集，后续增量 session D_t 只有少量样本（5-way 5-shot 或 10-way 5-shot）。目标是：在增量步骤 t 识别 D_t 中的新类，同时保持对已学类别的良好判别能力。模型由特征提取器 f_theta 和最近邻分类器 g_phi (Nearest Neighbor Classifier) 组成。分类器是非参数的，不受 backbone 更新的影响。

---

## 方法/框架（重点）

Lark 方法包含两个核心阶段：**Knowledge Localization（知识定位）** 和 **Incremental Editing（增量编辑）**。

### 第一阶段：Knowledge Localization

**目标**：识别模型中对旧知识敏感度最低、最适合学习新知识的参数所在的层。

**核心思想**：不直接计算每个参数的梯度（对大规模模型计算代价过高），而是利用隐藏状态 (hidden states) 作为间接度量指标，通过向隐藏状态注入高斯噪声并观察模型输出的变化，来评估不同层对已有知识的重要性。

**Fisher Information Matrix 计算流程**：

1. **添加扰动**：对第 l 层的隐藏状态（如 MHSA 输出 h_a^(l)）添加高斯扰动 epsilon ~ N(0, sigma^2)，得到扰动后的隐藏状态。

2. **计算敏感性指标 tau_a^l**（公式 3）：衡量第 l 层 token a 对最终输出的贡献。分子为扰动后 CLS token 和 patch tokens 梯度的 L2 范数平方和，分母为 N+1（1个CLS + N个patches）。损失函数 L = 1 - cos_sim(z_m^(L), z_tilde_m^(L))，即 1 减去原始与扰动后 CLS token 的余弦相似度。使用余弦相似度与最近邻分类器自然对齐。

3. **构建 Fisher Information Matrix F_k**（公式 4）：对第 k 个类别，随机选取 J 个样本，构建矩阵 F_k in R^(J x 2L)，每行对应一个样本，每列对应一层的一个模块（MHSA 或 MLP），共 2L 列。

4. **计算逐层敏感性**（公式 5）：对 F_k 的每一列求和，得到该层对第 k 类的影响力评分。对所有类别聚合后，使用 Lowest-K 准则选出对旧知识干扰最小的层/参数矩阵进行后续编辑。

**实现细节**：每类随机选 5 个样本计算感知差异（J=5），最终选出 3 个参数矩阵进行编辑（K=3）。

### 第二阶段：Incremental Editing

**目标**：以最小范数扰动原始参数矩阵，在引入新知识的同时最大程度保留旧知识。

**核心策略**：Rank-One 矩阵更新——将参数更新 Delta W 约束为秩为 1 的外积矩阵。

#### MHSA 模块中的编辑

**理论分析**：W_Query 和 W_Key 主导 token 间关系和注意力分布，但对注意力分布的改变很小。W_Value 主导最终输出，直接修改 W_Value 不会影响注意力分布。通过两个实验验证：(a) 训练前后 CLS token 对各 patch 的注意力分布变化很小（Figure 4）；(b) 训练前后第二层注意力矩阵的相似度均大于 0.75，最低值为 0.7617（Table 1）。因此只编辑 W_Value 即可。

**优化目标**（公式 7）：min ||Delta W_Value^(l)||_*  s.t.  nu = (W_Value^(l) + Delta W_Value^(l)) mu，其中 mu 是来自 A^sigma h_m^(l-1) 的单个 token 输入向量，nu 是该 token 的目标输出（通过对新类别进行常规训练获得，详见补充材料 Section 7.1）。

**Rank-One 更新形式**（公式 8）：Delta W_Value^(l) = (nu_z outer mu_z^T)/||mu_z||^2  union  {(nu_i outer mu_i^T)/||mu_i||^2 | i=1,...,N}，其中 nu_z outer mu_z^T 为 CLS token 的外积矩阵，nu_i outer mu_i^T 为各 patch token 的外积矩阵。

#### MLP 模块中的编辑

**理论分析**：Transformer 的 MLP 模块在知识存储中起关键作用。MLP 第一层编码输入的语义属性（充当 Key），第二层作为关联记忆检索对应事实（充当 Value）。因此编辑第二层线性层 W_(mlp,2)。

**更新公式**（公式 9）：与 MHSA 类似，Delta W_(mlp,2)^(l) = (nu_z outer mu_z^T)/||mu_z||^2  union  {(nu_i outer mu_i^T)/||mu_i||^2 | i=1,...,N}，其中 mu 来自 h_a^(l) W_(mlp,1)^(l)，nu 来自目标隐藏状态 h_m_hat^(l)。

#### SVD 秩一近似（公式 10）

由于每个 token 产生一个 Rank-One 矩阵，N+1 个 token 的并集产生的 Delta W 不一定是秩为 1 的。因此对 Delta W 进行奇异值分解 (SVD)，仅保留最大奇异值对应的分量：Delta W ≈ lambda_max u_max v_max^T。这种近似在 Frobenius 范数下提供最优低秩表示。


---

## 实验设置

### 数据集
| 数据集 | 基类数量 | 增量类数量 | 增量 session 数 | 每 session 设置 |
|--------|----------|------------|-----------------|-----------------|
| CIFAR100 | 60 | 40 | 8 | 5-way 5-shot |
| mini-ImageNet | 60 | 40 | 8 | 5-way 5-shot |
| CUB200 | 100 | 100 | 10 | 10-way 5-shot |

### Baselines
Lark 被集成到两种现有 FSCIL 方法中作为验证基础：
- **OrCo-ViT**：OrCo [CVPR 2024] 将 backbone 替换为 ViT-B/16
- **CLOSER-ViT**：CLOSER [ECCV 2024] 将 backbone 替换为 ViT-B/16

此外还与 CEC [CVPR 2021]、LIMIT [TPAMI 2022]、WaRP [ICLR 2023]、SV-T [ICCV 2023]、CPE-CLIP [ICME 2023]、MTE-FSCIL [TIP 2024]、LRT [TPAMI 2024] 等 state-of-the-art 方法对比。

### 评估指标
- **每 session 准确率 (Acc. in each session %)**
- **平均准确率 (Avg, %)**：所有 session（Base + 全部增量 sessions）准确率的平均值
- **性能下降率 (PD, %)**：PD = 基类 session 准确率 - 最后一个 session 准确率

### 实现细节
- 特征提取 backbone：ViT-B/16
- 分类器：Nearest Neighbor Classifier（非参数，不受 backbone 更新影响）
- 知识定位阶段：每类选 5 个样本计算 Fisher 信息，选 3 个参数矩阵进行编辑
- 增量编辑阶段：余弦学习率调度，最大学习率 0.1，训练 50 个 epoch
- 硬件：2 块 A100 GPU
- 结果汇报：3 次运行的平均值
- 所有数据集使用统一的超参数配置


---

## 核心结果与发现

### 主要实验结果（Table 2）

**CIFAR100 (8 sessions, 5-way 5-shot)**：
- OrCo-ViT baseline：Avg = 82.08%, PD = 19.99%
- **Lark in OrCo-ViT：Avg = 86.36%, PD = 13.50%**（Avg 提升 +4.28%, PD 降低 6.49%）
- CLOSER-ViT baseline：Avg = 83.19%, PD = 14.10%
- **Lark in CLOSER-ViT：Avg = 86.03%, PD = 7.74%**（Avg 提升 +2.84%, PD 降低 6.36%）

**mini-ImageNet (8 sessions, 5-way 5-shot)**：
- OrCo-ViT baseline：Avg = 80.82%, PD = 18.55%
- **Lark in OrCo-ViT：Avg = 86.15%, PD = 14.46%**（Avg 提升 +5.33%）
- CLOSER-ViT baseline：Avg = 84.11%, PD = 15.79%
- **Lark in CLOSER-ViT：Avg = 88.15%, PD = 9.43%**（Avg 提升 +4.04%, PD 降低 6.36%）

**CUB200 (10 sessions, 10-way 5-shot)**：
- OrCo-ViT baseline：Avg = 78.58%, PD = 14.79%
- **Lark in OrCo-ViT：Avg = 80.28%, PD = 12.44%**
- CLOSER-ViT baseline：Avg = 78.79%, PD = 14.05%
- **Lark in CLOSER-ViT：Avg = 82.00%, PD = 9.77%**

在所有三个数据集上，Lark 均显著超越 baseline 和多数现有方法。Lark 在不同 backbone 架构上表现出广泛适用性和鲁棒稳定性。在 PD 指标上，Lark 略逊于部分利用了 CLIP 额外语义信息的 prompt 方法（如 CPE-CLIP 在 CIFAR100 上 PD=7.31%, 而 Lark in CLOSER-ViT 为 7.74%），但 Lark 在 Avg 指标上表现更优，且完全不依赖外部预训练模型（如 CLIP），展现了方法的自足性和泛化能力。

### 消融实验（Table 3, CIFAR100, 基于 OrCo-ViT）

| 配置 | Avg (%) | PD (%) |
|------|---------|--------|
| Frozen（OrCo-ViT baseline，冻结 backbone） | 82.08 | 19.99 |
| 完全解冻微调（无 Localization, 无低秩约束） | 77.33 | 29.81 |
| 仅 Localization（定位后正常微调） | 81.07 | 22.32 |
| Localization + Rank-All（对各 token 更新矩阵求平均） | 84.38 | 16.59 |
| Localization + Rank-One（SVD 保留最大奇异值，即完整 Lark） | **86.36** | **13.50** |

关键发现：
1. 完全解冻微调最差（Avg 77.33, PD 29.81），直接微调严重破坏旧知识。
2. 仅做 Localization 不够（Avg 81.07 < 82.08），因为定位后微调仍使参数分布偏移，旧类识别能力下降。
3. Rank-All（对各 token 更新矩阵求平均）已优于 OrCo-ViT（84.38 > 82.08），证明低秩更新有潜力克服稳定性-可塑性困境。
4. Rank-One（SVD）效果最好（Avg 86.36），验证了秩一约束的必要性。

### MHSA 中编辑矩阵的选择分析（Table 4 左侧）

对比仅编辑 Q、仅编辑 K、仅编辑 V、三者同时编辑（结果基于 CLOSER-ViT）：

| 数据集 | Q only | K only | V only | Q,K,V 联合 |
|--------|--------|--------|--------|------------|
| CIFAR100 Avg/PD | 84.45/10.43 | 84.40/10.51 | **86.03/7.74** | 84.93/11.27 |
| CUB200 Avg/PD | 80.31/12.76 | 79.93/13.37 | **82.00/9.77** | 80.09/13.81 |
| mini-ImageNet Avg/PD | 85.19/12.83 | 84.72/13.33 | **88.15/9.43** | 84.30/14.49 |

**结论**：仅编辑 V 矩阵在所有数据集上取得最佳结果。原因是 Q、K 的改变会显著改变注意力分布（干扰旧类），而 V 仅是特征变换，能有效平衡稳定性与可塑性。三者联合更新导致性能退化。

### 秩的选择分析（Table 4 右侧）

对比 Rank-1, 4, 16, 64（结果基于 CLOSER-ViT）：

| 数据集 | Rank-1 | Rank-4 | Rank-16 | Rank-64 |
|--------|--------|--------|---------|---------|
| CIFAR100 Avg/PD | 86.03/7.74 | 85.99/7.60 | 85.99/7.91 | 85.85/7.95 |
| CUB200 Avg/PD | 82.00/9.77 | 81.91/9.63 | 81.83/9.57 | 81.72/9.49 |
| mini-ImageNet Avg/PD | 88.15/9.43 | 88.09/9.40 | 87.86/9.51 | 87.63/9.57 |

**结论**：Rank-1 在三个数据集上取得最优或接近最优的 Avg 和 PD。更高秩未带来显著提升，反而增加参数量和内存消耗，与 FSCIL 的轻量学习目标相悖。这支持了 Eckart-Young-Mirsky 定理——秩-1 外积在 Frobenius 范数下提供最优低秩表示。

### 可视化分析

**隐藏状态聚类可视化（Figure 5）**：使用 t-SNE 对比 OrCo-ViT 与 Lark 在 CIFAR100 Session 1 测试集上的特征分布。OrCo-ViT 在中间层（如 h_m^(10)）存在类别重叠（红色、绿色、紫色类别边界模糊），而 Lark 能画出清晰决策边界。

**参数分布可视化（Figure 6）**：对第 10 层 MLP 第二线性层的参数进行直方图和散点图分析。直方图显示 Lark 编辑后权重分布与原始权重非常接近，而仅做 Localization 的峰值和极端值频率显著偏移。散点图中 Lark 的大多数点紧贴参考线，Localization 明显偏离。

### 跨任务泛化：手部关键点检测（Table 5）

使用 RenderedHandPose (RHD) 作为基类 session，Hand3DStudio (H3D) 和 FreiHand (FHD) 作为增量 session：

| 方法 | Base | S1 | S2 | Avg | PD |
|------|------|----|----|-----|-----|
| Global（全参数更新） | 65.64 | 52.73 | 53.65 | 57.34 | 11.99 |
| Frozen（冻结 backbone） | 65.64 | 52.34 | 54.21 | 57.39 | 11.43 |
| Lark | 65.64 | 55.04 | 57.32 | **59.33** | **8.32** |

Lark Avg 提升 +1.94%（超过次优结果 57.39），PD 降低 3.11%（11.43 -> 8.32）。


---

## 主要贡献与局限性

### 主要贡献
1. **提出 Lark 方法**：首次将知识定位与低秩矩阵更新结合用于 FSCIL，适用于 ViT 等大视觉模型。
2. **高效的 Fisher 信息估计**：使用隐藏状态而非参数梯度计算 Fisher 信息矩阵，大幅降低计算复杂度，使大模型的敏感性分析变得可行。
3. **MHSA 模块编辑策略的深入分析**：通过注意力分布对比（Figure 4）和注意力矩阵相似度分析（Table 1），论证了仅编辑 W_Value 的充分性，为 Transformer 的模型编辑提供了更精细的指导。
4. **SVD 秩一近似**：通过保留最大奇异值分量实现秩为 1 的最优低秩逼近，最小化参数扰动（Figure 6 可视化验证）。
5. **跨任务泛化验证**：在手部关键点检测任务上验证了方法的通用性（Table 5），表明其适用于多种视觉任务中的增量学习场景。

### 局限性
1. **Fisher 信息矩阵基于固定数量的样本 (J=5)**：样本数量选择缺乏消融验证，不同样本数对定位精度的影响未知。
2. **编辑矩阵数量 K=3 为固定值**：缺乏对 K 的敏感性分析，不同数据集/任务的最优 K 可能不同。
3. **目标隐藏状态 nu 通过常规训练获得**：这意味着每次增量 session 仍需少量完整的训练过程，未完全消除计算开销。
4. **实验仅在 ViT-B/16 上进行**：未在更大规模 ViT（如 ViT-L）或其他 Transformer 变体（如 Swin、DeiT）上验证。
5. **某些 prompt/CLIP 方法在 PD 上仍有优势**：Lark 的 PD 略逊于 CPE-CLIP 等利用了额外语义信息和可学习 prompt 的方法。
6. **PD 指标未完全消除**：Lark 仍存在一定程度的性能下降（CIFAR100 上 7.74%~13.50%），距离零遗忘还有差距。
7. **仅验证图像分类和关键点检测**：未在分割、检测等更复杂视觉任务上评估。
8. **Fisher 信息估计需对全部旧类重新计算**：随着增量 session 增加，旧类别累积，计算量线性增长。

---

## 对后续研究的启发与潜在改进方向

1. **动态 K 值选择**：当前 K=3 固定，可设计自适应机制根据任务复杂度或增量类别数动态调整待编辑的层数。
2. **多尺度 Fisher 信息融合**：同时考虑隐藏状态和参数梯度两个层面的 Fisher 信息，实现更精准的知识定位。
3. **扩展到更多 Transformer 变体**：在 ViT-L、Swin Transformer 等架构上验证，探索不同注意力机制下的编辑策略差异。
4. **联合 prompt/低秩方法**：将 Lark 的低秩更新与可学习 prompt 结合，进一步提升性能。
5. **无需常规训练的目标状态估计**：探索不依赖完整训练过程即可获得目标隐藏状态 nu 的方法，降低增量阶段计算开销。
6. **扩展到检测与分割**：作者已在结论中提到这一方向，将 Lark 应用于目标检测、语义分割等复杂增量学习场景。
7. **理论分析**：对 Rank-One 更新与灾难性遗忘之间的关系进行深入理论分析，提供低秩逼近误差与遗忘率之间的理论界。
8. **时间与空间效率评估**：评估在大规模视觉模型上的实际时间/空间开销，确保部署可行性。
9. **多模态扩展**：将该方法推广到视觉-语言等多模态预训练模型的增量学习中。
10. **Fisher 信息的递归更新**：设计在线递归更新 Fisher 信息矩阵的方案，避免每次增量需重新估计全部类别的 Fisher 信息。
11. **J 值（每类样本数）的敏感性分析**：系统研究不同 J 值对定位精度和最终性能的影响，平衡计算开销与定位效果。

---

*总结生成时间：2026-05-18*
