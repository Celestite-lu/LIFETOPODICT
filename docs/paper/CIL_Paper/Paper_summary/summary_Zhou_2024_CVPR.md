# Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning —— 详细中文总结

---

## 一、基本信息

- **论文标题**：Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning
- **作者**：Da-Wei Zhou, Hai-Long Sun, Han-Jia Ye（通讯作者）, De-Chuan Zhan
- **单位**：南京大学 计算机软件新技术国家重点实验室 / 人工智能学院
- **发表信息**：CVPR 2024
- **代码地址**：https://github.com/sun-hailong/CVPR24-Ease

---

## 二、研究动机与问题定义

### 2.1 核心痛点

类增量学习（Class-Incremental Learning, CIL）面临稳定性-可塑性困境（stability-plasticity dilemma）：学习新类时不可避免地覆盖旧知识，造成灾难性遗忘（catastrophic forgetting）。在预训练模型（PTM）时代，虽然 PTM 的通用特征表示能力缓解了部分遗忘问题，但一个根本矛盾仍然存在——**过度修改网络导致遗忘，修改太少又无法充分拟合新类**。

### 2.2 现有方法的局限

**Prompt-based 方法**（L2P [CVPR 2022]、DualPrompt [ECCV 2022]、CODA-Prompt [CVPR 2023]）：在冻结的 PTM 上附加可学习的 prompt 池。虽然冻结了预训练权重，但优化新任务所需的 prompt 不可避免地覆盖旧任务的 prompt，本质上仍导致新旧任务间的冲突与遗忘。

**模型扩展方法**（DER [CVPR 2021]、FOSTER [ECCV 2022] 等 expandable networks）：为每个新任务学习一个独立的完整 backbone，避免跨任务特征冲突。但它存在两个关键缺陷：
1. **存储开销巨大**：对大型 PTM 而言，保存多个完整 backbone 不切实际；
2. **依赖 exemplar**：需要通过旧类样本（exemplar）来训练统一的大分类器以校准所有类别。

### 2.3 本文目标

本文旨在回答一个核心问题：**能否以低成本构建任务特定子空间（task-specific subspace），在不使用 exemplar 的前提下解决跨任务冲突？**

具体而言需要解决两个子挑战：
1. 用轻量级模块（而非整个 backbone）创建和保存任务特定子空间；
2. 在无 exemplar 条件下，开发能映射不断扩展的高维特征到所有类别的分类器。

---

## 三、方法/框架：EASE（Expandable Subspace Ensemble）

EASE 包含三个核心组件：(1) 基于 Adapter 的子空间扩展；(2) 语义引导的原型补全；(3) 子空间加权集成。

### 3.1 基于 Adapter 的子空间扩展（Subspace Expansion with Adapters）

**动机**：借鉴模型扩展思想（DER/FOSTER），但避免为每个任务保存完整 backbone。EASE 使用轻量级 Adapter 来创建任务特定的特征子空间。

**Adapter 结构**（遵循 AdaptFormer [8]）：
- 对 ViT 每一层的 MLP 模块添加一个旁路（side branch）bottleneck 模块
- 结构：下投影层 W_down (d x r) -> 非线性激活 sigma -> 上投影层 W_up (r x d)
- 公式（Eq. 4）：
  ```
  x_o = sigma(x_i W_down) W_up + MLP(x_i)
  ```
  其中 x_i 和 x_o 分别为 MLP 层的输入和输出，Adapter 以残差形式调节 MLP 输出

**训练策略**：
- 冻结所有预训练权重和之前任务的 Adapter，仅优化当前任务的新 Adapter
- 对每个增量任务 b，初始化一个全新的 Adapter A_b
- 训练目标（Eq. 5）：min_{A, W} sum_{(x,y) in D^b} L(W^T phi(x; A), y)，其中 L 为交叉熵损失
- 训练完成后，每个任务拥有独立的 Adapter，将所有子空间特征拼接得到整体嵌入（Eq. 6）：
  ```
  Phi(x) = [phi(x; A_1), phi(x; A_2), ..., phi(x; A_b)] in R^{b*d}
  ```

**参数开销**：保存所有 Adapter 的参数量为 B x L x 2dr，其中 B 为任务数，L 为 Transformer 层数，d 为特征维度，r 为投影维度。相比保存完整 backbone（需 B 个完整模型），开销极小。

**原型分类器**：
- 每个增量阶段训练后，提取各类在当前 Adapter 子空间中的类原型（Eq. 7）：
  ```
  p_{i,b} = (1/N) sum_{j=1}^{|D^b|} I(y_j = i) * phi(x_j; A_b)
  ```
  其中 N 为类 i 的样本数
- 即该类所有样本在该子空间中嵌入的均值向量
- 用原型拼接向量 P_i = [p_{i,1}, p_{i,2}, ..., p_{i,b}] 作为类 i 的分类器，采用余弦相似度进行预测
- 注意：训练时使用的临时线性分类器 W 在每个阶段结束后即被丢弃

### 3.2 语义引导原型补全（Semantic Guided Prototype Complement）

**问题**：当新 Adapter A_b 被训练后，旧类的训练数据不可用，旧类在新子空间中的原型无法计算，导致原型与嵌入之间的维度不匹配。

**符号化形式**：给定两个子空间（旧/新）和两个类集（旧/新），根据以下已知量估计 P_hat_{o,n}（旧类在新子空间中的原型）：
- P_{o,o}：旧类在旧子空间中的原型（可获取）
- P_{n,o}：新类在旧子空间中的原型（可获取，新类数据可通过旧 Adapter 前向传播得到）
- P_{n,n}：新类在新子空间中的原型（可获取）
- P_hat_{o,n}：旧类在新子空间中的原型（**需要估计**，无数据可用）

**核心假设**：类别间的语义相似性在不同子空间中是可共享的。例如，猫与狮子的语义关系在不同 Adapter 子空间中保持一致性。

**算法步骤**：

**Step 1**：在共现空间（co-occurrence space，即旧子空间，旧类和新类原型均存在）中计算旧类 i 与新类 j 之间的余弦相似度（Eq. 8）：
```
Sim_{i,j} = (P_{o,o}[i] / ||P_{o,o}[i]||_2) * (P_{n,o}[j]^T / ||P_{n,o}[j]||_2)
```

**Step 2**：对每个新类 j，在所有旧类 i 上进行 softmax 归一化：
```
S_tilde_{i,j} = exp(Sim_{i,j}) / sum_i exp(Sim_{i,j})
```
即对于每个新类 j，计算各个旧类 i 与 j 的相对相关性（在 i 维度上归一化）。

**Step 3**：在新子空间中，用新类原型的加权组合重建旧类原型（Eq. 9）：
```
P_hat_{o,n}[i] = sum_j S_tilde_{i,j} * P_{n,n}[j]
```
即旧类 i 在新子空间中的原型 = 所有新类的新子空间原型的加权和，权重为该旧类与各新类在共现空间中的归一化相似度。

**特点**：该原型补全过程完全无需训练（training-free），计算高效。

### 3.3 子空间加权集成（Subspace Ensemble via Subspace Reweight）

完成所有 Adapter 扩展和原型补全后，获得完整原型矩阵（Eq. 10）：

```
[ P_{1,1}   P_hat_{1,2}   ...   P_hat_{1,B} ]
[ P_{2,1}   P_{2,2}        ...   P_hat_{2,B} ]
[   ...       ...          ...      ...      ]
[ P_{B,1}   P_{B,2}        ...   P_{B,B}     ]
```

其中主对角线及以下为真实原型（用相应数据计算得出），主对角线以上为通过 Eq. 9 语义映射估计的原型。

**推理时 logit 计算**（Eq. 11）：任务 b 的 logit 等于所有子空间中原型-嵌入匹配得分的总和：
```
[P_{b,1}, P_{b,2}, ..., P_{b,B}]^T Phi(x) = sum_i P_{b,i}^T phi(x; A_i)
```

**重加权策略**（Eq. 12）：Adapter A_b 是专门为任务 b 训练的，其对应子空间的原型应该更适用于分类该任务类别。引入权重系数 alpha：
```
P_{b,b}^T phi(x; A_b) + alpha * sum_{i != b} P_{b,i}^T phi(x; A_i)
```
- alpha 为折中参数，默认设为 **0.1**
- 即：匹配子空间（diagonal, i=b）权重为 1，其他子空间（i != b）权重为 alpha = 0.1

### 3.4 训练流程总结

1. **对每个增量任务 b**：初始化新 Adapter A_b，冻结 PTM 和旧 Adapter，在 D^b 上训练（Eq. 5）
2. **原型提取**：用所有已有的 Adapter（A_1 至 A_b）分别计算当前数据 D^b 中各类的原型
3. **原型补全**：通过语义映射（Eq. 8-9）合成旧类在新子空间中的原型
4. **构建完整分类器**：组装原型矩阵（Eq. 10）
5. **推理**：使用重加权后的多子空间集成进行预测（Eq. 12）

---

## 四、实验设置

### 4.1 数据集

共 7 个基准数据集：

| 数据集 | 类别数 | 类型 |
|--------|--------|------|
| CIFAR100 | 100 | 标准 CIL 基准 |
| CUB200 | 200 | 细粒度分类 |
| ImageNet-R | 200 | OOD（包含多种风格/领域变体） |
| ImageNet-A | 200 | OOD（自然对抗样本） |
| ObjectNet | 200 | OOD（偏置控制数据集） |
| OmniBenchmark | 300 | OOD（多视觉领域基准） |
| VTAB | 50 | 多任务视觉适应基准 |

### 4.2 数据划分

- 采用 "B-m Inc-n" 表示法：第 1 阶段 m 个类，后续每个增量阶段 n 个类
- 随机种子 1993 打乱类顺序（遵循 iCaRL [46] 协议）
- 训练/测试集划分与 ADAM [77] 保持一致
- **完全 exemplar-free 设置**：不保存任何旧类样本

### 4.3 预训练模型

- **ViT-B/16-IN21K**：ImageNet-21K 预训练的 ViT-Base（patch=16）
- **ViT-B/16-IN1K**：在 IN21K 基础上用 ImageNet-1K 进一步微调

### 4.4 对比方法

**PTM-based CIL 方法**：L2P [CVPR 2022]、DualPrompt [ECCV 2022]、CODA-Prompt [CVPR 2023]、SimpleCIL [77]、ADAM [77] + 五种参数高效微调变体（Finetune、VPT-S、VPT-D、SSF、Adapter）

**传统 CIL 方法**（加装相同 PTM）：Finetune、Finetune Adapter、LwF [36]、SDC [67]、iCaRL [46]、DER [64]、FOSTER [56]、MEMO [76]

### 4.5 实现细节

- **硬件**：NVIDIA 4090
- **框架**：PyTorch + Pilot（PTM-based CIL 工具箱）
- **优化器**：SGD，batch size = 48，共 20 个 epoch
- **学习率**：初始 0.01，余弦退火衰减（cosine annealing）
- **Adapter 投影维度**：r = 16
- **重加权系数**：alpha = 0.1

### 4.6 评估指标

- **A_B**：最后一个增量阶段后的分类准确率（衡量最终性能）
- **A_bar**：所有增量阶段的平均准确率，A_bar = (1/B) sum_{b=1}^{B} A_b（衡量整体增量学习表现）

---

## 五、核心结果与发现

### 5.1 主要实验结果（Table 1，ViT-B/16-IN21K 作为 backbone）

EASE 在**全部 7 个数据集**上均取得最优性能。关键数据摘录：

| 数据集配置 | EASE A_bar | EASE A_B | 对比最强 Baseline |
|------------|-----------|----------|-------------------|
| CIFAR B0 Inc5 | 91.51 | 85.80 | ADAM+Adapter (90.65 / 85.15) |
| CUB B0 Inc10 | 92.23 | 86.81 | SimpleCIL (92.20 / 86.73) |
| IN-R B0 Inc5 | 78.31 | 70.58 | ADAM+Adapter (72.35 / 64.33) |
| IN-A B0 Inc20 | 65.34 | 55.04 | ADAM+SSF (61.30 / 50.03) |
| ObjNet B0 Inc10 | 70.84 | 57.86 | ADAM+SSF (69.15 / 56.64) |
| OmniBench B0 Inc30 | 81.11 | 74.85 | ADAM+VPT-D (81.05 / 74.47) |
| VTAB B0 Inc10 | 93.61 | 93.55 | CODA-Prompt (83.90 / 83.02) |

注：A_bar 为平均准确率，A_B 为最终阶段准确率。EASE 在 OOD 数据集（IN-R、IN-A、ObjectNet）上的增益尤为显著。

### 5.2 增量性能曲线（Figure 3，ViT-B/16-IN1K 作为 backbone）

包含 6 个子图（CIFAR B0 Inc20、ImageNet-A B0 Inc20、ImageNet-R B0 Inc10、ObjectNet B0 Inc20、Omnibenchmark B0 Inc30、VTAB B0 Inc10）。随着类别数增加，EASE 始终保持显著优势。在最后一个增量阶段，EASE 领先第二名 **约 4%~7.5%**（在 ImageNet-R/A、ObjectNet、VTAB 上尤为突出）。

### 5.3 大基类设置（Figure 4，ViT-B/16-IN21K）

在 B100 Inc50 设置（前 100 类为基类，每步新增 50 类）的 ImageNet-R 和 ImageNet-A 上，EASE 仍然保持显著优势，验证了方法对不同数据划分的鲁棒性。

### 5.4 与传统 exemplar-based 方法对比（Table 2）

传统方法（iCaRL、DER、FOSTER、MEMO）需**每类保存 20 个 exemplar**，而 EASE **零 exemplar**：

| 数据集 | EASE (0 exemplar) | FOSTER (20/class) | DER (20/class) |
|--------|-------------------|-------------------|----------------|
| IN-R B0 Inc20 (A_bar / A_B) | 81.73 / 76.17 | 81.34 / 74.48 | 80.48 / 74.32 |
| CIFAR B0 Inc10 (A_bar / A_B) | 92.35 / 87.76 | 89.87 / 84.91 | 86.04 / 77.93 |

EASE 在无 exemplar 条件下**超过所有使用 exemplar 的方法**。

### 5.5 参数效率（Figure 1）

在 ImageNet-R B100 Inc50 设置下：
- EASE 仅使用约 **1% 可训练参数**
- 准确率达到 **78.5%**，超过所有对比方法（第二名 FOSTER 77.0%，但 FOSTER 需训练 100% 参数；CODA-Prompt 使用 5% 参数达 76.8%）
- 相比 L2P (0.5% 参数, 74.8%) 和 DualPrompt (0.5% 参数, 72.0%)，EASE 在同等参数预算下性能大幅领先

### 5.6 消融实验（Figure 5，ImageNet-R B0 Inc20）

逐步验证三个组件的贡献（最终阶段即 200 类时的准确率）：

| 配置 | 描述 | 准确率趋势 |
|------|------|-----------|
| Vanilla PTM | 仅用预训练编码器+原型分类，不做任何适配 | 基线最低（200类时约 61%） |
| + Task-Specific Adapters | 加入可扩展 Adapter，仅用对角线原型分类 | 大幅提升（200类时约 73%） |
| + Prototype Complement | 加入语义映射补全全原型矩阵 | 进一步提升（200类时约 75%） |
| + Subspace Reweight | 加入子空间重加权（完整 EASE） | 最优（200类时约 77%） |

核心结论：
- PTM 的通用特征远不足以解决 CIL——**任务特定的适配（Adapter）至关重要**
- **原型补全**能有效利用跨任务语义信息提升分类性能
- **子空间重加权**通过强调匹配子空间（diagonal）进一步提升了决策准确性
- 三个组件均有正向贡献，且相互正交

### 5.7 t-SNE 可视化（Figure 6）

用 ImageNet-R B0 Inc5 训练两个 Adapter（A1 和 A2），t-SNE 可视化显示：
- 在 A1 子空间中：任务 1 的类（圆点）充分分离，任务 2 的类（三角形）未分离
- 在 A2 子空间中：任务 2 的类充分分离，任务 1 的类未分离
- 验证了**每个 Adapter 确实学到了任务特定的判别特征**，支持 Eq. 12 中用匹配子空间作为主要预测来源的设计合理性

### 5.8 超参数鲁棒性（Figure 7a）

在 ImageNet-R B0 Inc20 上联合扫描两超参数网格（r x alpha）：
- 投影维度 r in {8, 16, 32, 64, 128}
- 折中参数 alpha in {0.01, 0.05, 0.1, 0.3, 0.5}

结果：A_bar 在 81.5~82.9 之间小范围波动，方法对超参数非常鲁棒。推荐默认值 r=16, alpha=0.1。

### 5.9 原型补全策略对比（Figure 7b）

在 ImageNet-R/A/ObjectNet 上对比三种原型补全策略（均评估 A_bar）：

| 策略 | ImageNet-R | ImageNet-A | ObjectNet |
|------|-----------|-----------|-----------|
| Similarity（本文 Eq.8-9） | 77.3 | 58.5 | 61.4 |
| Optimal Transport (OT) | 76.0 | 57.3 | 60.9 |
| Linear Regression (LR) | 76.1 | 56.5 | 58.3 |

基于余弦相似度 + softmax 加权重组的策略效果最优，且无训练参数、计算最简单。

---

## 六、主要贡献

1. **提出了 EASE 框架**：首次将可扩展子空间思想引入 PTM-based CIL，用轻量级 Adapter 实现低成本的任务特定子空间构建，从根本上解决了跨任务特征冲突（无需覆盖旧知识即可学习新知识）。

2. **语义引导原型补全机制**：提出一种 training-free 的语义映射方法，利用类别间语义相似性在不同子空间中可共享的假设，在无 exemplar 条件下合成旧类在新子空间中的原型，实现了 exemplar-free 的完全分类器构建。

3. **子空间加权集成策略**：通过对匹配子空间赋予更高权重（alpha=0.1 用于非匹配子空间），有效融合多子空间信息进行统一决策。

4. **全面的实验验证**：在 7 个数据集（含 6 个 OOD 数据集）上全面超越 SOTA（L2P、DualPrompt、CODA-Prompt、ADAM 系列、SimpleCIL），在参数效率上显著优于传统模型扩展方法（DER、FOSTER），且无需任何 exemplar 即可超越使用 exemplar 的传统方法（每类保存 20 个 exemplar）。

---

## 七、主要局限性

1. **Adapter 的累积存储开销**：虽然每个 Adapter 仅占 backbone 参数的一小部分（文中结论部分指出约 0.3%），但随着增量任务数 B 的增长，总存储量线性增长（B x L x 2dr）。论文在结论部分明确将其列为潜在局限，并提出未来设计压缩算法来应对。

2. **原型补全的核心假设依赖**：语义引导原型补全依赖于类别间相似性在不同子空间中可共享这一假设。当域偏移极大或新类与旧类的语义关系断裂时（例如全部新类与全部旧类属于完全不同的语义域），该假设可能不成立，补全原型精度将受影响。

3. **推理计算开销随任务数线性增长**：每次推理需将输入通过所有 B 个 Adapter 分别前向传播（Eq. 6），推理延迟和计算量随任务数线性增长，可能成为实际部署的瓶颈。

4. **Adapter 之间的协同不足**：各 Adapter 在训练阶段完全独立，仅在后处理阶段（原型补全和重加权）进行集成。缺乏训练过程中的显式协同（例如对比学习约束或互信息最大化），可能导致子空间之间的冗余。

5. **类顺序效应未深入探讨**：原型补全效果依赖于每个增量阶段中至少有部分新类与旧类存在语义关联——极端情况下（如旧类全是动物、该阶段新类全是人造物），补全质量的下降程度未在论文中得到系统分析。

---

## 八、对后续研究的启发与潜在改进方向

1. **Adapter 压缩与参数共享**：论文已明确提出压缩 Adapter 是未来方向。潜在方案包括：（a）将多个旧 Adapter 知识蒸馏为一个紧凑 Adapter；（b）低秩分解与量化进一步压缩；（c）使用超网络（hypernetwork）根据任务 ID 或原型动态生成 Adapter，以消除存储的线性增长。

2. **跨 Adapter 协同训练**：在训练新 Adapter 时加入与旧 Adapter 的交互约束（如正交性约束、互信息最大化或对比学习），使不同子空间之间形成更明确的互补关系。

3. **更鲁棒的原型补全机制**：可探索基于图神经网络（GNN）捕捉高阶类间关系、cross-attention 机制做上下文感知的原型映射、或引入不确定性建模以评估补全原型的置信度并进行选择性使用。

4. **自适应 Adapter 选择/剪枝**：推理时可根据输入的语义特征选择性激活相关的 Adapter 子集，而非使用全部历史 Adapter，从而降低推理成本。

5. **非线性子空间融合**：当前采用简单的特征拼接 + 线性原型匹配，未来可设计端到端可学习的非线性融合模块来更有效地整合多子空间信息。

6. **扩展到更多持续学习场景**：EASE 的核心思想（低成本子空间扩展 + 无 exemplar 原型补全）可推广到 Task-IL、Domain-IL、FSCIL（少样本类增量学习）以及多模态持续学习场景。

7. **理论分析补全**：目前缺少原型补全误差界的理论分析（如补全误差与类间相似度、嵌入空间维度、Adapter 容量之间的关系），以及与完全模型扩展方法等价性或近似性的理论证明。

---

## 附录：关键技术公式速查

| 公式编号 | 含义 | 核心表达式 |
|----------|------|-----------|
| Eq. 4 | Adapter 残差输出 | x_o = sigma(x_i W_down) W_up + MLP(x_i) |
| Eq. 5 | 单任务 Adapter 训练目标 | min_{A,W} sum L(W^T phi(x; A), y) |
| Eq. 6 | 多子空间特征拼接 | Phi(x) = [phi(x;A_1), ..., phi(x;A_b)] |
| Eq. 7 | 类原型计算（类内样本均值） | p_{i,b} = (1/N) sum I(y_j=i) phi(x_j; A_b) |
| Eq. 8 | 类间余弦相似度（共现空间） | Sim_{i,j} = cosine(P_{o,o}[i], P_{n,o}[j]) |
| Eq. 9 | 原型语义补全（softmax加权重组） | P_hat_{o,n}[i] = sum_j Softmax(Sim)_{i,j} * P_{n,n}[j] |
| Eq. 12 | 子空间重加权推理 | P_{b,b}^T phi(x;A_b) + alpha sum_{i!=b} P_{b,i}^T phi(x;A_i) |

---

*本总结基于对 CVPR 2024 论文 "Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning" 原文的详细阅读与逐项核查生成。*
