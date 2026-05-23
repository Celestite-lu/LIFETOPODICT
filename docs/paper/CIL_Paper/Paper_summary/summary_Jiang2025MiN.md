# Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning -- 论文详细总结

---

## 基本信息

- **标题**: Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning (MIN)
- **作者**: Kai Jiang (西北工业大学), Zhengyan Shi (上海交通大学), Dell Zhang, Hongyuan Zhang (香港大学), Xuelong Li (中国电信人工智能研究院)
- **发表年份/会议**: NeurIPS 2025
- **代码链接**: https://github.com/ASCIIJK/MiN-NeurIPS2025

---

## 研究动机与问题定义

### 核心痛点

该论文针对**基于预训练模型 (Pre-Trained Model, PTM) 的类增量学习 (Class-Incremental Learning, CIL)** 中的一个关键问题：尽管 PTM 具有很强的泛化能力，但在连续下游任务的微调过程中，即使是轻量级微调仍会导致**参数漂移 (parameter drift)**，从而损害 PTM 的泛化能力，导致对旧任务知识的灾难性遗忘。

### 论文的独特视角

论文从一个新颖的**噪声视角**重新思考 CIL：
1. 参数漂移可以被概念化为一种**对旧任务具有破坏性的噪声**，它模糊了先前任务学习到的关键识别模式。
2. 然而，噪声并非总是有害的。预训练模型学到的大量视觉模式可能被单个任务"滥用"（即任务决策边界过度依赖某些冗余特征），而**引入适当的噪声可以抑制一些低相关性的特征**，从而为未来的任务留下空间（margin）。
3. 联合训练 (joint training) 引入的噪声能够抑制跨任务的混淆模式，产生积极作用。因此，论文的目标是**直接为每个新任务学习这种"有益的噪声" (beneficial noise / Positive-Incentive Noise, Pi-Noise)**，将其嵌入到中间特征中，以屏蔽低效模式的响应，从而在适应新任务的同时保持主干的泛化能力。

### 问题设定

- **Exemplar-free 设定**: 不保留任何旧任务的数据样本。
- **PTM-based CIL**: 使用预训练 ViT 作为主干，冻结大部分参数，仅微调少量参数。
- 模型记为 F_t = {F_t, W_t}，分解为预训练主干 F_t 和分类器 W_t。

---

## 方法/框架（重点）

### 总体思路

MIN 的核心思想是：**冻结 PTM 的原始参数，为每个任务学习一个任务特定的 Pi-Noise 生成器**，在 Transformer 块之间注入有益噪声来调整中间特征，从而在不改变主干参数的前提下适应新任务。同时，通过一个**噪声混合 (Noise Mixture)** 机制，将来自不同任务的噪声进行自适应加权组合，避免推理时多次通过主干网络。分类器权重使用**解析学习 (Analytic Learning)** 方法递归更新。

### 框架结构

#### 1. Pi-Noise Layer（Pi-噪声层）

- 插入在 ViT 的 Transformer 块之间。设主干 F = {f_1, ..., f_L} 包含 L 个块，则修改后的前向传播为：
  r_l = P_l(f_l(r_{l-1}))
  其中 P_l 为第 l 个 Pi-Noise 层。
- 每个 Pi-Noise 层包含：
  - 一个**共享的下投影层** W_down in R^{d1 x d2}：将中间特征从高维 d1 降到低维 d2（元素从标准正态分布采样，**不参与训练**）。
  - 一个**共享的上投影层** W_up in R^{d2 x d1}：将噪声重新映射回 d1 维（同样随机初始化且不训练）。

#### 2. Noise Expansion（噪声扩展）

- 为**每个任务 t** 学习一个独立的噪声生成器 P_t，由两个 MLP 层组成：
  - phi_t^mu: 生成均值向量 mu_t
  - phi_t^sigma: 生成方差向量 sigma_t
- 噪声生成过程（使用 reparameterization trick）：
  epsilon_t = epsilon * phi_t^sigma(r_l * W_down) + phi_t^mu(r_l * W_down)
  其中 epsilon ~ N(0, I) 从标准正态分布采样。
- P_t 结构非常简单：仅两个 MLP 层，由于 d2 << d1，每任务可训练参数量约等于两个 d2 x d2 方阵的参数。
- 学习新任务时，**仅训练 P_t，之前所有任务的噪声生成器保持冻结**。

#### 3. Noise Mixture（噪声混合）

- 经过 Noise Expansion，每个任务都会产生一个噪声 {epsilon_1, ..., epsilon_t}。
- 简单平均会忽略任务间的差异（相似任务应分配更高权重，冲突任务的噪声可能有害）。
- 使用一组**可学习的权重 omega** 动态调整各任务噪声的混合比例。
- **权重初始化**：基于任务原型（均值向量）的余弦相似度，通过 softmax 温度缩放初始化：
  s_{t,i} = (mu_t * mu_i) / (||mu_t|| * ||mu_i||)
  w_i = exp(s_{t,i} / tau) / SUM_{j=1}^{t} exp(s_{t,j} / tau)
  其中 tau 为温度系数，默认设为 2。
- 噪声混合操作：
  phi({epsilon_1, ..., epsilon_t}) = SUM_{i=1}^{t} epsilon_i * omega_i
- 混合后的噪声通过上投影层嵌入中间特征：
  r_hat_l = r_l + phi({epsilon_1, ..., epsilon_t}) * W_up

#### 4. 训练流程（Algorithm 1）

每个任务 t 的训练分为三个步骤：

**步骤 1 -- 解析更新分类器**：使用当前任务数据，通过 Analytic Learning 的递归公式（Eq. 3）更新分类器 W_t：
  W_t = [W_{t-1} - R_t * F(X_t)^T * F(X_t) * W_{t-1},  R_t * F(X_t)^T * Y_t]
  其中 R_t 为自相关矩阵 (autocorrelation matrix) 的递归更新：
  B_t = (I + F(X_t) * R_{t-1} * F(X_t)^T)^{-1}
  R_t = R_{t-1} - R_{t-1} * F(X_t)^T * B_t * F(X_t) * R_{t-1}

**步骤 2 -- 学习有益噪声**：
  - 扩展新的噪声生成器 P_t
  - 基于 Eq. 12 初始化混合权重 omega
  - 构建一个辅助分类器 W_aux（所有元素初始化为零）
  - 使用交叉熵损失优化 P_t、omega 和 W_aux，损失函数为：
    L_cls = l(z_L * W_aux,  y - z_L * W_t)
    其中 z_L 为骨干网络输出特征，W_t 在训练中**不参与梯度更新**。本质上是让辅助分类器拟合真实标签与当前主分类器输出的残差。

**步骤 3 -- 再次解析更新分类器**：用更新后的 P_t（即更新后的特征提取器）重新通过 Eq. 3 更新 W_t，然后丢弃 W_aux。

#### 5. 分类器（Analytic Learning Baseline）

- 基于 ACIL [Zhuang et al., NeurIPS 2022] 的解析学习方法，将分类问题转化为带 L2 正则化的线性回归：
  argmin_W ||Y - F(X)W||_2^2 + lambda * ||W||_2^2
  其闭式解为：
  W = (F(X)^T F(X) + lambda * I)^{-1} F(X)^T Y
- 该方法的递归形式使得增量更新仅需要当前任务数据、上一任务分类器权重和一个固定大小的自相关矩阵 R_t。

---

## 实验设置

### 数据集

在**6个基准数据集**上进行实验：

| 数据集 | 类别数 | 训练样本 | 测试样本 | 图像平均尺寸 |
|--------|--------|----------|----------|-------------|
| CIFAR-100 | 100 | 50,000 | 10,000 | 32x32 |
| CUB-200 | 200 | 9,430 | 2,358 | 467x386 |
| ImageNet-A | 200 | 5,960 | 1,515 | 443x427 |
| ImageNet-R | 200 | 24,000 | 6,000 | 443x427 |
| FOOD-101 | 101 | 75,750 | 25,250 | 496x475 |
| Omnibenchmark | 300 | 89,697 | 5,985 | 764x581 |

### 增量设定 (Protocols)

- 将每个数据集划分为 T 个任务（steps），每个任务包含相同数量的类别。
- T in {5, 10, 20, 50}，其中 50-steps 设置难度最大。
- 任务学习顺序由随机种子 1993 确定。
- 每个实验重复 **3次**，报告平均结果和标准差。

### 评估指标

- **平均准确率 A_bar**: 所有 T 个任务完成后的平均测试准确率，A_bar = (1/T) * SUM_{t=1}^{T} A_t
- **最终任务准确率 A_T**: 第 T 个任务完成后在所有已见类别上的测试准确率

### 预训练主干网络

两种 ViT-B/16 变体：
1. **ViT-B/16-IN21K**: 在 ImageNet-21K 上预训练
2. **ViT-B/16-IN1K**: 在 ImageNet-21K 预训练后，在 ImageNet-1K 上进一步微调

### Baselines（10种对比方法）

L2P [CVPR 2022]、DualPrompt [ECCV 2022]、CODA-Prompt [CVPR 2023]、SLCA [ICCV 2023]、FeCAM [NeurIPS 2023]、RanPAC [NeurIPS 2023]、APER [IJCV 2024]、EASE [CVPR 2024]、COFiMA [ECCV 2024]、MOS [AAAI 2025]。此外，ACIL [NeurIPS 2022] 作为 baseline。

### 实现细节

- **优化器**: SGD with momentum
- **批次大小**: 128
- **训练轮数**: 10 epochs
- **学习率**: 初始 0.001，按余弦退火衰减至 0 (cosine annealing)
- **隐藏层维度 d2**: 192（根据 5.3 节超参数分析选定）
- **温度系数 tau**: 2（用于 Eq. 12 权重初始化）
- **Baseline 超参数**: 正则化系数 gamma = 100 或 500，缓冲区大小 = 16384
- **硬件**: 双路 NVIDIA GeForce RTX 4090 GPU，Intel Xeon Gold 6244 CPU，256 GB (8x32GB DDR4) 内存
- **软件框架**: PyTorch [Paszke et al., NeurIPS 2019]，使用 Pilot [Zhou et al., TPAMI 2024] 库重现对比方法

---

## 核心结果与发现

### 主要实验结果

1. **CIFAR-100 和 CUB-200（Table 1, ViT-B/16-IN21K）**：
   - MIN 在全部 3 种设定（10/20/50 steps）下均取得最优，且在 **50-steps 下优势尤为显著**。
   - CIFAR-100 10-steps: MIN Avg 95.12%, Last 92.12%；按Avg排名第二为 COFiMA（Avg 93.87%, Last 89.77%），按Last排名第二为 MOS（Avg 93.83%, Last 90.19%）。
   - CIFAR-100 50-steps: MIN Avg 93.63%, Last 89.82%；第二名 MOS Avg 92.36%, Last 87.39%。
   - CUB-200 10-steps: MIN Avg 94.00%, Last 91.22%；第二名 RanPAC Avg 93.30%, Last 89.78%。
   - CUB-200 50-steps: MIN Avg 93.21%, Last 89.95%；第二名 MOS Avg 91.99%, Last 87.59%。

2. **ImageNet-A 和 ImageNet-R（Table 2, ViT-B/16-IN21K）**：
   - ImageNet-A 上 MIN 的优势最大：10-steps 最终准确率 64.32%，领先 MOS (57.14%) 约 7.2 个百分点；20-steps 领先 9.4 个百分点；50-steps 领先 5.9 个百分点。
   - ImageNet-R 上 MIN 领先约 2-3 个百分点。

3. **FOOD-101 和 Omnibenchmark（Table 3, ViT-B/16-IN21K）**：
   - MIN 在所有设定下均保持最优。FOOD-101 50-steps 表现尤为突出: MIN Avg 93.60%, Last 89.47%；第二名 MOS Avg 91.57%, Last 87.08%。

4. **不同随机种子的稳健性（Figure 6）**：
   - 使用 {1993, 1994, 1995, 1996, 1997, 1998} 六种随机种子在 ImageNet-R 上测试（两种 ViT 变体），MIN 在所有种子下均一致优于其他方法 SLCA、FeCAM、APER、MOS。

5. **增量趋势分析（Figure 3-5, ViT-B/16-IN1K）**：
   - MIN 的优势随任务数增加而**逐渐放大**：CIFAR-100 从 5-step 到 50-step，MIN 对第二名的性能差距从 0.76% 扩大到 4.28%。CUB-200 从 0.86% 扩大到 2.12%。
   - 表明 MIN 在更长的增量序列（更多任务）中具有明显更好的抗遗忘稳定性。

### 消融实验（Table 4）

在全部 6 个数据集上使用 ViT-B/16-IN21K 进行 10-steps 消融实验：

- **Baseline**: 仅使用解析分类器 + 冻结 PTM 骨架，不做任何噪声注入。例如 CIFAR-100 Avg 91.18%, Last 86.82%。
- **w/ NE + epsilon_bar**: 使用 Noise Expansion + 对多任务噪声取简单平均（无 Noise Mixture 机制，直接使用 W_t 计算 CE loss 而非辅助分类器）。相比 baseline 显著提升（CIFAR-100 Avg 94.07%, Last 90.22%）。
- **w/ NE + epsilon_mu**: 仅使用噪声的均值分量 mu（不使用方差分量 sigma）。效果大幅下降（CIFAR-100 Avg 89.27%, Last 85.63%），甚至低于 baseline，验证了方差分量 sigma 对噪声有益性的关键作用。
- **w/ NE + epsilon_sigma**: 仅使用噪声的方差分量 sigma。效果接近完整方法（CIFAR-100 Avg 94.37%, Last 91.32%），说明 sigma 分量是效果的主要贡献者。
- **w/ NE + epsilon_t**: 仅使用最后一个（当前）任务的噪声，不使用历史任务噪声。效果差（CIFAR-100 Avg 92.36%, Last 87.92%），验证了历史有益噪声的累积贡献。
- **w/ NE + epsilon_i**: 随机选择一个任务的噪声。效果一般（CIFAR-100 Avg 92.65%, Last 88.10%）。
- **MIN（完整方法）**: 在所有 6 个数据集上取得最佳结果，验证了 Noise Mixture 可学习权重 + 辅助分类器的有效性。

### 可视化分析

Grad-CAM 可视化（Figure 2 及补充材料 Figure 9）显示：
- Baseline 方法的激活区域更分散，对不相关区域（如背景墙壁、画作）产生较强响应，易导致错误分类。
- MIN 更关注目标主体关键部位（如花瓶主体、鸟类头部/羽色细节），有效**抑制了无关背景信息的激活**。
- 在 CUB 细粒度分类中，MIN 能够抑制不同鸟类间的共性特征（如体型轮廓），增强对判别性细粒度特征（如特定羽色斑纹）的激活响应，从而避免跨任务决策边界混淆。

### 参数效率分析（Figure 8）

- MIN 总参数量约 **94.2M**，略高于大多数对比方法（约 85-89M），因为下/上投影层引入了额外参数（约为主干 ViT-B 的 4%）。
- 但**每任务可训练参数量**仅约 **0.8M**（远小于大多数方法），增量学习时的计算和存储开销很低。
- Pi-Noise 与 Adapter、VPT-shallow、VPT-deep 等替代特征适配方案相比，在 ImageNet-R 10-steps 设定下表现出明显优势（Figure 8c）。

### 超参数稳健性

- **正则化系数 gamma 和缓冲区大小（Figure 7a）**: gamma 在 100-500 之间、缓冲区在 8192-16384 之间效果稳定且较好。
- **隐藏层维度 d2（Figure 7b/7c）**: d2 = 192 时性能最佳；继续增大维度（如 256, 384, 512, 768）导致性能下降（可能由于过拟合或噪声过于复杂）。此时可训练参数量相应增大（d2=768 时约 7.68M）。
- **温度系数 tau（Table 5）**: 在 {0.5, 1.0, 1.5, 2.0} 范围内，ImageNet-R 10-steps 下 A_bar 在 84.79%-85.18% 之间波动很小，方法对 tau 不敏感。

---

## 主要贡献与局限性

### 主要贡献

1. **新视角**: 从"噪声"的视角重新审视类增量学习中的遗忘问题，将参数漂移量化为破坏性噪声，并提出学习"有益噪声"（Pi-Noise）来抑制跨任务混淆模式，为该领域提供了新的概念框架。

2. **方法设计**:
   - **Noise Expansion**: 为每任务学习独立的 Pi-Noise 生成器（2层轻量 MLP），旧噪声生成器冻结，结构化简单但高效。
   - **Noise Mixture**: 通过可学习权重动态混合多任务噪声，权重以任务原型余弦相似度初始化（利用信息论启发），无需多次推理通过主干即可实现跨任务噪声整合。
   - 与 **Analytic Learning** 解析分类器（ACIL）无缝集成，三步训练流程清晰、数学基础扎实。

3. **卓越性能**: 在 6 个基准数据集、10 种对比方法、多种增量设定下取得 SOTA，特别是在 50-steps 长序列增量设定中优势显著（相对于第二名的性能优势随任务数放大），验证了方法的强可扩展性。

4. **高效参数**: 每任务仅需训练约 0.8M 参数，与需要大量可训练参数的方法（如 prompt-based、adapter-based）相比具有明显效率优势。

### 局限性（论文附录 H 自述 + 客观分析）

1. **对预训练质量的依赖**: MIN 依赖高质量 PTM 提取的先验信息来指导 Pi-Noise 生成器的适配。论文指出随着自监督预训练技术（如 DINOv2、MAE）的发展，高质量 PTM 越来越易得，这一限制正在减轻。

2. **额外的存储参数开销**: 每个 Pi-Noise 层需要额外的下/上投影层（随机初始化、不训练），引入约为主干 4% 的额外总参数量。论文认为这是实现简单性和低成本训练的合理权衡。

3. **架构验证范围有限**: 所有实验均基于 ViT-B/16，未验证在 CNN 架构、其他 ViT 变体（Swin、DeiT）或更大规模模型上的通用性。

4. **任务顺序敏感性问题未充分探讨**: 虽然 Figure 6 测试了 6 种随机顺序，但未系统分析任务语义相似度对 Pi-Noise 质量的调制效应。

5. **仅适用于任务增量 (Task-IL) 批处理设定**: 方法假定已知任务边界 (task boundary)，未探讨在在线 CIL、任务无关 (task-agnostic) 或域增量 (domain-IL) 设定下的扩展性。

6. **对第一个任务的处理**: 论文未详细说明第一个任务 (Task 1) 的 Pi-Noise 训练策略。从 Algorithm 1 看，步骤 1 首先更新分类器，然后才"扩展新的噪声生成器 P_t"。第一个任务仅有 W_1 的解析更新，P_1 在步骤 2 训练。但在推理时第一个任务也需要噪声，可能存在冷启动问题。

---

## 对后续研究的启发或潜在改进方向

1. **Pi-Noise 理论基础的深化**: 论文基于 Positive-Incentive Noise [Li, IEEE TNNLS 2024] 的框架，但"何为有益噪声"的理论理解仍较初步。未来可从信息论（如互信息最大化、条件熵最小化）或学习理论角度严格定义和量化噪声的"有益性"，为噪声生成提供原则性指导。

2. **跨架构泛化验证**: 将 Pi-Noise 机制扩展到 CNN（如 ResNet）、其他 ViT 变体（Swin, DeiT, MAE）、多模态基础模型（CLIP, LLaVA），甚至 NLP 大模型的持续微调场景。

3. **任务顺序自适应与课程学习**: 当前方法虽通过可学习权重实现一定程度的自适应，但噪声生成过程独立于任务间语义关系。可探索基于任务间相似度预分析（如通过 text embedding）的课程学习策略，或任务顺序鲁棒的正则化方法。

4. **噪声空间的共享与组合**: 当前每任务拥有完全独立的噪声生成器。可探索：共享的噪声子空间 + 任务特定投影、层次化噪声生成（粗粒度 + 细粒度）、或基于超网络 (hypernetwork) 的噪声生成器参数预测。

5. **与其他 CIL 策略的正交融合**: MIN（特征层面噪声注入）可与提示学习（prompt-based, 如 L2P/CODA-Prompt 的输入空间适配）、权重集成（weight ensemble, 如 COFiMA/MOS 的参数空间融合）等策略形成三维互补。

6. **扩展到更复杂的增量学习场景**: 包括在线/流式 CIL（去除任务边界假设）、任务无关 CIL（task-agnostic，推理时不知任务 ID）、少样本 CIL (few-shot CIL)、以及持续强化学习等场景。

7. **更大规模与领域的实际部署验证**: 在医学影像（病理、CT）、遥感、自动驾驶感知、工业缺陷检测等实际领域进行验证，评估 Pi-Noise 在分布外 (OOD)、领域漂移场景下的鲁棒性。

8. **噪声属性的多维度研究**: 探索噪声的极性（正向/负向/选择性）、强度（scale）、稀疏性（sparsity）、频率特性（低频/高频噪声）等多维度设计空间，以及任务语义距离与最优噪声属性之间的关系。

9. **理论收敛性与泛化界分析**: 为 Pi-Noise 辅助的增量学习过程建立理论收敛性保证和泛化误差界，特别是噪声注入对增量学习遗憾界 (regret bound) 的影响。

---

*总结生成日期: 2026年5月18日*
