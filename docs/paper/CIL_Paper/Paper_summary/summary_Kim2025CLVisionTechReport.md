# Kim2025CLVisionTechReport 论文详细总结

---

## 一、基本信息

- **标题**：Technical Report for CVPR 2024 5th CLVISION Challenge: Multi-Level Knowledge Distillation and Dynamic Self-Supervised Learning for Continual Learning.
- **作者**：Taeheon Kim, San Kim, Minhyuk Seo, Dongjae Jeon, Wonje Jeung, Jonghyun Choi
- **单位**：Yonsei University（延世大学）、Seoul National University（首尔大学）
- **发表信息**：CVPR 2024 第五届 CLVISION Challenge 技术报告，获得第 2 名（2nd place）
- **年份**：技术报告涉及 CVPR 2024 挑战赛，论文致谢中提及 EuroHPC 项目编号 EHPC-DEV-2025D08-065/088，表明撰文时间为 2025 年

---

## 二、研究动机与问题定义

### 2.1 背景与问题

传统类增量学习（Class-Incremental Learning, CIL）假设每个新任务引入的是完全未见过的类别，且每任务新增固定数量的类。然而，真实场景中模型可能反复遇到已见过和未见过类别的混合数据流。Hemati et al. [12] 因此提出了**带重复的类增量学习（Class-Incremental Learning with Repetition, CIR）**，允许每个任务包含新旧类别的混合。

### 2.2 核心痛点

1. **灾难性遗忘（Catastrophic Forgetting）**[20]：持续学习新任务时，模型会遗忘先前学到的知识。
2. **稳定性-可塑性困境（Stability-Plasticity Dilemma）**：需要在保持旧知识（稳定性）和快速学习新知识（可塑性）之间取得平衡。
3. **基于回放方法的局限性**：传统 rehearsal-based 方法 [3, 23, 24, 26] 需要存储以往样本，面临**数据隐私问题（data privacy concerns）**[27] 和**存储限制（storage limitations）**，在许多实际场景中不可行。
4. **外部无标签数据的利用机遇**：实践中可从互联网等开放源获取大量无标签外部数据（open-source external data），这些数据可下载、使用后丢弃，规避了存储敏感/大规模数据的问题。如何高效利用这些数据来同时增强模型的稳定性（stability）和可塑性（plasticity）是一个关键研究问题。

### 2.3 研究目标

在 CIR 设定下，利用外部无标签数据（external unlabeled data），同时提升模型的**稳定性**（防止遗忘旧知识）和**可塑性**（加速新类别的学习），最大化测试集上的图像分类最终准确率。

---

## 三、方法/框架（重点）

### 3.1 问题形式化

- **标记数据集**：D_l = {(x_i, y_i) | x_i in X_l, y_i in Y_l}, i=1 to N，标签空间 Y_l = {1, 2, ..., C}
- **无标签数据集**：D_u = {x_i' | x_i' in X_u}, i=1 to N'
- **小容量记忆缓冲区 M**：存储先前任务的相关信息（特征和 logits）
- 本文中，**特征（feature）** 定义为特征提取器（ResNet encoder）的输出；**logit** 定义为预测头（prediction head）的输出
- 目标：在测试集上最大化图像分类最终准确率

### 3.2 多层次知识蒸馏（Multi-Level Knowledge Distillation, MLKD）

MLKD 从多个旧模型（previous models）中，在多个层面（特征层和 logit 层）提取知识，以保留丰富的旧知识。该方法扩展了持续学习中常见的知识蒸馏方法 [16, 17]。

#### （1）特征级知识蒸馏（Feature KD, FKD）

- 对先前模型的特征输出施加 **L2 损失**，使模型学习到的表示在时间上保持一致性（consistency），保留嵌入在特征中的知识。
- **权重线性增长策略**：特征 KD 损失的权重随任务数量线性增长（具体为 beta = 0.002 * t，t 为任务编号）。原因是早期任务的特征可能包含的有用信息不足，若权重过大会阻碍新知识的学习。

#### （2）基于相关性的 Logit 知识蒸馏（Correlation-based Logit KD, CLKD）

- 传统基于实例对（instance-wise）的 logit 蒸馏在无标签数据上效果不佳，因为无标签数据产生的 logits 无法准确反映训练数据的知识（数据分布偏移，data distribution shift）。
- 引入基于 **Gram 矩阵**的蒸馏机制 [13]，从**批次内实例相关性**和**类别间相关性**两个层面捕捉知识：

```
L_LogitKD = (1/B) * sum_{k=1}^{K} ||G_curr^k - G_prev^k||^2
          + (1/C) * sum_{k=1}^{K} ||M_curr^k - M_prev^k||^2
```

- 第一项：蒸馏批次内实例间的相关性（instance-wise correlation within the batch）
- 第二项：蒸馏类别间的相关性（class-wise correlation）
- 其中 G 和 M 为 Gram 矩阵。对于 logit 矩阵 l in R^{B x C}（B 为 batch size，C 为类别数），G = l * l^T in R^{B x B}（实例间），M = l^T * l in R^{C x C}（类别间）
- K 为先前模型数量

#### （3）多先前模型集成（Multiple Previous Models, MPM）

- 保留每个任务结束时训练好的模型（每个任务保存一个）
- 在蒸馏时根据输出**置信度（confidence）**选择性地组合多个模型的 feature 和 logit，增强知识迁移的鲁棒性，补偿数据分布偏移

#### （4）指数移动平均更新（Exponential Moving Average, EMA）

- 通过 EMA 方式**逐步更新所有先前模型**，将当前模型的新知识平滑融入蒸馏过程，既保留了旧知识，又反映了新知识，促进了模型的可塑性（plasticity）

### 3.3 动态自监督学习（Dynamic Self-Supervised Learning）

#### （1）自监督学习的引入

- 在无标签数据集上施加自监督损失（SSL），促使模型提取**通用特征（general features）**，使其能够识别未见过类别的相关特征，从而加速新类别的学习（增强可塑性）。

#### （2）动态权重机制

- 简单粗暴地施加 SSL 会干扰主任务（图像分类）的学习。因此引入**动态衰减权重**以平衡 SSL 的影响：

```
L(X_l, X_u) = (1 - 0.1 * alpha) * L_ACE(X_l) + alpha * L_SSL(X_u)
```

- **alpha = c * omega^t**，其中 c 和 omega 是超参数，t 为任务编号（task number）
- omega 设置为**小于 1**（实际取值 0.95），确保 alpha 随任务数增加而**指数衰减**
- 效果：训练初期 SSL 权重较大，帮助模型快速建立通用特征提取能力；训练后期 SSL 权重递减至接近零，使模型更专注于分类主任务

### 3.4 最终损失函数

整体方法建立在 DER（Dark Experience Replay [3]）和 XDER [2] 的基线之上，最终损失为：

```
L_final(X_l, X_u, M) = (1 - 0.1*alpha) * L_ACE(X_l, M)
                      + alpha * L_SSL(X_u)
                      + gamma * L_LC(X_l)
                      + eta * L_der(M)
                      + beta * L_featureKD(X_u)
                      + delta * L_logitKD(X_u)
```

各损失项解释：

| 损失项 | 含义 | 参考文献 |
|--------|------|----------|
| L_ACE(X_l, M) | 异步交叉熵损失（Asymmetric Cross-Entropy），仅对批次中实际出现的类别计算损失 | [4] Caccia et al. |
| L_SSL(X_u) | 自监督损失，具体采用**旋转预测（Rotation Prediction）** | [10] Gidaris et al. |
| L_LC(X_l) | Logit 约束损失（Logit Constraint Loss），来自 XDER，帮助区分不同任务的类别 | [2] Boschini et al. |
| L_der(M) | Dark Experience Replay 损失，将样本的特征和 logit 存储在记忆缓冲区 M 中回放 | [3] Buzzega et al. |
| L_featureKD(X_u) | 特征级知识蒸馏损失（L2），权重 beta = 0.002t | 本文提出 |
| L_logitKD(X_u) | 基于 Gram 矩阵的 logit 知识蒸馏损失，权重 delta = 0.1 | 本文提出，借鉴 [13] |

### 3.5 推理策略

采用**模型集成（Model Ensemble）**：推理时将当前模型和一个先前模型的预测结果取平均。

---

## 四、实验设置

### 4.1 数据集与任务流

- **数据集**：ImageNet-1K [7] 子集，包含 130 个类别，主题为鸟类（birds）、海洋动物（sea animals）和昆虫（bugs）
- **标记数据流**：包含 100 个类别的数据
- **任务设置**：50 个 experiences（连续到达的数据流）
- **每 experience**：500 个标记样本 + 1000 个无标记样本
- **无标签数据来源**（视挑战赛场景设置而定）：
  1. 与同 experience 中标记数据相同类别的图像
  2. 来自标记数据流 100 类内部的图像
  3. 来自整个数据集的随机图像

### 4.2 模型与约束

- **模型架构**：ResNet-18 [11]（He et al., 2016）
- **GPU 显存限制**：8,000 MB
- **时间限制**：600 分钟（10 小时）
- **记忆缓冲区限制**：最多存储 200 个 exemplars，每个 exemplar 大小不超过 1,024 个浮点数（即 200 x 1024 floats）

### 4.3 实现细节

- **框架**：PyTorch [21] + Avalanche [19] 持续学习库
- **优化器**：Adam [15]，学习率 4e-4
- **批次大小**：标记数据 64，无标签数据 100
- **SSL 方法**：**旋转预测（Rotation Prediction）**[10]（Gidaris et al., 2018），预测图像的四个旋转角度之一作为自监督任务
- **超参数一览**：

| 参数 | 取值 | 含义 |
|------|------|------|
| c | 0.5 | SSL 动态权重参数 |
| omega | 0.95 | SSL 指数衰减因子（每任务衰减至 95%）|
| gamma | 0.1 | Logit 约束损失 L_LC 权重 |
| eta | 0.4 | DER 回放损失 L_der 权重 |
| beta | 0.002 * t（t 为任务编号）| Feature KD 损失权重（每任务增长 0.002）|
| delta | 0.1 | Logit KD 损失 L_logitKD 权重 |

- **硬件**：单张 RTX 2080Ti GPU

### 4.4 Baselines

- **Fine-tuning (FT)**：最简微调方法，不采取任何防遗忘措施
- **Baseline**：基于 DER/XDER 的基础方法（包含 L_ACE + L_LC + L_der，即异步交叉熵 + logit 约束 + Dark Experience Replay 回放）

---

## 五、核心结果与发现

### 5.1 主要实验结果（Table 1）

| 方法 | 最终准确率（Final Acc %） |
|------|--------------------------|
| Fine-tuning | 5.82 |
| Baseline | 19.54 |
| Baseline + Dynamic SSL | 23.82 |
| Baseline + MLKD | 39.82 |
| **Baseline + Dynamic SSL + MLKD (Ours)** | **42.00** |

- 朴素微调效果最差（5.82%），表明灾难性遗忘极为严重
- 基线方法提升至 19.54%
- 单独加入 Dynamic SSL 提升至 23.82%（相对 Baseline 提升 4.28 个百分点）
- 单独加入 MLKD 大幅提升至 39.82%（相对 Baseline 提升 20.28 个百分点）
- 两者结合达到最优 42.00%
- **MLKD 是性能提升的主要贡献者**，远大于 Dynamic SSL 的贡献

### 5.2 MLKD 组件消融实验（Table 2）

| 配置 | 最终准确率（%） |
|------|---------------|
| FKD only | 39.40 |
| FKD + EMA | 40.84 |
| FKD + EMA + CLKD | 41.08 |
| FKD + EMA + CLKD + MPM | 42.00 |

注：FKD=Feature KD, EMA=Exponential Moving Average, CLKD=Correlation-based Logit KD, MPM=Multiple Previous Models

- 从仅有 FKD（39.40%）开始，逐步加入 EMA（+1.44）、CLKD（+0.24）、MPM（+0.92），每个组件均带来正向收益
- 四个组件存在协同效应，全组合达到 42.00%
- EMA 的贡献（+1.44）大于 CLKD（+0.24），说明平滑更新旧模型比相关性蒸馏更为关键

### 5.3 先前模型数量的影响（Table 3）

| 先前模型数量 | 最终准确率（%） |
|------------|---------------|
| 1 个 | 40.34 |
| 2 个 | 41.56 |
| 3 个 | 42.00 |
| 4 个 | 41.84 |

- 更多模型通常带来更好性能，但需更多训练时间（更多前向传播和 EMA 更新）
- **3 个先前模型达到最优**（42.00%），4 个反而略微下降至 41.84%
- 存在边际递减效应，3 个模型在保留有用知识与避免冗余、保持合理训练时间之间达到最优平衡

### 5.4 动态加权的效果（Table 4）

| 方法 | 最终准确率（%） |
|------|---------------|
| 无动态加权（固定 SSL 权重） | 40.76 |
| **有动态加权（alpha 递减）** | **42.00** |

- 动态调整 SSL 影响带来 1.24% 的绝对提升
- 验证了衰减策略有助于模型在训练后期保持对主任务的聚焦（focus on the primary task），同时仍受益于 SSL 的泛化效果

### 5.5 SSL 方法选择的比较（Table 5）

| SSL 方法 | 最终准确率（%） |
|---------|---------------|
| **Rotation Prediction [10]** | **42.00** |
| SimCLR [5] | 39.10 |
| SimSiam [6] | 39.46 |
| VICReg [1] | 36.18 |
| VICReg-ctr [9] | 37.26 |

- 简单的**旋转预测**反而显著优于 SimCLR、SimSiam、VICReg 等最新对比学习方法
- 作者解释：对比学习类方法通常需要**大批次大小（large batch size）和长训练时间（long training time）**才能发挥效果，在挑战赛的资源约束下（单 GPU、600 分钟）无法达到最优性能，导致次优效果

### 5.6 关键结论归纳

1. **MLKD 是性能提升的决定性因素**：仅添加 MLKD 即可将准确率从 19.54% 提升至 39.82%，相对提升超过 100%。
2. **在资源受限设定下，简单 SSL 方法优于复杂对比学习方法**：旋转预测以 42.00% 胜出，而 SimCLR 仅为 39.10%。
3. **多先前模型的集成蒸馏优于单模型蒸馏**，但存在边际递减效应，3 个模型是最优选择。
4. **动态衰减的 SSL 权重策略是必要的**，固定权重会导致性能下降（40.76% vs 42.00%）。
5. EMA 更新旧模型（+1.44）比相关性 logit KD（+0.24）带来的增益更大，表明模型参数层面的知识保留比 logit 分布层面的保留在 CIR 中更为有效。

---

## 六、主要贡献与局限性

### 6.1 主要贡献

1. **提出了 MLKD（Multi-Level Knowledge Distillation）**：一种从多个先前模型的多个层面（特征层 L2 蒸馏 + logit 层基于 Gram 矩阵的相关性蒸馏）进行知识迁移的方法，有效缓解灾难性遗忘。特别地，基于 Gram 矩阵的相关性蒸馏巧妙地规避了无标签数据与训练数据之间的分布偏移问题。
2. **提出了动态自监督学习（Dynamic SSL）**：利用外部无标签数据加速新类别学习，并通过指数衰减权重 alpha = c * omega^t 平衡主任务与自监督任务之间的关系。
3. **无需存储原始标记数据（rehearsal-free 变体）**：方法利用可下载、使用后丢弃的外部无标签数据，规避了数据隐私风险和存储限制问题。
4. **实际竞赛验证**：在 CVPR 2024 第 5 届 CLVISION 挑战赛中取得第 2 名，证明了方法在严格资源约束下的实用有效性。

### 6.2 局限性

1. **计算开销增加**：维护多个先前模型、进行多次前向传播和 EMA 更新会显著增加训练时间，论文在 Table 3 分析中也明确提到 more training time due to more number of model forwarding and EMA updating。
2. **外部数据依赖性**：方法假设可以方便获取大量外部无标签数据，在数据获取受限或无相关外部数据的场景下适用性有限。
3. **简单 SSL 方法最优的结论有条件性**：旋转预测优于对比学习方法是在特定资源限制（单张 2080Ti、600 分钟、batch size 受限）下的结论，在资源充足的场景下对比学习方法可能更具优势。
4. **对比基线有限**：只与微调和自身基线进行了比较，缺乏与其他代表性 rehearsal-free CIL/CIR 方法（如 LwF、EWC、PASS、FeTrIL 等）的直接对比。
5. **单数据集验证**：仅在 ImageNet-1K 的一个子集（130 类，限定领域：鸟类/海洋动物/昆虫）上进行了验证，在 CIFAR-100、Tiny-ImageNet 等常用 CIL 基准上的泛化性未知。
6. **超参数较多**：引入了 c、omega、gamma、eta、beta、delta 共 6 个超参数，调参工作量大，且未见对这些超参数敏感性的系统性分析。

---

## 七、对后续研究的启发与潜在改进方向

### 7.1 方法层面的启发

1. **多层级/多模型蒸馏范式**：MLKD 同时蒸馏 feature 和 logit，并用多个旧模型集成和 EMA 平滑更新，该思路可推广到其他持续学习设定（task-incremental、domain-incremental、online CL），也可与 Transformer/ViT 架构结合。
2. **Gram 矩阵相关性蒸馏**：在无标签数据上通过批次内实例相关性和类别间相关性进行蒸馏，巧妙绕过了实例级分布偏移问题。该技术可独立用于其他需要跨分布进行知识迁移的场景（如无源域适应 source-free domain adaptation、联邦学习中的知识聚合）。
3. **动态权重衰减的简单范式**：alpha = c * omega^t 这种指数衰减形式简单有效，可推广到任何需要平衡主辅任务的多任务/辅助任务学习场景。

### 7.2 潜在改进方向

1. **降低多模型蒸馏的计算开销**：研究在不维护多个完整旧模型的前提下实现类似多模型蒸馏效果的方法，例如：
   - 模型参数融合（如模型汤 model soup、参数平均）
   - 蒸馏路径共享（多个旧模型共享特征提取器，仅分离分类头）
   - 知识缓存技术（如将旧模型的知识预先计算并存储，避免重复前向传播）
2. **自适应先前模型选择**：当前固定使用 3 个旧模型，可研究基于任务相似度或模型间差异性的自适应选择策略，例如根据验证集表现动态增减蒸馏的旧模型数量。
3. **资源高效的小批次对比学习**：探索使 SimCLR、VICReg 等对比学习方法在资源受限条件下也能有效工作的小批次训练技术（如动量编码器 memory bank、更高效的负样本利用策略）。
4. **多数据集多指标评估**：在更多数据集（CIFAR-100、Tiny-ImageNet、CoRe50）和更多指标（前向迁移 Forward Transfer、后向迁移 Backward Transfer、平均增量准确率 Average Incremental Accuracy）上验证方法的泛化性。
5. **与 rehearsal 方法的混合策略**：尽管方法定位于 rehearsal-free 变体，但与少量 exemplar 回放结合（混合 rehearsal + 无标签数据蒸馏）可能进一步提升性能。
6. **无标签数据质量与来源敏感度分析**：系统研究不同质量、不同域偏移（domain shift）程度的无标签数据对方法效果的影响，以及如何筛选/加权高质量无标签数据以最大化蒸馏和 SSL 效果。
7. **超参数自动搜索/自适应调整**：针对 6 个超参数的调优问题，可引入超参数自动搜索策略，或设计随任务进展自适应调整的机制，减少人工调参负担。

---

*总结完成于 2026-05-18*
