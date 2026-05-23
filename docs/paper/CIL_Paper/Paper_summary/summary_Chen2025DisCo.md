# Make Domain Shift a Catastrophic Forgetting Alleviator in Class-Incremental Learning -- 论文详细总结

---

## 一、基本信息

- **标题**：Make Domain Shift a Catastrophic Forgetting Alleviator in Class-Incremental Learning
- **作者**：Wei Chen, Yi Zhou（东南大学计算机科学与工程学院，新一代人工智能技术与交叉应用教育部重点实验室）
- **发表年份/出处**：2025 年（arXiv 预印本）
- **代码**：https://github.com/PixelChen24/DisCo

---

## 二、研究动机与问题定义

### 核心问题
类增量学习（Class-Incremental Learning, CIL）的核心挑战是灾难性遗忘（catastrophic forgetting）：模型在连续学习新类时，会严重丧失对旧类的分类能力。

### 出发点与反直觉发现
传统的 CIL 研究大多假设各任务的数据分布相同（即各任务域不变），忽视了域偏移（domain shift）的影响。已有将 CIL 与域增量学习（DIL）结合的工作（如 Kundu et al. 2020; Simon et al. 2022; Xie et al. 2022），但并未真正聚焦于域偏移本身对 CIL 的影响。

本文提出并回答了一个关键问题：**域偏移是否真的损害 CIL 方法？**

答案出乎意料——**将域偏移引入 CIL 会显著降低遗忘率**。作者通过系统的实证研究揭示了这一反直觉现象，并以此为出发点设计了方法 DisCo。

---

## 三、实证研究（Empirical Study）

### 3.1 实验场景设计

作者设计了两种对比场景：

- **CIL（标准类增量学习）**：各任务共享相同的数据分布，无域变化。
- **CILD（带域偏移的类增量学习）**：基于 CIL，每个任务引入不同的域变化，但标签空间与 CIL 一致。

两种 CILD 构建方式：
1. **合成方式**：使用预训练风格迁移模型 AvatarNet（Sheng et al. 2018）将 CIFAR-100 原始图像迁移到多个域，生成 DomainCIFAR-100。
2. **分割方式**：利用 DomainNet（Peng et al. 2019）本身具有多域特性的数据集，按域分割形成任务序列。

### 3.2 关键观察

在 CIFAR-100 和 DomainNet 上对 6 种代表性 CIL 方法（iCaRL、BiC、MEMO、LwF、DER、L2P，覆盖 rehearsal-based、regularization-based、architecture-based、prompt-based 四类）进行测试。

**核心发现（Table 1）**：在 CILD 场景下，大多数 CIL 方法的遗忘率（FM）显著低于 CIL 场景。代表性数据（CIFAR-100 上）：

| 方法 | CIL FM | CILD FM | 绝对降幅 |
|------|--------|---------|----------|
| iCaRL | 59.42 | 25.98 | -33.44 |
| BiC | 42.80 | 16.04 | -26.76 |
| MEMO | 16.91 | 7.56 | -9.35 |
| DER | 40.26 | 0.50 | -39.76 |

DomainNet 上同样大幅下降（如 iCaRL FM 从 41.27 降至 22.88; BiC FM 从 36.20 降至 13.12; DER FM 从 34.21 降至 9.56）。

**例外**：基于 prompt 的方法 L2P 在 CILD 下遗忘反而加剧（CIFAR-100: FM 从 4.42 升至 13.46, +9.04; DomainNet: FM 从 3.81 升至 5.04, +1.23），原因在于 L2P 的主干网络冻结，仅微调 prompt key-value 对，参数可扩展性不足，面对分布外样本容易过拟合而遗忘旧知识。

**t-SNE 可视化（Figure 3）**：CIL 场景下不同任务的特征高度重叠；CILD 场景下不同任务的特征形成清晰的独立簇，表明域偏移有助于保持任务特征的区分性，减少任务间干扰。

此外，CILD 的 AA（Average Accuracy）普遍低于 CIL（因为合成图像/异域图像对第2个及以后的任务更难分类，导致初始准确率较低），但如果仅关注第一个任务在整个增量过程中的性能，CILD 下模型遭受的遗忘明显更少（详见附录 B.3）。

### 3.3 定量分析：PIV 与 PFTS

为量化域偏移对参数更新的影响，作者提出了两个指标：

- **PIV（Parameter Interference Value）**：衡量不同任务之间参数更新的干扰程度。值越低表示任务间干扰越小。
- **PFTS（Parameter Forward Transfer Score）**：衡量模型在序贯任务中相对学习到的知识量。值越高表示模型学到了更多可区分特征。

数学定义见附录 A.2。

**关键结果（Table 1）**：
- CILD 下的 PIV 普遍低于 CIL（例如 iCaRL CIFAR-100: PIV 73.50 降至 56.00; BiC CIFAR-100: PIV 69.00 降至 53.00），表明域偏移降低了任务间的参数干扰。
- CILD 下的 PFTS 普遍高于 CIL（例如 MEMO CIFAR-100: PFTS 68.24 升至 95.30; BiC CIFAR-100: PFTS 56.21 升至 77.25），表明域偏移有助于模型有序地学习可区分表征。
- L2P 的 PIV 在两个场景均为 100%（因为可训练参数仅限 prompt），同时 CILD 下 PFTS 显著升高（CIFAR-100: 30.16 升至 59.75），说明共享 prompt 学习能力虽强但干扰严重，导致 CILD 表现更差。
- DER 在两种场景下 PIV 均较低（均为 1.00），因为该方法为每个任务引入新的网络层。

---

## 四、方法/框架：DisCo

### 4.1 设计动机

直接将域偏移引入 CIL 方法存在根本困难：推理时模型无法确定该对输入施加何种域偏移（需要 task ID），这与 CIL 的设定冲突。因此 DisCo 的思路是：**在特征空间而非输入空间中模拟域偏移的有利效果**——促使不同任务的特征分布相互分离，从而减少干扰。

DisCo 是一个**即插即用（plug-and-play）**的方法，可无缝集成到已有的基于 rehearsal 的 CIL 方法中。

### 4.2 原型池（Prototype Pool）构建

DisCo 维护一个轻量级原型池，为每个已学任务存储一个代表性原型向量。支持两种原型类型：

**（1）图像特征原型（DisCo-I）**：
- 将当前 batch 的图片经过特征提取器 F_theta^t 和投影器 psi^t，得到降维特征 tilde(x_j) in R^{N x d}
- Batch 级原型：p_i = (1/N) * sum_{j=1}^{N} tilde(x_j)

**（2）文本原型（DisCo-T）**：
- 收集当前 batch 所有类别名称，构造提示词："a photo of {class 0} or {class 1} or ... or {class n}"
- 将其输入 CLIP（Radford et al. 2021）文本编码器提取嵌入向量

**动量累积（Momentum Accumulation）**：
为让 batch 级原型 p_i 逼近整个任务的原型 P_t，使用动量更新：

    p_i = ((i-1)/i) * p_{i-1} + (1/i) * p_i

任务 t 的所有 N_t 个 mini-batch 处理完后，p_{N_t} 作为任务 t 的原型 P_t 存入原型池。

### 4.3 任务级和类级别正则化（Task-level & Class-level Regularization）

**任务级对比损失 L_tcon**：
- 目的：使当前任务的特征远离之前所有任务的原型，确保不同任务的特征空间存在明确边界。
- 将 batch 中每个样本 x_j 作为锚点（anchor），将本 batch 原型 p_i 作为唯一正样本（positive），将之前所有任务的原型 P_k（k < t）作为负样本（negative）。
- 每 batch 构造 N x (t-1) 个三元组：

    L_tcon = (1/(N*(t-1))) * sum_{j=1}^{N} sum_{k=1}^{t-1} Triplet(x_j, p_i, P_k)

    Triplet(a, p, n) = log(1 + exp(1 - S(a,p) + S(a,n)))

其中 S(x, y) 为余弦相似度函数（cosine similarity）。

**类级别对比损失 L_ccon**：
- 目的：在同一任务内学习有区分度的类别特征。
- 对每个样本 x_j，随机选择同标签样本作为正样本 x_p，不同标签样本作为负样本 x_n：

    L_ccon = (1/N) * sum_{j=1}^{N} Triplet(x_j, x_p, x_n)

### 4.4 跨任务对比蒸馏（Cross-task Contrastive Distillation, CCD）

- 目的：显式保护已学旧知识，使当前学生模型与旧教师模型的特征表示对齐。
- 任务 t 训练时，复制任务 t-1 训练好的模型作为教师模型 f_theta^{t-1}（冻结），当前模型作为学生模型 f_theta^t。
- 对于 rehearsal 样本集 R = {(x_j, y_j) | y_j not in C_t}（即旧类样本）中的每个样本：
  - 用教师模型提取特征 tilde(x_j)
  - 用学生模型提取特征 tilde(x_j)'
- 对比蒸馏损失：

    L_ccd = sum_{(x_j,y_j) in R} sum_{k in R, y_k != y_j} Triplet(tilde(x_j)', tilde(x_j), tilde(x_k)')

该损失强制学生模型：同一旧类样本的特征向教师模型的对应特征靠拢，同时远离其他不同旧类样本的特征——即同时实现"同类别特征拉近（pull together）"和"不同类别特征推远（push away）"。

### 4.5 总损失与推广

**总损失**：

    L = L_baseline + lambda_tcon * L_tcon + lambda_ccon * L_ccon + lambda_ccd * L_ccd

超参数默认设置：lambda_tcon = 0.5，lambda_ccon = 0.5，lambda_ccd = 1。其他可能取值的评估见附录 C.5。

**向非 rehearsal 方法的推广**：
- **基于正则化的方法（如 LwF）**：可直接集成，丢弃 CCD 模块（仅有轻微性能下降）。消融实验中对 LwF 的 CCD 标记为 "-"。
- **基于 prompt 的方法（如 L2P）**：对选中的 prompt keys 施加类似式(3)的对比正则化（代替对图像特征的正则化），详见附录 C.3。

---

## 五、实验设置

### 5.1 数据集与任务划分
- **CIFAR-100**（Krizhevsky & Hinton 2009）
- **Fashion-MNIST**
- **Tiny-ImageNet**

具体任务划分策略见附录 C.1。消融实验中使用了 B{X}-{Y} 格式，如 B50-5 表示第 0 个任务包含 50 个类，剩余 50 个类均分到 5 个增量任务中。

### 5.2 基线方法
- iCaRL（Rebuff et al. 2016）—— rehearsal-based
- BiC（Wu et al. 2019）—— rehearsal-based
- LwF（Li & Hoiem 2017）—— regularization-based
- DER（Buzzega et al. 2020）—— architecture-based
- L2P（Wang et al. 2022c）—— prompt-based
- Co2L（Cha et al. 2021）—— 基于对比学习的 rehearsal 方法，作为额外对比

### 5.3 评估指标
- **AA（Average Accuracy）**：所有已见任务的平均分类准确率，越高越好。
- **FM（Forgetting Measure）**：旧任务性能下降的度量，越低越好。
- 实证部分还使用了 **PIV** 和 **PFTS**（专门设计的参数层面指标）。
- 消融部分使用了 **IA（Initial Accuracy）**：每个新任务的初始分类准确率。

两个核心指标的详细数学定义见附录 A.1。

### 5.4 实现细节
- 框架：基于 PyTorch 和 LAMDA-PILOT（Sun et al. 2023）
- 所有任务训练 100 个 epoch，在第 60 和 80 个 epoch 衰减学习率
- **ResNet 主干**：weight decay = 5e-4，lr = 0.1，milestones 时乘以 0.1
- **ViT 主干**：weight decay = 2e-4，lr = 1e-3，milestones 时乘以 0.1
- 每组实验平均运行 3 次

---

## 六、核心结果与发现

### 6.1 主要实验结果（Table 2）

**CIFAR-100 上各方法集成 DisCo 后的完整性能数据**：

| 方法 | AA | FM |
|------|-----|-----|
| iCaRL（基线） | 64.24 | 51.34 |
| iCaRL + DisCo-I | 70.11 | 33.96 |
| iCaRL + DisCo-T | 63.35 | 35.26 |
| BiC（基线） | 67.04 | 46.51 |
| BiC + DisCo-I | 69.89 | 28.54 |
| BiC + DisCo-T | 67.68 | 23.08 |
| Co2L（独立对比方法） | 71.25 | 32.17 |
| LwF（基线） | 51.30 | 55.98 |
| LwF + DisCo-I | 56.42 | 36.52 |
| LwF + DisCo-T | 56.87 | 33.89 |
| DER（基线） | 63.91 | 40.18 |
| DER + DisCo-I | 64.87 | 36.41 |
| DER + DisCo-T | 69.51 | 33.74 |
| L2P（基线） | 82.65 | 7.62 |
| L2P + DisCo-I | 82.78 | 7.98 |
| L2P + DisCo-T | 83.12 | 6.80 |

**论文正文强调的典型提升（最佳变体）**：
- iCaRL + DisCo-I 在 CIFAR-100 上：AA +5.87（百分点），FM -17.38（百分点）
- iCaRL + DisCo-T 在 Tiny-ImageNet 上：AA +3.31（8.57 升至 11.88），FM -10.94（81.40 降至 70.46）
- LwF + DisCo-T 在 CIFAR-100 上：FM -22.09（55.98 降至 33.89）
- LwF + DisCo-T 在 Tiny-ImageNet 上：FM -8.35（85.50 降至 77.15）

**Tiny-ImageNet 完整数据总结**：
iCaRL: AA 8.57, FM 81.40; BiC: AA 8.42, FM 78.42; LwF: AA 5.06, FM 85.50; DER: AA 11.58, FM 77.99; L2P: AA 29.54, FM 44.30。集成 DisCo 后各方法遗忘率总体降低，但 L2P+DisCo-I 的 AA 反而从 29.54 降至 28.43。

**Fashion-MNIST 数据总结**：
iCaRL+DisCo-I: AA 69.41 升至 72.57, FM 47.44 降至 34.45; DER+DisCo-I: AA 68.21 升至 72.41, FM 37.85 降至 29.86。多数方法集成 DisCo 后 FM 降低。

**关键趋势**：
1. DisCo 几乎在所有基线与数据集组合上降低了 FM，提高了 AA。
2. 对 rehearsal-based 方法（iCaRL、BiC）的提升最为显著。
3. 对 L2P：DisCo-T 优于 DisCo-I。文本原型提供高层语义信息，更契合 prompt 机制。
4. DisCo 取得与 Co2L 相当或更优的性能（Co2L CIFAR-100: AA=71.25, FM=32.17）。
5. AA 提升幅度通常小于 FM 下降幅度——任务级正则化在分离任务特征分布的同时增加新类学习难度，降低新任务初始准确率（IA），消融实验证实了这一点。

### 6.2 消融实验（Table 3）

在 CIFAR-100 上对 DisCo 各组件进行消融：

以 **iCaRL + DisCo-I** 为例（完整 vs 各消融变体）：
- 完整 DisCo：AA=70.11, FM=33.96, IA=80.16
- 去掉 Ccon（类级别正则化）：AA=67.21, FM=32.15, IA=75.22（AA 和 IA 下降）
- 去掉 Tcon（任务级正则化）：AA=66.00, FM=47.41, IA=82.54（FM 从 33.96 恶化至 47.41，但 IA 反而更高）
- 去掉 CCD（跨任务对比蒸馏）：AA=69.35, FM=35.47, IA=80.34（FM 略增，AA 略降）

BiC + DisCo-I 各消融变体趋势一致：
- 去掉 Ccon：AA=63.32, FM=28.16, IA=72.52
- 去掉 Tcon：AA=65.15, FM=44.25, IA=74.67
- 去掉 CCD：AA=67.33, FM=26.68, IA=76.10

**综合结论**：**Tcon（任务级正则化）对降低遗忘贡献最大；Ccon（类级别正则化）对提升新类学习贡献最大；CCD（跨任务对比蒸馏）提供额外的旧知识保护**。三者协同效果最优。对于非 rehearsal-based 方法（LwF, L2P），CCD 不适用（标记为 "-"）。

### 6.3 任务长度消融（Figure 5）

在不同增量策略下（B50-5、B50-10、B40-10 等，B{X}-{Y} 表示初始任务含 X 个类，剩余类均分到 Y 个增量任务），遗忘趋势随任务长度增加而趋于缓和——每个任务包含的类越多，DisCo 表现越好。B50-5（初始50类，剩余50类分5个增量任务）的遗忘最小，因为更多样本能产生更准确的原型辅助后续任务的特征分布引导。

### 6.4 补充发现
- t-SNE 可视化（附录 C.4）显示 DisCo 能有效分离不同任务的特征分布
- 更细粒度数据集上的额外结果见附录 C.4

---

## 七、主要贡献与局限性

### 主要贡献
1. **发现反直觉现象**：首次发现将域偏移引入 CIL 可显著降低灾难性遗忘，并通过对 6 种方法（跨 4 个类别）和 2 个数据集的系统实证验证了该现象的普适性。
2. **定量分析工具**：设计 PIV（参数干扰值）和 PFTS（参数前向迁移分数）两个量化指标，揭示了域偏移降低遗忘的内在机制：减少任务间的参数更新干扰，促进有序的知识学习。
3. **简单有效的方法**：提出即插即用的 DisCo，仅通过对比损失在特征空间模拟域偏移的分离效应，无需修改基线方法结构，即可无缝集成并普遍提升性能。
4. **广泛的实验验证**：在 3 个数据集（CIFAR-100, Fashion-MNIST, Tiny-ImageNet）、5 种基线方法（跨 4 个方法类别）、两种主干网络（ResNet/ViT）上验证了 DisCo 的有效性和泛化性。

### 局限性
1. **AA 提升有限**：虽然遗忘率大幅下降，但平均准确率的提升相对有限。任务级正则化在分离任务特征分布的同时增加了新类学习难度，导致新任务初始准确率（IA）降低。
2. **对 rehearsal 机制的依赖**：DisCo 的 CCD 模块依赖 memory buffer 中的旧类样本；对于无 buffer 的方法（LwF, L2P），CCD 无法使用，性能增益受限。
3. **原型质量依赖样本量**：DisCo 的性能随每任务样本数量增加而提升（Figure 5），小样本增量场景下原型可能不够准确。
4. **超参数敏感性**：论文默认使用 lambda_tcon=0.5, lambda_ccon=0.5, lambda_ccd=1，跨数据集/场景可能需要额外调参。
5. **对 prompt-based 方法的不稳定性**：L2P+DisCo-I 在 CIFAR-100 上 FM 反升至 7.98（基线 7.62），Tiny-ImageNet 上 AA 降至 28.43（基线 29.54）。DisCo-T 对 L2P 才稳定正向。
6. **缺乏大规模数据集验证**：仅在中小规模数据集上实验，未在 ImageNet-1K 等大规模数据集上验证。

---

## 八、对后续研究的启发与潜在改进方向

1. **域偏移作为正则化工具的新视角**：将域偏移从"问题"重新定位为"工具"，开辟了利用数据增强/风格迁移主动构造任务间分布差异来对抗遗忘的可能性。核心挑战是不引入 task ID 泄露的前提下自动选择合适的域偏移策略。

2. **特征空间分离的理论分析**：论文提供了经验证据（t-SNE + PIV/PFTS），但缺乏严格理论解释。可从信息瓶颈理论或神经正切核（NTK）角度提供理论支撑。

3. **无 task ID 的域选择策略**：可研究基于无监督域检测/自适应的方法，在推理时自动推断输入所属的"隐式域"，进而选择合适的特征变换或原型引导。

4. **原型质量的提升**：
   - 可学习原型（learnable prototypes）：让原型通过梯度优化
   - 不确定性建模（如高斯原型）：更好地应对样本稀疏或噪声
   - 更强大的多模态基础模型：CLIP 以外的替代方案

5. **推广至更多 CIL 范式**：
   - 在线 CIL（online CIL）
   - 少样本 CIL（few-shot CIL）
   - 长尾 CIL（long-tailed CIL）
   - Blurry CIL（任务边界模糊的 CIL）

6. **与 prompt-based 方法的深度融合**：L2P+DisCo-T 优于 L2P+DisCo-I，说明文本原型在 prompt 机制中有独特价值。可探索将原型直接编码为 prompt 的一部分，或使用交叉注意力机制动态整合。

7. **动态超参数调度**：任务级正则化强度 lambda_tcon 可随时间动态调整——初期强约束建立分离边界，后期放松利于新类学习，从而实现更好的稳定性-可塑性动态平衡。

8. **计算开销优化**：随着任务数增加，L_tcon 中的负样本数量（旧任务原型数）和三元组数量（N x (t-1)）线性增长。可研究原型的层次化聚类或稀疏采样策略降低复杂度。

9. **跨架构泛化验证**：探索 DisCo 在 ConvNeXt、Swin Transformer、Mamba 等新兴架构上的表现，验证方法的架构无关性。

---

*本总结基于论文 "Make Domain Shift a Catastrophic Forgetting Alleviator in Class-Incremental Learning"（Chen & Zhou, 2025）全文整理。所有数据和数字均直接来源于论文正文及表格，已逐项核查。*
