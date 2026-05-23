# CLASS INCREMENTAL LEARNING VIA LIKELIHOOD RATIO BASED TASK PREDICTION (TPL) —— 详细中文总结

---

## 一、基本信息

- **标题**: Class Incremental Learning via Likelihood Ratio Based Task Prediction
- **作者**: Haowei Lin (Peking University), Yijia Shao (Stanford University), Weinan Qian (Peking University), Ningxin Pan (Peking University), Yiduo Guo (Peking University), Bing Liu (University of Illinois at Chicago)
- **发表年份/会议**: 2024年，ICLR 2024（International Conference on Learning Representations）
- **代码仓库**: https://github.com/linhaowei1/TPL

---

## 二、研究动机与问题定义

### 2.1 类增量学习（CIL）的核心挑战

类增量学习（Class Incremental Learning, CIL）要求模型顺序学习一系列任务，每个任务包含一组不相交的类别。CIL 面临两大核心难题：

1. **灾难性遗忘（Catastrophic Forgetting, CF）**：学习新任务时，模型对旧任务性能急剧下降。
2. **任务间类别分离（Inter-task Class Separation, ICS）**：在无法访问旧任务全部训练数据的情况下，难以建立新旧任务类别之间的决策边界。

CIL 区别于任务增量学习（Task Incremental Learning, TIL）的关键在于：**测试时不提供任务标识符（task-id）**。因此 CIL 模型必须自行推断每个测试样本属于哪个任务，再进行任务内分类。

### 2.2 已有 TIL+OOD 方法的局限性

Kim et al. (2022b) 提出了将 CIL 分类概率分解为两项乘积的理论框架：

P(y_j^(t) | x) = P(y_j^(t) | x, t) * P(t | x)

其中第一项是**任务内预测概率（Within-task Prediction, WP）**，第二项是**任务标识预测概率（Task-id Prediction, TP）**。WP 使用任务专属模型通常可以做到高精度，因此 TP 成为关键挑战。

Kim et al. (2022b) 进一步理论证明了 TP 与各任务模型的 OOD 检测能力相关——OOD 检测器既能做分布内（IND）分类，又能做分布外（OOD）检测，后者恰好可用于判断样本是否属于该任务。这一范式被称为 **TIL+OOD**，代表性工作包括 HAT+CSI、MORE、ROW 等。

**本文的核心论点**：传统 OOD 检测器并非为 CIL 设计，它们仅利用 IND 分布 P_t（某任务的数据分布），而对 OOD 分布 P_{t^c}（其他所有任务的数据分布）只能假设为均匀分布或使用代理分布。然而 CIL 场景中存在可利用的额外信息——**保存在 memory buffer 中的旧任务 replay 数据**可以用于估计 P_{t^c}，从而设计出更优、更有原则的任务 ID 预测方法。

### 2.3 理论差距的具体说明

在传统 OOD 检测中，OOD 的 universal set 包含世界上所有可能的类别（除了 IND 类），其规模极大（如果不是无限的话），且没有任何 OOD 数据可用于估计其分布。但在 CIL 中则不同——universal set U_{CIL} = {1, 2, ..., T} 包含所有已学习的任务，且每个任务都有 replay 数据保存，因此可以估计其补集分布 P_{t^c}（t^c = U_{CIL} - {t} 的等权重混合分布）。这为使用完整似然比（而非仅 IND 密度）进行任务 ID 预测提供了可行性。

论文通过一个具体反例说明（Section 4.1）：设 P_{t^c} = N(0, 0.01), P_t = N(0, 1)。虽然 p_t(0) > p_t(1)，但似然比 p_t(0)/p_{t^c}(0) < p_t(1)/p_{t^c}(1) = 0.1 * e^{49.5}。仅凭 IND 密度会做出错误选择，而基于似然比能做出正确判断。Appendix H 在真实 CIL 场景（C10-5T）中可视化了此类失败案例。

---

## 三、方法/框架（重点）

### 3.1 总体框架概述

TPL 系统由两部分组成：
1. **训练阶段**：基于 HAT 掩码机制训练每个任务的模型，采用特殊的多类别训练策略（引入"O"类）。
2. **推理阶段**：基于似然比（Likelihood Ratio）的 task-id 预测方法，包含特征空间的 LR 分数与 logit 空间分数的融合。

框架使用 HAT（Hard Attention to the Task, Serra et al., 2018）消除灾难性遗忘。HAT 是一种参数隔离方法，为每个任务学习一组注意力掩码——在反向传播时阻断对旧任务重要神经元的梯度更新；前向传播时不设阻断，从而允许跨任务参数共享。这种机制可以完全消除 CF。论文也指出 TPL 可以替换 HAT 为 SupSup、PackNet、PNN、Ternary Masks 等其他 TIL 方法（Appendix G.3）。

### 3.2 训练策略

每个任务的模型由**特征提取器** h(x; phi^(t))（通过 HAT 部分共享）和**任务专属分类器** f(z; theta^(t)) 组成。

训练第 t 个任务时，模型接收当前任务训练数据 D^(t) 和 replay buffer 中之前所有任务保存的数据 Buf_{<t}。关键创新在于：**每个任务模型不仅在其任务类别上做分类，而是额外引入一个"O"（Others）类**。具体而言，replay buffer 中来自旧任务的数据全部被标记为"O"类（赋予统一标签），当前任务的数据保留其原始类别标签。损失函数为：

L(theta^(t), phi^(t)) = E_{(x,y) ~ D^(t) U Buf_{<t}} [L_CE(f(h(x; phi^(t)); theta^(t)), y)] + L_HAT

其中 L_HAT 是 HAT 的稀疏正则化项，用于鼓励掩码稀疏性和参数共享（详见 Appendix G）。HAT 的正则化定义为：

L_reg = (sum_l sum_i a_{i,l}^(t) * (1 - a_{i,l}^{(<t)})) / (sum_l sum_i (1 - a_{i,l}^{(<t)}))
L_HAT = mu * L_reg

其中 mu = 0.75 是平衡超参数，温度退火参数 s = 400 控制 sigmoid 逼近 0-1 门的程度。

这一训练策略的动机是：让每个任务模型不仅学会识别自己的类别，还能区分"自己的数据"和"其他任务的数据"，从而为推理阶段基于似然比的 task-id 预测奠定基础。**"O"类在推理阶段不使用**——推理时仅用任务的原始类别 Y^(t) 做任务内分类（WP）。

### 3.3 推理阶段：Task-ID 预测的核心——似然比方法

#### 3.3.1 理论分析（Section 4.1）

将 task-id 预测形式化为二值假设检验问题：

H_0: x ~ P_t   v.s.   H_1: x ~ P_{t^c}

其中 P_t 为任务 t 的数据分布，P_{t^c} 为 t 的补集（所有已学习任务中除 t 外的任务的等权重混合分布）。

**定理 4.1**（基于 Neyman-Pearson 引理，Neyman & Pearson, 1933）：拒绝域定义为 R := {x: p_t(x) / p_{t^c}(x) < lambda_0} 的检验是唯一的**一致最优检验（Uniformly Most Powerful, UMP）**，其中 lambda_0 为满足指定显著性水平的阈值。

**定理 4.2**：该 UMP 检验最大化对 P_t 和 P_{t^c} 进行二分类的 AUC（受试者工作特征曲线下面积）。

这两个定理确立了似然比在 task-id 预测中的原则性最优地位。证明过程详见 Appendix E。

#### 3.3.2 特征空间似然比分数 S_LR^(t)(x)（Section 4.2.1）

为避免在高维原始图像空间直接估计分布，TPL 在低维特征空间中进行估计。利用能量模型（Energy-Based Model, EBM）参数化：

p_t(x) ∝ exp{E_t(x)},   p_{t^c}(x) ∝ exp{E_{t^c}(x)}

似然比对数为：

S_LR^(t)(x) = log(p_t(x) / p_{t^c}(x)) = E_t(x) - E_{t^c}(x) + log(Z_2 / Z_1)

省略常数项后：S_LR^(t)(x) := E_t(x) - E_{t^c}(x)

**关键设计选择与理由**：

- **E_t(x) 使用 Mahalanobis Distance (MD) 分数**：
  S_MD^(t)(x) = 1 / min_{c in Y^(t)} (h(x) - mu_c)^T Sigma^{-1} (h(x) - mu_c)
  即特征向量到各类别中心的最小马氏距离的倒数。选择 MD 的原因：其在测试时不需要访问任务训练数据，仅需在训练后保存类别中心 mu_c 和协方差矩阵 Sigma 即可。

- **E_{t^c}(x) 使用 KNN 距离的负值**：
  -E_{t^c}(x) = d_KNN(x, Buf_{t^c})
  即样本 x 的 L2 归一化特征到 replay buffer 中其他任务数据特征的第 k 近邻距离。选择 KNN 原因：非参数估计器在有限 replay 数据下性能优异（Yang et al., 2022），仅依赖 replay 数据即可估计。

最终 LR 分数（公式 7）：

S_LR^(t)(x) := alpha * S_MD^(t)(x) + d_KNN(x, Buf_{t^c})

附录 E.4 从理论上证明了 MD 分数和 KNN 分数本质上都是 IND 密度估计器——MD 在高斯假设下，S_MD 与最大对数似然成反比；KNN 的非参数密度估计 p(x) = k * Gamma(...) / (pi^(...) * n * r^(m-1))，因此 S_KNN ∝ p(x)^(-1/(m-1))。这一分析也揭示了 MORE 和 ROW 使用 MD 分数做 task-id 预测时实际仅估计了 IND 密度 p_t(x)，而忽略了补集密度 p_{t^c}(x)，因此理论上存在不足。

#### 3.3.3 融合 Logit 空间分数（Section 4.2.2）

为进一步提升性能，将特征空间的 S_LR 分数与 logit 空间的 OOD 分数融合。利用 EBM 的组合性质（OR 门，Du et al., 2020）：

E_composition(x) = log(exp{alpha_1 * S_logit^(t)(x)} + exp{alpha_2 * S_LR^(t)(x)})

选择 **MLS（Maximum Logit Score）**作为 logit 空间分数：

S_MLS^(t)(x) = max_{j=1}^{|Y_t|} f(h(x; phi^(t)); theta^(t))_j

即分类器输出的最大未归一化 logit 值。

**最终 TPL 分数（公式 9）**：

S_TPL^(t)(x) = log(exp{beta_1 * S_MLS^(t)(x)} + exp{beta_2 * S_MD^(t)(x) + d_KNN(x, Buf_{t^c})})

其中 beta_1 和 beta_2 是缩放参数，取值为各自分数在训练数据上经验均值的倒数（公式 10），使不同量级的分数可比：

1/beta_1 = (1/|D^(t)|) * sum_{x in D^(t)} S_MLS^(t)(x)
1/beta_2 = (1/|D^(t)|) * sum_{x in D^(t)} S_MD^(t)(x)

关于 EBM 使用的说明：EBM 以其灵活性著称但通常面临不可解性（intractability）问题。TPL 利用 EBM 的灵活性来推导原则性的 task-id 预测分数（遵循定理 4.1 和公式 8），同时通过在实践中使用 OOD 检测分数（MD、KNN、MLS）作为近似来保持可解性。这使得 TPL 在理论和实践上都站得住脚。

#### 3.3.4 分数转概率（Section 4.3）

通过带温度参数的 softmax 将各任务的未归一化分数转化为归一化 task-id 预测概率（公式 11）：

P(t | x) = softmax([S_TPL^(1)(x), S_TPL^(2)(x), ..., S_TPL^(T)(x)] / gamma)_t

选择低温 gamma = 0.05 以鼓励自信（低熵）的任务 ID 预测分布。

#### 3.3.5 最终 CIL 分类

结合 WP 概率和 TP 概率得到最终 CIL 概率（公式 3）：

P(y_j^(t) | x) = [softmax(f(h(x; phi^(t)); theta^(t)))]_j * P(t | x)

注意：WP 概率的 softmax 仅在任务 t 的原始类别 Y^(t) 上计算（不包括训练时的"O"类输出维度）。最终预测结果为概率最大的类别。不同任务的计算可以并行执行。

### 3.4 输出校准（Output Calibration, Appendix B）

由于各任务模型是分开训练的，不同任务模块的输出量级可能不一致。即使每个任务模型的 task-id 预测是完美的，如果跨任务输出的量级不同，系统仍会做出错误的最终预测。因此使用 replay buffer 数据学习每个任务的一组校准参数 {(sigma_1^(t), sigma_2^(t))}_(t=1)^T in R^{2T}，对最终预测进行线性缩放和平移：

P(y_j^(t) | x) = sigma_1^(t) * [softmax(f(h(x; phi^(t)); theta^(t)))]_j * S(x; t) + sigma_2^(t)

校准参数通过最小化 replay buffer 上的交叉熵损失优化得到（使用 Adam 优化器，batch size 64，100 epochs）。

### 3.5 关键超参数配置

- gamma = 0.05（温度参数，搜索范围 {0.01, 0.05, 0.10, 0.50, 1.0, 2.0, 5.0, 10.0}）
- k = 5（KNN 近邻数，搜索范围 {1, 2, 5, 10, 50, 100}）
- HAT: mu = 0.75, s = 400
- Adapter hidden dim: 64（CIFAR-10）/ 128（CIFAR-100, TinyImageNet）
- 训练 epochs: 20 (C10), 40 (C100), 15 (T-5T), 10 (T-10T)
- 优化器: SGD, momentum 0.9, batch size 64
- 学习率: 0.005 (C10-5T, T-5T, T-10T, C100-20T) / 0.001 (C100-10T)



---

## 四、实验设置

### 4.1 数据集与任务划分

| 数据集 | 任务划分 | 每任务类别数 | Replay Buffer 大小 |
|--------|---------|-------------|-------------------|
| CIFAR-10 | 5 任务 (C10-5T) | 2 | 200 |
| CIFAR-100 | 10 任务 (C100-10T) | 10 | 2000 |
| CIFAR-100 | 20 任务 (C100-20T) | 5 | 2000 |
| TinyImageNet | 5 任务 (T-5T) | 40 | 2000 |
| TinyImageNet | 10 任务 (T-10T) | 20 | 2000 |
| ImageNet-380 | 10 任务 | 38 | 7600 (20样本/类) |
ImageNet-380 为本文新构造：从 ImageNet-1k 中排除与 CIFAR/TinyImageNet 相似的 389 类后，随机选取 380 类（每类约 1,300 张图片）。所有实验采用随机类别顺序，生成 5 种不同顺序，报告均值加减标准差。

### 4.2 基线方法（17个）

**非 replay 方法（6个）**：OWM, ADAM, PASS, HAT_CIL, SLDA, L2P

**Replay 方法（11个）**：iCaRL, A-GEM, EEIL, GD, DER++, HAL, DER, FOSTER, BEEF, MORE, ROW

其中 MORE 和 ROW 是此前最强的 TIL+OOD 基线方法。L2P 和 SLDA 专为预训练骨干设计，无法适配无预训练设置。
### 4.3 骨干网络架构

**有预训练设置（主实验）**：
- Backbone: DeiT-S/16 (Touvron et al., 2021)，在 ImageNet 的 611 类上预训练（排除与 CIFAR/TinyImageNet 相似的 389 类，防止信息泄露）
- Transformer 参数固定（21.6M），仅 adapter 模块可训练
- Adapter (Houlsby et al., 2019) 插入每个 Transformer 层
- Adapter hidden dim: CIFAR-10 为 64，CIFAR-100 和 TinyImageNet 为 128
- 可训练部分：adapter modules + classifiers + layer norms，由 HAT 保护
- 结果记为 **TPL**

**全 ImageNet 预训练**（用于参考上限，Table 1 粉红行）：
- DeiT-S/16 使用完整 ImageNet 预训练（存在信息泄露），结果记为 **TPL_PFI**

**无预训练设置**：
- Backbone: ResNet-18 (He et al., 2016)
- 排除 L2P、SLDA、ADAM（专为预训练设计）
- MORE、ROW、TPL 通过将 HAT 应用于 ResNet-18 来适配
### 4.4 评估指标

- **Last ACC (A_last)**：学完最后一个任务后，在所有已学任务测试集上的总准确率。
- **AIA (Average Incremental Accuracy, A_AIA)**：每个任务学完后，在所有已学任务测试集上准确率的平均值。定义为 A_AIA = (1/T) * sum_{k=1}^T A^{(<=k)}。
- **遗忘率（Forgetting Rate）**：本文提出了专门针对 CIL 的修正遗忘率指标，通过与 Non-CL（联合训练所有任务）性能的差距来衡量：
  - F_{CIL, Last}^(t) = (1/t) * sum_{i=1}^t (A_i^{(t, NCL)} - A_i^{(t)})
  - F_{CIL, AIA}^(t) = (1/t) * sum_{i=1}^t F_{CIL, Last}^(i)
- 传统遗忘率公式仅适用于 TIL，不适用于 CIL（在 CIL 中，随着类别增多准确率自然下降，该下降不等同于遗忘）。

### 4.5 实现细节

- GPU: NVIDIA GeForce RTX-2080Ti
- 软件环境: Ubuntu Linux 16.04, Python 3.6
- OOD 分数计算使用 scikit-learn（均值/协方差）和 faiss（KNN 检索）
- 运行时间：TPL 约 20.7 min/T (C10-5T), 23.3 min/T (C100-10T), 11.7 min/T (C100-20T), 32.5 min/T (T-5T), 11.2 min/T (T-10T)，与 MORE 和 ROW 相当
---

## 五、核心结果与发现

### 5.1 主要实验：有预训练 CIL 准确率（Table 1）

5 个数据集的平均结果：

| 方法 | 平均 Last ACC (%) | 平均 AIA (%) |
|------|------------------|-------------|
| **TPL (ours)** | **76.21** | **83.23** |
| ROW | 73.72 | 81.50 |
| MORE | 71.59 | 80.77 |
| BEEF | 70.13 | 79.77 |
| DER | 69.77 | 79.81 |
| FOSTER | 68.17 | 79.13 |
| PASS (最佳无replay) | 68.25 | 75.38 |
| Non-CL (上限) | 81.27 | 85.16 |
| **TPL_PFI** | **85.22** | **89.42** |
| Non-CL_PFI (上限) | 87.08 | 90.70 |

逐数据集（Last ACC / AIA）：
- C10-5T: TPL 92.33 / 95.11 (ROW 90.97 / 94.45)
- C100-10T: TPL 76.53 / 84.10 (ROW 74.72 / 82.87)
- C100-20T: TPL 76.34 / 84.46 (ROW 74.60 / 83.12)
- T-5T: TPL 68.64 / 76.77 (ROW 65.11 / 74.16)
- T-10T: TPL 67.20 / 75.72 (ROW 63.21 / 72.91)

TPL 在所有数据集和指标上全面超越此前最优基线 ROW。与最佳无 replay 方法 PASS 相比，平均 Last ACC 差距达 7.96 个百分点（76.21 vs 68.25）。

使用全 ImageNet 预训练的 TPL_PFI 遗忘几乎为零：平均 AIA 89.42%，与 Non-CL_PFI (90.70%) 仅差 1.28 个百分点。
### 5.2 无预训练实验结果（Table 2 和 Table 8）

使用 ResNet-18，5 数据集的平均 Last ACC：

| 方法 | 平均 Last ACC (%) |
|------|------------------|
| **TPL (ours)** | **57.5** |
| DER | 54.2 |
| BEEF | 53.4 |
| ROW | 53.1 |
| FOSTER | 52.2 |
| MORE | 51.2 |

TPL 在 C10-5T (78.4%)、T-5T (48.2%)、T-10T (42.9%) 上最优；DER 在 C100-10T (64.5%) 和 C100-20T (62.5%) 上领先。

### 5.3 遗忘率分析（Table 6, Appendix C.2）

| 方法 | 平均 F_CIL_Last (%) | 平均 F_CIL_AIA (%) |
|------|---------------------|--------------------|
| **TPL** | **5.06** | **1.93** |
| ROW | 7.55 | 3.66 |
| MORE | 9.68 | 4.39 |
| BEEF | 11.14 | 5.39 |
| DER++ | 11.50 | 5.35 |
| **TPL_PFI** | **1.86** | **1.28** |

TPL 的遗忘率在所有对比方法中最低。由于 HAT 已有效消除灾难性遗忘，TPL 的遗忘率主要反映 ICS 问题导致的性能损失。
### 5.4 消融实验

#### 5.4.1 性能增益分解（Figure 2a, Section 5.3）

| 配置 | 5数据集平均 Last ACC (%) | 增益 |
|------|------------------------|------|
| HAT_CIL（基础） | 63.41 | - |
| HAT + MLS | 68.69 | +5.28 |
| HAT + LR | 71.25 | +7.84 |
| **TPL（HAT + LR + MLS）** | **76.21** | **+12.80** |

仅 LR 分数就带来 7.84 个百分点的显著提升（vs MLS 的 5.28），验证了似然比方法的核心有效性。最终融合方案总提升达 12.80 个百分点。

#### 5.4.2 E_t 与 E_t^c 的不同选择（Figure 2b）

核心发现：
1. 估计 P_t^c（非均匀假设）始终优于均匀分布假设（Constant），验证了利用补集分布信息的价值。
2. KNN 作为 E_t^c 估计器优于 MD（因 replay 数据有限，非参数估计更鲁棒）。
3. MD 作为 E_t 估计器优于 KNN（因测试时不需要完整任务训练数据，仅需存储统计量）。

#### 5.4.3 不同 Logit 分数的影响（Figure 2c）

| Logit 分数 | 5数据集平均 Last ACC (%) |
|-----------|------------------------|
| MSP (Max Softmax) | 71.32 |
| EBO (Energy-Based) | 75.76 |
| **MLS (Max Logit)** | **76.21** |

MLS 与 EBO 效果相近，均远超简单的 MSP。
#### 5.4.4 更小 Replay Buffer（Table 3 和 Table 9）

Buffer 减半设置（CIFAR-10: 100, CIFAR-100/TinyImageNet: 1000）下：

| 方法 | 平均 Last ACC (%) | vs 标准buffer下降 |
|------|------------------|------------------|
| **TPL** | **75.56** | -0.65 |
| ROW | 72.70 | -1.02 |
| MORE | 71.44 | -0.15 |

TPL 在更小 buffer 下依然鲁棒——replay 数据在 TPL 中主要用于估计似然比，而非直接防止遗忘（CF 已由 HAT 消除）。

#### 5.4.5 beta_1 和 beta_2 的敏感性（Table 10, Appendix D.4）

在 C10-5T 上的网格搜索：beta_1 in [15.0, 25.0], beta_2 in [0.5, 0.9]，Last ACC 在 91.0%-92.5% 范围波动。方法对这两个超参数不敏感。

### 5.5 ImageNet-380 实验结果（Table 7, Appendix D.1）

| 方法 | Last ACC (%) |
|------|-------------|
| **TPL** | **78.49 +/- 0.89** |
| ROW | 74.52 +/- 1.38 |
| MORE | 72.10 +/- 1.44 |

TPL 比最强 baseline ROW 提升 3.97 个百分点，在大规模设置下验证了方法的有效性。
### 5.6 20 种 OOD 方法的系统基准研究（Appendix A）

对 20 种 OOD 检测方法在统一 TIL+OOD 框架下的基准测试：

核心发现：
1. **OOD AUC 与 CIL ACC 之间存在强线性关系**（Figure 3）。5 个数据集上的 Pearson 相关系数：0.976 (C10-5T), 0.811 (C100-10T), 0.941 (C100-20T), 0.963 (T-5T), 0.980 (T-10T)。提高 OOD AUC 可带来 1.5-3.4 倍的 CIL ACC 线性提升（即拟合斜率）。
2. **不同 OOD 训练技巧对 TIL 准确率影响极小**：所有 20 种方法的 TIL ACC 均值约在 92.98%-93.21% 之间，方差极小。该发现与 Yang et al. (2022) 在 ResNet 上的发现一致，本文验证了在预训练 DeiT 上也成立。
3. 在 20 种 OOD 方法中，VIM（结合特征和 logit）表现最好（平均 CIL ACC 72.1%），但仍显著低于 TPL（76.21%）。
4. KNN 在标准 OOD benchmark 中很强，但在 CIL 中较弱（平均 CIL ACC 仅 57.5%），因为 CIL 中 KNN 只能用极小的 replay 数据计算 OOD 分数。
### 5.7 不同预训练模型的影响（Table 11, Appendix D.5）

| 视觉编码器 | 预训练方式 | 平均 TIL ACC (%) | 平均 CIL ACC (%) |
|-----------|----------|-----------------|-----------------|
| ViT-small | 监督 (完整ImageNet) | 95.30 | **85.35** |
| DeiT-small (TPL_PFI) | 监督 (完整ImageNet) | 95.22 | 85.22 |
| DeiT-small-IN661 (TPL) | 监督 (611类ImageNet) | 92.98 | 76.21 |
| ViT-tiny | 监督 (完整ImageNet) | 91.83 | 77.21 |
| DeiT-tiny | 监督 (完整ImageNet) | 92.29 | 76.44 |
| ViT-small-Dino | 自监督 | 90.69 | 73.80 |
| ViT-base-MAE | 自监督 | 90.12 | 70.97 |

关键发现：监督预训练优于自监督预训练；更大模型带来更高 CIL 性能。

### 5.8 特征分布可视化（Appendix H）

对 C10-5T 各任务提取的特征进行 PCA 降维（384->2 维）和 KDE（核密度估计）可视化：
- 任务分布 {P_t} 与其他任务分布 {P_{t^c}} 之间存在显著重叠区域。
- 具体反例（Figure 5）：某红色星号样本在 P_1 下有更高密度（0.9 vs 绿色星号 0.4），但其似然比仅为 3（0.9/0.3），而绿色星号为 20（0.4/0.02）。传统 TIL+OOD 方法（MORE/ROW）会错误选择红色星号，TPL 基于似然比会正确选择绿色星号。
- 这直接验证了论文的核心论点：仅依靠 IND 密度做任务预测存在失败风险，必须同时考虑补集密度。

### 5.9 运行效率（Table 13, Appendix I.3）

TPL 的训练时间与 MORE 几乎相同，略低于 ROW。推理时间可通过并行化技术（Appendix G.2）达到与标准单通道 CL 方法相近的时间效率——代价是 T 倍的 GPU 内存占用（T 为任务数），可通过调整并行度灵活权衡。
---

## 六、主要贡献与局限性

### 6.1 主要贡献

1. **原则性的 task-id 预测理论框架**：基于 Neyman-Pearson 引理，首次严格证明了似然比检验是任务 ID 预测的 UMP 检验，最大化 AUC。为 CIL 领域的 task-id 预测问题提供了坚实的理论基础。

2. **首次利用 CIL 场景特有的 replay 数据估计补集分布 P_{t^c}**：弥补了传统 OOD 检测方法无法估计真实 OOD 分布的理论缺陷，在方法论上与 TIL+OOD 范式形成本质差异化。

3. **创新的训练策略**：在每个任务模型中引入 O 类，用 replay 数据作为其他任务的代表样本进行联合训练，使模型天然具备区分本任务数据与其他任务数据的能力，为似然比推断提供更好的特征表示。

4. **特征空间与 logit 空间的原则性融合**：基于能量模型（EBM）的组合性质，将 MD（特征空间密度估计器）、KNN 距离（补集分布估计器）与 MLS（logit 空间分数）三大信号整合到统一框架中。

5. **全面且严谨的实验验证**：17 个基线方法、5 个标准数据集 + 自建 ImageNet-380、有/无预训练、多个指标；详尽消融实验；20 种 OOD 方法的系统基准研究。

6. **方法论贡献**：指出并修正了传统 CIL 遗忘率公式的不适当性，提出了以 Non-CL 性能为参照的修正公式（F_{CIL, Last} 和 F_{CIL, AIA}）。
### 6.2 局限性

1. **依赖 replay buffer**：需存储旧任务数据，存在隐私合规风险和额外存储开销（尽管对 buffer 大小较为鲁棒，但完全没有 replay 数据则方法无法工作）。

2. **简单的随机采样策略**：buffer 管理策略可能不是最优的，更优的 coreset selection 或主动学习策略可能进一步提升。

3. **仅适用于传统离线 CIL**：假设所有任务数据完整可用、任务边界清晰、标签空间不相交。未覆盖在线 CIL、模糊任务边界（Blurry Task Setting, Bang et al., 2022）或跨任务标签重叠等场景。

4. **协方差矩阵存储开销**：每个任务需保存 384x384 协方差矩阵和各类中心向量，在超长任务序列下可能成瓶颈（可考虑对角近似或低秩分解缓解）。

5. **推理时逐任务前向传播**：虽可通过并行化缓解计算时间，但 GPU 内存随任务数线性增长（T 倍于单通道 CL 方法）。
---

## 七、对后续研究的启发与潜在改进方向

### 7.1 理论与方法

1. **似然比框架的跨领域推广**：定理 4.1/4.2 的 UMP 检验框架可推广至 Open-Set Recognition（已知未知类别）、Anomaly Detection with Known Negatives、Few-Shot Class-Incremental Learning 等场景。

2. **无需 replay 的补集分布估计**：探索使用生成模型（Diffusion Models, Normalizing Flows）合成伪 replay 样本，或通过贝叶斯推断从模型参数恢复旧分布，消除对 replay 数据的依赖。

3. **更强大的分布估计器**：用 Score-based Models、Flow-based Models 等现代生成模型替代 MD/KNN，获得更精确的似然比估计。

4. **自适应/可学习超参数**：当前 beta_1、beta_2 使用经验均值的简单倒数计算，可探索元学习（meta-learning）确定最优缩放参数。

5. **与更多 TIL 方法的系统协同研究**：论文提及 TPL 可替换 HAT 为 SupSup、PackNet、PNN、Ternary Masks 等，系统研究不同 TIL 特征隔离策略与似然比框架的协同效应。

### 7.2 应用扩展

6. **在线 CIL 适配**：解决数据流式到达时 replay buffer 的动态更新、分布估计器的增量更新等问题。

7. **跨域/跨数据集 CIL**：似然比方法天然适合处理分布偏移，可扩展到源域和目标域不同的 CIL 场景。

8. **多模态 CIL**：从图像分类扩展到视觉问答、图文匹配等多模态持续学习任务。

9. **超长任务序列 CIL**：在 100+ 任务设置下测试 TPL，同时解决协方差矩阵存储扩展性问题。

### 7.3 实验与分析

10. **OOD-CIL 关系的理论解释**：附录 A 发现的线性关系（斜率 1.5-3.4）值得进一步理论分析其成因，可能涉及 task-id 预测的误差传播特性。

11. **预训练策略对 CIL 的影响**：附录 D.5 显示监督预训练优于自监督预训练，如何设计专门针对 CIL 的预训练策略（无需大规模标注数据）是一个有价值的开放问题。

12. **可解释性研究**：深入分析 O 类训练对特征空间几何结构的影响，以及似然比分数与分类置信度之间的关系。

13. **最优 Buffer 采样策略**：研究如何用最少的 replay 样本最大化补集分布估计精度（如不确定性采样、多样性采样、边界采样等策略）。
---

*本总结基于 Lin et al., "Class Incremental Learning via Likelihood Ratio Based Task Prediction", ICLR 2024 的全文阅读（含正文 Section 1-6 及附录 A-J）生成。*