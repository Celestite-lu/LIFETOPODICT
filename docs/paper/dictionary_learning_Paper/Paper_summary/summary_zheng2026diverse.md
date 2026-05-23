# Diverse Dictionary Learning 详细论文总结

## 一、基本信息

- **标题**：Diverse Dictionary Learning（多样化字典学习）
- **作者**：Yujia Zheng (CMU), Zijian Li (MBZUAI), Shunxing Fan (MBZUAI), Andrew Gordon Wilson (NYU), Kun Zhang (CMU & MBZUAI)
- **发表年份**：2026年
- **发表会议/期刊**：论文中未明确指出具体会议或期刊名称。论文包含致谢"anonymous reviewers, AC"，且主文中提到"Due to page limits"（因页数限制），表明其为有页数限制的会议投稿格式，但最终发表的会议/期刊信息需进一步确认。

---

## 二、研究动机与问题定义

### 2.1 核心问题

在一般的字典学习中，观测数据由隐藏变量通过未知函数生成：**X = g(Z)**，其中 Z 为隐变量（latent variables），g 为生成映射（generative map）。在没有任何额外假设的情况下，仅从观测数据 X 中恢复隐变量 Z 和生成过程 g 是根本病态的（ill-posed）。

### 2.2 现有方法的不足

1. **线性假设过于严格**：尽管字典学习本质上是非参数的（nonparametric），绝大多数现有方法将其线性化为 X = DZ（稀疏编码、SAE等），忽略了现实世界中普遍存在的非线性生成结构。例如，大模型中的 Sparse Autoencoder (SAE) 基于稀疏线性字典学习，难以捕获神经网络中固有的非线性表示结构。

2. **非线性ICA对附加信息的依赖**：现有非线性可识别性工作分为两类——(a) 利用辅助变量（auxiliary variables，如时间索引、域索引）作为弱监督实现统计独立下的可识别性；(b) 对混合函数 g 本身施加限制（如 post-nonlinear 模型）。在因果表示学习中，可识别性通常依赖于干预数据（interventional data）或反事实视图（counterfactual views），这些要求对数据生成过程有一定控制权。

3. **理论假设难以验证**：理论上，附加假设可带来恢复保证，但实践中几乎无法验证这些假设是否成立。更严重的是，大多数现有方法在假设即使是轻微违背的情况下也无法提供任何保证，使得其归纳偏置（如对比学习目标、弱监督）难以在不同设置间泛化。

4. **理论与实践的鸿沟**：既有工作关注的是理想条件下的完全可识别性，而实践中更需要知道"在一般条件下哪些方面仍能被恢复"以及"应当引入什么归纳偏置来引导恢复"。

### 2.3 本文的核心思想

论文提出了一个互补视角：**在一般设置中完全可识别性不可实现时，哪些方面仍然可以有保证地恢复？哪些归纳偏置可以普遍采用？**

具体提出两个研究问题：
- 潜在过程的哪些方面仍然可以被恢复？
- 应该引入哪些归纳偏置来引导恢复？

基于此，论文正式定义了 **Diverse Dictionary Learning（多样化字典学习）** 问题，旨在在最基本假设下提供可操作的可识别性保证（actionable identifiability）。

---

## 三、方法与框架

### 3.1 问题形式化

给定 X = g(Z)，其中：
- X = (X_1, ..., X_{d_x}) in R^{d_x} 为观测变量
- Z = (Z_1, ..., Z_{d_z}) in R^{d_z} 为隐变量
- X 和 Z 分别表示 X 和 Z 的支撑集（support）
- g 假设为 C^2 微分同胚（smooth and injective diffeomorphism onto its image），确保信息不丢失。这是非线性可识别性领域的标准假设（Hyvarinen & Pajunen, 1999），是线性情况中 Restricted Isometry Property (RIP) 的非线性类比。

### 3.2 依赖结构的定义

**定义1**（矩阵值函数的支撑）：矩阵值函数 M: Theta -> R^{m x n} 的支撑是满足存在 theta in Theta 使得 M(theta)_{i,j} != 0 的索引对 (i, j) 的集合。

**定义2**（依赖结构，Dependency Structure）：隐变量 Z 与观测变量 X 之间的依赖结构定义为 g 的 Jacobian 矩阵的支撑：

S := supp(D_z g; Z) = {(i, j) in [d_x] x [d_z] | 存在 z in Z, partial g_i(z)/partial z_j != 0}

该结构捕获了哪些隐变量在**功能上**（functionally）影响哪些观测变量，而非统计依赖性（statistical dependencies）。因此，它不要求 Z 的统计独立性，也不限于 ICA 中通常考虑的混合结构（mixing structures）。如 Figure 1 所示，Jacobian 结构甚至可以捕获隐变量之间的交互（如 Z_1 和 Z_2 的交互）。

### 3.3 广义可识别性框架

#### 观测等价（Defn. 3）
两个模型 theta = (g, p_Z) 和 theta_hat = (g_hat, p_Z_hat) 是观测等价的，记为 theta ~_obs theta_hat，当且仅当对所有 x in X，p(x; theta) = p(x; theta_hat)。

#### 隐变量索引集（Defn. 4）
对任意观测变量集 X_S，其隐变量索引集 I_S 包含于 [d_z] 定义为：

I_S := {i in [d_z] | partial X_S / partial Z_i != 0}

即影响 X_S 的隐变量索引的集合。

#### 集合论不确定度（Defn. 5，核心形式化定义）
两个模型之间存在集合论不确定度（set-theoretic indeterminacy），记为 theta ~_set theta_hat，当且仅当对任意两个观测变量集 X_K 和 X_V 及其隐变量索引集 I_K 和 I_V，存在一个置换 pi 使得 Z_i 不是 Z_hat_{pi(j)} 的函数（即 partial Z_i / partial Z_hat_{pi(j)} = 0），当 (i, j) 满足以下至少一个条件时：

1. **（交 Intersection）** i in I_K cap I_V, j in I_K Delta I_V —— 共享因素不可与独有因素纠缠
2. **（对称差 Symmetric difference）** i in I_K Delta I_V, j in I_K cap I_V —— 独有因素不可与共享因素纠缠
3. **（补 Complement）** i in I_K  I_V, j in I_V  I_K，或 i in I_V  I_K, j in I_K  I_V —— 两组各自的独有因素之间不得相互纠缠

**直观解释**（Figure 2）：I_K cap I_V 是共享隐变量，I_K Delta I_V 是 X_K 或 X_V 独有的隐变量。集合论不确定度保证这三类之间互不纠缠，从而实现基于多样性视角的结构化解耦。

#### 广义可识别性（Defn. 6）
模型 theta 具有广义可识别性（generalized identifiability），当且仅当对任意其他模型 theta_hat，有：

theta ~_obs theta_hat  =>  theta ~_set theta_hat

即观测等价性蕴含集合论不确定度。

### 3.4 广义可识别性的推论（Prop. 1）

若 theta ~_set theta_hat，则对任意两个观测变量集 X_K 和 X_V，Z_i 不是 Z_hat_{pi(j)} 的函数当 (i,j) 满足以下至少一个条件：

1. **对象中心解耦（Object-centric）**：i in I_K, j in I_V  I_K 或 i in I_V, j in I_K  I_V —— 每个"对象"相关的隐变量与另一对象的独有隐变量解耦，对齐对象中心学习中的模块性（modularity）
2. **个体中心解耦（Individual-centric）**：i in (I_K  I_V), j in I_V 或 i in (I_V  I_K), j in I_K —— 某组独有的隐变量与整个另一组解耦，支持域适应中隔离域特定因素
3. **共享中心解耦（Shared-centric）**：i in I_K cap I_V, j in I_K Delta I_V —— 共享部分与所有独有部分解耦，支持迁移学习和泛化

### 3.5 原子区域（Atomic Regions）与 Venn 图

如果隐变量索引集的并集覆盖整个隐变量空间，广义可识别性保证可以扩展到 Venn 图中的每个**原子区域**（atomic region）。原子区域被定义为在有限交集和补集操作下闭合的最小、不重叠区域。

**Example 4**（识别原子区域）：以 I_1, I_2, I_3 为三个观测变量的隐变量索引集，考虑原子区域 (I_1 cap I_2)  I_3。通过两次集合运算将其解耦：
- 步骤 1：取 X_K = X_1, X_V = X_2，利用对称差条件（case (ii) in Defn. 5）将 i in (I_1 cap I_2)  I_3 与 j in I_K Delta I_V 解耦
- 步骤 2：取 X_K = X_1 union X_2, X_V = X_3，利用补条件（case (iii) in Defn. 5）将 i 与 j in I_3 解耦

由此保证该原子区域与所有其他变量解耦，实现**块级可识别性**（block-wise identifiability）。

三个集合构成的标准 Venn 图共有 **7 个原子区域**（见 Appendix B.1）：
1. I_1  (I_2 union I_3)
2. I_2  (I_1 union I_3)
3. I_3  (I_1 union I_2)
4. (I_1 cap I_2)  I_3
5. (I_1 cap I_3)  I_2
6. (I_2 cap I_3)  I_1
7. I_1 cap I_2 cap I_3

论文附录 B.1 为每个原子区域给出了完整的 (X_K, X_V) 选择方案（通过两次 Defn. 5 条件覆盖所有 j not in A）。

**与块级可识别性文献的关系（Remark 2）**：本文的原子区域视角与现有块级可识别性工作（von Kugelgen et al., 2021; Yao et al., 2024b; Li et al., 2023）概念相关但本质不同。现有工作依赖多视图/域等额外信息实现块级可识别性；Yao et al. (2024b) 的"可识别性代数"是先借助多视图信号恢复组再取交集的"自顶向下"方案。本文则从基本假设出发，"自底向上"直接目标交和补的可识别性，不依赖任何外部弱监督。

### 3.6 核心定理

#### 关键假设

**假设1（充分非线性，Sufficient Nonlinearity）**：对每个 i in [d_x]，存在 ||(D_Z g)_{i,.}||_0 个点，使得对应的 Jacobian 向量：

(partial X_i/partial Z_1, ..., partial X_i/partial Z_{d_z})|_{z=z^{(k)}}, k in S_i

线性独立（linearly independent），且该 Jacobian 支撑在估计中不超过真实支撑。

**直观解释**：该假设确保 Jacobian 在足够多样本间变化以张成支撑空间。对每个观测变量 X_i，仅需 ||(D_Z g)_{i,.}||_0 个样本（即影响该观测变量的隐变量数量），通常远小于可用样本量。当 g 光滑且隐变量分布具有连续密度时，独立采样点的 Jacobian 随机向量以概率 1 处于一般位置（general position），因此少量随机样本即可满足条件（例如，若 X_i 依赖5个隐坐标，约5个样本通常就足够）。

第二个条件排除了真实依赖性在所有样本上被全局消除的情况，这类似于因果发现中的忠实性（faithfulness）违反。该假设在文献中被广泛使用（Lachapelle et al., 2022; Zheng et al., 2022; Kong et al., 2023; Yan et al., 2023）。

**条件 i**：Z 的概率密度在 R^{d_z} 上为正（标准条件，几乎所有可识别性结果均假设）。

**条件 ii（稀疏正则化, Sparsity Regularization）**：||D_{Z_hat} g_hat||_0 <= ||D_Z g||_0。**重点强调：这不是对数据生成过程的假设，而仅是在估计过程中施加的实用归纳偏置（inductive bias）**。因此，真实数据生成过程完全不需要是稀疏的。

#### 定理1（广义可识别性，Generalized Identifiability）

在上述假设下，若 theta ~_obs theta_hat，则 theta ~_set theta_hat（即广义可识别性成立）。具体来说，对任意两个观测变量集 X_K 和 X_V，交 I_K cap I_V、对称差 I_K Delta I_V、补 I_K  I_V 和 I_V  I_K 定义的隐变量子集在集合论不确定度意义下可被识别。

**核心证明思路**：通过链式法则 D_{Z_hat} g_hat = D_Z g . D_{Z_hat} phi^{-1}（其中 phi = g_hat^{-1} o g），利用 Jacobian 的线性独立性和 Hall 婚姻定理得到置换 pi 使得 (D_z g)_{i,j} != 0 <=> (D_{z_hat} g_hat)_{i,pi(j)} != 0。然后通过反证法逐一验证 Defn. 5 的三个条件。完整证明见附录 A.2，篇幅约 3 页。

#### 定理2（结构可识别性，Structure Identifiability）

在定理1相同假设下，若 theta ~_obs theta_hat，则 D_{Z_hat} g_hat 的支撑（support）与 D_z g 的支撑在列置换意义下完全相同：

supp(D_z g) = supp((D_{Z_hat} g_hat) P)

其中 P 为置换矩阵。即隐变量与观测变量之间的依赖结构本身可被识别（仅差重标记/relabeling）。

#### 假设2（充分多样性，Sufficient Diversity）

对每个隐变量 Z_i (i in [d_z])，存在一组观测变量 A 使得至少满足以下条件之一：

1. 存在 X_k in A 使得 所有 X_j in A 的 I_j 的并集 = [d_z]，且 I_k 减去 (所有 X_j in A 除 X_k 外的 I_j 的并集) = {i}
2. 存在 X_k in A 使得 所有 X_j in A 的 I_j 的并集 = [d_z]，且 (所有 X_j in A 除 X_k 外的 I_j 的交集) 减去 I_k = {i}
3. (Zheng et al., 2022) 所有 X_j in A 的 I_j 的交集 = {i} —— 结构稀疏条件

**阐明"多样化 vs 稀疏"**：
- 第三条（条件3）正好对应于 Zheng et al. (2022) 的结构稀疏条件，是该方向的现有工作
- 前两条（条件1和2）为本文新增，扩展了可容许结构的类别，提供更大灵活性
- **多样性（diversity）本质不同于稀疏性（sparsity）**：多样化仅要求在连接模式中存在某些变化（即使只有一个不同的边），而稀疏性严格要求稀疏结构。因此，即使在几乎全连接的结构中，只要存在连接模式变异，充分多样性仍可满足。作为对比，锚特征假设（anchor feature assumption, Arora et al., 2012; Moran et al., 2021）要求每个隐变量至少有2个专属观测变量，排除了稠密结构

#### 定理3（元素可识别性，Element Identifiability）

在定理1的所有假设加上充分多样性（假设2）下，theta ~_obs theta_hat => theta ~_elem theta_hat，即可以恢复所有隐变量，仅差**元素级不确定度**（element-wise indeterminacy）：

Z_hat = P_pi phi(Z)

其中 phi(Z) = (phi_1(Z_1), ..., phi_{d_z}(Z_{d_z})) 是逐元素微分同胚（element-wise diffeomorphism），P_pi 是对应于 d_z 置换 pi 的置换矩阵。这意味着每个恢复的隐变量对应于某个真实隐变量的一对一非线性变换，最多差一个全局置换。

**核心证明思路**：从充分多样性的三个条件出发，逐一证明对任意 r != i 有 partial Z_i / partial Z_hat_{pi(r)} = 0，且因 phi 可逆，必须有 partial Z_i / partial Z_hat_{pi(i)} != 0。完整证明见附录 A.4。

### 3.7 通用归纳偏置：依赖稀疏（Dependency Sparsity）

论文指出一个关键洞察：定理1中的稀疏正则化条件（条件 ii）**不是一个关于数据生成过程的假设，而仅是在估计过程中施加的实用归纳偏置**。该偏置反映了一种"奥卡姆剃刀"（Occam's razor）原则——倾向于剃除不必要的依赖关系（shave away unnecessary relations）。

这一归纳偏置的通用性和实用性体现在：
- **理论上**：与因果结构学习中使用的忠实性（faithfulness）、节俭性（frugality）和极小性（minimality）原则一致（Zhang, 2013）
- **实践上**：可集成到任何可访问其对隐变量梯度（Jacobian）的可微模型中，无论模型架构如何

---

## 四、实验设置

### 4.1 合成实验

**数据生成**：按照 Sec. 2 的数据生成过程，所有生成过程为非线性，由带 Leaky ReLU 的 MLP 实现。

**模型架构**：以变分自编码器（VAE）为主干模型，目标函数包含标准 ELBO 和依赖稀疏正则化项：

L = E_{q(Z|X)} [ln p(X|Z)] - beta D_KL(q(Z|X) || p(Z)) + alpha ||D_{Z_hat} g_hat||_0^2

其中 D_KL 为 KL 散度，q(Z|X) 为变分后验，p(Z) 为先验，p(X|Z) 为似然，alpha 和 beta 为正则化权重。**注意**：合成实验中目标函数使用 L0 范数的平方（||.||_0^2）作为稀疏惩罚；而在图像实验中（Sec. 4.2），论文描述为对 Jacobian 施加 L1 正则化（L1 regularization on Jacobian）。在实践中 L0 通常通过 L1 或其他可微代理实现。

**超参数**：使用 10,000 个样本，alpha = beta = 0.05（所有实验统一设置）。

**广义可识别性评估**：生成维度为 {3, 4, 5} 的数据集，将观测变量分为两组 X_K 和 X_V，计算 R^2 分数（越低表示越解耦）：
- **Int**：I_K cap I_V 与 I_K Delta I_V 之间的 R^2 —— 测试"交与对称差解耦"
- **SymDiff**：I_K Delta I_V 与 I_K cap I_V 之间的 R^2 —— 测试"对称差与交解耦"
- **Comp A 和 Comp B**：I_K  I_V 与 I_V  I_K 之间的双向 R^2 —— 测试"互补部分解耦"
- **Ref**：Z 与 Z_hat 之间的 R^2 —— 作为纠缠变量的基准水平（指示若变量已纠缠时预期的 R^2 水平）

**元素可识别性评估**：构建维度为 {3, 4, 5, 6, 7, 8, 9, 10} 的数据集，结构要么满足充分多样性（Assum. 2，"Ours"）要么通过全连接依赖违反（"Base"）。以 Mean Correlation Coefficient (MCC) 为评估指标（遵循 Hyvarinen et al., 2024）。

**额外合成实验**（附录 C.1）：
- 与 OroJAR (Wei et al., 2021) 和 Hessian Penalty (Peebles et al., 2020) 的比较（维度 3, 4, 5）
- 噪声鲁棒性测试（加性噪声下 MCC 对比）
- 正则化权重 lambda（即 alpha）从 0 到 0.05 的敏感性分析（{0, 0.001, 0.005, 0.01, 0.03, 0.05}）

### 4.2 图像实验（Visual Experiments）

**数据集**：三个标准解耦表示学习基准数据集，均包含已知生成因素（物体颜色、形状、尺度、朝向、视点等，从合成渲染到真实世界图像全覆盖）：
- **Cars3D** (Reed et al., 2015)
- **Shapes3D** (Kim & Mnih, 2018)
- **MPI3D** (Gondal et al., 2019)

**主干方法和基线**：将依赖稀疏损失集成到三种基于不同主流生成模型的方法中：
- **VAE-based**：FactorVAE (Kim & Mnih, 2018)
- **GAN-based**：DisCo (Ren et al., 2021)
- **Diffusion-based**：EncDiff (Yang et al., 2024)

每种方法设置三个变体：
1. 原始方法（Base）
2. + Latent Sparsity（对隐变量 Z 施加 L1 正则化）
3. + Dependency Sparsity（本文方法，对 Jacobian 施加正则化）

**评估指标**：FactorVAE score（Kim & Mnih, 2018）和 DCI Disentanglement score（Eastwood & Williams, 2018），均为越高越好。每个方法在 3 个随机种子上重复。

**其他图像实验**（附录 C.2）：
- Fashion 数据集 + Flow 模型上的定性可视化（latent traversals）
- Shapes3D、Cars3D、MPI3D 上的 latent swapping 可控性测试
- Cars3D 上 128x128 高分辨率扩展性测试
- 与 OroJAR 和 Hessian Penalty 在 Cars3D 和 MPI3D 上的全面对比

### 4.3 SAE 相关实验（附录 B.3）

在 OpenWebText 数据集上使用 GPT2-Small，隐变量维度设为 12,288，比较三种 SAE 方法的死特征（dead features）数量：
- Top-K SAE (Gao et al., 2025)
- Batch Top-K SAE (Bussmann et al.)
- JSAE (Farnik et al., 2025) —— 基于依赖稀疏（Jacobian稀疏）

---

## 五、核心结果与发现

### 5.1 广义可识别性验证（Figure 4）

在维度 {3, 4, 5} 上，集合论不确定度所隐含的所有解耦条件均得到满足：

| 变量数 | Int R^2 | SymDiff R^2 | Comp A R^2 | Comp B R^2 | Ref R^2（基准） |
|--------|---------|-------------|------------|------------|---------------|
| 3      | 0.25    | 0.05        | 0.08       | 0.15       | 0.75          |
| 4      | 0.25    | 0.05        | 0.02       | 0.20       | 0.70          |
| 5      | 0.05    | 0.10        | 0.10       | 0.25       | 0.60          |

**关键发现**：在结构上不相交的组件之间（Int、SymDiff、CompA、CompB），R^2 分数一致性地远低于已纠缠变量的基准水平（Ref）。例如，维度3中交与对称差的 R^2 仅为 0.25 和 0.05，而 Ref 高达 0.75。这直接支持了广义可识别性的有效性：集合论操作定义的隐变量子集确实可在估计中被解耦恢复。

### 5.2 元素可识别性验证（Figure 5）

| 变量数 | Ours MCC | Base MCC (全连接) |
|--------|---------|------------------|
| 3      | 0.83    | 0.39             |
| 4      | 0.85    | 0.56             |
| 5      | 0.82    | 0.46             |
| 6      | 0.86    | 0.54             |
| 7      | 0.71    | 0.31             |
| 8      | 0.73    | 0.31             |
| 9      | 0.82    | 0.31             |
| 10     | 0.84    | 0.32             |

**关键发现**：满足充分多样性条件的结构（Ours）在所有维度下均取得高 MCC（0.71-0.86），而全连接依赖结构（Base）的 MCC 显著较低（0.31-0.56）。这直接验证了元素级别可识别性仅在结构多样性存在时成立。全连接结构由于缺乏连接模式的变异而无法满足充分多样性，因而无法通过依赖稀疏正则化实现元素级恢复。这也体现了"多样性不等于稀疏性"：Ours 不要求结构稀疏，只需要连接模式中存在变化。

### 5.3 图像解耦实验结果（Table 1）

#### VAE-based (FactorVAE)

| 方法 | Shapes3D DCI | Cars3D FactorVAE | MPI3D DCI |
|------|-------------|-----------------|-----------|
| FactorVAE (Base) | 0.484 +/- 0.120 | 0.708 +/- 0.026 | 0.345 +/- 0.047 |
| + Latent Sparsity | 0.477 +/- 0.152 | 0.501 +/- 0.434 | 0.325 +/- 0.028 |
| + Dependency Sparsity | **0.575** +/- 0.032 | **0.752** +/- 0.040 | **0.384** +/- 0.031 |

#### Diffusion-based (EncDiff)

| 方法 | Shapes3D DCI | Cars3D FactorVAE | MPI3D DCI |
|------|-------------|-----------------|-----------|
| EncDiff (Base) | 0.901 +/- 0.050 | **0.779** +/- 0.060 | **0.676** +/- 0.018 |
| + Latent Sparsity | 0.891 +/- 0.057 | 0.729 +/- 0.003 | 0.684 +/- 0.020 |
| + Dependency Sparsity | **0.947** +/- 0.005 | 0.756 +/- 0.041 | 0.667 +/- 0.047 |

#### GAN-based (DisCo)

| 方法 | Shapes3D DCI | Cars3D FactorVAE | MPI3D DCI |
|------|-------------|-----------------|-----------|
| DisCo (Base) | 0.710 +/- 0.020 | 0.727 +/- 0.106 | 0.306 +/- 0.079 |
| + Latent Sparsity | 0.707 +/- 0.024 | 0.761 +/- 0.148 | 0.314 +/- 0.050 |
| + Dependency Sparsity | **0.712** +/- 0.018 | **0.789** +/- 0.029 | **0.324** +/- 0.059 |

**核心发现**：
1. **依赖稀疏在绝大多数设置下优于潜变量稀疏**：在 FactorVAE 和 DisCo 主干上，Dependency Sparsity 在所有数据集/指标组合中均为最佳。尤其在 VAE 主干上，改善最为显著（Cars3D FactorVAE 从 0.708 升至 0.752；Shapes3D DCI 从 0.484 升至 0.575）。
2. **Diffusion 主干下表现有分化**：EncDiff 本身已很强（如 Shapes3D 的 DCI 达 0.901），Dependency Sparsity 在 Shapes3D 上进一步将 DCI 推至 0.947、FactorVAE 至 1.0000，但在 Cars3D FactorVAE（0.756 vs Base 0.779）和 MPI3D DCI（0.667 vs Base 0.676）上略低于 Base。论文明确表述为"across **most** datasets and backbone methods"（并非所有），这符合 Occam's razor 偏置与强 Diffusion 模型之间的权衡。
3. **潜变量稀疏有时有害**：在多个设置中（如 Cars3D 的 FactorVAE，Latent Sparsity 将 FactorVAE 从 0.708 降至 0.501），L1 对 Z 的正则化反而损害了解耦性能。这与可解释性文献中对潜变量稀疏局限性的最新讨论一致（如 Sharkey et al., 2025 关于特征吸收、线性约束和高维问题）。
4. **跨生成模型类型的通用性**：依赖稀疏作为归纳偏置在 VAE、GAN 和 Diffusion 三种主流架构上均展示出一致的益处（在大多数组合中），验证了其作为"通用归纳偏置"的声称。

### 5.4 附加对比（附录 C.1、C.2）

**与 OroJAR/Hessian Penalty 对比（Table 4）**：

| 维度 | Ours MCC | OroJAR MCC | Hessian Penalty MCC |
|------|---------|-----------|-------------------|
| 3    | 0.8258 +/- 0.0085 | 0.7288 +/- 0.0280 | 0.8257 +/- 0.0240 |
| 4    | 0.8449 +/- 0.0043 | 0.6301 +/- 0.0810 | 0.8352 +/- 0.0396 |
| 5    | 0.8048 +/- 0.0080 | 0.5119 +/- 0.1482 | 0.7789 +/- 0.0174 |

**发现**：本文方法在各维度上均优于 OroJAR，差距随维度增加而扩大（dim 5 时 Ours 0.805 vs OroJAR 0.512）。Hessian Penalty 在低维（dim 3 时 0.8257 vs Ours 0.8258，基本持平）与本文方法接近，但随维度增加差距拉大（dim 5 时 0.779 vs 0.805）。这两种惩罚均不提供非参数可识别性保证，而以结构方式惩罚依赖映射有助于更好地恢复真实隐变量。

**噪声鲁棒性（Table 5）**：

| 维度 | Ours w/o noise | Ours w/ noise | Base |
|------|---------------|--------------|------|
| 3    | 0.8258 +/- 0.0085 | 0.8210 +/- 0.0088 | 0.3814 +/- 0.0369 |
| 4    | 0.8449 +/- 0.0043 | 0.8381 +/- 0.0093 | 0.5467 +/- 0.0326 |
| 5    | 0.8048 +/- 0.0080 | 0.7944 +/- 0.0134 | 0.4576 +/- 0.1075 |

**发现**：加入加性噪声后依赖稀疏方法的 MCC 仅有微小下降（如 dim 3 从 0.826 到 0.821），而 Base 模型严重恶化（dim 3 MCC 降至 0.381）。这验证了 Remark 1 的扩展性和依赖稀疏在噪声下稳定隐变量恢复的能力。

**正则化权重 lambda 敏感性（Table 6）**：
MCC 从 lambda = 0 时约 0.679-0.732（dim 3-5）平稳递增至 lambda = 0.03 时约 0.810-0.827，lambda = 0.05 时约 0.810-0.842 趋于平稳。方法在非欠正则化区域（lambda >= 0.03）表现稳定且对超参数不过分敏感。

**扩展性（Table 7）**：Cars3D 128x128 分辨率下，Dependency Sparsity 的 FactorVAE 为 0.723 +/- 0.023（vs 64x64 的 0.752 +/- 0.040）和 DCI 为 0.141 +/- 0.004（vs 64x64 的 0.144 +/- 0.053），性能与 64x64 设置一致（误差带重叠），表明改善来自结构正则化本身而非图像分辨率。

**更全面的图像消融（Table 8）**：加入 OroJAR 和 Hessian Penalty 到 FactorVAE 主干，在 Cars3D 上 OroJAR 的 FactorVAE 降至 0.165、DCI 降至 0.030，Hessian Penalty 的 FactorVAE 降至 0.321、DCI 降至 0.082，均远差于 Dependency Sparsity（FactorVAE 0.752, DCI 0.144）。

### 5.5 死特征实验（Table 3, 附录 B.3）

| 方法 | Dead Features 数量 |
|------|------------------|
| Top-K SAE | 439 |
| Batch Top-K SAE | 207 |
| JSAE (依赖稀疏) | **62** |

**发现**：基于依赖稀疏的 JSAE 死特征数量（62）远少于 Top-K SAE（439）和 Batch Top-K SAE（207），表明依赖稀疏在保持活跃有意义的隐变量特征方面具有优势。

### 5.6 定性可视化

- **Fashion + Flow + Dependency Sparsity（Figure 7）**：单个隐变量坐标清晰对应于性别、鞋跟高度、上衣宽度，因素间干扰极小。
- **Shapes3D + EncDiff + Dependency Sparsity（Figure 8）**：隐变量清晰解耦为墙面角度、墙面颜色、物体形状和物体颜色。
- **Latent Swapping 实验（Figures 9, 10, 11）**：在 Shapes3D（交换 floor/wall color）、Cars3D（交换 azimuth/color）、MPI3D（交换 rotation/background）上，交换单个因子的隐变量可干净地转移对应属性而不产生非预期副作用，验证了依赖稀疏鼓励局部化、不重叠的隐变量影响。

---

## 六、主要贡献

1. **提出新问题**：正式定义了 Diverse Dictionary Learning 问题，为在一般无约束设置下研究哪些隐变量结构仍可被恢复提供了全新的理论框架。

2. **集合论可识别性理论**：建立了基于集合论不确定度的广义可识别性概念（Defn. 5-6），在最基本假设（微分同胚 + 充分非线性 + 正密度 + 依赖稀疏正则化）下即可保证交、补、对称差定义的隐变量子集的可识别性。

3. **Venn 图与原子区域识别**：展示了通过集合代数（set algebra）的基本操作可以灵活组合构造 Venn 图并识别所有原子区域，实现块级可识别性。从"属-种差定义"（genus-differentia definitions, Granger, 1984）的角度提供了理解隐藏世界的原则性方法——交集捕获"属"（共享因素），补集和对称差隔离"种差"（独有因素）。

4. **结构可识别性定理**（Theorem 2）：证明了隐变量-观测变量间的完整依赖结构可在置换意义下被识别，不依赖于统计独立性假设。

5. **充分多样性条件**（Assum. 2）：提出推广的结构条件，包含三条（其中两条为本文新增），统一并扩展了 Zheng et al. (2022) 的结构稀疏条件。核心创新在于区分了"多样性"与"稀疏性"——多样性不要求稀疏结构，即使在几乎全连接图中仍可满足，仅需连接模式中存在变异。

6. **通用归纳偏置的揭示**：阐明了依赖稀疏（dependency sparsity/Jacobian稀疏）作为一个简单、可广泛集成的归纳偏置，能够解释从部分可识别性到完全可识别性的各种保证。该偏置不要求真实数据生成过程稀疏，仅作为估计中的正则化。

7. **全面的实验验证**：在合成数据（3-10维）和三个真实图像基准数据集上，跨 VAE、GAN、Diffusion 三种生成模型主干，验证了理论的有效性和归纳偏置的实用价值。额外在 LLM（GPT2-Small）上验证了依赖稀疏在减少死特征方面的优势。

---

## 七、主要局限性

1. **充分非线性假设（Assum. 1）**：虽然通常较温和（仅需少量样本的 Jacobian 向量线性独立），且在光滑函数加连续密度的条件下以概率 1 成立，但在局部过于平坦的函数区域（如梯度消失的饱和区域）可能不成立。

2. **可逆性（微分同胚）要求**：继承了非线性可识别性领域的标准假设。当生成过程存在信息丢失（非单射，如降维映射）时，丢失的信息在原则上无法恢复。论文在附录 B.6 中讨论了处理部分非可逆性的扩展方向（如结合 Chen et al., 2024 的时间信息方法），但这需要额外信息，超出当前框架范围。

3. **噪声处理的限制**：主要理论结果针对确定性函数 X = g(Z)。Remark 1 指出可自然扩展到加性噪声，且在附录 B.6 中引用了 Zheng et al. (2025) 基于 Hu-Schennach 定理（2008）处理一般噪声的扩展方向。但处理任意纠缠的非加性噪声仍需要更强的假设。

4. **大规模模型上的计算开销**：计算完整 Jacobian 对于大模型可能昂贵。论文在附录 B.5 中讨论了缓解策略——先识别活跃潜变量坐标（使 Jacobian 计算量缩小数个数量级，因活跃块通常远小于全隐变量空间）和使用高效闭式表达式（对带残差注意力和前馈结构的模型）。但依赖稀疏正则化的训练速度仍约为潜变量稀疏正则化的两倍（Farnik et al., 2025）。

5. **基础模型应用的开放性**：论文明确将"在基础模型上探索广义可识别性"列为未来工作方向和当前局限。虽然引用了 Farnik et al. (2025) 的 JSAE 工作作为实证支持，但本文方法在超大规模模型（如 GPT-4 级别）上的直接应用和验证仍然缺失。

6. **充分多样性的必要性未证明**：论文推测充分多样性条件可能在某些设置下是元素级可识别性的必要条件，但未给出严格的必要性证明。

---

## 八、对后续研究的启发与潜在改进方向

1. **基础模型与可识别性**：论文明确指出，当前基础模型主要受经验洞察驱动，基于可识别性的归纳偏置被系统性地忽视了，这可能为突破性进展提供新方向。随着海量数据和计算资源的可用，渐近保证（asymptotic guarantees）正变得越来越具有实际意义。在 LLM 中系统探索广义可识别性是重要开放方向。

2. **机械可解释性（Mechanistic Interpretability）的根本改进**：本文理论为当前 SAE（Sparse Autoencoder）的两大核心开放问题提供了原则性解决方案——(a) SAE 仅能处理线性生成假设，限制了非线性 LLM 的真实表示恢复；(b) 潜变量稀疏导致特征吸收（feature absorption）、极端高维（数百万维）和特征分裂（feature splitting）。Diverse Dictionary Learning 的非线性可识别性保证结合依赖稀疏，可能从根本上提升 LLM 可解释性研究的理论严谨性。

3. **域适应与迁移学习的灵活框架**：广义可识别性中的个体中心解耦（individual-centric disentanglement）直接支持迁移学习中分离不变内容（content/invariant）和变化风格（style/changing），而对象中心解耦（object-centric）支持模块化对象表示。这些提供了远比现有方法更灵活的理论基础。

4. **可控生成（Controllable Generation）**：论文通过 latent swapping 实验直观展示了依赖稀疏如何实现精准单属性操控而不产生非预期副作用（如戴眼镜操作意外增加年龄）。将可识别性指导的归纳偏置嵌入生成模型可减少语义操控中的"意外更改"，对视觉生成和编辑应用有直接价值。

5. **多模态对齐**：多模态场景可自然映射到字典学习框架——语义共享概念对应于多组观测变量的隐变量交集（I_K cap I_V），模态特有概念（如图像的纹理、音频的音量）对应于补集（I_K  I_V）。广义可识别性理论为多模态表示学习中的信息对齐与错位问题提供了理论基础。

6. **科学发现的形式化**：论文以牛顿观察苹果落地推断出引力（X = f(Z) -> 从 X 恢复 Z）为例，论述了科学发现本质上可形式化为字典学习问题。广义可识别性为保证恢复的隐变量与真实隐变量之间的有意义对应提供了必要的理论保证。这在气候科学、神经影像、基因组学等领域已有初步应用。

7. **集合代数的进一步探索**：论文开创性地将集合代数（set algebra）引入可识别性理论，将其作为构造多样化隐变量视角的组合工具。值得探索的方向包括：更复杂的集合运算能否提供更强的保证？是否能将可识别性运算形式化为一个代数系统？

8. **充分多样性的充要性证明**：论文推测充分多样性可能是（不施加分布或函数形式约束时的）必要条件，但未证明。严格的必要条件分析将进一步闭合理论框架。

9. **降低计算开销的算法设计**：设计更高效的 Jacobian 稀疏惩罚计算方法（如随机投影近似、仅用子网络梯度、层间局部 Jacobian 等），使依赖稀疏能够更轻量地扩展到超大规模模型是一个重要的工程方向。

10. **非微分同胚与更弱假设下的扩展**：当前框架要求 g 为微分同胚且充分非线性。探索在更弱（或更易验证）假设下的广义可识别性理论——例如，当仅能保证局部可逆性时，集合论可识别性能保留多少——是一个重要的理论方向。
