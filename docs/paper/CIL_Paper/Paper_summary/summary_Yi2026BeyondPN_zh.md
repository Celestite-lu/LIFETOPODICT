# Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for Class-Incremental Learning

Huiyu Yi 1 2 Zhiming Xu 1 2 Dunwei Tu 1 2 Zhicheng Wang 1 3 Baile Xu 1 2 Furao Shen 1 2

# Abstract

Nearest Class Mean (NCM) 分类器因其相比 Fully Connected 层更优越的抗 catastrophic forgetting 能力，在 Class-Incremental Learning (CIL) 中被广泛采用。虽然 Neural Collapse (NC) 理论通过假设特征坍缩为单一的点来支持 NCM 的最优性，但在 CIL 中，非线性 feature drift 和训练不充分常常阻碍这一理想状态的达成。因此，类别表现为复杂的 manifold 而非坍缩的点，使得单点 NCM 次优。为解决这一问题，我们提出了 Hierarchical-Cluster SOINN (HC-SOINN)，一种通过"local-to-global"表示来捕捉这些 manifold 拓扑结构的新型分类器。此外，我们引入了 Structure-Topology Alignment via Residuals (STAR) 方法，该方法采用细粒度的 pointwise trajectory tracking 机制来主动变形已学习的拓扑，使其能够精确适应复杂的非线性 feature drift。理论分析和 Procrustes distance 实验验证了我们框架对 manifold 变形的鲁棒性。我们将 HC-SOINN 集成到七种 state-of-the-art 方法中，替换它们原有的分类器，取得了一致的性能提升，突出了我们方法的有效性和鲁棒性。代码可在 https://github.com/yhyet/HC_SOINN 获取。

1 National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing, China 2 School of Artificial Intelligence, Nanjing University, Nanjing, China 3 School of Computer Science, Nanjing University, Nanjing, China. Correspondence to: Baile Xu <xubaile@nju.edu.cn>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

# 1. Introduction

Class-Incremental Learning (CIL) 旨在使模型能够顺序学习新类别，而不会 catastrophically forgetting 先前获得的知识 (Belouadah et al., 2021; Zhou et al., 2024a; McCloskey & Cohen, 1989)。传统的 Fully Connected (FC) 分类器由于对新任务存在严重的 weight bias 而极易遗忘 (Nguyen et al., 2019)，而 Nearest Class Mean (NCM) 分类器 (Rebuffi et al., 2017) 因其优越的鲁棒性已成为现代 CIL 框架的基石。这种经验偏好通过 Neural Collapse (NC) 理论的视角获得了重要的理论支持 (Papyan et al., 2020; Yang et al., 2023)。NC 理论证明，在训练的 terminal phase，intra-class features 坍缩到各自的 class means，使得 NCM 分类器等价于最优线性分类器。因此，基于 NCM 的方法已成为增量场景中维持 feature-classifier alignment 的事实标准 (Zhou et al., 2025; Yi, 2024; Zhu et al., 2024; Tu et al., 2025b)。

然而，在实际 CIL 设置中，模型达到这一理想 terminal phase 的假设经常由于三个关键因素而被违反。首先，初始任务中的模型通常不会被训练到 collapsed point 以避免严重的 overfitting。其次，在增量步骤中，新类的训练通常受到约束（例如，通过 regularization 或 distillation）以保持旧类性能，导致收敛不充分。最后，普遍存在的"feature drift"问题导致旧类的表示随着 backbone 的更新而偏移 (Yu et al., 2020)，使模型进一步远离 collapsed state。在这些约束下，NC1 property（zero intra-class variance）被破坏 (Papyan et al., 2020)。类别特征不再坍缩为单一的点，而是位于复杂的高维 manifold 上。虽然 NCM 对简单的线性平移具有鲁棒性，但当这些 manifold 经历非线性 drift，可能形成偏离 single-prototype 假设的"dumbbell"或"crescent"结构时，它无法表示这些 manifold (Allen et al., 2019)。

在本文中，我们挑战了对 single-prototype 分类器的依赖，并提出了 CIL 的 topology-aware 表示。我们引入了 HC-SOINN (Hierarchical-Cluster SOINN)，一种旨在捕捉类 manifold 内在结构的新型分类器。通过将 hierarchical clustering 与改进的 Self-Organizing Incremental Neural Network (SOINN) (Furao & Hasegawa, 2006) 相结合，HC-SOINN 通过"local centers"和"global center"的组合来表示每个类别。在推理过程中，这些拓扑点被融合以提供更忠实地尊重 manifold 几何的决策边界。

此外，我们的 Procrustes analysis 揭示了 CIL 中的 feature drift 远非简单的 rigid transformation (Goldberg & Ritov, 2009)，在后期的任务中 normalized distances 频繁达到 0.3−0.4。这一显著偏差表明 feature drift 主要是非线性的，涉及无法用简单 rigid transformation 近似的复杂变形。尽管所提出的 HC-SOINN 凭借其拓扑表示天生比 single-prototype NCM 更能抵抗此类变形，但其承受严重非线性扭曲的能力仍然有限。为弥补这一差距，我们提出了 STAR (Structure-Topology Alignment via Residuals)。利用 HC-SOINN 的多节点粒度，STAR 超越了 global alignment，执行灵活的 pointwise trajectory tracking。这一机制使已学习的 manifold 结构能够主动变形并精确适应 feature space 的复杂演化。

我们的贡献总结如下：

- 我们重新审视了 NCM 在 Neural Collapse 下的最优性，并使用 Procrustes analysis 经验性地揭示了 feature drift 导致显著的非线性结构变形，暴露了 single-prototype 方法的失败。
- 我们提出了 HC-SOINN 分类器，利用 hierarchical topological structure 比 single-prototype 方法更准确地表示类 manifold。
- 我们引入了 STAR，将范式从 drift resistance 转变为 drift adaptation。利用 pointwise trajectory tracking，STAR 使拓扑能够主动变形以匹配复杂的非线性 feature evolution。
- 在三个主流 CIL benchmark 上的大量实验以及与七种 state-of-the-art 方法的集成，证明了我们方法的有效性和鲁棒性。

# 2. Related Works

Continual Learning with Pre-trained Models 利用 PTM 的鲁棒表示来减轻 forgetting (Zhou et al., 2024a)。Parameter-Efficient Fine-Tuning (PEFT) 方法，如 DualPrompt (Wang et al., 2022b)、CODA-Prompt (Smith et al., 2023)、MQMK (Tu et al., 2025a)、SEMA (Wang et al., 2025) 和 CL-LoRA (He et al., 2025)，通过 task-specific prompts、attention-based weighted summation 或 modularized adapters 和 LoRA modules 来适配 backbone。另一种方法是 representation-based 方法，专注于利用 frozen features；例如，SimpleCIL (Zhou et al., 2025) 使用 class prototypes 作为 baseline，而 APER 和 EASE (Zhou et al., 2024b) 通过 embedding aggregation 或 expandable subspace ensembles 进一步增强性能。除此之外，KAC 引入了基于 Kolmogorov-Arnold networks 的非线性分类器来优化 decision boundaries 并减轻 drift (Hu et al., 2025)。

## Neural Collapse Theory

Neural Collapse 理论描述了在训练 terminal phase 观察到的现象，通过四个关键性质进行数学形式化 (Papyan et al., 2020)：Variability Collapse (NC1)，即 intra-class feature variance 趋近于零，样本收敛到其 class means $\mu _ { k }$；Convergence to Simplex ETF (NC2)，即这些 class means 形成几何最优的 Simplex Equiangular Tight Frame；Feature-Classifier Alignment (NC3)，意味着 classifier weights 与归一化的 class means 完美对齐；以及 Simplification to NCC (NC4)，即决策规则等价于 NCM 分类器。在 Few-Shot Class Incremental Learning (FSCIL) 的背景下 (Tao et al., 2020)，NC 理论证明了使用基于 NCM 的分类器在增量步骤中维持 feature-classifier alignment 的合理性，启发了利用固定 ETF 结构或 prototype alignment 来整合有限数据的新类别同时保持旧类知识的方法 (Yang et al., 2023; Tu et al., 2025b)。

Self-Organizing Incremental Neural Network (SOINN) 是一种经典的神经网络，专为 unsupervised incremental learning 而设计 (Furao & Hasegawa, 2006)。它通过动态调整 nodes 和 edges 来表示 non-stationary data streams 的拓扑。为了将其能力扩展到 supervised tasks，已开发了若干变体。例如，Enhanced SOINN 采用 single-layer 结构并改进了 density-based noise removal (Furao et al., 2007)。此外，Adjusted SOINN Classifier 将 class labels 纳入 incremental learning 过程 (Shen & Hasegawa, 2008)，使网络能够处理 online classification 并适应 overlapping class distributions。

# 3. Preliminary & Motivation

# 3.1. Problem Setting

Class-Incremental Learning (CIL) 旨在从任务流 ${ \mathcal { S } } = \{ { \mathcal { T } } _ { 1 } , { \mathcal { T } } _ { 2 } , \ldots , { \mathcal { T } } _ { T } \}$ 中顺序学习。对于每个任务 $\mathcal { T } _ { t }$，模型被提供训练数据集 $\mathcal { D } _ { t } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$，其中 $\mathbf { x } _ { i } \in \mathcal { X }$ 表示输入样本，$y _ { i } \in \mathcal { D } _ { t }$ 是其对应的标签，来自 label space $\mathcal { \partial } _ { t }$。CIL 的一个核心约束是 label spaces 的互斥性：对于任意 $j \neq k$，$\mathcal { V } _ { j } \cap \mathcal { V } _ { k } = \emptyset$。

当转换到任务 $\mathcal { T } _ { t }$ 时，来自先前任务的数据集 $\{ \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { t - 1 } \}$ 不再可访问。目标是获得一个模型，能够正确分类来自所有已观察类别的并集 $\textstyle { \mathcal { C } } _ { t } = \bigcup _ { j = 1 } ^ { t } y _ { j }$ 的样本。在推理过程中，给定一个测试样本 x，模型必须在不知道 task ID 的情况下预测 $\hat { y } \in \mathcal C _ { t }$，这被称为 class-incremental setting。

# 3.2. Pre-trained Models and the NCM Classifier Paradigm

在现代 CIL 的背景下，模型通常由一个强大的 pre-trained feature extractor $f _ { \theta } : \mathcal { X } \rightarrow \mathbb { R } ^ { d }$ 和一个分类机制组成。NCM 分类器是在增量步骤中维持性能的主流选择。对于每个类别 $c \in { \mathcal { C } } _ { t }$，prototype $\pmb { \mu } _ { c }$ 被定义为其训练特征的 mean embedding：

$$
\boldsymbol {\mu} _ {c} = \frac {1}{| \mathcal {D} _ {c} |} \sum_ {(\mathbf {x}, y) \in \mathcal {D} _ {c}} f _ {\theta} (\mathbf {x}), \tag {1}
$$

其中 $\mathcal { D } _ { c }$ 表示属于类别 c 的样本集。在推理过程中，新样本 x 的分类规则由 maximum cosine similarity 定义：

$$
\hat {y} = \underset {c \in \mathcal {C} _ {t}} {\operatorname{argmax}} \cos \left\langle f _ {\theta} (\mathbf {x}), \boldsymbol {\mu} _ {c} \right\rangle = \underset {c \in \mathcal {C} _ {t}} {\operatorname{argmax}} \frac {f _ {\theta} (\mathbf {x}) \cdot \boldsymbol {\mu} _ {c}}{\| f _ {\theta} (\mathbf {x}) \| _ {2} \| \boldsymbol {\mu} _ {c} \| _ {2}}. \tag {2}
$$

# 3.3. Motivation: Procrustes Distance Experiment

为了研究 feature drift，我们通过在 Split CIFAR-100 和 Split CUB-200 上追踪初始类别 $( \mathcal { C } _ { i n i t } )$ 的结构演化来进行几何分析。我们使用初始（Task 1）和当前（Task t）表示之间的 Average Procrustes Distance $( d _ { P } ^ { ( t ) } )$ 来量化 manifold deformation：

$$
d _ {P} ^ {(t)} = \frac {1}{| \mathcal {C} i n i t |} \sum_ {c \in \mathcal {C} _ {i n i t}} d _ {P} (\mathbf {H} _ {c} ^ {(1)}, \mathbf {H} _ {c} ^ {(t)}), \tag {3}
$$

其中 $\mathbf { H } _ { c } ^ { ( 1 ) }$ 和 $\mathbf { H } _ { c } ^ { ( t ) }$ 表示类别 c 的 feature matrices。我们追踪了代表性的 PEFT 方法（详见 Section 5）。

如图 1 所示，我们使用 0.1 作为 quasi-linear drift 的经验阈值。在 CIFAR-100 上，我们观察到 $d _ { P } ^ { ( t ) }$ 从 Task 2 开始就持续超过此界限，并稳步攀升至 $0.35-0.40$。在 CUB-200 上，像 DualPrompt 这样的方法表现出 manifold distortion 的快速增加。这一上升趋势证实了持续训练导致 feature space 从根本上偏离初始结构。因此，类别分布演化为复杂 manifold，而非维持 Neural Collapse 预测的紧致结构，挑战了实际 CIL 场景中 single-prototype NCM 假设的最优性。

![](images/77443a4a0379ca72991806ae921b9973abb9d9ab263e395bdc0fb613b56f7159.jpg)
(a) Split CIFAR-100

![](images/77763a92889682fb7f768a2560c6bd841ace7f296bbd8fd5cb23dd759fd70680.jpg)
(b) Split CUB-200
Figure 1. 初始类别在增量任务上的 Average Procrustes Distance $( d _ { P } ^ { ( t ) } )$ 的演化。红色虚线 $( y = 0.1 )$ 表示 quasi-linear drift 的经验阈值。

# 4. The Proposed Methods

# 4.1. HC-SOINN: Topological Manifold Representation

为了准确表示由于不完全 neural collapse 而形成的类 manifold，我们提出了 HC-SOINN (Hierarchical-Cluster Self-Organizing Incremental Neural Network) 分类器。与假设 unimodal Gaussian distribution（single prototype）的标准 NCM 不同，HC-SOINN 将每个类别 c 建模为 topology graph $\mathcal { G } _ { c } = ( \nu _ { c } , \mathcal { E } _ { c } )$，其中 ${ \mathcal { V } } _ { c } = \{ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { K _ { c } } \}$ 表示一组 local sub-prototypes，$\mathcal { E } _ { c }$ 表示捕捉 manifold 形状的 connectivity edges。

$\mathcal { G } _ { c }$ 的构建以 coarse-to-fine 的方式进行，融合了 hierarchical clustering 的稳定性和 SOINN 的适应性。

# 4.1.1. Hierarchical Initialization

直接在 raw data streams 上应用 incremental clustering（如 SOINN）通常对噪声和输入顺序敏感。为缓解这一问题，我们首先在 memory buffer 中累积 feature samples。一旦 buffer 满或任务结束，我们执行 Agglomerative Hierarchical Clustering 来初始化 manifold skeleton (Johnson, 1967)。

给定一组 normalized features $\mathbf { Z } _ { c } = \{ \tilde { f } _ { \theta } ( \mathbf { x } ) \ | \ y = c \}$，我们使用 average linkage (UPGMA) 准则结合 cosine distance metric 来计算 linkage matrix。具体而言，两个簇 $\mathcal { C } _ { i }$ 和 $\mathcal { C } _ { j }$ 之间的距离定义为所有 inter-cluster pairs 之间的平均 cosine distance：

$$
d (\mathcal {C} _ {i}, \mathcal {C} _ {j}) = \frac {1}{| \mathcal {C} _ {i} | | \mathcal {C} _ {j} |} \sum_ {\mathbf {u} \in \mathcal {C} _ {i}} \sum_ {\mathbf {v} \in \mathcal {C} _ {j}} (1 - \cos \langle \mathbf {u}, \mathbf {v} \rangle). \tag {4}
$$

对生成的 dendrogram 进行切割以获得目标簇数量 $K _ { i n i t }$，产生初始节点集 $\mathcal { V } _ { c } ^ { ( 0 ) }$。选择这种特定的 linkage criterion 是因为它对 outliers 具有鲁棒性，并且能够在 hypersphere 上产生紧致的 spherical clusters，为后续的 self-organizing phase 提供稳定的初始化。

![](images/3d84988b037718e98558ecf9cf73f29fd91f03f46b021389145a52a371fcff12.jpg)
Figure 2. HC-SOINN 和 STAR 的总体 pipeline。该方法首先在固定的 feature embeddings 上构建 topology-aware hierarchical classifier，以建模 class-incremental learning 中出现的复杂 class manifolds。然后采用 STAR 机制通过 pointwise tracking 主动将此拓扑适应于非线性 feature drift，实现 dual-view inference 而无需 retraining。

# 4.1.2. Spherical SOINN Refinement

为了捕捉 manifold 的细粒度几何和连接关系，我们对初始节点 $\mathcal { V } _ { c } ^ { ( 0 ) }$ 应用简化的 SOINN 机制。我们将标准的 Competitive Hebbian Learning (CHL) (Martinetz et al., 1991) 适配到由归一化强加的单位 hypersphere 几何。

对于每次迭代，给定输入信号 $\mathbf { z } \in \mathbf { Z } _ { c }$（或来自前一阶段的 cluster center），我们从 $\gamma _ { c }$ 中识别 winner node $s _ { 1 }$ 和 runner-up $s _ { 2 }$：

$$
s _ {1} = \underset {\mathbf {v} _ {i} \in \mathcal {V} _ {c}} {\operatorname{argmax}} \cos \langle \mathbf {z}, \mathbf {v} _ {i} \rangle , \quad s _ {2} = \underset {\mathbf {v} _ {i} \in \mathcal {V} _ {c} \backslash \{s _ {1} \}} {\operatorname{argmax}} \cos \langle \mathbf {z}, \mathbf {v} _ {i} \rangle . \tag {5}
$$

Topology Learning: 如果 $s _ { 1 }$ 和 $s _ { 2 }$ 不连通，则在 $\mathcal { E } _ { c }$ 中创建一条边 $\left( { { s _ { 1 } } , { s _ { 2 } } } \right)$。每条边有一个 age 属性，随时间递增；超过最大 age $a g e _ { m a x }$ 的边被移除。这一机制动态学习 manifold 内的 adjacency relations。

Spherical Update: 标准 SOINN 通过 linear interpolation 更新节点位置。然而，在我们的设置中，节点必须保持在 unit hypersphere 上。因此，我们采用 Spherical Linear Interpolation (SLERP) (Shoemake, 1985) 来更新 winner $s _ { 1 }$ 及其 topological neighbors $\mathcal { N } ( s _ { 1 } )$：

$$
\begin{array}{l} \mathbf {v} _ {s _ {1}} \leftarrow \operatorname{SLERP} (\mathbf {v} _ {s _ {1}}, \mathbf {z}, \eta_ {1}), \\ \mathbf {v} _ {j} \leftarrow \operatorname{SLERP} (\mathbf {v} _ {j}, \mathbf {z}, \eta_ {2}) \forall j \in \mathcal {N} (s _ {1}), \\ \end{array}
$$

其中 $\eta_1$, $\eta_2$ 是 learning rates。这确保了 sub-prototypes 沿 manifold surface 演化，而非漂移到 hypersphere 内部。

# 4.1.3. Dual-View Inference

在推理过程中，HC-SOINN 结合了由 class mean $\pmb { \mu } _ { c }$（来自 NCM）提供的"Global View"和由学习拓扑中最近 sub-prototype 提供的"Local View"。测试样本 x 关于类别 c 的 decision score 定义为：

$$
S (\mathbf {x}, c) = \alpha \cdot \cos \left\langle \tilde {f} _ {\theta} (\mathbf {x}), \boldsymbol {\mu} _ {c} \right\rangle + (1 - \alpha) \cdot \max _ {\mathbf {v} \in \mathcal {V} _ {c}} \cos \left\langle \tilde {f} _ {\theta} (\mathbf {x}), \mathbf {v} \right\rangle , \tag {7}
$$

其中 $\alpha \in [ 0 , 1 ]$ 平衡 global robustness 和 local precision。

Algorithm 1 HC-SOINN Construction and Inference
Input: Training stream $\mathcal{D}$, Feature Extractor $f_{\theta}$, Balance factor $\alpha$
Output: Class Topologies $\{\mathcal{G}_c\}_{c \in \mathcal{C}}$ and Global Means $\{\boldsymbol{\mu}_c\}_{c \in \mathcal{C}}$
1: Training Phase:
2: for 当前任务中的每个类别 c do
3: Accumulate features $\mathbf{Z}_c = \{\tilde{f}_{\theta}(\mathbf{x})\}$
4: Update global mean: $\boldsymbol{\mu}_c \leftarrow \text{Normalize}(\text{Mean}(\mathbf{Z}_c))$
5: // Phase 1: Hierarchical Initialization
6: $\mathcal{V}_c^{(0)} \leftarrow \text{HierarchicalClustering}(\mathbf{Z}_c, \text{metric} = \text{'cosine'})$
7: // Phase 2: Spherical SOINN Refinement
8: Initialize $\mathcal{V}_c \leftarrow \mathcal{V}_c^{(0)}, \mathcal{E}_c \leftarrow \emptyset$
9: for iter = 1 to $T_{soinn}$ do
10: for $\mathbf{z} \in \mathbf{Z}_c$ do
11: Find winner $s_1$ and runner-up $s_2$
12: Create or refresh edge ($s_1, s_2$) in $\mathcal{E}_c$
13: Update $\mathbf{v}_{s_1}$ and neighbors via SLERP($\cdot, \mathbf{z}, \eta$)
14: Remove edges with age > $age_{max}$
15: end for
16: Remove isolated nodes from $\mathcal{V}_c$
17: end for
18: end for
19: Inference Phase:
20: for test sample $\mathbf{x}$ do
21: $\hat{y} = \underset{c}{\text{argmax}}\left(\alpha \tilde{f}_{\theta}(\mathbf{x})^\top \boldsymbol{\mu}_c + (1 - \alpha) \underset{\mathbf{v} \in \mathcal{V}_c}{\text{max}} \tilde{f}_{\theta}(\mathbf{x})^\top \mathbf{v}\right)$
22: end for

完整的训练和推理过程总结在 Algorithm 1 中，详细的实现和参数设置在 Section G 中提供。

这种 dual-view inference 的基本原理是修正 decision boundary。全局项 $\cos \langle \tilde { f } _ { \theta } ( \mathbf { x } ) , \mu _ { c } \rangle$ 作为稳定的 baseline，确保样本与 class centroid 大致对齐。局部项 $\mathrm { m a x } _ { \mathbf { v } \in \mathcal { V } _ { c } } \cos \langle \tilde { f } _ { \theta } ( \mathbf { x } ) , \mathbf { v } \rangle$ 作为 topological refinement，如果样本落入类的特定 high-density region（local manifold），即使远离 global mean，也能赋予更高的置信度。选择具有最高 composite score 的类作为最终预测：

$$
\hat {y} = \underset {c \in \mathcal {C} _ {t}} {\operatorname{argmax}} S (\mathbf {x}, c). \tag {8}
$$

# 4.2. STAR: Structure-Topology Alignment via Residuals

为解决 Section 3.3 揭示的非线性 feature drift，我们提出了 STAR 模块。与传统假设 global rigid transformation 的方法不同，STAR 采用细粒度的 Pointwise Trajectory Tracking 策略。它利用 HC-SOINN 的多节点粒度独立追踪和纠正每个 topological sub-prototype 的 drift，使 manifold 结构能够主动变形并适应复杂的 feature space 演化。

对于每个已学习的类别 $c$，我们建立其拓扑节点集 $\mathcal { V } _ { c } = \{ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { K _ { c } } \}$ 与 anchor samples 集 $\mathcal { A } _ { c } = \{ ( \mathbf { x } _ { i } , \mathbf { h } _ { i } ^ { ( r e f ) } ) \} _ { i = 1 } ^ { K _ { c } }$ 之间的一一映射，其中 $\mathbf { x } _ { i }$ 是与 $\mathbf { v } _ { i }$ 关联的 anchor。在任务 $\mathcal { T } _ { t }$ 的增量学习之后，我们通过比较当前特征 $\mathbf { h } _ { i } ^ { ( t ) } = f _ { \theta _ { t } } ( \mathbf { x } _ { i } )$ 与存储的 reference $\mathbf { h } _ { i } ^ { ( r e f ) }$ 来计算每个节点的 pointwise drift。

为减轻 stochastic optimization steps 引起的不稳定性，我们采用 Exponential Moving Average (EMA) 来平滑 drift trajectory。设 $\delta _ { i }$ 表示第 i 个节点的追踪 drift vector。更新规则定义为：

$$
\boldsymbol {\Delta} _ {i} = \mathbf {h} _ {i} ^ {(t)} - \mathbf {h} _ {i} ^ {(r e f)}, \quad \boldsymbol {\delta} _ {i} \leftarrow (1 - \lambda) \boldsymbol {\delta} _ {i} + \lambda \boldsymbol {\Delta} _ {i}, \tag {9}
$$

其中 $\lambda \in ( 0 , 1 ]$ 是 momentum coefficient。然后，该 smoothed residual $\delta _ { i }$ 被应用于在当前 feature space 中传输相应的 unnormalized sub-prototype $\mathbf { v } _ { r a w , i }$：

$$
\mathbf {v} _ {\text { raw }, i} ^ {\prime} = \mathbf {v} _ {\text { raw }, i} + \boldsymbol {\delta} _ {i}. \tag {10}
$$

对齐后，节点被 re-normalized 到 unit hypersphere。类 global mean $\pmb { \mu } _ { c }$ 使用所有节点的平均 drift 同步更新：$\pmb { \mu } _ { c } ^ { \prime } = \pmb { \mu } _ { c } + \frac { \bar { 1 } } { K _ { c } } \sum _ { i } \delta _ { i }$。这一机制确保拓扑忠实地跟随 feature distribution 的非线性迁移，而无需昂贵的 retraining。STAR 的完整算法总结在 Section C 的 Algorithm 2 中。

我们框架的总体 pipeline 如图 2 所示。

# 5. Performance Experiments

# 5.1. Experimental Setup

Datasets and Metrics. 我们在 Class-Incremental Learning (CIL) 中广泛使用的三个 benchmark 上评估我们提出的方法：Split CIFAR-100（通用物体）、Split CUB-200（细粒度）和 Split ImageNet-R（out-of-distribution robustness）(Krizhevsky et al., 2009; Wah et al., 2011; Hendrycks et al., 2021)。对于所有数据集，我们遵循标准的 class-incremental setting，即在推理过程中不提供 task identity。详细的数据集统计和划分协议在 Section G 中提供。

遵循标准文献 (Zhou et al., 2024a)，我们报告两个关键指标来评估性能：

Average Accuracy $( A _ { A v g } )$：每个增量任务后获得的分类准确率的均值，定义为 $\begin{array} { r } { A _ { A v g } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathcal { A } _ { t } } \end{array}$，其中 $\boldsymbol { A } _ { t }$ 是任务 t 后的测试准确率。

Last-Task Accuracy $( A _ { L a s t } )$：在最后一个任务 T 完成后所有类别的准确率，即 $A _ { L a s t } = \mathcal { A } _ { T }$。

Table 1. Split CIFAR-100、Split CUB-200 和 Split ImageNet-R 上的性能比较。我们报告 Average Accuracy $( A _ { A v g } )$ 和 Last-Task Accuracy $( A _ { L a s t } )$。"+ HC-SOINN"表示我们的方法。Bold 表示每列的最佳结果，Red 表示第二佳结果。括号中的值表示与 base method 相比的性能 gain/loss。所有模型采用 ViT-B/16-IN1K 作为 backbone。

（表格1内容保持不变，包含各方法在三个 benchmark 上的 $A_{Avg}$ 和 $A_{Last}$ 数据）

# 5.2. Implementation Details

为了证明我们方法的通用性，我们将 HC-SOINN 集成到七种代表性的 PTM-based CIL 方法中，涵盖多样化的范式，如 prompt tuning (DualPrompt, CODA-Prompt)、adapter tuning (SEMA)、LoRA (CL-LoRA) 和 representation learning (SimpleCIL, APER, EASE) (Wang et al., 2022b; Smith et al., 2023; Wang et al., 2025; He et al., 2025; Zhou et al., 2025; 2024b)。

所有实验使用 LAMDA-PILOT framework (Sun et al., 2025) 进行，采用在 ImageNet-1K 上预训练的标准 ViT-B/16 backbone。我们严格遵循该框架的 alignment principles 以确保公平比较。在我们的方法中，我们简单地将 baseline 的原始分类器（如 FC layer 或 NCM）替换为我们的 HC-SOINN 分类器，保持 backbone 训练策略不变。具体的 hyper-parameter 设置和每个 baseline 的详细描述请参见 Section G。我们在所有 benchmark 上保持一组统一的 hyperparameters，以证明我们拓扑方法的 tuning-free robustness（sensitivity analysis 见 Section H）。

# 5.3. Main Results

三个 benchmark 上的定量比较总结在 Table 1 中。显然，HC-SOINN 的集成在所有三个 benchmark 和七个 baseline 架构上都产生了一致且通常显著的性能提升。特别值得注意的是，像 CODA-Prompt 和 SEMA 这样的方法原本依赖参数化的 Fully Connected (FC) layers，这使它们在细粒度（CUB-200）和长序列（ImageNet-R）任务中容易遭受严重的 catastrophic forgetting。观察到的显著提升（例如，SEMA 在 CUB-200 上 gain +21.00%）表明，用我们的 non-parametric、topology-aware 分类器替换 FC layer 有效地减轻了这种 forgetting。

此外，HC-SOINN 显著重振了已有方法，使它们能够媲美甚至超越最新的 SOTA baselines（SEMA 和 CL-LoRA）。例如，在 Split CIFAR-100 上，representation-based 方法 EASE 在配备 HC-SOINN 后，达到了 87.86% 的 $A _ { L a s t }$，超过了标准 SEMA（86.04%），并与 CL-LoRA（87.99%）持平。类似地，在 Split CUB-200 上，CODA-Prompt + HC-SOINN（85.75%）显著超越了 SEMA 和 CL-LoRA 的 baseline 性能。

最终，通过为 SEMA 和 CL-LoRA（CVPR 2025 的 SOTA 方法）配备 HC-SOINN，我们成功地进一步推动了性能边界，在评估的数据集上建立了新的 SOTA 性能。

Table 2. 三个 benchmark 上的分类器性能比较。我们固定 backbone 和 prompt tuning 机制（DualPrompt 或 CODA-Prompt），仅变化 classification head。"+ STAR"表示集成我们的 drift alignment 模块。Bold 表示每个 backbone 设置内的最佳结果，Red 表示第二佳结果。括号中的值表示与不带 STAR 的 HC-SOINN 相比的 gain/loss。所有模型采用 ViT-B/16-IN1K 作为 backbone。

（表格2内容保持不变，包含不同分类器在双 backbone 设置下的 $A_{Avg}$ 和 $A_{Last}$ 数据）

# 5.4. Classifier Performance Analysis

为了将分类器的贡献与 representation learning 隔离开来，我们采用 DualPrompt 和 CODA-Prompt 作为固定的 feature extractors，并在 frozen features 上评估了四种 classifier paradigms：

Original (FC): 使用 cross-entropy loss 训练的标准线性分类器。

NCM: Nearest Class Mean 分类器，代表从 NC 理论导出的 single-prototype 范式。

KAC: Kolmogorov-Arnold Classifier (Hu et al., 2025)，CVPR 2025 中提出的非线性分类器。

HC-SOINN (Ours): 我们提出的拓扑分类器，评估时包含和不包含 STAR alignment 模块。

Table 2 揭示了三个关键洞察。首先，parametric FC classifiers 在长序列任务中遭受 catastrophic forgetting，而 prototype-based 方法保持优越的稳定性。其次，虽然非线性 KAC 表现出有前途的 plasticity，但它是不稳定的；在像 Split CIFAR-100 配合 DualPrompt 这样的场景中，它的表现不如 FC baseline，且其在 out-of-distribution tasks 上的泛化能力有限。第三，STAR 从根本上提升了鲁棒性。通过使拓扑结构主动追踪非线性 feature drift，HC-SOINN + STAR 建立了决定性的领先优势。例如，在 Split ImageNet-R 上配合 DualPrompt，我们的框架达到了 69.75% 的 $A _ { L a s t }$，超过 KAC（63.85%）5.90%。这表明我们的 adaptive topological alignment 为复杂的增量 drift 提供了比 KAC 的 fixed non-linear boundaries 显著更可靠的解决方案。

# 6. Discussion & Analysis

# 6.1. Computational Efficiency Analysis

我们在 Split CIFAR-100 benchmark 上（在最终的 Task 10）使用 CODA-Prompt 作为 baseline 分析计算成本。我们还纳入 KAC 分类器进行比较。Table 3 总结了结果。

Training Efficiency. 我们的框架表现出极具竞争力的训练效率。虽然 KAC 方法带来了明显的计算负担，将训练时间增加了 10.24%，但我们的完整方法（STAR）仅增加了约 3.96% 的最小开销（从 906.57s 到 942.43s）。这一效率通过 class-wise parallelization 策略实现，每个类别的独立 topological refinement 被并发执行以最大化 CPU 利用率。

Inference Efficiency. 值得注意的是，我们的方法引入了几乎为零的推理延迟开销（+0.01%）。虽然模型计算到多个 topological sub-prototypes 的距离以捕捉复杂 manifold——这一过程通常会减慢推理——我们通过采用 GPU-accelerated matrix operations 来 vectorize distance computations，有效地消除了这一瓶颈。因此，我们的推理速度严格与 baseline 持平（26.84s），与 KAC（+0.00%）的效率相当，同时提供更鲁棒的拓扑。

Decomposition of Inference Latency. 为了进一步理解推理效率，我们将总延迟分解为 backbone processing（即通过 ViT backbone 和 prompt modules 进行 feature extraction）和 classifier evaluation，如 Table 4 所示。结果显示 backbone network 压倒性地主导了计算成本，在所有方法中一致占总推理时间的 99.7% 以上。相比之下，classification head 上花费的时间非常小，我们的完整 STAR 方法峰值为仅 0.26%（0.0686s）。这一分解确认了整体推理瓶颈本质上受限于深度 backbone architecture 而非分类器。因此，classification head 的推理速度对整体系统延迟影响可忽略不计，表明我们的 topological refinement 策略实现了显著的性能增益而没有带来实际的延迟损失。

Table 3. CODA-Prompt 在 Split CIFAR-100（Task 10）上的计算成本分析。我们使用 Intel Core i9-13900K CPU 和一块 NVIDIA RTX 4090 GPU 进行实验。所有结果取三次独立运行的平均值。括号中的百分比表示 overhead ratio，计算方式为 (Ours − Baseline)/Baseline。

（表格3内容保持不变，包含各方法的 Training Time 和 Inference Time 数据）

Table 4. 推理延迟的详细分解。总推理时间被分解为 backbone processing（包括 prompt mechanics）和 classifier evaluation。Backbone 在所有方法中一致占总计算成本的 99.70% 以上。

（表格4内容保持不变，包含各方法 Total Time、Backbone 和 Classifier 各自的 Time 和 Ratio 数据）

# 6.2. Additional Topological Visualization

为了提供 feature drift 和我们框架适应能力的全面定性分析，我们在 Figure 3 中展示了扩展的 t-SNE 可视化。我们特别追踪了初始 10 个类别（在 Task 1 中引入）在增量学习不同阶段的结构演化。

Initial Topological Fidelity. 如 Figure 3(a) 所示，在 base session（Task 1）结束时，所提出的 HC-SOINN 分类器（以大圆表示）成功地捕捉了 class manifolds 的复杂拓扑。与仅限于几何中心的 single-prototype NCM（以'×'标记）不同，我们的拓扑节点跨越了 query samples（小点）的整个分布。这验证了我们方法的基本前提："local-to-global"的 graph representation 在建模高维 feature spaces 方面更优越。

Table 5. 所提出组件在 Split CUB-200 上的 ablation study。"HC"表示 Hierarchical Clustering。"SOINN"表示 SOINN-based topological refinement。"STAR"表示 drift alignment 模块。

（表格5内容保持不变，包含不同配置下的消融实验结果）

The Challenge of Feature Drift. Figure 3(b) 说明了五个后续增量步骤后 feature drift 的严重影响。随着 backbone 为新任务更新，旧类的 feature embeddings 发生非线性偏移。标准的 HC-SOINN 节点由于缺乏主动 alignment 机制，无法跟上这种演化。可以看到明显的 spatial mismatch，拓扑节点落后于实际样本簇，留下大量 peripheral samples 未被覆盖。这可视化了"topological lag"现象，解释了为什么静态拓扑方法随时间性能下降。

Effectiveness of STAR Alignment. 相比之下，Figure 3(c) 展示了 STAR 模块在相同 drift 条件下的效力。通过利用存储的 anchor drift 来估计 pointwise trajectories，STAR 主动变形拓扑结构。Figure 3(c) 中对齐的节点重新建立了对 drifted manifolds 的精确覆盖，有效地匹配了新的分布。这一比较提供了令人信服的证据，表明从"drift resistance"转向"drift adaptation"对于在 Class-Incremental Learning 中维持长期 topological fidelity 至关重要。

Figure 3. 初始 10 个类别的 feature distributions 的 t-SNE 可视化。小点表示 query samples，大圆表示 HC-SOINN topological nodes，'×'标记表示 NCM prototypes。(a) 在 Task 1 时，HC-SOINN 节点完美地捕捉了初始 class manifolds。(b) 到 Task 6 时，没有主动 alignment，标准 HC-SOINN 节点无法覆盖 drifted features（由 spatial mismatch 突出显示）。(c) 启用 STAR 后，topological nodes 在 Task 6 被主动变形以匹配 drifted distribution，恢复了 representational fidelity。

# 6.3. Ablation Study

为严格验证每个组件的贡献，我们在 Split CUB-200 上使用 CODA-Prompt 进行了全面的 ablation study。我们评估了 Hierarchical Clustering (HC)、SOINN refinement 和 STAR alignment 模块的组合。结果详见表 Table 5。

Effectiveness of Topological Initialization and Refinement. 与 single-prototype Baseline（NCM，80.74%）相比，Pure SOINN（86.71%）和 Pure HC（88.20%）都取得了显著的性能提升，确认了拓扑表示的有效性。值得注意的是，这两个组件的组合（HC-SOINN）进一步提升至 89.43%，超越了两个单独的模块。这种协同效应源于它们的互补性：Hierarchical Clustering 提供了鲁棒的 global initialization，减轻了 SOINN 在 early learning phase 对 outliers 的敏感性，而 SOINN 的 incremental learning 机制比 static clustering 具有更强的表示细粒度 non-linear manifold features 的能力。

Orthogonality of STAR Alignment. 我们在不同拓扑设置下评估 STAR 模块以验证其鲁棒性。将 STAR 添加到 Pure HC（HC+STAR）将性能提升至 89.39%，证明 STAR 即使使用 coarse cluster centroids 也能有效对齐分布。此外，我们的完整方法（Ours），将 STAR 应用于 refined HC-SOINN topology，达到了 90.37% $( A _ { a v g } )$ 和 86.17% $( A _ { l a s t } )$ 的峰值性能。两种设置中观察到的一致增益表明，Drift Adaptation（通过 STAR）和 Topological Refinement（通过 HC-SOINN）是正交且互补的机制——一个修复 spatial misalignment，而另一个优化 decision boundary 结构。

# 7. Conclusion

在本文中，我们重新审视了 Nearest Class Mean (NCM) 分类器在 Class-Incremental Learning 背景下的理论最优性。我们发现由于不完全的 Neural Collapse，实际 CIL 场景中的类别表示表现为复杂 manifold 而非 collapsed points，使得 single-prototype 分类器次优。为了弥补这一差距，我们提出了 HC-SOINN，一种新颖的 topology-aware 分类器，通过 coarse-to-fine learning 机制捕捉 class manifolds 的内在几何特性。此外，我们引入了 STAR，一个 residual-based alignment 模块，将范式从"drift resistance"转变为"drift adaptation"。通过利用细粒度的 pointwise trajectory tracking，STAR 使拓扑结构能够主动变形，精确适应非线性 feature evolution。在三个多样化 benchmark 上的大量实验表明，我们的拓扑框架一致优于传统的基于 NCM 的方法。未来的工作将探索将这种拓扑表示扩展到 multi-modal continual learning 场景，以进一步验证其泛化能力。Limitation. 我们承认性能增益伴随着权衡。与轻量级 NCM 相比，我们的框架由于 topological maintenance 而引入了适度的计算开销，并需要为 STAR 的 anchor buffer 提供额外存储。此外，为保持严格的推理效率，我们当前的 scoring mechanism 仅依赖 node distances，在分类阶段未利用丰富的 edge connectivity。

# Acknowledgements

本工作部分得到 National Natural Science Foundation of China (Grant Nos. 62495090, 62495094 and 62276127)、Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118) 以及 "111 Center" (No. B26023) 的支持。

# Impact Statement

在本文中，我们介绍了 HC-SOINN 和 STAR，一个旨在解决 Class-Incremental Learning (CIL) 中非线性 feature drift 的框架。Broader Applications and Benefits: 我们的研究推进了 AI 系统从 streaming data 中持续学习的能力，这对于 autonomous robotics、personalized assistants 和 dynamic medical diagnosis systems 等应用至关重要。通过使模型能够适应新任务而不遗忘先前知识，我们的方法促进了 Lifelong Learning agents 在真实世界环境中的部署。此外，与从头 retraining 模型相比，我们的方法显著降低了模型生命周期内的计算成本和能耗，为 Green AI 和 sustainable computing 的目标做出了贡献。

在提高稳定性的同时，dynamic topological models 的部署也引入了特定挑战。与 static models 不同，我们的系统（由 STAR alignment 模块驱动）主动演化其 decision boundaries。这种动态性质使 safety verification 和 interpretability 复杂化，这在 autonomous driving 等安全关键领域是一个关注点。如果 pointwise trajectory tracking 因 noisy data 而产生 misalignment，可能导致不可预测的失败。存在风险的是，初始训练任务中存在的 biases 可能被编码到拓扑结构（HC-SOINN）中，并在后续增量步骤中被传播或放大。Dependence: 随着系统在自动适应方面变得更好，存在过度依赖模型 self-correction 能力的风险，可能减少对数据质量的人类监督。

我们鼓励未来的研究探索实时验证 evolving topologies 稳定性的方法。此外，研究人员应该研究检测和纠正已学习 manifold structures 中 biases 的机制，确保"drift adaptation"过程不会无意中固化不公平的决策模式。

# References

（参考文献部分保持不变，共40条引用，涵盖 Allen 2019 至 Zhu 2024 的所有文献）

# A. Theoretical Analysis

# A.1. Proposition 1: Error Bound Analysis under Feature Drift

在此，我们提供一个理论 bound 来证明我们的 topology-aware 表示（HC-SOINN）在"Feature Drift"问题下相比 single-prototype NCM 最小化了 representation error。

设 $\mathcal { M } _ { c } \subset \mathbb { R } ^ { d }$ 表示任务 t 时类别 c 的 feature manifold。设 $\phi : \mathbb { R } ^ { d } \rightarrow \mathbb { R } ^ { d }$ 是将特征从任务 $t$ 映射到 $t + 1$ 的 drift function（由于 backbone updates）。我们将 Representation Error 形式化定义为 drifted sample 与其对应 drifted prototype representation 之间的最大偏差。

Assumption A.1 (Lipschitz Continuous Drift). Feature drift function $\phi$ 是 L-Lipschitz continuous 的，即存在常数 $L > 0$，使得对于任意 $\mathbf { x } , \mathbf { y } \in \mathcal { M } _ { c }$：

$$
\| \phi (\mathbf {x}) - \phi (\mathbf {y}) \| \leq L \| \mathbf {x} - \mathbf {y} \|. \tag {11}
$$

此假设在深度学习中较为温和，因为 gradient updates 通常是有界的（例如通过 gradient clipping 或 weight decay），防止 feature mapping 中的任意不连续性。

Analysis of NCM Classifier. NCM 分类器使用单一 global centroid $\pmb { \mu } _ { c }$ 表示 manifold $\mathcal { M } _ { c }$。NCM 在 drift 后的 representation error，记为 $\mathcal { E } _ { N C M }$，以 drifted sample 与 drifted mean 之间的最坏情况距离为界：

$$
\begin{array}{l} \mathcal {E} _ {N C M} = \max _ {\mathbf {z} \in \mathcal {M} _ {c}} \| \phi (\mathbf {z}) - \phi (\boldsymbol {\mu} _ {c}) \| \\ \leq \max _ {\mathbf {z} \in \mathcal {M} _ {c}} L \| \mathbf {z} - \boldsymbol {\mu} _ {c} \| = L \cdot R _ {c}, \tag {12} \\ \end{array}
$$

其中 $R _ { c } = \operatorname* { m a x } _ { \mathbf { z } \in \mathcal { M } _ { c } } \left\| \mathbf { z } - \pmb { \mu } _ { c } \right\|$ 表示 class manifold 的 Global Radius。在 non-collapsed 场景中，$R _ { c }$ 显著较大。

Analysis of HC-SOINN Classifier. HC-SOINN 将 manifold $\mathcal { M } _ { c }$ 划分为以 sub-prototypes $\{ \mathbf { v } _ { 1 } , \dotsc , \mathbf { v } _ { K } \}$ 为中心的 K 个 Voronoi regions $\{ \gamma _ { 1 } , \dots , \gamma _ { K } \}$。样本 z 由其最近的 sub-prototype $\mathbf { v } _ { k ^ { * } }$ 表示。HC-SOINN 的 representation error $\mathcal { E } _ { H C }$（假设拓扑在 drift 过程中得以保持）以以下为界：

$$
\begin{array}{l} \mathcal {E} _ {H C} = \max _ {k} \max _ {\mathbf {z} \in \mathcal {V} _ {k}} \| \phi (\mathbf {z}) - \phi (\mathbf {v} _ {k}) \| \\ \leq \max _ {k} \max _ {\mathbf {z} \in \mathcal {V} _ {k}} L \| \mathbf {z} - \mathbf {v} _ {k} \| = L \cdot r _ {\text {local}}, \tag {13} \\ \end{array}
$$

其中 $r _ { l o c a l } = \mathrm { m a x } _ { k } \mathrm { m a x } _ { \mathbf { z } \in \mathcal { V } _ { k } } \| \mathbf { z } - \mathbf { v } _ { k } \|$ 是 sub-clusters 的最大 Local Radius。

Conclusion & Impact on Forgetting. 由于 hierarchical clustering 和 SOINN 明确最小化 quantization error，manifold 被分解为紧致的 local regions。根据划分的定义，对于任何复杂 manifold（例如 non-convex shapes），$r _ { l o c a l } \ll R _ { c }$ 成立。因此：

$$
\mathcal {E} _ {H C} \leq L \cdot r _ {\text { local }} \ll L \cdot R _ {c} = \text { Upper Bound of } \mathcal {E} _ {N C M}. \tag {14}
$$

这一不等式从理论上保证了 HC-SOINN 在 feature drift 下维持更紧的 error bound。即使 backbone 扭曲空间（按 $L$ 缩放），multi-prototype 结构也比单一 rigid centroid 更好地保持 local neighborhood relations。关键的是，这种最小化的 representation error 直接转化为减少的 catastrophic forgetting。在 class-incremental 场景中，当 feature drift 导致已建立的 decision boundaries 发生严重的非线性扭曲时，就会发生 forgetting。通过将 feature deviation 严格限制在更小的 local radius $r _ { l o c a l }$ 内，我们的 topology-aware 方法防止了 class decision half-spaces 的剧烈漂移。因此，旧类样本保持与对应 sub-prototypes 的正确对齐，直接保持了分类精度并在无需 rehearsal 的情况下减轻了 forgetting。

# A.2. Proposition 2: Bayesian Optimality via vMF Mixture

Proposition 1 建立了对 drift 的鲁棒性，我们进一步证明推理分数 $S ( \mathbf { x } , c )$（Eq. 6）的具体形式的合理性。由于 feature vectors $\tilde { f } _ { \theta } ( \mathbf { x } )$ 被归一化到 unit hypersphere $\mathbb { S } ^ { d - 1 }$，标准 NCM 采用的 Gaussian assumption 在几何上是次优的。相反，我们采用 von Mises-Fisher (vMF) distribution，它是球面上的 maximum entropy distribution。

Definition A.2 (vMF Distribution). 随机变量 $\mathbf { x } \in \mathbb { S } ^ { d - 1 }$ 服从 vMF distribution，具有 mean direction $\pmb { \mu } \in \mathbb { S } ^ { d - 1 }$ 和 concentration parameter $\kappa \geq 0$，记为 $v M F ( \mathbf { x } ; \pmb { \mu } , \kappa )$，如果其 probability density function 为：

$$
p (\mathbf {x} | \boldsymbol {\mu}, \kappa) = C _ {d} (\kappa) \exp \left(\kappa \boldsymbol {\mu} ^ {\top} \mathbf {x}\right), \tag {15}
$$

其中 $C _ { d } ( \kappa )$ 是 normalization constant，$\mu ^ { \top }$ x 对应于 cosine similarity。

为同时捕捉 global stability（NCM）和 local topological fidelity（HC-SOINN），我们将 likelihood $P ( \mathbf { x } | c )$ 建模为 constrained hybrid distribution。具体而言，我们假设 class conditional density 正比于 Global Prior（以 $\mu _ { c }$ 为中心）和 Local Mixture（以 sub-prototypes $\nu _ { c }$ 为中心）的乘积：

$$
\begin{array}{l} P (\mathbf {x} | c) \propto P _ {\text { global }} (\mathbf {x} | c) \times P _ {\text { local }} (\mathbf {x} | c) \\ \approx v M F (\mathbf {x}; \boldsymbol {\mu} _ {c}, \kappa_ {g}) \times \max _ {\mathbf {v} \in \mathcal {V} _ {c}} v M F (\mathbf {x}; \mathbf {v}, \kappa_ {l}), \tag {16} \\ \end{array}
$$

其中 $\kappa _ { g }$ 和 $\kappa _ { l }$ 分别表示 global mean 和 local sub-prototypes 的 concentration（confidence）。max operator 近似于 mixture sum，假设局部 high-density regions 互不相交。

在 uniform class priors $P ( c )$ 的假设下，Bayes Optimal Classifier 最大化 posterior $P ( c | \mathbf { x } )$，这等价于最大化 log-likelihood $\log P ( \mathbf { x } | c )$。将 vMF density 代入 log-likelihood，我们得到：

$$
\begin{array}{l} \log P (\mathbf {x} | c) = \log \left(C _ {d} \left(\kappa_ {g}\right) e ^ {\kappa_ {g} \boldsymbol {\mu} _ {c} ^ {\top} \mathbf {x}} \cdot C _ {d} \left(\kappa_ {l}\right) e ^ {\kappa_ {l} \max _ {\mathbf {v}} \mathbf {v} ^ {\top} \mathbf {x}}\right) \\ = \kappa_ {g} \boldsymbol {\mu} _ {c} ^ {\top} \mathbf {x} + \kappa_ {l} \max _ {\mathbf {v} \in \mathcal {V} _ {c}} \mathbf {v} ^ {\top} \mathbf {x} + \underbrace {\log (C _ {d} (\kappa_ {g}) C _ {d} (\kappa_ {l}))} _ {\text {class-independent constant}}. \tag {17} \\ \end{array}
$$

由于 feature vectors 经过 L2-normalized $( \| \mathbf { x } \| = \| \pmb { \mu } \| = \| \mathbf { v } \| = 1 )$，dot product 等价于 cosine similarity：$\mathbf { a } ^ { \top } \mathbf { b } = \cos \langle \mathbf { a } , \mathbf { b } \rangle$。此外，假设各类别间的 concentration parameters 均匀，log-partition term 关于类别 c 是常数，可以在优化过程中省略。

因此，分类规则简化为最大化 cosine similarities 的加权和：

$$
\hat {y} = \underset {c} {\operatorname{argmax}} \left(\kappa_ {g} \cos \langle \mathbf {x}, \boldsymbol {\mu} _ {c} \rangle + \kappa_ {l} \underset {\mathbf {v} \in \mathcal {V} _ {c}} {\max} \cos \langle \mathbf {x}, \mathbf {v} \rangle\right). \tag {18}
$$

为使其与我们提出的 dual-view metric 一致，我们利用 argmax operator 的 scale-invariance 性质。定义 total concentration $K _ { t o t a l } = \kappa _ { g } + \kappa _ { l }$ 并将目标除以这个正常数而不改变预测：

$$
\begin{array}{l} \hat {y} = \underset {c} {\operatorname{argmax}} \frac {1}{K _ {t o t a l}} \left(\kappa_ {g} \cos \langle \mathbf {x}, \boldsymbol {\mu} _ {c} \rangle + \kappa_ {l} \underset {\mathbf {v} \in \mathcal {V} _ {c}} {\max} \cos \langle \mathbf {x}, \mathbf {v} \rangle\right) \\ = \underset {c} {\operatorname{argmax}} \left(\frac {\kappa_ {g}}{K _ {t o t a l}} \cos \langle \mathbf {x}, \boldsymbol {\mu} _ {c} \rangle + \frac {\kappa_ {l}}{K _ {t o t a l}} \underset {\mathbf {v} \in \mathcal {V} _ {c}} {\max} \cos \langle \mathbf {x}, \mathbf {v} \rangle\right). \tag {19} \\ \end{array}
$$

最后，令 balancing factor $\begin{array} { r } { \alpha = \frac { \kappa _ { g } } { \kappa _ { g } + \kappa _ { l } } } \end{array}$，这意味着 $\begin{array} { r } { 1 - \alpha = \frac { \kappa _ { l } } { \kappa _ { g } + \kappa _ { l } } } \end{array}$，我们恢复了推理公式：

$$
\hat {y} = \underset {c} {\operatorname{argmax}} \left(\alpha \cos \langle \mathbf {x}, \boldsymbol {\mu} _ {c} \rangle + (1 - \alpha) \underset {\mathbf {v} \in \mathcal {V} _ {c}} {\max} \cos \langle \mathbf {x}, \mathbf {v} \rangle\right). \tag {20}
$$

Conclusion. 这一推导证明了我们的 heuristic score $S ( \mathbf { x } , c )$ 在理论上等价于 hybrid vMF model 下的 Maximum A Posteriori (MAP) 估计。超参数 α 不是任意的；它明确表示 global confidence $\kappa _ { g }$ 与 total confidence 的比率。较高的 α 表示 global prototype 更可靠（例如在 low-drift 场景中），而较低的 α 则强调局部拓扑细节。

# B. Implementation Insights and Geometric Rationale of HC-SOINN

在本节中，我们提供 HC-SOINN 分类器更详细的阐述，重点是其设计的数学论证及其相对于传统 incremental clustering 方法的优势。

# B.1. Robustness of Hierarchical Initialization

如 Section 4.1.1 所述，我们采用 Unweighted Pair Group Method with Arithmetic Mean (UPGMA) 作为 linkage criterion 的 Agglomerative Hierarchical Clustering。两个簇 $\mathcal { C } _ { i }$ 和 $\mathcal { C } _ { j }$ 之间的距离计算如下：

$$
l (\mathcal {C} _ {i}, \mathcal {C} _ {j}) = \frac {1}{| \mathcal {C} _ {i} | | \mathcal {C} _ {j} |} \sum_ {\mathbf {u} \in \mathcal {C} _ {i}} \sum_ {\mathbf {v} \in \mathcal {C} _ {j}} d _ {c o s} (\mathbf {u}, \mathbf {v}), \tag {21}
$$

其中 $\begin{array} { r } { d _ { c o s } ( \mathbf { u } , \mathbf { v } ) = 1 - \frac { \mathbf { u } ^ { \top } \mathbf { v } } { \| \mathbf { u } \| \| \mathbf { v } \| } } \end{array}$。

Order-Independence and Noise Suppression: 与标准 SOINN 或其他逐个处理信号的 online clustering 方法不同，HC-SOINN 在每个任务结束时对整个 feature buffer 进行初始化。这种 batch-style processing 确保初始化是 order-independent 的，减轻了 manifold skeleton 被输入序列扭曲的风险。此外，通过平均所有 inter-cluster pairs 的距离，UPGMA 有效地抑制了 individual outliers 的影响，为后续 refinement 提供了稳定且全局最优的"backbone"。

# B.2. Spherical Geometry and SLERP Dynamics

现代 CIL 中的一个关键挑战是利用 normalized feature spaces（unit hyperspheres $\mathbb { S } ^ { d - 1 }$）来促进基于 cosine similarity 的分类。经典 SOINN 中使用的标准 linear interpolation，定义为 $\mathbf { v } \leftarrow \mathbf { v } + \eta ( \mathbf { z } - \mathbf { v } )$，会将节点拉入 hypersphere 内部，违反 unit-norm constraint 并扭曲表示。

SLERP for Consistency: 为保持几何完整性，我们采用 Spherical Linear Interpolation (SLERP)：

$$
\operatorname{SLERP} (\mathbf {v}, \mathbf {z}; \eta) = \frac {\sin ((1 - \eta) \Omega)}{\sin \Omega} \mathbf {v} + \frac {\sin (\eta \Omega)}{\sin \Omega} \mathbf {z}, \tag {22}
$$

其中 $\Omega = \operatorname { a r c c o s } ( \mathbf { v } ^ { \top } \mathbf { z } )$。SLERP 确保 sub-prototypes 严格沿 manifold surface 迁移。这保证了 distance metrics 在整个训练和推理阶段保持一致，防止 Competitive Hebbian Learning (CHL) 阶段的 representational collapse。

# B.3. Topology as a Non-Linear Manifold Proxy

HC-SOINN 学习的 graph structure $\mathcal { G } _ { c } = ( \nu _ { c } , \mathcal { E } _ { c } )$ 作为底层 class manifold 的 piecewise-linear 近似。

- Local Connectivity: 连接 winner 和 runner-up nodes（Top-2 nodes）本质上定义了一个 Voronoi-like neighborhood structure，捕捉数据的局部密度和形状。
- Co-evolution: 当 winner node $s _ { 1 }$ 及其 topological neighbors $\mathcal { N } ( s _ { 1 } )$ 通过 SLERP 更新时，整个 local patch of the manifold "协同演化"朝向新的数据分布。
- Refining Decision Boundaries: 在我们的 Dual-View inference 中，Local Path score $\mathrm { m a x } _ { \mathbf { v } \in \mathcal { V } _ { c } } \cos ( \tilde { f } _ { \theta } ( \mathbf { x } ) , \mathbf { v } )$ 提供了细粒度的 membership test。与假设 convex, unimodal distribution 的 single-prototype NCM 不同，topology-aware grid 可以捕捉复杂的 non-convex shapes（例如"crescent"或"dumbbell"分布）。这使分类器能够为落入类实际 high-density regions 的样本赋予高置信度，即使它们远离 global centroid $\pmb { \mu } _ { c }$。

# C. The Algorithm of STAR

STAR 的完整算法总结在 Algorithm 2 中。

Algorithm 2 STAR Trajectory Alignment
Input: Old Classes $C_{old}$, Anchors A, Current Backbone $f_{\theta_i}$, HC-SOINN Classifier $\Psi$, Momentum $\lambda$
Output: Aligned Class Topologies

1: for each old class $c \in C_{old}$ do
2: $\Delta_{list} \leftarrow []$
3: for each node i in $\Psi.\text{get\_nodes}(c)$ do
4: Retrieve anchor ($x_i, h_i^{(ref)}$) and stored drift $\delta_i$
5: Extract current feature: $\mathbf{h}_i^{(t)} \leftarrow f_{\theta_t}(\mathbf{x}_i)$
6: Compute instant drift: $\Delta_i \leftarrow \mathbf{h}_i^{(t)} - \mathbf{h}_i^{(ref)}$
7: Update smoothed trajectory: $\delta_i \leftarrow (1 - \lambda)\delta_i + \lambda\Delta_i$
8: // Pointwise Transport & Re-normalization
9: $v'_{raw,i} \leftarrow v_{raw,i} + \delta_i$
10: $v'_i \leftarrow v'_{raw,i}/\|v'_{raw,i}\|$
11: $\Psi.\text{update\_node}(c, i, v'_{raw,i}, v'_i)$
12: Update Reference: $h_i^{(ref)} \leftarrow h_i^{(t)}$
13: Append $\delta_i$ to $\Delta_{list}$
14: end for
15: // Synchronize Global Mean
16: $\mu'_c \leftarrow \mu_c + \text{Mean}(\Delta_{list})$
17: $\Psi.\text{update\_global\_mean}(c, \text{Normalize}(\mu'_c))$
18: end for

# D. Topological Complexity Analysis

我们的 HC-SOINN 分类器的一个关键特性是它能够自适应地确定表示每个类别所需的 topological nodes 数量，而非使用固定的预算。为了研究我们框架学习的 manifold 的复杂性，我们报告了在三个 benchmark 上七种不同 backbone 下每个类别的平均节点数量。结果总结在 Table 6 中。

Adaptive yet Compact Representation. 如 Table 6 所示，节点数量在不同 feature extractors 之间表现出显著的稳定性（例如 SimpleCIL vs. CODA-Prompt），一致收敛到紧致的范围。对于标准数据集如 Split CIFAR-100 和 CUB-200，拓扑稳定在每个类别约 18∼19 个节点。对于由于多样化艺术风格而包含显著 intra-class variance 的 Split ImageNet-R，模型自适应地分配略多的资源（约 21 个节点）。

Efficiency of Manifold Approximation. 这些统计数据表明了我们"local-to-global"学习机制的效率。通过平均约 20 个 sub-prototypes，HC-SOINN 成功地捕捉了 class manifolds 的复杂非线性几何。这个数量代表了一个"sweet spot"——它比 NCM 的 single prototype 丰富得多，同时又保持足够稀疏以避免存储所有 training samples 所带来的高计算和存储负担。

Table 6. HC-SOINN 在不同方法和数据集上生成的每类平均 topological nodes 数量。

（表格6内容保持不变，包含各 base method 在三个数据集上的节点数数据）

Table 7. 使用 ViT-B/16-IN21K backbone 与 rehearsal-based 方法的比较。"Exemplars"表示每类存储图像的平均数量。我们的方法（SEMA + Ours）在与标准 20-image fixed buffer 相当或更低的存储预算下达到了 state-of-the-art 性能。Baselines 的结果来源于 (Zhou et al., 2024b)。

（表格7内容保持不变，包含各方法在 ImageNet-R 和 CIFAR-100 上的对比数据）

# E. Comparison with Traditional Exemplar-based Methods

为严格评估我们 exemplar 使用的有效性，我们将我们的框架与经典的 rehearsal-based CIL 方法进行 benchmark 比较，包括 iCaRL (Rebuffi et al., 2017)、DER (Yan et al., 2021)、FOSTER (Wang et al., 2022a) 和 MEMO (Zhou et al., 2022)。我们严格遵循 (Zhou et al., 2024b) 中的任务设置。我们采用 ViT-B/16-IN21K backbone 用于所有方法以确保公平比较，将我们的 HC-SOINN 和 STAR 模块集成到 SEMA (Wang et al., 2025) baseline 中。Split ImageNet-R（10 tasks，每个 20 classes）和 Split CIFAR-100（10 tasks，每个 10 classes）上的结果总结在 Table 7 中。

如结果所示，将我们的方法集成到 SEMA 中一致地优于所有比较的 rehearsal-based 方法。在 Split CIFAR-100 上，我们的方法达到了 94.25% 的 Average Accuracy，显著超过了最强的 rehearsal baseline FOSTER（89.87%）。类似地，在具有挑战性的 Split ImageNet-R 上，我们保持了性能优势（81.70% vs. 81.34%）。关键的是，这种优越的性能是在更低的存储预算下实现的。传统方法依赖每类 20 个 exemplars 的 fixed buffer，而我们的 adaptive topology 平均每个类别仅生成 18.48 和 18.87 个节点（分别为 CIFAR-100 和 ImageNet-R）。此外，存储样本的利用方式有一个根本区别：rehearsal 方法需要计算昂贵的 gradient-based retraining 来减轻 forgetting，而 STAR 将这些样本专门用于 inference-time spatial alignment，无需 backward propagation。这意味着我们的 drift adaptation 机制在理论上与 gradient-based replay 是正交的。未来的工作将探索将 STAR 的 alignment 与 replay buffers 相结合，以实现对存储 exemplars 的更全面利用，可能解锁进一步的性能增益。

# F. Integrating Original Classification Head Information

在我们的主要实验中，原始 parameterized classification head（即 Fully Connected 或 FC layer）被我们的 topology-aware HC-SOINN 分类器替换。这一设计选择主要是由于标准 FC layers 在 Class-Incremental Learning (CIL) 场景下容易受到严重的 task-recency bias 的影响。然而，很自然地会问原始 head 是否仍然保留了可以利用的 complementary discriminative information。

为探索整合这两种范式的边界，我们在推理过程中引入了 dynamic score fusion 机制：

$$
\text { Final Score } = (1 - w) \times \mathcal {S} _ {\mathrm{HC-SOINN}} + w \times \mathcal {S} _ {\text { Calibrated FC }}, \tag {23}
$$

其中 $w \in [ 0 , 1 ]$ 是分配给原始 classification head 的权重。我们使用 DualPrompt + HC-SOINN backbone 在三个 benchmark 上评估此策略，设置 $w \in \{ 0 , 0.3 , 0.5 \}$。结果总结在 Table 8 中。

经验结果揭示了明确的 trade-off：

- Moderate fusion is beneficial $( w = 0.3 )$: 为 FC layer 分配保守权重在大多数数据集上产生轻微的性能提升（例如，Split CIFAR-100 上的 $A _ { l a s t }$ 从 85.63% 增加到 86.14%）。这证实了 calibrated FC layer 仍然包含可以补充我们 topological geometric features 的辅助知识。

- Over-reliance is detrimental $( w = 0.5 )$: 当 FC layer 的影响进一步增加时，整体准确率明显下降（例如，Split CUB-200 上的 $A _ { l a s t }$ 下降到 85.24%，低于 baseline）。这验证了我们的初始担忧：FC layer 的过度参与重新引入了 catastrophic forgetting 和 task bias，从而掩盖了 topology-aware architecture 固有的抗遗忘优势。

总之，虽然适度整合 FC layer 的知识可能略微有益，但最优权重高度敏感。纯 HC-SOINN 表示（w = 0）仍然是持续评估的一个高度鲁棒、更简单且无偏见的默认选择。

Table 8. HC-SOINN 与原始 FC classifier 之间 dynamic score fusion 的性能比较。w = 0 表示我们默认的不使用 FC fusion 的设置。

（表格8内容保持不变，包含不同 w 值下的 $A_{avg}$ 和 $A_{last}$ 数据）

# G. Detailed Implementation and Parameter Settings

# G.1. Datasets and Benchmarks

我们在三个广泛采用的 Class-Incremental Learning (CIL) benchmark 上评估我们的框架：Split CIFAR-100、Split CUB-200 和 Split ImageNet-R。这些 benchmark 涵盖了多样化的视觉识别场景，包括 coarse-grained natural images、fine-grained categorization 和 domain-shifted artistic renditions，能够全面评估 incremental learning 性能。

所有实验遵循标准 CIL protocol，即类别按顺序在一系列 disjoint tasks 中引入。在训练期间，模型仅能访问当前任务的数据，而在评估期间需要对迄今为止遇到的所有类别的样本进行分类。

Split CIFAR-100: 包含 100 个自然图像类别，共 60,000 张分辨率为 32×32 的图像。按照惯例，数据集被划分为 10 个增量任务，每个任务包含 10 个类别。此 benchmark 被广泛用作评估有限分辨率设置下 class-incremental learning 方法的标准测试平台。

Split CUB-200: 一个 fine-grained visual classification 数据集，包含 200 种鸟类和 11,788 张图像。我们将数据集划分为 10 个任务，每个任务引入 20 个新类别。由于高 inter-class similarity 和细微的判别线索，Split CUB-200 对在 class-incremental learning 中维持判别性表示提出了重大挑战。

Split ImageNet-R: 包含 200 个类别约 30,000 张图像，其中图像是 artistic renditions，如 cartoons、sketches、graffiti 和 paintings。数据集被划分为 40 个任务，每个任务 5 个类别。与自然图像的严重 domain shift 使得 ImageNet-R 特别适合评估基于 pre-trained models 的 incremental learning 方法的鲁棒性和泛化能力。

# G.2. Baselines

我们将我们的方法与一套全面的 state-of-the-art class-incremental learning baselines 进行比较。这些方法可以大致分为 representation-based 方法和 parameter-efficient fine-tuning (PEFT) 方法。后者进一步包括 prompt-based 和 adapter-based 技术，它们对基于 transformer 的 pre-trained models 特别有效。

SimpleCIL 是一个强大的 baseline，冻结 pre-trained backbone 并使用 class prototypes 增量更新分类器，表明仅用良好泛化的 pre-trained representations 就能在 class-incremental learning 中取得竞争性能。

DualPrompt 引入跨任务共享的 global prompts 和专用于各个任务的 task-specific prompts，在 class-incremental learning 中实现知识共享和任务判别之间的平衡。

CODA-Prompt 通过使用 attention mechanisms 动态组合和选择 prompts 进一步增强 prompt learning，提高 prompt diversity 并减少长任务序列中的 task interference。

APER 认为，通过将高度泛化的 pre-trained features 与 adaptive classifier updates 相结合，可以有效解决 class-incremental learning。APER 避免 rehearsal 和 parameter expansion，转而依赖 adaptive prototype estimation 来平衡增量任务中的 stability 和 plasticity。

EASE 通过增量扩展 task-specific subspaces 来解决 representation drift 和 class interference。它构建一个 expandable feature subspaces 的 ensemble，以更好地容纳新引入的类别同时保留先前学习的知识。

SEMA 提出了一个 mixture-of-adapters 框架，允许模型动态扩展其容量。通过选择性激活和组合多个 adapters，SEMA 增强了模型表现力同时减轻 catastrophic forgetting。

CL-LoRA 采用 low-rank adaptation modules 来增量微调 pre-trained models。它通过引入 task-wise low-rank updates 同时冻结 backbone parameters 来实现 rehearsal-free class-incremental learning。

我们还在主论文中比较了一些 exemplar-based 方法：

iCaRL 将 nearest-class-mean classifier 与 herding-based exemplar selection 策略相结合。它利用 knowledge distillation 在学习新类时减轻 catastrophic forgetting。

DER 采用 dynamic architecture，为每个新任务扩展 feature extractor。它集成来自 frozen past extractors 和当前 learnable extractor 的特征，以保留旧知识同时获取新概念。

FOSTER 采用涉及 feature boosting and compression 的两阶段学习范式。它首先动态扩展模型以适应新任务 residuals，随后通过 distillation 压缩扩展参数以保持模型 compactness。

MEMO 作为 rehearsal-based baseline，利用存储的 exemplars 来维护先前类别的 decision boundaries，通过 memory-efficient optimization 平衡 stability-plasticity trade-off。

# G.3. Implementation Details

Backbone and Training. 遵循标准协议，我们采用在 ImageNet-1K (IN1K) 上预训练的 Vision Transformer (ViT-B/16) 作为所有实验的 feature extractor，除非在 Section E 中另有说明。在增量阶段 backbone parameters 保持冻结以防止 catastrophic forgetting，仅更新 PEFT modules（如适用）和分类器。

HC-SOINN Hyperparameters. 我们的拓扑分类器 HC-SOINN 配置如下 hyperparameters 以在 stability 和 plasticity 之间确保平衡：

Balance Factor (α): 设置为 0.5。此等值权重控制在推理时 global class center 与 local sub-prototypes 之间的贡献（Eq. 7）。

Target Cluster Count $( K _ { i n i t } )$: 设置为 60。此参数确定初始 hierarchical clustering 的粒度。

Max Edge Age $( a g e _ { m a x } )$: 设置为 20。SOINN graph 中 20 次迭代未被刷新的 edges 将被移除以修剪过时的 topological connections。

SOINN Iterations $( T _ { s o i n n } )$: 设置为 1。我们每个任务执行单次 refinement pass 以保持计算效率。

STAR Momentum (λ): 设置为 0.999。此系数控制在更新 pointwise drift vectors 时的 EMA，在 topological alignment 期间平滑 trajectory 以减轻由 stochastic optimization steps 引起的不稳定性。

除非另有说明，所有实验使用上述相同的 hyperparameter 设置进行。在这组单一 hyperparameter 设置下，我们的方法在所有评估中取得了强劲性能，证明了其鲁棒性。

所有实验在 NVIDIA GPUs 上使用 PyTorch 进行。对于 baseline 方法，我们使用其官方实现和推荐设置以确保公平比较。

# H. Comprehensive Hyperparameter Sensitivity and Robustness

为全面评估模型对 hyperparameters 的灵敏度，我们在 Split CIFAR-100（与 CODA-Prompt 集成）和 Split ImageNet-R（与 SimpleCIL 和 DualPrompt 集成）上进行了广泛分析。详细结果如 Figure 4 和 Figure 5 所示。

Detailed Parameter Analysis. 从广泛评估中可以清楚观察到，我们的框架在极其广泛的参数值范围内保持了高度的性能稳定性：

- Balance Factor (α): 如 Figure 4(a) 和 Figure 5(a) 所示，α 揭示了"inverted-U"性能趋势。两个极端——纯 NCM $(\alpha = 1.0，在 CIFAR-100 上 90.55\%)$ 和纯 local search $(\alpha = 0.0，89.34\%)$ ——都是次优的。平衡设置（α = 0.5）一致产生峰值准确率（91.62%），确认了结合 global stability 和 local plasticity 的必要性。
- Target Cluster Count $( K _ { i n i t } )$: 虽然将 nodes 从 20 增加到 500 通过精炼 manifold approximation 提高了准确率（在 CIFAR-100 上从 90.92% 到 92.28%，Figure 4b），但 marginal utility 最终递减。我们采用 $K _ { i n i t } = 60$ 作为高度鲁棒的"sweet spot"，在保持严格计算和存储效率的同时达到优秀准确率。
- Topological Maintenance & Alignment $( a g e _ { m a x } , T _ { s o i n n } , \lambda )$: 对于控制 topological graph updates 和 pointwise drift tracking 的其余参数，性能曲线非常平坦。例如，框架在宽范围的最大 edge ages $( a g e _ { m a x } \in [ 3 , 30 ] )$ 和 STAR momentum 值 $( \lambda \in [ 0.9 , 1.0 ] )$ 上表现出极大的 fault tolerance。

The "Tuning-Free" Philosophy. 值得注意的是，dataset-specific fine-tuning 可能会产生比我们主要结果中报告的更高的指标。例如，在 ImageNet-R 上（Figure 5a），将 α 调整到 0.6 可达到 70.43% 的 $A _ { A v g }$，明显超过了我们在统一 $\alpha = 0.5$ 设置下报告的 69.97%。然而，我们有意跳过了此类针对数据集的调优。通过在所有数据集上使用单一、统一的配置报告结果，我们证明了我们方法的"tuning-free"特性。HC-SOINN 本质上依赖 data-driven self-organizing growth 来动态描绘复杂 manifold，赋予其显著的自适应性，并消除了手动 parameter engineering 的实际负担。

Figure 4. Split CIFAR-100 上（与 CODA-Prompt 集成）的全面 hyperparameter sensitivity analysis。结果表明我们的框架在广泛的数值范围内保持了高度稳定的 average accuracy $( A _ { A v g } )$ 和 last-task accuracy $( A _ { L a s t } )$。

Figure 5. Split ImageNet-R 上的全面 hyperparameter sensitivity analysis。参数 (a)-(d) 与 SimpleCIL 一起评估，而 (e) 与 DualPrompt 一起评估。与 CIFAR-100 类似，在不同配置下性能保持显著鲁棒。
