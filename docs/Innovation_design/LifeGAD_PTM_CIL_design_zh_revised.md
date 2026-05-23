# LifeGAD：面向 PTM 类增量学习的生命周期字典原型（修订版）

> 版本：2026-05-19  
> 方法全称：**LifeGAD: Risk-Calibrated Lifelong Dictionary Prototypes for PTM-Based Class-Incremental Learning**  
> 定位：在 GAD 的共享字典原型与几何锚定基础上，引入 **字典生命周期管理**，使字典 atom 能在无 replay 的 CIL 序列中被保护、复用、扩容、隔离和裁剪。  
> 修订依据：`审查/LifeGAD_审查报告.md`、`审查/LifeGAD_审查报告_v2.md`、`审查/LifeGAD_审查报告_v3.md`、`dictionary_learning_Paper`、`DPR/01_dpr_architecture.md`、`CIL_Paper`。

---

## 0. 核心定位

<!-- 修订说明：根据 v3 审查报告收窄核心 claim；将 PMI 明确为 atom-error association proxy，弱化 MoAL/M2SD 机制继承表述，并把完整 lifecycle 与 MVP 证据边界分开。 -->

GAD 的原始设计将每个类别 prototype 表示为共享字典原子的稀疏组合，并通过 prototype anchor 与 coefficient anchor 稳定旧类几何。DPR 架构文档能够支持的基础事实是：共享字典、per-layer 或 per-class 系数、余弦 prototype readout、温度缩放分类和 L1 稀疏正则是可行的字典读出框架。LifeGAD 在此基础上提出的是一个更进一步的研究假设：

> CIL 中的关键问题不是一次性学习一个固定字典，而是在旧数据不可回看的条件下维护一个稳定、可复用、可诊断、可增长的共享坐标系。

因此 LifeGAD 不再把字典视为静态参数，而把每个 atom 视为具有生命周期状态的对象：

- `protected`：旧类高度依赖，默认冻结或强锚定；
- `plastic`：可被当前任务更新；
- `reserve`：base 阶段预留给未来类别，是受 D-FSCIL 前向兼容思想启发的经验设计，不声称机制继承；
- `quarantined`：与当前任务错误统计相关，暂不直接删除但限制使用；
- `inactive`：长期低利用，可裁剪或归档。

LifeGAD 的核心思想是：

1. 新类优先复用已有共享字典；
2. 当现有字典无法表达新类，或当前任务错误统计集中到某些 atoms 时，触发扩容、clone 或隔离；
3. 旧类支撑通过 protected atoms、class anchors 和选择性旧知识补强稳定；
4. 整个推理过程仍然 task-agnostic，不需要 task-ID。

需要强调的是，生命周期状态机是 LifeGAD 的创新贡献，但其中的阈值、clone、quarantine 和 prune 仍是经验设计。本文只把 PMI 写作当前任务的 **atom-error association proxy**，用于风险诊断；不把它解释为 SUE 不确定性估计器的原样迁移，也不把它解释为因果证据。首版论文的强 claim 应收窄到 protected reuse 与 residual-triggered growth；PMI、rumination、clone、quarantine、prune 和 transport 是校准诊断或 appendix 增强机制，完整 lifecycle 闭环需要额外实现与消融支撑。

受 MoAL 的知识反刍与动量适配思想启发，LifeGAD 额外吸收若干**头部侧**设计原则，但不改变其 classifier geometry 定位：第一，旧知识补强应是选择性的，只处理 anchor 漂移、旧新 margin 违规或 atom 风险集中的旧类；第二，当上游轻量适配器或外部特征系统发生漂移时，旧 anchors 和 dictionary atoms 需要可选的 feature-space transport；第三，字典更新可采用状态感知的动量约束，而不是对所有 atoms 统一冻结或统一适配；第四，被旧类强依赖但在当前任务中持续诱发错误的 atoms 可进入 stale-binding refresh 队列，通过 coefficient reallocation、选择性反刍、可选 clone 或隔离处理，而不是简单删除。LifeGAD 不继承 MoAL 的 adapter weight interpolation、解析/RLS 分类头或完整 PTM-CIL 框架，只把选择性补强与动量约束原则改写到 dictionary head 侧。

受 M2SD 的 base-session 前向准备思想远缘启发，LifeGAD 进一步引入若干**字典侧**经验原则：第一，base 阶段不能只拟合已知类，还应显式准备可被未来新类激活的 reserve atoms；第二，reserve candidates 不应只来自单一路径的类均值 mixup，可尝试使用多视角虚拟方向并约束其 head 输出一致；第三，base 训练可分为稳定 warmup 与 reserve preparation 两阶段，阶段比例作为消融项而非固定结论；第四，多视角 reserve 分支只作为训练时正则，推理时丢弃，不改变 LifeGAD 的单次 prototype matching 推理形式。M2SD 的 mixup/CutMix 是 FSCIL 图像级表示学习机制；LifeGAD 的 feature-mask reserve 只是 frozen-feature/head 侧扰动视角，不是机制复用，必须通过 linear-only / mask-only / dual-view 消融验证。

---

## 1. 问题设定与符号

<!-- 修订说明：保留原问题设定，同时补充校准/评估 split 符号；本轮进一步明确 feature-adapted setting 下工作字典也必须 transport 到当前空间。 -->

阅读提示：本版对 display math 公式块采用“公式后紧跟 `符号说明：`”的写法；重复出现的符号优先沿用第 1-2 节的全局定义，局部新引入的索引、阈值、集合和诊断量在对应公式旁说明。

### 1.1 类增量学习

给定任务序列：

```math
\mathcal D_t=\{(x_i^t,y_i^t)\}_{i=1}^{n_t},
\qquad
y_i^t\in \mathcal Y_t,
```

符号说明：\(t\) 表示增量学习中的任务编号；\(\mathcal D_t\) 是第 \(t\) 个任务的数据集；\(x_i^t\) 是该任务第 \(i\) 个输入样本，\(y_i^t\) 是对应类别标签；\(n_t\) 是任务 \(t\) 的样本数；\(\mathcal Y_t\) 是任务 \(t\) 引入的新类别集合。

不同任务类别互不重叠：

```math
\mathcal Y_i\cap \mathcal Y_j=\emptyset,\qquad i\ne j.
```

符号说明：\(\mathcal Y_i\cap\mathcal Y_j\) 表示两个任务类别集合的交集；该式说明不同任务不会重复引入同一类别。

到第 \(t\) 个任务为止，已见类别集合为：

```math
\mathcal C_t=\bigcup_{j=0}^{t}\mathcal Y_j.
```

符号说明：\(\mathcal C_t\) 是截至任务 \(t\) 已经见过的全部类别集合；\(\bigcup_{j=0}^{t}\) 表示从 base task \(0\) 到当前任务 \(t\) 的类别并集。

训练时只能访问当前任务数据 \(\mathcal D_t\)，不能访问旧任务样本。测试时不提供 task-ID，模型必须在所有已见类别上统一分类：

```math
\hat y=\arg\max_{c\in\mathcal C_t} S_c(x).
```

符号说明：\(\hat y\) 是模型预测类别；\(c\) 遍历所有已见类别；\(S_c(x)\) 是样本 \(x\) 对类别 \(c\) 的分类分数；\(\arg\max\) 返回分数最高的类别。

为避免触发信号和评估信号混用，当前任务数据在训练内部划分为：

```math
\mathcal D_t=\mathcal D_t^{train}\cup \mathcal V_t^{cal}\cup \mathcal V_t^{eval}.
```

符号说明：\(\mathcal V_t^{cal}\) 用于 residual、usage、PMI 等生命周期统计，\(\mathcal V_t^{eval}\) 仅用于验证 growth 或 quarantine 等动作是否改善下游指标。

符号说明：\(\mathcal D_t^{train}\) 是实际用于优化参数的训练 split；\(\mathcal V_t^{cal}\) 是校准 split，只用于触发统计和阈值判断；\(\mathcal V_t^{eval}\) 是内部评估 split，只用于验证某个生命周期动作是否有效。三者应互不重叠，避免用同一批样本同时触发和证明动作有效。

### 1.2 PTM 特征

使用冻结或轻量适配的预训练模型作为特征提取器：

```math
h(x)=f_\theta(x)\in\mathbb R^d.
```

符号说明：\(f_\theta\) 是预训练模型或轻量适配后的特征提取器；\(\theta\) 是其参数；\(h(x)\) 是未归一化的 \(d\) 维特征；\(\mathbb R^d\) 表示 \(d\) 维实向量空间。

LifeGAD 默认在 base task 上估计一个可选白化 / 标准化变换 \(B\)，用于降低 PTM 特征的各向异性：

```math
z(x)=
\frac{B(h(x))}{\|B(h(x))\|_2+\epsilon}.
```

符号说明：若不使用白化，则 \(B\) 为恒等映射。白化本身不是 LifeGAD 的必要贡献，需在消融中报告。

符号说明：\(B\) 是可选的白化或标准化变换；\(z(x)\) 是用于后续字典与 prototype 计算的归一化特征；\(\|\cdot\|_2\) 是欧氏范数；\(\epsilon>0\) 是数值稳定项，避免分母为 0。

#### 1.2.1 Appendix-only feature-space dictionary/anchor transport

LifeGAD 默认主协议使用 frozen PTM 特征，因此 \(z_t(x)=z_{t-1}(x)\)，旧类 anchors 与 dictionary atoms 不需要跨特征空间修正。Feature-space dictionary/anchor transport 不作为主文默认模块，只作为 LifeGAD-on-adapted-features 的 appendix 机制。当 LifeGAD 作为 plug-in readout 接到 EASE、SD-LoRA、DIA、MoAL 等轻量适配特征上时，上游表示可能随任务变化。此时直接复用 \(q_c^\star\)、\(d_m^\star\) 和 \(D_t\) 会把“旧类稳定”与“旧特征空间”混在一起。

为避免该问题，引入一个可选的低容量传输映射 \(T_t\)，只在 feature-adapted setting 中启用：

```math
T_t:\mathbb R^d\rightarrow\mathbb R^d,
\qquad
\tilde q_c^\star=
\frac{T_t(q_c^\star)}
{\|T_t(q_c^\star)\|_2+\epsilon}.
```

符号说明：\(T_t\) 是任务 \(t\) 中从旧特征空间到当前特征空间的可选传输映射；\(q_c^\star\) 是旧类 \(c\) 首次学习后保存的 anchor；\(\tilde q_c^\star\) 是传输并重新归一化后的当前空间 anchor；星号 \((\star)\) 表示历史快照，波浪号 \((\tilde{\ })\) 表示经 transport 对齐后的版本。

对 protected atom 快照和需要继续参与推理的旧 dictionary atoms 使用同一映射：

```math
\tilde d_m^\star=
\frac{T_t(d_m^\star)}
{\|T_t(d_m^\star)\|_2+\epsilon},
\qquad
\tilde d_m=
\frac{T_t(d_m)}
{\|T_t(d_m)\|_2+\epsilon}.
```

符号说明：\(d_m^\star\) 是 protected atom \(m\) 的冻结快照；\(d_m\) 是当前字典中的 atom；\(\tilde d_m^\star\) 和 \(\tilde d_m\) 分别是快照 atom 与当前 atom 经 \(T_t\) 传输后的当前空间版本。

在 feature-adapted setting 中，当前任务的工作字典必须先初始化到当前特征空间：

```math
D_t^{init}
=
\mathrm{RowNorm}(T_t(D_{t-1})),
\qquad
D_t\leftarrow D_t^{init}.
```

符号说明：\(D_{t-1}\) 是上一任务结束时保存的工作字典；\(D_t^{init}\) 是经 \(T_t\) 逐行传输并归一化后的当前空间字典；\(\mathrm{RowNorm}\) 表示对每个 atom 行向量分别做 \(L_2\) 归一化；赋值 \(D_t\leftarrow D_t^{init}\) 表示当前任务后续 readout、residual、PMI、rumination 和推理 prototype 都基于 current-space dictionary。

\(T_t\) 由当前任务样本同时经过上一阶段特征提取器和当前特征提取器得到的 paired features 训练：

```math
\mathcal L_{\mathrm{transport}}
=
\mathbb E_{x\in\mathcal D_t^{train}}
\left\|
z_t(x)-T_t(z_{t-1}(x))
\right\|_2^2.
```

符号说明：\(\mathcal L_{\mathrm{transport}}\) 是训练 \(T_t\) 的对齐损失；\(z_{t-1}(x)\) 和 \(z_t(x)\) 分别表示同一样本经过上一阶段与当前阶段特征提取器得到的归一化特征；\(\mathbb E_{x\in\mathcal D_t^{train}}\) 表示对当前训练 split 样本求平均。

默认候选为 affine / orthogonal Procrustes / 两层小 MLP，主文优先使用 affine 或 Procrustes，避免多次 MLP 映射造成早期 anchors 误差累积。Feature-adapted setting 中，传输后的 \(D_t\)、\(\tilde q_c^\star\)、\(\tilde d_m^\star\) 和 \(\tilde d_m\) 共同定义当前任务空间；原始旧空间 metadata 仍保留，便于回退和误差分析。若使用 frozen features，则 \(T_t\) 为恒等映射，\(D_t^{init}=D_{t-1}\)，且该项不进入训练。

---

## 2. 字典原型表示

<!-- 修订说明：根据 v3 审查报告修正 lifecycle 符号闭合性；保留五状态集合，将 grown/clone 改为 origin metadata，并明确 inactive active-mask、A_t 和 stale new-blocked flag。 -->

### 2.1 生命周期字典

LifeGAD 在任务 \(t\) 维护一个可增长字典：

```math
D_t=[d_1,\ldots,d_{M_t}]^\top\in\mathbb R^{M_t\times d}.
```

符号说明：\(D_t\) 是任务 \(t\) 的字典矩阵；\(d_m\) 是第 \(m\) 个 atom 的未归一化向量；\(M_t\) 是当前 atom 总数；上标 \(\top\) 表示把 atom 作为矩阵行堆叠；\(d\) 与 PTM 特征维度一致。

每个 atom 按行归一化：

```math
\hat d_m=\frac{d_m}{\|d_m\|_2+\epsilon},
\qquad
\hat D_t=[\hat d_1,\ldots,\hat d_{M_t}]^\top.
```

符号说明：\(\hat d_m\) 是 \(d_m\) 的单位长度版本；\(\hat D_t\) 是行归一化后的字典；后续 prototype、相似度和 residual 默认使用 \(\hat D_t\)，以便余弦几何可比。

每个 atom 有生命周期状态：

```math
s_m\in
\{\mathrm{protected},\mathrm{plastic},\mathrm{reserve},\mathrm{quarantined},\mathrm{inactive}\}.
```

符号说明：\(s_m\) 是 atom \(m\) 的生命周期状态；五个取值分别表示旧类保护、可塑更新、未来预留、风险隔离和逻辑停用。

`grown` 不再作为第六个状态，而作为来源元数据记录。这样 lifecycle 的训练行为只由 \(s_m\) 决定，扩容来源由 \(o_m\) 追踪：

```math
o_m\in
\{\mathrm{base},\mathrm{reserve},\mathrm{grown},\mathrm{clone}\}.
```

符号说明：\(o_m\) 是 atom \(m\) 的来源元数据，不决定训练行为本身；`base` 表示 base task 初始化得到，`reserve` 表示预留 atom，`grown` 表示由 residual growth 新增，`clone` 表示由旧 atom 克隆得到。

为使 logical inactive 在公式中闭合，定义 active atom 集合和二值掩码：

```math
\mathcal M_t^{act}
=
\{m:s_m\ne\mathrm{inactive}\},
\qquad
r_{t,m}^{act}=\mathbf 1[m\in\mathcal M_t^{act}].
```

符号说明：\(\mathcal M_t^{act}\) 是任务 \(t\) 中仍参与读出、训练、统计和初始化的 atom 索引集合；\(r_{t,m}^{act}\) 是 atom \(m\) 的 active mask；logical inactive atom 仍可保留在张量中，但默认不参与后续公式中的有效计算。

对于 inactive atoms，默认强制：

```math
a_{c,m}=0,\qquad
\nabla d_m=0,
\qquad
m\notin \mathrm{TopKInit}(c),
\quad
m\notin \mathrm{ReadoutSupport}(c).
```

符号说明：第一项保证 inactive atom 不进入 prototype readout；第二项保证 inactive atom 不接收梯度或 momentum 更新；后两项说明新类 top-k/ridge 初始化和推理 support 均排除 inactive atoms。

stale-binding 不改变单一生命周期状态本身，而通过独立使用限制 flag 表达“保旧类、禁新类”：

```math
b_m^{new\text{-}blocked}\in\{0,1\}.
```

符号说明：\(b_m^{new\text{-}blocked}=1\) 表示 atom \(m\) 不允许被新类 top-k/ridge 初始化、coefficient reallocation 或后续新类 support 继续选择；该 flag 不影响旧类 readout，因此 stale protected atom 可保持 \(s_m=\mathrm{protected}\)。

状态含义如下：

| 状态 | 含义 | 默认训练行为 |
|---|---|---|
| `protected` | 旧类高度依赖的 atom | 冻结或强 L2 anchor |
| `plastic` | 当前任务可更新 atom | 可训练 |
| `reserve` | base 阶段预留给未来类别 | 被新类激活后转为 plastic/protected |
| `quarantined` | 与当前任务错误统计强相关的 atom | 降权、限制新类初始化、不直接删除 |
| `inactive` | 长期低利用 atom | 逻辑停用；不参与梯度、momentum、新类 top-k/ridge 初始化或推理 support |

物理 prune 只允许发生在 \(m\notin\bigcup_{c\in\mathcal C_{t-1}}\mathcal S_c^\star\) 且 remapping 完成之后。若 remapping 不能同步更新 \(D_t,A_t,a_c^\star,q_c^\star,\mathcal S_c^\star\) 和 atom metadata，则只做 logical inactive，不从张量中删除该 atom。

### 2.2 类别系数与 prototype

每个类别维护稀疏系数：

```math
a_c\in\mathbb R^{1\times M_t}.
```

符号说明：\(a_c\) 是类别 \(c\) 在当前字典 \(D_t\) 上的系数行向量；其长度等于当前 atom 数 \(M_t\)；\(a_{c,m}\) 表示类别 \(c\) 对 atom \(m\) 的权重。

把所有已见类系数按行堆叠为：

```math
A_t=[a_c]_{c\in\mathcal C_t}
\in
\mathbb R^{|\mathcal C_t|\times M_t}.
```

符号说明：\(A_t\) 是任务 \(t\) 的全类别系数矩阵；第 \(c\) 行是类别 \(c\) 的系数 \(a_c\)；该定义用于 grow / clone / prune 后的维度同步说明。

类别 prototype 由字典线性组合后归一化得到：

```math
a_c^{act}=a_c\odot r_t^{act},
\qquad
u_c=a_c^{act}\hat D_t,
\qquad
p_c=\frac{u_c}{\|u_c\|_2+\epsilon}.
```

符号说明：\(r_t^{act}\) 是由 \(r_{t,m}^{act}\) 组成的长度 \(M_t\) 的 active mask 向量；\(\odot\) 表示逐元素乘法；\(a_c^{act}\) 是屏蔽 inactive atoms 后的有效系数；\(u_c\) 是未归一化类别 prototype；\(p_c\) 是归一化后的类别 prototype，用于余弦分类。

分类分数：

```math
S_c(x)=\frac{\langle z(x),p_c\rangle}{\tau}.
```

符号说明：\(\langle\cdot,\cdot\rangle\) 表示内积；\(S_c(x)\) 是样本 \(x\) 对类别 \(c\) 的 logit / score；\(\tau>0\) 是温度系数，越小则 softmax 分布越尖锐。

推理：

```math
\hat y=\arg\max_{c\in\mathcal C_t}S_c(x).
```

符号说明：这里的 \(\hat y\) 与第 1.1 节一致，表示在所有已见类别 \(\mathcal C_t\) 上分数最大的预测类别。

DPR 文档中温度 \(\tau=0.05\) 是当前主线常用设置，因此 LifeGAD 采用 \(\tau=0.05\) 作为默认初值，而不是重新声称新的理论推荐值。

### 2.3 类别锚点

每个类别首次学习完成后保存：

```math
q_c^\star=p_c,
\qquad
a_c^\star=a_c.
```

符号说明：\(q_c^\star\) 是类别 \(c\) 首次学习完成时保存的 prototype anchor；\(a_c^\star\) 是同步保存的系数 anchor；星号表示后续任务中作为旧类参考快照使用。

同时保存该类主要使用的 atom support：

```math
\mathcal S_c^\star=\{m: |a_{c,m}^\star|>\eta_a\}.
```

符号说明：\(\mathcal S_c^\star\) 是旧类 \(c\) 的历史 atom support；\(|a_{c,m}^\star|\) 是 anchor 系数在 atom \(m\) 上的绝对值；\(\eta_a\) 是判断某个 atom 是否被该类显著使用的系数阈值。

任意 grow、clone 或 prune 之后必须保持以下不变量：

1. \(D_t\)、所有 \(a_c\)、\(a_c^\star\) 的维度一致；
2. 旧类 \(q_c^\star\) 可重新对齐到当前 prototype 空间；
3. protected atom 有冻结快照 \(d_m^\star\)；
4. prune 仅允许删除不在任何旧类 support 中的 atom，且需要维护 index remapping。

---

## 3. 生命周期统计量

<!-- 修订说明：修正 old support 未归一化、PMI 使用条件过度迁移、预测内生性和 residual 方向不一致等问题。 -->

LifeGAD 的关键区别是：不固定字典容量，而是基于 atom-level 统计量决定冻结、扩容、clone、隔离和裁剪。这些统计量用于状态迁移，不直接作为最终论文的因果解释。

### 3.1 Atom 贡献

对样本 \(x\)，令预测类为 \(\hat y\)。定义 atom \(m\) 对当前预测的非负贡献：

```math
\psi_m^{pred}(x)=
\frac{
\max(0,a_{\hat y,m}\langle z(x),\hat d_m\rangle)
}{
\sum_{j\in\mathcal M_t^{act}} \max(0,a_{\hat y,j}\langle z(x),\hat d_j\rangle)+\epsilon
}.
```

符号说明：\(\psi_m^{pred}(x)\) 是 active atom \(m\) 对预测类 \(\hat y\) 的非负归一化贡献；\(a_{\hat y,m}\) 是预测类在 atom \(m\) 上的系数；\(\langle z(x),\hat d_m\rangle\) 是样本特征与 atom 的余弦相似度；\(\max(0,\cdot)\) 只保留正贡献；分母只对 active atoms \(j\in\mathcal M_t^{act}\) 求和，避免 inactive atoms 进入使用率和 PMI 统计。

该定义依赖预测类 \(\hat y\)，而错误变量 \(e(x)\) 也依赖 \(\hat y\)，因此存在预测内生性。为避免把自相关误读为独立风险来源，实验中同时报告三种变体：

```math
\psi_m^{true}(x)=
\frac{
\max(0,a_{y,m}\langle z(x),\hat d_m\rangle)
}{
\sum_{j\in\mathcal M_t^{act}} \max(0,a_{y,j}\langle z(x),\hat d_j\rangle)+\epsilon
},
```

符号说明：\(\psi_m^{true}(x)\) 与 \(\psi_m^{pred}(x)\) 形式相同，但使用真实标签 \(y\) 对应的系数 \(a_{y,m}\)，且分母同样只遍历 active atoms；因此只能在有标签的校准或验证 split 上计算。

```math
h_m^{coef}(c)=\mathbf 1[|a_{c,m}|>\eta_a].
```

符号说明：\(\psi_m^{true}\) 仅在有标签的 validation/calibration split 上使用，\(h_m^{coef}\) 用于区分“系数被选中”和“样本贡献实际激活”。

符号说明：\(h_m^{coef}(c)\) 是二值指示变量；\(\mathbf 1[\cdot]\) 在条件成立时取 1，否则取 0；\(\eta_a\) 是系数显著性阈值。

### 3.2 Atom 使用率

在当前任务的 calibration split \(\mathcal V_t^{cal}\) 上计算：

```math
U_{m,\mathrm{pred}}^{(t)}
=
\mathbb E_{x\in\mathcal V_t^{cal}}[\psi_m^{pred}(x)].
```

符号说明：\(U_{m,\mathrm{pred}}^{(t)}\) 是任务 \(t\) 中 atom \(m\) 的预测标签视角平均使用率；\(\mathbb E\) 表示对 calibration split 中样本求均值。

同时报告有标签 calibration split 上的 true-label 变体：

```math
U_{m,\mathrm{true}}^{(t)}
=
\mathbb E_{x\in\mathcal V_t^{cal}}[\psi_m^{true}(x)].
```

符号说明：\(U_{m,\mathrm{true}}^{(t)}\) 是真实标签视角的平均使用率，只用于诊断预测内生性，不在无标签推理中使用。

默认状态迁移仍使用 \(U_{m,\mathrm{pred}}^{(t)}\)，因为推理时没有真实标签；\(U_{m,\mathrm{true}}^{(t)}\) 只作为诊断，用于暴露 usage 与 error PMI 共享预测信号导致的内生性。

低使用率 atom 可能被标记为 `inactive`，但只有在旧类支撑强度也低时才允许裁剪。

### 3.3 旧类支撑强度

修正后的 atom \(m\) 对旧类的支撑强度为旧类均值，而不是随旧类数单调增长的求和：

```math
O_m^{(t)}
=
\frac{1}{|\mathcal C_{t-1}|}
\sum_{c\in\mathcal C_{t-1}}
\frac{|a_{c,m}|}{\|a_c\|_1+\epsilon}.
```

符号说明：\(O_m^{(t)}\) 是 atom \(m\) 对旧类的平均支撑强度；\(\mathcal C_{t-1}\) 是当前任务前已见旧类集合；\(\|a_c\|_1\) 是类别 \(c\) 系数绝对值之和，用于把不同类别系数尺度归一化。

若类别数较大，也可使用 top-q 旧类支撑均值：

```math
O_{m,q}^{(t)}=
\frac1q\sum_{c\in \mathrm{TopQ}_q(m)}
\frac{|a_{c,m}|}{\|a_c\|_1+\epsilon}.
```

符号说明：\(O_{m,q}^{(t)}\) 是 top-\(q\) 旧类支撑均值；\(\mathrm{TopQ}_q(m)\) 表示在所有旧类中选出对 atom \(m\) 归一化系数最大的 \(q\) 个类别；\(q\) 是人为设定的聚合宽度。

阈值 \(\theta_o\) 应由当前任务的分位数或 base 校准统计确定，不能使用随任务数隐式变化的固定求和阈值。

### 3.4 Atom 错误 PMI

定义 atom 激活指示变量：

```math
h_m(x)=\mathbf 1[\psi_m^{pred}(x)>\eta_\psi].
```

符号说明：\(h_m(x)\) 是样本级 atom 激活指示变量；\(\eta_\psi\) 是贡献阈值，只有当 atom \(m\) 对预测类贡献超过该阈值时认为被激活。

定义错误指示变量：

```math
e(x)=\mathbf 1[\hat y(x)\ne y].
```

符号说明：\(e(x)\) 是错误指示变量；\(\hat y(x)\) 是模型预测标签；\(y\) 是真实标签；预测错误时 \(e(x)=1\)。

在当前任务 held-out calibration split \(\mathcal V_t^{cal}\) 上估计 atom 激活与错误之间的 PMI：

```math
\widehat{\mathrm{PMI}}_m^{err}
=
\log
\frac{
\widehat P(h_m=1,e=1)+\epsilon
}{
(\widehat P(h_m=1)+\epsilon)(\widehat P(e=1)+\epsilon)
}.
```

符号说明：\(\widehat{\mathrm{PMI}}_m^{err}\) 是 atom \(m\) 激活与预测错误之间的经验 PMI；\(\widehat P(\cdot)\) 表示在 \(\mathcal V_t^{cal}\) 上估计的频率概率；分子估计 atom 激活且预测错误的联合概率；分母估计二者若独立时的概率乘积；\(\log\) 为自然对数。

该统计量只被解释为 **当前任务 atom-error association proxy**。SUE 的原始条件包括固定字典、独立校准集、仅正确样本训练字典、全校准集估计 PMI；LifeGAD 的 \(\mathcal V_t^{cal}\) 只是当前任务内 held-out split，与训练数据同分布，不是 SUE 意义上的外部分布独立校准集。再加上字典会演化且每个任务样本较少，因此不能把该 PMI 直接写成 SUE 不确定性估计器的理论迁移。

PMI 触发只使用当前任务的原始估计 \(\widehat{\mathrm{PMI}}_{m,t}^{err}\)，需要满足可靠性过滤：

```math
n_{11},n_{10},n_{01},n_{00}\ge n_{\min},
```

符号说明：\(n_{ab}\) 是 \(h_m=a,e=b\) 的 2x2 计数表条目，其中 \(a,b\in\{0,1\}\)；\(n_{\min}\) 是每个格子的最小可靠计数阈值。

并在该 2x2 计数表上计算原始置信区间：

```math
\mathrm{CI}_{m,t}^{raw}
=
\left[
\widehat{\mathrm{PMI}}_{m,t}^{err}
-z_{\alpha/2}\widehat\sigma_{m,t},
\widehat{\mathrm{PMI}}_{m,t}^{err}
+z_{\alpha/2}\widehat\sigma_{m,t}
\right],
```

符号说明：\(\widehat\sigma_{m,t}\) 可由 delta method 或 bootstrap 在当前任务 calibration split 上估计。若任一计数低于 \(n_{\min}\)，则该 atom 的 PMI 不参与 freeze / grow / clone / quarantine / stale-binding 决策，仅记为 `PMI-unreliable`。实验必须报告：

符号说明：\(\mathrm{CI}_{m,t}^{raw}\) 是任务 \(t\) 中 atom \(m\) 的原始 PMI 置信区间；\(z_{\alpha/2}\) 是标准正态分布的双侧分位数；\(\widehat\sigma_{m,t}\) 是 PMI 估计的标准误；\(\alpha\) 是显著性水平。

```math
\mathrm{PMIInvalidRate}_t
=
\frac1{|\mathcal M_t^{act}|+\epsilon}
\sum_{m\in\mathcal M_t^{act}}
\mathbf 1[
\min(n_{11},n_{10},n_{01},n_{00})<n_{\min}
].
```

符号说明：\(\mathrm{PMIInvalidRate}_t\) 是任务 \(t\) 中 PMI 计数不可靠的 active atom 比例；\(\sum_{m\in\mathcal M_t^{act}}\) 只遍历 active atoms；\(\min(n_{11},n_{10},n_{01},n_{00})<n_{\min}\) 表示至少一个计数格低于可靠阈值。

跨任务 EMA 仅作为可视化和趋势监控，不作为状态迁移触发条件。为避免字典演化后追踪非同一语义实体，EMA 只对 persistent atoms 更新。令 \(\chi_{m,t}\) 表示 atom \(m\) 在任务 \(t-1\rightarrow t\) 之间未发生 grow / clone / remap，且方向漂移仍小：

```math
\chi_{m,t}
=
\mathbf 1[
\mathrm{id}_{m,t}=\mathrm{id}_{m,t-1}
\ \mathrm{and}\
\langle \hat d_{m,t},\hat d_{m,t-1}\rangle>\theta_{\mathrm{id}}
].
```

符号说明：\(\chi_{m,t}\) 是 persistent-atom 指示变量；\(\mathrm{id}_{m,t}\) 是 atom 的身份标识；\(\langle \hat d_{m,t},\hat d_{m,t-1}\rangle\) 衡量相邻任务中同一 atom 的方向一致性；\(\theta_{\mathrm{id}}\) 是允许沿用 EMA 历史的方向相似度阈值。

则监控用 EMA 为：

```math
\widetilde{\mathrm{PMI}}_{m,t}^{err}
=
\begin{cases}
\rho\widetilde{\mathrm{PMI}}_{m,t-1}^{err}
+(1-\rho)\widehat{\mathrm{PMI}}_{m,t}^{err},
&\chi_{m,t}=1,\\
\widehat{\mathrm{PMI}}_{m,t}^{err},
&\chi_{m,t}=0.
\end{cases}
```

符号说明：\(\widetilde{\mathrm{PMI}}_{m,t}^{err}\) 是仅用于监控的 EMA PMI；\(\rho\in[0,1)\) 是 EMA 平滑系数；当 \(\chi_{m,t}=1\) 时沿用历史趋势，当 \(\chi_{m,t}=0\) 时重置为当前任务原始 PMI。

若原始置信区间跨过触发阈值，则不执行状态迁移，仅进入观察队列。本文不对 EMA 后的 \(\widetilde{\mathrm{PMI}}\) 声称置信区间。

### 3.5 新类残差

对当前任务新类 \(c\)，计算归一化类均值：

```math
\mu_c=
\frac1{|\mathcal D_t^c|}
\sum_{(x,y)\in\mathcal D_t^c}z(x),
\qquad
\hat\mu_c=
\frac{\mu_c}{\|\mu_c\|_2+\epsilon}.
```

符号说明：\(\mathcal D_t^c=\{(x,y)\in\mathcal D_t:y=c\}\) 是任务 \(t\) 中类别 \(c\) 的样本子集；\(\mu_c\) 是未归一化类均值；\(\hat\mu_c\) 是归一化类均值。

LifeGAD 区分向量残差和标量诊断残差。用于新增 atom 方向的是重构残差：

```math
r_c^{vec}=
\hat\mu_c-a_c\hat D_t,
\qquad
\hat r_c^{vec}=
\frac{r_c^{vec}}{\|r_c^{vec}\|_2+\epsilon}.
```

符号说明：\(r_c^{vec}\) 是类别 \(c\) 的向量重构残差，即类均值与当前字典重构 \(a_c\hat D_t\) 之间的差；\(\hat r_c^{vec}\) 是归一化残差方向，用于新增 atom 初始化。

用于诊断和阈值的余弦残差为：

```math
r_c^{cos}
=
1-
\left\langle
\hat\mu_c,
\frac{a_c\hat D_t}{\|a_c\hat D_t\|_2+\epsilon}
\right\rangle.
```

符号说明：\(r_c^{cos}\) 是标量余弦重构残差；值越大表示当前稀疏 prototype 与类均值方向越不一致；该值用于 grow 阈值判断，而不是直接作为新增 atom 方向。

这样 grow 判据和新增 atom 初始化处于同一 residual 空间，避免“标量 residual 用归一化方向、growth direction 用未说明残差”的不一致。

受 D-FSCIL 中 L2 协作表示稳定性的启发，LifeGAD 增加一个 **ridge shadow residual**，只用于诊断 grow 是否必要，不替代主模型的稀疏系数。对新类 \(c\)，计算 dense ridge shadow code：

```math
b_c
=
\hat\mu_c \hat D_t^\top
\left(\hat D_t\hat D_t^\top+\lambda_{\mathrm{sh}} I_{M_t}\right)^{-1}.
```

符号说明：\(b_c\) 是类别 \(c\) 的 dense ridge shadow code；\(\lambda_{\mathrm{sh}}\) 是 ridge 稳定系数；\(I_{M_t}\) 是 \(M_t\times M_t\) 单位矩阵；该式允许使用所有 atoms 形成 dense 表示，只用于诊断字典覆盖能力。

当 \(M_t>d\) 或 \(M_t\times M_t\) 逆计算更贵时，使用等价的 push-through 形式：

```math
b_c
=
\hat\mu_c
\left(\hat D_t^\top\hat D_t+\lambda_{\mathrm{sh}} I_d\right)^{-1}
\hat D_t^\top.
```

符号说明：这是与上一式等价的 \(d\times d\) push-through 形式；\(I_d\) 是 \(d\times d\) 单位矩阵；当 \(d<M_t\) 时该形式可能更省计算。

该变换使用 ridge 恒等式

```math
D^\top(DD^\top+\lambda I)^{-1}
=
(D^\top D+\lambda I)^{-1}D^\top,
```

符号说明：这里的 \(D\) 是泛指矩阵，不特指某一任务的 \(D_t\)；\(\lambda>0\) 是 ridge 正则；该恒等式说明两种 ridge 求解形式等价。

不是额外近似。

实现时选择 \(\min(M_t,d)\) 维矩阵做 Cholesky / ridge solve，而不是固定求 \(M_t\times M_t\) 逆。审查报告中“\(M_t=500+\) 即 \(M_t\gg d=768\)”这一量级表述不严谨；真正需要修正的是原方案没有说明按 \(M_t\) 与 \(d\) 自适应选择求解形式。

其 dense residual 为：

```math
r_c^{dense}
=
1-
\left\langle
\hat\mu_c,
\frac{b_c\hat D_t}{\|b_c\hat D_t\|_2+\epsilon}
\right\rangle.
```

符号说明：\(r_c^{dense}\) 是使用 dense ridge code \(b_c\) 后的余弦残差；若 \(r_c^{dense}\) 低而 \(r_c^{cos}\) 高，说明字典方向可能足够但稀疏选择不足。

解释规则如下：

| \(r_c^{cos}\) | \(r_c^{dense}\) | 解释 | 动作 |
|---|---|---|---|
| 高 | 低 | 字典覆盖足够，但稀疏预算或 top-k 选择不足 | 先放宽 \(k\) 或重解 top-k ridge，不 grow |
| 高 | 高 | 当前字典确实覆盖不足 | 允许 residual growth |
| 低 | 低 | 正常复用 | 不 grow |
| 低 | 高 | dense 解病态或字典条件数异常 | 检查 \(\kappa_{\mathrm{ridge}}\) 与 mutual coherence |

这样吸收 D-FSCIL 的 L2 稳定诊断，同时保持 LifeGAD 的稀疏 support 和 atom lifecycle 结构。

---

## 4. 初始化

<!-- 修订说明：将 top-k ridge 标注为工程启发式，删除 active atoms 固定比例矛盾，并把 reserve atoms 的来源降格为远缘动机。 -->

### 4.1 Base task 字典初始化

Base task 先提取所有类别均值：

```math
\{\hat\mu_c:c\in\mathcal Y_0\}.
```

符号说明：\(\mathcal Y_0\) 是 base task 类别集合；该集合表示为每个 base 类 \(c\) 计算一个归一化均值 \(\hat\mu_c\)，作为初始化候选。

可选初始化策略：

1. **Class means initialization**：直接用类均值或其扰动初始化 atoms；
2. **K-means initialization**：对样本特征或类均值聚类；
3. **Whitened k-means**：先白化再聚类；
4. **Pseudo-class reserve atoms**：对 base 类均值做 mixup，生成未来预留方向；
5. **Residual centers**：用初始表示残差聚类作为 reserve atoms。

Pseudo-class reserve atoms 只作为前向兼容思想的启发。D-FSCIL 使用 FSCIL、ResNet、L2 字典重建和增量字典适配，与 LifeGAD 的 PTM-CIL、稀疏系数和生命周期状态并不相同，因此不声称机制直接继承。M2SD 进一步提示：虚拟预留方向不应只依赖单一 mixing 视角，否则容易把随机增强噪声误当作未来类结构。LifeGAD 因此把原始 pseudo-class reserve 扩展为 **multi-view reserve atom consistency**，但只在 frozen feature / dictionary head 侧实现。

具体地，对 base 类 \(i,j\) 的归一化均值做类间 mixup：

```math
\tilde\mu_{ij}^{\mathrm{lin}}
=
\frac{
\gamma\hat\mu_i+(1-\gamma)\hat\mu_j
}{
\|\gamma\hat\mu_i+(1-\gamma)\hat\mu_j\|_2+\epsilon
},
\qquad
\gamma\sim U(0.4,0.6).
```

符号说明：\(\tilde\mu_{ij}^{\mathrm{lin}}\) 是由 base 类 \(i,j\) 的归一化均值线性混合得到的虚拟方向；\(\gamma\) 是混合权重；\(U(0.4,0.6)\) 表示在区间 \([0.4,0.6]\) 上均匀采样。

同时构造一个 CutMix-like 的 feature-side 视角，而不是图像级 CutMix。若 PTM 暴露 patch / token features，可在 token group 上采样掩码；若只有 pooled feature，则在白化后的 feature groups 或随机子空间上采样二值掩码 \(m\)，并与线性分支共享同一个 \(\gamma\)：

```math
\tilde\mu_{ij}^{\mathrm{mask}}
=
\frac{
m\odot \hat\mu_i+(1-m)\odot \hat\mu_j
}{
\|m\odot \hat\mu_i+(1-m)\odot \hat\mu_j\|_2+\epsilon
},
\qquad
\frac1d\sum_{\ell=1}^d m_\ell\approx \gamma.
```

符号说明：\(\tilde\mu_{ij}^{\mathrm{mask}}\) 是 feature-mask 视角的虚拟方向；\(m\in\{0,1\}^d\) 是二值掩码；\(\odot\) 表示逐维乘法；\(m_\ell\) 是掩码第 \(\ell\) 维；约束 \(\frac1d\sum_\ell m_\ell\approx\gamma\) 让 mask 分支与 linear 分支具有相近混合比例。

这两个分支只生成 reserve candidates 和训练时 virtual probes，不被加入正式类别集合。LifeGAD 不预设未来新类总数，而使用相对 base class atom budget 的固定 reserve 预算。令 \(M_{\mathrm{cls}}=mC_{\mathrm{base}}\)，则：

```math
M_{\mathrm{res}}
=
\lceil \rho_{\mathrm{res}}M_{\mathrm{cls}}\rceil,
\qquad
\rho_{\mathrm{res}}\in[0.1,0.3].
```

符号说明：\(M_{\mathrm{cls}}\) 是按 base 类数分配的正式 class atoms 预算；\(M_{\mathrm{res}}\) 是额外 reserve atoms 数；\(\rho_{\mathrm{res}}\) 是 reserve 相对 class atom 预算的比例；\(\lceil\cdot\rceil\) 表示向上取整。

从 \(\{\tilde\mu_{ij}^{\mathrm{lin}},\tilde\mu_{ij}^{\mathrm{mask}}\}\) 的候选池中通过 k-means 或 farthest-first selection 选出 \(M_{\mathrm{res}}\) 个 reserve atoms。Reserve atoms 在推理中默认不作为旧类 support 贡献；只有当新类 top-k 初始化或 residual growth 明确使用它们时，才转为 `plastic`。

Base 训练的 reserve preparation 阶段对两个视角加入轻量一致性正则。令 \(g_D(v)\) 为当前 LifeGAD head 在 virtual probe \(v\) 上对 base 类的 logit，软目标为 \(\bar y_{ij}=\gamma e_i+(1-\gamma)e_j\)，则：

```math
\mathcal L_{\mathrm{res-mv}}
=
\lambda_{\mathrm{vce}}
\sum_{(i,j)}
\left[
\mathrm{CE}(\bar y_{ij}, g_D(\tilde\mu_{ij}^{\mathrm{lin}}))
+
\mathrm{CE}(\bar y_{ij}, g_D(\tilde\mu_{ij}^{\mathrm{mask}}))
\right]
+
\lambda_{\mathrm{vkl}}
\sum_{(i,j)}
\mathrm{SKL}
\left(
\sigma(g_D(\tilde\mu_{ij}^{\mathrm{lin}})/\tau_{\mathrm{mv}}),
\sigma(g_D(\tilde\mu_{ij}^{\mathrm{mask}})/\tau_{\mathrm{mv}})
\right).
```

符号说明：\(\mathcal L_{\mathrm{res-mv}}\) 是 multi-view reserve consistency loss；\(\lambda_{\mathrm{vce}}\) 和 \(\lambda_{\mathrm{vkl}}\) 分别控制虚拟交叉熵项与双视角 KL 一致性项权重；\(\mathrm{CE}\) 是交叉熵；\(\mathrm{SKL}\) 是对称 KL；\(\sigma\) 是 softmax；\(\tau_{\mathrm{mv}}\) 是该一致性项的温度；\(\bar y_{ij}\) 是混合软标签；\(g_D(v)\) 是 LifeGAD head 对 virtual probe \(v\) 输出的 base 类 logits。

仅有 head 输出一致性可能主要更新 base prototype 边界，而不一定把梯度施加到 reserve atoms。为保证该模块确实准备 reserve 容量，Stage B1 还加入 reserve-atom alignment。令 \(\mathcal R_0\) 为 reserve atom 集合，\(r(i,j)\) 为候选选择阶段分配给 pair \((i,j)\) 的最近 reserve atom：

```math
\mathcal L_{\mathrm{res-align}}
=
\sum_{(i,j)}
\left[
1-\langle \hat d_{r(i,j)},\tilde\mu_{ij}^{\mathrm{lin}}\rangle
+
1-\langle \hat d_{r(i,j)},\tilde\mu_{ij}^{\mathrm{mask}}\rangle
\right].
```

符号说明：\(\mathcal L_{\mathrm{res-align}}\) 把 reserve atom 方向拉向虚拟方向；\(\mathcal R_0\) 是 base 初始化后的 reserve atom 集合；\(r(i,j)\) 是 pair \((i,j)\) 对应的最近 reserve atom 索引；\(\hat d_{r(i,j)}\) 是该 reserve atom 的归一化方向。

实现上可把 \(\lambda_{\mathrm{align}}\mathcal L_{\mathrm{res-align}}\) 并入 \(\mathcal L_{\mathrm{res-mv}}\)，也可作为单独项报告。Base support extraction 默认仍 mask 掉 reserve atoms，避免它们在尚未被新类激活前成为旧类正式支撑。

\(\mathcal L_{\mathrm{res-mv}}\) 的作用是让 reserve 区域在不同虚拟扰动下产生一致的 base-class 边界响应，并通过 alignment 把预留方向写入 reserve atoms，而不是把虚拟方向训练成真实未来类别。该项只在 base task 的 reserve preparation 阶段启用，推理时不保留任何双分支结构。

默认字典大小先定义 class atoms，再附加 reserve atoms，避免 reserve budget 对总量自引用：

```math
M_{\mathrm{cls}}=mC_{\mathrm{base}},
\qquad
M_{\mathrm{res}}=\lceil\rho_{\mathrm{res}}M_{\mathrm{cls}}\rceil,
\qquad
M_0=M_{\mathrm{cls}}+M_{\mathrm{res}},
\qquad
m\in\{3,5,8\}.
```

符号说明：\(C_{\mathrm{base}}=|\mathcal Y_0|\) 是 base 类别数；\(m\) 是每个 base 类分配的 class atom 数；\(M_0\) 是 base 初始化完成后的总字典大小。

\(m=5\) 和 \(\rho_{\mathrm{res}}\in[0.1,0.3]\) 仅作为工程先验，必须做容量曲线和 `w/o pseudo-class reserve` 消融。

### 4.2 新类系数初始化

对新类 \(c\)，优先使用 top-k ridge 初始化：

```math
\mathcal K_t^{new}
=
\{m\in\mathcal M_t^{act}: b_m^{new\text{-}blocked}=0,\ s_m\ne\mathrm{quarantined}\}.
```

符号说明：\(\mathcal K_t^{new}\) 是新类初始化可选择的 atom 候选集合；它排除 inactive atoms、被 stale-binding 标为 new-blocked 的 atoms，以及当前处于 quarantined 状态的 atoms。

```math
a_c^{(0)}
\approx
\arg\min_{\|a\|_0\le k,\ \mathrm{supp}(a)\subseteq\mathcal K_t^{new}}
\left\|
\hat\mu_c-a\hat D_t
\right\|_2^2
+\lambda_2\|a\|_2^2.
```

符号说明：\(a_c^{(0)}\) 是新类 \(c\) 的初始系数；\(\arg\min\) 表示寻找使目标最小的系数 \(a\)；\(\|a\|_0\le k\) 限制最多使用 \(k\) 个非零 atom；\(\mathrm{supp}(a)\subseteq\mathcal K_t^{new}\) 表示非零系数只能落在可选候选集合中；\(\lambda_2\) 是 ridge L2 正则系数；符号 \(\approx\) 表示工程上用 top-k ridge 近似求解。

工程实现是启发式近似，而不是精确 OMP/LASSO 解：

1. 计算 \(\hat\mu_c\) 与所有 atoms 的余弦相似度；
2. 在 \(\mathcal K_t^{new}\) 中选 top-k atoms；
3. 在支持集上解 ridge regression；
4. 其他系数置零；
5. 若 \(r_c^{cos}\) 或 \(\|r_c^{vec}\|_2\) 超过阈值，则触发 growth。

默认 \(k\in\{5,10\}\) 表示每类使用的绝对活跃 atom 数，不再写作“5%-15% atoms active”。若要使用比例预算，则明确改为：

```math
k_t=\lceil rM_t\rceil,\qquad r\in[0.05,0.15].
```

符号说明：\(k_t\) 是任务 \(t\) 的比例式 active atom 预算；\(r\) 是允许激活的 atom 比例；该变体与固定 \(k\in\{5,10\}\) 的默认设置不同。

---

## 5. 损失函数

<!-- 修订说明：补充 protected anchor snapshot 维护、完整 p-annealing 的 λ 系数退火、proxy old-logit 的可复现定义和复杂度；本轮修正 current-space anchor、active-mask 遍历和 sparse loss 双重 λ。 -->

### 5.1 当前任务分类损失

```math
\mathcal L_{\mathrm{ce}}
=
\mathbb E_{(x,y)\sim\mathcal D_t^{train}}
\mathrm{CE}(\{S_c(x)\}_{c\in\mathcal C_t},y).
```

符号说明：softmax 覆盖所有已见类别，不使用 task-ID。

符号说明：\(\mathcal L_{\mathrm{ce}}\) 是当前任务分类交叉熵；\((x,y)\sim\mathcal D_t^{train}\) 表示从当前训练 split 采样带标签样本；\(\{S_c(x)\}_{c\in\mathcal C_t}\) 是对所有已见类的 logit 向量；\(\mathrm{CE}\) 是多类交叉熵。

### 5.2 新类几何对齐

```math
\mathcal L_{\mathrm{new}}
=
\frac1{|\mathcal Y_t|}
\sum_{c\in\mathcal Y_t}
(1-\langle p_c,\hat\mu_c\rangle).
```

符号说明：\(\mathcal L_{\mathrm{new}}\) 是新类 prototype 与新类均值的几何对齐损失；\(|\mathcal Y_t|\) 是当前任务新类数；\(p_c\) 是学习得到的类别 prototype；\(\hat\mu_c\) 是该新类样本均值方向。

### 5.3 旧类 prototype anchor

```math
\mathcal L_{\mathrm{p-anchor}}
=
\frac1{|\mathcal C_{t-1}|}
\sum_{c\in\mathcal C_{t-1}}
(1-\langle p_c,\bar q_c^\star\rangle),
\qquad
\bar q_c^\star=
\begin{cases}
\tilde q_c^\star, & \text{feature-adapted setting},\\
q_c^\star, & \text{frozen-feature setting}.
\end{cases}
```

符号说明：这样写是为了避免在 feature-adapted setting 中把旧特征空间的 \(q_c^\star\) 与当前任务特征空间的 \(p_c\) 直接比较。主协议使用 frozen PTM 特征时 \(T_t=I\)，上式退化为原始 anchor loss。

符号说明：\(\mathcal L_{\mathrm{p-anchor}}\) 是旧类 prototype anchor 损失；\(\bar q_c^\star\) 是当前任务空间中用于约束旧类 \(c\) 的 anchor 目标；feature-adapted setting 使用 transport 后的 \(\tilde q_c^\star\)，frozen-feature setting 直接使用历史快照 \(q_c^\star\)。

### 5.4 旧类 coefficient anchor

```math
\mathcal L_{\mathrm{a-anchor}}
=
\frac1{|\mathcal C_{t-1}|}
\sum_{c\in\mathcal C_{t-1}}
\frac{\|a_c-a_c^\star\|_2^2}
{\|a_c\|_2^2+\|a_c^\star\|_2^2+\epsilon}.
```

符号说明：默认实现中可直接冻结旧类系数；该项用于 soft-anchor 变体。

符号说明：\(\mathcal L_{\mathrm{a-anchor}}\) 是旧类系数 anchor 损失；\(a_c\) 是当前系数，\(a_c^\star\) 是旧快照系数；分母按当前与历史系数范数归一化，避免大系数类别主导损失。

### 5.5 Protected atom 稳定项

令 \(\mathcal P_t\) 为 protected atoms 集合。每个 atom 首次进入 protected 状态时保存快照 \(d_m^\star\)。clone 或 grow 产生的新 atom 初始没有 protected 快照，只有在后续被旧类 support 使用并转为 protected 时才保存。

```math
\mathcal L_{\mathrm{D-freeze}}
=
\sum_{m\in\mathcal P_t}
\omega_m\|d_m-\bar d_m^\star\|_2^2,
\qquad
\bar d_m^\star=
\begin{cases}
\tilde d_m^\star, & \text{feature-adapted setting},\\
d_m^\star, & \text{frozen-feature setting}.
\end{cases}
```

符号说明：\(\omega_m\) 可由归一化旧类支撑强度 \(O_m^{(t)}\) 决定。

符号说明：\(\mathcal L_{\mathrm{D-freeze}}\) 是 protected atoms 稳定项；\(\mathcal P_t\) 是当前 protected atom 集合；\(\omega_m\) 是 atom \(m\) 的冻结权重；\(\bar d_m^\star\) 是当前任务空间中的 atom 快照目标，按是否启用 feature transport 选择 \(\tilde d_m^\star\) 或 \(d_m^\star\)。

### 5.5.1 状态感知 base-drift 正则

D-FSCIL 使用 \(\alpha\|M-M_0\|_F^2\) 约束增量字典不要远离 base 字典。LifeGAD 吸收这一“有限适配”原则，但改为 atom 状态与来源元数据共同感知的版本，避免新 grow atoms 被强行拉回 base 空间。

设 \(\mathcal B_0\) 为 base task 后已有的 atoms，\(d_m^{(0)}\) 为其 base 快照：

```math
\mathcal L_{\mathrm{base-drift}}
=
\sum_{m\in\mathcal B_0}
\rho(s_m,o_m)\|d_m-d_m^{(0)}\|_2^2.
```

符号说明：\(\mathcal L_{\mathrm{base-drift}}\) 是 base atoms 的轻量回拉项；\(\mathcal B_0\) 是 base task 后已经存在的 atom 索引集合；\(d_m^{(0)}\) 是 atom \(m\) 的 base 快照；\(\rho(s_m,o_m)\) 是由生命周期状态和来源共同决定的回拉权重。

权重由生命周期状态和来源决定：

```math
\rho(\mathrm{protected},\cdot) \gg
\rho(\mathrm{plastic},\mathrm{base}) >
\rho(\mathrm{reserve},\mathrm{reserve}) \ge 0,
\qquad
\rho(s_m,\mathrm{grown})=0.
```

符号说明：\(\rho(\mathrm{protected},\cdot)\) 表示任意来源的 protected atom 使用较大回拉；\(\cdot\) 是通配符；\(\rho(s_m,\mathrm{grown})=0\) 表示由 residual 新增的 grown-origin atom 不被拉回 base 空间。

该项与 \(\mathcal L_{\mathrm{D-freeze}}\) 的区别是：\(\mathcal L_{\mathrm{D-freeze}}\) 保护已被旧类强依赖的 atoms，\(\mathcal L_{\mathrm{base-drift}}\) 则对尚未 protected 的 base atoms 提供轻量回拉，减少无 replay 训练中的整体坐标系漂移。

### 5.6 稀疏正则：TopK 与完整 p-annealing

默认主实现采用 TopK projection 或 top-k ridge，因其可复现且不会引入额外 λ 退火机制。若使用 p-annealing，必须实现 Karvonen 等工作中的 coefficient annealing；否则只能称为 simplified Lp annealing。

完整 p-annealing 写为：

```math
\Omega_{\mathrm{sparse}}(s)
=
\frac1{|\mathcal C_t|\,|\mathcal M_t^{act}|}
\sum_{c\in\mathcal C_t}
\sum_{m\in\mathcal M_t^{act}}
(|a_{c,m}|+\epsilon)^{p_s}.
```

符号说明：\(\Omega_{\mathrm{sparse}}(s)\) 是不含权重的稀疏 penalty；\(\sum_{c\in\mathcal C_t}\sum_{m\in\mathcal M_t^{act}}\) 只对已见类别和 active atoms 求和；\(|\mathcal M_t^{act}|\) 是 active atom 数；\(p_s\) 是当前步的 \(L_p\) 指数。总目标中只在外部乘一次 \(\lambda_s\)，避免双重权重。

训练早期 \(p_s=1\)，随后逐步退火到：

```math
p_{\mathrm{end}}\in[0.2,0.7].
```

符号说明：\(p_{\mathrm{end}}\) 是 p-annealing 结束时的目标指数；从 \(p_s=1\) 退火到小于 1 的值会更强地鼓励稀疏。

系数退火用最近 \(q\) 个 batch 的激活队列保持稀疏惩罚强度局部恒定：

```math
A_q(p)=\sum_{j=s-q+1}^{s}\sum_i f_i(x_j)^p,
```

符号说明：\(A_q(p)\) 是最近 \(q\) 个 batch 上的激活强度统计；\(j\) 是 batch 或训练步索引；\(f_i(x_j)\) 表示第 \(j\) 个 batch 中第 \(i\) 个被惩罚激活量；\(p\) 是用于计算该统计的指数。

```math
\lambda_{s+1}
\leftarrow
\lambda_s
\frac{A_q(p_s)}
{A_q(p_{s+1})+\epsilon}.
```

符号说明：该式更新下一步稀疏权重 \(\lambda_{s+1}\)；\(p_s\) 与 \(p_{s+1}\) 是退火前后的指数；分式用于抵消 \(p\) 改变导致的惩罚尺度突变。

若未实现上式，实验表中必须把该变体标为 `simplified Lp`，不得作为默认 p-annealing 报告。

若报告 full p-annealing，还需给出收敛与稀疏质量诊断：最终 \(p_{\mathrm{end}}\) 附近的平均 active atoms、support Jaccard 稳定性、\(\lambda_s\) 退火曲线，以及与 TopK 的同容量准确率对比。否则 p-annealing 只作为 appendix exploratory variant。

### 5.7 字典互相干

```math
\mathcal L_{\mathrm{coh}}
=
\frac1{|\mathcal M_t^{act}|(|\mathcal M_t^{act}|-1)+\epsilon}
\sum_{\substack{i,j\in\mathcal M_t^{act}\\i\ne j}}
\max(0,|\hat d_i^\top \hat d_j|-\gamma)^2.
```

符号说明：\(\mathcal L_{\mathrm{coh}}\) 是字典互相干惩罚；\(i,j\) 只遍历 active atom 索引；\(|\hat d_i^\top\hat d_j|\) 是两个归一化 atoms 的绝对余弦相似度；\(\gamma\) 是允许的最大相干阈值；分母中的 \(\epsilon\) 处理 active atom 数小于 2 的退化情形。

### 5.8 Proxy old-logit 校准

无 replay 条件下旧类没有正样本，因此 LifeGAD 使用旧类 anchors 构造 prototype-neighborhood distillation，而不是保存所有 \(O(C^2)\) pairwise distribution。

对每个旧类 \(c\)，定义 anchor 邻域：

```math
\mathcal N_k(c)=
\mathrm{TopK}_{j\in\mathcal C_{t-1}\setminus\{c\}}
\langle \bar q_c^\star,\bar q_j^\star\rangle.
```

符号说明：\(\mathcal N_k(c)\) 是旧类 \(c\) 的 anchor 邻域；\(\mathrm{TopK}\) 选出与当前任务空间 anchor \(\bar q_c^\star\) 最相似的 \(k\) 个其他旧类；\(\mathcal C_{t-1}\setminus\{c\}\) 表示排除自身类别。Frozen-feature setting 中 \(\bar q=q\)，feature-adapted setting 中 \(\bar q=\tilde q\)。

历史分布和当前分布定义为：

```math
\pi_c^\star
=
\mathrm{softmax}
\left(
\left\{\frac{\langle \bar q_c^\star,\bar q_j^\star\rangle}{\tau_p}\right\}_{j\in\mathcal N_k(c)}
\right),
```

符号说明：\(\pi_c^\star\) 是当前任务空间中的旧 anchor 邻域目标分布；\(\tau_p\) 是 proxy distillation 温度；分布维度等于邻域大小 \(k\)。使用 \(\bar q^\star\) 可避免 feature-adapted setting 中把旧空间 anchor 与当前空间 prototype 混用。

```math
\pi_c
=
\mathrm{softmax}
\left(
\left\{\frac{\langle \bar q_c^\star,p_j\rangle}{\tau_p}\right\}_{j\in\mathcal N_k(c)}
\right).
```

符号说明：\(\pi_c\) 是当前模型中 current-space anchor \(\bar q_c^\star\) 对邻域类别当前 prototype \(p_j\) 的相似度分布；它与 \(\pi_c^\star\) 对齐以保持旧类相对几何。

Proxy loss：

```math
\mathcal L_{\mathrm{proxy}}
=
\frac1{|\mathcal C_{t-1}|}
\sum_{c\in\mathcal C_{t-1}}
\mathrm{KL}
\left(
\pi_c^\star
\|
\pi_c
\right)
+
\max(0,\xi+\max_{n\in\mathcal Y_t}\langle \bar q_c^\star,p_n\rangle-\langle \bar q_c^\star,p_c\rangle).
```

符号说明：该实现只需存 top-k 邻域或动态检索，复杂度为 \(O(Ck)\)，避免接近 \(O(C^2)\) 的不可控开销；\(C\) 表示当前已见类别数，\(k\) 表示每类邻域大小。

符号说明：\(\mathcal L_{\mathrm{proxy}}\) 包含 KL 邻域蒸馏和旧新 margin 约束；\(\mathrm{KL}(\pi_c^\star\|\pi_c)\) 衡量旧 anchor 目标分布与当前 prototype 分布差异；\(n\in\mathcal Y_t\) 遍历当前新类；\(\xi\) 是 current-space 旧类 anchor 与新类 prototype 之间的安全 margin。

#### 5.8.1 Selective anchor rumination

MoAL 的知识反刍强调：旧知识补强不应对所有旧类均匀施加，而应只强化当前模型尚未充分吸收的旧类关系。LifeGAD 将该思想改写为 classifier-head 侧的 **selective anchor rumination**，不保存旧样本，只使用旧 anchors、当前新类样本和风险掩码。

为避免样本分数和 pseudo feature 分数混用，先统一定义：对任意归一化特征向量 \(v\)，

```math
S_c(v)=\frac{\langle v,p_c\rangle}{\tau},
\qquad
S_c(x)=S_c(z_t(x)).
```

符号说明：\(v\) 是任意已归一化的特征向量，可是真实样本特征、anchor 或 pseudo feature；\(S_c(v)\) 是向量 \(v\) 对类别 \(c\) 的分数；\(S_c(x)\) 是把样本 \(x\) 映射成当前特征 \(z_t(x)\) 后的分数。

旧类 \(c\) 在向量 \(v\) 上的 one-vs-rest margin 为：

```math
\mathrm{margin}_c(v)
=
S_c(v)-\max_{j\in\mathcal C_t,j\ne c}S_j(v).
```

符号说明：\(\mathrm{margin}_c(v)\) 是类别 \(c\) 相对所有非 \(c\) 类的 one-vs-rest margin；值越大表示 \(v\) 越稳定地被判为类别 \(c\)。

本文把 margin drop 定义为旧 anchor margin 与当前 prototype margin 的差：

```math
\Delta_{\mathrm{margin}}(c)
=
\mathrm{margin}_c(\bar q_c^\star)
-
\mathrm{margin}_c(p_c).
```

符号说明：\(\Delta_{\mathrm{margin}}(c)\) 衡量旧类 \(c\) 当前 prototype 相对旧 anchor 的 margin 下降量；\(\bar q_c^\star\) 是当前空间 anchor，\(p_c\) 是当前 prototype。

因此 \(\Delta_{\mathrm{margin}}(c)>0\) 表示当前 prototype 相对旧 anchor 的 margin 下降；触发风险集合时使用 “大于阈值”，而不是把它写成负 margin。

atom 级旧类 margin drop 聚合为它所支撑旧类中的最大下降：

```math
\Delta_{\mathrm{old\text{-}margin}}(m)
=
\max_{c:m\in\mathcal S_c^\star}
\Delta_{\mathrm{margin}}(c).
```

符号说明：\(\Delta_{\mathrm{old\text{-}margin}}(m)\) 把 atom \(m\) 所支撑旧类中的最大 margin 下降作为 atom 级风险；索引条件 \(c:m\in\mathcal S_c^\star\) 表示只在历史 support 包含 atom \(m\) 的旧类上取最大值。

若某个 atom 不在任何旧类 support 中，则该量不参与 stale-binding 判定。

先定义需要反刍的旧类集合：

```math
\mathcal R_t^{old}
=
\left\{
c\in\mathcal C_{t-1}:
\mathrm{PAD}_c>\theta_{\mathrm{pad}}
\ \mathrm{or}\
\Delta_{\mathrm{margin}}(c)>\theta_{\mathrm{mar}}
\ \mathrm{or}\
\frac1{|\mathcal S_c^\star|+\epsilon}
\sum_{m\in\mathcal S_c^\star}
\max(0,\mathrm{LowerCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_{m}^{err}))
>\theta_{\mathrm{rum}}
\right\}.
```

符号说明：\(\mathrm{PAD}_c=1-\langle p_c,\bar q_c^\star\rangle\)，\(\bar q_c^\star\) 为当前任务特征空间中的旧 anchor 目标：feature-adapted setting 使用 \(\tilde q_c^\star\)，frozen-feature setting 使用 \(q_c^\star\)。对每个风险旧类 \(i\)，选取与其 anchor 最相似的新类 \(j(i)\)：

符号说明：\(\mathcal R_t^{old}\) 是任务 \(t\) 中需要 selective rumination 的旧类集合；\(\theta_{\mathrm{pad}}\)、\(\theta_{\mathrm{mar}}\)、\(\theta_{\mathrm{rum}}\) 分别是 PAD、margin drop 和 atom-error association proxy 的触发阈值；\(\mathrm{LowerCI}^{raw}_{m,t}\) 是 atom \(m\) 当前任务原始 PMI 置信区间下界；\(\mathcal S_c^\star\) 是旧类 \(c\) 的历史 support。若 \(\mathcal S_c^\star=\emptyset\)，空和为 0，分母由 \(\epsilon\) 防止除零，因此该 PMI 平均项不会触发 rumination。

```math
j(i)=
\arg\max_{j\in\mathcal Y_t}
\langle \bar q_i^\star,\hat\mu_j\rangle.
```

符号说明：\(j(i)\) 是与风险旧类 \(i\) 的 anchor 最相似的新类索引；\(\hat\mu_j\) 是新类 \(j\) 的归一化类均值。

用当前新类样本的类内残差生成旧类 pseudo feature：

```math
\tilde z_{i|j}(x)
=
\frac{
z_t(x)-\hat\mu_j+\bar q_i^\star
}{
\|z_t(x)-\hat\mu_j+\bar q_i^\star\|_2+\epsilon
},
\qquad x\in\mathcal D_t^{j(i)}.
```

符号说明：\(\tilde z_{i|j}(x)\) 是为旧类 \(i\) 临时构造的 pseudo feature；\(x\) 来自相似新类 \(j(i)\)；\(z_t(x)-\hat\mu_j\) 是新类样本的局部类内残差；把中心替换为 \(\bar q_i^\star\) 后再归一化。

这比直接平移 prototype 更保守：它只迁移新类样本围绕其类均值的局部残差，把中心替换为旧类 anchor。然后用当前 head 预测这些 pseudo features，并只选取仍然出错或 margin 不足的样本：

```math
M_{\mathrm{rum}}(\tilde z,i)
=
\mathbf 1
\left[
\arg\max_{c\in\mathcal C_t} S_c(\tilde z)\ne i
\ \mathrm{or}\
S_i(\tilde z)-\max_{c\ne i}S_c(\tilde z)<\delta_{\mathrm{rum}}
\right].
```

符号说明：\(M_{\mathrm{rum}}(\tilde z,i)\) 是反刍样本掩码；只有当 pseudo feature \(\tilde z\) 未被判为目标旧类 \(i\)，或其旧类 margin 低于 \(\delta_{\mathrm{rum}}\) 时取 1；\(\delta_{\mathrm{rum}}\) 是 rumination margin 阈值。

反刍损失为：

```math
\mathcal L_{\mathrm{rum}}
=
\frac{
\sum_{i\in\mathcal R_t^{old}}
\sum_{\tilde z\in\tilde{\mathcal Z}_i}
M_{\mathrm{rum}}(\tilde z,i)
\ \mathrm{CE}(\{S_c(\tilde z)\}_{c\in\mathcal C_t},i)
}{
\sum_{i,\tilde z}M_{\mathrm{rum}}(\tilde z,i)+\epsilon
}.
```

符号说明：\(\tilde{\mathcal Z}_i\) 不作为 replay buffer 保存，只在当前任务训练阶段临时生成。若风险集合为空或掩码全零，则 \(\mathcal L_{\mathrm{rum}}=0\)。该模块吸收 MoAL 的“选择性强化未吸收旧知识”原则，但不引入 MoAL 的解析递归分类头。

符号说明：\(\tilde{\mathcal Z}_i\) 是为旧类 \(i\) 临时生成的 pseudo feature 集合；分子只对被掩码选中的 pseudo features 施加交叉熵；分母是有效 pseudo features 数量，\(\epsilon\) 保证没有有效样本时不会除零。

### 5.9 总目标

```math
\mathcal L_t
=
\mathcal L_{\mathrm{ce}}
+\lambda_n\mathcal L_{\mathrm{new}}
+\lambda_p\mathcal L_{\mathrm{p-anchor}}
+\lambda_a\mathcal L_{\mathrm{a-anchor}}
+\lambda_D\mathcal L_{\mathrm{D-freeze}}
+\lambda_b\mathcal L_{\mathrm{base-drift}}
+\lambda_s\Omega_{\mathrm{sparse}}
+\lambda_{\mathrm{coh}}\mathcal L_{\mathrm{coh}}
+\lambda_{\mathrm{proxy}}\mathcal L_{\mathrm{proxy}}
+\lambda_{\mathrm{rum}}\mathcal L_{\mathrm{rum}}
+\lambda_{\mathrm{mv}}\mathcal L_{\mathrm{res-mv}}
+\lambda_T\mathcal L_{\mathrm{transport}}.
```

符号说明：\(\mathcal L_{\mathrm{res-mv}}\) 只在 base task 的 reserve preparation 阶段启用；增量阶段和推理阶段均置为 0。为降低搜索空间，主实验固定大部分 \(\lambda\)，只对 \(\lambda_D\)、\(\lambda_b\)、\(\lambda_{\mathrm{rum}}\)、\(\lambda_{\mathrm{mv}}\)、growth 阈值和稀疏机制做有限敏感性分析。Frozen-feature 主协议中 \(T_t=I\)，因此 \(\lambda_T\mathcal L_{\mathrm{transport}}\) 自动为 0；只有 plug-in feature-adapted setting 才启用。

符号说明：\(\mathcal L_t\) 是任务 \(t\) 的总训练目标；所有 \(\lambda_{\cdot}\) 都是对应损失项的非负权重；\(\lambda_n,\lambda_p,\lambda_a,\lambda_D,\lambda_b,\lambda_s,\lambda_{\mathrm{coh}},\lambda_{\mathrm{proxy}},\lambda_{\mathrm{rum}},\lambda_{\mathrm{mv}},\lambda_T\) 分别控制新类对齐、prototype anchor、coefficient anchor、atom freeze、base drift、稀疏 penalty \(\Omega_{\mathrm{sparse}}\)、互相干、proxy 校准、反刍、reserve multi-view 和 transport 损失。

---

## 6. 生命周期动作

<!-- 修订说明：定义 clone 扰动、修正 grow residual 空间、补充 prune/remapping 不变量，并把 quarantine/clone 明确为经验机制；本轮加入 support 空集 guard 和 stale-binding new-blocked flag。 -->

每个任务结束后，在当前任务 calibration split 上计算：

- atom usage \(U_{m,\mathrm{pred}}^{(t)}\) 与诊断用 \(U_{m,\mathrm{true}}^{(t)}\)；
- old-class support \(O_m^{(t)}\)；
- 当前任务原始 error PMI \(\widehat{\mathrm{PMI}}_{m,t}^{err}\)、其置信区间 \(\mathrm{CI}_{m,t}^{raw}\)，以及仅用于监控的 persistent-atom EMA；
- new-class residual \(r_c^{cos}\) 与 \(r_c^{vec}\)；
- old-new prototype margin。

状态迁移只在统计量通过最小样本数和置信过滤时执行。若某个 atom 的 PMI 计数不满足 \(n_{\min}\)，则不允许由 PMI 单独触发 grow / clone / quarantine / stale-binding；该 atom 只能由 residual、usage 和 old support 规则触发保守动作，并计入 \(\mathrm{PMIInvalidRate}_t\)。

### 6.1 Freeze

若 atom 支撑旧类且当前任务错误风险低：

```math
O_m^{(t)}>\theta_o
\quad\text{and}\quad
\mathrm{UpperCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_m^{err})<\theta_e,
```

符号说明：\(\theta_o\) 是旧类支撑强度阈值；\(\mathrm{UpperCI}^{raw}_{m,t}\) 是 atom \(m\) 当前任务原始 PMI 置信区间上界；\(\theta_e\) 是错误关联风险阈值。该条件表示“旧类依赖强且错误风险上界仍低”。

则：

```math
s_m=\mathrm{protected}.
```

符号说明：该动作把 atom \(m\) 的生命周期状态 \(s_m\) 设置为 `protected`，后续训练中冻结或强约束该 atom。

首次进入 protected 时保存 \(d_m^\star\)，后续使用 \(\mathcal L_{\mathrm{D-freeze}}\) 或直接冻结。

### 6.2 Grow

若新类稀疏残差高且 ridge shadow residual 也高：

```math
r_c^{cos}>\theta_r
\quad\text{and}\quad
r_c^{dense}>\theta_{r,dense},
```

符号说明：\(\theta_r\) 是稀疏余弦残差阈值；\(\theta_{r,dense}\) 是 dense ridge residual 阈值；二者同时高表示当前字典在稀疏和 dense 表示下都难以覆盖新类 \(c\)。

或新类主要支撑的 error PMI 高且统计可靠：

```math
\frac1{|\mathcal S_c|+\epsilon}
\sum_{m\in\mathcal S_c}
\mathrm{LowerCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_m^{err})
>
\theta_e,
```

符号说明：\(\mathcal S_c=\{m\in\mathcal M_t^{act}:|a_{c,m}|>\eta_a\}\) 是当前类别 \(c\) 的 active support；\(\mathrm{LowerCI}^{raw}_{m,t}\) 是 PMI 置信区间下界；该条件表示新类所依赖 atoms 的错误关联下界平均值仍超过风险阈值。若 \(\mathcal S_c=\emptyset\)，空和为 0，分母由 \(\epsilon\) 防止除零，因此该 PMI 平均项不会触发 grow。

则新增 atoms。新增 atom 由重构残差方向初始化：

```math
d_{\mathrm{new}}
=
\hat r_c^{vec}
=
\frac{
\hat\mu_c-a_c\hat D_t
}{
\|\hat\mu_c-a_c\hat D_t\|_2+\epsilon
}.
```

符号说明：\(d_{\mathrm{new}}\) 是新增 atom 的初始化方向；它直接使用新类 \(c\) 的归一化重构残差 \(\hat r_c^{vec}\)，使新增 atom 指向当前字典最难表达的方向。

新增后为所有旧类和新类系数补零维度，再对当前新类局部重解 top-k ridge。

若 \(r_c^{cos}>\theta_r\) 但 \(r_c^{dense}\le\theta_{r,dense}\)，说明当前字典在 dense L2 意义下仍能表示新类，问题更可能来自稀疏预算过紧或 top-k 选择错误。此时先执行：

```math
k\leftarrow \min(k+\Delta k,k_{\max}),
```

符号说明：该式临时放宽每类 active atom 数；\(\Delta k\) 是每次增加的预算；\(k_{\max}\) 是最大允许 active atom 数；\(\leftarrow\) 表示原地更新。

并重解 top-k ridge；只有放宽后 residual 仍高，才允许 grow。该规则吸收 D-FSCIL 的 L2 稳定性诊断，用来减少不必要扩容。

### 6.3 Clone

如果高风险 atom 同时被旧类强依赖：

```math
\mathrm{LowerCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_m^{err})>\theta_e
\quad\text{and}\quad
O_m^{(t)}>\theta_o,
```

符号说明：该 clone 条件表示 atom \(m\) 的错误关联风险下界高，同时又被旧类强依赖；因此不能直接删除或大幅改动原 atom。

不能直接删除或大幅更新。此时执行 clone。设 \(\xi\) 是与 \(\hat d_m\) 正交的单位噪声，残差正交分量为：

```math
r_{c,m}^{\perp}
=
(I-\hat d_m\hat d_m^\top)r_c^{vec}.
```

符号说明：\(r_{c,m}^{\perp}\) 是新类残差 \(r_c^{vec}\) 中与 atom \(m\) 正交的分量；\(I\) 是 \(d\times d\) 单位矩阵；\((I-\hat d_m\hat d_m^\top)\) 是去除 \(\hat d_m\) 方向投影的正交投影矩阵。

clone atom 初始化为：

```math
d_{m'}
=
\frac{
\hat d_m
+\alpha\frac{r_{c,m}^{\perp}}{\|r_{c,m}^{\perp}\|_2+\epsilon}
+\sigma\xi
}{
\left\|
\hat d_m
+\alpha\frac{r_{c,m}^{\perp}}{\|r_{c,m}^{\perp}\|_2+\epsilon}
+\sigma\xi
\right\|_2+\epsilon
}.
```

符号说明：\(d_{m'}\) 是 clone 出来的新 atom；\(m'\) 是新 atom 索引；\(\alpha\) 控制沿残差正交方向偏移的幅度；\(\sigma\) 控制随机正交噪声强度；\(\xi\) 是与 \(\hat d_m\) 正交的单位随机向量。

原 atom \(m\) 保留给旧类，新 atom \(m'\) 作为 plastic atom 供新类调整。第一版若不实现 clone，则主文不把 clone 写入核心贡献，只作为 appendix 变体。

Clone 与 grow 一样必须维护维度不变量：创建 \(m'=M_t+1\) 后，所有 \(a_c\)、\(a_c^\star\) 和相关 support metadata 先在新维度补零；旧类默认保持 \(a_{c,m'}=0\)，只有当前新类或后续任务能选择 \(m'\)。原 atom \(m\) 的旧类 support 与 protected snapshot 不被转移到 \(m'\)。

### 6.4 Quarantine

如果 atom 错误风险高但旧类支撑弱：

```math
\mathrm{LowerCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_m^{err})>\theta_e
\quad\text{and}\quad
O_m^{(t)}<\theta_o,
```

符号说明：该 quarantine 条件表示 atom \(m\) 与错误稳定相关，但旧类支撑较弱，因此可限制其后续使用而不伤害强旧类 support。

则：

```math
s_m=\mathrm{quarantined}.
```

符号说明：该动作把 atom \(m\) 的状态设为 `quarantined`；它仍保留在字典中，但新类初始化和后续更新会限制或降权使用它。

Quarantined atom 不直接删除，但限制新类初始化时选择它，并可对其 contribution 降权。该动作是风险控制启发式，需要通过 `w/o quarantine` 消融和 deletion/insertion 曲线验证。

### 6.5 Prune

Reserve atoms 使用 aging 机制。若 reserve atom 连续 \(L_{\mathrm{res}}\) 个任务未被任何新类 top-k 初始化、growth 或 support 使用：

```math
s_m=\mathrm{reserve}
\quad\text{and}\quad
A_m^{res}\ge L_{\mathrm{res}},
```

符号说明：\(A_m^{res}\) 是 reserve atom \(m\) 未被有效使用的 aging 计数；\(L_{\mathrm{res}}\) 是 reserve atom 允许闲置的最大任务数。

则将其转为：

```math
s_m=\mathrm{inactive}
```

符号说明：该动作把 atom \(m\) 标为 `inactive`，默认只做逻辑停用，不立即物理删除张量维度。

或放回 reserve candidate pool 重新初始化。该规则避免 D-FSCIL 式前向兼容预留在长序列中变成永久无效容量。

如果 atom 长期低利用：

```math
U_m^{(t)}<\theta_u
\quad\text{for }L\text{ consecutive tasks},
```

符号说明：\(U_m^{(t)}\) 表示 atom \(m\) 的使用率，可按实现选择 \(U_{m,\mathrm{pred}}^{(t)}\)；\(\theta_u\) 是低使用率阈值；\(L\) 是连续满足低使用条件的任务数。

且不支撑旧类：

```math
O_m^{(t)}<\theta_o
\quad\text{and}\quad
m\notin\bigcup_{c\in\mathcal C_{t-1}}\mathcal S_c^\star,
```

符号说明：该条件要求 atom \(m\) 既没有足够旧类支撑，也不属于任何旧类历史 support；\(\notin\) 表示“不属于”；满足后才允许逻辑停用或物理 prune。

则：

```math
s_m=\mathrm{inactive}.
```

符号说明：这里的 `inactive` 与 reserve aging 中含义相同，表示该 atom 后续不参与训练、初始化和推理 support，除非被显式重新激活或重初始化。

真正从推理字典中移除前，需要构造 prune remapping \(\pi_{\mathrm{prune}}\)，同步更新 \(D_t\)、所有 \(a_c\)、\(a_c^\star\)、support sets 和 atom metadata。若不能保证该不变量，则只做逻辑 inactive，不物理删除。

符号说明：\(\pi_{\mathrm{prune}}\) 是物理删除 atoms 时的新旧索引映射；不用 \(\rho\) 表示 remapping，是为了避免与 base-drift 权重 \(\rho(s_m,o_m)\) 以及 PMI EMA 系数 \(\rho\) 混淆。

### 6.6 State-aware atom momentum

MoAL 的动量插值说明，持续适配不必依赖重蒸馏，也不应让参数在每个任务中完全覆盖旧状态。LifeGAD 只把这一原则迁移到 dictionary atoms：每轮训练得到临时更新 \(\bar d_m\) 后，按 atom 状态和来源元数据执行动量合并：

```math
d_m
\leftarrow
\frac{
\alpha_D(s_m,o_m)d_m
+
(1-\alpha_D(s_m,o_m))\bar d_m
}{
\|\alpha_D(s_m,o_m)d_m
+
(1-\alpha_D(s_m,o_m))\bar d_m\|_2+\epsilon
}.
```

符号说明：\(\bar d_m\) 是当前任务梯度训练得到的临时 atom 更新；\(\alpha_D(s_m,o_m)\in[0,1]\) 是按状态和来源设置的动量保留系数；\(\alpha_D\) 越大，越接近保留旧 atom \(d_m\)。

状态权重满足：

```math
\alpha_D(\mathrm{protected},\cdot)
\ge
\alpha_D(\mathrm{plastic},\mathrm{base})
>
\alpha_D(\mathrm{plastic},\mathrm{grown})
\ge
\alpha_D(\mathrm{reserve},\mathrm{reserve}),
\qquad
\alpha_D(\mathrm{quarantined},\cdot)=1.
```

符号说明：该不等式指定不同状态/来源的相对动量强度；`protected` 最稳定，grown-origin plastic atoms 更可塑；`quarantined` 设置为 1 表示不接受当前任务更新。

默认实现中，protected atoms 仍可直接冻结；state-aware momentum 是 soft-freeze 变体，用来测试“有限适配”是否优于硬冻结。若该模块写入摘要或主贡献，主文必须报告 `hard-freeze vs state-aware momentum` 和 `momentum w/o protected freeze` 消融；否则应降为 appendix 机制。该模块不更新 PTM adapter，因此不会改变 LifeGAD 的 head-only 主定位。

### 6.7 Stale-binding refresh

仅用 protected freeze 可能把过时绑定永久保留下来。若某个 atom 被旧类强依赖，同时在当前任务中稳定关联错误，LifeGAD 不应直接删除它，而应把它标记为 stale binding 并进入刷新队列：

```math
\mathrm{stale}_m
=
\mathbf 1
\left[
O_m^{(t)}>\theta_o
\ \mathrm{and}\
\mathrm{LowerCI}^{raw}_{m,t}(\widehat{\mathrm{PMI}}_m^{err})>\theta_e
\ \mathrm{and}\
\Delta_{\mathrm{old\text{-}margin}}(m)>\theta_{\mathrm{oldmar}}
\right].
```

符号说明：\(\mathrm{stale}_m\) 是 stale-binding 指示变量；\(\theta_{\mathrm{oldmar}}\) 是 atom 支撑旧类的 margin drop 阈值；该条件同时要求旧类依赖强、错误关联高、旧类 margin 下降明显，才进入 refresh 队列。

若 \(\mathrm{stale}_m=1\)，首先保留其旧类保护语义，并阻断新类继续选择：

```math
s_m\leftarrow \mathrm{protected},
\qquad
b_m^{new\text{-}blocked}\leftarrow 1.
```

符号说明：该更新表示 stale atom 仍被旧类强依赖，因此不把它改成互斥的 `quarantined` 状态；\(b_m^{new\text{-}blocked}=1\) 单独表达“禁止新类继续绑定该 atom”的使用限制。

刷新策略按风险从低到高执行：

1. **coefficient reallocation**：限制新类继续选择该 atom，重解新类 top-k ridge；
2. **selective anchor rumination**：对依赖该 atom 的旧类加入 \(\mathcal R_t^{old}\)，只补强仍出错或 margin 不足的 pseudo old features；
3. **clone**：若旧类依赖强且新类仍需要相近方向，则保留原 atom 给旧类，新增 plastic clone 给新类；
4. **quarantine**：仅当刷新后旧类支撑也降到 \(O_m^{(t)}<\theta_o\) 或不再属于任何旧类 support 时，才允许改为 `quarantined`；否则保持 `protected` 并继续使用 \(b_m^{new\text{-}blocked}=1\)。

这一步吸收 MoAL 的“遗忘过时知识”思想，但 LifeGAD 的遗忘对象是 atom-class binding 和新类系数选择，而不是 PTM 表征本身。所有 stale-binding 事件都需要在 appendix 报告数量、触发原因和对 old-new HM 的影响。

---

## 7. 训练与推理流程

<!-- 修订说明：加入 held-out calibration/evaluation split、状态机不变量和 growth/clone 后系数补零/重映射流程。 -->

### 7.1 Base task

1. 提取 base task PTM 特征；
2. 估计 whitening / standardization 变换 \(B\)；
3. 计算 base 类均值；
4. 用 class means / k-means / multi-view pseudo-class reserve 初始化 \(D_0\)，并为 base atoms 保存 \(d_m^{(0)}\)；
5. 用 top-k ridge 初始化 base 类系数；
6. **Stage B0: geometry warmup**。先只用 base CE、base prototype 几何对齐、稀疏和互相干约束训练 \(D_0,A_0\)，不启用 \(\mathcal L_{\mathrm{res-mv}}\)，避免随机 reserve 信号在分类几何稳定前主导训练；
7. **Stage B1: reserve preparation**。启用 multi-view reserve candidates 与 \(\mathcal L_{\mathrm{res-mv}}\)，继续训练 \(D_0,A_0\) 和 reserve atoms，使预留区域对线性 mixup 与 mask/subspace mixing 两种虚拟视角保持一致；
8. Stage B0:B1 的 epoch 比例作为超参数或自适应切换策略。M2SD 的 20:80 只作为候选消融点，不作为 LifeGAD 默认事实；
9. 训练结束后丢弃 virtual probes、mask 分支和任何临时 reserve teacher，只保留 \(D_0,A_0\)、atom states 与 metadata；
10. 保存每个 base 类的 \((a_c^\star,q_c^\star,\mathcal S_c^\star)\)；
11. 根据 \(O_m\)、usage 和 support 初始化 atom states。

### 7.2 增量任务 \(t\)

1. 若处于 feature-adapted setting，先用当前任务 paired features 估计 \(T_t\)，将工作字典初始化为 \(D_t^{init}=\mathrm{RowNorm}(T_t(D_{t-1}))\)，并将旧 anchors / protected atom snapshots 传输到当前特征空间；
2. 若字典已扩容，为旧类系数和 anchors 补零维度；
3. 冻结旧类系数和 protected atoms，或启用 state-aware atom momentum 作为 soft-freeze 变体；
4. 计算新类均值，先用 top-k ridge 初始化新类系数，再计算 ridge shadow residual；
5. warmup 训练新类系数和 plastic atoms；
6. 在 \(\mathcal V_t^{cal}\) 上计算 residual、usage、PMI、PAD、old-new margin 及置信过滤；
7. 根据生命周期规则执行 grow / clone / quarantine / reserve aging / prune，并把 stale-binding atoms 放入 refresh queue；
8. 对 grow/clone 后的新维度重解当前新类 top-k ridge，对 stale-binding 相关新类执行 coefficient reallocation；
9. 生成 selective anchor rumination 的风险旧类集合 \(\mathcal R_t^{old}\) 和临时 pseudo features；
10. 继续训练完整 LifeGAD 损失，包括可选的 \(\mathcal L_{\mathrm{rum}}\) 和 \(\mathcal L_{\mathrm{transport}}\)；
11. 若启用 state-aware atom momentum，用 \(\alpha_D(s_m,o_m)\) 合并临时 atom 更新；
12. 在 \(\mathcal V_t^{eval}\) 上评估状态迁移、反刍和刷新是否改善下游指标；
13. 保存新类 anchors；
14. 更新 atom states 和 metadata。

### 7.3 推理

推理阶段不需要 task-ID：

1. 提取并归一化特征 \(z(x)\)；
2. 计算所有已见类 prototype；
3. 全类别余弦分类：

```math
\hat y=\arg\max_{c\in\mathcal C_t}
\frac{\langle z(x),p_c\rangle}{\tau}.
```

符号说明：推理不使用 PMI、validation labels 或 lifecycle 触发器。

符号说明：推理时只需要样本特征 \(z(x)\)、所有已见类 prototype \(p_c\)、温度 \(\tau\) 和类别集合 \(\mathcal C_t\)；PMI、\(\mathcal V_t^{cal}\)、\(\mathcal V_t^{eval}\)、状态迁移阈值只在训练或任务结束诊断时使用。

---

## 8. 与最相关 baseline 的逐环节对比

<!-- 修订说明：修正 HC-SOINN 机制与协议描述，补充 LifeGAD 与强 PTM-CIL 系统的公允边界。 -->

### 8.1 LifeGAD vs Original GAD

| 环节 | GAD | LifeGAD |
|---|---|---|
| 字典容量 | 固定或简单消融 | 由 residual、atom-error association proxy、usage 触发增长或限制 |
| atom 状态 | 无状态 | protected / plastic / reserve / quarantined / inactive |
| 旧类稳定 | prototype + coefficient anchor | class anchor + protected atom + support freeze |
| 新类适配 | 投影到固定字典 | 先复用，残差高则 grow；clone 为可选变体 |
| 错误诊断 | GADrift / CTSO | 当前任务 atom-error association proxy、signed atom risk、growth effectiveness |
| 稀疏机制 | L1 | TopK / top-k ridge；完整 p-annealing 为实现 λ 退火后的变体 |
| 主要风险 | 固定容量、atom collapse | 生命周期阈值和规则复杂，统计信号需校准 |

### 8.2 LifeGAD vs EASE

| 环节 | EASE | LifeGAD |
|---|---|---|
| 核心单元 | 任务 adapter 子空间 | 字典 atoms |
| 存储增长 | 每任务 adapter 增长 | 按风险触发 atom 增长 |
| 分类器 | prototype completion + subspace reweight | sparse dictionary prototypes |
| 旧知识 | adapter 和跨子空间原型 | protected atoms + anchors |
| 推理 | 多子空间相关机制，计算随 adapter 数增长 | 一次全类 prototype matching |
| 互补性 | 强 feature adaptation | 强 classifier geometry |

LifeGAD 不应简单声称“替换 EASE prototype completion”。若做 hybrid，需要说明字典是在单个 adapter 子空间、拼接空间还是最终融合特征上工作。

### 8.3 LifeGAD vs HC-SOINN

| 环节 | HC-SOINN | LifeGAD |
|---|---|---|
| 类表示 | 每类多节点拓扑图 \(G_c=(V_c,E_c)\) | 每类稀疏字典组合 prototype |
| 初始化 | 层次聚类初始化 | class means / k-means / reserve / residual centers |
| 精炼 | 球面 SOINN + 边年龄维护 | 字典 atom 更新 + lifecycle 状态 |
| 推理 | 全局类均值 + 局部最近子原型双视角 | 单输出 prototype matching |
| 漂移处理 | STAR 点轨迹追踪 | anchors + protected atoms |
| 容量 | 每类多节点拓扑表示，节点数由聚类/SOINN 决定 | 全局共享 atoms 风险驱动增长 |
| 优势 | 类内多模态和非线性流形建模强 | 共享语义复用、轻量、task-agnostic |
| 弱点 | 节点/锚点存储与拓扑维护 | 仍是单输出 prototype，类内多模态有限 |

HC-SOINN 公开主表协议中 CUB-200 为 20 tasks × 10 classes，ImageNet-R 为 40 tasks × 5 classes；LifeGAD 默认是 Base100 + 10 sessions × 10 classes。公开结果不能直接混入主表。主表必须统一重跑协议，公开数值只能作为背景 context。

### 8.4 LifeGAD vs 强 PTM-CIL 系统

EASE、SD-LoRA、CL-LoRA、DIA、MoAL、MiN 等方法大多是 feature adaptation 或完整 PTM-CIL 系统，不是纯 classifier head。LifeGAD 的公允定位是：

- 作为 frozen-feature classifier head，与 NCM、SimpleCIL、ACIL/RanPAC、GPA、GAD 等同特征方法比较；
- 作为 plug-in readout，接到 EASE/SD-LoRA/DIA 等强特征系统后验证是否改善 classifier geometry；
- 不在未统一协议时声称对 full PTM-CIL 系统的直接 SOTA。

对 MoAL 的吸收边界需要单独说明：LifeGAD 不继承 MoAL 的 adapter weight interpolation、随机缓冲解析头或 RLS 递归更新；这些属于 full PTM-CIL / analytical-learning 系统。LifeGAD 只吸收其四个可迁移原则：feature 漂移时 anchor 需要传输，适配应受动量约束，旧知识应选择性反刍，过时映射应进入刷新而非永久冻结。

### 8.5 LifeGAD vs D-FSCIL

D-FSCIL 是 LifeGAD 的重要近邻，但二者不应写成同一类方法。D-FSCIL 面向 FSCIL，在 base 阶段联合训练特征提取器、L2 字典和伪类原型，增量阶段冻结 backbone 并对整体字典做有限适配。LifeGAD 面向 PTM-CIL，把分类头写成稀疏共享字典 prototype，并在 atom 粒度维护状态、风险、容量和 metadata。

LifeGAD 吸收 D-FSCIL 的三个思想，但都改写为生命周期字典语境：

| D-FSCIL 思想 | LifeGAD 吸收方式 | 不直接继承的部分 |
|---|---|---|
| 伪类前向兼容 | pseudo-class reserve atoms，用 \(\rho_{\mathrm{res}}\) 控制预算 | 不预设未来新类总数，不把伪类当正式类别训练 |
| L2 协作表示稳定 | ridge shadow residual，用于判断是否真的需要 grow | 不替代稀疏系数和 support |
| \(\alpha\|M-M_0\|_F^2\) 有限适配 | state-aware base-drift regularizer | 不对新 grow atoms 做全局回拉 |

因此，D-FSCIL-style global-adapt GAD 必须作为消融 baseline：整体字典可适配、有 base drift，但没有 atom lifecycle。只有 LifeGAD 显著优于该 baseline，才能说明收益来自生命周期管理而不是普通字典适配。

### 8.6 LifeGAD vs M2SD

M2SD 面向 FSCIL 的 base-session 表示学习，核心是在训练 backbone 时用多混合自蒸馏扩展特征空间，并在增量阶段用简单 prototype append 更新分类器。LifeGAD 面向 frozen PTM-CIL 的 dictionary head，不训练 backbone，也不把分类器退化为普通类均值追加。

LifeGAD 只吸收 M2SD 的四个可迁移原则，并全部改写为字典侧机制：

| M2SD 启发 | LifeGAD 吸收方式 | 不直接继承的部分 |
|---|---|---|
| base 阶段为未来新类准备空间 | base Stage B1 的 reserve preparation | 不训练 PTM backbone |
| mixup + CutMix 双视角虚拟类一致性 | linear reserve + feature-mask reserve 的 \(\mathcal L_{\mathrm{res-mv}}\) | 不做图像级 CutMix，不把虚拟方向当真实新类 |
| Stage1/Stage2 阶段划分 | Stage B0:B1 比例消融或自适应切换 | 不把 20:80 写成跨数据集默认最优 |
| 训练时辅助结构推理时丢弃 | virtual probes 和 mask 分支只在 base 训练中存在 | 不增加推理分支或 task-ID 依赖 |

M2SD 的 ResNet block 注意力增强、BiFPN 临时分类器和原型追加策略只能作为表示学习或解耦分类器的背景参考，不作为 LifeGAD 主文机制。

---

## 9. 关键超参数

<!-- 修订说明：修正 active atoms 比例矛盾，补充 PMI 可靠性、p-annealing λ 退火和 memory 上界相关超参数。 -->

| 超参数 | 默认值 | 选择依据 |
|---|---:|---|
| \(m=M_{\mathrm{cls}}/C_{\mathrm{base}}\) | 5 | 与 GAD/DPR 字典规模先验一致，需消融 `{3,5,8}` |
| active atoms \(k\) | 5 or 10 | 固定绝对稀疏预算；不再声称固定比例 |
| 比例式 active atoms | \(k_t=\lceil rM_t\rceil\) | 仅在比例预算变体使用，\(r\in[0.05,0.15]\) |
| \(\tau\) | 0.05 | DPR 余弦 prototype readout 主线常用设置 |
| \(p_{\mathrm{end}}\) | 0.2-0.7 | 仅在完整 p-annealing 中使用，必须配 λ 退火 |
| \(\lambda_{\mathrm{sh}}\) | 0.1 或按条件数校准 | ridge shadow residual 的 L2 稳定系数 |
| \(\theta_{r,dense}\) | dense residual P90 或 MAD | 判断字典覆盖是否真的不足 |
| \(\gamma\) | 0.2 | 控制 atom mutual coherence |
| \(\theta_r\) | base residual P90 或 median+MAD | 避免测试集调参 |
| \(\theta_e\) | PMI CI lower bound > threshold | 统计触发高风险 atom |
| \(\theta_{\mathrm{oldmar}}\) | base / cal percentile | stale-binding refresh 的旧类 support margin drop 阈值 |
| \(n_{\min}\) | 20 或按样本量校准 | PMI 最小计数过滤 |
| \(\rho\) | 0.8-0.95 | 仅用于 persistent atoms 的 PMI EMA 监控，不用于触发 |
| \(\theta_{\mathrm{id}}\) | 0.98 或按 base 漂移校准 | atom EMA 历史是否可沿用的方向一致性阈值 |
| \(\theta_u\) | usage < 0.1% for 2 tasks | 防止误删短期未用 atoms |
| \(\rho_{\mathrm{res}}\) | 0.1-0.3 | \(M_{\mathrm{res}}=\lceil\rho_{\mathrm{res}}M_{\mathrm{cls}}\rceil\)，不对 \(M_0\) 自引用 |
| Stage B0:B1 | 20:80 / 50:50 / 80:20 / adaptive | base 几何 warmup 与 reserve preparation 的阶段比例 |
| \(\lambda_{\mathrm{mv}}\) | 0.1-1 | multi-view reserve consistency 总权重，只在 base Stage B1 启用 |
| \(\lambda_{\mathrm{vce}},\lambda_{\mathrm{vkl}}\) | fixed or small grid | 虚拟 soft target 与双视角一致性的相对权重 |
| \(\lambda_{\mathrm{align}}\) | 0.1-1 | reserve atom alignment 权重，确保梯度作用到预留 atoms |
| \(\tau_{\mathrm{mv}}\) | 1-4 | reserve consistency 的 KL 温度 |
| \(L_{\mathrm{res}}\) | 2-3 tasks | reserve atom aging 到 inactive 的等待期 |
| max growth / task | `min(0.1M_t, 2|Y_t|)` | 控制内存增长 |
| clone \(\alpha,\sigma\) | appendix search | clone 为可选变体，不作为 MVP 默认 |
| \(\lambda_D\) | 1-10 | protected atom 应强于普通 L2 |
| \(\lambda_b\) | 0.01-1 | base atoms 的轻量坐标系回拉 |
| \(\lambda_{\mathrm{proxy}}\) | 0.1-1 | 旧类校准辅助，不压过 CE |
| \(\lambda_{\mathrm{rum}}\) | 0.1-1 | 选择性旧类反刍，不压过当前任务 CE |
| \(\delta_{\mathrm{rum}}\) | validation percentile | 反刍 pseudo feature 的 margin 掩码 |
| \(\theta_{\mathrm{pad}},\theta_{\mathrm{rum}}\) | base / cal percentile | 风险旧类进入反刍集合的阈值 |
| \(\alpha_D(s_m,o_m)\) | state/origin-wise search | atom 动量系数，protected 最大，grown-origin plastic atoms 较小 |
| \(\lambda_T\) | 0 或 0.1-1 | frozen-feature 为 0；feature-adapted plug-in 才启用 |

若每任务增长 \(g_t\le \min(\eta M_t,\beta|\mathcal Y_t|)\)，则主文采用线性 budget cap：

```math
M_T\le M_0+\beta\sum_t|\mathcal Y_t|.
```

符号说明：指数界 \(M_T\le M_0(1+\eta)^T\) 只作为 worst-case 备注，不作为主文 memory fairness 口径。

符号说明：\(M_T\) 是最后一个任务后的字典 atom 总数；\(M_0\) 是 base task 后的初始字典大小；\(\beta\) 是每个新增类别最多允许增长的 atom 数系数；\(\sum_t|\mathcal Y_t|\) 是增量过程中所有任务新增类别数之和。若同时使用比例增长上限，\(\eta\) 表示每任务相对当前字典大小的最大增长比例。

实验需报告 dictionary、稀疏系数、prototype anchors、atom states 和 proxy metadata 的完整 memory。

---

## 10. 实验方案

<!-- 修订说明：重构协议与 baseline 分层，替换 LCE/GTP/PSD 等问题指标，压缩主文消融规模。 -->

### 10.1 数据集

| 数据集 | 类别数 | 选择理由 |
|---|---:|---|
| CIFAR-100 | 100 | 标准 CIL 基准，便于多 seed 和大量消融 |
| CUB-200 | 200 | 细粒度鸟类，检验单 prototype、容量和 atom 语义纯度 |
| ImageNet-R | 200 | 有 domain/rendition shift，检验负迁移和字典生命周期 |

可选 appendix：

- ImageNet-A：更强鲁棒性压力测试；
- ObjectNet：偏置控制和 OOD robustness；
- Stanford Cars / Aircraft：更强细粒度验证。

### 10.2 增量协议

默认协议：

| 数据集 | Base | Incremental |
|---|---:|---|
| CIFAR-100 | 50 classes | 10 sessions × 5 classes |
| CUB-200 | 100 classes | 10 sessions × 10 classes |
| ImageNet-R | 100 classes | 10 sessions × 10 classes |

统一设置：

- 3 个 class order seeds；
- exemplar-free；
- inference task-agnostic；
- 主表使用 frozen ViT-B/16-IN21K；
- 附表使用 ViT-B/16-IN1K；
- 所有方法报告完整 memory。

不直接混表的公开协议包括：HC-SOINN 的 CUB-200 20×10 和 ImageNet-R 40×5，EASE 的 B0 Inc5/10/20 或 B100 Inc50，SD-LoRA 的 N=5/10/20。若引用公开结果，只放在 background table，不能作为主结论。

若统一重跑 HC-SOINN，必须在表注中说明协议 bias：HC-SOINN 面向更长任务序列和多节点拓扑逐步精炼，Base100 + 10 sessions 的 LifeGAD 默认协议可能低估其长序列边年龄维护和类内多模态建模能力。因此 HC-SOINN 在该协议下只能作为同协议 context，而不能作为“LifeGAD 公平压制 HC-SOINN”的证据。

### 10.3 对比方法

主表分两层。

**Same-feature classifier heads**：

1. NCM；
2. SimpleCIL；
3. Free Prototype Head；
4. ACIL / RanPAC head；
5. GPA；
6. Original GAD；
7. LifeGAD。

所有 same-feature classifier heads 必须共享同一 frozen ViT feature cache、同一 train / calibration / eval split、同一 class order seeds，并使用同等调参预算。若 Free Prototype Head、GAD 或 LifeGAD 是自行实现，需报告每个方法的验证集搜索范围和搜索轮数，避免把实现调参差异误写成方法收益。

**Full PTM-CIL systems / upper-bound context**：

1. EASE；
2. SD-LoRA；
3. CL-LoRA；
4. DIA；
5. MoAL；
6. MiN；
7. 可选 prompt baselines：L2P、DualPrompt、CODA-Prompt。

若资源受限，最小主表保留：

```text
NCM, SimpleCIL, Free Prototype, ACIL/RanPAC, GPA, GAD, LifeGAD
```

强系统作为 context 或 plug-in setting 报告。

### 10.4 常规评估指标

- Average Accuracy；
- Final Accuracy；
- Average Forgetting；
- Old Accuracy；
- New Accuracy；
- **old-new harmonic mean**：作为主指标之一，而不是附属指标，用于暴露“过度保护旧类”和“过度扩容偏新类”的 tradeoff；
- ECE / calibration curve；
- Memory in MB；
- Inference latency / FLOPs。

### 10.5 LifeGAD 自定义指标

#### Dictionary Growth Ratio

```math
\mathrm{DGR}=\frac{M_T}{M_0}.
```

符号说明：衡量最终字典相对 base 字典的增长比例。

符号说明：\(\mathrm{DGR}\) 是 Dictionary Growth Ratio；\(M_T\) 是最终 atom 数；\(M_0\) 是 base 后 atom 数；值为 1 表示没有增长。

#### Budgeted Accuracy / Memory Pareto

删除原 LCE。原因是：

```math
\frac{\Delta Acc}{\Delta Mem}
```

符号说明：在 \(\Delta Mem=0\) 时无定义，在 \(\Delta Mem\approx0\) 时会发散。LifeGAD 改用：

符号说明：\(\Delta Acc\) 是相对 baseline 的准确率提升；\(\Delta Mem\) 是相对 baseline 的额外内存；该比值试图表示单位内存收益，但在额外内存接近 0 时不稳定。

- fixed memory budget 下的 AvgAcc；
- memory-accuracy Pareto frontier；
- memory-budget AUC。

若必须报告单位内存收益，只能使用稳定化版本：

```math
\frac{\Delta Acc}{\max(\Delta Mem,\rho\mathrm{Mem}_{GAD})}.
```

符号说明：\(\mathrm{Mem}_{GAD}\) 是 GAD baseline 的内存占用；\(\rho\) 是稳定化比例常数；\(\max(\Delta Mem,\rho\mathrm{Mem}_{GAD})\) 给分母设置下界，避免单位内存收益虚高。

#### Signed Atom Error PMI

```math
\mathrm{AEPMI}^{+}
=
\frac1{|\mathcal M_T^{act}|+\epsilon}
\sum_{m\in\mathcal M_T^{act}} \max(0,\widehat{\mathrm{PMI}}_{m,T}^{err}),
```

符号说明：\(\mathrm{AEPMI}^{+}\) 是最终任务正 PMI 的 active atom 平均值；只累计与错误正相关的 atom-error association；\(\widehat{\mathrm{PMI}}_{m,T}^{err}\) 是最终任务中 atom \(m\) 的原始错误 PMI；inactive atoms 不进入该平均。

```math
\mathrm{AEPMI}^{-}
=
\frac1{|\mathcal M_T^{act}|+\epsilon}
\sum_{m\in\mathcal M_T^{act}} \min(0,\widehat{\mathrm{PMI}}_{m,T}^{err}),
```

符号说明：\(\mathrm{AEPMI}^{-}\) 是最终任务负 PMI 的 active atom 平均值；负值表示 atom 激活与错误低于独立假设下的共现水平，可能对应保护性或稳定性 atom。

```math
\mathrm{AEPMI}^{signed}
=
\frac1{|\mathcal M_T^{act}|+\epsilon}
\sum_{m\in\mathcal M_T^{act}} \widehat{\mathrm{PMI}}_{m,T}^{err}.
```

符号说明：\(\mathrm{AEPMI}^{signed}\) 是不截断符号的 active atom 平均 PMI；它会让正负关联相互抵消，因此需与 \(\mathrm{AEPMI}^{+}\)、\(\mathrm{AEPMI}^{-}\) 同时报。

负 PMI 可能代表保护性 atom，因此不能只报告 \(\max(0,\mathrm{PMI})\)。AEPMI 默认使用当前任务原始 PMI；EMA 版本只能作为可视化趋势，不用于状态触发。

#### PMI Invalid Rate

为暴露少样本条件下 PMI 触发退化，报告每个任务中不满足 2x2 计数下限的 atom 比例：

```math
\mathrm{PMIInvalidRate}_t
=
\frac1{|\mathcal M_t^{act}|+\epsilon}
\sum_{m\in\mathcal M_t^{act}}
\mathbf 1[
\min(n_{11},n_{10},n_{01},n_{00})<n_{\min}
].
```

符号说明：若该比例较高，应明确说明 lifecycle 决策主要退化为 residual、usage 和 old support 规则，而不是由 PMI 驱动。

符号说明：这里的 \(\mathrm{PMIInvalidRate}_t\) 与第 3.4 节相同；它用于实验报告，而不仅是训练内部过滤信号。

#### Prototype Anchor Drift

原 PSD 更名为 PAD，因为公式衡量 prototype anchor drift，而不是 protected support drift：

```math
\mathrm{PAD}
=
\frac1{|\mathcal C_{\mathrm{old}}|}
\sum_{c\in\mathcal C_{\mathrm{old}}}
(1-\langle p_c,\bar q_c^\star\rangle).
```

符号说明：Frozen-feature setting 中 \(\bar q_c^\star=q_c^\star\)；feature-adapted setting 中 \(\bar q_c^\star=\tilde q_c^\star\)。这样 PAD 与 \(\mathcal L_{\mathrm{p-anchor}}\) 使用同一当前任务 anchor 目标。

符号说明：\(\mathrm{PAD}\) 是 Prototype Anchor Drift；\(\mathcal C_{\mathrm{old}}\) 是评估时的旧类集合，通常等于当前任务前已见类；\(1-\langle p_c,\bar q_c^\star\rangle\) 是旧类当前 prototype 与其 anchor 的余弦距离。

若要报告 support-level drift，另定义：

```math
\mathrm{SD}
=
\frac1{|\mathcal C_{\mathrm{old}}|}
\sum_c
\left(1-
\frac{|\mathcal S_c\cap\mathcal S_c^\star|}
{|\mathcal S_c\cup\mathcal S_c^\star|+\epsilon}
\right).
```

符号说明：\(\mathrm{SD}\) 是 support drift；\(\mathcal S_c\) 是当前类别 \(c\) 的 active support，\(\mathcal S_c^\star\) 是历史 support；交并比 \(|\mathcal S_c\cap\mathcal S_c^\star|/|\mathcal S_c\cup\mathcal S_c^\star|\) 衡量 support 保持程度。

#### Growth Trigger Effectiveness

删除原 GTP。原因是用 residual / PMI 降低判定 growth 有效会与触发信号循环论证。修订为局部和前向两个版本。局部版本只说明当前任务 held-out split 上的即时收益：

```math
\mathrm{GTE}_{local}
=
\frac{
\#\{g:\Delta \mathrm{AvgAcc}_g>0
\ \text{or}\ 
\Delta \mathrm{HM}_g>0
\ \text{or}\ 
\Delta \mathrm{ECE}_g<0\}
}{
\#\{\text{growth events}\}+\epsilon
}.
```

符号说明：所有 \(\Delta\) 均在独立 \(\mathcal V_t^{eval}\) 或最终 test protocol 的下游指标上计算，不使用触发时的 residual/PMI 作为有效性判据。为检查 growth 是否只是过拟合当前任务，还需报告前向版本：

符号说明：\(\mathrm{GTE}_{local}\) 是局部 Growth Trigger Effectiveness；\(g\) 表示一次 growth 事件；\(\#\{\cdot\}\) 表示集合计数；\(\Delta \mathrm{AvgAcc}_g\)、\(\Delta \mathrm{HM}_g\)、\(\Delta \mathrm{ECE}_g\) 是开启该 growth 相对不开启时在当前 held-out 指标上的变化；\(\mathrm{HM}\) 是 old-new harmonic mean，\(\mathrm{ECE}\) 是 expected calibration error。

```math
\mathrm{GTE}_{forward}^{(K)}
=
\frac{
\#\{g\in\mathcal G_{\le t}:
\Delta \mathrm{HM}_{t+K}(g)>0
\ \mathrm{or}\
\Delta \mathrm{AvgAcc}_{t+K}(g)>0\}
}{
\#\mathcal G_{\le t}+\epsilon
},
\qquad K\in\{1,2\}.
```

符号说明：\(\Delta_{t+K}\) 在未来任务 \(t+K\) 后回测。若实验资源不足，主文报告 \(\mathrm{GTE}_{local}\)，appendix 报告 \(\mathrm{GTE}_{forward}^{(1)}\)；不能把 local GTE 解释为长期泛化有效性。

符号说明：\(\mathrm{GTE}_{forward}^{(K)}\) 是向后回测 \(K\) 个任务后的 growth 有效性；\(\mathcal G_{\le t}\) 是截至任务 \(t\) 已发生的 growth 事件集合；\(\Delta \mathrm{HM}_{t+K}(g)\) 和 \(\Delta \mathrm{AvgAcc}_{t+K}(g)\) 是未来任务后重新评估该事件影响的指标变化。

#### Effective Dictionary Rank

为避免动态 growth 只增加冗余 atoms，报告归一化字典 Gram 矩阵的 effective rank：

```math
G_t=\hat D_t\hat D_t^\top,
\qquad
\mathrm{EffRank}(D_t)
=
\exp\left(
-\sum_i \bar\sigma_i\log(\bar\sigma_i+\epsilon)
\right),
```

符号说明：\(\bar\sigma_i=\sigma_i(G_t)/\sum_j\sigma_j(G_t)\)。若 \(M_t\) 增长但 \(\mathrm{EffRank}(D_t)\) 基本不变，说明新增 atoms 可能冗余。

符号说明：\(G_t\) 是归一化字典的 Gram 矩阵；\(\sigma_i(G_t)\) 是 \(G_t\) 的第 \(i\) 个奇异值或特征值幅度；\(\bar\sigma_i\) 是归一化谱权重；\(\mathrm{EffRank}(D_t)\) 是谱熵指数化后的有效秩。

#### Ridge Condition Number

为监控 ridge shadow residual 是否可信，报告：

```math
\kappa_{\mathrm{ridge}}
=
\mathrm{cond}
\left(
\hat D_t\hat D_t^\top+\lambda_{\mathrm{sh}}I
\right).
```

符号说明：若实现采用 \(d\times d\) push-through 形式，则同步报告 \(\mathrm{cond}(\hat D_t^\top\hat D_t+\lambda_{\mathrm{sh}}I_d)\)。\(\kappa_{\mathrm{ridge}}\) 过高说明 dense residual 可能由病态字典导致，不应单独作为 grow 或 no-grow 决策依据。

符号说明：\(\kappa_{\mathrm{ridge}}\) 是 ridge 系统的条件数；\(\mathrm{cond}(\cdot)\) 表示矩阵最大与最小奇异值之比；条件数越大，dense ridge residual 越可能受数值病态影响。

### 10.6 消融实验

消融分为主文核心消融和 appendix 诊断消融，避免形成不可执行的 18 项 × 3 数据集 × 3 seeds 矩阵。主文默认只在 3 个数据集、3 seeds 上跑 8-10 个核心项；其余诊断项先在 CIFAR-100 单 seed 或单数据集 3 seeds 上报告，只有结果关键时再扩展。

**主文核心消融**：

| 消融 | 目的 |
|---|---|
| w/o growth | 验证动态扩容是否必要 |
| fixed-growth schedule | 区分数据驱动增长与固定增长 |
| fixed growth events + random directions | 分离“发生 growth”与“残差信息方向”的贡献 |
| residual-only growth | 验证 residual 信号 |
| residual + ridge-shadow gate | 验证 L2 shadow residual 是否减少误扩容 |
| PMI vs error-rate-only / usage-only / random trigger | 验证 atom-error association proxy 是否比更简单触发信号有独立价值 |
| w/o protected freeze | 验证 protected atoms 对旧类稳定性 |
| w/o base-drift regularizer | 验证 D-FSCIL 式有限适配原则是否有益 |
| no rumination / all-old rumination / selective rumination | 三路共享 baseline，验证收益来自反刍本身还是选择性过滤 |
| D-FSCIL-style global-adapt GAD, \(m=3\) and \(m=M_T^{LifeGAD}\) | 整体字典适配 + base drift、无 atom lifecycle；两档容量避免容量差异混淆 |
| 3-state lifecycle baseline | 仅 plastic / protected / inactive，验证 5 状态 formalism 是否必要 |
| same-memory GAD vs LifeGAD | 验证内存公平性 |

**Appendix / 单数据集诊断消融**：

| 消融 | 目的 |
|---|---|
| hard-freeze vs state-aware atom momentum | 验证 MoAL 式有限动量适配是否优于硬冻结 |
| state-aware atom momentum w/o protected freeze | 排除 momentum 只是在 protected freeze 外重复发挥作用 |
| w/o feature-space anchor transport | 仅在 feature-adapted plug-in setting 中验证 anchor 传输必要性 |
| w/o stale-binding refresh | 验证过时 atom-class binding 刷新是否优于永久保护 |
| w/o quarantine/prune | 验证风险 atom 管理 |
| w/o pseudo-class reserve | 验证前向兼容 reserve 初始化 |
| w/o reserve preparation / w/o \(\mathcal L_{\mathrm{res-mv}}\) | 验证 base 阶段多视角 reserve consistency 是否带来额外收益 |
| linear-only reserve / mask-only reserve / dual-view reserve | 区分单一路径 mixing 与双视角一致性的贡献，检查 mask-like 分支是否单独有害 |
| Stage B0:B1 = 20:80 / 50:50 / 80:20 / adaptive | 验证阶段化 base 训练，而不是照搬 M2SD 的固定比例 |
| TopK vs simplified Lp vs full p-annealing | 验证稀疏机制与 λ 退火 |
| capacity \(m=\{3,5,8\}\) | 验证容量曲线 |
| w/o proxy calibration | 验证旧类邻域校准 |
| \(\psi^{pred}\) / \(\psi^{true}\) / coefficient-only statistics | 报告 atom 贡献定义对 usage、PMI 和触发的影响 |
| \(U_{pred}\) vs \(U_{true}\) | 检查 usage 信号的预测内生性 |

其余如 whitening、initialization、usage balancing、shared-private、clone \(\alpha,\sigma\)、support Jaccard、atom deletion/insertion curves 放 appendix 或单数据集分析。

---

## 11. Introduction 草稿

<!-- 修订说明：更新贡献表述，删除 LCE/PSD 等问题指标，避免把 PMI 写成因果或 SUE 原样迁移。 -->

Class-incremental learning (CIL) requires a model to learn a sequence of disjoint class sets and to recognize all seen classes at test time without access to task identity. Recent progress in pre-trained model based CIL has significantly changed the nature of this problem. With strong frozen or lightly adapted visual backbones, catastrophic representation forgetting is often less severe than in models trained from scratch. However, the classifier sitting on top of the representation remains a critical and under-examined source of instability. Prototype classifiers such as nearest class mean are simple and robust, but they assume that each class can be represented by a single stable direction. This assumption becomes fragile in long class sequences, fine-grained recognition, and domain-shifted data, where new classes continuously compete with old ones in a shared feature space.

A promising way to strengthen prototype classifiers is to parameterize each class prototype as a sparse combination of shared dictionary atoms. Instead of learning every prototype independently, a dictionary-prototype classifier represents classes through reusable coordinates. This design can reduce redundancy, encourage semantic sharing, and provide a compact memory of old knowledge through sparse supports. Yet a fixed dictionary is not sufficient for class-incremental learning. Standard dictionary learning is typically studied in a static setting where all data are available, while CIL is fundamentally sequential: old data disappear, new classes arrive, and the feature distribution encountered by the classifier changes over time.

This paper argues that the central question is not merely how to learn a dictionary prototype classifier, but how to manage the lifecycle of a dictionary under incremental constraints. A class-incremental dictionary must decide which atoms should be protected because they support old classes, which atoms remain plastic for future adaptation, when new atoms should be added, and when unreliable atoms should be quarantined or pruned. Existing prototype-based methods usually lack such a mechanism. They either keep class prototypes fixed, update them with limited geometric regularization, or expand task-specific modules without explicitly modeling the reliability and usage of shared prototype components.

We propose LifeGAD, a risk-calibrated lifelong dictionary prototype framework for PTM-based class-incremental learning. LifeGAD represents each class prototype as a sparse combination of dictionary atoms, but unlike a fixed dictionary classifier, it equips atoms with lifecycle states. Atoms that strongly support old classes are protected and anchored. New classes first attempt to reuse the existing dictionary through sparse projection. If the residual between a new class prototype and the current dictionary is high, LifeGAD grows new atoms initialized from residual directions. If validation errors are statistically associated with particular atoms under calibrated reliability filters, LifeGAD may place them in a refresh queue, quarantine them, or clone them depending on whether they are also needed by old classes.

The key signal behind LifeGAD is atom-level risk diagnostics. For each task, LifeGAD estimates atom usage, normalized old-class support, prototype drift, and the pointwise mutual information between atom activation and prediction errors. Unlike uncertainty estimation methods that assume fixed dictionaries and external calibration sets, LifeGAD treats PMI only as a current-task atom-error association proxy in an evolving dictionary. We therefore use held-out calibration splits, minimum-count filters, raw-PMI confidence intervals, persistent-atom EMA only for monitoring, and held-out downstream metrics to avoid over-interpreting this signal.

LifeGAD also provides a diagnostic perspective on CIL forgetting. Forgetting is not only a drop in accuracy or a shift in classifier weights; it can be decomposed into prototype anchor drift, overuse of shared atoms, error concentration on unreliable atoms, and insufficient dictionary capacity for new semantic regions. These diagnostics make it possible to ask whether performance gains come from better reuse of existing atoms, timely growth of new atoms, or improved calibration between old and new classes.

Our contributions are threefold. First, we introduce a lifelong dictionary prototype formulation for class-incremental learning, where dictionary atoms are managed entities with protected, plastic, reserve, quarantined, and inactive states plus source metadata. Second, we propose calibrated lifecycle diagnostics and head-side refinement mechanisms, including residual growth, normalized support usage, raw error-PMI confidence filtering, selective anchor rumination, state-aware atom momentum, and stale-binding refresh. The primary mechanisms are protected reuse and residual growth; PMI, rumination, clone, quarantine, prune, and transport are diagnostic or appendix mechanisms unless supported by dedicated ablations. Third, we define lifecycle-specific evaluation tools, including dictionary growth ratio, signed atom error PMI, prototype anchor drift, local/forward growth trigger effectiveness, and memory-budgeted accuracy, and evaluate LifeGAD under frozen PTM protocols. Feature-space anchor transport is kept as an appendix mechanism for optional plug-in feature-adapted protocols rather than a main contribution.

LifeGAD is not intended to replace feature adaptation methods such as adapters or LoRA. Instead, it targets the classifier geometry that remains after strong pre-trained representations are obtained. It can operate as a standalone exemplar-free classifier head or as a plug-in readout on top of stronger PTM-CIL feature learners. In the latter case, appendix feature-space anchor transport can align old anchors to the current representation, while selective rumination and stale-binding refresh still act only on the dictionary head. By treating the dictionary as a lifelong object rather than a fixed basis, LifeGAD bridges sparse dictionary learning and class-incremental recognition under the practical constraints of no replay, no task identity, and bounded memory.

---

## 12. 风险分析与应对

<!-- 修订说明：补充 PMI 统计风险、single-prototype 表达风险、理论边界和 reviewer 对指标/协议的质疑应对。 -->

### 12.1 实验结果不及预期的原因与预案

#### 原因 1：字典 head 不是主要瓶颈

强 PTM 特征下，SimpleCIL、EASE、SD-LoRA、DIA、MoAL 或 MiN 已经很强，LifeGAD 的提升可能有限。

预案：

- 明确定位为 classifier head，而不是完整 feature adaptation 方法；
- 主打 frozen ViT、GAD、Free Prototype、GPA 的替换收益；
- 增加 LifeGAD-on-SD-LoRA / LifeGAD-on-EASE-features 插件实验；
- 报告同等 feature 下的分类头对比。

#### 原因 2：生命周期触发噪声大

PMI、residual 或 usage 触发可能误扩容、误冻结。

预案：

- 用 base residual percentile / MAD 阈值，避免针对测试调参；
- 对 PMI 使用当前任务 held-out calibration split、最小计数和原始 PMI 置信区间；EMA 仅作 persistent-atom 可视化；
- 报告 \(\mathrm{PMIInvalidRate}_t\)，当计数不足时不由 PMI 单独触发状态迁移；
- 加 fixed-growth、residual-only、PMI vs error-rate-only / usage-only / random trigger 对照；
- 加 memory budget 上限；
- 用 \(\mathrm{GTE}_{local}\) 在 held-out 下游指标上评估即时 growth，用 \(\mathrm{GTE}_{forward}\) 回测未来任务影响。

#### 原因 3：细粒度类内多模态无法由单 prototype 解决

CUB 或 Cars 上可能输给 HC-SOINN。

预案：

- 承认 LifeGAD 仍是单输出 prototype；
- 增加 Mixture-LifeGAD appendix，每类允许 2 个 support modes；
- 主文强调轻量共享坐标和生命周期，而不是替代拓扑方法。

#### 原因 4：反刍与锚点传输引入额外假设

Selective anchor rumination 假设当前新类样本的局部残差可迁移到相似旧类 anchor 附近；feature-space transport 假设当前任务 paired features 足以估计旧特征空间到新特征空间的低容量映射。两者都可能在语义跨度大、样本少或上游特征强非线性漂移时失效。

预案：

- 主协议仍以 frozen-feature LifeGAD 为核心，不把 transport 作为默认收益来源；
- 反刍只对风险旧类和 margin violation 样本启用，避免 all-old pseudo replay；
- 报告 `all-old rumination`、`w/o selective rumination` 和 `w/o feature-space anchor transport`；
- 对 pseudo features 单独报告置信度、margin 改善、old-new HM，而不只报告最终 AvgAcc；
- 若 transport 连续任务误差累积，降级为恒等映射或只在 plug-in setting appendix 中使用。

#### 原因 5：multi-view reserve 可能引入虚拟方向噪声

M2SD 的消融显示，CutMix-like 分支单独使用可能损害 FSCIL 后期性能。LifeGAD 的 feature-mask reserve 分支同样不能被解释为稳定收益来源：它只是提供与 linear reserve 不同的扰动视角，若单独用于初始化或正则，可能把无语义的子空间拼接方向写入 reserve atoms。

预案：

- mask-like 分支默认只与 linear 分支配对使用，并通过 \(\mathcal L_{\mathrm{res-mv}}\) 做一致性约束；
- 主协议不做图像级 CutMix，不训练 backbone，不把 mask 分支生成物加入正式类别集合；
- 报告 `linear-only reserve`、`mask-only reserve`、`dual-view reserve` 与 `w/o reserve preparation`；
- 若 mask-only 或 dual-view 在 \(\mathcal V_0^{eval}\) 上降低 base margin 或后续 AvgAcc，则降级为 linear reserve 或关闭 \(\lambda_{\mathrm{mv}}\)；
- 对 reserve atoms 单独报告后续激活率、aging 到 inactive 的比例和被新类 top-k 使用后的收益，避免只看最终准确率。

### 12.2 Reviewer 可能质疑与 rebuttal 思路

#### Q1：这是不是 GAD 加一堆 heuristics？

Rebuttal：

LifeGAD 的核心不是单个 trick，而是 atom lifecycle formalism。它定义了 atom states、risk statistics、state transitions、memory-aware diagnostics 和状态机不变量。消融会证明 freeze、growth、atom-error association proxy、quarantine、pruning 分别贡献不同。

#### Q2：动态扩容是否不公平？

Rebuttal：

报告完整 memory，包括 dictionary、coefficients、anchors、metadata、proxy 邻域和状态向量。提供 same-memory GAD、fixed-capacity GAD、fixed-growth LifeGAD，并使用 memory Pareto / budgeted accuracy，而不是不稳定的 LCE。

#### Q3：error PMI 是否需要旧数据或 task-ID？

Rebuttal：

PMI 只使用当前任务 calibration split 和模型预测，不保存旧样本。旧类风险通过 class anchors、prototype margin 和 normalized old support 间接估计。推理阶段不使用 task-ID。PMI 只作为当前任务 atom-error association proxy，不作为不确定性估计器或因果解释。

#### Q4：为什么 HC-SOINN 结果不能直接比较？

Rebuttal：

HC-SOINN 公开主表的 CUB-200 和 ImageNet-R 协议与 LifeGAD 默认协议不同。主表将统一重跑同协议；公开数值只作为背景，不用于 SOTA claim。同时承认 LifeGAD 默认协议可能低估 HC-SOINN 在更长任务序列下的拓扑逐步精炼优势，因此 HC-SOINN 行必须带协议 bias 注释。

### 12.3 失效条件

LifeGAD 可能在以下条件下失效或收益有限：

- base 类太少，初始字典没有覆盖性；
- 未来任务与 base 语义完全无关，字典必须不断扩容；
- PTM 特征严重漂移，旧 anchors 不再可靠；
- feature-space transport 无法可靠对齐旧 anchors，导致 selective rumination 强化错误几何；
- 类内多模态是主瓶颈，单 prototype 表达不足；
- 当前任务样本极少，PMI 统计不稳定，大量 atoms 不满足 \(n_{\min}\) 时 lifecycle 退化为 residual / usage / old support 驱动；
- task-free online CIL 没有明确任务边界，anchor 更新困难；
- lifecycle 阈值对不同数据集或 class order 敏感。

### 12.4 适用范围

LifeGAD 最适合：

- PTM-based CIL；
- exemplar-free 或低内存 CIL；
- task-agnostic inference；
- 分类头几何和 old-new calibration 是主要瓶颈的场景；
- 中长序列、细粒度、语义跨度逐步扩大的类别流。

---

## 13. 时间线

<!-- 修订说明：把 12 周完整强版本降格为探索性估计，压缩消融并区分 MVP 与强系统对比。 -->

假设单张 A100/4090，每日约 12 小时。以下是探索性估计，不应写成保证可完成的投稿承诺。

| 阶段 | 时间 | 目标 |
|---|---:|---|
| Week 1 | 7 天 | 搭建 frozen ViT 特征缓存；复现 SimpleCIL、NCM、Free Prototype、Original GAD |
| Week 2 | 7 天 | 实现 LifeGAD core：atom states、protected freeze、residual growth、coefficient expansion |
| Week 3 | 7 天 | 实现 usage / atom-error association proxy、calibration split、confidence filtering、PAD / signed AEPMI / DGR |
| Week 4 | 7 天 | CIFAR-100 主实验与调试；确定默认阈值策略 |
| Week 5 | 7 天 | CUB-200 与 ImageNet-R 主实验；跑 3 seeds |
| Week 6 | 7 天 | same-feature baselines：ACIL/RanPAC、GPA、GAD、same-memory 对比 |
| Week 7 | 7 天 | 主文核心消融 8-10 项：growth 事件/方向、PMI、freeze、rumination、D-FSCIL-style baseline、same-memory |
| Week 8 | 7 天 | appendix 单数据集诊断：capacity、TopK vs Lp、prune/quarantine、momentum、PMI heatmap、\(\mathrm{GTE}_{local}\) |
| Week 9-10 | 14 天 | 插件实验或强系统 context：EASE/SD-LoRA/DIA 特征上接 LifeGAD |
| Week 11 | 7 天 | 失败案例分析、阈值敏感性、不同 class order、IN1K backbone |
| Week 12 | 7 天 | 论文初稿、表格统一、附录、代码整理 |

最小可投稿验证约 8 周：CIFAR-100 + CUB-200 + ImageNet-R，同特征主 baselines + 主文核心消融。完整 18+ 项消融、\(\mathrm{GTE}_{forward}\)、强 PTM-CIL 系统统一重跑和插件实验更可能需要 12-16 周，不应写成 Week 7 单周承诺。

---

## 14. 最小可行实现

<!-- 修订说明：删除 LCE/PSD，MVP 不再声称完整 lifecycle 闭环；在 residual-growth 与 protected-freeze 主线外，吸收 MoAL 启发的选择性反刍和状态感知动量作为可控增强。 -->

MVP 版本应优先实现以下组件：

1. Frozen ViT feature cache；
2. GAD base head；
3. atom state vector；
4. protected atom freeze；
5. multi-view pseudo-class reserve atom 初始化，但使用 \(\rho_{\mathrm{res}}\) 预算而非未来类总数；
6. base two-stage training：Stage B0 geometry warmup + Stage B1 reserve preparation，并在 B1 启用 \(\mathcal L_{\mathrm{res-mv}}\)；
7. ridge shadow residual gate，先区分“稀疏预算不足”和“字典覆盖不足”；
8. residual-triggered growth；
9. state-aware base-drift regularizer；
10. state-aware atom momentum，作为 protected hard-freeze 的 soft-freeze 对照；
11. selective anchor rumination，只对 PAD / margin / support PMI 触发的风险旧类生成临时 pseudo features；
12. stale-binding refresh queue，第一版至少实现 coefficient reallocation 和 rumination，不强制实现 clone；
13. usage-based inactive marking 和 reserve aging；
14. calibration split + raw PMI CI + PMIInvalidRate 统计；EMA 只作 persistent-atom 可视化；
15. LifeGAD metrics：DGR、Budgeted AvgAcc / memory Pareto、signed AEPMI、PMIInvalidRate、PAD、\(\mathrm{GTE}_{local}\)、EffRank、ridge condition number；
16. CIFAR-100 主实验、same-memory GAD 对照和 D-FSCIL-style global-adapt GAD 对照。

第一版可暂缓：

- clone；
- quarantine contribution reweighting；
- feature-space anchor transport，除非同时做 LifeGAD-on-adapted-features；
- shared-private dictionary；
- mixture prototype；
- full p-annealing（除非已实现 λ coefficient annealing）。

MVP 若能证明：

```text
LifeGAD > GAD > Free Prototype / SimpleCIL
```

且在 same-memory 或 memory-budgeted setting 下收益仍存在，则方法主线成立。完整 lifecycle claim 需要 clone/quarantine/prune 与 \(\mathrm{GTE}_{local}/\mathrm{GTE}_{forward}\) 的后续实验证明。

---

## 15. 文档修订方向

<!-- 修订说明：更新合并回 GAD 主文档时的写作边界，强调证据等级和公允对比。 -->

若将 LifeGAD 合并回 GAD 主文档，建议：

1. 将 GAD 作为 fixed dictionary baseline；
2. 将 LifeGAD 作为主方法；
3. 在方法章节新增 “Dictionary Lifecycle Management”；
4. 在实验章节新增 lifecycle metrics：DGR、PAD、signed AEPMI、\(\mathrm{GTE}_{local}/\mathrm{GTE}_{forward}\)、memory Pareto；
5. 在风险章节明确 PMI 不是因果解释，clone/quarantine/prune 是经验设计；
6. 在 related work 中把 D-FSCIL 写作前向兼容动机，而非机制继承；
7. 在 baseline 章节分开 same-feature head 和 full PTM-CIL systems；
8. 在 ablation 中加入 D-FSCIL-style global-adapt GAD，证明 LifeGAD 的收益不是简单整体字典适配；
9. 在 appendix 放 atom-level 可视化、PMI 置信过滤、deletion/insertion、ridge shadow residual 分析和 protocol background table。

推荐论文 narrative：

> Class-incremental learning requires not merely sparse prototypes, but a lifelong dictionary coordinate system whose atoms are stabilized, diagnosed, and selectively evolved without replay.

---

### 附录：理论边界与经验假设

<!-- 修订说明：根据审查报告补充命题级理论加固，同时明确哪些机制仍是经验设计或后续工作。 -->

LifeGAD 的创新在于把固定 GAD head 改为可诊断、可增长、可保护的生命周期字典。但本文只主张命题级边界；触发阈值、clone、quarantine、prune 属经验设计。

#### Prototype drift 传递界

对旧类 \(c\)，若 \(u_c^\star=a_c^\star \hat D^\star\) 且 \(\|u_c^\star\|\ge\kappa\)，令 \(\mathcal S_c=\mathcal S_c^\star\cup\mathcal S_c^{current}\)，则局部有：

```math
\|p_c-p_c^\star\|
\lesssim
\kappa^{-1}
\left(
\|a_{c,\mathcal S_c}-a_{c,\mathcal S_c}^\star\|_2
\|\hat D_{\mathcal S_c}\|_{\mathrm{op}}
+
\|a_{c,\mathcal S_c}^\star\|_2
\left(
\sum_{m\in\mathcal S_c}
\|\hat d_m-\hat d_m^\star\|_2^2
\right)^{1/2}
\right).
```

符号说明：该界描述旧类 prototype 漂移的局部上界；\(p_c^\star\) 是旧类历史 prototype；\(\hat D^\star\) 是旧归一化字典快照，与正文 \(u_c=a_c^{act}\hat D_t\) 的归一化字典口径一致；\(\kappa\) 是旧未归一化 prototype 范数下界；\(\mathcal S_c\) 是历史 support 与当前 support 的并集；\(a_{c,\mathcal S_c}\) 表示系数 \(a_c\) 限制在 support \(\mathcal S_c\) 上的子向量；\(\|\hat D_{\mathcal S_c}\|_{\mathrm{op}}\) 是 support 子字典的算子范数；\(\lesssim\) 表示忽略高阶项的近似上界。

令旧 anchor 的分类 margin 为

```math
\gamma_c^\star
=
S_c(q_c^\star)-\max_{j\ne c}S_j(q_c^\star).
```

符号说明：\(\gamma_c^\star\) 是旧 anchor \(q_c^\star\) 上类别 \(c\) 的历史分类 margin；它衡量旧 anchor 相对最强竞争类的分数优势。

若 \(\|p_c-\bar q_c^\star\|_2\le\epsilon_c\)，且其他 prototype 方向在同一步中的总扰动不超过同阶 \(\epsilon_c\)，则由余弦打分的 Lipschitz 性可得近似 margin preservation bound：

```math
\mathrm{margin}_c(p_c)
\gtrsim
\gamma_c^\star-\frac{2\epsilon_c}{\tau}.
```

符号说明：该式把 prototype drift \(\epsilon_c\) 转换为 margin 损失上界；\(\gtrsim\) 表示近似下界；\(\epsilon_c\) 是当前 prototype 与当前空间 anchor 的距离上界；\(\tau\) 是分类温度。

当右侧仍为正时，anchor drift 本身不足以解释旧类判别翻转；当其接近或小于 0 时，protected freeze 与 coefficient anchor 不再提供 margin 保证，需要通过 rumination、coefficient reallocation 或降低新类干扰处理。该命题只说明 drift-to-margin 的局部边界，不证明绝对不遗忘。实验需报告 \(\kappa=\|u_c^\star\|\)、\(\|\hat D_{\mathcal S_c}\|_{\mathrm{op}}\)、\(\epsilon_c\) 和旧类 margin 分布；若 \(\kappa\) 接近 0 或 support 条件数很差，该界会失去解释力。

#### PMI 估计可靠性界

\(\widehat{\mathrm{PMI}}(h_m,e)\) 只在有效计数充足时使用。令 \(n_{11},n_{10},n_{01},n_{00}\ge n_{\min}\)，用平滑估计和 Hoeffding / delta method 近似给出置信区间。若区间跨过触发阈值，则不执行状态迁移，仅进入观察队列。

#### Residual growth 局部改进 remark

若新类均值 \(\hat\mu_c\) 对当前字典的重构残差 \(r_c=\hat\mu_c-a_cD\) 超过阈值，新增 \(d_{\mathrm{new}}=r_c/\|r_c\|\) 并允许新类系数使用该 atom，则在不考虑稀疏预算和归一化扰动时，取新系数 \(\beta=\|r_c\|\) 可使当前类均值的平方重构误差至少下降 \(\|r_c\|^2\)。在 top-k、coherence 和 old-protected 约束下，只能保证局部候选解不差于不使用新 atom 的解；这不是全局准确率或泛化提升保证。主文应把该结论写成 residual-growth 的线性代数解释，而不是核心理论贡献。

为了分离“growth 事件”与“残差方向信息”的贡献，实验必须包含 fixed growth events + random directions 消融。

#### Selective rumination 的假设边界

Selective anchor rumination 依赖局部残差迁移假设：若新类 \(j\) 与旧类 \(i\) 在 anchor 空间相近，则当前新类样本的局部偏移 \(z_t(x)-\hat\mu_j\) 可以作为旧类 \(i\) 的局部扰动近似。该假设比直接生成全量旧样本更弱，但仍可能在语义跨度大或类内多模态强时失败。因此反刍只对 \(\mathcal R_t^{old}\) 中的风险旧类启用，并使用 \(M_{\mathrm{rum}}\) 掩码选择当前 head 确实未吸收的 pseudo features。不能把该模块写成 replay 或真实旧数据恢复。

#### Feature-space transport 误差边界

若 \(T_t\) 的 paired-feature 训练误差为 \(\epsilon_T\)，则旧 anchor 的对齐误差会进入后续 PAD、rumination 和 proxy calibration。连续任务中反复使用 MLP transport 可能造成误差累积，因此主文默认只在 feature-adapted plug-in setting 启用，并优先使用 affine / Procrustes 等低容量映射。Frozen-feature 协议下 \(T_t=I\)，不存在该误差源。

若 \(T_t\) 的 Lipschitz 常数为 \(L_t\)，单步对齐误差为 \(\epsilon_t\)，则递推误差满足：

```math
E_t
\le
L_t E_{t-1}+\epsilon_t,
\qquad
E_0=0.
```

符号说明：因此 \(L_t>1\) 时误差可能随任务放大。实现中若连续两任务的 held-out transport error 上升，应停止复合传输并回退到最近一次可靠 anchor 快照或恒等映射。

符号说明：\(E_t\) 是复合到任务 \(t\) 后的累计 transport 对齐误差；\(L_t\) 是当前传输映射 \(T_t\) 的 Lipschitz 常数；\(\epsilon_t\) 是任务 \(t\) 单步传输估计误差；\(E_0=0\) 表示 base 任务没有历史传输误差。

#### State-aware momentum 漂移界

对 atom 更新 \(\bar d_m\) 使用 \(\alpha_D(s_m,o_m)\) 动量合并时，单步漂移满足：

```math
\|d_m^{new}-d_m\|
\lesssim
(1-\alpha_D(s_m,o_m))\|\bar d_m-d_m\|.
```

符号说明：因此 protected atoms 可通过较大的 \(\alpha_D\) 获得更小漂移，但该界只约束 atom 位置，不保证旧类分类 margin 不下降。实际稳定性仍需结合 coefficient anchor、PAD 和 old-new HM 验证。

符号说明：\(d_m^{new}\) 是动量合并后的 atom；\(d_m\) 是合并前 atom；\(\bar d_m\) 是当前任务训练得到的临时 atom；右侧说明实际漂移被 \(1-\alpha_D(s_m,o_m)\) 缩放。

#### 状态机不变量

任意 grow / clone / prune 后必须保持：

- \(D_t\)、\(A_t\)、anchors 维度一致；
- 旧类 \(a_c^\star,q_c^\star\) 可索引；
- protected atom 有冻结快照；
- prune 仅允许删除无旧类 support 的 atom；
- 推理时所有已见类 prototype 均可计算。

#### Memory growth 上界

若每任务增长 \(g_t\le\min(\eta M_t,\beta|\mathcal Y_t|)\)，则主文 memory fairness 采用线性预算上界：

```math
M_T\le M_0+\beta\sum_t|\mathcal Y_t|.
```

符号说明：指数形式 \(M_T\le M_0(1+\eta)^T\) 只是递推增长的 worst-case bound，通常过松，不应用作证明动态容量公平的主依据。总存储需报告 dictionary、稀疏系数、prototype anchors、状态元数据和 proxy 统计缓存。

符号说明：这里的 \(M_T,M_0,\beta,\mathcal Y_t\) 与第 9 节 memory 上界相同；该附录公式用于说明线性预算假设下的容量边界，而不是自动保证动态容量公平。

#### PMI 非因果声明

PMI 仅表示 atom 激活与错误共现，不能解释为“atom 导致错误”。原因是 \(h_m\) 依赖预测类且与 \(e\) 共享模型决策信号。若要接近因果证据，需要 deletion / insertion 或 intervention curves，本文将其作为 appendix 诊断或后续工作。

---

### 附录：修订日志

<!-- 修订说明：补充 v3 审查报告中的硬伤、推测项和实验公平性问题，确保每个审查发现都有对应修订位置。 -->

| 审查发现 | 严重程度 | 修订动作 | 修订后位置 |
|---------|---------|---------|-----------|
| v3: `grown` 被误写成 lifecycle state | 🔴 | 保留五状态集合；将 `grown/clone` 改为来源元数据 \(o_m\)，动量/正则使用 \((s_m,o_m)\) | 第 2.1、5.5.1、6.6 节 |
| v3: `inactive` 行为未闭合 | 🔴 | 定义 logical inactive 不参与梯度、momentum、新类初始化或推理；物理 prune 需 remapping 不变量 | 第 2.1、2.3、6.5 节 |
| v3: feature-space transport 后 anchor/freeze loss 空间混用 | 🔴 | anchor 与 freeze loss 改用 \(\bar q_c^\star,\bar d_m^\star\)，frozen-feature 下退化为原 anchor | 第 5.3、5.5 节 |
| v3: margin drop 与 old-margin drop 未正式定义 | 🔴 | 定义 \(S_c(v)\)、\(\mathrm{margin}_c(v)\)、\(\Delta_{\mathrm{margin}}\)、\(\Delta_{\mathrm{old\text{-}margin}}\) 及 stale-binding 阈值 | 第 5.8.1、6.7、9 节 |
| v3: 完整 lifecycle claim 与 MVP 不匹配 | 🔴 | 首版强 claim 收窄到 protected reuse + residual-triggered growth；clone/quarantine/prune/transport 降为 appendix 或后续验证 | 第 0、13、14、15 节 |
| v3: “Uncertainty-Guided” 命名过强 | 🟡 | 方法全称改为 Risk-Calibrated；PMI 统一写作 atom-error association proxy | 标题、第 0、3.4、11、12 节 |
| v3: MoAL 启发边界过强 | 🟡 | 改为“受 MoAL 的知识反刍与动量适配思想启发”，明确不继承 adapter/RLS/完整框架 | 第 0、5.7、6.6、8.4 节 |
| v3: M2SD multi-view reserve 只能作远缘类比 | 🟡 | 标注 feature-mask reserve 为 frozen-feature/head 侧扰动视角，不称机制继承 | 第 0、4.1、8.6、12 节 |
| v3: memory growth 指数界过松 | 🟢 | 主文采用线性 budget cap；指数界仅作递推 worst-case 备注 | 第 9 节、理论附录 |
| v3: PMI trigger 独立价值仍需证明 | 🟡 | 增加 PMI vs error-rate-only / usage-only / random trigger 消融要求 | 第 10.6、12 节 |
| v3: dynamic capacity 公平性易被质疑 | 🟡 | 明确 same-memory GAD、fixed-capacity GAD、memory Pareto 和逐项 memory accounting | 第 8、10、12 节 |
| PSD 名称与公式不一致 | 🔴 | 修正为 PAD；support drift 另定义 SD | 第 10.5 节 |
| active atoms 默认值与比例说明矛盾 | 🔴 | 删除固定比例说法；比例预算改为 \(k_t=\lceil rM_t\rceil\) | 第 4.2、9 节 |
| SUE-PMI 使用条件被过度迁移 | 🔴 | 将 PMI 降格为当前任务 atom-error association proxy；加入 held-out calibration split、最小计数、raw CI、persistent-atom EMA 监控 | 第 3.4、12 节 |
| p-annealing 缺 λ coefficient annealing | 🔴 | 默认改为 TopK；完整 p-annealing 必须实现 λ 退火 | 第 5.6、10.6、14 节 |
| GTP 循环论证 | 🔴 | 删除 GTP，改为基于 held-out 下游指标的 \(\mathrm{GTE}_{local}/\mathrm{GTE}_{forward}\) | 第 10.5 节 |
| LCE 在 same-memory 或近零额外内存时退化 | 🔴 | 删除 LCE，改为 memory Pareto / budgeted accuracy | 第 10.5、12.2 节 |
| clone 扰动 \(\delta\) 未定义 | 🔴 | 定义 residual-orthogonal clone 公式；MVP 不默认启用 | 第 6.3、14 节 |
| proxy old-logit calibration 未定义 | 🔴 | 改为 top-k prototype-neighborhood distillation，复杂度 \(O(Ck)\) | 第 5.8 节 |
| \(O_m^{(t)}\) 未按旧类数量归一化 | 🟡 | 改为旧类均值或 top-q 支撑均值 | 第 3.3 节 |
| grow residual 方向与诊断 residual 不一致 | 🟡 | 区分 \(r_c^{vec}\) 和 \(r_c^{cos}\) | 第 3.5、6.2 节 |
| \(d_m^\star\) 快照生命周期未说明 | 🟡 | protected 首次进入状态时保存快照 | 第 5.5、6.1 节 |
| pruning 后旧系数和 anchors 维度同步未说明 | 🟡 | 增加 remapping 和状态机不变量 | 第 2.3、6.5、16.4 节 |
| AEPMI 丢弃负 PMI 信息 | 🟡 | 改为 AEPMI+、AEPMI-、signed AEPMI | 第 10.5 节 |
| PMI 与 \(\psi_m\) 存在预测内生性 | 🟡 | 增加 true-label / predicted-label / coefficient-only 变体 | 第 3.1、10.6 节 |
| HC-SOINN 协议不一致不能直接混表 | 🟡 | 明确统一重跑；公开数值只作背景 | 第 8.3、10.2 节 |
| HC-SOINN 容量增长措辞不准确 | 🟡 | 改为每类多节点拓扑表示，节点数由聚类/SOINN 决定 | 第 8.3 节 |
| baseline 缺少近年强 PTM-CIL 系统 | 🟡 | 分为 same-feature head 和 full PTM-CIL system 两层 | 第 8.4、10.3 节 |
| D-FSCIL / reserve atoms 只能作远缘动机 | 🟡 | 吸收为 pseudo-class reserve 初始化，但使用预算 \(\rho_{\mathrm{res}}\)，不称机制继承 | 第 4.1、8.5、15 节 |
| D-FSCIL 的 L2 协作表示稳定性未利用 | 🟢 | 补充 ridge shadow residual，只作 grow gate 诊断，不替代稀疏 support | 第 3.5、6.2 节 |
| D-FSCIL 的整体字典回拉可提供有限适配先验 | 🟢 | 改写为 state-aware base-drift regularizer | 第 5.5.1、5.9 节 |
| 需要证明 LifeGAD 不只是全局字典适配 | 🟢 | 新增 D-FSCIL-style global-adapt GAD baseline | 第 8.5、10.6、14 节 |
| 动态 grow 可能只增加冗余 atoms | 🟢 | 新增 EffRank 与 ridge condition number 诊断 | 第 10.5 节 |
| 消融矩阵过大 | 🟢 | 主文压缩为 8-10 项，其余 appendix | 第 10.6、13 节 |
| MVP 不足以支撑完整 lifecycle claim | 🔴 | MVP 只验证 residual-growth/protected-freeze 主线；完整 lifecycle 需 clone/quarantine/prune 与 transport 消融后再主张 | 第 0、14、15 节 |
| 缺少理论边界 | 🟢 | 补充 drift 传递界、PMI 可靠性、growth 局部改进、memory 上界 | 理论附录 |
| PMI 非因果解释未声明 | 🟢 | 明确 PMI 仅为统计关联，因果需干预曲线 | 第 3.4 节、理论附录 |
| v2: EMA 后 PMI CI 无操作性定义 | 🔴 | 触发只用当前任务原始 PMI CI；EMA 仅作 persistent-atom 可视化 | 第 3.4、6 节 |
| v2: PMI EMA 跨 grow/clone 追踪非同一语义实体 | 🔴 | 增加 atom id / cosine gating；grow/clone/remap 后重置 EMA | 第 3.4 节 |
| v2: ridge shadow 固定 \(M_t\times M_t\) 逆 | 🟡 | 增加 push-through 形式，按 \(\min(M_t,d)\) 选择求解维度 | 第 3.5、10.5 节 |
| v2: GTE 仍可能只验证局部过拟合 | 🟡 | 拆为 \(\mathrm{GTE}_{local}\) 和 \(\mathrm{GTE}_{forward}\) | 第 10.5 节 |
| v2: 消融矩阵不可执行 | 🔴 | 主文 8-10 项核心消融，其余降为 appendix / 单数据集诊断 | 第 10.6、13 节 |
| v2: HC-SOINN 重跑协议 bias 未标注 | 🟡 | 增加长序列拓扑精炼优势可能被低估的表注要求 | 第 10.2、12.2 节 |
| v2: rumination / momentum / growth 消融归因不足 | 🟡 | 增加三路 rumination、random growth direction、momentum w/o protected freeze | 第 10.6 节 |
| v2: D-FSCIL-style baseline 容量混淆 | 🟡 | 要求报告 \(m=3\) 与 \(m=M_T^{LifeGAD}\) 两档 | 第 10.6 节 |

### 附录：已知但未解决的弱点

<!-- 修订说明：按 v3 审查结论更新剩余风险边界，统一 PMI 为 atom-error association proxy，并明确完整 lifecycle 仍需实验闭环。 -->

1. PMI 仍可能受预测内生性影响。即使加入当前任务 held-out calibration split、原始置信区间和变体对比，它仍然只是 atom-error association proxy，不能证明 atom 导致错误。
2. 单 prototype 输出仍限制类内多模态表达。LifeGAD 可能在 CUB、Cars 等细粒度数据上输给 HC-SOINN 这类多节点拓扑方法。
3. 生命周期阈值仍是经验设计。不同数据集、class order、base 类覆盖度可能导致 growth / quarantine / prune 的最优策略不同。
4. clone、quarantine、prune 和 feature-space transport 的理论与实验支撑仍不足。它们可以作为工程机制和 appendix 消融，但不应成为首版论文的核心强 claim，完整 lifecycle 仍需后续实验闭环。
5. 强 PTM-CIL 系统对比成本高。EASE、SD-LoRA、CL-LoRA、DIA、MoAL、MiN 的统一重跑可能超出 MVP 时间线；需要明确主表和 context 表的边界。
6. Memory accounting 更复杂。字典、系数、anchors、状态元数据、proxy 邻域都要计入，same-memory 对比必须严格实现。
7. 完整 p-annealing 需要额外实现和调试。若 λ coefficient annealing 不稳定，主文应使用 TopK，并把 p-annealing 放 appendix。
8. 理论分析目前是命题级边界。它能解释稳定性和局部改进方向，但不能替代完整泛化界或全局收敛保证。
9. Pseudo-class reserve 可能在语义跨度很大的 base 类之间生成无效方向，必须通过 `w/o pseudo-class reserve` 和 reserve aging 统计验证。
10. Ridge shadow residual 引入 dense 诊断和矩阵条件数问题；若 \(\kappa_{\mathrm{ridge}}\) 过高，该信号不能单独决定 no-grow。
11. State-aware base drift 可能过度限制 plastic atoms；需要与 old-new HM 联合评估，避免只提高 old accuracy。
12. Selective anchor rumination 可能生成低质量 pseudo features。它必须只作为临时训练信号，并通过 `all-old rumination vs selective rumination`、margin 分布和 old-new HM 验证。
13. Feature-space anchor transport 只适用于 feature-adapted plug-in setting。若上游特征漂移非线性强或当前任务样本少，transport 可能比恒等映射更差。
14. State-aware atom momentum 增加了状态级超参数，可能与 protected freeze、base-drift regularizer 功能重叠；需要报告 hard-freeze 对照。
15. Stale-binding refresh 的触发依赖 PMI、old support 和 \(\theta_{\mathrm{oldmar}}\) margin 阈值的联合判断，错误触发可能削弱旧类支撑；第一版应优先实现 coefficient reallocation 和 selective rumination，clone/quarantine 继续作为 appendix 机制。
16. 当 \(\mathrm{PMIInvalidRate}_t\) 较高时，PMI 驱动的风险诊断 claim 会显著变弱；此时只能主张 residual / support / usage 驱动的保守生命周期管理。
17. 当前理论仍是组件级边界。新增的 prototype drift、transport 误差和 state momentum 界能解释局部风险，但尚未构成端到端遗忘概率保证。
