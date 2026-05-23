# 论文详细总结：Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models

## 一、基本信息

- **标题**：Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models
- **作者**：Adam Karvonen (Independent), Benjamin Wright (Independent), Can Rager (Independent), Rico Angell (UMass, Amherst), Jannik Brinkmann (University of Mannheim), Logan Smith (Independent), Claudio Mayrink Verdun (Harvard University), David Bau (Northeastern University), Samuel Marks (Northeastern University)
- **发表年份/会议**：2024 年，NeurIPS 2024
- **代码与模型**：SAE 模型开源在 HuggingFace（chess: `adamkarvonen/chess_saes/tree/main`，Othello: `adamkarvonen/othello_saes/tree/main`）

---

## 二、研究动机与问题定义

### 核心痛点

机制可解释性（Mechanistic Interpretability）旨在将神经网络逆向工程为人类可理解的组件。近期研究使用稀疏自编码器（Sparse Autoencoders, SAEs）对语言模型（LM）中间表征进行稀疏字典学习，以提取可解释特征。然而，**评估 SAE 的质量面临一个根本性困难**：缺乏真实可解释特征的 ground-truth 集合——我们不知道一个好的 SAE 应该恢复出什么特征。

### 现有评估方法的局限

- **合成数据设定**（如 Elhage et al. [24]）：所有 ground-truth 特征已知，但与真实场景差距较大。
- **无监督代理指标（Proxy Metrics）**：如稀疏度（L0 norm）、重构保真度（Loss Recovered）、基于语言模型的自动可解释性（autointerpretability, Bills et al. [6]），这些仅是间接近似，无法直接衡量特征的可解释性和完备性。
- **手动检查**：依赖研究者的主观判断，不可靠、不可扩展。

### 本文方案

本文提出在**棋盘游戏（chess 和 Othello）语言模型**这一中间设定下评估 SAE 质量。该设定介于合成数据和自然语言之间：棋盘游戏具有天然的可解释特征类别（例如"F3 上有一个马"、"f5 上的象被牵制"），可以形式化定义并用于有监督的 SAE 质量评估。本文提出两个新的有监督评估指标——Coverage 和 Board Reconstruction，并引入一种新的 SAE 训练方法——p-annealing。

---

## 三、方法/框架

### 3.1 棋盘游戏语言模型

**Othello 模型**：
- 架构：8 层 GPT，8 个注意力头，隐藏维度 n = 512
- 训练数据：2000 万局合法走法随机采样的合成对局，每局以走子序列（token 表示落子方格）输入
- 来源：Li et al. [36]，无游戏规则先验知识，从头训练
- 关键发现（Nanda et al. [47]）：模型内部出现了对棋盘状态的线性表征，可通过线性探针（linear probe）提取，支持线性表征假说（linear representation hypothesis）

**Chess 模型**：
- 架构：与 Othello 模型相同（8 层 GPT，8 头，d=512）
- 训练数据：Lichess 数据库 [38] 中 1600 万局人类对局，以 PGN（Portable Game Notation）格式输入（如 "1. e4 e5 2. Nf3 ..."）
- 来源：Karvonen [33]，同样无先验知识，从头训练
- 模型预测合法走子的准确率达到 99.8%，内部同样可提取棋盘状态表征 [33]

### 3.2 稀疏自编码器（SAE）基本框架

SAE 是一种具有编码器-解码器架构的神经网络，对数据集 D 中的向量 x in R^d 学习稀疏重构。

本文所有 SAE 均在**第 6 层 post-MLP 残差流**的激活上训练（在该层，逻辑回归训练的线性探针已能准确分类多种棋盘属性 [33, 47]）。实验中测试了四种 SAE 变体（两种架构 x 是否使用 p-annealing）：

#### (1) Standard SAE（基准方法，Bricken et al. [9]）

编码器权重 W_e in R^{m x n}，解码器权重 W_d in R^{n x m}（列向量 L2 范数约束为 1），编码器偏置 b_e in R^m，解码器偏置 b_d in R^n。

编码器：f(x) = ReLU(W_e (x - b_d) + b_e)
解码器：x_hat = W_d f(x) + b_d

损失函数（基于开源 dictionary_learning 库 [43] 的实现）：
L_standard = E_{x ~ D_train} [ ||x - x_hat||_2 + lambda * ||f(x)||_1 ]

其中 lambda > 0 控制稀疏度强度。

#### (2) Gated SAE（Rajamanoharan et al. [54]）

Gated SAE 的核心创新是分离了"选择哪些字典元素参与重构"和"估计这些元素的激活系数"两个过程，以解决 L1 惩罚导致的特征激活收缩（shrinkage）问题：

门控信号：pi_gate(x) := W_gate (x - b_d) + b_gate
特征激活（Heaviside 阶跃函数与 ReLU 的逐元素乘积）：f_tilde(x) := I[pi_gate(x) > 0] * ReLU(W_mag (x - b_d) + b_mag)
重构：x_hat(f_tilde(x)) = W_d f_tilde(x) + b_d

损失函数使用了解码器的冻结副本 x_hat_frozen：
L_gated := E_{x ~ D_train} [ ||x - x_hat(f_tilde(x))||_2^2 + lambda * ||ReLU(pi_gate(x))||_1 + ||x - x_hat_frozen(ReLU(pi_gate(x)))||_2^2 ]

Gated SAE 的前向传播比 Standard SAE 多约 50% 的计算量 [54]。

### 3.3 p-Annealing（p-退火）—— 本文核心方法

#### 动机

L1 惩罚是真实稀疏度量（L0 范数）的凸松弛（convex relaxation）——L1 范数是 L0 范数的凸包（convex hull），因此是一个可优化的替代方案 [63]。但代理损失函数并不等同于直接优化稀疏度，导致两个问题：
- **特征收缩（shrinkage/feature suppression）** [62]：L1 惩罚不仅鼓励稀疏，还同时倾向于使特征激活值变小
- 可能学不到足够稀疏的特征表示

L0 范数不可微且直接最小化是 NP-hard [48, 18]，不适合训练。本文提出使用**非凸 L_p^p 最小化（p < 1）** 替代标准 L1 最小化。在压缩感知（compressive sensing）领域，L_p 最小化已成功用于获得比 L1 更稀疏的解 [11, 61, 64, 60]。

#### p-Annealing 算法细节

受压缩感知中 p-continuation 技术 [66] 启发，本文引入 p-annealing：

**核心思想**：训练过程中，p 从 1（凸优化，对应 L1）逐步减小到某个 p < 1（非凸目标），使其越来越逼近真实的 L0 范数。

**稀疏惩罚项**（当前训练步 s 的函数）：
L_sparse(x, s) = lambda_s * || f(x) ||_{p_s}^{p_s} = lambda_s * Sum_i f_i(x)^{p_s}

**p 的调度策略**：
- 先在初始阶段保持 **p = 1** 训练若干步（先用易于训练的 L1 惩罚到达较优参数区域）
- 然后开始**线性减小 p**，到训练结束时达到 p_end > 0
- 本文设置 **p_end = 0.2**，退火起始步为 10,000

为什么要从 p=1 开始而不是直接用固定的较低 p？因为 p 越低，L_p^p 范数越凹（非凸），越容易陷入局部最优。先通过 L1 训练到达较优区域再逐步改变损失面，可以规避该问题。

**系数退火（Coefficient Annealing）**：

改变 p 会改变 L_p^p 范数的数值尺度。若不做调整，稀疏惩罚强度会在训练中剧烈变化。经验上发现，若保持 lambda 不变，较大 p 的初始阶段稀疏惩罚太弱，效果甚至不如从头用固定 p 训练。因此引入系数退火来**自适应调整 lambda，保持稀疏惩罚强度局部恒定**：

lambda_{s+1} <- lambda_s * ( Sum_{j=s-q+1}^{s} Sum_i f_i(x_j)^{p_s} ) / ( Sum_{j=s-q+1}^{s} Sum_i f_i(x_j)^{p_{s+1}} )

实现上维护一个最近 q 个 batch 的特征激活队列，用于校准 lambda 更新。

#### p-Annealing 的通用性

p-annealing 仅修改损失函数中的 L1 项，不影响 SAE 架构，因此可以**正交地与其他 SAE 改进方法组合**。本文创建了 **Gated-Annealed SAE**：将 Gated SAE 损失函数中的稀疏惩罚项替换为 p-annealing 版本。实验表明 p-annealing 和 Gated SAE 的最优 lambda 值不同。

### 3.4 棋盘状态属性（Board State Properties, BSPs）

形式化定义：一个 BSP g 是一个函数 g : {game board} -> {0, 1}。本文考虑以下两类：

**G_boardstate（棋盘状态属性）**：
- **Chess**：8x8 格 x 12 种棋子类型（白王、白后、白车、白马、白象、白兵、黑王、黑后、黑车、黑马、黑象、黑兵）= **768 个 BSPs**
- **Othello**：8x8 格 x 2 种棋子类型（黑、白）= **128 个 BSPs**

**G_strategy（策略属性，仅 Chess）**：
由作者基于国际象棋领域知识和 AlphaZero 可解释性研究 [45] 手工选取，共 14 个概念：

| 概念 | BSP 数量 | 描述 |
|---|---|---|
| check | 1 | 轮到走的玩家是否被将军 |
| can_check | 1 | 轮到走的玩家是否可以将军对手 |
| queen | 1 | 轮到走的玩家是否有后 |
| can_capture_queen | 1 | 轮到走的玩家是否可以吃掉对手的后 |
| bishop_pair | 1 | 轮到走的玩家是否拥有双象 |
| castling_rights | 1 | 轮到走的玩家是否仍可王车易位 |
| kingside_castling_rights | 1 | 是否仍可短易位 |
| queenside_castling_rights | 1 | 是否仍可长易位 |
| fork | 1 | 轮到走的玩家是否有对对手大子的捉双 |
| pin | 1 | 棋盘上是否存在牵制 |
| legal_en_passant | 1 | 轮到走的玩家是否有合法的吃过路兵 |
| ambiguous_moves | 1 | 是否存在需要进一步指定的走法（多个同类型子可走到同一格） |
| threatened_squares | 64 | 哪些格被对手威胁 |
| legal_moves | 64 | 哪些格轮到走的玩家可以合法走到 |

### 3.5 有监督评估指标

#### (1) Coverage（覆盖率）

衡量 SAE 在多大程度上"覆盖"了研究者预设的 BSPs 集合——即 SAE 是否针对每个 BSP 都学到了对应的特征分类器。

对每个 SAE 特征 f_i 和阈值 t in [0,1]，定义二值化分类器：
phi_{f_i, t}(x) = I[ f_i(x) > t * f_i^{max} ]

其中 f_i^{max} 是 f_i 在数据集 D 上的（经验估计的）最大值。该分类器的直觉是：将 SAE 特征激活的 t 倍最大阈值以上标记为"开"。

对每个 BSP g in G，在所有 SAE 特征和所有阈值中找出 F1-score 最高的组合：
Cov({f_i}, G) := (1/|G|) * Sum_{g in G} max_t max_{f_i} F1(phi_{f_i, t}; g)

阈值搜索范围：t in {0, 0.1, 0.2, ..., 0.9}。作者观察到最优 t 通常在 {0, 0.1, 0.2} 中。

Coverage = 1 意味着对每个 BSP，SAE 都包含一个完美 F1 分类器。

#### (2) Board Reconstruction（棋盘重构）

评估能否以人类可解释的方式从 SAE 特征激活中恢复完整棋盘状态。

**核心假设**（基于 Templeton et al. [58]）：可解释的 SAE 特征往往是某些 BSP 组合的**高精度**（但不一定高召回率）分类器。例如，一个特征可能对"白兵在 e4 且白马在 f3"精确度很高——即使它不总被激活，一旦激活就非常可靠。

**两步过程**：

**步骤一（训练阶段）**：在 D_train（1000 局棋）上，对每个 SAE 特征 f_i，找出它在哪些 BSPs 上能达到 precision >= 0.95。

**步骤二（推理/预测规则）**：在 D_test（另外 1000 局棋）上，对每个 BSP g 的预测规则为：
P_g({f_i(x)}) = 1 如果存在任何对 g 在训练集上高精度的 f_i 当前被激活，否则为 0。

最终指标为所有测试棋盘状态上预测 vs 真实的平均 F1-score：
Rec({x_i}, D_test) = (1/|D_test|) * Sum_{x in D_test} max_t F1(P({f_i(x)}); b)

其中 b 表示真实棋盘状态，P 表示对所有 64 格（或所有 BSPs）的完整预测。

注意：Board Reconstruction 和 Coverage 的高低关系取决于特征类型——当存在许多对组合配置（如"白兵在 e4 + 白马在 f3"）高精度的特征时，Reconstruction 会高于 Coverage；反之，当很少有针对某 BSP 的超过 95% 精度的高精度特征时，Coverage 会更高。

---

## 四、实验设置

### 4.1 数据提取策略

- 视角一致性考虑：SAE 特征通常追踪"轮到谁走棋"的相对属性（如"我的王被牵制"而非绝对颜色"白王被牵制"）。因此**仅从白方走子前的 token 提取激活**（即 PGN 字符串中的每个句点位置）
- Chess 模型：**不测量初始位置的棋子**，因为初始位置与 PGN 字符串位置强相关，会引入虚假信号。若测量初始位置棋子，第 1 层 SAE 的 board reconstruction F1 会从 0.01 飙升到 0.52，说明该虚假信号确实存在
- 同样不测量中途走子 token（如 "Nf3" 中的 "f"）的棋盘状态，因为此时没有明确的 ground-truth 棋盘状态

### 4.2 SAE 训练超参数

| 参数 | 值 |
|---|---|
| 训练 token 数 | 3 亿（300M） |
| 优化器 | Adam (beta1=0.9, beta2=0.999) |
| 线性预热步数 | 1,000 |
| Batch size | 8,192 |
| 学习率 | 3e-4 |
| 扩展因子（expansion factor） | {8, 16}（对应隐藏维度 4096 和 8192） |
| 退火起始步（annealing start） | 10,000 |
| p_end | 0.2 |
| lambda_init | [0.02, 2.0] |

- 共计训练并开源了 **超过 500 个 SAE**（chess 和 Othello 各 500+）
- 使用**单张 NVIDIA A100 GPU**，单个 SAE 训练时间远小于 24 小时（在 3 亿 token 上）。训练完成后评估仅需不到 5 分钟

### 4.3 SAE 方法对比

四种 SAE 训练方法：
1. **Standard SAE**：基线方法，恒定 L1 惩罚
2. **Standard SAE + p-annealing**：仅本文的 p-annealing 方法
3. **Gated SAE**：Rajamanoharan et al. [54] 的架构
4. **Gated SAE + p-annealing**：p-annealing 与 Gated 架构的组合

### 4.4 评估指标汇总

**无监督代理指标**（已有文献使用 [9, 16, 54]）：
- **L0**：平均活跃特征数，即 E_{x~D} || f(x) ||_0
- **Loss Recovered**（损失恢复率）：(H_* - H_0) / (H_orig - H_0)，其中 H_orig 为原始棋盘游戏模型的 next-token 交叉熵损失，H_* 为用 SAE 重构激活替代原激活后的损失，H_0 为零化该激活的损失

**有监督指标**（本文提出）：
- Coverage（覆盖所有 BSPs）
- Board Reconstruction

### 4.5 基准对比

- **SAE on random GPT**：在随机初始化权重的模型上训练 SAE，作为下界，验证 SAE 性能确实来源于模型学习到的表征而非架构本身的归纳偏置
- **线性探针（Linear Probe）**：逻辑回归探针作为每个 BSP 的"天花板"性能参考

---

## 五、核心结果与发现

### 5.1 SAE 特征可准确重构棋盘状态（Table 1 与 Figure 2, 4）

在 G_boardstate 上：

| 方法 | Chess Coverage | Chess Reconstruction | Othello Coverage | Othello Reconstruction |
|---|---|---|---|---|
| SAE (random GPT) | 0.11 | 0.01 | 0.27 | 0.08 |
| SAE (trained GPT) | 0.48 | 0.85 | 0.52 | 0.95 |
| Linear Probe | 0.98 | 0.98 | 0.99 | 0.99 |

关键发现：
- SAE 在 trained GPT 上远优于 random GPT（例如 Chess Reconstruction: 0.01 -> 0.85），证明 SAE 成功捕获了模型学到的棋盘结构
- **SAE 性能未达到线性探针的水平**：Chess Reconstruction 0.85 vs 0.98，Othello Reconstruction 0.95 vs 0.99。这表明 SAE 并未捕获模型内部表征中的全部信息

### 5.2 p-Annealing 使 Standard SAE 性能与 Gated SAE 持平

- **Standard SAE + p-annealing** 在所有指标上持续优于使用恒定 L1 惩罚的 Standard SAE（见 Figure 2a, 4a 的 Coverage 对比）
- Standard SAE + p-annealing 的 **Coverage 得分与不使用 p-annealing 的 Gated SAE 相当**
- 两者在解决 shrinkage 问题上均显著优于 Standard SAE（见 Appendix E / Figure 5：p-annealing 和 Gated SAE 的 gamma 接近 1.0，Standard SAE 的 gamma 则明显偏低）
- Gated SAE 在某些情况下 Loss Recovered 更高（Figure 2），但 Coverage 相差无几
- **计算效率优势**：Standard SAE + p-annealing 的前向传播无需 Gated SAE 的额外门控参数和冻结解码器计算，**比 Gated SAE 少约 50% 的每步计算量**

### 5.3 有监督指标揭示了传统指标无法捕获的性能差异（Figure 2, 4）

关键发现：本文的两个有监督指标在多个方面补充了传统指标的信息盲区：

1. **扩展因子效应**：在无监督指标（L0 vs Loss Recovered 散点图，Figure 2a, 2c）上，hidden dim 4096 和 8192 的 SAE 散点几乎完全重合，无法区分。而在 Coverage 和 Board Reconstruction 上（Figure 2b, 2d），8192 维度的 SAE **明显优于** 4096 维度的 SAE（在 Standard 架构中表现为两条平行的紫色菱形线）。这表明较大容量的 SAE 确实学到了更好的特征，但传统代理指标无法检测到这一改进

2. p-annealing 在标准代理指标上与 Gated SAE 差异不大，但在有监督指标上表现出更清晰的改进

### 5.4 有监督指标与无监督指标的总体一致性（Figure 2, 3, 4）

Coverage 和 Board Reconstruction 均在**Pareto 前沿的"肘部"（elbow）区域**达到最优——该区域 SAE 以最少特征数量高效重构内部激活。这与研究者手工检查中"Pareto 前沿肘部区域的 SAE 提供最内聚的语义解读"的经验一致。本文工作为这一经验提供了**精确定量验证**。

### 5.5 策略 BSPs 的性能差异显著（Table 4）

不同策略属性的 SAE 表现差异很大（使用最佳 SAE 的 Coverage 和 Reconstruction）：

| 概念 | 线性探针 F1 | 最佳 SAE Reconstruction | 最佳 SAE Coverage |
|---|---|---|---|
| check | 1.00 | 1.00 | 1.00 |
| queen | 1.00 | 0.97 | 0.96 |
| can_check | 0.93 | 0.27 | 0.54 |
| fork | 0.68 | 0.13 | 0.38 |
| pin | 0.67 | 0.20 | 0.33 |
| board_state | 0.98 | 0.67 | 0.41 |

关键观察：
- 15 个属性中**有 6 个的线性探针 F1 < 0.95**（can_check 0.93, can_capture_queen 0.66, fork 0.68, pin 0.67, ambiguous_moves 0.72, legal_moves 0.92），说明 ChessGPT 可能不以线性方式编码这些属性
- Board Reconstruction 有时高于 Coverage（如 queen: 0.97 vs 0.96），原因是存在许多对"组合配置"高精度的 SAE 特征
- Coverage 有时高于 Reconstruction（如 can_check: 0.54 vs 0.27），原因是虽有某个特征对 can_check 有中等 F1（0.54），但其精度不到 95%，不足以参与 Reconstruction

### 5.6 相对重构偏置（Relative Reconstruction Bias）分析（Figure 5）

使用 Rajamanoharan et al. [54] 的度量：gamma := argmin_{gamma'} E_{x~D} [ || x_hat_SAE(x) / gamma' - x ||_2^2 ]
- gamma < 1 表示收缩（shrinkage），gamma = 1 表示完全无偏
- **p-annealing 和 Gated SAE 均实现了接近无偏的重构**（gamma 约 1.0），远优于 Standard SAE
- 不同领域表现不同：Chess SAE 的 gamma 集中在 0.98 附近，Othello 则在约 0.80 附近，可能反映底层模型或数据分布的差异
- L0 接近 0 的 SAE 出现不稳定 gamma 值（退化情况）

### 5.7 发现的可解释 SAE 特征示例

- **棋盘位置检测器**："f3 上有马"(Figure 1 左)，精度 > 0.95
- **战术威胁检测器**："车正在威胁后"(Figure 1 中)，与具体位置和被威胁棋子无关
- **牵制检测器**："对角线上存在牵制"(Figure 1 右)，识别需要解决将军的牵制着法
- **吃过路兵可用检测器**(Figure 6a)
- **镜像视角特征**："e2 或 e7 上的马"(Figure 6b, 6c)——特征以"轮到谁走棋"为参照系，对"对方马在 e 列、距对方底线 1 格"做出响应

---

## 六、主要贡献与局限性

### 主要贡献

1. **大规模实证研究**：训练并开源了超过 500 个 SAE（覆盖 chess 和 Othello 模型、四种训练方法、多种超参数设置），为社区提供了丰富的研究资源

2. **两个有监督 SAE 质量评估指标**：
   - **Coverage**：量化衡量特征的单义性（monosemanticity）和特征提取质量——SAE 学到了多少个我们"认为应该存在"的可解释特征
   - **Board Reconstruction**：量化衡量 SAE 对模型内部信息的穷举表示程度（comprehensiveness）——SAE 特征能否联合重建完整的棋盘状态
   - 这两个指标在棋盘游戏领域提供了比无监督代理指标更客观的评估方式

3. **p-Annealing 训练方法**：
   - 核心思想：将稀疏惩罚从凸 L1 范数逐步退火到非凸 L_p 范数（p -> 0.2），更接近真实的 L0 稀疏度量
   - **效果可媲美 Gated SAE**（Coverage 和反收缩效果相当），但计算成本更低（前向传播约减少 50%）
   - 具有**通用性**：仅修改损失函数，可与任何 SAE 架构正交组合（论文演示了与 Gated SAE 的组合）

4. **系数退火（Coefficient Annealing）技术**：通过最近 q 个 batch 的激活队列自适应调整 lambda，使 p 变化时稀疏惩罚强度保持局部恒定

### 局限性

1. **领域泛化难**：Coverage 和 Board Reconstruction 指标**严重依赖棋盘游戏特有的可枚举 BSPs**，无法直接推广到自然语言处理等更广泛领域。如何为自然语言等通用领域设计类似的有监督指标仍是一个重大开放挑战

2. **研究者偏见（Researcher Preconceptions）**：BSPs 集合完全由研究者基于领域知识手工选定，可能遗漏模型实际学到的、但研究者未预料到的重要特征或策略概念。指标的完备性受限于人类已有认知框架

3. **评估与应用的脱节**：本文仅评估 SAE 捕获模型内部表征的"质量"，**未涉及如何将学到的特征用于下游可解释性任务**（如稀疏特征电路发现 [44]、模型编辑、行为干预等），而这些才是可解释性研究的最终目标

4. **SAE 的信息损失**：SAE 在所有指标上均未达到线性探针的性能上限，说明当前 SAE 方法仍无法完整捕获模型内部表征中的所有信息。这种信息丢失的本质和原因尚不清楚

5. **模型规模有限**：实验仅在 8 层、d=512 的小型 GPT 上完成，p-annealing 在更大规模模型（GPT-2、Llama 等）上的效果有待验证

---

## 七、对后续研究的启发或潜在改进方向

1. **将 p-Annealing 扩展到更大规模模型**：本文仅在小型棋盘游戏模型上验证，鉴于 p-annealing 计算开销极低（仅修改损失函数中的范数计算和 lambda 更新），核心挑战在于验证其在 Llama、GPT-2 等大模型上的有效性，以及在大规模训练中 p 的最优调度策略是否改变

2. **在其他结构化领域设计类比的有监督评估指标**：拓展本文思路的核心难点是找到类似棋盘 BSPs 的可枚举、可验证 ground-truth 特征集合。潜在方向包括：
   - **编程语言模型**：利用 AST 节点类型、变量绑定、类型信息、控制流等作为 ground-truth BSPs
   - **数学证明模型**：以证明中间状态（已证明的子目标、可用引理）作为 ground-truth
   - **结构化知识领域**：以知识图谱中的实体和关系作为 ground-truth

3. **探索 p-Annealing 的更深理论基础**：研究 p-continuation（从 p=1 退火到 p<1）在 SAE 训练场景下的收敛性质，是否可进一步降低到更小的 p_end 值，甚至直接逼近 L0

4. **p-Annealing 与新兴 SAE 架构的组合**：p-annealing 与架构设计正交，可以尝试结合：
   - TopK SAE (Gao et al. [27])
   - BatchTopK / JumpReLU SAE
   - 其他替代 L1 惩罚的方案（如 tanh 惩罚 [31]、平方根 L1 [56]）

5. **自动化 BSPs 发现**：减少研究者偏见的方法——通过聚类 SAE 特征、使用对抗/对比方法自动挖掘潜在的 BSPs 类别，使 Coverage 更完备

6. **SAE 特征的下游应用验证**：在棋盘游戏设定中测试 SAE 特征在下游任务中的实用性（特征电路发现、概念编辑、干预实验），将 Coverage/Reconstruction 等"描述性"指标与"实用性"指标关联起来

7. **SAE 信息损失的系统研究**：线性探针与 SAE 之间的性能差距（如 Chess Reconstruction 0.98 vs 0.85）提示 SAE 存在固有信息损失。系统研究这种损失是来自架构限制（如 ReLU 激活函数、字典大小）、训练过程（如 L1/L_p 惩罚的副作用）还是稀疏性约束本身的信息瓶颈，对进一步提高 SAE 质量至关重要
