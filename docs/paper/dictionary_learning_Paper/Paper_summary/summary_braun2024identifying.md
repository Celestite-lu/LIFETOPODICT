# Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning

## 基本信息

- **标题**: Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning
- **作者**: Dan Braun, Jordan Taylor, Nicholas Goldowsky-Dill, Lee Sharkey
- **机构**: Apollo Research
- **发表年份/会议**: 2024 年，发表于 NeurIPS 2024
- **代码**: https://github.com/ApolloResearch/e2e_sae
- **Neuronpedia 交互页面**: https://www.neuronpedia.org/gpt2sm-apollojt
- **模型权重**: https://huggingface.co/apollo-research/e2e-saes-gpt2

---

## 研究动机与问题定义

### 核心痛点

在机械可解释性（mechanistic interpretability）领域，稀疏自编码器（Sparse Autoencoders, SAEs）已被广泛用于识别神经网络内部的特征。传统 SAEs 的训练目标是同时最小化重构 MSE 和稀疏惩罚项，以重建某一层的激活值。

然而，这一训练范式存在一个根本性问题：**MSE 重构误差所衡量的特征重要性，与这些特征在解释网络输出行为方面的功能重要性（functional importance）之间可能相关性很弱**。具体表现为：

1. **特征分裂（feature splitting）问题**：Bricken et al. (2023) 发现，随着字典尺寸增大和稀疏度提高，SAEs 会将一个特征分裂为多个更细粒度的特征，每个特征覆盖数据分布中越来越小的子集。在极限情况下，甚至可能将每个单独数据点表示为一个字典元素。
2. **容量浪费**：由于 MSE 重构损失不显式区分功能重要特征与次要特征，SAE 会将其固定容量大量浪费在学习功能上不重要的特征上。
3. **残差中的信息**：Marks et al. (2024) 观察到，当用 SAE 特征组成的 circuit 解释网络行为时，相当可观的信息量并非由 SAE 特征介导，而是由重构残差（即 SAE 未能捕获的部分）介导。

因此，论文提出核心问题：**如何识别网络中功能重要的特征？** 作者将功能重要特征定义为对解释网络在训练分布上行为起关键作用的特征。如果能优先学习功能重要特征，则应该可以用更少的每数据点特征、以及更少的总特征来维持对网络性能的解释力。

---

## 方法/框架

### 2.1 数学形式化

考虑一个具有 L 层的解码器-only Transformer，隐藏激活为 a^{(l)}。SAE 由编码器网络（仿射变换 + ReLU 激活）和单位范数字典特征组成，字典表示为矩阵 D，附带偏置向量 b_d：

```
Enc(a^{(l)}(x)) = ReLU(W_e * a^{(l)}(x) + b_e)
SAE(a^{(l)}(x)) = D^T * Enc(a^{(l)}(x)) + b_d
```

其中字典 D 和编码器权重 W_e 均为 (N_dict_elements x d_hidden) 矩阵，b_e 为 N_dict_elements 维向量，b_d 和 a^{(l)}(x) 为 d_hidden 维向量。

**重要设计**：每次前向传播时，字典 D 被归一化为单位范数（每个字典元素都是单位范数方向）。

主实验针对 GPT2-small 的残差流（residual stream），插入位置为 **attention layer 6 之前**，同时也在 layer 2 和 layer 10 进行了验证。

### 2.2 Baseline：局部 SAE 训练损失 L_local

标准的、基准的 SAE 训练方法。SAE 输出被训练为通过 MSE 损失重建其输入，同时对编码器激活加 L1 稀疏惩罚：

```
L_local = L_reconstruction + L_sparsity
        = ||a^{(l)}(x) - SAE_local(a^{(l)}(x))||_2^2 + phi * ||Enc(a^{(l)}(x))||_1
```

其中 phi = lambda / dim(a^{(l)})，lambda 为稀疏系数。除以激活维度是为了使稀疏系数对模型尺寸具有鲁棒性。

### 2.3 方法一：端到端 SAE 训练损失 L_e2e

本方法**不再训练 SAE 重构激活**，而是将原始模型激活替换为 SAE 输出，并将其前向传递通过后续网络层：

```
a_hat^{(l)}(x) = SAE_e2e(a^{(l)}(x))
a_hat^{(k)}(x) = f^{(k)}(a_hat^{(l)}(x)),  for k = l, ..., L-1
y_hat = softmax(f^{(L)}(a_hat^{(L-1)}(x)))
```

损失函数：**惩罚原始模型输出分布与插入 SAE 后的模型输出分布之间的 KL 散度**：

```
L_e2e = L_KL + L_sparsity = KL(y_hat, y) + phi * ||Enc(a^{(l)}(x))||_1
```

关键设计：**冻结模型所有参数，仅训练 SAE 参数**。这与 Tamkin et al. (2023) 形成对比，后者同时训练模型参数和 codebook。

### 2.4 方法二：端到端 + 下游层重构训练损失 L_e2e+downstream

L_e2e 的一个合理担忧是：即使冻结了原始模型参数，插入 SAE 后的模型可能通过不同的计算路径达到相似的输出分布。为缓解此问题，L_e2e+downstream 在 KL 散度和稀疏惩罚之外，额外最小化下游层的 MSE 重构误差：

```
L_e2e+downstream = KL(y_hat, y) + phi * ||Enc(a^{(l)})||_1
    + (beta_l / (L-l)) * SUM_{k=l+1}^{L-1} ||a_hat^{(k)}(x) - a^{(k)}(x)||_2^2
```

其中 beta_l 是控制下游重构损失项的超参数：
- **Layer 2 和 Layer 6**：beta_l = 2.5
- **Layer 10**：beta_l = 0.05

总系数 beta_l 被平均分配到所有下游层。

该损失函数兼具两个优势：
1. 激励 SAE 输出导致与原始网络相似的下游层计算
2. 允许 SAE "清除"一些非功能特征，因为不在 SAE 所在层引入重构损失

但论文也指出，下游重构项的引入可能鼓励 SAE 学习功能重要性相对较低的特征。

**实现细节**：SAE_e2e+ds 的 KL 损失项在实现中乘以 0.5。这是一个工程选择——如果固定为 1 并变化其他系数，则需要相应调整学习率和有效批次大小。

### 2.5 实验指标

1. **CE Loss Increase（交叉熵损失增量）**：原始模型与插入 SAE 后模型的 CE 损失之差。增量越低，SAE 对网络性能的解释力越强（有时也称作"性能解释量"）。
2. **L0**：每个数据点平均激活的 SAE 特征数量。
3. **Alive Dictionary Elements（存活字典元素数）**：训练期间未"死亡"（即在 500k tokens 上至少激活过一次）的特征数量。更低意味着更高效率。
4. **下游层重构误差**：SAE 插入后在下游层的 MSE，衡量计算路径的一致性。
5. **自动化可解释性评分**：基于 Bills et al. (2023) 的方法。用 GPT-4-turbo-2024-04-09 基于 5 个最大激活样本生成特征解释，然后用 GPT-3.5-turbo 基于该解释预测 20 个最大激活样本上的真实激活值。

---

## 实验设置

### 模型

- **GPT2-small**（Radford et al., 2019）：12 层 Transformer，残差流维度 768。评估集 CE loss = 3.139。
- **Tinystories-1M**（Eldan and Li, 2023）：8 层，1M 参数。评估集 CE loss = 2.306。

### 数据集

- Open Web Text（Gokaslan and Cohen, 2019），使用 GPT2 tokenizer 进行 tokenization。
- 训练：400k 样本，上下文长度 1024。
- 评估：500 个样本（不同随机 seed）。
- 字典元素"存活"标准：在 500k 训练 tokens 上至少激活过一次。

### 字典配置

- 字典大小为残差流尺寸的 **60 倍**：60 x 768 = 46,080 个初始字典元素。
- 编码器和解码器均有可训练偏置，使用 Kaiming 初始化。
- 每次前向传播时将解码器归一化为单位范数。
- **不使用重采样技术**（不同于 Bricken et al. 2023），因为其对于 e2e 训练中特征发现的影响不明确。

### 训练超参数

- **学习率**：5e-4（0.0005），warmup 20k 样本，cosine 调度衰减至最大学习率的 10%
- **有效批次大小**：16
- **优化器**：Adam（默认超参数：beta1=0.9, beta2=0.999）
- **梯度范数裁剪**：10（针对 GPT2-small；仅影响训练开始时非常大的梯度和后续偶发的尖峰）
- 稀疏系数 lambda 除以 dim(a^{(l)}) 以缩放

### 学习率选择

每种 SAE 类型的学习率根据其 L0 vs CE loss increase 曲线的 Pareto 最优性选择。实验设计以最大化 SAE_local 的 L0 vs CE loss increase Pareto 前沿为目标，然后将相同的设计选择用于 SAE_e2e 和 SAE_e2e+ds。大部分设计迭代在较小的 Tinystories-1M 上进行。

### 训练时长（单张 NVIDIA A100 80GB GPU）

| Layer | SAE_local | SAE_e2e | SAE_e2e+ds |
|-------|-----------|---------|------------|
| 2     | 3h45m     | 12h24m  | 12h30m     |
| 6     | 4h45m     | 11h20m  | 11h24m     |
| 10    | 5h19m     | 10h12m  | 10h20m     |

e2e SAEs 训练时间约为 SAE_local 的 **2-3.5 倍**。

### GPU

NVIDIA A100 80GB VRAM（使用较小批次大小时 GPU 未被充分利用，占用 40GB 或更少）。

---

## 核心结果与发现

### 3.1 端到端 SAEs 相比局部 SAEs 是严格的 Pareto 改进

**关键定量结果（GPT2-small Layer 6 主实验）**：

- 对于相同的 L0，SAE_e2e 和 SAE_e2e+ds 的 CE 损失增量不到 SAE_local 的 **45%**（即 SAE_local 需要超过 2.2 倍的 CE 损失增量才能达到同等的 L0）。

**三个等 CE 损失增量 SAEs 的详细对比（Table 1, Layer 6）**：

| SAE 类型 | lambda (稀疏系数) | L0   | 存活元素 | CE 损失增量 |
|----------|-------------------|------|---------|------------|
| Local    | 4.0               | 69.4 | 26k     | 0.145      |
| e2e      | 3.0               | 27.5 | 22k     | 0.144      |
| e2e+ds   | 50.0              | 36.8 | 15k     | 0.125      |

关键发现：e2e+ds 使用约 **53%** 的 L0（36.8 vs 69.4）和约 **58%** 的存活字典元素（15k vs 26k），同时实现了更低的 CE 损失增量（0.125 vs 0.145）。e2e 仅用不到一半的 L0（27.5）达到了几乎相同的性能（CE 损失增量差仅 0.001）。

**三个核心声明得到验证**：
1. 对于相同性能解释量，SAE_local 需要激活**超过两倍**的每数据点特征（相比 e2e SAEs）。
2. SAE_e2e+ds 在 L0 vs CE 方面与 SAE_e2e 表现相当，但其激活路径与 SAE_local 更为相似。
3. SAE_local 需要更多的数据集总体特征来解释同等的网络性能（相比 e2e SAEs）。

**计算量对比测试**：将字典尺寸从 60x768 扩大到 100x768，或将训练样本从 400k 增加到 800k，均**未显著改善** SAE_local 的 Pareto 前沿。说明 e2e SAE 的优势不仅仅来自更多计算量。

**其他层和模型的验证**：
- GPT2-small 的 Layer 2 和 Layer 10 上定性一致（Appendix A.1）
- Tinystories-1M 的 Layer 5（共 8 层）上定性一致（Appendix A.2）

注：GPT2-small 在评估集上的 CE loss = 3.139；Tinystories-1M CE loss = 2.306。

### 3.2 尽管输出分布相似，端到端 SAE 在各层的重构误差更差

SAE_e2e 在下游层的 MSE 重构误差比 SAE_local 高出一个数量级（Layer 6 约 14.5 vs 约 0.5），SAE_e2e+ds 则为约 3.5。这说明 SAE_e2e 的激活可能通过不同的计算路径达到相似的输出。

然而，在下游层中 SAE_e2e+ds 的重构误差与 SAE_local 极为接近。因此 **SAE_e2e+ds 实现了理想的平衡**：
- 既能解释与原模型相近的网络性能（与 SAE_e2e 相当）
- 又能保持与原模型相似的计算路径（与 SAE_local 接近）

Layer 6 本身的重构差异是预期的——SAE_e2e+ds 不在该层训练重构损失。Appendix B 进一步分析了该差异在多大程度上由特征缩放（feature scaling）解释。Appendix G.3 发现了一个具体实例：SAE_local 忠实重建了一个功能重要性低的方向，而 SAE_e2e+ds 忽略了它。

### 3.3 不同 SAE 类型的特征几何差异

#### 3.3.1 端到端 SAE 特征正交性更强（特征分裂更少）

衡量每个字典特征与同字典中最近邻特征的余弦相似度（作为特征分裂的代理指标）：

- SAE_local 的特征更加紧凑聚集，表明更严重的特征分裂。
- 相比 SAE_e2e+ds，SAE_local 的平均余弦相似度**高 0.04**（bootstrap 95% CI: [0.037, 0.043]）。
- 相比 SAE_e2e，SAE_local 的平均余弦相似度**高 0.166**（95% CI: [0.163, 0.168]）。

该差异并非由存活字典元素数量不同所致——对所有 Pareto 前沿中的 run 进行测量后发现关系成立（Appendix A.5）。

#### 3.3.2 SAE_e2e 特征对随机种子不鲁棒，但 SAE_e2e+ds 和 SAE_local 鲁棒

- SAE_local 和 SAE_e2e+ds 在不同随机种子下学到相似的特征。
- SAE_e2e 在不同种子下学到的特征差异很大，说明存在多个不同的特征集合可以实现相同的输出分布（但通过不同的计算路径）。

#### 3.3.3 SAE_e2e 与 SAE_local 特征并非总是对齐

- SAE_e2e 与 SAE_local 之间的平均最近邻余弦相似度较低，包含一组相似度很低的特征。
- SAE_e2e+ds 与 SAE_local 更相似，但余弦相似度分布呈**双峰性**，表明 SAE_e2e+ds 也学习到了一些与 SAE_local 差异很大的方向。
- 这暗示 SAE_local 可作为 SAE_e2e+ds 的良好初始化，减少训练时间。

### 3.4 可解释性分析

使用自动化可解释性库，对每个 SAE 随机采样 198-201 个特征进行评分：

- **相似 L0 配对比较（Table 3）**：SAE_e2e+ds 与 SAE_local 的自动化可解释性得分无显著差异（Layer 2: p=0.61; Layer 6: p=0.71; Layer 10: p=0.33）。
- **相似 CE 损失增量配对比较（Table 2）**：
  - Layer 2：SAE_e2e+ds **更可解释**（均值差异 +0.08, 95% CI [0.02, 0.13], p=0.0057）
  - Layer 6：SAE_e2e+ds **更可解释**（均值差异 +0.10, 95% CI [0.05, 0.16], p=0.00044）
  - Layer 10：无显著差异（均值差异 +0.02, p=0.41）

**结论**：e2e SAEs 在效率上的提升**未以牺牲可解释性为代价**。在 Layer 2 和 6 的相似 CE 损失增量比较中，SAE_e2e+ds 甚至显著更具可解释性。

### 3.5 定性分析：UMAP 可视化（Appendix G）

使用 UMAP 降维可视化（McInnes et al., 2018）比较 SAE_e2e+ds 与 SAE_local 的特征分布（Figure 19, Layer 6），识别出多个差异区域：

- **Region A**（位置特征）：SAE_local 91 个 vs SAE_e2e+ds 18 个特征——涉及后期上下文位置特征。SAE_e2e+ds 的位s features 更少。
- **Region B**（文档边界）：SAE_e2e+ds 48 个 vs SAE_local 2 个——主要在 `<|endoftext|>` token 和换行符上激活。SAE_e2e+ds 具有更多文档边界特征。
- **Region C**（"by/from" 区域）：SAE_local 31 个 vs SAE_e2e+ds 20 个。SAE_local 表现出极端的特征分裂："goes by"、"led by"、". By"、"stop by"、"Posted by"、"Directed by"、"killed by" 等各自对应独立特征，而 SAE_e2e+ds 使用更广泛的上下文特征。
- **Region D**（"at" 区域）：类似模式——SAE_local 19 个 vs SAE_e2e+ds 11 个，SAE_local 分裂更细粒度。
- **Region E**（段落转接）：SAE_local 67 个 vs SAE_e2e+ds 仅 3 个——SAE_local 有大量特征用于区分不同上下文中的 "Moreover"、"Furthermore"、"However" 等词（如烘培食谱 vs 技术性写作）。
- **Region G**（混合区域）：SAE_local 71 个 vs SAE_e2e+ds 41 个，SAE_e2e+ds 特征倾向于更专一地在更少 token 上激活。

总体模式：SAE_local 持续表现出更多的特征分裂（更多特征覆盖更细粒度的区分）。

### 3.6 功能无关方向的个案研究（Appendix G.3）

在 Layer 10 的 UMAP 中，Region H 显示 SAE_local 包含 593 个特征，而 SAE_e2e+ds 仅有 2 个。这 593 个特征均指向远离第 0 个 PCA 方向的方向。

第 0 个 PCA 方向对应于 position 0 处的异常激活（图 22 显示该 PCA 分量呈三模态分布：position 0 处的大离群值、end-of-text 处的小离群值和其他位置）。

该方向在非零位置**功能上几乎无关**：对 position > 0 的该方向进行 resample ablation 后，KL 散度仅为 **0.01**，远低于对其他 PCA 方向（前 30 个）执行相同干预的 KL 散度（范围约 0.02-0.22）。

**重建忠实度对比**：
- SAE_e2e+ds 在 position 0 忠实重建该方向（相关系数 r = 0.996），但在 position > 0 重建极差（r = 0.262）
- SAE_local 在所有位置忠实重建（r 约等于 1.0）

**启示**：SAE_local 将大量容量浪费在功能无关的特征上，而 SAE_e2e+ds 通过"放弃"对功能无关方向的准确重建，获得了更干净（更少冗余）的字典。这是为何优化 MSE 会导致学习非功能特征的具体实例。

### 3.7 补充实验：主语-动词一致性任务（Appendix I）

在 Finlayson et al. (2021) 的四类任务（Simple、Across PP、Across RC、Within RC）上评估。结果表明：**e2e SAEs 未在任何特定下游任务组合上呈现明显的一致优势**（Table 7a 和 7b）。不同 SAE 类型和层表现无一致规律。这些任务特定结果噪音较大，需系统地聚合大量任务和模板才能区分各种 SAE 训练方法。如 layer 10 的 e2e+ds 在 Simple 上忠实度为 104.7%，而 layer 6 的 e2e+ds 在 Across PP 上仅为 79.2%。

---

## 补充实验发现（附录）

### 学习率效应（Appendix C）

- 高于默认 5e-4 的学习率可改善 L0 vs CE loss increase 和 Alive Elements vs CE loss increase 的 Pareto 前沿。
- 此效应在 SAE_e2e 和 SAE_e2e+ds 上比 SAE_local 更显著。
- **缺点**：高学习率（>= 0.001）导致 L0 在初始下降后持续增加（尤其在后层），因此主实验坚持使用 5e-4。使用稀疏度调度可能解决此问题。

### 字典尺寸扫描（Appendix E.1）

- L0 vs CE loss increase 随字典尺寸增加而改善，但有递减收益，在 ratio 约等于 60 处趋于饱和。
- 更大字典带来更多存活元素。
- 在 Tinystories-1M 上，ratio 低至 5 仍能实现良好的权衡。

### 训练样本数扫描（Appendix E.2）

- SAE_local：200k、400k、800k 样本间差异很小（50k 时明显不足）。
- SAE_e2e：从 50k 到 800k 有递减改善；在 alive dict elements vs CE loss increase 上持续改善。
- SAE_e2e+ds：400k 后性能饱和。
- 800k 样本的 e2e 或 e2e+ds 训练约需 23 小时（单张 A100）。

### 特征抑制分析（Appendix B）

e2e SAE 存在特征抑制（SAE 输出范数显著小于输入范数）：

L2 Ratio = E[||a_hat(x)|| / ||a(x)||]，Layer 6 相似 CE SAEs：

| SAE 类型 | Position 0 | Position > 0 |
|----------|-----------|--------------|
| local    | 1.00      | 0.92         |
| e2e      | 0.11      | 0.31         |
| e2e+ds   | 0.99      | 0.56         |

- SAE_e2e 的特征抑制最严重（position > 0 的 L2 ratio 仅 0.31）。
- 残差流上的 LayerNorm 可重新归一化被压缩的激活，使网络维持相似输出。
- 对激活进行 LayerNorm 归一化后比较，e2e SAE 的解释方差大幅提升，但相对曲线形状不变（Appendix B.3）。

### 梯度范数与存活字典元素的关系（Appendix C）

存活字典元素数量与训练期间的梯度范数呈负相关。更高的学习率通常导致更大的梯度范数和更少的存活元素。然而，超大学习率带来的训练不稳定性限制了这一收益。

### 跨种子鲁棒性（Appendix F）

在不同稀疏系数和层上，训练过程对随机种子具有良好的鲁棒性（包括 SAE 权重初始化和训练/评估数据采样）。

---

## 主要贡献

1. **提出端到端字典学习方法**：首次通过最小化原始模型与插入 SAE 后模型输出分布之间的 KL 散度来训练 SAE，直接优化功能重要性而非仅重构激活。模型参数保持冻结，仅 SAE 被训练。
2. **两种端到端变体**：SAE_e2e（纯 KL 散度）最大化每特征性能解释力；SAE_e2e+ds（KL 散度 + 下游层 MSE 重构）在功能重要性和计算路径一致性之间取得平衡。
3. **严格实验验证**：在 GPT2-small 和 Tinystories-1M 上证明 e2e SAEs 在 L0 vs CE loss increase 和 Alive Elements vs CE loss increase 两个 Pareto 前沿上均构成改进。相同性能解释量下所需每数据点特征不到一半（e2e SAEs 的 CE loss increase 不到 SAE_local 的 45%）。
4. **全面几何和可解释性分析**：通过余弦相似度、跨种子比较、跨类型比较、自动化可解释性评分（约 200 个特征/SAE）和 UMAP 定性分析，多维证明效率提升不牺牲可解释性。
5. **完整开源工具链**：训练库（https://github.com/ApolloResearch/e2e_sae）、W&B 训练指标报告、Neuronpedia 交互页面（https://www.neuronpedia.org/gpt2sm-apollojt）、HuggingFace 模型权重（https://huggingface.co/apollo-research/e2e-saes-gpt2）。

---

## 局限性

1. **训练时间增加**：e2e SAEs 的训练时间是 SAE_local 的 2-3.5 倍（每批需额外前向传播通过模型剩余层）。
2. **模型和架构范围有限**：仅在 GPT2-small（解码器-only Transformer）和 Tinystories-1M 上测试，尚未验证在其他架构（如编码器-解码器、更大规模模型）上的泛化性。
3. **仅在残差流上测试**：未测试 attention 输出、MLP 输出等其他激活位置的效果。
4. **下游任务评估 inconclusive**：主语-动词一致性实验未显示 e2e SAEs 的明确优势，具体受益的下游任务仍需探索。
5. **SAE_e2e 计算路径不可靠**：下游层 MSE 比 SAE_local 高一个数量级以上，可能采用与原始模型不同的计算机制。
6. **e2e 方法超参数未充分调优**：beta_l 仅粗略设置（Layer 2/6 为 2.5，Layer 10 为 0.05），未探索不同层的不同权重。KL 项乘 0.5 为临时工程选择。所有超参数针对 SAE_local 优化，非 e2e SAEs。
7. **高学习率训练不稳定**：高 LR 改善 Pareto 前沿但导致 L0 在初始下降后持续增长。稀疏度调度可能是解决方向。
8. **特征抑制问题**：e2e SAE 特征值被系统性压缩（SAE_e2e 最严重），SAE 输出范数显著小于输入范数（Layer 6 position > 0 的 L2 ratio 仅 0.31）。

---

## 对后续研究的启发与潜在改进方向

1. **架构改进**：结合 Gated SAE（Rajamanoharan et al., 2024）等架构创新，将门控机制与端到端训练结合，同时抑制特征抑制并改善 Pareto 前沿。
2. **损失函数优化**：探索不同层不同下游重构权重、在 SAE 所在层加入适量重构损失以更好地平衡功能重要性和几何忠实度、使用 JS 散度或其他散度替代 KL 散度。
3. **加速训练**：用 SAE_local 初始化 SAE_e2e+ds（两者特征有相似性）、同时训练多层 SAE、或减少字典 ratio（Tinystories-1M 上 ratio=5 即可）和训练样本数以显著提速。
4. **更大规模验证**：在更大模型（Pythia、Llama 系列）和各种架构上测试 e2e SAE 方法，验证可扩展性和泛化能力。
5. **Circuit Discovery 应用**：将 e2e SAEs 用于构建 Sparse Feature Circuits（Marks et al., 2024），测试是否能发现更稀疏、更准确的 circuits。
6. **任务特定的端到端目标**：将 KL 散度替换为或补充以任务特定的目标函数（如行为一致性、可控生成等），使 SAE 偏向特定功能特征。
7. **稀疏性调度**：利用稀疏度调度（训练早期鼓励更稀疏表示）稳定高学习率训练，在实现更少存活字典元素的同时维持甚至提升性能。
8. **更深入的几何理解**：研究为何 SAE_e2e 特征跨种子不鲁棒而 SAE_e2e+ds 鲁棒，以及这一差异对可解释性和下游使用的理论与实践影响。
9. **特征抑制的根本解决方案**：将 Wright and Sharkey (2024) 的训练后微调方法或 Jermyn et al. (2024) 的 Tanh 惩罚等替代稀疏惩罚引入 e2e 训练框架。
10. **多任务评估策略**：系统聚合大量任务和模板以差异化评估各种 SAE 训练方法，因为单一任务评估结果噪音较大。