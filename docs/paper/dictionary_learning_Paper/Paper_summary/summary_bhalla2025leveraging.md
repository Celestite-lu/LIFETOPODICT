# 论文详细结构化总结：Leveraging the Sequential Nature of Language for Interpretability

## 一、基本信息

- **标题**：Leveraging the Sequential Nature of Language for Interpretability
- **作者**：Usha Bhalla\*, Alex Oesterling\*, Claudio Mayrink Verdun, Flavio P. Calmon, Himabindu Lakkaraju（\* 表示同等贡献）
- **单位**：Harvard University / Kempner Institute
- **发表会议/年份**：ICML 2025 Workshop on Assessing World Models (2025)
- **论文类型**：Regular Research Paper（非综述）

---

## 二、研究动机与问题定义

### 核心痛点

1. **现有字典学习方法忽视数据的模态特定结构**：Sparse Autoencoders（SAEs）作为当前无监督概念发现的主流方法，已被广泛应用于语言、视觉、X光、蛋白质序列、时间序列等多种模态，但几乎没有任何针对不同数据类型的架构调整。这种"开箱即用"的直接应用假设数据无底层结构，虽然带来了灵活性，但放弃了对数据模态的已有先验知识和归纳偏置（inductive biases）。

2. **语言的时间/序列维度未被利用**：自然语言具有天然的序列性质（sequential nature），当前 SAE 方法将训练数据中的每个 token 视为独立同分布（i.i.d.），完全忽略了 token 之间的时间依赖关系。

### 问题定义

如何在字典学习（尤其是 SAE）中利用自然语言的时序结构，以更好地从语言模型的中间表示中解耦和提取语义特征（semantic features）与句法特征（syntactic features）。

### 认知科学依据

论文借鉴认知科学和计算语言学的研究成果（Griffiths et al., 2004）：**语义内容具有长程行为（long-range behavior）**——同一文档中不同词或句子持有一致的语义内容；**句法主要依赖短程交互（short-range interactions）**。这为设计时序一致的 SAE 训练目标提供了理论基础。

---

## 三、方法/框架：Temporal Sparse Autoencoders

### 3.1 数据生成过程（Data-Generating Process, DGP）

论文为语言数据的生成过程构建了一个概率图模型：

考虑一个说话者生成 token 序列 tau_1, ..., tau_T。每个 token tau_t 的生成取决于以下因素：
- **高层/全局隐变量 h_t**：捕获语义和意图等与特定 token 无关的特征（如主题、意图、情感）
- **低层/局部隐变量 l_t**：捕获特定时间步的 token 级特征（如语法性别、词性）
- **上文 tau^{t-1}**：即之前已生成的 token 序列 tau_1, ..., tau_{t-1}

生成过程形式化为随机映射函数：

```
tau_t = phi(tau^{t-1}, h_t, l_t)
```

其中 phi 是一个随机函数。给定语言模型 M，将完整序列 tau^T 输入 M，在第 L 层得到隐向量序列 {x_t^L} in R^d（t = 1, ..., T）。为简化分析，论文聚焦于单一层并省略上标 L。

**DGP 的两个核心假设**：

1. **h_t 是时间不变的（time invariant）**：来自同一序列的任意两个 token x_t 和 x_{t'} 应具有相似的高层隐变量，即 h_t approx h_{t'}。直观上，对于给定的一段文本，其语义在时间上应相对恒定。

2. **h_t 和 l_t 在语言模型中以加法方式表示**：语言模型的隐向量 x_t 可分解为 x_t = x_t^h + x_t^l，其中 x_t^h 捕获高层特征（用于生成 h_t），x_t^l 捕获低层特征。

### 3.2 模型架构

Temporal SAE 将标准 SAE 的特征空间划分为两个部分：
- **前 h 个特征**（W_{0:h}^dec in R^{d x h}）：对应高层特征 h_t，用于重建语义/上下文信息
- **后 m-h 个特征**（W_{h:m}^dec in R^{d x (m-h)}）：对应低层特征 l_t，用于重建 token 级细节和句法信息
- 总特征数：m
- 特征划分比例（手动设定）：Pythia-160M 上高层占 20%，Gemma2-2b 上高层占 10%

**标准 SAE 操作**（输入 x_t in R^d）：

```
f(x_t) = sigma(W^enc x_t + b^enc)          （编码器，sigma 为非线性激活函数）
x_hat(f) = W^dec f(x_t) + b^dec           （解码器，线性重构）
```

其中 W^enc in R^{m x d} 为编码器矩阵，W^dec in R^{d x m} 为解码器矩阵（由高层特征 W_{0:h}^dec 和低层特征 W_{h:m}^dec 拼接而成），b^enc in R^d 和 b^dec in R^d 分别为编码器和解码器偏置。

### 3.3 损失函数（核心创新）

总损失由三部分组成，超参数 alpha 控制对比损失的权重：

```
L(x_t) = L_H + L_L + alpha * L_contr
```

**（1）高层重构损失 L_H**：仅使用前 h 个高层特征进行重构，强制这些特征捕获输入中最主要的信息（即语义和上下文信号）。

```
L_H = ||x_t - W_{0:h}^dec * f_{0:h}(x_t) + b^dec||_2^2
```

**（2）全部特征重构损失 L_L**：使用全部 m 个特征进行重构。其中低层特征（后 m-h 个）自然负责重构高层特征未能解释的残差。该设计在形式上与 Matryoshka SAE（Bussmann et al., 2025）的目标类似，但 Matryoshka SAE 没有引入时序约束。

```
L_L = ||x_t - W^dec * f(x_t) + b^dec||_2^2
```

**（3）时序对比损失 L_contr（核心创新）**：该损失鼓励同一序列中相邻 token 的高层特征表示彼此相似，而与来自不同序列（batch 内其他随机序列）的 token 的高层特征表示相互远离。令 z_t = f_{0:h}(x_t) 为高层特征向量（即编码器输出的前 h 维），s(x, y) 为余弦相似度，lambda 为温度超参数：

```
L_contr = - (1/N) SUM_{i=1}^{N} log [ exp(s(z_t^{(i)}, z_{t-1}^{(i)}) / lambda) / SUM_{j=1}^{N} exp(s(z_t^{(i)}, z_{t-1}^{(j)}) / lambda) ]
          - (1/N) SUM_{j=1}^{N} log [ exp(s(z_{t-1}^{(j)}, z_t^{(j)}) / lambda) / SUM_{i=1}^{N} exp(s(z_{t-1}^{(i)}, z_t^{(j)}) / lambda) ]
```

其中 N 为 batch size，z_t^{(i)} 为 batch 中第 i 个样本的高层隐向量。这是一个**双向 InfoNCE 对比损失**（与 SimCLR 和 CLIP 的形式类似）：

- 第一个求和项以 z_t 为锚点（anchor），正样本为同一序列的 z_{t-1}，负样本为 batch 中其他序列的 z_{t-1}
- 第二个求和项以 z_{t-1} 为锚点，正样本为同一序列的 z_t，负样本为 batch 中其他序列的 z_t

**实际训练中的数据处理**：
- 以成对数据 (x_t, x_{t-1}) 的形式加载激活值
- 对 batch 内这些成对数据进行打乱（shuffle），以确保负样本的多样性
- 论文还探索了从历史所有 token {x_1, ..., x_{t-1}} 中均匀随机采样第二个 token 的变体（而非固定使用前一个 token x_{t-1}），以鼓励**长程序列语义一致性**而不仅是相邻 token 间的短程一致性（详见附录 A.1.1）

**直觉解释**：对于高级、抽象、语义性的特征，来自同一序列的 token 应具有比来自两个随机序列的 token 更相似的高层激活。时序对比损失正是约束高层字典将这些语义相近的表示在 SAE 特征空间中编码得彼此靠近。对于低层的 token 特异性特征，由于无需满足相同的时序平滑要求，同时通过拟合高层模块留下的残差，低层特征自然被引导去捕获序列中方差较高的特征。

### 3.4 训练与实现细节

- **激活函数**：所有模型使用 BatchTopK 激活函数（Bussmann et al., 2024）
- **辅助损失**：使用 Gao et al. (2024) 提出的辅助损失，鼓励死亡隐变量（dead latents）重构已有重构留下的误差
- **稀疏度**：Pythia-160M 使用 k-sparsity = 20；Gemma2-2b 使用 k = 40
- **训练数据**：RedPajama-Data-1T-Sample 数据集（Weber et al., 2024）

---

## 四、实验设置

### 4.1 模型与数据配置

| 项目 | 设置 |
|---|---|
| **语言模型** | Pythia-160M（第 8 层 residual stream，768 维，SAE 含 32k 特征）、Gemma2-2b（第 18 层 residual stream，SAE 含 16k 特征） |
| **训练数据** | RedPajama-Data-1T-Sample 数据集（Weber et al., 2024） |
| **评测数据** | Pile 数据集中的 128k token 样本（Gao et al., 2020）。语义标签来自 FineFineWeb 和 Wikipedia 两个数据集 |
| **分析的激活位置** | Residual stream（残差流） |

### 4.2 基线方法（Baselines）

- **BatchTopK SAE**（Bussmann et al., 2024）：使用 BatchTopK 稀疏激活机制的标准 SAE，未划分高/低层特征空间
- **Matryoshka SAE**（Bussmann et al., 2025）：学习多层嵌套特征的 SAE，有类似的高/低层特征划分设计（递归重构），但不包含任何时序约束

### 4.3 评测指标

**（1）重构质量指标**（在 128k token 的 Pile 评测子集上计算）：

- **FVU（Fraction of Variance Unexplained，未解释方差比例）**：衡量 SAE 重构 x_hat 未能解释的方差比例。公式为 1 - Var(x - x_hat) / Var(x)，其中 Var 为经验方差。越低越好。
- **L2 Loss**：均方重构误差，即 ||x - x_hat||_2^2。越低越好。
- **Cosine Similarity（余弦相似度）**：重构向量与输入向量的余弦相似度，<x, x_hat> / (||x||_2 * ||x_hat||_2)。越高越好。
- **% Dead Latents（死亡隐变量比例）**：在评测数据集上任何 token 都未激活的 SAE 隐特征的百分比。越低越好，代表特征利用率更高。

**（2）语义/句法/上下文特征提取质量指标**：

使用**线性探针（linear probing）**方法，在 SAE 分解得到的表示上训练分类器以预测以下三类标签：
- **语义标签（Semantics）**：来自 FineFineWeb（类别如 "literature", "food", "drama and film", "mathematics", "medical"）和 Wikipedia（类别如 "Geography", "History", "Knowledge", "People", "Religion"）
- **句法标签（Syntax）**：使用词性标注器（part-of-speech tagger）自动构造
- **上下文标签（Context）**：按序列/文章来源将 token 分组，即区分 token 来自哪个文档或序列

同时以直接对基础语言模型隐变量（未经 SAE 处理）进行 probing 的结果作为上界基线（"Base Model"），代表语言模型原生编码的信息量。目标是 SAE 在将表示转换到更可解释空间的过程中尽可能保留这些信息。

**（3）定性可视化**：使用 TSNE（Van der Maaten & Hinton, 2008）对 SAE 特征激活进行 2D 降维可视化，并按语义类别、上下文来源、词性标签分别着色观察聚类效果。

**（4）PCA 降维分析**（附录 A.1.3）：通过 PCA 将表示投影到不同维度的子空间后训练 l2 正则化线性探针，使用 80%-20% 训练-测试划分，考察不同信息（语义/句法/上下文）在表示各维度的分布特征。

---

## 五、核心结果与发现

### 5.1 重构质量对比（Table 1）

Temporal SAE 加入时序约束并未显著降低重构质量，与基线基本持平：

| 模型 | SAE 类型 | FVU down | % Dead down | L2 down | CosSim up |
|---|---|---|---|---|---|
| **Pythia-160M** | **Temporal** | **0.09** | **34%** | **10.4** | **0.89** |
| | Matryoshka | 0.07 | 44% | 9.2 | 0.91 |
| | BatchTopK | 0.07 | 62% | 9.3 | 0.91 |
| **Gemma2-2b** | **Temporal** | **0.41** | **1%** | **131** | **0.86** |
| | Matryoshka | 0.38 | 1% | 119 | 0.88 |
| | BatchTopK | 0.38 | 1% | 122 | 0.88 |

**关键发现**：
- Temporal SAE 在重构指标上仅产生极小损失。对于 Pythia，FVU 从 0.07 到 0.09（增加 0.02），L2 从 9.2-9.3 到 10.4（增加约 12%）；对于 Gemma，三者的 FVU 和 L2 基本接近。
- **Temporal SAE 在 Pythia 上显著减少了死亡特征**：34%（Temporal）vs 44%（Matryoshka）vs 62%（BatchTopK），说明时序约束反而促进了更多特征的激活和利用。在 Gemma 上三类方法均为 1%，这可能是 Gemma 的更大容量和训练充分性使得死亡特征问题本身就不严重。

### 5.2 语义/句法/上下文分类性能

**FineFineWeb 数据集的探针准确率（Table 2）**：

| 模型 | 方法 | 语义 up | 句法 up | 上下文 up |
|---|---|---|---|---|
| **Pythia-160M** | Base Model（上界） | 0.703 | 0.865 | 0.835 |
| | **Temporal SAE** | **0.647** | 0.796 | **0.799** |
| | Matryoshka SAE | 0.576 | 0.848 | 0.430 |
| | BatchTopK SAE | 0.587 | 0.847 | 0.654 |
| **Gemma2-2b** | Base Model（上界） | 0.720 | 0.875 | 0.974 |
| | **Temporal SAE** | **0.682** | 0.857 | **0.964** |
| | Matryoshka SAE | 0.633 | 0.877 | 0.895 |
| | BatchTopK SAE | 0.648 | 0.876 | 0.918 |

**Wikipedia 数据集的探针准确率（Table 3）**：

| 模型 | 方法 | 语义 up | 句法 up | 上下文 up |
|---|---|---|---|---|
| **Pythia-160M** | Base Model（上界） | 0.718 | 0.832 | 0.869 |
| | **Temporal SAE** | **0.633** | 0.770 | **0.731** |
| | Matryoshka SAE | 0.568 | 0.824 | 0.582 |
| | BatchTopK SAE | 0.559 | 0.815 | 0.605 |
| **Gemma2-2b** | Base Model（上界） | 0.939 | 0.873 | 0.927 |
| | **Temporal SAE** | **0.896** | 0.865 | **0.882** |
| | Matryoshka SAE | 0.882 | 0.877 | 0.817 |
| | BatchTopK SAE | 0.896 | 0.875 | 0.846 |

**核心分析**：

1. **语义和上下文能力大幅领先**：Temporal SAE 在语义和上下文两个维度上全面超越所有基线。在 FineFineWeb 上，相对于最佳基线的相对提升约为：Pythia 语义 +10.2%、上下文 +22.2%；Gemma 语义 +5.2%、上下文 +5.0%。在 Wikipedia 上，Temporal SAE 同样在语义和上下文上领先（除 Gemma 的 Wikipedia 语义与 BatchTopK 持平外）。

2. **基线 SAE 丧失语义/上下文信息**：Matryoshka SAE 和 BatchTopK SAE 虽然在句法上表现良好（接近 Base Model），但上下文分类能力极差——尤其是 Pythia 上的 Matryoshka SAE，上下文准确率仅 0.430（FineFineWeb）和 0.582（Wikipedia），不到 Base Model（0.835/0.869）的一半。这意味着标准 SAE 在以重构为导向的训练中几乎丢失了上下文的区分能力。

3. **句法性能的轻微权衡**：Temporal SAE 在句法分类上略低于基线 SAE（如 Pythia FineFineWeb 上 0.796 vs 0.848），但仍保持在 Base Model（0.865）的约 92% 水平。这是一种有意义的权衡（trade-off）——用少量句法精度换取了显著的语义和上下文增益。

4. **Gemma2-2b 上差距较小**：由于 Gemma2-2b 本身表示能力更强（Base Model 语义和上下文准确率更高），SAE 之间的差距相应缩小，但 Temporal SAE 的优势趋势仍然一致。

### 5.3 TSNE 可视化分析（Figure 1, Appendix Figure 2）

- **Matryoshka SAE（基线）**：TSNE 图上仅在按词性（Part-of-Speech）标签着色时出现清晰的聚类模式；按语义类别着色时散点杂乱无章，按上下文着色时也几乎无规律可循。这表明 Matryoshka SAE 几乎只保留了句法信息，丧失了全部语义和上下文结构。

- **Temporal SAE**：在三种标签（语义类别、上下文来源、词性）下均可观察到明显的聚类结构。直观上，上下文聚类最为清晰（因为对比损失直接优化这一目标），语义聚类次之，句法聚类也存在。这表明 Temporal SAE 成功地同时保留了语义、上下文和句法三类信息。

- 在 Wikipedia 数据集上的 TSNE 图（Appendix Figure 2）呈现出一致的定性结果。

### 5.4 消融实验：相邻 Token vs 随机历史 Token 采样（Table 4）

| 方法 | Pythia 语义 up | Pythia 句法 up | Pythia 上下文 up |
|---|---|---|---|
| Base Model（上界） | 0.703 | 0.865 | 0.835 |
| Temporal SAE (t-1) | 0.647 | 0.796 | 0.799 |
| **Temporal SAE (random)** | **0.687** | **0.839** | **0.830** |
| Matryoshka SAE | 0.576 | 0.848 | 0.430 |
| BatchTopK SAE | 0.587 | 0.847 | 0.654 |

**关键发现**：
- 从历史所有 token 中均匀随机采样第二个 token（而非固定使用前一个 token x_{t-1}），在三个维度上均有提升，尤其**句法准确率从 0.796 大幅提升到 0.839**（接近 Matryoshka/BatchTopK 的 0.848/0.847），且语义和上下文也分别从 0.647 和 0.799 提升到 0.687 和 0.830。
- 论文的假设性解释：相邻 token（x_{t-1} 和 x_t）之间的句法高度可预测（如冠词后大概率跟名词），使用固定相邻 token 的对比损失时，SAE 可能"取巧"地利用这种局部可预测性而非真正学习句法特征。随机采样打破了短程可预测性，迫使 SAE 真正编码句法信息。

### 5.5 PCA 降维与 Superposition 假说验证（Appendix Figure 3）

通过对表示进行 PCA 降维后训练探针，论文分析了语义、句法、上下文信息在表示各主成分中的分布密度：

- **语义任务**：基础语言模型的探针准确率随 PCA 维度增加而快速上升（从极低值迅速攀升），而 SAE 的曲线上上升更为平缓。这验证了 **Superposition 假说**：语义信号在基础模型的隐空间中广泛分布在大量神经元上（需要较多主成分才能恢复全部性能），而 SAE 成功将语义信息压缩到更少、更集中的特征中。

- **上下文任务**：呈现与语义任务相似的趋势——基础模型需要更多主成分来恢复全部表现。

- **句法任务**：Temporal SAE 的句法探针性能紧随基础模型和基线 SAE，但其准确率随维度增加的攀升趋势比基线 SAE 更陡峭。这表明相对于专门只学句法的基线 SAE，Temporal SAE 对句法的表示被分布到了更多的特征维度上（更 distributed），而非像基线 SAE 那样极度集中。

---

## 六、主要贡献与局限性

### 主要贡献

1. **为语言序列结构建立数据生成过程模型（DGP）**：将语言模型的隐变量分解为时间不变的高层语义特征（h_t）和时间变化的低层句法特征（l_t），并以加法方式组合，为后续方法设计提供了理论框架。

2. **提出 Temporal SAE 方法**：设计了一种新颖的 SAE 训练范式，引入双向 InfoNCE 时序对比损失，首次在稀疏自编码器训练中显式建模语言的时序结构。该方法与现有 SAE 架构兼容（可作为插件式改进），且计算开销可控。

3. **系统的实证验证**：在 Pythia-160M 和 Gemma2-2b 两个不同规模的语言模型、FineFineWeb 和 Wikipedia 两个数据集上进行了全面的定量和定性评估，证明 Temporal SAE 在保持竞争性重构性能的前提下，显著提升了对语义和上下文特征的提取和保留能力，同时保持了可接受的句法特征质量。

### 局限性

1. **特征划分比例需手动设定**：高层/低层特征的比例（Pythia 20%/80%，Gemma 10%/90%）是基于经验的启发式选择，论文未提供自动化或基于理论指导的确定方法，也未分析该比例对结果敏感性的影响。

2. **超参数 alpha 和 lambda 未公开**：论文未报告对比损失权重 alpha 和温度系数 lambda 的具体取值，也未进行相应消融实验，降低了结果的可复现性。

3. **对比损失仅约束高层特征**：低层特征仅通过"重构残差"的隐式机制实现与高层特征的分离，缺乏显式的正则化约束来保证两类特征的解耦质量。如果高层与低层特征未能有效解耦，对比损失的效益可能受限。

4. **仅在 Residual Stream 的单一层上验证**：实验仅限于 Pythia 第 8 层和 Gemma 第 18 层的 residual stream，未探索其他层或其他激活位置（如 MLP 输出、Attention 输出）的效果，也未分析不同层之间的交互。

5. **缺乏特征级可解释性案例分析**：论文主要依赖定量探针和 TSNE 可视化来评估，未展示高层特征字典中具体方向对应的语义概念实例（如某个高层特征方向是否对应 "medical"、"mathematics" 等可解释语义），也未进行人工可解释性评估。

6. **模型规模有限**：仅在 160M 和 2B 参数规模的模型上实验，未在更大规模模型（如 7B、13B、70B 等）上验证方法的可扩展性和适用性。

---

## 七、对后续研究的启发与潜在改进方向

### 启发

1. **模态特定先验对字典学习的关键价值**：该工作展示了一个重要范式——为不同数据模态（语言、视觉、蛋白质序列等）的字典学习方法引入该模态的结构化先验，可以在基本不牺牲重构质量的前提下大幅提升特征的可解释性和语义保真度。这为其他模态定制化 SAE 设计提供了方法论参考（如视频中的时空维度、图像中的空间层级结构）。

2. **序列内自然配对作为自监督信号**：对比损失利用同一序列内部 token 间的天然配对关系作为"免费"的自监督信号，无需任何外部标注。这种思路可推广到其他序列数据（代码、基因序列、音频、用户行为轨迹等）的无监督概念发现。

3. **认知科学与可解释性研究的学科交叉**：论文借鉴认知科学中关于语义长程 / 句法短程的经典洞见来设计机器学习方法，提示了跨学科知识迁移在可解释性研究中的巨大潜力。

### 潜在改进方向

1. **自适应特征划分机制**：基于信息论准则（如 mutual information）或变分推断，自动学习高层与低层特征的最优维度分配，避免依赖手工比例设定。可考虑引入可学习的门控机制（gating mechanism）动态确定每个特征的归属。

2. **显式的高/低层特征解耦正则化**：加入正交性约束（orthogonality constraint）或互信息最小化项（mutual information minimization），强制高层与低层特征空间的分离，进一步提升解耦质量。

3. **更丰富的语言结构先验**：除了时序平滑性，还可利用语法解析树（parse tree）、依存关系（dependency parsing）、篇章结构（discourse structure）等更丰富的语言结构信息来指导特征学习，实现更精细的语义/句法/语用多层次解耦。

4. **多层级与跨层分析**：将 Temporal SAE 扩展到语言模型的多个连续层，研究高层语义特征和低层句法特征在模型不同深度层级之间的演化、转换和交互规律。

5. **扩展到其他序列模态**：将方法适配到 Vision Transformer（patch 序列天然具有空间顺序，以及视频的时空顺序）、蛋白质语言模型、语音模型等，验证时序先验在跨模态场景下的通用性。

6. **更大规模验证与下游应用评估**：在更大规模模型（7B+）上进行扩展性实验，并探索 Temporal SAE 学到的语义特征在模型操控（steering）、安全性审计、偏见检测与缓解、知识编辑等下游任务中的实际效果与价值。

7. **长程依赖的增强建模**：当前随机采样的变体已初步探索了长程序列一致性，可进一步设计能显式建模远程依赖（如使用注意力机制或递归结构在特征层面建模序列依赖）的 SAE 架构变体。
