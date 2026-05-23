# 论文详细总结：Interpreting Large Text-to-Image Diffusion Models with Dictionary Learning

## 一、基本信息

- **标题**：Interpreting Large Text-to-Image Diffusion Models with Dictionary Learning（用字典学习解释大型文本到图像扩散模型）
- **作者**：Stepan Shabalin (Georgia Tech), Ayush Panda (Georgia Tech), Dmitrii Kharlapenko (ETH Zurich), Abdur Raheem Ali (Trajectory Labs), Yixiong Hao (Georgia Tech), Arthur Conmy（标注*，未注明隶属机构）
- **发表时间**：2025年
- **发表会议/期刊**：未明确标注（论文为预印本/workshop论文形式；其参考的ITDA方法[31]发表于 ICML 2025）
- **代码仓库**：SAE训练代码 github.com/neverix/fae；自动解释代码 github.com/kisate/flux-saes-gpu

---

## 二、研究动机与问题定义

### 2.1 背景与痛点

近年来，文本到图像（T2I）生成模型的能力迅速提升。这些模型发展出了对世界物理结构的内部表征（internal representations of the physical structure of the world），但以下问题仍存在争议：

1. **模型可解释性不足**：模型内部的特征组成（feature compositions）尚不清楚，难以理解模型如何生成特定图像。
2. **可控性受限**：如何精确控制模型输出是一个开放问题。
3. **训练数据记忆问题**：T2I模型可能在多大程度上记忆了训练数据（memorize training data）尚不明确。

### 2.2 研究目标

本文旨在通过**学习模型激活值的稀疏分解（sparse decompositions of model activations）**来探索大型文本到图像扩散模型的隐藏表征可解释性，具体评估该方法的**可行性和可扩展性（feasibility and scalability）**。核心思路是将在大语言模型（LLM）领域取得成功的**稀疏自编码器（Sparse Autoencoders, SAEs）**和新兴的**推理时激活分解方法（Inference-Time Decomposition of Activations, ITDA）**应用于大规模T2I扩散模型 FLUX.1（120亿参数，MMDiT架构），并评估其可解释性和操控（steering）能力。

---

## 三、方法与框架

### 3.1 目标模型：FLUX.1

FLUX.1 是基于 **Multimodal Diffusion Transformer (MMDiT)** 架构的最新T2I模型，参数量为 **120亿（12B）**。其架构特征：

- **19个 double blocks**（双流块，同时处理文本和图像的双模态信息）
- **38个 single blocks**（单流块）
- 使用 **AdaLN（Adaptive Layer Normalization）**层（Huang & Belongie, 2017），比标准 LayerNorm更复杂，具有时间步和prompt依赖性
- 基于 **Flow Matching / Rectified Flow** 训练范式（Lipman et al. 2023; Liu et al. 2022）
- 提供**步蒸馏变体 FLUX.1 Schnell**，可在单步内生成图像；另有 FLUX.1 Dev（非蒸馏版）

SAE/ITDA 训练在 double blocks 和 single blocks 的**残差流（residual stream）**上进行，这是Transformer架构中信息传递的核心路径。

### 3.2 SAE 公式化（TopK SAE）

本文采用 **TopK SAE**（Gao et al. 2024）的简化变体，**不使用 AuxK 辅助损失**。

**编码过程**（仿射变换 + TopK稀疏化）：
$$e = \text{Top}_k(W_{enc} (x - b_{post} + b_{pre}))$$

其中：
- $x \in \mathbb{R}^n$：输入激活向量（残差流中的一层激活值）
- $e \in \mathbb{R}^d$：稀疏潜在表示，满足 $L_0(e) \leq k$
- $k$：稀疏性常数（超参数），控制激活的最大特征数量
- $W_{enc} \in \mathbb{R}^{d \times n}$：编码器权重矩阵
- $b_{post}, b_{pre} \in \mathbb{R}^n$：偏置项

**解码过程**（仿射变换）：
$$y = W_{dec} e + b_{dec}$$

其中 $y \approx x$ 为重构的激活值，$W_{dec}$ 的每一行构成字典中的一个特征方向。

**损失函数**：纯 MSE 重构损失，无额外惩罚项（无 AuxK、无 L1正则）：
$$\text{MSE}(x, y) = \sum_{i=1}^{n} \frac{1}{n} (y_i - x_i)^2$$

**关键超参数**：
- $k$（稀疏度）：越高重构越好，但可能降低可解释性
- $d/n$（扩展比，expansion factor）：字典大小与输入维度的比值，控制模型容量
- SAE 学习的是过完备（overcomplete）字典基（$d \gg n$）

### 3.3 ITDA 公式化

**ITDA（Inference-Time Decomposition of Activations）**由 Leask et al. [31] 提出，是对 SAE 的重大改进——**完全移除可学习编码器**。

**核心思想**：用推理时优化代替学习编码器，直接从训练数据中选取解码器字典行。

**编码**（无编码器权重，使用梯度追踪）：
$$e = \text{ITO}(x, W_{dec})$$

其中 ITO（Inference-Time Optimization, Smith 2024）是**梯度追踪（Gradient Pursuit, Blumensath & Davies 2008）**的高效实现，可在无编码器的情况下，基于给定字典进行稀疏编码。

**字典构建算法**（贪心数据选择）：
$$W_{dec, t+1} = W_{dec, t} \cup \{ x \mid x \in \text{data} \land \text{MSE}(x, \text{ITO}(x, W_{dec, t})) > \text{threshold} \}$$

具体流程：
1. 从空字典或初始字典开始
2. 遍历训练数据，对每个数据点用当前字典进行 ITO 编码和重构
3. 如果某个数据点的重构 **MSE 超过阈值 threshold**，将该数据点（原始激活向量）直接加入字典作为新的一行
4. 重复直到字典大小达到目标或遍历完数据
5. **threshold** 是通过超参数搜索（sweeps）确定的

**ITDA 的核心优势**：
- **训练速度极快**：几分钟 vs SAE 需数小时/数天（数量级加速）
- **可解释性更直接**：每个潜在特征对应一个具体的训练数据点（token），可追溯到 SAE 训练数据集中的具体实例
- **字典构建无需梯度下降**：纯粹的贪心选择策略

**本文对 ITDA 的修改尝试**（Section 3.5，**均未显著改善**字典大小-重构精度权衡）：
1. 通过 Top-K 限制早期迭代的特征增长速度
2. 将残差重构误差（residual reconstruction error）而非原始数据点加入字典
3. 使用 K-Means 聚类对训练后的 ITDA 字典进行剪枝

**ITDA 训练损失**：使用 **FVU（Fraction of Variance Unexplained）**作为损失函数：
$$\text{FVU}(y, y^*) = \frac{\text{MSE}(y, y^*)}{\text{MSE}(y^*, \mu_{y^*})}$$

FVU 可以超过 1，典型值在 **0.1 到 0.5** 之间。该指标归一化了 MSE，使不同尺度的激活值具有可比性。

### 3.4 PCA 白化归一化（PCA Whitening Normalization）——关键创新

这是本文**最重要的训练方法改进**，直接解决了一项关键训练障碍。

**问题诊断**（Section 4.2）：
- FLUX.1 残差流具有极高的**各向异性（anisotropy）**——大部分方差（variance）集中在少数几个 PCA 特征值/维度上
- 标准训练（无归一化）产生巨量死亡特征：**> 99% 的潜在特征从不激活（dead latents）**
- 虽然这些 SAE 仍能解释 > 60% 的方差（variance explained），但所有活跃特征集中在低维子空间中，与训练数据的方差分布一致
- 无论是否使用 AuxK 损失，问题同样严重

**解决方案——PCA 白化三步流程**：

**第 1 步 - PCA 投影**：基于训练数据的前几个 batch，计算 PCA 分解的正交投影矩阵 $W \in O(n)$：
$$\vec{x_1} = W \vec{x_0}$$
变换后各分量具有零协方差，但方差各向异性（anisotropic variance），均值可能很大。

**第 2 步 - 标准化（去均值、除标准差）**：
$$\vec{x_2} = \vec{x_1} \odot \frac{1}{\vec{\sigma}(x_1)}$$
$$\vec{x_3} = \vec{x_2} - \vec{\mu}(x_2)$$
其中 $\odot$ 表示逐元素乘法，$\vec{\sigma}(x_1)$ 和 $\vec{\mu}(x_2)$ 分别为标准差和均值向量。

**第 3 步 - 训练与推理**：
- **训练**：在 $\vec{x_3}$（白化空间）中训练 SAE
- **推理重构**：将 SAE 输出从白化空间变换回原始残差流空间：
  $$y^* = W^T (y \odot \vec{\sigma}(x_1) + \vec{\mu}(x_2))$$

**效果**：
- 死亡特征比例从 **> 99% 骤降至 < 30%**
- 仅使用均值/标准差归一化（不做 PCA 旋转）效果类似，但本文所有 SAE 实验均采用完整 PCA 变换
- 注意：**ITDA 训练也进行归一化，但不使用 PCA**，仅做简单的均值/标准差归一化

### 3.5 训练配置详情

| 配置项 | 详情 |
|--------|------|
| **模型** | FLUX.1 Schnell（步蒸馏版，单步生成） |
| **生成分辨率** | 256x256（受计算资源限制） |
| **主要训练层** | **Double blocks 第 18 层**（位于模型参数量的中间位置，共19个double blocks） |
| **其他训练层** | Double blocks 第 9 层、Single blocks 第 9 层、Single blocks 第 18 层 |
| **训练数据量** | 30k 训练步 / 3000万 tokens |
| **时间步** | 单步生成（纯噪声输入，对应 Schnell 的单步特性） |
| **硬件平台** | **Google TPU v4-8**（共128GB HBM） |
| **深度学习框架** | **Jax**（论文声称首个 FLUX.1 的 Jax 实现） |
| **模型加载** | 扩散Transformer + T5文本编码器共占48GB（bfloat16精度） |
| **量化策略** | **4-bit NormalFloat（NF4）量化**（Dettmers et al. 2023 QLoRA）+ 自定义推理内核，避免在内存中物化（materialize）去量化权重矩阵 |
| **模型并行** | **FSDP（Fully Sharded Data Parallel）**沿输出轴（output axis）进行权重分区 |
| **激活值收集** | 使用 **Oryx** 和 **jax.lax.cond clobber** 技术跨多个时间步收集激活值（参考 Kramar 2024）；激活值先发送到CPU内存，吞吐量未受影响 |
| **解码器加速** | 使用 **vmap 批处理**实现；未编写TPU稀疏矩阵乘法内核，但设计了批量实现以减少HBM占用；参考 Gao et al. [17] 的 TopK SAE 解码器加速方法 |

---

## 四、实验设置

### 4.1 评估指标

**1. 重构质量（Reconstruction Quality）**：

- **VE（Variance Explained，方差解释率）**：多维度协方差加权 R-squared 指标：
  $$\text{VE}(y, y^*) = \sum_{i=0}^{n} \frac{1}{\sigma_u^2 \sigma_{,*}^2} \left( \frac{1}{|y|} \sum_{j=0}^{|y|} (y - \mu_y)_{j,i} (y^* - \mu_{y^*})_{j,i} \right)^2$$

- **FVU（Fraction of Variance Unexplained，未解释方差比例）**：主要对比指标，值越低表示重构越好；被用作 ITDA 训练的损失函数和 SAE vs ITDA 对比的核心指标：
  $$\text{FVU}(y, y^*) = \frac{\text{MSE}(y, y^*)}{\text{MSE}(y^*, \mu_{y^*})}$$

**2. 自动可解释性（Autointerpretation）**：

引入**视觉自动解释流水线（Visual Autointerpretation Pipeline）**，由两个组件构成：

- **解释器（Explainer）**：接收一组图像（激活区域标注为蓝色），对每张图先描述哪些区域被高亮，然后生成一个共享的共同模式/特征描述（shared explanation）。Prompt 见论文 Appendix D。

- **评判器（Classifier / Scorer）**：接收单张图像和一个特征描述（"The activation pattern is {pattern}"），判断该描述是否准确，输出 **0-1 分数**。要求非常苛刻（"Be very critical"），模糊或笼统的描述应得低分。

- **方法学借鉴**：采用 **Detection Scoring** 方法（Juang et al. 2024）——将回归任务（预测神经元在每个token上的激活强度）替换为分类任务（判断该特征标签是否正确），大幅减少 token 使用量。

- **后端 VLM**：`google/gemini-2.0-flash-001`

- **评估设置**：在 **Layer 18, k=64, d=64000** 上运行，仅考虑各方法中**实际激活的特征**（未激活/死亡特征不参与评估）。使用最大激活样本（max-activating examples）。

**3. Baselines**：

- **MLP 神经元**（类比 Huben et al. [21] 的自动解释方法）
- SAE 与 ITDA 之间的直接对比

**4. Steering（操控）评估**：

通过残差流上的激活向量加法进行定向操控，公式为：
$$y[a:b, c:d] \mathrel{+}= W_{dec}[f]$$

即在图像特定空间区域 $[a:b, c:d]$ 上，加上字典中特征 $f$ 对应的解码器向量 $W_{dec}[f]$。

- **主研究层**：Layer 18
- **操控步数**：仅在生成过程的前 **5 步（共 7 步）**施加操控，全步操控会引入明显伪影
- **评估维度**：正/负向操控效果、prompt 依赖程度

### 4.2 实验配置总结

| 参数 | 设定 |
|------|------|
| SAE 类型 | TopK SAE（无 AuxK） |
| SAE 归一化 | PCA 白化（训练空间） |
| ITDA 归一化 | 均值/标准差归一化（无 PCA） |
| ITDA 损失 | FVU（Fraction of Variance Unexplained） |
| 主要实验层 | Double blocks Layer 18 |
| 自动解释层 | Layer 18, k=64, d=64000 |
| 操控层 | Layer 18，前5/7步 |
| 分辨率 | 256x256 |
| 训练数据量 | 3000万 tokens |
| 硬件 | TPU v4-8 (128GB HBM) |

---

## 五、核心结果与发现

### 5.1 基础 FLUX 可解释性分析（Section 4）

**残差流范数分析（Section 4.1, Figure 3 & 4）**：
- Double blocks 的残差流范数整体随层数**平滑变化**（从 Layer 0 到 Layer 18），但文本分量（text component）在**最后一层（第19层，即最后一层double block）的范数急剧跳跃**
- 这可能对应一个**独特的推理阶段（distinct stage of inference）**，类似 Lad et al. [30] 对 LLM 推理阶段鲁棒性的发现
- 残差流范数大致随去噪时间步增加而增加，但变化幅度仅**一个数量级**以内
- Single blocks 呈现类似的范数变化模式

**潜在空间频谱分析（Section 4.2, Figure 5）**：

核心发现：**FLUX.1 残差流极度各向异性**——大部分方差集中在少数维度上。

各层第一 PCA 分量的解释方差比（Explained Variance Ratio）：

| Double Block 层 | 第一 PCA 分量解释方差比 |
|:---:|:---:|
| Layer 1 | **58.4%** |
| Layer 2 | 37% |
| Layer 9 | 30% |
| Layer 18 | **21%**（中间层最低点） |
| Layer 19 | **57.7%** |

观察：
- 首尾层方差高度集中（> 57%），中间层相对分散（最低21%）
- 方差最大的维度与基底对齐（basis-aligned），这与语言模型中的**异常维度（outlier dimensions, Kovaleva et al. 2021）**现象有关
- 推测这些高方差维度可能编码了**CLIP 图像嵌入、VAE 空间中的噪声编码或位置嵌入**
- 这种极端各向异性是导致未归一化 SAE 训练失败的根本原因（> 99% 死亡特征）

### 5.2 重构性能对比（Section 5.1）

**Layer 18 上的 SAE vs ITDA 主对比（Figure 6, FVU值越低越好）**：

| 稀疏度 k | ITDA (d=16.0k) | ITDA (d=24.0k) | ITDA (d=64.0k) | SAE (d=59.0k) |
|:--------:|:--------------:|:--------------:|:--------------:|:-------------:|
| 15 | **0.49** | 0.57 | 0.57 | 0.57 |
| 25 | **0.53** | 0.62 | 0.62 | 0.62 |
| 35 | **0.57** | 0.65 | 0.65 | 0.65 |
| 45 | **0.61** | 0.68 | 0.68 | 0.68 |
| 55 | **0.64** | 0.70 | 0.70 | 0.70 |
| 65 | **0.65** | 0.71 | 0.71 | 0.71 |

关键结论：
1. **ITDA 在重构性能上普遍优于 SAE**：即使在最小的 ITDA 字典配置 d=16.0k（约为 SAE d=59.0k 的 27%），FVU 也更低
2. 随着字典大小 d 和稀疏度 k 增大，ITDA 增益递减——d 从 16k 增大到 24k 有显著提升（k=15时 0.49 vs 0.57），但 24k 以上**完全饱和（saturate）**，k=65时所有 d>=24k 的 ITDA 均为 0.71
3. SAE 在 k=65 时达到 0.71，与最大字典配置的 ITDA 持平
4. ITDA 在 d=16k 时就能在 k=15 处达到 **0.49** 的 FVU，远优于 SAE d=59k 的 0.57

**跨层重构对比（Figure 8, Appendix A，d=64k, k=64）**：

| 层 | SAE FVU | ITDA FVU | 优势方 |
|:--:|:-------:|:--------:|:------:|
| 0 | **0.58** | 0.88 | SAE |
| 10 | **0.65** | 0.85 | SAE |
| 20 | **0.67** | 0.75 | SAE |
| 30 | 0.65 | 0.65 | 持平 |
| 35 | **0.50** | 0.60 | SAE |

关键结论：
- **SAE 在大多数层重构优于 ITDA**，尤其是在浅层（Layer 0：0.58 vs 0.88，差距达 0.30）和深层（Layer 35：0.50 vs 0.60）
- 仅在中间层（Layer 30）二者完全持平
- 这暗示 ITDA 在特定中间层（如 Layer 18 in Figure 6）有局部优势，而 SAE 在跨层泛化上更稳健

### 5.3 自动可解释性结果（Section 5.2）

**Autointerp 评分分布（Figure 7 直方图）**：

- **SAE 和 ITDA 的可解释性评分分布相近（comparable）**
- **MLP 神经元整体可解释性显著较低**，但其分布中存在若干**评分极高（exceedingly high autointerp scores）**的离群特征
- 从视觉检查来看，高分 MLP 神经元激活在图像的**连续区域**上，具有**固定语义含义**，可能对图像生成过程具有**因果相关性**（causally relevant）
- SAE 特征整体可解释性优于 MLP 神经元——这与 LLM 领域的发现一致（Huben et al. [21]）

**定性差异的重要发现（Section 7 结论部分，通过手动检查揭示）**：

尽管 Autointerp 分数相似，但 **ITDAs 和 SAEs 的特征类型存在本质区别**：

| 维度 | ITDA 特征 | SAE 特征 |
|------|----------|----------|
| 特征类型 | **通用属性**（颜色、纹理、区域） | **具体物体**（帽子、面孔等具体语义概念） |
| 像素覆盖 | 较稀疏的激活模式 | **更好的像素级覆盖**（pixel-wise coverage） |
| 语义级别 | 低层视觉属性 | 高层语义概念 |
| 可操控性 | 未明确展示 | 可通过激活加法进行定向操控 |

这一发现揭示了当前 **Autointerp 指标的根本局限**：自动评分无法区分"抽象视觉属性"与"可操控的语义概念"——两者在统计意义上有相似的检测分数，但在实用价值和因果意义上差异巨大。

### 5.4 Steering（操控）结果（Section 5.3）

**操控效果总结**：

1. **正向操控**：SAE 特征操控通常能产生与**最大激活模式（max activating patterns）**相关联的视觉变化效果

2. **条件受限性（关键发现）**：
   - 操控效果**高度依赖初始 prompt**：初始 prompt 必须与操控特征语义相关
   - 具体案例："动漫风格"（anime style）特征仅对 "A cartoon" prompt 有效，对 "A person" prompt 无效
   - 这暗示图像的特征空间具有**局部化（localized）**特性：后期层特征如果与早期层生成的图像表征不直接相关，操控效果很弱

3. **步数约束**：
   - **仅在前 5 步（共 7 步）施加操控**效果最佳
   - **全步操控（7/7步）会引入明显伪影（noticeable artifacts）**

4. **负向操控**：
   - 在某些情况下也能成功（subtractive steering）
   - 但**只能移除图像的细小细节**，效果远不如正向操控
   - 需要比正向操控**更复杂的 prompt** 才能生效

---

## 六、主要贡献

1. **首次将 SAE 扩展到大规模 T2I 扩散模型（FLUX.1，12B参数）**，验证了字典学习方法在 MMDiT 架构上的可行性和可扩展性。在 Jax + TPU 平台上实现了多项训练效率优化：
   - 首个 FLUX.1 的 Jax 实现
   - 4-bit NormalFloat 量化 + 自定义推理内核
   - FSDP 沿输出轴的模型并行
   - 基于 Oryx 和 jax.lax.cond clobber 的激活值收集
   - vmap 批处理解码器加速

2. **识别并解决了关键训练障碍**：发现 FLUX.1 残差流高度各向异性（前几个 PCA 分量集中大部分方差）导致 > 99% 死亡特征，提出**PCA 白化归一化**方案，将死亡特征比例降至 < 30%。此外验证了纯均值/标准差归一化（无PCA旋转）也有类似效果。

3. **首次在 T2I 扩散模型上评估和对比了 ITDA 与 SAE**：
   - ITDA 在 Layer 18 重构性能上优于同等/更大尺寸的 SAE（ITDA d=16k FVU 0.49 vs SAE d=59k FVU 0.57 @k=15）
   - ITDA 训练效率优势显著（分钟级 vs 小时/天级）
   - 但 SAE 特征捕捉具体物体，ITDA 特征捕捉通用属性——定性差异无法被 Autointerp 指标捕获

4. **引入了视觉自动解释流水线（Visual Autointerpretation Pipeline）**：基于 Gemini 2.0 Flash 的双组件系统（解释器 + 分类评判器），采用 Detection Scoring 方法减少 token 使用量

5. **展示了基于 SAE 特征的通用图像生成操控**：通过残差流激活加法实现定向操控，并系统分析了操控的条件约束特性（prompt依赖性、步数敏感性、局部vs全局操控差异）

---

## 七、局限性

1. **自动解释流水线的根本局限**：
   - Autointerp 分数无法区分抽象视觉属性（ITDAs特征）与可操控语义概念（SAEs特征）——两种截然不同的特征类型获得了相似的评分
   - 可能被简单图像特征（如颜色）误导，类似 Heap et al. [18] 发现的"随机初始化Transformer也能被SAE解释"的偏差
   - 仅使用了简单的分类评分（0/1判断），缺乏对特征**因果性、实用性、细粒度操控能力**的评估

2. **操控方法的约束**：
   - 条件受限：仅在与特征相关的初始 prompt 下有效，限制了实际应用场景
   - 空间局部：只能进行空间区域的局部操控（$y[a:b, c:d] \mathrel{+}= W_{dec}[f]$），非全局引导
   - 步数敏感：全步操控引入伪影，仅部分步数有效——需手动调优
   - 负向操控弱：只能移除细小细节，效果和适用范围有限

3. **实验范围受限**：
   - **仅 FLUX.1 Schnell**（步蒸馏版），未涉及 FLUX.1 Dev；无法分析蒸馏对特征的影响
   - **仅 256x256 分辨率**（受计算资源限制），未探究分辨率缩放效果
   - **仅纯噪声时间步**（single timestep, Schnell单步生成特性），未覆盖去噪全过程的中间时间步
   - **仅图像流**（image activations），未探索 MMDiT 文本流特征及双流间的跨模态特征共享
   - 训练数据量有限（3000万tokens / 30k步）

4. **SAE 架构缺乏图像特定适配**：
   - 直接套用 LLM 的 SAE 架构（全连接编码器），未加入**空间归纳偏置**
   - 图像中邻近空间位置的潜在激活比随机位置更相似（nostalgebraist, 2024），但编码器未利用这一先验（如使用卷积编码器）
   - 没有针对图像的空间位置或时间步做特征专门化设计

5. **特征利用深度不足**：
   - 仅用重构和 Autointerp 两个指标评估特征质量
   - 未探索 SAE/ITDA 特征在下游视觉任务中的应用（分类、分割、深度估计等），这相当于 LLM 领域的"稀疏探测"（sparse probing, Gao et al. 2024）
   - 缺乏对特征**因果重要性**的系统评估

---

## 八、对后续研究的启发与潜在改进方向

### 8.1 SAE 架构改进

- **空间归纳偏置（Spatial Inductive Bias）**：将 SAE 编码器改造为**卷积架构（convolutional encoder）**，利用图像中邻近空间位置激活更相似的特性——这是图像与文本序列的根本差异
- **多尺度架构**：针对不同空间位置/时间步设计专门的 SAE 结构或训练策略
- **跨模态字典学习**：利用 MMDiT 双流架构，使用 **Crosscoder（Lindsey et al. 2024）**技术同时学习文本流和图像流的共享特征及模态特有特征

### 8.2 操控能力改进

- **类微调的操控方法**：参考 Kwon et al. [28] 的语义潜在空间操控和 Zhang et al. [53] 的 ControlNet 方法，使 SAE 操控更接近微调的效果——更加稳健、全局化
- **操控效应感知**：在操控时预测和考虑对下游层的级联影响，避免引入伪影
- **全步/全局操控**：解决全步操控的伪影问题，实现更鲁棒的特征引导，甚至从纯特征引导生成图像（类比 Surkov et al. 2024 在 SDXL Turbo 上的工作）

### 8.3 评估体系完善

- **多维度评估框架**：区分特征的统计相关性、因果重要性、可操控性三个维度
- **下游视觉任务评测（Sparse Probing）**：用 SAE/ITDA 特征进行分类、分割、深度估计等任务，检验特征的语义表达能力——这是本文明确提出的未来方向之一
- **自动解释偏差系统性研究**：分析颜色、纹理等低级特征对 Autointerp 评分的干扰，建立更可靠的可解释性评估基准

### 8.4 模型分析扩展

- **FLUX.1 Dev vs Schnell 对比分析**：通过 Crosscoder 识别步蒸馏模型特有的特征（distillation-specific features）以及两变体间的对应特征对（corresponding feature pairs）——本文明确提出的未来方向之一
- **层间特征因果分析**：追踪特征从早期层到后期层的传递和演化，研究扩散模型内部的电路机制（circuits-based mechanistic interpretability, Cammarata et al. 2020）
- **时间步维度分析**：使用 **Activation Atlas（Carter et al. 2019）**等技术可视化中间时间步的特征演化
- **高方差维度的深入解释**：理解残差流中方差高度集中的维度具体编码了什么信息（CLIP嵌入/VAE噪声编码/位置嵌入的假设需要验证）

### 8.5 ITDA 方法优化

- **特征质量改进**：使 ITDA 特征从"通用属性"级提升到"具体物体"级，缩小与 SAE 在语义概念捕捉上的差距
- **归一化策略优化**：本文仅对 ITDA 做了简单归一化（无PCA），探索 PCA 白化或其他归一化方式对 ITDA 的影响
- **字典构建算法改进**：本文尝试的三种改进（Top-K限制、残差添加、K-Means剪枝）均未显著改善字典大小-重构精度权衡，但仍有组合策略或新策略的探索空间

---

## 关键公式速查

| 组件 | 公式 |
|------|------|
| SAE 编码 | $e = \text{Top}_k(W_{enc}(x - b_{post} + b_{pre}))$ |
| SAE 解码 | $y = W_{dec} e + b_{dec}$ |
| SAE 损失 | $\text{MSE}(x, y) = \sum_{i=1}^{n} \frac{1}{n}(y_i - x_i)^2$ |
| ITDA 编码 | $e = \text{ITO}(x, W_{dec})$ |
| ITDA 字典更新 | $W_{dec, t+1} = W_{dec, t} \cup \{x \mid \text{MSE}(x, \text{ITO}(x, W_{dec, t})) > \text{threshold}\}$ |
| FVU | $\text{FVU}(y, y^*) = \text{MSE}(y, y^*) / \text{MSE}(y^*, \mu_{y^*})$ |
| PCA 白化前向 | $\vec{x_3} = (\vec{x_1} \odot 1/\vec{\sigma}(x_1)) - \vec{\mu}(x_2)$，其中 $\vec{x_1} = W \vec{x_0}$ |
| 逆白化变换 | $y^* = W^T (y \odot \vec{\sigma}(x_1) + \vec{\mu}(x_2))$ |
| Steering 操控 | $y[a:b, c:d] \mathrel{+}= W_{dec}[f]$ |
| 方差解释率 (VE) | $\text{VE}(y, y^*) = \sum_{i=0}^{n} \frac{1}{\sigma_u^2 \sigma_{,*}^2} \left( \frac{1}{|y|} \sum_{j} (y - \mu_y)_{j,i} (y^* - \mu_{y^*})_{j,i} \right)^2$ |
