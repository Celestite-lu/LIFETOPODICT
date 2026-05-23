# RESTYLING UNSUPERVISED CONCEPT BASED INTERPRETABLE NETWORKS WITH GENERATIVE MODELS -- 论文详细总结

## 一、基本信息

- **标题**：RESTYLING UNSUPERVISED CONCEPT BASED INTERPRETABLE NETWORKS WITH GENERATIVE MODELS
- **作者**：Jayneel Parekh (ISIR, Sorbonne Universite; LTCI, Telecom Paris), Quentin Bouniot (LTCI, Telecom Paris; TU Munich; Helmholtz Munich; MCML), Pavlo Mozharovskyi (LTCI, Telecom Paris), Alasdair Newson (ISIR, Sorbonne Universite), Florence dAlche-Buc (LTCI, Telecom Paris)
- **发表年份**：2025
- **发表会议/期刊**：论文采用标准学术会议格式（含附录共约12页），但正文中未明确注明所发表的会议或期刊名称，推测为预印本或投稿状态。项目主页：https://jayneelparekh.github.io/VisCoIN_project_page/
- **缩写**：VisCoIN（Visualizable CoIN）

## 二、研究动机与问题定义

### 2.1 背景：基于概念的可解释网络（CoIN）

基于概念的可解释网络（Concept-based Interpretable Networks, CoINs）是内在可解释预测模型（by-design interpretable models）的一个子类，其核心思想是学习一个高层概念字典（dictionary of high-level concepts）用于预测。无监督CoIN通过精心设计的损失函数来约束概念学习，而不依赖人工标注的概念标签。其输出可解释性体现在：用户可以查看每个概念的激活值以及它们如何组合产生最终预测。

### 2.2 核心痛点：无监督CoIN的概念可视化存在严重局限

在无监督CoIN中，学到的概念需要通过可视化来理解，现有两类方法都存在严重缺陷：

1. **最大化激活样本（MAS）可视化**：从数据集中找出对某概念激活最高的自然图像。缺点是缺乏粒度（granularity），无法精确突出该概念具体编码了输入中的哪个特征。
2. **激活最大化（Activation Maximization）可视化**（如FLINT采用的输入空间优化）：通过优化输入图像使概念激活最大化。缺点是在大尺度图像上生成的图像往往充斥着重复模式（repeated patterns），不够自然，用户难以从中提取人类可理解的语义信号。例如论文Fig.1所示，从FLINT的可视化中很难识别出概念对应的是Yellow-colored head（黄色头部）。

此外，之前的CoIN系统**未能将可视化过程纳入对概念质量的定量评估**中。

### 2.3 本文目标

在大尺度图像上解决无监督CoIN概念可视化的上述局限性，具体做法是：
- 新增 **viewability（可视图性）** 属性：要求系统能够从学到的概念中重建高质量图像
- 通过预训练生成模型将概念特征映射到生成模型的隐空间，实现高质量可视化
- 定义一套新的定量评估指标，将对概念可视化的评价纳入评估体系


## 三、方法/框架（重点）

### 3.1 标准CoIN系统概述

给定训练集 S = {(x_i, y_i)}_{i=1}^{N}，分类任务有C个类别。一个标准CoIN系统将预测g(x)分解为两部分：
- **概念提取函数Phi**：将输入x映射为概念激活Phi(x)
- **预测网络Theta**：基于概念激活做出预测，即 g(x) = Theta∘Phi(x)

无监督CoIN学到的Phi通常需要满足以下三个属性：
1. **Fidelity to output（输出保真度）**：Phi(x)需能通过Theta预测真实标签或预训练分类器的输出
2. **Fidelity to input（输入保真度）**：Phi(x)需能通过解码器重建输入x，以使概念编码语义上有意义的输入特征
3. **Sparsity of activations（激活稀疏性）**：对于任何输入，仅少量概念被同时激活，增强可解释性

### 3.2 VisCoIN整体架构

VisCoIN在标准CoIN的基础上增加了一个**概念翻译器Omega（concept translator）**，将概念特征Phi(x)映射到预训练生成模型G的隐空间，整体架构如下（参见论文Fig.2右图）：

- **固定/预训练组件**：
  - 预训练分类器f（ResNet50，提供来源表示和监督信号）
  - 预训练生成模型G（StyleGAN2-ADA，用于高质量图像生成）

- **可训练组件**（紫色模块）：
  - **Psi**：在f的选定隐藏层之上学习的轻量网络，输出概念激活Phi(x)
  - **Theta**：预测网络，对Phi(x)做池化后通过线性层+softmax得到最终预测g(x)
  - **Omega**：概念翻译器，将Phi(x)映射到G的隐空间

### 3.3 各组件详细设计

#### 3.3.1 概念提取函数Phi与Psi

概念字典Phi包含K个概念函数phi_1, ..., phi_K。对于输入x，每个概念激活phi_k(x)是一个非负的小型卷积特征图（3x3），即Phi(x) in R_+^{K x b}，其中b = 9（3x3特征图的元素总数）。

Psi的网络结构（以ResNet50作为f时）：
- 从ResNet50的**三个隐藏层**提取特征（比FLINT多一个）：
  - Block 2输出：512x32x32
  - Block 3输出：1024x16x16
  - Block 4的倒数第二个bottleneck层输出：2048x8x8
- 每个层的输出通过一个卷积层投影到统一形状512x8x8
- 拼接所有特征图后，经过两个卷积层和一个池化层，最终输出形状为Kx3x3

#### 3.3.2 预测网络Theta

Theta的设计非常简洁：
g(x) = Theta(Phi(x)) = softmax(Theta_W^T . pool(Phi(x)))
其中Theta_W in R^{K x C}为线性层权重，pool为对每个特征图的空间池化操作（将Kxb压缩为K维向量）。

这种简化设计使得估计每个概念phi_k对于任意预测的重要性非常直接：概念k对预测类别c的贡献即为pool(phi_k(x)) . Theta_W^{k, c}。

#### 3.3.3 概念翻译器Omega与重建流程

Omega的设计取决于底层生成模型的架构：

**(a) 通用情况（ProgressiveGAN、beta-VAE）**：
Omega为一个全连接（FC）层，接受Phi(x)作为输入，预测隐向量w_x：
x_tilde = G(w_x)，其中 w_x = Omega(Phi(x))

**(b) StyleGAN2-ADA情况（主要实验配置）**：
由于StyleGAN使用扩展隐空间W+（256x256分辨率对应14个不同分辨率的隐向量），且Phi(x)维度远小于W+，作者引入了一个**支持表示Phi_prime(x)（supporting representation）**：

- Phi_prime(x)由Psi的第二个并行分支输出（两个全连接层）辅助Omega将输入嵌入W+
- Omega由一组独立的FC层组成，每个FC层预测W+中的一个隐向量
- 隐向量分配策略（基于Katzir et al., 2022的发现）：
  - W+的14个隐向量按分辨率分为：粗粒度4个、中粒度4个、细粒度6个
  - 前3个（最粗粒度）和最后2个（最细粒度）隐向量由Phi_prime(x)预测
  - 其余9个隐向量由Phi(x)预测
- 重建公式：x_tilde = G(w_x^+), w_x^+ = Omega(Phi(x), Phi_prime(x)) in W+

这种设计使与分类相关的语义特征（由Phi(x)控制）对应到中粒度和细粒度（除去最细）的隐向量，而Phi_prime(x)负责粗粒度和最细粒度特征。

### 3.4 训练损失函数

总训练损失由三部分组成：

#### (1) 输出保真度损失 L_of

L_of(x; Psi, Theta) = alpha . CE(g(x), f(x))

- CE为广义交叉熵（generalized cross entropy），衡量g(x)与预训练分类器f(x)输出分布之间的差异
- 使用f(x)作为软标签而非真实标签，便于同时对真实样本和来自G的合成样本进行训练
- alpha = 0.5（所有数据集固定）

#### (2) 重建损失 L_rec^G（最关键的部分）

L_rec^G(x; Psi, Omega) = ||x_tilde - x||_2^2 + ||x_tilde - x||_1 + beta . LPIPS(x_tilde, x) + gamma . CE(f(x_tilde), f(x))

- **L2和L1惩罚**：逐像素重建损失，保证fidelity to input
- **LPIPS感知损失**（权重beta = 3.0）：基于深度特征的感知相似度，是实现viewability的核心损失项
- **重建分类损失**（权重gamma）：鼓励生成模型重建出与输入x在分类器f视角下具有相同判别特征的图像。CE(f(x_tilde), f(x))要求f对重建图像的分类输出与对原始图像的分类输出一致

#### (3) 正则化损失 L_reg

L_reg(x; Psi, Omega) = L_reg-Psi(x; Psi) + L_reg-Omega(x; Omega)

其中：
- **L_reg-Psi** = delta . ||Phi(x)||_1 + L_orth(Psi)
  - L1损失（权重delta）：鼓励概念激活稀疏性
  - **核正交损失L_orth**：作用在Psi最后一层卷积层的权重上，鼓励L2归一化后的权重列向量两两正交（Xie et al., 2017; Wang et al., 2020），以减少概念冗余
- **L_reg-Omega** = ||w_x - w_bar||_2^2
  - 鼓励Omega预测的隐向量接近G的平均隐向量w_bar（生成模型逆映射系统的常见做法，Tov et al., 2021）

#### 最终优化目标

训练时同时优化Psi、Theta、Omega的参数，f和G保持固定：

min_{Psi,Theta,Omega} (1/N) Sum_{x in S} [L_of + L_rec^G + L_reg]

超参数汇总（Table 3）：

| 参数 | 含义 | CelebA-HQ | CUB-200 | Stanford Cars |
|------|------|-----------|---------|---------------|
| K | 概念字典大小 | 64 | 256 | 256 |
| alpha | 输出保真度权重 | 0.5 | 0.5 | 0.5 |
| beta | LPIPS权重 | 3.0 | 3.0 | 3.0 |
| gamma | 重建分类损失权重 | 0.2 | 0.1 | 0.05 |
| delta | 稀疏性权重 | 2 | 0.2 | 0.2 |

### 3.5 解释阶段（Interpretation Phase）

解释过程分为两个步骤：

#### (1) 概念相关性估计（Concept Relevance Estimation）

- **局部相关性r_k(x)**：概念phi_k对样本x的预测类别c_hat的对数几率（logit）贡献，归一化到[-1, 1]：
  r_k(x) = alpha_k(x) / max_l |alpha_l(x)|，其中 alpha_k(x) = pool(phi_k(x)) . Theta_W^{k, c_hat}
- **全局相关性r_{k,c}**：概念phi_k对类别c的平均局部相关性：
  r_{k,c} = E(r_k(x) | g(x) = c)

注意：此步骤不涉及解码器/生成模型G，仅依赖Theta的线性结构。

#### (2) 概念可视化（Concept Visualization）

核心创新：直接修改概念激活值并通过G可视化修改效果。

- 对于输入x和相关概念phi_k，按因子lambda（lambda >= 0）修改其激活值：phi_k(x) -> lambda x phi_k(x)，保持Phi(x)中其他概念的激活不变
- 观察lambda从1（原始重建x_tilde）到更大值（如4）时生成输出的变化
- lambda = 0（移除概念激活）对应删除该概念的语义信息
- 定性实验发现lambda在3到4以内保持可靠且一致的修改效果；极端大的lambda会推动隐向量远离w_bar，生成不可靠的输出

这种互动的、分层的可视化方式使用户能够直观理解每个概念编码的语义信息。


## 四、实验设置

### 4.1 数据集

论文在三个不同领域的大尺度图像识别任务上进行实验：

1. **CelebA-HQ**：二分类年龄分类（young/old），图像分辨率256x256
2. **Caltech-UCSD-Birds-200 (CUB-200)**：200类细粒度鸟类分类，图像分辨率256x256
3. **Stanford Cars**：196类细粒度汽车型号分类，图像分辨率256x256

### 4.2 主要Baseline

1. **FLINT**（Parekh et al., 2021）：基于输入重建的无监督CoIN，使用标准非生成式解码器
2. **FLAEM**（Sarkar et al., 2022）：另一个无监督CoIN系统
3. **Pretrained f (ResNet50)**：不可解释的预训练分类器，仅用于准确性基准
4. **Random baseline**（仅用于faithfulness评估）：随机选择概念将其激活设为0

注意：SENN（Alvarez-Melis & Jaakkola, 2018a）因需计算概念字典关于输入像素的雅可比矩阵，在大尺度图像上存在计算问题，未被纳入对比。

### 4.3 评估指标

#### 基本指标
- **预测准确率**：g在测试集上的分类准确率（%）

#### 重建质量指标
- **MSE**：平均逐像素均方误差
- **LPIPS**：感知图像块相似度距离（Zhang et al., 2018b），衡量感知质量
- **FID**：Frechet Inception Distance（Heusel et al., 2017），衡量重建图像与原始图像的分布距离

#### 新提出的可解释性指标

**(a) Faithfulness（忠实性）**：被识别为重要的概念是否真正编码了与预测相关的信息。

- 计算方式：对于测试样本x，将所有局部相关性r_k(x) > tau的概念激活设为0，生成Phi_rem(x)，计算FF_x = g(x_tilde)_c_hat - g(x_rem)_c_hat
- 理想情况下FF_x > 0（移除重要概念后预测概率下降）
- 报告1000个随机测试样本上FF_x的中位数，对不同阈值tau in {0.1, 0.2, 0.4}
- 注意：此处x_tilde = G(Omega(Phi(x)))，x_rem = G(Omega(Phi_rem(x)))

**(b) Consistency of Concept Visualization（概念可视化一致性）**：

- 动机：如果概念phi_k在不同图像上产生一致的语义修改，那么高phi_k激活和零phi_k激活生成的图像在f的嵌入空间中应该可以被区分
- 实现：
  - 选取phi_k最相关的N_cc = 100个样本，找到最大激活phi_k^{max}
  - 对每个样本生成两个版本的图像：x_k+（激活设为lambda . phi_k^{max}，lambda=2）和x_k-（激活设为0）
  - 构建二分类数据集S_k = {(x_k+, 1), (x_k-, 0)}，包含200个样本
  - 使用训练数据构建S_k^{train}训练线性SVM分类器varphi_k
  - 在测试数据构建的S_k^{test}上计算二分类准确率作为CC_k
- 报告所有概念CC_k的均值和标准差（%）

#### 附加评估（Appendix E）
- **Top-N概念激活过滤**：仅保留激活最强的N个概念后评估准确率
- **AUC-Faithfulness指标**：逐步增加概念后准确率曲线的AUC
- **Part Consistency**（CUB-200特有）：利用312种鸟部位标注评估概念修改对部位概率一致性
- **Sparsity评估**：每类全局相关性超过阈值的概念数量
- **Visualization FID**：FLINT激活最大化可视化与VisCoIN可视化的分布距离比较

### 4.4 实现细节

#### 预训练f（ResNet50）
- 初始化自ImageNet预训练权重
- 微调轮数：CelebA-HQ 10 epochs、CUB-200 30 epochs、Stanford Cars 90 epochs
- 优化器：CUB-200/Adam（lr=0.0001），Stanford Cars/SGD（起始lr=0.1，每30/60 epochs衰减0.1），CelebA-HQ/Adam（lr=0.001）
- 全部在256x256分辨率下处理

#### 预训练G（StyleGAN2-ADA）
- CelebA-HQ：使用NVIDIA提供的预训练checkpoint
- CUB-200：ImageNet预训练checkpoint微调，判别器训练2M真实图像（21小时）
- Stanford Cars：LSUN Cars预训练checkpoint微调，判别器训练1.8M真实图像（18.5小时）；水平翻转增强强度从默认1降至0.1
- CUB-200预训练G的FID约为9.4，Stanford Cars约为8.3
- 单V100-32GB GPU，batch_size=16

#### VisCoIN训练
- 优化器：Adam，学习率0.0001
- 迭代次数：CelebA-HQ 50K，CUB-200和Stanford Cars各100K
- Batch构成：8个真实训练样本 + 8个随机采样自G的合成样本
- 数据增强：随机裁剪 + 随机水平翻转
- 图像归一化到[-1, 1]，分辨率256x256
- 所有实验在单张V100-32GB GPU上完成

### 4.5 不同架构实验（Appendix D）

- **ProgressiveGAN作为G**（CelebA-HQ）：Acc 87.82%，FID 6.98，CC_k 93.8+/-6.8%
- **beta-VAE作为G**（FashionMNIST，K=25，beta=2，隐维度48，LeNet作为f）：Acc 88.06%
- **ResNet101作为f**（CUB-200）：Acc 79.44%，FID 11.24，CC_k 94.0+/-6.2%
- **ViT-B/16作为f**（CUB-200）：Acc f=86.66%，Acc g=85.86%，FF_x(tau=0.2)=0.081

## 五、核心结果与发现

### 5.1 预测准确率（Table 1a）

| 数据集 | Original-f | FLINT | FLAEM | VisCoIN |
|--------|-----------|-------|-------|---------|
| CelebA-HQ | 87.71 | 87.25 | **88.18** | 87.71 |
| CUB-200 | **80.56** | 77.20 | 51.76 | 79.44 |
| Stanford Cars | **82.28** | 75.95 | 50.02 | 79.89 |

- VisCoIN在CUB-200（79.44%）和Stanford Cars（79.89%）上显著优于FLINT和FLAEM，准确率接近不可解释的预训练f（80.56%、82.28%）
- 在CelebA-HQ上VisCoIN（87.71%）与f持平（87.71%），FLAEM略高（88.18%）
- FLAEM在复杂多类别任务上严重退化（CUB-200仅51.76%，Stanford Cars仅50.02%），显示了其在大规模细粒度分类上的不足

### 5.2 重建质量（Table 2a）

| 数据集 | 指标 | FLINT | FLAEM | VisCoIN |
|--------|------|-------|-------|---------|
| CelebA-HQ | MSE | **0.051** | 0.119 | 0.094 |
| | LPIPS | 0.533 | 0.688 | **0.405** |
| | FID | 30.45 | 39.73 | **8.55** |
| CUB-200 | MSE | **0.113** | 0.217 | 0.161 |
| | LPIPS | 0.712 | 0.75 | **0.545** |
| | FID | 53.16 | 51.15 | **15.85** |
| Stanford Cars | MSE | **0.121** | 0.278 | 0.179 |
| | LPIPS | 0.697 | 0.734 | **0.488** |
| | FID | 64.16 | 69.44 | **6.77** |

- **关键发现**：虽然FLINT在MSE（逐像素）指标上占优（因为其解码器直接优化逐像素重建），但VisCoIN在**感知相似度（LPIPS）和分布距离（FID）上大幅领先**。例如Stanford Cars上VisCoIN的FID为6.77，而FLINT为64.16，差距约10倍。
- 这表明预训练生成模型对实现viewability至关重要：标准解码器的逐像素重建指标虽好，但生成的图像在感知上远不如基于生成模型的重建。

### 5.3 Faithfulness（Table 2b）

| 数据集 | tau | Random | FLINT | FLAEM | VisCoIN |
|--------|-----|--------|-------|-------|---------|
| CelebA-HQ | 0.1 | 0.03 | 0.254 | 0.091 | **0.267** |
| | 0.2 | 0.018 | **0.201** | 0.151 | 0.171 |
| | 0.4 | 0.005 | 0.07 | **0.107** | 0.074 |
| CUB-200 | 0.1 | 0.034 | 0.004 | <0.001 | **0.251** |
| | 0.2 | 0.007 | 0.002 | <0.001 | **0.146** |
| | 0.4 | 0.001 | <0.001 | <0.001 | **0.044** |
| Stanford Cars | 0.1 | 0.035 | 0.001 | <0.001 | **0.161** |
| | 0.2 | 0.016 | <0.001 | <0.001 | **0.118** |
| | 0.4 | 0.002 | <0.001 | <0.001 | **0.034** |

- **关键发现**：
  - Random baseline的FF_x在任何阈值下都接近0，证明随机移除概念对预测基本无影响
  - 在复杂数据集（CUB-200、Stanford Cars）上，FLINT和FLAEM的忠实性非常差（接近0），甚至不优于Random baseline，说明其学到的相关概念实际上并未编码对分类真正重要的信息
  - VisCoIN在复杂数据集上忠实性远优于baseline：CUB-200上tau=0.1时达0.251，而FLINT仅0.004；Stanford Cars上tau=0.1时达0.161，而FLINT仅0.001
  - 在CelebA-HQ上三者表现竞争性但无明显差距

### 5.4 概念可视化一致性（Table 1b, lambda=2）

| 数据集 | FLINT | FLAEM | VisCoIN |
|--------|-------|-------|---------|
| CelebA-HQ | 82.6+/-22.7 | 57+/-17.3 | **85.5+/-13.9** |
| CUB-200 | 72.6+/-18 | 55.6+/-13.6 | **85+/-8.4** |
| Stanford Cars | 70+/-16.3 | 54.9+/-13.3 | **82.7+/-8.3** |

- VisCoIN在所有数据集上一致性均值最高，且标准差最低（一致性更稳定）
- FLAEM一致性最低（约55-57%），接近随机水平
- 将lambda从2增大到3时（Table 8, Appendix E），VisCoIN一致性进一步提升（如CUB-200从85+/-8.4到93.4+/-6.2），证实更大的概念激活修改会带来更强的语义一致性

### 5.5 可视化质量FID（Table 9, Appendix E）

| 数据集 | FLINT（激活最大化） | VisCoIN（lambda=4） |
|--------|-------------------|----------------|
| CelebA-HQ | 21.12 | **9.83** |
| CUB-200 | 26.55 | **11.71** |
| Stanford Cars | 45.72 | **8** |

- 评估了FLINT激活最大化生成的图像与VisCoIN生成图像（lambda=4）分别与原始数据分布之间的FID
- VisCoIN可视化在所有数据集上的FID显著低于FLINT（Stanford Cars上差距最大：8 vs 45.72），定量验证了VisCoIN生成的解释图像远比FLINT的激活最大化图像更接近自然图像分布
- 注意：FLINT可视化需要1000次反向传播迭代（计算昂贵），而VisCoIN仅需1次前向传播

### 5.6 消融实验与设计选择分析（Appendix F）

**(a) 重建分类损失权重gamma（Table 14, CUB-200）**：

| gamma | Faithfulness (tau=0.2) | LPIPS | FID |
|-------|----------------------|-------|-----|
| 0 | 0.001 | 0.52 | 13.11 |
| 0.1 | 0.146 | 0.545 | 15.85 |
| 0.2 | 0.236 | 0.607 | 8.84 |
| 0.5 | 0.24 | 0.634 | 11.54 |

- 揭示了**faithfulness与感知重建质量之间的tradeoff**：
  - 高gamma提升faithfulness（0.1到0.2：0.146到0.236），但LPIPS恶化（0.545到0.607）
  - 原因：更高的gamma促使模型生成被分类器f捕获的伪特征（spurious features），牺牲了输入质量
- 最终选择gamma=0.1（CUB-200）以优先保证足够的感知质量支持viewability

**(b) 支持表示Phi_prime（Table 15, CUB-200）**：

使用Phi_prime + K=256的LPIPS=0.545，优于不使用Phi_prime的K=512（LPIPS=0.568）和K=256（LPIPS=0.584）。结论：Phi_prime使得在不增加概念字典大小的情况下改善重建质量成为可能。

**(c) 概念数量K（Table 16, CUB-200）**：

K从64到512，准确率从78.91%提升至79.78%，LPIPS从0.624改善至0.537。K从256增至512收益递减（LPIPS仅从0.545到0.537），最终选择K=256。

**(d) 正交损失L_orth（Table 17, CUB-200）**：

使用L_orth带来轻微优势：LPIPS 0.545 vs 0.556，FID 15.85 vs 9.43。定性观察表明该损失有助于减少多个概念对同类捕捉相同特征的现象。

**(e) 概念翻译器Omega（Table 20, CUB-200）**：

不使用Omega（直接使用Phi(x)作为隐向量）导致重建质量下降（LPIPS 0.572 vs 0.545）。Omega的线性层允许模型为每个概念学习隐空间中的任意方向，而非强制将其与前K个标准基向量对齐。

**(f) 输出保真度损失权重alpha（Table 18, CUB-200）**：

alpha=0.1时准确率降至76.9%；alpha=2时准确率仅微增至79.63%但LPIPS从0.545恶化至0.586。最终选择alpha=0.5实现平衡。

**(g) 稀疏性权重delta（Table 19, CUB-200）**：

delta=20严重损害准确率（76%）和重建质量（LPIPS 0.629）。对CUB-200/Stanford Cars选择delta=0.2，CelebA-HQ选择delta=2。

**(h) 输出保真度损失vs交叉熵损失（Table 21, CUB-200）**：

使用交叉熵损失的VisCoIN在准确率（78.89% vs 79.44%）、LPIPS（0.559 vs 0.545）和faithfulness（0.076 vs 0.146）上均差于使用输出保真度损失的版本。输出保真度损失的一大优势是可以用于无标注的合成样本。

### 5.7 定性结果

- 论文Fig.4展示了CUB-200上的概念Red-eye（类别25 Bronzed-cowbird）、Blue upperparts（类别15）；CelebA-HQ上的Makeup（Young）、Eye squint（Old）；Stanford Cars上的Silver front（类别2）、Radiator grille（类别60）
- 在大多数情况下，增加lambda会强烈强调与概念名称对应的特定语义特征
- **局限性观察**：有时单个概念函数会同时修改多个高层特征（如Fig.4d中Eye squint同时增加眯眼和胡须特征）

## 六、主要贡献

1. **提出VisCoIN架构**：首个将概念翻译器Omega整合到无监督CoIN训练中的方法，通过将概念表示映射到预训练生成模型的隐空间，实现高质量概念可视化。

2. **引入Viewability新属性**：在CoIN训练中新增可视图性约束，通过LPIPS感知损失和重建分类损失强制系统从概念中重建高质量图像，使得通过生成输出看到输入样本成为可能。

3. **定义新颖的概念可视化流程**：通过修改单个概念激活值（乘以因子lambda）并观察生成输出的变化，实现交互式的、分层的概念解释。比之前的MAS和激活最大化方法更直观、更高效（1次前向传播 vs 1000次反向传播）。

4. **提出新评估指标**：在CoIN评估中首次引入faithfulness（忠实性）和consistency（一致性）两个功能基础（functionally-grounded）指标，更贴合用户在实际解释过程中体验的概念质量。

5. **多架构兼容性验证**：在StyleGAN2-ADA、ProgressiveGAN、beta-VAE等不同生成模型及ResNet50、ResNet101、ViT-B/16等不同骨干网络上验证了方法的通用性。

## 七、局限性

1. **概念语义纯度无法保证**：与其他无监督CoIN一样，VisCoIN不能保证学到的概念精确对应人类概念且不编码额外信息。虽然改进的可视化使识别偏差变得更容易，但无法从根本上解决此问题。

2. **受限于预训练生成模型G的质量**：如果G无法生成某特定特征（例如训练数据中稀缺的模式），则难以可视化编码该特征的概念。系统的上限由G的生成能力决定。

3. **线性隐空间遍历假设**：当Omega使用单FC层时，修改概念激活等价于在G的隐空间中沿线性轨迹遍历。最近研究（Song et al., 2023）表明线性轨迹不一定是隐空间遍历的最优选择。

4. **单概念可同时修改多特征**：定性观察发现，某些概念函数可能同时影响多个高层语义特征，降低解释的粒度精度。

5. **稀疏性不如FLINT**：由于优先优化重建/viewability而非使用熵基压缩损失，VisCoIN的概念稀疏性在同等阈值下低于FLINT（例如tau=0.5时VisCoIN平均每类6.1个概念 vs FLINT 3.4个）。

## 八、对后续研究的启发或潜在改进方向

1. **扩展到有监督CoIN和语言模型CBMs**：论文在Appendix A中指出，当前基于语言模型的CBMs（如Oikarinen et al., 2023; Panousis et al., 2024）虽有文字描述但缺乏可靠的可视化管道来验证概念检测是否与描述一致。将VisCoIN的viewability思想和可视化管道扩展到有监督场景，可有效识别概念泄露（concept leakage）等已知问题。

2. **结合扩散模型（Diffusion Models）**：论文在Appendix B中提出，当前扩散模型的隐空间理解（如语义隐方向发现）尚不成熟，难以设计Omega实现有意义的隐空间遍历。随着h-space和semantic latent directions研究的深入，将VisCoIN扩展到潜在扩散模型（如Stable Diffusion）是一个有价值的方向。

3. **改进隐空间遍历方式**：使用非线性或更复杂的隐空间遍历策略（而非简单的线性轨迹）可能使概念可视化效果更优。

4. **扩展到其他数据模态**：论文在结论中提到，将该框架适配到音频、文本、医疗影像等其他数据模态是未来的研究方向。例如结合多模态生成模型，可能实现跨模态的概念可视化。

5. **优化概念纯度与多特征解耦**：设计更精细的正则化或概念结构，使每个概念函数编码更单一的语义属性，减少一个概念修改多个特征的问题。

6. **人类评估研究**：设计严格的用户研究（human evaluation），比较文字描述与可视化在真实应用场景中对模型决策理解的帮助程度。这有助于量化可视化相对于纯语言描述的附加价值。

7. **提高稀疏性而不损失viewability**：在保持高质量重建的同时提升概念稀疏性，使可解释性更强。可能的途径包括调整稀疏性正则化策略或采用后处理的概念过滤方法。
