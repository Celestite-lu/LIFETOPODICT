# Concept-Guided Dictionary Learning for Interpretable Concept Extraction and Attribution in Large Vision-Language Models

## 一、基本信息

- **标题**：Concept-Guided Dictionary Learning for Interpretable Concept Extraction and Attribution in Large Vision-Language Models
- **作者**：Anonymous authors（双盲审稿中）
- **发表年份/会议**：投稿中（under double-blind review），具体会议/期刊未披露
- **代码**：https://anonymous.4open.science/r/xl-vlms-30C1

---

## 二、研究动机与问题定义

### 2.1 核心痛点

自回归大视觉-语言模型（Autoregressive LVLMs，如 Qwen2.5-VL、Gemma-3n）逐 token 生成文本，每一步的条件概率依赖于不断演化的多模态状态。这种机制带来了独特的可解释性挑战：

1. **难以判断预测是否基于真实视觉概念**：模型输出可能源于虚假相关性（spurious correlations）或幻觉（hallucination），而非视觉证据。
2. **现有概念发现方法无法用于自回归 LVLMs**：
   - TCAV、CRAFT、CLIP-Dissect、Holistic 等方法仅适用于 encoder-only 或对比学习模型，无法处理自回归架构（见 Table 1）。
   - CoX-LMM 虽然适配了 LVLMs，但存在严重局限：(i) 依赖标注 token（labeled concept tokens）；(ii) 仅支持单物体（single-object）场景；(iii) 全图提取引入背景噪声；(iv) 残差流中持续存在多义性（polysemanticity）。
3. **多义性（Polysemanticity）问题**：一个特征方向可能同时响应多个语义上无关的概念（例如同一特征 f3 同时激活 "chair" 和 "cat"），导致归因不可靠（见 Appendix A, Figure 4）。
4. **可扩展性不足**：现有方法无法从单物体设置扩展到多物体、大规模概念词汇表（如 1000 个概念）的场景。

### 2.2 问题定义

给定一个自回归 LVLM，将其分解为嵌入函数 g（视觉编码器 + 桥接 + 解码器注意力）和输出函数 h（投影 + softmax）。在倒数第二层 norm 层（language_model.norm）提取残差激活（residual activations）a_t^(l) in R^p（p 为残差维度），通过对每个响应的 token 序列进行平均池化得到概念级别的嵌入 s_m in R^p。目标是从这些激活中学习 K 个单义（monosemantic）概念向量 U = [u_1, ..., u_K] in R^{p x K}，使其能忠实地解释模型的预测行为。

---

## 三、方法/框架（CGDL）

### 3.1 整体流程概览

CGDL 是一个弱监督框架，包含四个主要阶段：

1. **概念候选发现**（Concept Candidate Discovery）：利用 LVLM 自身从数据集中自动识别文本概念。
2. **正负样本构建**（Positive/Negative Patch Construction）：使用 SAM 生成前景裁剪块，随机背景块作为负样本，构建概念袋（concept-example bags）。
3. **对比残差提取**（Contrastive Residual Extraction）：通过二元提示（binary prompt）获取概念对齐的残差激活。
4. **字典学习与概念向量识别**（Dictionary Learning and Concept Vector Identification）：对每个概念袋执行二基分解，分离概念信号与噪声。

### 3.2 概念候选发现（Section 4.1）

给定无标签图像集 T = {I_k}_{k=1}^N，LVLM f 使用结构化提示预测每张图像的候选概念：

- **提示模板**："Identify every visible object, item, concept, and pattern in the image. Output only a single-word, comma-separated list. No explanations or sentences."
- 全局概念词汇表通过并集构建：C = Union_{k=1}^N C(I_k)
- 对每个概念 c，收集支持图像集 Phi(c) = {I_k | c in C(I_k)}

### 3.3 正负样本构建（Section 4.1）

对每个概念 c，使用两种裁剪算子构建概念袋：

- P(I_k, c) in {randomcrop(I_k), sam(I_k, c)}
  - **SAM**：利用 Segment Anything Model (Kirillov et al., 2023) 对概念相关区域进行分割裁剪，提供弱定位。
  - **RandomCrop**：随机裁剪作为基线。
- 概念袋 B(c) = B+(c) Union B-(c)，包含正样本（含概念）和负样本（不含概念）的混合。
- **关键优势**：这种弱监督方式无需手动标注，自动扩展到多物体数据集。例如，"stripes" 概念可能来自斑马、老虎或猫的图像。

**实现细节**：
- ImageNet：保留最高频的 1000 个概念
- MSCOCO：保留最高频的 10 个概念
- 每个概念袋上限 1600 个裁剪块
- 图像缩放至 500 像素宽，裁剪为 200x200 窗口，0.2 重叠率，平均每张图像生成 5-6 个裁剪块

### 3.4 概念引导的字典学习（Section 4.2）——核心创新

这是 CGDL 最核心的技术贡献。与 CoX-LMM 使用开放式的 token 生成不同，CGDL 通过**二元提示（binary prompt）**约束生成范围，限制 token 生成以减少噪声和纠缠（entanglement）。



四种变体模板（P1-P4）被消融实验证明具有鲁棒性（Table 4）：
- **P1**: "Detect whether the image contains c_k. If yes, return c_k; otherwise, return UNK."
- **P2**: "Does the image contain c_k? If yes, output c_k; otherwise, output No-c_k."
- **P3**: "Is there a clear instance of c_k in this image? Reply with c_k or thing, nothing else."
- **P4**: "Recognize whether the concept c_k is present in the picture. Use only c_k or UNK as your answer."

**残差提取流程**：
1. 对每个概念袋 B(c_k) 中的图像裁剪 x^{c_k}，使用上述提示查询模型。
2. 提取倒数第二层 norm 层（language_model.norm）的残差激活：a_t^(l) = g^(l)(x^{c_k}, prompt_cg, y_hat_{<t}) in R^p
3. 对每个响应的 token 序列进行平均池化（支持多 token 概念如 "hot dog"，而非将其拆分为独立 token "hot" 和 "dog"）：s_m = (1/T_m) * sum_{t=1}^{T_m} a_t^(l) in R^p
4. 收集 M 个样本得到激活矩阵 S = [s_1, ..., s_M] in R^{M x p}

**二基字典分解（Two-Basis Decomposition）**：

对每个概念袋，字典学习将 S 分解为概念基和否定基：

S approx V U^T,  V in R^{M x 2},  U in R^{p x 2}

其中 U 的两个列向量分别对应 c_k（概念存在）的基和 No-c_k（概念不存在）的基，V 捕获每个样本在两个基上的激活权重。

**核心设计理由**：
- 每个概念袋限制为 2 个基（概念 vs. 否定），类比一对一分类（one-vs-all classification），捕获概念间关系。
- **计算复杂度**：每个概念袋的成本为 O(M_c * p + M_c^2)，其中 M_c 是 B(c_k) 的样本数。K 个概念总复杂度为 O(K(M_c * p + M_c^2))，与 Semi-NMF 的每次迭代复杂度一致（Kim and Park, 2008; Ding et al., 2010）。
- 对比完整 NMF（当 K >> 2 时呈立方复杂度依赖）和 SAE（需要反向传播百万参数），CGDL 的二基设计大幅提高了可扩展性。
- 避免了大词汇表中字典原子（atom）之间的干扰和坍塌（collapse），这是先前字典学习方法的常见问题。

### 3.5 正概念向量识别（Section 4.3）

从二基分解的 U 中识别哪个基对应正概念：

1. **最大激活裁剪（MAC, Maximum Activated Crops）**：对 U 中的每个特征，提取在 V 矩阵对应列中激活值最高的 top-alpha_MAC 个图像块：X_{k,MAC} = {i | v_i^(k) is among the top-alpha_MAC values of v^(k)}
2. **基选择**：选择与多数正样本输出对齐的基（即该基的高激活样本中，模型实际输出 c_k 的比例最高）：k* = arg max_{j in {0,1}} |{i in X_{k,MAC} | y_hat_i = c_k}|
3. 定义 u_{k*} in R^p 为概念向量，X_{k*,MAC} 为其支持裁剪块。

**文本接地（Textual Grounding）**：
- 利用 LVLM 的输出函数 h^(l)（投影层 + softmax），将概念向量直接映射到词汇空间：q_k = h^(l)(u_k) in R^{|V|}，其中 |V| 为模型的输出词汇表大小。
- 选择 top-tau（文中设定 tau=50）个最高分 token，去除停用词和噪声，得到文本接地集 X_{k,MAT}。
- 汇总所有概念得到视觉接地集 X_MAC = {X_{1,MAC}, ..., X_{K,MAC}} 和文本接地集 X_MAT = {X_{1,MAT}, ..., X_{K,MAT}}。
### 3.6 概念归因与多模态对齐（Section 4.4）

受 TCAV (Kim et al., 2018) 启发，将残差激活投影到学到的概念子空间 U_hat = [u_1, ..., u_K] in R^{p x K}：

- **Token 级别**：alpha_j^(t) = cos_sim(u_j, a_t^(l))，提供细粒度 token 归因。
- **短语/句子级别**：使用平均池化激活 s_m，计算 alpha_{m,j} = cos_sim(u_j, s_m)。
- **最激活概念**：k* = arg max_j alpha_{m,j}。
- 其视觉接地 X_{MAC}[k*] 和文本接地 X_{MAT}[k*] 共同提供多模态解释。
- **纯文本模式**：评估纯文本提示是否激活与多模态输入相同的特征方向，由此量化多模态对齐（multimodal alignment）程度。

### 3.7 与现有字典学习方法的统一视角（Section 3）

CGDL 将现有方法统一在字典学习的框架下。给定激活 S in R^{n x d}，求解：

```
arg min_{U in R^{d x K}, V in R^{n x K}} ||S - V U^T||_F^2
```

不同方法通过对 (U, V) 施加不同的约束来区分：

| 方法 | U 约束 | V 约束 |
|------|--------|--------|
| K-Means (ACE, Ghorbani et al., 2019) | 无 | v_i in {e_1, ..., e_K}（one-hot 编码） |
| PCA (Graziani et al., 2023) | U^T U = I（正交基） | 无 |
| NMF (CRAFT, Fel et al., 2023a;b) | S >= 0, U >= 0（非负） | V >= 0（非负） |
| Semi-NMF (SNMF, Trigeorgis et al., 2014; CoX-LMM) | U 自由（混合符号） | V >= 0（非负激活） |
| Sparse AE (SAE, Templeton, 2024; Pach et al., 2025) | 无 | V = psi(S), ||v_i||_0 <= s（稀疏编码） |
| **CGDL（本方法）** | **每概念包仅 2 基（K=2 per bag）** | **二基分解（concept vs. negation）** |

CGDL 强调：其关键区别不在于分解算法本身，而在于**数据的呈现方式**。前序工作（Grobrugge et al., 2025; Sun et al., 2023）指出，概念质量更多地取决于数据如何呈现给字典而非具体的分解算法。CGDL 通过二元提示分离概念信号与噪声，二基分解防止原子坍塌——这与 NMF + 预处理的简单组合有本质区别。

---

## 四、实验设置

### 4.1 模型与数据

**模型**（全部冻结以保持后验可解释性）：
- Qwen2-VL-7B (Wang et al., 2024)
- Qwen2.5-VL-7B (Team, 2025b)
- Gemma-3n-E4B (Team, 2025a) ——正文主要报告模型（Qwen 模型结果见附录 E）

**数据集**：

| 数据集 | 类别数 | 训练样本/类 | 验证样本/类 |
|--------|--------|------------|------------|
| ImageNet-1k | 1000 | 300 | 50 |
| MSCOCO (10) | 10 | 300 | 50 |
| CIFAR100（附录） | 100 | 300 | 100 |
| DTD（附录） | 47 | 95 | 24 |

**激活提取**：倒数第二层 norm 层（language_model.norm），维度 p = 2048。

### 4.2 Baselines

- **CoX-LMM (Parekh et al., 2024)**：分别在 SNMF 和 SAE 两种字典学习变体下比较，CoX-LMM 依赖 ImageNet 标签和 MSCOCO 标注 token。
- **SIMPLE**：简单基线，测试具有最高 l2-norm 的残差是否与概念对齐。
- **Random**：随机分配概念的基线。

### 4.3 评估指标

1. **概念发现质量（Discovery Quality，遵循 Fel et al., 2023a 协议）**：
   - **Sparsity（稀疏性，向上箭头表示越高越好）**：概念表示中非零元素的比例，越高表示概念越集中。
   - **Stability（稳定性，向下箭头表示越低越好）**：多次运行间概念向量的一致性。
   - **Overlap（重叠度，向下箭头表示越低越好）**：不同概念向量之间的余弦相似度重叠（越低单义性越强）。

2. **归因质量（Attribution Quality）**：
   - **CLIPScore**（Radford et al., 2021b）：跨模态对比对齐（图像-文本）。
   - **BERTScore**（Devlin et al., 2019）：上下文语义相似度（文本-文本）。

3. **忠实性（Faithfulness）**：
   - **C-Deletion（概念删除）**：逐步将 top-1、top-2、top-3 最有影响力的概念方向坐标置零，记录模型输出概率的下降。排名由梯度幅值（相对最高概率 token）决定。
   - **C-Insertion（概念插入）**：从零向量开始按相同排名逐步添加概念坐标，记录概率上升。

4. **可扩展性（Scalability）**：在 10 和 1000 个概念上评估。

### 4.4 实现细节

- **GPU**：单张 NVIDIA RTX 3090 (24GB)
- **CPU**：AMD Ryzen 9 5950X (16 cores)
- 固定随机种子
- SNMF 稀疏权重 alpha = 20（经消融实验选择，见 Table 9）
- 字典大小 K = 2（每概念包，见 Table 8 消融）
- 训练 10 个概念的端到端时间：CGDL 421s vs. CoX-LMM 4375s

---

## 五、核心结果与发现

### 5.1 概念向量质量（Table 2，Gemma-3n-E4B）

| 方法 | 字典学习 | ImageNet Spars.(向上) | ImageNet Stab.(向下) | ImageNet Overlap(向下) | MSCOCO Spars.(向上) | MSCOCO Stab.(向下) | MSCOCO Overlap(向下) |
|------|---------|----------------------|----------------------|------------------------|---------------------|---------------------|-----------------------|
| CoX-LMM | SNMF | 0.96 | 0.13 | 0.25 | 1.00 | 0.02 | 0.16 |
| CoX-LMM | SAE | 0.84 | 0.21 | 0.27 | 0.97 | 0.17 | 0.28 |
| CoX-LMM | SIMPLE | 0.07 | 0.79 | 0.68 | 0.63 | 0.91 | 0.84 |
| **CGDL** | **SNMF** | **1.00** | **0.02** | **0.08** | **1.00** | **0.00** | **0.06** |
| CGDL | SAE | 0.90 | 0.16 | 0.10 | 1.00 | 0.05 | 0.16 |
| CGDL | SIMPLE | 0.52 | 0.64 | 0.67 | 0.80 | 0.49 | 0.54 |

**关键发现**：
- CGDL-SNMF 在所有指标上取得最佳结果：稀疏性最高（1.00），稳定性最低（ImageNet 0.02 / MSCOCO 0.00），重叠度最低（ImageNet 0.08 / MSCOCO 0.06）。
- 相比 CoX-LMM SNMF：ImageNet 上稀疏性提升 4%（0.96 -> 1.00），稳定性提升 11%（0.13 -> 0.02），重叠度降低 17%（0.25 -> 0.08），与摘要声明的 "up to 4% higher sparsity, 11% greater stability, 17% lower overlap" 一致。
- SAE 同样从概念引导（Concept-Guidance）中获益，但提升幅度不如 SNMF。
- SIMPLE 基线始终薄弱（最低稀疏性、最高重叠度），表明仅靠高 l2-norm 残差无法有效对齐概念。

### 5.2 归因质量（Table 3，Gemma-3n-E4B）

**ImageNet（1000 个概念）**：

| 方法 | 模态 | CLIPScore | BERTScore |
|------|------|-----------|-----------|
| CGDL | Text-only | - | 0.78 +/- 0.07 |
| CGDL | Image-only | 0.62 +/- 0.08 | 0.84 +/- 0.08 |
| CGDL | Combined | **0.67 +/- 0.09** | **0.86 +/- 0.10** |
| CoX-LMM | Text-only | - | 0.74 +/- 0.08 |
| CoX-LMM | Image-only | 0.57 +/- 0.03 | 0.75 +/- 0.06 |
| CoX-LMM | Combined | 0.58 +/- 0.05 | 0.82 +/- 0.09 |
| Random | - | 0.52 +/- 0.04 | 0.71 +/- 0.06 |

**MSCOCO（10 个概念）**：

| 方法 | 模态 | CLIPScore | BERTScore |
|------|------|-----------|-----------|
| CGDL | Combined | **0.64 +/- 0.08** | **0.93 +/- 0.07** |
| CoX-LMM | Combined | 0.55 +/- 0.05 | 0.73 +/- 0.11 |

**关键发现**：
- CGDL 在所有三种模态（Text-only、Image-only、Combined）下均优于 CoX-LMM。
- ImageNet Combined：CGDL 的 CLIPScore 达 0.67（CoX-LMM 0.58，提升 +9%），BERTScore 达 0.86（CoX-LMM 0.82，提升 +4%），与摘要声明的 "up to +9% CLIPScore, +4% BERTScore"（在 1k 概念下）一致。
- MSCOCO Combined：CGDL BERTScore 达 0.93，远超 CoX-LMM 的 0.73（提升 +20%）；CLIPScore 从 0.55 提升至 0.64（+9%）。
- 优势同时在**小规模（10 概念）和大规模（1000 概念）**评估中保持。
- CGDL 尽管需要**更少的监督信号**（弱监督 vs. CoX-LMM 需要类别标签和标注 token），仍持续超越 CoX-LMM。
### 5.3 忠实性评估（Figure 1）

**C-Deletion（概念删除）**：
- CGDL 呈现更陡峭的概率下降曲线，Top-1 概念删除导致最剧烈的概率下降，Top-2 和 Top-3 依次递减，排名之间区分度清晰。
- CoX-LMM 的曲线相对平坦，各排名间的分离较弱。

**C-Insertion（概念插入）**：
- CGDL 呈现更强劲且有序的概率恢复（Top-1 > Top-2 > Top-3）。
- CoX-LMM 的恢复曲线区分度不足。

**结论**：CGDL 产生的概念排名更具**忠实性（faithfulness）和区分度（discriminativeness）**，证明学到的概念向量能可靠地代表模型预测中实际使用的特征。

### 5.4 Qwen 模型上的跨模型验证（附录 E, Table 6 和 Table 7）

**Qwen2-VL-7B（Table 6）**：CGDL 在全部四个数据集（ImageNet-1k, CIFAR100, DTD, MSCOCO）上均优于 CoX-LMM。
- MSCOCO Combined BERTScore：CGDL 0.94 +/- 0.08 vs. CoX-LMM 0.82 +/- 0.03（+12%）
- MSCOCO Combined CLIPScore：CGDL 0.62 +/- 0.06 vs. CoX-LMM 0.58 +/- 0.06（+4%）
- CIFAR100 Combined BERTScore：CGDL 0.91 +/- 0.08 vs. CoX-LMM 0.73 +/- 0.07（+18%）

**Qwen2.5-VL-7B（Table 7）**：CGDL 同样全面领先。
- MSCOCO Combined BERTScore：CGDL 0.95 +/- 0.08 vs. CoX-LMM 0.90 +/- 0.07（+5%）
- CIFAR100 Combined BERTScore：CGDL 0.92 +/- 0.07 vs. CoX-LMM 0.87 +/- 0.06（+5%）
- DTD Combined BERTScore：CGDL 0.89 +/- 0.08 vs. CoX-LMM 0.88 +/- 0.05（+1%）

**跨模型一致性**：CGDL 在三个不同架构、不同规模的 LVLM 上均表现出一致且显著的优势，证明了其模型无关（model-agnostic）特性。

### 5.5 定性分析

- **单义 vs. 多义概念（Figure 2）**：CoX-LMM 产生的概念向量存在严重多义性（如 "canine" 链接到 "hot dog"、"tiger" 混入 "lion" 多种动物），而 CGDL 发现细粒度、单义概念（如 "fur"、"dog"、"stripes"），每个概念向量只对应一个明确的语义。
- **多模态归因展示（Figure 3）**：三种归因设置均被验证有效：(a) 对齐的图像-文本概念触发强特征归因；(b) 仅图像输入（无概念文本）仍能触发相关特征；(c) 纯文本（无图像）激活语义有意义的特征方向。证明了 CGDL 鲁棒的多模态对齐能力。
- **多义 vs. 单义特征对比（Figure 4, 附录 A）**：直观图示多义特征 f3 同时激活 "chair" 和 "cat"，而 CGDL 鼓励每个特征仅对应一个概念，如 f1 -> "cat", f3 -> "table"。

### 5.6 概念向量的泛化性（附录 E.4, Figure 9）

即使目标物体的概念不在字典中（概念不匹配），LVLM 仍然将预测激活与最语义相关的可用概念对齐。例如，模型会将 "stripe-like patterns" 的视觉输入对齐到 "zebra" 概念向量。

### 5.7 抽象概念归因（附录 E.5, Figure 10）

CGDL 能捕获抽象概念与预测 token 之间的语义关系：
- "macaw"（金刚鹦鹉）对 "colored, colorful, rainbow" 概念显示高相似度。
- "ladybug"（瓢虫）和 "dalmatian dog"（斑点狗）对 "dotted, polka"（波点）概念显示高相似度。
- "jellyfish"（水母）对 "skin" 和 "soft" 概念显示高相似度。

### 5.8 Token 级别后验解释（附录 E.3）

CGDL 是首个能在自回归 LVLM 中提供 token 级别后验概念解释的方法。使用 2x2 图像网格故意提高任务难度，CGDL 可以逐 token 展示 top-2 激活概念及其视觉/文本接地，而 CoX-LMM 因多义性无法提供可靠的 token 级别映射。

---

## 六、消融实验

### 6.1 提示模板鲁棒性（Table 4）

在四种二元提示变体（P1-P4）上评估 CLIPScore（MSCOCO-10），三个模型得分仅有微小波动：
- Qwen-2：P1 0.57+/-0.10 至 P4 0.59+/-0.11（方差略高）
- Qwen2.5：P1 0.65+/-0.13 至 P4 0.63+/-0.08
- Gemma-3n：P1 0.62+/-0.08 至 P4 0.64+/-0.09

**结论**：方法对提示措辞具有鲁棒性，不需要精心调校提示模板。

### 6.2 字典大小 K 消融（Table 8）

在 Gemma-3n 上测试 K = {2, 10, 30, 50, 100}（5 个 ImageNet 类）：

| K | BERT@1 (向上) | CLIP@1 (向上) |
|---|--------------|--------------|
| 2 | 0.881 +/- 0.013 | 0.616 +/- 0.044 |
| 10 | 0.881 +/- 0.013 | 0.603 +/- 0.047 |
| 30 | 0.884 +/- 0.040 | 0.475 +/- 0.033 |
| 50 | 0.885 +/- 0.040 | 0.509 +/- 0.028 |
| 100 | 0.887 +/- 0.041 | 0.488 +/- 0.035 |

- BERTScore 基本恒定（约 0.88），不受 K 增大显著影响。
- CLIPScore 随 K 增大而显著下降（从 0.616 降至约 0.49），表明大字典损害视觉-文本对齐质量。
- **结论**：K=2（每概念袋二基分解）在归因质量与计算效率间取得最佳平衡。

### 6.3 SNMF 稀疏权重 alpha 消融（Table 9）

测试 alpha = {0, 20, 100, 150, 200}（Gemma-3n, 5 个 ImageNet 类）：

| alpha | BERT@1 (向上) | CLIP@1 (向上) |
|-------|--------------|--------------|
| 0 | 0.881 +/- 0.013 | 0.610 +/- 0.039 |
| 20 | 0.881 +/- 0.013 | 0.616 +/- 0.044 |
| 100 | 0.881 +/- 0.013 | 0.615 +/- 0.043 |
| 150 | 0.881 +/- 0.013 | 0.617 +/- 0.043 |
| 200 | 0.881 +/- 0.013 | 0.621 +/- 0.039 |

- BERTScore 在 alpha 变化时保持恒定。
- CLIPScore 随 alpha 增大而轻微提升。
- **选择 alpha=20**：效果稳定且避免过度稀疏化，所有主实验均使用此设置。

### 6.4 SAM vs. 随机裁剪定位消融（Table 10）

在 Gemma-3n、Qwen-2.5、Qwen-2.0 三个模型上对比（5 个 ImageNet 类）：

| 模型 | SAM BERTScore@1 | SAM CLIPScore@1 | Random BERTScore@1 | Random CLIPScore@1 |
|------|----------------|-----------------|---------------------|---------------------|
| Gemma-3n | **0.93 +/- 0.04** | **0.61 +/- 0.02** | 0.88 +/- 0.01 | 0.59 +/- 0.04 |
| Qwen-2.5 | **0.92 +/- 0.03** | **0.67 +/- 0.03** | 0.91 +/- 0.06 | 0.66 +/- 0.03 |
| Qwen-2.0 | **0.93 +/- 0.04** | **0.63 +/- 0.04** | 0.91 +/- 0.06 | 0.61 +/- 0.04 |

**结论**：SAM 在 BERTScore 和 CLIPScore 上均优于随机裁剪，因其分割更好地定位原始图像中的概念区域。两者在归因排名上表现相似，SAM 的主要优势在于提供更干净、更视觉连贯的概念样本可视化（qualitative interpretability）。
### 6.5 层消融（Figure 12）

在 Gemma-3n 的不同 norm 层（0-35）提取概念向量并评估：
- **CLIPScore**：随层深增加而整体提升（浅层约 0.50，最深第 35 层达 0.67），与先前工作一致，因为深层包含更多全局图像特征，CLIPScore 主要测量全局图像-文本相似度。
- **BERTScore**：呈非单调变化（某些层高至 0.9，某些层低至 0.1-0.3），因为 BERTScore 仅比较文本描述，不区分视觉特征的层级深度。高 BERTScore 的层提供语义更丰富、更连贯的文本概念。

### 6.6 计算成本分析（Table 11）

基于 Gemma-3n-4B 骨干网络（约 4 x 10^9 参数，每 token 约 8 x 10^9 FLOPs）估算：

| 类别 | CGDL | CoX-LMM | 说明 |
|------|------|---------|------|
| 训练 GPU FLOPs | 约 4.28 x 10^16 | 约 7.20 x 10^16 | CGDL 约减少 40% |
| 训练 CPU ops（SNMF 字典学习） | 约 3.28 x 10^11 | 约 1.43 x 10^11 | CGDL SNMF 迭代次数更多（5000 vs 200），但样本更少（1600 vs 35000） |
| 推理 FLOPs / 图像 | 约 1.6 x 10^11 | 约 1.6 x 10^11 | 几乎相同（余弦相似度计算可忽略） |

- **端到端训练时间**（10 个概念）：CGDL 421s vs. CoX-LMM 4375s（约 **10 倍加速**）。
- CGDL 的 GPU FLOPs 减少主要来自：(1) 使用更少的残差样本（16000 vs. 35000）；(2) 更短的 token 序列（227 tokens vs. 257 tokens）。

**FLOPs 分解详情**（CGDL）：
- 概念袋创建（3000 张图像，T=436）：约 1.05 x 10^16 FLOPs
- SAM 分割（3000 张图像，每张约 1.1 x 10^12 FLOPs）：约 3.3 x 10^15 FLOPs
- 残差提取（16000 样本，T=227）：约 2.91 x 10^16 FLOPs
- SNMF 字典学习（CPU，N=1600, D=2048, K=2, 5000 iters, 10 concepts）：约 3.28 x 10^11 CPU ops

---

## 七、失效模式、适用前提与局限性

### 7.1 方法适用前提

CGDL 要求目标 LVLM 能理解基本语言指令（如附录 B 中的 prompt），这对于大多数现代 LVLM 是成立的。但这也意味着方法的适用性与模型的 prompt 理解能力绑定——若模型无法正确遵循二元提示，概念提取质量将退化。

### 7.2 失效模式：文本接地偏移（附录 E.6, Figure 11）

**核心问题**：概念向量的视觉接地正确，但文本接地可能偏移。

**具体案例**：
- LVLM 将 bears 的图像错误预测为 beaver（视觉幻觉），CGDL 的概念解释显示激活更多响应了 rabbit 而非 bear。
- 更关键的是：视觉接地正确激活了 beaver-like 的身体区域，但文本接地错误地映射到了 "base"、"prediction" 等无关 token（可能因 "water-based" 的干扰）。
- 概念 cat 的文本接地偏移为 "dog"、"category"、"kat"。

**原因分析**：概念向量在视觉上是稳定且可靠的（跨不同输入保持一致的视觉模式激活），但从概念向量到文本 token 的解码映射对 LVLM 解码器分布的局部变化仍然敏感。视觉和文本接地之间存在不完全对齐。

### 7.3 论文声明的局限性

1. **缺乏层次化概念组织**：CGDL 发现的概念是扁平列表，未捕获概念间的层次关系（如 animal -> mammal -> dog）。
2. **仅在 LVLM 上验证**：尚未在其他类型模型架构上评估。
3. **依赖 prompt 理解能力**：对 LVLM 的指令跟随能力有最低要求。
4. **训练与推断分离**：当前是纯后验（post-hoc）方法，不参与模型训练。

---

## 八、主要贡献

1. **提出 CGDL 框架**：首个弱监督、可扩展的自回归 LVLM 概念发现框架，克服了先前方法仅支持单物体（single-object）设置的根本限制。在 Table 1 中对 9 种方法的系统对比中，CGDL 是唯一同时支持文本接地（TG）、自回归模型（AutoReg）、且可处理多物体/无限物体场景的方法。

2. **SAM 驱动的弱概念定位**：利用 Segment Anything Model 为概念袋自动构建高质量的正/负样本图像块，无需任何人工标注，使方法能扩展到大规模概念词汇表。

3. **概念引导的对比残差提取方案**：二元提示（binary prompt）约束 + 每概念二基分解（two-basis decomposition per bag）的创新设计，强制分离概念信号与背景噪声，实现概念向量的单义性（monosemanticity）。核心洞见是：概念质量取决于数据如何呈现给字典，而非分解算法本身。

4. **多模态忠实归因**：通过稀疏性、稳定性、重叠度、CLIPScore、BERTScore、C-Deletion/C-Insertion 六项指标的全方位评估，证明 CGDL 在概念质量和归因忠实性上全面超越 CoX-LMM。可扩展到 1000 个概念且不损失纯度与唯一性。

5. **模型无关性与可复现性**：在三个不同架构 LVLM 上验证有效性，提供完整可复现流程（单张 RTX 3090, 24GB），代码已匿名开源。

6. **Token 级别后验解释能力**：CGDL 是首个能在自回归 LVLM 中提供逐 token 概念归因的方法，这是之前所有方法均不具备的能力。

---

## 九、对后续研究的启发与潜在改进方向

1. **层次化概念结构**：论文明确指出的未来工作方向是将 CGDL 扩展到捕获层次化概念（hierarchical concepts），例如同时学习 animal -> mammal -> dog -> husky 这种粗细粒度的层次关系，使概念解释更具结构化。

2. **超越 LVLM 的应用**：将 CGDL 的评估扩展到 LVLM 以外的模型架构，如纯文本大语言模型或其他多模态架构（如 video-LLM、audio-LLM）。

3. **解决文本接地偏移问题**：视觉接地正确但文本接地偏移是 CGDL 已知的失效模式。可能的改进方向：
   - 对解码器投影层施加正则化约束，强制概念向量与目标 token 的对齐。
   - 使用集成策略融合多层的文本接地结果以提高鲁棒性。
   - 引入对比学习损失，显式优化视觉接地与文本接地之间的一致性。

4. **训练阶段整合概念**：当前 CGDL 是完全的后验（post-hoc）方法。将概念向量融入模型训练过程——类似于 Concept Bottleneck Models 的思路——可能同时提升模型的内在可解释性和预测性能。

5. **与稀疏自编码器（SAE）的深度结合**：CGDL 的二基分解与 SAE 的稀疏编码具有天然互补性——CGDL 提供干净、低噪声的数据表示，SAE 在此基础上学习更丰富的稀疏特征。大词汇量场景下 SAE 的训练效率问题可能在 CGDL 的数据预处理下得到缓解。

6. **动态/增量式概念发现**：当前概念词汇表是离线静态构建的。开发在线或增量式概念发现机制，使 CGDL 能够持续适应新数据和新概念的出现，对于部署环境中的持续监控尤为重要。

7. **偏见检测与缓解工具**：CGDL 能够揭示模型内部学到的偏见概念（例如将某些职业与特定性别/种族关联）。可基于 CGDL 开发自动化偏见检测工具，并进一步设计概念级别的模型编辑（concept editing）方法以缓解这些偏见。

8. **更丰富的评估体系**：当前评估主要依赖 CLIPScore 和 BERTScore 等自动化指标。未来可引入：
   - 人工评估（human evaluation）的概念质量判断。
   - 因果干预测试（causal intervention），验证操纵概念向量是否能可靠地改变模型输出。
   - 更多样的下游任务评估（如 VQA、captioning 中的概念归因质量）。

9. **多概念交互建模**：当前 CGDL 将每个概念独立处理（per-bag two-basis）。建模概念之间的交互关系（如共现、互斥、组合）可能进一步提升概念向量的语义精度。

---

## 十、关键技术细节补充

### 10.1 与 CoX-LMM 的本质区别

CGDL 论文明确指出，其贡献不是简单的 "Semi-NMF + 预处理"，而是一个**通用的、模型无关的框架**，将概念发现重新定义为**对比的一对多分解（contrastive one-vs-all decomposition）**。关键差异如下表所示。

### 10.2 与 Concept Bottleneck Models (CBMs) 的区别

CBMs（Oikarinen et al., 2023; Yang et al., 2023）使用 LVLM 提取文本概念，然后**训练线性代理模型**预测模型输出。CGDL 同样使用 LVLM 识别概念，但**不训练任何代理模型**——发现的概念直接用于从同一 LVLM 的隐藏状态中事后提取概念向量，解释该 LVLM 的预测行为。这是一种更纯粹的后验可解释性范式。

### 10.3 模型分解架构

遵循 Koh et al. (2020) 和 Alam et al. (2025) 的分解方案，LVLM 被形式化为：
- **嵌入函数 g**：视觉编码器 + 桥接层（bridging）+ 解码器注意力层
- **输出函数 h**：投影层（projection）+ softmax

这种分解允许在不修改模型（frozen）的前提下从任意中间层提取概念向量。文中选择倒数第二层 norm 层作为主要分析层，因为前序工作证明该层产生高质量嵌入（Parekh et al., 2024; Kim et al., 2018），且层消融实验（Figure 12）验证了该选择的合理性——最深层的 CLIPScore 最高，BERTScore 也保持在合理水平。
