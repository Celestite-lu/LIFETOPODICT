# 论文总结：Dynamic Integration of Task-Specific Adapters for Class Incremental Learning

## 基本信息

- **标题**：Dynamic Integration of Task-Specific Adapters for Class Incremental Learning (DIA)
- **作者**：Jiashuo Li, Shaokun Wang (共同通讯), Bo Qian, Yuhang He, Xing Wei, Qiang Wang, Yihong Gong (通讯)
- **单位**：西安交通大学（软件学院 & 人工智能学院）
- **发表信息**：CVPR 2025
- **骨干网络**：基于 ViT-B/16，在 ImageNet-21K 上预训练

---

## 研究动机与问题定义

该论文聚焦于 **Non-Exemplar Class Incremental Learning (NECIL)**，即不存储旧任务样本的类增量学习场景。核心动机是其认为现有 PTM-based NECIL 方法存在两大缺陷：

### 1. 组合性不足 (Compositionality Deficiency)

- **共享参数空间导致任务干扰**：许多方法在共享参数空间中为所有任务进行调优，参数混合导致灾难性遗忘（如图1a所示）。
- **Prompt-based 方法的局限性**：现有 PET 方法（如 L2P, DualPrompt, CPrompt 等）通过设计 task-specific prompts 来隔离任务参数，但 prompt 选择与 class token ([CLS] token) 之间建立了过强的耦合关系，忽视了 **patch tokens 中蕴合的丰富语义信息**以及**旧任务参数中保存的完整任务特定知识**。此外，这些方法需要多次前向传播，计算开销大（如图1b所示，EASE 等方法 GFLOPS 极高）。

### 2. 模型对齐不足 (Model Alignment Deficiency)

- **特征漂移未得到充分约束**：增量学习过程中，旧任务样本的表示会因新知识的引入而改变（feature drift）。现有 PET 方法忽视了跨增量阶段模型一致性维护的必要性。
- **决策边界不正确**：新任务引入后旧任务的决策边界会发生改变，现有方法未能有效适配。
- **基于正则化的方法**（如对 class token 施加约束）过于严格，阻碍新任务学习。
- **基于原型的方法**（如使用高斯分布生成旧类特征再对齐分类器）会导致生成的特征与实际特征分布逐渐偏离（如图1c所示），造成不准确的决策边界。

---

## 方法/框架（重点）

DIA 框架由两大组件构成：**Task-Specific Adapter Integration (TSAI)** 和 **Patch-Level Model Alignment**（包含 PDL 和 PFR）。

### 4.1 总体流程

在增量任务 t，算法在每个 Transformer block b 中引入：
- 一个 **task-specific adapter** A^{t,b}（parallel to MLP）；
- 一个 **task signature vector** tau^{t,b} in R^d。

每个 image token 通过 signature vectors [tau^{i,b}]_{i=1}^t 被路由到相关的 adapters，各 adapter 独立处理输入 token，其输出通过由 task signature vectors 决定的标量进行加权合并。**在任务 t 期间，仅 A^{t,b} 和 tau^{t,b} 是可训练的，其余参数冻结。**

### 4.2 Task-Specific Adapter Integration (TSAI)

**Adapter 结构**：
- 采用 bottleneck 模块，包含下投影层 W_down in R^{d x r} 和上投影层 W_up in R^{r x d}（论文主实验中 r=8）。
- 通过残差连接修改 MLP 输出，并利用从 tau^t 导出的缩放向量 s^t 进行加权。

**Token-Level Routing 机制**（Eq.1-3）：
- 输入 X = [p_j^T]_{j=0}^L in R^{(L+1) x d}，其中 p_0 为 class token，p_j (j>0) 为 patch tokens。
- Task signature vector tau^t 为每个 token p_j 分配标量 s_j^t：

  s_j^t = <p_bar_j, tau_bar^t>,   s^t = [s_j^t]_{j=0}^L

  其中 p_bar_j = p_j / ||p_j||_2，tau_bar^t = tau^t / ||tau^t||_2（归一化后内积即余弦相似度）。

- 单 adapter 的 token 输出：

  tilde_p_j^t = s_j^t * A^t(p_j) = s_j^t * ReLU(p_j W_down) W_up
  hat_p_j = MLP(p_j) + tilde_p_j^t + p_j

**多 Adapter 集成（t>1）**（Eq.4-6）：
- 使用 softmax 归一化各 adapter 的路由权重后加权求和：

  [s_hat_j^i]_{i=1}^t = softmax([s_j^i]_{i=1}^t)
  tilde_p_j^t = sum_{i=1}^t s_hat_j^i * A^i(p_j)
  hat_p_j = MLP(p_j) + tilde_p_j^t + p_j

**知识保留与重建的理论分析**（Eq.7-10）：
- 对 adapter 权重矩阵 W = W_down W_up 进行 SVD（秩为 r）：

  W = U * diag(sigma) * V = sum_i u_i sigma_i v_i^T

- 在线性情况下，adapter 输出为：

  o = A(p) = sum_i rho_i v_i = V^T g(p),   g(p) = diag(sigma)^T U^T p

  其中 rho_i = sigma_i * (u_i^T p)。

- **核心洞察**：无论输入 p 如何变化，adapter 输出的判别信息 o 始终保持在由 v_i（右奇异向量）张成的**任务子空间**内。该子空间由 adapter 权重内在决定，独立于输入特征。加入 ReLU 后函数变为非线性，但子空间约束性质在补充材料中有进一步分析。此特性使得 TSAI 无需 exemplars 即可有效保留和复现旧任务知识。

### 4.3 Patch-Level Model Alignment

#### 4.3.1 Patch-Level Distillation Loss (PDL)

**目的**：在训练新任务 t 时维护旧任务特征一致性，允许贡献度高的 patch tokens 有更大灵活性，约束贡献度低的 patch tokens 与旧模型输出对齐。

**机制**（Eq.11-14）：
1. 获取当前模型（训练 t）输出 X^n = [p_j^t] 和旧模型（t-1 后冻结）输出 X^o = [p_j^{t-1}]，各含 L+1 个 tokens。
2. 基于 **角度相似度** 计算每个 patch token p_j 对当前新任务的贡献度（以 class token p_0^t 为参考）：

   alpha_cos = (p_0^t * p_j^t) / (||p_0^t||_2 * ||p_j^t||_2)
   alpha_ang = (pi - arccos(-alpha_cos)) / pi     （取值范围 [0, 1]）

3. 对 patch tokens 进行 L2 归一化后计算特征漂移：

   D(p_j^t, p_j^{t-1}) = ||p_bar_j^t - p_bar_j^{t-1}||_2

4. **PDL 损失**（仅对 patch tokens，从 j=1 到 L，显式排除 class token）：

   L_pdl = (1/L) * sum_{j=1}^L alpha_ang(p_0^t, p_j^t) * D(p_j^t, p_j^{t-1})

**设计思想**：贡献度大的 patch token 获得更大漂移容忍度（plasticity）；贡献度小的被约束对齐旧模型（stability）。这是一种自适应的 stability-plasticity 平衡机制。

消融实验（Table 5）证实：(a) 角度相似度 > 余弦相似度 > 欧氏距离；(b) 仅约束 patch tokens 优于同时约束 class token；(c) PDL 优于传统的对所有 tokens 施加 L1 特征蒸馏（L_fd）。

#### 4.3.2 Patch-Level Feature Reconstruction (PFR)

**目的**：在无 exemplars 条件下重建更贴合真实分布的旧类特征，为分类器对齐提供更准确的伪特征（替代高斯采样）。

**机制**（Eq.15-18）：
1. **原型存储**：在每任务 t-1 结束时计算各类别特征均值作为原型 mu_k^{t-1}。
2. **相对相似度差异**：训练 batch 中对每个 patch token p_{(i,j)}（第 i 样本第 j patch），衡量其是否比 class token 更接近旧类原型：

   delta_{(k,i,j)} = (alpha_cos(mu_k^{t-1}, p_{(i,j)}) - alpha_cos(p_{(i,0)}, p_{(i,j)}))
                    / (alpha_cos(mu_k^{t-1}, p_{(i,j)}) + alpha_cos(p_{(i,0)}, p_{(i,j)}))

   delta > 0 表示该 patch 更偏向旧类知识，更适合用于重建。

3. **筛选并加权**：delta_hat = max(0, delta)，softmax 归一化得 omega_{(i,j)}。
4. **凸组合重建**（Eq.17）：

   mu_hat_k^{t-1} = beta * mu_k^{t-1} + (1-beta) * sum_{i,j} omega_{(i,j)} * p_{(i,j)}

   其中 beta = 0.7 为超参数，控制原型与 patch tokens 信息混合比例。

5. **分类器对齐**：每 batch 随机采样 N=32 个类原型（涵盖 C^{1:t}），通过 PFR 生成伪特征，与真实训练样本一起输入分类器，以 L_CE 微调分类头（此阶段不更新 backbone/adapter）。

### 4.4 优化目标与训练流程

**两阶段训练**（遵循 SLCA [45] / PASS [51] 的流程）：

**阶段一 —— 新任务学习**（更新 adapter + signature vector + 分类头）：
- 联合优化：

  L_obj = L_CE + lambda * L_pdl

  其中 lambda = 0.1。

**阶段二 —— 分类器对齐**（仅微调分类头，其余冻结）：
- 随机采样 N=32 个类原型（涵盖 C^{1:t}），通过 PFR 生成伪特征。
- 伪特征与当前训练样本混合，以 L_CE 微调分类头。

---

## 实验设置

### 数据集
1. **Cifar-100**（100 类）
2. **CUB-200**（200 类，细粒度鸟类）
3. **ImageNet-R**（200 类，含卡通/艺术等与预训练 domain gap 大的风格）
4. **ImageNet-A**（200 类，自然对抗样本集）

类别划分遵循 EASE [49]，各类别间无重叠。

### Baselines（15 个方法，三大类）
1. **Prompt-based**：L2P [39], DualPrompt [38], CODA-Prompt [33], CPrompt [7], ConvPrompt [31], Adam-Prompt (shallow/deep) [47]
2. **Adapter-based**：Adam-Adapter [47], EASE [49], LAE [5], InfLoRA [25], C-ADA [6]
3. **Finetuning-based**：FT (Full Fine-tuning), SLCA [45], Adam-Ft [47]

### 评估指标
- A^T：最终任务后的准确率
- A_bar^T = (1/T) sum_{i=1}^T A^i：平均准确率（主要指标）
- 同时报告可训练参数量 (Params) 和推理浮点运算数 (FLOPS)

### 实现细节
- **骨干网络**：ViT-B/16，ImageNet-21K 预训练
- **初始学习率**：0.015，余弦退火衰减
- **训练**：每任务 20 epochs，batch size 32，Nvidia 3090 GPU (24GB)
- **Adapter 下投影维度**：r = 8
- **超参数**：beta = 0.7（PFR），lambda = 0.1（PDL）
- **分类器对齐**：N = 32 个类原型采样
- 所有对比方法结果均使用其公开源代码复现

### 实验配置
- **10 Task**：Cifar-100 (10类/任务)，CUB-200/ImageNet-R/ImageNet-A (20类/任务)
- **20 Task（长序列）**：ImageNet-R (10类/任务)，Cifar-100 (5类/任务)

---

## 核心结果与发现

### 10 任务设置（Table 1）

DIA 在四个数据集上的 A^{10} / A_bar^{10}（粗体=所有方法最佳）：

| 数据集 | DIA A^{10} | DIA A_bar^{10} | vs CPrompt A_bar^{10} | vs EASE A_bar^{10} |
|--------|-----------|---------------|----------------------|--------------------|
| ImageNet-R | **79.03%** | **85.61%** | +2.69% | +3.88% |
| ImageNet-A | **61.69%** | **71.58%** | +6.16% | +6.24% |
| CUB-200 | **86.73%** | **93.21%** | +5.55% | +2.70% |
| Cifar-100 | 90.80% (SLCA=91.26%) | **94.29%** | +1.76% | +1.94% |

**DIA 在四个数据集上均取得 A_bar^{10} SOTA**，ImageNet-R 和 ImageNet-A 上 A^{10} 也最优。

### 关键效率指标

- **参数量**：每任务 **0.17M**，为 SLCA/Adam-Ft (86M) 的约 **0.20%**（节省 **99.80%**）。
- **推理 FLOPS**：**17.91B**：
  - 比 EASE (177.11B) 降低约 **90%**
  - 比 CPrompt (23.62B) 降低 **24.17%**
  - 在几乎最低的参数和 FLOPS 开销下实现 SOTA 精度，达到效率与精度的最优平衡。

### 各类方法对比

**vs Prompt-based**：
- 相比 CPrompt (CVPR 2024)：四个数据集 A_bar^{10} 分别领先 2.69%、6.16%、5.55%、1.76%。参数减少 32%，FLOPS 减少 24.17%。
- 相比 ConvPrompt (CVPR 2024)：ImageNet-R A_bar^{10} 高出 4.06%（85.61% vs 81.55%）。

**vs Adapter-based**：
- 仅用 EASE 14.28% 的参数量，在 domain gap 大数据集（ImageNet-R +3.88%, ImageNet-A +6.24%）上提升尤为显著。

**vs Finetuning-based**：以 0.17M 的极轻量参数方案与之竞争，大多指标超越或持平。

### 20 任务长序列（Table 2）

| 数据集 | DIA A^{20} | DIA A_bar^{20} | 对比 |
|--------|-----------|---------------|------|
| ImageNet-R | **76.32%** | **83.51%** | 超第二名 CPrompt 2.05% (A_bar^{20}) |
| Cifar-100 | **88.74%** | **93.41%** | A_bar^{20} 与 SLCA(93.85%) 可比（-0.44%），仅用 0.17M vs 86M 参数 |

### 消融实验

#### 组件消融（Table 3, 10-task）

| 配置 | ImageNet-R A^{10} / A_bar^{10} | Cifar-100 A^{10} / A_bar^{10} |
|------|-------------------------------|-------------------------------|
| 仅预训练+分类器 (=FT) | 20.93 / 40.35 | 22.17 / 41.83 |
| + TSAI | 77.13 / 83.87 | 88.37 / 92.31 |
| + TSAI + PDL | 78.18 / 84.55 | 89.32 / 93.48 |
| + TSAI + PFR | 78.22 / 84.25 | 89.81 / 93.78 |
| + TSAI + PDL + PFR (**Full DIA**) | **79.03 / 85.61** | **90.80 / 94.29** |

- TSAI 独立使用即从 FT 的 ~40% 提升至接近 SOTA 水平，验证了 patch-level adapter 集成缓解任务干扰的核心作用。
- PDL (+0.68%/+1.17%) 和 PFR (+0.38%/+1.47%) 各自带来可观增益，叠加效果最佳。

#### 特征重建方法对比（Table 4）

| 方法 | ImageNet-R A^{10} / A_bar^{10} | Cifar-100 A^{10} / A_bar^{10} |
|------|-------------------------------|-------------------------------|
| DIA-Gau（高斯采样） | 76.07 / 83.72 | 89.04 / 93.51 |
| DIA-SLCA | 78.37 / 84.48 | 89.47 / 93.79 |
| **DIA-PFR** | **79.03 / 85.61** | **90.80 / 94.29** |

PFR 在 ImageNet-R 上 A_bar^{10} 比高斯采样高 1.89%。可视化（Fig.3）证实 PFR 生成的伪特征与实际特征高度重合（单一紧密簇），而高斯采样形成两个分离的簇。

#### PDL 相似度度量与蒸馏方式消融（Table 5）

| 配置 | ImageNet-R A^{10} / A_bar^{10} | Cifar-100 A^{10} / A_bar^{10} |
|------|-------------------------------|-------------------------------|
| PDL w/ alpha_eu（欧氏距离） | 76.38 / 81.73 | 87.42 / 92.91 |
| PDL w/ alpha_cos（余弦相似度） | 78.28 / 85.16 | 89.67 / 93.88 |
| **PDL w/ alpha_ang（角度相似度）** | **79.03 / 85.61** | **90.80 / 94.29** |
| L_fd（传统 L1 约束所有 tokens [37,51,52]） | 76.67 / 83.82 | 88.31 / 92.97 |
| PDL w/ cls（也约束 class token [43]） | 77.01 / 83.87 | 88.87 / 93.02 |
| **PDL w/o cls（仅 patch tokens）** | **79.03 / 85.61** | **90.80 / 94.29** |

关键发现：
- 角度相似度最优，欧氏距离因数值不稳定表现最差。
- 仅约束 patch tokens 优于约束所有 tokens（含 class token），因 class token 约束限制新任务学习。
- 传统 L_fd（L1 约束所有 tokens）过于严格，劣于 PDL。

#### 超参数消融（Fig. 4, Cifar-100）

- **lambda（PDL 权重）**：过大（>0.3）严重阻碍新任务学习，最优 lambda = 0.1。
- **beta（PFR 原型权重）**：过小（0.3）过度依赖 patch tokens 导致对齐不准，过大（0.9）过拟合原型，最优 beta = 0.7。

---

## 主要贡献

1. **提出 DIA 统一框架**：首次将 task-specific adapter 的 patch-level 动态集成与双层次（特征级+分类器级）模型对齐统一在 NECIL 中，同时解决组合性不足和模型对齐不足。
2. **TSAI 模块**：通过 patch-level adapter integration + token-level soft routing 实现灵活的增量多任务集成。SVD 理论分析证明 adapter 输出始终保留在任务特定子空间内，天然具备知识保留与复现能力。
3. **Patch-Level Model Alignment**：
   - **PDL**：基于角度相似度的贡献评估+差异化特征漂移约束，自适应平衡 stability-plasticity。
   - **PFR**：相对相似度差异筛选旧类相关 patch tokens + 原型凸组合，生成比高斯采样更准确的旧类伪特征，显著提升决策边界精度。
4. **SOTA 性能与极致效率**：在 4 个 NECIL 基准、2 种任务划分（10/20-task）下全面达到或接近 SOTA，同时仅用 0.17M 参数和 17.91B FLOPS（比 EASE 降 ~90%），实现精度与效率的最优平衡。

---

## 局限性

1. **对预训练模型质量的依赖**：性能高度依赖 ViT-B/16 在 ImageNet-21K 上的预训练质量。在 domain gap 极大的 ImageNet-A 上绝对精度仍偏低（A_bar^{10}=71.58%），受限于预训练特征泛化能力。
2. **仅验证 ViT-B/16**：未在其他 ViT 规模（ViT-L/H）或非 ViT 架构（Swin Transformer, ConvNeXt）上验证泛化性。
3. **固定 Adapter 容量**：所有任务统一使用 r=8，未根据任务复杂度自适应调整。
4. **存储开销随任务数线性增长**：每任务存储一个 adapter + signature vector。论文主要对比训练参数和 FLOPS，未详细量化总模型存储的扩展性。
5. **假设任务边界已知**：遵循标准 CIL 设定（已知任务 ID），未讨论 task-agnostic 场景。
6. **PFR 的原型鲁棒性未在极端数据条件下验证**：长尾分布或含噪标签场景下原型质量可能下降，影响对齐效果。

---

## 对后续研究的启发与潜在改进方向

### 短期/直接方向
1. **动态 Adapter 容量分配**：根据任务难度或类数量自适应调整 r。
2. **多骨干验证**：推广至 ViT-L/H、Swin Transformer、CLIP 等模型。
3. **原型构建鲁棒化**：引入 EMA 更新、多原型表示或少量合成样本提升 PFR 鲁棒性。
4. **Adapter 压缩/共享**：通过共享低秩基 + 任务特定系数控制超长序列下的存储增长。

### 中长期/深度方向
5. **Task-Agnostic CIL**：将 token-level soft routing 扩展到无任务 ID 场景，自动推理任务上下文。
6. **Adapter-Prompt 混合架构**：TSAI 的 soft routing 与 prompt selection 功能相似，联合优化可能进一步提升灵活性。
7. **维度级漂移控制**：以特征维度（而非 patch token 整体）为单位进行重要性加权约束（如基于 Fisher Information），实现更细粒度的 stability-plasticity 平衡。
8. **生成式旧类特征重建**：在 PFR 中引入轻量 diffusion/VAE 替代凸组合，生成更高质量的伪特征。
9. **在线/流式增量学习**：将 patch-level alignment 扩展到 single-pass 数据流场景。
10. **跨模态 CIL**：从纯视觉推广至 VQA、captioning 等多模态场景。
