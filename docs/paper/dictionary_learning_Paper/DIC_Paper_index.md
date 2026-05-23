# Dictionary Learning Paper Index

> 本索引由所有论文的详细总结自动生成，用于快速定位参考论文。
> 实际论文内容请点击对应总结文件查看。
> 生成日期：2026-05-19
> 总论文数：19
> 综述论文数：2
> 研究论文数：17

---

## 一、论文主表

| 序号 | 论文标题 | 年份 | 核心方法/关键词 | 一句话贡献 | 总结文件路径 |
|------|----------|------|-----------------|-----------|-------------|
| 1 | RoseCDL: Robust and Scalable Convolutional Dictionary Learning for Rare-event and Anomaly Detection | 2026 | RoseCDL、随机窗口化、内联异常检测 | 同时解决卷积字典学习的可扩展性与异常鲁棒性两大瓶颈，将CDL重新定义为局部patch分布估计问题，实现无监督罕见事件检测 | Paper_summary/summary_yehya2025rosecdl.md |
| 2 | Diverse Dictionary Learning | 2026 | 依赖稀疏 (Dependency Sparsity)、广义可识别性、集合论不确定度 | 提出多样化字典学习理论框架，在最基本假设下保证隐变量子集的集合论可识别性，揭示依赖稀疏作为跨VAE/GAN/Diffusion的通用归纳偏置 | Paper_summary/summary_zheng2026diverse.md |
| 3 | Share Your Attention: Transformer Weight Sharing via Matrix-based Dictionary Learning | 2026 | MASA (Matrix Atom Sharing in Attention)、Matrix PCA | 将注意力矩阵压缩建模为字典学习问题，通过共享矩阵原子实现跨层权重复用，66.7%压缩率下性能接近完整Transformer，推理零额外开销 | Paper_summary/summary_zhussip2026share.md |
| 4 | Concept-Guided Dictionary Learning for Interpretable Concept Extraction and Attribution in Large Vision-Language Models | 2025 | CGDL、二基字典分解、概念引导 | 首个面向自回归大视觉语言模型的弱监督概念提取框架，通过二元提示与二基分解实现可扩展的单义概念发现与多模态归因，可处理1000个概念 | Paper_summary/summary_kadirconcept.md |
| 5 | Leveraging the Sequential Nature of Language for Interpretability | 2025 | Temporal SAE、时序对比损失、语义/句法解耦 | 首次在SAE训练中显式建模语言的时序结构，将特征空间分为高层语义与低层句法并通过InfoNCE对比损失约束，语义和上下文分类显著超越标准SAE | Paper_summary/summary_bhalla2025leveraging.md |
| 6 | Towards Scientific Discovery with Dictionary Learning: Extracting Biological Concepts from Microscopy Foundation Models | 2025 | ICFL (Iterative Codebook Feature Learning)、PCA白化、batched-OMP | 首次将字典学习应用于无监督科学数据领域，从显微图像MAE模型中提取与细胞类型/遗传扰动相关的生物学概念，实现单细胞精度可解释性 | Paper_summary/summary_donhauser2024towards.md |
| 7 | Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models | 2025 | A-SAE / RA-SAE、原型约束、数据凸包锚定 | 识别并系统量化SAE训练的严重不稳定性（跨运行间约50%概念不一致），通过将字典原子约束在数据凸包内实现稳定性从0.54提升至0.93，重建质量基本持平 | Paper_summary/summary_fel2025archetypal.md |
| 8 | SUE: Sparsity-based Uncertainty Estimation via Sparse Dictionary Learning | 2025 | SUE、稀疏字典学习+PMI不确定性估计 | 提出基于稀疏字典学习的不确定性估计框架，通过字典原子激活与预测错误的逐点互信息量化置信度，在低精度/高不确定性场景中显著优于MC Dropout等基线 | Paper_summary/summary_ficsor2025sue.md |
| 9 | Dictionary Learning: The Complexity of Sparse Superposed Features with Feedback | 2025 | 反馈复杂度、低秩特征矩阵、O(p^2)理论紧界 | 给出通过智能体三元组反馈学习字典特征矩阵的反馈复杂度紧界——通用采样下为Theta(p(p+1)/2)，低秩结构可降至Theta(r^2+p)，在ChessGPT/Pythia SAE字典上实验验证 | Paper_summary/summary_kumar2025complexity.md |
| 10 | A Comprehensive Survey of Transfer Dictionary Learning | 2025 | 迁移字典学习综述 (TDL)、CDDL/DADL六分类体系 | [综述] 首篇迁移字典学习系统性综述，按交叉域/域自适应和监督层次建立六分类体系，覆盖26篇核心算法，总结五个基准数据集上的实验规律 | Paper_summary/summary_li2025comprehensive.md |
| 11 | Restyling Unsupervised Concept Based Interpretable Networks with Generative Models | 2025 | VisCoIN、概念翻译器Omega、Viewability | 将概念激活映射到预训练生成模型隐空间，实现高质量交互式概念可视化，新定义忠实性与一致性评估指标，在细粒度分类任务上大幅超越FLINT和FLAEM | Paper_summary/summary_parekh2025restyling.md |
| 12 | Interpreting Large Text-to-Image Diffusion Models with Dictionary Learning | 2025 | PCA白化归一化、ITDA vs SAE、FLUX.1 | 首次将SAE和ITDA扩展到12B参数的FLUX.1文生图扩散模型，发现残差流极度各向异性导致>99%死亡特征，提出PCA白化方案将其降至<30% | Paper_summary/summary_shabalin2025interpreting.md |
| 13 | Robust Deep Dictionary Learning via Self-Expression Neighbor Atom Enhancement | 2025 | DNADL、邻域自表达流形、原子增强 | 将图拉普拉斯信息深度嵌入字典原子更新核心过程，通过谱聚类邻域划分与自表达权重增强实现图结构与字典的动态协同优化，在6个数据集上全面超越SOTA | Paper_summary/summary_song2025robust.md |
| 14 | A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima | 2025 | 分段双凸优化、伪局部极小值、Feature Anchoring | 建立SAE/Transcoder/Crosscoder的统一SDL理论框架，证明其分段双凸结构，从理论上解释多义性/死神经元/特征吸收的必然性，并提出Feature Anchoring正则化方法 | Paper_summary/summary_tang2025theoretical.md |
| 15 | Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning | 2024 | e2e SAE、KL散度端到端训练、功能重要性 | 首次通过最小化模型输出分布KL散度而非重构MSE训练SAE，相同性能解释量所需每数据点特征不到标准SAE的一半，且不牺牲可解释性 | Paper_summary/summary_braun2024identifying.md |
| 16 | Fast Structured Orthogonal Dictionary Learning using Householder Reflections | 2024 | Householder反射、O(np)复杂度、样本复杂度理论 | 首次给出结构化正交字典学习的样本复杂度理论，基于Householder反射实现O(np)非迭代字典恢复，样本受限场景显著优于Procrustes方法 | Paper_summary/summary_dash2025fast.md |
| 17 | Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models | 2024 | p-Annealing、Coverage/Board Reconstruction评估指标 | 在棋类GPT上提出有监督SAE评估指标（Coverage和Board Reconstruction），并引入p-Annealing方法——将L1惩罚退火为非凸L_p范数，效果媲美Gated SAE但前向计算量减半 | Paper_summary/summary_karvonen2024measuring.md |
| 18 | Learning the Sparse Prior: Modern Approaches | 2024 | 稀疏先验综述、算法展开 (Unrolling)、近端梯度法 | [综述] 系统综述稀疏编码模型/优化算法/字典学习三大技术脉络，在统一实验设置下对比7种优化算法在4种场景的性能，得出APG/AFPG最优、联合优化优于交替优化等实验规律 | Paper_summary/summary_peng2024learning.md |
| 19 | Evolving Dictionary Representation for Few-shot Class-incremental Learning | 2023 | D-FSCIL、伪类增强学习、增量字典适配 | 首次将深度字典学习引入小样本类增量学习，通过类间Mixup生成伪类预留表示空间并配合字典正则化微调，在CIFAR100/miniImageNet/CUB200上全面超越SOTA | Paper_summary/summary_han2023evolving.md |

---

## 二、关键词/主题索引

### 1. Sparse Autoencoders (SAE) 用于机械可解释性
- [5] Bhalla et al. 2025 — Temporal SAE、时序对比损失、语义/句法解耦 → `Paper_summary/summary_bhalla2025leveraging.md`
- [7] Fel et al. 2025 — A-SAE/RA-SAE、原型约束、跨运行稳定性 → `Paper_summary/summary_fel2025archetypal.md`
- [9] Kumar et al. 2025 — 反馈复杂度、低秩特征矩阵恢复 → `Paper_summary/summary_kumar2025complexity.md`
- [12] Shabalin et al. 2025 — PCA白化归一化、FLUX.1 SAE/ITDA训练 → `Paper_summary/summary_shabalin2025interpreting.md`
- [14] Tang et al. 2025 — SDL统一分段双凸理论、Feature Anchoring → `Paper_summary/summary_tang2025theoretical.md`
- [15] Braun et al. 2024 — 端到端SAE、KL散度训练、功能重要性 → `Paper_summary/summary_braun2024identifying.md`
- [17] Karvonen et al. 2024 — p-Annealing、Coverage/Board Reconstruction评估 → `Paper_summary/summary_karvonen2024measuring.md`

### 2. 多模态/视觉模型中的概念提取与归因
- [4] Kadir et al. 2025 — CGDL、自回归LVLM概念发现与二基字典分解 → `Paper_summary/summary_kadirconcept.md`
- [6] Donhauser et al. 2025 — ICFL、显微图像MAE生物学概念提取 → `Paper_summary/summary_donhauser2024towards.md`
- [11] Parekh et al. 2025 — VisCoIN、生成模型驱动的交互式概念可视化 → `Paper_summary/summary_parekh2025restyling.md`

### 3. 字典学习理论与可识别性
- [2] Zheng et al. 2026 — 多样化字典学习、依赖稀疏、广义可识别性 → `Paper_summary/summary_zheng2026diverse.md`
- [14] Tang et al. 2025 — 分段双凸优化景观、伪局部极小值必然性 → `Paper_summary/summary_tang2025theoretical.md`
- [9] Kumar et al. 2025 — 基于反馈的特征矩阵学习复杂度界 → `Paper_summary/summary_kumar2025complexity.md`

### 4. 经典与结构化字典学习算法
- [13] Song et al. 2025 — DNADL、图拉普拉斯引导的邻域自表达原子增强 → `Paper_summary/summary_song2025robust.md`
- [16] Dash et al. 2024 — Householder反射正交字典、O(np)快速算法 → `Paper_summary/summary_dash2025fast.md`
- [18] Peng 2024 — [综述] 稀疏先验：近端梯度/ADMM/K-SVD/算法展开 → `Paper_summary/summary_peng2024learning.md`

### 5. 卷积字典学习与信号处理应用
- [1] Yehya et al. 2026 — RoseCDL、随机窗口化、罕见事件检测 → `Paper_summary/summary_yehya2025rosecdl.md`
- [13] Song et al. 2025 — DNADL、深度字典学习图像分类 → `Paper_summary/summary_song2025robust.md`

### 6. 迁移/域自适应与增量学习中的字典学习
- [10] Li et al. 2025 — [综述] 迁移字典学习：CDDL/DADL六分类 → `Paper_summary/summary_li2025comprehensive.md`
- [19] Han et al. 2023 — D-FSCIL、伪类增强的少样本类增量学习 → `Paper_summary/summary_han2023evolving.md`

### 7. 字典学习在分类、不确定性与异常检测中的应用
- [8] Ficsor et al. 2025 — SUE、稀疏字典学习+PMI不确定性估计 → `Paper_summary/summary_ficsor2025sue.md`
- [1] Yehya et al. 2026 — RoseCDL、内联异常检测与罕见事件发现 → `Paper_summary/summary_yehya2025rosecdl.md`
- [19] Han et al. 2023 — D-FSCIL、少样本类增量图像分类 → `Paper_summary/summary_han2023evolving.md`

### 8. 模型效率与权重压缩
- [3] Zhussip et al. 2026 — MASA、字典学习驱动的跨层注意力权重复用 → `Paper_summary/summary_zhussip2026share.md`
- [15] Braun et al. 2024 — e2e SAE、更少特征实现同等性能解释 → `Paper_summary/summary_braun2024identifying.md`

---

## 三、按年份统计

| 年份 | 论文数 | 综述数 |
|------|--------|--------|
| 2026 | 3 | 0 |
| 2025 | 11 | 1 |
| 2024 | 4 | 1 |
| 2023 | 1 | 0 |
| **合计** | **19** | **2** |

---

## 四、注意事项

1. **bhalla2025leveraging 已修复**：序号5（Bhalla et al. 2025）的总结文件此前为空占位，现已包含完整内容。Bhall2025为ICML 2025 Workshop on Assessing World Models论文，提出Temporal SAE方法利用语言时序结构解耦语义/句法特征。

2. **dash2025fast 年份标注差异**：序号16（Dash et al.）的文件名标注为2025，但实际arXiv预印本发布于2024年9月（arXiv:2409.09138），本表按其arXiv发表时间归类为2024年。

3. **kadirconcept 双盲审稿中**：序号4（Kadir et al.）论文处于双盲审稿（double-blind review）阶段，作者和发表会议/期刊信息尚未公开。年份基于上下文推断为2025年，待最终发表后确认。

4. **donhauser2024towards 文件名与发表年不一致**：序号6（Donhauser et al.）的文件名含"2024"，但论文实际发表于ICML 2025（PMLR 267），本表按实际发表年归类为2025年。

5. **yehya2025rosecdl 文件名与发表年不一致**：序号1（Yehya et al.）的文件名含"2025"，但论文实际发表于AISTATS 2026（PMLR Volume 300），本表按实际发表年归类为2026年。

6. **tang2025theoretical 审稿中**：序号14（Tang et al. 2025）文中有"upon acceptance"字样，表明当前处于审稿阶段，最终发表信息待确认。
