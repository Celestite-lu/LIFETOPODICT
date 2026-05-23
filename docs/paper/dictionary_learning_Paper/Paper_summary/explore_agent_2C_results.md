# Explore Agent 2-C: 字典学习理论与优化实验经验挖掘结果

## Q1: 优化算法实验对比

| 对比维度 | 涉及方法 | 具体发现 | 论文来源 |
|----------|----------|----------|----------|
| 凸/非凸求解器收敛性对比 | PG, FPG, APG, AFPG, ADMM, AADMM | APG 和 AFPG 在所有 4 类稀疏编码场景（传统 SC、CSC、ML-CSC）中均达到最低目标函数值。传统 ADMM 在非凸 l0 目标上完全发散（目标值从 ~10.0 升至 ~12.4），而 AADMM 通过凸初始化和自适应 rho 调整平稳收敛至 ~9.2 | Peng 2024 (Section 4.2, Figure 7) |
| 回溯+动量联合加速效果 | APG vs PG, AFPG vs FPG | APG 在所有稀疏度 c 值（2-14）下优于 PG（c=2 时目标值 6700 vs 9500）；AFPG 在 Lagrangian 稀疏性设定下 10 次迭代即达到 ~7.8，ADMM 需更多迭代且终值更差 | Peng 2024 (Section 4.1.1, Figure 6; Section 4.2, Figure 7) |
| 极度稀疏条件下的算法选择 | MP vs PG vs APG | c=2（极度稀疏）时 MP 目标值（8500）甚至优于 PG（9500），但 APG 仍最优（6700）。说明在极端稀疏条件下贪心策略 MP 比基础梯度法 PG 更有效，但带回溯的 APG 仍为最佳 | Peng 2024 (Section 4.1.1, Figure 6) |
| 交替优化 vs 联合优化 | AO (Alternating Optimization) vs Joint (Joint Direct Optimization) | Joint 在所有字典学习实验（传统 DL、CDL）中以更少的外层迭代次数达到更优的最终函数值。l0 目标下 Joint 优势更大（差距 ~0.11 vs l1 下 ~0.005），非凸问题受益更显著 | Peng 2024 (Section 6.2, Figures 11-12; Section 6.3, Figures 13-14) |
| K-SVD 与 Proximal Gradient 字典更新对比 | MP+K-SVD vs MP+PG | K-SVD 早期迭代更优（10 轮：目标值 8.18 vs 8.15），得益于同时更新字典原子及其系数。后期差距缩小（100 轮：8.08 vs 8.07）。K-SVD 要求系数矩阵高度稀疏 | Peng 2024 (Section 6.1, Figure 10) |
| 非迭代统计方法 vs 迭代谱方法 | Householder 统计法 (Dash) vs Orthogonal Procrustes vs H0DLA/HmDLA | 在样本受限场景（p << n），本文方法 Frobenius 误差为 Procrustes 的 1/6 至 1/10。p=n 时 Procrustes 精确恢复（误差 0.0），统计方法的优势消失。总计算复杂度 O(nmp) vs O(n^2 p) per iter（Procrustes） | Dash 2025 (Section 5.2, Fig.1, Fig.2) |
| 算法展开 vs 传统优化的理论与实验鸿沟 | ISTA-Net, ADMM-Net 等 vs PG/ADMM | Peng 综述明确指出现有文献"缺乏与传统优化方法在效率-精度权衡上的量化对比"。算法展开的优势在于可解释性和端到端训练，但展开层数有限，表达能力受限于优化算法结构本身 | Peng 2024 (Section 3.4, Section 7.2 局限#4) |
| 多层结构的优化瓶颈 | PG vs FPG (ML-CSC/ML-CDL) | FPG 约 30 次迭代即接近收敛（目标值 7.70），PG 需 80 次以上（7.80）。**关键空白**：APG/AFPG 的参数自适应方案尚未成功推广到 ML-CSC/ML-CDL，联合优化方法也未实现 | Peng 2024 (Section 4.4, Figure 9; Section 6.4, Figure 15) |
| 超参数对收敛的敏感性 | eta (步长) 和 rho (惩罚参数) | eta=0.01（传统 SC）、eta=0.001（CSC）；非凸 ADMM 的 rho 需自适应调整（Peng 2019 方法），否则发散。APG/AFPG 的 eta 通过回溯自动估计，降低了对初始步长设置的敏感性 | Peng 2024 (Section 4.2, 4.3) |
| ICFL vs TopK SAE 重构质量 | ICFL (Batched-OMP + 梯度下降) vs TopK SAE | ICFL 在所有稀疏度水平上的余弦重构相似度均远高于 TopK SAE。TopK SAE + PCA 白化时训练很不稳定，ICFL 表现稳定。ICFL 死亡特征仅 55-341 个（/8192），TopK SAE 高达 7640-8026 个 | Donhauser 2025 (Section 5.1, 5.5, Figure 4C, Figure 9) |
| CGDL vs CoX-LMM 训练效率 | CGDL-SNMF vs CoX-LMM-SNMF/SAE | 端到端训练时间（10 个概念）：CGDL 421s vs CoX-LMM 4375s（约 10 倍加速）。CGDL 的 GPU FLOPs 约 4.28x10^16 vs CoX-LMM 的 7.20x10^16（减少约 40%） | Kadir (CGDL) (Section 6.6, Table 11) |

## Q2: 正交性约束实验影响

| 约束类型 | 涉及方法 | 对字典质量影响 | 对分类性能影响 | 论文来源 |
|----------|----------|---------------|---------------|----------|
| Householder 乘积结构正交字典 | Dash Householder 方法 (Algorithm 1/3) | 在样本受限场景（p=20, n=1000, m=10）Frobenius 误差仅 5-7，Orthogonal Procrustes 误差高达 43。p 增大至接近 n 时 Procrustes 反超（p=n 时误差 0.0）。p 增大误差单调递减，theta（稀疏度）越高误差越低（theta>=0.4 时误差稳定在 0.02-0.03） | 未涉及分类实验；论文目标为信号恢复和字典恢复精度 | Dash 2025 (Section 5) |
| 正交性假设下的可恢复性（Lemma 1） | Householder 字典 Cholesky 分解 | 若特征矩阵 Phi = L^T L 且 L 行正交，则可从 Phi 恢复 L 到正交变换和符号变化（分块对角正交矩阵，对应不同特征值块；重数为 1 的特征值仅允许 +/-1 符号变化）。若无正交性假设，仅恢复到正缩放等价 | 未涉及 | Kumar 2025 (Lemma 1) |
| 多 Householder 乘积不可唯一识别性（Lemma 1） | Householder 多因子 | 对任意 Householder 矩阵 H，存在 H1, H2, H3 != H 使得 H H1 = H2 H3。因此 m>1 时无法唯一识别各 H_i，只能评估整体 V 的 Frobenius 误差 | 未涉及 | Dash 2025 (Section 3.3, Lemma 1) |
| 过完备字典的列单位范数约束 | K-SVD, Proximal Gradient 字典更新 | 所有传统字典学习实验（Peng 2024）中字典列约束为 Gamma_C(d_m) 限制列位于单位范数球面/球内，属于半正交（列不要求相互正交）。非凸 l0 学到的字典元素比凸 l1 更锐利（边缘清晰、Gabor-like 结构明显） | 未涉及分类性能的直接对比 | Peng 2024 (Section 3.3.1, Section 6.2, Figure 12) |
| 核正交损失 L_orth（概念级别的正交性） | VisCoIN 的 L_orth 正则化 | 施加于 Psi 最后一层卷积层权重，鼓励 L2 归一化后权重列向量两两正交。定性观察表明该损失有助于减少多个概念对同类捕捉相同特征的现象（LPIPS 0.545 vs 0.556, FID 15.85 vs 9.43） | 使用 L_orth 未显著改变预测准确率，但提升了概念多样性，减少冗余 | Parekh 2025 (Section 3.4, Table 17) |
| 余弦相似度随机重置（隐式正交性维护） | ICFL 随机重置机制 | 每 100 步 SGD 后，对所有余弦相似度超过 0.9 的 W_dec 列对，随机重新初始化其中一个。可视为投影梯度下降中投影步骤的廉价替代——以计算成本换取算法简洁性，在实践中有效防止特征坍缩（collapse） | 直接影响字典质量和下游线性探测性能（特征坍缩会降低 BTA） | Donhauser 2025 (Section 3.3.3) |
| 字典正交性对样本复杂度的影响 | Householder 统计方法 | 验证了需要 p = Omega(log n) 样本即可 l_inf 恢复单 Householder 字典；m>1 时无法唯一恢复各因子。正交性假设使得字典学习从迭代优化转化为非迭代统计估计，计算复杂度从 O(n^2 p) 降至 O(nmp) | 未涉及 | Dash 2025 (Theorem 1, Section 5.1) |
| 半正交（过完备 64 列 x 96 行）与正交的实践差异 | 传统 DL: D in R^{64x96}（过完备），结构化正交（Householder, n x n 方阵） | 过完备字典（列数 > 行数）不可能同时全正交（列向量数超过空间维数）。Peng 实验中仅约束列单位范数，学到的字典原子呈结构化碎片（边缘、纹理等局部特征）。Dash 的 Householder 方法仅适用于方阵正交，不能直接扩展到过完备情形 | Peng 未涉及正交性分类比较；Dash 未处理过完备情形 | Peng 2024 (Section 6.1); Dash 2025 |

## Q3: 字典可识别性实验验证

| 验证方法 | 涉及方法 | 条件/发现 | 论文来源 |
|----------|----------|----------|----------|
| 集合论不确定度 R^2 验证（广义可识别性） | Diverse DL (Zheng 2026) — VAE + 依赖稀疏 | 在 3-5 维合成数据上，所有解耦条件的 R^2（Int, SymDiff, CompA, CompB）一致性地远低于已纠缠变量的基准 R^2（如 dim3: Int R^2=0.25 vs Ref=0.75）。直接支持广义可识别性：集合论操作定义的隐变量子集可在估计中被解耦恢复 | Zheng 2026 (Section 5.1, Figure 4) |
| MCC 元素可识别性验证 | Diverse DL (Zheng 2026) — VAE | 满足充分多样性条件的结构在所有 3-10 维均取得高 MCC（0.71-0.86）；全连接依赖结构 MCC 显著较低（0.31-0.56）。直接验证元素级可识别性仅在结构多样性存在时成立 | Zheng 2026 (Section 5.2, Figure 5) |
| 依赖结构可识别性（Theorem 2） | Diverse DL (Zheng 2026) — VAE | 定理证明：在充分非线性 + 依赖稀疏正则化条件下，隐变量-观测变量间的完整依赖结构可在置换意义下被识别（仅差重标记）。关键：不依赖于统计独立性假设，仅需 Jacobian 稀疏正则化作为估计中的归纳偏置 | Zheng 2026 (Theorem 2) |
| 可识别性条件违反实验 | Diverse DL (Zheng 2026) — VAE | 当充分多样性被违反（全连接结构）时 MCC 大幅下降；当 alpha=0（无正则化）时 MCC 从 0.83 降至约 0.68-0.73。加性噪声下依赖稀疏方法 MCC 仅有微小下降（dim3: 0.826 -> 0.821），而 Base 模型严重恶化（0.381） | Zheng 2026 (Section 5.2, Appendix C.1 Tables 5-6) |
| l_inf 恢复保证的有效性验证 | Dash 2025 Householder 方法 | m=1 时通过 u 的 l_inf 误差验证 Theorem 1：p 增大误差单调递减，theta 越高误差越低，p>=2 且 theta>=0.7 时 l_inf 误差稳定在约 0.02。验证了 p = Omega(log n) 样本充足性 | Dash 2025 (Section 5.1, Fig.3) |
| 多 Householder 不可识别性验证 | Dash 2025 Algorithm 3 | 实验确认 Lemma 1：m>1 时无法唯一恢复各 H_i，只能评估整体 V 的 Frobenius 误差。误差随 m 增大轻微上升（累积误差），但本文方法误差始终显著优于 Procrustes | Dash 2025 (Section 5.2, Fig.1) |
| 线性探测 BTA（Balanced Test Accuracy）作为字典质量代理 | ICFL (Donhauser 2025) | 在 5 个分类任务（从易到难）上比较原始表示 vs ICFL 重构表示的线性探测准确率。简单概念（细胞类型、批次、CRISPR）几乎全部信号被保留；困难任务（siRNA 1138 类、功能基因组 39 类）大量信号被保留但有明显下降 | Donhauser 2025 (Section 5.4, Figure 4A) |
| Pearson 相关系数与 CellProfiler 手工特征对比 | ICFL (Donhauser 2025) | ICFL 特征的最大平均选择性分数与领域专家设计的 CellProfiler 特征呈现强相关（r=0.71）。ICFL 仅用约 100 个非零元素就获得了 CP 在约 300 个非零元素时的可比选择性，说明更稀疏但等质量的表示 | Donhauser 2025 (Section 5.3) |
| 选择性分数作为字典原子特异性度量 | ICFL (Donhauser 2025) | 在五个分类任务上系统计算特征选择性：细胞类型任务有 73 个特征平均选择性 > 0.5；批次效应 11 个 > 0.5；即使最困难的功能基因组任务也有 >100 个特征选择性达 0.1 以上。提供了"哪些原子真正编码了有意义概念"的定量证据 | Donhauser 2025 (Section 5.2) |
| 零空间 vs 行空间的线性探测（特征存储位置的可识别性） | ICFL (Donhauser 2025) — ViT MAE 架构 | Token 分解为 x = x_row + x_null（行空间传递到解码器，零空间仅用于内部处理）。零空间分量的线性探测准确率始终与完整 token 几乎相同；行空间分量显著较低，甚至低于随机 512 维子空间。结论：生物学概念主要存储在零空间中 | Donhauser 2025 (Appendix A, Figure 7) |
| 特征等价性（仅恢复到正缩放） | Kumar 2025 反馈复杂度框架 | 学习者只能学到特征等价 Phi' = lambda * Phi*（正缩放）。除非在强正交性假设（Lemma 1）下才可恢复到正交变换和符号变化。完整字典恢复仍是开放问题 | Kumar 2025 (Definition 2, Lemma 1) |
| 低秩结构对可恢复性的影响 | Kumar 2025 Theorem 1 | 当特征矩阵具有低秩性质（rank=r），反馈复杂度可从 O(p^2) 降至 Theta(r^2 + p)。实验验证：RFM 特征（r=4, p=10）仅需 55 个反馈（=p(p+1)/2）达到 ground-truth MSE；特征分解法仅需 r(r+1)/2 + p - r = 38 个反馈 | Kumar 2025 (Section 5.1, Appendix C.1) |

## Q4: 字典可视化/可解释性诊断

| 诊断方法 | 涉及方法 | 具体技术 | 发现/教训 | 论文来源 |
|----------|----------|----------|----------|----------|
| 概念激活修改 + 生成模型可视化（VisCoIN 核心） | VisCoIN (Parekh 2025) | 通过概念翻译器 Omega 将概念激活 Phi(x) 映射到 StyleGAN2-ADA 的 W+ 隐空间；修改单个概念激活（乘以因子 lambda = 0-4）并观察生成输出的变化。1 次前向传播即可完成可视化（vs FLINT 需 1000 次反向传播） | 定性展示了细粒度语义修改（如 CUB-200 上的 Red-eye、Blue upperparts；CelebA-HQ 上的 Makeup、Eye squint）。局限性：单个概念函数有时同时修改多个高层特征 | Parekh 2025 (Section 3.5, Figure 4) |
| 概念可视化一致性（CC_k）定量指标 | VisCoIN (Parekh 2025) | 选取概念 phi_k 最相关的 100 个样本，生成高激活 vs 零激活两组图像（各 100 张），训练线性 SVM 区分两组。分类准确率即为 CC_k（%），越高表示概念在不同图像上产生一致的语义修改 | VisCoIN 在所有三个数据集上一致性均值最高（82.7%-85.5%）且标准差最低（8.3-13.9）。FLAEM 一致性接近随机水平（~55%）。Lambda 从 2 增大到 3 时一致性进一步提升（CUB-200: 85 -> 93.4） | Parekh 2025 (Table 1b, Table 8) |
| 激活最大化可视化 FID 对比 | VisCoIN (Parekh 2025) vs FLINT 激活最大化 | 用 FID 衡量生成解释图像与原始数据分布的距离。VisCoIN（lambda=4）的 FID 在所有数据集上显著低于 FLINT 激活最大化图像（Stanford Cars 差距最大：8 vs 45.72，约 5.7 倍） | 预训练生成模型是实现高质量、高自然度概念可视化的关键。标准解码器的逐像素重建指标（MSE）虽好，但感知质量远不如生成模型 | Parekh 2025 (Table 9) |
| Token 级热力图分析（单细胞精度） | ICFL (Donhauser 2025) | 生成每个图像中 8x8 patch token 与特定特征方向的余弦相似度热力图；结合通道特异性分析（线粒体、内质网、RNA、质膜/高尔基体通道） | 成功区分同一图像中的对照样细胞（87.1% 召回率）与受扰动细胞（100% 召回率）。Mann-Whitney U 检验确认两组 token 对齐分数存在统计学显著差异（p < 0.0001） | Donhauser 2025 (Section 5.6, Figure 5-6, Table 3) |
| 选择性分数（Selectivity Score） | ICFL (Donhauser 2025) | 源自神经科学（Hubel & Wiesel, 1968）的指标，衡量特征对特定标签的单义性：P(feature active \| label i) - P(feature active \| other labels) | 提供每个字典原子"专一性"的量化度量。选择性 > 0.5 的特征被认为是高度专一的。可系统筛选出真正编码生物学概念的特征方向 | Donhauser 2025 (Section 4.5, 5.2) |
| CGDL 视觉接地 + 文本接地双模态解释 | CGDL (Kadir) | **视觉接地（X_MAC）**：对每个概念向量的二基分解 V 矩阵中 top-alpha_MAC 最高激活图像块进行可视化。**文本接地（X_MAT）**：利用 LVLM 的输出投影层 h^(l) 将概念向量映射到词汇空间，取 top-tau（50）个最高分 token | 视觉接地可靠——正确激活了概念对应的图像区域（如 "beaver-like" 身体区域）；但文本接地可能偏移（如视觉对应 beaver 但文本映射为 "base"/"prediction"）。视觉-文本接地间存在不完全对齐 | Kadir (CGDL) (Section 3.5, Section 4.3, Figure 11) |
| 多义性 vs 单义性原子可视化对比 | CGDL (Kadir) vs CoX-LMM | 对比 CoX-LMM 产生多义特征（f3 同时激活 "chair" 和 "cat"）与 CGDL 产生单义特征（f1 -> "cat", f3 -> "table"）的直观图示 | 多义性是字典原子可解释性的核心障碍。CGDL 通过每概念袋二基分解强制分离信号与噪声，有效抑制多义性 | Kadir (CGDL) (Figure 2, Figure 4) |
| 字典原子可视化（图像块/Gabor 滤波器） | K-SVD + PG 字典学习 (Peng 2024) | 将学习到的字典列向量 reshape 为 8x8 或 6x6 图像块进行可视化 | 非凸 l0 学到的字典元素比凸 l1 学到的更锐利（边缘清晰、Gabor-like 结构明显）。ML-CDL 第一层字典捕捉到类似 Gabor 滤波器的简单特征，第二层捕捉到更复杂的组合特征 | Peng 2024 (Section 6.1, 6.2, 6.4) |
| 概念稀疏性、稳定性、重叠度三指标 | CGDL (Kadir) | **Sparsity**（向上箭头）：概念表示中非零元素比例，越高越集中。**Stability**（向下箭头）：多次运行间概念向量的一致性。**Overlap**（向下箭头）：不同概念向量间的余弦相似度重叠，越低单义性越强 | CGDL-SNMF 在这三个指标上均最优：稀疏性 1.00，稳定性 ImageNet 0.02 / MSCOCO 0.00，重叠度 ImageNet 0.08 / MSCOCO 0.06。相比 CoX-LMM SNMF：稀疏性提升 4%，稳定性提升 11%，重叠度降低 17% | Kadir (CGDL) (Table 2) |
| 忠实性诊断（C-Deletion / C-Insertion） | CGDL (Kadir) | **C-Deletion**：逐步将 top-1/2/3 最有影响力的概念方向坐标置零，记录模型输出概率下降。**C-Insertion**：从零向量开始按相同排名逐步添加概念坐标，记录概率上升 | CGDL 呈现更陡峭、更有序的删除/插入曲线，Top-1 > Top-2 > Top-3 区分度清晰。CoX-LMM 曲线相对平坦且区分度不足。证明 CGDL 概念排名更具忠实性和区分度 | Kadir (CGDL) (Figure 1) |
| 忠实性（Faithfulness FF_x）概念移除效果 | VisCoIN (Parekh 2025) | 将局部相关性 r_k(x) > tau 的概念激活设为 0，计算预测概率下降 FF_x = g(x_tilde)_c_hat - g(x_rem)_c_hat。理想情况下 FF_x > 0 | 在复杂数据集上 VisCoIN 忠实性远超 baseline：CUB-200 tau=0.1 时 FF_x=0.251（FLINT 仅 0.004）；Stanford Cars 0.161（FLINT 仅 0.001）。Random baseline 的 FF_x 接近 0 | Parekh 2025 (Table 2b) |

## Q5: 稀疏正则化系数与字典大小搜索策略

| 搜索维度 | 涉及方法 | 推荐策略 | 失败模式 | 论文来源 |
|----------|----------|----------|----------|----------|
| 稀疏正则化系数 lambda（l0 vs l1） | 传统 SC 和 CSC 的 APG/AFPG (Peng 2024) | Peng 2024 在所有实验中使用固定值：l0 lambda=0.01（CSC）或 0.05-0.1（传统 SC）；l1 lambda=0.05（CSC）或 0.1-1.0（传统 SC）。未提供系统化搜索策略。l0 与 l1 的 lambda 不可直接比较，因为两者惩罚尺度不同（硬阈值 vs 软阈值） | 过大的 lambda 导致过度稀疏化（过少活跃原子，重建质量下降）。lambda 的选择需配合算法步长 eta（APG 中 eta 通过回溯自动适应，降低了对 lambda 的联合敏感性） | Peng 2024 (Section 4.2, 4.3, 6.2, 6.3) |
| 稀疏度上界 c 的网格搜索 | 传统 SC 固定稀疏度问题 (Peng 2024) | c 从 2 到 15 的网格实验。目标函数值随 c 增大单调递减（更多非零元素可更好重建）。c 的选择是重建质量与可解释性/计算成本的权衡 | 未讨论过稀疏化或欠稀疏化的系统性症状 | Peng 2024 (Section 4.1.1, Figure 6) |
| 字典大小（特征总数 M）的选择 | ICFL (Donhauser 2025) | 默认 M=8192（经验选择），未提供搜索策略。稀疏度 k=100（J=20, L=5），也未给出与 M 交互关系的分析 | 未讨论 | Donhauser 2025 (Section 4.2) |
| 字典大小 K 消融实验（每概念袋） | CGDL (Kadir) | K 从 {2, 10, 30, 50, 100} 的消融实验表明：BERTScore 基本恒定（约 0.88），不受 K 增大显著影响；CLIPScore 随 K 增大而显著下降（从 0.616 降至约 0.49）。**推荐 K=2（每概念袋二基分解）**——在归因质量与计算效率间取得最佳平衡 | 大字典损害视觉-文本对齐质量（CLIPScore 下降），但文本语义质量（BERTScore）基本不变。过度增加字典大小会导致原子间干扰和坍塌 | Kadir (CGDL) (Table 8) |
| SNMF 稀疏权重 alpha 的消融选择 | CGDL (Kadir) | alpha 从 {0, 20, 100, 150, 200} 的消融实验：BERTScore 在所有 alpha 下保持恒定（约 0.881）；CLIPScore 随 alpha 增大轻微提升（0.610 -> 0.621）。**推荐 alpha=20**——效果稳定且避免过度稀疏化 | 过度稀疏化的症状：CLIPScore 虽可能略提升但视觉接地可能丢失细节。实验表明 alpha 在较大范围内（0-200）表现稳定 | Kadir (CGDL) (Table 9) |
| 依赖稀疏正则化权重 alpha 敏感性 | Diverse DL (Zheng 2026) | alpha（即 lambda）从 {0, 0.001, 0.005, 0.01, 0.03, 0.05} 的敏感性分析：MCC 从 alpha=0 时约 0.679-0.732 平稳递增至 alpha=0.03 时约 0.810-0.827，alpha=0.05 时约 0.810-0.842 趋于平稳。**方法在非欠正则化区域（alpha >= 0.03）表现稳定** | 欠正则化（alpha=0）导致 MCC 显著下降（~0.68-0.73 vs ~0.83 at alpha>=0.03）；过正则化（alpha>0.05 未测试）未发现副作用。方法对超参数不过分敏感 | Zheng 2026 (Appendix C.1, Table 6) |
| 稀疏性权重 delta 的消融 | VisCoIN (Parekh 2025) | delta 从 0.2 到 20 的消融实验（CUB-200）：delta=20 严重损害准确率（从 79.44% 降至 ~76%）和重建质量（LPIPS 从 0.545 恶化为 0.629）。**推荐 CUB-200/Stanford Cars: delta=0.2；CelebA-HQ: delta=2**（因二分类任务更简单） | 过度稀疏化（delta 过大）的症状：准确率大幅下降、重建质量恶化、概念信息被"挤压"丢失。需根据任务复杂度调节 | Parekh 2025 (Table 19) |
| 概念数量 K 的缩放实验 | VisCoIN (Parekh 2025) | K 从 64 到 512（CUB-200）：准确率从 78.91% 提升至 79.78%，LPIPS 从 0.624 改善至 0.537。**推荐 K=256（收益递减点）**：K 从 256 增至 512 时 LPIPS 仅从 0.545 到 0.537（边际收益小） | K 过小（如 64）时概念承载能力不足，重建和分类性能受限。K 过大（如 512）边际收益递减但计算成本增加 | Parekh 2025 (Table 16) |
| 重建分类损失权重 gamma 的 tradeoff | VisCoIN (Parekh 2025) | gamma 从 0 到 0.5（CUB-200）：揭示了 faithfulness 与感知重建质量之间的 tradeoff。高 gamma 提升 faithfulness（0 -> 0.1: FF_x 从 0.001 到 0.146），但 LPIPS 恶化（0.52 到 0.545）。**推荐 gamma=0.1（CUB-200）以优先保证感知质量支持 viewability** | gamma=0 时 faithfulness 几乎为零（0.001）——概念学得与预测无关。gamma=0.5 时 LPIPS 严重恶化（0.634）——生成图像出现伪影 | Parekh 2025 (Table 14) |
| 稀疏采样中稀疏度与样本效率的关系 | Kumar 2025 反馈复杂度框架 | Theorem 4 给出了稀疏采样下的理论反馈复杂度上界。实验显示：高稀疏度（mu=0.9，非零概率仅 10%）在标准反馈量（55）下 MSE 高达 5.636（远差于 ground-truth 0.0016）；低稀疏度（mu=0.1，非零概率 90%）在 55 反馈下达 ground-truth 质量 | 过度稀疏（mu 过高）导致需要指数级更多反馈才能恢复特征矩阵。稀疏性未必有利——"稀疏表示稠密特征可能反而需要更多反馈" | Kumar 2025 (Section 5.1, Fig.1, Theorem 2 Remark) |
| 来自 Peng 综述的跨方法 lambda 实践经验 | Peng 综述的所有方法 | 综述总结了大量方法中 lambda 的设定但没有给出系统化选择策略。不同任务/模型/loss 的 lambda 值差异很大（从 0.01 到 1.0），说明 lambda 高度依赖具体场景 | 综述未涉及 | Peng 2024 (贯穿 Section 4, 6) |

## 覆盖确认
- [x] Sparse Prior Survey (summary_peng2024learning.md)
- [x] Householder Orth (summary_dash2025fast.md)
- [x] Diverse DL (summary_zheng2026diverse.md)
- [x] Feedback Complexity (summary_kumar2025complexity.md)
- [x] ICFL (summary_donhauser2024towards.md)
- [x] CGDL (summary_kadirconcept.md)
- [x] VisCoIN (summary_parekh2025restyling.md)
