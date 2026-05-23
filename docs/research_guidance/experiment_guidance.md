# LifeTopoDict 实验推进指南

> 基于 42 篇论文的系统挖掘 | 证据强度：【多篇验证】【两篇验证】【单篇报告】【理论推测】
> 生成日期：2026-05-21 | 对应设计文档：LIFE_TOPO_DICT_README.md
> 2026-05-22 更新：LifeTopoDict 后续实验已关闭 residual dictionary growth 与 additive edge-aware scoring。固定使用 `use_dictionary_growth=false`、`dict_max_growth_per_task=0`、`use_edge_aware_scoring=false`。本文中关于 growth 触发、growth cap、growth 方向质量、additive edge-aware scoring 的内容仅作历史诊断和失败模式参考，不再作为后续实验设置、消融项或参数搜索项。

---

## 一、实验设置与协议

### 1.1 公平比较协议要点

> 通用 CIL 设置

**原则一：内存预算必须统一换算模型存储 + 数据存储。**【多篇验证】

动态网络方法（DER 9.27M, MEMO 7.14M, DyTox 10.7M）享有隐式内存优势。统一到 23.5MB 后 DER 对 iCaRL 的领先从 9.07% 降至 1.74%。参考换算：1 个 ResNet32 约等于 603 张 CIFAR 图像（Zhou2024CILSurvey, Ashtekar2025CILFirstPrinciples）。LifeTopoDict 引入共享字典 + 增量 growth，必须报告 **dict_atoms + 每节点 sparse coefficients + SOINN node states + node-atom assignment indices + class means** 的完整内存 breakdown。

**原则二：计算约束应和内存约束同等对待。**【两篇验证】

大多数 CIL 论文只约束内存不约束计算（Kim2025CLVisionTechReport: 部分高引用方法计算量超过全量重训练）。LifeTopoDict 推理包含 dual-view inference（alpha-balanced global mean + local node matching），训练包含 sparse_encode 循环（已知瓶颈）、growth 判断和 SOINN 节点更新。需报告训练时间/FLOPs，且与同特征 baseline 使用相同 epoch 预算。

**原则三：预训练模型必须与从头训练方法分区比较。**【多篇验证】

L2P (ViT-B/16 IN21K 85.7M) vs iCaRL (ResNet32 0.46M) 差距 80+pp 大部分来自预训练差异（Zhou2024CILSurvey, Lin2024TPL, Ashtekar2025CILFirstPrinciples）。LifeTopoDict 主表使用 frozen ViT-B/16-IN21K，同特征 classifier heads 必须共享同一 frozen feature cache。预训练收益不能归因于 lifecycle 机制。

**原则四：任务顺序显著影响方法排名，必须多顺序报告。**【多篇验证】

单一顺序下几个百分点差异在统计上可能不显著（Lin2024TPL: 5 种顺序; MiN: 6 种随机种子）。LifeTopoDict 默认 3 个 class order seeds，报告 mean +/- std。

**原则五：超参数搜索预算在方法间需统一。**【单篇报告】

EWC lambda 仅搜索 4 值；BiC 需验证集学习校正（Zhou2024CILSurvey）。LifeTopoDict 所有 same-feature heads 使用同等调参预算，且 base/val split 保持固定。

**原则六：无免费午餐——不同内存预算下最优方法不同。**【单篇报告】

小内存下 Replay/iCaRL 更强，大内存下动态网络更强（Zhou2024CILSurvey）。LifeTopoDict 在不同 memory budget 下的表现应通过 AUC-A 或 memory Pareto 评估，而非单一 LCE。

| 公平性维度 | LifeTopoDict 应报告的内容 | 依据 |
|-----------|--------------------------|------|
| 内存 | dict_atoms + sparse codes + node states + assignment indices + class means | Zhou2024CILSurvey, Ashtekar2025 |
| 计算 | 训练 wall-clock time / FLOPs；sparse_encode 瓶颈报告 | Kim2025CLVisionTechReport |
| 预训练 | 同骨干同冻结状态比较 | Lin2024TPL, Ashtekar2025 |
| 顺序 | 3 seeds, mean ± std | Lin2024TPL, MiN |
| 调参 | 统一网格预算 | Zhou2024CILSurvey |
| 内存效率 | AUC-A / Memory Pareto 替代 LCE | Zhou2024CILSurvey |

**LifeTopoDict 内存 breakdown（对比 HC-SOINN baseline）：**

| 组件 | LifeTopoDict 存储 | HC-SOINN 存储 |
|------|-------------------|--------------|
| Prototype/Node 存储 | M_t × d (共享字典) + C_t × k × d (sparse codes) | C_t × (18~21) × d (per-class nodes, 无共享) |
| SOINN 节点状态 | C_t × N_c × state_metadata | C_t × N_c × state_metadata (相同) |
| 分类均值 | C_t × d (class means) | C_t × d (class means, 相同) |
| Assignment 索引 | C_t × N_c × k (int indices) | 无 |
| 总计（相对） | 更低（字典共享压缩） | ~18-21 nodes/class，与 < 20 exemplars/class rehearsal 相当 (E14) |

### 1.2 数据集与协议选择

> 通用 CIL 设置

**推荐最小数据集组合：**

| 数据集 | 用途 | LifeTopoDict 特定意义 | 依据 |
|--------|------|----------------------|------|
| CIFAR-100 | 大量消融 | 验证共享字典在通用分类的表达力；单 prototype 假设相对安全 | 综合 PTM-CIL 论文 |
| CUB-200 | **核心测试平台** | 单 prototype 表达力天花板——HC-SOINN 在此 +21.00pp (65.56% -> 86.56%)，LifeTopoDict 目标：用更低内存接近此增益 | HC-SOINN (Yi 2026) |
| ImageNet-R | 附录级域偏移测试 | 域偏移可能影响字典原子泛化能力；不同骨干间的有效性差异在此最大 | SD-LoRA, MoAL, MiN |
| ImageNet-A | 极限压力测试 (appendix) | Full Fine-Tuning 在此完全崩溃 16.31% (E7)；方法间差距最大 (MiN vs MOS ~7.2pp) | SD-LoRA (Table 3), MoAL (Table 1) |

**HC-SOINN 的协议（LifeTopoDict 继承）：**

```
CUB-200:   Base 100 + 10 sessions × 10 classes (20×10 变体: Base 100 + 19×5 + 1×5)
CIFAR-100: Base 50 + 5 sessions × 10 classes (10×10)
ImageNet-R: 40×5 (200 classes, 5 per session)
```

**骨干网络：** ViT-B/16-IN21K 是 PTM-based CIL 的事实标准（6 篇方法论文全部使用，EASE/CL-LoRA/SD-LoRA/DIA/MoAL/MiN）。附表使用 ViT-B/16-IN1K。**关键风险：HC-SOINN 在不同骨干上效果不均**——EASE backbone 上 CUB-200 出现 -0.08% 负增益，CL-LoRA 仅 +0.13% 微小增益。LifeTopoDict 必须跨多个 backbone 验证。

**关键注意事项：**

1. **不同论文公开协议不可直接混表。** HC-SOINN 使用 CUB-200 20×10 / ImageNet-R 40×5；EASE 使用 B0 Inc5/10/20；SD-LoRA 使用 N=5/10/20 (A17)。主表必须统一重跑所有同特征 baseline，公开数值仅作背景。
2. **不同数据集的字典原子-节点最优比不同。** D-FSCIL: CIFAR-100 0.7 atoms/class, CUB-200 2.5, miniImageNet 5.0。LifeTopoDict 的 dict_sparse_k 按数据集消融 {3, 5, 8, 12}（A16）。
3. **基类初始性能决定全序列性能上限。** 几乎所有 SA Top5 的方法也获 AA Top5 (FSCIL Survey, Zhang 2025)。LifeTopoDict base 阶段的字典初始化质量 + SOINN 聚类初始化直接决定全序列性能上限。Stage B0:B1 比例需消融 {20:80, 50:50, 80:20}（A18, F16-F17）。
4. **data split 必须分离触发信号和评估信号。** 当前任务数据划分为 D_t^{train} / V_t^{cal} / V_t^{eval}，三者互不重叠。V_t^{cal} 用于 residual/usage 统计触发 growth 和 lifecycle 决策；V_t^{eval} 仅用于验证 lifecycle 动作是否改善下游指标。

### 1.3 评估指标选择与解读

| 指标 | 公式/说明 | 选择理由 | 依据 |
|------|----------|---------|------|
| **Average Accuracy** (主指标) | 所有任务结束后的平均 acc | 反映全程稳定性，避免"早期糟糕但后期补救"的方法被掩盖 | Zhou2024CILSurvey, Lin2024TPL |
| **Final Accuracy** (主指标) | 最后任务后全类别 acc | 最终部署性能 | Zhou2024CILSurvey (A8) |
| **Average Forgetting** (主指标) | 标准 CIL 遗忘率 | 同 feature baseline 间的可比指标 | Lin2024TPL (A7) |
| **Old/New Accuracy** (主指标) | 分别报告旧类/新类 acc | 暴露 stability-plasticity tradeoff | 所有 CIL 论文惯例 |
| **Old-New Harmonic Mean** (主指标) | 2 × A_old × A_new / (A_old + A_new) | 避免单方面优化旧类或新类 | GPA 消融验证; LifeTopoDict 设计文档 (A11) |
| **任务内/跨任务准确率** (辅助) | ATD: Acc on all non-target classes | ATD 比任务内遗忘更决定 CIL 性能；强任务内+强跨任务=强 CIL | Ashtekar2025 (A9) |
| **PAD** (LifeTopoDict 专用) | 1 - <p_c, q_c*> per old class | 监控 prototype drift 累积效应 | LifeTopoDict 设计文档 (D1-D3) |
| **EffRank** (LifeTopoDict 专用) | exp(-sum sigma_i log sigma_i), sigma_i 为字典原子矩阵奇异值 | 监控 growth 是否真正增加表达能力而非冗余 | SAE (B12, C1) |
| **GTE** (LifeTopoDict 专用) | Growth Treatment Effect: (acc after growth) - (acc before growth) on V_t^{eval} | 量化每次 growth 事件的即时收益 | LifeTopoDict 设计文档 |
| **avg_residual** (LifeTopoDict 专用) | 所有新类节点在现有字典下的平均重构残差 | 驱动 growth 判定的核心信号 | LifeTopoDict 设计文档 |
| **protected/inactive ratio** (LifeTopoDict 专用) | \|M_t^{protected}\| / \|M_t^{inactive}\| | 监控生命周期健康度 | LifeTopoDict 设计文档 |
| **OOD AUC vs CIL ACC** (诊断) | Pearson r 最高 0.980 | 提高 OOD AUC 可带来 1.5-3.4x CIL ACC 线性提升 | Lin2024TPL (A10) |

**指标陷阱：**

- 传统遗忘率公式混淆"类别增多效应"与"遗忘效应"，应使用修正版 F_CIL 以 Joint Training 为参照（Lin2024TPL, Ashtekar2025）。
- LCE 在 Delta_Mem=0 时无定义，不能作为 memory efficiency 指标（E13）。改用 memory-budgeted accuracy / memory Pareto / AUC-A。
- 预训练 VLM 特征下需报告零样本泛化退化 (Transfer)。DPW 取 Transfer/Avg./Last 三者的 Mean 作为最终排名（Jang2026DPW, Kim2025CLVisionTechReport, A12）。

---

## 二、训练稳定性

### 2.1 字典训练常见问题与解决方案

> LifeTopoDict 使用 Top-K Ridge 稀疏编码（dict_sparse_k + dict_ridge_lambda），字典训练问题与 SAE/D-FSCIL 共享诊断框架。

| 问题 | 症状 | 根因 | 解决方案 | 依据 |
|------|------|------|---------|------|
| **死亡原子 (Dead Atoms)** | >99% atoms 从未被任何节点选中 | PTM 特征极度各向异性（第一 PCA 分量解释 21%-58.4% 方差） | PCA 白化：death rate 从 >99% 降至 <30% (SAE)；或 class means / k-means 初始化替代随机初始化 | Shabalin et al. (FLUX.1 SAE), B9-B10 |
| **随机初始化不稳定** | 跨 seed 余弦稳定性仅 ~0.5，约 50% 概念在运行间变化 | 随机初始化方向不匹配数据流形 | K-Means 蒸馏：Stability 0.542 -> 0.927 (Archetypal SAE)；Feature Anchoring：M_GT 0% -> 24.13% (SDL Theory) | Fel et al., Tang et al., B13-B15 |
| **字典秩膨胀 (Rank Inflation)** | TopK SAE Stable Rank=141.6 vs RA-SAE=5.89（受控）；EffRank 停滞但 M_t 增长 | 过参数化，字典方向冗余 | 监控 EffRank；当 EffRank 停滞但 M_t 增长时说明新增 atom 冗余 | Fel et al., B12 |
| **字典过大导致精度下降** | D-FSCIL m=90 准确率 51.5% < m=70 的 52.2% | 冗余 atoms 引入噪声 | EffRank 监控 + 定期 cosine similarity 检查；当 atom 间余弦相似度 >0.9 时随机重置其一 (ICFL) | D-FSCIL, ICFL (Donhauser 2025), C1-C2 |
| **字典过小导致信息压缩** | m=10 -> 46.0% vs m=70 -> 52.2% (D-FSCIL) | 字典容量不足，无法表达类别差异 | Base 字典大小 M_0 需充足；按 dict_sparse_k 消融 {3, 5, 8, 12} | D-FSCIL, C2 |
| **O 投影对压缩高度敏感** | MASA-QKV (PPL=112.23) vs MASA-QKVO (PPL=133.62) | 输出侧信息压缩损失大 | LifeTopoDict 不压缩分类侧 prototype 输出，避免类似退化 | MASA (Zhussip 2026), C5 |
| **跨投影共享字典削弱专门化** | MASA 共享字典削弱每类投影能力 | 独立字典 > 共享字典 | 若 LifeTopoDict 使用多层 ViT 特征，需权衡 per-layer 独立字典 vs 共享字典 (C4) | MASA |

**LifeTopoDict 专项诊断原则：**

1. 每任务结束后监控 `inactive` atom 比例。若 base 训练结束后大量 atom 从未被任何节点 top-k 选中，说明字典初始化方向与 PTM 特征流形不匹配——优先考虑 PCA 白化预处理 (B9-B10)。
2. 存活字典元素数与梯度范数呈负相关 (Braun et al., e2e SAE)。若训练中梯度范数异常升高而 EffRank 下降，是 atom 塌缩的早期预警信号 (C3)。
3. L_coh 惩罚 + ICFL 式高余弦随机重置构成字典冗余的双层防护 (C6, X1)。L_coh 默认阈值 gamma=0.2；当任意两个 active atom 余弦相似度 >0.9 时，触发随机重置 (ICFL)。

**LifeTopoDict 特有的字典一致性问题（已知 bug）：**

| 问题 | 影响 | 临时对策 |
|------|------|---------|
| **dict_atoms 归一化不一致** | base atoms 存储未归一化的类均值，grown atoms 存储归一化的残差方向。center_raw 语义在二者间不同，可能导致 sparse_encode 对 base atoms 和 grown atoms 产生不同尺度的系数 | 在 compress() 中统一归一化语义；实验前检查所有 atom 的 L2 norm 分布 |
| **sparse_encode 逐样本循环瓶颈** | 当前实现为 Python for-loop per sample，在 CUB-200 全量（~12k 样本 × 数百 atoms）下训练速度可能是主要瓶颈 | 报告 per-task sparse_encode wall-clock time；考虑 batch 向量化或 GPU 批处理优化 |

### 2.2 Loss 收敛诊断流程（简化）

> LifeTopoDict 的损失结构比 LifeGAD 更简单：无 ridge shadow residual、无 proxy loss、无交替优化。诊断聚焦于 Top-K Ridge 的数值稳定性。

**Step 1: 检查 ridge_lambda 是否启用。**

D-FSCIL 中 lambda=0 导致模型完全失效（0% 准确率），lambda=0.1 恢复至 52.2% (B7, X4)。LifeTopoDict 的 dict_ridge_lambda 是 Top-K Ridge 闭式解的核心稳定器。若 sparse code 数值爆炸（系数 > 10^3），立即检查 ridge_lambda。建议范围 0.01-0.1；默认 0.1。若 kappa_ridge = condition number of (D_k^T D_k + lambda I) 极高（>10^4），以 2x 步长倍增 lambda 直至稳定。

**Step 2: 检查学习率是否过高。**

SAE 中 lr >= 0.001 导致 L0 在初期下降后持续增长（尤其深层，Braun et al. B4）。LifeTopoDict 字典训练中高学习率可能导致稀疏度失控。

**Step 3: 检查 atom 初始化质量。**

随机初始化导致严重跨运行不稳定性 (B14)。确认使用了 class means / k-means / whitened k-means 初始化。LifeTopoDict 的 SOINN 聚类初始化提供天然的 k-means 引导。

**Step 4: 区分"稀疏不足"和"覆盖不足"。**

使用 residual 诊断表（简化版，无 ridge shadow）：

| r_c^{cos} | 诊断 | 动作 |
|-----------|------|------|
| 高 | 字典覆盖不足或 k 太小 | 先放宽 k (k <- k+2, up to k_max)，若仍高则允许 growth |
| 低 | 正常复用 | 不 grow |
| 低（但新类 acc 低） | 字典方向泛化但不具判别力 | 检查 atom 选择分数（ICFL Selectivity Score），可能需增加训练 |

**Step 5: 检查 dict_sparse_k 与 dict_ridge_lambda 的交互。**

Top-K Ridge 中 k 和 lambda 不是独立的：较大的 k 引入更多共线性风险，需要更大的 lambda 稳定。建议在 k 消融时同步调整 lambda 或报告 k-lambda 联合敏感性。

### 2.3 学习率与优化器建议

| 配置项 | 推荐值 | 依据 | 注意 |
|--------|--------|------|------|
| **LR 调度** | Cosine Annealing | 全部 6 篇 PTM-CIL 论文统一使用 (EASE, CL-LoRA, SD-LoRA, DIA, MoAL, MiN)，无一使用阶梯衰减 (B1) | LifeTopoDict 默认 |
| **优化器** | Adam (lr=0.001-0.008) | 简单 head 类方法 (FCS, PASS++) 多用 Adam lr=1e-3；完整系统多用 SGD (B2) | LifeTopoDict 作为 classifier head 可选用 Adam，与同特征 baseline 统一 |
| **字典训练 LR** | <= 0.0005 | 高 LR (>=0.001) 导致 SAE L0 失控 (Braun et al., B4) | 字典学习需比分类头低 2-4x |
| **Base task epochs** | 监控饱和点（验证集 EffRank 连续 5 epoch 波动 < 1% 视为饱和） | SAE_e2e+ds 在 400k 样本后性能饱和 (Braun et al., F19) | 避免无谓训练时间 |
| **Stage B0:B1 比例** | 需消融 {20:80, 50:50, 80:20} | M2SD 的 20:80 为候选消融点，不能照搬 (F16) | Base task 训练阶段划分对后续增量性能有决定性影响 |

**正则化系数的"甜蜜区"：**

| 参数 | 建议范围 | 过强后果 | 依据 |
|------|---------|---------|------|
| dict_ridge_lambda | 0.01-0.1 | lambda=0 数值崩溃；lambda>1.0 过度收缩 sparse code -> 表达力衰减 | D-FSCIL lambda=0 完全失效 (B7) |
| lifecycle_theta_support | 0.3-0.7 | 过高 -> 所有原子都 protected，新类无法适应；过低 -> 无保护，等同无 lifecycle | DIA: lambda>0.3 严重阻碍新任务 (E8) |
| lambda_b (base drift) | 0.01-0.1 | 限制非 protected atoms 适应新类 | D-FSCIL: alpha=100 过度约束 (E8) |
| dict_sparse_k | {3, 5, 8, 12, 15} | k 过小类似 VisCoIN delta=20 过度稀疏化 (CUB-200 79.44% -> ~76%)；k 过大类似 CGDL CLIPScore 从 0.616 降至 ~0.49 | VisCoIN, CGDL (D10, F3) |
| tau (温度) | 0.05 (默认) | 数据集依赖性：细粒度可能受益于低 tau; 域偏移可能有害 (D7) | MiN: ImageNet-R 10-steps tau 不敏感 (<0.4pp 波动) |

**损失权重的数据依赖性。** FCS 的 lambda 在 5/10-stage (0.1) 和 20-stage/ImageNet (0.03) 间切换；VisCoIN delta 按 CUB-200 (0.2) 和 CelebA-HQ (2) 调节 (B8)。LifeTopoDict 阈值/权重不应假设跨数据集恒定，至少区分通用分类 vs 细粒度 vs 域偏移三个场景。

---

## 三、字典生命周期诊断

> LifeTopoDict 使用简化生命周期：仅 **protected** 和 **inactive** 两种状态。
> 无 quarantined、selective rumination、stale-binding、reserve atoms、PMI gating、ridge shadow residual。
> 旧类支持由 coefficient magnitude（非 PMI）判定；growth 由 residual 阈值触发。

### 3.1 Atom Collapse/冗余诊断

> LifeTopoDict 特有诊断

**四级诊断指标：**

| 指标 | 计算方式 | 健康范围 | 阈值/警告 | 来源 |
|------|---------|---------|----------|------|
| **存活 atom 比例 (Alive Ratio)** | \|M_t^{act}\| / M_t (active = protected + used-above-threshold) | > 70% | < 50% 需白化或重新初始化 | SAE (Shabalin, X1) |
| **atom 间平均余弦相似度** | mean_{i!=j} \|d_i^T d_j\| | < 0.3 | > 0.5 需增强 L_coh 或重置 | SAE Coherence (X1) |
| **字典有效秩 (EffRank)** | exp(-sum sigma_i log sigma_i), sigma_i 为 D_t 奇异值 | 稳定增长或持平 | 停滞而 M_t 增长 -> 冗余 (C1, B12) | SAE Stable Rank (X1) |
| **选择分数 (Selectivity Score)** | S_m = (max activation on class c) / (sum activation across all classes) | S_m > 0.5 = 高度专门化 atom | S_m 普遍偏低 -> atom 方向泛化、缺乏类别特异性 | ICFL (Donhauser 2024) (C17) |
| **最长连续 inactive 任务数** | 连续 usage < lifecycle_theta_usage 的任务数 | < lifecycle_T_inactive (默认 3) | >= T_inactive -> 标记 inactive | LifeTopoDict 设计 |
| **低使用率比例** | \|{m: U_m < theta_u}\| / M_t | < 30% | > 50% 需诊断 | LifeTopoDict 设计 |

**LifeTopoDict 特有的使用率偏差问题：**

`_compute_atom_usage` 累加器从不衰减（已知 bug）——usage 统计天然偏向早期 atoms（因为它们参与更多次累计）。这导致：
- 后期新增的 grown atoms 可能被错误地标记为 "低使用率"
- 早期 base atoms 可能因历史累计过高而永远不会被标记为 inactive
- 对策：在诊断报告中区分 "原始累计 usage" 和 "滑动窗口 usage（最近 L 个任务）"，对比二者差异

**诊断流程（分步）：**

1. **Base 训练结束后**：检查 alive ratio。若 < 70%，尝试 PCA 白化（B9-B10）或增强互相关惩罚 gamma 从 0.2 降至 0.1。
2. **每任务结束后**：检查 EffRank 变化。若 EffRank 停滞但 M_t 持续增长，停止 growth 并考虑标记冗余 atoms。
3. **每任务结束后**：检查 atom 间余弦相似度分布直方图。若尾部分布接近 1.0，对相似度 >0.9 的 atom 对随机重置其一 (ICFL, C6)。
4. **跨任务**：追踪 alive ratio 和 EffRank 趋势。若持续下降，说明字典在增量过程中方向退化——增大 base atom 保护强度（提高 lifecycle_theta_support），同时监控 alive ratio 是否恢复。
5. **注意 dict_atoms 归一化不一致的连锁效应**：若 base atoms（未归一化）和 grown atoms（归一化）的 L2 norm 分布明显不同，则 cosine similarity、EffRank、Selectivity Score 等依赖归一化向量的指标在跨来源 atom 间不可直接比较。建议在诊断前统一归一化。

### 3.2 旧类支持可靠性（系数幅度 vs PMI）

> LifeTopoDict 使用 coefficient magnitude-based old support 替代 PMI 判定旧类原子保护。

**系数幅度作为旧支持信号的工作原理：**

对于每个旧类节点 n_c，其 sparse code a_c ∈ R^{M_t}（仅 k 个非零元素）。原子 m 对类 c 的支持度定义为该原子在所有类 c 节点中的归一化平均系数幅度：

```
support(m, class_c) = mean_{n in nodes(c)} |a_c[m]| / max_{m'} mean_{n in nodes(c)} |a_c[m']|
```

若 support(m, class_c) > lifecycle_theta_support（默认 0.5），则原子 m 在该类上视为 "old-supported" -> protected。

**系数幅度的可靠性条件：**

| 条件 | 说明 | 不满足后果 | 依据 |
|------|------|-----------|------|
| 1. 每类节点数充足 | 每类至少 >= 3 个 SOINN 节点 | 均值估计不可靠（类似 PMI 的 n_min 条件） | SUE, LifeTopoDict 设计 |
| 2. k 值适度 | k 不宜过大导致 coefficient 被稀释到过多 atoms | 所有系数都接近均值，无区分力 | VisCoIN, CGDL (D10, F3) |
| 3. 特征分布稳定 | PTM frozen 特征不会跨任务漂移 | 节点位置不变但系数幅度可能因新增 atoms 而重新分布 | LifeTopoDict 设计 |
| 4. 归一化一致 | 所有 dict atoms L2 归一化一致 | base vs grown atom 系数幅度不可比——已知 bug | LifeTopoDict 设计 |

**系数幅度何时失效：**

- **少样本节点（每类 1-2 个节点）**：均值支撑度不可靠，类似 PMI 在 < 5 个校准样本时退化。此时 lifecycle 决策退化为仅由 usage 驱动。
- **高度相关的 atoms**：若多个 atoms 高度相关，系数会分散到相关 atoms 上，单个 atom 的 support 被稀释——即使它对该类确实重要。这是 Top-K Ridge 的已知弊端（k 个 atoms 可能互相竞争）。
- **grown atoms 的起始 disadvantage**：新 grown atoms 刚加入时支持度为 0（因为旧类节点在 growth 前已完成 sparse encode），不会立即被标记为 protected。这可能导致 lifecycle 倾向于保护 base atoms——与 usage 累加器偏差叠加。

**与 PMI 的对比：**

| 维度 | PMI (原 LifeGAD) | Coefficient Magnitude (LifeTopoDict) |
|------|------------------|--------------------------------------|
| 复杂度 | O(C × M × N_cal) 每任务 | O(C × M) —— 直接从 sparse code 取 |
| 校准需求 | 需要独立 V_t^{cal} 样本 | 不需要额外校准（系数 = 训练副产品） |
| 信号粒度 | atom-error 关联（反映"原子是否导致错误"） | atom-usage 关联（反映"原子是否被使用"） |
| 少样本可靠性 | PMIInvalidRate 可能很高（< 500 samples） | 仅依赖节点数，不依赖额外校准样本 |
| 信息内容 | 包含错误方向信息（区分"被使用"和"被正确使用"） | 仅包含使用强度信息——丢失了 atom-error 关联 |
| 主要风险 | 少样本下 PMI 点估计不可靠 | 系数幅度高不意味着使用是正确的 |
| 硬阈值问题 | SUE 报告的硬阈值二值化丢失连续信息 (C9) | 同样适用——support threshold 也是硬阈值 |

**建议实验验证：**

| 变体 | 说明 | 目的 |
|------|------|------|
| `support_magnitude` (默认) | 仅用系数幅度判定 old support | LifeTopoDict 标准 |
| `support_usage_only` | 仅用 usage 判定（忽略系数幅度） | 消融系数幅度的独立价值 |
| `support_magnitude + error_weight` | 系数幅度 × (1 - 节点错误率) 加权 | 引入错误信息，接近 PMI 但更简单 |
| `sliding_window_support` | 仅用最近 L 个任务的系数（对抗累加器偏差） | 测试 usage 累加器偏差对 support 判定的影响 |

### 3.3 Growth 有效性验证

> LifeTopoDict 的 growth 由 residual 触发，无 secondary PMI gate，无 ridge shadow gate。

**Growth 触发条件（单门控）：**

1. 新类节点在现有字典上的 avg_residual > dict_theta_residual（默认 0.3）
2. 新增原子数 capped by dict_max_growth_per_task（默认 5）
3. 新增原子方向 = 残差方向（归一化）

**与 LifeGAD 的简化：** 无 r_c^{cos} vs r_c^{dense} 区分、无 secondary PMI gate、无 safety guard（k 放宽回退）。Growth 判定更简单但假阳性率可能更高。

**Growth 有效性验证消融：**

| 消融实验 | 对比 | 验证目标 |
|---------|------|---------|
| `w/o growth` (ablation_no_growth) | 固定字典无扩容 vs 完整 LifeTopoDict | growth 的整体贡献 |
| `fixed growth events + random directions` | growth 事件次数相同但新增 atom 方向随机 | 分离"发生 growth"与"残差方向信息" |
| `growth cap 消融` | dict_max_growth_per_task = {0, 1, 3, 5, 10, unlimited} | growth 激进程度对性能的影响 |
| `theta_residual 消融` | dict_theta_residual = {0.1, 0.2, 0.3, 0.5, 0.7} | 残差阈值的敏感性 |

**Growth 效果的量化指标：**

| 指标 | 计算方式 | 期望趋势 |
|------|---------|---------|
| **GTE (Growth Treatment Effect)** | (acc after growth) - (acc before growth)，在 V_t^{eval} 上 | > 0，至少不退化 |
| **Delta EffRank per growth** | EffRank 增长量 / 新增 atom 数 | > 0.1，否则新增仅冗余 |
| **New class residual reduction** | avg_residual 在 growth 前后的变化 | 应显著下降 (> 30% reduction) |
| **Growth rate per task** | 新增 atom 数 / 任务新类数 | 应递减或稳定，不应线性增长 |

**字典容量与边际收益：**

D-FSCIL 固定字典 m=10->70 持续提升 (46.0%->52.2%)，但 m=70->90 反降 (51.5%)。CLOVER 专家数 1->4 持续提升，5+ 后 Avg 下降 (C18, F15)。SAE 字典 size 从 16k 到 24k 有显著改善但 24k+ 饱和 (Braun et al., F9)。

LifeTopoDict 的容量上限设计：dict_max_growth_per_task 硬性 cap。若 EffRank 停滞，即使 avg_residual > dict_theta_residual 也不应继续 growth——转而考虑字典冗余诊断（3.1 节）。

**Growth 方向质量的验证：**

- Rank-1 是少样本下最有效的参数更新方式，优于 Rank-4/16/64 (Lark, E18)。LifeTopoDict 的 growth 方向来自残差方向（归一化），本质也是 Rank-1——但少样本下残差方向可能不准确。
- 验证方法：提前在 base 数据上做 leave-one-class-out 实验，验证残差方向能否在少样本下恢复 held-out class。

**已知风险：protected indices 映射 bug。** `compress()` 中 protected 索引映射错误导致 protected gate 在 MVP 中实际禁用。这意味着即使 lifecycle 正确标记了 protected atoms，它们仍可能在 growth 或字典更新中被覆盖。在修复前，growth 实验结果需标注此限制。

---

## 四、分类头几何

> LifeTopoDict 构建在 HC-SOINN 之上，共享其 dual-view inference、SLERP 球面更新和层次聚类初始化。
> 本节以 HC-SOINN 为核心参照系，所有设计决策从 HC-SOINN 的实证发现和理论推导出发。

### 4.1 Prototype Drift 与 HC-SOINN 的理论基础

**漂移累积效应——HC-SOINN 的核心动机：**

三类证据共同构成 HC-SOINN 的理论-实证闭环：

| 证据 | 内容 | 来源 |
|------|------|------|
| **实证：NC1 性质被破坏** | 初始任务训练不充分 + 增量步训练受限 + 特征漂移 → 类特征不再坍缩为单点，呈"哑铃形"或"新月形"复杂流形 | HC-SOINN Section 3.1, FCS (D1) |
| **实证：Procrustes 量化非线性漂移** | d_P^(t) 从 Task 2 起持续超 0.1 经验阈值，后期达 0.35-0.40；归一化距离无法被刚性变换近似 | HC-SOINN Figure 3, D2 |
| **理论：Proposition 1 误差界分析** | 假设漂移函数 phi 为 L-Lipschitz：NCM 误差上界 = L×R_c（全局半径），HC-SOINN 误差上界 = L×r_local（最大局部子簇半径）。层次聚类+SOINN 最小化量化误差 → r_local << R_c → 更紧的误差界 → 更少的灾难性遗忘 | HC-SOINN Appendix A |
| **理论：Proposition 2 贝叶斯最优性** | 假设特征服从混合 vMF 分布，双视角推理公式 S(x,c) = alpha×cos<f, mu_c> + (1-alpha)×max cos<f, v> 等价于混合 vMF 模型下的 MAP 估计；alpha = kappa_g/(kappa_g+kappa_l) 具有明确的统计解释——全局原型越可靠，alpha 越大 | HC-SOINN Appendix A |

**对 LifeTopoDict 的核心启示：** Proposition 1 直接论证了"多节点优于单原型"——共享字典 atom 提供的多节点表示基底继承了更紧的误差界。Proposition 2 为 alpha=0.5 的默认选择提供了贝叶斯最优性背书（当全局浓度 kappa_g 与局部浓度 kappa_l 相当时 alpha 自然取 0.5）。

**HC-SOINN 的 3 级洞察与 LifeTopoDict 的响应：**

| 层级 | HC-SOINN 发现 | LifeTopoDict 的响应 |
|------|-------------|-------------------|
| 第 1 级：单 prototype NCM 失效 | 单均值无法捕捉非线性流形漂移 | 继承多节点表示——通过 shared atoms 的 sparse 组合实现 |
| 第 2 级：多节点拓扑成功 | CUB-200 +21.00pp (65.56%→86.56%)；ImageNet-R +10.60pp | 用共享字典复现此增益，同时大幅降低内存 |
| 第 3 级：内存成本过高 | 每类 ~18-21 节点×d 维，无类间共享 (~9.2MB CUB-200) | 共享字典压缩至 ~0.43MB（~1/20），牺牲少量精度换取内存效率 |

**LifeTopoDict 的 Prototype Drift 防护（对应 HC-SOINN 三层防御）：**

| 防护层 | 机制 | HC-SOINN 对应 | LifeTopoDict 增强 |
|--------|------|-------------|-----------------|
| Level 1: 多节点表示 | shared atoms 提供跨类共享基底 | HC-SOINN 多节点拓扑 (HC+SOINN) | 共享字典 → 类间信息复用，压缩率 ~1/20 |
| Level 2: Atom 保护 | lifecycle_theta_support 保护旧类强依赖 atoms | SLERP 球面更新（减少方向漂移） | 系数幅度判定 + usage 双信号 |
| Level 3: 节点更新 | SOINN SLERP 保持节点在单位球面 | HC-SOINN Spherical Update | 继承 SLERP——无修改 |

**PAD 监控协议：**

```
- 每任务结束后计算所有旧类的 PAD_c = 1 - <p_c, q_c*>
- PAD 基线值应参考 HC-SOINN 的 Procrustes 漂移量级 (0.35-0.40 at late stages)
- 期望：LifeTopoDict 的 PAD 应 ≤ HC-SOINN 的 d_P（共享字典提供更稳定的表示基底）
- 告警阈值：PAD_max > 0.3 或 PAD_mean 环比增长 > 50% → 检查 protected gate
```

### 4.2 STAR 机制与"漂移适应"范式（可选增强）

> STAR (Structure-Topology Alignment via Residuals) 是 HC-SOINN 的可选增强模块，实现从"漂移抵抗"到"漂移适应"的范式转变。

**STAR 核心机制（5 步，Algorithm 2）：**

1. **计算瞬时漂移**：Delta_i = h_i^(t) - h_i^(ref)，其中 h_i 为锚点样本在当前/参考 backbone 下的特征
2. **EMA 平滑**：delta_i := (1-lambda)×delta_i + lambda×Delta_i，lambda=0.999
3. **点对点传输+重归一化**：v' = (v_raw + delta_i) / ||v_raw + delta_i||
4. **全局均值同步更新**：mu'_c = mu_c + (1/K_c)×sum_i delta_i，随后归一化
5. **刷新参考特征**：h_i^(ref) := h_i^(t)

**关键特性：**
- **无需重训练**：锚点样本仅用于推理时对齐，无梯度反传
- **计算开销极小**：+3.96% 训练时间，+0.01% 推理时间（HC-SOINN Table 3/4）
- **在 HC-SOINN 上额外增益**：STAR 贡献 A_Last +8.53 (DualPrompt+CUB-200)，+0.94 A_Avg (CODA-Prompt+CUB-200, Table 5)
- **与 rehearsal 正交**：理论上可与梯度 replay 结合

**对 LifeTopoDict 的启示：** STAR 的锚点样本存储与 LifeTopoDict 的 shared atoms 正交——可同时启用。若启用 STAR：每类需额外存储 ~18-21 个锚点样本（与 HC-SOINN 相同）。不启用时：LifeTopoDict 本身的多节点+保护机制已提供基础漂移抵抗力。LifeTopoDict MVP 中 STAR 未实现，可作为 P1 增强方向。

**STAR 的 HC-SOINN 验证效果：**

| Backbone | 数据集 | w/o STAR A_Last | w/ STAR A_Last | Delta |
|----------|--------|----------------|----------------|-------|
| DualPrompt | CUB-200 | 77.31 | 85.84 | **+8.53** |
| DualPrompt | CIFAR-100 | 82.98 | 85.63 | +2.65 |
| CODA-Prompt | CUB-200 | 85.75 | 86.17 | +0.42 |
| CODA-Prompt | ImageNet-R | 70.67 | 71.85 | +1.18 |

STAR 在长细粒度序列上增益最大——与 HC-SOINN 本身的增益模式一致。

### 4.3 HC-SOINN 组件调参与"tuning-free"哲学

> HC-SOINN 论文的核心设计哲学：**使用单一统一超参数配置，展示方法的 tuning-free 特性。**

**HC-SOINN 的"免调参"设计原则：**

论文有意不对各数据集单独调参（尽管 ImageNet-R 上调 alpha=0.6 可从 69.97% 提升至 70.43%），以展示统一配置的泛化能力。LifeTopoDict 应继承此哲学——**主表使用跨数据集统一配置**，附录报告 per-dataset 调参的潜力上限。

**HC-SOINN 已验证的 Sweet Spot（全部跨 3 数据集统一配置）：**

| 参数 | 最优值 | 容忍范围 | 跨数据集一致性 | LifeTopoDict 继承方式 |
|------|--------|---------|--------------|---------------------|
| alpha (global-local balance) | 0.5 | [0.3, 0.7] | ✅ 3/3 数据集最优 | dual-view inference 直接继承，保持 0.5 |
| K_init (初始聚类数) | 60 | [40, 100] → 边际递减 | ✅ CUB-200 验证；<30 退化 | SOINN 聚类初始化使用 K_init=60 |
| age_max (边老化) | 20 | [10, 50] 高度容忍 | ✅ 跨数据集几乎无影响 | 保持 20 |
| T_soinn (更新间隔) | 1 | [1, 5] 几乎无影响 | ✅ 无需调整 | 保持 1（每任务单次精炼） |
| lambda (STAR decay) | 0.999 | [0.99, 0.9999] 稳定 | ✅ 跨 CIFAR-100, CUB-200 | 保持 0.999（若启用 STAR） |

**HC-SOINN 的分类器对比洞察（Table 2——对 LifeTopoDict 的定位参考）：**

| 分类器 | CIFAR-100 A_Last | CUB-200 A_Last | ImageNet-R A_Last | 关键特点 |
|--------|-----------------|----------------|------------------|---------|
| NCM | 86.21 | 84.18 | 68.68 | 单原型——细粒度下表达力不足 |
| KAC (KAN-based) | 87.39 | 76.76 | 70.73 | **不稳定**——CUB-200 上远低于 NCM (76.76 vs 84.18) |
| HC-SOINN | 87.56 | 85.75 | 70.67 | 多节点——所有场景鲁棒 |
| HC-SOINN + STAR | **89.67** | **86.17** | **71.85** | 漂移适应增强 |

**核心洞察**：KAC 虽在部分设置下表现亮眼（CIFAR-100 A_Avg 92.34%），但极不稳定——在 DualPrompt+CUB-200 上 A_Last (76.76%) 远低于 NCM (84.18%)。这论证了**非参数化拓扑方法优于参数化非线性分类器**。LifeTopoDict 的非参数化 shared dictionary + SOINN 框架继承了 HC-SOINN 的鲁棒性优势。

**HC-SOINN 的 Rehearsal 对比（Table 7——存储效率评估）：**

| 方法 | Exemplars/类 | CIFAR-100 A_Avg | ImageNet-R A_Avg | 存储方式 |
|------|-------------|----------------|-----------------|---------|
| iCaRL | 20 | 82.46 | 72.42 | 原始图像 + 梯度重训练 |
| FOSTER | 20 | 89.87 | 81.34 | 原始图像 + 梯度重训练 |
| SEMA (Base) | **0** | 92.56 | 80.51 | 无存储（仅模型参数） |
| SEMA + HC-SOINN | **~18.5** | **94.25** | **81.70** | 特征锚点（无梯度重训练） |

LifeTopoDict 定位在"更低存储的 HC-SOINN"——共享字典将 ~18.5 anchors/class 压缩为 M_t 共享 atoms + k-sparse codes。应在此表基础上添加 LifeTopoDict 行（同 SEMA backbone，对比存储量和准确率）。

**LifeTopoDict 独有的关键参数消融：**

| 参数 | 建议消融范围 | 替代关系 |
|------|-------------|---------|
| dict_sparse_k | {3, 5, 8, 12} | 替代 HC-SOINN 的 per-class 节点数（18-21 -> k per node） |
| dict_ridge_lambda | {0.01, 0.05, 0.1, 0.5} | 无 HC-SOINN 对应——Top-K Ridge 独有 |
| lifecycle_theta_support | {0.3, 0.5, 0.7} | 无 HC-SOINN 对应——生命周期独有 |

**dict_sparse_k 的消融逻辑：**

- k=3: 最激进压缩（3 atoms/node），可能欠表达
- k=5: 默认值，在表达力与压缩率间平衡
- k=8: 更丰富表达，接近 HC-SOINN 的节点内多样性
- k=12: 可能过参数化，引入冗余 noise

**温度 tau 的数据集依赖性：**

| 数据集 | 建议 tau | 理由 | 依据 |
|--------|---------|------|------|
| CIFAR-100 | 0.05 (默认) | DPR 主线 | LifeTopoDict 设计 |
| CUB-200 | 0.05-0.1 | 细粒度场景低温度可能有益 | MiN: tau 不敏感 (<0.4pp) (D7) |
| ImageNet-R | 0.05-0.1 | 域偏移下低温度可能有害 | 域偏移特征形成独立簇 (DisCo, X10) |

tau 越小 softmax 分布越尖锐——在细粒度场景可能有益（增大类间区分度），在域偏移场景可能有害（过拟合到源域特征）。

### 4.3 Old-New Tradeoff 平衡策略

> LifeTopoDict 的 old-new 平衡比 LifeGAD 更简单：无 selective rumination、无 state-aware 连续梯度。

**LifeTopoDict 的 old-new 平衡机制：**

| 机制 | 偏向 | 强度控制 | 消融验证 |
|------|------|---------|---------|
| Protected atoms (lifecycle_theta_support) | 稳定性(旧类) | lifecycle_theta_support | ablation_no_lifecycle (all protected) |
| Residual growth (dict_theta_residual) | 可塑性(新类) | dict_theta_residual + dict_max_growth_per_task | ablation_no_growth |
| HC-SOINN dual-view inference | 平衡(alpha) | alpha = 0.5 | alpha 消融 {0.3, 0.5, 0.7} |

**关键阈值对 old-new 偏向的影响：**

| 调高参数 | 效果 | 风险 |
|---------|------|------|
| lifecycle_theta_support | 更多 atoms 被保护 | 新类缺乏可适应 atoms (类似 DIA lambda>0.3 阻碍新类 E8) |
| dict_theta_residual | 更难触发 growth | 新类 cover 不足 |
| dict_sparse_k | 新类 sparse 表达力增强 | 更多 atoms 参与 update 可能干扰旧类 |
| lifecycle_T_inactive | 更少 atoms 被标记 inactive | inactive atoms 积累浪费内存 |

**GPA-style 分解验证：**

GPA Phase 2 贡献 Acc_T +5.6pp (可塑性)，Phase 3 贡献遗忘率 -13.66pp (稳定性) (D12)。LifeTopoDict 需同样报告：

1. Protected atoms (稳定性贡献): delta Old Acc / delta Forgetting
2. Residual growth (可塑性贡献): delta New Acc / delta Overall Acc
3. 消融对比：no_lifecycle (all protected) 的 old-new HM vs 完整 LifeTopoDict 的 old-new HM

**HC-SOINN 的 old-new 模式（LifeTopoDict 继承）：**

| 数据集 | HC-SOINN 增益 | 序列特征 | 对 LifeTopoDict 的预期 |
|--------|--------------|---------|----------------------|
| CUB-200 | +21.00pp A_Last | 长细粒度序列 (20×10) | **最大增益场景**——共享字典的跨类表示对细粒度区分最有效 |
| CIFAR-100 | +0.53pp | 短通用序列 (10×10) | 增益最小——通用分类下单 prototype 已足够 |
| ImageNet-R (40×5) | 中等 | 域偏移长序列 | 需验证域偏移下字典原子的跨域泛化 |

**已知风险：HC-SOINN 在不同 backbone 上的效果不均。** EASE backbone 上 CUB-200 出现 -0.08% 负增益，CL-LoRA 仅 +0.13% 微小增益。LifeTopoDict 必须在多 backbone 上测试（frozen ViT-B/16-IN21K + frozen ViT-B/16-IN1K + 至少一个 adapter 变体）。

**GPA 式消融建议（验证 HC-SOINN 论文 Table 5 结构）：**

| 组件组合 | 预期贡献方向 | HC-SOINN 对应消融 | CUB-200 A_Last (ref) |
|---------|-------------|------------------|---------------------|
| NCM baseline (单 prototype) | 基础 | Baseline (FC/NCM) | ~70% |
| + HC (层次聚类多节点) | 拓扑初始化 | Pure HC | 83.80 |
| + SOINN (球面更新精炼) | 拓扑精炼 | HC + SOINN | 85.75 |
| + Shared Dict (LifeTopoDict) | 跨类共享压缩 | LifeTopoDict 独有 | **目标: ~85%+ 但 < 1/10 内存** |

**HC-SOINN 的已知局限——对 LifeTopoDict 的启示：**

| HC-SOINN 局限 | 对 LifeTopoDict 的影响 | LifeTopoDict 的应对/机会 |
|--------------|----------------------|------------------------|
| **边连接信息未用于推理**：E_c 拓扑边仅在训练时构建，分类时仅用节点距离 | 同样存在——sparse coding 不利用 SOINN 边的信息 | 未来方向：edge-aware scoring（README 中列为 P1）可利用边的流形连通性提升细粒度判别 |
| **非均匀有效性**：EASE+CUB-200 -0.08%，CL-LoRA +0.13% | shared dictionary 的跨类共享可能放大或缩小此效应 | 需跨 backbone 验证——若 shared atoms 提供更稳定基底，可能缓解此问题 |
| **KAC 在某些设置下仍有竞争力**：CODA-Prompt+CIFAR-100 KAC A_Avg 92.34% vs HC-SOINN 91.62% | KAC 的不稳定性意味着它不能作为鲁棒 baseline | 仅需在 CIFAR-100 附表列出 KAC 对比 |
| **FC 分类头融合实验（Table 8）**：w=0.3 融合 FC + HC-SOINN 略微提升 (+0.51pp CIFAR-100)，但 w=0.5 在 CUB-200 上反向 (-0.93pp) | dual-view inference (alpha=0.5) 已经是最鲁棒的默认选择 | 不推荐融合外部分类器——HC-SOINN 的 standalone 鲁棒性已充分验证 |
| **论文内部不一致**：CUB-200 划分正文 (20×10) vs 附录 (10×20) 矛盾 | 复现时需明确使用 20×10 协议（正文表格标注） | 所有实验记录中注明使用 20×10 协议 |

---

## 五、已知陷阱与预防

### 5.1 论文中报告的失败案例

| 失败模式 | 具体表现 | 根因 | 预防措施 | 来源 |
|---------|---------|------|---------|------|
| **高斯采样生成旧类伪特征失败** | 生成特征与实际特征形成两个分离簇；PFR 生成的与实际特征高度重合。A_bar 低 1.89% | 单高斯假设无法捕捉类内分布 | LifeTopoDict 不使用伪特征生成——SOINN 节点来自真实样本聚类 | DIA (E1) |
| **仅用噪声均值 (无 sigma) 效果骤降** | 低于 Baseline (无噪声)；sigma 是效果主要贡献者 | 类内方差/多模态信息比均值方向更关键 | SOINN multi-node 自然捕捉类内多模态 | MiN (E3) |
| **可训练参数全放开导致性能崩塌** | CL-LoRA CIFAR l=12: 固定 84.0 vs 可训练 71.0 (差 13pp) | 冻结底层结构稳定 > 全部可训练 | protected atoms 冻结 + SOINN 节点 SLERP 限制更新幅度 | CL-LoRA (E4) |
| **仅用当前任务噪声无历史累积** | 效果显著下降 | 历史有益信息被丢弃 | 共享字典天然跨任务累积 | MiN (E5) |
| **全解冻微调在少样本下严重破坏旧知识** | Avg 82.08% -> 77.33%, PD 19.99% -> 29.81% | 少样本下全参数更新无约束 | frozen-feature 主协议天然避免 | Lark (E6) |
| **长序列 Full Fine-Tuning 完全崩溃** | SD-LoRA N=5->10->20: FFT 64.92->60.57->49.95 | 无保护的全模型更新在长序列中累积破坏 | protected atoms + SOINN SLERP 双重约束 | SD-LoRA (E7) |
| **欧氏距离数值不稳定** | ImageNet-R A_bar: 角度 85.61% > 余弦 85.16% > 欧氏 81.73% | 特征空间非各向同性 | LifeTopoDict 所有几何操作使用余弦相似度，正确 (F13) | DIA (E2) |
| **KD 蒸馏单独使用不足** | MoAL: ImageNet-R ~82%->~64%；CL-LoRA: KD alone only +1.16~+2.63pp over baseline | KD 单独无法提供充分的稳定性约束 | LifeTopoDict 不使用 KD——protected atoms 提供结构稳定性 | MoAL, CL-LoRA (B5) |
| **长尾分布下基线方法完全失效** | Long-Tail CIL (Wang 2024 CVPR) baseline 蒸馏在 rho=0.01 时仅 11.3% 准确率 | 类不平衡使 feature drift 分布高度偏斜 | LifeTopoDict 当前无长尾专项设计，如实验涉及长尾分布需补充策略 | Long-Tail CIL (Wang 2024 CVPR) (E15) |

**LifeTopoDict 特有的已知问题（来自 README）：**

| 失败模式 | 具体表现 | 根因 | 优先级 | 影响范围 |
|---------|---------|------|--------|---------|
| **Protected indices 映射 bug** | `compress()` 中 protected 索引映射错误；protected gate 在 MVP 中实际禁用 | compress 后 indices 未正确重映射 | P0 | lifecycle 核心功能失效——protected atoms 可能被覆盖 |
| **dict_atoms 归一化不一致** | base atoms 未归一化（类均值），grown atoms 归一化（残差方向）；center_raw 语义不同 | 不同来源的 atoms 采用不同归一化约定 | P1 | sparse coding 质量不一致；EffRank/cosine similarity 诊断不可靠 |
| **_compute_atom_usage 累加器不衰减** | usage 指标偏向早期 atoms；后期 grown atoms 的系统性 disadvantage | 累加器无衰减或滑动窗口 | P1 | lifecycle 决策（inactive 判定）偏向；usage-based 诊断失真 |
| **sparse_encode 逐样本循环瓶颈** | Python for-loop per sample，大规模数据下训练慢 | 未批量化/向量化 | P2 | 训练速度——CUB-200 ~12k 样本时显著 |
| **ablation 语义不精确** | ablation_no_lifecycle 实际含义是 "全部 protected"（非"无生命周期"）；ablation_no_dict 切换到不同 model class（HC-SOINN） | 命名与语义不一致 | P3 | 消融结果解读——必须注释实际语义 |
| **P1 组件未实现** | quarantined 状态、edge-aware scoring、STAR anchor confidence gate、state-aware STAR delta scaling、selective rumination 均未实现 | MVP scope 限制 | P3 | 这些是设计文档中的高级功能——不影响 MVP 实验但需在消融中标注 |

**N2L 闭式解等价性说明：** N2L 的递推岭回归闭式解与全量数据联合训练解完全等价（消除遗忘的最强理论保证）。LifeTopoDict 的 Top-K Ridge 每任务重新计算，不具有此性质——这是 LifeTopoDict 与闭式解方法的核心理论差异。Source: N2L (E19)。

**中等任务相似性反直觉陷阱：** "几乎相同但略有不同"比"完全无关"更危险——这是中等相似性造成的最严重遗忘 (Ashtekar2025, Ramasesh 2021, Evron 2022, E10)。LifeTopoDict 的 coefficient magnitude-based support 可能在中等相似性任务上误判：相似但权重错位的 atoms 可能系数高但使用方向错误，support 阈值无法区分"高使用率"与"正确使用"。

### 5.2 阈值敏感性应对

**阈值设定的三项原则：**

1. **基于 base 数据统计设定，不在测试集上调参。** 所有 lifecycle 阈值 (dict_theta_residual, lifecycle_theta_support, lifecycle_theta_usage, lifecycle_T_inactive) 应在 base task 的 V_0^{cal} 上按 percentile / MAD 确定。

2. **报告跨 class order 的阈值敏感性。** 默认 3 个 class order seeds，报告各 seed 下的 lifecycle 统计 (growth 次数, protected count, inactive count) 均值与方差。若不同 seed 间生命周期行为显著不同，说明阈值对数据顺序高度敏感 (E9)。

3. **分数据集独立阈值，不接受跨数据集常数。** 至少区分：
   - 通用分类 (CIFAR-100): 类间区分度中等
   - 细粒度 (CUB-200): 类间区分度低，residual 阈值应更宽松（细粒度下 residual 天然更高），support 阈值应更严格
   - 域偏移 (ImageNet-R): 域差异可能导致 residual 偏高但非表达不足

**LifeTopoDict 阈值初始化方案：**

| 阈值 | 初始化方式 | 说明 |
|------|-----------|------|
| dict_theta_residual | V_0^{cal} 上所有 base 类的 avg_residual 第 90 百分位 | "显著高于正常复用残差" |
| lifecycle_theta_support | 0.5 (固定初始值)；按 V_0^{cal} 的 support 分布验证合理性 | 保护阈值——均值 support 低于此则不受保护 |
| lifecycle_theta_usage | V_0^{cal} 上所有 atoms usage 的第 20 百分位 | 低使用率阈值——排除底部 20% |
| lifecycle_T_inactive | 3 (固定，由序列长度验证) | 连续低使用任务数——长序列 (20 tasks) 可能需增至 5-7 |
| dict_max_growth_per_task | 5 (默认)；按每任务新类数缩放：max(5, |Y_t|) | 若新类多，cap 需相应增大 |

### 5.3 内存公平性计算

**LifeTopoDict 完整内存 breakdown：**

| 组件 | 存储量 | 公式 |
|------|--------|------|
| Dictionary D_t | M_t × d × sizeof(float32) | M_t × d × 4 |
| Node Sparse Codes | sum_c N_c × k × sizeof(float32) | ~ C_t × N_avg × k × 4 |
| SOINN Node States | sum_c N_c × state_metadata | ~ C_t × N_avg × 2 × 4 (state + position info) |
| Class Means | C_t × d × sizeof(float32) | C_t × d × 4 |
| Node-Atom Assignment Indices | sum_c N_c × k × sizeof(int) | ~ C_t × N_avg × k × 4 (int typically 4 bytes) |
| Atom Metadata | M_t × (state + usage) × sizeof(float32) | ~ M_t × 8 |

**简化对比（近似）：**

| 方法 | 内存公式 | CUB-200 示例 (d=768, C=200, N_avg=15) |
|------|---------|--------------------------------------|
| HC-SOINN | C × N_avg × d × 4 | 200 × 15 × 768 × 4 ≈ 9.2 MB |
| LifeTopoDict | M_t × d × 4 + C × N_avg × k × 8 | M_0=100: 100×768×4 + 200×15×5×8 ≈ 0.31 + 0.12 = 0.43 MB |
| Rehearsal (20/cls) | C × 20 × d × 4 | 200 × 20 × 768 × 4 ≈ 12.3 MB |

**LifeTopoDict 随任务增长的内存变化：** base 之后 M_t 最多增至 M_0 + T × dict_max_growth_per_task。在 CUB-200 20-task 场景：M_T max ≈ 100 + 20×5 = 200 atoms。M_T max 内存 = 200×768×4 ≈ 0.61 MB —— 仍远低于 HC-SOINN 的 9.2 MB。

**关键对照：**

1. **Same-memory HC-SOINN**: 将 HC-SOINN 节点数压缩到与 LifeTopoDict 相同内存预算，对比二者在相同约束下的表现。
2. **Fixed-capacity LifeTopoDict**: 允许 growth 但 M_T 上限与 base M_0 相同（通过标记 inactive 而非增长维持），验证 lifecycle 管理 vs 简单扩容。
3. **HC-SOINN memory**: 每类 ~18-21 anchors/class，与 < 20 exemplars/class 的 rehearsal budget 相当 (E14)。

**错误指标避免：** LCE = Delta_Acc / Delta_Mem 在 Delta_Mem=0 时无定义 (E13)。改用 memory-budgeted accuracy / memory Pareto / AUC-A。

---

## 六、调参与消融策略

### 6.1 参数搜索优先级排序

> **继承 HC-SOINN 的"tuning-free"哲学：** HC-SOINN 的所有超参数使用跨数据集统一配置，不对各数据集单独调参。LifeTopoDict 主表同样使用**跨数据集统一配置**——仅有 dict_sparse_k 和 dict_ridge_lambda 因机制特殊性需要 per-dataset 消融。附录报告 per-dataset 调参的潜力上限。

**HC-SOINN 的拓扑复杂度基线（Table 6——LifeTopoDict 的压缩目标参考）：**

| 数据集 | HC-SOINN 平均节点/类 | 特征 | LifeTopoDict 的压缩目标 |
|--------|---------------------|------|----------------------|
| CIFAR-100 | 18.68 | 通用物体——类内方差中等 | k=5 时每节点 5 atoms × ~19 nodes ≈ 等效 ~5 atoms 跨类共享 |
| CUB-200 | 18.61 | 细粒度——类间相似性高 | k=5 → 更丰富的稀疏编码补偿细粒度区分 |
| ImageNet-R | 20.61 | 艺术风格多样性 → 类内方差更大 | k=8 候选——域偏移下可能需要更多表达力 |

约 20 子原型这一"sweet spot"贯穿三个数据集——LifeTopoDict 的 dict_sparse_k 在此背景下应消融 {3, 5, 8, 12}，目标是找到接近 HC-SOINN 表达力但内存大幅降低的 k 值。

按对性能影响程度从高到低：

| 优先级 | 参数 | 候选取值 | 对 LifeTopoDict 的影响 | 依据 |
|--------|------|---------|----------------------|------|
| **P0 (必调)** | dict_sparse_k | {3, 5, 8, 12} | 核心压缩率——每个节点的原子数；替代 HC-SOINN per-class 节点数 | D-FSCIL, F1 |
| **P0 (必调)** | dict_ridge_lambda | {0.01, 0.05, 0.1, 0.5} | Top-K Ridge 数值稳定性核心参数；lambda=0 完全失效 | D-FSCIL (B7) |
| **P0 (必调)** | lifecycle_theta_support | {0.3, 0.5, 0.7} | 控制 stability-plasticity 的核心参数——多少 atoms 被保护 | DIA, GPA (B6, E8) |
| **归档** | dict_theta_residual | 固定不搜索 | growth 已关闭，仅作历史诊断参数 | LifeTopoDict 历史设计 |
| **归档** | dict_max_growth_per_task | 固定 0 | growth 已关闭，后续实验不纳入 | LifeTopoDict 历史设计 |
| **P1** | lifecycle_T_inactive | {2, 3, 5, 7} | inactive 标记延迟——长序列（20 tasks）需要更长的 T | LifeTopoDict 设计 |
| **P2** | lifecycle_theta_usage | {0.005, 0.01, 0.02, 0.05} | 低使用率阈值——过高低估 inactive，过低过度清理 | LifeTopoDict 设计 |
| **P2** | SOINN alpha | {0.3, 0.5, 0.7} | global-local inference 平衡；HC-SOINN 验证 alpha=0.5 最优 | HC-SOINN |
| **P2** | SOINN K_init | {30, 60, 100} | 初始聚类数；HC-SOINN 验证 60 sweet spot，<30 退化 | HC-SOINN |
| **P3** | SOINN age_max | {10, 20, 50} | 边老化阈值；HC-SOINN 验证 [10, 50] 高度容忍 | HC-SOINN |
| **P3** | SOINN T_soinn | {1, 3, 5} | 更新间隔；HC-SOINN 验证 [1, 5] 几乎无影响 | HC-SOINN |
| **P3** | SOINN lambda (STAR) | {0.99, 0.999, 0.9999} | STAR decay rate；HC-SOINN 验证 0.999 稳定 | HC-SOINN |

**参数搜索策略：** P0 参数做完整网格搜索（3×4×3 = 36 组合）；lifecycle 补充参数用固定 P0 最优值做局部网格；P2/P3 用 HC-SOINN 验证过的默认值并报告敏感性。全参数联合搜索在实验和审查中均不可接受，必须明确搜索顺序和固定策略。growth/edge additive 参数不进入搜索。

**关键交互效应需联合报告：**
- dict_sparse_k × dict_ridge_lambda: 大 k 引入更多共线性，需要更大 lambda
- dict_theta_residual × dict_max_growth_per_task: 低阈值+高 cap 可导致爆炸式 growth（历史风险；后续固定关闭）
- lifecycle_theta_support × lifecycle_T_inactive: 高 support + 短 T 可导致过度清理

### 6.2 核心消融实验设计

**主文核心消融（LifeTopoDict 保留 ablation）：**

```
Ablation 1: ablation_no_lifecycle  — 无生命周期管理（所有 atoms = protected，即全部保护）
Ablation 2: ablation_no_dict       — 等效 HC-SOINN baseline (不同 model class，无字典共享)
```

**补充消融（验证各组件独立贡献）：**

| 消融实验 | 对比 | 验证目标 | 依据 |
|---------|------|---------|------|
| **dict_sparse_k 消融** | k = {3, 5, 8, 12} at fixed all-else | 压缩率 vs 准确率 tradeoff | F1, F3 |
| **dict_max_growth_per_task 消融** | 不再执行 | growth 已关闭 | 历史归档 |
| **growth 方向质量消融** | 不再执行 | growth 已关闭 | 历史归档 |
| **lifecycle 消融** | no_lifecycle vs full lifecycle vs usage-only lifecycle | lifecycle 三态的独立价值 | LifeTopoDict 设计 |
| **HC-SOINN 分解消融** | HC-only vs SOINN-only vs HC+SOINN (no dict) vs HC+SOINN+Dict (full) | 复现 HC-SOINN Table 5 + 验证 dict 的增量贡献 | HC-SOINN |
| **ridge_lambda 消融** | lambda = {0, 0.01, 0.05, 0.1, 0.5, 1.0} | ridge 稳定性的必要性和 sweet spot | D-FSCIL (B7) |
| **usage 累加器消融** | 原始累计 usage vs 滑动窗口 usage (last L tasks) | usage 累加器不衰减的 bias 影响 | LifeTopoDict 设计 |
| **归一化一致性消融** | 统一归一化 vs 原始混合归一化 | dict_atoms 归一化不一致的性能影响 | LifeTopoDict 设计 |

**分离 stability vs plasticity 贡献（参照 GPA 消融模板, F6）：**

| 组件组合 | 预期贡献方向 | 度量 |
|---------|-------------|------|
| 无 protected atoms (no_lifecycle = all protected) | 最大稳定性，最小可塑性 | Old Acc, Forgetting |
| 无 growth (no_growth) | 最大可塑性受限 | New Acc, Overall Acc |
| 完整 LifeTopoDict | 平衡 | Old-New HM |
| 完整 LifeTopoDict + dict 共享 (vs HC-SOINN no dict) | 压缩+共享贡献 | Memory-budgeted accuracy |

**HC-SOINN Table 5 式分解消融（强推荐）：**

证明 LifeTopoDict 继承了 HC-SOINN 的组件结构并增加了 dict 共享压缩：

| 配置 | 组件 | 预期 A_Last (CUB-200) | 内存 |
|------|------|----------------------|------|
| NCM baseline | 单 prototype per class | ~65% | C × d |
| HC-only | 层次聚类多节点 | ~75% | C × N_avg × d |
| SOINN-only | SOINN 拓扑更新 | ~78% | C × N_avg × d |
| HC + SOINN | 完整 HC-SOINN (no dict) | ~86% (ref) | C × N_avg × d |
| HC + SOINN + Dict (LifeTopoDict) | 共享字典编码 | **目标: ~85%+ 但内存 < 1/10** | M_t × d + codes |

### 6.3 论文验证过的有效配置

**已验证的有效设计选择：**

| 设计选择 | 有效性证据 | 对 LifeTopoDict 的启示 |
|---------|-----------|----------------------|
| **余弦相似度优于欧氏距离** | DIA: 角度 85.61% > 欧氏 81.73% (ImageNet-R) (F13) | LifeTopoDict 全程余弦是正确的 |
| **Prototype/解析分类器逐步取代端到端线性分类器** | 6 篇 PTM-CIL 中 4 篇采用非梯度/半非梯度分类器 (F10) | LifeTopoDict 的 dual-view inference (alpha-balanced) 符合趋势 |
| **"任务特定参数隔离"是跨方法的共同设计原则** | EASE/CL-LoRA/SD-LoRA/MiN 均隔离不同任务参数 (F11) | LifeTopoDict 的 protected atoms 是"共享坐标系中的任务隔离" |
| **仅编辑 V_Value 矩阵对少样本最关键** | Q/K 改变显著干扰旧类注意力 (Lark, F12) | LifeTopoDict 的 head-only 更新不会干扰 backbone attention |
| **恒等初始化 + 单层线性最优** | FCS: 恒等映射保证训练初期稳定 (F14) | Feature-space transport T_t 应 affine/Procrustes, 多层 MLP 有害 |
| **动量插值系数必须接近 1.0** | MoAL alpha=0.999 最优，alpha=0.99 降约 8pp (F2) | LifeTopoDict SOINN 的 SLERP 隐式提供球面插值等价动量 |

**HC-SOINN 验证的 Sweet Spot 汇总：**

| 参数 | 最优值 | 容忍范围 | 验证数据集 |
|------|--------|---------|-----------|
| alpha (global-local balance) | 0.5 | [0.3, 0.7] | CIFAR-100, CUB-200, ImageNet-R |
| K_init (初始聚类数) | 60 | [40, 100] | CUB-200 |
| age_max (边老化) | 20 | [10, 50] | CUB-200 |
| T_soinn (更新间隔) | 1 | [1, 5] | CIFAR-100 |
| lambda / STAR (decay) | 0.999 | [0.99, 0.9999] | CIFAR-100, CUB-200 |

**D-FSCIL 验证的字典正则组合：**

| 组件 | 参数 | 效果 |
|------|------|------|
| DDL (字典学习) | alpha=10 正则 | 基类字典训练核心 |
| PC (伪类预留) | 20-60 个伪类 | 性能仅波动 0.5% (C15) |
| DA (字典适配) | alpha×\|\|M-M_0\|\|_F^2 | 防止增量字典远离 base |

**其他已知 Sweet Spot：**

- Adapter/LoRA 秩: 8-16 (EASE r=16, CL-LoRA r=10, SD-LoRA r=10, DIA r=8) (F1)
- 稀疏正则化在非欠正则化区域稳定: alpha >= 0.03 时 MCC ~0.81-0.83 平稳 (F4)
- MoAL 动量: alpha=0.999 最优 (>0.995)，alpha=0.99 降约 8pp (F2)

---

## 附录 A: LifeTopoDict 关键风险检查清单

基于 LifeTopoDict 架构特性和已知问题，实验设计必须验证以下 10 个关键风险点：

**风险 1: Protected gate 因 indices 映射 bug 实际禁用。**【已知 bug / MVP 限制】

`compress()` 中 protected 索引映射错误，导致 protected atoms 可能被覆盖。对策：
- [ ] 在实验前修复此 bug 或在所有结果中明确标注"protected gate 未生效"
- [ ] 通过检查 protected atoms 在 training 前后的 cosine similarity 变化来验证 gate 是否实际工作
- [ ] 若 gate 未修复：所有 lifecycle 结论需限定为"usage-based inactive 判定仅部分工作"

**风险 2: dict_atoms 归一化不一致导致 sparse coding 质量下降。**【已知 bug / 设计缺陷】

base atoms（未归一化类均值）vs grown atoms（归一化残差方向）的 scale 差异导致 Top-K Ridge 系数不可比。对策：
- [ ] 实验前检查所有 atoms 的 L2 norm 分布直方图
- [ ] 对比"统一归一化"vs"原始混合"的性能差异
- [ ] cosine similarity、EffRank 等指标在归一化不一致时不可跨来源比较

**风险 3: Usage 累加器偏差扭曲生命周期决策。**【已知 bug / 设计缺陷】

`_compute_atom_usage` 从不衰减 -> early atoms 系统性优势 -> lifecycle 偏向保护 base atoms。对策：
- [ ] 同时报告原始累计 usage 和滑动窗口 usage（最近 L 个任务）
- [ ] 对比二者差异——若排序差异 > 30%，累加器偏差显著
- [ ] 建议修复：添加 EMA 衰减或滑动窗口

**风险 4: Growth 边际收益递减。**【多篇验证】(X9, C1-C2, F15)

D-FSCIL m=90 < m=70；CLOVER 5+ experts 下降；SAE 24k+ 饱和。对策：
- [ ] 通过 EffRank 曲线和 GTE 回测 growth 的长期有效性
- [ ] 当 EffRank 停滞时即使 avg_residual > dict_theta_residual 也不 growth
- [ ] 设定 dict_max_growth_per_task 硬 cap

**风险 5: 单 prototype 在细粒度场景的表达力天花板。**【单篇报告】(D2, D16-D17)

CUB-200 上单 prototype NCM ~65% vs HC-SOINN ~86%。HC-SOINN 的多节点拓扑是核心贡献——LifeTopoDict 的共享字典旨在用更低内存复现此增益。对策：
- [ ] CUB-200 是核心测试平台——必须在此数据集上充分消融
- [ ] 若 LifeTopoDict 在 CUB-200 上显著低于 HC-SOINN（如 < 80%），说明共享字典压缩过度
- [ ] 预案：增加 dict_sparse_k 或降低压缩率

**风险 6: HC-SOINN 在不同 backbone 上效果不均。**【单篇报告】

EASE backbone CUB-200 -0.08%，CL-LoRA +0.13%（微小）。LifeTopoDict 作为 HC-SOINN 的插件式改进，同样面临此风险。对策：
- [ ] 至少在 2 个 backbone 上测试：frozen ViT-B/16-IN21K + frozen ViT-B/16-IN1K
- [ ] 若某 backbone 上增益为负，分析是否 backbone 特征空间不适合 shared dictionary

**风险 7: sparse_encode 逐样本循环的性能瓶颈。**【已知 bug / 工程限制】

CUB-200 ~12k 样本下 per-sample Python for-loop 可能显著拖慢训练。对策：
- [ ] 报告 per-task sparse_encode wall-clock time
- [ ] 比较总训练时间中 sparse_encode 占比
- [ ] 若 > 30% 训练时间：优先向量化优化

**风险 8: 系数幅度 old-support 在少样本下不可靠。**【理论推测】

每类仅 1-2 个 SOINN 节点时，系数均值估计不可靠（类似 PMI 少样本退化）。对策：
- [ ] 报告每类平均节点数——若某些类节点过少，support 判定不可靠
- [ ] 设定最小节点数 n_min_nodes（如 3），低于此的类不参与 support 判定
- [ ] 当某类节点不足时，其 support 判定退回 usage-only

**风险 9: Ablation 语义不精确。**【已知设计问题】

`ablation_no_lifecycle` 实际含义是 "全部 protected"（非"无生命周期"）；`ablation_no_dict` 切换到不同 model class。对策：
- [ ] 所有消融报告中明确注释实际语义
- [ ] 建议改名：`ablation_all_protected` 和 `ablation_hc_soinn_baseline`

**风险 10: 内存比较公平性。**【多篇验证】(E12-E13, A1)

LifeTopoDict 的"低内存"主张必须跟 HC-SOINN 和 rehearsal baseline 公平对比。对策：
- [ ] 提供完整 memory breakdown 表格（含所有组件）
- [ ] Same-memory HC-SOINN 对照（压缩节点数到与 LifeTopoDict 相同预算）
- [ ] Memory Pareto / AUC-A 作为 memory efficiency 指标
- [ ] 与 HC-SOINN (18-21 anchors/class) 和 rehearsal (20 exemplars/class) 的量级对比

---

## 附录 B: 证据来源索引

### 核心 CIL 论文

| 缩写 | 全称 | 关键发现 | 主要引用节点 |
|------|------|---------|-------------|
| **HC-SOINN** | **Hierarchical Clustering SOINN for CIL** | **CUB-200 +21.00pp、Procrustes 漂移 0.35-0.40、多子原型 SLERP、alpha=0.5 optimal、K_init=60 sweet spot** | **D2, D16-D18, E14; LifeTopoDict 基础架构** |
| D-FSCIL | Dictionary Learning for Few-Shot Class-Incremental Learning | 字典正则 alpha、容量消融 m=10-90、lambda=0 崩溃 | X1, X3, X4, X9, B7, C1-C2, C15, F5, F15 |
| SUE | Sparse Uncertainty Estimation | PMI 可靠性条件、校准集设计、硬阈值二值化局限 | X2, C7-C12 |
| EASE | Expandable Subspace Ensemble | task adapter 隔离、prototype completion、r=16 | A13, F11, F13 |
| CL-LoRA | Continual LoRA | 可训练 Bs 退化、tau=2、task-specific LoRA | X1, B1, E4, F11 |
| SD-LoRA | State-Driven LoRA | 方向冻结+幅度重标定、FFT 崩溃 | A15, B1, E7, F11 |
| DIA | Distribution-based Instance Augmentation | 高斯采样失败、欧氏 vs 角度对比、PDL lambda 甜蜜区 | E1, E2, E8, F13, F18 |
| MoAL | Mixture of Adapters for Lifelong Learning | 动量插值 alpha=0.999、选择性反刍、仅 KD 崩塌 | X7, B5, D14, F2, F11 |
| MiN | Multi-Noise | tau 不敏感、mu 无 sigma 效果骤降、Pi-Noise | A15, D7, E3, E5, F11 |
| GPA | Geometry-Aware Prototype Anchoring | Phase 1-3 分工、超球面投影、anchor lambda=0.15 | X6, X7, D5-D6, D12, F6 |
| FCS | Feature Calibration Scheme | FCN 增益随阶段增大、恒等初始化+单层最优 | X6, B16, D1, D3-D4, F14 |
| SPB/SimpleCIL | Simple Continual Instance Learning | RAW 动态平衡消融 -24.5pp | X7, D11 |
| PASS++ | Prototype Augmentation and Self-Supervision | 双偏置必须同时解决、难度感知双参考点 +6.23pp | D13, D15, D19 |
| Lark | Low-Rank Adaptation | Rank-1 少样本最优、V-only 编辑 | X4, X8, E6, E18, F12 |
| N2L | No-Forgetting Learning | 递推岭回归闭式解等价联合训练 | X4, E11, E19 |
| DisCo | Disentangled Continual Learning | 域偏移引入 CIL 降低遗忘、B50-5 遗忘最小 | X2, X10, E17 |
| Long-Tail CIL | Long-Tailed Class-Incremental Learning | 子原型空间+特征重采样 +24.3pp、baseline 11.3% | X8, E15-E16, F17 |
| TPL | Test-time Prototype Learning | OOD-CIL 线性关系 r=0.980、多顺序报告 | A4, A10, D7 |
| CGDL | Concept-Guided Dictionary Learning | K=2 最优概念袋、K 增大 CLIPScore 下降 | F3, F15 |
| DPW | Decomposed Prototype Weighting | Transfer/Avg/Last 三者 Mean 排名 | A12 |

### 字典学习 / SAE 论文

| 缩写 | 全称 | 关键发现 | 主要引用节点 |
|------|------|---------|-------------|
| SAE (Shabalin) | FLUX.1 Sparse Autoencoders | death rate >99% -> <30% (PCA 白化)、残差流各向异性 | X1, B9-B10 |
| SAE (Braun) | End-to-End SAE | 存活元素-梯度范数负相关、扩展因子 ~60x 饱和 | B4, B11, C3, F9, F19 |
| SAE (Fel/Archetypal) | Archetypal SAE | K-Means Stability 0.542->0.927、随机初始化 50% 概念变化 | X1, B12-B14 |
| SAE (Bhalla/Temporal) | Temporal SAE | BatchTopK 62% -> Temporal 34% death rate | B9 |
| ICFL (Donhauser) | Interpretable Concept Feature Learning | 余弦相似度 >0.9 随机重置、Selectivity Score >0.5 | X1, C6, C17 |
| SDL Theory (Tang) | Sparse Dictionary Learning Theory | Feature Anchoring M_GT 0%->24.13%、双凸性定理 | B15, B17, C19 |
| p-Annealing (Karvonen) | p-Annealing for SAE | Coverage 与 Gated SAE 相当、gamma~1.0、速度快 50% | X4, B3, D8 |
| Sparse Prior Survey (Peng) | Dictionary Learning with Sparse Priors | Joint > Alternating、非凸 ADMM 发散、APG 最优 | B18-B20, D9 |
| Diverse DL (Zheng) | Diverse Dictionary Learning | alpha >= 0.03 时 MCC 平稳 | F4 |
| Transfer DL Survey (Li) | Transfer Dictionary Learning | 跨域字典迁移性能下降、迁移方向不对称 | X10 |
| MASA (Zhussip) | Multi-Head Sparse Attention | 独立字典 > 共享、O 投影压缩敏感 | C4-C5 |
| CGDL (Kadir) | Concept-Guided DL | K 消融 2->100 | F3 |

### 综述与系统论文

| 缩写 | 全称 | 关键发现 | 主要引用节点 |
|------|------|---------|-------------|
| Zhou2024CILSurvey | Continual Learning Survey | 内存公平性、预训练优势区分、评估指标 | A1, A3, A5-A6, A8 |
| Ashtekar2025CILFirstPrinciples | CIL First Principles | 内存/计算公平性、ATD、中等相似性反直觉陷阱 | A1-A2, A9, E10 |
| Lin2024TPL | Test-time Prototype Learning Survey | OOD-CIL r=0.980、多顺序、修正遗忘率 | A4, A7-A8, A10 |
| Kim2025CLVisionTechReport | CL Vision Technical Report | 计算公平性、VLM Transfer 退化 | A2, A12 |
| FSCIL Survey (Zhang) | Few-Shot CIL Survey | 基类性能决定上限、新类 mAP 1%-10% | X2, A18 |

### LifeTopoDict 设计文档 (内部)

| 引用 | 内容 |
|------|------|
| LIFE_TOPO_DICT_README.md | LifeTopoDict 完整架构设计、参数表、度量定义、已知问题列表 |
| HC-SOINN (Yi 2026) | LifeTopoDict 的基础架构：dual-view inference, SLERP, 层次聚类初始化 |

---

*本指南基于 Phase 3-A 交叉综合知识清单（88 条发现、10 个交叉洞察）和 LIFE_TOPO_DICT_README.md 设计文档编写。*
*所有推荐均标注证据强度：【多篇验证】23 条、【两篇验证】15 条、【单篇报告】42 条、【理论推测】8 条。*
