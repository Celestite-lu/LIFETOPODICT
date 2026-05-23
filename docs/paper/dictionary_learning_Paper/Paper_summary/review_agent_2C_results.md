# Review 2-C: 字典学习理论与优化挖掘审查

## Coverage Assessment

| Paper | Covered? | Key Missing Items |
|-------|----------|-------------------|
| Peng 2024 Sparse Prior Survey | Yes | (1) 缺失像素填充案例研究（Section 8.2, Figure 16：50%像素缺失时PSNR从9.97dB恢复至26.10dB，+16.13dB提升）; (2) 综述自身提出的5个未来研究方向（Section 9）完全未提及; (3) 综述的6条局限性（如非系统性综述、作者偏向等）部分提及但未覆盖章节7.2的#2,#6点 |
| Dash 2025 Householder | Yes | (1) Fig.4 X恢复误差与字典恢复趋势相反（theta越高误差越大，因硬阈值在稠密场景更难区分零/非零值）这一反直觉发现未提及; (2) Theorem 1的两个条件（c=Omega(n^{alpha}), alpha>1/4; u_i*c有界）仅在数据结构部分提及但未在Q3中作为限制条件讨论 |
| Zheng 2026 Diverse DL | Partial | (1) 真实图像实验（Cars3D/Shapes3D/MPI3D，Table 1）完全未提及——依赖稀疏在VAE/GAN/Diffusion三种主干上一致改善解耦性能的核心实践结论缺失; (2) 与OroJAR/Hessian Penalty的对比实验（Appendix C.1/C.2）未覆盖; (3) LLM SAE死特征实验（Table 3：JSAE仅62 vs Top-K 439）未提及; (4) Latent Swapping定性可视化（Figures 9-11）未覆盖 |
| Kumar 2025 Feedback Complexity | Yes | (1) Theorem 3通用采样反馈Theta(p(p+1)/2)几乎必然下界仅简要提及，但Theorem 3与Theorem 4的构造区别（重缩放技术、Sard定理应用）未展开; (2) ChessGPT和Pythia-70M的具体反馈数量实验（Table）未在Q1或Q3中引用; (3) Algorithm 3梯度下降优化实现细节（低秩分解Phi=UU^T，lambda=1e-4）未覆盖 |
| Donhauser 2025 ICFL | Yes | (1) PCA白化作为弱监督范式的核心设计（Section 3.4/4.1白化矩阵构建细节）仅一笔带过; (2) 消融实验中注意力输出vs残差流对比（Table 4/Figure 17/18）、模型大小扩展（MAE-G 1.9B vs MAE-L 330M）未提及; (3) 细胞类型/批次/CRISPR/siRNA/功能基因组五项任务的原始表示BTA基准值（97.2%/87.8%/94.6%/51.6%/32.1%）仅在数据结构中隐含，未在评估部分完整列出 |
| Kadir CGDL | Yes | (1) SAM vs Random Crop定位消融（Table 10）未提及; (2) 层消融实验（Figure 12：CLIPScore随层深提升，BERTScore非单调）未覆盖; (3) 概念向量的泛化性实验（Figure 9：即使目标概念不在字典中也能语义对齐）和抽象概念归因（Figure 10）未覆盖; (4) 四种反馈提示模板P1-P4消融的具体数值（仅在Q5中概括为"鲁棒性"） |
| Parekh 2025 VisCoIN | Yes | (1) Omega概念翻译器架构细节（W+隐空间的14个隐向量分配策略、Phi_prime支持表示）完全未提及; (2) 输出保真度损失vs标准交叉熵损失的对比消融（Table 21）未提及; (3) 多架构兼容性验证（ProgressiveGAN/beta-VAE/ResNet101/ViT-B/16的结果）未覆盖; (4) FLINT可视化需1000次反向传播vs VisCoIN仅需1次前向传播的效率对比在Q4中提到，但作为核心方法优势可更突出 |

## Accuracy Issues

| Claim (Explore Agent Row) | Verification | Issue |
|---------------------------|-------------|-------|
| Q1 Row 1: "APG 和 AFPG 在所有 4 类稀疏编码场景（传统 SC、CSC、ML-CSC）中均达到最低目标函数值" | Partial | 源文件Section 4.4明确写道"APG 和 AFPG 尚未成功应用于 ML-CSC"，ML-CSC仅对比了PG和FPG。APG/AFPG只在3/4场景中被测试。Explore Agent在Q1 Row 8中自行承认了这一空白，但Row 1的"所有4类"表述存在自相矛盾。 |
| Q1 Row 13: "FPG 约 30 次迭代即接近收敛（目标值 7.70），PG 需 80 次以上（7.80）" | Minor | 源文件Table（Section 4.4）：20 iter时FPG=7.70，80 iter时PG=7.80。30次迭代是插值估计（20到80之间的合理近似），但7.70是20次的精确数据点而非30次。 |
| Q1 Row 11: "总计算复杂度 O(nmp) vs O(n^2 p) per iter（Procrustes）" | Minor | 该表述将Dash的非迭代总复杂度与Procrustes的单次迭代复杂度对比，维度不统一。源文件中Dash为O(nmp)总量（固定m步），Procrustes为O(n^2 max{n,p})每次迭代且需多轮迭代才收敛，实际效率差距被低估。 |
| Q3 Row 8: "RFM 特征（r=4, p=10）仅需 55 个反馈...特征分解法仅需 r(r+1)/2 + p - r = 38 个反馈" | Minor | 55个反馈来自r=4实验（Fig.1），38个反馈来自r=8实验（Appendix C.1）。两句并列呈现，给人同一实验的错觉。对于r=4，r(r+1)/2+p-r=4*5/2+10-4=16而非38。未注明不同秩数。 |
| Q2 Row 5: "L_orth...LPIPS 0.545 vs 0.556, FID 15.85 vs 9.43" | Accurate | 数值与源文件完全一致。但源文件自身的表述矛盾（L_orth对LPIPS有利但对FID不利）未被Explore Agent指出——paper中FID 15.85（含L_orth）劣于9.43（不含L_orth）。Explore Agent在这一行仅做数值复述，未分析这一定量矛盾。 |
| Q5 Row 7: "delta=20 严重损害准确率（从 79.44% 降至 ~76%）" | Accurate | 源文件：delta=0.2时Acc=79.44%，delta=20时Acc=76%（源文未写"~"但写为"76%"）。精确数值匹配，~表述可接受。 |
| Q1 Row 16: "CGDL 421s vs CoX-LMM 4375s（约 10 倍加速）" | Accurate | 源文件Table 11/Section 6.6: 421 vs 4375，4375/421≈10.4。 |
| Q3 Row 10: "Pearson 相关系数 r=0.71" | Accurate | 源文件Section 5.3明确写为"Pearson 相关系数 r = 0.71"。 |
| Q4 Row 3: "1 次前向传播即可完成可视化（vs FLINT 需 1000 次反向传播）" | Accurate | 源文件Section 5.5 Table 9注释和局限性部分有提及。 |

## Missing Critical Findings

### 1. Peng 综述的应用案例与未来方向（高优先级）

Explore Agent完全遗漏了Peng 2024的两块高价值内容：
- **缺失像素填充案例研究（Section 8.2, Figure 16）**：50%像素随机缺失时，CSC+字典学习将PSNR从9.97dB提升至26.10dB（+16.13dB），即使10%缺失也能恢复至34.75dB。这是该综述最具体的端到端应用验证，对评估稀疏字典学习的实际价值有直接参考意义。
- **5个未来研究方向（Section 9）**：(1) 开拓带有稀疏先验的新型数据模型；(2) ML-CSC/ML-CDL的非凸优化高效算法；(3) 扩展算法展开应用范围（AADMM、联合优化的展开等）；(4) 释放算法展开在无监督学习中的潜力；(5) 展开网络理论基础建设（泛化界、最优层数）。这些为后续2-3年的研究提供了直接的技术攻关路线图。

### 2. Zheng 2026 真实图像实验（高优先级）

Explore Agent仅在Q3中覆盖了合成实验（3-5维的R^2，3-10维的MCC），完全未提及图像实验结果。这个遗漏有实质性影响：
- Cars3D/Shapes3D/MPI3D三个真实图像基准上，依赖稀疏在VAE（FactorVAE）、GAN（DisCo）上一致优于latent sparsity和base模型
- 跨Diffusion主干的结果有分化（EncDiff在Shapes3D上从0.901提升至0.947，但Cars3D上略降），这一"不是在所有主干上均改善"的诚实结论对实践选择很重要
- 与OroJAR/Hessian Penalty的对比展示了依赖稀疏随维度增加而扩大优势（dim 5: 0.805 vs 0.512 vs 0.779）

### 3. Donhauser 2025 五项任务的原始基准值（中优先级）
Q3 Row 7的线性探测BTA分析仅定性描述"几乎全部信号被保留"和"明显下降"，但未列出五项具体任务的原始表示基准准确率（细胞类型97.2%、批次87.8%、CRISPR 94.6%、siRNA 51.6%、功能基因组32.1%），使得读者无法量化"信号保留"和"下降"的程度。

### 4. VisCoIN 核心架构创新（中优先级）
Omega概念翻译器的设计——特别是W+隐空间的14个隐向量分配策略（粗粒度4+中粒度4+细粒度6，Phi_primes负责前3和后2，Phi负责中间9）——是VisCoIN实现高质量可视化的关键技术细节，但Explore Agent完全未提及。这一细节对打算复现或扩展VisCoIN的读者至关重要。

### 5. Kumar 2025大规模LLM实验（中优先级）
ChessGPT SAE上稀疏构造以840万反馈达PCC=0.9773、特征分解仅需13.5万反馈达PCC=0.9427；Pythia-70M上稀疏采样从20万到2000万反馈PCC从0.02提升至0.77。这些实验是验证理论界的直接证据，但Explore Agent在Q1-Q4中完全未引用，仅在Q5稀疏采样部分一笔带过。

## Overall Verdict: 通过（有条件）

Explore Agent 2-C的产出覆盖了全部7篇论文，Q1-Q5五个问题的组织框架逻辑清晰，绝大多数数值与源文件精确吻合。主要的缺失集中在：(1) Peng综述的应用案例和未来方向；(2) Zheng 2026的真实图像实验；(3) Kumar 2025的大规模LLM验证；(4) VisCoIN Omega架构细节。这些缺失不影响结论的核心正确性，但限制了产出对实践者的完整指导价值。

## Fix Items

以下修复项按优先级排列：

**P0 (必须修复):**
1. Q1 Row 1将"所有 4 类稀疏编码场景"修正为"3/4类已测试场景（传统SC、CSC及Lagrangian形式均验证；ML-CSC中APG/AFPG尚未测试，当前仅PG/FPG可用）"，与Q1 Row 8的自述保持一致。

**P1 (强烈建议):**
2. 新增Q6段落或补充Q4，覆盖Peng 2024的应用案例（缺失像素填充PSNR恢复数据）和5个未来研究方向。
3. 补充Zheng 2026的真实图像实验表格（Cars3D/Shapes3D/MPI3D上的DCI和FactorVAE分数），至少在Q3中加入一行关于"依赖稀疏在真实图像上的Vae/GAN/Diffusion跨主干改善"的总结。
4. 补充Kumar 2025的ChessGPT和Pythia-70M大规模实验数据到Q3。

**P2 (建议):**
5. Q3 Row 8修正：将"r=4...38个反馈"拆分为两个独立实验陈述，明确标注r=8的38反馈界来自Appendix C.1。
6. Q2 Row 5补充说明L_orth对FID不利的量化矛盾（含L_orth时FID 15.85 vs 不含9.43），避免读者误以为L_orth在所有维度上都有改善。
7. 补充VisCoIN Omega架构中W+隐向量分配策略（一行即可），确保读者理解1次前向传播实现可视化的技术前提。
8. Q3 Row 7补充五项线性探测任务的原始表示基准BTA值。
