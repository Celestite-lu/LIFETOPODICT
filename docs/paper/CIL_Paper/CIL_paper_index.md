# 类增量学习（CIL）论文索引

> 本索引由 23 篇论文的详细总结自动生成，用于快速定位参考论文。各论文的详细内容请点击对应总结文件查看。
> 生成日期：2026-05-18

---

## 主表格

| 序号 | 论文标题 | 年份 | 会议/期刊 | 核心方法/关键词 | 一句话贡献 | 总结文件 |
|:----:|----------|:----:|-----------|----------------|-----------|----------|
| 1 | Striking a Balance between Stability and Plasticity for Class-Incremental Learning | 2021 | ICCV | SPB / RAW / 倒数自适应权重 / 多视角学习 | 提出倒数自适应权重动态平衡稳定性-可塑性，结合类无关学习和多视角知识实现无回放CIL | [summary_Wu_2021_ICCV.md](summary_Wu_2021_ICCV.md) |
| 2 | Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning (EASE) | 2024 | CVPR | EASE / Adapter子空间扩展 / 语义原型补全 | 用轻量级Adapter构建任务特定子空间，以语义引导原型补全实现无exemplar的PTM-based CIL | [summary_Zhou_2024_CVPR.md](summary_Zhou_2024_CVPR.md) |
| 3 | FCS: Feature Calibration and Separation for Non-Exemplar Class Incremental Learning | 2024 | CVPR | FCS / FCN / PIC / 最优传输 | 从四维分析NECIL遗忘根源，提出最优传输驱动的原型校准网络与原型参与对比学习 | [summary_Li_2024_CVPR.md](summary_Li_2024_CVPR.md) |
| 4 | Class Incremental Learning via Likelihood Ratio Based Task Prediction (TPL) | 2024 | ICLR | TPL / 似然比任务预测 / HAT参数隔离 | 基于Neyman-Pearson引理提出似然比原则性任务ID预测，首次利用replay数据估计补集分布 | [summary_Lin2024TPL.md](summary_Lin2024TPL.md) |
| 5 | Class-Incremental Learning: A Survey | 2024 | IEEE TPAMI | CIL综述 / 七维分类法 / AUC-A/L / 内存公平比较 | 构建七维CIL方法分类法，倡导内存预算公平比较协议并提出AUC-A/L内存无关评估指标 | [summary_Zhou2024CILSurvey.md](summary_Zhou2024CILSurvey.md) |
| 6 | Class Incremental Learning from First Principles: A Review | 2025 | OpenReview | 第一性原理 / 资源约束 / ATD跨任务判别 | 从第一性原理重新审视CIL，提出用资源约束替代"数据不可访问"并强调跨任务判别的核心地位 | [summary_Ashtekar2025CILFirstPrinciples.md](summary_Ashtekar2025CILFirstPrinciples.md) |
| 7 | Long-Tail Class Incremental Learning via Independent Sub-prototype Construction | 2024 | CVPR | 子原型空间 / 回忆空间 / LT-CIL / 非回放 | 提出跨类别共享子原型空间和回忆空间双重机制，无需回放显著超越所有LT-CIL方法 | [summary_Wang_2024_CVPR.md](summary_Wang_2024_CVPR.md) |
| 8 | PASS++: A Dual Bias Reduction Framework for Non-Exemplar Class-Incremental Learning | 2025 | IEEE TPAMI | PASS++ / SST / protoAug / 双偏置削减 | 将CIL遗忘分解为表示偏置与分类器偏置，提出自监督变换与原型增强的双偏置削减框架 | [summary_Zhu2024PASSpp.md](summary_Zhu2024PASSpp.md) |
| 9 | CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning | 2025 | CVPR | CL-LoRA / 双适配器架构 / 梯度重分配 | 提出任务共享与任务特定双LoRA适配器，以早退KD和梯度重分配实现高效rehearsal-free CIL | [summary_He_2025_CVPR.md](summary_He_2025_CVPR.md) |
| 10 | SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning | 2025 | ICLR (Oral) | SD-LoRA / 幅度-方向解耦LoRA | 将LoRA的幅度和方向解耦，方向逐步累积冻结、幅度持续重标定，同时满足三个理想属性 | [summary_Wu2025SDLoRA.md](summary_Wu2025SDLoRA.md) |
| 11 | A Tiny Change, A Giant Leap: Long-Tailed CIL via Geometric Prototype Alignment (GPA) | 2025 | ICCV | GPA / 几何原型对齐 / 超球面投影 | 提出即插即用的几何原型对齐模块，通过超球面投影初始化和动态锚定解决LT-CIL分类器不对齐 | [summary_Lai_2025_ICCV.md](summary_Lai_2025_ICCV.md) |
| 12 | Lark: Low-Rank updates after knowledge localization for FSCIL | 2025 | ICCV | Lark / 知识定位 / Fisher信息 / 秩一更新 | 首次将知识定位与秩一矩阵编辑结合用于FSCIL，通过Fisher信息定位低敏感性层并最小扰动更新 | [summary_Shi_2025_ICCV.md](summary_Shi_2025_ICCV.md) |
| 13 | Dynamic Integration of Task-Specific Adapters for Class Incremental Learning (DIA) | 2025 | CVPR | DIA / TSAI / Patch级模型对齐 | 提出任务特定Adapter的token级动态集成和patch级双层次模型对齐，以极低参数实现NECIL SOTA | [summary_Li_2025_CVPR_DIA.md](summary_Li_2025_CVPR_DIA.md) |
| 14 | Knowledge Memorization and Rumination for PTM-based CIL (MoAL) | 2025 | CVPR | MoAL / 动量适配器插值 / 知识反刍 | 以动量适配器权重插值替代知识蒸馏，结合仿生知识反刍机制在不破坏解析学习前提下持续增强适应性 | [summary_Gao_2025_CVPR.md](summary_Gao_2025_CVPR.md) |
| 15 | Mixture of Noise for Pre-Trained Model-Based CIL (MiN) | 2025 | NeurIPS | MiN / Pi-Noise / 噪声混合 | 从噪声视角重新定义CIL，为每任务学习有益噪声注入中间特征以抑制跨任务混淆模式 | [summary_Jiang2025MiN.md](summary_Jiang2025MiN.md) |
| 16 | Technical Report for CVPR 2024 CLVISION Challenge (MLKD+Dynamic SSL) | 2025 | CVPR Workshop | MLKD / 多层次知识蒸馏 / 动态SSL | 提出多层次知识蒸馏和动态自监督学习，在CVPR 2024 CLVISION挑战赛CIR赛道获第2名 | [summary_Kim2025CLVisionTechReport.md](summary_Kim2025CLVisionTechReport.md) |
| 17 | CLOVER: Confusion-Driven Self-Supervised Ensemble for Non-Exemplar CIL | 2025 | NeurIPS | CLOVER / CDSSL / PWP / 混淆驱动SSL | 通过混淆驱动的自监督学习和渐进式加权预测，缓解NECIL中的表征重叠和不可靠专家干扰问题 | [summary_Hu2025CLOVER.md](summary_Hu2025CLOVER.md) |
| 18 | Make Domain Shift a Catastrophic Forgetting Alleviator in CIL (DisCo) | 2025 | arXiv | DisCo / 域偏移作为遗忘缓解器 | 发现域偏移可显著降低CIL遗忘率，提出在特征空间模拟域偏移分离效应的即插即用方法 | [summary_Chen2025DisCo.md](summary_Chen2025DisCo.md) |
| 19 | Few-Shot Class-Incremental Learning for Classification and Object Detection: A Survey | 2025 | IEEE TPAMI | FSCIL综述 / 分类+检测 / 三维方法分类 | 首篇同时覆盖FSCIL分类与目标检测的系统综述，从数据、结构、优化三维度构建分类体系 | [summary_Zhang2025FSCILSurvey.md](summary_Zhang2025FSCILSurvey.md) |
| 20 | Enhancing Continual Learning of VLMs via Dynamic Prefix Weighting (DPW) | 2026 | 预印本 | DPW / RePA / CondAct / RWM / Token级Prefix | 首次在prefix-tuning中实现token级细粒度权重分配，通过动态prefix加权解决VLM持续学习 | [summary_Jang2026DPW.md](summary_Jang2026DPW.md) |
| 21 | Naming to Learn: CIL for VLM with Unlabeled Data (N2L) | 2026 | ICLR | N2L / 无标签CIL / 解析学习 / 伪标签精炼 | 提出首个仅用无标签数据和类别名称的CIL范式，以SVD降维精炼和递推闭式解消除遗忘 | [summary_Li2026N2L.md](summary_Li2026N2L.md) |
| 22 | Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for CIL (HC-SOINN) | 2026 | ICML | HC-SOINN / STAR / 拓扑感知分类器 | 批判NCM单原型假设，提出拓扑感知层次分类器和结构-拓扑残差对齐机制实现漂移适应 | [summary_Yi2026BeyondPN.md](summary_Yi2026BeyondPN.md) |
| 23 | M2SD: Multiple Mixing Self-Distillation for Few-Shot Class-Incremental Learning | 2024 | AAAI | M2SD / 双分支虚拟类蒸馏 / 注意力增强自蒸馏 | 首次将自蒸馏引入FSCIL，mixup+CutMix双分支虚拟类蒸馏扩展特征空间，注意力增强自蒸馏提升判别力 | [summary_lin2024m2sd.md](summary_lin2024m2sd.md) |

---

## 按方法类别分组的关键词索引

### 基于预训练模型的方法 (PTM-based CIL)

- [1] SPB — 余弦分类器+倒数自适应权重，无预训练但首次系统探索无回放CIL基线
- [2] EASE — Adapter子空间扩展+语义原型补全，exemplar-free PTM-based CIL
- [9] CL-LoRA — 共享+特定双LoRA适配器架构，梯度重分配保留跨任务知识
- [10] SD-LoRA — LoRA幅度方向解耦，方向累积+幅度重标定，推理高效
- [13] DIA — Token级Adapter动态集成+Patch级双层次模型对齐
- [14] MoAL — 动量适配器权重插值+仿生知识反刍机制，解析学习兼容
- [15] MiN — Pi-Noise有益噪声注入中间特征，冻结主干仅训噪声生成器
- [20] DPW — Token级Prefix动态加权，RePA简化QK点积为仿射变换
- [21] N2L — 无标签CIL，解析学习闭式解+伪标签降维精炼
- [16] MLKD+Dynamic SSL — 多层级知识蒸馏（特征+logit Gram矩阵）+动态SSL权重

### 基于LoRA/Adapter的方法

- [9] CL-LoRA — 双适配器架构，早退KD+梯度重分配+块权重正交约束
- [10] SD-LoRA — 幅度方向解耦LoRA，逐步累积+幅度重标定
- [2] EASE — Adapter子空间独立训练+多子空间加权集成
- [13] DIA — Task-Specific Adapter token级soft routing集成
- [14] MoAL — 单一Adapter动量插值，配合解析分类头
- [12] Lark — 秩一LoRA更新+Fisher信息知识定位（FSCIL场景）

### 非样本回放方法 (Non-Exemplar / Rehearsal-Free CIL)

- [1] SPB/SPB-I/SPB-M — 倒数权重+类无关学习+多视角知识
- [2] EASE — Adapter子空间+语义原型补全
- [3] FCS — 最优传输原型校准+原型参与对比学习
- [7] 子原型空间+回忆空间 — LT-CIL场景，双空间协同
- [8] PASS++ — 自监督变换+原型增强+难度感知protoAug
- [9] CL-LoRA — 双LoRA适配器架构，完全rehearsal-free
- [10] SD-LoRA — 方向累积+幅度重标定，rehearsal-free
- [13] DIA — Token级Adapter集成+Patch级对齐
- [14] MoAL — 动量插值+知识反刍，rehearsal-free
- [15] MiN — 有益噪声注入，rehearsal-free
- [17] CLOVER — CDSSL混淆驱动SSL+渐进式加权集成
- [20] DPW — Token级Prefix动态加权
- [22] HC-SOINN — 拓扑感知分类器替代NCM
- [23] M2SD — 双分支虚拟类蒸馏+注意力增强自蒸馏，训练后丢弃辅助结构

### 基于回放的方法 (Replay-based)

- [4] TPL — 基于似然比的Task-ID预测，HAT参数隔离，需replay buffer
- [18] DisCo — 基于rehearsal方法的即插即用模块，特征空间域偏移模拟
- [21] N2L — 无回放方法但性能超越存储20样本/类的回放方法

### 长尾类增量学习 (LT-CIL)

- [7] 子原型空间+回忆空间 — 跨类别共享子原型缓解任务内不平衡
- [11] GPA — 几何原型对齐即插即用模块，超球面投影解耦幅度与方向

### 小样本类增量学习 (FSCIL)

- [12] Lark — 知识定位+秩一矩阵编辑，适用于ViT大模型
- [19] FSCIL综述 — 首篇覆盖分类+检测的系统综述
- [23] M2SD — 双分支mixup+CutMix虚拟类蒸馏+注意力增强自蒸馏，首次将自蒸馏引入FSCIL

### 特定场景与特殊设定

- [16] CLVision Challenge (CIR) — 带重复的类增量学习，多层次KD+动态SSL
- [18] DisCo — 域偏移下的CIL，发现域偏移可降低遗忘率
- [21] N2L — 无标签CIL，仅需类别名称无需标注数据
- [22] HC-SOINN — 非理想Neural Collapse下的拓扑感知分类器

### 综述/调研论文

- [5] Zhou et al., TPAMI 2024 — CIL七维分类法+内存公平比较+PyCIL工具
- [6] Ashtekar et al., 2025 — CIL第一性原理审视，资源约束框架+ATD跨任务判别
- [19] Zhang et al., TPAMI 2025 — FSCIL综述，覆盖分类与检测两任务

### 理论分析与新视角

- [6] CIL from First Principles — ATD形式化、资源约束替代数据不可访问、五大问题维度
- [22] HC-SOINN — 非理想NC下NCM最优性批判、vMF贝叶斯分类理论、Procrustes漂移分析
- [4] TPL — Neyman-Pearson引理证明似然比是Task-ID预测的UMP检验
- [11] GPA — Fisher最优性证明、收敛加速定理、泛化误差界分析
- [15] MiN — 噪声视角重新概念化CIL参数漂移

### 其他/经典方法

- [1] SPB — 2021年无回放CIL经典基线，RAW倒数权重思想影响深远
- [3] FCS — 最优传输理论在NECIL中的首次应用，FCN单层线性设计
- [8] PASS++ — 从PASS扩展的难度感知原型增强+多视图推理集成
- [17] CLOVER — 混淆驱动SSL思想，"制造混淆来增强判别"的独特视角
- [23] M2SD — 首次将自蒸馏引入FSCIL，双分支虚拟类混合蒸馏+注意力增强自蒸馏

---

## 按年份分组

### 2021

| 序号 | 论文 | 会议/期刊 | 关键词 |
|:----:|------|-----------|--------|
| 1 | SPB / SPB-I / SPB-M | ICCV | 倒数自适应权重、多视角学习、无回放CIL基线 |

### 2024

| 序号 | 论文 | 会议/期刊 | 关键词 |
|:----:|------|-----------|--------|
| 2 | EASE | CVPR | Adapter子空间扩展、语义原型补全、PTM-based CIL |
| 3 | FCS | CVPR | 最优传输原型校准、原型参与对比学习、NECIL |
| 4 | TPL | ICLR | 似然比任务预测、HAT参数隔离、原则性Task-ID预测 |
| 5 | CIL Survey (Zhou et al.) | IEEE TPAMI | CIL七维分类法、内存公平比较、AUC-A/L指标 |
| 7 | LT-CIL Sub-prototype | CVPR | 子原型空间、回忆空间、长尾CIL |
| 23 | M2SD | AAAI | 双分支虚拟类蒸馏、注意力增强自蒸馏、FSCIL |

### 2025

| 序号 | 论文 | 会议/期刊 | 关键词 |
|:----:|------|-----------|--------|
| 6 | CIL from First Principles | OpenReview | 资源约束框架、ATD、第一性原理 |
| 8 | PASS++ | IEEE TPAMI | 双偏置削减、自监督变换、原型增强 |
| 9 | CL-LoRA | CVPR | 双适配器架构、梯度重分配、LoRA CIL |
| 10 | SD-LoRA | ICLR (Oral) | 幅度方向解耦、LoRA累积、推理高效 |
| 11 | GPA | ICCV | 几何原型对齐、超球面投影、LT-CIL |
| 12 | Lark | ICCV | 知识定位、秩一更新、FSCIL、ViT |
| 13 | DIA | CVPR | Token级Adapter集成、Patch级对齐 |
| 14 | MoAL | CVPR | 动量插值、知识反刍、解析学习 |
| 15 | MiN | NeurIPS | Pi-Noise、噪声混合、噪声视角CIL |
| 16 | MLKD+Dynamic SSL | CVPR Workshop | 多层次知识蒸馏、动态SSL、CIR |
| 17 | CLOVER | NeurIPS | CDSSL混淆驱动SSL、渐进式加权集成 |
| 18 | DisCo | arXiv | 域偏移缓解遗忘、即插即用对比正则化 |
| 19 | FSCIL Survey (Zhang et al.) | IEEE TPAMI | FSCIL综述、分类+检测、三维分类法 |

### 2026

| 序号 | 论文 | 会议/期刊 | 关键词 |
|:----:|------|-----------|--------|
| 20 | DPW | 预印本 | Token级Prefix加权、RePA、VLM持续学习 |
| 21 | N2L | ICLR | 无标签CIL、解析学习、伪标签降维精炼 |
| 22 | HC-SOINN | ICML | 拓扑感知分类器、STAR漂移适应、非理想NC |
