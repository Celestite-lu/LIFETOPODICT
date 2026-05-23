# Zhang et al. 2025 FSCIL 综述论文详细总结

---

## 一、基本信息

- **标题**：Few-Shot Class-Incremental Learning for Classification and Object Detection: A Survey
- **作者**：Jinghua Zhang, Li Liu, Olli Silven, Matti Pietikainen (Fellow, IEEE), Dewen Hu
- **发表信息**：IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)，2025年1月14日接收发表，2025年3月6日当前版本
- **综述类型**：Survey（系统综述）
- **覆盖范围**：主要覆盖2020年至2023年间的FSCIL研究，以计算机视觉领域的深度学习方法为核心，涵盖图像分类和目标检测两大任务
- **搜索策略**：基于主流学术会议和期刊论文，包括CVPR、ICCV、ECCV、NeurIPS、AAAI、TPAMI等，同时依托Awesome-Incremental-Learning资源库
- **与现有综述的区别**：该综述是第一篇系统全面地同时覆盖FSCIL分类和目标检测方法的综述，提供了从数据、结构、优化三个角度的分类体系，补充了现有FSL综述和IL综述的空白

---

## 二、综述范围与分类体系

### 2.1 覆盖的子方向/变体

该综述明确定义了FSCIL与相关问题的边界：

| 问题设定 | 训练数据（初始阶段） | 训练数据（后续阶段） | 测试数据 | 核心挑战 |
|---------|-------------------|-------------------|---------|---------|
| **FSCIL** | 充足基类样本 | N-way K-shot新类样本 | 所有见过的类 | 不可靠经验风险最小化 + 稳定性-可塑性困境 |
| CIL | 充足基类样本 | 充足新类样本 | 所有见过的类 | 稳定性-可塑性困境 |
| TIL | 充足基类样本 | 充足新类样本 | 所有见过的类（已知任务ID） | 跨任务共享特征识别 |
| DIL | 充足样本（领域0） | 充足样本（新领域） | 基类多领域数据 | 跨领域特征共享 |
| FSL | 基类数据 | 新类小样本 | 仅新类 | 小样本泛化 |
| gFSL | 基类数据 | 基类+新类数据 | 基类+新类 | 不遗忘基类的小样本学习 |

### 2.2 分类体系

```
FSCIL
├── FSCIC (Few-Shot Class-Incremental Classification)
│   ├── Data-based（数据层面）
│   │   ├── Data Replay → Raw Replay / Generative Replay
│   │   └── Pseudo Scenarios → Pseudo Class / Pseudo Session
│   ├── Structure-based（结构层面）
│   │   ├── Dynamic Structure → Graph-based / Other
│   │   └── Attention-based
│   └── Optimization-based（优化层面）
│       ├── Representation Learning → Metric Learning / Feature Space / Feature Fusion
│       ├── Knowledge Distillation → KD with Balanced Data / Optimized KD
│       ├── Meta Learning
│       └── Other (F2M)
└── FSCIOD (Few-Shot Class-Incremental Object Detection)
    ├── Anchor-free → CentreNet-based / FCOS-based / DETR-based
    └── Anchor-based → Mask RCNN-based
```

### 2.3 方法论大类一句话概括

| 大类 | 核心思想 |
|-----|---------|
| Data-based | 从数据角度解决样本不足/不可重用的问题，通过数据回放或伪数据构造缓解灾难性遗忘 |
| Structure-based | 利用模型结构设计或结构特性（动态调整结构、图拓扑、注意力机制）来解决FSCIL挑战 |
| Optimization-based | 通过优化问题的复杂度处理（表征学习、知识蒸馏、元学习等）来解决FSCIL的核心困难 |

### 2.4 总体框架

两种主要框架：
1. **Feature Extractor + Softmax Classifier**：feature extractor 在整个IL过程中可训练，常借助KD缓解遗忘
2. **Feature Embedding + Nearest Mean Classifier**：基类训练后固定backbone，用最近邻类别均值分类器
---

## 三、FSCIC 分类方法详细描述

### 3.A.1 Data-based Approaches（基于数据的方法）

**核心思想**：从数据层面解决FSCIL中因训练数据有限或不可重用导致的挑战。包括向后兼容的Data Replay和向前兼容的Pseudo Scenarios。

#### 3.A.1.1 Data Replay（数据回放）

##### (a) Raw Replay（原始回放）

1. **LCwoF** (Kukleva et al., ICCV 2021)：多阶段方法——CE训练backbone，KD+base-normalized CE联合监督，最后随机采样新旧类数据联合DR校准。首个FSCIL原始回放方法。

2. **FDD** (Zhu et al., PRAI 2022)：存储与每新类等量旧样本形成联合集；新旧模型分别提取特征表示，CE+KD联合损失约束新模型。在特征分布层面蒸馏，固定浅层微调深层。

##### (b) Generative Replay（生成式回放）

3. **ERDFR** (Liu et al., ECCV 2022)：无数据回放。GAN训练生成器，熵正则化不确定性约束使生成数据接近决策边界。首次证明DR在FSCIL中的有效性。

4. **Shankarampeta & Yamauchi** (ICPRAM 2021)：Wasserstein GAN + MAML，特征蒸馏+特征回放在分类器层面解决遗忘。

5. **FSIL-GAN** (Agarwal et al., ACM MM 2022)：语义投影模块约束合成特征与潜在语义向量对齐，确保多样性和可分辨性。KD保证新旧生成器间知识传递。

#### 3.A.1.2 Pseudo Scenarios（伪场景构造）

##### (a) Pseudo Class（伪类别）

6. **FACT** (Zhou et al., CVPR 2022)：Forward Compatible FSCIL框架。约束真实样本紧凑化+masked features推向伪类别，为虚拟类别预留特征空间。首次提出FSCIL中前向兼容概念。不足：需预知增量类总数。

7. **ALICE** (Peng et al., ECCV 2022)：合并两个不同基类生成伪类别+数据增强。使用angular penalty loss训练特征提取器，余弦相似度分类。将angular penalty loss引入FSCIL。与FACT区别：ALICE通过两类合并生成伪类别，使用角度惩罚损失。

8. **SAVC** (Song et al., CVPR 2023)：监督对比学习+虚拟类别初始化backbone。语义感知虚拟对比约束。miniImageNet基类81.12%、CUB-200基类81.85%（最佳）。

##### (b) Pseudo Session（伪会话）

9. **CEC** (Zhang et al., CVPR 2021)：对基类大幅旋转变换构建伪增量会话，元学习训练GAT。真实增量时GAT在原型间传递上下文信息调整关系。首个GNN+元学习的FSCIL方法，代码被广泛用作baseline。

10. **SPPR** (Zhu et al., CVPR 2021)：随机片段选择(RES)随机采样5类构造伪会话；动态关系投影(DRP)将原型映射到共享潜在空间，余弦相似度计算关系矩阵作为过渡系数优化原型。

### 3.A.2 Structure-based Approaches（基于结构的方法）

#### 3.A.2.1 Dynamic Structure（动态结构）

**(a) Graph-based（基于图）：**

11. **TOPIC** (Tao et al., CVPR 2020)：首个提出FSCIL问题设定和评估基准。神经气体网络定义无向图保持拓扑特性；竞争Hebbian学习中增长节点和边；Stability Loss抑制遗忘，Adaptability Loss减少过拟合。miniImageNet AA: 39.64%（性能较低）。

12. **CEC**：GAT+元学习实现跨会话上下文信息传递（详见3.A.1.2）。

**(b) Other Dynamic Structure：**

13. **DSN** (Yang et al., TPAMI 2022)：Dynamic Support Network。每次会话临时扩展网络节点增强表示，通过节点自激活动态压缩缓解过拟合；选择性调用旧类分布避免类间混淆。CUB-200: AA 71.02%, PD 17.65%。

14. **Ahmad et al.** (2022)：输出节点随类别数增长；旧类参数固定，新节点随机初始化仅用当前数据训练。

#### 3.A.2.2 Attention-based（基于注意力）

15. **BiDistFSCIL** (Zhao et al., CVPR 2023)：类感知双边蒸馏框架，双教师模型（基类通用知识+上一会话自适应知识），注意力聚合模块选择性融合两分支预测。

16. **MetaFSCIL** (Chi et al., CVPR 2022)：双层元学习优化+双向引导调制。

17. **SaKD** (Cheraghian et al., CVPR 2021)：语义感知KD。标签映射到词向量，多头注意力超类聚合训练，映射模型对齐图像特征与词向量。基于Transformer的注意力校正模型。首个将语义词向量融入FSCIL，为CLIP应用提供参考。

18. **LIMIT** (Zhou et al., TPAMI 2022)：伪增量任务元学习训练，Transformer校正模型通过自注意力调整旧类分类器与新类原型间的偏置。
### 3.A.3 Optimization-based Approaches（基于优化的方法）

#### 3.A.3.1 Representation Learning（表征学习）

**(a) Metric Learning（度量学习）：**

19. **FSLL+SS** (Mazumder et al., AAAI 2021)：自监督学习增强泛化；参数重要性分析仅对不重要参数更新；Triplet loss+Regularization loss+Cosine similarity loss联合。首次将自监督特征引入FSCIL。

20. **CLOM** (Zou et al., NeurIPS 2022)：首次识别类级过拟合（class-level overfitting）——大margin基类好但阻碍新类，小/负margin反之。将margin理论与NN结构结合（浅层学共同特征，深层学高级特征），分层约束损失。

**(b) Feature Space（特征空间）：**

21. **MgSvF** (Zhao et al., TPAMI 2021)：发现低频信息更有利于保留旧知识。设计快慢子空间（不同学习率），快速子空间学新知识，慢速子空间保留旧知识。miniImageNet PD 17.33%（最优之一）。

22. **C-FSCIL** (Hersche et al., CVPR 2022)：超维嵌入（hyperdimensional embedding），利用准正交性、丰富表达空间、良好语义表示。预定义分类器引导优化。

23. **NC-FSCIL** (Yang et al., ICLR 2023)：基于神经坍缩（neural collapse）理论，预分配ETF原型固定，投影层将类别分配到对应原型。

24. **WaRP** (Kim et al., ICLR 2023)：重量空间旋转（weight space rotation），旧知识压缩到关键参数。CIFAR-100 SA: 80.31%。

**(c) Feature Fusion（特征融合）：**

25. **FeSSSS** (Ahmad et al., CVPR 2022)：监督+自监督feature extractors + 高斯生成器合成回放特征 + 轻量级增量融合分类。

26. **S3C** (Kalla and Biswas, ECCV 2022)：随机分类器（stochastic classifier）+自监督学习。随机分类器权重减轻有限样本影响。

27. **RE** (Yao et al., JEI 2022)：表示增强，根据已有知识相似度加权融合原型，模仿人类认知。

28. **TEEN** (Wang et al., NeurIPS 2023)：免训练校准（training-free calibration），融合新类原型与加权基类原型。

#### 3.A.3.2 Knowledge Distillation（知识蒸馏）

**(a) KD with Balanced Data：**

29. **ERL++** (Dong et al., AAAI 2021)：关系知识蒸馏。构建样本关系图（基于角度去除冗余），样本关系损失发现类别间关系知识。

30. **Us-KD** (Cui et al., IEEE TMM 2022)：不确定性引导半监督FSCIL，不确定性引导模块选择/标注未标记数据迭代更新。

31. **Cui et al.后续** (IEEE TNNLS 2023)：类别均衡（Class Equilibrium）+ 不确定性感知蒸馏（精细化+自适应蒸馏损失权重）。

**(b) Optimized KD：**

32-33. **SaKD** / **BiDistFSCIL**（详见3.A.2.2）

#### 3.A.3.3 Meta Learning（元学习）

34. **CSR** (Zheng and Zhang, ICDMW 2021)：元学习类结构正则化器（方向向量+对齐核），确保可分辨类原型互不干扰。

35-38. **C-FSCIL / LIMIT / CEC / MetaFSCIL**（详见前文）

#### 3.A.3.4 Other Methods（其他方法）

39. **F2M** (Shi et al., NeurIPS 2021)：在基类训练中寻找损失函数的平坦局部最小值（flat local minima），增量时在该平坦区域内微调。通过对参数添加随机噪声近似平坦最小值。独特视角：在基类而非增量阶段解决遗忘。CIFAR-100 PD 20.04%（最优之一），CUB-200 SA 81.07%。
---

### 3.A.4 FSCIC 性能对比总结

#### 数据集

| 数据集 | 类别数 | 图片数 | 划分方式 |
|--------|-------|--------|---------|
| miniImageNet | 100类 | 60,000张 | 60基类+40增量类（8个session，每session 5类，5-shot） |
| CIFAR-100 | 100类 | 60,000张 | 60基类+40增量类（8个session，每session 5类，5-shot） |
| CUB-200 | 200类 | 11,788张 | 100基类+100增量类（10个session，每session 10类，10-shot） |

#### 评估指标

- **AA (Average Accuracy)**：所有会话的平均准确率，AA = 1/(B+1) * sum(Acc_i)，越高越好
- **PD (Performance Dropping)**：最后会话与初始会话的精度差，PD = Acc_B - Acc_0，越低越好

#### 关键性能数据（基于原文Table IV）

**miniImageNet：**
SA Top5: SAVC(81.12%), ALICE(80.60%), BiDistFSCIL(74.65%), TEEN(73.53%), CLOM(73.08%)
AA Top5: SAVC(67.05%), ALICE(63.99%), BiDistFSCIL(61.42%), TEEN(61.45%), FACT(60.70%)
PD Top5: MgSvF(17.33%), SFbFSCIL(19.14%), SPPR(19.53%), ERL++(20.94%), DSN(21.06%)

**CIFAR-100：**
SA Top5: WaRP(80.31%), BiDistFSCIL(79.45%), ALICE(79.00%), SAVC(78.77%), TEEN(74.92%)
AA Top5: BiDistFSCIL(66.14%), WaRP(65.82%), SAVC(63.63%), ALICE(63.21%), TEEN(63.10%)
PD Top5: F2M(20.04%), SFbFSCIL(20.31%), SPPR(20.65%), TEEN(22.28%), FACT(22.50%)

**CUB-200：**
SA Top5: SAVC(81.85%), F2M/FSIL-GAN(81.07%), DSN(80.86%), BiDistFSCIL(79.12%), CLOM(79.57%)
AA Top5: DSN(71.02%), SAVC(69.35%), F2M(69.49%), FSIL-GAN(69.26%), BiDistFSCIL(67.34%)
PD Top5: ALICE(17.30%), DSN(17.65%), MgSvF(17.96%), TEEN(18.13%), BiDistFSCIL(18.19%)

#### 从结果提炼的重要规律

1. **基类性能对整体性能影响显著**：几乎所有SA Top5的方法也获得了AA Top5。
2. **虚拟类策略优势突出**：SAVC和ALICE在基类性能上表现最佳。
3. **反遗忘策略多样化且有效**：KD（ERL++）、伪增量场景（SPPR）、动态结构（DSN）、特征优化（MgSvF）均能有效缓解遗忘。
4. **PD与AA指标解耦**：PD最优方法不一定AA最优，体现反遗忘与新类学习之间的权衡。
5. **跨数据集性能不一致**：方法在不同数据集上排名不同。
---

## 四、FSCIOD 目标检测方法详细描述

### 4.0 与FSCIC的区别

FSCIOD要求模型在持续学习新类的同时准确定位（边界框回归或分割）和分类图像中的每个个体对象。训练数据通常分为一个基类训练集和一个增量训练集（通常所有新类打包为一个增量session，而非多session）。评估方式：(1)COCO标准评估；(2)COCO到VOC跨数据集评估。评估指标：mAP，mAP50，mAR。

### 4.1 Anchor-free Frameworks（无锚框框架）

#### CentreNet-based

1. **ONCE** (Perez-Rua et al., CVPR 2020)：首个提出FSCIOD设定。CentreNet分解为固定通用特征提取器+元学习对象定位器。增量时仅需前向传播注册。5-shot: COCO整体mAP 13.70%, 新类mAP 1.00%。

2. **SS方法** (Cheng et al., TCSVT 2021)：MAML元学习+CentreNet。关键发现：限制新类性能的主因是特征提取器对基类过拟合。5-shot: COCO整体mAP 22.30%, 新类mAP 1.40%。

3. **MS方法** (Cheng et al., TCSVT 2021)：SS增强版。5-shot: COCO整体mAP 25.60%, 新类mAP 2.50%。

#### FCOS-based

4. **Sylph** (Yin et al., CVPR 2022)：FCOS类别无关检测器+多二元sigmoid分类器替换softmax（避免类别间干扰），每分类器独立参数。5-shot: COCO整体mAP 32.18%, 基类mAP 42.40%, 新类mAP 1.50%。

5. **MCH** (Feng et al., PRL 2022)：脑启发，新类出现时添加新分类分支。基于FCOS/ATSS。5-shot: COCO整体mAP 28.30%, 新类mAP 5.50%, VOC新类mAP 14.30%。

6. **BPMCH** (Feng et al., PRL 2022)：新增backbone（基类权重初始化）传递知识。5-shot: COCO整体mAP 28.60%, 新类mAP 6.40%, VOC新类mAP 16.40%。

#### DETR-based

7. **Incremental-DETR** (Dong et al., AAAI 2023)：首个DETR框架FSCIOD。两阶段：(1)基类预训练+Selective Search自监督微调；(2)固定CNN/Transformer/回归头，增量few-shot微调+KD类别特定组件。5-shot新类mAP 8.30%; **10-shot新类mAP 14.40%（所有方法最高）**。

### 4.2 Anchor-based Frameworks（基于锚框框架）

8. **iMTFA** (Ganea et al., CVPR 2021)：首个FSCIL实例分割方法。TFA+分割分支=MTFA，扩展为iMTFA（类别无关回归/掩码头+可分辨特征提取器+原型拼接）。免训练增量。5-shot: COCO整体mAP 19.62%, 新类mAP 6.07%。

9. **iFS-RCNN** (Nguyen and Todorovic, CVPR 2022)：Mask R-CNN第二阶段扩展：(1)probit贝叶斯新类分类器；(2)不确定性引导边界框预测器。**COCO 5-shot整体mAP 32.52%（最佳），新类mAP 9.91%（最佳）**。

### 4.3 FSCIOD 性能对比总结

**COCO 5-shot关键数据（原文Table V）：**

| 方法 | 类型 | 整体mAP | 基类mAP | 新类mAP |
|------|------|---------|---------|---------|
| ONCE | CentreNet | 13.70% | 17.90% | 1.00% |
| SS | CentreNet | 22.30% | 29.20% | 1.40% |
| MS | CentreNet | 25.60% | 33.30% | 2.50% |
| Sylph | FCOS | 32.18% | 42.40% | 1.50% |
| MCH | FCOS | 28.30% | 36.00% | 5.50% |
| BPMCH | FCOS | 28.60% | 36.00% | 6.40% |
| Inc-DETR | DETR | 24.90% | 30.50% | 8.30% |
| iMTFA | Mask RCNN | 19.62% | 24.13% | 6.07% |
| iFS-RCNN | Mask RCNN | **32.52%** | 40.06% | **9.91%** |

#### 关键规律

1. **整体性能远低于监督学习**：新类mAP普遍在1%-10%之间（5-shot），距实用差距巨大。
2. **类不可知策略有优势**：ONCE（免训练注册）、iMTFA（原型拼接）效率好但性能不高。
3. **Anchor-based整体更优但更复杂**：iFS-RCNN取得最佳整体性能；Sylph和MCH以简单架构取得次优性能。
4. **DETR在新类检测上有独特潜力**：Incremental-DETR在10-shot下新类mAP（14.40%）超过所有其他方法。
5. **知识传递策略重要**：BPMCH通过新增backbone传递知识的策略优于直接微调。
6. **新类检测是最主要瓶颈**：基类mAP可达30%-42%，但新类mAP通常小于10%。
---

## 五、关键发现与领域洞察

### 5.1 综述揭示的核心结论

1. **基类初始化的决定性作用**：基类session的性能几乎直接影响最终AA指标。
2. **方法论呈现三维分化**：数据、结构、优化三个维度各有代表性优秀方法，且跨维度互有交叉。
3. **无单一方法在所有维度上最优**：SA/AA最优方法和PD最优方法通常是不同的方法。
4. **虚拟/伪策略的两面性**：虚拟类合成在基类表现卓越，但预留空间假设在实际中不一定成立。
5. **FSCIOD与FSCIC的性能差距悬殊**：检测任务远低于分类任务。

### 5.2 不同方法族之间的优劣对比

| 方法族 | 优势 | 不足 |
|--------|------|------|
| Data Replay | 直接缓解遗忘；实现简单 | 原始回放受存储/隐私限制；生成回放增加复杂性 |
| Pseudo Scenarios | 前向兼容；不依赖存储旧数据 | 伪场景与真实场景差异不确定；需先验知识 |
| Dynamic Structure | 灵活适应新类；保留旧结构 | 参数/结构增长控制挑战 |
| Attention-based | 灵活特征重加权 | 通常为辅助模块非独立方案 |
| Metric Learning | 生成可分辨特征；理论扎实 | 类级过拟合（CLOM指出） |
| Feature Space | 系统化特征表示优化 | 设计依赖对空间的理解 |
| Feature Fusion | 利用多源信息增强泛化 | 额外训练开销；融合策略需精心设计 |
| KD | 成熟抗遗忘技术 | FSCIL下数据不平衡/过拟合挑战 |
| Meta Learning | 强快速适应能力 | 伪任务质量影响大；训练开销大 |
| Flat Minima (F2M) | 独特的基线阶段方案 | 近似平坦区间的质量不确定 |

### 5.3 已解决 vs 未解决的问题

**已取得的进展：**
- 多种有效反遗忘策略（KD, DR, 动态结构, 特征优化等）
- 明确了基类初始化对FSCIL的重要性
- 建立了相对完善的评估基准和数据集
- 虚拟类/伪增量场景策略在特定数据集表现突出

**仍未解决：**
- 缺少全面评估指标（AA/PD无法反映持续学习详细表现）
- 实验条件不公平（不同backbone、额外数据引入导致无法公平比较）
- 当前FSCIL设定过于理想化（Tao et al. 2020设定）
- FSCIOD新类检测性能极低（小于10% mAP）
- 隐私保护方法研究缺失
- 跨领域FSCIL研究空白
---

## 六、未来研究方向

### 6.1 弥合人机差距（Human-Machine Gap in FSCIL）

- 人脑在编码、存储、检索三阶段高效处理少量信息；当前FSCIL模型缺乏联想学习和抽象思维
- 建议：引入生物启发式智能系统性解决FSCIL；探索多模块知识存储（替代单模型存储所有知识）；研究工作记忆/长短期记忆机制；主动的知识巩固、更新和个性化管理（主动遗忘低频知识、强化困难知识、整合一致知识）

### 6.2 更实用的FSCIL设定（Practical Settings）

- **已有改进设定**：FSCIL-lb（Kalla & Biswas, ECCV 2022，减少基类样本数量要求）；FSCIL-im（不平衡增量会话）；Semi-supervised FSCIL（Cui et al. 系列工作，利用无标签数据）
- **待探索方向**：Cross-domain FSCIL（处理域变化）；FSCIL with repetition（允许类别重复出现）；Incomplete FSCIL（增量类中既有充足也有稀缺的样本）；Federated FSCIL（结合联邦学习的隐私和分布式特性）

### 6.3 知识获取与更新（Knowledge Acquisition and Update）

- **基类阶段**：增加数据多样性（数据增强、数据生成、引入无监督数据）；优化backbone学习方法；**引入基础模型（Foundation Models）**：如CLIP、SAM、GPT等预训练模型，利用自监督/半监督预训练+提示工程的泛化和迁移能力。已有初步尝试：D-Alessandro et al.对CLIP使用提示学习策略；Zhang et al.使用RETFound基础模型增强特征学习。注意实验对比的公平性。
- **增量阶段**：当前主要策略（冻结backbone+类别原型平均、保留关键参数进行新类学习、KD）均面临挑战。新类增加时原型平均可能降低性能；参数重要性评估困难；KD中如何高效选择旧样本仍需探索。

### 6.4 应用场景与安全性（Applications and Safety）

- **应用场景**：当前研究主要针对图像分类，在目标检测、NLP、唇读、遥感、机器人等领域有初步但不系统的探索。视频分析、酒店服务机器人、自动驾驶等场景对FSCIL技术需求明显。
- **隐私与安全**：数据回放方法可能引发隐私泄露风险；对抗样本和数据投毒攻击威胁值得关注；FSCIL的隐私保护研究极为有限。深度研究安全性和隐私保护对FSCIL广泛应用至关重要。

---

## 七、综述本身的贡献与局限

### 7.1 贡献

1. **系统性**：首篇系统全面地同时覆盖FSCIL分类和目标检测两大任务的综述
2. **结构化分类体系**：提出从data-based、structure-based、optimization-based三维度的FSCIC分类法，以及从anchor-free和anchor-based角度的FSCIOD分类法
3. **详细的问题背景阐述**：明确定义FSCIL问题、核心挑战（不可靠经验风险最小化和稳定性-可塑性困境，用公式化方式阐述）、与相关问题的区别（FSL, gFSL, CIL, TIL, DIL）
4. **全面覆盖**：覆盖从2020年至2023年的主要FSCIL方法，包含详细的性能对比表（Table IV和Table V）
5. **前瞻性**：提出五个主要未来方向：人机差距、实用设定、知识获取与更新、应用场景、隐私与安全
6. **理论深度**：用公式化的方式阐述了不可靠经验风险最小化（式2-7）和稳定性-可塑性困境（式8）两个核心挑战

### 7.2 局限

1. **性能对比受限**：并非所有相关方法开源，实施条件和配置（backbone网络、特征融合方式、学习范式）各异，使得公平比较困难
2. **文献覆盖时间范围**：主要覆盖2020-2023年，2024年的最新进展未被完全纳入（该综述2024年10月修订）
3. **基础模型讨论较浅**：虽然提到了Foundation Models（CLIP、SAM、GPT）作为未来方向，但只有简要提及，未能深入讨论已有方法
4. **非深度学习方法的排除**：综述范围限定为计算机视觉领域的深度FSCIL算法，排除了非深度学习方法和其他领域的相关工作
5. **FSCIOD覆盖不够深入**：目标检测部分的方法比分类部分少很多，反映出该子方向研究还处于初期
6. **理论分析不足**：主要集中在方法描述和性能总结上，各方法之间的理论联系和统一框架的分析较少
7. **评价指标体系建议不够具体**：虽然指出了现有指标的不足（AA/PD不能反映持续学习细节），但未提出具体的改进方案或新指标设计