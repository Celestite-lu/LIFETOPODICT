# SD-LORA: SCALABLE DECOUPLED LOW-RANK ADAP-TATION FOR CLASS INCREMENTAL LEARNING

Yichen Wu1,2∗, Hongming Piao1∗, Long-Kai Huang4†, Renzhen Wang3, Wanhua Li2, Hanspeter Pfister2, Deyu Meng3,6, Kede Ma1†, Ying Wei5†

1City University of Hong Kong, 2Harvard University, 3Xi’an Jiaotong University,

4Tencent AI Lab, 5Zhejiang University, 6Pengcheng Laboratory

# ABSTRACT

Continual Learning (CL) with foundation models has recently emerged as a promising paradigm to exploit abundant knowledge acquired during pre-training for tackling sequential tasks. However, existing prompt-based and Low-Rank Adaptationbased (LoRA-based) methods often require expanding a prompt/LoRA pool or retaining samples of previous tasks, which poses significant scalability challenges as the number of tasks grows. To address these limitations, we propose Scalable Decoupled LoRA (SD-LoRA) for class incremental learning, which continually separates the learning of the magnitude and direction of LoRA components without rehearsal. Our empirical and theoretical analysis reveals that SD-LoRA tends to follow a low-loss trajectory and converges to an overlapping low-loss region for all learned tasks, resulting in an excellent stability-plasticity trade-off. Building upon these insights, we introduce two variants of SD-LoRA with further improved parameter efficiency. All parameters of SD-LoRAs can be end-to-end optimized for CL objectives. Meanwhile, they support efficient inference by allowing direct evaluation with the finally trained model, obviating the need for component selection. Extensive experiments across multiple CL benchmarks and foundation models consistently validate the effectiveness of SD-LoRA. The code is available at https://github.com/WuYichen-97/SD-Lora-CL.

# 1 INTRODUCTION

Continual Learning (CL, Rolnick et al. 2019,Wang et al. 2024b,Zhou et al. 2024,Wang et al. 2022b) aims to develop computational learning systems capable of continually adapting to evolving environments while retaining previously acquired knowledge. In contrast to standard supervised learning, which assumes that training data are independent and identically distributed (i.i.d.), CL trains models on non-stationary data where tasks are presented sequentially. This departure from the i.i.d. assumption introduces the central challenge of catastrophic forgetting (French, 1999; McClelland et al., 1995; McCloskey & Cohen, 1989; Kirkpatrick et al., 2017), indicated by a significant performance degradation of previous tasks when new tasks are introduced.

Over the last five years, foundation models—large-scale pre-trained neural networks—have proven highly effective at transferring knowledge while exhibiting strong resistance to catastrophic forgetting (Wang et al., 2022b;a; Smith et al., 2023; Huang et al., 2024; Wang et al., 2024a; Liang & Li, 2024) in the context of CL. One prominent approach centers on adapting input and intermediate representations (i.e, prompts) of foundation models to accommodate new tasks. For example, L2P (Wang et al., 2022b) and DualPrompt (Wang et al., 2022a) incrementally learn a prompt pool and selectively insert the most relevant prompts based on their match with incoming test samples. CODA-Prompt (Smith et al., 2023) refines this strategy by end-to-end optimizing the prompt selection module for CL objectives. Despite obviating manual task identifiers, these methods rely heavily on accurately identifying task-relevant prompts from a potentially growing pool, raising concerns about inference scalability.

Table 1: Comparisons of existing CL methods with foundation models in terms of three desirable properties: 1) Rehearsal-free (i.e, without memory for sample storage), 2) inference efficiency (i.e, without additional computational overhead during inference), and 3) end-to-end optimization (of all model parameters for CL objectives). 

<table><tr><td>Method</td><td>Rehearsal-free</td><td>Inference Efficiency</td><td>End-to-end Optimization</td></tr><tr><td>L2P (Wang et al., 2022b)</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>DualPrompt (Wang et al., 2022a)</td><td>√</td><td>✗</td><td>✗</td></tr><tr><td>CODA-Prompt (Smith et al., 2023)</td><td>√</td><td>✗</td><td>√</td></tr><tr><td>HiDe-Prompt (Wang et al., 2024a)</td><td>✗</td><td>✗</td><td>√</td></tr><tr><td>InfLoRA (Liang &amp; Li, 2024)</td><td>✗</td><td>√</td><td>√</td></tr><tr><td>SD-LoRA(Ours)</td><td>√</td><td>√</td><td>√</td></tr></table>

Another line of work has taken a more memory-intensive route by retaining samples of previous tasks to bolster performance. Building upon CODA-Prompt, HiDe-Prompt (Wang et al., 2024a) continues to learn prompts incrementally but stores large quantities of samples. Likewise, InfLoRA (Liang & Li, 2024) leverages Low-Rank Adaptation (LoRA) (Hu et al., 2022) to remain parameterefficient, yet similarly rehearses extensive samples during incremental LoRA optimization. This reliance on large-scale memory makes them less scalable in real-world deployments, particularly in resource-constrained or large-scale CL settings.

Table 1 outlines three desirable properties for an ideal CL method with foundation models:

• Rehearsal-free: The method should eliminate the need to store samples from previous tasks, thereby ensuring learning scalability;   
• Inference Efficiency: The method should maintain computational efficiency during inference, preferably without added computational costs, thereby ensuring inference scalability;   
• End-to-end Optimization: All method parameters should be end-to-end optimized for CL objectives, rather than through segmented and separate optimization stages, thereby maximizing CL performance.

To achieve these properties, we propose Scalable and Decoupled LoRA (SD-LoRA), which incrementally adds LoRA components by separating the magnitude and direction learning. By directly employing the finally trained model for testing—without task-specific component selection—SD-LoRA supports computationally efficient inference (Huang et al., 2024), while being rehearsal-free. Through an in-depth empirical and theoretical analysis, we show that SD-LoRA learns to follow a low-loss path that converges to an overlapping low-loss region for all learned tasks, thus achieving an excellent stability-plasticity trade-off. Meanwhile, the importance of the incrementally learned LoRA directions diminishes as CL progresses. Building upon these observations, we introduce two variants of SD-LoRA with improved parameter efficiency through rank reduction and knowledge distillation, respectively. All SD-LoRA parameters can be end-to-end optimized for CL objectives.

In summary, the principal contributions of our work include

• A CL method with foundation models—SD-LoRA, offering a rehearsal-free, inference-efficient, and end-to-end optimized solution. We additionally include two SD-LoRA variants that improve parameter efficiency;   
• An empirical and theoretical analysis of SD-LoRA, elucidating its plausible working mechanism that eliminates task-specific component selection;   
• A comprehensive experimental evaluation of SD-LoRAs, demonstrating their effectiveness across multiple CL benchmarks and foundation models.

# 2 RELATED WORK

Continual Learning (CL). CL seeks to sequentially learn from new tasks while retaining previously acquired knowledge, with the central goal of mitigating catastrophic forgetting. Broadly, existing CL methods can be categorized according to three main design philosophies: Rehearsal-based, regularization-based, and architecture-based approaches. Rehearsal-based methods (Riemer et al.,

2018; Chaudhry et al., 2019; Tiwari et al., 2022) selectively retain and replay samples from previous tasks to alleviate catastrophic forgetting. Regularization-based methods (Kirkpatrick et al., 2017; Li & Hoiem, 2017; Lee et al., 2019) introduce penalty terms into the training objective to constrain updates on parameters deemed important for learned tasks. Architecture-based methods (Mallya et al., 2018; Ebrahimi et al., 2020; Ramesh & Chaudhari, 2021) expand or adapt the model architecture to account for new tasks. By allocating additional task-specific parameters or modules, these methods prevent the overwriting of learned important weights. Among various CL settings, this paper focuses on particularly challenging and practical class-incremental learning, in which the model must perform all learned tasks with no access to task identity at test time. Conventional class-incremental learning methods often require expensive training from scratch or parameter-intensive tuning, which can lead to overfitting and interference among tasks.

CL with Foundation Models. Foundation models have recently demonstrated their effectiveness in CL by facilitating knowledge transfer across tasks and reducing catastrophic forgetting (Wang et al., 2022b;a; Smith et al., 2023; Wang et al., 2024a; Liang & Li, 2024). Specifically, methods like L2P (Wang et al., 2022b), DualPrompt (Wang et al., 2022a), and CODA-Prompt (Smith et al., 2023) integrate Vision Transformers (ViTs) with prompt-tuning strategies (Lester et al., 2021; Jia et al., 2022), thereby improving knowledge retention as new tasks are introduced. Building on these, HiDe-Prompt (Wang et al., 2024a) further stores a large number of samples to boost performance. Rather than prompt-tuning, InfLoRA (Liang & Li, 2024) adopts a LoRA-based Parameter-Efficient Fine-Tuning (PEFT) approach, which similarly necessitates substantial sample storage. Despite their promise, none of the existing CL methods with foundation models simultaneously satisfy the three desirable properties outlined in Table 1. To fill this gap, we introduce SD-LoRA, which can also be viewed as a form of model-merging techniques (Chitale et al., 2023; Ilharco et al., 2023), developed in parallel with, yet complementary to ongoing CL research.

Parameter-Efficient Fine-Tuning (PEFT). The integration of PEFT methods with CL with foundation models is essential because full fine-tuning for each individual task is prohibitive in terms of computation and storage requirements. Representative PEFT methods include adapters (Houlsby et al., 2019), which insert lightweight learnable modules into Transformer layers; prompt-tuning (Qin & Eisner, 2021; Jia et al., 2022) and prefix-tuning (Li & Liang, 2021), which introduce learnable input representations into Transformer layers; and LoRA (Hu et al., 2022), which adds and tunes low-rank branches as updates to the pre-trained weights. While these techniques have proven effective in single-task and multi-task offline learning settings, their performance boundaries in CL with foundation models remain insufficiently explored. The proposed SD-LoRA adopts a rehearsal-free, LoRA-based PEFT approach for CL with foundation models.

# 3 PROPOSED METHOD: SD-LORA

In this section, we first present the necessary preliminaries. We then present in detail the SD-LoRA method for CL with foundation models, accompanied by an empirical and theoretical analysis. Finally, we describe two SD-LoRA variants with improved parameter efficiency.

# 3.1 PRELIMINARIES

Problem Formulation. Let $\{ \mathcal { T } _ { 1 } , \mathcal { T } _ { 2 } , \ldots , \mathcal { T } _ { N } \}$ be N sequential classification tasks. The training split of $\mathcal { T } _ { t } ,$ , denoted as $\mathcal { D } _ { t } = \{ \boldsymbol { x } _ { t } ^ { ( i ) } , \boldsymbol { y } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { | \mathcal { D } _ { t } | }$ ( , comprises of $| \mathcal { D } _ { t } |$ training example pairs, where $\mathbf { \boldsymbol { x } } _ { t } ^ { ( i ) }$ representing the input image and $y _ { t } ^ { ( i ) }$ its corresponding label. We consider a classification model $f _ { \theta } ,$ , parameterized by θ. When training on $\mathcal { D } _ { t }$ , no data from previous tasks $\{ \mathcal { T } _ { k } \} _ { k = 1 } ^ { t - 1 }$ is accessible. Accordingly, the training objective is given by

$$
\ell \left(\mathcal {D} _ {t}; \boldsymbol {\theta}\right) = \frac {1}{\left| \mathcal {D} _ {t} \right|} \sum_ {i = 1} ^ {\left| \mathcal {D} _ {t} \right|} \ell \left(f _ {\boldsymbol {\theta}} \left(\boldsymbol {x} _ {t} ^ {(i)}\right), y _ {t} ^ {(i)}\right), \tag {1}
$$

where $\ell ( \cdot , \cdot )$ is a per-sample loss function such as cross-entropy. For model evaluation, we may compute the average loss of $f _ { \theta }$ across all tasks encountered so far: $\begin{array} { r } { \frac { 1 } { t } \sum _ { k = 1 } ^ { t } \ell ( \mathcal { V } _ { k } ; \pmb { \theta } ) } \end{array}$ , where $\nu _ { k }$ denotes the test split of $\mathcal { T } _ { k }$ . That is, the overarching goal is to ensure that $f _ { \theta }$ performs well on both the current task and all previous tasks. We generally follow the class-incremental learning setting described in (Wang et al., 2022b; Liang & Li, 2024).

![](images/1a8dcbbe39e850f6ce0488c69fd167413827087723d2c99988da0c2938cc6d4f.jpg)

<details>
<summary>text_image</summary>

Frozen
Trainable
Decoupled Low-rank Matrix {αi} Learnable LoRA Magnitude
W0 ∈ Rm×n
Pre-trained Weights
+
A
r
B
r
ΔW = AB ∈ Rm×n
+
A1
r1
r2
α1
α2
B1/√||A1B1||F
r1
r2
ΔW = α1A1B1 + α2A2B2
(a)
(b)
</details>

Figure 1: Illustration of the parameter update in (a) Vanilla LoRA and (b) the proposed SD-LoRA, where the current task index is $t = 2$ and $r , r _ { 1 } , r _ { 2 } \ll$ min $\{ m , n \}$ .

Low-Rank Adaptation (LoRA). As illustrated in Fig. 1(a), LoRA (Hu et al., 2022) constrains the parameter updates during fine-tuning to lie in a low-rank subspace. Concretely, let $\mathbf { W } _ { 0 } \in \mathbb { R } ^ { m \times n }$ denote the original weight matrix of a layer in the classifier $f _ { \pmb { \theta } } .$ . LoRA expresses the parameter update $\Delta \mathbf { W } \in \mathbb { R } ^ { m \times n }$ as the product of two learnable matrices $\mathbf { A } \in \mathbb { R } ^ { m \times \hat { r } }$ and B $\in \mathbb { R } ^ { r \times n }$ , i.e., $\Delta \mathbf { \bar { W } } = \mathbf { A } \mathbf { B }$ , with $r \ll$ min $\{ m , n \}$ . For a given layer of $f _ { \theta } ,$ , the LoRA-updated output is

$$
\boldsymbol {h} ^ {\prime} = \mathbf {W} _ {0} \boldsymbol {x} + \Delta \mathbf {W} \boldsymbol {x} = (\mathbf {W} _ {0} + \mathbf {A B}) \boldsymbol {x}. \tag {2}
$$

Throughout fine-tuning, the original weight matrix $\mathbf { W } _ { 0 }$ remains fixed.

# 3.2 SD-LORA

In LoRA, the parameter update ∆W can be decomposed as follows:

$$
\Delta \mathbf {W} = \| \mathbf {A B} \| _ {F} \cdot \overline {{{{\mathbf {A B}}}}} = \| \mathbf {A B} \| _ {F} \cdot \frac {\mathbf {A B}}{\| \mathbf {A B} \| _ {F}}. \tag {3}
$$

This decomposition highlights two crucial elements of the update: The magnitude $( i . e ,$ the Frobenius norm $\| \mathbf { A B } \| _ { F } )$ and direction $( i . e ,$ the normalized matrix AB). Recent work (Liu et al., 2024) has demonstrated that compared to full fine-tuning, LoRA exhibits limited flexibility in precisely adjusting these two elements. This drawback hinders its performance on complex tasks that demand fine-grained control over both magnitude and direction. Furthermore, Qiu et al. (2023) highlighted a more critical role of the direction in fine-tuning than the magnitude.

Motivated by these observations, we describe SD-LoRA for CL with foundation models. In a nutshell, SD-LoRA incrementally decouples the learning of the magnitude and direction of LoRA components, while fixing the directions learned from previous tasks as CL progresses. Concretely, let $\mathcal { M } = \{ \alpha _ { k } \} _ { k = 1 } ^ { t }$ denote the learnable LoRA magnitudes, and $\mathcal { W } = \{ \bar { \mathbf { A } _ { k } \mathbf { B } _ { k } } \} _ { k = 1 } ^ { t - 1 }$ represent the previously learned directions. As illustrated in Fig. 1(b), during learning on Tt, SD-LoRA computes the output of a given layer of the classifier $f _ { \theta }$ by

$$
\boldsymbol {h} ^ {\prime} = \left(\mathbf {W} _ {0} + \alpha_ {1} \overline {{\mathbf {A} _ {1} \mathbf {B} _ {1}}} + \alpha_ {2} \overline {{\mathbf {A} _ {2} \mathbf {B} _ {2}}} + \dots + \alpha_ {t} \overline {{\mathbf {A} _ {t} \mathbf {B} _ {t}}}\right) \boldsymbol {x}, \tag {4}
$$

where the color-highlighted terms $\{ \alpha _ { k } \} _ { k = 1 } ^ { t }$ and $\overline { { \mathbf { A } _ { t } \mathbf { B } _ { t } } }$ are learnable. The original weight matrix $\mathbf { W } _ { 0 }$ and the previously learned directions $\{ \overline { { \mathbf { A } _ { k } \mathbf { B } _ { k } } } \} _ { k = 1 } ^ { t - 1 }$ remain fixed.

# 3.3 EMPIRICAL ANALYSIS OF SD-LORA

By incrementally decoupling the magnitude and direction of LoRA components while preserving the directions learned from previous tasks, we observe substantial performance gains across various CL benchmarks (see Sec. 4). Nevertheless, the underlying working mechanism—particularly how SD-LoRA mitigates catastrophic forgetting—remains poorly understood. To shed light on this, we conduct a series of experiments and distill our observations into three key findings.

Finding 1: When fine-tuning the foundation model directly on different downstream tasks, the resulting task-specific weights end up closer to each other than the original model weights. To illustrate this, we consider five tasks drawn from ImageNet-R (Boschini et al., 2022), and fine-tune the ViT-B-16 model (Dosovitskiy et al., 2020) for each task, thereby obtaining five sets of optimal task-specific weights $\{ \mathbf { W } _ { i } ^ { \star } \} _ { i = 1 } ^ { 5 }$ . As shown in Fig. 2(a), measuring the relative distances in parameter space reveals that these task-specific weights cluster more closely with one another than the original weights of the foundation model $\mathbf { W } _ { 0 }$ .

![](images/128df2b380211466e40c5f19e5e5e5430b2510d39cf11b9952c38d2a5b5f095e.jpg)

<details>
<summary>heatmap</summary>

| Task Index i | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.8 | 0.6 | 0.4 | 0.2 | 0.0 |
| 2 | 0.7 | 0.5 | 0.3 | 0.1 | 0.0 |
| 3 | 0.6 | 0.4 | 0.2 | 0.1 | 0.0 |
| 4 | 0.5 | 0.3 | 0.1 | 0.0 | 0.0 |
| 5 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 |
</details>

(a)

![](images/f48aed59fda35ea6fbfcc888768a636f7ce71121f2c698ced0cda2cac18cf987.jpg)

<details>
<summary>line</summary>

| Number of Tasks | Vanilla LoRA | LoRA with Fixed Direction |
| --------------- | ------------ | ------------------------- |
| 1               | 88.0         | 88.0                      |
| 2               | 82.0         | 84.0                      |
| 3               | 79.0         | 82.0                      |
| 4               | 77.0         | 80.0                      |
| 5               | 76.0         | 78.0                      |
</details>

![](images/b6cf90b04848a38b4e3075e18c2c3bdcbb7c361b6acf96adc144fc88153cfc79.jpg)

<details>
<summary>line</summary>

| Number of Tasks | Vanilla LoRA | LoRA with Fixed Direction |
| --------------- | ------------ | ------------------------- |
| 1               | 90.0         | 90.0                      |
| 2               | 87.5         | 88.0                      |
| 3               | 85.0         | 86.0                      |
| 4               | 82.5         | 84.0                      |
| 5               | 80.0         | 82.0                      |
| 6               | 78.0         | 80.0                      |
| 7               | 77.0         | 79.0                      |
| 8               | 76.0         | 78.0                      |
| 9               | 75.0         | 77.0                      |
| 10              | 73.0         | 76.0                      |
</details>

Figure 2: (a) Distances between the five optimal weights $\{ \mathbf { W } _ { i } ^ { \star } \}$ on ImageNet-R $( N = 5 )$ relative to the foundation model weights $\mathbf { W } _ { 0 }$ . All relative distances are much smaller than one, indicating that $\{ \mathbf { W } _ { i } ^ { \star } \}$ are closer to each other than to $\mathbf { W } _ { 0 } .$ . (b) and (c) Performance comparison of Vanilla LoRA versus LoRA with the first learned direction fixed, on ImageNet-R across five and ten tasks, respectively. Shaded regions indicate standard error.

We further conduct a CL experiment in which only the LoRA magnitude is continually optimized, while the direction remains fixed after the first task. Consequently, the updated output for a given layer at the current $\mathcal { T } _ { t }$ becomes $\begin{array} { r l } { \pmb { h } ^ { \prime } = ( \mathbf { W } _ { 0 } + } & { { } \overline { { \mathbf { A } _ { 1 } \mathbf { B } _ { 1 } } } ) \pmb { x } } \end{array}$ . As shown in Figs. 2(b) and (c), the average accuracy up to the current task consistently surpasses that of the vanilla LoRA baseline, $i . e ,$ , $\pmb { h } ^ { \prime } = ( \mathbf { W } _ { 0 } + \mathbf { \Sigma } \mathbf { \Sigma } ) \pmb { x }$ . Aligning well with (Entezari et al., 2022; Gueta et al., 2023), our results further indicate that the fine-tuned weights for different tasks lie in close proximity, enabling relatively strong performance even when fixing a single learned direction.

![](images/7baba363c333559225d339e7476375dcf64c1dc6e9af88dabfefbd329b0b8cfb.jpg)

<details>
<summary>line</summary>

| Iteration | Mean     | Standard error |
| --------- | -------- | -------------- |
| 0         | 0.0000   | 0.0000         |
| 100       | 1.2000   | 1.0000         |
| 200       | 2.5000   | 2.0000         |
| 300       | 3.8000   | 3.0000         |
| 400       | 4.5000   | 4.0000         |
| 450       | 4.8000   | 4.5000         |
</details>

![](images/a2979fae718e5de0c9d4314dc3d5eb83d078c320034f4eb5ca12a369793dfc0d.jpg)

<details>
<summary>line</summary>

| α    | After training on T₂ | After training on T₃ | After training on T₄ | After training on T₅ |
| ---- | --------------------- | --------------------- | --------------------- | --------------------- |
| α₁   | 2.1                   | 1.9                   | 1.8                   | 1.7                   |
| α₂   | 1.6                   | 1.4                   | 1.3                   | 1.3                   |
| α₃   | 1.4                   | 1.3                   | 1.2                   | 1.2                   |
| α₄   | 1.2                   | 1.1                   | 1.0                   | 1.0                   |
| α₅   | 1.0                   | 1.0                   | 1.0                   | 1.0                   |
</details>

(b)

![](images/1beae5eb3757507c1022208f7de336970087d500a46394ca8788d9a52fd82d13.jpg)

<details>
<summary>line</summary>

|        | T2    | T3    | T4    | T5    | T6    | T7    | T8    | T9    | T10   |
| ------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| α₁     | 1.6   | 1.5   | 1.4   | 1.3   | 1.2   | 1.1   | 1.0   | 1.0   | 1.0   |
| α₂     | 1.4   | 1.3   | 1.2   | 1.1   | 1.0   | 0.9   | 0.8   | 0.8   | 0.8   |
| α₃     | 1.2   | 1.1   | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.6   | 0.6   |
| α₄     | 1.1   | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.5   | 0.5   |
| α₅     | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
| α₆     | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
| α₇     | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
| α₈     | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
| α₉     | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
| α₁₀    | 1.0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.4   | 0.4   |
</details>

(c)   
Figure 3: Analysis of the learning process of SD-LoRA. (a) Least squares fitting residual between the newly learned direction $\overline { { \mathbf { A } _ { t } \mathbf { B } _ { t } } }$ and all previous directions $\{ \overline { { \mathbf { A } _ { k } \mathbf { B } _ { k } } } \} _ { k = 1 } ^ { t - 1 }$ over time. (b) and (c) Learned magnitudes $\{ \alpha _ { k } \} _ { k = 1 } ^ { N }$ on ImageNet-R across five and ten tasks, respectively.

Finding 2: The directions preserved from previous tasks $( i . e . , \left\{ \overline { { \mathbf { A } _ { k } \mathbf { B } _ { k } } } \right\} _ { k = 1 } ^ { t - 1 } )$ play a significant role in the CL process—particularly those learned in the initial tasks. To reveal this, we first compute the least squares fitting residual between $\overline { { \mathbf { A } _ { t } \mathbf { B } _ { t } } }$ and $\{ \overline { { \mathbf { A } _ { k } \mathbf { B } _ { k } } } \} _ { k = 1 } ^ { t - 1 }$ , which increases over time (see Fig. 3(a)). Initially, the newly learned direction strongly aligns with earlier ones, enabling learned direction reuse. As training continues, $\overline { { \mathbf { A } _ { t } \mathbf { B } _ { t } } }$ gradually diverges, incorporating subtle variations that distinguish it from previous directions.

Further analysis of the learned magnitudes, all initialized to ones, reveals that $\alpha _ { k }$ values corresponding to earlier tasks rise rapidly, while those for later tasks exhibit a general decline trend (see Figs. 3(b) and (c) as well as Appendix $\mathrm { A } . 2 )$ . This pattern suggests that the classifier increasingly relies on directions learned from the earlier tasks, whereas the recently introduced directions serve primarily as slight adjustments to accommodate the specific requirements of the later tasks.

Finding 3: SD-LoRA effectively uncovers a low-loss path by leveraging the fixed directions from previous tasks with the learned magnitudes $\{ \alpha _ { k } \} _ { k = 1 } ^ { N }$ , toward a low-loss region shared by all tasks. To verify this, we conduct weight interpolation experiments to examine the linear path between the two sets of model weights for two sequential tasks. As depicted in Fig. 4, along the linear path from $\mathbf { W } _ { 1 } ^ { \mathrm { S D } }$ to $\mathbf { W } _ { 2 } ^ { \mathrm { S D } }$ learned by SD-LoRA, the performance on $\dot { \mathcal { T } } _ { 2 }$ steadily improves without loss in accuracy on $\mathcal { T } _ { 1 }$ . However, this is not the case for vanilla LoRA, where the performance improvement of $\mathcal { T } _ { 2 }$ is at the expense of $\mathcal { T } _ { 1 }$ , indicative of catastrophic forgetting. These observations suggest that SD-LoRA selectively scales the parameter update along the previously learned directions, effectively enabling the classifier to trace a low-loss path that ultimately settles on an overlapping low-loss region for all tasks (see Fig. 4 (a)).

![](images/e347b2b74e65e0665944a5396f0233f91e3dc984bb282d268d715ce23789515b.jpg)

<details>
<summary>text_image</summary>

W0
A1B1
W1LoRA
W1SD
A2B2
W2SD
W2LoRA
/ Low-loss region for T1
</details>

(a)

![](images/921f214cb0bbad8f00461cf927839f058c33309b71f48ede0cf8c96d4f744329.jpg)

<details>
<summary>line</summary>

| Interpolation Point | Task 1 Acc | Task 2 Acc |
| ------------------ | ---------- | ---------- |
| w₁^ααα             | 90.0       | 78.0       |
| 0.1                | 89.5       | 79.0       |
| 0.2                | 88.5       | 80.0       |
| 0.3                | 87.5       | 81.0       |
| 0.4                | 87.0       | 82.0       |
| 0.5                | 86.5       | 82.5       |
| 0.6                | 86.0       | 83.0       |
| 0.7                | 85.5       | 83.0       |
| 0.8                | 85.0       | 83.0       |
| w₁^ααα             | 86.0       | 79.0       |
</details>

![](images/2e83f79d495254c4d14ad801c0bf957f082c1e81d568a8393f311c55fd99fb6c.jpg)

<details>
<summary>line</summary>

| Interpolation Point | Task 1 Acc | Task 2 Acc |
| ------------------- | ---------- | ---------- |
| w₁°                 | 90.5       | 88.5       |
| 0.1                 | 91.0       | 89.0       |
| 0.3                 | 91.5       | 89.5       |
| 0.4                 | 91.7       | 90.0       |
| 0.5                 | 91.6       | 90.2       |
| 0.6                 | 91.8       | 90.3       |
| 0.7                 | 91.7       | 90.4       |
| 0.8                 | 91.5       | 90.5       |
| w₂°                 | 91.6       | 90.4       |
</details>

(c)   
Figure 4: Learning trajectory comparison of vanilla LoRA and SD-LoRA. (a) Toy illustration of the ng trajectories for vanilla LoRA across two sequential tasks. ( $( \mathbf { W } _ { 0 } \to \mathbf { W } _ { 1 } ^ { \mathrm { L o R A } } \to \mathbf { W } _ { 2 } ^ { \mathrm { L o R A } } )$ and SD-LoRAong the vanilla $( \mathbf { W } _ { 0 } \to \mathbf { W } _ { 1 } ^ { \mathrm { S D } } \to$ $\mathbf { W } _ { 2 } ^ { \mathrm { S D } } )$ improvement on $\mathcal { T } _ { 2 }$ but degradation on $\mathcal { T } _ { 1 }$ indicates that vanilla LoRA suffers from catastrophic forgetting. (c) Classification accuracy along the SD-LoRA path, showing that it successfully lands on an overlapping low-loss region.

These findings elucidate why SD-LoRA excels in CL with foundation models. Initially, it identifies critical LoRA directions during learning earlier tasks, and relies heavily on these directions to guide the classifier toward an overlapping low-loss region for learned tasks. Subsequently, by progressively incorporating LoRA components, SD-LoRA refines these directions to converge on the shared low-loss region for both earlier and later tasks. This mechanism of tracing a low-loss trajectory eliminates the need to store samples from previous tasks for task-specific component selection, making SD-LoRA strong and rehearsal-free.

# 3.4 THEORETICAL ANALYSIS OF SD-LORA

In this subsection, based on the results in (Jiang et al., 2023), we present a theoretical analysis to explain why the initially learned LoRA directions are so critical (as in Finding 2).

Let $\Delta \mathbf { W } ^ { \star } \in \mathbb { R } ^ { m \times n }$ be the optimal update matrix lying in the overlapping low-loss region for all N sequential tasks. Additionally, let $\{ \Delta \mathbf { W } _ { i } ^ { \star } \} _ { i = 1 } ^ { N }$ represent the optimal update matrices in their respective low-loss regions. Denote the singular values of $\Delta \mathbf { W } _ { t } ^ { \star }$ as $\sigma _ { 1 } \geq . . . \geq \sigma _ { \operatorname* { m i n } \{ m , n \} } \geq 0$ . The matrices $\mathbf { A } \in \mathbb { R } ^ { m \times r }$ and $\mathbf { B } \in \mathbb { R } ^ { r \times n }$ are updated iteratively, starting from initial values given by $\frac { \rho } { 3 \sqrt { m + n + r } } \big ( \mathbf { A } _ { 0 } , \mathbf { B } _ { 0 } \big )$ ρ , where the entries of ${ \bf A } _ { 0 }$ and $\mathbf { B } _ { 0 }$ are i.i.d. according to $\mathcal { N } ( 0 , \sigma _ { 1 } )$ , and $\rho$ is the initialization scalingcondition number as $\begin{array} { r } { \kappa _ { j } = \frac { \sigma _ { 1 } } { \sigma _ { j } } } \end{array}$ For an intege. Finally, let $\| \cdot \| _ { \mathrm { o p } }$ he range  denote t $\{ 0 , 1 , \ldots , \operatorname* { m i n } \{ r , m , n \} \}$ , define the $j \cdot$

Theorem 1. Suppose the assumptions stated in Appendix A.1 hold, where $\epsilon _ { 1 }$ is a small constant. Let δ ∈ (0, 1) be such that δ ≤ mink∈{1,...,j} $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ $\begin{array} { r } { \delta \le \operatorname* { m i n } _ { k \in \{ 1 , \dots , j \} } \frac { \sigma _ { k } - \sigma _ { k + 1 } } { \sigma _ { k } } } \end{array}$ σk−σk+1 . Fix any tolerance level ϵ2 satisfying σk $\epsilon _ { 2 }$ $\begin{array} { r } { \epsilon _ { 2 } \leq \frac { 1 } { m + n + r } } \end{array}$ . Let η denote the learning rate for updating the matrices A and B, and define $\Delta \mathbf { W } ^ { [ : i ] }$ as the rank-i approximation of $\Delta \mathbf { W } ^ { \star }$ , obtained by retaining the top-i principal components.

Then, there exist some numerical constants c and $c ^ { \prime } ,$ , and a sequence of iteration indices:

$$
i _ {1} \leq i _ {2} \leq \ldots \leq i _ {j} \leq \frac {c ^ {\prime}}{\delta \eta \sigma_ {j}} \log \left(\frac {\kappa_ {j}}{\delta \epsilon_ {2}}\right)
$$

such that, with high probability, gradient descent with step size $\eta \leq$ c min $\begin{array} { r } { \{ \delta , 1 - \delta \} \frac { \sigma _ { j } ^ { 2 } } { \sigma _ { 1 } ^ { 3 } } } \end{array}$ and initialization scaling factor $\begin{array} { r } { \rho \le \big ( \frac { c \delta \epsilon _ { 2 } } { \kappa _ { j } } \big ) } \end{array}$ 1cδ ensures that the approximation error satisfies

$$
\left\| \mathbf {A} _ {i _ {k}} \mathbf {B} _ {i _ {k}} - \Delta \mathbf {W} ^ {[: k ]} \right\| _ {\mathrm{op}} \leq \epsilon_ {2} \sigma_ {1} + \epsilon_ {1}, \quad \forall k = 1, 2, \dots , j. \tag {5}
$$

In Theorem 1, we formulate the learning process of SD-LoRA as a matrix factorization problem, and prove that gradient descent with small initialization drives the learned product AB to approximate the principal components of $\Delta \mathbf { W } ^ { \star } , i . e , \Delta \mathbf { W } ^ { [ 1 ] } , \Delta \mathbf { W } ^ { [ : 2 ] } , \ldots , \Delta \mathbf { W } ^ { [ : j ] }$ , sequentially. This theoretical insight explains the observed decreasing trend in the learned magnitudes, and further supports the feasibility of the subsequent parameter-efficient variants of SD-LoRA.

# 3.5 TWO VARIANTS OF SD-LORA

Our analysis in Sec. 3.3 reveals an interesting behavior of SD-LoRA: LoRA directions acquired during learning on earlier tasks are heavily reused and contribute substantially to classification accuracy. In contrast, directions learned through later tasks primarily serve as minor refinements, exhibiting a diminishing utility. We leverage this behavior and propose two variants of SD-LoRA with improved parameter efficiency, which integrate rank reduction and knowledge distillation, termed SD-LoRA-RR and SD-LoRA-KD, respectively.

SD-LoRA-RR. To mitigate incremental parameter expansion, we implement an empirical rankreduction strategy for the learnable matrices $\mathbf { A } _ { t } \in \mathbb { R } ^ { m \times r _ { t } }$ and $\mathbf { B } _ { t } \in \dot { \mathbb { R } } ^ { r _ { t } \times n }$ associated with later tasks. Specifically, the rank $r _ { t }$ is reduced in a stepwise manner:

$$
r _ {1} = r _ {2} = \dots > r _ {\mu} = r _ {\mu + 1} = \dots > r _ {\nu} = r _ {\nu + 1} = \dots = r _ {N}, \tag {6}
$$

where $\mu$ and $\nu$ are predefined task indices. The set of hyperparameters include $\{ \mu , \nu , r _ { 1 } , r _ { \mu } , r _ { \nu } \}$ . This stepwise reduction ensures that later tasks, which contribute less, are encoded with lower-rank approximations, thereby curbing computational and memory overhead.

SD-LoRA-KD. While the rank-reduction strategy limits parameter expansion, each new task still grows parameters incrementally. To fully address this issue, we propose a knowledge distillation approach based on least squares. Our method evaluates whether a newly introduced LoRA direction $\overline { { \mathbf { A } _ { t } \mathbf { B } _ { t } } }$ can be linearly represented by the subspace spanned by previously learned directions $\{ \overline { { \mathbf { A } _ { k } \mathbf { B } _ { k } } } \} _ { k = 1 } ^ { t - 1 }$ . If a sufficient linear approximation exists, the fitting coefficients will be absorbed into existing learned magnitudes M rather than expanding the direction set W. Formally, after training on $\mathcal { T } _ { t } .$ , we solve the least squares optimization problem:

$$
\left\{\Delta \alpha_ {k} \right\} _ {k = 1} ^ {t - 1} = \underset {\left\{\alpha_ {k} ^ {\prime} \right\} _ {k = 1} ^ {t - 1}} {\arg \min} \left\| \overline {{{\mathbf {A} _ {t} \mathbf {B} _ {t}}}} - \sum_ {k = 1} ^ {t - 1} \alpha_ {k} ^ {\prime} \overline {{{\mathbf {A} _ {k} \mathbf {B} _ {k}}}} \right\| _ {F} ^ {2}, \tag {7}
$$

where $\Delta \alpha _ { k }$ denotes the optimal coefficient for the k-th learned direction. If the fitting residual is less than a predefined threshold $\tau ,$ we assimilate the new direction by updating the first t − 1 learned magnitudes from $\mathcal { T } _ { t }$ as

$$
\boldsymbol {h} ^ {\prime} = \left(\mathbf {W} _ {0} + (\alpha_ {1} + \Delta \alpha_ {1}) \overline {{\mathbf {A} _ {1} \mathbf {B} _ {1}}} + (\alpha_ {2} + \Delta \alpha_ {2}) \overline {{\mathbf {A} _ {2} \mathbf {B} _ {2}}} + \dots + (\alpha_ {t - 1} + \Delta \alpha_ {t - 1}) \overline {{\mathbf {A} _ {t - 1} \mathbf {B} _ {t - 1}}}\right) \boldsymbol {x}, \tag {8}
$$

which prevents parameter expansion while preserving knowledge through coefficient fusion. The complete implementations of SD-LoRAs are detailed in Algorithm 1.

# 4 EXPERIMENTS

In this section, we first present the experimental setups, and then compare SD-LoRAs with state-ofthe-art CL methods across multiple benchmarks and foundation models.

# 4.1 EXPERIMENTAL SETUPS

Evaluation Benchmarks and Protocols. Following (Gao et al., 2023; Liang & Li, 2024), we evaluate SD-LoRAs on three standard CL benchmarks: ImageNet-R (Boschini et al., 2022), ImageNet-A (Hendrycks et al., 2021), and DomainNet (Peng et al., 2019). Specifically, ImageNet-R consists of 200 ImageNet classes (Deng et al., 2009) rendered in artistic styles. ImageNet-A features 200 classes with natural adversarial examples, often misclassified by standard ImageNet-trained models. DomainNet includes 345 classes across six distinct domains. As common practices (Liang & Li, 2024; Huang et al., 2024), we split ImageNet-R into 5/10/20 tasks (40/20/10 classes per task), ImageNet-A into 10 tasks (20 classes each), and DomainNet into 5 tasks (69 classes each). Additionally, we include CIFAR100 (Krizhevsky, 2009) and CUB200 (Wah et al., 2011) results in Appendix A.3.

Algorithm 1 SD-LoRA and its Variants on the Current Task $\mathcal { T } _ { t }$   
Input: Weight matrix from the foundation model $W_{0}$ , current task $T_{t}$ , learned LoRA directions from previous tasks $W = \{\overline{A_{k}B_{k}}\}_{k=1}^{t-1}$ , rank parameters $\{\mu, \nu, r_{1}, r_{\mu}, r_{\nu}\}$ in Eqn. (6), residual threshold $\tau$ for Eqn. (7), and maximum number of iterations MaxIter.

Output: Sets of learned LoRA magnitudes M and directions W.

1: Initialize $A_{t} \in R^{m \times r_{t}}$ , $B_{t} \in R^{r_{t} \times n}$ , and $M = \{\alpha_{k}\}_{k=1}^{t}$ 2: if $t = \mu$ or $\nu$ then / Only for SD-LoRA-RR

3: Reduce the lower dimension of $A_{t}$ and $B_{t}$ to $r_{\mu}$ or $r_{\nu}$ 4: end if

5: for Iter = 0 to MaxIter do

6: Compute the cross-entropy loss on the current task $T_{t}$ using Eqn. (4)

7: Update $\{\alpha_{k}\}_{k=1}^{t}$ and $\overline{A_{t}B_{t}}$ by minimizing Eqn. (1) using some stochastic optimizer

8: end for

9: $W \leftarrow W \cup \{\overline{A_{t}B_{t}}\}$ 10: Solve Problem (7) to obtain the optimal fitting coefficients $\{\Delta\alpha_{k}\}_{k=1}^{t-1}$ / Only for SD-LoRA-KD

11: if the fitting residual $\left\|\overline{A_{t}B_{t}} - \sum_{k=1}^{t-1} \Delta\alpha_{k} \overline{A_{k}B_{k}}\right\|_{F} \leq \tau$ then

12: $M \leftarrow \{\alpha_{k} + \Delta\alpha_{k}\}_{k=1}^{t-1}$ and $W \leftarrow \{\overline{A_{k}B_{k}}\}_{k=1}^{t-1}$

Table 2: Performance comparison on ImageNet-R across different task lengths. 

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-R (N = 5)</td><td colspan="2">ImageNet-R (N = 10)</td><td colspan="2">ImageNet-R (N = 20)</td></tr><tr><td>Acc ↑</td><td>AAA↑</td><td>Acc ↑</td><td>AAA ↑</td><td>Acc ↑</td><td>AAA ↑</td></tr><tr><td>Full Fine-Tuning</td><td>64.92(0.87)</td><td>75.57(0.50)</td><td>60.57(1.06)</td><td>72.31(1.09)</td><td>49.95(1.31)</td><td>65.32(0.84)</td></tr><tr><td>L2P</td><td>73.04(0.71)</td><td>76.94(0.41)</td><td>71.26(0.44)</td><td>76.13(0.46)</td><td>68.97(0.51)</td><td>74.16(0.32)</td></tr><tr><td>DualPrompt</td><td>69.99(0.57)</td><td>72.24(0.41)</td><td>68.22(0.20)</td><td>73.81(0.39)</td><td>65.23(0.45)</td><td>71.30(0.16)</td></tr><tr><td>CODA-Prompt</td><td>76.63(0.27)</td><td>80.30(0.28)</td><td>74.05(0.41)</td><td>78.14(0.39)</td><td>69.38(0.33)</td><td>73.95(0.63)</td></tr><tr><td>HiDe-Prompt</td><td>74.77(0.25)</td><td>78.15(0.24)</td><td>74.65(0.14)</td><td>78.46(0.18)</td><td>73.59(0.19)</td><td>77.93(0.19)</td></tr><tr><td>InfLoRA</td><td>76.95(0.23)</td><td>81.81(0.14)</td><td>74.75(0.64)</td><td>80.67(0.55)</td><td>69.89(0.56)</td><td>76.68(0.57)</td></tr><tr><td>SD-LoRA</td><td>79.15(0.20)</td><td>83.01(0.42)</td><td>77.34(0.35)</td><td>82.04(0.24)</td><td>75.26(0.37)</td><td>80.22(0.72)</td></tr><tr><td>SD-LoRA-RR</td><td>79.01(0.26)</td><td>82.50(0.38)</td><td>77.18(0.39)</td><td>81.74(0.24)</td><td>74.05(0.51)</td><td>80.65(0.35)</td></tr><tr><td>SD-LoRA-KD</td><td>78.85(0.29)</td><td>82.47(0.58)</td><td>77.03(0.67)</td><td>81.52(0.26)</td><td>74.12(0.66)</td><td>80.11(0.75)</td></tr></table>

We adopt two standard and widely used CL metrics: Average accuracy (Acc) and average anytime accuracy (AAA). The Acc metric measures the overall performance by computing the average accuracy across all N tasks upon the completion of CL. AAA further accumulates the average accuracy of all encountered tasks after training on each new task.

Competing Methods and Implementation Details. We compare SD-LoRAs against state-of-the-art ViT-based CL methods, including L2P (Wang et al., 2022b), DualPrompt (Wang et al., 2022a), CODA-Prompt (Smith et al., 2023), HiDe-Prompt (Wang et al., 2024a), and InfLoRA (Liang & Li, 2024). We also incorporate full fine-tuning as a form of performance lower bound. Following prior work (Gao et al., 2023; Huang et al., 2024), we employ ViT-B/16 (Dosovitskiy et al., 2020), pretrained on ImageNet-21K and fine-tuned on ImageNet-1K as the foundation model for classification. We also experiment with a self-supervised ViT-B/16 from DINO (Caron et al., 2021). The SD-LoRA components are inserted into the attention layers of all Transformer blocks, modifying the query and value projections, with a fixed rank of $r _ { 1 } = 1 0$ . It is noteworthy that we utilize a shared set of LoRA magnitudes for all projections, each with different LoRA directions. For SD-LoRA-RR, we set the additional rank parameters as $\mu = 4 , \nu = 8 , r _ { \mu } = 8 ,$ and $r _ { \nu } = 6$ . For SD-LoRA-KD, we set the threshold for the fitting residual to $\mathfrak { r } = 9 \times 1 0 ^ { - 4 }$ . For all methods, training is carried out by

Table 3: Performance comparison on ImageNet-A and DomainNet across different task lengths. 

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-A (N = 10)</td><td colspan="2">DomainNet (N = 5)</td></tr><tr><td>Acc ↑</td><td>AAA ↑</td><td>Acc ↑</td><td>AAA ↑</td></tr><tr><td>Full Fine-Tuning</td><td> $16.31_{(7.89)}$ </td><td> $30.04_{(13.18)}$ </td><td> $51.46_{(0.47)}$ </td><td> $67.08_{(1.13)}$ </td></tr><tr><td>L2P (Wang et al., 2022b)</td><td> $42.94_{(1.27)}$ </td><td> $51.40_{(1.95)}$ </td><td> $70.26_{(0.25)}$ </td><td> $75.83_{(0.98)}$ </td></tr><tr><td>DualPrompt (Wang et al., 2022a)</td><td> $45.49_{(0.96)}$ </td><td> $54.68_{(1.24)}$ </td><td> $68.26_{(0.90)}$ </td><td> $73.84_{(0.45)}$ </td></tr><tr><td>CODA-Prompt (Smith et al., 2023)</td><td> $45.36_{(0.78)}$ </td><td> $57.03_{(0.94)}$ </td><td> $70.58_{(0.53)}$ </td><td> $76.68_{(0.44)}$ </td></tr><tr><td>HiDe-Prompt (Wang et al., 2024a)</td><td> $42.70_{(0.60)}$ </td><td> $56.32_{(0.40)}$ </td><td> $72.20_{(0.08)}$ </td><td> $77.01_{(0.04)}$ </td></tr><tr><td>InfLoRA (Liang &amp; Li, 2024)</td><td> $49.20_{(1.12)}$ </td><td> $60.92_{(0.61)}$ </td><td> $71.59_{(0.23)}$ </td><td> $78.29_{(0.50)}$ </td></tr><tr><td>SDLoRA</td><td> $55.96_{(0.73)}$ </td><td> $64.95_{(1.63)}$ </td><td> $72.82_{(0.37)}$ </td><td> $78.89_{(0.50)}$ </td></tr><tr><td>SD-LoRA-RR</td><td> $55.59_{(1.08)}$ </td><td> $64.59_{(1.91)}$ </td><td> $72.58_{(0.40)}$ </td><td> $78.79_{(0.78)}$ </td></tr><tr><td>SD-LoRA-KD</td><td> $54.24_{(1.12)}$ </td><td> $63.89_{(0.58)}$ </td><td> $72.15_{(0.50)}$ </td><td> $78.44_{(0.66)}$ </td></tr></table>

![](images/5bc9c294124e8047489fb907eb2e9ae56d80f6c68f38fd8fd1e6770c2361bf87.jpg)

<details>
<summary>line</summary>

| Number of Tasks | L2P   | DualPrompt | CODA-Prompt | HiDe-Prompt | InfLoRA | SD-LoRA | SD-LoRA-RR | SD-LoRA-KD |
| --------------- | ----- | ---------- | ----------- | ----------- | ------- | ------- | ---------- | ---------- |
| 1               | 75.0  | 78.0       | 80.0        | 82.0        | 80.0    | 82.0    | 84.0       | 86.0       |
| 2               | 65.0  | 72.0       | 78.0        | 80.0        | 78.0    | 80.0    | 82.0       | 84.0       |
| 3               | 55.0  | 68.0       | 75.0        | 78.0        | 75.0    | 78.0    | 80.0       | 82.0       |
| 4               | 50.0  | 65.0       | 72.0        | 75.0        | 72.0    | 75.0    | 78.0       | 80.0       |
| 5               | 48.0  | 62.0       | 70.0        | 72.0        | 70.0    | 72.0    | 75.0       | 78.0       |
| 6               | 46.0  | 60.0       | 68.0        | 70.0        | 68.0    | 70.0    | 72.0       | 75.0       |
| 7               | 44.0  | 58.0       | 65.0        | 68.0        | 65.0    | 68.0    | 70.0       | 72.0       |
| 8               | 42.0  | 55.0       | 62.0        | 65.0        | 62.0    | 65.0    | 68.0       | 70.0       |
| 9               | 41.0  | 53.0       | 60.0        | 63.0        | 60.0    | 63.0    | 65.0       | 68.0       |
| 10              | 40.0  | 51.0       | 58.0        | 61.0        | 58.0    | 61.0    | 63.0       | 65.0       |
</details>

(a)

![](images/dd4b9074cc809e90742280440d34873ccdf11d8c5ffdea800267b6eda6f21255.jpg)

<details>
<summary>line</summary>

| Number of Tasks | L2P   | DualPrompt | CODA-Prompt | HiDe-Prompt | InfLoRA | SD-LoRA | SD-LoRA-RR | SD-LoRA-KD |
| --------------- | ----- | ---------- | ----------- | ----------- | ------- | ------- | ---------- | ---------- |
| 1               | 89.5  | 89.0       | 90.0        | 89.5        | 90.5    | 90.0    | 90.5       | 90.5       |
| 2               | 87.0  | 86.0       | 88.0        | 87.0        | 88.0    | 87.5    | 88.0       | 88.0       |
| 3               | 84.0  | 83.0       | 86.0        | 84.0        | 85.0    | 84.5    | 85.0       | 85.0       |
| 4               | 81.0  | 80.0       | 83.0        | 81.0        | 82.0    | 81.5    | 82.0       | 82.0       |
| 5               | 78.0  | 77.0       | 80.0        | 78.0        | 79.0    | 78.5    | 79.0       | 79.0       |
| 6               | 76.0  | 75.0       | 78.0        | 76.0        | 77.0    | 76.5    | 77.0       | 77.0       |
| 7               | 74.0  | 73.0       | 76.0        | 74.0        | 75.0    | 74.5    | 75.0       | 75.0       |
| 8               | 73.0  | 72.0       | 75.0        | 73.0        | 74.0    | 73.5    | 74.0       | 74.0       |
| 9               | 72.0  | 71.0       | 74.0        | 72.0        | 73.0    | 72.5    | 73.0       | 73.0       |
| 10              | 71.0  | 70.0       | 73.0        | 71.0        | 72.0    | 71.5    | 72.0       | 72.0       |
</details>

(b)

![](images/bcd492f24bceb9a812d13a14ddd0fa368c2b4d569fc338a23a2120f161070f32.jpg)

<details>
<summary>line</summary>

| Number of Tasks | L2P   | DualPrompt | CODA-Prompt | HiDe-Prompt | InfLoRA | SD-LoRA | SD-LoRA-RR | SD-LoRA-KD |
| --------------- | ----- | ---------- | ----------- | ----------- | ------- | ------- | ---------- | ---------- |
| 1               | 80.0  | 78.0       | 82.0        | 84.0        | 86.0    | 85.0    | 83.0       | 84.0       |
| 2               | 75.0  | 72.0       | 78.0        | 80.0        | 82.0    | 80.0    | 78.0       | 79.0       |
| 3               | 70.0  | 68.0       | 75.0        | 77.0        | 79.0    | 77.0    | 75.0       | 76.0       |
| 4               | 65.0  | 65.0       | 72.0        | 73.0        | 75.0    | 73.0    | 71.0       | 72.0       |
| 5               | 60.0  | 62.0       | 68.0        | 69.0        | 71.0    | 70.0    | 69.0       | 70.0       |
</details>

(c)   
Figure 5: Average accuracy during sequential training on (a) ImageNet-A (N = 10), (b) ImageNet-R (N = 10), and (c) ImageNet-R (N = 5) using ViT-B/16 from DINO (Caron et al., 2021).

Adam (Kingma & Ba, 2014) with a learning rate of 0.008 and a minibatch size of 128 for 30 epochs on ImageNet-R, 10 epochs on DomainNet, and 20 epochs on all other datasets. We report mean results across five runs with standard errors.

# 4.2 EXPERIMENTAL RESULTS

Results on Different CL benchmarks Using Different Backbones. In Tables 2 and 3, it is clear that SD-LoRA achieves significant improvements over existing methods. Specifically, on ImageNet-R (N = 20), SD-LoRA surpasses InfLoRA by margins of 7.68% in Acc and 4.62% in AAA. Similarly, on ImageNet-A, SD-LoRA outperforms HiDe-prompt by approximately 31.05% in Acc and 15.32% in AAA. Even on the more complex DomainNet, comprising six distinct domains, SD-LoRA consistently attains the best performance. To demonstrate the generality of SD-LoRA, we also evaluate it using the self-supervised ViT-B/16 from DINO. The results in Fig. 5(c) show that SD-LoRA continues to deliver superior performance under both Acc and AAA using different backbones. Finally, we observe that the two variants SD-LoRA-RR and SD-LoRA-KD, exhibit only marginal performance degradations relative to the full SD-LoRA model, confirming the effectiveness of their parameter-efficient designs.

Results across Varied Task Lengths. To evaluate the scalability and generalizability of SD-LoRAs under different task lengths, we follow (Liang & Li, 2024; Huang et al., 2024) and partition ImageNet-R into 5, 10, and 20 sequential tasks containing 40, 20, and 10 classes per task, respectively. As shown in Table 2, SD-LoRAs demonstrate consistent superiority over existing methods, with performance margins growing as the number of tasks. This underscores the suitability of SD-LoRAs for scenarios requiring resource-efficient CL without compromising accuracy.

Ablation Studies. To validate the contributions of design choices in the proposed SD-LoRA, we conduct a series of ablation experiments, with quantitative results summarized in Table 4. First, we fix the singly learned LoRA direction while allowing its magnitude to adapt during training. This simplified configuration already achieves nontrivial performance, highlighting the critical role of the initial LoRA direction in CL. Second, we decouple the magnitude and direction learning, but restrict the classifier to a single LoRA component. The inferior performance relative to SD-LoRA suggests that the performance gains of SD-LoRA cannot be attributed solely to decoupling. Instead, the synergistic effect of training multiple decoupled LoRA components is essential for achieving satisfactory results. Last, we fix learned LoRA components without magnitude rescaling, and also observe a noticeable performance decline. This suggests that the rescaling mechanism enables SD-LoRA to navigate low-loss paths by reweighting contributions from earlier components.

Table 4: Ablation analysis of the proposed SD-LoRA. Trainable parameters are highlighted in orange. 

<table><tr><td rowspan="2">Training Strategy</td><td colspan="2">ImageNet-R (N = 5)</td><td colspan="2">ImageNet-R (N = 10)</td></tr><tr><td>Acc ↑</td><td>AAA ↑</td><td>Acc ↑</td><td>AAA ↑</td></tr><tr><td> $\mathbf{W}_{0} + \alpha \overline{\mathbf{A}_{1}\mathbf{B}_{1}}$ </td><td>78.17(0.27)</td><td>81.93(0.51)</td><td>74.82(0.96)</td><td>80.63(0.63)</td></tr><tr><td> $\mathbf{W}_{0} + \alpha \overline{\mathbf{A}\mathbf{B}}$ </td><td>73.24(0.31)</td><td>78.80(0.13)</td><td>70.62(0.78)</td><td>76.32(0.16)</td></tr><tr><td> $\mathbf{W}_{0} + \overline{\mathbf{A}_{1}\mathbf{B}_{1}} + \ldots + \alpha \overline{\mathbf{A}_{t}\mathbf{B}_{t}}$ </td><td>78.28(0.59)</td><td>82.02(0.71)</td><td>74.29(0.32)</td><td>79.74(0.71)</td></tr><tr><td> $\mathbf{W}_{0} + \alpha_{1}\overline{\mathbf{A}_{1}\mathbf{B}_{1}} + \ldots + \alpha_{t}\overline{\mathbf{A}_{t}\mathbf{B}_{t}}$  (SD-LoRA)</td><td>79.15(0.20)</td><td>83.01(0.42)</td><td>77.34(0.35)</td><td>82.04(0.24)</td></tr></table>

Table 5: Comparison on ImageNet-R (N = 20) in terms of computation (GFLOPs), parameter, and storage efficiency. 

<table><tr><td>Method</td><td>GFLOPs</td><td>Learnable Parameters (M)</td><td>Stored Features (M)</td></tr><tr><td>L2P (Wang et al., 2022b)</td><td>70.14</td><td>0.48</td><td>0</td></tr><tr><td>DualPrompt (Wang et al., 2022a)</td><td>70.26</td><td>0.06</td><td>0</td></tr><tr><td>CODA-Prompt (Smith et al., 2023)</td><td>70.61</td><td>0.38</td><td>0</td></tr><tr><td>HiDe-Prompt (Wang et al., 2024a)</td><td>70.36</td><td>0.08</td><td>0.15</td></tr><tr><td>InfLoRA (Liang &amp; Li, 2024)</td><td>35.12</td><td>0.37</td><td>0.10</td></tr><tr><td>SD-LoRA</td><td>35.12</td><td>0.37</td><td>0</td></tr><tr><td>SD-LoRA-RR</td><td>35.12</td><td>0.23</td><td>0</td></tr></table>

Analysis of Computation, Parameter, and Storage Efficiency. As presented in Table 5, we compare the inference computations in terms of GFLOPs, trainable parameters, and feature storage requirements of various CL methods. Notably, InfLoRA (Liang & Li, 2024) and the proposed SD-LoRA eliminate the need for task-specific prompt selection during inference, thereby enjoying the highest inference efficiency. Moreover, our proposed SD-LoRA-RR is capable of further reducing the number of LoRA parameters without reliance on sample rehearsal, making it an ideal choice for resource-constrained CL scenarios.

# 5 CONCLUSION AND DISCUSSION

In this paper, we have introduced SD-LoRA, a computational method designed to address scalability challenges in class-incremental learning with foundation models. By decoupling the learning of magnitude and direction of LoRA components, SD-LoRA provides a rehearsal-free, inferenceefficient, and end-to-end optimized solution. Our empirical and theoretical analysis demonstrates that SD-LoRA uncovers a low-loss trajectory that converges to an overlapping low-loss region for all learned tasks, effectively balancing stability and plasticity. Extensive experiments confirmed the effectiveness of SD-LoRA in mitigating catastrophic forgetting while maintaining adaptability to new tasks. Additionally, our two parameter-efficient variants, SD-LoRA-RR and SD-LoRA-KD, further enhance its practicality for resource-constrained applications.

While SD-LoRA has shown promise, several avenues for future research warrant exploration. First, extending SD-LoRA to other foundation models beyond ViTs could provide valuable insights into its generality and effectiveness across different backbone architectures. Second, integrating SD-LoRA with other PEFT techniques, such as adapters or prefix-tuning, may further enhance its performance and scalability. Finally, developing more theoretically grounded strategies for rank reduction and knowledge distillation within SD-LoRA could lead to additional improvements in parameter efficiency and overall performance.

# ACKNOWLEDGEMENTS

We would like to thank Ziye Ma and Xinyuan Song for helping formalize the proofs, and Xuelin Liu for assistance with the plots and diagrams. This work was supported in part by the National Key R&D Program of China (2020YFA0713900), the Hong Kong RGC General Research Fund (11220224), the CityU Applied Research Grant (9667264), the National Natural Science Foundation of China under the Tianyuan Fund for Mathematics (12426105) and under Grant 62306233, and the Major Key Project of PCL (PCL2024A06).

# REFERENCES

Matteo Boschini, Lorenzo Bonicelli, Angelo Porrello, Giovanni Bellitto, Matteo Pennisi, Simone Palazzo, Concetto Spampinato, and Simone Calderara. Transfer without forgetting. In European Conference on Computer Vision, pp. 692–709, 2022.   
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision Transformers. In IEEE/CVF International Conference on Computer Vision, pp. 9650–9660, 2021.   
Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019.   
Rajas Chitale, Ankit Vaidya, Aditya Kane, and Archana Ghotkar. Task arithmetic with LoRA for continual learning. arXiv preprint arXiv:2311.02428, 2023.   
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16 × 16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.   
Sayna Ebrahimi, Franziska Meier, Roberto Calandra, Trevor Darrell, and Marcus Rohrbach. Adversarial continual learning. In European Conference on Computer Vision, pp. 386–402, 2020.   
Rahim Entezari, Hanie Sedghi, Olga Saukh, and Behnam Neyshabur. The role of permutation invariance in linear mode connectivity of neural networks. In International Conference on Learning Representations, 2022.   
Robert M French. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3 (4):128–135, 1999.   
Qiankun Gao, Chen Zhao, Yifan Sun, Teng Xi, Gang Zhang, Bernard Ghanem, and Jian Zhang. A unified continual learning framework with general parameter-efficient tuning. In IEEE/CVF International Conference on Computer Vision, pp. 11483–11493, 2023.   
Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, and Leshem Choshen. Knowledge is a region in weight space for fine-tuned language models. In Findings of the Association for Computational Linguistics, pp. 1350–1370, 2023.   
Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Christina Kwon, and Jacob Steinhardt. Natural adversarial examples. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15262–15271, 2021.   
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning, pp. 2790–2799, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.   
Wei-Cheng Huang, Chun-Fu Chen, and Hsiang Hsu. OVOR: OnePrompt with virtual outlier regularization for rehearsal-free class-incremental learning. In International Conference on Learning Representations, 2024.   
Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2023.   
Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Visual prompt tuning. In European Conference on Computer Vision, pp. 709–727, 2022.   
Liwei Jiang, Yudong Chen, and Lijun Ding. Algorithmic regularization in model-free overparametrized asymmetric matrix factorization. SIAM Journal on Mathematics of Data Science, pp. 723–744, 2023.   
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521–3526, 2017.   
Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL https://www.cs.toronto.edu/\~kriz/ learning-features-2009-TR.pdf.   
Kibok Lee, Kimin Lee, Jinwoo Shin, and Honglak Lee. Overcoming catastrophic forgetting with unlabeled data in the wild. In IEEE/CVF International Conference on Computer Vision, pp. 312–321, 2019.   
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, 2021.   
Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021.   
Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12):2935–2947, 2017.   
Yan-Shuo Liang and Wu-Jun Li. InfLoRA: Interference-free low-rank adaptation for continual learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 23638– 23647, 2024.   
Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang F Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: Weight-decomposed low-rank adaptation. In International Conference on Machine Learning, pp. 32100–32121, 2024.   
Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggyback: Adapting a single network to multiple tasks by learning to mask weights. In European Conference on Computer Vision, pp. 67–82, 2018.   
James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102(3):419–457, 1995.   
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of Learning and Motivation, pp. 109–165, 1989.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In IEEE/CVF International Conference on Computer Vision, pp. 1406–1415, 2019.   
Guanghui Qin and Jason Eisner. Learning how to ask: Querying LMs with mixtures of soft prompts. arXiv preprint arXiv:2104.06599, 2021.   
Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. In Advances in Neural Information Processing Systems, pp. 79320–79362, 2023.   
Rahul Ramesh and Pratik Chaudhari. Model Zoo: A growing “brain” that learns continually. arXiv preprint arXiv:2106.03027, 2021.   
Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu, Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn without forgetting by maximizing transfer and minimizing interference. arXiv preprint arXiv:1810.11910, 2018.   
David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy Lillicrap, and Gregory Wayne. Experience replay for continual learning. In Advances in Neural Information Processing Systems, pp. 350–360, 2019.   
James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. CODA-Prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11909–11919, 2023.   
Rishabh Tiwari, Krishnateja Killamsetty, Rishabh Iyer, and Pradeep Shenoy. GCR: Gradient coreset based replay buffer selection for continual learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 99–108, 2022.   
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. URL https://authors.library.caltech.edu/27452/.   
Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, and Jun Zhu. Hierarchical decomposition of prompt-based continual learning: Rethinking obscured sub-optimality. In Advances in Neural Information Processing Systems, pp. 69054–69076, 2024a.   
Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(8):5362–5383, 2024b.   
Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoxi Sun, Han Zhang, Chen-Yu Lee, Xiaoqi Ren, Guolong Su, Vincent Perot, and Jennifer Dy. DualPrompt: Complementary prompting for rehearsalfree continual learning. In European Conference on Computer Vision, pp. 631–648, 2022a.   
Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 139–149, 2022b.   
Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Classincremental learning: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9851–9873, 2024.

# A APPENDIX

# A.1 PROOF OF THEOREM 1

In this section, we prove Theorem 1 as a specific case of Theorem 2.

Assumption 1. The optimal updates $\Delta \mathbf { W } _ { t } ^ { \star } , t \in \{ 1 , 2 , \ldots , N \}$ lie within a small neighborhood in the loss landscape, meaning there exists a small constant $\epsilon _ { 1 } > 0$ such that $\| \Delta \mathbf { W } ^ { \star } - \Delta \mathbf { \bar { W } } _ { t } ^ { \star } \| _ { \mathrm { o p } } < \epsilon _ { 1 }$ .

Assumption 2. The first $j + 1$ singular values of $\Delta \mathbf { W } _ { t } ^ { \star }$ are distinct, $i . e , \sigma _ { 1 } > . . . > \sigma _ { j } > \sigma _ { j + 1 }$ .

Theorem 2. Fix any $j \leq r .$ . Suppose $\sigma _ { j + 1 } < \sigma _ { j }$ , choose any $\gamma \in ( 0 , 1 )$ ) such that $\begin{array} { r } { \frac { \sigma _ { j + 1 } } { \sigma _ { j } } \leq \gamma . } \end{array}$ . Pick any stepsize η ≤ min{ 600σ31 $\begin{array} { r } { \eta \leq \operatorname* { m i n } \{ \frac { \gamma \sigma _ { j } ^ { 2 } } { 6 0 0 \sigma _ { 1 } ^ { 3 } } , \frac { ( 1 - \gamma ) \sigma _ { j } } { 2 0 \sigma _ { j } ^ { 2 } } \} } \end{array}$ γ σ2j , (1−γ)σj20σ2 }. For any cρ < 1, let the initialization size ρ satisfy j $c _ { \rho } < 1$

$$
\rho \leq \min \left\{\frac {1}{3}, \frac {1 - \gamma}{2 4}, \frac {c _ {\rho} \sigma_ {1}}{1 2 (m + n + r) \sqrt {\frac {1 - \gamma}{2 4} \sqrt {\sigma_ {j}}}} \right\},
$$

and

$$
\rho \leq \min \left\{\left(\frac {(1 - \gamma) c _ {\rho} \sigma_ {j}}{1 2 0 0 (m + n + r) j \sigma_ {1}}\right) ^ {\frac {2 (1 + \gamma)}{1 - \gamma}}, \left(\frac {\gamma \sigma_ {j} ^ {2}}{1 6 0 0 0 r \sigma_ {j} ^ {2}}\right) ^ {\frac {1 + \gamma}{1 - \gamma}}, \frac {\gamma \sigma_ {j} \sqrt {2 j}}{1 6 \sigma_ {j} \sqrt {m + n + r}} \right\}
$$

Define

$$
T _ {1} = \left\lfloor \frac {\log (\frac {1 2 (m + n + r) \sqrt {\frac {1 - \gamma}{2 4}} \sqrt {\sigma_ {j}})}{c _ {\rho} \rho \sqrt {\sigma_ {j}}})}{\log (1 + \frac {1 + \gamma}{2} \eta \sigma_ {j})} \right\rfloor + 1, \qquad T _ {2} = \left\lfloor \frac {\log (\sqrt {\frac {2 4}{1 - \gamma}})}{\log (1 + 0 . 1 \eta \sigma_ {j})} \right\rfloor + 1,
$$

$$
T _ {3} = \left\lfloor \frac {\log (\rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} / 3)}{\log (1 - \frac {3}{2} \eta \sigma_ {j})} \right\rfloor + 1, \qquad T = \left\lfloor \frac {\log (\rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} / \rho)}{\log (1 + \gamma \eta \sigma_ {j})} \right\rfloor
$$

Define $T _ { 0 } : = T _ { 1 } + T _ { 2 } + T _ { 3 }$ , then we have

$$
\frac {T _ {0}}{T} \leq 1 - \frac {(3 - 2 \gamma) (1 - \gamma)}{6 (3 \gamma + 1)}.
$$

Furthermore, there exists a universal constant C such that with probability at least $1 - ( C c _ { \rho } ) ^ { r - j + 1 } -$ $C \exp ( - r / C )$ , for all $T _ { 0 } \leq i \leq T$ , we have

$$
\left\| \mathbf {A} _ {i} \mathbf {B} _ {i} - \Delta \mathbf {W} ^ {\left[: j \right]} \right\| _ {\mathrm{op}} \leq 8 \rho^ {\frac {\delta}{2 (2 - \sigma)}} \sigma_ {1} + 4 \rho^ {\frac {\delta}{2 (2 - \delta)}} \sqrt {2 j} \sigma_ {1} + \epsilon_ {1}. \tag {A.9}
$$

Proof. The sequential training of LoRA can be conceptualized as a matrix approximation problem (Jiang et al., 2023):

$$
\ell (\mathbf {A}, \mathbf {B}) = \frac {1}{2} \| \mathbf {A B} - \Delta \mathbf {W} _ {t} ^ {\star} \| _ {F} ^ {2}, \tag {A.10}
$$

where $\Delta \mathbf { W } _ { t } ^ { \star }$ denotes the optimal update matrix for the current t-th task. We compute the gradients of ℓ with respect to A and B, respectively:

$$
\nabla_ {\mathbf {A}} \ell = (\mathbf {A B} - \Delta \mathbf {W} _ {t} ^ {\star}) \mathbf {B} ^ {\intercal}
$$

$$
\nabla_ {\mathbf {B}} \ell = \mathbf {A} ^ {\intercal} (\mathbf {A B} - \Delta \mathbf {W} _ {t} ^ {\star}).
$$

Then, the gradient descent updates with a step size of η are

$$
\mathbf {A} _ {+} = \mathbf {A} - \eta \nabla_ {\mathbf {A}} \ell = \mathbf {A} + \eta (\Delta \mathbf {W} _ {t} ^ {\star} - \mathbf {A B}) \mathbf {B} ^ {\intercal},
$$

$$
\mathbf {B} _ {+} = \mathbf {B} - \eta \nabla_ {\mathbf {B}} \ell = \mathbf {B} + \eta \mathbf {A} ^ {\intercal} (\mathbf {A B} - \Delta \mathbf {W} _ {t} ^ {\star}).
$$

By performing the singular value decomposition (SVD) of $\Delta \mathbf { W } _ { t } ^ { \star }$ , i.e,

$$
\Delta \mathbf {W} _ {t} ^ {\star} = \boldsymbol {\Phi} _ {t} \boldsymbol {\Sigma} _ {t} \boldsymbol {\Psi} _ {t} ^ {\intercal},
$$

and exploiting the rotational invariance of the Frobenius norm in Eqn. (A.10), i.e, by substituting $\mathbf { A } \to \bar { \Phi } _ { \mathbf { t } } ^ { \intercal } \mathbf { A }$ and $\mathbf { B } \to \Psi _ { t } ^ { \intercal } \mathbf { B }$ , we may assume without loss of generality that $\Delta \mathbf { W } _ { t } ^ { \star }$ is diagonal.

We then rewrite $\Delta \mathbf { W } _ { t } ^ { \star }$ as

$$
\Delta \mathbf {W} _ {t} ^ {\star} = \left( \begin{array}{c c} \boldsymbol {\Sigma} _ {t} & \mathbf {0} \\ \mathbf {0} & \boldsymbol {\Sigma} _ {t} ^ {\prime} \end{array} \right)
$$

with $\Sigma _ { t } ~ = ~ \operatorname { d i a g } ( \sigma _ { 1 } , \ldots , \sigma _ { j } ) ~ \in ~ \mathbb { R } ^ { j \times j }$ and $\Sigma _ { t } ^ { \prime } ~ \in ~ \mathbb { R } ^ { ( m - j ) \times ( n - j ) }$ be a diagonal matrix with $\sigma _ { j + 1 } , \ldots , \sigma _ { \operatorname* { m i n } \{ m , n \} }$ on the diagonals. We next introduce the following block partitions:

$$
\mathbf {A} = \binom{\mathbf {U}}{\mathbf {J}} \quad \text { and } \quad \mathbf {B} = (\mathbf {V} \mathbf {K})  ,
$$

where

$$
\mathbf {U} \in \mathbb {R} ^ {j \times r}, \quad \mathbf {J} \in \mathbb {R} ^ {(m - j) \times r}, \quad \mathbf {V} \in \mathbb {R} ^ {r \times j}, \quad \mathbf {K} \in \mathbb {R} ^ {r \times (n - j)}.
$$

In this way, we can decompose $\mathbf { A B } - \Delta \mathbf { W } _ { t } ^ { \star }$ as

$$
\mathbf {A} \mathbf {B} - \Delta \mathbf {W} _ {t} ^ {\star} = \left( \begin{array}{c c} \mathbf {U V} - \boldsymbol {\Sigma} _ {t} & \mathbf {U K} \\ \mathbf {J V} & \mathbf {J K} - \boldsymbol {\Sigma} _ {t} ^ {\prime} \end{array} \right),
$$

and we are ready to bound the difference $\mathbf { A B } - \Delta \mathbf { W } ^ { \star }$ :

$$
\begin{array}{l} \| \mathbf {A B} - \Delta \mathbf {W} ^ {\star} \| _ {\mathrm{op}} = \| \mathbf {A B} - \Delta \mathbf {W} _ {t} ^ {\star} + \Delta \mathbf {W} _ {t} ^ {\star} - \Delta \mathbf {W} ^ {\star} \| _ {\mathrm{op}} \\ \leq \left\| \mathbf {A} \mathbf {B} - \Delta \mathbf {W} _ {t} ^ {\star} \right\| _ {\mathrm{op}} + \left\| \Delta \mathbf {W} _ {t} ^ {\star} - \Delta \mathbf {W} ^ {\star} \right\| _ {\mathrm{op}} \\ \leq \| \mathbf {U V} - \boldsymbol {\Sigma} _ {t} \| _ {\mathrm{op}} + \| \mathbf {U K} \| _ {\mathrm{op}} + \| \mathbf {J V} \| _ {\mathrm{op}} + \| \mathbf {J K} - \boldsymbol {\Sigma} _ {t} ^ {\prime} \| _ {\mathrm{op}} + \epsilon_ {1}, \\ \end{array}
$$

where we note that $\pmb { \Sigma } _ { t } = \Delta \mathbf { W } _ { t } ^ { [ : j ] }$ (after diagonalization of $\Delta \mathbf { W } _ { t } ^ { \star } )$ . To ensure convergence, it suffices to show that the dominant term UV approaches $\Sigma _ { t }$ while the error terms $( \mathbf { J } , \mathbf { K } )$ remain small. Let us first extract the gradient descent update of the top left block U:

$$
\begin{array}{l} \mathbf {U} _ {+} = \mathbf {U} + \eta \Big [ (\mathbf {\Sigma} _ {t} - \mathbf {U V}) \mathbf {V} ^ {\intercal} + (- \mathbf {U K}) \mathbf {K} ^ {\intercal} \Big ] \\ = \mathbf {U} + \eta \left(\boldsymbol {\Sigma} _ {t} \mathbf {V} ^ {\intercal} - \mathbf {U} \left(\mathbf {V V} ^ {\intercal} + \mathbf {K K} ^ {\intercal}\right)\right). \\ \end{array}
$$

Similarly, we have

$$
\mathbf {V} _ {+} = \mathbf {V} + \eta \left(\mathbf {U} ^ {\intercal} \boldsymbol {\Sigma} _ {t} - \left(\mathbf {U} ^ {\intercal} \mathbf {U} + \mathbf {J} ^ {\intercal} \mathbf {J}\right) \mathbf {V}\right),
$$

$$
\mathbf {J} _ {+} = \mathbf {J} + \eta \left(\boldsymbol {\Sigma} _ {t} ^ {\prime} \mathbf {K} ^ {\intercal} - \mathbf {J} \left(\mathbf {V V} ^ {\intercal} + \mathbf {K K} ^ {\intercal}\right)\right),
$$

$$
\mathbf {K} _ {+} = \mathbf {K} + \eta \Big (\mathbf {J} ^ {\intercal} \boldsymbol {\Sigma} _ {t} ^ {\prime} - \big (\mathbf {U} ^ {\intercal} \mathbf {U} + \mathbf {J} ^ {\intercal} \mathbf {J} \big) \mathbf {K} \Big).
$$

To account for the potential imbalance of U and V, we introduce the following quantities,

$$
\mathbf {F} = \frac {\mathbf {U} + \mathbf {V} ^ {\intercal}}{2} \quad \text { and } \quad \mathbf {G} = \frac {\mathbf {U} - \mathbf {V} ^ {\intercal}}{2},
$$

so that

$$
\mathbf {U} = \mathbf {F} + \mathbf {G} \quad \text { and } \quad \mathbf {V} ^ {\intercal} = \mathbf {F} - \mathbf {G}.
$$

Then, the updates for F and G are given by

$$
\begin{array}{l} \mathbf {F} _ {+} = \frac {1}{2} \left(\mathbf {U} _ {+} + \mathbf {V} _ {+} ^ {\intercal}\right) \\ = \frac {1}{2} \left[ \mathbf {U} + \mathbf {V} ^ {\intercal} + \eta \left(\boldsymbol {\Sigma} _ {t} \mathbf {V} ^ {\intercal} + \boldsymbol {\Sigma} _ {t} \mathbf {U}\right) - \eta \left(\mathbf {U} \left(\mathbf {V V} ^ {\intercal} + \mathbf {K K} ^ {\intercal}\right) + \mathbf {V} ^ {\intercal} \left(\mathbf {U} ^ {\intercal} \mathbf {U} + \mathbf {J} ^ {\intercal} \mathbf {J}\right)\right) \right] \\ = \mathbf {F} + \eta \boldsymbol {\Sigma} _ {t} \mathbf {F} - \frac {\eta}{2} \Big [ (\mathbf {F} + \mathbf {G}) (\mathbf {V V ^ {\intercal}} + \mathbf {K K ^ {\intercal}}) + (\mathbf {F} - \mathbf {G}) (\mathbf {U ^ {\intercal} U} + \mathbf {J ^ {\intercal} J}) \Big ]. \\ \end{array}
$$

A similar computation gives

$$
\begin{array}{l} \mathbf {G} _ {+} = \frac {1}{2} \left(\mathbf {U} _ {+} - \mathbf {V} _ {+} ^ {\intercal}\right) \\ = \mathbf {G} - \eta \boldsymbol {\Sigma} _ {t} \mathbf {G} - \frac {\eta}{2} \left[ (\mathbf {F} + \mathbf {G}) (\mathbf {V V} ^ {\intercal} + \mathbf {K K} ^ {\intercal}) - (\mathbf {F} - \mathbf {G}) (\mathbf {U} ^ {\intercal} \mathbf {U} + \mathbf {J} ^ {\intercal} \mathbf {J}) \right]. \\ \end{array}
$$

It is now natural to introduce the following equations:

$$
\mathbf {P} = \boldsymbol {\Sigma} _ {t} - \mathbf {F} \mathbf {F} ^ {\intercal} + \mathbf {G} \mathbf {G} ^ {\intercal}, \quad \mathbf {Q} = \mathbf {F} \mathbf {G} ^ {\intercal} - \mathbf {G} \mathbf {F} ^ {\intercal},
$$

so that one can verify that

$$
\mathbf {P} + \mathbf {Q} = \boldsymbol {\Sigma} _ {t} - \mathbf {U V}.
$$

According to the Proposition B.2 and B.5 of Jiang et al. (2023), it holds with high probability that for any $T _ { 1 } + \mathbf { \bar { \it { T } } } _ { 2 } + T _ { 3 } \leq { \dot { \it { i } } } \leq T$ ,

$$
\| \mathbf {U} _ {i} \mathbf {K} _ {i} \| _ {\mathrm{op}} \leq 3 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sigma_ {1}, \| \mathbf {J} _ {i} \mathbf {V} _ {i} \| _ {\mathrm{op}} \leq 3 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sigma_ {1},
$$

$$
\| \mathbf {J} _ {i} \mathbf {K} _ {i} \| _ {\mathrm{op}} \leq \rho^ {\frac {1 - \gamma}{(1 + \gamma)}} \sigma_ {1}, \| \mathbf {Q} _ {i} \| _ {\mathrm{op}} \leq 4 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sqrt {2 j} \sigma_ {1}.
$$

$$
\begin{array}{l} \| \mathbf {P} _ {i} \| _ {\mathrm{op}} \leq (1 - 0. 7 9 \eta \sigma_ {j}) ^ {2} + 6 \eta^ {2} \sigma_ {1} ^ {2} \| \mathbf {P} _ {i - 1} \| _ {\mathrm{op}} + 8 0 \eta \rho^ {\frac {1 - \gamma}{1 + \gamma} j} j \sigma_ {1} \\ \leq (1 - \frac {3 \eta \sigma_ {j}}{2}) \| \mathbf {P} _ {i - 1} \| _ {\mathrm{op}} + 8 0 \eta \rho^ {\frac {1 - \gamma}{1 + \gamma} j \sigma_ {1} ^ {2}} \\ \leq 2 \left(1 - \frac {3 \eta \sigma_ {j}}{2}\right) ^ {i - T _ {1} - T _ {2}} \sigma_ {1} + \frac {8 0 \rho^ {\frac {1 - \gamma}{1 + \gamma}} j \sigma_ {1} ^ {2}}{\sigma_ {r}}. \\ \end{array}
$$

Given that $\begin{array} { r } { T _ { 3 } = \left\lfloor \frac { \log ( \rho ^ { \frac { 1 - \gamma } { 2 ( 1 + \gamma ) } } / 3 ) } { \log ( 1 - \frac { 3 } { 2 } \eta \sigma _ { j } ) } \right\rfloor + 1 } \end{array}$ , it follows that for all i satisfying $T _ { 1 } + T _ { 2 } + T _ { 3 } \leq i \leq T$ , we 1 −γ have $\| \mathbf { P } _ { i } \| _ { \mathrm { o p } } \leq \rho ^ { \frac { 1 - \gamma } { 2 ( 1 + \gamma ) } } \sigma _ { 1 }$ .

$$
\| \mathbf {U} _ {i} \mathbf {V} _ {i} - \boldsymbol {\Sigma} _ {t} \| _ {\mathrm{op}} = \| \mathbf {P} _ {i} + \mathbf {Q} _ {i} \| _ {\mathrm{op}} \leq \| \mathbf {P} _ {i} \| _ {\mathrm{op}} + \| \mathbf {Q} _ {i} \| _ {\mathrm{op}} \leq \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sigma_ {1} + 4 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sqrt {2 j} \sigma_ {1}.
$$

By combining these parts, we can have

$$
\| \mathbf {A} _ {i} \mathbf {B} _ {i} - \Delta \mathbf {W} ^ {\left[: j \right]} \| _ {\mathrm{op}} \leq 8 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sigma_ {1} + 4 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sqrt {2 j} \sigma_ {1} + \epsilon_ {1} = (8 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} + 4 \rho^ {\frac {1 - \gamma}{2 (1 + \gamma)}} \sqrt {2 j}) \sigma_ {1} + \epsilon_ {1}
$$

![](images/e63d123d50035f87c62da6a199b84d9b27452b7c9babb88f1dead378313aa934.jpg)

# A.2 ADDITIONAL RESULTS FOR SEC. 3.3

To further investigate the temporal evolution of learned LoRA magnitudes $\mathcal { M } = \{ \alpha _ { k } \} _ { k = 1 } ^ { N }$ in SD-LoRA, we conduct additional experiments on ImageNet-R (Boschini et al., 2022) with an extended task length of $N = 2 0$ and a more challenging DomainNet dataset (Peng et al., 2019). As visualized in Fig. 6, both experimental configurations reveal a systematic decrease in $\alpha _ { k }$ values throughout the training process. These results corroborate the descending trend observed in our main experiments, demonstrate the consistent behaviors of SD-LoRA across extended task horizons and diverse domain distributions, and align with the theoretical analysis presented in Sec. 3.4.

![](images/f88b5719148719166f1c4d0b95a3cd211962b0d20370baf307f6d95f91891860.jpg)

<details>
<summary>line</summary>

| α   | τ₂    | τ₉    | τ₁₅   | τ₃    | τ₁₀   | τ₁₆   | τ₄    | τ₁₁   | τ₁₇   | τ₅    | τ₁₂   | τ₁₈   | τ₆    | τ₁₃   | τ₁₉   | τ₇    | τ₁₄   | τ₂₀   |
| --- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 1   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 1.3   | 0.95  |
| 2   | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | 1.25  | nan   | nan   | nan   |
| 3   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | 1.2   | nan   | nan   | nan   |
| 4   | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | 1.15  | nan   | nan   | nan   |
| 5   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | 1.1   | nan   | nan   | nan   |
| 6   | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | 1.08  | nan   | nan   | nan   |
| 7   | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | 1.06  | nan   | nan   | nan   |
| 8   | 1.04  | 1.04  | 1.04  | 1.04  | 1.04  | 1.04  | 1.04  | 1.04  | 1.04  | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   |
| 9   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   |
| 10+| nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   | nan   |
| ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...   ...| ...    ...| ...    ...| ...    ...| ...    ...| ... |
| ?* (α) vs x: - τ₂, ..., x: - τ₈, y: - τ₁, etc., Legend: τ₂, τ₉, τ₁₀, ..., τ₂₀, etc., The index n is estimated based on the provided code.
</details>

(a)

![](images/4df95a5b8d8e3ef40bb639856ef95c8964a919240f3b1289283d64f9c42aefb4.jpg)

<details>
<summary>line</summary>

| α    | After training on T₂ | After training on T₃ | After training on T₄ | After training on T₅ |
| ---- | -------------------- | -------------------- | -------------------- | -------------------- |
| α₁   | 2.4                  | 2.3                  | 2.1                  | 2.0                  |
| α₂   | 1.6                  | 1.6                  | 1.6                  | 1.6                  |
| α₃   | 1.5                  | 1.5                  | 1.5                  | 1.5                  |
| α₄   | 1.4                  | 1.4                  | 1.4                  | 1.4                  |
| α₅   | 1.3                  | 1.3                  | 1.3                  | 1.3                  |
</details>

Figure 6: Learned LoRA magnitudes $\mathcal { M } = \{ \alpha _ { k } \} _ { k = 1 } ^ { N }$ in SD-LoRA on (a) ImageNet-R (N = 20) and (b) DomainNet (N = 5).

![](images/2b9ab4901e88a3adc323784db884c8f6b9d770790ecdbb9a7c6194c38712714c.jpg)

<details>
<summary>heatmap</summary>

| Task Index i | 1    | 2    | 3    | 4    | 5    |
| ------------ | ---- | ---- | ---- | ---- | ---- |
| 1            | 0.0  | 0.8  | 0.9  | 0.7  | 0.6  |
| 2            | 0.0  | 0.8  | 0.9  | 0.7  | 0.6  |
| 3            | 0.0  | 0.8  | 0.9  | 0.7  | 0.6  |
| 4            | 0.0  | 0.8  | 0.9  | 0.7  | 0.6  |
| 5            | 0.0  | 0.8  | 0.9  | 0.7  | 0.6  |
</details>

![](images/b7c3a0bc00e0ed5ffd45f8e758015e3dcd09bd7b6dd6c4043aa26151ec41b17f.jpg)

<details>
<summary>heatmap</summary>

| Task Index i | 1    | 2    | 3    | 4    | 5    |
| ------------ | ---- | ---- | ---- | ---- | ---- |
| 1            | 0.8  | 0.6  | 0.7  | 0.9  | 0.8  |
| 2            | 0.6  | 0.8  | 0.9  | 0.7  | 0.6  |
| 3            | 0.7  | 0.9  | 0.8  | 0.6  | 0.5  |
| 4            | 0.9  | 0.7  | 0.6  | 0.8  | 0.7  |
| 5            | 0.5  | 0.6  | 0.7  | 0.9  | 0.8  |
</details>

(b)

![](images/6a1a6cd1501d61949964ee92ef450aee6b0a8691a28e025ca9dcef5a607ecc6f.jpg)

<details>
<summary>heatmap</summary>

| Task Index i | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 2            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 3            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 4            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 5            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 6            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 7            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 8            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 9            | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| 10           | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |
</details>

(c)   
Figure 7: Relative distances computed on (a) DomainNet $( N = 5 )$ , (b) ImageNet-R $( N = 5 )$ using ViT-B/16 from DINO, and (c) ImageNet-R(N = 10), respectively.

# A.3 RESULTS ON OTHER CL BENCHMARKS

In addition to ImageNet-R, ImageNet-A, and DomainNet, we evaluate the proposed SD-LoRA on two other widely recognized benchmarks: CIFAR-100 (Krizhevsky, 2009) and CUB-200 (Wah et al., 2011). CIFAR-100 is a standard dataset for image classification, comprising 60, 000 images evenly distributed across 100 classes, with 600 images per class. For our experiments, we split CIFAR-100 into ten tasks, each containing ten classes. Similarly, CUB-200 is a fine-grained dataset specifically designed for bird classification, which consists of 11, 788 images across 200 classes. We divide this dataset into ten tasks, with each task encompassing 20 species. As shown in Table 6, the proposed SD-LoRAs consistently deliver outstanding performance on both datasets.

Additionally, we provide supplementary results to those in Fig. 2(a) by analyzing the relative distances between fine-tuned and pre-trained weights across different benchmarks, backbones, and task lengths. As shown in Fig. 7, the observed trends remain consistent with Finding 1.

Table 6: Performance comparison on CIFAR100 and CUB200. 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR100</td><td colspan="2">CUB200</td></tr><tr><td>Acc ↑</td><td>AAA ↑</td><td>Acc ↑</td><td>AAA ↑</td></tr><tr><td>Full Fine-Tuning</td><td>69.49(0.50)</td><td>80.35(0.87)</td><td>51.43(1.41)</td><td>69.74(0.93)</td></tr><tr><td>L2P (Wang et al., 2022b)</td><td>83.18(1.20)</td><td>87.69(1.05)</td><td>65.18(2.49)</td><td>76.12(1.27)</td></tr><tr><td>DualPrompt (Wang et al., 2022a)</td><td>81.48(0.86)</td><td>86.41(0.66)</td><td>68.00(1.06)</td><td>79.40(0.88)</td></tr><tr><td>CODA-Prompt (Smith et al., 2023)</td><td>86.31(0.12)</td><td>90.67(0.22)</td><td>71.92(0.33)</td><td>78.76(0.65)</td></tr><tr><td>InfLoRA (Liang &amp; Li, 2024)</td><td>86.75(0.35)</td><td>91.72(0.15)</td><td>70.82(0.23)</td><td>81.39(0.14)</td></tr><tr><td>SD-LoRA</td><td>88.01(0.31)</td><td>92.54(0.18)</td><td>77.48(0.20)</td><td>85.59(0.44)</td></tr><tr><td>SD-LoRA-RR</td><td>87.26(0.22)</td><td>92.05(0.31)</td><td>76.35(0.28)</td><td>83.89(0.35)</td></tr><tr><td>SD-LoRA-KD</td><td>87.09(0.45)</td><td>92.01(0.33)</td><td>75.95(0.55)</td><td>83.21(0.31)</td></tr></table>