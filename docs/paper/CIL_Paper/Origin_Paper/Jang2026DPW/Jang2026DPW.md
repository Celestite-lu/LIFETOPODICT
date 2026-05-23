# Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting

Hyeonseo Jang Hyuk Kwon Kibok Lee

Yonsei University

{jhyeonseo715, kh12043, kibok}@yonsei.ac.kr

# Abstract

We investigate recently introduced domain-class incremental learning scenarios for vision-language models (VLMs). Recent works address this challenge using parameter-efficient methods, such as prefix-tuning or adapters, which facilitate model adaptation to downstream tasks by incorporating task-specific information into input tokens through additive vectors. However, previous approaches often normalize the weights of these vectors, disregarding the fact that different input tokens require different degrees of adjustment. To overcome this issue, we propose Dynamic Prefix Weighting (DPW), a framework that dynamically assigns weights to prefixes, complemented by adapters. DPW consists of 1) a gating module that adjusts the weights of each prefix based on the importance of the corresponding input token, and 2) a weighting mechanism that derives adapter output weights as a residual of prefix-tuning weights, ensuring that adapters are utilized only when necessary. Experimental results demonstrate that our method achieves state-of-the-art performance in domain-class incremental learning scenarios for VLMs. The code is available at: https://github.com/YonseiML/dpw.

# 1. Introduction

Continual learning (CL) enables models to adapt to sequential data streams while mitigating catastrophic forgetting of previously acquired knowledge, offering an alternative to retraining from scratch in ever-changing environments [5, 13, 30]. Recent advances in CL have extended its applicability to multi-modal learning scenarios [44], including vision-language models (VLMs) such as CLIP [28]. Despite their impressive zero-shot capabilities, VLMs often yield suboptimal performance in real-world applications, underscoring the need for CL to improve downstream task adaptation while preserving the foundational knowledge established during pretraining [49]. To address this, previous studies have introduced benchmarks evaluating both the zero-shot and CL performance of VLMs on downstream tasks [18, 49]. However, the large-scale nature of VLMs poses significant challenges in fine-tuning the entire model due to the computational cost [11, 45].

Recent studies have explored parameter-efficient finetuning (PEFT) methods to mitigate this challenge. These approaches typically freeze the backbone model and learn taskspecific knowledge by introducing lightweight adapters [20, 45] or employing prefix-tuning techniques with learnable prompt vectors [18, 22, 32]. Such methods enable effective adaptation to new tasks while requiring only a small number of additional parameters. In the context of CL, recent PEFT advances primarily focus on adaptively weighting the influence of newly introduced parameters based on input samples, thereby mitigating catastrophic forgetting [32, 45]. However, as illustrated in Fig. 1, existing weighting mechanisms generally operate at the sample level, treating all tokens within a sample uniformly and assigning them the same amount of task-specific information. As prior studies [7, 52] have shown that dynamically adjusting the amount of injected information according to the task relevance of each token can substantially improve performance, the lack of token-level weighting thus represents a fundamental limitation.

We further observe that token-level weighting is hindered not only by the weighting strategies used in existing methods, but also by inherent properties of the pretrained backbone. For instance, within the prefix-tuning framework, existing methods typically rely on attention mechanisms to assign prefix weights, which can blur the token-wise distinctions required for fine-grained weighting. Because attention mechanisms tend to capture global contextual relationships within a sample [27, 34], they often project tokens with distinct characteristics closer together in the feature space. This behavior obscures the distinctions between task-relevant and irrelevant tokens, making fine-grained weighting difficult. Moreover, this limitation extends beyond individual samples to the task level, hindering the disentanglement of token embeddings across tasks. As a result, knowledge transfer in CL settings becomes less effective, and task interference increasingly degrades overall performance [9].

![](images/add4c58b705aead836c84c604355f48a4d685413eb1eb90d47ba2fd0753368db.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Adapter 1"] --> B["Router"]
    C["Adapter 2"] --> D["Adapter 1"]
    E["Adapter 1"] --> F["Adapter 2"]
    G["Adapter 1"] --> H["CLS"]
    I["Adapter 2"] --> J["CLS"]
    K["Adapter 1"] --> L["CLS"]
    M["Adapter 2"] --> N["CLS"]
    O["Adapter 1"] --> P["CLS"]
    Q["Adapter 2"] --> R["CLS"]
    S["Adapter 1"] --> T["CLS"]
    U["Adapter 2"] --> V["CLS"]
    W["Adapter 1"] --> X["CLS"]
    Y["Adapter 2"] --> Z["CLS"]
    AA["Adapter 1"] --> AB["CLS"]
    AC["Adapter 2"] --> AD["CLS"]
    AE["Adapter 1"] --> AF["CLS"]
    AG["Adapter 2"] --> AH["CLS"]
    AI["Adapter 1"] --> AJ["CLS"]
    AK["Adapter 2"] --> AL["CLS"]
    AM["Adapter 1"] --> AN["CLS"]
    AO["Adapter 2"] --> AP["CLS"]
    AQ["Adapter 1"] --> AR["CLS"]
    AS["Adapter 2"] --> AT["CLS"]
    AU["Adapter 1"] --> AV["CLS"]
    AW["Adapter 2"] --> AX["CLS"]
    AY["Adapter 1"] --> AZ["CLS"]
    BA["Adapter 2"] --> BB["CLS"]
    BC["Adapter 1"] --> BD["CLS"]
    BE["Adapter 2"] --> BF["CLS"]
    BG["Adapter 1"] --> BH["CLS"]
    BI["Adapter 2"] --> BJ["CLS"]
    BK["Adapter 1"] --> BL["CLS"]
    BM["Adapter 2"] --> BN["CLS"]
    BO["Adapter 1"] --> BP["CLS"]
    BZ["Adapter 2"] --> CA["CLS"]
```
</details>

(a) MoE-Adapters

![](images/5e600c6623b1c0bfed9cae96b402ec440a3f764c18ede18e31da3d8f486a3e3d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Mouse Image"] --> B{Total prefix attention weights are uniform across tokens}
    B --> C["Sum(·)"]
    C --> D["P_{K1}"]
    C --> E["P_{KL}"]
    D --> F["×"]
    E --> G["×"]
    F --> H["P_{V1}"]
    F --> I["P_{VL}"]
    G --> J["×"]
    H --> K["×"]
    I --> L["×"]
    J --> M["P_{V1}"]
    J --> N["P_{VL}"]
    K --> O["×"]
    L --> P["×"]
    M --> Q["Q_i"]
    N --> R["Q_j"]
    O --> S["..."]
    P --> T["..."]
    Q --> U["W^Q"]
    R --> U
    S --> U
    T --> U
    U --> V["..."]
    U --> W["..."]
    U --> X["..."]
    U --> Y["..."]
```
</details>

(b) DIKI

![](images/cc4544514fcd9c4b83fc69822cbcfcc6e443a7a29c5053f8d74228e851dd7e93.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Token-level dynamic weighting"] --> B["Sum(·)"]
    C["Balancing prefix & adapter"] --> B
    B --> D["Gating"]
    D --> E["Adapter"]
    E --> F["Output"]
    subgraph Input
        G["Mouse Icon"] --> H["×"]
        I["×"] --> J["×"]
        K["×"] --> L["×"]
        M["×"] --> N["×"]
        O["×"] --> P["×"]
        Q["×"] --> R["×"]
        S["×"] --> T["×"]
        U["×"] --> V["×"]
        W["×"] --> X["×"]
        Y["×"] --> Z["×"]
        AA["×"] --> AB["×"]
        AC["×"] --> AD["×"]
        AE["×"] --> AF["×"]
        AG["×"] --> AH["×"]
        AI["×"] --> AJ["×"]
        AK["×"] --> AL["×"]
        AM["×"] --> AN["×"]
        AO["×"] --> AP["×"]
        AQ["×"] --> AR["×"]
        AS["×"] --> AT["×"]
        AU["×"] --> AV["×"]
        AW["×"] --> AX["×"]
        AY["Input"] --> AZ["Sum(·)"]
    end
```
</details>

(c) Ours   
Figure 1. Comparison of the weighting mechanisms of various popular PEFT methods. Light-green boxes denote the output vectors added to the input tokens. (a): MoE-Adapters [45] relies solely on the CLS token for routing and assigns equal weights to the outputs of each adapter, with their sum always fixed at one. (b): DIKI [32] always normalizes the total weight of the prefixes added to each input, forcing them to sum up to one. (c): Our method dynamically adjusts prefixes and adapters weights based on the importance of the input tokens.

In this paper, we investigate how to incorporate an appropriate amount of task-specific information into each input token in CL. To this end, we propose dynamic prefix weighting (DPW), a framework that assigns refined weights to both prefixes and adapters based on the task relevance of the corresponding input token. First, we develop a novel gating module that replaces the conventional attention mechanism used in prefix-tuning. Our gating module consists of reparametrized prefix attention (RePA) and conditional activation (CondAct), which replace the query-key dot product and softmax activation, respectively. Unlike conventional prefix-tuning methods that rely on query-key projection, our RePA computes task relevance scores directly from input tokens using a learnable affine transformation. Subsequently, CondAct replaces the softmax activation with conditional normalization and filtering, ensuring that the total prefix weight is dynamically adjusted within an appropriate range. Second, we introduce a residual weighting mechanism (RWM) that assigns weights to adapters based on residual prefix weights generated during conditional normalization. RWM selectively incorporates adapter outputs for input tokens with higher prefix scores, enabling adapters to focus on more informative tokens according to refined prefix scores. As illustrated in Fig. 1(c), DPW provides finegrained weighting for both prefixes and adapters, enabling precise token-level modulation and improving knowledge retention in CL settings.

# 2. Related Work

Continual Learning (CL). Existing CL approaches can be categorized into three types: regularization-based, memorybased, and architecture-based methods. Regularizationbased methods [1, 2, 12, 15, 48] mitigate catastrophic forgetting by introducing penalty terms into the loss function, thereby preserving the prior knowledge acquired from earlier tasks. Memory-based approaches [3, 4, 21, 23, 42, 43] store historical data or synthetic samples generated by models, allowing selective replay to retain past knowledge. Architecture-based methods [6, 16, 19, 24, 31, 36–38] expand model parameters to incorporate new tasks while retaining knowledge from previous ones. PEFT based approaches, such as adapter and prompt-tuning, are typically categorized as architecture-based methods, as they introduce a small set of task-specific parameters while keeping the backbone frozen. Our method adopts a similar strategy, enabling efficient adaptation to downstream tasks with few additional parameters.

Continual Learning of VLMs. CL of VLMs seeks to enable these models to sequentially adapt to diverse domains and classes while preserving their zero-shot capabilities [18, 49]. One of the pioneering approaches is ZSCL [49], which addresses this challenge by leveraging a reference dataset and employing knowledge distillation from a frozen CLIP. Subsequent work has explored advanced distillation strategies, such as incorporating CLIP models trained on previous tasks into the distillation process with sample-wise adaptive balancing between two teacher models [47], or utilizing diffusion models to generate reference datasets for distillation [40]. However, these approaches require finetuning the entire CLIP model and rely on an additional reference dataset, which reduces training efficiency. An alternative line of work leverages PEFT techniques, such as adapters [8, 20, 45], for more efficient adaptation. For example, MoE-Adapters [45] introduce two adapters per task and use a router to assign weights within a mixture-of-experts framework, combining their outputs with the input tokens. C-CLIP [20] learns LoRA with a distillation loss and integrates it into the backbone after training each task. Another PEFT branch focuses on prompt-based methods [18, 22, 32], which learn task-specific information through a learnable vector called a prompt or a prefix. CoLeCLIP [18] extends to new tasks using a dedicated vocabulary and task-specific prompts. DIKI [32] mitigates prefix interference during CL by using cross-attention to separately assign weights to prefixes, with a batch-wise distribution-aware calibration factor. However, as widely reported in the prompt-based learning literature, these methods often suffer from performance degradation when applied to domains for which the prompts were not trained [36, 46, 51]. Unlike existing approaches, our method dynamically balances the use of prefixes and adapters to leverage the strengths of both.

# 3. Preliminaries

# 3.1. Continual Learning Protocol for VLMs

Following [32, 45, 49], we consider a collection of $T$ tasks, denoted by \ifmmode \lbrace \else \textbraceleft \fi \mathcal {T}^t\}\_{t=1}^{T} $\{ \mathcal { T } ^ { t } \} _ { t = 1 } ^ { T }$ . Each task $\mathcal { T } ^ { t }$ consists of a dataset $D _ { t }$ and a set of class names $C _ { t }$ . The class names in $C _ { t }$ are paired with manually crafted prompts and fed into the text encoder. The dataset $D _ { t }$ , consisting of $N _ { t }$ samples, is represented as $D _ { t } = \{ ( I _ { t n } , y _ { t n } ) \} _ { n = 1 } ^ { N _ { t } } ,$ , where $I _ { t n }$ is the input image and $y _ { t n }$ is its corresponding one-hot label from $C _ { t } .$ . In CL, models incrementally learn by sequentially processing each task $\mathcal { T } ^ { t } = \{ D _ { t } , C _ { t } \}$ . The primary objective of CL in VLMs is to maintain strong performance on both learned and unseen tasks, with the latter evaluated for zero-shot capabilities after each task to measure a new type of forgetting that arises in the context of CL for VLMs, known as forward forgetting. We consider two relevant benchmarks: Multi-domain Task Incremental Learning (MTIL) [49] and Open-Domain Continual Learning (ODCL-CIL) [18], where both domain and class distributions change across tasks. In MTIL, the model has access to the task ID during evaluation, whereas ODCL-CIL requires classification over all learned classes without task ID information.

# 3.2. Prompt-Based Continual Learning

Prompt-based methods are widely used in CL to adapt transformer-based pretrained models that process data samples in token units [10, 31, 36–38]. These methods adjust the model by introducing prompt tokens of length L, consisting of learnable vectors of dimension $d ,$ while keeping the model’s backbone frozen. For each task t, a unique set of vectors $P _ { t } \in \mathbb { R } ^ { L \times d }$ is learned, constructing a pool $\{ P _ { 1 } , P _ { 2 } , \dots , P _ { T } \}$ that spans $T$ tasks. At inference time, the prompt $P _ { t }$ is selected from the pool and attached to the pretrained model to restore task-specific knowledge [32], allowing input tokens $\boldsymbol { X } \in \mathbb { R } ^ { m \times d }$ to absorb task-specific information through the attention mechanism by modifying the inputs of multi-head self-attention layers [37, 41]. Among prompt-based methods, prefix-tuning [17] learns a pair of task-specific prompts $P _ { K } , P _ { V } \in \mathbb { R } ^ { L \times d }$ instead of a single $P ,$ , and separately projects them to the key and value in the attention mechanism.

Specifically, for the i-th head, input tokens and prefix are first transformed into query $Q _ { i } .$ key $K _ { i } ,$ and value $V _ { i }$ by corresponding projection matrices $\dot { W } _ { i } ^ { Q } , \dot { W } _ { i } ^ { K } , W _ { i } ^ { V }$ and their respective biases. With the input token $X$ , this can be formulated as follows:

$$
Q _ {i} = X W _ {i} ^ {Q} + \mathbf {1} _ {n} b _ {i} ^ {Q} \in \mathbb {R} ^ {n \times \frac {d}{h}},
$$

$$
K _ {i} = \left[ \begin{array}{c} K _ {X _ {i}} \\ K _ {P _ {i}} \end{array} \right] = \left[ \begin{array}{c} X W _ {i} ^ {K} + \mathbf {1} _ {n} b _ {i} ^ {K} \\ P _ {K} W _ {i} ^ {K} + \mathbf {1} _ {L} b _ {i} ^ {K} \end{array} \right] \in \mathbb {R} ^ {(n + L) \times \frac {d}{h}}, \tag {1}
$$

$$
V _ {i} = \left[ \begin{array}{c} V _ {X _ {i}} \\ V _ {P _ {i}} \end{array} \right] = \left[ \begin{array}{c} X W _ {i} ^ {V} + \mathbf {1} _ {n} b _ {i} ^ {V} \\ P _ {V} W _ {i} ^ {V} + \mathbf {1} _ {L} b _ {i} ^ {V} \end{array} \right] \in \mathbb {R} ^ {(n + L) \times \frac {d}{h}}.
$$

The self-attention computes the attention scores $S _ { i }$ using the inner product of query and key $Q _ { i } ( K _ { i } ) ^ { \top }$ :

$$
S _ {i} = \left[ S _ {X X} ^ {(i)}, S _ {X P} ^ {(i)} \right] = \left[ Q _ {i} \left(K _ {X _ {i}}\right) ^ {\top}, Q _ {i} \left(K _ {P _ {i}}\right) ^ {\top} \right]. \tag {2}
$$

Here, S\_{XX}^{(i)} r $S _ { X X } ^ { ( i ) }$ epresents the attention scores among the input tokens, whil e S\_{XP}^{(i)} $S _ { X P } ^ { ( i ) }$ represents the attention scores between the input tokens and the prefixes. Next, these scores are normalized to attention weights using a Softmax function with temperature $\tau ,$ which are then used to weight the corresponding value matrix $V _ { i }$ :

$$
h _ {i} = \text { softmax } \left(\frac {S _ {i}}{\sqrt {d / h}}\right) V _ {i} \in \mathbb {R} ^ {m \times \frac {d}{h}}. \tag {3}
$$

Finally, the outputs from all heads are concatenated and projected through the projection matrix $W _ { o }$ .

# 4. Methods

The main goals of our method are two-fold. First, we aim to ensure that each input token receives a proper amount of information from the prefixes and adapters throughout CL tasks. Second, we aim to establish an effective synergy between the prefixes and adapters. Our method addresses both motivations within a unified framework. Specifically, we first introduce a gating module that assigns refined weights to the prefixes. Then, guided by these weights, tokens that require additional adaptation receive information from the adapter. Fig. 2 provides an overview of the proposed framework.

# 4.1. Reparameterized Prefix Attention

In conventional prefix-tuning for CL, the learnable prefix key $P _ { K }$ is appended for each task and projected onto the pretrained key distribution through a key projection matrix. The $P _ { K }$ is then optimized to align with the pretrained query distribution of the corresponding task, determining the attention scores that weight the prefix value $P _ { V }$ . However, because the pretrained model typically lacks sufficient taskspecific knowledge for new tasks [49], these attention scores often fail to accurately capture the task relevance of input tokens, as illustrated in Fig. 3. Consequently, the model fails to effectively regulate the amount of information injected per token, which has been shown to impair downstream task performance [7, 52].

![](images/7d03bbf4fbf08d30f9bd12ebdd10aaa1ca331b869397dfaa49592d56c44aec81.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original flow"] --> B["Element-wise Sum"]
    C["Additional flow"] --> B
    B --> D["Concat"]
    D --> E["Scaled Dot-Product Attention"]
    E --> F["Image branch"]
    F --> G["Text branch"]
    G --> H["X ∈ R^M×d"]
    I["Deactivated"] --> J["Scalar"]
    K["Activated"] --> L["Vector"]
    M["Deactivated"] --> N["Scalar"]
    O["Activated"] --> P["Vector"]
    Q["Deactivated"] --> R["Scalar"]
    S["Activated"] --> T["Vector"]
    U["Deactivated"] --> V["Scalar"]
    W["Activated"] --> X["Vector"]
    Y["Deactivated"] --> Z["Scalar"]
    AA["Activated"] --> AB["Vector"]
    AC["Deactivated"] --> AD["Scalar"]
    AE["Activated"] --> AF["Vector"]
    AG["Deactivated"] --> AH["Scalar"]
    AI["Activated"] --> AJ["Vector"]
    AK["Deactivated"] --> AL["Scalar"]
    AM["Activated"] --> AN["Vector"]
    AO["Deactivated"] --> AP["Scalar"]
    AQ["Activated"] --> AR["Vector"]
    AS["Deactivated"] --> AT["Scalar"]
    AU["Activated"] --> AV["Vector"]
    AW["Deactivated"] --> AX["Scalar"]
    AY["Original flow"] --> AZ["W^o"]
    BA["Additional flow"] --> BB["W^o"]
    BC["Image branch"] --> BD["P_V ∈ R^{h'×L×(d/h')}"]
    BEText branch --> BF["P_Vi ∈ R^{h'/R^d/h'"]
    BF --> BG["P_Vi1 ∈ R^{d/h}"]
    BH["Text branch"] --> BI["P_ViL ∈ R^{h'/R^d/h}"]
    BI --> BJ["P_Vi1 ∈ R^{d/h}"]
    BK["Image branch"] --> BL["P_Vi1 ∈ R^{d/h}"]
    BM["Text branch"] --> BN["P_Vi1 ∈ R^{d/h}"]
    BO["X ∈ R^M×d"] --> BP["P_V ∈ R^{h'/R^d/h}"]
    BP --> BQ["P_Vi1 ∈ R^{d/h}"]
    BR["X ∈ R^M×d"] --> BS["P_Vi1 ∈ R^{d/h}"]
    BT["X ∈ R^M×d"] --> BU["P_Vi1 ∈ R^{d/h}"]
    BV["X ∈ R^M×d"] --> BW["P_Vi1 ∈ R^{d/h}"]
    BX["X ∈ R^M×d"] --> BY["P_Vi1 ∈ R^{d/h}"]
    
    subgraph Zoom-in view (for i-th head)
        Z["Oprefixi"] --> AA
        AB["Filtering"] --> AC
        AD["Conditional Norm"] --> AE["Sigmoid"]
        AF["Sijk"] --> AG["Sijk"]
        AH["SijL"] --> AI["SijL"]
        AJ["Sij = xjW_i^G + B_i^G"] --> AK
        AK --> BA
        BB["Sij = xjW_i^G + B_i^G"] --> BC
        BD --> BD
        BE --> BE
        BF --> BF
        BG --> BG
        BH --> BH
        BI --> BI
        BJ --> BJ
        BK --> BK
        BL --> BL
        BM --> BM
        BN --> BN
        BO --> BO
        CA["O_Ri ∈ R^d/h'"] --> AD
        AD --> AE
        AE --> AF
        AF --> AG
        AG --> AH
        AH --> AI
        AI --> AJ
        AJ --> AK
        AK --> AL
        AL --> AM["S_ij = xjW_i^G + B_i^G"]
        AM --> AN["S_ij = xjW_i^G + B_i^G"]
        AN --> AO["S_ij = xjW_i^G + B_i^G"]
        AP["S_ij = xjW_i^G + B_i^G"] --> AJ
        AJ --> AN
        AO --> AN
        AP --> AN
        AN --> AO
        AP -.->|Δ_ij| AO
        AO -.->|ε(x_j) ∈ R^d/h'| AP
        AP -.->|Δ_ij| AO
        AP -.->|ε(x_j) ∈ R^d/h'| AO
        AP -.->|Δ_ij| AO
        AP -.->|ε(x_j) ∈ R^d/h'| AP
        AP -.->|Δ_ij| AO
        AP -.->|ε(x_j) ∈ R^d/h'| AP
        AP -.->|Δ_ij| AO
    
    subgraph Zoom-in view (for i-th head)
        BI["Sigmoid"] --> AJ
        AK["Sijk"] --> AJ
        AL["SijL"] --> AJ
        AM["Sij = xjW_i^G + B_i^G"] --> AN
        AN -.->|Sij = xjW_i^G + B_i^G| AO
        AO -.->|Sij = xjW_i^G + B_i^G| AP
        AP -.->|Sij = xjW_i^G + B_i^G| AO
    
    subgraph Zoom-in view (for i-th head)
        AQ["Sigmoid"] --> AR["Sijk"]
        AQ["SijL"] --> AR["SijL"]
        AQ["Sij = xjW_i^G + B_i^G"] --> AN
        AN -.->|Sij = xjW_i^G + B_i^G| AO
    
    subgraph Zoom-in view (for i-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i^G + B_i^G"]
        AN -.->|Sij = xjW_i^G + B_i^G| AO
    
    subgraph Zoom-in view (for i-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i^G + B_i^G"]
        AN -.->|Sij = xjW_i^G + B_i^G| AO
    
    subgraph Zoom-in view (for ii-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i^G + B_i^G"]
        AN -.->|Sij = xjW_i^G + B_i^G| AO
    
    subgraph Zoom-in view (for ii-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i ^G + B_i^G"]
        AN -.->|Sij = xjW_i ^G + B_i^G| AO
    
    subgraph Zoom-in view (for iii-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i ^G + B_i^G"]
        AN -.->|Sij = xjW_i ^G + B_i^G| AO
    
    subgraph Zoom-in view (for iv-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i ^G + B_i^G"]
        AN -.->|Sij = xjW_i ^G + B_i^G| AO
    
    subgraph Zoom-in view (for iv-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i^G + B_i^G"]
        AN -.->|Sij = xjW_i ^G + B_i^G| AO
    
    subgraph Zoom-in view (for iv-th head)
        AR["Sijk"] --> AR["SijL"]
        AR["SijL"] --> AR["Sij = xjW_i ^G + B_i^G"]
        AN -.->|Sij = xjW_i ^G + B_i^G| AO
    
    end
    
    style Zoom-in view fill:#f9f,stroke:#333,stroke-width:2px
```
</details>

Figure 2. Overall framework of the proposed method. The right side of the figure shows how each input token assigns weights to both the prefixes and the adapter. Specifically, we compute prefix scores for each input token using a RePA module, and then convert those scores into weights with the CondAct module. We also set an upper limit on the total weight that can be assigned to the prefixes. Any weight beyond this limit is applied to the adapter’s output. This design enables token-level weighting for the adapter without the need for a router.

A straightforward way to address this issue is to learn a separate query-key projection matrix for each task, dedicated exclusively to computing the attention scores of the prefix. Although this approach improves performance in practice, the large number of additional parameters, together with the limited data in downstream tasks, often leads to suboptimal results. Instead, we propose a reparameterized prefix attention (RePA), which simplifies the query-key dot product process into a single affine transformation to compute the prefix scores. Specifically, for the i-th attention head, the original attention module computes the prefix score through $S _ { X P } ^ { ( i ) } = Q _ { i } ( K _ { P _ { i } } ) ^ { \top } = ( X W _ { i } ^ { Q } + \mathbf { 1 } _ { m } b _ { Q } ^ { \top } ) ( P _ { K } W _ { i } ^ { K } + \mathbf { 1 } _ { L } b _ { K } ^ { \top } ) ^ { \top }$ . This expression can be rewritten by separating the terms that depend on the input token matrix X from those that remain constant, yielding the following equation:

$$
\begin{array}{l} S _ {X P} ^ {(i)} = X \left[ W _ {i} ^ {Q} (P _ {K} W _ {i} ^ {K}) ^ {\top} + W _ {i} ^ {Q} b _ {K} \mathbf {1} _ {L} ^ {\top} \right] \tag {4} \\ + \left[ \mathbf {1} _ {m} b _ {Q} ^ {\top} (P _ {K} W _ {i} ^ {K}) ^ {\top} + \left(b _ {Q} ^ {\top} b _ {K}\right) \mathbf {1} _ {m \times L} \right]. \\ \end{array}
$$

We can group the terms into two matrices, arriving at RePA:

$$
S _ {X P} ^ {(i)} = X W _ {i} ^ {G} + B _ {i} ^ {G}. \tag {5}
$$

Note that training $W _ { i } ^ { G }$ and $B _ { i } ^ { G }$ effectively unifies the original attention projection parameters and the learnable prefix key $P _ { K }$ into a single reparameterized form [26]:

$$
W _ {i} ^ {G} \approx W _ {i} ^ {Q} \left(P _ {K} W _ {i} ^ {K}\right) ^ {\top} + W _ {i} ^ {Q} b _ {K} \mathbf {1} _ {L} ^ {\top}, \tag {6}
$$

$$
B _ {i} ^ {G} \approx \mathbf {1} _ {m} b _ {Q} ^ {\top} (P _ {K} W _ {i} ^ {K}) ^ {\top} + (b _ {Q} ^ {\top} b _ {K}) \mathbf {1} _ {m \times L}. \tag {7}
$$

By training these composite parameters, $W _ { i } ^ { G }$ and $B _ { i } ^ { G }$ , we can better capture task relevance by learning task-specific projections applied exclusively to the prefixes, while preserving pretrained knowledge by leaving the original projection matrices unchanged. Consequently, as illustrated in Fig. 3, our RePA assigns higher attention scores to tokens that are directly relevant to each task, thereby facilitating more effective adaptation to downstream tasks [7, 52].

![](images/c5e7292a8241617e8f38cfa58f2b43ba443dc9e17c053229672f3f20ec2a1507.jpg)

<details>
<summary>text_image</summary>

Aircraft
Oxford Pets
Original Image
Attention: (QK^T)
Ours: (XW^G + B^G)
</details>

Figure 3. Comparison of prefix score maps between attention and our method on the Aircraft and Oxford Pets images. Our approach produces more task-relevant scores.

# 4.2. Conditional Activation

Applying the above method, we obtain an enhanced score matrix S\_{XP}^{(i)} . $S _ { X F } ^ { ( i ) }$ However, prior work [32] has demonstrated that concatenating $S _ { X P } ^ { ( i ) }$ with the original attention score $S _ { X X } ^ { ( i ) }$ before applying the softmaained knowledge encoded in ction disrupts the, leading to signif-$S _ { X X } ^ { ( i ) }$ icant degradation in zero-shot performance. Alternatively, DIKI [32] proposes applying a separate softmax to $S _ { X P } ^ { ( i ) }$ While this approach avoids interference with the pretrained scores, it imposes a fixed total weight across all input tokens, hindering dynamic adjustment of the additional information for each token, thereby limiting token-wise modulation.

To overcome these limitations, we introduce Conditional Activation (CondAct), a novel alternative to the conventional softmax activation. CondAct processes the score matrix $S _ { X P }$ through two conditional operations, conditional normalization and conditional filtering, applied after a sigmoid function $\sigma ( \cdot )$ . Let $s _ { i j k }$ denote the score between the j-th input token and the k-th prefix, corresponding to the $( j , k ) \ – \mathrm { t h }$ element of the score matrix $S _ { X P } ^ { ( i ) }$ . The corresponding prefix weight $\widetilde { g } _ { i j k }$ is computed via the following procedure:

(Step 1: Conditional Norm)

$$
g _ {i j k} = \left\{ \begin{array}{l l} \frac {\sigma (s _ {i j k})}{\sum_ {k = 1} ^ {L} \sigma (s _ {i j k})}, & \text { if   } \sum_ {k = 1} ^ {L} \sigma (s _ {i j k}) \geq 1, \\ \sigma (s _ {i j k}), & \text { otherwise. } \end{array} \right. \tag {8a}
$$

(Step 2: Conditional Filtering)

$$
\widetilde {g} _ {i j k} = g _ {i j k} \cdot \mathbb {I} \big (g _ {i j k} \geq \text { cutoff } \big). \tag {8b}
$$

Here, I(·) is an indicator function that returns 1 if the condition is met and 0 otherwise.

The conditional normalization step in Eq. (8a) enables the total sum of prefix weights, $\textstyle \sum _ { k = 1 } ^ { L } g _ { i j k }$ , to flexibly range up to one—allowing the model to dynamically adjust the influence of each input token. This upper bound prevents taskspecific information from becoming overly dominant, a common issue when using a sigmoid function alone [29]. Consequently, CondAct mitigates forward forgetting in VLMs and enhances zero-shot generalization in CL tasks.

Next, the conditional filtering step in Eq. (8b) removes prefixes that are less relevant to the current task. Empirically, we observe that the attention scores between the [CLS] token and each prefix follow a Gaussian distribution. Leveraging this property, we determine the cutoff value dynamically for each sample. Specifically, for each prefix, we compute the mean and variance of its attention scores with the [CLS] token across the training data of task t and model its distribution as $\mathcal { N } ( \mu _ { t } , \sigma _ { t } ^ { 2 } )$ . Given a test sample, we evaluate Ours (??෨ )the log probability density of the observed attention score $s _ { i , \mathrm { c l s } , k }$ under this distribution and transform it with a sigmoid function to obtain a likelihood score, as in [32]. The cutoff for the k-th prefix is defined as one minus this score:

$$
\operatorname{cutoff} _ {i j k} = 1 - \sigma \left(\log \varphi (s _ {i, \mathrm{cls}, k}; \mu_ {t}, \sigma_ {t} ^ {2})\right). \tag {9}
$$

If the likelihood score is sufficiently high, we set the cutoff to zero so that the corresponding prefix fully contributes to the output. Since each prefix exhibits distinct activation patterns across input tokens, this prefix-wise adaptive cutoff enables token-wise dynamic filtering conditioned on task-specific likelihoods, allowing the model to selectively suppress irrelevant prefixes. For the i-th head, the additional information derived from the prefixes is computed as follows:

$$
O _ {\text { prefix } _ {i}} = \widetilde {G} _ {i}   P _ {V _ {i}} \quad \text { with } \quad \widetilde {G} _ {i} = (\widetilde {g} _ {i j k}) _ {j, k} \in \mathbb {R} ^ {m \times L}. \tag {10}
$$

# 4.3. Residual Weighting Mechanism

Our proposed CondAct dynamically adjusts the total weight of the prefixes for each token, constraining it within an upper limit of one to preserve zero-shot performance. However, strictly enforcing this limit may restrict further improvements, as tokens with higher task relevance often require stronger adaptation. To address this, we introduce the Residual Weighting Mechanism (RWM), which selectively applies the adapter output to tokens that demand additional adaptation. For parameter efficiency, we employ LoRA as the adapter [45] and follow the structure proposed in [33], where a shared down-projection matrix D is used across tasks and a task-specific up-projection $U _ { t }$ is assigned to each task t. To further adapt this design for CL, we freeze the shared projection, initializing it with the top-k left singular vectors of the value projection matrix $W _ { i } ^ { V }$ as [25, 35]. This constraint ensures that the adapter is fine-tuned within the row space of $D ,$ , effectively leveraging pretrained knowledge to complement the learned prefixes [19]. The adapter’s output, denoted as $\mathcal { E } _ { i } ^ { t } ( X )$ , is then weighted using RWM:

$$
O _ {\text { adapter } _ {i}} = \Delta_ {i} \odot \mathcal {E} _ {i} ^ {t} (X),
$$

$$
\Delta_ {i} = \text { concat } \left[ \Delta_ {i j} \right] _ {j = 1} ^ {m}, \tag {11}
$$

$$
\Delta_ {i j} = \max \left(0, \sum_ {k = 1} ^ {L} \sigma \left(s _ {i j k}\right) - 1\right),
$$

where \odot denotes element-wise multiplication.

<table><tr><td>Method</td><td>Extra data</td><td>Params.</td><td>Trans.</td><td>Avg.</td><td>Last</td><td>Mean</td></tr><tr><td>Zero-shot</td><td>-</td><td>-</td><td>69.4</td><td>65.3</td><td>65.3</td><td>66.7</td></tr><tr><td>ZSCL [49]</td><td>√</td><td>149.6 M</td><td>68.1</td><td>75.4</td><td>83.6</td><td>75.7</td></tr><tr><td>DIKI [32]</td><td>×</td><td>1.8 M</td><td>68.7</td><td>76.3</td><td>85.1</td><td>76.7</td></tr><tr><td>MoE-Adapter [45]</td><td>×</td><td>59.6 M</td><td>68.9</td><td>76.7</td><td>85.0</td><td>76.9</td></tr><tr><td>GIFT [40]</td><td>√</td><td>149.6 M</td><td>69.3</td><td>77.3</td><td>86.0</td><td>77.5</td></tr><tr><td>Ours†</td><td>×</td><td>4.6 M</td><td>70.0</td><td>78.6</td><td>87.6</td><td>78.7</td></tr><tr><td>Ours</td><td>×</td><td>30.8 M</td><td>70.4</td><td>79.3</td><td>88.3</td><td>79.3</td></tr></table>

Table 1. Comparison of various SOTA methods on MTIL Order I benchmark in terms of “Transfer”, “Average”, and “Last” scores (%). Best and second-best results are highlighted in bold and underline, respectively.

<table><tr><td>RePA</td><td>CondAct</td><td>RWM</td><td>Trans.</td><td>Avg.</td><td>Last</td></tr><tr><td>-</td><td>-</td><td>-</td><td>68.1</td><td>76.6</td><td>85.9</td></tr><tr><td>√</td><td>-</td><td>-</td><td>68.0</td><td>76.8</td><td>86.4</td></tr><tr><td>-</td><td>√</td><td>-</td><td>69.5</td><td>77.8</td><td>86.8</td></tr><tr><td>√</td><td>√</td><td>-</td><td>69.9</td><td>78.9</td><td>87.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Table 3. Ablation study on the MTIL benchmark evaluating each module’s contribution.

If the sum of prefix weights, $\scriptstyle \sum _ { k = 1 } ^ { L } \sigma ( s _ { i j k } )$ , does not exceed one, the corresponding element of $\Delta _ { i }$ becomes zero, meaning that the adapter does not contribute to that token. In this way, RWM leverages the residual prefix weights as adaptive scaling factors, enabling each token to receive additional information proportional to its remaining adaptation demand. Through this combination of prefixes and adapters, we achieve a balance between the aggressive modification introduced by the prefix and the conservative update performed by the adapter. For each head i, the output of the proposed DPW module, denoted as $O _ { R _ { i } }$ is computed as follows:

$$
O _ {R _ {i}} = O _ {\text { prefix } _ {i}} + O _ {\text { adapter } _ {i}}. \tag {12}
$$

# 5. Experiments

# 5.1. Experimental Setting

Benchmarks. We evaluate our method in two domain-class incremental settings: Multi-domain Task Incremental Learning (MTIL) [49] and Open-Domain Continual Learning (ODCL-CIL) [18]. MTIL uses task IDs for task-specific classification, while ODCL-CIL classifies across all seen classes without task IDs. Both benchmarks include 11 datasets from various domains, covering 1201 classes.

Metrics. We evaluate our method on both the MTIL and ODCL-CIL benchmarks using the metrics proposed in [18, 49]: Transfer, Avg., and Last. The Transfer score measures a model’s ability to generalize to unseen data through zero-shot evaluation, and it is used to quantify forward forgetting in the context of CL for VLMs [49]. Because task IDs are used when evaluating zero-shot capabilities in both benchmarks, the Transfer score is identical across MTIL and ODCL-CIL. The Last score represents the average performance across all tasks at the end of CL, whereas the Avg. score measures the average accuracy across all datasets and training stages.

<table><tr><td>Method</td><td>Trans.</td><td>Avg.</td><td>Last</td><td>Mean</td></tr><tr><td>ZSCL [49]</td><td>68.0</td><td>71.8</td><td>77.6</td><td>72.5</td></tr><tr><td>MoE-Adapter [45]</td><td>69.1</td><td>66.2</td><td>66.9</td><td>67.4</td></tr><tr><td>CoLeCLIP [18]</td><td>68.8</td><td>73.7</td><td>79.7</td><td>74.1</td></tr><tr><td>DPeCLIP [22]</td><td>69.1</td><td>76.1</td><td>84.6</td><td>76.6</td></tr><tr><td>Ours†</td><td> $\underline{70.0}$ </td><td> $\underline{77.9}$ </td><td> $\underline{85.8}$ </td><td> $\underline{77.9}$ </td></tr><tr><td>Ours</td><td> $\underline{70.4}$ </td><td> $\underline{78.6}$ </td><td> $\underline{86.6}$ </td><td> $\underline{78.5}$ </td></tr></table>

Table 2. Comparison of various SOTA methods on ODCL-CIL benchmark in terms of “Transfer,” “Average,” and “Last” scores (%).

Compared Methods. We compare our method against various state-of-the-art (SOTA) approaches, including prompt-based, adapter-based, and full fine-tuning methods. For prompt-based learning, we compare CoLeCLIP [18], DIKI [32] and DPeCLIP [22]. For adapter-based learning, we compare MoE-Adapters [45]. For full fine-tuning, we compare ZSCL [49] and GIFT [40]. In addition to our default model (denoted Ours), we also introduce a parameterefficient variant, Ours†, which reduces the number of trainable parameters. Ours†computes the prefix score for each head using only its corresponding sub-dimension $( \mathrm { i } . \mathrm { e } . , d / h ^ { \prime } )$ instead of the full embedding, reducing the number of parameters by a factor of h′. We further apply LoRA with a lower rank for additional efficiency. All experiments follow the baseline setup [32], with details provided in the Appendix.

# 5.2. Main Results.

Performance on MTIL. Tab 1 presents a comparison between the proposed method and several SOTA approaches on the MTIL benchmark. For the sequence of training tasks, we follow the predefined Order I configuration described in [49], where tasks are arranged alphabetically. Detailed results, including Order II, are provided in the Appendix. Overall, the results show that both Ours and Ours† outperform competing methods across all evaluation metrics. In particular, they surpass the second-best approach, GIFT [40], which performs full fine-tuning of the entire CLIP model using additional data generated by an auxiliary diffusion model. Compared to PEFT methods, such as MoE-Adapter [45] and DIKI [32], our approach consistently achieves superior performance while maintaining a similarly low number of trainable parameters. The strong performance on both Transfer and Last scores indicates that our method not only adapts effectively to new tasks but also preserves previously learned knowledge, underscoring its suitability for CL.

Performance on ODCL-CIL. We evaluate our method on class-incremental setting using the ODCL-CIL benchmark, as presented in Tab. 2. Compared with the MTIL benchmark, the ODCL-CIL scenario is more challenging because task IDs are unavailable during inference. Following the baseline method [32], we adopt its task-identification strategy, which estimates the likelihood of the current image feature under each task distribution and selects the parameters associated with the highest-likelihood task. To avoid potential advantages from batch voting [50], we further evaluate our method using both the default batch size (e.g., 256) and a batch size of one, reporting the lower score. Across all evaluation metrics, our method consistently achieves SOTA performance in this class-incremental setting.

![](images/7f9cd65270557e755c70a2bf03ce0e8708bc231b1f3b1b76f26515ececdddb69.jpg)

<details>
<summary>text_image</summary>

Base token
Air One
</details>

![](images/4b8ce92ff222eef414d6b4d1d4018adf9d13b1b2cfdd162542a88c43435ad386.jpg)

<details>
<summary>bar</summary>

| Token Type | Before query projection | After query projection |
| :--- | :--- | :--- |
| Base token | Sim=0.34 | Sim=0.37 |
| Task-relevant token | Sim=0.02 | Sim=0.47 |
| Task-irrelevant token | Dataset: Aircraft | |
</details>

Figure 4. Cosine similarities comparison between the base token and both the task-relevant and task-irrelevant tokens, shown before and after query projection. 

<table><tr><td>Method</td><td>Params.</td><td>Trans.</td><td>Avg.</td><td>Last</td></tr><tr><td> $QK^{\top}$  (freeze  $W^{Q}, W^{K}$ )</td><td>2.7 M</td><td>68.7</td><td>76.4</td><td>85.2</td></tr><tr><td> $QK^{\top}$  (train  $W^{Q}, W^{K}$ )</td><td>228.0 M</td><td>68.9</td><td>76.8</td><td>85.9</td></tr><tr><td> $QW_{G} + B_{G}$ </td><td>30.8 M</td><td>69.7</td><td>78.6</td><td>87.6</td></tr><tr><td> $XW_{G} + B_{G}$  (Ours)</td><td>30.8 M</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Table 4. Comparison between RePA and the traditional attention mechanism.

<table><tr><td>Sigmoid</td><td>CondNorm</td><td>Filtering</td><td>Trans.</td><td>Avg.</td><td>Last</td></tr><tr><td>-</td><td>-</td><td>-</td><td>68.0</td><td>76.8</td><td>86.4</td></tr><tr><td>√</td><td>-</td><td>-</td><td>68.6</td><td>78.3</td><td>88.2</td></tr><tr><td>√</td><td>√</td><td>-</td><td>69.9</td><td>79.0</td><td>88.2</td></tr><tr><td>√</td><td>√</td><td>√</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Table 5. Ablation study evaluating the impact of different components within the CondAct.

![](images/68874e301efd94d76101cb9dccefb550fda05e6044089aef1aa32e8cfffeac17.jpg)

<details>
<summary>text_image</summary>

Aircraft tokens (task 1)
OxfordPet tokens (task 9)
(a) Before query projection
(b) After query projection
</details>

Figure 5. UMAP visualization of token embeddings from the Aircraft and OxfordPet datasets, illustrating their distributions before and after applying query projection. 

<table><tr><td>Method</td><td>Params.</td><td>Trans.</td><td>Avg.</td><td>Last</td></tr><tr><td>Baseline</td><td>19.8 M</td><td>69.9</td><td>78.9</td><td>87.9</td></tr><tr><td>Prefix length ×2</td><td>39.6 M</td><td>69.5</td><td>78.4</td><td>87.6</td></tr><tr><td>Learnable router</td><td>32.7 M</td><td>69.9</td><td>79.0</td><td>88.1</td></tr><tr><td>RWM (Ours)</td><td>30.8 M</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Aircraft tokens OxforTable 6. Comparison between the RWM and traditional routing mechanism.

# 5.3. Analysis

Ablation Study. Our framework consists of three components: RePA to compute prefix scores (Eq. (5)), CondAct to transform the prefix scores into weights (Eq. (8)), and RWM to incorporate the adapter (Eq. (11)). We evaluate each module on the MTIL benchmark in Tab. 3. First, by replacing the traditional query-key dot product with the RePA, we obtain a refined prefix score matrix S\_{XP} , leading to an improvement in the Last score. However, this component alone cannot overcome the inherent limitation of softmax, which forces scores to sum to one and thus restricts the optimal assignment of weight per token. As shown in the third row, replacing softmax with our CondAct improves overall performance by dynamically controlling the amount of information added to each token, and combining it with the refined scores from RePA yields additional gains by better capturing task-specific characteristics in the scoring process. This token-wise modulation not only improves performance on the trained tasks, but also facilitates effective knowledge transfer across tasks during CL and mitigates forward forgetting, as evidenced by the improved Transfer scores. Moreover, by leveraging the weights assigned to each prefix, our RWM enables adapters to add information only to tokens that are not sufficiently adapted by the prefixes, thereby effectively complementing the prefixes and leading to notable improvements in both the Last and Transfer scores.

Limitations of Pretrained Attention Projections in CL. To further investigate the limitations of existing attention mechanisms in prefix-tuning for CL, we conduct a comparative analysis of token embeddings before and after the query projection. As shown in Fig. 4, we extract tokens from a single image and label those from object regions (e.g., the body of an airplane) as task-relevant and those from background regions as task-irrelevant, then compute cosine similarities (i) among task-relevant tokens (purple arrow) and (ii) between task-relevant and task-irrelevant tokens (green arrow). Specifically, we find that before the query projection, the similarity between task-relevant and task-irrelevant tokens is merely 0.02. After the projection, however, this similarity increases substantially to 0.47, exceeding even the similarity among task-relevant tokens. This indicates that the query projection disrupts the model’s ability to maintain distinctions between task-relevant and task-irrelevant features. Consequently, as shown in Fig. 3, prefix struggles to route attention to the appropriate tokens. Notably, this effect is not limited to individual samples; it also appears at the task level in our CL setting. Fig. 5 illustrates UMAP visualizations of tokens from multiple samples across two tasks. Before the projection, tokens from different tasks are clearly separated. However, after projection, they collapse into overlapping regions, revealing diminished task separability. Such reduced separability increases task interference and weakens effective knowledge transfer across CL tasks [9]. Overall, these findings highlight inherent limitations in approaches that rely on pretrained attention projections [22, 32, 37, 38].

Analysis of RePA. To evaluate whether RePA can effectively replace the conventional attention mechanism, we conduct experiments summarized in Tab. 4. As shown in the second row, re-training the attention projection matrices leads to improved CL performance. However, this approach substantially increases the number of learnable parameters to

![](images/a4ad8dc7a45abe70cd5ebf8b9f48aac82f2ecf06c5f6c8e72200472eabdffdba.jpg)

<details>
<summary>bar</summary>

| Dataset     | Total Sum of Weights | Ratio (Prefix / Total) |
| ----------- | --------------------- | ---------------------- |
| Aircraft    | 0.68                  | 0.9                    |
| Caltech101  | 0.69                  | 0.88                   |
| CIFAR100    | 0.79                  | 0.85                   |
| DTD         | 0.76                  | 0.82                   |
| EuroSAT     | 1.45                  | 0.6                    |
| Flowers     | 0.12                  | 0.2                    |
| Food        | 0.11                  | 0.2                    |
| MNIST       | 0.20                  | 0.25                   |
| OxfordPet   | 0.13                  | 0.2                    |
| Cars        | 0.10                  | 0.25                   |
| SUN397      | 0.12                  | 0.2                    |
</details>

Figure 6. Weight distribution after training on five tasks. Green bars show the sum of prefixes and adapters weights, while purple lines indicate the ratio of prefix weights to the total.

228.0M, while the performance gain remains significantly lower than that achieved by RePA. In the third row, we preserve the original query projection process of the attention mechanism and apply RePA on top of it. Similar to the previous case, the performance remains inferior to that obtained when RePA is directly applied to the input tokens. These results highlight the intrinsic limitations of the attention mechanism and demonstrate that RePA serves as an effective and efficient alternative.

Analysis of CondAct. To assess the contribution of each component in the proposed CondAct mechanism, we conduct an ablation study summarized in Tab. 5. Consistent with recent findings [14], which report improved performance when combining softmax and sigmoid activations in prefix tuning, we observe that incorporating a sigmoid activation into RePA also yields notable gains. However, the sigmoid function tends to produce overly large total weights for input tokens [29], potentially distorting their pretrained representations and degrading Transfer performance. Our conditional normalization module (8a) mitigates this issue by maintaining the total prefix information within a stable range, leading to a substantial improvement in Transfer accuracy. Furthermore, the conditional filtering mechanism (8b), which leverages the distribution of prefix scores to assign prefix-wise cutoffs, enables dynamic and token-wise selection of informative prefixes. This adaptive filtering further enhances the Transfer score by refining how prefix information is selectively utilized across tokens.

Analysis of RWM. To assess the effectiveness of our RWM, we conduct experiments summarized in Tab. 6. We first observe that simply increasing the number of learnable parameters does not yield better performance. As shown in the second block, doubling the prefix length degrades performance, highlighting the limitation of performance gains achievable through prefix alone. The third block introduces an adapter with a learnable router, but the resulting improvements remain marginal, suggesting that the adapter is ineffective without an appropriate weighting mechanism. In contrast, our proposed RWM module enables more effective weighting, leading to consistent improvements in both the Transfer and Last scores. Empirically, we find that the adapter outputs become more orthogonal to the prefixes when trained with RWM, indicating that the adapter complements the prefix by extending beyond the subspace spanned by a fixed number of prefix vectors.

![](images/84fb1b448712283f8bbcee50faa405dc4cfbf4f94a094850daa9474cbba67bf8.jpg)

<details>
<summary>line</summary>

| Learnable parameters (M) | Ours (Mean Performance %) | DIKI (Mean Performance %) |
| ------------------------ | -------------------------- | -------------------------- |
| 1M                       | 77.6                       | 76.5                       |
| 2M                       | 78.3                       | 76.7                       |
| 3M                       | 78.4                       | 76.4                       |
| 5M                       | 78.7                       | 76.6                       |
</details>

Figure 7. Comparison of mean performance and the number of learnable parameters between Ours† and DIKI [32] under varying prefix lengths. Bold indicates configurations reported in Tab. 1. Our method consistently outperforms DIKI across all settings.

Analysis of Weight Distribution. Fig. 6 presents an analysis of the weights assigned to the prefixes and the adapters. We compute the average weights for each and visualize their total sums and relative ratios. This suggests that the model can distinguish between learned and unseen tasks, highlighting the suitability of our method for CL scenarios for VLMs. Moreover, we observe that trained tasks rely more heavily on prefix information, whereas zero-shot tasks depend primarily on the adapter. This adaptive weighting mechanism represents a core property of our framework: when the model encounters distributions that deviate from the training data, it decreases reliance on the prefixes and activates the adapter, which is designed to preserve generalizable knowledge.

Parameter Efficiency. Fig. 7 shows how performance scales with the number of learnable parameters for both our method and the baseline DIKI [32]. We control the parameter count by varying the number of prefixes L in each method. The results demonstrate that our approach consistently outperforms DIKI, even when using fewer learnable parameters.

# 6. Conclusion

In this work, we propose a method for CL of VLMs that overcomes the limitations of attention-based weighting by learning task-aware weights for prefixes and adapters. Specifically, we introduce a novel mechanism that evaluates the task relevance of each token and assigns corresponding weights accordingly. As prefix-tuning and adapters have recently gained popularity, we believe our method can be extended to improve their effectiveness beyond the CL setting.

# 7. Acknowledgments

This work was partially supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT) of the Korean government (RS-2024-00341749), and Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by MSIT (RS-2023-00259934, RS-2025-02283048).

# References

[1] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. Memory aware synapses: Learning what (not) to forget. In Proceedings of the European Conference on Computer Vision (ECCV), pages 139–154, 2018. 2   
[2] Rahaf Aljundi, Klaas Kelchtermans, and Tinne Tuytelaars. Task-free continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11254–11263, 2019. 2   
[3] Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha, and Jonghyun Choi. Rainbow memory: Continual learning with a memory of diverse samples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8218–8227, 2021. 2   
[4] Arslan Chaudhry, Puneet K. Dokania, Thalaiyasingam Ajanthan, and Philip H. S. Torr. Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Proceedings of the European Conference on Computer Vision (ECCV), pages 532–547, 2018. 2   
[5] Pramit Dhar, Rajeev Ranjan Singh, Kuan-Chuan Peng, Ziyan Wu, and Rama Chellappa. Learning without memorizing. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5138–5146, 2019. 1   
[6] A. Douillard, A. Ramé, G. Couairon, and M. Cord. Dytox: Transformers for continual learning with dynamic token expansion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9285–9295, 2022. 2   
[7] Chin-Lun Fu, Zih-Ching Chen, Yun-Ru Lee, and Hung-Yi Lee. Adapterbias: Parameter-efficient token-dependent representation shift for adapters in nlp tasks. In Findings of the Association for Computational Linguistics, pages 2608–2621, 2022. 1, 4   
[8] Haoyuan Gao, Zicong Zhang, Yuqi Wei, Linglan Zhao, Guilin Li, Yexin Li, Linghe Kong, and Weiran Huang. Enhanced continual learning of vision-language models with model fusion. In ICLR 2025 Workshop. ICLR, 2025. 2   
[9] Naoki Hiratani. Disentangling and mitigating the impact of task similarity for continual learning. arXiv preprint, 2024. 1, 7   
[10] D. Jung, D. Han, J. Bang, and H. Song. Generating instancelevel prompts for rehearsal-free continual learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 11813–11823, 2023. 3   
[11] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-

modal prompt learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19113–19122, 2023. 1   
[12] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13): 3521–3526, 2017. 2   
[13] Matthias De Lange, Rahaf Aljundi, Mateusz Masana, Sophie Parisot, Xu Jia, Ales Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 44(7):3366–3385, 2021.   
[14] Minh Le, An Nguyen The, Huy Nguyen, Thien Trang Nguyen Vu, Huyen Trang Pham, Linh Ngo Van, and Nhat Ho. Mixture of experts meets prompt-based continual learning. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 8   
[15] Sang-Woo Lee, Jin-Hwa Kim, Jaehyun Jun, Jung-Woo Ha, and Byoung-Tak Zhang. Overcoming catastrophic forgetting by incremental moment matching. Advances in Neural Information Processing Systems (NeurIPS), 30, 2017. 2   
[16] Xilai Li, Yuezhou Zhou, Tianjun Wu, Richard Socher, and Caiming Xiong. Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting. In Proceedings of the International Conference on Machine Learning (ICML), pages 3925–3934. PMLR, 2019. 2   
[17] X. L. Li and P. Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00119, 2021. 3   
[18] Yukun Li, Guansong Pang, Wei Suo, Chenchen Jing, Yuling Xi, Lingqiao Liu, Hao Chen, Guoqiang Liang, and Peng Wang. Coleclip: Open-domain continual learning via joint task prompt and vocabulary learning. arXiv preprint, 2024. 1, 2, 3, 6, 7   
[19] Yan-Shuo Liang and Wu-Jun Li. Inflora: Interference-free low-rank adaptation for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23638–23647, 2024. 2, 5   
[20] Wenzhuo Liu, Fei Zhu, Longhui Wei, and Qi Tian. C-clip: Multimodal continual learning for vision-language model. In International Conference on Learning Representations (ICLR), 2025. 1, 2, 3   
[21] Yaoyao Liu, Yuting Su, An-An Liu, Bernt Schiele, and Qianru Sun. Mnemonics training: Multi-class incremental learning without forgetting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12245–12254, 2020. 2, 6   
[22] Yadong Lu, Shitian Zhao, Boxiang Yun, Dongsheng Jiang, Yin Li, Qingli Li, and Yan Wang. Boosting open-domain continual learning via leveraging intra-domain category-aware prototype. arXiv preprint, 2024. 1, 3, 6, 7   
[23] Zilin Luo, Yaoyao Liu, Bernt Schiele, and Qianru Sun. Classincremental exemplar compression for class-incremental learning. In Proceedings of the IEEE/CVF Conference on

Computer Vision and Pattern Recognition (CVPR), pages 11371–11380, 2023. 2   
[24] Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7765–7773, 2018. 2   
[25] Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. In Advances in Neural Information Processing Systems, 2024. 5   
[26] Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, and Christos Thrampoulidis. On the role of attention in prompttuning. arXiv preprint arXiv:2306.03435, 2023. 4   
[27] Xu Pan, Aaron Philip, Ziqian Xie, and Odelia Schwartz. Dissecting query-key interaction in vision transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2024. Spotlight Presentation. 1   
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. arXiv preprint, 2021. 1, 2   
[29] Jason Ramapuram, Federico Danieli, Eeshan Dhekane, Floris Weers, Dan Busbridge, Pierre Ablin, Tatiana Likhomanenko, Jagrit Digani, Zijin Gu, Amitis Shidani, and Russ Webb. Theory, analysis, and best practices for sigmoid self-attention. arXiv preprint, 2025. 5, 8   
[30] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H. Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2001–2010, 2017. 1, 6   
[31] James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogerio Feris, and Zsolt Kira. Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. Accepted. 2, 3   
[32] Longxiang Tang, Zhuotao Tian, Kai Li, Chunming He, Hantao Zhou, Hengshuang Zhao, Xiu Li, and Jiaya Jia. Mind the interference: Retaining pre-trained knowledge in parameter efficient continual learning of vision-language models. In European Conference on Computer Vision (ECCV), pages 346–365. Springer, 2024. 1, 2, 3, 5, 6, 7, 8, 4   
[33] Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, and Cheng zhong Xu. Hydralora: An asymmetric lora architecture for efficient fine-tuning. In Advances in Neural Information Processing Systems, 2024. 5, 2   
[34] Feng Wang, Jieru Mei, and Alan Yuille. Sclip: Rethinking self-attention for dense vision-language inference. In Proceedings of the European Conference on Computer Vision (ECCV), 2024. 1   
[35] Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen, and Yun Chen. Milora: Harnessing minor singular components for parameter-efficient llm finetuning. arXiv preprint arXiv:2406.09044, 2024. 5

[36] Yabin Wang, Zhiwu Huang, and Xiaopeng Hong. S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 2, 3   
[37] Zifeng Wang, Zizhao Zhang, Sayna Ebrahimi, Ruoyu Sun, Haohan Zhang, Ching-Yao Lee, Xinlei Ren, Guodong Su, Vincent Perot, Jennifer Dy, et al. Dualprompt: Complementary prompting for rehearsal-free continual learning. In European Conference on Computer Vision (ECCV), pages 631–648. Springer, 2022. 3, 7   
[38] Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 139–149, 2022. 2, 3, 7   
[39] Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, and Ludwig Schmidt. Robust fine-tuning of zeroshot models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 6   
[40] Bin Wu, Wuxuan Shi, Jinqiao Wang, and Mang Ye. Synthetic data is an elegant gift for continual vision-language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 6, 4, 7   
[41] Yinghui Xing, Qirui Wu, De Cheng, Shizhou Zhang, Guoqiang Liang, Peng Wang, and Yanning Zhang. Dual modality prompt tuning for vision-language pre-trained model. arXiv preprint, 2022. 3   
[42] Qingsen Yan, Dong Gong, Yuhang Liu, Anton van den Hengel, and Javen Qinfeng Shi. Learning bayesian sparse networks with full experience replay for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 109–118, 2022. 2   
[43] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3014–3023, 2021. 2   
[44] Dianzhi Yu, Xinni Zhang, Yankai Chen, Aiwei Liu, Yifei Zhang, Philip S. Yu, and Irwin King. Recent advances of multimodal continual learning: A comprehensive survey. arXiv preprint, 2024. 1   
[45] Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Ping Hu, Dong Wang, Huchuan Lu, and You He. Boosting continual learning of vision-language models via mixture-of-experts adapters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 23219–23230, 2024. 1, 2, 3, 5, 6, 4, 7   
[46] Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, and Xinchao Wang. Task residual for tuning vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 3   
[47] Yu-Chu Yu, Chi-Pin Huang, Jr-Jen Chen, Kai-Po Chang, Yung-Hsuan Lai, Fu-En Yang, and Yu-Chiang Frank Wang.

Select and distill: Selective dual-teacher knowledge transfer for continual learning on vision-language models. In European Conference on Computer Vision (ECCV). Springer, 2024. 2   
[48] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In Proceedings of the International Conference on Machine Learning (ICML), pages 3987–3995. PMLR, 2017. 2   
[49] Zangwei Zheng, Mingyuan Ma, Kai Wang, Ziheng Qin, Xiangyu Yue, and Yang You. Preventing zero-shot transfer degradation in continual learning of vision-language models. arXiv preprint, 2023. 1, 2, 3, 4, 6, 7   
[50] Da-Wei Zhou, Hai-Long Sun, Jingyi Ning, Han-Jia Ye, and De-Chuan Zhan. Continual learning with pre-trained models: A survey. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2024. 7   
[51] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16816–16825, 2022. 3   
[52] Nan Zhou, Jiaxin Chen, and Di Huang. ivpt: Improving task-relevant information sharing in visual prompt tuning by cross-layer dynamic connection. arXiv preprint, 2024. 1, 4

# Enhancing Continual Learning of Vision-Language Models via Dynamic Prefix Weighting

Supplementary Material

![](images/06d388615214725f40d60a10a2ee38129222b6255dfa2c35dbfc100f650f9002.jpg)

<details>
<summary>line</summary>

| Cutoff threshold | Transfer Score (%) | Last Score (%) |
| ---------------- | ------------------ | -------------- |
| 0.0              | 70.4               | 88.3           |
| 0.2              | 70.3               | 88.3           |
| 0.4              | 70.2               | 88.3           |
| 0.6              | 69.9               | 88.3           |
| 0.8              | 69.6               | 88.3           |
</details>

Figure A.1. Comparison between fixed cutoff and the proposed dynamic cutoff strategy.

# A. Additional Experimental Results

# A.1. Analysis of Conditional Filtering

Our CondAct module refines the prefix weight $g _ { i j k }$ through a conditional filtering process, rather than directly applying the outputs of conditional normalization. This additional step selectively suppresses the influence of less relevant prefixes by zeroing out weights that fall below a certain cutoff. Importantly, instead of using a fixed threshold, our method determines this cutoff dynamically, based on the likelihood of each prefix score under the trained task distributions. In Fig. A.1, we compare our dynamic cutoff strategy against several fixed thresholds. In the figure, dashed lines denote the results using our dynamic approach. Our method consistently outperforms fixed thresholds in both Transfer and Last scores, clearly demonstrating the effectiveness of the proposed filtering mechanism.

# A.2. Relationship Between Prefixes and the Adapter

To further understand the relationship between prefixes and the adapter, we analyze the cosine similarity between $O _ { \mathrm { p r e f i x } _ { i } }$ which is generated by a linear combination of prefix vectors, and the adapter output $O _ { \mathrm { a d a p t e r } _ { i } }$ . As shown in Fig. A.2, with our residual weighting mechanism (RWM), orthogonality emerges between the prefixes and the adapter. This shows that the adapter complements the prefixes by generating outputs that go beyond the subspace defined by the fixed prefix vectors, thus capturing information that linear combinations of those prefixes alone cannot express. However, as the figure shows, such orthogonality does not emerge when the adapter output is uniformly weighted to all tokens. This

![](images/01e08c7e39baa2ebc639c0a6ddde352df99c6a3e2c3950e29c21a3e3a48d4529.jpg)

<details>
<summary>bar</summary>

| Dataset     | Ours (RWM) | Uniform Weighting |
| ----------- | ---------- | ----------------- |
| Aircraft    | -0.01      | 0.17              |
| Caltech101  | -0.01      | 0.10              |
| CIFAR100    | -0.08      | 0.26              |
| DTD         | -0.05      | 0.26              |
| EuroSAT     | -0.11      | 0.30              |
| Flowers     | -0.01      | 0.18              |
| Food        | -0.02      | 0.14              |
| MNIST       | -0.12      | 0.35              |
| OxfordPet   | -0.01      | 0.12              |
| Cars        | -0.01      | 0.17              |
| SUN397      | -0.01      | 0.17              |
</details>

Figure A.2. Cosine similarity between the prefix output and the adapter output across datasets. Our RWM yields near-orthogonal representations, whereas uniform weighting produces substantially higher cosine similarity.

<table><tr><td>Weighting Method</td><td>Transfer</td><td>Avg.</td><td>Last</td></tr><tr><td> $\sum_{k} g_{ijk}$ </td><td>69.8</td><td>79.1</td><td>88.2</td></tr><tr><td> $\sum_{k} \widetilde{g}_{ijk}$ </td><td>70.1</td><td>79.1</td><td>88.0</td></tr><tr><td>RWM (Ours)</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Table A.1. Comparison of weighting strategies for adapter. $\sum _ { k } g _ { i j k }$ and $\sum _ { k } \widetilde { g } _ { i j k }$ use normalized and filtered prefix weights, respectively.

highlights that the proposed weighting mechanism plays a crucial role in enabling the complementary synergy between the adapter and the prefixes.

# A.3. Comparison with Alternative Weighting Mechanisms

Our weighting mechanism (RWM) applies token-wise weighting to the adapter’s output using the total prefix score $\textstyle \sum _ { k } \sigma ( s _ { i j k } )$ assigned to each input token. Notably, our method does not utilize the normalized attention weights $\sum _ { k } \widetilde { g } _ { i j k }$ , which are bounded within the interval [0,1] , but instead relies directly on the raw prefix scores prior to normalization. Rather than adopting an approach that directly transfers the weights assigned to the prefix onto the adapter, our method enables selective focusing by subtracting the prefix contribution from the overall activation score and using only the resulting values that remain non-negative. This design allows the adapter to selectively address tokens that are insufficiently adapted by the prefix alone. As illustrated in Tab. A.1, we compare our RWM with alternative weighting strategies that employ either the sum of normalized weights $\sum _ { k } g _ { i j k }$ or filtered weights $\sum _ { k } \widetilde { g } _ { i j k }$ , obtained through an additional filtering process. The results confirm the effectiveness of our approach, which uses the residual of the raw prefix scores.

<table><tr><td>Params.</td><td>Single D</td><td>Frozen D</td><td>SVD Init.</td><td>Transfer</td><td>Avg.</td><td>Last</td></tr><tr><td>21.6 M</td><td>-</td><td>-</td><td>-</td><td>69.7</td><td>79.0</td><td>88.1</td></tr><tr><td>11.0 M</td><td>√</td><td>-</td><td>-</td><td>58.4</td><td>50.9</td><td>25.7</td></tr><tr><td>11.0 M</td><td>√</td><td>√</td><td>-</td><td>69.7</td><td>78.8</td><td>88.0</td></tr><tr><td>11.0 M</td><td>√</td><td>√</td><td>√</td><td>70.4</td><td>79.3</td><td>88.3</td></tr></table>

Table A.2. Ablation study of adapter designs. D denotes the down projection matrix. The first row uses standard LoRA, and the second row applies HydraLoRA [33] with a shared, continually updated D across tasks. Params. represents the number of additional parameters.

# A.4. Analysis of Adapter Architecture

In this paper, we adopt the HydraLoRA [33] architecture as our LoRA-based adapter, which achieves parameter efficiency through a single down projection matrix D. To better suit the CL scenarios, we modify this architecture by initializing D using the top-k left singular vectors of the value projection matrix from the CLIP model, and freeze this matrix throughout the CL process. To evaluate the effectiveness of our proposed adapter architecture, we conduct an ablation study on various adapter designs, as summarized in Tab. A.2. The first row in the table corresponds to a baseline using the standard LoRA architecture, which exhibits lower performance compared to our modified adapter. The second row shows results obtained by directly applying the original HydraLoRA [33] architecture, where the D matrix is continually updated for each task. This continual update introduces catastrophic forgetting, resulting in significant performance degradation in the CL scenario. The third row corresponds to another variant, where the down projection matrix D is randomly initialized and then frozen. In this case, the row space of D may lie within a minor subspace, and restricting fine-tuning to this subspace [19] fails to improve performance. Taken together, our proposed adapter design enables fine-tuning directly on the principal subspace of the value projection matrix, effectively leveraging the pretrained knowledge from CLIP. This approach further supports prefixes containing task-specific knowledge and facilitates additional adaptation, thereby improving performance in CL scenarios.

# A.5. Computational Costs.

In Tab. A.3, we compare the inference time of our method with other PEFT approaches on the MTIL benchmark. Our method consistently achieves both performance improvements and lower inference time. This efficiency arises from the simplicity of our design: (i) replacing the quadratic query–key dot-product operations with affine transformations, and (ii) removing router operations by directly mapping the residual weights from CondAct to the adapter weights. Together, these design choices significantly reduce computational overhead, resulting in lower inference time.

<table><tr><td>Method</td><td>Inference Time</td><td>Mean (%)</td></tr><tr><td>MoE-Adapters</td><td>35m 47s</td><td>76.9</td></tr><tr><td>DIKI</td><td>3m 46s</td><td>76.7</td></tr><tr><td>Ours†</td><td>3m 8s</td><td>78.7</td></tr><tr><td>Ours</td><td>3m 16s</td><td>79.3</td></tr></table>

Table A.3. Comparison of methods in terms of throughput. Mean is computed as the average of Transfer, Avg., and Last. 

<table><tr><td>RePA</td><td>CondAct</td><td>RWM</td><td>Inference Time</td><td>Mean (%)</td></tr><tr><td>-</td><td>-</td><td>-</td><td>3m 44s</td><td>76.9</td></tr><tr><td>√</td><td>-</td><td>-</td><td>2m 56s</td><td>77.1</td></tr><tr><td>-</td><td>√</td><td>-</td><td>3m 29s</td><td>78.0</td></tr><tr><td>√</td><td>√</td><td>-</td><td>2m 53s</td><td>78.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>3m 8s</td><td>79.3</td></tr></table>

Table A.4. Analysis of how each component of our method contributes to the increase in inference time.

In Tab. A.4, we further analyze how each component of our method affects inference time. Starting from the baseline, incorporating RePA substantially reduces inference time by replacing attention with lightweight affine transformations. Using CondAct alone also provides a modest speedup by suppressing less relevant prefixes. When combined, RePA and CondAct further reduce inference time, making the model even faster than RePA alone. This improvement likely arises because RePA produces prefix scores that better capture task relevance, allowing CondAct to suppress more irrelevant prefixes and zero out additional prefix weights, thereby reducing effective computation. Finally, adding RWM slightly increases inference time due to the additional adapter branch, but it achieves the best overall performance while still remaining faster than DIKI.

# B. Implementation Details

# B.1. Experimental Setup

We follow the baseline setup of DIKI [32] for all our experiments. Specifically, for each task, we use the same handcrafted text prompts as DIKI, combining them with class names to construct the input to the text encoder. Following the baseline, we train the model for 10 epochs using a cosine learning rate scheduler and select the one with the highest validation accuracy, using a consistent data split throughout the process. To ensure a fair comparison, we adopt the CLIP ViT-B/16 [28] model, which is commonly used by other methods on the same benchmark.

Building upon this configuration, we minimize crossentropy loss between model predictions and ground truth labels during training. The learning rate is set to 1.25, and a batch size of 32 is used. The prefix length L is set to 8. The row rank dimension of the LoRA adapter is set to 64 in our default setting (Ours) and reduced to 4 in the parameter-efficient variant (Ours†). Both prefix and adapter modules are integrated into all 12 layers of the visual and text encoders. All experiments are conducted using a single NVIDIA 4090 GPU. For RePA, the bias matrix $B _ { i } ^ { G }$ is initialized to − for the visual branch and − for the text branch. The weight matrix $W _ { i } ^ { G }$ and the prefix vector $P _ { V }$ are initialized as orthogonal vectors across all tasks.

<table><tr><td>Head Adjustment</td><td>Transfer (%)</td><td>Avg (%)</td><td>Last (%)</td><td>Params.</td></tr><tr><td>Original Head Count (×1)</td><td>69.8</td><td>78.9</td><td>88.2</td><td>29.6 M</td></tr><tr><td>Reduced Head Count (×0.5)</td><td>70.1</td><td>79.1</td><td>88.1</td><td>21.0 M</td></tr><tr><td>Expanded Head Count (×2)</td><td>69.8</td><td>78.9</td><td>88.2</td><td>46.8 M</td></tr><tr><td>Adaptive Head Count (Ours)</td><td>70.4</td><td>79.3</td><td>88.3</td><td>30.8 M</td></tr></table>

Table B.1. Comparison of various head counts and our adaptive head-count selection method.

# B.2. Adaptive Head Count

In this paper, we propose a novel adaptive method for determining the number of heads $h ^ { \prime }$ for each task, where $h ^ { \prime }$ is a separate set of heads used exclusively in the prefix and adapter modules, distinct from the original number of heads h in CLIP. Specifically, we compute a task-specific scaling factor that captures the discriminability of the feature representations of each task and use it to scale the original number of heads to derive $h ^ { \prime } .$ . To obtain this factor, we first measure the inter-class and intra-class variances of the features for each task. For image features, let $\mathbf { v } _ { i , j }$ denote the normalized feature vector of the j-th sample in class i, and let $N _ { i }$ be the number of samples in class i. We compute the normalized class mean for each class i as:

$$
\hat {\mu} _ {i} = \frac {\mu_ {i}}{\| \mu_ {i} \|}, \quad \text { with } \quad \mu_ {i} = \frac {1}{N _ {i}} \sum_ {j = 1} ^ {N _ {i}} \mathbf {v} _ {i, j}.
$$

The inter-class variance for image features is defined as one minus the average cosine similarity between different class means:

$$
\text { inter\_variance } _ {\text { image }} = 1 - \frac {1}{K (K - 1)} \sum_ {i \neq j} \hat {\mu} _ {i} \cdot \hat {\mu} _ {j},
$$

where K is the total number of classes. The intra-class variance for image features is computed based on the average pairwise cosine similarity within each class:

$$
\text { intra\_variance } _ {\text { image }} = 1 - \frac {1}{K} \sum_ {i = 1} ^ {K} \left(\frac {1}{N _ {i} (N _ {i} - 1)} \sum_ {j \neq k} \mathbf {v} _ {i, j} \cdot \mathbf {v} _ {i, k}\right).
$$

For text features, assuming that each class i is represented by a normalized embedding $\mathbf { t } _ { i } ,$ the inter-class variance is similarly defined as:

$$
\text { inter\_variance } _ {\text { text }} = 1 - \frac {1}{K (K - 1)} \sum_ {i \neq j} \mathbf {t} _ {i} \cdot \mathbf {t} _ {j}.
$$

Next, we combine these variances into an easiness score, which quantifies task difficulty by jointly considering class inter-separability and within-class consistency, with an additional correction for the number of classes. The proposed easiness score is computed as follows:

$$
\text { easiness } = \frac {\frac {1}{2} \left(\text { inter\_variance } _ {\text { image }} + \text { inter\_variance } _ {\text { text }}\right)}{\text { intra\_variance } _ {\text { image }} + \left(\alpha / n _ {\text { cls }}\right)},
$$

where $n _ { \mathrm { c l s } }$ is the number of classes, and \alpha (set to 10) is used as a scaling factor to balance the units of the class count term. A higher score indicates well-separated and consistent class features, implying an easier classification task. This score is then divided by the model’s zero-shot accuracy (\protect \text {zs\_accuracy} ) to yield the composite scaling factor F for each task:

$$
F = \frac {\text { easiness }}{\text { zs\_accuracy }}.
$$

Dividing the easiness score by the zero-shot performance allows the model to assess how well its pretrained knowledge aligns with each task. When a task has a high easiness score but the model shows low zero-shot accuracy, this indicates that the task is considered to require more adaptation capacity, as the pretrained features are not sufficiently informative. The original number of heads h is scaled by $F$ and adjusted to the nearest power of two to maintain architectural consistency:

$$
h ^ {\prime} = \left\{ \begin{array}{l l} 1, & \text { if } h \times F = 0, \\ 2 ^ {\operatorname{round} (\log_ {2} (h \times F))}, & \text { otherwise }. \end{array} \right.
$$

As shown in Tab. B.1, this adaptive adjustment yields superior performance compared to simply increasing the number of parameters uniformly, highlighting the effectiveness of our adaptive approach.

# C. Details about Benchmark

# C.1. Datasets

Both MTIL and ODCL-CIL benchmarks were evaluated using two distinct dataset orders, followed by [49]. MTIL is designed for the task-incremental setting, while ODCL-CIL, also referred to as MCIL, targets the class-incremental setting. In Order-I, the datasets are arranged alphabetically as follows: Aircraft, Caltech101, CIFAR100, DTD, EuroSAT, Flowers, Food, MNIST, OxfordPet, StanfordCars, and SUN397. In contrast, Order-II lists the datasets in a random sequence: StanfordCars, Food, MNIST, OxfordPet, Flowers, SUN397, Aircraft, Caltech101, DTD, EuroSAT, and CIFAR100.

<table><tr><td>Method</td><td>Trans</td><td>Avg</td><td>Last</td><td>Mean</td></tr><tr><td>DIKI [32]</td><td>68.7</td><td>72.5</td><td>78.6</td><td>73.3</td></tr><tr><td>MoE-Adapter [45]</td><td>69.3</td><td>75.3</td><td>82.3</td><td>75.6</td></tr><tr><td>Ours</td><td>69.6</td><td>75.9</td><td>83.7</td><td>76.4 (+0.8)</td></tr></table>

Table C.1. Comparison with SOTA methods on 16-shot MTIL-FS benchmark. Results for DIKI and MoE-Adapter are based on our replications.

# C.2. Metrics

We further formulate the Transfer, Avg, and Last metrics, which were originally introduced in [49]. Let $p _ { j } ^ { ( i ) }$ denote the accuracy achieved on task j after the model has been trained on task i. Given a total of T tasks, these metrics are computed as follows.

The Transfer metric, which measures the model’s forward forgetting by evaluating its zero-shot performance after completing training on task j is defined as:

$$
\operatorname{Transfer} _ {j} = \frac {1}{j - 1} \sum_ {i = 1} ^ {j - 1} p _ {j} ^ {(i)}, \quad j = 2, 3, \dots , T.
$$

The Avg metric, representing the average performance across all tasks, is defined as:

$$
\operatorname{Avg} _ {j} = \frac {1}{T} \sum_ {i = 1} ^ {T} p _ {j} ^ {(i)}, \quad j = 1, 2, \dots , T.
$$

Finally, the Last metric, which reflects the final performance after training on all tasks, is given by:

$$
\operatorname{Last} _ {j} = p _ {j} ^ {(T)}, \quad j = 1, 2, \dots , T.
$$

# C.3. Few-Shot Setting

In contrast to the baseline few-shot setup used in DIKI [32], we conduct experiments across all datasets included in the MTIL benchmark. Specifically, DIKI excluded certain datasets, arguing that prompt-based CL methods fail to capture sufficient information from few-shot samples in some tasks. However, our method effectively addresses this limitation through a conditional filtering process that suppresses the influence of noisy prefixes while simultaneously leveraging adapters to supplement additional required information. For a fair comparison, we exclude GIFT [40], which utilizes additional synthetic data generated via diffusion models. Instead, we replicate MoE-Adapter [45], another PEFT-based method, alongside DIKI [32], adapting both to our experimental setting for comparison. A summary of the results is provided in Tab.C.1, with the complete table in Tab.E.1. In the few-shot setting, our method outperforms other SOTA methods, demonstrating non-trivial improvements. These results indicate that our approach performs well even in data-scarce environments.

# C.4. Detailed Results

Tab. E.2 shows the performance comparison with SOTA methods on the second-order setting of the MTIL benchmark. Our method consistently outperforms other approaches in the second-order setting as well. The complete results for both orders can be found in Tab. E.3 and Tab .E.4, respectively. The complete results for the ODCL-CIL setting are included in Tab. E.5.

# D. Theoretical Analysis of CondAct for Forward Forgetting

In this section, we provide a theoretical analysis demonstrating how our proposed CondAct mechanism mitigates forward forgetting by bounding the prefix-induced drift in token-level representations.

In the pretrained model, the token-level representation of the j-th input token at the i-th attention head is computed as

$$
h _ {i j} ^ {\text { orig }} = \sum_ {u = 1} ^ {m} \alpha_ {i j u} V _ {i u}, \quad \text { where } \sum_ {k = 1} ^ {m} \alpha_ {i j k} = 1.
$$

Here, $\alpha _ { i j k }$ denotes the attention weights, and $V _ { i u }$ denotes the corresponding projected value vectors. We define the prefix-induced representation as:

$$
p _ {i j} = \sum_ {k = 1} ^ {L} w _ {i j k} P _ {V _ {k}} ^ {(i)},
$$

where $P _ { V _ { k } } ^ { ( i ) }$ Vk is the value vector associated with the k-th prefix token, and $w _ { i j k }$ represents the attention weight between the j-th input token and k-th prefix token. These weights are obtained by applying a sigmoid gate to the attention score. Specifically, let $s _ { i j k }$ denote the attention score between the j-th input token and the k-th prefix token; then we consider two cases:

$$
w _ {i j k} = \left\{ \begin{array}{l l} \sigma (s _ {i j k}), & \text { without   CondAct,   Sigmoid - only. } \\ g _ {i j k} \text {   or   } \tilde {g} _ {i j k}, & \text { with   CondAct,   Eq.   (8). } \end{array} \right.
$$

Assumption (Bounded prefix values). There exists a constant $c _ { V } > 0$ such that

$$
\left\| P _ {V _ {k}} ^ {(i)} \right\| _ {2} \leq c _ {V} \quad \text {   for   all   prefix   tokens   } k.
$$

Proposition (Prefix-induced drift bound). For any token j at head i, the norm of the prefix-induced representation $p _ { i j }$ satisfies

$$
\left\| p _ {i j} \right\| _ {2} \leq L c _ {V} \quad (\text { without   CondAct }, \text { Sigmoid - only }),
$$

$$
\left\| p _ {i j} \right\| _ {2} \leq c _ {V} \quad (\text { with   CondAct,   Eq.   (8) }).
$$

![](images/3f4f223e7815ab65fefe4a773d6176a396aad7830a5ffa61d27873495d73f2cc.jpg)

<details>
<summary>histogram</summary>

| attention score | empirical density | Gaussian fit |
| --------------- | ----------------- | ------------ |
| -5.0            | 0.0               | 0.0          |
| -4.5            | 0.5               | 0.5          |
| -4.0            | 1.4               | 1.4          |
| -3.5            | 0.7               | 0.7          |
| -3.0            | 0.0               | 0.0          |
</details>

![](images/8a21d63070faf10504ed08b41576a3754468a7669f096a7371f160e0687d74c8.jpg)

<details>
<summary>histogram</summary>

| attention score | empirical density | Gaussian fit |
| --------------- | ----------------- | ------------ |
| -3.9            | 0.0               | 0.0          |
| -3.8            | 2.5               | 2.5          |
| -3.7            | 5.5               | 5.5          |
| -3.6            | 3.5               | 3.5          |
| -3.5            | 0.5               | 0.5          |
</details>

Figure C.1. Mean and variance of attention scores between prefix tokens and the [CLS] token measured on the Aircraft dataset. The left plot shows results from Layer 3, and the right plot shows results from Layer 9. The observed distributions closely match their Gaussian counterparts, supporting the validity of the proposed cutoff design.

In particular, CondAct reduces the worst-case prefix-induced drift per token from LcV to cV .

Proof. By the definition of prefix-induced representation,

$$
\left\| p _ {i j} \right\| _ {2} = \left\| \sum_ {k = 1} ^ {L} w _ {i j k} P _ {V _ {k}} ^ {(i)} \right\| _ {2} \leq \sum_ {k = 1} ^ {L} | w _ {i j k} | \left\| P _ {V _ {k}} ^ {(i)} \right\| _ {2},
$$

where we used the triangle inequality. In the Sigmoid-only setting, $w _ { i j k } = \sigma ( s _ { i j k } ) \in [ 0 , 1 ]$ , so $| w _ { i j k } | \le 1$ for all k. Together with the previous assumption, this yields

$$
\left\| p _ {i j} \right\| _ {2} \leq \sum_ {k = 1} ^ {L} \left\| P _ {V _ {k}} ^ {(i)} \right\| _ {2} \leq \sum_ {k = 1} ^ {L} c _ {V} = L c _ {V},
$$

We now consider the CondAct case. By construction (Eq. (8a)), CondAct applies conditional normalization such that the total prefix weight per token is explicitly bounded,

$$
\sum_ {k = 1} ^ {L} g _ {i j k} \leq 1, \quad g _ {i j k} \geq 0 \quad \text { for   all } i, j, k.
$$

In this case as well, under the same assumption as above, we obtain

$$
\left\| p _ {i j} \right\| _ {2} \leq \sum_ {k = 1} ^ {L} g _ {i j k} \left\| P _ {V _ {k}} ^ {(i)} \right\| _ {2} \leq c _ {V} \sum_ {k = 1} ^ {L} g _ {i j k} \leq c _ {V},
$$

![](images/777c7173b610531471c9a4b841f72e3c1bb8c998572eafb7137f1701c0c0fe22.jpg)

Corollary (Relative prefix-induced drift). Suppose the pretrained representation satisfies

$$
\left\| h _ {i j} ^ {\mathrm{orig}} \right\| _ {2} \geq c _ {h} > 0
$$

for some constant $c _ { h } .$ . Then the relative drift satisfies

$$
\frac {\left\| p _ {i j} \right\| _ {2}}{\left\| h _ {i j} ^ {\text { orig }} \right\| _ {2}} \leq \left\{ \begin{array}{l l} L \cdot \frac {c _ {V}}{c _ {h}}, & \text { without   CondAct }, \\ \frac {c _ {V}}{c _ {h}}, & \text { with   CondAct }. \end{array} \right.
$$

In summary, CondAct reduces the worst-case relative drift by a factor of up to L.

# E. Prefix Score Distribution

In Eq. (9), we leverage the empirical observation that the attention scores between prefix tokens and the [CLS] token follow a Gaussian distribution, which enables us to determine the distribution used for computing the cutoff. To validate this assumption, we measure the prefix scores on the Aircraft dataset at Layers 3 and 9, which we selected arbitrarily. As shown in Fig. C.1, the empirical distributions of the prefix scores at both layers closely align with the corresponding Gaussian fits, indicating that the Gaussian assumption provides a reasonable approximation. This observation supports the stability of our cutoff design.

<table><tr><td>Task</td><td>Method</td><td>Aircraft</td><td>Caltech101</td><td>CIFAR100</td><td>DTD</td><td>EuroSAT</td><td>Flowers</td><td>Food</td><td>MNIST</td><td>OxfordPet</td><td>Cars</td><td>SUN397</td><td>Average</td></tr><tr><td rowspan="12">MTIL-FS</td><td colspan="13">Transfer</td></tr><tr><td>DIKI [32]</td><td>-</td><td>92.7</td><td>68.6</td><td>44.1</td><td>47.8</td><td>69.9</td><td>86.1</td><td>58.4</td><td>89.0</td><td>65.1</td><td>65.2</td><td>68.7</td></tr><tr><td>MoE-Adapters [45]</td><td>-</td><td>88.4</td><td>68.2</td><td>42.9</td><td>53.5</td><td>70.1</td><td>88.5</td><td>63.3</td><td>89.0</td><td>64.7</td><td>65.1</td><td>69.3</td></tr><tr><td>Ours</td><td>-</td><td>92.9</td><td>68.9</td><td>45.2</td><td>53.7</td><td>71.1</td><td>86.4</td><td>60.4</td><td>89.7</td><td>66.0</td><td>65.8</td><td>69.6 (+0.3)</td></tr><tr><td colspan="13">Avg.</td></tr><tr><td>DIKI [32]</td><td>39.7</td><td>95.6</td><td>76.7</td><td>61.4</td><td>61.2</td><td>83.3</td><td>86.3</td><td>69.3</td><td>90.2</td><td>67.6</td><td>66.1</td><td>72.5</td></tr><tr><td>MoE-Adapters [45]</td><td>43.3</td><td>92.1</td><td>77.2</td><td>66.7</td><td>80.0</td><td>84.8</td><td>88.3</td><td>74.4</td><td>89.0</td><td>67.1</td><td>65.9</td><td>75.3</td></tr><tr><td>Ours</td><td>51.3</td><td>95.7</td><td>77.5</td><td>65.9</td><td>74.1</td><td>85.3</td><td>86.7</td><td>71.2</td><td>91.0</td><td>68.8</td><td>67.1</td><td>75.9 (+0.6)</td></tr><tr><td colspan="13">Last</td></tr><tr><td>DIKI [32]</td><td>39.6</td><td>95.8</td><td>78.5</td><td>68.0</td><td>67.0</td><td>94.4</td><td>86.4</td><td>88.3</td><td>93.5</td><td>78.6</td><td>74.3</td><td>78.6</td></tr><tr><td>MoE-Adapters [45]</td><td>43.3</td><td>92.6</td><td>79.2</td><td>75.6</td><td>95.1</td><td>97.1</td><td>88.1</td><td>93.8</td><td>89.1</td><td>78.0</td><td>73.7</td><td>82.3</td></tr><tr><td>Ours</td><td>51.4</td><td>96.0</td><td>79.4</td><td>73.8</td><td>90.2</td><td>97.4</td><td>87.0</td><td>91.8</td><td>93.9</td><td>83.4</td><td>76.6</td><td>83.7 (+1.4)</td></tr></table>

Table E.1. Comparison with SOTA methods on MTIL-FS benchmark (Order I) in terms of “Transfer”, “Average”, and “Last” scores (%). Best results are highlighted in bold. Ours† indicates our reduced-parameter variant.

<table><tr><td>Method</td><td>Extra data</td><td>Train Params.</td><td>Transfer</td><td>Avg.</td><td>Last</td><td>Mean</td></tr><tr><td>Zero-shot</td><td>-</td><td>-</td><td>65.4</td><td>65.3</td><td>65.3</td><td>65.3</td></tr><tr><td>LwF [21]</td><td>√</td><td>149.6 M</td><td>53.2</td><td>62.2</td><td>71.9</td><td>62.4</td></tr><tr><td>iCaRL [30]</td><td>√</td><td>149.6 M</td><td>50.9</td><td>56.9</td><td>71.6</td><td>59.8</td></tr><tr><td>WiSE-FT [39]</td><td>√</td><td>149.6 M</td><td>51.0</td><td>61.5</td><td>72.2</td><td>61.6</td></tr><tr><td>ZSCL [49]</td><td>√</td><td>149.6 M</td><td>64.2</td><td>74.5</td><td>83.4</td><td>74.0</td></tr><tr><td>DIKI [32]</td><td>×</td><td>1.8 M</td><td>64.4</td><td>74.5</td><td>85.5</td><td>74.8</td></tr><tr><td>MoE-Adapter [45]</td><td>×</td><td>59.6 M</td><td>64.3</td><td>74.7</td><td>84.1</td><td>74.4</td></tr><tr><td>GIFT [40]</td><td>√</td><td>149.6 M</td><td>65.9</td><td>75.7</td><td>85.3</td><td>75.6</td></tr><tr><td>Ours†</td><td>×</td><td>4.6 M</td><td>65.3</td><td>76.0</td><td>87.6</td><td>76.2 (+0.6)</td></tr><tr><td>Ours</td><td>×</td><td>30.8 M</td><td>65.7</td><td>76.4</td><td>88.1</td><td>76.7 (+1.1)</td></tr></table>

Table E.2. Comparison of SOTA methods on MTIL Order II.

<table><tr><td>Task</td><td>Method</td><td>Aircraft</td><td>Caltech101</td><td>CIFAR100</td><td>DTD</td><td>EuroSAT</td><td>Flowers</td><td>Food</td><td>MNIST</td><td>OxfordPet</td><td>Cars</td><td>SUN397</td><td>Average</td></tr><tr><td rowspan="21">MTIL</td><td colspan="13">Transfer</td></tr><tr><td>ZSCL [49]</td><td>-</td><td>86.0</td><td>67.4</td><td>45.4</td><td>50.4</td><td>69.1</td><td>87.6</td><td>61.8</td><td>86.8</td><td>60.1</td><td>66.8</td><td>68.1</td></tr><tr><td>DIKI [32]</td><td>-</td><td>92.9</td><td>69.0</td><td>43.2</td><td>48.2</td><td>67.4</td><td>85.2</td><td>63.0</td><td>87.9</td><td>63.8</td><td>66.2</td><td>68.7</td></tr><tr><td>MoE-Adapters [45]</td><td>-</td><td>87.9</td><td>68.2</td><td>44.4</td><td>49.9</td><td>70.7</td><td>88.7</td><td>59.7</td><td>89.1</td><td>64.5</td><td>65.5</td><td>68.9</td></tr><tr><td>GIFT [40]</td><td>-</td><td>88.5</td><td>69.8</td><td>46.0</td><td>49.4</td><td>68.5</td><td>87.1</td><td>69.9</td><td>88.9</td><td>57.7</td><td>67.7</td><td>69.3</td></tr><tr><td>Ours $\dagger$ </td><td>-</td><td>92.9</td><td>68.9</td><td>45.2</td><td>53.7</td><td>71.1</td><td>86.4</td><td>60.4</td><td>89.7</td><td>66.0</td><td>65.8</td><td>70.0 (+0.7)</td></tr><tr><td>Ours</td><td>-</td><td>92.9</td><td>69.0</td><td>45.3</td><td>54.2</td><td>71.1</td><td>86.2</td><td>63.8</td><td>89.3</td><td>65.7</td><td>66.5</td><td>70.4 (+1.1)</td></tr><tr><td colspan="13">Avg.</td></tr><tr><td>ZSCL [49]</td><td>45.1</td><td>92.0</td><td>80.1</td><td>64.3</td><td>79.5</td><td>81.6</td><td>89.6</td><td>75.2</td><td>88.9</td><td>64.7</td><td>68.0</td><td>75.4</td></tr><tr><td>DIKI [32]</td><td>45.1</td><td>95.5</td><td>83.1</td><td>64.8</td><td>79.9</td><td>83.5</td><td>87.0</td><td>76.2</td><td>89.6</td><td>67.0</td><td>67.1</td><td>76.3</td></tr><tr><td>MoE-Adapters [45]</td><td>50.2</td><td>91.9</td><td>83.1</td><td>69.4</td><td>78.9</td><td>84.0</td><td>89.1</td><td>73.7</td><td>89.3</td><td>67.7</td><td>66.9</td><td>76.7</td></tr><tr><td>GIFT [40]</td><td>51.9</td><td>93.9</td><td>81.4</td><td>67.7</td><td>80.3</td><td>82.8</td><td>89.3</td><td>80.6</td><td>90.3</td><td>63.1</td><td>68.9</td><td>77.3</td></tr><tr><td>Ours $\dagger$ </td><td>57.8</td><td>96.2</td><td>83.9</td><td>69.2</td><td>82.1</td><td>85.8</td><td>87.8</td><td>74.6</td><td>91.2</td><td>69.6</td><td>67.0</td><td>78.6 (+1.3)</td></tr><tr><td>Ours</td><td>60.4</td><td>96.2</td><td>84.5</td><td>70.8</td><td>82.4</td><td>85.8</td><td>87.7</td><td>76.8</td><td>90.9</td><td>69.4</td><td>67.6</td><td>79.3 (+2.0)</td></tr><tr><td colspan="13">Last</td></tr><tr><td>ZSCL [49]</td><td>40.6</td><td>92.2</td><td>81.3</td><td>70.5</td><td>94.8</td><td>90.5</td><td>91.9</td><td>98.7</td><td>93.9</td><td>85.3</td><td>80.2</td><td>83.6</td></tr><tr><td>DIKI [32]</td><td>45.2</td><td>95.7</td><td>86.3</td><td>72.9</td><td>98.0</td><td>97.0</td><td>89.2</td><td>99.4</td><td>94.2</td><td>81.6</td><td>76.6</td><td>85.1</td></tr><tr><td>MoE-Adapters [45]</td><td>49.8</td><td>92.2</td><td>86.1</td><td>78.1</td><td>95.7</td><td>94.3</td><td>89.5</td><td>98.1</td><td>89.9</td><td>81.6</td><td>80.0</td><td>85.0</td></tr><tr><td>GIFT [40]</td><td>47.9</td><td>95.6</td><td>82.8</td><td>75.1</td><td>97.3</td><td>94.2</td><td>91.7</td><td>99.2</td><td>94.2</td><td>87.0</td><td>80.9</td><td>86.0</td></tr><tr><td>Ours $\dagger$ </td><td>57.8</td><td>96.5</td><td>87.3</td><td>78.2</td><td>98.3</td><td>98.1</td><td>89.5</td><td>99.5</td><td>94.9</td><td>85.6</td><td>78.1</td><td>87.6 (+1.6)</td></tr><tr><td>Ours</td><td>60.5</td><td>96.6</td><td>87.9</td><td>80.4</td><td>98.5</td><td>98.1</td><td>89.4</td><td>99.6</td><td>95.1</td><td>86.3</td><td>78.8</td><td>88.3 (+2.3)</td></tr><tr><td>Task</td><td>Method</td><td>Cars</td><td>Food</td><td>MNIST</td><td>OxfordPet</td><td>Flowers</td><td>SUN397</td><td>Aircraft</td><td>Caltech101</td><td>DTD</td><td>EuroSAT</td><td>CIFAR100</td><td>Average</td></tr><tr><td rowspan="21">MTIL</td><td colspan="13">Transfer</td></tr><tr><td>ZSCL [49]</td><td>-</td><td>88.3</td><td>57.5</td><td>84.7</td><td>68.1</td><td>64.8</td><td>21.1</td><td>88.2</td><td>45.3</td><td>55.2</td><td>68.2</td><td>64.2</td></tr><tr><td>DIKI [32]</td><td>-</td><td>85.8</td><td>59.8</td><td>89.1</td><td>71.8</td><td>62.6</td><td>24.3</td><td>93.3</td><td>42.7</td><td>46.8</td><td>67.8</td><td>64.4</td></tr><tr><td>MoE-Adapters [45]</td><td>-</td><td>88.8</td><td>59.5</td><td>89.1</td><td>69.9</td><td>64.4</td><td>18.1</td><td>86.9</td><td>43.7</td><td>54.6</td><td>68.2</td><td>64.3</td></tr><tr><td>GIFT [40]</td><td></td><td>88.3</td><td>63.4</td><td>88.1</td><td>70.8</td><td>67.7</td><td>22.8</td><td>90.4</td><td>46.7</td><td>51.8</td><td>68.8</td><td>65.9</td></tr><tr><td>Ours†</td><td>-</td><td>86.9</td><td>61.6</td><td>89.3</td><td>71.3</td><td>63.2</td><td>24.3</td><td>93.3</td><td>44.8</td><td>50.4</td><td>68.6</td><td>65.3 (-0.6)</td></tr><tr><td>Ours</td><td>-</td><td>85.9</td><td>63.3</td><td>89.4</td><td>72.2</td><td>64.2</td><td>24.6</td><td>94.0</td><td>44.6</td><td>49.4</td><td>69.1</td><td>65.7 (-0.2)</td></tr><tr><td colspan="13">Avg.</td></tr><tr><td>ZSCL [49]</td><td>81.7</td><td>91.3</td><td>91.1</td><td>91.0</td><td>82.9</td><td>72.5</td><td>33.6</td><td>89.7</td><td>53.3</td><td>62.8</td><td>69.9</td><td>74.5</td></tr><tr><td>DIKI [32]</td><td>81.9</td><td>88.9</td><td>92.1</td><td>92.8</td><td>87.7</td><td>70.3</td><td>34.3</td><td>94.2</td><td>51.5</td><td>56.1</td><td>69.5</td><td>74.5</td></tr><tr><td>MoE-Adapters [45]</td><td>84.9</td><td>89.9</td><td>89.3</td><td>91.4</td><td>86.2</td><td>72.2</td><td>33.4</td><td>89.4</td><td>53.3</td><td>61.4</td><td>69.9</td><td>74.7</td></tr><tr><td>GIFT [40]</td><td>83.2</td><td>90.8</td><td>92.6</td><td>92.8</td><td>85.8</td><td>74.1</td><td>36.0</td><td>92.1</td><td>54.7</td><td>60.0</td><td>70.4</td><td>75.7</td></tr><tr><td>Ours†</td><td>85.7</td><td>89.1</td><td>92.4</td><td>93.2</td><td>88.4</td><td>71.3</td><td>39.1</td><td>94.5</td><td>53.2</td><td>59.1</td><td>70.3</td><td>76.0 (+0.3)</td></tr><tr><td>Ours</td><td>86.6</td><td>88.9</td><td>92.8</td><td>93.6</td><td>88.5</td><td>71.5</td><td>40.4</td><td>94.9</td><td>54.2</td><td>58.3</td><td>70.9</td><td>76.4 (+0.7)</td></tr><tr><td colspan="13">Last</td></tr><tr><td>ZSCL [49]</td><td>78.2</td><td>91.1</td><td>97.6</td><td>92.5</td><td>87.4</td><td>78.2</td><td>45.0</td><td>92.3</td><td>72.7</td><td>96.2</td><td>86.3</td><td>83.4</td></tr><tr><td>DIKI [32]</td><td>81.9</td><td>89.2</td><td>99.4</td><td>94.3</td><td>96.8</td><td>76.7</td><td>46.3</td><td>95.9</td><td>74.8</td><td>98.3</td><td>86.6</td><td>85.5</td></tr><tr><td>MoE-Adapters [45]</td><td>84.1</td><td>88.5</td><td>94.0</td><td>91.8</td><td>94.1</td><td>77.8</td><td>50.4</td><td>93.3</td><td>77.1</td><td>87.7</td><td>86.6</td><td>84.1</td></tr><tr><td>GIFT [40]</td><td>81.0</td><td>90.2</td><td>98.6</td><td>94.0</td><td>91.5</td><td>78.6</td><td>51.7</td><td>94.6</td><td>75.6</td><td>95.4</td><td>86.6</td><td>85.3</td></tr><tr><td>Ours†</td><td>85.7</td><td>89.4</td><td>99.5</td><td>94.7</td><td>98.1</td><td>78.1</td><td>56.9</td><td>96.6</td><td>78.4</td><td>98.4</td><td>87.3</td><td>87.6 (+2.3)</td></tr><tr><td>Ours</td><td>86.6</td><td>89.2</td><td>99.6</td><td>95.1</td><td>97.9</td><td>78.4</td><td>59.4</td><td>96.4</td><td>79.8</td><td>98.4</td><td>87.9</td><td>88.1 (+2.8)</td></tr></table>

Table E.3. Comparison with SOTA methods on MTIL benchmark (Order I) in terms of “Transfer”, “Average”, and “Last” scores (%). Best results are highlighted in bold. Ours† indicates our reduced-parameter variant.

Table E.4. Comparison with SOTA methods on MTIL benchmark (Order II) in terms of “Transfer”, “Average”, and “Last” scores (%). Best results are highlighted in bold. Ours† indicates our reduced-parameter variant.

<table><tr><td>Task</td><td>Method</td><td>Aircraft</td><td>Caltech101</td><td>CIFAR100</td><td>DTD</td><td>EuroSAT</td><td>Flowers</td><td>Food</td><td>MNIST</td><td>OxfordPet</td><td>Cars</td><td>SUN397</td><td>Average</td></tr><tr><td rowspan="21">ODCL-CIL (MCIL)</td><td colspan="13">Transfer</td></tr><tr><td>ZSCL [49]</td><td>-</td><td>84.6</td><td>67.5</td><td>44.8</td><td>51.5</td><td>69.0</td><td>87.6</td><td>62.3</td><td>87.1</td><td>59.7</td><td>66.4</td><td>68.0</td></tr><tr><td>MoE-Adapters [45]</td><td>-</td><td>88.2</td><td>66.8</td><td>44.7</td><td>54.1</td><td>70.6</td><td>88.4</td><td>59.5</td><td>89.0</td><td>64.7</td><td>65.0</td><td>69.1</td></tr><tr><td>CoLeCLIP [18]</td><td>-</td><td>88.2</td><td>65.1</td><td>44.7</td><td>54.1</td><td>68.8</td><td>88.5</td><td>59.5</td><td>89.0</td><td>64.7</td><td>65.1</td><td>68.8</td></tr><tr><td>DPeCLIP [22]</td><td>-</td><td>88.2</td><td>67.2</td><td>44.7</td><td>54.0</td><td>70.6</td><td>88.2</td><td>59.5</td><td>89.0</td><td>64.7</td><td>64.8</td><td>69.1</td></tr><tr><td>Ours $\dagger$ </td><td>-</td><td>92.9</td><td>68.9</td><td>45.2</td><td>53.7</td><td>71.1</td><td>86.4</td><td>60.4</td><td>89.7</td><td>66.0</td><td>65.8</td><td>70.0 (+0.9)</td></tr><tr><td>Ours</td><td>-</td><td>92.9</td><td>69.0</td><td>45.3</td><td>54.2</td><td>71.1</td><td>86.2</td><td>63.8</td><td>89.3</td><td>65.7</td><td>66.5</td><td>70.4 (+1.3)</td></tr><tr><td colspan="13">Avg.</td></tr><tr><td>ZSCL [49]</td><td>46.3</td><td>68.3</td><td>74.3</td><td>56.3</td><td>79.1</td><td>81.4</td><td>89.5</td><td>74.0</td><td>89.0</td><td>64.4</td><td>67.5</td><td>71.8</td></tr><tr><td>MoE-Adapters [45]</td><td>37.2</td><td>65.3</td><td>79.5</td><td>67.6</td><td>19.7</td><td>83.1</td><td>80.5</td><td>74.0</td><td>88.5</td><td>67.5</td><td>65.3</td><td>66.2</td></tr><tr><td>CoLeCLIP [18]</td><td>48.2</td><td>77.8</td><td>71.7</td><td>65.7</td><td>76.8</td><td>83.8</td><td>89.6</td><td>72.2</td><td>90.3</td><td>68.0</td><td>66.4</td><td>73.7</td></tr><tr><td>DPeCLIP [22]</td><td>49.9</td><td>85.3</td><td>81.5</td><td>65.3</td><td>81.6</td><td>84.3</td><td>89.9</td><td>74.0</td><td>90.4</td><td>68.3</td><td>66.2</td><td>76.1</td></tr><tr><td>Ours $\dagger$ </td><td>57.3</td><td>92.8</td><td>83.2</td><td>67.5</td><td>81.7</td><td>85.0</td><td>87.7</td><td>74.5</td><td>90.6</td><td>69.5</td><td>66.9</td><td>77.9 (+1.8)</td></tr><tr><td>Ours</td><td>60.2</td><td>93.5</td><td>83.9</td><td>68.9</td><td>82.0</td><td>84.8</td><td>87.6</td><td>76.7</td><td>90.4</td><td>69.4</td><td>67.6</td><td>78.6 (+2.5)</td></tr><tr><td colspan="13">Last</td></tr><tr><td>ZSCL [49]</td><td>42.5</td><td>64.4</td><td>67.2</td><td>54.8</td><td>89.7</td><td>90.4</td><td>91.7</td><td>95.8</td><td>93.4</td><td>85.2</td><td>78.3</td><td>77.6</td></tr><tr><td>MoE-Adapters [45]</td><td>34.1</td><td>47.6</td><td>80.9</td><td>75.5</td><td>0.00</td><td>93.0</td><td>70.8</td><td>99.4</td><td>86.4</td><td>79.8</td><td>68.9</td><td>66.9</td></tr><tr><td>CoLeCLIP [18]</td><td>48.1</td><td>73.1</td><td>65.2</td><td>69.6</td><td>84.0</td><td>96.2</td><td>90.9</td><td>94.6</td><td>93.5</td><td>82.6</td><td>79.3</td><td>79.7</td></tr><tr><td>DPeCLIP [22]</td><td>49.9</td><td>84.2</td><td>83.2</td><td>71.1</td><td>97.0</td><td>95.8</td><td>92.0</td><td>99.4</td><td>93.9</td><td>84.5</td><td>80.2</td><td>84.6</td></tr><tr><td>Ours $\dagger$ </td><td>57.2</td><td>89.9</td><td>85.2</td><td>74.1</td><td>97.7</td><td>96.5</td><td>89.1</td><td>99.2</td><td>92.7</td><td>85.1</td><td>77.4</td><td>85.8 (+1.2)</td></tr><tr><td>Ours</td><td>60.1</td><td>91.1</td><td>85.9</td><td>75.7</td><td>97.9</td><td>96.2</td><td>89.1</td><td>99.2</td><td>93.0</td><td>86.0</td><td>78.3</td><td>86.6 (+2.0)</td></tr></table>

Table E.5. Comparison with SOTA methods on ODCL-CIL benchmarks in terms of “Transfer,” “Average,” and “Last” scores (%). Best results are highlighted in bold. Ours† indicates our reduced-parameter variant.