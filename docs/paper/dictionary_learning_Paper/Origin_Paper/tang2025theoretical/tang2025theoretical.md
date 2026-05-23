# A Unified Theory of Sparse Dictionary Learning in Mechanistic Interpretability: Piecewise Biconvexity and Spurious Minima

Yiming Tang

National University of Singapore

yiming@nus.edu.sg

Harshvardhan Saini

Indian Institute of Technology, Dhanbad

Zhaoqian Yao

Chinese University of Hong Kong

Zheng Lin

Hong Kong University of Science and Technology

Yizhen Liao

National University of Singapore

Jingyi Cui

Peking University

Yisen Wang

Peking University

Mengnan Du

Chinese University of Hong Kong

Dianbo Liu

National University of Singapore

dianbo@nus.edu.sg

# Abstract

As AI models achieve remarkable capabilities across diverse domains, understanding what representations they learn and how they encode concepts has become increasingly important for both scientific progress and trustworthy deployment. Recent works in mechanistic interpretability have widely reported that neural networks represent meaningful concepts as linear directions in their representation spaces and often encode diverse concepts in superposition. Various sparse dictionary learning (SDL) methods, including sparse autoencoders, transcoders, and crosscoders, are utilized to address this by training auxiliary models with sparsity constraints to disentangle these superposed concepts into monosemantic features. These methods are the backbone of modern mechanistic interpretability, yet in practice they consistently produce polysemantic features, feature absorption, and dead neurons, with very limited theoretical understanding of why these phenomena occur. Existing theoretical work is limited to tied-weight sparse autoencoders, leaving the broader family of SDL methods without formal grounding. We develop the first unified theoretical framework that casts all major SDL variants as a single piecewise biconvex optimization problem, and characterize its global solution set, non-identifiability, and spurious optima. This analysis yields principled explanations for feature absorption and dead neurons. To expose these pathologies under full ground-truth access, we introduce the Linear Representation Bench. Guided by our theory, we propose feature anchoring, a novel technique that restores SDL identifiability, substantially improving feature recovery across synthetic benchmarks and real neural representations.

# 1 Introduction

As artificial intelligence systems scale to frontier capabilities, understanding their internal mechanisms has become essential for safe deployment (Lipton, 2017; Rudin, 2019). A central insight from mechanistic interpretability is that neural networks encode interpretable concepts as linear directions in superposition (Park et al., 2024; Elhage et al., 2022), where individual neurons respond to multiple unrelated concepts—a phenomenon known as polysemanticity (Bricken et al., 2023; Templeton et al., 2024). To disentangle these superposed representations, Sparse Dictionary Learning methods, including sparse autoencoders (SAEs) (Cunningham et al., 2023), transcoders (Dunefsky et al., 2024), and crosscoders (Lindsey et al., 2024), have emerged as the dominant paradigm, achieving remarkable empirical success on frontier language models (Templeton et al., 2024; Gao et al., 2024) and enabling applications from feature steering (Wang et al., 2025) to circuit analysis (Marks et al., 2025) and medical diagnosis (Abdulaal et al., 2024).

Despite this empirical success across diverse applications, SDL methods consistently exhibit persistent failure modes: learned features remain polysemantic (Chanin et al., 2025), "dead neurons" fail to activate on any data samples (Bricken et al., 2023), and "feature absorption" occurs where one neuron captures specific subconcepts while another responds to the remaining related concepts (Chanin et al., 2025). Practitioners have developed techniques to address these issues, including neuron resampling (Bricken et al., 2023), auxiliary losses (Gao et al., 2024), and careful hyperparameter tuning, yet these fixes remain ad-hoc engineering solutions without principled justification. Critically, these phenomena persist even with careful training on clean data, suggesting they are not mere implementation artifacts but reflect fundamental structural properties of the SDL optimization problem itself.

We argue that the root cause of these failure modes is the non-identifiability of SDL methods: even under the idealized Linear Representation Hypothesis, SDL optimization admits multiple solutions achieving perfect reconstruction loss, some recovering no interpretable ground-truth features at all, necessitating comprehensive theoretical analysis on SDL methods. While classical dictionary learning theory provides identifiability guarantees under strict conditions (Spielman et al., 2012; Gribonval & Schnass, 2010), and recent work establishes necessary conditions for tied-weight SAEs (Cui et al., 2025), no unified theoretical framework explains why diverse SDL methods, SAEs, transcoders, crosscoders, and their variants, systematically fail in predictable ways. Without theoretical grounding, the development of improved SDL methods remains largely empirical, potentially leaving highly effective techniques underexplored.

In this work, we develop a unified theoretical framework that formalizes SDL as a general optimization problem, encompassing various SDL methods (Bussmann et al., 2024; 2025; Tang et al., 2025b; Luo et al., 2023) as special cases. We demonstrate how these diverse methods instantiate our framework through different choices of input-output representation pairs, activation functions, and loss designs. We establish rigorous conditions under which SDL methods provably recover ground-truth interpretable features, characterizing the roles of feature sparsity, latent dimensionality, and activation functions. Through detailed analysis of the optimization landscape, we demonstrate that global minima correspond to correct feature recovery and provide necessary and sufficient conditions for achieving zero loss. We establish the prevalence of spurious partial minima exhibiting un-disentangled polysemanticity, providing novel theoretical explanations for feature absorption (Chanin et al., 2025) and the effectiveness of neuron resampling (Bricken et al., 2023). We design the Linear Representation Bench, a synthetic benchmark that strictly follows the Linear Representation Hypothesis, to evaluate SDL methods with fully accessible ground-truth features. Motivated by our theoretical insights, we propose feature anchoring, a technique applicable to all SDL methods which achieves improved feature recovery by constraining learned features to known anchor directions.

Our main contributions are as follows:

• We build the first theoretical framework for SDL in mechanistic interpretability as a general optimization problem encompassing diverse SDL methods.

• We theoretically prove that SDL optimization is biconvex, bridging mechanistic interpretability methods with traditional biconvex optimization theory.   
• We characterize SDL optimization landscape and prove its non-identifiability, providing novel explanations for various phenomena observed empirically.   
• We design the Linear Representation Bench, a benchmark with fully accessible ground-truth features, enabling fully transparent evaluation of SDL methods.   
• We propose a novel technique, feature anchoring, that can achieve largely improved feature recovery performance applicable for all SDL methods. We validate the effectiveness of feature anchoring with extensive experiments across diverse SDL methods and settings.

# 2 Preliminaries

In this section, we present a unified theoretical framework for Sparse Dictionary Learning (SDL). We begin with the formal definitions of foundational concepts and our assumptions on representations. Then we introduce our framework and how various SDL methods instantiate it.

# 2.1 Input Distribution and Model Representation

Definition 2.1 (Input Distribution). Let D denote the distribution over possible inputs, X , to a neural network. For example, D could be the distribution of natural images or the distribution of text sequences (Notations in Appendix B).

Definition 2.2 (Model Representation). For a given model representation x, let $n \in \mathbb { N }$ be its dimensionality. For each $s \sim \mathcal { D }$ , the network produces a representation vector $\mathbf { x } ( s )$ , which is directly observable by running the model on s.

# 2.2 Assumptions

Empirical studies in mechanistic interpretability have observed that neural network representations encode meaningful concepts as linear directions, often in superposition (Marks & Tegmark, 2024; Nanda et al., 2023; Jiang et al., 2024; Park et al., 2025). Following Elhage et al. (2022) and Park et al. (2024), we primarily consider model representations $\mathbf { x } _ { p } \in \mathbb { R } ^ { n _ { p } }$ satisfying the following assumptions.

Assumption 2.1 (Representation Assumptions). There exists a feature function $\mathbf { x } : \mathcal { X }  \mathbb { R } ^ { n }$ and a feature matrix $W _ { p } \in \mathbb { R } ^ { n _ { p } \times n }$ such that:

1. Linear Decomposition: For all $s \sim \mathcal { D }$ ,

$$
\mathbf {x} _ {p} (s) = W _ {p} \mathbf {x} (s).
$$

2. Unit Norm: The feature matrix $W _ { p } \in \mathbb { R } ^ { n _ { p } \times n }$ has unit-norm columns:

$$
\| W _ {p} [:, i ] \| _ {2} = 1 \quad \forall i \in [ n ]
$$

3. Non-negativity: $\mathbf { x } ( s ) \in \mathbb { R } _ { + } ^ { n }$

4. Sparsity: There exists $S \in [ 0 , 1 ]$ such that $\forall i \in [ n ]$ ,

$$
\operatorname * {P r} _ {s \sim \mathcal {D}} (x _ {i} (s) = 0) \geq S.
$$

We refer to $\mathbf { x } ( s )$ as the ground-truth features, where each $x _ { i } ( s )$ represents the activation level of concept i for input s. Researchers typically assume each component $x _ { i }$ of x corresponds to a human-interpretable concept (Bricken et al., 2023; Cunningham et al., 2023; Elhage et al., 2022).

# 2.3 General Optimization Framework for SDL

Now we formalize SDL as an optimization problem under the Representation Assumptions (Assumption 2.1). We follow previous approaches Cui et al. (2025); Elhage et al. (2022) to omit the bias terms for mathematical simplicity.

Definition 2.3 (Sparse Dictionary Learning). A SDL model maps an input representation $\mathbf { x } _ { p } ( s ) \in \mathbb { R } ^ { n _ { p } }$ to a target representation $\mathbf { x } _ { r } ( s ) \in \mathbb { R } ^ { n _ { \tau } }$ through a two-layer architecture:

(i) An encoder layer that maps $\mathbf { x } _ { p } ( s )$ to a latent space:

$$
\mathbf {x} _ {q} (s) = \sigma (W _ {E} \mathbf {x} _ {p} (s)) \tag {1}
$$

(ii) A decoder layer that maps the latents to $\mathbf { x } _ { r } ( s )$ :

$$
\hat {\mathbf {x}} _ {r} (s) = W _ {D} \mathbf {x} _ {q} (s) \tag {2}
$$

where $\mathbf { x } _ { q } ( s ) \in \mathbb { R } ^ { n _ { q } } , W _ { E } \in \mathbb { R } ^ { n _ { q } \times n _ { p } } , W _ { D } \in \mathbb { R } ^ { n _ { r } \times n _ { q } }$ , and $\sigma : \mathbb { R } ^ { n _ { q } }  \mathbb { R } ^ { n _ { q } }$ is a sparsity-inducing activation function.

The SDL objective minimizes mean square error:

$$
\mathcal {L} _ {S D L} = \mathbb {E} _ {s \sim \mathcal {D}} \left[ \| \mathbf {x} _ {r} (s) - W _ {D} \sigma (W _ {E} \mathbf {x} _ {p} (s)) \| _ {2} ^ {2} \right] \tag {3}
$$

Under the Linear Representation Hypothesis (Assumption 2.1), both representations admit linear decompositions $\mathbf { x } _ { p } ( s ) = W _ { p } \mathbf { x } ( s )$ and $\mathbf { x } _ { r } ( s ) = W _ { r } \mathbf { x } ( s )$ in terms of ground-truth features $\mathbf { x } ( s )$ . The loss can thus be expressed as:

$$
\mathcal {L} _ {\mathrm{SDL}} = \mathbb {E} _ {s \sim \mathcal {D}} \left[ \| W _ {r} \mathbf {x} (s) - W _ {D} \sigma (W _ {E} W _ {p} \mathbf {x} (s)) \| _ {2} ^ {2} \right] \tag {4}
$$

# 2.4 Instantiations: Existing SDL Methods

We now demonstrate how existing SDL methods instantiate our framework through adopting different choices of input-target pairs $\left( \mathbf { x } _ { p } , \mathbf { x } _ { r } \right)$ and activation functions σ, and proposing variants on the loss function $\mathcal { L } _ { \mathrm { S D L } }$ (See Appendix A).

Sparse Autoencoders (SAEs). SAEs (Cunningham et al., 2023) decompose polysemantic activations into monosemantic components through sparsity constraints. In our framework, SAEs are characterized by setting ${ \bf x } _ { r } = { \bf x } _ { p }$ (self-reconstruction). The encoder projects to a higher-dimensional sparse latent space, encouraging $\mathbf { x } _ { q } ( s )$ to capture the underlying ground-truth features $\mathbf { x } ( s )$ (Figure 1).

![](images/56c391d7ee7fbceafa08842b1c1d64208ac2291d40b370fa7a470e044ed4da1a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Model"] --> B["x_p"]
    B --> C["Model Representation"]
    C --> D["W_E"]
    D --> E["σ"]
    E --> F["x_q"]
    F --> G["W_D"]
    G --> H["x_r"]
    style A fill:#cce5ff,stroke:#333
    style H fill:#cce5ff,stroke:#333
    subgraph "Activated"
        I["Light pink"]
        J["Light gray"]
    end
    subgraph "Non-activated"
        K["Light gray"]
        L["Light gray"]
    end
    M["x_r = x_p"]
```
</details>

Figure 1: Sparse Autoencoder: encoder $W _ { E }$ maps $\mathbf { x } _ { p }$ to sparse latents $\mathbf { x } _ { q } .$ decoder $W _ { D }$ reconstructs from $\mathbf { x } _ { q }$

Transcoders. Transcoders (Dunefsky et al., 2024; Paulo et al., 2025) capture interpretable features in layer-to-layer transformations. Unlike ${ \mathrm { S A E s } } .$ , transcoders approximate the input-output function of a target component, such as a MLP, using a sparse bottleneck. In our proposed theoretical framework, transcoders set $\mathbf { x } _ { p } = \mathbf { x } _ { \mathrm { m i d } } ( s )$ and $\mathbf { x } _ { r } = \mathbf { x } _ { \mathrm { p r e } } ( s )$ , where $\mathbf { x } _ { \mathrm { m i d } } ( s )$ denotes the inputs of one MLP block, and $\mathbf { x } _ { \mathrm { p r e } } ( s )$ denotes the prediction of MLP’s outputs (Figure 2).

![](images/e2a23c20b49f8488553fa7d401157d585d3e4d44f2f83eaf2fd426770593c6f6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Model"] --> B["Attention"]
    A --> C["MLP"]
    A --> D["x_p"]
    D --> E["W_E"]
    E --> F["σ"]
    F --> G["x_q"]
    G --> H["W_D"]
    H --> I["x_r"]
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style F fill:#bfb,stroke:#333
    style G fill:#ffb,stroke:#333
    style H fill:#bbf,stroke:#333
    style I fill:#f9f,stroke:#333
```
</details>

Figure 2: Transcoder: encoder $W _ { E }$ maps $\mathbf { x } _ { p } ( s )$ to sparse latents $\mathbf { x } _ { q } ( s )$ , decoder $W _ { D }$ gives $\mathbf { x } _ { r } ( s )$ as a prediction of MLP’s output.

Crosscoders. Crosscoders (Lindsey et al., 2024) discover shared features across multiple representation sources by jointly encoding and reconstructing concatenated representations. In our framework, crosscoders set $\mathbf { x } _ { p } = [ \mathbf { x } _ { p } ^ { ( 1 ) } ; \ldots ; \mathbf { x } _ { p } ^ { ( m ) } ]$ (m) and $\mathbf { x } _ { r } = [ \mathbf { x } _ { r } ^ { ( 1 ) } ; \ldots ; \mathbf { x } _ { r } ^ { ( m ) } ]$ where each superscript denotes a different source $\mathrm { ( F i g \mathrm { - } }$ - ure 3).

![](images/2eaad3f215cfbbcd55b56750f898e656ad3bda1e236e6cdb0c03e9a5ae3deaa7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Model¹"] --> B["x_p^(1)"]
    C["Model²"] --> D["x_p^(2)"]
    E["..."] --> F["x_p^(m)"]
    G["Model^m"] --> H["x_p^(m)"]
    B --> I["x_p"]
    D --> I
    F --> I
    H --> I
    I --> J["W_E"]
    J --> K["σ"]
    K --> L["x_q"]
    L --> M["W_D"]
    M --> N["x_r"]
    N --> O["Model¹"]
    N --> P["Model²"]
    N --> Q["..."]
    N --> R["Model^m"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#ccf,stroke:#333
    style J fill:#cfc,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#cfc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#fff,stroke:#333
    style P fill:#fff,stroke:#333
    style Q fill:#fff,stroke:#333
    style R fill:#fff,stroke:#333
```
</details>

Figure 3: Crosscoder: encoder $W _ { E }$ maps concatenated multi-layer input $\mathbf { x } _ { p }$ to $\mathbf { x } _ { q } ,$ decoder $W _ { D }$ reconstructs multi-layer output xr.

Variants of SDL Methods. Various SDL methods fit into our theoretical framework but differ in their choices of activation functions and loss designs. Bricken et al. (2023) and Templeton et al. (2024) use ReLU activation $\sigma _ { \mathrm { R e L U } } ( z ) = \operatorname* { m a x } ( 0 , z )$ with $L _ { 1 }$ regularization on latents:

$$
\mathcal {L} = \mathbb {E} _ {s \sim \mathcal {D}} \left[ \| \mathbf {x} _ {r} (s) - W _ {D} \sigma_ {\mathrm{ReLU}} (W _ {E} \mathbf {x} _ {p} (s)) \| _ {2} ^ {2} + \lambda \| \mathbf {x} _ {q} (s) \| _ {1} \right] \tag {5}
$$

Rajamanoharan et al. (2024b) propose to utilize JumpReLU as the activation function:

$$
\sigma_ {\text { JumpReLU }} (z) = z \cdot H (z - \theta) \tag {6}
$$

combined with a smoothed Heaviside function to directly penalize the $L _ { 0 }$ norm. Makhzani & Frey (2014) introduce Top-k activation:

$$
\sigma_ {\text {Top-} k} (z) _ {i} = \left\{ \begin{array}{l l} z _ {i} & \text {if} z _ {i} \text {is among the} k \text {largest components} \\ 0 & \text {otherwise} \end{array} \right. \tag {7}
$$

Bussmann et al. (2024) extend this to Batch Top-k activation, which applies Top-k selection across a batch to allow different samples to have different numbers of activated features. Rajamanoharan et al. (2024a) use ReLU activation with an additional Heaviside gating function. Gao et al. (2024) employ Top-k activation and introduce an auxiliary loss using $\mathrm { T o p } { - } k _ { \mathrm { a u x } }$ dead latents:

$$
\begin{array}{l} \mathcal {L} = \mathbb {E} _ {s \sim \mathcal {D}} \left[ \| \mathbf {x} _ {r} (s) - W _ {D} \sigma_ {\text {Top-} k} \left(W _ {E} \mathbf {x} _ {p} (s)\right) \| _ {2} ^ {2} \right. \tag {8} \\ \left. + \lambda_ {\mathrm{aux}} \| \mathbf {x} _ {p} (s) - W _ {D} ^ {\prime} \sigma_ {\mathrm{Top-} k _ {\mathrm{aux}}} (W _ {E} \mathbf {x} _ {p} (s)) \| _ {2} ^ {2} \right] \\ \end{array}
$$

to resurrect dead neurons. Bussmann et al. (2025) and Tang et al. (2025b) use multiple k values with a multi-scale loss:

$$
\mathcal {L} = \sum_ {i = 1} ^ {m} \lambda_ {i} \mathbb {E} _ {s \sim \mathcal {D}} \left[ \| \mathbf {x} _ {r} (s) - W _ {D} \sigma_ {\text {Top-} k _ {i}} (W _ {E} \mathbf {x} _ {p} (s)) \| _ {2} ^ {2} \right] \tag {9}
$$

that sums reconstruction errors of different sparsity levels.

The variants described above demonstrate that diverse SDL methods can be unified under our general framework (Definition 2.3) through specific choices of activation functions σ and loss modifications. Critically, this unification enables subsequent theoretical analysis to apply to the entire family of SDL methods, rather than just a single architecture. We also state a key property for SDL activation functions:

$$
\sigma (z) _ {i} \in \{0, z _ {i} \} \quad \forall i \in [ n _ {q} ] \tag {10}
$$

This property holds for ReLU, JumpReLU, Top-k, Batch Top-k, and their compositions, the primary sparsity mechanisms used in practice.

# 3 Theoretical Results

Despite the empirical success of SDL methods across diverse applications, a significant gap exists between their practical use and our theoretical understanding of their optimization dynamics. This gap has important consequences: practitioners employ techniques like dead neuron resampling (Bricken et al., 2023) and observe phenomena like feature absorption (Chanin et al., 2025) without rigorous explanations for why these occur or how to systematically address them. Without theoretical grounding, the development of improved SDL methods remains largely empirical, potentially leaving highly effective techniques underexplored.

Our theoretical analysis bridges this gap by characterizing the SDL optimization landscape under the Linear Representation Hypothesis. Section 3.1 establishes a loss approximation enabling tractable analysis (Theorem 3.1). Section 3.2 proves SDL exhibits piecewise biconvex structure within activation pattern regions (Theorem 3.2). Section 3.3 characterizes the global minimum and shows the optimization is underdetermined (Theorem 3.3 and 3.4). Section 3.4 establishes the prevalence of spurious partial minima exhibiting polysemanticity (Theorem 3.6). Section 3.5 shows that hierarchical concept structures naturally induce feature absorption patterns that manifest as partial minima (Theorem 3.8). We provide full proofs in Appendix I.

# 3.1 Approximate Loss with Feature Reconstruction

We establish the bound for S satisfying $\textstyle ( 1 - S ) \leq { \frac { 1 } { n } }$ , the $\mathcal { L } _ { \mathrm { S D I } }$ can be approximated by per-feature reconstruction.

Theorem 3.1 (Loss Approximation). Under Assumption 2.1, let $\begin{array} { r } { C = \operatorname* { s u p } _ { s \sim \mathcal { D } } \| \mathbf { x } _ { r } ( s ) - W _ { D } \sigma ( W _ { E } \mathbf { x } _ { p } ( s ) ) \| _ { 2 } ^ { 2 } < } \end{array}$ < ∞. Define the approximate loss as:

$$
\tilde {\mathcal {L}} _ {S D L} \left(W _ {D}, W _ {E}\right) := \sum_ {d = 1} ^ {n} M _ {d} \left\| w _ {r} ^ {d} - W _ {D} \sigma \left(W _ {E} w _ {p} ^ {d}\right) \right\| ^ {2} \tag {11}
$$

where $M _ { d } = \operatorname* { P r } ( \mathbf { x } ( s ) = x _ { d } ( s ) e _ { d } ) \cdot \mathbb { E } [ x _ { d } ( s ) ^ { 2 } \mid x _ { d } ( s ) > 0 ]$ . Then for any S satisfying $\textstyle ( 1 - S ) \leq { \frac { 1 } { n } }$

$$
\left| \mathcal {L} _ {S D L} - \tilde {\mathcal {L}} _ {S D L} \right| \leq n ^ {2} C (1 - S) ^ {2} \tag {12}
$$

This approximation isolates the contribution of each ground-truth feature, making the optimization landscape amenable to analysis techniques applied subsequently.

# 3.2 SDL Optimization is Piecewise Biconvex

While the SDL loss is non-convex globally due to activation discontinuities, it exhibits favorable convex structure within regions of fixed activation patterns.

Theorem 3.2 (Bi-convex Structure of SDL). Consider the approximate loss $\tilde { \mathcal { L } } _ { S D L } ( W _ { D } , W _ { E } )$ . For an activation pattern $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { n _ { q } } )$ (See Definition 3.1), define the corresponding activation pattern region as:

$$
\Omega_ {\mathcal {P}} = \Big \{W _ {E} \in \mathbb {R} ^ {n _ {q} \times n _ {p}} \Big | \forall i \in [ n _ {q} ],
$$

$$
\mathcal {F} _ {i} = \left\{d \in [ n ]: \left(\sigma (W _ {E} w _ {p} ^ {d})\right) _ {i} > 0 \right\}, \tag {13}
$$

Then $\tilde { \mathcal { L } } _ { S D L }$ exhibits bi-convex structure over $\mathbb { R } ^ { n _ { r } \times n _ { q } } \times \Omega _ { \mathcal { P } }$ :

1. For any fixed $W _ { E } \in \Omega _ { \mathcal { P } } , W _ { D } \mapsto \tilde { \mathcal { L } } _ { S D L } ( W _ { D } , W _ { E } )$ is convex in $W _ { D }$   
2. For any fixed $W _ { D } \in \mathbb { R } ^ { n _ { r } \times n _ { q } } , W _ { E } \mapsto \tilde { \mathcal { L } } _ { S D L } ( W _ { D } , W _ { E } )$ is convex in $W _ { E }$ over $\Omega _ { \mathcal { P } }$

This establishes SDL as a biconvex optimization problem within each activation pattern, bridging mechanistic interpretability with classical biconvex optimization theory, enabling both theoretical analysis of the optimization landscape and implementation of biconvex methods for SDL.

# 3.3 Characterizing the Global Minimum of SDL

Characterizing the global minimum reveals both the success conditions for SDL training and the fundamental underdetermined nature of the optimization problem.

Theorem 3.3 (Successful Reconstruction Achieves Near-Zero Loss). Consider the approximate SDL loss

$$
\tilde {\mathcal {L}} _ {S D L} (W _ {D}, W _ {E}) = \sum_ {d = 1} ^ {n} M _ {d} \left\| w _ {r} ^ {d} - W _ {D} \sigma (W _ {E} w _ {p} ^ {d}) \right\| ^ {2} \tag {14}
$$

where $M _ { d } = \operatorname* { P r } ( \mathbf { x } ( s ) = x _ { d } ( s ) e _ { d } ) \cdot \mathbb { E } [ x _ { d } ( s ) ^ { 2 } \mid x _ { d } ( s ) > 0 ]$ , and let $M = \operatorname* { m a x } _ { i \neq j } \langle w _ { p } ^ { i } , w _ { p } ^ { j } \rangle$ denote the maximum interference. When $n _ { q } \geq n$ and σ satisfies $\sigma ( z ) _ { i } \in \{ 0 , z _ { i } \}$ for all i, the configuration

$$
W _ {D} ^ {*} = \left[ W _ {r}, \mathbf {0} \right], \quad W _ {E} ^ {*} = \left[ \begin{array}{c} W _ {p} ^ {\top} \\ \mathbf {0} \end{array} \right] \tag {15}
$$

where 0 denotes zero padding to dimension $n _ { q } ,$ , satisfies:

$$
\tilde {\mathcal {L}} _ {S D L} (W _ {D} ^ {*}, W _ {E} ^ {*}) \leq n ^ {2} M ^ {2} \sum_ {d = 1} ^ {n} M _ {d} \tag {16}
$$

Theorem 3.4 (Necessary and Sufficient Conditions for Zero Loss). The approximate loss satisfies $\tilde { \mathcal { L } } _ { S D L } ( W _ { D } , W _ { E } ) = 0$ if and only if

$$
w _ {r} ^ {d} = W _ {D} \sigma (W _ {E} w _ {p} ^ {d}) \quad \text {   for   all   } d \in [ n ] \tag {17}
$$

Theorem 3.3 provides a constructive global minimum that recovers ground-truth features, while Theorem 3.4 reveals the complete solution space. This system of n vector equations is underdetermined when $n _ { q } > n _ { : }$ admitting multiple solutions beyond the feature-recovering configuration. Critically, some solutions achieve zero reconstruction loss without recovering any ground-truth features (Figure 4).

![](images/d83605b7007c0057374406255c24c08d02fd3586a3c0386b59c7fd2e5dd43dfe.jpg)

<details>
<summary>scatter</summary>

| Point | Dimension 1 | Dimension 2 |
|-------|-------------|-------------|
| f1    | 0.0         | 0.0         |
| f2    | 0.0         | 1.5         |
| f3    | -1.0        | 0.0         |
| f4    | -1.0        | -1.5        |
</details>

![](images/d9d8d7f260d2d298dd6d3dc04f0fc2695419296d857a78335b33a791d0b578b6.jpg)

<details>
<summary>scatter</summary>

| Point | Dimension 1 | Dimension 2 |
|-------|-------------|-------------|
| e1    | -1.0        | 1.0         |
| e2    | 1.0         | -1.0        |
| e3    | -1.5        | -0.5        |
| e4    | 1.5         | 0.5         |
</details>

![](images/864e9474d4f2e1b15745db93a9f624a306bffc0d7bffca61f40fc8cadbb92e0f.jpg)

<details>
<summary>line</summary>

| Point | Dimension 1 | Dimension 2 |
|-------|-------------|-------------|
| d1    | -0.5        | 1.2         |
| d2    | 0.8         | -1.3        |
| d3    | -1.0        | -1.0        |
| d4    | 1.0         | 0.9         |
</details>

Figure 4: Zero reconstruction loss without recovering ground-truth features. We design the Linear Representation Bench that enable full knowledge of the ground truth features to study SDL methods. We observe one concerning phenomenon that these methods can achieve zero loss without recovering any ground truth features. Left: Four ground-truth feature directions. Middle: Learned encoder directions fail to align with ground truth. Right: Learned decoder directions are rotated accordingly. Although $\tilde { \mathcal { L } } _ { \mathrm { S D L } } \approx 0$ , the learned features bear no correspondence to interpretable ground-truth concepts, demonstrating the underdetermined nature of SDL optimization.

# 3.4 Characterizing Spurious Partial Minima of SDL

Beyond the global minimum, SDL optimization exhibits spurious partial minima where neurons exhibit polysemanticity—responding to multiple unrelated features.

Example 3.5 (Spurious Partial Minimum). Consider $n = n _ { p } = n _ { q } = n _ { r } = 2 , \sigma = \sigma _ { T o p - 1 } \cdot \sigma _ { R e L U } , S \to 1$ , and $M _ { 1 } = M _ { 2 } = 1$ . Let:

$$
w _ {p} ^ {1} = \left[ \begin{array}{l} 1 \\ 0 \end{array} \right], \quad w _ {p} ^ {2} = \left[ \begin{array}{l} 0 \\ 1 \end{array} \right], \quad w _ {r} ^ {1} = \left[ \begin{array}{l} 1 \\ 0 \end{array} \right], \quad w _ {r} ^ {2} = \left[ \begin{array}{l} 0 \\ 1 \end{array} \right] \tag {18}
$$

Consider the configuration where neuron 1 activates for both features while neuron 2 remains dead:

$$
W _ {E} ^ {*} = \left[ \begin{array}{l l} 1 & 1 \\ 0 & 0 \end{array} \right], \quad W _ {D} ^ {*} = \left[ \begin{array}{l l} 1 / 2 & 0 \\ 1 / 2 & 0 \end{array} \right] \tag {19}
$$

Within the activation pattern region $\Omega _ { \cal A } \ = \ \{ W _ { E } \ : \ \langle w _ { E } ^ { 1 } , w _ { p } ^ { 1 } \rangle \ > \ 0 , \langle w _ { E } ^ { 1 } , w _ { p } ^ { 2 } \rangle \ > \ 0 \}$ , direct calculation shows $\nabla _ { W _ { D } } \tilde { \mathcal { L } } \ = \ 0$ and $\nabla _ { W _ { E } } \tilde { \mathcal { L } } \ = \ 0$ . By traditional biconvex optimization theory (Gorski et al., 2007), $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ is a partial optimum. However, this configuration exhibits polysemanticity with suboptimal loss: $\tilde { \mathcal { L } } _ { S D L } ( W _ { D } ^ { * } , W _ { E } ^ { * } ) = 1 > 0$ .

Definition 3.1 (Activation Pattern). An activation pattern is a collection $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { N } )$ where $\mathcal { F } _ { i } \subseteq [ n ]$ denotes the set of ground-truth features that activate neuron i.

An activation pattern P is called:

• Polysemantic $i f \exists i \in [ N ]$ such that $| \mathcal { F } _ { i } | \geq 2$ (at least one neuron responds to multiple features).   
• Realizable if there exists an encoder $\sigma ( W _ { E } ( \cdot ) ) \in \mathbb { R } ^ { N \times n _ { p } }$ such that:

$$
\forall i \in [ N ], \quad \mathcal {F} _ {i} = \left\{d \in [ n ]: \left(\sigma (W _ {E} w _ {p} ^ {d})\right) _ {i} > 0 \right\} \tag {20}
$$

That is, neuron i activates for exactly the features in $\mathcal { F } _ { i }$ when they appear in isolation.

Theorem 3.6 (Prevalence of Spurious Partial Minima). Under Assumptions 2.1 with $n \geq 2$ and $n _ { q } \geq n$ , for any activation pattern $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { n _ { q } } )$ that is realizable, polysemantic, and forms a partition of [n], there exists a partial minimum $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ of $\tilde { \mathcal { L } } _ { S D L }$ exhibiting this pattern with positive loss.

![](images/22fabd4fdaf1386e5bc434c5b022be2e971cb03e56ecd6519d4e57aee91ec9c8.jpg)  
Figure 5: Feature absorption emerges from hierarchical concept structure. Left: Ideal SDL features without absorption. Right: hierarchical concept structure exists and only a proportion of the sub-concepts of "Dog" can activate the SDL feature.

This establishes that partial minima are pervasive in SDL optimization: every realizable polysemantic activation pattern corresponds to a partial optima point where gradient descent can become trapped, a persistent challenge for SDL.

# 3.5 Theoretical Explanation for Feature Absorption

Feature absorption—where one neuron captures, or "absorbs", a specific sub-concept while another responds to remaining related concepts—frequently occurs in SDL training (Chanin et al., 2025). Though prevalently encountered and unwanted, researchers have limited understanding about why feature absorption occurs in SDL training. Here we show hierarchical concept structures naturally introduce realizable activation patterns exhibiting feature absorption and therefore connected with the framework’s partial minima.

Example 3.7 (Feature Absorption). Consider a representation space with a parent concept "Dog" and four sub-concepts: "Border Collie", "Golden Retriever", "Husky", and "German Shepherd". Ideally, SDL learns separate monosemantic neurons for Dog, Cat, Horse, and Elephant, with each dog breed activating only the Dog neuron. However, feature absorption can result in the pattern in Figure 5: one neuron exclusively captures "Border Collie" (absorbed feature), while another responds to the remaining three breeds collectively (main line interpretation).

Definition 3.2 (Hierarchical Concept Structure). A set of ground-truth features exhibits hierarchical structure if it is composed of a parent concept p and a set of sub-concepts $c _ { 1 } , \ldots , c _ { k }$ satisfying: $p ( x ) > 0 \iff$ $\exists i \in [ k ] , c _ { i } ( x ) > 0$ .

Theorem 3.8 (Feature Absorption from Hierarchical Structure). Suppose there exist M parent concepts with hierarchical decompositions into sub-concepts: for each $i \in [ M ]$ , parent concept $d _ { i }$ decomposes into sub-concepts $\mathcal { F } _ { i } = \{ d _ { i , 1 } , \ldots , d _ { i , k _ { i } } \}$ where $k _ { i } \geq 2$ .

If the activation pattern $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { M } )$ is realizable, then $\forall i ^ { * } \in [ M ] , \exists j ^ { * } \in [ k _ { i ^ { * } } ]$ such that the following activation pattern exhibiting feature absorption is realizable:

$$
\mathcal {P} ^ {\prime} = \left(\mathcal {F} _ {1}, \dots , \mathcal {F} _ {i ^ {*}} \backslash \left\{d _ {i ^ {*}, j ^ {*}} \right\}, \left\{d _ {i ^ {*}, j ^ {*}} \right\}, \dots , \mathcal {F} _ {M}\right) \tag {21}
$$

Theorem 3.8 explains why feature absorption persists even with careful training: hierarchical concept structures naturally induce realizable polysemantic patterns that manifest as partial minima, providing theoretical grounding for this widely observed empirical phenomenon.

# 4 Method

Our theoretical analysis reveals a fundamental challenge in SDL optimization: the underdetermined nature of the loss landscape (Theorem 3.4), and the solution space admits multiple configurations—some achieving zero reconstruction loss without recovering any interpretable ground-truth features (Figure 4). These theoretical findings motivate our method, feature anchoring, a technique that constrains a subset of features to align with known semantic directions.

# 4.1 Anchor Feature Extraction

Feature anchoring requires identifying k anchor features $\{ \tilde { \mathbf { w } } _ { p } ^ { ( i ) } , \tilde { \mathbf { w } } _ { r } ^ { ( i ) } \} _ { i = 1 } ^ { k }$ that represent semantically meaningful directions in the representation space. We present two methods for obtaining these anchors:

Ground-Truth Features (Linear Representation Bench). When ground-truth features are available—as in our Linear Representation Bench where features are known by construction—we directly use a random subset of k ground-truth feature directions:

$$
\tilde {w} _ {p} ^ {(i)} = W _ {p} ^ {\mathrm{true}} [:, i ], \quad \tilde {w} _ {r} ^ {(i)} = W _ {r} ^ {\mathrm{true}} [:, i ], \quad i \in \mathcal {K} \tag {22}
$$

where $\kappa \subset [ n ]$ is a randomly selected subset of size k.

Subpopulation Mean Embeddings. For real-world datasets where ground-truth features are unknown, we identify semantic subpopulations Luo et al. (2024b) and compute their mean representations. Specifically, given a labeled dataset $\bar { \mathcal { D } } = \{ ( s _ { j } , y _ { j } ) \} _ { j = 1 } ^ { N }$ , where $y _ { j } \in \{ 1 , \ldots , C \}$ , we compute the mean of representations for one subpopulation cluster, which is considered as a proxy for ground-truth features:

$$
\bar {\mathbf {x}} _ {p} ^ {(c)} = \frac {1}{| \{j : y _ {j} = c \} |} \sum_ {j: y _ {j} = c} \mathbf {x} _ {p} (s _ {j}) \tag {23}
$$

Then we normalize each mean representation to obtain anchor directions:

$$
\tilde {\mathbf {w}} _ {p} ^ {(c)} = \frac {\bar {\mathbf {x}} _ {p} ^ {(c)}}{\| \bar {\mathbf {x}} _ {p} ^ {(c)} \| _ {2}} \tag {24}
$$

# 4.2 SDL Loss Function with Feature Anchoring

Given k anchor features $\{ \tilde { \mathbf { w } } _ { p } ^ { ( i ) } , \tilde { \mathbf { w } } _ { r } ^ { ( i ) } \} _ { i = 1 } ^ { k }$ , we modify the SDL optimization objective to include an anchoring penalty that constrains the first k encoder rows and decoder columns to align with these anchors.

The complete anchored SDL objective is:

$$
\mathcal {L} _ {\mathrm{SDL-FA}} = \mathcal {L} _ {\mathrm{SDL}} + \lambda_ {\text { anchor }} \mathcal {L} _ {\text { anchor }} \tag {25}
$$

where $\mathcal { L } _ { \mathrm { S D L } }$ is the standard SDL loss (Equation 3) and the anchoring loss is:

$$
\mathcal {L} _ {\text {anchor}} = \left\| W _ {E} [ 1: k,: ] - \left[ \tilde {w} _ {p} ^ {(1)}, \dots , \tilde {w} _ {p} ^ {(k)} \right] ^ {\top} \right\| _ {F} ^ {2} \tag {26}
$$

$$
+ \| W _ {D} [:, 1: k ] - [ \tilde {w} _ {r} ^ {(1)}, \ldots , \tilde {w} _ {r} ^ {(k)} ] \| _ {F} ^ {2}
$$

This feature anchoring technique (Equation 26) reduces the underdetermined nature of SDL training, and is method-agnostic: it applies equally to SAEs, transcoders, crosscoders, and their variants (TopK SAEs, Matryoshka SAEs, etc.) since it only constrains the encoder $W _ { E }$ and decoder $W _ { D }$ matrices that are common to all SDL architectures. We believe this universality makes feature anchoring a broadly useful technique for improving performance across SDL methods.

Table 1: Feature recovery results on the Linear Representation Bench $( n = 1 0 0 0 , n _ { p } = n _ { r } = 7 6 8 , n _ { q } =$ 16000, ${ \cal S } = 0 . 9 9 )$ . Evaluation is performed only on features not used for anchoring. Feature anchoring (FA) consistently improves both GT Recovery $\left( \mathbf { M } _ { \mathbf { G T } } \right)$ and Maximum Inner Product $\left( \mathbf { M } _ { \mathbf { I P } } \right)$ across all SDL methods. \*: For some methods, $\mathbf { M } _ { \mathbf { G T } }$ is 0% because all best-match similarities fall below the evaluation threshold (0.95). In these cases, $\mathbf { M _ { I P } }$ , which remains well-aligned with $\mathbf { M } _ { \mathbf { G T } }$ , serves as a more informative indicator of recovery performance. See Figure 6 for a threshold-dependent view of $\mathbf { M } _ { \mathbf { G T } }$ . 

<table><tr><td>Method</td><td> $M_{GT} \uparrow$ </td><td> $M_{IP} \uparrow$ </td></tr><tr><td>ReLU SAE</td><td>0.00%*</td><td>0.205</td></tr><tr><td>+ Feature Anchoring</td><td>0.00%*</td><td>0.246</td></tr><tr><td>JumpReLU SAE</td><td>0.00%*</td><td>0.237</td></tr><tr><td>+ Feature Anchoring</td><td>0.00%*</td><td>0.333</td></tr><tr><td>TopK SAE</td><td>84.90%</td><td>0.983</td></tr><tr><td>+ Feature Anchoring</td><td>87.63%</td><td>0.986</td></tr><tr><td>BatchTopK SAE</td><td>84.80%</td><td>0.981</td></tr><tr><td>+ Feature Anchoring</td><td>89.38%</td><td>0.988</td></tr><tr><td>Matryoshka SAE</td><td>83.70%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>87.32%</td><td>0.985</td></tr><tr><td>Transcoder</td><td>23.60%</td><td>0.838</td></tr><tr><td>+ Feature Anchoring</td><td>25.05%</td><td>0.838</td></tr><tr><td>Crosscoder</td><td>56.42%</td><td>0.940</td></tr><tr><td>+ Feature Anchoring</td><td>57.71%</td><td>0.941</td></tr></table>

# 5 Experimental Results

We first evaluate feature anchoring with various SDL methods on the Linear Representation Bench (Section 5.1), a synthetic benchmark with fully accessible ground-truth features that precisely instantiates our theoretical assumptions. Second, we validate feature anchoring on real-world data by training SDL methods on CLIP embeddings of ImageNet-1K. Third, we provide empirical evidence showing that dead neuron resampling helps escape spurious local minima in large language models. Ablation studies are in Appendix G.

# 5.1 Results on The Linear Representation Bench

To validate our theoretical predictions, we design the Linear Representation Bench, a benchmark that precisely instantiates Assumptions 2.1 with fully known GT features.

Data Generation. We generate synthetic representations $\mathbf { x } _ { p } ( s ) = W _ { p } ^ { \mathrm { t r u e } } \mathbf { x } ( s )$ where ground-truth features $\mathbf { x } ( s ) \in \mathbb { R } ^ { n }$ follow shifted exponential distributions with sparsity S. The feature matrix $W _ { p } ^ { \mathrm { t r u e } } \in \mathbb { R } ^ { n _ { p } \times n }$ is constructed via gradient-based optimization (Detailed in Appendix C).

Metrics. Given learned features $W _ { p } ^ { \mathrm { l e a r n e d } } \ : = \ : W _ { E } ^ { \top } \in \mathbb { R } ^ { n _ { p } \times n _ { q } }$ and ground truth features $W _ { o } ^ { \mathrm { t r u e } } \in \mathbb { R } ^ { n _ { p } \times n }$ ∈ Rnp×n (columns are unit-norm feature directions), we compute the similarity matrix $\mathbf { S } = | W _ { p } ^ { \mathrm { l e a r n e d T } } W _ { p } ^ { \mathrm { t r u e } } | \in \mathbb { R } ^ { n _ { q } \times n }$ . For each ground truth feature $i \in [ n ]$ , we find its best match score: $s _ { i } = \operatorname* { m a x } _ { j } S _ { j i }$ .

• GT Recovery is the fraction of ground truth features with $s _ { i } { \mathrm { ~  ~  ~ } } \tau , { \bf M } _ { { \bf G T } } ( W _ { p } ^ { \mathrm { l e a r n e d } } ) = $ $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ s _ { i } > \tau \}$ .   
• Maximum Inner Product is the mean of best match scores: $\begin{array} { r } { \mathbf { M _ { I P } } ( W _ { p } ^ { l e a r n e d } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } s _ { i } } \end{array}$

As shown in Table 1 and Figure 6, feature anchoring significantly improves feature recovery performances.

![](images/db5b95ca6563775454922a5145fcbb3000466d20d70b790208d350928799bb46.jpg)  
Figure 6: $\mathbf { M } _ { \mathbf { G T } }$ vs. Threshold. Across all SDL methods, feature anchoring consistently improves feature recovery, as measured by $\mathbf { M } _ { \mathbf { G T } }$ over a range of thresholds.

# 5.2 Results on CLIP Embeddings of ImageNet

To validate that feature anchoring generalizes beyond synthetic benchmarks to real-world representations, we apply our method to CLIP embeddings of ImageNet-1K (Russakovsky et al., 2015). We extract CLIP-ViT-$\mathrm { B / 3 2 }$ (Radford et al., 2021) image embeddings for all ImageNet training images and compute ground-truth anchor features as normalized class mean embeddings: $\tilde { w } _ { p } ^ { ( c ) } = \bar { x } _ { p } ^ { ( c ) } / \| \bar { x } _ { p } ^ { ( c ) } \| _ { 2 }$ where $\bar { x } _ { p } ^ { ( c ) }$ is the mean embedding over all images in class c. We train TopK, BatchTopK, and Matryoshka SAEs with $n _ { p } = n _ { r } = 7 6 8 .$ , $n _ { q } = 1 6 , 3 8 4$ , and use $k = 3 0$ anchors with $\lambda _ { \mathrm { a n c h o r } } = 1 . 0$ . Table 2 shows feature anchoring consistently improves feature recovery (Examples in Appendix H).

Table 2: Feature recovery on CLIP embeddings. As ground-truth features are not explicitly available in this setting, we use subpopulation mean embeddings as a proxy for the underlying feature directions. \*: 0% recovery resulted from the same issue as Talble 1. 

<table><tr><td>Method</td><td> $M_{GT} \uparrow$ </td><td> $M_{IP} \uparrow$ </td></tr><tr><td>TopK SAE</td><td>0.00%*</td><td>0.517</td></tr><tr><td>+ Feature Anchoring</td><td>24.13%</td><td>0.851</td></tr><tr><td>BatchTopK SAE</td><td>0.00%*</td><td>0.508</td></tr><tr><td>+ Feature Anchoring</td><td>24.13%</td><td>0.847</td></tr><tr><td>Matryoshka SAE</td><td>0.00%*</td><td>0.683</td></tr><tr><td>+ Feature Anchoring</td><td>24.45%</td><td>0.858</td></tr></table>

# 5.3 Neuron Resampling Helps Escape Partial Minima

Our partial minima analysis (Theorem 3.6) proves the prevalence of spurious partial minima and connects them to dead neurons $( \mathcal { F } _ { i } = \varnothing )$ . To address this, we utilize neuron resampling (Bricken et al., 2023) to help SDL training overcome these partial minima. We argue that resampling reinitializes dead neurons toward under-reconstructed directions, perturbing the optimization away from the partial minima.

To validate this, we train SAEs on Llama 3.1 8B Instruct (layer 12, dimension 4096) with latent dimension 131,072 for 30,000 steps on FineWeb-Edu (Penedo et al., 2024), processing 4096 token activations per step. We compare standard training against feature resampling after 5000 steps. As shown in Figure 7, resampling enables the optimizer to escape spurious local minima, achieving lower final loss.

![](images/438d6c24ffdb471c5548a4fd0fcdef415c37f56a0e59021af0566a8d917808bd.jpg)

<details>
<summary>line</summary>

| Step | resample | no_resample |
| ---- | -------- | ----------- |
| 0    | 5.0      | 5.0         |
| 50   | 3.4      | 2.0         |
| 100  | 1.6      | 1.9         |
| 150  | 1.5      | 1.8         |
| 200  | 1.4      | 1.8         |
| 250  | 1.3      | 1.7         |
| 300  | 1.3      | 1.7         |
| 350  | 1.3      | 1.7         |
</details>

Figure 7: Feature resampling accelerates convergence and improves final loss. Training curves on Llama 3.1 8B comparing standard SAE training (blue) with periodic dead neuron resampling after 5000 steps (in the plot x axis is scaled by 100).

# 6 Conclusion

We develop the first unified theoretical framework for Sparse Dictionary Learning in mechanistic interpretability, demonstrating how diverse SDL methods instantiate a single optimization problem. We prove that SDL exhibits piecewise biconvex structure, bridging mechanistic interpretability with classical optimization theory. We characterize the global minimum and establish that the optimization is fundamentally underdetermined, admitting solutions that achieve zero reconstruction loss without recovering interpretable features. We demonstrate that spurious partial minima exhibiting polysemanticity are pervasive, and prove that hierarchical concept structures naturally induce feature absorption patterns that manifest as partial minima. To validate our theory, we design the Linear Representation Bench with fully accessible groundtruth features, and propose feature anchoring—a technique applicable to all SDL methods that addresses the underdetermined nature of optimization.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# Code Availability

We will provide full scripts of the Linear Representation Bench and feature anchoring upon acceptance.

# Declaration of LLM Usage

We admit the usage of Claude Sonnet-4.6 for coding and polishing the manuscript.

# References

Ahmed Abdulaal, Hugo Fry, Nina Montaña-Brown, Ayodeji Ijishakin, Jack Gao, Stephanie Hyland, Daniel C. Alexander, and Daniel C. Castro. An x-ray is worth 15 features: Sparse autoencoders for interpretable radiology report generation, 2024. URL https://arxiv.org/abs/2410.03334.   
Michal Aharon, Michael Elad, and Alfred Bruckstein. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on signal processing, 54(11):4311–4322, 2006.   
Leonard Bereska and Efstratios Gavves. Mechanistic interpretability for ai safety – a review, 2024. URL https://arxiv.org/abs/2404.14082.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nicholas L Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Chris Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. URL https://transformer-circuits.pub/2023/monosemantic-features.   
Bart Bussmann, Patrick Leask, and Neel Nanda. Batchtopk sparse autoencoders, 2024. URL https: //arxiv.org/abs/2412.06410.   
Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. Learning multi-level features with matryoshka sparse autoencoders, 2025. URL https://arxiv.org/abs/2503.17547.   
David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, Satvik Golechha, and Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2025. URL https: //arxiv.org/abs/2409.14507.   
Jingyi Cui, Qi Zhang, Yifei Wang, and Yisen Wang. On the theoretical understanding of identifiable sparse autoencoders and beyond, 2025. URL https://arxiv.org/abs/2506.15963.   
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL https://arxiv.org/abs/2309.08600.   
David L Donoho. Compressed sensing. IEEE Transactions on information theory, 52(4):1289–1306, 2006.   
Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders find interpretable llm feature circuits, 2024. URL https://arxiv.org/abs/2406.11944.   
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. URL https://arxiv.org/abs/2209.10652.   
Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear, 2025. URL https://arxiv.org/abs/2405.14860.   
Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders, 2024. URL https://arxiv.org/abs/ 2406.04093.   
Rong Ge, Jason D. Lee, and Tengyu Ma. Matrix completion has no spurious local minimum, 2018. URL https://arxiv.org/abs/1605.07272.   
Jochen Gorski, Frank Pfeuffer, and Kathrin Klamroth. Biconvex sets and optimization with biconvex functions: a survey and extensions. Mathematical methods of operations research, 66(3):373–407, 2007.   
Remi Gribonval and Karin Schnass. Dictionary identification - sparse matrix-factorisation via ℓ1- minimisation, 2010. URL https://arxiv.org/abs/0904.4774.   
Onkar Gujral, Mihir Bafna, Eric Alm, and Bonnie Berger. Sparse autoencoders uncover biologically interpretable features in protein language model representations. Proceedings of the National Academy of Sciences, 122(34):e2506316122, 2025.   
Prateek Jain and Purushottam Kar. Non-convex optimization for machine learning. Foundations and Trends® in Machine Learning, 10(3–4):142–336, 2017. ISSN 1935-8245. doi: 10.1561/2200000058. URL http://dx.doi.org/10.1561/2200000058.   
Yibo Jiang, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam, and Victor Veitch. On the origins of linear representations in large language models, 2024. URL https://arxiv.org/abs/2403.03867.

Adam Karvonen, Benjamin Wright, Can Rager, Rico Angell, Jannik Brinkmann, Logan Smith, Claudio Mayrink Verdun, David Bau, and Samuel Marks. Measuring progress in dictionary learning for language model interpretability with board game models, 2024. URL https://arxiv.org/abs/2408.00113.   
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav), 2018. URL https://arxiv.org/abs/1711.11279.   
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models, 2020. URL https://arxiv.org/abs/2007.04612.   
Daniel D Lee, P Pham, Y Largman, and A Ng. Advances in neural information processing systems 22. Tech Rep, 2009.   
Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, and Christopher Olah. Sparse crosscoders for cross-layer features and model diffing. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/crosscoders/index.html.   
Zachary C. Lipton. The mythos of model interpretability, 2017. URL https://arxiv.org/abs/1606.03490.   
Scott Lundberg and Su-In Lee. A unified approach to interpreting model predictions, 2017. URL https: //arxiv.org/abs/1705.07874.   
Yifan Luo, Yiming Tang, Chengfeng Shen, Zhennan Zhou, and Bin Dong. Prompt engineering through the lens of optimal control, 2023. URL https://arxiv.org/abs/2310.14201.   
Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model.   
Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model. In European Conference on Computer Vision, pp. 235–252. Springer, 2024a.   
Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model, 2024b. URL https://arxiv.org/ abs/2405.02363.   
Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro. Online dictionary learning for sparse coding. In Proceedings of the 26th annual international conference on machine learning, pp. 689–696, 2009.   
Alireza Makhzani and Brendan Frey. k-sparse autoencoders, 2014. URL https://arxiv.org/abs/1312. 5663.   
Ziming Mao, Jia Xu, Zeqi Zheng, Haofang Zheng, Dabing Sheng, Yaochu Jin, and Guoyuan Yang. Sparse autoencoders bridge the deep learning model and the brain, 2025. URL https://arxiv.org/abs/2506. 11123.   
Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https://arxiv.org/abs/2310.06824.   
Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models, 2025. URL https://arxiv.org/abs/2403.19647.   
Ibomoiye Domor Mienye and Nobert Jere. A survey of decision trees: Concepts, algorithms, and applications. IEEE access, 12:86716–86727, 2024.   
Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models, 2023. URL https://arxiv.org/abs/2309.00941.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. URL https://distill.pub/ 2020/circuits/zoom-in/.   
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022. URL https://arxiv.org/abs/2209.11895.   
Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models, 2024. URL https://arxiv.org/abs/2311.03658.   
Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models, 2025. URL https://arxiv.org/abs/2406.01506.   
Gonçalo Paulo, Stepan Shabalin, and Nora Belrose. Transcoders beat sparse autoencoders for interpretability, 2025. URL https://arxiv.org/abs/2501.18823.   
Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. URL https://arxiv.org/abs/2406.17557.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.   
Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda. Improving dictionary learning with gated sparse autoencoders, 2024a. URL https://arxiv.org/abs/2404.16014.   
Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders, 2024b. URL https://arxiv.org/abs/2407.14435.   
Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead, 2019. URL https://arxiv.org/abs/1811.10154.   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge, 2015. URL https://arxiv.org/abs/1409.0575.   
Itay Safran and Ohad Shamir. Spurious local minima are common in two-layer relu neural networks, 2018. URL https://arxiv.org/abs/1712.08968.   
Harshvardhan Saini, Yiming Tang, and Dianbo Liu. Bridging mechanistic interpretability and prompt engineering with gradient ascent for interpretable persona control, 2026. URL https://arxiv.org/abs/ 2601.02896.   
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. International Journal of Computer Vision, 128(2):336–359, October 2019. ISSN 1573-1405. doi: 10.1007/s11263-019-01228-7. URL http://dx.doi.org/10.1007/s11263-019-01228-7.   
Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi Schoots, Joseph Miller, Eric J. Michaud, Stephen Casper, Max Tegmark, William Saunders, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, and Tom McGrath. Open problems in mechanistic interpretability, 2025. URL https://arxiv.org/abs/2501.16496.

Dong Shu, Xuansheng Wu, Haiyan Zhao, Daking Rai, Ziyu Yao, Ninghao Liu, and Mengnan Du. A survey on sparse autoencoders: Interpreting the internal mechanisms of large language models, 2025. URL https://arxiv.org/abs/2503.05613.   
Elana Simon and James Zou. Interplm: Discovering interpretable features in protein language models via sparse autoencoders, 2024. URL https://arxiv.org/abs/2412.12101.   
Daniel A. Spielman, Huan Wang, and John Wright. Exact recovery of sparsely-used dictionaries, 2012. URL https://arxiv.org/abs/1206.5882.   
Ruoyu Sun and Zhi-Quan Luo. Guaranteed matrix completion via non-convex factorization. IEEE Transactions on Information Theory, 62(11):6535–6579, November 2016. ISSN 1557-9654. doi: 10.1109/tit.2016. 2598574. URL http://dx.doi.org/10.1109/TIT.2016.2598574.   
Yiming Tang and Bin Dong. Demonstration notebook: Finding the most suited in-context learning example from interactions, 2024. URL https://arxiv.org/abs/2406.10878.   
Yiming Tang, Arash Lagzian, Srinivas Anumasa, Qiran Zou, Yingtao Zhu, Ye Zhang, Trang Nguyen, Yih-Chung Tham, Ehsan Adeli, Ching-Yu Cheng, Yilun Du, and Dianbo Liu. Human-like content analysis for generative ai with language-grounded sparse encoders, 2025a. URL https://arxiv.org/abs/2508.18236.   
Yiming Tang, Abhijeet Sinha, and Dianbo Liu. How does my model fail? automatic identification and interpretation of physical plausibility failure modes with matryoshka transcoders, 2025b. URL https: //arxiv.org/abs/2511.10094.   
Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread, 2024. URL https: //transformer-circuits.pub/2024/scaling-monosemanticity/.   
Viswanathan Visweswaran and CA Floudast. A global optimization algorithm (gop) for certain classes of nonconvex nlps—ii. application of theory and test problems. Computers & chemical engineering, 14(12): 1419–1434, 1990.   
Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Jeffrey Wang, Achyuta Rajaram, Johannes Heidecke, Tejal Patwardhan, and Dan Mossing. Persona features control emergent misalignment, 2025. URL https://arxiv.org/abs/2506.19823.   
Haiyan Zhao, Zirui He, Yiming Tang, Fan Yang, Ali Payani, Dianbo Liu, and Mengnan Du. Rep2text: Decoding full text from a single llm token representation, 2026. URL https://arxiv.org/abs/2511. 06571.   
Qiran Zou, Hou Hei Lam, Wenhao Zhao, Yiming Tang, Tingting Chen, Samson Yu, Tianyi Zhang, Chang Liu, Xiangyang Ji, and Dianbo Liu. Fml-bench: Benchmarking machine learning agents for scientific research, 2026. URL https://arxiv.org/abs/2510.10472.

# A Taxonomy for Sparse Dictionary Learning in Mechanistic Interpretability

![](images/c4f51dac4739d36525bbcc43737c236a12d2f4f771e40651a9366254a4e95a96.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["SDL in MechInterp"] --> B["Architecture"]
    A --> C["Activation"]
    A --> D["Benchmarks"]
    A --> E["Phenomena"]
    A --> F["Applications"]

    B --> B1["Sparse Autoencoders (Cunningham et al., 2023)"]
    B --> B2["Transcoders (Dunefsky et al., 2024)"]
    B --> B3["Crosscoders (Lindsey et al., 2024)"]
    B --> B4["Binary Autoencoder (Cho et al., 2025)"]

    C --> C1["ReLU SAE (Bricken et al., 2023)"]
    C --> C2["Top-k SAE (Makhzani & Frey, 2014)"]
    C --> C3["Batch Top-k SAE (Bussmann et al., 2024)"]
    C --> C4["JumpReLU SAE (Rajamanoharan et al., 2024b)"]
    C --> C5["Gated SAE (Rajamanoharan et al., 2024a)"]

    D --> D1["Neuron Resampling (Bricken et al., 2023)"]
    D --> D2["Auxiliary Loss (Ghost Grads) (Gao et al., 2024)"]
    D --> D3["Tied Encoder-Decoder Init (Gao et al., 2024)"]
    D --> D4["Decoder Normalization (Bricken et al., 2023)"]
    D --> D5["Matryoshka SAEs (Bussmann et al., 2025)"]
    D --> D6["Matryoshka Transcoders (Tang et al., 2025b)"]

    E --> E1["SAEBench (Karvonen et al., 2024)"]
    E --> E2["Automated Interpretability (Bills et al., 2023)"]
    E --> E3["Sparse Probing (Gurnee et al., 2023)"]
    E --> E4["SDL Scaling Laws (Gao et al., 2024)"]
    E --> E5["MIB Benchmark (Mueller et al., 2025)"]
    E --> E6["AXBench (Wu et al., 2025)"]

    F --> F1["Feature Superposition (Elhage et al., 2022)"]
    F --> F2["Feature Absorption (Chanin et al., 2025)"]
    F --> F3["Feature Splitting (Chanin et al., 2025)"]
    F --> F4["Dead Neurons (Bricken et al., 2023)"]
    F --> F5["Polysemanticity (Scherlis et al., 2022)"]

    G --> G1["Circuit Analysis (Marks et al., 2025)"]
    G --> G2["Feature Steering (Templeton et al., 2024)"]
    G --> G3["Model Editing (Farrell et al., 2024)"]
    G --> G4["Adversarial Robustness (Bereska & Gavves, 2024)"]
    G --> G5["Medical Diagnosis (Abdulaal et al., 2024)"]
    G --> G6["Scientific Discovery (Gujral et al., 2025)"]
    G --> G7["Failure Mode Analysis (Tang et al., 2025b)"]
    G --> G8["Content Analysis (Tang et al., 2025a)"]
```
</details>

Figure 8: Hierarchical taxonomy of Sparse Dictionary Learning research in Mechanistic Interpretability.

# B Notations

<table><tr><td>Notation</td><td>Description</td></tr><tr><td colspan="2">Distributions and Indexing</td></tr><tr><td> $\mathcal{D}$ </td><td>Distribution over inputs</td></tr><tr><td> $\mathcal{X}$ </td><td>Input space</td></tr><tr><td>s</td><td>Sample drawn from  $\mathcal{D}$ </td></tr><tr><td> $\mathbf{e}_{d}$ </td><td>Standard basis vector (1 in position d, 0 elsewhere)</td></tr><tr><td colspan="2">Representations and Dimensions</td></tr><tr><td> $\mathbf{x}(s)$ </td><td>Ground-truth features for input s~ $\mathcal{D}$ </td></tr><tr><td> $\mathbf{x}_{p}(s)$ </td><td>Input representation to SDL model (observed)</td></tr><tr><td> $\mathbf{x}_{r}(s)$ </td><td>Target/output representation for SDL model</td></tr><tr><td> $\mathbf{x}_{q}(s)$ </td><td>Latent activations in SDL bottleneck</td></tr><tr><td>n</td><td>Number of ground-truth features (dimension of x)</td></tr><tr><td> $n_{p}$ </td><td>Dimension of input representation  $\mathbf{x}_{p}$ </td></tr><tr><td> $n_{r}$ </td><td>Dimension of target representation  $\mathbf{x}_{r}$ </td></tr><tr><td> $n_{q}$ </td><td>Dimension of latent space  $\mathbf{x}_{q}$ </td></tr><tr><td colspan="2">SDL Architecture</td></tr><tr><td> $W_{E}$ </td><td>Encoder matrix ( $n_{q} \times n_{p}$ )</td></tr><tr><td> $W_{D}$ </td><td>Decoder matrix ( $n_{r} \times n_{q}$ )</td></tr><tr><td> $\sigma(\cdot)$ </td><td>Sparsity-inducing activation function</td></tr><tr><td> $\mathbf{w}_{E}^{i}$ </td><td>i-th row of encoder  $W_{E}$  (encodes to neuron i)</td></tr><tr><td> $\mathbf{w}_{D}^{i}$ </td><td>i-th column of decoder  $W_{D}$  (decodes from neuron i)</td></tr><tr><td colspan="2">Linear Representation Hypothesis</td></tr><tr><td> $W_{p}$ </td><td>Feature matrix for  $\mathbf{x}_{p}$  ( $n_{p} \times n$ )</td></tr><tr><td> $W_{r}$ </td><td>Feature matrix for  $\mathbf{x}_{r}$  ( $n_{r} \times n$ )</td></tr><tr><td> $\mathbf{w}_{p}^{d}$ </td><td>d-th column of  $W_{p}$  (feature direction for feature d)</td></tr><tr><td> $\mathbf{w}_{r}^{d}$ </td><td>d-th column of  $W_{r}$  (feature direction for feature d)</td></tr><tr><td> $W_{p}^{\text{true}}$ </td><td>Ground-truth feature matrix (Linear Representation Bench)</td></tr><tr><td> $W_{p}^{\text{learned}}$ </td><td>Learned feature matrix ( $W_{E}^{\top}$ )</td></tr><tr><td colspan="2">Sparsity and Interference</td></tr><tr><td>S</td><td>Sparsity level: Pr( $\mathbf{x}_{i}(s)=0$ ) ≥ S</td></tr><tr><td>M</td><td>Maximum interference: max $_{i\neq j}\langle W_{p}[:,i],W_{p}[:,j]\rangle$ </td></tr><tr><td> $M_{d}$ </td><td>Weight for feature d: Pr( $\mathbf{x}(s)=x_{d}(s)\mathbf{e}_{d}$ )· $\mathbb{E}[x_{d}(s)^{2}|x_{d}(s)>0]$ </td></tr><tr><td colspan="2">Feature Anchoring</td></tr><tr><td>k</td><td>Number of anchor features</td></tr><tr><td> $\tilde{\mathbf{w}}_{p}^{(i)},\tilde{\mathbf{w}}_{r}^{(i)}$ </td><td>i-th anchor feature pair</td></tr><tr><td> $\lambda_{anchor}$ </td><td>Anchoring loss weight</td></tr><tr><td> $\mathcal{K}$ </td><td>Set of indices for selected anchor features</td></tr><tr><td colspan="2">Loss Functions</td></tr><tr><td> $\mathcal{L}_{SDL}$ </td><td>Standard SDL reconstruction loss</td></tr><tr><td> $\tilde{\mathcal{L}}_{SDL}$ </td><td>Approximate SDL loss (extreme sparsity regime)</td></tr><tr><td> $\mathcal{L}_{anchor}$ </td><td>Feature anchoring penalty</td></tr><tr><td> $\mathcal{L}_{SDL-FA}$ </td><td>Anchored SDL objective:  $\mathcal{L}_{SDL}+\lambda_{anchor}\mathcal{L}_{anchor}$ </td></tr><tr><td colspan="2">Activation Patterns</td></tr><tr><td> $\mathcal{P}=(\mathcal{F}_{1},\ldots,\mathcal{F}_{N})$ </td><td>Activation pattern</td></tr><tr><td> $\mathcal{F}_{i}$ </td><td>Set of ground-truth features activating neuron i</td></tr><tr><td> $\mathcal{A}(d)$ </td><td>Set of neurons activated by feature d</td></tr><tr><td> $\Omega_{\mathcal{A}}$ </td><td>Activation pattern region (fixed activation structure)</td></tr><tr><td colspan="2">Evaluation Metrics</td></tr><tr><td> $\mathbf{M}_{GT}$ </td><td>GT Recovery: fraction of features with &gt;τ alignment</td></tr><tr><td> $\mathbf{M}_{IP}$ </td><td>Maximum Inner Product: mean best alignment per feature</td></tr><tr><td>τ</td><td>Threshold for GT Recovery (typically τ = 0.9 or 0.95)</td></tr></table>

# C The Linear Representation Bench

The Linear Representation Bench is a synthetic benchmark designed to precisely instantiate the Representation Assumptions (Assumption 2.1) with fully accessible ground-truth features (See Figure 9 for a visualization of the benchmark). This enables rigorous evaluation of SDL methods under controlled conditions where feature recovery can be directly measured.

![](images/261b961f0979c1cd183b05eb0463ad8aa48ac22f3c79eaa46b13bc7b18c1b79b.jpg)

<details>
<summary>surface_3d</summary>

| Feature | Dimension 1 | Dimension 2 |
|---------|-------------|-------------|
| W₁      | 0.0         | -0.5        |
| W₂      | 0.5         | 1.0         |
| W₃      | 0.5         | -0.5        |
| W₄      | 0.0         | 0.0         |
</details>

![](images/71c8ededa22c35b2771798625219694a110b66a38cb5d5185aef92ccc8e7892e.jpg)

<details>
<summary>scatter_3d</summary>

| Dimension 1 | Dimension 2 |
|-------------|-------------|
| -6          | -6          |
| -4          | -4          |
| -2          | -2          |
| 0           | 0           |
| 2           | 2           |
| 4           | 4           |
| 6           | 6           |
</details>

Figure 9: Visualization of the Linear Representation Bench. The figure illustrates a $D = 3$ dimensional representation space generated using $N = 4$ feature directions $( \mathbf { W } _ { p } )$ .

# C.1 Ground-Truth Feature Matrix Generation

We construct the feature matrix $\mathbf { W } _ { p } \in \mathbb { R } ^ { n _ { p } \times n }$ to satisfy the representation assumptions (Assumption 2.1).

Initialization. Initialize $\mathbf { W } _ { p }$ with random Gaussian entries and normalize each column to unit $\ell _ { 2 } { \mathrm { - n o r m } } \colon$

$$
\mathbf {w} _ {p} ^ {d} \leftarrow \frac {\mathbf {w} _ {p} ^ {d}}{\| \mathbf {w} _ {p} ^ {d} \| _ {2}}, \quad \forall d \in [ n ] \tag {27}
$$

Interference Minimization. We minimize pairwise interference via projected gradient descent. Define the soft-thresholded interference loss:

$$
\mathcal {L} _ {\mathrm{int}} (\mathbf {W} _ {p}) = \sum_ {i \neq j} \left[ \max \left(0, \langle \mathbf {w} _ {p} ^ {i}, \mathbf {w} _ {p} ^ {j} \rangle - (M - \epsilon)\right) \right] ^ {2} + \lambda \sum_ {i \neq j} \left[ \max \left(0, \langle \mathbf {w} _ {p} ^ {i}, \mathbf {w} _ {p} ^ {j} \rangle\right) \right] ^ {2} \tag {28}
$$

where $\epsilon > 0$ is a tolerance margin and $\lambda > 0$ is a small regularization weight encouraging negative interference.

At each iteration, we compute the gradient with respect to $\mathbf { W } _ { p } ,$ perform a gradient step, and project back to the unit sphere:

$$
\mathbf {W} _ {p} \leftarrow \mathbf {W} _ {p} - \eta \nabla_ {\mathbf {W} _ {p}} \mathcal {L} _ {\mathrm{int}}, \quad \mathbf {w} _ {p} ^ {d} \leftarrow \frac {\mathbf {w} _ {p} ^ {d}}{\| \mathbf {w} _ {p} ^ {d} \| _ {2}}, \quad \forall d \in [ n ] \tag {29}
$$

This procedure continues until the maximum interference satisfies ma $\mathrm { x } _ { i \neq j } \langle \mathbf { w } _ { p } ^ { i } , \mathbf { w } _ { p } ^ { j } \rangle \leq M$ .

# C.2 Sparse Coefficient Generation

For each sample $i \in [ N ]$ , we generate sparse ground-truth features $\mathbf { x } ^ { ( i ) } \in \mathbb { R } _ { + } ^ { n }$ as follows:

Sparsity Mask. Each feature $d \in [ n ]$ is independently activated with probability $( 1 - { \mathcal { S } } )$ :

$$
m _ {d} ^ {(i)} \sim \text { Bernoulli } (1 - \mathcal {S}) \tag {30}
$$

Feature Magnitudes. Active features follow a shifted exponential distribution:

$$
x _ {d} ^ {(i)} = m _ {d} ^ {(i)} \cdot (c _ {\min} + \mathrm{Exp} (\beta)) \tag {31}
$$

where $c _ { \operatorname* { m i n } } \geq 0$ is a minimum activation threshold and $\beta > 0$ is the scale parameter.

This construction ensures:

• Non-negativity: $x _ { d } ^ { ( i ) } \geq 0$ x(i)d ≥ 0 for all d, i $d , i$   
• Sparsity: $\operatorname* { P r } ( x _ { d } ^ { ( i ) } = 0 ) = \mathcal { S }$ for all d   
$\{ x _ { d } ^ { ( i ) } \} _ { d = 1 } ^ { n }$

# C.3 Data Synthesis

The observed representations are computed via linear combination:

$$
\mathbf {x} _ {p} ^ {(i)} = \mathbf {W} _ {p} \mathbf {x} ^ {(i)} = \sum_ {d = 1} ^ {n} x _ {d} ^ {(i)} \mathbf {w} _ {p} ^ {d} \tag {32}
$$

By construction, this dataset exactly satisfies Assumptions 2.1, enabling evaluation of SDL methods with complete knowledge of ground-truth features.

# C.4 Default Configuration

Table 3 lists the default parameters used in our experiments.

Table 3: Default configuration for the Linear Representation Bench. 

<table><tr><td>Parameter</td><td>Symbol</td><td>Default Value</td></tr><tr><td>Number of features</td><td>n</td><td>1000</td></tr><tr><td>Representation dimension</td><td>np</td><td>768</td></tr><tr><td>Superposition ratio</td><td>n/np</td><td>1.30×</td></tr><tr><td>Number of samples</td><td>N</td><td>100,000</td></tr><tr><td>Sparsity level</td><td>S</td><td>0.99</td></tr><tr><td>Maximum interference</td><td>M</td><td>0.1</td></tr></table>

This controlled setting enables direct measurement of feature recovery metrics with ground-truth access, complementing evaluations on real neural network representations where true features are unknown.

# D Related Works

# D.1 Mechanistic Interpretability

Interpretability is crucial for deploying AI in high-stakes domains such as medical diagnosis and financial modeling, where understanding model decisions is essential for safety and trust (Simon & Zou, 2024; Abdulaal et al., 2024; Zhao et al., 2026). Traditional approaches include interpretability-by-design methods like Concept Bottleneck Models (Koh et al., 2020) and decision trees (Mienye & Jere, 2024). Concept Activation Vectors (Kim et al., 2018) extend this by identifying human-defined concept directions in neural network representation spaces. Post-hoc explanation methods like GradCAM (Selvaraju et al., 2019) and SHAP (Lundberg & Lee, 2017) provide local explanations without modifying the model architecture. Mechanistic interpretability (Sharkey et al., 2025; Bereska & Gavves, 2024) aims to reverse-engineer neural networks by understanding their internal computational mechanisms. SAEs (Shu et al., 2025) and related dictionary learning methods (Tang et al., 2025a; Dunefsky et al., 2024) decompose neural activations into sparse, interpretable features. Circuit analysis (Olah et al., 2020; Olsson et al., 2022) investigates how these features compose into algorithms.

# D.2 Sparse Dictionary Learning

Sparse dictionary learning has a rich history predating its application to mechanistic interpretability. K-SVD (Aharon et al., 2006) established foundational methods for learning overcomplete dictionaries, while theoretical work in compressed sensing (Donoho, 2006) characterized recovery conditions, with Spielman et al. (2012) providing polynomial-time algorithms for exact reconstruction under sparsity assumptions. Safran & Shamir (2018) demonstrated that spurious local minima are common even in simple two-layer ReLU networks, highlighting optimization challenges that persist in modern applications. Recent work has adapted these principles to mechanistic interpretability. SAEs (Cunningham et al., 2023) apply dictionary learning to language model activations, with LLMs widely utilized to analyze discovered features (Luo et al., 2024a; Luo et al.; Tang & Dong, 2024; Zou et al., 2026). Various variants are further developed, including transcoders (Dunefsky et al., 2024), crosscoders (Gao et al., 2024), Matryoshka SAEs (Bussmann et al., 2025), and hybrid approaches like Language-Grounded Sparse Encoders (Tang et al., 2025a). These methods have found applications beyond language models, including protein structure analysis (Simon & Zou, 2024; Gujral et al., 2025), medical imaging (Abdulaal et al., 2024), model evaluation (Tang et al., 2025b), prompt engineering (Saini et al., 2026), board game analysis (Karvonen et al., 2024), and fMRI data analysis Mao et al. (2025).

# D.3 Biconvex Optimization

Biconvex optimization studies problems where the objective is convex in each variable block when the other is fixed. Gorski et al. (2007) provide a comprehensive survey establishing theoretical foundations and algorithmic approaches, while Visweswaran & Floudast (1990) develop the GOP algorithm providing global optimality guarantees via branch-and-bound. Matrix factorization problems exhibit similar bilinear structure; Lee et al. (2009) introduce non-negative matrix factorization with multiplicative updates, and subsequent work establishes landscape properties, with Ge et al. (2018) proving matrix completion has no spurious local minima and Sun & Luo (2016) providing convergence guarantees for alternating minimization. More broadly, Jain & Kar (2017) survey non-convex optimization in machine learning, while Safran & Shamir (2018) demonstrate that spurious local minima are common in two-layer ReLU networks. Our work bridges these optimization-theoretic foundations with mechanistic interpretability by proving SDL exhibits biconvex structure, enabling the application of established algorithms and analysis techniques to this emerging field.

# E Limitations

While our theoretical framework provides valuable insights into SDL methods, several limitations warrant discussion:

• Assumption Violations. Our analysis relies on the Representation Assumptions (Assumption 2.1), which may not hold perfectly in real-world neural networks. In particular, there exists features that are not one-dimentional linear (Engels et al., 2025) .   
• Extreme Sparsity. Several key results, including Theorem 3.1, rely on the extreme sparsity assumption $\textstyle ( 1 - S < { \frac { 1 } { n } } )$ . The bounds may degrade substantially for moderate sparsity levels commonly observed in practice.   
• Feature Independence. We assume mutual independence among ground-truth features, but realworld concepts often exhibit correlations that our analysis does not capture.   
• Convergence Guarantees. We characterize global and partial minima of the optimization landscape but do not provide guarantees on whether gradient-based methods converge to global minima.   
• Anchor Availability. Feature anchoring requires access to known semantic directions, which may not always be available. The quality and coverage of anchors significantly impact performance.

# F Future Works

Our theoretical framework opens several avenues for future research:

• Global Optimization via GOP. Having established that SDL is piecewise biconvex (Theorem 3.2), applying Global Optimization for biconvex Problems (GOP) (Gorski et al., 2007) could provide certificates of global optimality and systematically escape spurious partial minima.   
• Alternating Convex Search. Alternating convex search (ACS) (Gorski et al., 2007) directly exploits biconvex structure by alternately solving convex subproblems for $W _ { D }$ and $W _ { E }$ . This may offer faster convergence and better solutions than standard gradient descent.   
• Convergence and Sample Complexity. Building on our landscape characterization, future work should establish convergence rates for gradient descent and sample complexity bounds for feature recovery.   
• Moderate Sparsity Analysis. Extending results beyond the S → 1 regime to handle feature co-activation would broaden practical applicability.   
• Classical Dictionary Learning. Adapting established algorithms like K-SVD (Aharon et al., 2006) and online dictionary learning (Mairal et al., 2009) to the SDL setting could yield improved optimization methods.   
• Automatic Anchor Discovery. Developing methods to automatically discover high-quality anchors without external supervision would make feature anchoring more practical.

# G Ablation Studies

Our theoretical analysis reveals that SDL optimization becomes increasingly underdetermined as the number of ground-truth features n grows relative to the representation dimension $n _ { p }$ (Theorem 3.4). To comprehensively validate our theoretical framework and assess the robustness of feature anchoring across different conditions, we conduct extensive ablation studies on the Linear Representation Bench, systematically varying superposition ratio, interference level, sparsity, and activation function parameters.

Effect of Superposition Ratio. We first examine how feature anchoring performs under varying degrees of superposition by training ReLU SAEs with $n _ { q } = 1 6 , 0 0 0$ latent dimensions while varying the number of ground-truth features n ∈ {800, 900, 1000, 1100, 1200, 1300, 1400}. The representation dimension is fixed at $n _ { p } = n _ { r } = 7 6 8$ , yielding superposition ratios from 1.04× to 1.82×. For feature anchoring, we randomly select k = 100 ground-truth features as anchors with $\lambda _ { \mathrm { a n c h o r } } = 0 . 1$ .

As shown in Table 4, standard SAE training achieves 0% GT Recovery across all feature counts, demonstrating the severity of the underdetermined optimization problem. With feature anchoring, GT Recovery reaches 100% at $n = 8 0 0$ and remains high (98.89%) at $n = 9 0 0$ , confirming that anchoring a small subset of features (12.5% and 11.1% respectively) provides sufficient constraint to guide optimization toward the global minimum. As superposition increases, GT Recovery gradually decreases (85.50% at 1.30×, 58.09% at 1.43×, 30.58% at 1.56×), aligning with our theoretical expectation that higher superposition expands the space of spurious solutions. Notably, feature anchoring consistently improves Maximum Inner Product across all settings, indicating better feature quality even when full recovery is not achieved.

Table 4: Feature recovery under varying superposition ratios. We fix $n _ { p } = n _ { r } = 7 6 8$ and vary n from 800 to 1400. Feature anchoring uses k = 100 randomly selected anchors with $\lambda _ { \mathrm { a n c h o r } } = 0 . 1$ . 

<table><tr><td>Method</td><td>Num Features</td><td>GT Recovery ↑</td><td>Max Inner Product ↑</td></tr><tr><td>SAE</td><td>800</td><td>0.00%</td><td>0.361</td></tr><tr><td>+ Feature Anchoring</td><td>800</td><td>100.00%</td><td>0.436</td></tr><tr><td>SAE</td><td>900</td><td>0.00%</td><td>0.298</td></tr><tr><td>+ Feature Anchoring</td><td>900</td><td>98.89%</td><td>0.345</td></tr><tr><td>SAE</td><td>1000</td><td>0.00%</td><td>0.254</td></tr><tr><td>+ Feature Anchoring</td><td>1000</td><td>85.50%</td><td>0.292</td></tr><tr><td>SAE</td><td>1100</td><td>0.00%</td><td>0.223</td></tr><tr><td>+ Feature Anchoring</td><td>1100</td><td>58.09%</td><td>0.258</td></tr><tr><td>SAE</td><td>1200</td><td>0.00%</td><td>0.203</td></tr><tr><td>+ Feature Anchoring</td><td>1200</td><td>30.58%</td><td>0.233</td></tr><tr><td>SAE</td><td>1300</td><td>0.00%</td><td>0.192</td></tr><tr><td>+ Feature Anchoring</td><td>1300</td><td>15.23%</td><td>0.216</td></tr><tr><td>SAE</td><td>1400</td><td>0.00%</td><td>0.181</td></tr><tr><td>+ Feature Anchoring</td><td>1400</td><td>8.57%</td><td>0.207</td></tr></table>

Effect of Maximum Interference. To examine how feature anchoring performs under different interference levels, we fix $n = 1 0 0 0$ features in $n _ { p } = n _ { r } = 7 6 8$ dimensions and vary the maximum interference $M \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 5 \}$ during ground-truth feature matrix generation. Recall that $M : = \mathrm { m a x } _ { i \neq j } \langle W _ { p } [ $ : $, i ] , W _ { p } [ : , j ] \rangle$ quantifies the maximum dot product between distinct feature directions, characterizing the degree of feature overlap in the representation space.

Table 5 presents results across different interference levels. For TopK, BatchTopK, and Matryoshka SAEs, feature anchoring consistently improves GT Recovery across all interference values, with improvements ranging from 2-7 percentage points. For instance, at $M = 0 . 5$ , BatchTopK SAE improves from 83.7% to 89.6% with anchoring. ReLU and JumpReLU SAEs, which struggle with feature recovery even with anchoring (0% GT Recovery), still show improved Maximum Inner Product (e.g., ReLU at $M = 0 . 2 \colon 0 . 2 0 4  0 . 2 4 5 )$ , indicating better feature alignment. Interestingly, performance degrades slightly at very low interference $( M = 0 . 0 5 )$ compared to moderate interference $( M = 0 . 1 , 0 . 2 )$ , likely because extremely low interference creates near-orthogonal features that are easier to recover without additional constraints, making the anchoring less critical.

Table 5: Feature recovery under varying maximum interference M. All experiments use n = 1000 features with $n _ { p } = n _ { r } = 7 6 8$ and k = 100 anchors. 

<table><tr><td>Method</td><td>Max Interference</td><td>GT Recovery ↑</td><td>Max Inner Product ↑</td></tr><tr><td colspan="4">Low Interference (M = 0.05)</td></tr><tr><td>ReLU SAE</td><td>0.05</td><td>0.0%</td><td>0.205</td></tr><tr><td>+ Feature Anchoring</td><td>0.05</td><td>0.0%</td><td>0.246</td></tr><tr><td>JumpReLU SAE</td><td>0.05</td><td>0.0%</td><td>0.239</td></tr><tr><td>+ Feature Anchoring</td><td>0.05</td><td>0.0%</td><td>0.326</td></tr><tr><td>TopK SAE</td><td>0.05</td><td>84.3%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>0.05</td><td>87.3%</td><td>0.985</td></tr><tr><td>BatchTopK SAE</td><td>0.05</td><td>86.1%</td><td>0.984</td></tr><tr><td>+ Feature Anchoring</td><td>0.05</td><td>87.5%</td><td>0.986</td></tr><tr><td>Matryoshka SAE</td><td>0.05</td><td>83.8%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>0.05</td><td>85.8%</td><td>0.985</td></tr><tr><td colspan="4">Moderate Interference (M = 0.2)</td></tr><tr><td>ReLU SAE</td><td>0.2</td><td>0.0%</td><td>0.204</td></tr><tr><td>+ Feature Anchoring</td><td>0.2</td><td>0.0%</td><td>0.245</td></tr><tr><td>JumpReLU SAE</td><td>0.2</td><td>0.0%</td><td>0.237</td></tr><tr><td>+ Feature Anchoring</td><td>0.2</td><td>0.0%</td><td>0.333</td></tr><tr><td>TopK SAE</td><td>0.2</td><td>85.2%</td><td>0.984</td></tr><tr><td>+ Feature Anchoring</td><td>0.2</td><td>86.8%</td><td>0.986</td></tr><tr><td>BatchTopK SAE</td><td>0.2</td><td>84.7%</td><td>0.981</td></tr><tr><td>+ Feature Anchoring</td><td>0.2</td><td>90.7%</td><td>0.989</td></tr><tr><td>Matryoshka SAE</td><td>0.2</td><td>84.7%</td><td>0.983</td></tr><tr><td>+ Feature Anchoring</td><td>0.2</td><td>87.6%</td><td>0.986</td></tr><tr><td colspan="4">High Interference (M = 0.5)</td></tr><tr><td>ReLU SAE</td><td>0.5</td><td>0.0%</td><td>0.203</td></tr><tr><td>+ Feature Anchoring</td><td>0.5</td><td>0.0%</td><td>0.243</td></tr><tr><td>JumpReLU SAE</td><td>0.5</td><td>0.0%</td><td>0.237</td></tr><tr><td>+ Feature Anchoring</td><td>0.5</td><td>0.0%</td><td>0.325</td></tr><tr><td>TopK SAE</td><td>0.5</td><td>84.2%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>0.5</td><td>88.2%</td><td>0.987</td></tr><tr><td>BatchTopK SAE</td><td>0.5</td><td>83.7%</td><td>0.981</td></tr><tr><td>+ Feature Anchoring</td><td>0.5</td><td>89.6%</td><td>0.987</td></tr><tr><td>Matryoshka SAE</td><td>0.5</td><td>83.9%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>0.5</td><td>86.4%</td><td>0.985</td></tr></table>

Effect of Feature Sparsity. We investigate how feature sparsity S affects SDL performance by varying $S \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 \}$ where S denotes the probability that each feature is inactive. Lower S values correspond to denser activation patterns, while higher S approaches the extreme sparsity regime analyzed in our theoretical results.

Table 6 reveals several key insights. At very low sparsity $( S = 0 . 0 0 5 )$ , where features co-activate frequently, feature anchoring provides the most dramatic improvements. JumpReLU SAE improves from 11.8% to 74.8% GT Recovery, and all TopK-family methods show 2-6 percentage point gains. This validates that anchoring is particularly valuable when the extreme sparsity assumption is violated and feature co-occurrence complicates optimization. At moderate sparsity $( S \ : = \ : 0 . 0 5 )$ , performance drops significantly across all methods, as increased co-activation creates more complex interference patterns. At high sparsity $( S = 0 . 1 )$ , approaching our theoretical regime, TopK-family methods achieve near-perfect recovery (98.8%-99.7%), with anchoring providing marginal improvements. Interestingly, Matryoshka SAE benefits least from anchoring at extreme sparsity, likely because its multi-scale architecture already provides sufficient constraints to recover features.

Table 6: Feature recovery under varying sparsity levels S. All experiments use $n = 1 0 0 0$ features with $n _ { p } = n _ { r } = 7 6 8$ and k = 100 anchors. 

<table><tr><td>Method</td><td>Sparsity</td><td>GT Recovery ↑</td><td>Max Inner Product ↑</td></tr><tr><td colspan="4">Very Low Sparsity (S = 0.005)</td></tr><tr><td>ReLU SAE</td><td>0.005</td><td>0.0%</td><td>0.231</td></tr><tr><td>+ Feature Anchoring</td><td>0.005</td><td>0.0%</td><td>0.326</td></tr><tr><td>JumpReLU SAE</td><td>0.005</td><td>11.8%</td><td>0.890</td></tr><tr><td>+ Feature Anchoring</td><td>0.005</td><td>74.8%</td><td>0.970</td></tr><tr><td>TopK SAE</td><td>0.005</td><td>82.2%</td><td>0.978</td></tr><tr><td>+ Feature Anchoring</td><td>0.005</td><td>84.3%</td><td>0.981</td></tr><tr><td>BatchTopK SAE</td><td>0.005</td><td>80.5%</td><td>0.976</td></tr><tr><td>+ Feature Anchoring</td><td>0.005</td><td>83.2%</td><td>0.980</td></tr><tr><td>Matryoshka SAE</td><td>0.005</td><td>79.3%</td><td>0.974</td></tr><tr><td>+ Feature Anchoring</td><td>0.005</td><td>86.1%</td><td>0.983</td></tr><tr><td colspan="4">Moderate Sparsity (S = 0.01)</td></tr><tr><td>ReLU SAE</td><td>0.01</td><td>0.00%</td><td>0.205</td></tr><tr><td>+ Feature Anchoring</td><td>0.01</td><td>0.00%</td><td>0.246</td></tr><tr><td>JumpReLU SAE</td><td>0.01</td><td>0.00%</td><td>0.237</td></tr><tr><td>+ Feature Anchoring</td><td>0.01</td><td>0.00%</td><td>0.333</td></tr><tr><td>TopK SAE</td><td>0.01</td><td>84.90%</td><td>0.983</td></tr><tr><td>+ Feature Anchoring</td><td>0.01</td><td>87.63%</td><td>0.986</td></tr><tr><td>BatchTopK SAE</td><td>0.01</td><td>84.80%</td><td>0.981</td></tr><tr><td>+ Feature Anchoring</td><td>0.01</td><td>89.38%</td><td>0.988</td></tr><tr><td>Matryoshka SAE</td><td>0.01</td><td>83.70%</td><td>0.982</td></tr><tr><td>+ Feature Anchoring</td><td>0.01</td><td>87.32%</td><td>0.985</td></tr><tr><td colspan="4">High Sparsity (S = 0.1)</td></tr><tr><td>ReLU SAE</td><td>0.1</td><td>0.0%</td><td>0.186</td></tr><tr><td>+ Feature Anchoring</td><td>0.1</td><td>0.0%</td><td>0.191</td></tr><tr><td>JumpReLU SAE</td><td>0.1</td><td>0.0%</td><td>0.191</td></tr><tr><td>+ Feature Anchoring</td><td>0.1</td><td>0.0%</td><td>0.193</td></tr><tr><td>TopK SAE</td><td>0.1</td><td>98.8%</td><td>0.981</td></tr><tr><td>+ Feature Anchoring</td><td>0.1</td><td>98.4%</td><td>0.979</td></tr><tr><td>BatchTopK SAE</td><td>0.1</td><td>95.6%</td><td>0.975</td></tr><tr><td>+ Feature Anchoring</td><td>0.1</td><td>96.2%</td><td>0.976</td></tr><tr><td>Matryoshka SAE</td><td>0.1</td><td>99.7%</td><td>0.991</td></tr><tr><td>+ Feature Anchoring</td><td>0.1</td><td>99.7%</td><td>0.991</td></tr></table>

Effect of TopK Sparsity Parameter. Finally, we examine how the sparsity-inducing parameter k in TopK activation affects feature recovery. We train TopK SAEs with $k \in \{ 3 2 , 6 4 , 1 2 8 \}$ , corresponding to average activation rates of 0.2%, 0.4%, and 0.8% of the $n _ { q } = 1 6 , 0 0 0$ latent dimensions.

As shown in Table 7, both standard and anchored TopK SAEs achieve perfect (100%) GT Recovery at $k = 3 2$ , where extreme sparsity closely matches our theoretical assumptions. $\mathrm { A t } ~ k = 6 4$ , GT Recovery remains high (93.4%-93.9%) with minimal difference between standard and anchored training, indicating that moderate sparsity provides sufficient constraint for feature recovery without additional anchoring. $\mathrm { A t } ~ k = 1 2 8$ , where activation density increases, feature anchoring becomes beneficial again, improving GT Recovery from 86.7% to 88.0%. The Maximum Inner Product remains consistently high (> 0.985) across all settings, confirming that TopK activation is generally effective for feature recovery, with anchoring providing incremental benefits at higher k values where the optimization becomes more challenging.

Table 7: Feature recovery under varying TopK sparsity parameter k. All experiments use n = 1000 features with $n _ { p } = n _ { r } = 7 6 8 , n _ { q } = 1 6 , 0 0 0$ , and k = 100 anchors. 

<table><tr><td>Method</td><td>TopK Parameter k</td><td>GT Recovery ↑</td><td>Max Inner Product ↑</td></tr><tr><td>TopK SAE</td><td>32</td><td>100.0%</td><td>0.999</td></tr><tr><td>+ Feature Anchoring</td><td>32</td><td>100.0%</td><td>0.999</td></tr><tr><td>TopK SAE</td><td>64</td><td>93.9%</td><td>0.992</td></tr><tr><td>+ Feature Anchoring</td><td>64</td><td>93.4%</td><td>0.991</td></tr><tr><td>TopK SAE</td><td>128</td><td>86.7%</td><td>0.986</td></tr><tr><td>+ Feature Anchoring</td><td>128</td><td>88.0%</td><td>0.987</td></tr></table>

Theoretical Interpretation. These comprehensive ablation studies validate our theoretical framework across diverse conditions. The consistent failure of ReLU and JumpReLU SAEs without sufficient sparsity constraints corroborates Theorem 3.4: the underdetermined solution space admits infinitely many configurations achieving low reconstruction loss without recovering interpretable features. Feature anchoring addresses this by constraining encoder-decoder pairs, reducing degrees of freedom and steering optimization away from spurious partial minima (Theorem 3.7). The varying effectiveness across interference levels, sparsity regimes, and activation functions reflects the fundamental trade-off in our assumptions: more severe feature compression (higher M ), denser activations (lower S), or insufficient activation sparsity (higher k) create more complex optimization landscapes where even anchored methods struggle to disentangle all features completely.

# H Qualitative Examples

Beyond quantitative metrics, we provide qualitative evidence demonstrating how feature anchoring improves feature monosemanticity. Figure 10 shows features learned by Matryoshka SAE with feature anchoring on CLIP, while Figure 11 shows features from the same architecture trained without feature anchoring.

With Feature Anchoring. As shown in Figure 10, the learned features exhibit clear monosemanticity: each feature responds to a single, well-defined visual concept. The “African Grey” feature activates exclusively on African Grey parrots, the “American Black Bear” feature captures only black bears, and the “Digital Clock” feature responds specifically to digital time displays. This monosemantic behavior aligns with our theoretical prediction that feature anchoring reduces the underdetermined nature of SDL optimization (Theorem 3.4).

Without Feature Anchoring. In contrast, Figure 11 illustrates the polysemanticity that emerges without anchoring. Features respond to multiple unrelated concepts: one activates on “Various Boxes” (dishwashers, file cabinets, chests), another on “Various Screens” (slot machines, scoreboards, televisions). While achieving low reconstruction loss, these features fail to capture semantically meaningful concepts—exactly the spurious partial minima characterized in Theorem 3.6.

![](images/8f9dd427511031502986454ca9a183d5570f2ddcac3683647dd7cb13119d6c51.jpg)

<details>
<summary>natural_image</summary>

Six photos of parrots and a bird, each with distinct head and body features (no text or symbols visible)
</details>

Feature from Matryoshka SAE with Feature Anchoring: African Grey.

![](images/cd652491970cdf1e6683ed664afddfc04592181f7d585cc5c013d71c5ec751ac.jpg)

<details>
<summary>natural_image</summary>

Collage of seven black-and-white photos showing animals in natural settings (no text or symbols)
</details>

Feature from Matryoshka SAE with Feature Anchoring: American Black Bear.   
![](images/8f6d158ac43fb41bd16d069b00ded8fe1537f3ede34a04e28f868fc73c853762.jpg)

<details>
<summary>text_image</summary>

12:00 9:00 9:17 23:29 02:25 21:22 15:35
</details>

Feature from Matryoshka SAE with Feature Anchoring: Digital Clock.

Figure 10: Features learned with feature anchoring exhibit monosemanticity. Each row shows the top-activating images for a single feature from Matryoshka SAE trained on CLIP with feature anchoring.   
![](images/ee57ee628d1863040e7952c4b3ffcd5c822a21b7f71f71cdc95c0c0dd5955827.jpg)

<details>
<summary>natural_image</summary>

Panoramic street view of Times Square with multiple yellow taxis and city skyline at night (no visible text or signage)
</details>

Feature from Matryoshka SAE without Feature Anchoring: Cab.

![](images/af84f2850c759c78cb3b9751677ac7552bd40add7b070701f13407e4cb56599d.jpg)

<details>
<summary>text_image</summary>

Gia Street: New York, IL
6 100
Sight
5 727 557 0-80-
Snapchat
9 950 133 0-9+
Video Player
1 181
Super Game 10
Tobacco
1 178 0 0-67-
Drama
4 12 0 0-22-
Real Epc
1 53 2 2-02-
00:01:08
Talip Admit T2 Track 2
1 1
2 1
3 1
4 1
5 1
6 1
7 1
8 1
9 1
10 1
11 1
12 1
13 1
14 1
15 1
16 1
17 1
18 1
19 1
20 1
21 1
22 1
23 1
24 1
25 1
26 1
27 1
28 1
29 1
30 1
31 1
32 1
33 1
34 1
35 1
36 1
37 1
38 1
39 1
40 1
41 1
42 1
43 1
44 1
45 1
46 1
47 1
48 1
49 1
50 1
51 1
52 1
53 1
54 1
55 1
56 1
57 1
58 1
59 1
60 1
61 1
62 1
63 1
64 1
65 1
66 1
67 1
68 1
69 1
70 1
71 1
72 1
73 1
74 1
75 1
76 1
77 1
78 1
79 1
80 1
81 1
82 1
83 1
84 1
85 1
86 1
87 1
88 1
89 1
90 1
91 1
92 1
93 1
94 1
95 1
96 1
97 1
98 1
99 1
00:00:00
</details>

Feature from Matryoshka SAE without Feature Anchoring: Various Screens (slot, scoreboard, television).

![](images/069adb71bd9df0b0dafa4046975428eb566b84ce6ba3dc6dd7f7cd103fbdf096.jpg)

<details>
<summary>natural_image</summary>

Collage of various kitchen appliances including refrigerators, washing machines, and vending machines (no visible text or labels)
</details>

Feature from Matryoshka SAE without Feature Anchoring: Various Boxes (dishwaser, file, chest).   
Figure 11: Features learned without feature anchoring exhibit polysemanticity. Each row shows the top-activating images for a single feature from Matryoshka SAE trained without feature anchoring.

# I Proofs of the Theorems

# I.1 Proof of Theorem 3.1

Proof. Let $\begin{array} { r } { C = \operatorname* { s u p } _ { s \sim \mathcal { D } } \| \mathbf { x } _ { r } ( s ) - W _ { D } \sigma ( W _ { E } \mathbf { x } _ { p } ( s ) ) \| _ { 2 } ^ { 2 } < } \end{array}$ ∞ denote the uniform upper bound on the squared reconstruction error. The loss decomposes over the number of active features:

$$
\mathcal {L} _ {\mathrm{SDL}} = \sum_ {m = 0} ^ {n} \operatorname * {P r} (\| \mathbf {x} (s) \| _ {0} = m) \cdot \mathbb {E} \left[ \| \mathbf {x} _ {r} (s) - W _ {D} \sigma \left(W _ {E} \mathbf {x} _ {p} (s)\right) \| ^ {2} \mid \| \mathbf {x} (s) \| _ {0} = m \right]. \tag {33}
$$

The $m = 0$ term contributes zero. For $m = 1$ , exactly one feature d is active, giving $\mathbf { x } _ { r } ( s ) = x _ { d } ( s ) w _ { r } ^ { d }$ and $\begin{array} { r } { \mathbf { x } _ { p } ( s ) = x _ { d } ( s ) w _ { p } ^ { d } ; } \end{array}$ , so the total contribution equals:

$$
\sum_ {d = 1} ^ {n} M _ {d} \left\| w _ {r} ^ {d} - W _ {D} \sigma (W _ {E} w _ {p} ^ {d}) \right\| ^ {2} = \tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D}, W _ {E}), \tag {34}
$$

where $M _ { d } : = \operatorname* { P r } ( \mathbf { x } ( s ) = x _ { d } ( s ) \mathbf { e } _ { d } ) \cdot \mathbb { E } [ x _ { d } ( s ) ^ { 2 } \mid x _ { d } ( s ) > 0 ]$ . For $m \geq 2$ , since features are independently activated with probability $( 1 - S )$ , a union bound over all pairs $i \neq j$ gives:

$$
\operatorname * {P r} (\| \mathbf {x} (s) \| _ {0} \geq 2) \leq \sum_ {i, j} \operatorname * {P r} (x _ {i} > 0, x _ {j} > 0) \leq n ^ {2} (1 - S) ^ {2}. \tag {35}
$$

Since the reconstruction error is bounded by C, combining all cases yields:

$$
\left| \mathcal {L} _ {\mathrm{SDL}} - \tilde {\mathcal {L}} _ {\mathrm{SDL}} \right| \leq n ^ {2} C (1 - S) ^ {2}. \tag {36}
$$

# I.2 Proof of Theorem 3.2

Proof. Preliminary observation. By definition of $\Omega _ { \mathcal { P } }$ (Definition 3.1), for any $W _ { E } \in \Omega _ { \mathcal { P } }$ , the activation $z _ { d } ( W _ { E } ) : = \sigma ( W _ { E } w _ { p } ^ { d } )$ satisfies:

$$
z _ {d} (W _ {E}) _ {i} = \left\{ \begin{array}{l l} \langle w _ {E} ^ {i}, w _ {p} ^ {d} \rangle & \text { if } i \in \mathcal {F} _ {d} \\ 0 & \text { otherwise }, \end{array} \right. \tag {37}
$$

where $\mathcal { F } _ { d } = \{ i \in [ n _ { q } ] : ( \sigma ( W _ { E } w _ { p } ^ { d } ) ) _ { i } > 0 \}$ . Hence $z _ { d } ( W _ { E } )$ is linear in $W _ { E }$ within $\Omega _ { \mathcal { P } }$

Per-feature decomposition. Define $f _ { d } ( W _ { D } , W _ { E } ) : = \| w _ { r } ^ { d } - W _ { D } z _ { d } ( W _ { E } ) \| ^ { 2 }$ . Then:

$$
\tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D}, W _ {E}) = \sum_ {d = 1} ^ {n} M _ {d} f _ {d} (W _ {D}, W _ {E}), \quad M _ {d} > 0. \tag {38}
$$

Since positive linear combinations preserve biconvexity, it suffices to show each $f _ { d }$ is biconvex.

Convexity in $W _ { D }$ . For fixed $W _ { E } \in \Omega _ { \mathcal { P } } , z _ { d }$ is a fixed vector, so $f _ { d } ( W _ { D } ) = \| w _ { r } ^ { d } - W _ { D } z _ { d } \| ^ { 2 }$ is a squared affine function of $W _ { D }$ , hence convex. Its Hessian $\nabla _ { W _ { D } } ^ { 2 } f _ { d } = 2 I _ { n _ { r } } \otimes ( z _ { d } z _ { d } ^ { \top } ) \succeq 0$ confirms this.

Convexity in $W _ { E }$ . For fixed $W _ { D }$ , linearity of $z _ { d } ( W _ { E } )$ in $W _ { E }$ gives:

$$
W _ {D} z _ {d} (W _ {E}) = \sum_ {i \in \mathcal {F} _ {d}} \langle w _ {E} ^ {i}, w _ {p} ^ {d} \rangle w _ {D} ^ {i}, \tag {39}
$$

whichconve affine in For each $W _ { E }$ ence with $f _ { d } ( W _ { E } ) = \| w _ { r } ^ { d } - W _ { D } z _ { d } ( W _ { E } ) \| ^ { 2 }$ unction of , and rows $W _ { E }$ hich isdo not $w _ { E } ^ { i }$ $i \in \mathcal { F } _ { d }$ $\nabla _ { w _ { E } ^ { i } } ^ { 2 } \bar { f } _ { d } = 2 \| w _ { D } ^ { i } \| ^ { 2 } ( w _ { p } ^ { d } ( w _ { p } ^ { d } ) ^ { \top } ) \succeq 0$ $i \not \in \mathcal { F } _ { d }$ affect $f _ { d } .$ .

Conclusion. Each $f _ { d }$ is biconvex, so $\tilde { \mathcal { L } } _ { \mathrm { S D L } }$ is biconvex over $\mathbb { R } ^ { n _ { r } \times n _ { q } } \times \Omega _ { \mathcal { P } }$

# I.3 Proof of Theorem 3.3

Proof. We analyze $\tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } ^ { * } , W _ { E } ^ { * } )$ by examining the reconstruction error for each feature $d \in [ n ]$ active in isolation.

The encoder output is $W _ { E } ^ { * } w _ { p } ^ { d } = W _ { p } ^ { \top } w _ { p } ^ { d }$ with k-th component $( W _ { E } ^ { * } w _ { p } ^ { d } ) _ { k } \ = \ \langle w _ { p } ^ { k } , w _ { p } ^ { d } \rangle$ . By the unit-norm condition (Assumption 2.1), $( W _ { E } ^ { * } w _ { p } ^ { d } ) _ { d } = 1$ . Define the set of neurons activated by feature d:

$$
\mathcal {A} _ {d} = \{k \in [ n ]: \sigma (W _ {E} ^ {*} w _ {p} ^ {d}) _ {k} > 0 \}. \tag {40}
$$

Since $\sigma ( z ) _ { i } \in \{ 0 , z _ { i } \}$ , we have $\sigma ( W _ { E } ^ { * } w _ { p } ^ { d } ) _ { k } > 0$ if and only if $\langle w _ { p } ^ { k } , w _ { p } ^ { d } \rangle > 0$ . Therefore $d \in { \mathcal { A } } _ { d }$ and for $k \neq d \colon$

$$
\mathcal {A} _ {d} \setminus \{d \} \subseteq \{k \in [ n ]: 0 <   \left\langle w _ {p} ^ {k}, w _ {p} ^ {d} \right\rangle \leq M \}. \tag {41}
$$

The reconstruction is $\begin{array} { r } { W _ { D } ^ { * } \sigma ( W _ { E } ^ { * } w _ { p } ^ { d } ) = \sum _ { k \in \mathcal { A } _ { d } } \langle w _ { p } ^ { k } , w _ { p } ^ { d } \rangle w _ { r } ^ { k } } \end{array}$ , so the reconstruction error for feature d is:

$$
w _ {r} ^ {d} - W _ {D} ^ {*} \sigma (W _ {E} ^ {*} w _ {p} ^ {d}) = w _ {r} ^ {d} - \sum_ {k \in \mathcal {A} _ {d}} \langle w _ {p} ^ {k}, w _ {p} ^ {d} \rangle w _ {r} ^ {k} = - \sum_ {k \in \mathcal {A} _ {d} \setminus \{d \}} \langle w _ {p} ^ {k}, w _ {p} ^ {d} \rangle w _ {r} ^ {k}. \tag {42}
$$

Applying the triangle inequality and using $\| w _ { r } ^ { k } \| _ { 2 } = 1$ and $\langle w _ { p } ^ { k } , w _ { p } ^ { d } \rangle \leq M$ for $k \in \mathcal { A } _ { d } \setminus \{ d \}$ :

$$
\left\| w _ {r} ^ {d} - W _ {D} ^ {*} \sigma (W _ {E} ^ {*} w _ {p} ^ {d}) \right\| _ {2} \leq \sum_ {k \in \mathcal {A} _ {d} \setminus \{d \}} \langle w _ {p} ^ {k}, w _ {p} ^ {d} \rangle \leq M (K _ {d} - 1), \tag {43}
$$

where $K _ { d } = | { \mathcal { A } } _ { d } | \leq n$ . Squaring and summing over all features:

$$
\tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D} ^ {*}, W _ {E} ^ {*}) \leq M ^ {2} \sum_ {d = 1} ^ {n} M _ {d} (K _ {d} - 1) ^ {2} \leq n ^ {2} M ^ {2} \sum_ {d = 1} ^ {n} M _ {d}, \tag {44}
$$

where the second inequality uses $( K _ { d } - 1 ) ^ { 2 } \leq ( n - 1 ) ^ { 2 } \leq n ^ { 2 }$ .

# I.4 Proof of Theorem 3.4

Proof. We prove both directions of the equivalence.

Sufficient condition (⇒). Suppose $w _ { r } ^ { d } = W _ { D } \sigma ( W _ { E } w _ { p } ^ { d } )$ for all $d \in [ n ]$ . By definition of the approximate loss:

$$
\tilde {\mathcal {L}} _ {\mathrm{SDL}} \left(W _ {D}, W _ {E}\right) = \sum_ {d = 1} ^ {n} M _ {d} \left\| w _ {r} ^ {d} - W _ {D} \sigma \left(W _ {E} w _ {p} ^ {d}\right) \right\| ^ {2} \tag {45}
$$

$$
= \sum_ {d = 1} ^ {n} M _ {d} \| w _ {r} ^ {d} - w _ {r} ^ {d} \| ^ {2} \tag {46}
$$

$$
= 0 \tag {47}
$$

Necessary condition (⇐). Suppose $\tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } , W _ { E } ) = 0$ . Then:

$$
\sum_ {d = 1} ^ {n} M _ {d} \| w _ {r} ^ {d} - W _ {D} \sigma (W _ {E} w _ {p} ^ {d}) \| ^ {2} = 0 \tag {48}
$$

Since ${ \cal M } _ { d } > 0$ for all $d \in [ n ]$ (by definition, $M _ { d } = \operatorname* { P r } ( \mathbf { x } ( s ) = x _ { d } ( s ) e _ { d } ) \cdot \mathbb { E } [ x _ { d } ( s ) ^ { 2 } | x _ { d } ( s ) > 0 ]$ where both factors are positive under Assumptions 2.1):

$$
\left\| w _ {r} ^ {d} - W _ {D} \sigma \left(W _ {E} w _ {p} ^ {d}\right) \right\| ^ {2} = 0 \quad \forall d \in [ n ] \tag {49}
$$

Therefore:

$$
w _ {r} ^ {d} = W _ {D} \sigma (W _ {E} w _ {p} ^ {d}) \quad \forall d \in [ n ] \tag {50}
$$

![](images/19a559cd81290afe7146a1787acf84b768b5c131854b6295dbea6de9f338c48b.jpg)

# I.5 Proof of Theorem 3.6

Proof. We construct a configuration $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ exhibiting the polysemanticity pattern and explicitly verify that both gradients vanish.

Step 1: Given encoder $W _ { E } ^ { * }$ from realizability.

Since the activation pattern $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { n _ { q } } )$ is realizable and forms a partition of [n], by Definition 3.1, there exists an encoder $W _ { E } ^ { * } \in \mathbb { R } ^ { n _ { q } \times n _ { p } }$ such that:

$$
\forall i \in [ n _ {q} ], \quad \mathcal {F} _ {i} = \left\{d \in [ n ]: \left(\sigma (W _ {E} ^ {*} w _ {p} ^ {d})\right) _ {i} > 0 \right\} \tag {51}
$$

Fix this $W _ { E } ^ { * }$ and define the activation vectors:

$$
z _ {d} := \sigma (W _ {E} ^ {*} w _ {p} ^ {d}) \in \mathbb {R} ^ {n _ {q}}, \quad d = 1, \dots , n \tag {52}
$$

Since P forms a partition, for each feature d there exists a unique neuron $i ( d )$ such that $d \in { \mathcal { F } } _ { i ( d ) }$ , and:

$$
z _ {d} = \left\langle w _ {E} ^ {*, i (d)}, w _ {p} ^ {d} \right\rangle \cdot e _ {i (d)} \tag {53}
$$

That is, only neuron i(d) activates for feature $d .$

# Step 2: Construct optimal decoder $W _ { D } ^ { * }$ .

For each neuron $i \in [ n _ { q } ]$ , we construct the decoder column $W _ { D } ^ { * , i }$ by minimizing the loss over features activating that neuron.

Dead neurons: If ${ \mathcal { F } } _ { i } = \emptyset .$ , set $W _ { D } ^ { * , i } = 0$

Active neurons: If $\mathcal { F } _ { i } \neq \emptyset$ , solve:

$$
W _ {D} ^ {*, i} = \arg \min _ {W _ {D} ^ {i} \in \mathbb {R} ^ {n _ {r}}} \sum_ {d \in \mathcal {F} _ {i}} M _ {d} \left\| w _ {r} ^ {d} - W _ {D} ^ {i} (z _ {d}) _ {i} \right\| ^ {2} \tag {54}
$$

This is a least squares problem with solution:

$$
W _ {D} ^ {*, i} = \frac {\sum_ {d \in \mathcal {F} _ {i}} M _ {d} (z _ {d}) _ {i} w _ {r} ^ {d}}{\sum_ {d \in \mathcal {F} _ {i}} M _ {d} (z _ {d}) _ {i} ^ {2}} \tag {55}
$$

Step 3: Verify $\nabla _ { W _ { D } } \tilde { \mathcal { L } } _ { \mathbf { S D L } } ( W _ { D } ^ { * } , W _ { E } ^ { * } ) = 0 ,$ .

The loss is $\begin{array} { r } { \tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } , W _ { E } ) = \sum _ { d = 1 } ^ { n } M _ { d } \| w _ { r } ^ { d } - W _ { D } z _ { d } \| ^ { 2 } } \end{array}$ . The gradient with respect to column $W _ { D } ^ { i }$ is:

$$
\nabla_ {W _ {D} ^ {i}} \tilde {\mathcal {L}} _ {\mathrm{SDL}} = - 2 \sum_ {d = 1} ^ {n} M _ {d} (w _ {r} ^ {d} - W _ {D} z _ {d}) (z _ {d}) _ {i} \tag {56}
$$

Since P is a partition, $( z _ { d } ) _ { i } \neq 0$ only when $i = i ( d )$ , i.e., when $d \in { \mathcal { F } } _ { i }$ . Therefore:

$$
\nabla_ {W _ {D} ^ {i}} \tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D} ^ {*}, W _ {E} ^ {*}) = - 2 \sum_ {d \in \mathcal {F} _ {i}} M _ {d} (w _ {r} ^ {d} - W _ {D} ^ {*, i} (z _ {d}) _ {i}) (z _ {d}) _ {i} \tag {57}
$$

By construction (55), $W _ { D } ^ { * , i }$ satisfies the normal equation:

$$
\sum_ {d \in \mathcal {F} _ {i}} M _ {d} (z _ {d}) _ {i} (w _ {r} ^ {d} - W _ {D} ^ {*, i} (z _ {d}) _ {i}) = 0 \tag {58}
$$

Therefore $\nabla _ { W _ { D } ^ { i } } \tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } ^ { * } , W _ { E } ^ { * } ) = 0$ for all $i \in [ n _ { q } ]$

Step 4: Verify $\nabla _ { W _ { E } } \tilde { \mathcal { L } } _ { \mathbf { S D L } } ( W _ { D } ^ { * } , W _ { E } ^ { * } ) = 0 .$

Within the activation pattern region $\Omega _ { \mathcal { A } }$ , for feature $d \in { \mathcal { F } } _ { i }$ , the activation is:

$$
z _ {d} = \left\langle w _ {E} ^ {i}, w _ {p} ^ {d} \right\rangle \cdot e _ {i} \tag {59}
$$

The gradient with respect to encoder row $w _ { E } ^ { i }$ is:

$$
\nabla_ {w _ {E} ^ {i}} \tilde {\mathcal {L}} _ {\mathrm{SDL}} = \sum_ {d = 1} ^ {n} M _ {d} \nabla_ {w _ {E} ^ {i}} \left\| w _ {r} ^ {d} - W _ {D} z _ {d} \right\| ^ {2} \tag {60}
$$

Only features $d \in { \mathcal { F } } _ { i }$ have non-zero contribution. For such d:

$$
\frac {\partial z _ {d}}{\partial w _ {E} ^ {i}} = w _ {p} ^ {d} \otimes e _ {i} \tag {61}
$$

where ⊗ denotes outer product. Therefore:

$$
\nabla_ {w _ {E} ^ {i}} \left\| w _ {r} ^ {d} - W _ {D} z _ {d} \right\| ^ {2} = - 2 (w _ {r} ^ {d} - W _ {D} z _ {d}) ^ {\top} W _ {D} \frac {\partial z _ {d}}{\partial w _ {E} ^ {i}} \tag {62}
$$

$$
= - 2 (w _ {r} ^ {d} - W _ {D} ^ {*, i} \langle w _ {E} ^ {*, i}, w _ {p} ^ {d} \rangle) ^ {\top} W _ {D} ^ {*, i} w _ {p} ^ {d} \tag {63}
$$

The total gradient is:

$$
\nabla_ {w _ {E} ^ {i}} \tilde {\mathcal {L}} _ {\mathrm{SDL}} = - 2 \sum_ {d \in \mathcal {F} _ {i}} M _ {d} (w _ {r} ^ {d} - W _ {D} ^ {*, i} \langle w _ {E} ^ {*, i}, w _ {p} ^ {d} \rangle) ^ {\top} W _ {D} ^ {*, i} w _ {p} ^ {d} \tag {64}
$$

$$
= - 2 (W _ {D} ^ {*, i}) ^ {\top} \sum_ {d \in \mathcal {F} _ {i}} M _ {d} \langle w _ {E} ^ {*, i}, w _ {p} ^ {d} \rangle (w _ {r} ^ {d} - W _ {D} ^ {*, i} \langle w _ {E} ^ {*, i}, w _ {p} ^ {d} \rangle) \tag {65}
$$

From the normal equation in Step 3 with $( \boldsymbol { z } _ { d } ) _ { i } = \langle \boldsymbol { w } _ { E } ^ { * , i } , \boldsymbol { w } _ { p } ^ { d } \rangle$

$$
\sum_ {d \in \mathcal {F} _ {i}} M _ {d} \left\langle w _ {E} ^ {*}, i, w _ {p} ^ {d} \right\rangle \left(w _ {r} ^ {d} - W _ {D} ^ {*}, i \left\langle w _ {E} ^ {*}, i, w _ {p} ^ {d} \right\rangle\right) = 0 \tag {66}
$$

Therefore:

$$
\nabla_ {w _ {E} ^ {i}} \tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D} ^ {*}, W _ {E} ^ {*}) = - 2 (W _ {D} ^ {*, i}) ^ {\top} \cdot 0 = 0 \tag {67}
$$

This holds for all active neurons. For dead neurons $( \mathcal { F } _ { i } = \varnothing )$ , the gradient is automatically zero.

Step 5: $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ is a partial optimum.

By Theorem $3 . 2 , \tilde { \cal { L } } _ { \mathrm { S D I } }$ L is convex in $W _ { D }$ for fixed $W _ { E } ^ { * }$ , and convex in $W _ { E }$ for fixed $W _ { D } ^ { * }$ within $\Omega _ { \mathcal { A } }$ Since we have verified:

$\bullet \nabla _ { W _ { D } } \tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } ^ { * } , W _ { E } ^ { * } ) = 0 ( \mathrm { S t e p \ 3 } )$   
$\bullet \nabla _ { W _ { E } } \tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } ^ { \ast } , W _ { E } ^ { \ast } ) = 0 \mathrm { w i t h i n } \Omega _ { A } \mathrm { ( S t e p ~ 4 ) }$

By the first-order optimality condition for convex functions:

• $W _ { D } ^ { * }$ minimizes $\tilde { \mathcal { L } } _ { \mathrm { { S D L } } } ( \cdot , W _ { E } ^ { * } )$

• $W _ { E } ^ { * }$ minimizes $\tilde { \mathcal { L } } _ { \mathrm { S D L } } ( W _ { D } ^ { * } , \cdot )$ over $\Omega _ { \mathcal { A } }$

By standard results in biconvex optimization (Gorski et al., 2007), $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ is a partial optimum.

# Step 6: Non-zero loss.

Since the pattern is polysemantic, there exists $i \in [ n _ { q } ]$ with $| \mathcal { F } _ { i } | \geq 2$ . Choose distinct $d _ { 1 } , d _ { 2 } \in \mathcal { F } _ { i }$ .

By Theorem 3.4, zero loss requires:

$$
w _ {r} ^ {d _ {1}} = W _ {D} ^ {*} z _ {d _ {1}} = W _ {D} ^ {*, i} (z _ {d _ {1}}) _ {i}, \quad w _ {r} ^ {d _ {2}} = W _ {D} ^ {*} z _ {d _ {2}} = W _ {D} ^ {*, i} (z _ {d _ {2}}) _ {i} \tag {68}
$$

If both equations held, then:

$$
\frac {w _ {r} ^ {d _ {1}}}{(z _ {d _ {1}}) _ {i}} = W _ {D} ^ {*, i} = \frac {w _ {r} ^ {d _ {2}}}{(z _ {d _ {2}}) _ {i}} \tag {69}
$$

Since $( z _ { d _ { 1 } } ) _ { i } = \langle w _ { E } ^ { * , i } , w _ { p } ^ { d _ { 1 } } \rangle > 0$ and $( z _ { d _ { 2 } } ) _ { i } = \langle w _ { E } ^ { * , i } , w _ { p } ^ { d _ { 2 } } \rangle > 0$ , this requires:

$$
w _ {r} ^ {d _ {1}} = \frac {\left(z _ {d _ {1}}\right) _ {i}}{\left(z _ {d _ {2}}\right) _ {i}} w _ {r} ^ {d _ {2}} \tag {70}
$$

This means $w _ { r } ^ { d _ { 1 } } \propto w _ { r } ^ { d _ { 2 } }$ , contradicting Assumption 2.1 (distinct features have linearly independent reconstruction vectors). Therefore:

$$
\tilde {\mathcal {L}} _ {\mathrm{SDL}} (W _ {D} ^ {*}, W _ {E} ^ {*}) = \sum_ {d = 1} ^ {n} M _ {d} \| w _ {r} ^ {d} - W _ {D} ^ {*} z _ {d} \| ^ {2} > 0 \tag {71}
$$

Conclusion. We have constructed $( W _ { D } ^ { * } , W _ { E } ^ { * } )$ exhibiting the polysemanticity pattern, explicitly verified both gradients vanish, and shown it achieves positive loss, completing the proof. □

# I.6 Proof of Theorem 3.8

Proof. We construct a realizable pattern exhibiting feature absorption by scaling down the parent neuron’s encoder until one sub-concept separates, then adding a dedicated neuron for the separated feature.

# Step 1: Initial configuration and feature geometry.

Since $\mathcal { P } = ( \mathcal { F } _ { 1 } , \ldots , \mathcal { F } _ { M } )$ is realizable, there exists an encoder $W _ { E } \in \mathbb { R } ^ { M \times n _ { p } }$ and threshold $c > 0$ such that:

$$
\forall i \in [ M ], \quad \mathcal {F} _ {i} = \{d \in [ n ]: \langle w _ {E} ^ {i}, w _ {p} ^ {d} \rangle > c \} \tag {72}
$$

Fix any parent neuron $i ^ { * } \in [ M ]$ with sub-concepts $\mathcal { F } _ { i ^ { * } } = \{ d _ { i ^ { * } , 1 } , \dotsc , d _ { i ^ { * } , k _ { i ^ { * } } } \}$ where $k _ { i ^ { * } } \geq 2 .$

For each sub-concept, compute its activation strength:

$$
a _ {j} := \left\langle w _ {E} ^ {i ^ {*}}, w _ {p} ^ {d _ {i ^ {*}, j}} \right\rangle > c, \quad \forall j \in \left[ k _ {i ^ {*}} \right] \tag {73}
$$

Since there are finitely many sub-concepts, define:

$$
a _ {\min} := \min _ {j \in \left[ k _ {i ^ {*}} \right]} a _ {j} > c \tag {74}
$$

Let $j ^ { * } \in [ k _ { i ^ { * } } ]$ be the index achieving this minimum: $a _ { j ^ { * } } = a _ { \operatorname* { m i n } }$ .

# Step 2: Scale down parent neuron to separate one sub-concept.

We shrink the encoder row $w _ { E } ^ { i ^ { * } }$ by multiplying with scaling factor $\lambda \in ( 0 , 1 )$ :

$$
\tilde {w} _ {E} ^ {i ^ {*}} := \lambda \cdot w _ {E} ^ {i ^ {*}} \tag {75}
$$

The new activation strengths become:

$$
\tilde {a} _ {j} := \left\langle \tilde {w} _ {E} ^ {i ^ {*}}, w _ {p} ^ {d _ {i ^ {*}, j}} \right\rangle = \lambda \cdot a _ {j} \tag {76}
$$

Choose λ such that:

$$
\lambda := \frac {c + a _ {\min}}{2 a _ {\min}} \in \left(\frac {c}{a _ {\min}}, 1\right) \tag {77}
$$

This gives:

• For the minimum-activation feature: $\begin{array} { r } { \tilde { a } _ { j ^ { * } } = \lambda a _ { \mathrm { m i n } } = \frac { c + a _ { \mathrm { m i n } } } { 2 } < c + \epsilon } \end{array}$ for any small $\epsilon > 0$   
• For all other features: $\begin{array} { r } { \tilde { a } _ { j } = \lambda a _ { j } > \lambda a _ { \mathrm { m i n } } = \frac { c + a _ { \mathrm { m i n } } } { 2 } } \end{array}$

By choosing the threshold to be $\begin{array} { r } { c ^ { \prime } = \frac { c + a _ { \mathrm { m i n } } } { 2 } } \end{array}$ , we have:

$$
\tilde {a} _ {j ^ {*}} = c ^ {\prime} \quad (\text {at boundary}), \quad \tilde {a} _ {j} > c ^ {\prime} \text {for all} j \neq j ^ {*} \tag {78}
$$

Slightly increasing c′ by infinitesimal $\epsilon > 0$ gives $c ^ { \prime \prime } = c ^ { \prime } + \epsilon$ , ensuring:

$$
\tilde {a} _ {j ^ {*}} <   c ^ {\prime \prime} <   \tilde {a} _ {j} \quad \forall j \neq j ^ {*} \tag {79}
$$

Therefore, with encoder row $\tilde { w } _ { E } ^ { i ^ { * } }$ and threshold $c ^ { \prime \prime } { \mathrm { : } }$

$$
\{d \in [ n ]: \langle \tilde {w} _ {E} ^ {i ^ {*}}, w _ {p} ^ {d} \rangle > c ^ {\prime \prime} \} = \mathcal {F} _ {i ^ {*}} \setminus \left\{d _ {i ^ {*}, j ^ {*}} \right\} \tag {80}
$$

# Step 3: Construct dedicated neuron for separated feature.

We create a new encoder row parallel to the separated feature:

$$
w _ {E} ^ {\text { new }} := \alpha \cdot w _ {p} ^ {d _ {i ^ {*}, j ^ {*}}} \tag {81}
$$

where $\alpha > 0$ is chosen sufficiently large.

By the unit-norm assumption $( \| w _ { p } ^ { d _ { i ^ { * } , j ^ { * } } } \| = 1 )$ wdi∗,j∗p ∥ = 1):

$$
\left\langle w _ {E} ^ {\text { new }}, w _ {p} ^ {d _ {i ^ {*}, j ^ {*}}} \right\rangle = \alpha \| w _ {p} ^ {d _ {i ^ {*}, j ^ {*}}} \| ^ {2} = \alpha \tag {82}
$$

For any other feature d ̸= di∗,j∗ :

$$
\left\langle w _ {E} ^ {\text { new }}, w _ {p} ^ {d} \right\rangle = \alpha \left\langle w _ {p} ^ {d _ {i ^ {*}}, j ^ {*}}, w _ {p} ^ {d} \right\rangle \leq \alpha M \tag {83}
$$

where M < 1 is the maximum interference.

Choose α large enough such that:

$$
\alpha > c ^ {\prime \prime}, \quad \alpha M <   c ^ {\prime \prime} \tag {84}
$$

This is possible by taking $\begin{array} { r } { \alpha = \frac { 2 c ^ { \prime \prime } } { 1 + M } } \end{array}$ , giving:

$$
\alpha = \frac {2 c ^ {\prime \prime}}{1 + M} > c ^ {\prime \prime}, \quad \alpha M = \frac {2 c ^ {\prime \prime} M}{1 + M} <   c ^ {\prime \prime} \tag {85}
$$

where the first inequality uses M < 1 and the second uses $M < 1$ .

Therefore:

• $\langle w _ { E } ^ { n e w } , w _ { p } ^ { d _ { i ^ { * } , j ^ { * } } } \rangle = \alpha > c ^ { \prime \prime }$ wdi∗,j∗p ⟩ = α > c′′ (separated feature activates new neuron)

• $\langle w _ { E } ^ { n e w } , w _ { p } ^ { d } \rangle \leq \alpha M < c ^ { \prime \prime }$ for all $d \neq d _ { i ^ { * } , j ^ { * } }$ (no other features activate)

# Step 4: The new activation pattern.

Construct the expanded encoder $W _ { E } ^ { \prime } \in \mathbb { R } ^ { ( M + 1 ) \times n _ { p } }$ :

$$
W _ {E} ^ {\prime} = \left[ \begin{array}{c} w _ {E} ^ {1} \\ \vdots \\ w _ {E} ^ {i ^ {*} - 1} \\ \tilde {w} _ {E} ^ {i ^ {*}} \\ w _ {E} ^ {i ^ {*} + 1} \\ \vdots \\ w _ {E} ^ {M} \\ w _ {E} ^ {\text {new}} \end{array} \right] \tag {86}
$$

With threshold $c ^ { \prime \prime } .$ , the resulting activation pattern is:

$$
\mathcal {P} ^ {\prime} = (\mathcal {F} _ {1}, \ldots , \mathcal {F} _ {i ^ {*} - 1}, \mathcal {F} _ {i ^ {*}} \setminus \{d _ {i ^ {*}, j ^ {*}} \}, \mathcal {F} _ {i ^ {*} + 1}, \ldots , \mathcal {F} _ {M}, \{d _ {i ^ {*}, j ^ {*}} \}) \tag {87}
$$

This pattern is realizable by construction.

![](images/4b56a235a649a2384e11e41474f9d224e22b069cc6f99cd55ad5af6ae78da759.jpg)