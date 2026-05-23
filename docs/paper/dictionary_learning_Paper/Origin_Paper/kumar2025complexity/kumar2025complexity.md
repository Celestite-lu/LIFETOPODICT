# Dictionary Learning: The Complexity of Sparse Superposed Features with Feedback

Akash Kumar 1

# Abstract

The success of deep networks is crucially attributed to their ability to capture latent features within a representation space. In this work, we investigate whether the underlying learned features of a model can be efficiently retrieved through feedback from an agent, such as a large language model (LLM), in the form of relative triplet comparisons. These features may represent various constructs, including dictionaries in LLMs or a covariance matrix of Mahalanobis distances. We analyze the feedback complexity associated with learning a feature matrix in sparse settings. Our results establish tight bounds when the agent is permitted to construct activations and demonstrate strong upper bounds in sparse scenarios when the agent’s feedback is limited to distributional information. We validate our theoretical findings through experiments1 on two distinct applications: feature recovery from Recursive Feature Machines and dictionary extraction from sparse autoencoders trained on Large Language Models.

# 1. Introduction

In recent years, neural network-based models have achieved state-of-the-art performance across a wide array of tasks. These models effectively capture relevant features or concepts from samples, tailored to the specific prediction tasks they address (Yang and Hu, 2021b; Bordelon and Pehlevan, 2022a; Ba et al., 2022b). A fundamental challenge lies in understanding how these models learn such features and determining whether these features can be interpreted or even retrieved directly (Radhakrishnan et al., 2024). Recent 1Department of Computer Science & Engineering, University of California, San Diego, USA. Correspondence to: Akash Kumar <akk002@ucsd.edu>.

Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

1(https://github.com/akashkumar-d/ learnsparsefeatureswithfeedback.git)

advancements in mechanistic interpretability have opened multiple avenues for elucidating how transformer-based models, including Large Language Models (LLMs), acquire and represent features (Bricken et al., 2023; Doshi-Velez and Kim, 2017). These advances include uncovering neural circuits that encode specific concepts (Marks et al., 2024b; Olah et al., 2020), understanding feature composition across attention layers (Yang and Hu, 2021b), and revealing how models develop structured representations (Elhage et al., 2022). One line of research posits that features are encoded linearly within the latent representation space through sparse activations, a concept known as the linear representation hypothesis (LRH) (Mikolov et al., 2013; Arora et al., 2016). However, this hypothesis faces challenges in explaining how neural networks function, as models often need to represent more distinct features than their layer dimensions would theoretically allow under purely linear encoding. This phenomenon has been studied extensively in the context of large language models through the lens of superposition (Elhage et al., 2022), where multiple features share the same dimensional space in structured ways.

Recent efforts have addressed this challenge through sparse coding or dictionary learning, proposing that any layer ℓ of the model learns features linearly:

$$
\boldsymbol {x} \approx \mathsf {D} _ {\ell} \cdot \alpha_ {\ell} (\boldsymbol {x}) + \epsilon_ {\ell} (\boldsymbol {x}),
$$

where $\pmb { x } \in \mathbb { R } ^ { d } , \mathsf { D } _ { \ell } \in \mathbb { R } ^ { d \times p }$ is a dictionary2 matrix, $\alpha _ { \ell } ( \pmb { x } ) \in$ $\mathbb { R } ^ { p }$ is a sparse representation vector, and $\boldsymbol { \epsilon } _ { \ell } ( \boldsymbol { \mathscr { x } } ) \in \mathbb { R } ^ { p }$ represents error terms. This approach enables retrieval of interpretable features through sparse autoencoders (Bricken et al., 2023; Marks et al., 2024b), allowing for targeted monitoring and modification of network behavior. The linear feature decomposition not only advances model interpretation but also suggests the potential for developing compact, interpretable models that maintain performance by leveraging universal features from larger architectures.

In this work, we explore how complex features encoded as a dictionary can be distilled through feedback from either advanced language models (e.g., ChatGPT, Claude 3.0 Sonnet) or human agents. Let’s define a dictionary $\mathsf { D } \in \mathbb { R } ^ { d \times p }$ where each column represents an atomic feature vector. These atomic features, denoted as $u _ { 1 } , u _ { 2 } , \dotsc , u _ { p } \subset$ $\mathbb { R } ^ { d } .$ , could correspond to semantic concepts like "tree", "house", or "lawn" that are relevant to the task’s sample space. The core mechanism involves an agent (either AI or human) determining whether different sparse combinations of these atomic features are similar or dissimilar. Specifically, given sparse activation vectors $\alpha , \alpha ^ { \prime } \in \mathbb { R } ^ { p }$ , the agent evaluates whether linear combinations such as $\alpha _ { 1 } v ( " \mathrm { t r e e " } ) + \alpha _ { 2 } v ( " \mathrm { c a r " } ) + . . . + \alpha _ { d } v ( " \mathrm { h o u s e " } )$ are equivalent to other combinations using different activation vectors. Precisely, we formalize these feedback relationships using relative triplet comparisons $( \alpha , \beta , \zeta ) \in \mathcal { V }$ , where $\nu \subseteq \mathbb { R } ^ { p }$ is the activation or representation space. These comparisons express that a linear combination of features using coefficients α is more similar to a combination using coefficients $\beta$ than to one using coefficients ζ.

The objective is to determine the extent to which an oblivious learner—one who learns solely by satisfying the constraints of the feedback and randomly selecting valid features—can identify the feature vectors of D up to normal transformation. The fundamental protocol is as follows:

• The agent either constructs or selects (from a sampled pool) sparse triplets of activations $( \alpha , \beta , \zeta ) \in \mathbb { R } ^ { 3 p }$ and designs relative feedback of similarity $\ell \in \{ + 1 , 0 , - 1 \}$ satisfying sgn $( \| \mathsf { D } ( \alpha - \beta ) \| - \| \mathsf { D } ( \alpha - \zeta ) \| ) = \ell ,$ and provides them to the learner.   
• The learner solves for

$$
\left\{\operatorname{sgn} \left(\| \hat {\mathsf {D}} (\alpha - \beta) \| - \| \hat {\mathsf {D}} (\alpha - \zeta) \|\right) = \ell \right\}
$$

and outputs a solution $\hat { \mathsf { D } } ^ { \top } \hat { \mathsf { D } } .$

Semantically, these relative distances provide the relative information on how ground truth samples, e.g. images, text among others, relate to each other. We term the normal transformation ${ \mathsf { D } } { \mathsf { D } } ^ { \top }$ for a given dictionary D as feature matrices $\Phi \in \mathbb { R } ^ { p \times p }$ , which is exactly a covariance matrix. Alternatively, for the representation space $\nu \subseteq \mathbb { R } ^ { p }$ , this transformation defines a Mahalanobis distance function d : $\mathcal { V } \times \mathcal { V }  \mathbb { R }$ , characterized by the square symmetric linear transformation $\Phi \succeq 0$ such that for any pair of activations $( x , y ) \in \mathcal { V } ^ { 2 }$ , their distance is given by:

$$
d (x, y) := (x - y) ^ {\top} \boldsymbol {\Phi} (x - y)
$$

When Φ embeds samples into Rr, it admits a decomposition $\Phi = \mathsf { L } ^ { \top } \mathsf { L } \operatorname { f o r } \mathsf { L } \in \mathbb { R } ^ { \bar { r } \times p }$ , where L serves as a dictionary for this distance function—a formulation well-studied in metric learning literature (Kulis, 2013). In this work, we study the minimal number of interactions, termed as feedback complexity of learning feature matrices—normal transformations to a dictionary—of the form $\Phi ^ { * } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ . We consider two types of feedback: general activations and sparse activations, examining both constructive and distributional settings. Our primary contributions are:

I. We investigate feedback complexity in the constructive setting, where agents select activations from Rp, establishing strong bounds for both general and sparse scenarios. (see Section 4)   
II. We analyze the distributional setting with sampled activations, developing results for both general and sparse representations. For sparse sampling, we extend the definition of a Lebesgue measure to accommodate sparsity constraints. (see Section 5)   
III. We validate our theoretical bounds through experiments with feature matrices from Recursive Feature Machines and dictionaries trained for sparse autoencoders in Large Language Models, including Pythia-70M (Biderman et al., 2023) and Board Game models (Karvonen et al., 2024). (see Section 6)

Table 1 summarizes our feedback complexity bounds for different feedback types.

# 2. Related Work

Dictionary learning Recent work has explored dictionary learning to disentangle the semanticity (mono- or polysemy) of neural network activations (Faruqui et al., 2015; Arora et al., 2018; Zhang et al., 2019; Yun et al., 2021). Dictionary learning (Mallat and Zhang, 1993; Olshausen and Field, 1997) (aka sparse coding) provides a systematic approach to decompose task-specific samples into sparse signals. The sample complexity of dictionary learning (or sparse coding) has been extensively studied as an optimization problem, typically involving non-convex objectives such as $\ell _ { 1 }$ regularization (see (Gribonval et al., 2015)). While traditional methods work directly with ground-truth samples, our approach differs fundamentally as the learner only receives feedback on sparse signals or activations. Prior work in noiseless settings has established probabilistic exact recovery up to linear transformations (permutations and sign changes) under mutual incoherence conditions (Gribonval and Schnass, 2010; Agarwal et al., 2014). Our work extends these results by proving exact recovery (both deterministic and probabilistic) up to normal transformation, which generalizes to rotational and sign changes under strong incoherence properties (see Lemma 1). In the sampling regime, we analyze k-sparse signals, building upon the noisy setting framework developed in Arora et al. (2013); Gribonval et al. (2015).

Feature learning in neural networks and Linear representation hypothesis Neural networks demonstrate a remarkable ability to discover and exploit task-specific features from data (Yang and Hu, 2021b; Bordelon and Pehlevan, 2022b; Shi et al., 2022). Recent theoretical advances have significantly enhanced our understanding of feature evolution and emergence during training (Abbe et al., 2022; Ba et al., 2022a; Damian et al., 2022; Yang and Hu, 2021a; Zhu et al., 2022). Particularly noteworthy is the finding that the outer product of model weights correlates with the gradient outer product of the classifier averaged over layer preactivations (Radhakrishnan et al., 2024), which directly relates to the covariance matrices central to our investigation. Building upon these insights, Elhage et al. (2022) proposed that features in large language models follow a linear encoding principle, suggesting that the complex feature representations learned during training can be decomposed into interpretable linear components. This interpretability, in turn, could facilitate the development of simplified algorithms for complex tasks (Fawzi et al., 2022; Romera-Paredes et al., 2024). Recent research has focused on extracting these interpretable features in the form of dictionary learning by training sparse autoencoder for various language models including Board Games Models (Marks et al., 2024b; Bricken et al., 2023). Our work extends this line of inquiry by investigating whether such interpretable dictionaries can be effectively transferred to a weak learner using minimal comparative feedback.

<table><tr><td>Feedback type</td><td>Standard Constructive</td><td>Sparse Constructive</td><td>Standard Sampling</td><td>Sparse Sampling</td></tr><tr><td>Feedback Complexity</td><td> $\Theta(\frac{r(r+1)}{2} + p - r + 1)$ </td><td> $O(\frac{p(p+1)}{2})$ </td><td> $^{*}\Theta(\frac{p(p+1)}{2})$ </td><td> $^{*}cp^{2}(\frac{2}{p_{s}^{2}}\log\frac{2}{\delta})^{\frac{1}{p^{2}}}$ </td></tr></table>

<table><tr><td rowspan="2">Prior Works</td><td rowspan="2">SAE(Sharkey et al., 2025)</td><td rowspan="2">CRAFT(Fel et al., 2023)</td><td colspan="3">Probing (Marks and Tegmark, 2024)</td></tr><tr><td>LR</td><td>CCS (Burns et al., 2023)</td><td>LDA</td></tr><tr><td>Learning Complexity</td><td>Tnpd</td><td>npk</td><td>Tnp</td><td>Tnp</td><td> $\mathcal{O}(np^{2} + p^{3})$ </td></tr></table>

Table 1: Comparison of feedback complexity in this work against prior feature retrieval (learning) methods. T : number of iterations, n: number of samples, $p \mathrm { : }$ activation dimension, d: input space dimension, k: number of latent components, r: the rank of the feature matrix, $c > 0$ is a constant, and $p _ { \mathsf { S } }$ depends on activation distribution and sparsity s. We use ∗ to denote “almost surely” and ∗ to denote “with high probability” guarantees.

Triplet learning a covariance matrix Learning a feature matrix (for a dictionary) up to normal transformation can be viewed through two established frameworks: covariance estimation (Chen et al., 2013; Li and Voroninski, 2013) and learning Mahalanobis distances (Kulis, 2013). While these frameworks traditionally rely on exact or noisy measurements, our work introduces a distinct mechanism based solely on relative feedback, aligning more closely with the semantic structure of Mahalanobis distances. The study of such distances has been central to metric learning research (Bellet et al., 2015; Kulis, 2013), encompassing both supervised approaches (Weinberger and Saul, 2009; Xing et al., 2002) and unsupervised methods such as LDA (Fisher, 1936) and PCA (Jolliffe, 1986). Schultz and Joachims (2003) and Kleindessner and von Luxburg (2016) have extended this framework to incorporate relative comparisons on distances. Particularly relevant to our work are studies by Schultz and Joachims (2003) and Mason et al. (2017) that employ triplet comparisons, though these typically assume i.i.d. triplets with potentially noisy measurements. Our approach differs by incorporating an active learning element: while signals are drawn i.i.d, an agent selectively provides feedback on informative instances. This constructive triplet framework for covariance estimation represents a novel direction, drawing inspiration from machine teaching, where a teaching agent provides carefully chosen examples to facilitate learning (Zhu et al., 2018; Kumar et al., 2021).

# 3. Problem Setup

We denote by $\nu \subseteq \mathbb { R } ^ { p }$ the space of activations or representations and by $\mathcal { X } \subseteq \mathbb { R } ^ { d }$ the space of samples. For the space of feature matrices (for a dictionary or Mahalanobis distances), denoted as $M _ { \mathsf { F } } .$ , we consider the family of symmetric positive semi-definite matrices in $\mathbb { R } ^ { p \times p }$ , i.e. $\mathcal { M } _ { \sf F } = \left\{ \Phi \in \sf S y m _ { + } ( \mathbb { R } ^ { p \times p } ) \right\}$ . We denote a feedback set as F which consists of triplets $( x , y , z ) \in \mathcal { V } ^ { 3 }$ with corresponding signs $\ell \in \{ + 1 , 0 , - 1 \}$ .

We use the standard notations in linear algebra over a space of matrices provided in Appendix B.

Triplet feedback An agent provides feedback on activations in V through relative triplet comparisons $( x , y , z ) \in \mathcal { V }$ . Each comparison evaluates linear combinations of feature vectors:

$$
\begin{array}{l} \sum_ {i = 1} ^ {p} x _ {i} u _ {i} \left(" \text { feature } i"\right) \text { is   more   similar   to } \\ \sum_ {i = 1} ^ {p} y _ {i} u _ {i} \left(" \text {feature} i"\right) \text {than to} \sum_ {i = 1} ^ {p} z _ {i} u _ {i} \left(" \text {feature} i"\right) \\ \end{array}
$$

We study both sparse and non-sparse activation feedbacks, where sparsity is defined as:

Definition 1 (s-sparse activations). An activation $\alpha \in \mathbb { R } ^ { p }$ is s-sparse if at most s many indices of α are non-zero.

![](images/a50bb187edc9b1d63cc47e0f3d76faf651e61f0058da356473d815e517c9e333.jpg)  
Figure 1: Features via Recursive Feature Machines. We perform monomial regression on $z \sim \mathcal { N } ( 0 , 0 . 5 I _ { 1 0 } )$ with target $f ^ { * } ( z ) = z _ { 0 } z _ { 1 } { \bf 1 } ( z _ { 5 } > 0 )$ . An RFM kernel machine $\begin{array} { r } { \hat { f } _ { \Phi } ( z ) = \sum _ { { u } _ { i } \in { \mathcal { D } _ { \mathrm { t r a i n } } } } a _ { i } K _ { \Phi } ( y _ { i } , z ) } \end{array}$ is trained for 5 iterations on 4000 samples to produce the ground-truth feature matrix $\Phi ^ { * }$ of rank 4 (Radhakrishnan et al., 2024). We then query an agent for feedback via: eigendecomposition (Theorem 1), sparse constructive (Theorem 2), random Gaussian sampling (Theorem 3), and sparse sampling with $\mu = 0 . 9$ (Theorem 4). Eigendecomposition, sparse constructive, and random sampling achieve the ground-truth MSE with only 55 feedbacks, whereas high-sparsity sampling yields inferior features and larger MSE.

Since triplet comparisons are invariant to positive scaling of feature matrices, we define:

Definition 2 (Feature equivalence). For a feature family $\mathcal { M } _ { F } ,$ feature matrices $\Phi ^ { \prime }$ and $\Phi ^ { * }$ are equivalent if there exists $\lambda > 0$ such that $\Phi ^ { \prime } = \lambda \cdot \Phi ^ { * }$ .

We study a learning framework where the learner merely satisfies the constraints provided by the agent’s feedback:

Definition 3 (Oblivious learner). A learner is oblivious if it randomly selects a feature matrix from the set of valid solutions to a given feedback set F , i.e., arbitrarily chooses $\Phi \in { \mathcal { M } } _ { F } ( \mathcal { F } )$ , where $\scriptstyle { \mathcal { M } } _ { F } ( { \mathcal { F } } )$ represents the set of feature matrices satisfying the constraints in F .

This framework aligns with version space learning, where $\mathsf { V S } ( \mathcal { F } , \mathcal { M } _ { \mathsf { F } } )$ denotes the set of feature matrices in $\mathcal { M } _ { \sf F } ( \mathcal { F } )$ compatible with feedback set ${ \mathcal F } .$

Prior work on dictionary learning has established recovery up to linear transformation under weak mutual incoherence (Gribonval and Schnass, 2010). In our setting, with the agent’s feature feedback corresponding to D (or $\mathsf { L } ) \in \mathbb { R } ^ { d \times p } ,$ the learner recovers L up to normal transformation. Moreover, when L has orthogonal rows (strong incoherence), we can recover L up to rotation and sign changes as stated below, with proof deferred to Appendix D.

Lemma 1 (Recovering orthogonal representations). Assume $\Phi \in S y m _ { + } ( \mathbb { R } ^ { p \times p } )$ . Define the set of orthogonal Cholesky decompositions of Φ as

$$
\mathcal {W} _ {\boldsymbol {C D}} = \left\{\boldsymbol {U} \in \mathbb {R} ^ {p \times r} \mid \boldsymbol {\Phi} = \boldsymbol {U} \boldsymbol {U} ^ {\top} \& \boldsymbol {U} ^ {\top} \boldsymbol {U} = d i a g (\lambda_ {1}, \dots , \lambda_ {r}) \right\},
$$

where $r = r a n k ( \Phi )$ and $\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { r }$ are the eigenvalues of Φ in descending order. Then, for any two matrices $\pmb { U } , \pmb { U } ^ { \prime } \in \mathcal { W } _ { C D } ,$ , there exists an orthogonal matrix $\pmb { R } \in \mathbb { R } ^ { r \times r }$ such that ${ \pmb U } ^ { \prime } = { \pmb U } { \pmb R } _ { \mathrm { : } }$ , where R is block diagonal with orthogonal blocks corresponding to any repeated diagonal entries $\lambda _ { i }$ in $\pmb { U } ^ { \top }$ U. Additionally, each column of U′ can differ from the corresponding column of U by a sign change.

Algorithm 1 Model of Feature learning with Feedback

Given: Representation space $\overline { { \mathcal { V } \subseteq \mathbb { R } ^ { p } } } ,$ Feature family MF

In batch setting:

1. Teacher picks triplets $\mathcal { F } ( \mathcal { V } , \Phi ^ { * } ) =$

$$
\left\{(x, y, z) \in \mathcal {V} ^ {3} \mid (x - y) ^ {\top} \boldsymbol {\Phi} ^ {*} (x - y) \geq (x - z) ^ {\top} \boldsymbol {\Phi} ^ {*} (x - z) \right\}
$$

2. Learner receives F, and obliviously picks a feature matrix $\Phi \in { \mathcal { M } } _ { \sf F }$ that satisfy the set of constraints in $\mathcal { F } ( \nu , \Phi ^ { * } )$

3. Learner outputs $\Phi .$

We note that the recovery of L is pertaining to the assumption that all the rows of L are orthogonal, and thus the rank of L is $r \ = \ d .$ In cases where $r \ < \ d ,$ one needs additional information in the form of ground sample $\pmb { x } = \pmb { \mathrm { L } } \alpha$ for some activation α to recover L up to a linear transformation. Finally, we provide the general interaction protocol in Algorithm 1.

# 4. Sparse Feature Learning with Constructive Feedback

Here, we study the feedback complexity in the setting where agent is allowed to pick/construct any activation from $\mathbb { R } ^ { p }$ . This setup allows us to study the best-case informativeness of activation vectors for feature learning.

Reduction to Pairwise Comparisons The general triplet feedbacks with potentially inequality constraints in Algorithm 1 can be simplified to pairwise comparisons with equality constraints with a simple manipulation as follows.

Lemma 2. Let $\Phi ^ { * } \in { \mathcal { M } } _ { \sf F }$ be a target feature matrix in representation space $\mathbb { R } ^ { p }$ used for oblivious learning. Given

a feedback set

$$
\mathcal {F} = \left\{(x, y, z) \in \mathbb {R} ^ {3 p} \mid (x - y) ^ {\top} \Phi^ {*} (x - y) \geq \right.
$$

$$
(x - z) ^ {\top} \boldsymbol {\Phi} ^ {*} (x - z) \},
$$

such that any $\Phi ^ { \prime } \in V S ( \mathcal { F } , \mathcal { M } _ { \sf F } )$ is feature equivalent to $\Phi ^ { * }$ , there exists a pairwise feedback set

$$
\mathcal {F} ^ {\prime} = \left\{(y ^ {\prime}, z ^ {\prime}) \in \mathbb {R} ^ {2 p} \mid y ^ {\prime \top} \Phi^ {*} y ^ {\prime} = z ^ {\prime \top} \Phi^ {*} z ^ {\prime} \right\}
$$

such that $\Phi ^ { \prime } \in V S ( \mathcal { F } ^ { \prime } , \mathcal { M } _ { \sf F } )$ .

Proof. WLOG, assume x $\neq z$ for all $( x , y , z ) \in { \mathcal { F } }$ . For any triplet $( x , y , z ) \in { \mathcal { F } }$ : Case (i): If $( x - y ) ^ { \top } \Phi ^ { * } ( x - y ) =$ $( x - z ) ^ { \top } \Phi ^ { * } ( x - z )$ , then $( x - y , x - z )$ satisfies the equality.

Case (ii): If $( x - y ) ^ { \top } \Phi ^ { * } ( x - y ) > ( x - z ) ^ { \top } \Phi ^ { * } ( x - z )$ , then for some $\lambda > 0 \colon$

$$
(x - y) ^ {\top} \boldsymbol {\Phi} ^ {*} (x - y) = (1 + \lambda) (x - z) ^ {\top} \boldsymbol {\Phi} ^ {*} (x - z)
$$

implying $( x - y , \sqrt { 1 + \lambda } ( x - z ) )$ satisfies the equality.

Thus, each triplet in $\mathcal { F }$ maps to a pair in ${ \mathcal { F } } ^ { \prime }$ , preserving feature equivalence under positive scaling.

This implies that if triplet comparisons are used in Algorithm 1, equivalent pairwise comparisons exist satisfying:

$$
\boldsymbol {\Phi} ^ {\prime} = \lambda \cdot \boldsymbol {\Phi} ^ {*}, \quad \lambda > 0, \tag {1a}
$$

$$
\boldsymbol {\Phi} ^ {\prime} \in \left\{\boldsymbol {\Phi} \in \mathcal {M} _ {\mathsf {F}}   \middle |   \forall (y, z) \in \mathcal {F} ^ {\prime},   y ^ {\top} \boldsymbol {\Phi} y = z ^ {\top} \boldsymbol {\Phi} z \right\}. \tag {1b}
$$

Now, we show a reformulation of the oblivious learning problem for a feature matrix using pairwise comparisons that provide a unique geometric interpretation. Consider a pair $( y , z )$ and a matrix Φ. An equality constraint implies

$$
y ^ {\top} \Phi y = z ^ {\top} \Phi z \iff \langle \Phi , y y ^ {\top} - z z ^ {\top} \rangle = 0
$$

where $\langle \cdot , \cdot \rangle$ denotes the Frobenius inner product. Now, given a set of pairwise feedbacks

$$
\mathcal {F} (\mathbb {R} ^ {p}, \mathcal {M} _ {\mathsf {F}}, \boldsymbol {\Phi} ^ {*}) = \{(y _ {i}, z _ {i}) \} _ {i = 1} ^ {k}
$$

corresponding to the target feature matrix $\Phi ^ { * }$ , the learning problem defined by Eq. (1b) can be formulated as:

$$
\forall (y, z) \in \mathcal {F} (\mathbb {R} ^ {p}, \mathcal {M} _ {\mathsf {F}}, \boldsymbol {\Phi} ^ {*}), \quad \langle \boldsymbol {\Phi}, y y ^ {\top} - z z ^ {\top} \rangle = 0. \tag {2}
$$

Geometrically, the condition in Eq. (2) implies that any solution Φ should annihilate the subspace of the orthogonal complement that is spanned by the matrices $\{ y y ^ { \top } -$ $z z ^ { \top } \} _ { ( y , z ) \in \mathcal { F } }$ . Formally, this complement is defined as:

$$
\mathcal {O} _ {\boldsymbol {\Phi} ^ {*}} := \left\{S \in \operatorname{Sym} \left(\mathbb {R} ^ {p \times p}\right) \mid \langle \boldsymbol {\Phi} ^ {*}, S \rangle = 0 \right\}.
$$

# 4.1. Constructive feedbacks: Worst-case lower bound

To learn a symmetric PSD matrix, learner needs at most $p ( p + 1 ) / 2$ constraints for linear programming corresponding to the number of degrees of freedom. So, the first question is are there pathological cases of feature matrices in $\mathcal { M } _ { \sf F }$ which would require at least $p ( p + 1 ) / 2$ many triplet feedbacks in Algorithm 1. This indeed is the case, if a target matrix $\Phi ^ { * } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ is full rank.

In the following proposition proven in Appendix E, we show a strong lower bound on the worst-case $\Phi ^ { * }$ that turns out to be of order $\Omega ( p ^ { 2 } )$ .

Proposition 1. In the constructive setting, the worst-case feedback complexity of the class $\mathcal { M } _ { F }$ with general activations is at the least $( p ( p + 1 ) / 2 - 1 )$ .

Proof Outline. As discussed in Eq. (1) and Eq. (2), for a fullrank feature matrix $\Phi ^ { * } \in { \mathcal { M } } _ { \sf F }$ , the span of any feedback set ${ \mathcal { F } } ,$ i.e., span $\langle \{ x x ^ { \top } - y y ^ { \top } \} _ { ( x , y ) \in { \mathcal F } } \rangle$ , must lie within the orthogonal complement $\mathcal { O } _ { \Phi ^ { * } }$ of $\Phi ^ { * }$ in the space of symmetric matrices $\bar { \mathsf { S y m } } ( \mathbb { R } ^ { p \times p } )$ . Conversely, if $\Phi ^ { * }$ has full rank, then $O _ { \Phi } ,$ ∗ is contained within this span. This necessary condition requires the feedback set to have a size of at least $\textstyle { \frac { p ( p + 1 ) } { 2 } } - 1$ , given that dim $\begin{array} { r } { ( \mathsf { S y m } ( \mathbb { R } ^ { p \times p } ) ) = \frac { p ( p + 1 ) } { 2 } } \end{array}$ . 2 2

Since the worst-case bound is pessimistic for oblivious learning of Eq. (1) a general question is how feedback complexity varies over the feature model $\mathcal { M } _ { \sf F }$ . Now, we study the feedback complexity for feature model based on the rank of the matrix, showing that the bounds can be drastically reduced.

# 4.2. Feature learning of low-rank matrices

As stated in Proposition 1, the learner requires at least p(p+1)2 − 1 feedback pairs to annihilate the orthogonal com- $\textstyle { \frac { p ( p + 1 ) } { 2 } } - 1$ 2 plement $\mathcal { O } _ { \Phi ^ { * } }$ . However, this requirement decreases with a lower rank of $\Phi ^ { * }$ . We illustrate this in Fig. 1 for a feature matrix $\Phi \in \mathbb { R } ^ { 1 0 \times 1 0 }$ of rank 4 trained via Recursive Feature Machines (Radhakrishnan et al., 2024) (see Section 6 for experimental details).

Consider an activation $\boldsymbol { \alpha } ~ \in ~ \mathbb { R } ^ { p }$ in the nullspace of $\Phi ^ { * }$ . Since $\Phi ^ { * } \alpha = 0 .$ , it follows that $\alpha ^ { \top } \Phi ^ { * } \alpha = 0$ . Moreover, for another activation β /∈ span⟨α⟩ in the nullspace, any linear combination aα $+ b \beta$ satisfies

$$
(a \alpha + b \beta) ^ {\top} \Phi^ {*} (a \alpha + b \beta) = 0.
$$

This suggests a strategy for designing effective feedback based on the kernel $\mathrm { K e r } ( \Phi ^ { * } )$ and the null space $\mathrm { { n u l l } } ( \Phi ^ { * } )$ of $\Phi ^ { * }$ (see Appendix B for table of notations). This intuition is formalized by the eigendecomposition of the feature matrix:

$$
\boldsymbol {\Phi} ^ {*} = \sum_ {i = 1} ^ {r} \lambda_ {i} u _ {i} u _ {i} ^ {\top}, \tag {3}
$$

where $\{ \lambda _ { i } \}$ are the eigenvalues and $\{ u _ { i } \}$ are the orthonormal eigenvectors. Since $\Phi ^ { * } \succeq 0$ this decomposition is unique with non-negative eigenvalues.

To teach $\Phi ^ { * }$ , the agent can employ a dual approach: teaching the kernel associated with the eigenvectors in this decomposition and the null space separately. Specifically, the agent can provide feedbacks corresponding to the eigenvectors of $\Phi ^ { * } \mathbf { \bar { s } }$ kernel and extend the basis $\{ u _ { i } \}$ for the null space. We first present the following useful result (see proof in Appendix F).

Lemma 3. Let $\{ v _ { i } \} _ { i = 1 } ^ { r } \subset \mathbb { R } ^ { p }$ be a set of orthogonal vectors. Then, the set of rank-1 matrices

$$
\mathcal {B} := \left\{v _ {i} v _ {i} ^ {\top}, (v _ {i} + v _ {j}) (v _ {i} + v _ {j}) ^ {\top} \mid 1 \leq i <   j \leq r \right\}
$$

is linearly independent in the space of symmetric matrices $S y m ( \mathbb { R } ^ { p \times p } )$ .

Using this construction, the agent can provide feedbacks of the form $( u _ { i } , \sqrt { c _ { i } } y )$ for some $y \in \mathbb { R } ^ { p }$ with $\Phi ^ { * } y \neq 0$ and $v _ { i } ^ { \top } \Phi ^ { * } v _ { i } = c _ { i } \dot { y } ^ { \top } \Phi ^ { * } y$ to teach the kernel of $\Phi ^ { * }$ . For an orthogonal extension $\{ u _ { i } \} _ { i = r + 1 } ^ { p }$ where $\Phi ^ { * } u _ { i } = 0$ for all $i = r + 1 , \dotsc , p ,$ , feedbacks of the form $( u _ { i } , 0 )$ suffice to teach the null space of $\Phi ^ { * }$ .

This is the key idea underlying our study on feedback complexity in the general constructive setting that is stated below with the full proof deferred to Appendices F and G.

Theorem 1 (General Activations). Let $\Phi ^ { * } \in \mathcal { M } _ { F }$ be a target feature matrix with ran $k ( \Phi ^ { * } ) ~ = ~ r .$ . Then, in the setting of constructive feedbacks with general activations, the feedback complexity has a tight bound of $\begin{array} { r } { \Theta \left( \frac { r ( r + 1 ) } { 2 } + ( p - r ) - 1 \right) f o r E q . \ ( \ ) , } \end{array}$ .

Proof Outline. As discussed above we decompose the feature matrix $\Phi ^ { * }$ into its eigenspace and null space, leveraging the linear independence of the constructed feedbacks to ensure that the span covers the necessary orthogonal complements. The upper bound is established with a simple observation: $r ( r + 1 ) / 2 - 1$ many pairs composed of B are sufficient to teach $\Phi ^ { * }$ if the null space of $\Phi ^ { * }$ is known, whereas the agent only needs to provide $( p - r )$ many feedbacks corresponding to a basis extension to cover the null space, and hence the stated upper bound is achieved.

The lower bound requires showing that a valid feedback set possesses two spanning properties of $\langle x x ^ { \top } - y y ^ { \top } \rangle$ for all $( x , y ) \in { \mathcal { F } } \colon$ (1) it must include any $\Phi \in { \mathcal { O } } _ { \Phi }$ ∗ whose column vectors are within the span of eigenvectors of $\Phi ^ { * }$ , and (2) it must include any $v v ^ { \top }$ for some subset U that spans the null space of $\Phi ^ { * }$ and $y \in U$ . □

Learning with sparse activations In the discussion above, we demonstrated a strategy for reducing the feedback complexity when general activations are allowed. Now, we aim to understand how this complexity changes when activations are s-sparse (see Definition 1) for some $s < p .$ Notably, there exists a straightforward construction of rank-1 matrices using a sparse set of activations.

Consider this sparse set of activations B consisting of $\frac { p ( p + 1 ) } { 2 }$ items in $\mathbb { R } ^ { p }$ (see Kumar and Dasgupta (2024)):

$$
B = \left\{e _ {i} \mid 1 \leq i \leq p \right\} \cup \left\{e _ {i} + e _ {j} \mid 1 \leq i <   j \leq p \right\}, \tag {4}
$$

where $\left\{ e _ { i } \right\}$ forms the standard basis. Using a similar argument to Lemma 3, we note that the set of rank-1 matrices

$$
\mathcal {B} _ {\text { sparse }} := \left\{u u ^ {\top} \mid u \in B \right\}
$$

is linearly independent in the space of symmetric matrices $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ and forms a basis. Moreover, every activation in $B _ { \mathsf { e x t } }$ is at most 2-sparse (see Definition 1). With this, we state the main result on learning with sparse constructive feedback here.

Theorem 2 (Sparse Activations). Let $\Phi ^ { * } \in { \mathcal { M } } _ { F }$ be the target feature matrix. If an agent can construct pairs of activations from a representation space $\mathbb { R } ^ { p } ,$ , then the feedback complexity of the feature model $\mathcal { M } _ { F }$ with 2-sparse activations is upper bounded by p(p+1)2 . $\frac { p ( p + 1 ) } { 2 }$

Remark: While the lower bound from Theorem 1 applies here, sparse settings may require even more feedbacks. Consider a rank-1 matrix $\Phi ^ { * } = v v ^ { \top }$ with sparsity $( v ) \ = \ p$ . By the Pigeonhole principle, representing this using ssparse activations requires at least $( p / s ) ^ { 2 }$ rank-1 matrices. Thus, for constant sparsity $s = O ( 1 )$ , we need $\Omega ( p ^ { 2 } )$ ) feedbacks—implying sparse representation of dense features might not exploit the low-rank structure to minimize feedbacks.

# 5. Sparse Feature Learning with Sampled Feedback

In general, the assumption of constructive feedback may not hold in practice, as ground truth samples from nature or induced representations of a model are typically independently sampled from the representation space. The literature on Mahalanobis distance learning/dictionary learning has explored distributional assumptions on the sample/activation space (cf Gribonval et al. (2014)).

![](images/372bdf281633368d4f8795f2510c36d658e1de4b182435540b81d934d1f6c528.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Value   |
|-----|--------|---------|
| 0   | 0      | 0.020   |
| 0   | 1      | 0.015   |
| 0   | 2      | 0.010   |
| 0   | 3      | 0.005   |
| 0   | 4      | 0.000   |
| 0   | 5      | 0.005   |
| 0   | 6      | 0.010   |
| 0   | 7      | 0.015   |
| 0   | 8      | 0.020   |
| 1   | 0      | 0.020   |
| 1   | 1      | 0.015   |
| 1   | 2      | 0.010   |
| 1   | 3      | 0.005   |
| 1   | 4      | 0.000   |
| 1   | 5      | 0.005   |
| 1   | 6      | 0.010   |
| 1   | 7      | 0.015   |
| 1   | 8      | 0.020   |
| 2   | 0      | 0.020   |
| 2   | 1      | 0.015   |
| 2   | 2      | 0.010   |
| 2   | 3      | 0.005   |
| 2   | 4      | 0.000   |
| 2   | 5      | 0.005   |
| 2   | 6      | 0.010   |
| 2   | 7      | 0.015   |
| 2   | 8      | 0.020   |
| ... | ...    | ...     |
| 8   | ...    | ...     |
</details>

MSE:0.00161579, Trainingpoints:4000

![](images/3436f1289b96178c2d9ee7b2da2f734192358ced352189be985807bfe8f26670.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|-------|
| 0 | 0 | 1.0 |
| 0 | 2 | 1.0 |
| 0 | 4 | 1.0 |
| 0 | 6 | 1.0 |
| 0 | 8 | 1.0 |
| 2 | 0 | 1.0 |
| 2 | 2 | 1.0 |
| 2 | 4 | 1.0 |
| 2 | 6 | 1.0 |
| 2 | 8 | 1.0 |
| 4 | 0 | 1.0 |
| 4 | 2 | 1.0 |
| 4 | 4 | 1.0 |
| 4 | 6 | 1.0 |
| 4 | 8 | 1.0 |
| 6 | 0 | 1.0 |
| 6 | 2 | 1.0 |
| 6 | 4 | 1.0 |
| 6 | 6 | 1.0 |
| 6 | 8 | 1.0 |
| 8 | 0 | 1.0 |
| 8 | 2 | 1.0 |
| 8 | 4 | 1.0 |
| 8 | 6 | 1.0 |
| 8 | 8 | 1.0 |
</details>

MSE: 0.00152222, Feedbacks:55,mu:0.97

![](images/2c05e822c0841ff8420dc9178927c019047bc724ad420b57cb1780d9b55c41d7.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 2 | 0.8 |
| 0 | 4 | 0.6 |
| 0 | 6 | 0.4 |
| 0 | 8 | 0.2 |
| 2 | 0 | 0.8 |
| 2 | 2 | 0.6 |
| 2 | 4 | 0.4 |
| 2 | 6 | 0.2 |
| 2 | 8 | 0.0 |
| 4 | 0 | 0.8 |
| 4 | 2 | 0.6 |
| 4 | 4 | 0.4 |
| 4 | 6 | 0.2 |
| 4 | 8 | 0.0 |
| 6 | 0 | 0.8 |
| 6 | 2 | 0.6 |
| 6 | 4 | 0.4 |
| 6 | 6 | 0.2 |
| 6 | 8 | 0.0 |
| 8 | 0 | 0.8 |
| 8 | 2 | 0.6 |
| 8 | 4 | 0.4 |
| 8 | 6 | 0.2 |
| 8 | 8 | 0.0 |
</details>

MSE:5.63603468, Feedbacks:55,mu:0.7

![](images/f5c79950c798ad2468dea445e56fa4e0d01922c0e8f9d73e00f553a08a52b8a8.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 1 | 0.9 |
| 0 | 2 | 0.8 |
| 0 | 3 | 0.7 |
| 0 | 4 | 0.6 |
| 0 | 5 | 0.5 |
| 0 | 6 | 0.4 |
| 0 | 7 | 0.3 |
| 0 | 8 | 0.2 |
| 1 | 0 | 0.9 |
| 1 | 1 | 0.8 |
| 1 | 2 | 0.7 |
| 1 | 3 | 0.6 |
| 1 | 4 | 0.5 |
| 1 | 5 | 0.4 |
| 1 | 6 | 0.3 |
| 1 | 7 | 0.2 |
| 1 | 8 | 0.1 |
| 2 | 0 | 0.8 |
| 2 | 1 | 0.7 |
| 2 | 2 | 0.6 |
| 2 | 3 | 0.5 |
| 2 | 4 | 0.4 |
| 2 | 5 | 0.3 |
| 2 | 6 | 0.2 |
| 2 | 7 | 0.1 |
| 2 | 8 | 0.0 |
| 3 | 0 | 0.7 |
| 3 | 1 | 0.6 |
| 3 | 2 | 0.5 |
| 3 | 3 | 0.4 |
| 3 | 4 | 0.3 |
| 3 | 5 | 0.2 |
| 3 | 6 | 0.1 |
| 3 | 7 | 0.0 |
| 3 | 8 | -0.1 |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
</details>

MSE:0.00156753, Feedbacks:55,mu: 0.1

![](images/f38458db5d979d5d8ea28640a6392a7500ae5ac1c6ad670e5578f331ec1f7c63.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 |
| 1 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |
| 2 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| 3 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 |
| 4 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | -0.1 | -0.2 | -0.3 |
| 5 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | -0.1 | -0.2 | -0.3 | -0.4 |
| 6 | 0.4 | 0.3 | 0.2 | 0.1 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 |
| 7 | 0.3 | 0.2 | 0.1 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 | -0.6 |
| 8 | 0.2 | 0.1 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 | -0.6 | -0.7 |
The heatmap visualizes a sparse sampling pattern where each cell represents a unique sample's value at its respective coordinate point on the x-axis and y-axis.
</details>

MSE:0.00152041, Feedbacks:110,mu:0.97

![](images/8bb80e3e1a2723c601e0e1e4196765f69eace1f80b9f11e40f1e065daf9cef3a.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 0.9 |
| 0 | 2 | 0.9 |
| 0 | 4 | 0.9 |
| 0 | 6 | 0.9 |
| 0 | 8 | 0.9 |
| 2 | 0 | 0.9 |
| 2 | 2 | 0.9 |
| 2 | 4 | 0.9 |
| 2 | 6 | 0.9 |
| 2 | 8 | 0.9 |
| 4 | 0 | 0.9 |
| 4 | 2 | 0.9 |
| 4 | 4 | 0.9 |
| 4 | 6 | 0.9 |
| 4 | 8 | 0.9 |
| 6 | 0 | 0.9 |
| 6 | 2 | 0.9 |
| 6 | 4 | 0.9 |
| 6 | 6 | 0.9 |
| 6 | 8 | 0.9 |
| 8 | 0 | 0.9 |
| 8 | 2 | 0.9 |
| 8 | 4 | 0.9 |
| 8 | 6 | 0.9 |
| 8 | 8 | 0.9 |
</details>

MSE: 0.00152517, Feedbacks:275,mu:0.97

![](images/78d3dae695913049cfc8ea5ec178bd21c5e371f0f82c60e7818409c4310f42a6.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Value |
|---|---|---|
| 0 | 0 | 0.9 |
| 0 | 1 | 0.9 |
| 0 | 2 | 0.9 |
| 0 | 3 | 0.9 |
| 0 | 4 | 0.9 |
| 0 | 5 | 0.9 |
| 0 | 6 | 0.9 |
| 0 | 7 | 0.9 |
| 0 | 8 | 0.9 |
| 1 | 0 | 0.9 |
| 1 | 1 | 0.9 |
| 1 | 2 | 0.9 |
| 1 | 3 | 0.9 |
| 1 | 4 | 0.9 |
| 1 | 5 | 0.9 |
| 1 | 6 | 0.9 |
| 1 | 7 | 0.9 |
| 1 | 8 | 0.9 |
| 2 | 0 | 0.9 |
| 2 | 1 | 0.9 |
| 2 | 2 | 0.9 |
| 2 | 3 | 0.9 |
| 2 | 4 | 0.9 |
| 2 | 5 | 0.9 |
| 2 | 6 | 0.9 |
| 2 | 7 | 0.9 |
| 2 | 8 | 0.9 |
| 3 | 0 | 0.9 |
| 3 | 1 | 0.9 |
| 3 | 2 | 0.9 |
| 3 | 3 | 0.9 |
| 3 | 4 | 0.9 |
| 3 | 5 | 0.9 |
| 3 | 6 | 0.9 |
| 3 | 7 | 0.9 |
| 3 | 8 | 0.9 |
| 4 | 0 | 0.9 |
| 4 | 1 | 0.9 |
| 4 | 2 | 0.9 |
| 4 | 3 | 0.9 |
| 4 | 4 | 0.9 |
| 4 | 5 | 0.9 |
| 4 | 6 | 0.9 |
| 4 | 7 | 0.9 |
| 4 | 8 | 0.9 |
| 5 | 0 | 0.9 |
| 5 | 1 | 0.9 |
| 5 | 2 | 0.9 |
| 5 | 3 | 0.9 |
| 5 | 4 | 0.9 |
| 5 | 5 | 0.9 |
| 5 | 6 | 0.9 |
| 5 | 7 | 0.9 |
| 5 | 8 | 0.9 |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
</details>

MSE:0.00156868, Feedbacks:440,mu:0.97

![](images/5d377cfed0b4a640c142e8d228bfa845d661701f52aaa36c392c270640092842.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 |
| 1 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 |
| 2 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 |
| 3 | 0.6 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 | -0.2 |
| 4 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 | -0.2 | -0.3 |
| 5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 | -0.2 | -0.3 | -0.4 |
| 6 | 0.3 | 0.2 | 0.1 | 0.0 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 |
| 7 | 0.2 | 0.1 | 0.0 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 | -0.6 |
| 8 | 0.1 | 0.0 | -0.1 | -0.2 | -0.3 | -0.4 | -0.5 | -0.6 | -0.7 |
The image displays a heatmap with color intensity representing a scalar value that varies across the matrix, where darker colors indicate higher values and lighter colors indicate lower values for each variable's output.
</details>

MSE:0.00158344, Feedbacks:110o,mu:0.97   
Figure 2: Sparse sampling: We consider the same setup as Fig. 1 for the target function $f ^ { * } ( z ) = z _ { 0 } z _ { 1 } z _ { 3 } { \bf 1 } ( z _ { 5 } > 0 )$ . In these plots, we employ sparse sampling feedback methods where an agent provides feedback based on $\Phi ^ { * }$ with different sparsity probability (mu: probability of 0 being sampled). Thus, as mu decreases, the theorized complexity of $p ( p + 1 ) / 2 = 5 5$ obtains a close approximation of $\Phi ^ { * }$ . But for $m u = . 9 7$ , the agent needs to sample more number of activations to approximate properly, i.e., from $5 5 , 1 1 0 , . . . ,$ and 1100 approximation gradually improves as shown in Theorem 4.

Algorithm 2 Feature learning with Sampled Representations

Given: Representation space $\mathcal { V } \subset \mathbb { R } ^ { p } ,$ , Distribution over representations $\mathcal { D } _ { \mathcal { V } }$ , Feature family $\mathcal { M } _ { \sf F }$ .

In batch setting:

1. Teacher receives sampled representations $\mathcal { V } _ { n } \sim \mathcal { D } _ { \mathcal { V } }$ .   
2. Teacher picks pairs $\mathcal { F } ( \mathcal { V } _ { n } , \Phi ^ { * } ) =$

$$
\left\{(x, \sqrt {\lambda_ {x}} y) \mid (x, y) \in \mathcal {V} _ {n} ^ {2}, x ^ {\top} \boldsymbol {\Phi} ^ {*} x = \lambda_ {x} \cdot y ^ {\top} \boldsymbol {\Phi} ^ {*} y \right\}
$$

3. Learner receives $\mathcal { F } ;$ and obliviously picks a feature matrix $\Phi \in { \mathcal { M } } _ { \sf F }$ that satisfy the set of constraints in $\mathcal { F } ( \mathcal { V } _ { n } , \Phi ^ { * } )$   
4. Learner outputs Φ.

In this section, we consider a more realistic scenario where the agent observes a set of representations/activations $\nu _ { n } : =$ $\{ \alpha _ { 1 } , \alpha _ { 2 } , \ldots , \alpha _ { n } \} \sim { \mathcal { D } } _ { \boldsymbol { \nu } }$ , with $\mathcal { D } _ { \mathcal { V } }$ being an unknown measure over the continuous space $\nu \subseteq \mathbb { R } ^ { p }$ . With these observations, the agent designs pairs of activations to teach a target feature matrix $\Phi ^ { * } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ .

As shown in Lemma 2, we can reduce inequality constraints with triplet comparisons to equality constraints with pairs in the constructive setting. However, when the agent is restricted to selecting activations from the sampled set $\nu _ { n }$ rather than arbitrarily from V , this reduction no longer holds. Observe that if $\alpha , \beta \sim \mathrm { i i d } \ D _ { \nu }$ and $\Phi ^ { * } \neq 0$ a non-degenerate feature matrix, then

$$
\alpha^ {\top} \Phi^ {*} \alpha = \beta^ {\top} \Phi^ {*} \beta \implies \sum_ {i, j} (\alpha_ {i} \alpha_ {j} - \beta_ {i} \beta_ {j}) \Phi_ {i j} ^ {*} = 0.
$$

This equation represents a non-zero polynomial. According to Sard’s Theorem, the zero set of a non-zero polynomial has Lebesgue measure zero. Therefore,

$$
\mathcal {P} _ {(\alpha , \beta)} \left(\left\{\alpha^ {\top} \boldsymbol {\Phi} ^ {*} \alpha = \beta^ {\top} \boldsymbol {\Phi} ^ {*} \beta \right\}\right) = 0.
$$

Given this, the agent cannot reliably construct pairs that satisfy the required equality constraints from independently sampled activations. Since a general triplet feedback only provides 3 bits of information, exact recovery up to feature equivalence is impossible. To address these limitations, we consider rescaling the sampled activations to enable the agent to design effective pairs for the target feature matrix $\Phi ^ { * } \in \mathcal { M } _ { \sf F }$ .

Rescaled Pairs For a given matrix $\Phi \neq 0 ,$ a sampled input $x \sim D _ { \nu }$ is almost never orthogonal, i.e., almost surely $\Phi x \neq 0$ . This property can be utilized to rescale an input and construct pairs that satisfy equality constraints. Specifically, there exist scalars $\gamma , \lambda ~ > ~ 0$ such that (assuming without loss of generality $x ^ { \top } \Phi x > y ^ { \top } \Phi y )$ ,

$$
x ^ {\top} \Phi x = \lambda \cdot y ^ {\top} \Phi y + y ^ {\top} \Phi y = (\sqrt {1 + \lambda}) y ^ {\top} \Phi (\sqrt {1 + \lambda}) y.
$$

![](images/d1bcbf7730aee3a6bdf72ef59c3cbc60a7fd3696fd2eb71e44fcd718f10938b6.jpg)

(a) Visualization of 100 dimensions: Feature learning on a dictionary retrieved for an MLP layer of ChessGPT of dimension 4096 × 512. From left-to-right, top-to-bottom: ground truth SAE, Eigendecomposition (PCC= .9427, 134912 feedbacks), Sparse Constructive (PCC=.9773, 8390656 feedbacks), Sparse Sampling @200000, @1000000, @2000000, @4000000, @10000000 feedbacks. 

<table><tr><td>Method</td><td>Eigendecomp.</td><td>Sparse Cons.</td><td colspan="4">Sparse Sampling</td></tr><tr><td>Feedbacks</td><td>134912</td><td>8390656</td><td>10 M</td><td>4 M</td><td>2 M</td><td>1 M</td></tr><tr><td>PCC</td><td>0.9427</td><td>0.9773</td><td>0.9741</td><td>0.9625</td><td>0.8256</td><td>0.7152</td></tr></table>

(b) Pearson correlation coefficient and total feedback count for each method on the same SAE dictionary.   
Figure 3: Top: Feature-recovery quality as a function of feedback for a dictionary (of dimension $4 0 9 6 \times 5 1 2 )$ from an SAE trained for ChessGPT. Bottom: numeric PCC and feedback for each method. Sparse constructive achieves almost perfect correlation (0.9773) in only ≈ 8.4M queries; sampling with smaller feedback sizes struggle until $\gtrsim$ 4M samples.

Thus, the pair $( x , ( { \sqrt { 1 + \lambda } } ) y )$ satisfies the equality constraints. With this understanding, we reformulate Algorithm 1 into Algorithm 2. In this section, we analyze the feedback complexity in terms of the minimum number of sampled activations required for the agent to construct an effective feedback set achieving feature equivalence which is illustrated in Fig. 2. Our first result establishes complexity bounds for general activations (without sparsity constraints) sampled from a Lebesgue distribution, with the complete proof provided in Appendix H.

Theorem 3 (General Sampled Activations). Consider a representation space $\mathcal { V } \subseteq \mathbb { R } ^ { p } .$ . Assume that the agent receives activations sampled i.i.d from a Lebesgue distribution $\mathcal { D } _ { \mathcal { V } }$ . Then, for any target feature matrix $\Phi ^ { * } \in { \mathcal { M } } _ { F } ,$ with a tight bound of $\begin{array} { r } { n = \Theta \left( \frac { p ( p + 1 ) } { 2 } \right) } \end{array}$ on the feedback complexity, the oblivious learner (almost surely) learns $\Phi ^ { * }$ up to feature equivalence using the feedback set $\mathcal { F } ( \nu _ { n } , \Phi ^ { * } )$ , i.e.,

$$
\mathcal {P} _ {\mathcal {V}} \left(\forall \Phi^ {\prime} \in \mathcal {F} (\mathcal {V} _ {n}, \Phi^ {*}), \exists \lambda > 0, \Phi^ {\prime} = \lambda \cdot \Phi^ {*}\right) = 1.
$$

Proof Outline. The key observation is that almost surely for any $n \leq p ( p + 1 ) / 2$ sampled activations on a unit sphere $\mathbb { S } ^ { p }$ under Lebesgue measure, the corresponding rank-1 matrices are linearly independent. This is a direct application of Sard’s theorem on the zero set of a non-zero polynomial equation, yielding the upper bound. For the lower bound, we use some key necessary properties of a feedback set as elucidated in the proof of Theorem 1. This result essentially fixes activations that need to be spanned by a feedback set, but under a Lebesgue measure on a continuous domain, the probability of sampling a direction is zero. □

We consider a fairly general distribution over sparse activations similar to the signal model in (Gribonval et al., 2015).

Assumption 1 (Sparse-Distribution). Each index of a sparse activation vector $\alpha \in \mathbb { R } ^ { p }$ is sampled i.i.d from a sparse distribution defined as: for all i,

$$
\mathcal {P} \left(\alpha_ {i} = 0\right) = p _ {i}, \quad \alpha_ {i} \mid \alpha_ {i} \neq 0 \sim L e b e s g u e ((0, 1 ]).
$$

With this we state the main theorem of the section with the proof deferred to Appendix I.

Theorem 4 (Sparse Sampled Activations). Consider a representation space $\mathcal { V } \subseteq \mathbb { R } ^ { p } .$ . Assume that the agent receives representations sampled i.i.d from a sparse distribution $\mathcal { D } _ { \mathcal { V } }$ . Fix a threshold $\delta > 0 ,$ and sparsity parameter $s < p .$ Then, for any target feature matrix $\Phi ^ { * } \in { \mathcal { M } } _ { F } ,$ , with a bound of $\begin{array} { r } { n = O \left( p ^ { 2 } ( \frac { 2 } { p _ { s } ^ { 2 } } \log \frac { 2 } { \delta } ) ^ { 1 / p ^ { 2 } } \right) } \end{array}$ on the feedback complexity using s-sparse feedbacks,the oblivious learner learns $\Phi ^ { * }$ up to feature equivalence with high probability using the feedback set $\mathcal { F } ( \mathcal { V } _ { n } , \Phi ^ { * } )$ , i.e.,

$$
\mathcal {P} _ {\mathcal {V}} (\forall \boldsymbol {\Phi} ^ {\prime} \in \mathcal {F} (\mathcal {V} _ {n}, \boldsymbol {\Phi} ^ {*}), \exists \lambda > 0, \boldsymbol {\Phi} ^ {\prime} = \lambda \cdot \boldsymbol {\Phi} ^ {*}) \geq (1 - \delta),
$$

where $p _ { s }$ depends on $\mathcal { D } _ { \mathcal { V } }$ , and sparasity parameter s.

Proof Outline. Using the formulation of Eq. (2), we need to estimate the number of activations the agent needs to receive/sample before an induced set of $p ( p + 1 ) / 2$ many rank-1 linearly independent matrices are found. To estimate this, first we generalize the construction of the set B from the proof of Theorem 2 to

$$
\widehat {U} _ {g} = \left\{\lambda_ {i} ^ {2} e _ {i} ^ {\otimes 2}: i \in [ p ] \right\} \cup \left\{\left(\lambda_ {i j i} e _ {i} + \lambda_ {i j j} e _ {j}\right) ^ {\otimes 2}: i <   j \in [ p ] \right\}
$$

We then analyze a design matrix M of rank-1 matrices from sampled activations and compute the probability of finding columns with entries semantically similar to those in $\widehat { U } _ { g } .$ , ensuring a non-trivial determinant. The quantity $p _ { \mathsf { S } }$ is the probability that a pattern of these columns is sampled with sparsity at most s. The final complexity bound is derived using the application of Hoeffding’s inequality and a simplification via Sterling’s approximation. □

# 6. Experimental Setup

We empirically validate our theoretical framework for learning feature matrices. Our experiments examine different feedback mechanisms and teaching strategies across both synthetic tasks and large-scale neural networks. In the following, we discuss our experimental setup in detail.

Feedback Methods: We evaluate four feedback mechanisms: (1) Eigendecomposition uses Lemma 3 to construct feedback based on $\Phi ^ { \prime } \mathbf { s }$ low rank structure, (2) Sparse Constructive builds 2-sparse feedbacks using the basis in Eq. (4), (3) Random Sampling generates feedbacks spanning ${ \mathcal { O } } _ { \Phi ^ { * } }$ from a Lebesgue distribution, and (4) Sparse Sampling creates feedbacks using s-sparse samples drawn from a sparse distribution (see Definition 1).

Teaching Agent: We implement a teaching agent with access to the target feature matrix to enable numerical analysis.

# Algorithm 3 Optimization via Gradient Descent

1. Given a dictionary $U \in \mathbb { R } ^ { p \times r }$ , minimize the loss ${ \mathcal { L } } ( U ) : =$ $\mathcal { L } _ { \mathrm { M S E } } ( U ) + \mathcal { L } _ { \mathrm { r e g } } ( U )$ : where MSE loss and regularization term are:

$$
\mathcal {L} _ {\mathrm{MSE}} (U) = \frac {1}{| B |} \sum_ {i \in B} (\| U \cdot u _ {i} \| ^ {2} - c _ {i} \| U \cdot y \| ^ {2}) ^ {2}, \mathcal {L} _ {\mathrm{reg}} (U) = \lambda \| U \| _ {F} ^ {2}
$$

where B represents the batch of samples, $\lambda = 1 0 ^ { - 4 }$ is the regularization coefficient, and $y = e _ { 1 }$ is the fixed unit vector.

2. For each batch containing indices i, values v, and targets c:

(a) Construct sparse vectors ui using $( i , v )$ pairs   
(b) Compute projections: $U ^ { \top } u _ { i }$ and $U ^ { \top } { \mathfrak { z } }$ y where $y = e _ { 1 }$   
(c) Calculate residuals: $r _ { i } = \| U ^ { \top } u _ { i } \| ^ { 2 } - c _ { i } \| U ^ { \top } y \| ^ { 2 }$

3. Update U using Adam optimizer with gradient clipping

4. Enforce fixed entries in U after each update $( U [ 0 , 0 ] = 1 )$ is enforced to be 1.)

The agent constructs either specific basis vectors or receives activations from distributions (Lebesgue or Sparse) based on the chosen feedback method. For problems with small dimensions, we utilize the cvxpy package to solve constraints of the form $\{ \alpha \alpha ^ { \top } - y y ^ { \top } \}$ . When handling larger dimensional features (5000 × 5000), where constraints scale to millions $( p ( p + 1 ) / 2 \approx 1 2 . 5 M )$ , we employ batch-wise gradient descent for matrix regression.

Features via RFM: RFM (Radhakrishnan et al., 2024) considers a trainable kernel $K _ { \Phi } ~ : ~ \mathcal { X } \times \mathcal { X } \ \to ~ \mathbb { R }$ corresponding to a symmetric, PSD matrix Φ. At each iteration, the matrix $\Phi _ { t }$ is updated for the classifier fΦ(z) = Pyi∈Dtrain a $\begin{array} { r c l } { f _ { \Phi } ( z ) } & { = } & { \sum _ { y _ { i } \in \mathcal D _ { \mathrm { t r a i n } } } a _ { i } K _ { \Phi } ( y _ { i } , z ) } \end{array}$ as follows: $\begin{array} { r l } { \Phi _ { t + 1 } } & { { } = } \end{array}$ Pz∈Dtrain ∂z ${ \sum _ { z \in { \mathcal { D } } _ { \mathrm { t r a i n } } } \left( \frac { \partial f _ { \Phi _ { t } } } { \partial z } \right) \left( \frac { \partial f _ { \Phi _ { t } } } { \partial z } \right) ^ { \top } }$  ∂fΦt . We train target functions corresponding to monomials over samples in $\mathbb { R } ^ { 1 0 }$ using 4000 training samples. The feature matrix $\Phi _ { t }$ obtained after t iterations is used as ground truth against learning with feedbacks. Plots are shown in Fig. 1 and Fig. 2.

SAE features of Large-Scale Models: We analyze dictionaries from trained sparse autoencoders on Pythia-70M (Biderman et al., 2023) (see Appendix C) and Board Game Models (Karvonen et al., 2024), with dictionary dimensions of 32k × 512 and $4 0 9 6 \times 5 1 2$ , respectively. We use the dictionaries corresponding to the SAEs trained for various MLP layers of Board Games models: ChessGPT and OthelloGPT considered in (Karvonen et al., 2024), with dimension 4096 × 512. Note that $p ( p + 1 ) / 2 \approx 8 . 3 M$ . For the experiments, we use 3-sparsity on uniform sparse distributions. We present the plots for ChessGPT in Fig. 3 for different feedback methods. Additionally, we provide a table showing the Pearson Correlation Coefficient between the learned feature matrix and the target Φ∗ in Table 3b.

Memory-efficient constraint storage The high dimensionality of model dictionaries makes storing complete activation indices for each feature prohibitively memoryintensive. We address this by enforcing constant sparsity constraints, limiting activations to a maximum sparsity of 3. This constraint enables efficient storage of largedimensional arrays while preserving the essential characteristics of the features.

Computational optimization To efficiently handle constraint satisfaction at scale, we reformulate the problem as a matrix regression task, as detailed in Algorithm 3. The learner maintains a low-rank decomposition of the feature matrix Φ, assuming $\Phi = U U ^ { \top }$ , where U represents the learned dictionary. This formulation allows for efficient batch-wise optimization over the constraint set while maintaining feasible memory requirements.

Since there could be numerical issues in computation for these large dictionaries, to compare the learnt dictionaries, we compute the Pearson Correlation Coefficient (PCC) of the trained feature matrix $\Phi ^ { \prime }$ with the target matrix $\Phi ^ { * }$ to show their closeness.

$$
\rho (\boldsymbol {\Phi} ^ {\prime}, \boldsymbol {\Phi} ^ {*}) = \frac {\sum_ {i , j} (\boldsymbol {\Phi} _ {i j} ^ {\prime} - \bar {\boldsymbol {\Phi}} ^ {\prime}) (\boldsymbol {\Phi} _ {i j} ^ {*} - \bar {\boldsymbol {\Phi}} ^ {*})}{\sqrt {\sum_ {i , j} (\boldsymbol {\Phi} _ {i j} ^ {\prime} - \bar {\boldsymbol {\Phi}} ^ {\prime}) ^ {2}} \sqrt {\sum_ {i , j} (\boldsymbol {\Phi} _ {i j} ^ {*} - \bar {\boldsymbol {\Phi}} ^ {*}) ^ {2}}}.
$$

where Φ¯ denotes the mean of all elements in the matrice Φ. Note that the highest value of ρ is 1.

# 7. Discussion

# 7.1. Limitations and Future Work

The similarity–based feature-learning framework has some major limitations: the learner observes features only up to a normal transformation, so except under strong coherence assumptions (Lemma 1)—full recovery of the underlying dictionary remains open. A natural next step is to relax exact feature equivalence and ask instead for an ε-accurate approximation in Frobenius norm. The complexity bounds derived here already translate to the classical statisticallearning setting, but an intriguing open question is whether the gap between these bounds and practical sample requirements can be tightened, perhaps by exploiting the structural insights developed in this work.

# 7.2. Conclusion

Our theoretical bounds reveal that recovering the feature dictionary of a network layer (or a trained SAE) demands at least quadratic sample–complexity in the ambient dimension, which applies across standard settings, e.g., i.i.d. learning, active learning, or machine teaching. This establishes an expressiveness-versus-recoverability trade-off: the more complex or high-dimensional the dictionary, the more feedback/data is required. The quadratic scaling can, however, be reduced under additional structure—e.g., low-rank assumptions—suggesting that leveraging such structure is essential for efficiency. Empirically, we observe that recovery indeed becomes harder in higher dimensions, while incorporating dimensionality-reduction techniques substantially improves performance, motivating future work along these lines. Our results complement the Neural Feature Ansatz (Radhakrishnan et al. (2024)) by clarifying when efficient feature recovery is possible: if task-relevant directions lie in a low-dimensional subspace, the required feedback can be sharply reduced. This insight also informs modeldistillation, suggesting that smaller students can inherit features efficiently when such a low-rank structure is present.

# Impact Statement

“This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.”

# Acknowledgments

Author thanks anonymous reviewers for insightful reviews which helped revise the work in a better context. Author thanks the National Science Foundation for support under grant IIS-2211386 in the duration of this project. Author thanks Sanjoy Dasgupta (UCSD) for helping to develop the preliminary ideas of the work. Author thanks Geelon So (UCSD) for many extended discussions on the project. Author also thanks Mikhail (Misha) Belkin (UCSD) and Enric Boix-Adserà (MIT) for helpful discussion during a visit to the Simons Insitute (UC Berkeley). The general idea of writing was developed while the author was visiting the Simons Institute (UC Berkeley) for a workshop for which the travel was supported by the National Science Foundation (NSF) and the Simons Foundation for the Collaboration on the Theoretical Foundations of Deep Learning through awards DMS-2031883 and #814639.

# References

E. Abbe, E. Boix-Adsera, and T. Misiakiewicz. The mergedstaircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In Conference on Learning Theory,

pages 4782–4887. PMLR, 2022.   
A. Agarwal, A. Anandkumar, P. Jain, P. Netrapalli, and R. Tandon. Learning sparsely used overcomplete dictionaries. In M. F. Balcan, V. Feldman, and C. Szepesvári, editors, Proceedings of The 27th Conference on Learning Theory, volume 35 of Proceedings of Machine Learning Research, pages 123–137, Barcelona, Spain, 13–15 Jun 2014. PMLR. URL https://proceedings.mlr. press/v35/agarwal14a.html.   
S. Arora, R. Ge, and A. Moitra. New algorithms for learning incoherent and overcomplete dictionaries. In Annual Conference Computational Learning Theory, 2013. URL https://api.semanticscholar. org/CorpusID:6978132.   
S. Arora, Y. Li, Y. Liang, T. Ma, and A. Risteski. A latent variable model approach to PMI-based word embeddings. Transactions of the Association for Computational Linguistics, 4:385–399, 2016. doi: 10.1162/tacl\_ a\_00106. URL https://aclanthology.org/ Q16-1028/.   
S. Arora, Y. Li, Y. Liang, T. Ma, and A. Risteski. Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics, 6:483–495, 2018.   
J. Ba, M. A. Erdogdu, T. Suzuki, Z. Wang, D. Wu, and G. Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. arXiv preprint arXiv:2205.01445, 2022a.   
J. Ba, M. A. Erdogdu, T. Suzuki, Z. Wang, D. Wu, and G. Yang. High-dimensional asymptotics of feature learning: How one gradient step improves the representation. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022b. URL https://openreview.net/forum? id=akddwRG6EGi.   
A. Bellet, A. Habrard, and M. Sebban. Metric learning, volume 30 of Synth. Lect. Artif. Intell. Mach. Learn. San Rafael, CA: Morgan & Claypool Publishers, 2015. ISBN 978-1-62705-365-5; 978-1-627-05366-2. doi: 10.2200/ S00626ED1V01Y201501AIM030.   
S. Biderman, H. Schoelkopf, Q. Anthony, H. Bradley, K. O’Brien, E. Hallahan, M. A. Khan, S. Purohit, U. S. Prashanth, E. Raff, A. Skowron, L. Sutawika, and O. Van Der Wal. Pythia: a suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

B. Bordelon and C. Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. Journal of Statistical Mechanics: Theory and Experiment, 2023, 2022a. URL https: //api.semanticscholar.org/CorpusID: 248887466.   
B. Bordelon and C. Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 32240–32256. Curran Associates, Inc., 2022b.   
T. Bricken, A. Templeton, J. Batson, B. Chen, A. Jermyn, T. Conerly, N. Turner, C. Anil, C. Denison, A. Askell, R. Lasenby, Y. Wu, S. Kravec, N. Schiefer, T. Maxwell, N. Joseph, Z. Hatfield-Dodds, A. Tamkin, K. Nguyen, B. McLean, J. E. Burke, T. Hume, S. Carter, T. Henighan, and C. Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.   
C. Burns, H. Ye, D. Klein, and J. Steinhardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations (ICLR), 2023.   
Y. Chen, Y. Chi, and A. J. Goldsmith. Exact and stable covariance estimation from quadratic sampling via convex programming. IEEE Transactions on Information Theory, 61:4034–4059, 2013. URL https://api. semanticscholar.org/CorpusID:210337.   
A. Damian, J. Lee, and M. Soltanolkotabi. Neural networks can learn representations with gradient descent. In Conference on Learning Theory, pages 5413–5452. PMLR, 2022.   
F. Doshi-Velez and B. Kim. Towards a rigorous science of interpretable machine learning, 2017. URL https: //arxiv.org/abs/1702.08608.   
N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. Mc-Candlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition. Transformer Circuits Thread, 2022. https://transformercircuits.pub/2022/toy\_model/index.html.   
M. Faruqui, Y. Tsvetkov, D. Yogatama, C. Dyer, and N. A. Smith. Sparse overcomplete word vector representations. arXiv preprint arXiv:1506.02004, 2015.

A. Fawzi, M. Balog, A. Huang, T. Hubert, B. Romera-Paredes, M. Barekatain, A. Novikov, F. J. R. Ruiz, J. Schrittwieser, G. Swirszcz, D. Silver, D. Hassabis, and P. Kohli. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature, 610(7930):47–53, Oct. 2022. ISSN 1476-4687. doi: 10.1038/s41586-022-05172-4. URL https://doi. org/10.1038/s41586-022-05172-4.   
T. Fel, A. Picard, L. Béthune, T. Boissin, D. Vigouroux, J. Colin, R. Cadène, and T. Serre. Craft: Concept recursive activation factorization for explainability. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2711–2721, June 2023.   
R. A. Fisher. The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2):179–188, 1936. doi: https://doi.org/10.1111/j.1469-1809.1936.tb02137. x.   
R. Gribonval and K. Schnass. Dictionary identification—sparse matrix-factorization via ℓ1 -minimization. IEEE Transactions on Information Theory, 56(7):3523– 3539, 2010. doi: 10.1109/TIT.2010.2048466.   
R. Gribonval, R. Jenatton, and F. R. Bach. Sparse and spurious: Dictionary learning with noise and outliers. IEEE Transactions on Information Theory, 61:6298–6319, 2014. URL https://api.semanticscholar. org/CorpusID:217787222.   
R. Gribonval, R. Jenatton, and F. Bach. Sparse and spurious: Dictionary learning with noise and outliers. IEEE Transactions on Information Theory, 61(11):6298–6319, 2015. doi: 10.1109/TIT.2015.2472522.   
I. T. Jolliffe. Principal Component Analysis. Springer-Verlag, New York, 1986.   
A. Karvonen, B. Wright, C. Rager, R. Angell, J. Brinkmann, L. R. Smith, C. M. Verdun, D. Bau, and S. Marks. Measuring progress in dictionary learning for language model interpretability with board game models. In The Thirtyeighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=SCEdoGghcw.   
M. Kleindessner and U. von Luxburg. Kernel functions based on triplet comparisons. In Neural Information Processing Systems, 2016.   
B. Kulis. Metric learning: A survey. Foundations and Trends® in Machine Learning, 5(4):287–364, 2013. ISSN 1935-8237. doi: 10.1561/2200000019. URL http: //dx.doi.org/10.1561/2200000019.

A. Kumar and S. Dasgupta. Learning smooth distance functions via queries, 2024. URL https://arxiv.org/ abs/2412.01290.   
A. Kumar, Y. Chen, and A. Singla. Teaching via best-case counterexamples in the learning-with-equivalencequeries paradigm. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 26897–26910. Curran Associates, Inc., 2021. URL https://proceedings.neurips. cc/paper\_files/paper/2021/file/ e22dd5dabde45eda5a1a67772c8e25dd-Paper. pdf.   
X. Li and V. Voroninski. Sparse signal recovery from quadratic measurements via convex programming. SIAM Journal on Mathematical Analysis, 45(5):3019–3033, 2013. doi: 10.1137/120893707. URL https://doi. org/10.1137/120893707.   
S. Mallat and Z. Zhang. Matching pursuits with timefrequency dictionaries. IEEE Transactions on Signal Processing, 41(12):3397–3415, 1993. doi: 10.1109/78. 258082.   
S. Marks and M. Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets, 2024. URL https://arxiv. org/abs/2310.06824.   
S. Marks, A. Karvonen, and A. Mueller. dictionary\_learning. https://github.com/ saprmarks/dictionary\_learning, 2024a.   
S. Marks, C. Rager, E. J. Michaud, Y. Belinkov, D. Bau, and A. Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models, 2024b. URL https://arxiv.org/abs/2403. 19647.   
B. Mason, L. P. Jain, and R. D. Nowak. Learning lowdimensional metrics. In Neural Information Processing Systems, 2017.   
T. Mikolov, W.-t. Yih, and G. Zweig. Linguistic regularities in continuous space word representations. In L. Vanderwende, H. Daumé III, and K. Kirchhoff, editors, Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751, Atlanta, Georgia, June 2013. Association for Computational Linguistics. URL https://aclanthology. org/N13-1090/.

C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, and S. Carter. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.   
B. A. Olshausen and D. J. Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision Research, 37(23):3311–3325, 1997. ISSN 0042-6989. doi: https://doi.org/10.1016/S0042-6989(97)00169-7. URL https://www.sciencedirect.com/ science/article/pii/S0042698997001697.   
A. Radhakrishnan, D. Beaglehole, P. Pandit, and M. Belkin. Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science, 383(6690):1461–1467, 2024. doi: 10.1126/science. adi5639. URL https://www.science.org/doi/ abs/10.1126/science.adi5639.   
B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. R. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, P. Kohli, and A. Fawzi. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, Jan. 2024. ISSN 1476-4687. doi: 10.1038/ s41586-023-06924-6. URL https://doi.org/10. 1038/s41586-023-06924-6.   
M. Schultz and T. Joachims. Learning a distance metric from relative comparisons. In S. Thrun, L. Saul, and B. Schölkopf, editors, Advances in Neural Information Processing Systems, volume 16. MIT Press, 2003.   
L. Sharkey, B. Chughtai, J. Batson, J. Lindsey, J. Wu, L. Bushnaq, N. Goldowsky-Dill, S. Heimersheim, A. Ortega, J. Bloom, S. Biderman, A. Garriga-Alonso, A. Conmy, N. Nanda, J. Rumbelow, M. Wattenberg, N. Schoots, J. Miller, E. J. Michaud, S. Casper, M. Tegmark, W. Saunders, D. Bau, E. Todd, A. Geiger, M. Geva, J. Hoogland, D. Murfet, and T. McGrath. Open problems in mechanistic interpretability, 2025. URL https://arxiv.org/abs/2501.16496.   
Z. Shi, J. Wei, and Y. Lian. A theoretical analysis on feature learning in neural networks: Emergence from inputs and advantage over fixed features. In International Conference on Learning Representations, 2022.   
K. Q. Weinberger and L. K. Saul. Distance metric learning for large margin nearest neighbor classification. J. Mach. Learn. Res., 10:207–244, June 2009. ISSN 1532-4435.   
E. P. Xing, A. Ng, M. I. Jordan, and S. J. Russell. Distance metric learning with application to clustering with sideinformation. In Neural Information Processing Systems, 2002.

G. Yang and E. J. Hu. Tensor Programs IV: Feature learning in infinite-width neural networks. In International Conference on Machine Learning, 2021a.   
G. Yang and E. J. Hu. Tensor programs iv: Feature learning in infinite-width neural networks. In M. Meila and T. Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11727–11737. PMLR, 18–24 Jul 2021b. URL https://proceedings.mlr. press/v139/yang21c.html.   
Z. Yun, Y. Chen, B. A. Olshausen, and Y. LeCun. Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors. arXiv preprint arXiv:2103.15949, 2021.   
J. Zhang, Y. Chen, B. Cheung, and B. A. Olshausen. Word embedding visualization via dictionary learning. arXiv preprint arXiv:1910.03833, 2019.   
L. Zhu, C. Liu, A. Radhakrishnan, and M. Belkin. Quadratic models for understanding neural network dynamics. arXiv preprint arXiv:2205.11787, 2022.   
X. Zhu, A. Singla, S. Zilles, and A. N. Rafferty. An overview of machine teaching, 2018. URL https://arxiv. org/abs/1801.05927.

# A. Table of Contents

Here, we provide the table of contents for the appendix of the supplementary.

- Appendix C provides supplementary experimental results validating our theoretical findings.   
- Appendix B provides a comprehensive table of additional notations used throughout the paper and supplementary material.   
- Appendix D contains the proof for Lemma 1, establishing conditions for recovering orthogonal representations.   
- Appendix E completes the proof of Proposition 1, establishing a worst-case lower bound on feedback complexity in the constructive setting.   
- Appendix F presents the proof for the upper bound in Theorem 1 for low-rank feature matrices.   
- Appendix G establishes the proof for the lower bound in Theorem 1 for low-rank feature matrices.   
- Appendix H details the proof of Theorem 3 which asserts tight bounds on feedback complexity for general sampled activations.   
- Appendix I demonstrates the proof of Theorem 4 establishing an upper bound on the feedback complexity for sparse sampled activations.

# B. Notations

Here we provide the glossary of notations followed in the supplementary material.

<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\alpha, \beta, x, y, z$ </td><td>Activations</td></tr><tr><td>col( $\Phi$ )</td><td>Set of columns of matrix  $\Phi$ </td></tr><tr><td> $\mathcal{D}, \mathcal{D}_{\text{sparse}}$ </td><td>Distributions over activations</td></tr><tr><td> $d$ </td><td>Dimension of ground-truth sample space</td></tr><tr><td> $\mathsf{D}$ </td><td>Dictionary matrix</td></tr><tr><td> $\gamma, \lambda, \gamma_i, \lambda_i$ </td><td>Eigenvalues of a matrix</td></tr><tr><td> $\langle \Phi', \Phi \rangle$ </td><td>Element-wise product (inner product) of matrices</td></tr><tr><td> $\text{Ker}(\Phi)$ </td><td>Kernel of matrix  $\Phi$ </td></tr><tr><td> $\mu_i, u_i, v_i$ </td><td>Eigenvectors (orthogonal vectors)</td></tr><tr><td>null( $\Phi$ )</td><td>Null set of matrix  $\Phi$ </td></tr><tr><td> $\mathcal{O}_{\Phi^*}$ </td><td>Orthogonal complement of  $\Phi^*$  in  $\text{Sym}(\mathbb{R}^{p \times p})$ </td></tr><tr><td> $p$ </td><td>Dimension of representation space</td></tr><tr><td> $\Phi, \Sigma$ </td><td>Feature matrix</td></tr><tr><td> $\Phi_{ij}$ </td><td>Entry at  $i$ th row and  $j$ th column of  $\Phi$ </td></tr><tr><td> $\Phi^*$ </td><td>Target feature matrix</td></tr><tr><td> $r$ </td><td>Rank of a feature matrix</td></tr><tr><td> $\text{Sym}(\mathbb{R}^{p \times p})$ </td><td>Space of symmetric matrices</td></tr><tr><td> $\text{Sym}_+( \mathbb{R}^{p \times p})$ </td><td>Space of PSD, symmetric matrices</td></tr><tr><td> $\text{VS}(\mathcal{F}, \mathcal{M}_{\mathsf{F}})$ </td><td>Version space of  $\mathcal{M}_{\mathsf{F}}$  wrt feedback set  $\mathcal{F}$ </td></tr><tr><td> $V_{[r]}$ </td><td>The set  $\{v_1, v_2, \ldots, v_r\}$ </td></tr><tr><td> $V_{[p-r]}$ </td><td>The set  $\{v_{r+1}, \ldots, v_p\}$ </td></tr><tr><td> $V_{[p]}$ </td><td>Complete orthonormal basis  $\{v_1, v_2, \ldots, v_p\}$ </td></tr><tr><td> $\mathcal{V} \subset \mathbb{R}^p$ </td><td>Activation/Representation space</td></tr><tr><td> $\mathcal{X} \subset \mathbb{R}^d$ </td><td>Ground truth sample space</td></tr></table>

# C. Additional Experiments

In Section 6, we provided details of our experimental setup. In this appendix, we will show the results for some additional experiments: 1) Large-scale SAEs trained on Pythia-70M (Biderman et al., 2023), and 2) extensive experimental results (in Appendix C.1) on a synthetic task as considered in Fig. 1 and Fig. 2.

Dictionary features of Pythia-70M We use the publicly available repository for dictionary learning via sparse autoencoders on neural network activations (Marks et al., 2024a). We consider the dictionaries trained for Pythia-70M (Biderman et al., 2023) (a general-purpose LLM trained on publicly available datasets). We retrieve the corresponding autoencoders for the attention output layers, which have dimensions $3 2 7 6 8 \times 5 1 2$ . Note that $p ( p + 1 ) / 2 \approx , 5 1 2 M$ . For the experiments, we use 3-sparsity on uniform sparse distributions. We present the plots for ChessGPT in two parts in Fig. 4 and Fig. 5 for different feedback methods.

![](images/833b0923f0bcc6a98e3ad1744765a12a7ed63d1dfb7c65975cad2355effa9972.jpg)

![](images/518f5a1c2c6a30974ea8e2966a643f0fa6dc3e2627b5478cbbb6708c47dcd527.jpg)

<details>
<summary>heatmap</summary>

Feedback: Eigendecomposition
| X | Y | Value |
|---|---|---|
| 0 | 99 | 96 |
| 3 | 93 | 90 |
| 6 | 87 | 84 |
| 9 | 81 | 78 |
| 12 | 75 | 72 |
| 15 | 69 | 69 |
| 18 | 63 | 60 |
| 21 | 57 | 54 |
| 24 | 51 | 48 |
| 27 | 45 | 42 |
| 30 | 39 | 36 |
| 33 | 33 | 30 |
| 36 | 27 | 24 |
| 39 | 21 | 18 |
| 42 | 15 | 12 |
| 45 | 9 | 0 |
| 48 | 5 | -2 |
| 51 | -1 | -4 |
| 54 | -3 | -6 |
| 57 | -5 | -8 |
| 60 | -7 | -10 |
| 63 | -9 | -12 |
| 66 | -11 | -14 |
| 69 | -13 | -16 |
| 72 | -15 | -18 |
| 75 | -17 | -20 |
| 78 | -19 | -22 |
| 81 | -21 | -24 |
| 84 | -23 | -26 |
| 87 | -25 | -28 |
| 90 | -27 | -30 |
| 93 | -29 | -32 |
| 96 | -31 | -34 |
| 99 | -33 | -36 |
Legend: 
- Color scale: 
- Dark purple = 
- Light purple = 
- Light blue = 
- Dark green = 
- Dark red = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark red = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark green = 
- Dark blue = 
- Dark yellow = 
- Dark red = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark teal = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark pink = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Dark magenta = 
- Black line (PCC) is not explicitly labeled in the image. The chart displays a color-coded legend for the data series. The legend indicates that the color corresponds to the value of the measured variable (e.g., PCC or threshold).
</details>

Figure 4: Feature learning on a subsampled dictionary of dimension 4500 × 512 of SAE trained for Pythia-70M. Theorem 1 states that Eigendecompostion method requires 135316 constructive feedback. After a few 100 iterations of gradient descent as shown in Algorithm 3, a PCC of 93% is achieved on ground truth. For visualization, only the first 100 dimensions are used.

![](images/8c71676250c7aa4bde8370a1b23247e314bf97b636133aab641dbb39cae761d7.jpg)

![](images/61b4a4d4e47a9bac9535d2fe0a8d52349db440df0a8c82192e8d629c08075d33.jpg)

<details>
<summary>scatter</summary>

| x  | y  |
|----|----|
| 0  | 99 |
| 1  | 98 |
| 2  | 97 |
| 3  | 96 |
| 4  | 95 |
| 5  | 94 |
| 6  | 93 |
| 7  | 92 |
| 8  | 91 |
| 9  | 90 |
| 10 | 89 |
| 11 | 88 |
| 12 | 87 |
| 13 | 86 |
| 14 | 85 |
| 15 | 84 |
| 16 | 83 |
| 17 | 82 |
| 18 | 81 |
| 19 | 80 |
| 20 | 79 |
| 21 | 78 |
| 22 | 77 |
| 23 | 76 |
| 24 | 75 |
| 25 | 74 |
| 26 | 73 |
| 27 | 72 |
| 28 | 71 |
| 29 | 70 |
| 30 | 69 |
| 31 | 68 |
| 32 | 67 |
| 33 | 66 |
| 34 | 65 |
| 35 | 64 |
| 36 | 63 |
| 37 | 62 |
| 38 | 61 |
| 39 | 60 |
| 40 | 59 |
| 41 | 58 |
| 42 | 57 |
| 43 | 56 |
| 44 | 55 |
| 45 | 54 |
| 46 | 53 |
| 47 | 52 |
| 48 | 51 |
| 49 | 50 |
| 50 | 49 |
| 51 | 48 |
| 52 | 47 |
| 53 | 46 |
| 54 | 45 |
| 55 | 44 |
| 56 | 43 |
| 57 | 42 |
| 58 | 41 |
| 59 | 40 |
| 60 | 39 |
| 61 | 38 |
| 62 | 37 |
| 63 | 36 |
| 64 | 35 |
| 65 | 34 |
| 66 | 33 |
| 67 | 32 |
| 68 | 31 |
| 69 | 30 |
| 70 | 29 |
| 71 | 28 |
| 72 | 27 |
| 73 | 26 |
| 74 | 25 |
| 75 | 24 |
| 76 | 23 |
| 77 | 22 |
| 78 | 21 |
| 79 | 20 |
| 80 | 19 |
| 81 | 18 |
| 82 | 17 |
| 83 | 16 |
| 84 | 15 |
| 85 | 14 |
| 86 | 13 |
| 87 | 12 |
| 88 | 11 |
| 89 | 10 |
| 90 |   |
|   |   |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |    |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    |   |
    | nan
</details>

![](images/c1d0d544d2db97dc0260c477ac951dfb0f94517215d11b882977eeddd76425a0.jpg)

<details>
<summary>heatmap</summary>

Feedback: Sparse Sampling
| X | Y | Value |
|---|---|---|
| 0 | 0 | 99.96 |
| 3 | 3 | 93.90 |
| 6 | 6 | 87.81 |
| 9 | 9 | 78.75 |
| 12 | 12 | 72.72 |
| 15 | 15 | 69.69 |
| 18 | 18 | 64.66 |
| 21 | 21 | 59.53 |
| 24 | 24 | 54.48 |
| 27 | 27 | 50.54 |
| 30 | 30 | 45.48 |
| 33 | 33 | 40.54 |
| 36 | 36 | 35.48 |
| 39 | 39 | 30.54 |
| 42 | 42 | 25.48 |
| 45 | 45 | 20.54 |
| 48 | 48 | 15.48 |
| 51 | 51 | 10.54 |
| 54 | 54 | 5.48 |
| 57 | 57 | 0.00 |
| 60 | 60 | -0.54 |
| 63 | 63 | -1.00 |
| 66 | 66 | -1.56 |
| 69 | 69 | -2.12 |
| 72 | 72 | -2.78 |
| 75 | 75 | -3.44 |
| 78 | 78 | -4.00 |
| 81 | 81 | -4.66 |
| 84 | 84 | -5.32 |
| 87 | 87 | -5.98 |
| 90 | 90 | -6.64 |
| 93 | 93 | -7.30 |
| 96 | 96 | -7.96 |
| 99 | 99 | -8.62 |
</details>

![](images/b00feec6b228f9708f5f58fbecc4874be8e534efd8f2dfa3a21a5536a0be1e68.jpg)

<details>
<summary>scatter</summary>

| x  | y  |
|----|----|
| 0  | 99 |
| 3  | 98 |
| 6  | 97 |
| 9  | 96 |
| 12 | 95 |
| 15 | 94 |
| 18 | 93 |
| 21 | 92 |
| 24 | 91 |
| 27 | 90 |
| 30 | 89 |
| 33 | 88 |
| 36 | 87 |
| 39 | 86 |
| 42 | 85 |
| 45 | 84 |
| 48 | 83 |
| 51 | 82 |
| 54 | 81 |
| 57 | 80 |
| 60 | 79 |
| 63 | 78 |
| 66 | 77 |
| 69 | 76 |
| 72 | 75 |
| 75 | 74 |
| 78 | 73 |
| 81 | 72 |
| 84 | 71 |
| 87 | 70 |
| 90 | 69 |
| 93 | 68 |
| 96 | 67 |
| 99 | 66 |
</details>

![](images/1fefbb35f50a82b54b15dc3789f88eae398751d33b4dd42ec084c0cf1a0e0564.jpg)

![](images/b196aa9242815b52d7ffaf5eeffbe45e1ff07391d7e943a5bb83a277aeaab01b.jpg)

<details>
<summary>heatmap</summary>

Feedback: Sparse Sampling
| X | Y | Value |
|---|---|---|
| 99 | 0 | 0 |
| 98 | 1 | 0 |
| 97 | 2 | 0 |
| 96 | 3 | 0 |
| 95 | 4 | 0 |
| 94 | 5 | 0 |
| 93 | 6 | 0 |
| 92 | 7 | 0 |
| 91 | 8 | 0 |
| 90 | 9 | 0 |
| 89 | 10 | 0 |
| 88 | 11 | 0 |
| 87 | 12 | 0 |
| 86 | 13 | 0 |
| 85 | 14 | 0 |
| 84 | 15 | 0 |
| 83 | 16 | 0 |
| 82 | 17 | 0 |
| 81 | 18 | 0 |
| 80 | 19 | 0 |
| 79 | 20 | 0 |
| 78 | 21 | 0 |
| 77 | 22 | 0 |
| 76 | 23 | 0 |
| 75 | 24 | 0 |
| 74 | 25 | 0 |
| 73 | 26 | 0 |
| 72 | 27 | 0 |
| 71 | 28 | 0 |
| 70 | 29 | 0 |
| 69 | 30 | 0 |
| 68 | 31 | 0 |
| 67 | 32 | 0 |
| 66 | 33 | 0 |
| 65 | 34 | 0 |
| 64 | 35 | 0 |
| 63 | 36 | 0 |
| 62 | 37 | 0 |
| 61 | 38 | 0 |
| 60 | 39 | 0 |
| 59 | 40 | 0 |
| 58 | 41 | 0 |
| 57 | 42 | 0 |
| 56 | 43 | 0 |
| 55 | 44 | 0 |
| 54 | 45 | 0 |
| 53 | 46 | 0 |
| 52 | 47 | 0 |
| 51 | 48 | 0 |
| 50 | 49 | 0 |
| 49 | 50 | 0 |
| 48 | 51 | 0 |
| 47 | 52 | 0 |
| 46 | 53 | 0 |
| 45 | 54 | 0 |
| 44 | 55 | 0 |
| 43 | 56 | 0 |
| 42 | 57 | 0 |
| 41 | 58 | 0 |
| 40 | 59 | 0 |
| 39 | 60 | 0 |
| 38 | 61 | 0 |
| 37 | 62 | 0 |
| 36 | 63 | 0 |
| 35 | 64 | 0 |
| 34 | 65 | 0 |
| 33 | 66 | 0 |
| 32 | 67 | 0 |
| 31 | 68 | 0 |
| 30 | 69 | -1 |
| ... (repeated) for all rows and columns are integers from -1 to +1. The values in the matrix represent the magnitude of the matrix at each point. The PCC value is calculated as P = .7716, with Feedback = .2000000. The color bar on the right indicates the magnitude of the matrix. The chart is a scatter plot with a color scale ranging from -1 to +1.
</details>

Figure 5: Sparse sampling for Pythia-70M: Dimension of feature matrix: $3 2 7 6 8 \times 5 1 2$ and the rank is 215. Plots for varying feedback complexity sizes. Note that $p ( p + 1 ) / 2 \approx 5 1 2  { \mathrm { M } }$ . We run experiments with 3-sparse activations for uniform sparse distributions. The Pearson Correlation Coefficient (PCC) to feedback size (PCC, Feedback size) improves as follows: (200k, .0242), (2M, .38), (5M, .54), (10M, .65), and (20M, .77).

# C.1. Verification of theoretical results on a synthetic task

To validate our theoretical results, we compare the upper bounds derived in Theorem 1–4 against empirical performance on a controlled synthetic task. This experiment aims to assess how tightly the theoretical feedback complexity aligns with the actual number of feedback queries required to achieve feature recovery up to linear scaling equivalence (Definition 2).

We consider a monomial regression task defined by

$$
y = f ^ {*} (\pmb {x}) = \pmb {x} _ {1} \pmb {x} _ {2} \pmb {x} _ {3} \pmb {x} _ {4} \cdot \mathbf {1} (\pmb {x} _ {5} > 0),
$$

which induces a target feature matrix $\Phi ^ { * }$ (as constructed by the Recursive Feature Machine (Radhakrishnan et al., 2024), see Section 6).

Setup. Inputs $\pmb { x } \in \mathbb { R } ^ { 1 0 }$ are sampled from a Gaussian distribution $\mathcal { N } ( 0 , 0 . 5 \mathbb { I } _ { 1 0 } )$ . We train an RFM classifier on 5000 training samples to obtain $\Phi ^ { * }$ , and the teaching agent has access to this feature matrix for generating feedback.

We evaluated the following four feedback mechanisms: Eigendecomposition, Sparse Constructive, Random Sampling, and Sparse Sampling (Section 4 and Section 5).

For each method, we report:

1. The number of feedbacks provided.   
2. The empirical mean squared error (MSE) compared to the target MSE achieved using $\Phi ^ { * }$ .   
3. The theoretical upper bound on the number of feedbacks.

Theoretical vs Empirical Observations. The target feature matrix $\Phi ^ { * }$ has rank $r = 8 .$ , and the ambient input dimension is $p = 1 0 , \mathrm { g i v i n g } p ( p + 1 ) / 2 = 5 5 $ as the total number of degrees of freedom used in the stated bounds.

• Eigendecomposition: Theoretical bound (Theorem 1) is $\frac { r ( r + 1 ) } { 2 } + p - r = 3 8$ . As shown in Figure $^ { 6 , }$ this exact number of feedbacks is sufficient to match the target MSE (mean squared error) empirically.   
• Sparse Constructive: Using 2-sparse feedbacks (Theorem 2), the theoretical bound remains 55. As illustrated in Figure 6, the empirical performance saturates at the target MSE within this bound.   
• Random Sampling: Feedback is sampled uniformly at random. We evaluate empirical performance at 20%, 30%, 50%, 70%, and 100% of the theoretical bound: 55 (as computed using Theorem 3), as shown in Figure 6. The gradual reduction in MSE confirms that the learning curve aligns well with the theoretical complexity.

Remark: Given that these are sampled runs (not averaged), in some cases, the MSE might be higher even if the feedback set is increased (implying that an increase in feedback didn’t lead to relevant independent directions). But, averaging over runs, we note that the MSE gradually reduces in MSE with the stated theoretical bound.

• Sparse Sampling: Our first experiment is for 4-sparse activations in Figure 7, where each coordinate is nonzero with probability $1 - \mu = 0 . 2$ , and the nonzero values are drawn from $\mathcal { U } ( 0 , 1 )$ . Using a success threshold of $\delta = 0 . 0 5$ , Theorem 4 yields a bound of 117 feedbacks. Figure 7 shows MSE values at multiples (30% to 2000%) of the total number of degrees of freedom (55). As expected, MSE converges to the target MSE once the feedback size reaches the theoretical threshold.

We perform several experiments with different values of $\delta , \mu ,$ and sparsity level as shown in Figure 8-11.

Remark: Since the bounds are independent of the distribution of a coordinate being non-zero, the bounds don’t change even if we use a distribution other than the uniform distribution.

![](images/e8200c0581186e2226bae4e9637f0b615e55efd7e86f2d373b62f6ca28b7b621.jpg)

Figure 6: Empirical performance for Eigendecomposition, Sparse Constructive, and Random Sampling.   
![](images/7424c91c9c66e415f22bf3257a137dffb424a54dc6b145e3be1761f793c2a0e5.jpg)

<details>
<summary>heatmap</summary>

|        | 0    | 2    | 4    | 6    | 8    |
| ------ | ---- | ---- | ---- | ---- | ---- |
| 0      | 1.0  | 0.8  | 0.6  | 0.4  | 0.2  |
| 2      | 0.8  | 0.6  | 0.4  | 0.2  | 0.0  |
| 4      | 0.6  | 0.4  | 0.2  | 0.0  | -0.2 |
| 6      | 0.4  | 0.2  | 0.0  | -0.2 | -0.4 |
| 8      | 0.2  | 0.0  | -0.2 | -0.4 | -0.6 |
</details>

MSE:0.00101869, Training points:5000

![](images/144fed4622cd004ba36c1a39235c41797388aaff37d50248972e83903314e899.jpg)  
MSE:3.74420298, Feedbacks:16,mu:0.8, Delta:0.05,spasity:4, Theory:117

![](images/42ebab57e2bd73fe6dc4439ab5955da317ade9e5f19610c692f6a6902bec58c6.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
| 2 | 0.8 | 0.6 | 0.4 | 0.2 | 0.0 |
| 4 | 0.6 | 0.4 | 0.2 | 0.0 | 0.0 |
| 6 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 |
| 8 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
</details>

MSE:28.05099124, Feedbacks:27,mu:0.8, Delta:0.05,asity:4, Theory:117

![](images/4f346cdac40e1f7c2a0ca3cc6130fc841d492fc5d456ace9c78144ee6fb26bbf.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 6 | | | | | |
| 8 | | | | | |
The heatmap visualizes a sparse sampling pattern across the grid. Values are estimated based on the color scale ranging from 0.0 to 1.0.
</details>

MSE:3.09841886, Feedbacks:38,mu:0.8, Delta:0.05,spasity:4, Theory:117

![](images/9ecc25037f55a0c31c3aefdbe215d18d20be575377d9b4197e3db52cdde6e67b.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
|-----|------|------|------|------|------|------|------|------|------|
| 0   | 1.0  |      |      |      |      |      |      |      |      |
| 1   |      |      |      |      |      |      |      |      |      |
| 2   |      |      |      |      |      |      |      |      |      |
| 3   |      |      |      |      |      |      |      |      |      |
| 4   |      |      |      |      |      |      |      |      |      |
| 5   |      |      |      |      |      |      |      |      |      |
| 6   |      |      |      |      |      |      |      |      |      |
| 7   |      |      |      |      |      |      |      |      |      |
| 8   |      |      |      |      |      |      |      |      |      |
| 9   |      |      |      |      |      |      |      |      |      |
| 10  |      |      |      |      |      |      |      |      |      |
| 11  |      |      |      |      |      |      |      |      |      |
| 12  |      |      |      |      |      |      |      |      |      |
| 13  |      |      |      |      |      |      |      |      |      |
| 14  |      |      |      |      |      |      |      |      |      |
| 15  |      |      |      |      |      |      |      |      |      |
| 16  |      |      |      |      |      |      |      |      |      |
| 17  |      |      |      |      |      |      |      |      |      |
| 18  |      |      |      |      |      |      |      |      |      |
| 19  |      |      |      |      |      |      |      |      |      |
| 20  |      |      |      |      |      |      |      |      |      |
| 21  |      |      |      |      |      |      |      |      |      |
| 22  |      |      |      |      |      |      |      |      |      |
| 23  |      |      |      |      |      |      |      |      |      |
| 24  |      |      |      |      |      |      |      |      |      |
| 25  |      |      |      |      |      |      |      |      |      |
| 26  |      |      |      |      |      |      |      |      |      |
| 27  |      |      |      |      |      |      |      |      |      |
| 28  |      |      |      |      |      |      |     nan|     nan|     nan|
| 29  |     nan|     nan|     nan|     nan|     nan|     nan|     nan|     nan|     nan|
| 30  |     nan|     nan|     nan|     nan|     nan|     nan|     nan|     nan|     nan|
| Note: The actual values in the CSV data are not provided in the code. The code generates random values for these values in the heatmap. The actual values in the heatmap will vary each time the code is run. There is no label for the data series. The actual values in the heatmap will be the result of the random number generation. The actual values in the heatmap will vary each time the code is run. However, since the code does not provide the exact values, I have indicated that the actual values are not available. Therefore, I can provide the exact values as required. However, I can provide the exact values as required. The correct output is an empty string.          The correct output is: "Random number generation" for all rows.        The correct output is: "Random number sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling" for all rows.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The correct output is: "Sparse sampling", "Sparse sampling", etc.        The error bars represent the deviation from the exact values.         The error bars represent the absolute error of the error bar.         The error bars represent the absolute error of the error bar.
</details>

MSE:0.00101873, Feedbacks:55,mu:0.8, Delta:0.05,spasity:4, Theory:117

![](images/25f6f34484e6502e4e7fb1423a0847bbdbee4a0c872b586c2b09b237e1944df7.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
| 2 | 0.8 | 0.6 | 0.4 | 0.2 | 0.0 |
| 4 | 0.6 | 0.4 | 0.2 | 0.0 | 0.0 |
| 6 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 |
| 8 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
</details>

MSE:0.00101869, Feedbacks:110,mu:0.8, Delta:0.05,spasity:4 Theory:117

![](images/0e15b8bfbc8a0c2567866748c60383e745d0d07ee8a2f01af8be1a7b739156eb.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 6 | | | | | |
| 8 | | | | | |
The heatmap visualizes a sparse sampling pattern across the grid. Values are represented by a color scale ranging from 0.0 to 1.0.
</details>

MSE:0.00101869, Feedbacks:275,mu:0.8, Delta:0.05,spasity:4, Theory:117

![](images/0ea0e9a546323ef3d56ba7f956c54951fc291b9f6156726ab9de181ac321cef2.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | 0.8 | 0.6 | 0.4 | 0.2 |
| 2 | 0.8 | 0.6 | 0.4 | 0.2 | 0.0 |
| 4 | 0.6 | 0.4 | 0.2 | 0.0 | 0.0 |
| 6 | 0.4 | 0.2 | 0.0 | 0.0 | 0.0 |
| 8 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 |
</details>

MSE:0.00101869, Feedbacks:440,mu:0.8, Delta:0.05,spasity:4, Theory:117

Figure 7: Empirical performance for the Sparse Sampling feedback mechanism.   
![](images/c14bb85977a87f7599724c581cbb20c0173fb07cdc0b034bb209b67467a91351.jpg)

<details>
<summary>heatmap</summary>

| Row | Col 1 | Col 2 | Col 3 | Col 4 | Col 5 | Col 6 | Col 7 | Col 8 |
|-----|-------|-------|-------|-------|-------|-------|-------|-------|
| 0   | 0.9   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.3   | 0.2   |
| 1   | 0.8   | 0.7   | 0.6   | 0.5   | 0.4   | 0.3   | 0.2   | 0.1   |
| 2   | 0.7   | 0.6   | 0.5   | 0.4   | 0.3   | 0.2   | 0.1   | 0.0   |
| 3   | 0.6   | 0.5   | 0.4   | 0.3   | 0.2   | 0.1   | 0.0   | -0.1  |
| 4   | 0.5   | 0.4   | 0.3   | 0.2   | 0.1   | 0.0   | -0.1  | -0.2  |
| 5   | 0.4   | 0.3   | 0.2   | 0.1   | 0.0   | -0.1  | -0.2  | -0.3  |
| 6   | 0.3   | 0.2   | 0.1   | 0.0   | -0.1  | -0.2  | -0.3  | -0.4  |
| 7   | 0.2   | 0.1   | 0.0   | -0.1  | -0.2  | -0.3  | -0.4  | -0.5  |
| 8   | 0.1   | 0.0   | -0.1  | -0.2  | -0.3  | -0.4  | -0.5  | -0.6  |
</details>

MSE:0.00105147, Training points:5000

![](images/6b5d95b1cc933944710f683fed3842a1cf886cd4366ae8051a4cf5dd55bcca7e.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 2 | 0.8 |
| 0 | 4 | 0.6 |
| 0 | 6 | 0.4 |
| 0 | 8 | 0.2 |
| 2 | 0 | 0.0 |
| 2 | 2 | 0.8 |
| 2 | 4 | 0.6 |
| 2 | 6 | 0.4 |
| 2 | 8 | 0.2 |
| 4 | 0 | 0.0 |
| 4 | 2 | 0.8 |
| 4 | 4 | 0.6 |
| 4 | 6 | 0.4 |
| 4 | 8 | 0.2 |
| 6 | 0 | 0.0 |
| 6 | 2 | 0.8 |
| 6 | 4 | 0.6 |
| 6 | 6 | 0.4 |
| 6 | 8 | 0.2 |
| 8 | 0 | 0.0 |
| 8 | 2 | 0.8 |
| 8 | 4 | 0.6 |
| 8 | 6 | 0.4 |
| 8 | 8 | 0.2 |
</details>

MSE:3.53717631, Feedbacks:16,mu:0.6, Delta:O.1,sty：4, Theory：54

![](images/9b445598c2278ce6048e7bc80bf0967941de9eebd839ea17fc0c8bdc973f9346.jpg)

<details>
<summary>heatmap</summary>

| X\Y | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| 0 | 1.0 | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 6 | | | | | |
| 8 | | | | | |
The heatmap visualizes a sparse sampling pattern across the grid. Values are estimated based on the color scale ranging from 0.0 to 1.0.
</details>

MSE:0.24345435, Feedbacks:27,mu:0.6, Delta:O.1,sparsity:4, Theory:54

![](images/c923fffc7e736c8310ceec39b8fa26a890deff599ea7b831905dc7d8349ff766.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 2 | 0.8 |
| 0 | 4 | 0.6 |
| 0 | 6 | 0.4 |
| 0 | 8 | 0.2 |
| 2 | 0 | 0.0 |
| 2 | 2 | 0.2 |
| 2 | 4 | 0.4 |
| 2 | 6 | 0.6 |
| 2 | 8 | 0.8 |
| 4 | 0 | 1.0 |
| 4 | 2 | 0.8 |
| 4 | 4 | 0.6 |
| 4 | 6 | 0.4 |
| 4 | 8 | 0.2 |
| 6 | 0 | 0.0 |
| 6 | 2 | 0.2 |
| 6 | 4 | 0.4 |
| 6 | 6 | 0.6 |
| 6 | 8 | 0.8 |
| 8 | 0 | 1.0 |
| 8 | 2 | 0.8 |
| 8 | 4 | 0.6 |
| 8 | 6 | 0.4 |
| 8 | 8 | 0.2 |
</details>

MSE:184.23201961, Feedbacks:38,mu:0.6, Delta:O.1,sprsity:4, Theory:54

![](images/0f8499a5ae761cc55c457571568bbd7bf0f2103785fa1cc6f0ddfb8831817f06.jpg)  
MSE:7.35539692, Feedbacks:55,mu:0.6, Delta:O.1,sparsity:4, Theory：54

![](images/e98462f1d3db74c484ad9dee77db65e37b74820c56241bc936a1875b828adf3e.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 2 | 0.8 |
| 0 | 4 | 0.6 |
| 0 | 6 | 0.4 |
| 0 | 8 | 0.2 |
| 2 | 0 | 0.0 |
| 2 | 2 | 0.8 |
| 2 | 4 | 0.6 |
| 2 | 6 | 0.4 |
| 2 | 8 | 0.2 |
| 4 | 0 | 0.0 |
| 4 | 2 | 0.8 |
| 4 | 4 | 0.6 |
| 4 | 6 | 0.4 |
| 4 | 8 | 0.2 |
| 6 | 0 | 0.0 |
| 6 | 2 | 0.8 |
| 6 | 4 | 0.6 |
| 6 | 6 | 0.4 |
| 6 | 8 | 0.2 |
| 8 | 0 | 0.0 |
| 8 | 2 | 0.8 |
| 8 | 4 | 0.6 |
| 8 | 6 | 0.4 |
| 8 | 8 | 0.2 |
</details>

MSE:0.00105155, Feedbacks:110,mu:0.6, Delta:0.i,sparsity:4 Theory:54

![](images/14e08a300d44a36da10652c96dfa98c2303f810a77dbbe61fb72de259e4c3dae.jpg)

<details>
<summary>heatmap</summary>

| X | Y | Value |
|---|---|---|
| 0 | 0 | 1.0 |
| 0 | 2 | 0.8 |
| 0 | 4 | 0.6 |
| 0 | 6 | 0.4 |
| 0 | 8 | 0.2 |
| 2 | 0 | 0.8 |
| 2 | 2 | 0.6 |
| 2 | 4 | 0.4 |
| 2 | 6 | 0.2 |
| 2 | 8 | 0.0 |
| 4 | 0 | 0.8 |
| 4 | 2 | 0.6 |
| 4 | 4 | 0.4 |
| 4 | 6 | 0.2 |
| 4 | 8 | 0.0 |
| 6 | 0 | 0.8 |
| 6 | 2 | 0.6 |
| 6 | 4 | 0.4 |
| 6 | 6 | 0.2 |
| 6 | 8 | 0.0 |
| 8 | 0 | 0.8 |
| 8 | 2 | 0.6 |
| 8 | 4 | 0.4 |
| 8 | 6 | 0.2 |
| 8 | 8 | 0.0 |
</details>

MSE:0.00105147, Feedbacks:275,mu:0.6, Delta:.i，pity:4 Theory:54

![](images/563dd17d8c03a495f8712c2b14b60fd6db87c39ccbb38aad039b3277bc6e4929.jpg)

<details>
<summary>heatmap</summary>

| Row | Column | Value |
|---|---|---|
| 0 | 0 | 0.9 |
| 0 | 1 | 0.8 |
| 0 | 2 | 0.7 |
| 0 | 3 | 0.6 |
| 0 | 4 | 0.5 |
| 0 | 5 | 0.4 |
| 0 | 6 | 0.3 |
| 0 | 7 | 0.2 |
| 0 | 8 | 0.1 |
| 1 | 0 | 0.8 |
| 1 | 1 | 0.7 |
| 1 | 2 | 0.6 |
| 1 | 3 | 0.5 |
| 1 | 4 | 0.4 |
| 1 | 5 | 0.3 |
| 1 | 6 | 0.2 |
| 1 | 7 | 0.1 |
| 1 | 8 | 0.0 |
| 2 | 0 | 0.7 |
| 2 | 1 | 0.6 |
| 2 | 2 | 0.5 |
| 2 | 3 | 0.4 |
| 2 | 4 | 0.3 |
| 2 | 5 | 0.2 |
| 2 | 6 | 0.1 |
| 2 | 7 | 0.0 |
| 2 | 8 | -0.1 |
| 3 | 0 | 0.6 |
| 3 | 1 | 0.5 |
| 3 | 2 | 0.4 |
| 3 | 3 | 0.3 |
| 3 | 4 | 0.2 |
| 3 | 5 | 0.1 |
| 3 | 6 | -0.1 |
| 3 | 7 | -0.2 |
| 3 | 8 | -0.3 |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |
</details>

MSE:0.00105147, Feedbacks:440,mu:0.6, Delta:0.1,sparsity:4, Theory:54   
Figure 8: Empirical performance for the Sparse Sampling feedback mechanism.

![](images/dc36dda3d812f43d917d480785dfa39883154acab4b943ea413d7021f131b39f.jpg)

Figure 9: Empirical performance for the Sparse Sampling feedback mechanism.   
![](images/bee3c6f752a82291b24c65fa39e26761ff1738c598c5d62f355fa5c9a8bc3611.jpg)

Figure 10: Empirical performance for the Sparse Sampling feedback mechanism.   
![](images/75478df867ec6d5d45fa11434793e61fe46d9dfe2f583fdce09810c7bb3ae8e5.jpg)  
Figure 11: Empirical performance for the Sparse Sampling feedback mechanism.

# D. Proof of Lemma 1

In this appendix we restate and provide the proof of Lemma 1.

Lemma 1 (Recovering orthogonal atoms). Let $\Phi \in \mathbb { R } ^ { p \times p }$ be a symmetric positive semi-definite matrix. Define the set of orthogonal Cholesky decompositions of Φ as

$$
\mathcal {W} _ {\mathcal {C D}} = \left\{\boldsymbol {U} \in \mathbb {R} ^ {p \times r}   \middle |   \boldsymbol {\Phi} = \boldsymbol {U} \boldsymbol {U} ^ {\top}   a n d   \boldsymbol {U} ^ {\top} \boldsymbol {U} = d i a g (\lambda_ {1}, \ldots , \lambda_ {r}) \right\},
$$

where $r ~ = ~ r a n k ( \Phi )$ and $\lambda _ { 1 } , \lambda _ { 2 } , \ldots , \lambda _ { r }$ are the eigenvalues $o f \Phi$ in descending order. Then, for any two matrices $\pmb { U } , \pmb { U } ^ { \prime } \in \mathcal { W } _ { C D } ,$ , there exists an orthogonal matrix $R \in \mathbb { R } ^ { r \times r }$ such that

$$
\boldsymbol {U} ^ {\prime} = \boldsymbol {U R},
$$

where R is block diagonal with orthogonal blocks corresponding to any repeated diagonal entries $d _ { i }$ in $\pmb { U } ^ { \top } \pmb { U } .$ . Additionally, each column of U′ can differ from the corresponding column of U by a sign change.

Proof. Let $ { \mathbf { U } } ,  { \mathbf { U } } ^ { \prime } \in \mathcal { W } _ { \mathsf { C D } }$ be two orthogonal Cholesky decompositions of Φ. Define $\mathbf { R } = \mathbf { U } ^ { \top } \mathrm { d i a g } ( 1 / \lambda _ { 1 } , \dots , 1 / \lambda _ { r } ) \mathbf { U } ^ { \prime }$ . We will show that this matrix satisfies our requirements through the following steps:

First, we show that R is orthogonal. Note,

$$
\begin{array}{l} \mathbf {R} ^ {\top} \mathbf {R} = (\mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime}) ^ {\top} (\mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime}) \\ = \mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime} \\ = \mathbf {U} ^ {\top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \boldsymbol {\Phi} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} ^ {\prime} \\ = \mathbf {U} ^ {\prime \top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} ^ {\prime} \mathbf {U} ^ {\prime \top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} ^ {\prime} \\ = \mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime} \\ = \mathbf {I} _ {r} \\ \end{array}
$$

Similarly,

$$
\begin{array}{l} \mathbf {R} \mathbf {R} ^ {\top} = \mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime} (\mathbf {U} ^ {\prime}) ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} \\ = \mathbf {U} ^ {\top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \boldsymbol {\Phi} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} \\ = \mathbf {U} ^ {\top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} \mathbf {U} ^ {\top} \mathbf {U} \\ = \mathbf {U} ^ {\top} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \mathbf {U} \operatorname{diag} \left(\lambda_ {1}, \dots , \lambda_ {r}\right) \\ = \mathbf {I} _ {r} \\ \end{array}
$$

Now we show that $\mathbf { U } ^ { \prime } = \mathbf { U } \mathbf { R }$ .

$$
\begin{array}{l} \mathbf {U} \mathbf {R} = \mathbf {U} \mathbf {U} ^ {\top} \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime} \\ = \Phi \operatorname{diag} (1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}) \mathbf {U} ^ {\prime} \\ = \mathbf {U} ^ {\prime} \mathbf {U} ^ {\top} \mathbf {U} ^ {\prime} \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \\ = \mathbf {U} ^ {\prime} \operatorname{diag} \left(\lambda_ {1}, \dots , \lambda_ {r}\right) \operatorname{diag} \left(1 / \lambda_ {1}, \dots , 1 / \lambda_ {r}\right) \\ = \mathbf {U} ^ {\prime} \\ \end{array}
$$

To show that R is block diagonal with orthogonal blocks corresponding to repeated eigenvalues, consider the partitioning based on distinct eigenvalues. Let ${ \mathcal { T } } _ { k } = \{ i \mid \lambda _ { i } = \gamma _ { k } \}$ be the set of indices corresponding to the k-th distinct eigenvalue $\gamma _ { k }$ of Φ, for $k = 1 , \ldots , K$ , where K is the number of distinct eigenvalues. Let $m _ { k } = | \mathcal { T } _ { k } |$ denote the multiplicity of $\gamma _ { k }$ .

Define ${ \bf U } _ { k }$ and $\mathbf { U } _ { k } ^ { \prime }$ as the submatrices of U and U′ consisting of columns indexed by $\mathcal { T } _ { k } .$ , respectively.

Now, consider the block ${ \bf R } _ { k \ell }$ of R corresponding to eigenvalues $\gamma _ { k }$ and $\gamma _ { \ell }$ . For $k \neq \ell , \mathbf { U } _ { k }$ and $\mathbf { U } _ { \ell } ^ { \prime }$ correspond to different eigenspaces (as $\gamma _ { k } \neq \gamma _ { \ell } )$ , and thus their inner product is zero. Hence,

$$
\mathbf {U} _ {k} ^ {\top} \operatorname{diag} \left(\frac {1}{\lambda_ {1}}, \dots , \frac {1}{\lambda_ {r}}\right) \mathbf {U} _ {\ell} ^ {\prime} = \mathbf {0} _ {m _ {k} \times m _ {\ell}}.
$$

This implies ${ \bf R } _ { k \ell } = { \bf 0 } _ { m _ { k } \times m _ { \ell } }$ for $k \neq \ell .$

But then R must be block diagonal:

$$
\mathbf {R} = \left[ \begin{array}{c c c c} \mathbf {R} _ {1} & \mathbf {0} & \dots & \mathbf {0} \\ \mathbf {0} & \mathbf {R} _ {2} & \dots & \mathbf {0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf {0} & \mathbf {0} & \dots & \mathbf {R} _ {K} \end{array} \right],
$$

where each $\mathbf { R } _ { k } \in \mathbb { R } ^ { m _ { k } \times m _ { k } }$ is an orthogonal matrix. For eigenvalues with multiplicity one $( m _ { k } = 1 )$ , the corresponding block ${ \bf R } _ { k }$ is a $1 \times 1$ orthogonal matrix. The only possibilities are:

$$
\mathbf {R} _ {k} = [ 1 ] \quad \mathrm{or} \quad \mathbf {R} _ {k} = [ - 1 ],
$$

representing a sign change in the corresponding column of U. For eigenvalues with multiplicity greater than one $( m _ { k } > 1 )$ , each block ${ \bf R } _ { k }$ can be any $m _ { k } \times m _ { k }$ orthogonal matrix. This allows for rotations within the eigenspace corresponding to the repeated eigenvalue $\gamma _ { k }$ .

Combining all steps, we have shown that:

$$
\mathbf {U} ^ {\prime} = \mathbf {U R},
$$

where R is an orthogonal, block-diagonal matrix. Each block ${ \bf R } _ { k }$ corresponds to a distinct eigenvalue $\gamma _ { k }$ of Φ and is either a $1 \times 1$ matrix with entry ±1 (for unique eigenvalues) or an arbitrary orthogonal matrix of size equal to the multiplicity of $\gamma _ { k }$ (for repeated eigenvalues). This completes the proof of the lemma.

![](images/dae8db0f26ed61139b02bd9a467a6b91219b101d04e9001b2722dc62014f4ffc.jpg)

# E. Worst-case bounds: Constructive case

In this Appendix, we provide the proof of the lower bound as stated in Proposition 1. Before we prove this lower bound, we state a useful property of the sum of a symmetric, PSD matrix and a general symmetric matrix in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ .

Lemma 5. Let $\Phi \in S y m _ { + } ( \mathbb { R } ^ { p \times p } )$ be a symmetric matrix with full rank, i.e., $r a n k ( \Phi ) = p .$ For any arbitrary symmetric matrix $\Phi ^ { \prime } \in S y m ( \mathbb { R } ^ { p \times p } )$ , there exists a positive scalar $\lambda > 0$ such that the matrix $( \Phi + \lambda \Phi ^ { \prime } )$ is positive semidefinite.

Proof. Since Φ is symmetric and has full rank, it admits an eigendecomposition:

$$
\boldsymbol {\Phi} = \sum_ {i = 1} ^ {p} \lambda_ {i} u _ {i} u _ {i} ^ {\top},
$$

where $\{ \lambda _ { i } \} _ { i = 1 } ^ { p }$ are the positive eigenvalues and $\{ u _ { i } \} _ { i = 1 } ^ { p }$ are the corresponding orthonormal eigenvectors of $\Phi$

Define the constant γ as the maximum absolute value of the quadratic forms of $\Phi ^ { \prime }$ with respect to the eigenvectors of $\Phi$ :

$$
\gamma := \max _ {1 \leq i \leq p} \left| u _ {i} ^ {\top} \Phi^ {\prime} u _ {i} \right|.
$$

Let λ be chosen as:

$$
\lambda := \frac {\min _ {1 \leq i \leq p} \lambda_ {i}}{\gamma}.
$$

For each eigenvector ui, consider the quadratic form of $( \pmb { \Phi } + \lambda \pmb { \Phi } ^ { \prime } )$ :

$$
u _ {i} ^ {\top} (\pmb {\Phi} + \lambda \pmb {\Phi} ^ {\prime}) u _ {i} = \lambda_ {i} + \lambda u _ {i} ^ {\top} \pmb {\Phi} ^ {\prime} u _ {i} \geq \lambda_ {i} - \lambda \gamma = \lambda_ {i} - \frac {\min \lambda_ {i}}{\gamma} \gamma = \lambda_ {i} - \min \lambda_ {i} \geq 0.
$$

This shows that each eigenvector $u _ { i }$ satisfies:

$$
u _ {i} ^ {\top} (\Phi + \lambda \Phi^ {\prime}) u _ {i} \geq 0.
$$

Since $\{ u _ { i } \} _ { i = 1 } ^ { p }$ forms an orthonormal basis for $\mathbb { R } ^ { p }$ , for any vector $x \in \mathbb { R } ^ { p }$ , we can express x as $\textstyle x = \sum _ { i = 1 } ^ { p } a _ { i } u _ { i }$ . Then:

$$
x ^ {\top} (\boldsymbol {\Phi} + \lambda \boldsymbol {\Phi} ^ {\prime}) x = \sum_ {i = 1} ^ {p} a _ {i} ^ {2} u _ {i} ^ {\top} (\boldsymbol {\Phi} + \lambda \boldsymbol {\Phi} ^ {\prime}) u _ {i} \geq 0,
$$

since each term in the sum is non-negative.

Therefore, $( \Phi + \lambda \Phi ^ { \prime } )$ is positive semidefinite.

![](images/b1be11b7c7b6f0c0e5de6b6b4fed3f3f0a0c5073239ef670e7e7c1b79ae81d68.jpg)

Now, we provide the proof of Proposition 1 in the following:

Proof of Proposition 1. Assume, for contradiction, that there exists a feedback set $\mathcal { F } ( \mathcal { V } , \mathcal { M } _ { \sf F } , \Phi ^ { * } )$ for Eq. (1) with size $\begin{array} { r } { | \mathcal { F } | < \left( \frac { p ( p + 1 ) } { 2 } - 1 \right) } \end{array}$ .

For each pair $( y , z ) \in \mathcal { F } , \Phi ^ { * }$ is orthogonal to $( y y ^ { \top } - z z ^ { \top } )$ , implying that $( y y ^ { \top } - z z ^ { \top } ) \in { \mathcal { O } } _ { \Phi } ,$ , the orthogonal complement of $\Phi ^ { * }$ . Therefore,

$$
\operatorname{span} \left\langle \left\{y y ^ {\top} - z z ^ {\top} \right\} _ {(y, z) \in \mathcal {F}} \right\rangle \subset \mathcal {O} _ {\Phi^ {*}}.
$$

This leads to

$$
\Phi^ {*} \perp \operatorname{span} \left\langle \left\{y y ^ {\top} - z z ^ {\top} \right\} _ {(y, z) \in \mathcal {F}} \right\rangle .
$$

Since $\begin{array} { r } { | \mathcal { F } | < \frac { p ( p + 1 ) } { 2 } - 1 } \end{array}$ , we have

$$
\dim \left(\operatorname{span} \left\langle \left\{y y ^ {\top} - z z ^ {\top} \right\} \right\rangle\right) <   \frac {p (p + 1)}{2} - 1.
$$

Adding $\Phi ^ { * }$ to this span increases the dimension by at most one:

$$
\dim \left(\operatorname{span} \left\langle \Phi^ {*} \cup \left\{y y ^ {\top} - z z ^ {\top} \right\} _ {(y, z) \in \mathcal {F}} \right\rangle\right) \leq \frac {p (p + 1)}{2} - 1.
$$

Since $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ is a vector space with dim $\begin{array} { r } { ( \mathsf { S y m } ( \mathbb { R } ^ { p \times p } ) ) = \frac { p ( p + 1 ) } { 2 } } \end{array}$ , there exists a symmetric matrix $\Phi ^ { \prime } \in \mathcal { O } _ { \Phi }$ ∗ such that

$$
\boldsymbol {\Phi} ^ {\prime} \perp (y y ^ {\top} - z z ^ {\top}) \quad \forall (y, z) \in \mathcal {F}.
$$

By Lemma 5, there exists $\lambda > 0$ such that $\Phi ^ { * } + \lambda \Phi ^ { \prime }$ is PSD and symmetric. Since $\Phi ^ { \prime } \in \mathcal { O } _ { \Phi } \mathrm { : }$ ∗ and $\Phi ^ { \prime }$ is not a scalar multiple of $\Phi ^ { * }$ , the matrix $\Phi ^ { * } + \lambda \Phi ^ { \prime }$ is not related to $\Phi ^ { * }$ via linear scaling. However, it still satisfies Eq. (1), contradicting the minimality of ${ \mathcal F } .$ .

Thus, any feedback set must satisfy

$$
| \mathcal {F} | \geq \frac {p (p + 1)}{2} - 1.
$$

This establishes the stated lower bound on the feedback complexity of the feedback set.

![](images/158952fae4a79ccd8c2b46cd491545709a9ac743ccacef83979dfaa3270bad4e.jpg)

# F. Proof of Theorem 1: Upper bound

Below we provide proof of the upper bound stated in Theorem 1.

Consider the eigendecomposition of the matrix $\Phi ^ { * }$ . There exists a set of orthonormal vectors $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { r } \}$ with corresponding eigenvalues $\{ \gamma _ { 1 } , \gamma _ { 2 } , \ldots , \gamma _ { r } \}$ such that

$$
\boldsymbol {\Phi} ^ {*} = \sum_ {i = 1} ^ {r} \gamma_ {i} v _ {i} v _ {i} ^ {\top} \tag {5}
$$

Denote the set of orthogonal vectors $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { r } \}$ as $V _ { [ r ] }$ .

Let $\{ v _ { r + 1 } , \ldots , v _ { p } \}$ , denoted as $V _ { [ p - r ] }$ , be an orthogonal extension to the vectors in $V _ { [ r ] }$ such that

$$
V _ {[ r ]} \cup V _ {[ p - r ]} = \{v _ {1}, v _ {2}, \ldots , v _ {p} \}
$$

forms an orthonormal basis for Rp. Denote the complete basis $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { p } \}$ as $V _ { [ p ] }$ .

Note that $\{ v _ { r + 1 } , \ldots , v _ { p } \}$ precisely defines the null space of $\Phi ^ { * }$ , i.e.,

$$
\operatorname{null} \left(\boldsymbol {\Phi} ^ {*}\right) = \operatorname{span} \left\langle \left\{v _ {r + 1}, \dots , v _ {p} \right\} \right\rangle .
$$

The key idea of the proof is to manipulate this null space to satisfy the feedback set condition in Eq. (2) for the target matrix $\Phi ^ { * }$ . Since $\Phi ^ { * }$ has rank $r \leq p ,$ , the number of degrees of freedom is exactly $\textstyle { \frac { r ( r + 1 ) } { 2 } }$ . Alternatively, the span of the null space of $\Phi ^ { * }$ , which has dimension exactly $p - r$ , fixes the remaining entries in $\Phi ^ { * }$ .

Using this intuition, the teacher can provide pairs $( y , z ) \in \mathcal { V } ^ { 2 }$ to teach the null space and the eigenvectors $\{ v _ { 1 } , v _ { 2 } , \ldots , v _ { r } \}$ separately. However, it is necessary to ensure that this strategy is optimal in terms of sample efficiency. We confirm the optimality of this strategy in the next two lemmas.

# F.1. Feedback set for the null space of $\Phi ^ { * }$

Our first result is on nullifying the null set of $\Phi ^ { * }$ in the Eq. (2). Consider a partial feedback set

$$
\mathcal {F} _ {\mathrm{null}} = \{(0, v _ {i}) \} _ {i = r + 1} ^ {p}
$$

Lemma 6. If the teacher provides the set $\mathcal { F } _ { n u l I }$ then the null space of any PSD symmetric matrix $\Phi ^ { \prime }$ that satisfies Eq. (2) contains the span of $\{ v _ { r + 1 } , \ldots , v _ { p } \}$ , i.e.,

$$
\left\{v _ {r + 1}, \dots , v _ {p} \right\} \subseteq \operatorname{null} \left(\boldsymbol {\Phi} ^ {\prime}\right).
$$

Proof. Let $\Phi ^ { \prime } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ be a matrix that satisfies Eq. (2) (note that $\Phi ^ { * }$ satisfies Eq. (2)). Thus, we have the following equality constraints:

$$
\forall (0, v) \in \mathcal {F} _ {\text { null }}, \quad v ^ {\top} \Phi^ {\prime} v = 0.
$$

Since $\{ v _ { r + 1 } , \ldots , v _ { p } \}$ is a set of linearly independent vectors, it suffices to show that

$$
\forall v \in V _ {[ d - r ]}, \quad v ^ {\top} \boldsymbol {\Phi} ^ {\prime} v = 0 \implies \boldsymbol {\Phi} ^ {\prime} v = 0. \tag {6}
$$

To prove Eq. (6), we utilize general properties of the eigendecomposition of a symmetric, positive semi-definite matrix. We express $\Phi ^ { \prime }$ in its eigendecomposition as

$$
\boldsymbol {\Phi} ^ {\prime} = \sum_ {i = 1} ^ {s} \gamma_ {i} ^ {\prime} u _ {i} u _ {i} ^ {\top},
$$

where $\{ u _ { i } \} _ { i = 1 } ^ { s }$ are the eigenvectors and $\{ \gamma _ { i } ^ { \prime } \} _ { i = 1 } ^ { s }$ are the corresponding eigenvalues of $\Phi ^ { \prime } .$ Assume that x $\neq 0 \in \mathbb { R } ^ { p }$ satisfies

$$
x ^ {\top} \Phi^ {\prime} x = 0.
$$

Consider the decomposition $\begin{array} { r } { x = \sum _ { i = 1 } ^ { s } a _ { i } u _ { i } + v ^ { \prime } } \end{array}$ for scalars $a _ { i }$ and $v ^ { \prime } \bot \{ u _ { i } \} _ { i = 1 } ^ { s }$ . Now, expanding the equation above, we get

$$
\begin{array}{l} x ^ {\top} \boldsymbol {\Phi} ^ {\prime} x = \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i} + v ^ {\prime}\right) ^ {\top} \boldsymbol {\Phi} ^ {\prime} \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i} + v ^ {\prime}\right) \\ = \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) ^ {\top} \boldsymbol {\Phi} ^ {\prime} \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) + v ^ {\prime \top} \boldsymbol {\Phi} ^ {\prime} \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) + \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) \boldsymbol {\Phi} ^ {\prime} v ^ {\prime} + v ^ {\prime \top} \boldsymbol {\Phi} ^ {\prime} v ^ {\prime} \\ = \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) ^ {\top} \left(\sum_ {i = 1} ^ {s} \gamma_ {i} ^ {\prime} u _ {i} u _ {i} ^ {\top}\right) \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) + \underbrace {2 v ^ {\prime \top} \left(\sum_ {i = 1} ^ {s} \gamma_ {i} ^ {\prime} u _ {i} u _ {i} ^ {\top}\right) \left(\sum_ {i = 1} ^ {s} a _ {i} u _ {i}\right) + v ^ {\prime \top} \left(\sum_ {i = 1} ^ {s} \gamma_ {i} ^ {\prime} u _ {i} u _ {i} ^ {\top}\right) v ^ {\prime}} _ {= 0 \text {as} v ^ {\prime} \bot \{u _ {i} \}} \\ = \sum_ {i, j, k} a _ {i} u _ {i} ^ {\top} (\gamma_ {j} ^ {\prime} u _ {j} u _ {j} ^ {\top}) a _ {k} u _ {k} \\ = \sum_ {i = 1} ^ {s} a _ {i} ^ {2} \gamma_ {i} ^ {\prime} = 0 \\ \end{array}
$$

Since $\gamma _ { i } ^ { \prime } > 0$ for all $i = 1 , \dots , s$ (because $\Phi ^ { \prime }$ is PSD), it follows that each $a _ { i } = 0$ . Therefore,

$$
\Phi^ {\prime} x = \Phi^ {\prime} v ^ {\prime} = 0.
$$

This implies that $x \in \mathrm { n u l l } ( \Phi ^ { \prime } )$ , thereby proving Eq. (6).

Hence, if the teacher provides $\mathcal { F } _ { \mathrm { { n u l l } } }$ , any solution $\Phi ^ { \prime }$ to Eq. (2) must satisfy

$$
\left\{v _ {r + 1}, \dots , v _ {p} \right\} \subseteq \operatorname{null} \left(\Phi^ {\prime}\right).
$$

With this we will argue that the feedback setup in Eq. (2) can be decomposed in two parts: first is teaching the null set nu $\mathfrak { l l } ( \Phi ^ { * } ) : = \operatorname { s p a n } \left. \{ v _ { i } \} _ { i = r + 1 } ^ { n } \right.$ , and second is teaching $S _ { \Phi ^ { * } } = \operatorname { s p a n } \left. \{ v _ { i } \} _ { i = 1 } ^ { r } \right.$ in the form of $\begin{array} { r } { \Phi ^ { * } = \sum _ { i = 1 } ^ { r } \gamma _ { i } v _ { i } v _ { i } ^ { \top } } \end{array}$ .

Lemma 6 implies that using a feedback set of the form $\mathcal { F } _ { \mathrm { { n u l l } } }$ any solution $\Phi ^ { \prime } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ to Eq. (2) satisfies the property $V _ { [ d - r ] } \subset \mathrm { n u l l } ( \Phi ^ { \prime } )$ ). Furthermore, $| \mathcal { F } _ { \sf n u l l } | = p - r$ .

# F.2. Feedback set for the kernel of $\Phi ^ { * }$

Next, we discuss how to teach $V _ { [ r ] } ,$ i.e. $V _ { [ r ] }$ span the rows of any solution $\Phi ^ { \prime } \in \mathsf { S y m } _ { + } ( \mathbb { R } ^ { p \times p } )$ to Eq. (2) with the corresponding eigenvalues $\left\{ \gamma _ { i } \right\} _ { i = 1 } ^ { r }$ . We show that if the search space of metrics in $\operatorname { E q . } ( \mathbf { \mu } )$ is the version space $\mathsf { V S } ( \mathcal { M } _ { \sf F } , \mathcal { F } _ { \sf n u l l } )$ which is a restriction of the space $\mathcal { M } _ { \sf F }$ to feedback set $\mathcal { F } _ { \mathrm { { n u l l } } }$ , then a feedback set of size at most r(r+1) − 1 is sufficient to $\textstyle { \frac { r ( r + 1 ) } { 2 } } - 1$ 2 teach $\Phi ^ { * }$ up to feature equivalence. Thus, we consider the reformation of the problem in Eq. (2) as

$$
\forall (y, z) \in \mathcal {F} (\mathcal {X}, \mathrm{VS} \left(\mathcal {M} _ {\mathrm{F}}, \mathcal {F} _ {\text { null }}\right), \Phi^ {*}), \quad \boldsymbol {\Phi} \cdot \left(y y ^ {\top} - z z ^ {\top}\right) = 0 \tag {7}
$$

where the feedback set $\mathcal { F } ( \mathcal { X } , \mathsf { V S } ( \mathcal { M } _ { \mathsf { F } } , \mathcal { F } _ { \mathsf { n u l l } } ) , \Phi ^ { * } )$ is devised to solve a smaller space $\begin{array} { r l } { \mathsf { V S } ( \mathcal { M } _ { \mathsf { F } } , \mathcal { F } _ { \mathsf { n u l l } } ) } & { { } : = } \end{array}$ $\{ \Phi \in \mathcal { M } _ { \sf F } | \Phi v = 0 , \forall ( 0 , v ) \in \mathcal { F } _ { \sf n u l l } \}$ . With this state the following useful lemma on the size of the restricted feedback set $\mathcal { F } ( \mathcal { X } , \mathsf { V S } ( \mathcal { M } _ { \mathsf { F } } , \mathcal { F } _ { \mathsf { n u l l } } ) , \Phi ^ { * } )$ .

Lemma 7. Consider the problem as formulated in Eq. (7) in which the null set nul $( \Phi ^ { * } )$ of the target matrix $\Phi ^ { * }$ is known. Then, the teacher sufficiently and necessarily finds a set $\mathcal { F } ( \mathcal { X } , V S ( \mathcal { F } _ { n u l l } ) , \Phi ^ { * } )$ of size $\textstyle { \frac { r ( { \dot { r } } + 1 ) } { 2 } } - 1$ ) − 1 for oblivious learning up to feature equivalence.

Proof. Note that any solution $\Phi ^ { \prime }$ of Eq. (7) has its columns spanned exactly by $V _ { [ r ] }$ . Alternatively, if we consider the eigendecompostion of $\Phi ^ { \prime }$ then the corresponding eigenvectors exists in span $\left. V _ { [ r ] } \right.$ . Furthermore, note that $\Phi ^ { * }$ is of rank r which implies there are only r(r+1)2 $\frac { r ( r + 1 ) } { 2 }$ degrees of freedom, i.e. entries in the matrix $\Phi ^ { * }$ , that need to be fixed. 2

Thus, there are exactly r linearly independent columns of $\Phi ^ { * }$ , indexed as $\{ j _ { 1 } , j _ { 2 } , \dots , j _ { r } \}$ . Now, consider the set of matrices

$$
\left\{\boldsymbol {\Phi} ^ {(i, j)} \mid i \in [ d ], j \in \{j _ {1}, j _ {2}, \ldots , j _ {r} \}, \boldsymbol {\Phi} _ {i ^ {\prime} j ^ {\prime}} ^ {(i, j)} = \mathbb {1} [ i ^ {\prime} \in \{i, j \}, j ^ {\prime} \in \{i, j \} \setminus \{i ^ {\prime} \} ] \right\}
$$

This forms a basis to generate any matrix with independent columns along the indexed set. Hence, the span of $S _ { \Phi } { \mathrm { : } }$ ∗ induces a subspace of symmetric matrices ofindexed set is spanned by elements of ension ∗ . Thus, $\frac { r ( r + 1 ) } { 2 }$ in the vector space lear that picking a fe $\mathsf { s y m m } ( \mathbb { R } ^ { p } )$ , i.e. tf size vectors along the in the orthogonal $S _ { \Phi ^ { * } }$ ${ \frac { r ( r + 1 ) } { 2 } } - 1$ complement of $\Phi ^ { * }$ , i.e. $O _ { \Phi } .$ ∗ restricted by this span sufficiently teaches $\Phi ^ { * }$ if null(Φ∗) is known. One exact form of this set is proven in Lemma 3. Since any solution $\Phi ^ { \prime }$ is agnostic to the scaling of the target matrix $\Phi ^ { \prime }$ , we have shown that the sufficiency on the feedback complexity for $\Phi ^ { * }$ up to feature equivalence.

Now, we show that the stated feedback set size is necessary. The argument is similar to the proof of Lemma 5.

For the sake of contradiction assume that there is a smaller sized feedback set $\mathcal { F } _ { \mathsf { s m a l l } }$ . This implies that there is some matrix in $\mathsf { V S } ( \mathcal { M } _ { \sf F } , \mathcal { F } _ { \sf n u l l } )$ , a subspace induced by span $S _ { \Phi ^ { * } }$ , orthogonal to $( \Phi ^ { * } )$ is not in the span of $\mathcal { F } _ { \mathsf { s m a l l } }$ , denoted as $\Phi ^ { \prime } .$ . If $\Phi ^ { \prime }$ is PSD then it is a solution to $\operatorname { E q . } \left( \mathbf { \Sigma } \right)$ and $\Phi ^ { \prime }$ is not a scalar multiple of $\Phi ^ { * }$ . Now, if $\Phi ^ { \prime }$ is not PSD we show that there exists scalar $\lambda > 0$ such that

$$
\boldsymbol {\Phi} ^ {*} + \lambda \boldsymbol {\Phi} ^ {\prime} \in \mathsf {S y m} _ {+} (\mathbb {R} ^ {p \times p}),
$$

i.e. the sum is PSD. Consider the eigendecompostion of $\Phi ^ { \prime }$ (assume rank $( \Phi ^ { \prime } ) = r ^ { \prime } )$

$$
\boldsymbol {\Phi} ^ {\prime} = \sum_ {i = 1} ^ {r ^ {\prime}} \delta_ {i} \mu_ {i} \boldsymbol {\mu} _ {i} ^ {\top}
$$

for orthogonal eigenvectors $\left\{ \mu _ { i } \right\} _ { i = 1 } ^ { r ^ { \prime } }$ and the corresponding eigenvalues $\left\{ \delta _ { i } \right\} _ { i = 1 } ^ { r ^ { \prime } }$ . Since (assume) $r _ { 0 } \le r ^ { \prime }$ of the eigenvalues are negative we can rewrite $\Phi ^ { \prime }$ as

$$
\boldsymbol {\Phi} ^ {\prime} = \sum_ {i = 1} ^ {r _ {0}} \delta_ {i} \mu_ {i} \boldsymbol {\mu} _ {i} ^ {\top} + \sum_ {j = r _ {0} + 1} ^ {r ^ {\prime}} \delta_ {j} \mu_ {j} \boldsymbol {\mu} _ {j} ^ {\top}
$$

Thus, if we can regulate the values of $\mu _ { i } ^ { \top } \Phi ^ { * } \mu _ { i }$ , for all $i = 1 , 2 , \ldots , r _ { 0 }$ , noting they are positive, then we can find an appropriate scalar $\lambda > 0$ . Let $\begin{array} { r } { m ^ { * } : = \operatorname* { m i n } _ { i \in [ r _ { 0 } ] } \mu _ { i } ^ { \top } \Phi ^ { * } \mu _ { i } } \end{array}$ and $\ell ^ { * } : = \mathrm { m a x } _ { i \in [ r _ { 0 } ] } | \delta _ { i } |$ . Now, setting $\begin{array} { r } { \lambda \le \frac { m ^ { * } } { \ell ^ { * } } } \end{array}$ achieves the desired property of $\Phi ^ { * } + \lambda \Phi ^ { \prime }$ as shown in the proof of Lemma 5.

Consider that both $\Phi ^ { \prime }$ and $\Phi ^ { * }$ are orthogonal to every element in the feedback set $\mathcal { F } _ { \mathsf { s m a l l } }$ . This orthogonality implies that $\Phi ^ { * }$ is not a unique solution to equation Eq. (7) up to a positive scaling factor.

Therefore, we have demonstrated that when the null set nul $\lfloor ( \Phi ^ { * } )$ of the target matrix $\Phi ^ { * }$ is known, a feedback set of size exactly r(r+1)2 − 1 is both necessary and sufficient. $\textstyle { \frac { r ( r + 1 ) } { 2 } } - 1$ □ 2

# F.3. Proof of Lemma 3 and construction of feedback set for $\mathrm { K e r } ( \Phi ^ { * } )$

Up until this point we haven’s shown how to construct this $\textstyle { \frac { r ( r + 1 ) } { 2 } } - 1$ sized feedback set. Consider the following union:

$$
\left\{v _ {1} v _ {1} ^ {\top} \right\} \cup \left\{v _ {2} v _ {2} ^ {\top}, (v _ {2} + v _ {1}) (v _ {2} + v _ {1}) ^ {\top} \right\} \cup \ldots \cup \left\{v _ {r} v _ {r} ^ {\top}, (v _ {1} + v _ {r}) (v _ {1} + v _ {r}) ^ {\top}, \ldots , (v _ {r - 1} + v _ {r}) (v _ {r - 1} + v _ {r}) ^ {\top} \right\}
$$

We can show that this union is a set of linearly independent matrices of rank 1 as stated in Lemma 3 below.

Lemma 3. Let $\{ v _ { i } \} _ { i = 1 } ^ { r } \subset \mathbb { R } ^ { p }$ be a set of orthogonal vectors. Then, the set of rank-1 matrices

$$
\mathcal {B} := \left\{v _ {i} v _ {i} ^ {\top}, (v _ {i} + v _ {j}) (v _ {i} + v _ {j}) ^ {\top} \mid 1 \leq i <   j \leq r \right\}
$$

is linearly independent in the space of symmetric matrices $S y m ( \mathbb { R } ^ { p \times p } )$ .

Proof. We prove the claim by considering two separate cases. For the sake of contradiction, suppose that the set B is linearly dependent. This implies that there exists at least one matrix of the form $v _ { i } v _ { i } ^ { \top }$ or $( v _ { i } + v _ { j } ) ( v _ { i } + v _ { j } ) ^ { \top }$ that can be expressed as a linear combination of the other matrices in B. We now examine these two cases individually.

Case 1: First, we assume that for some $i \in [ r ] , v _ { i } v _ { i } ^ { \top }$ can be written as a linear combination. Thus, there exists scalars that satisfy the following property

$$
v _ {i} v _ {i} ^ {\top} = \sum_ {j = 1} ^ {r ^ {\prime}} \alpha_ {j} v _ {i j} v _ {i j} ^ {\top} + \sum_ {k = 1} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) (v _ {l _ {k}} + v _ {m _ {k}}) ^ {\top} \tag {8}
$$

$$
\forall j, k, \quad \alpha_ {j}, \beta_ {k} > 0, i _ {j} \neq i, l _ {k} <   m _ {k} \tag {9}
$$

Now, note that we can write

$$
\sum_ {k = 1} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) (v _ {l _ {k}} + v _ {m _ {k}}) ^ {\top} = \sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {l _ {k}} ^ {\top} + \sum_ {k = 1, l _ {k} \neq i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {l _ {k}} ^ {\top} + \sum_ {k = 1} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {m _ {k}} ^ {\top}
$$

But the following sum

$$
\sum_ {j = 1} ^ {r ^ {\prime}} \alpha_ {j} v _ {i _ {j}} v _ {i _ {j}} ^ {\top} + \sum_ {k = 1, l _ {k} \neq i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {l _ {k}} ^ {\top} + \sum_ {k = 1} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {m _ {k}} ^ {\top}
$$

doesn’t span (as column vectors) a subspace that contains the column vector $v _ { i }$ because $\left\{ v _ { i } \right\} _ { i = 1 } ^ { r }$ is a set of orthogonal vectors. Thus, we can write

$$
v _ {i} v _ {i} ^ {\top} = \sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) v _ {l _ {k}} ^ {\top} = \left(\sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} v _ {l _ {k}} + \sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} v _ {m _ {k}}\right) v _ {i} ^ {\top} \tag {10}
$$

This implies that

$$
\sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} v _ {m _ {k}} = 0 \implies \text { if } l _ {k} = i, \beta_ {k} = 0 \tag {11}
$$

Since not all $\beta _ { k } = 0$ corresponding to $l _ { k } = i$ (otherwise $\begin{array} { r } { \sum _ { k = 1 , l _ { k } = i } ^ { r ^ { \prime \prime } } \beta _ { k } v _ { l _ { k } } = 0 ) } \end{array}$ we have shown that $v _ { i } v _ { i } ^ { \top }$ can not be written as a linear combination of elements in $B \setminus \{ v _ { i } v _ { i } ^ { \top } \}$ .

Case 2: Now, we consider the second case where there exists some indices $i , j$ such that $( v _ { i } + v _ { j } ) ( v _ { i } + v _ { j } ) ^ { \top }$ is a sum of linear combination of elements in B. Note that this linear combination can’t have an element of type $v _ { k } v _ { k } ^ { \top }$ as it contradicts the first case. So, there are scalars such that

$$
(v _ {i} + v _ {j}) (v _ {i} + v _ {j}) ^ {\top} = \sum_ {k = 1} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) (v _ {l _ {k}} + v _ {m _ {k}}) ^ {\top} \tag {12}
$$

$$
\forall k, \quad l _ {k} <   m _ {k} \tag {13}
$$

But we rewrite this as

$$
\begin{array}{l} (v _ {i} + v _ {j}) v _ {i} ^ {\top} + (v _ {i} + v _ {j}) v _ {j} ^ {\top} \\ = \sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {i} + v _ {m _ {k}}) v _ {i} ^ {\top} + \sum_ {k = 1, m _ {k} = j} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {j}) v _ {j} ^ {\top} + \sum_ {k = 1, l _ {k} \neq i, \atop m _ {k} \neq j} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) (v _ {l _ {k}} + v _ {m _ {k}}) ^ {\top} \\ \end{array}
$$

Note that if $l _ { k } = i$ then the corresponding $m _ { k } \neq j$ and vice versa. Since $\left\{ v _ { i } \right\} _ { i = 1 } ^ { r }$ are orthogonal, the decomposition above implies

$$
(v _ {i} + v _ {j}) v _ {i} ^ {\top} = \sum_ {k = 1, l _ {k} = i} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {i} + v _ {m _ {k}}) v _ {i} ^ {\top} \tag {14}
$$

$$
(v _ {i} + v _ {j}) v _ {j} ^ {\top} = \sum_ {k = 1, m _ {k} = j} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {j}) v _ {j} ^ {\top} \tag {15}
$$

$$
\sum_ {\substack {k = 1, l _ {k} \neq i, \\ m _ {k} \neq j}} ^ {r ^ {\prime \prime}} \beta_ {k} (v _ {l _ {k}} + v _ {m _ {k}}) (v _ {l _ {k}} + v _ {m _ {k}}) ^ {\top} = 0 \tag{16}
$$

But using the arguments in Eq. (10) and Eq. (11), we can achieve Eq. (14) or Eq. (15).

Thus, we have shown that the set of rank-1 matrices as described in B are linearly independent.

In Lemma 7, we discussed that in order to teach $\Phi ^ { * }$ sufficiently agent needs a feedback set of size $\frac { r ( r + 1 ) } { 2 } - 1$ if the null set of $\Phi ^ { * }$ is known. We can establish this feedback set using the basis shown in Lemma 3. We state this result in the following lemma.

Lemma 9. For a given target matrix $\begin{array} { r } { \Phi ^ { * } = \sum _ { i = 1 } ^ { r } \gamma _ { i } v _ { i } v _ { i } ^ { \top } } \end{array}$ and basis set of matrices B as shown in Lemma 3, the following set spans a subspace of dimension $\frac { r ( r + 1 ) } { 2 } - 1$ in $S y m ( \mathbb { R } ^ { p \times p } )$ .

$$
\mathcal {O} _ {\mathcal {B}} := \left\{ \begin{array}{l} v _ {1} v _ {1} ^ {\top} - \lambda_ {1 1} y y ^ {\top}, v _ {2} v _ {2} ^ {\top} - \lambda_ {2 2} y y ^ {\top}, (v _ {1} + v _ {2}) (v _ {1} + v _ {2}) ^ {\top} - \lambda_ {1 2} y y ^ {\top}, \ldots , \\ v _ {r} v _ {r} ^ {\top} - \lambda_ {r r} y y ^ {\top}, (v _ {1} + v _ {r}) (v _ {1} + v _ {r}) ^ {\top} - \lambda_ {1 r} y y ^ {\top}, \ldots , \\ (v _ {r - 1} + v _ {r}) (v _ {r - 1} + v _ {r}) ^ {\top} - \lambda_ {(r - 1) r} y y ^ {\top} \end{array} \right\}
$$

$$
y \boldsymbol {\Phi} ^ {*} y ^ {\top} \neq 0
$$

$$
\forall i, j, \quad \lambda_ {i i} = \frac {v _ {i} \boldsymbol {\Phi} ^ {*} v _ {i} ^ {\top}}{y \boldsymbol {\Phi} ^ {*} y ^ {\top}}, \quad \lambda_ {i j} = \frac {(v _ {i} + v _ {j}) \boldsymbol {\Phi} ^ {*} (v _ {i} + v _ {j}) ^ {\top}}{y \boldsymbol {\Phi} ^ {*} y ^ {\top}} \quad (i \neq j)
$$

Proof. Since $\Phi ^ { * }$ has at least r positive eigenvalues there exists a vector $y \in \mathbb { R } ^ { p }$ such that $y \Phi ^ { * } y ^ { \top } \ne 0$ . It is straightforward to note that $\mathcal { O } _ { B }$ is orthogonal to $\Phi ^ { * } . \operatorname { A s } \mathcal { O } _ { B } \subset \operatorname { s p a n } \langle B \rangle$ and $\begin{array} { r } { \Phi ^ { \ast } \bot \mathcal { O } _ { B } , \mathrm { d i m } ( \mathsf { s p a n } \langle \mathcal { O } _ { B } \rangle ) = \frac { r ( r + 1 ) } { 2 } - 1 } \end{array}$ . □ 2

Now, we will complete the proof of the main result of the appendix here.

Proof of Theorem 1. Combining the results from Lemma 6, Lemma 7, and Lemma 9, we conclude that the feedback setup in Eq. (2) can be effectively decomposed into teaching the null space and the span of the eigenvectors of $\Phi ^ { * }$ . The constructed feedback sets ensure that $\Phi ^ { * }$ is uniquely identified up to a linear scaling factor with optimal sample efficiency. □

# G. Proof of Theorem 1: Lower bound

In this appendix, we provide the proof of the lower bound as stated in Theorem 1. We proceed by first showing some useful properties on a valid feedback set ${ \mathcal { F } } ( \mathbb { R } ^ { p } , \mathcal { M } _ { \sf F } , \Phi ^ { * } )$ for a target feature matrix $\Phi ^ { * }$ . They are stated in Lemma 10 and Lemma 11.

First, we consider a basic spanning property of matrices $( x x ^ { \top } - y y ^ { \top } )$ for any pair $( x , y ) \in { \mathcal { F } }$ in the space of symmetric matrices $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ .

Lemma 10. ${ \cal I } f \Phi \in { \mathcal { O } } _ { \Phi } ,$ ∗ such that span $\langle c o I ( \Phi ) \rangle \subset s p a n \left. V _ { [ r ] } \right.$ then $\Phi \in s p a n \left. \mathcal { F } \right.$ .

Proof. Consider an $\Phi \in { \mathcal { O } } _ { \Phi ^ { * } }$ such that span $\langle \mathsf { c o l } ( \Phi ) \rangle \subset \mathsf { s p a n } \left. V _ { [ r ] } \right.$ . Note that the eigendecompostion of $\Phi$ (assume ran $\operatorname { k } ( \Phi ) = r ^ { \prime } < r )$

$$
\Phi = \sum_ {i = 1} ^ {r ^ {\prime}} \delta_ {i} \mu_ {i} \mu_ {i} ^ {\top}
$$

for orthogonal eigenvectors $\left\{ \mu _ { i } \right\} _ { i = 1 } ^ { r ^ { \prime } }$ 1 and the corresponding eigenvalues $\left\{ \delta _ { i } \right\} _ { i = 1 } ^ { r ^ { \prime } }$ has the property that span $\left. \left\{ \mu _ { i } \right\} _ { i = 1 } ^ { r ^ { \prime } } \right. \subset$ span $\left. V _ { [ r ] } \right.$ . Using the arguments exactly as shown in the second half of the proof of Lemma 7 we can show there exists $\lambda > 0$ such that $\Phi ^ { * } + \lambda \Phi \in \mathsf { V S } ( \mathcal { F } , \mathcal { M } _ { \mathsf { F } } )$ . But then Φ is not feature equivalent to $\Phi ^ { * }$ . But this contradicts the assumption of $\mathcal { F }$ being a valid feedback set. □

Lemma 11. There exists vectors $U _ { [ p - r ] } \subset \mathrm { n u l l } ( \Phi ^ { * } ) ( o f s i z e p - r )$ such that span $\left. U _ { [ p - r ] } \right. = \mathrm { n u l l } ( \Phi ^ { * } )$ and for any vector $v \in U _ { [ p - r ] } , v v ^ { \top } \in s p a n \left. { \mathcal { F } } \right.$ .

Proof. Assuming the contrary, there exists $v \in$ span $\langle \mathrm { n u l l } ( \Phi ^ { * } ) \rangle$ such that $v v ^ { \top } \notin$ span $\langle \mathcal { F } \rangle$ .

Now if $v v ^ { \top } \bot \mathcal { F }$ , then for any scalar $\lambda > 0 , \Phi ^ { * } + \lambda v v ^ { \top }$ is both symmetric and positive semi-definite and satisfies all the conditions in Eq. (1) wrt $\mathcal { F }$ a contradiction as $\Phi ^ { * } + \lambda v v ^ { \intercal }$ is not feature equivalent to $\Phi ^ { * }$ .

$\mathrm { S o } ,$ consider the case when vv $\ l ^ { \top } \ \nearrow \ \mathcal { F }$ . Let $\{ v _ { r + 1 } , \dotsc , v _ { p - 1 } \}$ be an orthogonal extension3 of v such that $\{ v _ { r + 1 } , \ldots , v _ { p - 1 } , v \}$ forms a basis of nul $\mathrm { l } ( \Phi ^ { * } )$ , i.e., in other words

$$
v \bot \left\{v _ {r + 1}, \dots , v _ {p - 1} \right\} \& \operatorname{span} \left\langle \left\{v _ {r + 1}, \dots , v _ {p - 1}, v \right\} \right\rangle = \operatorname{null} (\Phi^ {*}).
$$

We will first show that there exists some $\Phi ^ { \prime } \left( \neq \lambda \Phi ^ { * } \right.$ , for some $\lambda > 0 ) \in \mathbb { S } \mathsf { y m } ( \mathbb { R } ^ { p \times p } )$ orthogonal to $\mathcal { F }$ and furthermore $\{ v _ { r + 1 } , \ldots , v _ { p - 1 } \} \subset \mathrm { n u l l } ( \Phi ^ { \prime } )$ .

Consider the intersection (in the space $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } ) )$ of the orthogonal complement of the matrices $\left\{ v _ { r + 1 } v _ { r + 1 } ^ { \top } , \ldots , v _ { p - 1 } v _ { p - 1 } ^ { \top } \right\}$ , denote it as $\mathcal { O } _ { \mathrm { r e s t } } , \mathrm { i . e . }$ ,

$$
\mathcal {O} _ {\text { rest }} := \bigcap_ {i = r + 1} ^ {p - 1} \mathcal {O} _ {v _ {i} v _ {i} ^ {\top}}
$$

Note that

$$
\dim (\mathcal {O} _ {\mathrm{rest}}) = p (p + 1) / 2 - p + r
$$

Since $v v ^ { \top }$ is in $\mathcal { O } _ { \mathsf { r e s t } }$ and dim $( \mathscr { O } _ { \sf r e s t } ) > 1$ there exists some $\Phi ^ { \prime }$ such that $\Phi ^ { \prime } \perp \Phi ^ { * }$ , and also orthogonal to elements in the feedback set ${ \mathcal F } .$ Thus, $\Phi ^ { \prime }$ has a null set which includes the subset $\{ v _ { r + 1 } , \dotsc , v _ { p - 1 } \}$ .

Now, the rest of the proof involves showing existence of some scalar $\lambda > 0$ such that $\Phi ^ { * } + \lambda \Phi ^ { \prime }$ satisfies the conditions of Eq. (1) for the feedback set ${ \mathcal F } .$ . Note that if $v \Phi ^ { \prime } v ^ { \top } = 0$ then the proof is straightforward as span $\langle \{ v _ { r + 1 } , \ldots , v _ { p - 1 } , v \} \rangle \subset$ null $\left( \Phi ^ { \prime } \right)$ , which implies span $\langle \mathsf { c o l } ( \Phi ^ { \prime } ) \rangle \subset \mathrm { s p a n } \left. V _ { [ r ] } \right.$ . But this is precisely the condition for Lemma 10 to hold.

Without loss of generality assume that $v \Phi ^ { \prime } v ^ { \top } > 0$ . First note that the eigendecomposition of $\Phi ^ { \prime }$ has eigenvectors that are contained in $V _ { [ r ] } \cup \{ v \}$ . Consider some arbitrary choice of $\lambda > 0$ , we will fix a value later. It is straightforward that $\Phi ^ { * } + \lambda \Phi ^ { \prime }$ is symmetric for $\Phi ^ { * }$ and $\Phi ^ { \prime }$ are symmetric. In order to show it is positive semi-definite, it suffices to show that

$$
\forall u \in \mathbb {R} ^ {p}, u ^ {\top} (\boldsymbol {\Phi} ^ {*} + \lambda \boldsymbol {\Phi} ^ {\prime}) u \geq 0 \tag {17}
$$

Since $\{ v _ { r + 1 } , \ldots , v _ { p - 1 } \} \subset \left( \mathrm { n u l l } ( \Phi ^ { * } ) \cap \mathrm { n u l l } ( \Phi ^ { \prime } ) \right)$ we can simplify Eq. (17) to

$$
\forall u \in \operatorname{span} \left\langle V _ {[ r ]} \cup \{v \} \right\rangle , u ^ {\top} (\boldsymbol {\Phi} ^ {*} + \lambda \boldsymbol {\Phi} ^ {\prime}) u \geq 0 \tag {18}
$$

Consider the decomposition of any arbitrary vector u ∈ span $\left. V _ { [ r ] } \cup \{ v \} \right.$ as follows:

$$
u = u _ {[ r ]} + v ^ {\prime}, \text {   such   that   } u _ {[ r ]} \in \operatorname{span} \left\langle V _ {[ r ]} \right\rangle , v ^ {\prime} \in \operatorname{span} \left\langle \{v \} \right\rangle \tag {19}
$$

$$
u _ {[ r ]} := \sum_ {i = 1} ^ {r} \alpha_ {i} v _ {i}, \forall i \alpha_ {i} \in \mathbb {R} \tag {20}
$$

From here on we assume that $u _ { [ r ] } \neq 0$ . The alternate case is trivial as $v ^ { \prime \top } \Phi ^ { \prime } v ^ { \prime } > 0$ .

Now, we write the vectors as scalar multiples of their corresponding unit vectors

$$
u _ {[ r ]} = \delta_ {r} \cdot \hat {u} _ {r}, \quad \hat {u} _ {r} := \frac {u _ {[ r ]}}{| | u _ {[ r ]} | | _ {V _ {[ r ]}} ^ {2}}, | | u _ {[ r ]} | | _ {V _ {[ r ]}} ^ {2} := \sum_ {i = 1} ^ {r} \alpha_ {i} ^ {2} \tag {21}
$$

$$
v ^ {\prime} = \delta_ {v ^ {\prime}} \cdot \hat {v}, \quad \hat {v} := \frac {v}{\left| | v | \right| _ {2} ^ {2}} \tag {22}
$$

Remark: Although we have computed the norm of $u _ { [ r ] } \mathrm { a s } | | u _ { [ r ] } | | _ { V _ { [ r ] } } ^ { 2 }$ in the orthonormal basis $V _ { [ r ] }$ , note that the norm remains unchanged (same as the $\ell _ { 2 } ) . \ell _ { 2 }$ is used for ease of analysis later on.

Using the decomposition in Eq. (19)-(20), we can write Eq. (18) as

$$
\begin{array}{l} u ^ {\top} \left(\boldsymbol {\Phi} ^ {*} + \lambda \boldsymbol {\Phi} ^ {\prime}\right) u = \left(u _ {[ r ]} + v ^ {\prime}\right) ^ {\top} \left(\boldsymbol {\Phi} ^ {*} + \lambda \boldsymbol {\Phi} ^ {\prime}\right) \left(u _ {[ r ]} + v ^ {\prime}\right) \\ = u _ {[ r ]} ^ {\top} \boldsymbol {\Phi} ^ {*} u _ {[ r ]} + \lambda (u _ {[ r ]} + v ^ {\prime}) ^ {\top} \boldsymbol {\Phi} ^ {\prime} (u _ {[ r ]} + v ^ {\prime}) \\ = \delta_ {r} ^ {2} \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {*} \hat {u} _ {r} + \lambda \left(\delta_ {r} ^ {2} \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {u} _ {r} + 2 \delta_ {r} \delta_ {v ^ {\prime}} \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v} + \delta_ {v ^ {\prime}} ^ {2} \cdot \hat {v} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v}\right) \tag {23} \\ \end{array}
$$

Since we want $u ^ { \top } ( \Phi ^ { * } + \lambda \Phi ^ { \prime } ) u \geq 0$ we can further simplify Eq. (23) as

$$
\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {*} \hat {u} _ {r} + \lambda \left(\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {u} _ {r} + 2 \frac {\delta_ {r} \delta_ {v ^ {\prime}}}{\delta_ {r} ^ {2}} \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v} + \frac {\delta_ {v ^ {\prime}} ^ {2}}{\delta_ {r} ^ {2}} \cdot \hat {v} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v}\right) \geq 0 \tag {24}
$$

$$
\Longleftrightarrow \underbrace {\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {*} \hat {u} _ {r}} _ {(1)} + \lambda \left(\underbrace {\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {u} _ {r}} _ {(3)} + \underbrace {2 \xi \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v} + \xi^ {2} \cdot \hat {v} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v}} _ {(2)}\right) \geq 0 \tag {25}
$$

where we have used $\begin{array} { r } { \xi = \frac { \delta _ { v ^ { \prime } } } { \delta _ { r } } } \end{array}$ . The next part of the proof we show that (1) is lower bounded by a positive constant whereas (2) is upper bounded by a positive constant and there is a choice of λ so that (3) is always smaller than (1).

Considering (1) we note that $\hat { u } _ { r }$ is a unit vector wrt the orthonormal set of basis $V _ { [ r ] }$ . Expanding using the eigendecomposition of Eq. (5)

$$
\hat {u} _ {r} ^ {\top} \Phi^ {*} \hat {u} _ {r} = \sum_ {i = 1} ^ {r} \frac {\alpha_ {i} ^ {2}}{\sum_ {i = 1} ^ {r} \alpha_ {i} ^ {2}} \cdot \gamma_ {i} \geq \min _ {i} \gamma_ {i} > 0
$$

The last inequality follows as all the eigenvalues in the eigendecompostion are (strictly) positive. Denote this minimum eigenvalue as $\gamma _ { \mathrm { m i n } } : = \operatorname* { m i n } _ { i } \gamma _ { i }$ .

Considering (2) note that only terms that are variable (i.e. could change value) is ξ as $\hat { u } _ { r } ^ { \top } \Phi ^ { \prime } \hat { v }$ is

Note that vˆ is a fixed vector and $\hat { u } _ { r }$ has a fixed norm (using Eq. (21)-(22)), so $| \hat { u } _ { r } ^ { \top } \Phi ^ { \prime } \hat { v } | \leq C$ for some bounded constant $C > 0$ whereas $\hat { v } ^ { \top } \Phi ^ { \prime } \hat { v }$ is already a constant. Now, $| 2 \boldsymbol { \xi } \cdot \hat { \boldsymbol { u } } _ { r } ^ { \top } \Phi ^ { \prime } \boldsymbol { \hat { v } } |$ exceeds $\xi ^ { 2 } \cdot \hat { v } ^ { \top } \tilde { \Phi ^ { \prime } } \hat { v }$ only if

$$
| 2 \xi \cdot \hat {u} _ {r} ^ {\top} \Phi^ {\prime} \hat {v} | \geq | \xi^ {2} \cdot \hat {v} ^ {\top} \Phi^ {\prime} \hat {v} | \Longleftrightarrow \frac {| \hat {u} _ {r} ^ {\top} \Phi^ {\prime} \hat {v} |}{\hat {v} ^ {\top} \Phi^ {\prime} \hat {v}} \geq \xi \implies \frac {C}{\hat {v} ^ {\top} \Phi^ {\prime} \hat {v}} \geq \xi
$$

Rightmost inequality implies that $2 \xi \cdot \hat { u } _ { r } ^ { \top } \Phi ^ { \prime } \hat { v } + \xi ^ { 2 } \cdot \hat { v } ^ { \top } \Phi ^ { \prime } \hat { v }$ is negative only for an $\xi$ bounded from above by a positive constant. But since $\xi$ is non-negative

$$
\left| 2 \xi \cdot \hat {u} _ {r} ^ {\top} \Phi^ {\prime} \hat {v} + \xi^ {2} \cdot \hat {v} ^ {\top} \Phi^ {\prime} \hat {v} \right| \leq C ^ {\prime} (\text { bounded   constant })
$$

Now using an argument similar to the second half of the proof of Lemma 7, it is straight forward to show that there is a choice of $\lambda ^ { \prime } > 0$ so that (3) is always smaller than (1).

Now, for $\begin{array} { r } { \lambda = \frac { \lambda ^ { \prime } } { 2 \lceil C ^ { \prime } \rceil \lambda ^ { \prime \prime } } } \end{array}$ where $\lambda ^ { \prime \prime }$ is chosen so that $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } \ge \frac { \lambda ^ { \prime } } { \lambda ^ { \prime \prime } } } \end{array}$ , we note that

$$
\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {*} \hat {u} _ {r} + \lambda (\hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {u} _ {r} + 2 \xi \cdot \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v} + \xi^ {2} \cdot \hat {v} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {v}) \geq \lambda_ {\mathrm{min}} + \frac {\lambda^ {\prime}}{2 \lceil C ^ {\prime} \rceil \lambda^ {\prime \prime}} \hat {u} _ {r} ^ {\top} \boldsymbol {\Phi} ^ {\prime} \hat {u} _ {r} - \frac {\lambda^ {\prime}}{2 \lambda^ {\prime \prime}} > 0.
$$

Using the equivalence in Eq. (23), Eq. (24) and Eq. (25), we have a choice of $\lambda > 0$ such that $u ^ { \top } ( \Phi ^ { * } + \lambda \Phi ^ { \prime } ) u \geq 0$ for any arbitrary vector u ∈ span $\langle V _ { [ r ] } \cup \{ v \} \rangle$ . Hence, we have achieved the conditions in Eq. (18), which is the simplification of Eq. (17). This implies that $\bar { \Phi ^ { * } } + \lambda \Phi ^ { \prime }$ is positive semi-definite.

This implies that there doesn’t exist a $v \in \operatorname { s p a n } \left. \operatorname { n u l l } ( \Phi ^ { * } ) \right.$ such that $v v ^ { \top } \notin \operatorname { s p a n } \left. \mathcal { F } \right.$ otherwise the assumption on $\mathcal { F }$ to be an oblivious feedback set for $\Phi ^ { * }$ is violated. Thus, the statement of Lemma 11 has to hold. □

# G.1. Proof of lower bound in Theorem 1

In the following, we provide proof of the main statement on the lower bound of the size of a feedback set.

If any of the two lemmas (10-11) are violated, we can show there exists $\lambda > 0$ and Φ such that $\Phi ^ { * } + \lambda \Phi \in \mathsf { V S } ( \mathcal { F } , \mathcal { M } _ { \mathsf { F } } )$ In order to ensure these statements, the feedback set should have $\begin{array} { r l r } {  { ( \frac { r ( r + 1 ) } 2 + ( d - r ) - 1 ) } } \end{array}$  r(r+1)2 + (d − r) − 1 many elements which proves the lower bound on ${ \mathcal F } .$ .

But using Lemma 7 and Lemma 9 we know that the dimension of the span of matrices that satisfy the condition in Lemma 10 is at the least $\textstyle { \frac { r ( r + 1 ) } { 2 } } - 1$ . We can use Lemma 9 where $\textstyle y = \sum _ { i = 1 } ^ { r } v _ { r }$ (note $\Phi ^ { * } v \neq 0 )$ . Thus, any basis matrix in $\mathcal { O } _ { B }$ satisfy the conditions in Lemma 10.

Since the dimension of nul $. ( \Phi ^ { * } )$ is at least $( d - r )$ thus there are at least $( d - r )$ ) directions or linearly independent matrices (in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } ) ;$ ) that need to be spanned by F.

Thus, Lemma 10 implies there are $\frac { r ( r + 1 ) } { 2 } \ : - \ : 1$ linearly independent matrices (in $\mathcal { O } _ { \Phi ^ { * } } )$ that need to be spanned by ${ \mathcal F } .$ Similarly, Lemma 11 implies there are $p - r$ linearly independent matrices (in $\mathcal { O } _ { \Phi ^ { * } } )$ that need to be spanned by ${ \mathcal F } .$ . Note that the column vectors of these matrices from the two statements are spanned by orthogonal set of vectors, i.e. one by $V _ { [ r ] }$ and the other by $\mathrm { n u l l } ( \Phi ^ { * } )$ respectively. Thus, these $\begin{array} { r } { \frac { r ( r + 1 ) } { 2 } - 1 + ( p - r ) } \end{array}$ are linearly independent in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ , but this forces a lower bound on the size of $\mathcal { F }$ (a lower dimensional span can’t contain a set of vectors spanning higher dimensional space). This completes the proof of the lower bound in Theorem 1.

# H. Proof of Theorem 3: General Activations Sampling

We aim to establish both upper and lower bounds on the feedback complexity for oblivious learning in Algorithm 2. The proof revolves around the linear independence of certain symmetric matrices derived from random representations and the dimensionality required to span a target feature matrix.

Let us define a positive index $\textstyle P = { \frac { p ( p + 1 ) } { 2 } }$ . The agent receives $P$ representations:

$$
\mathcal {V} _ {n} := \{v _ {1}, v _ {2}, \dots , v _ {P} \} \sim \mathcal {D} _ {\mathcal {V}}.
$$

For each $i ,$ we define the symmetric matrix $V _ { i } = v _ { i } v _ { i } ^ { \top }$ .

Consider the matrix M formed by concatenating the vectorized $V _ { i } { : }$

$$
\mathbb {M} = \left[ \begin{array}{c c c c} \operatorname{vec} (V _ {1}) & \operatorname{vec} (V _ {2}) & \dots & \operatorname{vec} (V _ {P}) \end{array} \right],
$$

where each ${ \mathrm { v e c } } ( V _ { i } )$ is treated as a column vector in $\mathbb { R } ^ { P }$ . The vectorization operation for a symmetric matrix $A \in { \mathsf { S y m } } ( \mathbb { R } ^ { p \times p } )$ is defined as:

$$
\operatorname{vec} (A) _ {k} = \left\{ \begin{array}{l l} A _ {i i} & \text { if   } k \text {   corresponds   to   } (i, i), \\ A _ {i j} + A _ {j i} & \text { if   } k \text {   corresponds   to   } (i, j), i <   j. \end{array} \right.
$$

The determinant det(M) is a non-zero polynomial in the entries of $v _ { 1 } , v _ { 2 } , \ldots , v _ { P }$ . Since the vectors $v _ { i }$ are drawn from a continuous distribution $\mathcal { D } _ { \mathcal { V } }$ , using Sard’s theorem the probability that det $( \mathbb { M } ) = 0$ is zero, i.e.,

$$
\mathcal {P} _ {\mathcal {V} _ {n}} (\det (\mathbb {M}) = 0) = 0.
$$

This implies that, with probability 1, the set $\{ V _ { 1 } , V _ { 2 } , \ldots , V _ { P } \}$ is linearly independent in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ ):

$$
\mathcal {P} _ {\mathcal {V} _ {n}} \left(\{v _ {i} v _ {i} ^ {\top} \} \text {   is   linearly   independent   in   } \mathbf {S y m} (\mathbb {R} ^ {p \times p})\right) = 1. \tag {26}
$$

Next, let $\Sigma ^ { * } \neq 0$ be an arbitrary target feature matrix for learning with feedback in Algorithm 2. Without loss of generality, assume $v : = v _ { 1 } \neq 0$ . Define the set $\mathcal { F }$ of rescaled pairs as:

$$
\mathcal {F} = \left\{\left(v, \sqrt {\gamma_ {i}} v _ {i}\right) \Bigg | \Sigma^ {*} \cdot \left(v v ^ {\top} - \gamma_ {i} v _ {i} v _ {i} ^ {\top}\right) = 0, \sqrt {\gamma_ {i}} > 0 \right\},
$$

noting that $| \mathcal { F } | = P - 1$ .

Assume, for contradiction, that the elements of $\mathcal { F }$ are linearly dependent in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ . Then, there exist scalars $\{ a _ { i } \}$ (not all zero) such that:

$$
\sum_ {i = 2} ^ {P} a _ {i} \left(v v ^ {\top} - \gamma_ {i} v _ {i} v _ {i} ^ {\top}\right) = 0 \quad \Rightarrow \quad \left(\sum_ {i = 2} ^ {P} a _ {i}\right) v v ^ {\top} = \sum_ {i = 2} ^ {P} a _ {i} \gamma_ {i} v _ {i} v _ {i} ^ {\top}.
$$

However, since $\{ v _ { i } v _ { i } ^ { \top } \}$ are linearly independent with probability 1, it must be that:

$$
\sum_ {i = 2} ^ {P} a _ {i} = 0 \quad \text { and } \quad a _ {i} \gamma_ {i} = 0 \quad \forall i.
$$

Given that $\gamma _ { i } > 0 .$ , this implies $a _ { i } = 0$ for all $i ,$ contradicting the assumption of linear dependence. Therefore, matrices induced by $\mathcal { F }$ are linearly independent.

This implies that $\mathcal { F }$ induces a set of linearly independent matrices, i.e., $\{ v v ^ { \top } - \gamma _ { i } v _ { i } v _ { i } ^ { \top } \}$ in the orthogonal complement $\mathcal { O } _ { \Sigma ^ { * } }$ , and since $\Sigma ^ { * }$ has at most $P$ degrees of freedom, any matrix $\Sigma ^ { \prime } \in \mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ satisfying:

$$
\Sigma^ {\prime} \cdot \left(v v ^ {\top} - \gamma_ {i} v _ {i} v _ {i} ^ {\top}\right) = 0 \quad \forall i
$$

must be a positive scalar multiple of $\Sigma ^ { * }$ .

Thus, using Eq. (26), with probability 1, the feedback set $\mathcal { F }$ is valid:

$$
\mathcal {P} _ {\mathcal {V} _ {n}} \left(\mathcal {F} \text {   is   a   valid   feedback   set }\right) = 1.
$$

Since $\Sigma ^ { * }$ was arbitrary, the worst-case feedback complexity is almost surely upper bounded by $P - 1$ for achieving feature equivalence.

For the lower bound, consider the proof of the lower bound in Theorem 1, specifically Lemma 10, which asserts that for any feedback set F in Algorithm 1, given any target matrix $\Sigma ^ { * } \in \mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ , if $\Sigma \in { \mathcal { O } } _ { \Sigma }$ ∗ such that span $\left. \mathsf { c o l } ( \Sigma ) \right. \subset$ span $\langle Z _ { [ r ] } \rangle$ then $\Sigma \in$ span $\langle \mathcal { F } \rangle$ where $Z _ { [ r ] } \left( r \leq d \right)$ is defined as the set of eigenvectors in the eigendecompostion of $\Sigma ^ { * }$ (see Eq. (5)).

This implies that any feedback set $\mathcal { F } ( \mathbb { V } _ { n } , \Sigma ^ { * } )$ must span certain matrices $\Sigma ^ { \prime } \in \mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ . Suppose the agent receives ℓ representations $v _ { 1 } , v _ { 2 } , \ldots , v _ { \ell } \sim \mathcal { D } _ { \nu }$ and constructs:

$$
\mathbb {M} = \left[ \begin{array}{c c c c} \operatorname{vec} (\Sigma^ {\prime}) & \operatorname{vec} (V _ {1}) & \dots & \operatorname{vec} (V _ {\ell}) \end{array} \right].
$$

Now, consider the polynomial equation $\operatorname* { d e t } ( \mathbb { M } ) = 0$ . Since every entry of M is semantically different, the determinant det(M) zero set olynomial. Note that there are  has Lebesgue measure zero if $\textstyle { \frac { p ( p + 1 ) } { 2 } }$ y degrees of freedom for, i.e. M requires at least ws. Thus, it is clear that the columns for det(M) to be identically zero. But this implies that set $\{ \operatorname* { d e t } ( \mathbb { M } ) = 0 \}$ $\left\{ v _ { i } v _ { i } ^ { \top } \right\} _ { i = 1 } ^ { \ell }$ 2 can’t span $\ell < { \frac { p ( p + 1 ) } { 2 } }$ $\Sigma ^ { \prime }$ (almost surely) if $\ell \leq \frac { p ( p + 1 ) } { 2 } - 1$ $\frac { p ( p + 1 ) } { 2 }$ . Hence, (almost surely) 2 the agent can’t devise a feedback set for oblivious learning in Algorithm 2. In other words, if $\ell \leq \frac { p ( p + 1 ) } { 2 } - 1$ − 1,

$$
\mathcal {P} _ {\mathcal {V} _ {\ell}} \left(\text { agent   devises   a   feedback   set } \mathcal {F} \text { up   to   feature   equivalence }\right) = 0
$$

Hence, to span Σ′, it almost surely requires at least p(p+1)2 $\Sigma ^ { \prime }$ $\textstyle { \frac { p ( p + 1 ) } { 2 } }$ representations. Therefore, the feedback complexity cannot be lower than $\Omega \left( { \frac { p ( p + 1 ) } { 2 } } \right)$ . 2

Combining the upper and lower bounds, we conclude that the feedback complexity for oblivious learning in Algorithm 2 is tightly bounded by $\Theta \left( { \textstyle { \frac { p ( p + 1 ) } { 2 } } } \right)$ .

# I. Proof of Theorem 4: Sparse Activations Sampling

Here we consider the analysis for the case when the activations V are sampled from the sparse distribution as stated in Definition 1.

In Theorem 4, we assume that the activations are sampled from a Lebesgue distribution. This, sufficiently, ensures that (almost surely) any random sampling of P activations induces a set of linearly independent rank-1 matrices. Since the distribution in Assumption 1 is not a Lebesgue distribution over the entire support [0, 1], requiring an understanding of certain events of the sampling of activations which could lead to linearly independent rank-1 matrices.

In the proof of Theorem 2, we used a set of sparse activations using the standard basis of the vector space $\mathbb { R } ^ { p }$ . We note that the idea could be generalized to arbitrary choice of scalars as well, i.e.,

$$
U _ {g} = \{\lambda_ {i} e _ {i}: \lambda_ {i} \neq 0, 1 \leq i \leq p \} \cup \{(\lambda_ {i j i} e _ {i} + \lambda_ {i j j} e _ {j}): \lambda_ {i j i}, \lambda_ {i j j} \neq 0, 1 \leq i <   j \leq p \}.
$$

Here $e _ { i }$ is the ith standard basis vector. Note that the corresponding set of rank-1 matrices, denoted as $\widehat { U } _ { g }$

$$
\widehat {U} _ {g} = \left\{\lambda_ {i} ^ {2} e _ {i} e _ {i} ^ {T}: 1 \leq i \leq p \right\} \cup \left\{(\lambda_ {i j i} e _ {i} + \lambda_ {i j j} e _ {j}) (\lambda_ {i j i} e _ {i} + \lambda_ {i j j} e _ {j}) ^ {T}: 1 \leq i <   j \leq p \right\}
$$

is linearly independent in the space of symmetric matrices on $\mathbb { R } ^ { p }$ , i.e., $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ ).

Assume that activations are sampled P times, denoted as $\mathcal { V } _ { P }$ . Now, consider the design matrix $\mathbb { M } = [ V _ { 1 } \quad V _ { 2 } \quad . . . V _ { P } ]$ ] as shown in the proof of Theorem 3. We know that if det(M) is non-zero then $\left\{ V _ { i } \right\} ^ { \prime }$ s are linearly independent in $\mathsf { S y m } ( \mathbb { R } ^ { p \times p } )$ . To show if a sampled set $\mathcal { V } _ { P }$ exhibits this property we need to show that det(M) is not identically zero, which could be possible for activations sampled from sparse distributions as stated in Assumption 1, i.e. $\mathcal { P } _ { v \sim \mathcal { D } _ { \mathsf { s p a r s e } } } ( v _ { i } \neq 0 ) > 0$ .

Note that $\begin{array} { r } { \operatorname* { d e t } ( \mathbb { M } ) = \sum _ { \sigma \in \mathsf { P } _ { P } } \prod _ { i } \mathbb { M } _ { i \sigma ( i ) } } \end{array}$ . Consider the diagonal of M. Consider the situation where all the entries are non-zero. This corresponds to sampling a set of activations of the form $\widehat { U } _ { g }$ . Consider the following random design matrix M.

$$
\mathbb {M} = \left[ \begin{array}{c c c c c c c c} \lambda_ {1} ^ {2} & . & . & \dots & . & \lambda_ {1 2 1} ^ {2} & \dots & . \\ . & \lambda_ {2} ^ {2} & . & \dots & . & \lambda_ {1 2 2} ^ {2} & \dots & \vdots \\ . & . & \lambda_ {3} ^ {2} & \dots & . & . & \dots & \lambda_ {(p - 1) p (p - 1)} ^ {2} \\ \vdots & \dots & \dots & \dots & \lambda_ {p} ^ {2} & \dots & \dots & \lambda_ {(p - 1) p p} ^ {2} \\ \vdots & \dots & \dots & \dots & . & \lambda_ {1 2 1} \lambda_ {1 2 2} & \dots & \dots \\ \vdots & \dots & \dots & \dots & . & . & \dots & \dots \\ . & \dots & \dots & \dots & . & . & . & \lambda_ {(p - 1) p (p - 1)} \lambda_ {(p - 1) p p} \end{array} \right].
$$

Now, a random design matrix M is not identically zero for any set of P randomly sampled activations that satisfy the following indexing property:

$$
\mathcal {R} := \{v: v _ {i} \neq 0, 1 \leq i \leq p \} \cup \{v: v _ {i}, v _ {j} \neq 0, 1 \leq i <   j \leq p \}. \tag {27}
$$

This is so because for identity permutation, we have $\textstyle \prod _ { i } \mathbb { M } _ { i i } \neq 0$ . Now, we will compute the probability that R is sampled from $\mathcal { D } _ { \mathsf { s p a r s e } }$ . Using the independence of sampling of each index of an activation, the probabilities for the two subsets of R can be computed as follows:

• p activations $\{ \alpha _ { 1 } , \alpha _ { 2 } , \cdot \cdot \cdot , \alpha _ { p } \} \sim \mathcal { D } _ { \mathsf { s p a r s e } } ^ { p }$ such that $\alpha _ { i i } \neq 0$ . Using independence, we have

$$
\mathcal {P} _ {1} = \sum_ {i = 0} ^ {s - 1} \binom {p - 1} {i} p _ {n z} ^ {i + 1} (1 - p _ {n z}) ^ {p - 1 - i},
$$

• Rest of $p ( p - 1 ) / 2$ activations of R in Eq. (27) require at least two indices to be non-zero. This could be computed as

$$
\mathcal {P} _ {2} = \sum_ {i = 0} ^ {s - 2} \binom {p - 2} {i} p _ {n z} ^ {i + 2} (1 - p _ {n z}) ^ {p - 2 - i}.
$$

Now, note that these P activations can be permuted in P ! ways and thus

$$
\mathcal {P} _ {\mathcal {V} _ {p}} (\mathcal {V} _ {P} \equiv \mathcal {R}) \geq P! \cdot \mathcal {P} _ {1} ^ {p} \cdot \mathcal {P} _ {2} ^ {(P - p)} = \underbrace {P ! \cdot \left(\sum_ {i = 0} ^ {s - 1} \binom {p - 1} {i} p _ {n z} ^ {i + 1} (1 - p _ {n z}) ^ {p - 1 - i}\right) ^ {p} \left(\sum_ {i = 0} ^ {s - 2} \binom {p - 2} {i} p _ {n z} ^ {i + 2} (1 - p _ {n z}) ^ {p - 2 - i}\right) ^ {(P - p)}} _ {p _ {\mathrm{s}}} \tag {28}
$$

Now, we will complete the proof of the theorem using Hoeffding’s inequality. Assume that the agent samples N activations, we will compute the probability that $\mathcal { R } \subset \mathcal { V } _ { N }$ . Consider all possible P -subsets of N items, enumerated as $\left\{ 1 , 2 , \ldots , \binom { N } { P } \right\}$ . Now, define random variables $X _ { i }$ as

$$
X _ {i} = \left\{ \begin{array}{l} 1 \text {   if   } i \text { th   subset   equals   } \mathcal {R}, \\ 0 \text {   o.w. } \end{array} \right.
$$

Now, define sum random variable $X = \sum _ { i } ^ { \left( { N } \right) } X _ { i }$ = P ( NP )i X . We want to understand the probability $\mathcal { P } _ { \mathcal { V } _ { N } } ( X \ge 1 )$ . Now note that,

$$
\underset {\mathcal {V} _ {N} \sim \mathcal {D} _ {\mathrm{sparse}}} {\mathbb {E}} [ X ] = \sum_ {i} \mathbb {E} [ X _ {i} ] = \binom{N}{P} \cdot \mathcal {P} _ {\mathcal {V} _ {P}} (\mathcal {V} _ {P} \equiv \mathcal {R})
$$

Now, using Hoeffding’s inequality

$$
\mathcal {P} _ {\mathcal {V} _ {N}} (X > 0) \geq 1 - 2 \exp^ {- 2 \mathbb {E} [ X ] ^ {2}} \geq 1 - 2 \exp^ {- 2 \binom {N} {P} ^ {2} p _ {\mathrm{s}} ^ {2}}
$$

Now, for a given choice of of $\delta > 0$ , we want $\delta \geq 2 \exp ^ { - 2 \binom { N } { P } ^ { 2 } p _ { s } ^ { 2 } }$ . Using Sterling’s approximation

$$
\binom {N} {P} \geq \frac {1}{p _ {\mathsf {s}}} \sqrt {\log \frac {4}{\delta^ {2}}} \implies \left(\frac {e N}{P}\right) ^ {P} \geq \frac {1}{p _ {\mathsf {s}}} \sqrt {\log \frac {4}{\delta^ {2}}} \implies N \geq \frac {P}{e} \left(\frac {1}{p _ {\mathsf {s}} ^ {2}} \log \frac {4}{\delta^ {2}}\right) ^ {1 / 2 P}
$$