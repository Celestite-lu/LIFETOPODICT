# Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models

<table><tr><td>Thomas Fel*Kempner InstituteHarvard University</td><td colspan="2">Ekdeep Singh Lubana*CBS-NTT Program in Physics of IntelligenceHarvard University</td><td>Jacob S. PrinceDept. of PsychologyHarvard University</td></tr><tr><td>Matthew KowalFAR AIYork University</td><td>Victor BoutinCerCoCNRS</td><td>Isabel PapadimitriouKempner InstituteHarvard University</td><td>Binxu WangKempner InstituteHarvard University</td></tr><tr><td>Martin Wattenberg†Harvard UniversityGoogle DeepMind</td><td>Demba BaKempner InstituteHarvard University</td><td colspan="2">Talia KonkleDept. of Psychology &amp; Kempner InstituteHarvard University</td></tr></table>

# Abstract

Sparse Autoencoders (SAEs) have emerged as a powerful framework for machine learning interpretability, enabling the unsupervised decomposition of model representations into a dictionary of abstract, human-interpretable concepts. However, we reveal a fundamental limitation: existing SAEs exhibit severe instability, as identical models trained on similar datasets can produce sharply different dictionaries, undermining their reliability as an interpretability tool. To address this issue, we draw inspiration from the Archetypal Analysis framework introduced by Cutler & Breiman (1994) and present Archetypal SAEs (A-SAE), wherein dictionary atoms are constrained to the convex hull of data. This geometric anchoring significantly enhances the stability of inferred dictionaries, and their mildly relaxed variants RA-SAEs further match state-of-the-art reconstruction abilities. To rigorously assess dictionary quality learned by SAEs, we introduce two new benchmarks that test (i) plausibility, if dictionaries recover “true” classification directions and (ii) identifiability, if dictionaries disentangle synthetic concept mixtures. Across all evaluations, RA-SAEs consistently yield more structured representations while uncovering novel, semantically meaningful concepts in large-scale vision models.

![](images/1a305c812c3a9c0d049e8a1ab29f512bea029193a55c182b795f0b99ddbd957b.jpg)

![](images/2155c4ca926ecc2b2534c6ea76c5271748171708a74636b16f2fce51890d33e7.jpg)  
Cosine(u1， u2) = 0.960

Figure 1. A) Archetypal-SAE. Archetypal-SAEs constrain dictionary atoms (decoder directions) to the data’s convex hull, improving stability. A relaxed variant (RA-SAE) allows mild relaxation, matching standard SAEs in reconstruction while maintaining stability. Both integrate with any SAE variant (e.g., TopK, JumpReLU). B) Instability Problem. Standard SAEs produce inconsistent dictionaries across runs, undermining interpretability. For example, in classical SAEs, the second most important concept for “rabbit” in one run has no counterpart in another run (cos = 0.58). In contrast, Archetypal-SAEs maintain consistent concept correspondences across runs, ensuring stability.

# 1. Introduction

Artificial Neural Networks (ANNs) have revolutionized computer vision, setting new benchmarks across a wide range of tasks. Despite these successes, the “black-box” nature of ANNs poses significant challenges, particularly in domains requiring transparency, accountability, and adherence to strict ethical and regulatory standards (Tripicchio & D’Avella, 2020). Beyond compliance, there is a growing curiosity within the scientific community to better understand these models’ internal mechanisms, both to satisfy fundamental questions about their function, leverage insights for debugging (Adebayo et al., 2020) and improvement, and even explore parallels with neuroscience (Goodwin et al., 2022; Vilas et al., 2024). These motivations have spurred the rapid growth of explainable artificial intelligence (XAI), a field dedicated to enhancing the interpretability of ANNs, thereby bridging the gap between machine intelligence and human understanding (Doshi-Velez & Kim, 2017).

Among XAI approaches, concept-based methods (Kim et al., 2018) have emerged as a powerful framework for uncovering intelligible visual concepts embedded within the intricate activation patterns of ANNs (Ghorbani et al., 2019; Zhang et al., 2021; Fel et al., 2023b; Graziani et al., 2023; Vielhaben et al., 2023). These methods excel in making the internal representations of ANNs more comprehensible by associating them with human-interpretable concepts. Recently, concept extraction methods have been shown to be instances of dictionary learning (Fel et al., 2023a), where the goal is to map the activation space of an ANN into a (higher-dimensional), sparse “concept space”. The resulting concept basis is often considered more interpretable. For such representations to be effective, they must be as sparse as possible while still enabling accurate reconstruction of the original activations from the learned basis using a dictionary of atoms—also called prototypes. Historically, dictionary learning methods have included techniques such as Non-negative Matrix Factorization (NMF) (Lee & Seung, 2001; Kowal et al., 2024a; Jourdan et al., 2023) and K-Means (Gersho & Gray, 1991; Ghorbani et al., 2019), while more recent approaches like Sparse Autoencoders (SAEs) (Cunningham et al., 2023; Bricken et al., 2023; Rajamanoharan et al., 2024; Gao et al., 2025; Thasarathan et al., 2025; Poche et al. ´ , 2025) have emerged as a powerful alternative. SAEs achieve a good balance between sparsity and reconstruction quality, and their optimization frameworks scale well to large datasets. However, compared to traditional methods, SAEs suffer from a critical limitation: instability. As illustrated in Figure 1, training two identical SAEs on the same data can yield significantly different dictionaries (concept bases), undermining their reliability and interpretability.

In this work, we address the instability of current SAEs by introducing two novel variants: Archetypal-SAE (A-SAE) and its relaxed counterpart (RA-SAE). Building upon archetypal analysis (Cutler & Breiman, 1994), A-SAE constrains each dictionary atom to reside within the convex hull of the training data, ensuring a more stable and consistent set of basis elements across different training runs by virtue of this geometrical constraint. RA-SAE further extend this framework by incorporating a small relaxation term, allowing for slight deviations from the convex hull to enhance modeling flexibility while maintaining stability. Overall, our work makes the following contributions.

1. Instability in SAEs. We identify a critical limitation of current SAEs paradigms: two training runs on identical data can yield concept dictionaries that are largely distinct, hence compromising their reliability as an interpretability protocol.   
2. A-SAE: Archetypal anchoring to overcome instability. To address the challenge above, we take inspiration from Cutler & Breiman (1994)’s Archetypal analysis of dictionary learning and propose A-SAE, an SAE paradigm wherein the dictionary atoms (decoder directions) are forced to lie in the convex hull of sample representations. As we show, this geometrical anchoring yields substantial stability across training runs. Moreover, we show a mild relaxation of this protocol, which we title RA-SAE, uncovers meaningful and interpretable concepts in large-scale vision models.   
3. Rigorous evaluations with novel metrics and benchmarks. We introduce novel metrics to evaluate the quality of dictionaries learned using different SAE paradigms, while proposing two new benchmarks for testing SAEs’ ability to recover classification directions and to disentangle synthetic image mixtures, inspired by identifiability theory (Locatello et al., 2019; 2020; Higgins et al., 2017). Our results provide substantial evidence that A-SAEs find more structured and coherent concepts. Further, to enable reproduction, we open-source our extensive codebase for large-scale SAE training on modern vision models.

# 2. Related Work

Sparse Coding & Dictionary Learning. Dictionary learning (Tosiˇ c & Frossard´ , 2011; Rubinstein et al., 2010; Elad, 2010; Mairal et al., 2014; Dumitrescu & Irofti, 2018) emerged as a fundamental approach for uncovering latent factors of a data-generating process in signal processing and machine learning, building upon early work in sparse coding (Olshausen & Field, 1996; 1997; Lee et al., 2006; Foldiak & Endres, 2008; Rentzeperis et al., 2023). The primary objective of these methods is to find a sparse representation of input data (Hurley & Rickard, 2009; Eamaz et al., 2022), such that each data sample can be accurately reconstructed using a linear combination of only a small subset of dictionary atoms. The field gained momentum with compressed sensing theory (Donoho, 2006; Candes et al.\` , 2006; Candes & Wakin\` , 2008; Lopes, 2013; Rencker et al., 2019), which established theoretical foundations for sparse signal recovery. Early dictionary learning methods evolved from vector quantization and K-means clustering (Gersho & Gray, 1991; Lloyd, 1982), leading to more sophisticated approaches like Non-negative Matrix Factorization (NMF) and its variants (Lee & Seung, 1999; 2001; Gillis, 2020; Ding et al., 2008; Kersting et al., 2010; Thurau et al., 2009; Gillis & Kumar, 2015), Sparse PCA, (d’Aspremont et al., 2004; Zou et al., 2006) and K-SVD (Aharon et al., 2006a; Elad & Aharon, 2006). The field further expanded rapidly (Wright et al., 2010; Chen et al., 2021; Tasissa et al., 2023) with online methods (Mairal et al., 2009; Kasiviswanathan et al., 2012; Lu et al., 2013) and structured sparsity (Jenatton et al., 2010; Bach et al., 2012; Sun et al., 2014). Theoretical guarantees for dictionary learning emerged (Aharon et al., 2006b; Spielman et al., 2012; Hillar & Sommer, 2015; Fu et al., 2018; Barbier & Macris, 2022; Hu & Huang, 2023), alongside connections to deep learning (Baccouche et al., 2012; Tariyal et al., 2016; Papyan et al., 2017; Mahdizadehaghdam et al., 2019; Tamkin et al., 2023; Yu et al., 2023). Parallel developments in archetypal analysis from Cutler & Breiman (1994) provided complementary perspectives on dictionary learning by focusing on extreme points in a set of observations (Dubins, 1962).

Vision Explainability. Early work in the field of Explainable AI primarily revolved around attribution methods, which highlight the input regions that most influence a model’s prediction (Simonyan et al., 2013; Zeiler & Fergus, 2014; Bach et al., 2015; Springenberg et al., 2014; Smilkov et al., 2017; Sundararajan et al., 2017; Selvaraju et al., 2017; Fong & Vedaldi, 2017; Fel et al., 2021; Novello et al., 2022; Muzellec et al., 2024)—in other words, where the network focuses its attention to produce its prediction. While valuable, attribution methods exhibit two core limitations: (i) they provide limited information about the semantic organization of learned representations (Hase & Bansal, 2020; Hsieh et al., 2021; Nguyen et al., 2021; Colin et al., 2021; Kim et al., 2022; Sixt et al., 2020), and (ii) they can produce incorrect explanations (Adebayo et al., 2018; Ghorbani et al., 2017; Slack et al., 2021; Sturmfels et al., 2020; Hsieh et al., 2021; Hase et al., 2021). In other words, just because the explanations make sense to humans, we cannot conclude that they accurately reflect what is actually happening within the model—as shown by ongoing efforts to develop robust evaluation metrics for explainability (Petsiuk et al., 2018; Bhatt et al., 2020; Jacovi & Goldberg, 2020; Hedstrom et al.¨ , 2022; Fel et al., 2022; Hsieh et al., 2021; Boopathy et al., 2020; Lin et al., 2019; Idrissi et al., 2021).

To overcome the constraints above, concept-based interpretability (Kim et al., 2018) has gained traction. Its central objective is to pinpoint semantically meaningful directions— revealing not just where the model is looking, but also what patterns or concepts it employs—and to link these systematically to latent activations (Bau et al., 2017; Ghorbani et al., 2019; Zhang et al., 2021; Fel et al., 2023b; Graziani et al., 2023; Vielhaben et al., 2023; Kowal et al., 2024a;b). In line with this perspective, Fel et al. (2023a) demonstrate that popular concept-extraction methods : ACE (Ghorbani et al., 2017), ICE (Zhang et al., 2021), CRAFT (Fel et al., 2023b) and more recently SAEs (Cunningham et al., 2023; Bricken et al., 2023; Rajamanoharan et al., 2024; Gao et al., 2025; Surkov et al., 2024; Gorton, 2024; Bhalla et al., 2024a) all address essentially the same dictionary learning task, albeit subject to distinct constraints (see Eq. 1).

Within this broader context, we note Sparse Autoencoders (SAEs) have emerged as a highy scalable special case of dictionary learning: unlike NMF, Sparse-PCA, or other optimization problem, SAEs can be trained with the same algorithms and architectures used in modern deep learning pipelines, making them especially well-suited for largescale concept extraction. However, motivated by more ambitious use-cases of interpretability, e.g., to develop transparency and accountability (Anwar et al., 2024), recent work has started to demonstrate limitations in existing SAE frameworks and propose improvements. Examples of such limitations include learning of overly specific or sensitive features (Bricken et al., 2023; Chanin et al., 2024), challenges in compositionality (Wattenberg & Viegas, 2024; Clarke et al., 2024), and limited effects of latent interventions (Bhalla et al., 2024b; Menon et al., 2024). In this paper, we aim to bring to attention an underappreciated challenge in SAEs’ training –instability – wherein mere reruns of SAE training yield inconsistent interpretations (Figure 1).

# 3. (In)Stability of SAEs

We first establish the challenge of instability in $\operatorname { s A E s } ^ { \prime }$ training. To this end, we start by defining notations and providing background on the SAEs analyzed in this work, then offering a formal measure of instability in SAE training.

Notation. Throughout this work, $| | \cdot | | _ { 2 }$ and $| | \cdot | | _ { F }$ represent the $\ell _ { 2 }$ and Frobenius norms, respectively. We define $[ n ] \ = \ \{ 1 , \ldots , n \}$ , and consider a general representation learning framework where a deep learning model $f : \mathcal { X } \to \mathcal { A }$ maps inputs from an input space X to a representation space $\mathcal { A } \subseteq \mathbb { R } ^ { d }$ . The representation is captured as a set of n points arranged in a matrix $\ b { A } \in \mathbb { R } ^ { n \times d }$ , with $A _ { i }$ denoting its i-th row, where $i \in [ n ]$ and $A _ { i } \in \mathbb { R } ^ { d }$ . For any matrix $X , X \geq 0$ denotes that all elements of $\boldsymbol { X }$ are non-negative, and ${ \mathcal { P } } ( n )$ is the set of $n \times n$ signed permutation matrices. Given $\ b { A } \in \mathbb { R } ^ { n \times d }$ , we define $\mathrm { c o n e } ( A ) =$ $\{ \pmb { x } \ | \ x = A \pmb { v } , \ \pmb { v } \in \mathbb { R } ^ { n } , \ \pmb { v } \geq 0 \}$ , and conv $\mathbf { \partial } ( A ) = \{ \pmb { x } \mid$

![](images/2a7e9a009c1e0e3f90ca2411e3f96925ba6af54592aca2a3c0055023e665358f.jpg)  
● Convex-NMF   
● Semi-NMF   
● Vanilla SAE   
● Top-K SAE   
● JumpReLU SAE

Figure 2. SAEs are a promising direction for scalable concept extraction in vision. Comparison of reconstruction error $( \ell _ { 2 }$ Loss) and sparsity across four large-scale vision models: ConvNext, DINO, SigLIP, and ViT. The figure compares the performance of various dictionary learning methods, including classical approaches (Convex-NMF, Semi-NMF) and modern Sparse Autoencoders (Vanilla SAE, Top-K SAE, JumpReLU SAE). Each SAE is trained up to 250 million tokens per epoch over 50 epochs, demonstrating the scalability of SAEs and their ability to achieve superior trade-offs between reconstruction fidelity and sparsity compared to traditional methods.

${ \pmb x } = A { \pmb v } , \ { \pmb v } \in \mathbb { R } ^ { n } , \ { \pmb v } \geq 0 , \ \mathbf { 1 } ^ { \top } { \pmb v } = 1 \}$ as the conical and convex hulls, respectively, generated by the columns of A.

Concept Extraction as Dictionary Learning. Concept extraction can be naturally framed as a dictionary learning problem, wherein a set of n activations, represented by the matrix $\ b { A } \in \mathbb { R } ^ { n \times d }$ , is approximated using a structured decomposition. The goal is to learn a Dictionary D—also referred to as atoms (Serre, 2006), prototypes, or a codebook (Tamkin et al., 2023)—such that activations can be reconstructed as sparse linear combinations of these learned directions. The corresponding coefficients, known as Codes Z, capture the latent structure of the activations and enforce interpretability by promoting sparsity or nonnegativity. This leads to the general optimization framework:

$$
(\boldsymbol {Z} ^ {\star}, \boldsymbol {D} ^ {\star}) = \underset {\boldsymbol {Z}, \boldsymbol {D}} {\arg \min} | | \boldsymbol {A} - \boldsymbol {Z} \boldsymbol {D} ^ {\mathsf {T}} | | _ {F} ^ {2},
$$

$$
\text { s.t. } \left\{ \begin{array}{l l} \forall i, \mathbf {Z} _ {i} \in \{\boldsymbol {e} _ {1}, \ldots , \boldsymbol {e} _ {k} \}, & (\mathbf {A C E - K - M e a n s}), \\ \mathbf {D} ^ {\top} \mathbf {D} = \mathbf {I}, & (\mathbf {I C E - P C A}), \\ \mathbf {Z} \geq 0, \mathbf {D} \geq 0, & (\mathbf {C R A F T - N M F}), \\ \mathbf {Z} = \mathbf {\Psi} _ {\theta} (\mathbf {A}), | | \mathbf {Z} | | _ {0} \leq K, & (\text { SAEs }). \end{array} \right.
$$

Here, $e _ { i }$ denotes the i-th canonical basis vector, I is the identity matrix, and $\Psi _ { \theta }$ is a neural network parameterized by θ (typically a single feedforward layer with bias). Notably, in Sparse Autoencoder (SAE) literature, the dictionary D is often identified with the decoder weight matrix, denoted as $W _ { \mathrm { d e c } }$ . This optimization framework unifies various classical methods for concept extraction, ranging from clusteringbased approaches (Ghorbani et al., 2019), orthogonal factorization methods (Zhang et al., 2021; Graziani et al., 2021), and nonnegative matrix factorization (Fel et al., 2023b) to modern sparse autoencoding techniques (SAEs) (Cunningham et al., 2023; Bricken et al., 2023).

Despite similar formulations to SAEs, solving K-Means, PCA, or NMF typically does not rely on backpropagation and lacks inherent batch-learning capabilities. This make SAEs particularly appealing for large-scale applications. Additionally, optimization problems solved in multiple steps, such as NMF, Semi-NMF, or K-Means, can be interpreted as having a multi-layer nonlinear encoding (akin to a $\Psi _ { \theta }$ with multiple layers)4. We note that the decoding process in these methods remains linear.

To compare these approaches, we generally evaluate the trade-off between two metrics: sparsity, measured as the $\ell _ { 0 }$ norm of $z ,$ and fidelity, measured as the $\ell _ { 2 }$ reconstruction error. As a starting point, we propose to study this pareto frontier using state-of-the-art SAEs, including Jump-ReLU (Rajamanoharan et al., 2024), TopK (Gao et al., 2025), and a vanilla SAE with $\ell _ { 1 }$ regularization (Bricken et al., 2023), alongside classical sparse dictionary learning methods. Since PCA is non-sparse, K-Means sparsity is fixed, and NMF is applicable only to non-negative activations, we adopt modified versions of these methods that are broadly applicable for concept extraction (Kowal et al., 2024a; Parekh et al., 2024), such as Convex-NMF and Semi-NMF. For SAEs, we use the following standard formulation:

$$
\boldsymbol {\Psi} _ {\theta} (\boldsymbol {A}) = \boldsymbol {\sigma} (\boldsymbol {A} \boldsymbol {W} _ {\theta} + \boldsymbol {b}), \tag {1}
$$

where A denotes a linear encoder layer and $\sigma ( \cdot )$ is an elementwise non-linearity that depends on the specific SAE architecture. For all SAEs, the resulting codes $Z = \Psi _ { \theta } ( A ) \geq$ 0 holds. We employ a Silverman kernel (Silverman, 1984) for instantiating Jump-ReLU SAEs. For vanilla SAEs, an $\ell _ { 1 }$ regularization on Z is applied until the desired sparsity is achieved. Top-K and Jump-ReLU SAEs directly control or optimize an $\ell _ { 0 }$ constraint.

Results in Figure 2 illustrate that the SAE methods discussed above outperform optimization based dictionary learning methods in terms of reconstruction fidelity for fixed sparsity levels. While this positions SAEs as a compelling solution for concept extraction, as we show in the following, a significant drawback of current methods lies in their instability: minor changes to the dataset can lead to substantial variations in the learned dictionary.

Measuring Instability. To formally define a notion of stability in SAEs’ training, we seek inspiration from the dictionary learning literature and prior works (Spielman et al., 2012), yielding the following metric for two dictionaries $D \in \mathbb { R } ^ { n \times d }$ and $D ^ { \prime } ;$ :

$$
\operatorname{Stability} \left(\boldsymbol {D}, \boldsymbol {D} ^ {\prime}\right) = \max _ {\boldsymbol {\Pi} \in \mathcal {P} (n)} \frac {1}{n} \operatorname{Tr} \left(\boldsymbol {D} ^ {\top} \boldsymbol {\Pi} \boldsymbol {D} ^ {\prime}\right), \tag {2}
$$

where we assume that each atom lies on the unit $\ell _ { 2 } { \mathrm { - n o r m } }$ ball, i.e., $| | D _ { i } | | _ { 2 } = 1$ for all $i \in [ n ]$ . This metric essentially measures the optimal average cosine similarity between the dictionaries after finding the best alignment via the Hungarian algorithm5.

We evaluate stability across various vision models by altering only the random seed of the algorithm, while keeping the dataset unchanged across 4 runs. A similar trend is observed when the dataset is perturbed by as little as 5% or 10%, consistent with prior findings (Fel et al., 2023a; Braun et al., 2024; Paulo & Belrose, 2025). Results in Figure 3 show that while SAEs outperform classical dictionary learning methods in terms of reconstruction error $( \ell _ { 2 } { \mathrm { - } } 1 \mathrm { o s s } )$ , they exhibit lower stability, with cosine stability values around 0.5 for TopK SAE trained on DinoV2 with over 250 million tokens. As a first approximation, this implies that re-running the same algorithm with a different seed can result in dictionaries where only half the concepts remain, while the other half are new orthogonal concepts. By contrast, the proposed Archetypal SAE, introduced below, achieves stability comparable to classical dictionary learning methods without compromising reconstruction fidelity.

# 4. Towards Archetypal SAEs

In their seminal work, Cutler & Breiman (1994) proposed representing each data point as a convex combination of Archetypes, which are themselves defined as convex combinations of data points. Concretely, the dictionary (the collection of Archetypes) can be constructed by multiplying the data by a row-stochastic matrix. Drawing on these ideas, we propose a solution to the problem of instability in SAEs’ training: the Archetypal SAE (A-SAE), which acts as a plug-and-play parameterization of the dictionary— i.e., the decoder matrix—and can be seamlessly integrated into any existing SAE, e.g., TopK (Gao et al., 2025) or Jump-ReLU (Rajamanoharan et al., 2024).

Formulation. Let $\pmb { A } \in \mathbb { R } ^ { n \times d }$ represent the data matrix (with n data points in $\mathbb { R } ^ { d } )$ , and $\Delta ^ { n } = \{ \pmb { x } \in \mathbb { R } ^ { n } \ | \ x _ { i } \geq$ $0 , \mathbf { 1 } ^ { \mathsf { T } } { \pmb x } = 1 \}$ denote the (n−1)-dimensional simplex in Rn. $\mathbf { A }$ matrix $\dot { W } \in \mathbb { R } ^ { k \times n }$ is row-stochastic if each row $W _ { i }$ belongs to $\Delta ^ { n }$ . Define the set of row-stochastic matrices as

$$
\Omega_ {k, n} \triangleq \{\boldsymbol {W} \in \mathbb {R} ^ {k \times n} \mid \boldsymbol {W} \geq 0, \boldsymbol {W} \mathbf {1} _ {n} = \mathbf {1} _ {k} \}. \tag {3}
$$

An archetypal dictionary D is then defined as follows.

$$
\boldsymbol {D} = \boldsymbol {W} \boldsymbol {A}, \quad \text { where } \boldsymbol {W} \in \Omega_ {k, n}. \tag {4}
$$

Hence, each row of D is a convex combination of the rows of A, ensuring that each archetype originates from the data.

Geometric Interpretation. In standard SAEs (and most dictionary learning approaches), the dictionary D is free in the sense that each atom $D _ { i } \in \mathbb { R } ^ { d }$ can be placed anywhere in the ambient space. From a geometric perspective, this flexibility allows the reconstructions ZD to span regions that may exceed the convex hull of the data A. While this unconstrained setting enables greater expressivity, it comes with significant drawbacks. Specifically, small perturbations in the data or random initializations can lead to unstable solutions, resulting in dictionaries that differ drastically across training runs. Moreover, if the dictionary atoms $\mathbf { \nabla } D _ { i }$ point in directions unrelated to the data, probing these directions may fail to activate any meaningful mechanisms within the underlying model (Makelov et al., 2023). This highlights the importance of ensuring that the dictionary aligns with “real” directions inherent to the data.

In contrast to above, the Archetypal dictionary imposes a crucial geometric restriction: every dictionary atom is constrained to lie within the convex hull of A, i.e., each dictionary atom $\mathbf { \nabla } D _ { i }$ is a convex combination of samples’ representations. Thus, once multiplied by a nonnegative $z ,$ the reconstructions ZD remain within the conic hull of A. This anchoring within the data manifold precludes the emergence of pathological or out-of-sample directions, yielding stability gains shown empirically in Figure 3 and formalized in Proposition F.2. Concretely, we always have:

$$
\boldsymbol {D} \in \operatorname{conv} (\boldsymbol {A}), \quad \boldsymbol {Z D} \in \operatorname{cone} (\boldsymbol {A}).
$$

![](images/a32c6caa7c735b250b3cd00b1481edffd514bd08edb3d651e22b2e675ec8ae84.jpg)  
● Convex-NMF   
● Semi-NMF   
● Vanilla SAE   
● Top-K SAE   
JumpReLUSAE   
● Archetypal SAE

Figure 3. Stability-Reconstruction tradeoff (optimal: top-left). We implement 5 dictionary learning methods on 4 models at 5 levels of sparsity each, as well as our A-SAE method. We show that SAEs exhibit instability (minor perturbations in the dataset can lead to significant changes in the learned dictionary), while traditional dictionary learning methods are more stable but worse at reconstructing the data. Archetypal-SAEs (ours) help mitigate this issue. We measure stability based on Equation (2): the optimal average cosine similarity between the dictionaries across 4 runs after finding the best alignment via the Hungarian algorithm. Archetypal-SAEs improve stability without compromising reconstruction fidelity, performing better on the stability-reconstruction tradeoff than existing methods.

Moreover, one can restrict D to be formed from a subset $C \subseteq \mathbb { R } ^ { n ^ { \prime } \times d }$ (rather than all of A) without losing expressivity, provided C contains the extreme points of A (Dubins, 1962). Indeed, in that case cone(C) = cone(A), ensuring the same representational power (see Proposition F.1).

We must now ask the core question that makes SAEs exciting: does an SAE with an Archetypal dictionary, as defined above, scale? Indeed, optimizing the matrix W of size $k \times n$ , where n is the number of points and k is the number of concepts, is often infeasible in practice (e.g., when the number of tokens $n > 1 0 ^ { 8 } )$ . This limitation speaks to the importance of the matrix C, which can either be a subset of A or elements within conv(A), such as mixtures of points. Specifically, accessing extreme points is necessary to achieve perfect reconstruction (Simon, 2011), but this is also intractable in practice for such high dimensions. However, we propose to address this with a relaxation.

Scaling Archetypal-SAE. To maintain the desirable properties of A-SAE while addressing scalability, we propose using a reduced subset of points $^ { C , }$ with $n ^ { \prime } \ll n ,$ , chosen as centroids of A obtained via K-Means. We fix C and train only W . We apply a ReLU activation and normalize each row of W at each step to ensure that $W \in \Omega _ { k , n }$ . Experiments indicate that K-Means forms the most reliable method for distilling A into C (see Figure 11), compared to alternatives such as isolation forests (Liu et al., 2008), convex hull computation in reduced dimensions, or outlier detection methods (Scholkopf et al., 1999; Breunig et al., 2000) (details in Appendix D). Hereafter, we use A-SAE to refer to this implementation where we directly optimize W to

![](images/560f30c9ac1db68fa37d13b0510d8669b095e25ff6c9cc7d9771c695ee5e83e0.jpg)  
. Baseline   
. δ=0.0   
. δ=0.5   
● δ=1.0   
δ=2.0

Figure 4. Impact of the Relaxation Parameter (δ). Enumerating extreme points is infeasible in practice; therefore, we introduce a small relaxation parameter (δ) that allows exploration beyond the convex hull of C. The magnitude of this relaxation enables the Archetypal SAE to achieve performance comparable to the unconstrained TopK SAE denoted as Baseline (left) while maintaining excellent stability (right).

find the best convex combination of points that reconstruct data. To enable a controlled degree of flexibility beyond conv(C), we introduce a mild relaxation term $\pmb { \Lambda } \in \mathbb { R } ^ { k \times d }$ , a matrix of the same dimensions as the dictionary, with a small norm constraint $| | \mathbf { A } | | _ { 2 } ^ { 2 } \leq \delta$ . This leads to a relaxed formulation, which we call the Relaxed Archetypal SAE (RA-SAE). Unlike standard A-SAEs, RA-SAE learns both the convex combination weights W and the relaxation term Λ, while ensuring that deviations from the convex hull remain minimal. Formally, we define the dictionary as:

$$
\boldsymbol {D} = \boldsymbol {W} \boldsymbol {C} + \boldsymbol {\Lambda}, \quad s. t. \quad \boldsymbol {W} \in \Omega_ {k, n} \text {and} | | \boldsymbol {\Lambda} | | _ {2} ^ {2} \leq \delta .
$$

Here, only W and Λ are trainable parameters, as detailed in the pseudocode (Figure 5). This implementation ensures that W remains row-stochastic and that the deviation term Λ stays within the prescribed norm constraint. As shown in Figure 4, RA-SAE achieves reconstruction performance on par with conventional Top-K SAEs while maintaining the stability benefits of the archetypal constraint.

```python
class ArchetypalDictionary(nn.Module):
    def __init__(self, C, k, δ=1.0):
    super().__init__()
    n', d = C.shape
    self.register_buffer("C", C)
    self.W = nn.Parameter(torch.eye(k, n'))
    self.A = nn.Parameter(torch.zeros(k, d))
    self.δ = δ

def forward(self, Z):
    with torch.no_grad():
    W = torch.relu(self.W)
    W /= W.sum(dim=-1, keepdim=True)
    Λ *= torch.clamp(
    self.δ / Λ.norm(dim=-1, keepdim=True),
    max=1
    )
    D = W @ self.C + Λ
    return Z @ D 
```  
Figure 5. Pseudocode for Relaxed Archetypal SAE (RA-SAE). This implementation ensures that dictionary atoms remain close to convex hull of the data conv(C) while allowing controlled deviations for better flexibility.

# 5. Experiments

This section is organized into five parts. We begin by describing the experimental setup in detail. Then, we introduce a novel set of theoretical metrics designed to better understand the differences between optimization-based dictionary learning methods, standard SAEs, and the proposed Archetypal SAEs. Following this, we present a new benchmark inspired by identifiability theory, which serves to evaluate the plausibility and uniqueness of the learned representations. We then propose a second, more practical benchmark to assess whether the models can retrieve “true” directions or concepts effectively utilized by the models. Finally, we conclude with qualitative examples showcasing the concepts discovered by SAEs, particularly when applied to the DI-NOv2 model, providing insight into the interpretability and utility of the learned representations.

Setup. We evaluate five models: DINOv2 (Darcet et al., 2023; Oquab et al., 2023), SigLip (Zhai et al., 2023), ViT (Dosovitskiy et al., 2020), ConvNeXt (Liu et al., 2022), and ResNet50 (He et al., 2016), sourced from the timm library (Wightman et al., 2019). Unless specified, we trained overcomplete dictionaries with size k = 5× the feature dimension (e.g., 768 × 5 for DINOv2 and 2048 × 5 for ConvNeXt). Models were trained on the ImageNet dataset (∼ 1.28M images), resulting in over 60M tokens per epoch for ConvNeXt (7 × 7 tokens/image) and over 250M tokens per epoch for DINOv2 (14 × 14 patches/image) across 50 epochs. Semi-NMF and Convex-NMF were trained using gradient descent with accumulation and $\ell _ { 1 }$ regularization to control sparsity, while the RA-SAE was applied atop a TopK SAE to maintain consistent sparsity. To compute C, we applied K-Means clustering to the entire dataset, reducing it to 32,000 centroids, which achieved reconstruction error comparable to the unconstrained SAE. The data matrix A was element-wise standardized. All the SAEs were trained using the Overcomplete library6.

# 5.1. Dictionary Learning Metrics

To improve SAEs, it is essential to deepen our understanding of the solutions they yield. We thus evaluate both standard and novel sets of metrics that evaluate dictionary learning methods across four key dimensions: (i) sparse reconstruction, (ii) consistency, (iii) structure in the dictionary (D), and (iv) structure in the codes (Z), all reported in Tab. 1 (see App. C for formal definitions). Overall, we find Archetypal SAEs achieve a strong balance between reconstruction performance, consistency, and identification of structure.

i) Sparse Reconstruction. Prior work (Bricken et al., 2023) commonly assesses the quality of the learned dictionary to reconstruct the original data under sparsity constraints via metrics such as $R ^ { 2 }$ , sparsity $( \ell _ { 0 }$ norm), and the effective usage of dictionary atoms (e.g., dead codes). We find that Archetypal SAEs perform on-par with existing SAEs and outperform NMF methods in reconstruction, achieving higher ${ \bar { R } } ^ { 2 }$ scores for comparable sparsity levels.

ii) Consistency. Excelling in reconstruction does not guarantee that the learned solution aligns with the underlying data distribution or is stable across training runs. To this end, we measure stability, which assesses the consistency of learned dictionaries across different initializations or perturbations in the data, and the OOD score, which quantifies how close the dictionary atoms are to real data points, i.e., whether learned concepts remain grounded and interpretable. Our findings indicate that SAEs perform poorly in consistency, showing both low stability (as evidenced in Figure 3) and suboptimal OOD scores. In contrast, Archetypal SAEs significantly enhance stability and OOD score without sacrificing reconstruction performance. In Proposition F.4, we provide theoretical arguments showing that, under mild assumptions, the OOD score of Archetypal SAEs is inherently lower-bounded, ensuring that learned dictionary atoms remain well-anchored within the data.

Archetypal SAE 

<table><tr><td>Metric</td><td>Van. SAE</td><td>TopK SAE</td><td>Jump SAE</td><td>SNMF</td><td>CNMF</td><td>RA-SAE</td></tr><tr><td> $R^2$ (↑)</td><td>83.94</td><td>89.52</td><td>89.92</td><td>67.43</td><td>55.48</td><td>89.34</td></tr><tr><td>Dead Codes (↓)</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.064</td><td>0.031</td><td>0.02</td></tr><tr><td>Stability (↑)</td><td>0.710</td><td>0.542</td><td>0.539</td><td>0.925</td><td>0.933</td><td>0.927</td></tr><tr><td>Max Cosine (↑)</td><td>0.997</td><td>0.993</td><td>0.994</td><td>0.999</td><td>0.999</td><td>0.999</td></tr><tr><td>OOD Score (↓)</td><td>0.451</td><td>0.551</td><td>0.551</td><td>0.430</td><td>0.087</td><td>0.060</td></tr><tr><td>Stable Rank (↓)</td><td>86.8</td><td>141.6</td><td>128.0</td><td>5.38</td><td>6.65</td><td>5.89</td></tr><tr><td>Eff. Rank (↓)</td><td>363</td><td>372</td><td>371</td><td>186</td><td>289</td><td>310</td></tr><tr><td>Coherence (↓)</td><td>0.838</td><td>0.728</td><td>0.560</td><td>0.999</td><td>0.999</td><td>0.973</td></tr><tr><td>Connectivity (↑)</td><td>0.000</td><td>0.002</td><td>0.003</td><td>0.243</td><td>0.138</td><td>0.159</td></tr><tr><td>Neg. Inter. (↓)</td><td>39.99</td><td>135.7</td><td>243</td><td>0.005</td><td>0.002</td><td>0.012</td></tr></table>

Table 1. Quantitative comparison of the dictionary learning methods on DINOv2, using a 90% sparse, overcomplete dictionary with 2000 concepts. SAE methods achieve the highest reconstruction performance. C-NMF, S-NMF, and Archetypal methods excel in consistency, ensuring stability across runs and that found concepts are close to real data (OOD). Additionally, these methods demonstrate superior dictionary structure (Stable and Eff. Rank) and codes structure (Connectivity and Neg Inter.), indicating patterns in the inferred concepts and structured sparsity.

iii) Structure in the Dictionary (D). This dimension examines whether the learned dictionary exhibits meaningful patterns. A well-structured dictionary may reveal metaconcepts or decomposable higher-level abstractions. Metrics such as stable rank, effective rank, and coherence provide insights into the compactness and interpretability of the dictionary. In particular, among solutions with comparable reconstruction performance, a more structured dictionary is preferable due to its potential for higher interpretability and organization. Across runs, methods like CNMF, SNMF, and Archetypal dictionary learning consistently yield dictionaries with better structure, reflecting their capacity for capturing higher-level patterns within the data. We also provide theoretical arguments in Proposition F.3, demonstrating that the rank of Archetypal dictionaries is inherently bounded by the rank of the data matrix.

iv) Structure in the Codes (Z). The structure in the encoding space is equally important, as it determines how concepts are combined to reconstruct the data. Connectivity, measured as the $\ell _ { 0 }$ norm of $Z Z ^ { \mathsf { T } }$ , reflects the combinatorial diversity of the concepts. High connectivity enables complex reconstructions, while low connectivity highlights structural sparsity and simpler patterns. Additionally, negative interference (Neg. Inter.) quantifies the simultaneous activation of conflicting concepts, which can cancel each other out. Archetypal SAEs and optimization-based dictionary learning approaches reliably produce more structured codes with reduced interference, enhancing the coherence of their representations.

# 5.2. Plausibility Benchmark

We evaluate ${ \mathrm { s A E s } } ^ { \prime }$ ability to recover true classification directions by assessing whether the learned dictionary D aligns with the classifier’s final layer weights $\{ \pmb { v } _ { 1 } , \ldots , \pmb { v } _ { c } \}$ , where c is the number of classes. Specifically, for each class vector $v _ { i } ,$ we compute the most aligned dictionary

<table><tr><td>Dict. size (k)</td><td>512</td><td>1k</td><td>2k</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td></tr><tr><td colspan="8">ConvNeXt</td></tr><tr><td>Baseline (TopK SAE)</td><td>0.1681</td><td>0.1686</td><td>0.1668</td><td>0.1671</td><td>0.1684</td><td>0.1692</td><td>0.1684</td></tr><tr><td>A-SAE ( $\delta = 0$ )</td><td>0.2172</td><td>0.3046</td><td>0.3597</td><td>0.3887</td><td>0.3957</td><td>0.3984</td><td>0.3999</td></tr><tr><td>RA-SAE ( $\delta = 0.01$ )</td><td>0.1973</td><td>0.2887</td><td>0.3581</td><td>0.3900</td><td>0.4007</td><td>0.4038</td><td>0.4045</td></tr><tr><td>RA-SAE ( $\delta = 0.1$ )</td><td>0.1270</td><td>0.1596</td><td>0.2106</td><td>0.2681</td><td>0.3280</td><td>0.3674</td><td>0.3845</td></tr><tr><td>RA-SAE ( $\delta = 1.0$ )</td><td>0.1172</td><td>0.1475</td><td>0.2124</td><td>0.3116</td><td>0.4342</td><td>0.5203</td><td>0.5581</td></tr><tr><td colspan="8">ResNet</td></tr><tr><td>Baseline (TopK SAE)</td><td>0.2295</td><td>0.2484</td><td>0.3055</td><td>0.3203</td><td>0.3301</td><td>0.3014</td><td>0.3125</td></tr><tr><td>A-SAE ( $\delta = 0$ )</td><td>0.5920</td><td>0.5985</td><td>0.5992</td><td>0.6013</td><td>0.6029</td><td>0.6105</td><td>0.6133</td></tr><tr><td>RA-SAE ( $\delta = 0.01$ )</td><td>0.5905</td><td>0.5985</td><td>0.5991</td><td>0.6013</td><td>0.6039</td><td>0.6106</td><td>0.6136</td></tr><tr><td>RA-SAE ( $\delta = 0.1$ )</td><td>0.5777</td><td>0.5932</td><td>0.6002</td><td>0.6039</td><td>0.6046</td><td>0.6067</td><td>0.6083</td></tr><tr><td>RA-SAE ( $\delta = 1.0$ )</td><td>0.6151</td><td>0.6165</td><td>0.6173</td><td>0.6189</td><td>0.6200</td><td>0.6208</td><td>0.6213</td></tr><tr><td colspan="8">ViT</td></tr><tr><td>Baseline (TopK SAE)</td><td>0.1317</td><td>0.1589</td><td>0.1984</td><td>0.2265</td><td>0.2595</td><td>0.2807</td><td>0.2939</td></tr><tr><td>A-SAE ( $\delta = 0$ )</td><td>0.2079</td><td>0.2459</td><td>0.2820</td><td>0.3096</td><td>0.3361</td><td>0.3581</td><td>0.3721</td></tr><tr><td>RA-SAE ( $\delta = 0.01$ )</td><td>0.2102</td><td>0.2490</td><td>0.2861</td><td>0.3132</td><td>0.3200</td><td>0.3642</td><td>0.3821</td></tr><tr><td>RA-SAE ( $\delta = 0.1$ )</td><td>0.2278</td><td>0.2747</td><td>0.3153</td><td>0.3410</td><td>0.3496</td><td>0.3912</td><td>0.4103</td></tr><tr><td>RA-SAE ( $\delta = 1.0$ )</td><td>0.1497</td><td>0.2123</td><td>0.2936</td><td>0.3624</td><td>0.4277</td><td>0.4786</td><td>0.5014</td></tr></table>

Table 2. Plausibility Benchmark Results. The Plausibility Score measures the alignment between the learned dictionary concepts and the classification head’s directions. RA-SAE achieves a significantly higher score compared to a TopK SAE (baseline). Best scores are wrapped in green, and worst scores are wrapped in red.

atom $D _ { j }$ and average the alignment score: Plausibility = 1c Pci=1 maxj ⟨vi, Dj ⟩. A score of 1 indicates perfect align- $\begin{array} { r } { \frac { 1 } { c } \sum _ { i = 1 } ^ { c } \mathrm { \bar { m a x } } _ { j } \langle \pmb { v } _ { i } , \pmb { D } _ { j } \rangle } \end{array}$ ment, while a score of 0 implies that all concepts lie in the classifier’s null space, making probing ineffective. This metric could help us detect potential hallucinations if concepts diverge too much from true classification directions, and is broadly similar to Karvonen et al. (2024); Mayne et al. (2024)’s evaluation of whether SAEs infer known linear features in toy settings. Results are reported in Table 2. We find that classical SAEs, even with extremely large dictionaries, achieve limited alignment with the classification directions (inline with prior work). In contrast, RA-SAE significantly enhances this alignment, recovering a substantial portion of the true classification directions. This demonstrates RA-SAE’s efficacy in producing semantically meaningful dictionaries.

![](images/4f8f3a99141644664860664d3e4c7778338ba29d8239804afd04248cbdd3d3ca.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["Image Collection"]
    B --> C["Visual encoder"]
    C --> D["SAE"]
    D --> E["Output: X₁, Z₁ with matching grid"]
```
</details>

1.Images (X)are generated from a ground truth generating process involving 9 objects forming a hidden dictionary.The model never directly observes this dictionary but only sees the resulting images.   
2.The visual encoder extracts meaningful internal representations (A) from the input images.   
3.SAEs processes the activations (A） to produce codes Z,containing 9 distinct concepts for each images.The ideal code identifies the true structure of the generative process, disentangling A into interpretable components.

Figure 6. Soft Identifiability benchmark. This example uses the “gems” dataset, part of the 12 identifiability benchmarks we introduce. The goal is to evaluate whether SAEs (or any dictionary learning method under study) can disentangle and recover each object from the hidden ground truth generative process. By analyzing the model’s ability to assign distinct codes to the underlying concepts, we test its capacity to reconstruct and interpret the true structure of the data. 

<table><tr><td>Method</td><td>DINOv2</td><td>ResNet</td><td>SigLIP</td><td>ViT</td></tr><tr><td>KMeans</td><td>0.7678</td><td>0.7624</td><td>0.7684</td><td>0.7702</td></tr><tr><td>ICA</td><td>0.8092</td><td>0.8370</td><td>0.8243</td><td>0.8267</td></tr><tr><td>Sparse PCA</td><td>0.7981</td><td>0.8318</td><td>0.8069</td><td>0.8082</td></tr><tr><td>SVD</td><td>0.7979</td><td>0.8291</td><td>0.8062</td><td>0.8075</td></tr><tr><td>SemiNMF</td><td>0.8297</td><td>0.8327</td><td>0.8358</td><td>0.8423</td></tr><tr><td>ConvexNMF</td><td>0.7645</td><td>0.7582</td><td>0.7639</td><td>0.7634</td></tr><tr><td>PCA</td><td>0.7979</td><td>0.8291</td><td>0.8062</td><td>0.8075</td></tr><tr><td>Vanilla</td><td>0.8047</td><td>0.8167</td><td>0.8126</td><td>0.8223</td></tr><tr><td>TopK</td><td>0.8135</td><td>0.8150</td><td>0.8289</td><td>0.8328</td></tr><tr><td>Jump</td><td>0.8010</td><td>0.7988</td><td>0.8131</td><td>0.8053</td></tr><tr><td>A-SAE</td><td>0.9482</td><td>0.9631</td><td>0.9602</td><td>0.9615</td></tr><tr><td>RA-SAE</td><td>0.9447</td><td>0.9602</td><td>0.9585</td><td>0.9586</td></tr></table>

Table 3. Soft Identifiability benchmark Across Models and Methods. This table presents the average accuracy scores for various methods evaluated across four different models: DINO, ResNet, SigLIP, and ViT. Best-performing scores for each model are in bold and second best are underlined. Full results are available in Appendix H.

# 5.3. Soft Identifiability Benchmark

Recent work on disentangled representation learning often evaluates whether an autoencoder trained to reconstruct samples from a data-generating process learns to represent its underlying concepts (Locatello et al., 2019; 2020; Von Kugelgen et al., 2021; Gresele et al., 2020; Khemakhem et al., 2020; Schott et al., 2021; Zimmermann et al., 2021; Menon et al., 2024). Identifiability theorems on the topic (Locatello et al., 2019; 2020) have however shown that unless the autoencoding architecture possesses the right inductive biases that match the generative process, there is no guarantee concepts underlying the data will map onto the autoencoder’s latents. Since these results do not make assumptions about the data modality, they remain valid for a standard SAE training setup, e.g., similar to our experiments above. Then, an intriguing experiment involves evaluating whether when trained on representations of samples from a toy datagenerating process with predefined concepts, A-SAE, or any other SAE architecture, develops latents capturing said concepts—if it does, then that is strongly suggestive of the SAE possessing the right inductive biases, i.e., it captures the mechanism via which a model encodes concepts in its representations.

Motivated by the above, we propose a Soft Identifiability Benchmark. Specifically, we construct twelve synthetic datasets, each comprising images formed by collaging four distinct objects (e.g., different types of gems) sampled from a pre-defined set. Each dataset is processed through a pretrained vision model to obtain pooled activations. Ideally, when trained on these activations, the SAE is able to recover the original objects as distinct concepts within its dictionary. We then assess performance by checking whether each object class has a corresponding concept that activates above a threshold λ when an object $y _ { j }$ appears in the image. Formally, for each image, we feed it into a model and then into the trained SAE to get a concept-label pair $( z , y )$ , where $z \in \mathbb { R } ^ { k }$ represents the k concept values and $\ b { y } \in \mathbb { R } ^ { c }$ denotes the c class labels. We then define the accuracy for the class j as: $\begin{array} { r } { \operatorname { A c c u r a c y } _ { j } = \operatorname* { m a x } _ { \lambda \in \mathbb { R } , i \in [ k ] } \mathbb { P } _ { ( z , \pmb { y } ) } ( ( z _ { i } > \lambda ) = y _ { j } ) } \end{array}$ .

# 5.4. Qualitative Examples

This section provides qualitative insights into the concepts learned by DinoV2 (Oquab et al., 2023; Darcet et al., 2023). We trained an Archetypal SAE on DinoV2-B with 4 registers and reported key performance metrics in Figure 10. RA-SAE uncovers unexpected concepts, such as shadowbased features (potentially linked to depth reasoning), a context-dependent “barber” concept (activating for barbers but not clients), and fine-grained edge detection in flower petals (Figure 9). It also learns more structured within-class distinctions (e.g., separating rabbit ears, faces, and paws) compared to TopK-SAEs (Figure 8). Finally, its dictionary forms clear clusters, grouping semantically related features like fine-grained animal faces or spatial concepts such as

![](images/5739d1322a39134c189ef89cbc3b619e114fb6f6feeccffb43867d05d101f2f7.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D facial images and a heatmap visualization overlay, showing various animal and object categories (no text or symbols)
</details>

Figure 7. Examples of 3 Concept Clusters in DinoV2. Each cluster contains 4 example concepts. • Complex hand positions, ranging from hands in pockets to hands on another person. • Abstract “under” concepts, linking animals and objects, such as birds, zebras, felines, and airplanes, while focusing on lower regions. • Fine-grained animal facial features, including ears, eyebrows, and cheeks.

# “underneath” (Figure 7).

To explore the dictionary learned by the model, we analyzed three clusters of concepts, as illustrated in Figure 7. Specifically: • A cluster of concepts capturing complex hand positions, ranging from hands in pockets to hands resting on another person (e.g., on a shoulder). • An abstract “under” concept cluster, linking entities such as birds, zebras, felines, and airplanes, while highlighting the lower regions of objects. • A cluster representing fine-grained facial concepts in animals, including ears, eyebrows, and cheeks.

In addition, we identified surprising and specific concepts among the 16, 000 dictionary atoms, shown in Figure ??. These include: A) A concept highlighting tokens corresponding to shadows of dogs, suggesting that DinoV2 may use shadows as a feature, potentially contributing to its depth estimation or 3D reasoning capabilities. B) A “barber” concept that activates exclusively on tokens representing barbers but not on the individuals receiving haircuts or shaves. C) A fine-grained visual concept that activates along the contours or edges of flower petals.

Finally, we present the top-5 concepts associated with the rabbit class on DinoV2 in Figure 8. The concepts learned by RA-SAE are distinct and exhibit greater structure compared to their TopK counterparts. For instance, RA-SAE successfully identifies separate concepts for rabbit ears, body, face, and paws, demonstrating its ability to disentangle finegrained features within a class. These results suggest that RA-SAE provides a more organized and meaningful decomposition of concepts.

![](images/b78cc8f3145010783883c755ee9c81898b04762e6747bd3be245c298988667be.jpg)  
Figure 8. Top-5 Concepts for the Rabbit Class in DinoV2. The RA-SAE on top of TopK identifies distinct and fine-grained concepts, including rabbit ears, body, face, and paws. These concepts exhibit greater structure and granularity compared to those found by the unconstrained TopK SAE (baseline) method.

# 6. Conclusion

We identified a fundamental instability in Sparse Autoencoders, where identical training runs can yield divergent dictionaries, limiting their reliability for interpretability. To address this, we introduced Archetypal SAEs (A-SAE), which constrain dictionary atoms to the convex hull of the data, significantly enhancing stability while preserving expressivity. We further proposed a relaxed variant, RA-SAE, which balances reconstruction quality with meaningful concept discovery in large-scale vision models. To rigorously

assess these models, we developed novel evaluation metrics and benchmarks inspired by identifiability theory, providing a systematic framework for measuring dictionary quality and disentanglement. Our findings suggest that archetypal constraints not only stabilize SAEs but also improve the structure and plausibility of learned representations. Beyond vision, our approach lays the groundwork for more reliable concept discovery in broader modalities, including large language models and other structured data domains.

# Impact Statement

This work confronts a critical obstacle in sparse interpretability methods: instability. Archetypal SAEs enforce geometric constraints that yield dictionaries more consistent across runs, more grounded in the data manifold, and better aligned with true classification and generative directions. These advances enhance both the reliability and scientific utility of sparse autoencoders, enabling stable concept discovery at scale in large vision models. Our benchmarks provide a principled framework for evaluating concept plausibility and identifiability, supporting reproducible and trustworthy progress in interpretability research.

A）  
![](images/efe9ee583b34f8b9dbdd5c286d3aab676df537a20c8dc849965b6772aff0f36a.jpg)

<details>
<summary>natural_image</summary>

Grid of nine image processing images showing animals and corresponding thermal or segmentation overlays (no text or symbols)
</details>

![](images/7055270bb4affa02a2e603b5cd5f665f08df74975db9c220ae382d9619e71bf6.jpg)

<details>
<summary>natural_image</summary>

Collage of nine photos showing elderly men in various settings, including hair styling, facial recognition, and color-coded heatmaps (no text or symbols)
</details>

![](images/f35d2abd645804e68136d5a5dbd0960b085c367457461ddc14e765f56a086684.jpg)

<details>
<summary>natural_image</summary>

Collage of flower images including a bee, a ladybug, and various flowers with pixelated textures (no text or symbols)
</details>

Figure 9. Exotic Concepts in DinoV2. A) Highlights tokens in shadows of dogs, suggesting shadow-based features and potential depth reasoning. B) A “barber” concept exclusively active for barbers, not their clients. C) A fine-grained concept focusing on petal edges or contours.

# Acknowledgments

This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University. MW acknowledges support from a Superalignment Fast Grant from OpenAI, Effective Ventures Foundation, Effektiv Spenden Schweiz, and the Open Philanthropy Project.

# References

Abrol, V. and Sharma, P. A geometric approach to archetypal analysis via sparse projections. In International Conference on Machine Learning (ICML), 2020.   
Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. Sanity checks for saliency maps. Advances in Neural Information Processing Systems (NIPS), 2018.   
Adebayo, J., Muelly, M., Liccardi, I., and Kim, B. Debugging tests for model explanations. Advances in Neural Information Processing Systems (NeurIPS), 2020.   
Aharon, M., Elad, M., and Bruckstein, A. K-svd: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on Signal Processing, 2006a.   
Aharon, M., Elad, M., and Bruckstein, A. M. On the uniqueness of overcomplete dictionaries, and a practical way to retrieve them. Linear Algebra and its Applications, 2006b.   
Anwar, U., Saparov, A., Rando, J., Paleka, D., Turpin, M., Hase, P., Lubana, E. S., Jenner, E., Casper, S., Sourbut, O., et al. Foundational challenges in assuring alignment and safety of large language models. ArXiv e-print, 2024.   
Arora, S., Ge, R., and Moitra, A. Learning topic models – going beyond svd. In IEEE Symposium on Foundations of Computer Science (FOCS), 2013.   
Baccouche, M., Mamalet, F., Wolf, C., Garcia, C., and Baskurt, A. Spatio-temporal convolutional sparse autoencoder for sequence classification. Proceedings of the British Machine Vision Conference (BMVC), 2012.   
Bach, F., Jenatton, R., Mairal, J., and Obozinski, G. Structured sparsity through convex optimization. Statistical Science, 2012.   
Bach, S., Binder, A., Montavon, G., Klauschen, F., Muller, ¨ K.-R., and Samek, W. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. Public Library of Science (PloS One), 2015.   
Barbier, J. and Macris, N. Statistical limits of dictionary learning: random matrix theory and the spectral replica method. Physical Review E, 2022.   
Bau, D., Zhou, B., Khosla, A., Oliva, A., and Torralba, A. Network dissection: Quantifying interpretability of deep visual representations. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Bauckhage, C. and Manshaei, K. Kernel archetypal analysis for clustering web search frequency time series. In International Conference on Pattern Recognition (ICPR), 2014.   
Bauckhage, C., Kersting, K., Hoppe, F., and Thurau, C. Archetypal analysis as an autoencoder. In Neural Computation Workshop.   
Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F. P., and Lakkaraju, H. Interpreting clip with sparse linear concept embeddings (splice). ArXiv e-print, 2024a.   
Bhalla, U., Srinivas, S., Ghandeharioun, A., and Lakkaraju, H. Towards unifying interpretability and control: Evaluation via intervention. ArXiv e-print, 2024b.   
Bhatt, U., Weller, A., and Moura, J. M. F. Evaluating and aggregating feature-based model explanations. Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2020.   
Boopathy, A., Liu, S., Zhang, G., Liu, C., Chen, P.-Y., Chang, S., and Daniel, L. Proper network interpretability helps adversarial robustness in classification. Proceedings of the International Conference on Machine Learning (ICML), 2020.   
Boyd, S. and Vandenberghe, L. Convex optimization. 2004.   
Braun, D., Taylor, J., Goldowsky-Dill, N., and Sharkey, L. Identifying functionally important features with end-toend sparse dictionary learning. ArXiv e-print, 2024.   
Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. Lof: identifying density-based local outliers. ACM SIGMOD International Conference on Management of Data, 2000.   
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.   
Candes, E. J. and Wakin, M. B. An introduction to com-\` pressive sampling. IEEE Signal Processing Magazine, 2008.   
Candes, E. J., Romberg, J., and Tao, T. Robust uncertainty \` principles: Exact signal reconstruction from highly incomplete frequency information. IEEE Transactions on Information Theory, 2006.   
Canhasi, E. and Kononenko, I. Weighted hierarchical archetypal analysis for multi-document summarization. Computer Speech & Language, 2016.

Chan, B., Mitchell, D., and Cram, L. Archetypal analysis of galaxy spectra. Monthly Notices of the Royal Astronomical Society (MNRAS), 2003.   
Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., and Bloom, J. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. ArXiv e-print, 2024.   
Chen, J., Mao, H., Wang, Z., and Zhang, X. Low-rank representation with adaptive dictionary learning for subspace clustering. Knowledge-Based Systems, 2021.   
Chen, Y., Mairal, J., and Harchaoui, Z. Fast and robust archetypal analysis for representation learning. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014.   
Clarke, M., Bhatnagar, H., and Bloom, J. Compositionality and ambiguity: Latent co-occurrence and interpretable subspaces. LessWrong, 2024.   
Colin, J., Fel, T., Cadene, R., and Serre, T. What i cannot \` predict, i do not understand: A human-centered evaluation framework for explainability methods. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. ArXiv e-print, 2023.   
Cutler, A. and Breiman, L. Archetypal analysis. Technometrics, 1994.   
Cutler, A. and Stone, E. Moving archetypes. Physica D: Nonlinear Phenomena, 1997.   
Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. Vision transformers need registers. ArXiv e-print, 2023.   
d’Aspremont, A., Ghaoui, L., Jordan, M., and Lanckriet, G. A direct formulation for sparse pca using semidefinite programming. Advances in Neural Information Processing Systems (NeurIPS), 2004.   
Dawson, J. A. and Kendziorski, C. Survival-supervised latent dirichlet allocation models for genomic analysis of time-to-event outcomes. arXiv, 2012.   
Ding, C. H., Li, T., and Jordan, M. I. Convex and seminonnegative matrix factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2008.   
Ding, W., Ishwar, P., and Saligrama, V. A provably efficient algorithm for separable topic discovery. Arxiv, 2016.   
Donoho, D. L. Compressed sensing. IEEE Transactions on Information Theory, 2006.

Doshi-Velez, F. and Kim, B. Towards a rigorous science of interpretable machine learning. ArXiv e-print, 2017.   
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. Proceedings of the International Conference on Learning Representations (ICLR), 2020.   
Dubins, L. E. On extreme points of convex sets. Journal of Mathematical Analysis and Applications, 1962.   
Dumitrescu, B. and Irofti, P. Dictionary learning algorithms and applications. 2018.   
Eamaz, A., Yeganegi, F., and Soltanalian, M. On the building blocks of sparsity measures. IEEE Signal Processing Letters, 2022.   
Elad, M. Sparse and redundant representations: from theory to applications in signal and image processing. 2010.   
Elad, M. and Aharon, M. Image denoising via sparse and redundant representations over learned dictionaries. IEEE Transactions on Image Processing, 2006.   
Epifanio, I., Ibanez, V., and Sim ˜ o, A. Archetypal analysis ´ with missing data: see all samples by looking at a few based on extreme profiles. The American Statistician, 2019.   
Eugster, M. and Leisch, F. Weighted and robust archetypal analysis. Computational Statistics & Data Analysis, 2011.   
Fel, T., Cadene, R., Chalvidal, M., Cord, M., Vigouroux, D., and Serre, T. Look at the variance! efficient blackbox explanations with sobol-based sensitivity analysis. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Fel, T., Hervier, L., Vigouroux, D., Poche, A., Plakoo, J., Cadene, R., Chalvidal, M., Colin, J., Boissin, T., Bethune, L., Picard, A., Nicodeme, C., Gardes, L., Flandin, G., and Serre, T. Xplique: A deep learning explainability toolbox. Workshop on Explainable Artificial Intelligence for Computer Vision (CVPR), 2022.   
Fel, T., Boutin, V., Moayeri, M., Cadene, R., Bethune, L., Chalvidal, M., and Serre, T. A holistic approach to unifying automatic concept extraction and concept importance estimation. Advances in Neural Information Processing Systems (NeurIPS), 2023a.   
Fel, T., Picard, A., Bethune, L., Boissin, T., Vigouroux, D., Colin, J., Cadene, R., and Serre, T. Craft: Concept \` recursive activation factorization for explainability. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023b.

Foldiak, P. and Endres, D. M. Sparse coding. ArXiv e-print, 2008.   
Fong, R. C. and Vedaldi, A. Interpretable explanations of black boxes by meaningful perturbation. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.   
Fotiadou, E., Panagakis, Y., and Pantic, M. Temporal archetypal analysis for action segmentation. In IEEE Conference on Automatic Face and Gesture Recognition (FG), 2017.   
Fu, X., Huang, K., and Sidiropoulos, N. D. On identifiability of nonnegative matrix factorization. IEEE Signal Processing Letters, 2018.   
Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. Proceedings of the International Conference on Learning Representations (ICLR), 2025.   
Gersho, A. and Gray, R. M. Vector quantization and signal compression. 1991.   
Ghorbani, A., Abid, A., and Zou, J. Interpretation of neural networks is fragile. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2017.   
Ghorbani, A., Wexler, J., Zou, J. Y., and Kim, B. Towards automatic concept-based explanations. Advances in Neural Information Processing Systems (NeurIPS), 2019.   
Gillis, N. Nonnegative matrix factorization. 2020.   
Gillis, N. and Kumar, A. Exact and heuristic algorithms for semi-nonnegative matrix factorization. SIAM Journal on Matrix Analysis and Applications, 2015.   
Gimbernat-Mayol, J., Dominguez Mantes, A., Bustamante, C. D., Mas Montserrat, D., and Ioannidis, A. G. Archetypal analysis for population genetics. PLoS Computational Biology, 2022.   
Goodwin, N. L., Nilsson, S. R., Choong, J. J., and Golden, S. A. Toward the explainability, transparency, and universality of machine learning for behavioral classification in neuroscience. Current Opinion in Neurobiology, 2022.   
Gorton, L. The missing curve detectors of inceptionv1: Applying sparse autoencoders to inceptionv1 early vision. ArXiv e-print, 2024.   
Graziani, M., Palatnik de Sousa, I., Vellasco, M. M., Costa da Silva, E., Muller, H., and Andrearczyk, V. Sharp- ¨ ening local interpretable model-agnostic explanations for histopathology: improved understandability and reliability. Medical Image Computing and Computer Assisted Intervention, 2021.

Graziani, M., Nguyen, A.-p., O’Mahony, L., Muller, H., and ¨ Andrearczyk, V. Concept discovery and dataset exploration with singular value decomposition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.   
Gresele, L., Rubenstein, P. K., Mehrjou, A., Locatello, F., and Scholkopf, B. The incomplete rosetta stone problem: Identifiability results for multi-view nonlinear ica. Uncertainty in Artificial Intelligence, 2020.   
Hase, P. and Bansal, M. Evaluating explainable ai: Which algorithmic explanations help users predict model behavior? Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2020.   
Hase, P., Xie, H., and Bansal, M. The out-of-distribution problem in explainability and search methods for feature importance explanations. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.   
Hedstrom, A., Weber, L., Bareeva, D., Motzkus, F., Samek, ¨ W., Lapuschkin, S., and Hohne, M. M.-C. Quantus: an ¨ explainable ai toolkit for responsible evaluation of neural network explanations. The Journal of Machine Learning Research (JMLR), 2022.   
Higgins, I., Matthey, L., Pal, A., Burgess, C. P., Glorot, X., Botvinick, M. M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. Proceedings of the International Conference on Learning Representations (ICLR), 2017.   
Hillar, C. J. and Sommer, F. T. When can dictionary learning uniquely recover sparse data from subsamples? IEEE Transactions on Information Theory, 2015.   
Hinrich, J., Bardenfleth, S., Røge, R., Churchill, N., Madsen, K., and Mørup, M. Archetypal analysis for modeling multi-subject fmri data. IEEE Journal of Selected Topics in Signal Processing, 2016.   
Hsieh, C.-Y., Yeh, C.-K., Liu, X., Ravikumar, P., Kim, S., Kumar, S., and Hsieh, C.-J. Evaluations and methods for explanation through robustness analysis. Proceedings of the International Conference on Learning Representations (ICLR), 2021.   
Hu, J. and Huang, K. Global identifiability of l1-based dictionary learning via matrix volume optimization. Advances in Neural Information Processing Systems (NeurIPS), 2023.

Hurley, N. and Rickard, S. Comparing measures of sparsity. IEEE Transactions on Information Theory, 2009.   
Idrissi, M. I., Chabridon, V., and Iooss, B. Developments and applications of shapley effects to reliability-oriented sensitivity analysis with correlated inputs. Environmental Modelling & Software, 2021.   
Jacovi, A. and Goldberg, Y. Towards faithfully interpretable nlp systems: How should we define and evaluate faithfulness? Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2020.   
Javadi, H. and Montanari, A. Nonnegative matrix factorization via archetypal analysis. Journal of the American Statistical Association, 2019.   
Jenatton, R., Obozinski, G., and Bach, F. Structured sparse principal component analysis. International Conference on Artificial Intelligence and Statistics, 2010.   
Jourdan, F., Picard, A., Fel, T., Risser, L., Loubes, J. M., and Asher, N. Cockatiel: Continuous concept ranked attribution with interpretable elements for explaining neural net classifiers on nlp tasks. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2023.   
Karvonen, A., Wright, B., Rager, C., Angell, R., Brinkmann, J., Smith, L., Verdun, C. M., Bau, D., and Marks, S. Measuring progress in dictionary learning for language model interpretability with board game models. ArXiv e-print, 2024.   
Kasiviswanathan, S., Wang, H., Banerjee, A., and Melville, P. Online l1-dictionary learning with application to novel document detection. Advances in Neural Information Processing Systems (NeurIPS), 2012.   
Keller, S. M., Samarin, M., Torres, F. A., Wieser, M., and Roth, V. Learning extremal representations with deep archetypal analysis. International Journal of Computer Vision, 2020.   
Kersting, K., Wahabzada, M., Thurau, C., and Bauckhage, C. Hierarchical convex nmf for clustering massive data. The Journal of Machine Learning Research (JMLR), 2010.   
Khemakhem, I., Kingma, D., Monti, R., and Hyvarinen, A. Variational autoencoders and nonlinear ica: A unifying framework. Proceedings of the International Conference on Machine Learning (ICML), 2020.   
Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). Proceedings of the International Conference on Machine Learning (ICML), 2018.

Kim, S. S. Y., Meister, N., Ramaswamy, V. V., Fong, R., and Russakovsky, O. HIVE: Evaluating the human interpretability of visual explanations. Proceedings of the IEEE European Conference on Computer Vision (ECCV), 2022.   
Kowal, M., Dave, A., Ambrus, R., Gaidon, A., Derpanis, K. G., and Tokmakov, P. Understanding video transformers via universal concept discovery. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024a.   
Kowal, M., Wildes, R. P., and Derpanis, K. G. Visual concept connectome (vcc): Open world concept discovery and their interlayer connections in deep models. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2024b.   
Lax, P. D. Functional analysis. 2014.   
Lee, D. D. and Seung, H. S. Learning the parts of objects by non-negative matrix factorization. Nature, 1999.   
Lee, D. D. and Seung, H. S. Algorithms for non-negative matrix factorization. Advances in Neural Information Processing Systems (NeurIPS), 2001.   
Lee, H., Battle, A., Raina, R., and Ng, A. Efficient sparse coding algorithms. Advances in Neural Information Processing Systems (NeurIPS), 2006.   
Lin, Z. Q., Shafiee, M. J., Bochkarev, S., Jules, M. S., Wang, X. Y., and Wong, A. Do explanations reflect decisions? a machine-centric strategy to quantify the performance of explainability algorithms. Advances in Neural Information Processing Systems (NIPS), 2019.   
Liu, F. T., Ting, K. M., and Zhou, Z.-H. Isolation forest. IEEE International Conference on Data Mining, 2008.   
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
Lloyd, S. Least squares quantization in pcm. IEEE Transactions on Information Theory, 1982.   
Locatello, F., Bauer, S., Lucic, M., Raetsch, G., Gelly, S., Scholkopf, B., and Bachem, O. Challenging common assumptions in the unsupervised learning of disentangled representations. Proceedings of the International Conference on Machine Learning (ICML), 2019.   
Locatello, F., Poole, B., Ratsch, G., Scholkopf, B., Bachem, O., and Tschannen, M. Weakly-supervised disentanglement without compromises. Proceedings of the International Conference on Machine Learning (ICML), 2020.

Lopes, M. Estimating unknown sparsity in compressed sensing. Proceedings of the International Conference on Machine Learning (ICML), 2013.   
Lu, C., Shi, J., and Jia, J. Online robust dictionary learning. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013.   
Mahdizadehaghdam, S., Panahi, A., Krim, H., and Dai, L. Deep dictionary learning: A parametric network approach. IEEE Transactions on Image Processing, 2019.   
Mair, S. and Brefeld, U. Coresets for archetypal analysis. In Advances in Neural Information Processing Systems (NeurIPS), 2019.   
Mairal, J., Bach, F., Ponce, J., and Sapiro, G. Online dictionary learning for sparse coding. Proceedings of the International Conference on Machine Learning (ICML), 2009.   
Mairal, J., Bach, F., and Ponce, J. Task-driven dictionary learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2011.   
Mairal, J., Bach, F., and Ponce, J. Sparse modeling for image and vision processing. Foundations and Trends in Computer Graphics and Vision, 2014.   
Makelov, A., Lange, G., and Nanda, N. Is this the subspace you are looking for? an interpretability illusion for subspace activation patching. ArXiv e-print, 2023.   
Mayne, H., Yang, Y., and Mahdi, A. Can sparse autoencoders be used to decompose and interpret steering vectors? ArXiv e-print, 2024.   
Mei, J., Wang, C., and Zeng, W. Online dictionary learning for approximate archetypal analysis. In European Conference on Computer Vision (ECCV), 2018.   
Menon, A., Shrivastava, M., Krueger, D., and Lubana, E. S. Analyzing (in) abilities of saes via formal languages. ArXiv e-print, 2024.   
Moliner, J. and Epifanio, I. Robust multivariate and functional archetypal analysis with application to financial time series. Physica A: Statistical Mechanics and its Applications, 2019.   
Mørup, M. and Hansen, L. K. Archetypal analysis for machine learning and data mining. Neurocomputing, 2012.   
Muzellec, S., Andeol, L., Fel, T., VanRullen, R., and Serre, T. Gradient strikes back: How filtering out high frequencies improves explanations. Proceedings of the International Conference on Learning Representations (ICLR), 2024.

Nguyen, G., Kim, D., and Nguyen, A. The effectiveness of feature attribution methods and its correlation with automatic evaluation scores. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Novello, P., Fel, T., and Vigouroux, D. Making sense of dependence: Efficient black-box explanations using dependence measure. Advances in Neural Information Processing Systems (NeurIPS), 2022.   
Olshausen, B. A. and Field, D. J. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 1996.   
Olshausen, B. A. and Field, D. J. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision Research, 1997.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. Dinov2: Learning robust visual features without supervision. ArXiv e-print, 2023.   
Papyan, V., Romano, Y., and Elad, M. Convolutional dictionary learning via local processing. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.   
Parekh, J., Khayatan, P., Shukor, M., Newson, A., and Cord, M. A concept-based explainability framework for large multimodal models. ArXiv e-print, 2024.   
Paulo, G. and Belrose, N. Sparse autoencoders trained on the same data learn different features. ArXiv e-print, 2025.   
Petsiuk, V., Das, A., and Saenko, K. Rise: Randomized input sampling for explanation of black-box models. Proceedings of the British Machine Vision Conference (BMVC), 2018.   
Poche, A., Jacovi, A., Picard, A. M., Boutin, V., and Jourdan, ´ F. Consim: Measuring concept-based explanations’ effectiveness with automated simulatability. ArXiv e-print, 2025.   
Prabhakaran, S., Raman, S., Vogt, J. E., and Roth, V. Automatic model selection in archetype analysis. In DAG-M/OAGM Symposium, 2012.   
Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramar, J., and Nanda, N. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. ArXiv e-print, 2024.   
Rasti, B., Zouaoui, A., Mairal, J., and Chanussot, J. Sunaa: Sparse unmixing using archetypal analysis. IEEE Geoscience and Remote Sensing Letters, 2023.

Rencker, L., Bach, F., Wang, W., and Plumbley, M. D. Sparse recovery and dictionary learning from nonlinear compressive measurements. IEEE Transactions on Signal Processing, 2019.   
Rentzeperis, I., Calatroni, L., Perrinet, L. U., and Prandi, D. Beyond l1 sparse coding in v1. PLoS Computational Biology, 2023.   
Rubinstein, R., Bruckstein, A. M., and Elad, M. Dictionaries for sparse representation modeling. Proceedings of the IEEE, 2010.   
Scholkopf, B., Williamson, R. C., Smola, A., Shawe-Taylor, J., and Platt, J. Support vector method for novelty detection. Advances in Neural Information Processing Systems (NeurIPS), 1999.   
Schott, L., Von Kugelgen, J., Trauble, F., Gehler, P., Russell, C., Bethge, M., Scholkopf, B., Locatello, F., and Brendel, W. Visual representation learning does not generalize strongly within the same domain. ArXiv e-print, 2021.   
Seiler, C. and Wohlrabe, K. Archetypal scientists. Journal of Informetrics, 2013.   
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-cam: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017.   
Serre, T. Learning a dictionary of shape-components in visual cortex: Comparison with neurons, humans and machines. 2006.   
Seth, S. and Eugster, M. J. A. Probabilistic archetypal analysis. Machine Learning, 2016.   
Sifa, R. and Bauckhage, C. Archetypical motion: Supervised game behavior learning with archetypal analysis. In FDG Workshop on Game Mining, 2013.   
Silverman, B. W. Spline smoothing: the equivalent variable kernel method. The Annals of Statistics, 1984.   
Simon, B. Convexity: an analytic viewpoint. 2011.   
Simonyan, K., Vedaldi, A., and Zisserman, A. Deep inside convolutional networks: Visualising image classification models and saliency maps. Proceedings of the International Conference on Learning Representations (ICLR), 2013.   
Sixt, L., Granz, M., and Landgraf, T. When explanations lie: Why many modified bp attributions fail. Proceedings of the International Conference on Machine Learning (ICML), 2020.

Slack, D., Hilgard, A., Lakkaraju, H., and Singh, S. Counterfactual explanations can be manipulated. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Smilkov, D., Thorat, N., Kim, B., Viegas, F., and Watten- ´ berg, M. Smoothgrad: removing noise by adding noise. Proceedings of the International Conference on Machine Learning (ICML), 2017.   
Spielman, D. A., Wang, H., and Wright, J. Exact recovery of sparsely-used dictionaries. The Journal of Machine Learning Research (JMLR), 2012.   
Springenberg, J. T., Dosovitskiy, A., Brox, T., and Riedmiller, M. Striving for simplicity: The all convolutional net. Workshop Proceedings of the International Conference on Learning Representations (ICLR), 2014.   
Stone, E. and Cutler, A. Introduction to archetypal analysis of spatio-temporal dynamics. Elsevier, 1996.   
Sturmfels, P., Lundberg, S., and Lee, S.-I. Visualizing the impact of feature attribution baselines. Distill, 2020.   
Sun, Y., Liu, Q., Tang, J., and Tao, D. Learning discriminative dictionary for group sparse representation. IEEE Transactions on Image Processing, 2014.   
Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. Proceedings of the International Conference on Machine Learning (ICML), 2017.   
Surkov, V., Wendler, C., Terekhov, M., Deschenaux, J., West, R., and Gulcehre, C. Unpacking sdxl turbo: Interpreting text-to-image models with sparse autoencoders. ArXiv e-print, 2024.   
Tamkin, A., Taufeeque, M., and Goodman, N. D. Codebook features: Sparse and discrete interpretability for neural networks. ArXiv e-print, 2023.   
Tariyal, S., Majumdar, A., Singh, R., and Vatsa, M. Deep dictionary learning. IEEE Access, 2016.   
Tasissa, A., Tankala, P., Murphy, J. M., and Ba, D. K-deep simplex: Manifold learning via local dictionaries. IEEE Transactions on Signal Processing, 2023.   
Thasarathan, H., Forsyth, J., Fel, T., Kowal, M., and Derpanis, K. Universal sparse autoencoders: Interpretable cross-model concept alignment. ArXiv e-print, 2025.   
Thurau, C. and Bauckhage, C. Archetypal images in large photo collections. In IEEE International Conference on Semantic Computing, 2009.   
Thurau, C., Kersting, K., and Bauckhage, C. Convex nonnegative matrix factorization in the wild. IEEE International Conference on Data Mining, 2009.

Tosiˇ c, I. and Frossard, P. Dictionary learning. ´ IEEE Signal Processing Magazine, 2011.   
Tripicchio, P. and D’Avella, S. Is deep learning ready to satisfy industry needs? Procedia Manufacturing, 2020.   
Vielhaben, J., Blucher, S., and Strodthoff, N. Multi- ¨ dimensional concept discovery (mcd): A unifying framework with completeness guarantees. The Journal of Transactions on Machine Learning Research (TMLR), 2023.   
Vilas, M. G., Adolfi, F., Poeppel, D., and Roig, G. Position: An inner interpretability framework for ai inspired by lessons from cognitive neuroscience. Proceedings of the International Conference on Machine Learning (ICML), 2024.   
Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al. Scipy 1.0: fundamental algorithms for scientific computing in python. Nature Methods, 2020.   
Von Kugelgen, J., Sharma, Y., Gresele, L., Brendel, W., Scholkopf, B., Besserve, M., and Locatello, F. Selfsupervised learning with data augmentations provably isolates content from style. Advances in Neural Information Processing Systems (NeurIPS), 2021.   
Wattenberg, M. and Viegas, F. B. Relational composition in neural networks: A survey and call to action. ArXiv e-print, 2024.   
Wedenborg, A. E. J. and Mørup, M. Archetypal analysis for binary data. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.   
Wightman, R. et al. Pytorch image models, 2019.   
Wright, J., Ma, Y., Mairal, J., Sapiro, G., Huang, T. S., and Yan, S. Sparse representation for computer vision and pattern recognition. Proceedings of the IEEE, 2010.   
Wynen, D., Schmid, C., and Mairal, J. Unsupervised learning of artistic styles with archetypal style analysis. In Advances in Neural Information Processing Systems (NeurIPS), 2018.   
Yu, Y., Buchanan, S., Pai, D., Chu, T., Wu, Z., Tong, S., Haeffele, B., and Ma, Y. White-box transformers via sparse rate reduction. Advances in Neural Information Processing Systems (NeurIPS), 2023.   
Zeiler, M. D. and Fergus, R. Visualizing and understanding convolutional networks. Proceedings of the IEEE European Conference on Computer Vision (ECCV), 2014.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2023.   
Zhang, R., Madumal, P., Miller, T., Ehinger, K. A., and Rubinstein, B. I. Invertible concept-based explanations for cnn models with non-negative concept activation vectors. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2021.   
Zimmermann, R. S., Sharma, Y., Schneider, S., Bethge, M., and Brendel, W. Contrastive learning inverts the data generating process. Proceedings of the International Conference on Machine Learning (ICML), 2021.   
Zou, H., Hastie, T., and Tibshirani, R. Sparse principal component analysis. Journal of Computational and Graphical Statistics, 2006.   
Zouaoui, A., Muhawenayo, G., Rasti, B., Chanussot, J., and Mairal, J. Entropic descent archetypal analysis for blind hyperspectral unmixing. IEEE Transactions on Image Processing, 2023.

# Appendix

# A. Qualitative Examples

![](images/df6115099253bb41a3a4fe6c71d80978c09c8ec829027a8a9c41c8a4c1952a16.jpg)

<details>
<summary>histogram</summary>

| Bin Range       | Frequency |
| --------------- | --------- |
| -0.6 to -0.4    | 0.0       |
| -0.4 to -0.2    | 0.0       |
| -0.2 to 0.0     | 1.5       |
| 0.0 to 0.2      | 3.5       |
| 0.2 to 0.4      | 1.0       |
| 0.4 to 0.6      | 0.0       |
| 0.6 to 0.8      | 0.0       |
</details>

![](images/8df3906fa37499c9f18068f5a64ae4b95bd4173c6b0005c8e7bc7d4d8038eaf6.jpg)

<details>
<summary>histogram</summary>

Concept firing rate
||Z_i||_0
| Bin Range | Frequency |
| :--- | :--- |
| 0.0 - 0.1 | 2700 |
| 0.1 - 0.2 | 2600 |
| 0.2 - 0.3 | 2300 |
| 0.3 - 0.4 | 1400 |
| 0.4 - 0.5 | 800 |
| 0.5 - 0.6 | 400 |
| 0.6 - 0.7 | 200 |
| 0.7 - 0.8 | 100 |
| 0.8 - 0.9 | 50 |
| 0.9 - 1.0 | 30 |
| 1.0 - 1.1 | 20 |
| 1.1 - 1.2 | 15 |
| 1.2 - 1.3 | 10 |
| 1.3 - 1.4 | 8 |
| 1.4 - 1.5 | 5 |
| 1.5 - 1.6 | 3 |
| 1.6 - 1.7 | 2 |
| 1.7 - 1.8 | 1 |
| 1.8 - 1.9 | 1 |
| 1.9 - 2.0 | 1 |
| 2.0 - 2.1 | 1 |
| 2.1 - 2.2 | 1 |
| 2.2 - 2.3 | 1 |
| 2.3 - 2.4 | 1 |
| 2.4 - 2.5 | 1 |
| 2.5 - 2.6 | 1 |
| 2.6 - 2.7 | 1 |
| 2.7 - 2.8 | 1 |
| 2.8 - 2.9 | 1 |
| 2.9 - 3.0 | 1 |
</details>

![](images/d666545a27691b0f27952f0fc78e440ab593d99a860e299c021bc81aaaa4f493.jpg)

<details>
<summary>heatmap</summary>

| CIs token | Register tokens |
| --------- | --------------- |
| 0         | 1.0             |
| 2         | 0.9             |
| 4         | 0.8             |
| 6         | 0.7             |
| 8         | 0.6             |
| 10        | 0.5             |
| 12        | 0.4             |
| 14        | 0.3             |
| 16        | 0.2             |
</details>

Figure 10. Performance, Auto-Correlation, and Firing Rate of DinoV2 for Archetypal SAE. The Archetypal SAE analyzed here demonstrates a well-structured dictionary with low auto-correlation, indicating that the dictionary atoms are not collinear. The firing rate exhibits a long-tail distribution, and reconstruction error is uniformly distributed across tokens, except for the CLS token, which shows the highest reconstruction error.

# B. Extended Related Work on Archetypal Analysis

Archetypal analysis (AA) was introduced by Cutler and Breiman in 1994 as a method to represent observations as convex combinations of extremal “pure types” called archetypes (Cutler & Breiman, 1994). Each archetype lies on the boundary of the data’s convex hull and is itself constrained to be a convex combination of data points, yielding an interpretable factorization where data are explained in terms of extreme exemplars. This approach was proposed as an alternative to principal component analysis for uncovering latent structure, providing data-like representative factors instead of orthogonal directions (Cutler & Stone, 1997). Shortly after its introduction, AA was extended to spatio-temporal settings by identifying archetypes that vary over time (Stone & Cutler, 1996) and (Prabhakaran et al., 2012) proposed an automatic model selection criterion for AA, alleviating the need to fix k a priori. After its initial formulation, AA did not gain widespread popularity, partly due to computational limitations and the availability of more established techniques like k-means or NMF. Nonetheless, several works in the 2000s applied AA successfully in scientific domains. For instance (Chan et al., 2003) applied AA to astronomical spectra. In genetics, (Gimbernat-Mayol et al., 2022) found that archetypes learned from human genotype data corresponded to ancestral population prototypes. AA was also utilized in socio-economic and bibliometric analyses (Seiler & Wohlrabe, 2013), and to model player behavior in video games (Sifa & Bauckhage, 2013). Various enhancements to the basic AA model have been proposed, (Eugster & Leisch, 2011) introduced weighted and robust AA, (Seth & Eugster, 2016) proposed to treats archetypes as latent factors in a probabilistic generative model. In a similar vein, contemporary work has developed AA for binary data specifically (Wedenborg & Mørup, 2025) and (Epifanio et al., 2019) tackled the presence of missing values. Likewise (Moliner & Epifanio, 2019) formulated robust AA for multivariate functional data.

Another important direction has been the development of non-linear and kernel variants of AA (Mørup & Hansen, 2012; Bauckhage & Manshaei, 2014). These kernel and multi-layer approaches connect AA with manifold learning techniques. (Javadi & Montanari, 2019) cast NMF as a special case of AA under separability assumptions. In term of algorithmic and scalability improvements (Chen et al., 2014) proposed a fast active-set algorithm, implemented in the SPAMS toolbox, making AA tractable for large-scale computer vision tasks. (Bauckhage et al.) adapted a Frank–Wolfe algorithm for AA, reducing runtime. (Mair & Brefeld, 2019) proposed coreset approximations to speed up archetype discovery, and (Abrol & Sharma, 2020) developed a greedy AA algorithm that extended naturally to robust and kernelized variants. Finally, (Mei et al., 2018) proposed an online algorithm for streaming data scenarios.

Mairal et al.’s task-driven dictionary learning framework inspired supervised AA models (Mairal et al., 2011). Related topic modeling approaches such as SurvLDA (Dawson & Kendziorski, 2012) and anchor-based models (Ding et al., 2016; Arora et al., 2013) similarly identify extremal structure. More recently, deep learning has been integrated with AA to handle complex, non-linear data. Wynen et al. (Wynen et al., 2018) applied AA to deep features for archetypal style analysis in artwork. Bauckhage et al. (Bauckhage et al.) described AA as an autoencoder. Keller et al. (Keller et al., 2020) developed a VAE-based deep AA that accommodates supervision. Applications are widespread. In vision, AA has been used to discover prototypical images (Thurau & Bauckhage, 2009), representative behaviors (Fotiadou et al., 2017), and fMRI brain patterns (Hinrich et al., 2016). In hyperspectral imaging, recent contributions such as SUnAA (Rasti et al., 2023) and entropic descent AA (Zouaoui et al., 2023) set new benchmarks. In NLP, AA was used in multi-document summarization (Canhasi & Kononenko, 2016) and anchor-based topic modeling (Ding et al., 2016; Arora et al., 2013).

Comparison to Our Work Our method departs from AA by constraining only the decoder to the convex hull of data, while using a linear encoder, maintaining full compatibility with modern SAEs architectures like TopK and JumpReLU. Unlike AA, which jointly optimizes over convex atoms and codes, we retain end to end SAE training. Our goal is not AA approximation or generation, but improved stability and semantic consistency of SAEs. We further introduce a scalable relaxation of the convex constraint, adapted to large-scale training and distinct from prior AA relaxations (e.g., (Mørup & Hansen, 2012)).

# C. Formal Definitions of Metrics

To comprehensively evaluate SAEs and their archetypal variants, we have defined in Sec.5.1 a set of metrics that assess four key dimensions of the SAEs: (i) sparse reconstruction, (ii) consistency, (iii) structure in the dictionary (D), and (iv) structure in the codes (Z).

# C.1. Sparse Reconstruction

As explained in the main paper, these metrics evaluate the ability of the model to accurately reconstruct activations while enforcing sparsity constraints. We believe they are well understood and already used by the interpretability community. The Reconstruction Error $( R ^ { 2 } )$ measures the fidelity of the reconstruction by quantifying how well the learned dictionary approximates the input activations:

$$
R ^ {2} = 1 - \frac {\left| \left| \boldsymbol {A} - \hat {\boldsymbol {A}} \right| \right| _ {F} ^ {2}}{\left| \left| \boldsymbol {A} - \bar {\boldsymbol {A}} \right| \right| _ {F} ^ {2}}, \tag {5}
$$

with $\hat { A } = z D$ the predicted activation and A¯ the mean activation matrix. Essentially, $R ^ { 2 }$ measures how much we improve on explaining variance upon the best possible predictor that uses only a single bias. The Dead Codes measure the fraction of dictionary atoms that remain unused across the dataset, highlighting inefficiencies in the learned representation, for a set of n codes and k concepts $\boldsymbol { Z } \in \mathbb { R } ^ { n \times k }$ :

$$
\text { Dead   Codes } = 1 - \frac {1}{k} | | \sum_ {i} ^ {n} \mathbf {Z} _ {i} | | _ {0} \tag {6}
$$

# C.2. Consistency

The second category of metrics assesses how consistent and well-grounded the learned dictionary is. Specifically, we evaluate (i) the Stability of the learned solution across training runs and (ii) its proximity to real data, measured via the Out-of-Distribution (OOD) Score.

Stability, introduced in Eq. 2, quantifies how consistent the learned dictionary remains when training is repeated with different initializations. Given two independently trained dictionaries, D and D′, stability is defined as:

$$
\operatorname{Stability} \left(\boldsymbol {D}, \boldsymbol {D} ^ {\prime}\right) = \max _ {\boldsymbol {\Pi} \in \mathcal {P} (n)} \frac {1}{n} \operatorname{Tr} \left(\boldsymbol {D} ^ {\top} \boldsymbol {\Pi} \boldsymbol {D} ^ {\prime}\right), \tag {7}
$$

where ${ \mathcal { P } } ( n )$ is the set of $n \times n$ signed permutation matrices. A score of 1 indicates perfect alignment—each concept in D has a direct equivalent in D′, while a score of 0 implies that all concepts are seed-specific and change arbitrarily across runs.

A looser measure of stability is the Max Cosine Similarity, which only considers the best-matching concept between two training runs:

$$
\text { Max   Cosine } = \max _ {i, j} \langle D _ {i}, D _ {j} ^ {\prime} \rangle . \tag {8}
$$

This metric provides an upper bound on alignment but does not enforce global consistency across the dictionary.

Beyond stability, we assess how well the learned dictionary aligns with real data. The Out-of-Distribution (OOD) Score measures the deviation of dictionary atoms from real data points by computing the cosine similarity between each dictionary atom $\mathbf { \nabla } D _ { i }$ and its closest real activation $A _ { j } { \mathrm { : } }$ :

$$
\text { OOD   Score } = 1 - \frac {1}{k} \sum_ {i = 1} ^ {k} \max _ {j \in [ n ]} \langle \boldsymbol {D} _ {i}, \boldsymbol {A} _ {j} \rangle . \tag {9}
$$

A score of 0 indicates that every dictionary atom $\mathbf { \nabla } D _ { i }$ exactly matches an existing data point $A _ { j } ,$ , meaning the model purely reconstructs real activations. Notably, methods like Separable-NMF (Gillis, 2020), which enforce the “purepixel” assumption by explicitly selecting dictionary atoms from the dataset, naturally achieve an OOD score of 0.

Together, these metrics provide a comprehensive evaluation of how stable, interpretable, and grounded the learned dictionary remains across training runs and relative to real data.

# C.3. Structure in the Dictionary (D)

The third category assesses the internal organization of the dictionary, providing insights into its effective dimensionality, redundancy, and (we hope) overall interpretability. Unlike reconstruction or consistency metrics, which evaluate external properties of the learned dictionary, these metrics focus on how well-formed and structured the set of concepts is. Again, a dictionary with lower effective dimensionality suggests structure like compositionality and/or hierarchical concepts. Stable Rank is the first metric we propose that provides an effective measure of the intrinsic dimensionality of the dictionary:

$$
\text { Stable   Rank } = \frac {| | D | | _ {F} ^ {2}}{| | D | | _ {2} ^ {2}}. \tag {10}
$$

Unlike the traditional matrix rank, which is sensitive to numerical precision (all dictionaries are nearly full rank in practice because of numerical error), the stable rank remains well-behaved even in high-dimensional settings, serving as a smooth proxy for rank estimation. Effective Rank offers an alternative perspective by measuring the entropy of the singular value distribution of D:

$$
\text { Eff.   Rank } = \exp \left(- \sum_ {i = 1} ^ {k} \sigma_ {i} \log \sigma_ {i}\right), \tag {11}
$$

where $\sigma _ { i }$ are the normalized singular values of D (i.e., $\textstyle \sum _ { i } \sigma _ { i } = 1 )$ . An effective rank close to k suggests that all dictionary atoms are equally important, whereas a low effective rank implies that only a few dominant concepts capture most of the variation in the data. A fully orthogonal dictionary (thus not overcomplete) would achieve an effective rank of k. Coherence is a common notion in dictionary learning and compress sensing, it quantifies redundancy between dictionary atoms by measuring the maximum pairwise cosine similarity:

$$
\text { Coherence } = \max _ {i \neq j} | \boldsymbol {D} _ {i} ^ {\top} \boldsymbol {D} _ {j} |. \tag {12}
$$

We still admit that each $\mathbf { \nabla } D _ { i }$ is on the $\ell _ { 2 }$ ball. Lower coherence indicates that dictionary atoms are more diverse and span independent directions, which is desirable for disentangled representations. Conversely, high coherence suggests that multiple dictionary atoms encode nearly identical features, reducing the efficiency of the learned basis. Notably, coherence is closely related to the concept of mutual incoherence in compressed sensing (Donoho, 2006), where low-coherence bases are preferred for sparse signal recovery.

# C.4. (iv) Structure in the Codes (Z)

The last category measures how much structure we have in the codes. While dictionary structure (iii) focuses on the learned basis D, the structure of the codes Z determines how these dictionary atoms are used to reconstruct activations. A well-structured encoding should exhibit meaningful combinations of concepts while avoiding redundancy and destructive interference. Connectivity measures the diversity of concept usage by counting the number of unique co-activations within the code matrix:

$$
\text { Connectivity } = 1 - \left(\frac {1}{d ^ {2}} | | \mathbf {Z} ^ {\top} \mathbf {Z} | | _ {0}\right). \tag {13}
$$

This metric quantifies how many distinct pairs of concepts (i, j) are activated together across samples. A high connectivity score suggests that a broad range of concepts can be meaningfully combined, leading to more complex representations. Conversely, low connectivity implies a highly structured representation, and in some sense a group-sparsity representation where only a subset of concepts can fire together. We note that connectivity in sparse coding has been linked to compositionality (Olshausen & Field, 1996). We believe that none of the SAEs currently studied achieve interesting performance on this metric. Finally, Negative Interference quantifies the extent to which co-activated concepts cancel each other out, reducing their effectiveness:

$$
\text { Neg. Inter. } = | | \operatorname{ReLU} (- (\mathbf {Z} ^ {\top} \mathbf {Z}) \odot (\mathbf {D D} ^ {\top})) | | _ {2}. \tag {14}
$$

Where ⊙ is the Hadamard product, this metric captures cases where two concepts i and j frequently co-activate (as measured by $Z ^ { \mathsf { T } } Z )$ , yet their dictionary atoms are negatively correlated (indicated by a negative dot product in $D D ^ { \dagger } )$ ). The ReLU function ensures that only destructive interactions are counted, where activation of both concepts leads to mutual cancellation rather than constructive combination. A high negative interference score suggests that the learned dictionary contains redundant or antagonistic concepts. In extreme cases, we could imagine a strong negative interference can lead to concept pairs that consistently suppress each other to comply with some sparsity constraint.

![](images/72097baa34b4c23efd591fbebe878c96c3dee08516a56a21401f74bf7cec52e3.jpg)

<details>
<summary>line</summary>

| Sparsity (%) | Ocsvm | Kmeans | Convex_hull | Lof | Iso_forest |
| ------------ | ----- | ------ | ----------- | --- | ---------- |
| 99.5         | 0.22  | 0.26   | 0.19        | 0.32 | 0.21       |
| 99.0         | 0.32  | 0.35   | 0.23        | 0.31 | 0.28       |
| 97.5         | 0.45  | 0.53   | 0.26        | 0.45 | 0.39       |
| 95.0         | 0.57  | 0.60   | 0.27        | 0.57 | 0.41       |
| 90.0         | 0.67  | 0.68   | 0.27        | 0.69 | 0.44       |
</details>

Figure 11. Comparison of Distillation Methods. To scale up the Archetypal SAE, it is impractical to utilize the entire data matrix A for identifying archetypes. Instead, we first reduce the dataset to a smaller subset of points, denoted as C, and construct the archetypes/dictionary elements from this reduced set. Among the distillation methods evaluated, K-Means proves to be the most effective approach, generating points within the convex hull of the data and achieving strong performance scores. These experiments were conducted on DINOv2 using an Archetypal SAE without relaxation.

# D. Distilling A into C

As a recall, Archetypal SAEs construct dictionary atoms as convex combinations of data points,

$$
\boldsymbol {D} = \boldsymbol {W} \boldsymbol {A}, \quad \text { with } \quad \boldsymbol {W} \in \Omega_ {k, n}, \quad \boldsymbol {A} \in \mathbb {R} ^ {n \times d}, \tag {15}
$$

requiring access to the full data matrix A. However, the original problem is intractable, as the number of points n requires storing and processing millions of activations at each gradient step, which is computationally prohibitive, particularly for large-scale datasets of tokens. To address this, we proposed in Sec. 4 a distillation step, reducing A to a compact subset C. The dictionary D is then formed using only $C \in \mathbb { R } ^ { n ^ { \prime } \times d }$ , with $n ^ { \prime } \ll n ,$ ensuring a tractable optimization while remaining representative of the original distribution.

We investigated five different methods to distill A into C, as illustrated in Fig. 11:

• K-Means: Groups data into m clusters and selects centroids as representatives. This ensures that C remains within the convex hull of A while capturing its most frequent patterns.   
• Local Outlier Factor (LOF) (Breunig et al., 2000): Identifies statistically atypical points compared to their neighbors, helping remove rare or extreme cases.   
• Isolation Forest (Iso) (Liu et al., 2008): Uses recursive

partitioning to isolate anomalies, providing an efficient method for detecting outliers.

• Convex Hull on Reduced Dimensions: Computes the convex hull of A after projecting it onto its 10 principal components via PCA. This ensures that extreme points defining the overall shape of the distribution are retained while reducing computational complexity.   
• One-Class SVM (OC-SVM) (Scholkopf et al., 1999): A support vector method that learns a boundary around high-density regions, effectively isolating representative points while filtering outliers.

Among these, K-Means emerges as the most effective method. Furthermore, while convex hull approaches theoretically guarantee coverage of extreme points, their computational cost scales poorly with high dimensions, making them impractical for large datasets.

# E. Implementation Details

We provide an efficient PyTorch implementation of the relaxed dictionary learning module used in Archetypal SAEs. Given a distilled set of centroids $C \in \mathbb { R } ^ { n ^ { \prime } \times d }$ , our goal is to construct a dictionary D as a convex combination of elements in C, while allowing a controlled degree of relaxation via additive perturbations.

- The weight matrix $W \in \mathbb { R } ^ { k \times n ^ { \prime } }$ is constrained to the probability simplex, ensuring convex combinations of centroids. This is enforced via a convex param function that projects W onto the simplex using ReLU and rowwise normalization.   
- The relaxation term $\pmb { \delta } \in \mathbb { R } ^ { k \times d }$ allows mild deviations from strict convexity. To prevent excessive drift, δ is regularized by adaptively scaling its norm to remain within a relaxation factor.   
- Dictionary atoms are computed as $D = W C + \delta$   
- The model supports gradient updates for W and δ while keeping C fixed.

class ArchetypalDictionary(nn.Module):
    """Relaxed Archetypal SAE (RA-SAE) dictionary.

Constructs a dictionary where each atom is a convex combination of data points from C, with a small relaxation term $\Lambda$ constrained by $\delta$ .

def __init__(self, C, k, delta=1.0):
    """
    Parameters
    ----
    C : Tensor
    Candidate archetypes of shape (n', d).
    k : int
    Number of dictionary atoms.
    delta : float
    Upper bound on the norm of the relaxation term $\Lambda$ .

    """
    super().__init__()
    n_prime, d = C.shape
    self.register_buffer("C", C) # store C as a fixed buffer (non-trainable)
    self.W = nn.Parameter(torch.eye(k, n_prime)) # trainable param (row-stochastic)
    self.Lambda = nn.Parameter(torch.zeros(k, d)) # small relaxation term
    self.delta = delta # constraint on the relaxation term norm

def forward(self, Z):
    """
    Parameters
    ----
    Z : Tensor
    Sparse codes of shape (n, k).

Returns
    ----
    Tensor
    Reconstructed activations of shape (n, d).
    """

    with torch.no_grad():
    # ensure W remains row-stochastic (positive and row sum to one)
    W = torch.relu(self.W)
    W /= W.sum(dim=-1, keepdim=True)
    self.W.data = W

    # enforce the norm constraint on $\Lambda$ to limit deviation from conv(C)
    norm_Lambda = self.Lambda.norm(dim=-1, keepdim=True) # norm per row
    scaling_factor = torch.clamp(self.delta / norm_Lambda, max=1) # safe scaling factor
    self.Lambda *= scaling_factor # scale $\Lambda$ to satisfy || $\Lambda$ || ≤ $\delta$ # compute the dictionary as a convex combination plus relaxation
D = self.W @ self.C + self.Lambda

return Z @ D   
Figure 12. Detailed Pytorch code for Relaxexd Archetypal SAE (RA-SAE).

# F. Theoretical properties of Archetypal Dictionary

In this section, we provide theoretical insights into Archetypal Sparse Autoencoders (A-SAEs) by addressing three key aspects: (i) why standard SAEs can produce dictionaries that drift away from the data manifold, (ii) a geometric interpretation of the A-SAE solution and the conditions under which distillation is optimal, and (iii) bounds on the stability, rank, and out-of-distribution (OOD) score of A-SAEs.

A simple explanation for why standard SAEs may drift away from the data manifold can be found by examining the gradient descent (GD) update rule for the dictionary D. Given a dataset $\pmb { A } \in \mathbb { R } ^ { n \times d }$ , nonnegative codes $Z \ge 0 \in$ $\mathbb { R } ^ { n \times k }$ , and a dictionary $D \in \mathbb { R } ^ { k \times d }$ with unit-norm rows $( \| D _ { j } \| _ { 2 } = 1 )$ , the standard SAE optimization problem is:

$$
\min _ {\boldsymbol {Z}, \boldsymbol {D}} \| \boldsymbol {A} - \boldsymbol {Z} \boldsymbol {D} ^ {\top} \| _ {F} ^ {2} \quad \text { s.t. } \quad \boldsymbol {Z} \geq 0, \quad \| \boldsymbol {D} _ {j} \| _ {2} = 1. \tag {16}
$$

The gradient of the reconstruction loss with respect to D is given by:

$$
\nabla_ {D} \| \boldsymbol {A} - \boldsymbol {Z} \boldsymbol {D} ^ {\mathsf {T}} \| _ {F} ^ {2} = 2 (\boldsymbol {Z} ^ {\mathsf {T}} \boldsymbol {Z} \boldsymbol {D} - \boldsymbol {Z} ^ {\mathsf {T}} \boldsymbol {A}). \tag {17}
$$

This gradient can be decomposed into two components:

$$
\Delta D = \underbrace {Z ^ {\mathsf {T}} A} _ {\text { data - anchored   term }} - \underbrace {Z ^ {\mathsf {T}} Z D} _ {\text { out - of - data   term }}.
$$

The first term, $Z ^ { \mathsf { T } } A$ , pulls dictionary atoms toward a conic combination of data points, anchoring them $\mathrm { t o ~ c o n e } ( A )$ . However, the second term, $Z ^ { \mathsf { T } } Z D$ , introduces a drift effect that pushes the dictionary away from the data, as it depends on the correlations within Z and the original seed D.

Empirically, in high-dimensional settings where the codes Z exhibit sufficient variability could induce the second term to dominate. This could explain why classical SAEs have a relatively low OOD score, as shown in Sec. 1 and why minor perturbations in initialization or the training set can potentially yield different dictionaries, as the second term is entirely dependent of the seed.

# F.1. Geometric interpretation of A-SAE

Proposition F.1 (Archetypal Dictionary, Convex and Conic Hulls). Given $\boldsymbol { A } \in \mathbb { R } ^ { n \times d }$ as a set of n data points and $W \ \in \ \Omega _ { k , n }$ as any row-stochastic matrix, parameterizing $D = W A$ ensures that each concept $\mathbf { \nabla } D _ { i }$ lies within the convex hull of the data, i.e., $\mathbf { \delta } _ { D _ { i } } \in \mathbf { \delta } $ conv(A) for all $i \in [ k ]$ . Moreover, for any nonnegative codes $Z \ge 0 ,$ , the reconstruction ZD lies within the conic hull of the data, $i . e . , Z D \subseteq \mathrm { c o n e } ( A )$ . More generally, let $C \in \mathbb { R } ^ { n ^ { \prime } : }$ ×d such that $\operatorname { c o n v } ( C ) \subseteq \operatorname { c o n v } ( A )$ be a subset of A or any set of points within conv(A). Then $D ^ { \prime } = W C$ satisfies $D _ { i } ^ { \prime } \in \operatorname { c o n v } ( C ) \subseteq \operatorname { c o n v } ( A )$ and $Z D ^ { \prime } \subseteq \mathrm { c o n e } ( A )$ . Finally, if C includes the extreme points of A, then no representational power is lost.

Proof. Since $W$ is row-stochastic, each $D _ { i } = W _ { i } { \cal A }$ is a convex combination of the rows of $A , { \mathrm { i . e . , } } D _ { i } \in \operatorname { c o n v } ( A )$ . Furthermore, for nonnegative Z, we have $Z D = ( Z W ) A$ , with $z W \ge 0$ , which implies that $Z D \subseteq \mathrm { c o n e } ( A )$ . With $D ^ { \prime } = W C$ for $\operatorname { c o n v } ( C ) \subseteq \operatorname { c o n v } ( A )$ , each row $\pmb { D } _ { i } ^ { \prime }$ lies within conv $( C ) \subseteq \operatorname { c o n v } ( A )$ . Lastly, if C contains the extreme points (Boyd & Vandenberghe, 2004) of A, then by simple application of the Krein-Milman theorem (Lax, 2014), conv $( C ) = \operatorname { c o n v } ( A )$ and cone $( Z W C ) =$ cone(A), ensuring no loss in expressivity.

The constraints imposed by A-SAE on D lead to a straightforward yet crucial geometric property: each dictionary atom $D _ { i }$ remains within the convex hull of the data. This result, while simple, has interesting implications for some of the metrics we study, notably stability and OOD. In fact, we will now see that this implies a bounded OOD score, prevents the rank of the dictionary (and thus its structure) to higher than the rank of the data, and induces some loose form of algorithmic stability.

# F.2. Stability of Archetypal Dictionary

Proposition F.2 (Geometric Stability of Archetypal Dictionaries). Let $\pmb { A } , \pmb { A } ^ { \prime } \in \mathbb { R } ^ { n \times d }$ be two data matrices such that $| | A - A ^ { \prime } | | _ { F } \leq \varepsilon .$ . Suppose W , $W ^ { \prime } \in \Omega _ { k , n }$ are rowstochastic matrices (i.e. each row belongs to the probability simplex). Define the archetypal dictionaries $D = W A$ and $D ^ { \prime } = W ^ { \prime } A ^ { \prime }$ . Then,

$$
\left| \left| \boldsymbol {D} - \boldsymbol {D} ^ {\prime} \right| \right| _ {F} \leq \sqrt {k} \varepsilon + 2 \sqrt {k} \min \left(\left| \left| \boldsymbol {A} \right| \right| _ {F}, \left| \left| \boldsymbol {A} ^ {\prime} \right| \right| _ {F}\right).
$$

Proof. Using triangle inequality, we have

$$
\begin{array}{l} \left| \left| \boldsymbol {D} - \boldsymbol {D} ^ {\prime} \right| \right| _ {F} = \left| \left| \boldsymbol {W} \boldsymbol {A} - \boldsymbol {W} ^ {\prime} \boldsymbol {A} ^ {\prime} \right| \right| _ {F} \leq \left| \left| \boldsymbol {W} \boldsymbol {A} - \boldsymbol {W} \boldsymbol {A} ^ {\prime} \right| \right| _ {F} + \\ \left| \left| \boldsymbol {W} \boldsymbol {A} ^ {\prime} - \boldsymbol {W} ^ {\prime} \boldsymbol {A} ^ {\prime} \right| \right| _ {F}. \\ \end{array}
$$

We bound each term separately, since D and $\pmb { D } ^ { \prime }$ are archety-√ pal dictionaries, W is row-stochastic, hence $| | W | | _ { 2 } \leq { \sqrt { k } }$ . Therefore,

$$
\left| \left| \boldsymbol {W} \boldsymbol {A} - \boldsymbol {W} \boldsymbol {A} ^ {\prime} \right| \right| _ {F} = \left| \left| \boldsymbol {W} \left(\boldsymbol {A} - \boldsymbol {A} ^ {\prime}\right) \right| \right| _ {F} \leq
$$

$$
\left| \left| \boldsymbol {W} \right| \right| _ {2} \left| \left| \boldsymbol {A} - \boldsymbol {A} ^ {\prime} \right| \right| _ {F},
$$

and $| | \boldsymbol { W } | | _ { 2 } | | \boldsymbol { A } - \boldsymbol { A } ^ { \prime } | | _ { F } \le \sqrt { k } \varepsilon$ . For the second term,

$$
\left| \left| \boldsymbol {W} \boldsymbol {A} ^ {\prime} - \boldsymbol {W} ^ {\prime} \boldsymbol {A} ^ {\prime} \right| \right| _ {F} = \left| \left| (\boldsymbol {W} - \boldsymbol {W} ^ {\prime}) \boldsymbol {A} ^ {\prime} \right| \right| _ {F} \leq
$$

$$
\left| \left| \boldsymbol {W} - \boldsymbol {W} ^ {\prime} \right| \right| _ {F} \left| \left| \boldsymbol {A} ^ {\prime} \right| \right| _ {F},
$$

with $| | W - W ^ { \prime } | | _ { F } \leq 2 \sqrt { k }$ summing these yield:

$$
\left| \left| \boldsymbol {D} - \boldsymbol {D} ^ {\prime} \right| \right| _ {F} \leq \sqrt {k} \varepsilon + 2 \sqrt {k} \left| \left| \boldsymbol {A} ^ {\prime} \right| \right| _ {F}.
$$

It is straightforward to repeat the above process with a $| | A ^ { \prime } | | _ { F }$ factor in the right hand side. Taking the minimum over the two factors then completes the proof.

The key observation is that a row-stochastic matrix $W$ cannot stretch data arbitrarily, imposing a natural control on how $D = W A$ changes if A is slightly perturbed. By contrast, in the unconstrained setting, where the dictionary D is free to move anywhere (subject only to norms or regularization), there is no comparably simple bound ensuring stability. Even small perturbations in A (or in the random seed, initialization, etc.) can shift the solution significantly: as there is no requirement that D stay close to $\operatorname { c o n v } ( A )$ , the learned atoms can drift to entirely different regions of the space.

# F.3. Controlled Rank and Stability in A-SAE Dictionaries

We now show that the rank of the dictionary obtained using the Archetypal parametrization are inherently controlled. Specifically, the rank of the dictionary cannot exceed the rank of the data. This property is interesting as it prevents the dictionary from becoming arbitrarily complex, promoting solutions that are more structured and aligned with the data. Consequently, the dictionary is more likely to uncover meta-concepts or low-rank representations.

Proposition F.3 (Rank Bound of Archetypal Dictionaries). Let $\pmb { A } \in \mathbb { R } ^ { n \times d }$ be a data matrix with rank $( A ) = r \leq d ,$ assuming n $\gg d$ and $k \gg d .$ . Let $W \in \Omega _ { k , n }$ be a rowstochastic matrix, and define the dictionary as $D = W A$ . Then, the rank of D is bounded by the rank of the data:

$$
\operatorname{rank} (\boldsymbol {D}) \leq \min (\operatorname{rank} (\boldsymbol {A}), \operatorname{rank} (\boldsymbol {W})) \leq d.
$$

Proof. Since $D = W A$ , the column space of D is contained in the column space of A, implying

$$
\operatorname{rank} (\boldsymbol {D}) \leq \operatorname{rank} (\boldsymbol {A}) = r.
$$

Additionally, since $W \in \mathbb { R } ^ { k \times n }$ is row-stochastic, its rank is at most min $( k , n )$ , giving

$$
\operatorname{rank} (\boldsymbol {W}) \leq \min (k, n).
$$

it follows that

$$
\operatorname{rank} (\boldsymbol {D}) = \operatorname{rank} (\boldsymbol {W A}) \leq \min (r, \operatorname{rank} (\boldsymbol {W})).
$$

# F.4. Bounding OOD score with Archetypal Constraints

We now demonstrate that the OOD measure of dictionary atoms obtained under some assumption is inherently lowerbounded. Specifically, the measure is directly tied to the weights in the row-stochastic matrix, with a maximum value of 1 achieved when a dictionary atom perfectly aligns with a data point. This property is particularly interesting as it ensures that dictionary atoms remain well-grounded in the data. Furthermore, the sparsity of the weight matrix plays a crucial role in maintaining orthogonality, thereby preserving the plausibility of the assumption and the robustness of the bounds.

Proposition F.4 (OOD Measure with Non-Interfering Archetypes). Let $\ b { A } \in \mathbb { R } ^ { n \times d }$ be our data matrix where each point Ai is normalized $( \lVert A _ { i } \rVert _ { 2 } = 1$ for all $i \in [ n ] )$ . Let our archetypal dictionary D = W A, where W ∈ Rk×n $D = W A$ $W \in \mathbb { R } ^ { k \times n }$ is a row-stochastic matrix $( W \in \Omega _ { k , n } ) .$ We assume non-interfering Archetypes, meaning two non-orthogonal archetypes cannot be active at the same time (but can exist in the bank of points A). Formally, for each row $W _ { i }$ , the active rows of A (those $A _ { j }$ with $W _ { i j } > 0 )$ are orthogonal, $i . e .$ ,

$$
\langle \boldsymbol {A} _ {j}, \boldsymbol {A} _ {j ^ {\prime}} \rangle = 0 \quad f o r j \neq j ^ {\prime} a n d \boldsymbol {W} _ {i j}, \boldsymbol {W} _ {i j ^ {\prime}} > 0.
$$

Then, the out-of-distribution (OOD) measure for each $\mathbf { \nabla } D _ { i }$ admits the upper bound:

$$
\operatorname{OOD} \left(\boldsymbol {D} _ {i}\right) \leq 1 - \max _ {j \in [ n ]} \boldsymbol {W} _ {i j}.
$$

Proof. By definition, $\begin{array} { r } { D _ { i } = \sum _ { j = 1 } ^ { n } W _ { i j } A _ { j } } \end{array}$ , so

$$
\langle \boldsymbol {D} _ {i}, \boldsymbol {A} _ {j} \rangle = \left\langle \sum_ {k = 1} ^ {n} \boldsymbol {W} _ {i k} \boldsymbol {A} _ {k}, \boldsymbol {A} _ {j} \right\rangle = \sum_ {k = 1} ^ {n} \boldsymbol {W} _ {i k} \langle \boldsymbol {A} _ {k}, \boldsymbol {A} _ {j} \rangle .
$$

Under the orthogonality assumption $\langle { \cal A } _ { k } , { \cal A } _ { j } \rangle = 0$ for k ̸= $j ,$ only the $k = j$ term remains:

$$
\langle \boldsymbol {D} _ {i}, \boldsymbol {A} _ {j} \rangle = \boldsymbol {W} _ {i j} \langle \boldsymbol {A} _ {j}, \boldsymbol {A} _ {j} \rangle .
$$

Since $\| A _ { j } \| _ { 2 } = 1$ , we have $\langle A _ { j } , A _ { j } \rangle = 1$ , so:

$$
\left\langle \boldsymbol {D} _ {i}, \boldsymbol {A} _ {j} \right\rangle = \boldsymbol {W} _ {i j}.
$$

By definition:

$$
\| \boldsymbol {D} _ {i} \| _ {2} ^ {2} = \left\| \sum_ {j = 1} ^ {n} \boldsymbol {W} _ {i j} \boldsymbol {A} _ {j} \right\| _ {2} ^ {2}.
$$

By the orthogonality of the active rows of A, the contribu-

![](images/7403d25724a547c3fb012d82cc25cbb03908c06dbe86c5c4f6664f21a3fb11aa.jpg)

<details>
<summary>line</summary>

| x    | Rectangle | Gaussian | Triangular | Cosine | Epanechnikov | Quartic | Silverman | Cauchy |
| ---- | --------- | -------- | ---------- | ------ | ------------ | ------- | --------- | ------ |
| -3.0 | 0.0       | 0.0      | 0.0        | 0.0    | 0.0          | 0.0     | 0.0       | 0.0    |
| -2.0 | 0.0       | 0.0      | 0.0        | 0.0    | 0.0          | 0.0     | 0.0       | 0.0    |
| -1.0 | 0.0       | 0.0      | 0.0        | 0.0    | 0.0          | 0.0     | 0.0       | 0.0    |
| 0.0  | 1.0       | 0.4      | 2.0        | 1.5    | 1.5          | 1.8     | 1.4       | 0.6    |
| 1.0  | 1.0       | 0.4      | 1.5        | 1.0    | 1.0          | 1.5     | 1.4       | 0.6    |
| 2.0  | 0.0       | 0.0      | 0.0        | 0.0    | 0.0          | 0.0     | 0.0       | 0.0    |
| 3.0  | 0.0       | 0.0      | 0.0        | 0.0    | 0.0          | 0.0     | 0.0       | 0.0    |
</details>

Figure 13. Example of kernel functions for JumpReLU over the interval [−3, 3] with a bandwidth of 1. The Silverman kernel produced more stable and positive results, leading to its selection for all experiments with a smaller bandwidth of $1 0 ^ { - 2 }$ .

tions of different rows do not interfere, so:

$$
\| \boldsymbol {D} _ {i} \| _ {2} ^ {2} = \sum_ {j = 1} ^ {n} \boldsymbol {W} _ {i j} ^ {2} \| \boldsymbol {A} _ {j} \| _ {2} ^ {2}.
$$

Since $\| A _ { j } \| _ { 2 } = 1$ , this simplifies to:

$$
\| \boldsymbol {D} _ {i} \| _ {2} ^ {2} = \sum_ {j = 1} ^ {n} \boldsymbol {W} _ {i j} ^ {2}.
$$

Substituting these yields our bound:

$$
\operatorname{OOD} \left(\boldsymbol {D} _ {i}\right) = 1 - \max _ {j \in [ n ]} \frac {\left\langle \boldsymbol {D} _ {i} , \boldsymbol {A} _ {j} \right\rangle}{\| \boldsymbol {D} _ {i} \| _ {2}} \leq 1 - \max _ {j \in [ n ]} \boldsymbol {W} _ {i j}.
$$

![](images/97d546b32740c444940a7b5f83ad1973b118f639b13136d73079adb3a3c1581b.jpg)

As a notable special case, we observe that $\mathrm { O O D } ( D _ { i } ) = 0$ when $W _ { i j } = 1$ for some $j \in [ n ]$ , as the dictionary atom aligns perfectly with a data point. In practice, the sparsity of W plays a crucial role: it limits interference between (nonorthogonal) components, ensuring that the orthogonality assumption remains plausible and that the derived bounds hold robustly.

# G. Kernel for JumpReLU

JumpReLU (Rajamanoharan et al., 2024) is a recently introduced activation mechanism for SAEs designed to optimize $\ell _ { 0 }$ sparsity by controlling the discontinuities of ReLU through a parameter θ. Its optimization relies on a kernel for density estimation. To assess the effect of kernel choice, we evaluated several options, including Gaussian, Cauchy, and Silverman, on DinoV2. As shown in Figure 13, the Silverman kernel consistently yielded the most stable and accurate reconstruction results. For our experiments, we selected the Silverman kernel with a bandwidth of $1 0 ^ { - 2 }$ , although the choice of kernel appears to have only a modest impact on performance.

# H. Soft Identifiability Benchmark

In this appendix, we provide additional details regarding the experimental setup and evaluation criteria used in the Soft Identifiability Benchmark. We recall that the goal is to assess the ability of SAEs to recover distinct concepts from synthetic image mixtures, where the underlying generative factors are known.

We generate twelve synthetic datasets using Midjourney API7. For each of these datasets, we programmatically create 4,000 images. These images are constructed by collaging four distinct objects selected from the predefined set, such as different types of gems. Each dataset is generated from between 9 and 20 unique objects, with the dictionary size set exactly to the number of true generative factors (unique object number). An example of some datasets used in the benchmark is shown in Figure 14.

Each dataset is split into a training set of 2,000 images and a test set of 2,000 images. The images are processed through a pre-trained vision model, in our case DinoV2, ResNet50, SigLIP and ViT. The resulting pooled activations serve as the input representations for the SAE.

Metrics. To quantitatively evaluate identifiability, we define an accuracy metric that measures whether each object class in the dataset is correctly assigned a distinct concept in the SAE dictionary. Given an image, we pass it through the vision model and then through the trained SAE to obtain a concept-label pair $( z , y )$ , where $z \in \mathbb { R } ^ { k }$ represents the k learned concept activations, and $\ b { y } \in \mathbb { R } ^ { c }$ denotes the c ground-truth class labels.

We define the accuracy for class $j$ as:

$$
\text { Accuracy } _ {j} = \max _ {\lambda \in \mathbb {R}, i \in [ k ]} \mathbb {P} _ {(\boldsymbol {z}, \boldsymbol {y})} ((z _ {i} > \lambda) = y _ {j}), \tag {18}
$$

where λ is a threshold determining whether a concept is activated.

To find an appropriate λ, we use the empirical percentiles of the concept activations Z, ranging from the 1st to the 100th percentile. This ensures that the threshold is adaptive to the distribution of activations, optimizing for the best classification accuracy.

![](images/cf88ed2e70e281baab704ce4553350e6d3e381657d6357e6be5bba1d9b1e99d4.jpg)  
Figure 14. Examples of synthetic datasets used in the Identifiability Benchmark.

# H.1. Complete Results

The full set of results across all methods and datasets is provided in Table 4. We also provide a comprehensive breakdown, including per-dataset accuracy scores and additional analysis.

Table 4. Accuracy Scores for Various Methods Across Models and Classes 

<table><tr><td>Model</td><td>Method</td><td>Animals</td><td>Birds</td><td>Books</td><td>Candy</td><td>Cards</td><td>Cocktails</td><td>Flowers</td><td>Gems</td><td>Landscapes</td><td>Planets</td><td>Potions</td><td>Watches</td><td>Avg</td></tr><tr><td rowspan="12">DINO</td><td>KMeans</td><td>0.7679</td><td>0.7678</td><td>0.7715</td><td>0.7709</td><td>0.7724</td><td>0.8095</td><td>0.6670</td><td>0.8137</td><td>0.7147</td><td>0.7786</td><td>0.8104</td><td>0.7693</td><td>0.7678</td></tr><tr><td>ICA</td><td>0.8113</td><td>0.7967</td><td>0.8099</td><td>0.8182</td><td>0.8212</td><td>0.8296</td><td>0.7497</td><td>0.8569</td><td>0.7807</td><td>0.8002</td><td>0.8297</td><td>0.8068</td><td>0.8092</td></tr><tr><td>Sparse PCA</td><td>0.8033</td><td>0.7919</td><td>0.8013</td><td>0.7939</td><td>0.8129</td><td>0.8245</td><td>0.7126</td><td>0.8322</td><td>0.7733</td><td>0.8022</td><td>0.8307</td><td>0.7986</td><td>0.7981</td></tr><tr><td>SVD</td><td>0.8037</td><td>0.7916</td><td>0.8018</td><td>0.7935</td><td>0.8142</td><td>0.8245</td><td>0.7116</td><td>0.8320</td><td>0.7716</td><td>0.8023</td><td>0.8301</td><td>0.7978</td><td>0.7979</td></tr><tr><td>SemiNMF</td><td>0.8175</td><td>0.8059</td><td>0.8560</td><td>0.8660</td><td>0.8464</td><td>0.8516</td><td>0.7360</td><td>0.8564</td><td>0.8111</td><td>0.8261</td><td>0.8569</td><td>0.8264</td><td>0.8297</td></tr><tr><td>ConvexNMF</td><td>0.7726</td><td>0.7658</td><td>0.7739</td><td>0.7759</td><td>0.7711</td><td>0.8108</td><td>0.6264</td><td>0.8163</td><td>0.7013</td><td>0.7688</td><td>0.8205</td><td>0.7711</td><td>0.7645</td></tr><tr><td>PCA</td><td>0.8037</td><td>0.7916</td><td>0.8018</td><td>0.7935</td><td>0.8142</td><td>0.8245</td><td>0.7116</td><td>0.8320</td><td>0.7716</td><td>0.8023</td><td>0.8301</td><td>0.7978</td><td>0.7979</td></tr><tr><td>Vanilla</td><td>0.7968</td><td>0.7878</td><td>0.8087</td><td>0.8161</td><td>0.8062</td><td>0.8468</td><td>0.7202</td><td>0.8518</td><td>0.7601</td><td>0.8262</td><td>0.8385</td><td>0.7977</td><td>0.8047</td></tr><tr><td>TopK</td><td>0.7906</td><td>0.7942</td><td>0.8104</td><td>0.8283</td><td>0.8243</td><td>0.8407</td><td>0.7728</td><td>0.8501</td><td>0.7744</td><td>0.8184</td><td>0.8387</td><td>0.8191</td><td>0.8135</td></tr><tr><td>Jump</td><td>0.7863</td><td>0.7956</td><td>0.7970</td><td>0.8174</td><td>0.8121</td><td>0.8251</td><td>0.7389</td><td>0.8418</td><td>0.7504</td><td>0.7989</td><td>0.8438</td><td>0.8042</td><td>0.8010</td></tr><tr><td>A-SAE</td><td>0.9433</td><td>0.9413</td><td>0.9692</td><td>0.9722</td><td>0.9750</td><td>0.8901</td><td>0.9590</td><td>0.9590</td><td>0.9677</td><td>0.9277</td><td>0.9129</td><td>0.9606</td><td>0.9482</td></tr><tr><td>RA-SAE</td><td>0.9402</td><td>0.9313</td><td>0.9703</td><td>0.9614</td><td>0.9686</td><td>0.8905</td><td>0.9613</td><td>0.9503</td><td>0.9666</td><td>0.9227</td><td>0.9094</td><td>0.9642</td><td>0.9447</td></tr><tr><td rowspan="12">ResNet</td><td>PCA</td><td>0.8249</td><td>0.8086</td><td>0.8556</td><td>0.8643</td><td>0.8622</td><td>0.8501</td><td>0.7908</td><td>0.8446</td><td>0.7814</td><td>0.8203</td><td>0.8326</td><td>0.8134</td><td>0.8291</td></tr><tr><td>KMeans</td><td>0.7670</td><td>0.7647</td><td>0.7720</td><td>0.7668</td><td>0.7669</td><td>0.8123</td><td>0.6418</td><td>0.8109</td><td>0.6964</td><td>0.7713</td><td>0.8109</td><td>0.7686</td><td>0.7624</td></tr><tr><td>ICA</td><td>0.8169</td><td>0.8231</td><td>0.8315</td><td>0.8358</td><td>0.8295</td><td>0.8852</td><td>0.7902</td><td>0.8900</td><td>0.8378</td><td>0.8206</td><td>0.8619</td><td>0.8213</td><td>0.8370</td></tr><tr><td>Sparse PCA</td><td>0.8256</td><td>0.8162</td><td>0.8599</td><td>0.8689</td><td>0.8661</td><td>0.8501</td><td>0.7937</td><td>0.8461</td><td>0.7841</td><td>0.8230</td><td>0.8330</td><td>0.8155</td><td>0.8318</td></tr><tr><td>SVD</td><td>0.8249</td><td>0.8086</td><td>0.8556</td><td>0.8643</td><td>0.8623</td><td>0.8501</td><td>0.7910</td><td>0.8445</td><td>0.7816</td><td>0.8201</td><td>0.8324</td><td>0.8134</td><td>0.8291</td></tr><tr><td>SemiNMF</td><td>0.8259</td><td>0.8274</td><td>0.8536</td><td>0.8611</td><td>0.8419</td><td>0.8577</td><td>0.7307</td><td>0.8632</td><td>0.8198</td><td>0.8195</td><td>0.8486</td><td>0.8432</td><td>0.8327</td></tr><tr><td>ConvexNMF</td><td>0.7647</td><td>0.7648</td><td>0.7699</td><td>0.7669</td><td>0.7659</td><td>0.8086</td><td>0.6102</td><td>0.8111</td><td>0.6939</td><td>0.7686</td><td>0.8098</td><td>0.7644</td><td>0.7582</td></tr><tr><td>Vanilla</td><td>0.8062</td><td>0.8143</td><td>0.8270</td><td>0.8228</td><td>0.8397</td><td>0.8412</td><td>0.7380</td><td>0.8393</td><td>0.7894</td><td>0.8147</td><td>0.8475</td><td>0.8198</td><td>0.8167</td></tr><tr><td>TopK</td><td>0.8078</td><td>0.8083</td><td>0.8384</td><td>0.8269</td><td>0.8176</td><td>0.8496</td><td>0.7437</td><td>0.8397</td><td>0.7753</td><td>0.8193</td><td>0.8482</td><td>0.8052</td><td>0.8150</td></tr><tr><td>Jump</td><td>0.7949</td><td>0.8027</td><td>0.8082</td><td>0.8274</td><td>0.8144</td><td>0.8345</td><td>0.6648</td><td>0.8373</td><td>0.7485</td><td>0.8161</td><td>0.8287</td><td>0.8084</td><td>0.7988</td></tr><tr><td>A-SAE</td><td>0.9633</td><td>0.9703</td><td>0.9638</td><td>0.9673</td><td>0.9713</td><td>0.9738</td><td>0.9894</td><td>0.9722</td><td>0.9658</td><td>0.9342</td><td>0.9539</td><td>0.9315</td><td>0.9631</td></tr><tr><td>RA-SAE</td><td>0.9613</td><td>0.9577</td><td>0.9694</td><td>0.9834</td><td>0.9709</td><td>0.9625</td><td>0.9497</td><td>0.9640</td><td>0.9629</td><td>0.9371</td><td>0.9554</td><td>0.9479</td><td>0.9602</td></tr><tr><td rowspan="12">SigLIP</td><td>PCA</td><td>0.8253</td><td>0.7957</td><td>0.8264</td><td>0.8030</td><td>0.8157</td><td>0.8286</td><td>0.7367</td><td>0.8270</td><td>0.7678</td><td>0.7931</td><td>0.8291</td><td>0.8261</td><td>0.8062</td></tr><tr><td>KMeans</td><td>0.7733</td><td>0.7672</td><td>0.7733</td><td>0.7691</td><td>0.7742</td><td>0.8094</td><td>0.6690</td><td>0.8121</td><td>0.7171</td><td>0.7733</td><td>0.8101</td><td>0.7724</td><td>0.7684</td></tr><tr><td>ICA</td><td>0.8372</td><td>0.8171</td><td>0.8151</td><td>0.8254</td><td>0.8382</td><td>0.8341</td><td>0.7918</td><td>0.8333</td><td>0.8420</td><td>0.8056</td><td>0.8362</td><td>0.8158</td><td>0.8243</td></tr><tr><td>Sparse PCA</td><td>0.8251</td><td>0.7962</td><td>0.8374</td><td>0.8021</td><td>0.8152</td><td>0.8269</td><td>0.7369</td><td>0.8286</td><td>0.7676</td><td>0.7941</td><td>0.8286</td><td>0.8240</td><td>0.8069</td></tr><tr><td>SVD</td><td>0.8253</td><td>0.7957</td><td>0.8264</td><td>0.8030</td><td>0.8157</td><td>0.8286</td><td>0.7367</td><td>0.8269</td><td>0.7678</td><td>0.7931</td><td>0.8291</td><td>0.8261</td><td>0.8062</td></tr><tr><td>SemiNMF</td><td>0.8574</td><td>0.8386</td><td>0.8413</td><td>0.8559</td><td>0.8521</td><td>0.8393</td><td>0.7780</td><td>0.8524</td><td>0.7979</td><td>0.8320</td><td>0.8389</td><td>0.8463</td><td>0.8358</td></tr><tr><td>ConvexNMF</td><td>0.7706</td><td>0.7706</td><td>0.7769</td><td>0.7704</td><td>0.7687</td><td>0.8090</td><td>0.6314</td><td>0.8121</td><td>0.6972</td><td>0.7718</td><td>0.8134</td><td>0.7745</td><td>0.7639</td></tr><tr><td>Vanilla</td><td>0.8240</td><td>0.8160</td><td>0.8097</td><td>0.8202</td><td>0.8429</td><td>0.8436</td><td>0.7167</td><td>0.8493</td><td>0.7775</td><td>0.7976</td><td>0.8433</td><td>0.8108</td><td>0.8126</td></tr><tr><td>TopK</td><td>0.8254</td><td>0.8190</td><td>0.8215</td><td>0.8439</td><td>0.8664</td><td>0.8460</td><td>0.7576</td><td>0.8547</td><td>0.8022</td><td>0.8253</td><td>0.8483</td><td>0.8364</td><td>0.8289</td></tr><tr><td>Jump</td><td>0.8199</td><td>0.8351</td><td>0.8048</td><td>0.8206</td><td>0.8222</td><td>0.8367</td><td>0.7307</td><td>0.8493</td><td>0.7813</td><td>0.7946</td><td>0.8352</td><td>0.8269</td><td>0.8131</td></tr><tr><td>A-SAE</td><td>0.9727</td><td>0.9613</td><td>0.9517</td><td>0.9686</td><td>0.9753</td><td>0.9445</td><td>0.9622</td><td>0.9669</td><td>0.9553</td><td>0.9479</td><td>0.9457</td><td>0.9704</td><td>0.9602</td></tr><tr><td>RA-SAE</td><td>0.9655</td><td>0.9654</td><td>0.9411</td><td>0.9681</td><td>0.9749</td><td>0.9366</td><td>0.9632</td><td>0.9594</td><td>0.9543</td><td>0.9325</td><td>0.9546</td><td>0.9861</td><td>0.9585</td></tr><tr><td rowspan="12">ViT</td><td>PCA</td><td>0.7994</td><td>0.8107</td><td>0.8226</td><td>0.8258</td><td>0.7963</td><td>0.8352</td><td>0.7780</td><td>0.8274</td><td>0.7556</td><td>0.8029</td><td>0.8199</td><td>0.8164</td><td>0.8075</td></tr><tr><td>KMeans</td><td>0.7706</td><td>0.7719</td><td>0.7744</td><td>0.7776</td><td>0.7752</td><td>0.8095</td><td>0.6721</td><td>0.8134</td><td>0.7120</td><td>0.7841</td><td>0.8099</td><td>0.7722</td><td>0.7702</td></tr><tr><td>ICA</td><td>0.8134</td><td>0.8251</td><td>0.8243</td><td>0.8437</td><td>0.8159</td><td>0.8623</td><td>0.7983</td><td>0.8539</td><td>0.7998</td><td>0.8229</td><td>0.8415</td><td>0.8188</td><td>0.8267</td></tr><tr><td>Sparse PCA</td><td>0.8003</td><td>0.8109</td><td>0.8225</td><td>0.8287</td><td>0.7963</td><td>0.8342</td><td>0.7780</td><td>0.8282</td><td>0.7563</td><td>0.8022</td><td>0.8201</td><td>0.8208</td><td>0.8082</td></tr><tr><td>SVD</td><td>0.7994</td><td>0.8107</td><td>0.8226</td><td>0.8258</td><td>0.7963</td><td>0.8351</td><td>0.7780</td><td>0.8274</td><td>0.7556</td><td>0.8029</td><td>0.8201</td><td>0.8164</td><td>0.8075</td></tr><tr><td>SemiNMF</td><td>0.8232</td><td>0.8414</td><td>0.8436</td><td>0.8410</td><td>0.8598</td><td>0.8661</td><td>0.8133</td><td>0.8584</td><td>0.8064</td><td>0.8418</td><td>0.8492</td><td>0.8629</td><td>0.8423</td></tr><tr><td>ConvexNMF</td><td>0.7709</td><td>0.7685</td><td>0.7684</td><td>0.7714</td><td>0.7689</td><td>0.8117</td><td>0.6254</td><td>0.8132</td><td>0.7017</td><td>0.7736</td><td>0.8134</td><td>0.7738</td><td>0.7634</td></tr><tr><td>Vanilla</td><td>0.8169</td><td>0.8548</td><td>0.8459</td><td>0.8234</td><td>0.8438</td><td>0.8481</td><td>0.7312</td><td>0.8598</td><td>0.7512</td><td>0.8249</td><td>0.8285</td><td>0.8387</td><td>0.8223</td></tr><tr><td>TopK</td><td>0.8494</td><td>0.8430</td><td>0.8577</td><td>0.8264</td><td>0.8405</td><td>0.8503</td><td>0.7849</td><td>0.8534</td><td>0.7910</td><td>0.8215</td><td>0.8310</td><td>0.8451</td><td>0.8328</td></tr><tr><td>Jump</td><td>0.8042</td><td>0.7995</td><td>0.8316</td><td>0.8156</td><td>0.8064</td><td>0.8377</td><td>0.6872</td><td>0.8515</td><td>0.7681</td><td>0.8143</td><td>0.8300</td><td>0.8172</td><td>0.8053</td></tr><tr><td>A-SAE</td><td>0.9647</td><td>0.9847</td><td>0.9519</td><td>0.9838</td><td>0.9733</td><td>0.9578</td><td>0.9938</td><td>0.9488</td><td>0.9384</td><td>0.9369</td><td>0.9325</td><td>0.9719</td><td>0.9615</td></tr><tr><td>RA-SAE</td><td>0.9699</td><td>0.9847</td><td>0.9683</td><td>0.9696</td><td>0.9694</td><td>0.9620</td><td>0.9576</td><td>0.9455</td><td>0.9281</td><td>0.9484</td><td>0.9253</td><td>0.9745</td><td>0.9586</td></tr></table>