Survey Paper

# A comprehensive survey of transfer dictionary learning

![](images/edbedfba9f998eb5efa776e654b14f063cac16d108c54c6ca96c00beb3d8c25a.jpg)

Mengyao Li a, Yang Li a, Zhengming Li a,b,∗

a School of Cyber Security, Guangdong Polytechnic Normal University, 510630, Guangzhou, China b Guangdong Provincial Key Laboratory of Intellectual Property and Big Dat, 510665, Guangzhou, China

# A R T I C L E I N F O

Communicated by R. Yang

Keywords:

Dictionary learning

Transfer learning

Cross-domain dictionary learning

Domain-adaptive dictionary learning

# A B S T R A C T

Despite the noteworthy advancements in the field of transfer dictionary learning, a comprehensive survey remains conspicuously absent. Moreover, since cross-domain learning and domain-adaptive learning are two main directions of transfer learning. Consequently, this paper aims to extensively review the advancements achieved in the cross-domain dictionary learning algorithms as well as the domain-adaptive dictionary learning algorithms. Firstly, based on the learning strategies, the cross-domain dictionary learning algorithms can be segmented into supervised cross-domain dictionary learning algorithms, semi-supervised cross-domain dictionary learning algorithms, and unsupervised cross-domain dictionary learning algorithms. Correspondingly, the domain-adaptive dictionary learning algorithms can also be categorized into three distinct groups: supervised domain-adaptive dictionary learning algorithms, semi-supervised domain-adaptive dictionary learning algorithms, and unsupervised domain-adaptive dictionary learning algorithms. Then, we provide detailed analyses of the research progress and faced challenges, and elaborate on the working principles of a typical algorithm for each category. Secondly, to validate the competence of the domain-adaptive dictionary learning and cross-domain dictionary learning algorithms, we conducted a comparative evaluation against several forefront dictionary learning algorithms, shallow transfer learning algorithms, and deep transfer learning algorithms. Experimental results show that the transfer dictionary learning algorithms exhibit superior performance than the forefront dictionary learning algorithms and transfer learning algorithms on the five well-known datasets. Compared to the single-domain limitation of traditional dictionary learning, cross-domain dictionary learning uses multi-domain data to enrich feature representation. This not only enhances the dictionary discriminative but also improves model generalization. Moreover, domain-adaptive dictionary learning integrates domainadaptive characteristics while retaining the advantages of dictionary learning. This achieves dual optimization for sparse representation of features and cross-domain transfer. Lastly, we analyze the challenges faced by the transfer dictionary learning algorithms and present five future research directions. This survey would aid readers in achieving a systematic understanding of dictionary learning and transfer learning, selecting appropriate cross-domain and domain-adaptive dictionary learning algorithms for different scenarios.

# 1. Introduction

Transfer learning facilitates the dissemination of knowledge gained in one domain to other related domains, and it can effectively augment the generalized ability [1–3]. Recently, the utilization of transfer learning spans various disciplines such as speech recognition [4], natural language processing [5], computer vision [6], and reinforcement learning [7,8]. To improve the transfer performance, many transfer learning methods have been proposed [9–11].

In the realm of transfer learning, cross-domain learning, and domain-adaptive learning are two main research directions. Crossdomain learning involves transferring knowledge between entirely different domain data distributions, where the data in the target and source domains may have distinct feature spaces and distributions [12– 15]. Domain-adaptive learning focuses on transferring knowledge between similar yet not identical domain data distributions, where the feature spaces between target and source domain data are similar but their distributions differ [16]. Although the cross-domain learning and domain-adaptive learning algorithms have achieved competitive performance, there are still numerous challenges that need to be addressed. One challenge is efficiently extracting pertinent features that possess across-domain applicability and can overcome data distribution mismatch. Furthermore, the presence of domain-specific features adds complexity to this task. Effectively capturing and utilizing common features across domains while ignoring or reducing the influence of domain-specific features is also a key challenge. Additionally, the scarcity or outright absence of labeled target domain data poses a significant impediment to knowledge transfer, particularly when labels are limited. To furnish a comprehensive overview and analysis of the advancements in transfer learning, researchers have conducted reviews of the transfer learning algorithms from diverse perspectives $[ 1 , 1 7 -$ 20]. Notably, [1] provides a systematic and exhaustive analysis of the current transfer learning mechanisms and strategies, encompassing both data and model perspectives.

Dictionary Learning (DL) has achieved remarkable results in different applications due to sparsity and flexibility, such as image recognition [21,22], denoising [23,24], and fault diagnosis [25,26]. In general, DL can be divided into three categories: synthesis dictionary, analysis dictionary, and analysis-synthesis dictionary. (1) Synthesis dictionary learning model is used to learn dictionaries for reconstructing the samples. Researchers have proposed a large number of synthesis DL algorithms [27–29]. To address the requirements of extensive data processing, [27] presented an efficient online algorithm for synthesis DL. To improve the discriminative ability of the learned synthesis dictionary, [28] adopted a hierarchical strategy to establish two levels of discriminative synthesis dictionaries: class-level synthesis dictionary and image-level synthesis dictionary. In addition, [29] introduced a discriminative synthesis dictionary framework, using the similarity characteristics of coding coefficients and atoms to design the discriminative term. (2) The analysis DL model primarily aims to analyze samples while promoting sparsity in the analysis coefficients. Due to its poor discriminative ability, the analysis DL model is often employed for reconstruction [30] and denoising [31]. However, the analysis DL model provides a more intuitive description of the role of analyzing atoms, while also being highly efficient. Therefore, the analysis DL model has garnered widespread attention in discriminative pattern classification tasks [32]. To enhance the discriminability of the learned analysis dictionary, [33] constructed the code consistent and local topology terms, and added them into the basic analysis DL model. Moreover, [34] proposed a discriminative structured analysis dictionary by constructing the union of subspace to improve the discriminative ability of analysis DL. (3) The analysis-synthesis DL model learns the synthesis dictionaries and analysis dictionaries simultaneously [35]. In contrast to the analysis dictionary that exhibits inferior discrimination performance, the analysis-synthesis DL model is an effective approach to improve discrimination performance of the learned dictionaries. Recently, [36] proposed an analysis-synthesis dictionary pair learning algorithm. And, it utilized the local constraints term to enhance intra-class similarity in the learned representations. Additionally, the local order-preserving constraints [37] and scale constraints [38] have been incorporated into the analysis-synthesis dictionary pair learning model. Although the above DL algorithms have achieved excellent performance, they cannot extract the complex and deep features of data. Therefore, researchers have proposed deep DL model [39,40]. In 2016, [41] proposed the concept of deep DL. Recently, [42] introduced an intra-class and inter-class induced discriminative deep DL algorithm for visual recognition. Moreover, a hierarchical local-aware deep learning framework is proposed in [43]. Those deep DL algorithms all achieved better performance than the shallow DL algorithms and some deep learning algorithms. Furthermore, several surveys on DL have been proposed [44–46]. Notably, [44] provided a comprehensive overview of DL algorithms and discussed the discriminative power of sparse representation.

The sparsity constraint plays a significant role in enhancing the representation efficiency of the dictionary, improving the generalization ability of the learned model, and optimizing computational performance. The sparsity constraint usually contains the $\ell _ { 0 } .$ -norm, $\ell _ { 1 }$ -norm, $\ell _ { p } { \cdot } \mathrm { n o r m } , \ell _ { 1 / 2 } { \cdot } \mathrm { n o r m } _ { }$ , and group log-regularizer. The $\ell _ { 0 } .$ -norm is defined as the number of non-zero elements in a vector. In DL, it serves to promote sparsity in the representation, implying that only a small subset of dictionary atoms is used to represent a signal. In [47], the $\ell _ { 0 } { \cdot } \mathbf { n o r m }$ is leveraged to constrain the coding coefficients. However, the $\ell _ { 0 } .$ -norm is an NP-hard problem, it is usually intractable to solve directly. The $\ell _ { 1 }$ -norm represents the summation of the absolute values of the vector’s elements. It finds extensive application in spare coding and DL [48]. Unlike the $\ell _ { 0 } { \cdot } \mathbf { n o r m } ,$ , the $\ell _ { 1 } { \mathrm { - n o r m } }$ is convex and easy to derive of the closed-form solutions. Nevertheless, the $\ell _ { 1 } { \mathrm { - n o r m } }$ is not sufficiently sparse. The $\ell _ { p } .$ -norm refers to the sum of the absolute values of the elements raised to the power of $p .$ To effectively obtain the strong sparsity-promoting solutions, the $\ell _ { p }$ -norm was proposed in [49]. Moreover, the $\ell _ { p } .$ -norm is another popular sparsity constraint method in many fields such as hyperspectral imagery target detection [50], image reconstruction [51], and face recognition [52]. The $\ell _ { p } { \bf - n o r m }$ provides a balance between sparsity and smoothness, and the choice of $p$ influencing the level of sparsity enforced. [53] proposed $\ell _ { 1 / 2 ^ { - } }$ norm as a representative of $\ell _ { p } { \cdot } \mathrm { n o r m } ~ ( 0 < p < 1 )$ . The $\ell _ { 1 / 2 } { \cdot } \mathrm { n o r m }$ has been introduced to promote sparsity in DL [54–56]. Furthermore, the nonconvex $\ell _ { 1 / 2 }$ -norm needs a fast and reliable solver. [57] proposed an iterated shrinkage algorithm to swiftly and efficiently deal with the $\ell _ { 1 / 2 } { \cdot } \mathrm { n o r m }$ optimization problem. The group log-regularizer applies a logarithmic penalty to the norm of each group of dictionary atoms, encouraging sparsity within each group. In [58], the group log-regularizer was employed to enforce sparsity at the group level.

Recently, DL, as an effective feature extraction method, has been extensively utilized in the transfer learning frameworks [59–62]. To tackle cross-scene classification problems, [60] proposed a Semi-Supervised Double Dictionary Non-Negative Matrix Factorization (SS-DDNMF) algorithm to mitigate differences in features across domains. Moreover, [61] designed a cross-domain discriminant DL algorithm based on Projection Dual Reconstruction (PDR), which can align data from different domains. Furthermore, [62] proposed a multikernel Domain-Adaptive DL (DADL) method that leveraged dictionaries to learn discrimination features from both domains. The aforementioned research demonstrates that DL can bolster the effectiveness of the transfer learning algorithms to some degree. However, there is a lack of a comprehensive literature review on the Transfer DL (TDL) algorithms. Such a review would seek to analyze and synthesize the current research progress, pinpoint existing challenges, and propose future research directions.

Overall, this paper offers a comprehensive overview of TDL, providing readers with a systematic understanding of both DL and its application in transfer learning. Specifically, this survey will assist in selecting suitable Cross-Domain DL (CDDL) and DADL algorithms for diverse practical applications. CDDL aims to learn a shared dictionary capable of representing data from several domains. The main challenge in CDDL is finding a common dictionary that captures both domain-invariant features and domain-specific features. The learning dictionary is adaptable enough to handle variations between domains while remaining general. Moreover, CDDL deals with more complex scenarios where domains have different feature spaces, label spaces, or distributions. Different from CDDL, DADL primarily focuses on scenarios where there are discrepancies in data distributions, but the label spaces and feature spaces are the same or similar. DADL adjusts the dictionary to match the specific features or distributions of each domain. This often involves learning separate dictionaries for each domain or learning a common dictionary to fit the specific features of each domain. Firstly, we divide CDDL algorithms into three distinct categories: Supervised CDDL (SCDDL) algorithms, Semi-Supervised CDDL (SSCDDL) algorithms, and Unsupervised CDDL (UCDDL) algorithms. Subsequently, we analyze their advantages and limitations, and elucidate the working principles of typical algorithms within each category. Secondly, according to the level of supervision, the DADL algorithms are categorized as Supervised DADL (SDADL) algorithms, Semi-Supervised DADL (SSDADL) algorithms, and Unsupervised DADL (UDADL) algorithms. Likewise, typical algorithms of each category are presented along with an analysis of their working principles, and a summary of the pros and cons of each category. Finally, we compare some typical DL algorithms, Shallow Transfer Learning (STL) algorithms, and Deep Transfer Learning (DTL) algorithms with the TDL algorithms on the five renowned datasets. The experiment results indicate that the TDL algorithms show better performance than the compared algorithms. Furthermore, we explore the challenges faced in TDL and offer five promising avenues for future research.

Table 1 Notations used throughout the paper. 

<table><tr><td>Description</td><td>Notation</td></tr><tr><td>Number of target domains</td><td> $F$ </td></tr><tr><td>Number of atoms in dictionary  $D$ </td><td> $h$ </td></tr><tr><td>Number of sample</td><td> $N$ </td></tr><tr><td>Number of class</td><td> $C$ </td></tr><tr><td>Rearrange source domain data</td><td> $S$ </td></tr><tr><td>Rearrange target domain data</td><td> $T$ </td></tr><tr><td>Rearrange source domain data in the cth class</td><td> $S_c$ </td></tr><tr><td>Rearrange target domain data in the cth class</td><td> $\frac{T_c}{S_c}$ </td></tr><tr><td>Complementary matrices of  $S_c$ </td><td> $\frac{T_c}{S_c}$ </td></tr><tr><td>Complementary matrices of  $T_c$ </td><td> $Y_s$ </td></tr><tr><td>Source domain data</td><td> $Y_t$ </td></tr><tr><td>Target domain data</td><td> $Y$ </td></tr><tr><td>Source domain data and target domain data</td><td> $Y^{(m)}$ </td></tr><tr><td>Source domain data and target domain data at the mth layer</td><td> $D'$ </td></tr><tr><td>Common dictionary shared among the domains</td><td> $\Delta D_T^{(f)}$ </td></tr><tr><td>Individual dictionary for the fth target domain</td><td> $\Lambda$ </td></tr><tr><td>Label vector matrix</td><td> $Z$ </td></tr><tr><td>Sparse coding coefficient matrix</td><td> $Q_S$ </td></tr><tr><td>Representation coefficient matrix for the source domain</td><td> $Q_T^f$ </td></tr><tr><td>Representation coefficient matrix for the fth target domain</td><td> $Y_s^{(m)}$ </td></tr><tr><td>Source domain data at the mth layer</td><td> $Y_t^{(m)}$ </td></tr><tr><td>Target domain data at the mth layer</td><td> $Y_{s,c}^{(m)}$ </td></tr><tr><td>Source domain data in class c at the mth layer</td><td> $Y_{t,c}^{(m)}$ </td></tr><tr><td>Target domain data in class c at the mth layer</td><td> $Y_{s}^{(m)}$ </td></tr><tr><td>The sparse coding coefficient matrix for  $Y_s^{(m)}$ </td><td> $Z_s^{(m)}$ </td></tr><tr><td>The sparse coding coefficient matrix for  $Y_t^{(m)}$ </td><td> $Z_t^{(m)}$ </td></tr><tr><td>The sparse coding coefficient matrix for  $Y_{s,c}^{(m)}$ </td><td> $Z_{s,c}^{(m)}$ </td></tr><tr><td>The sparse coding coefficient matrix for  $Y_{t,c}^{(m)}$ </td><td> $Z_{t,c}^{(m)}$ </td></tr><tr><td>Learned dictionary</td><td> $D$ </td></tr><tr><td>Projection matrix</td><td> $P$ </td></tr><tr><td>Projection matrix for the source domain</td><td> $P_S$ </td></tr><tr><td>Projection matrix for the target domain</td><td> $P_T$ </td></tr><tr><td>Transformation matrix</td><td> $A$ </td></tr><tr><td>Input data matrix (Training data matrix)</td><td> $X$ </td></tr><tr><td>Weight matrix</td><td> $W$ </td></tr><tr><td>The synthesis dictionary</td><td> $D_s$ </td></tr><tr><td>The cth class weight matrix</td><td> $W_c$ </td></tr><tr><td>The analysis dictionary</td><td> $D_p$ </td></tr><tr><td>Frobenius norm</td><td> $\| \cdot \|_F$ </td></tr><tr><td> $\ell_1$ norm</td><td> $\| \cdot \|_1$ </td></tr></table>

The paper’s primary contributions consist of the following points:

(1) As far as we are aware, the paper represents the inaugural survey on the TDL algorithms. We primarily analyze the research progress of the CDDL algorithms and DADL algorithms, as well as how DL can enhance the performance of transfer learning.   
(2) We categorize the TDL algorithms into six categories and provide a detailed analysis of each category. Moreover, we also present the working principles of typical algorithms of each class and discuss their strengths and weaknesses.   
(3) From an experimental perspective, the performance of the TDL algorithms is analyzed. Some typical algorithms from both CDDL and DADL are selected for comparative experiments against some typical DL algorithms, STL algorithms, and DTL algorithms on the five well-known datasets. Furthermore, current challenges faced by the TDL algorithms are summarized, and five future research directions are proposed.

The remainder of the review is organized as follows. Section 2 presents the relevant theories, primarily introducing transfer learning and DL. Section 3 offers a survey of the CDDL algorithms. The survey of the DADL algorithms is given in Section 4. Section 5 comprises experimental comparative analysis. Section 6 discusses the challenges of the TDL algorithms and proposes five future research directions. Finally, the presentation of conclusions in Section 7. The organization of this survey paper is shown in Fig. 1.

# 2. Related works

In this section, we provide details on transfer learning and present two typical algorithms in this field: cross-domain learning and domainadaptive learning. Additionally, we also elaborate on DL, delving into the supervised DL algorithms, semi-supervised DL algorithms, and unsupervised DL algorithms. The notations used in the paper are shown in Table 1, and some commonly used abbreviations in the paper are listed in Table 2.

# 2.1. Transfer learning

Machine learning often assumes that the data used for training and testing adheres to a consistent distribution. However, the real-world data often comes from different distributions. For example, source and target domains may differ in feature space, class space, or marginal distribution. Transfer learning is a machine learning technique that improves performance in a target domain by leveraging knowledge from a related source domain. In the realm of transfer learning, a domain can be understood as a specific area at a given time. For instance, sketching and oil painting can be viewed as two different domains. The domain from which knowledge is transferred, containing knowledge, is usually called the source domain, while the learning domain is referred to as the target domain. A task denotes the specific work or goal that needs to be accomplished within a particular domain, such as sentiment analysis and image recognition, which represent different tasks. According to [63], transfer learning can be defined as follows: transfer learning is employed to enhance the learning effectiveness of the target predictive function in the target domain. This is achieved by leveraging knowledge from a source domain and a source task when the source domain differs from the target domain or the source task differs from the target task for a given target learning task.

In general, transfer learning contains cross-domain learning, domain-adaptive learning, domain generalization, and so on. In this survey, we are only concerned with the CDDL and the DADL algorithms.

# 2.1.1. The cross-domain learning algorithms

To achieve knowledge transfer between different domains, researchers have proposed numerous cross-domain learning algorithms, such as feature transformation [64], support vector machines [65], and generative adversarial networks [66]. In particular, a typical crossdomain learning algorithm was proposed by Fang et al. [67], which was called Projective Cross-Reconstruction (PCR). The objective function of PCR can be articulated in the following manner:

$$
\min _ {P _ {S}, P _ {T}} \| P _ {S} ^ {T} S - P _ {S} ^ {T} T P _ {T} ^ {T} S \| _ {F} ^ {2} + \| P _ {S} ^ {T} T - P _ {S} ^ {T} S P _ {T} ^ {T} T \| _ {F} ^ {2} \tag {1}
$$

where $P _ { S } ^ { T }$ is the transpose matrix of $P _ { S } \mathrm { : }$ , and $P _ { T } ^ { T }$ is the transpose matrix of $P _ { T } .$ . Specifically, the application of these projection matrices $P _ { S }$ and $P _ { T }$ results in obtaining projected data from the source and target domains, respectively. The purpose of PCR is to use the projected data for reconstruction. The data from the source and target domains are rearranged accordingly. PCR ensures that data from disparate domains is integrated, leading to a reduction in distribution mismatch.

# 2.1.2. The domain-adaptive learning algorithms

Domain-adaptive learning aims to bridge different distribution domains by utilizing data and model from the source domain [68]. The target domain, though correlated, differs in marginal distributions, while feature and class spaces remain consistent [63]. According to [69], domain-adaptive is defined as follows: domain-adaptive assumes that although the feature spaces and tasks in different domains remain consistent, there are discrepancies in the distributions of data between the source and target domains. Two domains, source and target, are defined on the joint feature and label space with different distributions. Domain-adaptive aims to transfer the learned knowledge from the source domain to the target domain to perform a specific task, which is common to both domains. Domain-adaptive algorithms aim to address data distribution mismatch in cross-domain learning. These algorithms adjust model parameters or structures through various techniques to adapt the data distributions of different domains. The main domain-adaptive algorithms are summarized below.

Table 2 Abbreviations used throughout the paper. 

<table><tr><td>Abbreviation</td><td>Meaning</td></tr><tr><td>SS-DDNMF</td><td>Semi-Supervised Double Dictionary Non-Negative Matrix Factorization</td></tr><tr><td>CDDL</td><td>Cross-Domain Dictionary Learning</td></tr><tr><td>DADL</td><td>Domain-Adaptive Dictionary Learning</td></tr><tr><td>CCA</td><td>Canonical Correlation Analysis</td></tr><tr><td>CNN</td><td>Convolutional Neural Network</td></tr><tr><td>MMD</td><td>Maximum Mean Discrepancy</td></tr><tr><td>EEG</td><td>Electroencephalogram</td></tr><tr><td>ECG</td><td>Electrocardiogram</td></tr><tr><td>PPG</td><td>Photoplethysmogram</td></tr><tr><td>LPP</td><td>Locality Preserving Projection</td></tr></table>

![](images/99d536a967fdf9dbaf08d922ad0e37c31081746389a6444fa7769541e5d9f062.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Transfer dictionary learning"] --> B["Section 1: Introduction"]
    A --> C["Section 2: Related works"]
    A --> D["Section 3: The cross-domain dictionary learning algorithms"]
    A --> E["Section 4: The domain-adaptive dictionary learning algorithms"]
    A --> F["Section 5: Experiments"]
    A --> G["Section 6: Future directions"]
    A --> H["Section 7: Conclusions"]

    B --> I["Transfer learning [1, 2, 3, 4"]]
    I --> J["The cross-domain learning algorithms [65, 66, 67, 78"]]
    I --> K["The domain-adaptive learning algorithms [64, 69, 70, 71"]]
    C --> L["Dictionary learning [21, 22, 23, 24"]]
    L --> M["The supervised dictionary learning algorithms [72, 73, 74, 75"]]
    L --> N["The semi-supervised dictionary learning algorithms [35, 76"]]
    L --> O["The unsupervised dictionary learning algorithms [77, 78"]]
    D --> P["The supervised cross-domain dictionary learning algorithms [61, 80, 81"]]
    D --> Q["The semi-supervised cross-domain dictionary learning algorithms [82, 83, 84, 85"]]
    D --> R["The unsupervised cross-domain dictionary learning algorithms [87, 88, 89, 90"]]
    E --> S["The supervised domain-adaptive dictionary learning algorithms [92, 93"]]
    E --> T["The semi-supervised domain-adaptive dictionary learning algorithms [62, 94, 95"]]
    E --> U["The unsupervised domain-adaptive dictionary learning algorithms [96, 97, 98, 99"]]
```
</details>

Fig. 1. The organization of this survey paper.

(1) Instance reweighting domain-adaptive: Instance reweighting seeks to directly deduce resampling weights through nonparametric

feature distribution alignment across diverse domains. The reweightingbased domain-adaptive methods are applicable in situations where the difference between the source and target domain is not too large. By adjusting the weights of samples in the source domain, the model focuses more on samples that are similar to the target domain, thereby reducing the distribution discrepancy between the source and target domains.

(2) Feature alignment domain-adaptive: To adapt data from multiple domains, the technique aims to find a common feature space between the source domain and the target domain. Among these, DADL is a representative algorithm that emphasizes finding a set of bases $( \mathrm { i . e . } ,$ , dictionary) and representation coefficients in different domains. This feature alignment strategy helps the model capture common features across domains, thereby enhancing generalization performance in the target domain.

(3) Adversarial domain-adaptive: This method introduces a domain classifier to distinguish whether the input data comes from the source domain or the target domain. Meanwhile, the model attempts to deceive the domain classifier, making it unable to accurately determine the source domain of the data. Through this adversarial competition, the model can learn a domain-independent representation that generalizes well to data from different domains. The effectiveness of the adversarial training strategy lies in the model can learn a universal feature that transcends domain boundaries, effectively overcoming data distribution mismatches.

The domain-adaptive algorithms are prevalently applied in image recognition, with a notable algorithm introduced by Lu et al. [70] being a Domain-Adaptive Deep Neural Network Model for Fault Diagnosis (DAFD). The objective function of DAFD comprises three distinct components:

(1) The fundamental loss term $L _ { a e }$ for constructing the deep neural network is identical in form to Eq. (2):

$$
L _ {a e} = \frac {1}{2 N} \| R - X \| _ {F} ^ {2} \tag {2}
$$

The decoder, denoted as $R ,$ aims to reconstruct the input data from the extracted features.

(2) The Maximum Mean Discrepancy (MMD) term $L _ { M M D }$ is designed to minimize the difference between the distributions of the source domain and target domain;

Let $\{ \omega _ { i } \} _ { i = 1 } ^ { C }$ 1 denotes the set of categories that are commonly shared by the domains, and the MMD term is then evaluated as follows:

$$
L _ {M M D} = \sum_ {c = 1} ^ {C} \| \frac {1}{N _ {\omega c} ^ {s}} \sum_ {p = 1} ^ {N _ {\omega c} ^ {s}} \phi (X _ {\omega c} ^ {s}, p) - \frac {1}{N _ {\omega c} ^ {t}} \sum_ {q = 1} ^ {N _ {\omega c} ^ {t}} \phi (X _ {\omega c} ^ {t}, q) \| _ {H} ^ {2} \tag {3}
$$

where $X _ { \omega c } ^ { s }$ and $X _ { \omega c } ^ { t }$ are the samples of the $\omega _ { c }$ category from $Y _ { s }$ and $Y _ { t } ,$ respectively. Additionally, $N _ { \omega c } ^ { s }$ and $N _ { \omega c } ^ { t }$ are the sample numbers of the $\omega _ { c }$ category from $Y _ { s }$ and $Y _ { t } ,$ respectively. The function ??(⋅) represents the process of extracting features from the input data via the deep neural network. ?? denotes the element of the category set. The ?? (superscript) and ?? (superscript) represent the source and target. ?? is the feature matrix of ??.

(3) The regularization term $L _ { w e i g h t }$ on the weights serves to enhance the representational properties of the original data features.

$$
L _ {\text { weight }} = \sum_ {i = 1} ^ {e} \exp (- \| W _ {i} \| _ {F / \sigma} ^ {2}) \tag {4}
$$

When referring to the weight matrix $( \{ W _ { i } \} _ { i = 1 } ^ { e } )$ set related to the transferable features. The parameter ?? represents the scope of values within which the weights will be strengthened, also known as the punishment factor. As the value of $\Vert \ W _ { i } \Vert _ { F } ^ { 2 }$ increases, the value of $L _ { w e i g h t }$ decreases. Therefore, by optimizing the $L _ { w e i g h t }$ term with a suitable punishment coefficient, the desired goal can be achieved.

The integration of Eqs. (2), (3), and (4) yields Eq. (5). The objective function of DADF is as shown in Eq. (5):

$$
L _ {D A F D} = L _ {a e} + \nu L _ {M M D} + \frac {\varphi}{2} L _ {w e i g h t} \tag {5}
$$

where $\nu > 0$ and $\varphi > 0$ are the tunable parameters, they regulate the balance between the $L _ { M M D }$ term and the $L _ { w e i g h t }$ term, respectively. The DAFD seeks to learn transferable features that reduce domain discrepancies while enhancing the inherent recognizability of the original data.

# 2.2. Dictionary learning

DL, an important direction in sparse theory, aims to obtain the optimal representation of data with certain strategies. DL is generally classified into supervised, semi-supervised, and unsupervised learning based on whether training sample labels are utilized.

# 2.2.1. The supervised dictionary learning algorithms

In supervised $\mathrm { D L } ,$ supervision information, i.e., data labels, is utilized during the DL process. It can employ the label information of training data or testing data to learn dictionaries, taking full advantage of label information to optimize dictionary elements [21,71–73]. A typical supervised DL algorithm is the Label Consistent K-SVD (LC-KSVD) proposed by Jiang et al. [74], whose objective function is given by Eq. (6):

$$
\arg \min _ {D, M, A, Z} \| X - D Z \| _ {2} ^ {2} + \alpha \| B - A Z \| _ {2} ^ {2} + \beta \| \Lambda - M Z \| _ {2} ^ {2}, s. t. \| z _ {i} \| _ {0} \leq b \tag {6}
$$

where ?? denotes the parameters of the classifier and ?? represents the discriminative and sparse representation of ??. ?? is a sparsity constraint factor and $Z = [ z _ { 1 } , \dots , z _ { g } ] ( i \in \{ 1 , 2 , \dots , g \} ) . \parallel \cdot \parallel _ { 0 }$ is the $\ell _ { 0 }$ norm. ?? and $\beta$ represent two balancing factors. The reconstruction error is indicated by the first term. The classification error is represented by the third term, and the discriminative sparse code error is represented by the second term. The objective function aims to learn a dictionary that possesses both reconstructive and discriminative abilities. The amount of labeled training data has a significant impact on the LC-KSVD’s performance. Consequently, a lexicon with insufficient labeled training data may exhibit poor generalization performance.

# 2.2.2. The semi-supervised dictionary learning algorithms

The semi-supervised DL algorithms are utilized the labeled and unlabeled data to learn dictionaries for different applications [75]. The key is to classify unlabeled data by using label information from labeled data as a guide. The initial dictionary is formed using labeled data, and unlabeled data is classified accordingly. Then, correctly classified, unlabeled data is added to the dictionary, forming a semi-supervised DL algorithm [35]. The general framework of the semi-supervised DL algorithms is shown in Eq. (7):

$$
\arg \min _ {D, Z _ {L}, Z _ {U}, O} Y (X _ {L}, X _ {U}, D, Z _ {L}, Z _ {U}) + \lambda_ {1} \Psi (Z _ {L}, Z _ {U}, O) + \lambda_ {2} \Phi (Z _ {L}) \tag {7}
$$

where the labeled training data is represented by $X _ { L } .$ . The unlabeled training data is denoted by $X _ { U }$ . Moreover, $Z _ { L }$ represents the coding coefficient matrix for $X _ { L }$ . For $X _ { U } ,$ , the coding coefficient matrix is denoted by $Z _ { U } .$ . As part of the regularization second term, ?? is the parameter of the model matrix that needs to be trained. The tradeoff parameters $\lambda _ { 1 }$ and $\lambda _ { 2 }$ serve to balance the different terms. The first term stands for the reconstruction error of $X _ { L }$ and $X _ { U }$ concerning ??. The regularization term for $Z _ { L }$ and $Z _ { U }$ is represented by the second term. The discriminative term for $Z _ { L }$ is represented by the third term. The semi-supervised DL algorithms are capable of leveraging a vast amount of unlabeled training data, making it suitable for scenarios where labeled training data is limited but unlabeled training data is abundant.

# 2.2.3. The unsupervised dictionary learning algorithms

The unsupervised DL algorithms do not make use of label information [76]. To optimize dictionary atoms using the latent structure of data to capture its intrinsic structure and features. The K-Means algorithm is a representative of unsupervised DL [77], whose objective function is given by $\operatorname { E q . }$ . (8):

$$
J (X, V) = \sum_ {k = 1} ^ {K} \sum_ {x _ {i} \in X _ {k}} \| x _ {i} - v _ {k} \| _ {2} ^ {2} \tag {8}
$$

![](images/380be1d2449d595ee0c46bfa08683021f41fc12fa34082948616824fc9995b5d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["2013: Ni et al. [88"]] --> B["2018: Ding and Fu [84"]]
    B --> C["2019: Li et al. [87"]]
    C --> D["2020: Ni et al. [79"]]
    D --> E["2021: Wang et al. [89"]]
    E --> F["2022: Chen et al. [86"]]
    F --> G["2023: Tian et al. [82"]]
    H["2013: Qi et al. [78"]] --> I["2019: Han et al. [61"]]
    I --> J["2020: Tian et al. [81"]]
    J --> K["2021: Huang et al. [90"]]
    K --> L["2022: Zhang et al. [85"]]
    L --> M["2023: Luo et al. [83"]]
    N["Chen et al. [60"]] --> O["2020"]
    O --> P["2021"]
```
</details>

Fig. 2. Temporal distribution of the CDDL algorithms.

Let $X = [ x _ { 1 } , \dots , x _ { N } ] ( i \in \{ 1 , 2 , \dots , N \} )$ . The data points belonging to the ??th cluster are contained in each $x _ { k } .$ The cluster center of the ??th cluster is $v _ { k } ,$ , and the number of clusters is denoted by ??. The K-Means clustering algorithm aims to achieve an optimal partitioning of the data by minimizing the total sum of squared distances between data points and their respective nearest cluster centers. The K-Means algorithm stands out for its simplicity, intuitive concept, and straightforward implementation, making it highly accessible and interpretable. Nevertheless, its sensitivity to noise and outliers poses a challenge, as these factors can significantly compromise the precision and reliability of clustering outcomes.

# 3. The cross-domain dictionary learning algorithms

The CDDL algorithms are designed to solve the problem of transferring knowledge across domains through DL. Their main objective is using DL to extract the most fundamental and representative features from various domains. By accomplishing this, they aim to enable similar distributions of data from different domains and promoting cross-domain transfer. Based on the utilization of class label information in the target domain, the CDDL algorithms can be generally divided into three types: supervised, semi-supervised, and unsupervised CDDL algorithms. The averaged recognition accuracy of the CDDL algorithms is shown in Table 6.

The temporal distribution of the CDDL algorithms is shown in Fig. 2. The timeline includes the development of CDDL algorithms in recent years. Each point on the timeline represents the year an algorithm was published, along with the corresponding algorithms introduced in that year. As shown in the figure, a total of 14 CDDL algorithms were proposed between 2013 and 2023. Notably, 2020 was the year with the most research activity, with five algorithms proposed during this period. To better describe the evolution process of different CDDL algorithms, the development of the CDDL algorithms is shown in Fig. 3. The development process consists of two main parts: foundational works and developmental works. In Fig. 3, ‘‘add’’ represents the incorporation of methods. For example, as indicated by the red arrow in the figure, [78] proposed a CDDL algorithm that utilizes a synthetic dictionary and transfer matrix. Building upon [78], [79] introduced metric learning and proposed an improved CDDL algorithm.

# 3.1. The supervised cross-domain dictionary learning algorithms

The SCDDL algorithms acquire transferable feature dictionaries from labeled source data, thereby improving their adaptability to various data distributions in cross-domain applications. A representative algorithm was the PDR, which was proposed by Han et al. [61]. The formulation of PDR’s objective function is outlined as follows in Eq. (9):

$$
\min _ {D, P, G \geq 0} \| S - D P T \| _ {F} ^ {2} + | S - D P S \| _ {F} ^ {2} + \beta \{\| P T -
$$

$$
(U + J \odot G) \| _ {F} ^ {2} + \| P S - (U + J \odot G) \| _ {F} ^ {2} \} + \gamma \| \tag {9}
$$

$$
\varLambda - A P S \parallel_ {F} ^ {2}, s. t. d _ {j} ^ {T} d _ {j} = 1, j \in \{1, 2, \ldots , h \}
$$

where ?? denotes a non-negative matrix and ?? is a different matrix. ?? represents a term for label consistency regularization that merges ?? with ?? to create a new, enhanced label consistency regularization term, denoted as $U + J \odot G . \odot$ represents the element-wise product operation. Moreover, ?? and ?? represent two balancing factors in the optimization process. Furthermore, the initial term represents the error in reconstructing the data across different reconstructions, and the second term represents the reconstruction error of the data itself. The third term represents a discriminative label consistency regularization term, stemming from an expansive iteration of [80]. The third term enables the comprehensive utilization of label confidence, thereby augmenting its discriminatory prowess. The fourth term represents a classification error. The PDR algorithm uses the PDR strategy to learn dictionaries for both its source and target domains. By simultaneously reconstructing own and cross-domain data, and leveraging the correlations between the originating and target domains, better data alignment can be achieved.

In order to better extract domain-invariant features, [81] (2020) proposed the Cross-Domain Joint Dictionary Learning (XDJDL) framework for reconstructing Electrocardiogram (ECG) signals from Photoplethysmogram (PPG) signals. The XDJDL algorithm learns a joint dictionary that contained multi-domain characteristics, establishing of connections between the two domains. By utilizing the learned joint dictionary to extract domain-invariant features, it facilitates crossdomain knowledge transfer. Subsequently, [82] (2023) introduced a label consistency constraint, proposing Label-Consistent XDJDL (LC-XDJDL), which aids in understanding PPG signals and further enhances the quality of ECG signal reconstruction.

The SCDDL algorithms can fully utilize label information to learn dictionaries specific to certain tasks, generally resulting in better performance on target tasks. However, they entirely depend on labeled data. The insufficient or low-quality labeled data can substantially impact performance, leading to poor generalization and the labeling process being costly and time-consuming. The summary of the SCDDL algorithms is shown in Table 3.

# 3.2. The semi-supervised cross-domain dictionary learning algorithms

The SSCDDL algorithms primarily leverage a limited quantity of labeled data within the target domain to learn a transfer dictionary. The main objective is to acquire a dictionary that better represents transferred data, thereby facilitating knowledge transfer between the source and target domains. A typical algorithm of SSCDL, as proposed by Luo et al. [83], is the Locality-Adaptive Structured Dictionary Learning (LASDL) algorithm, with its objective function expressed as $\operatorname { E q . }$ . (10):

![](images/a2b964ae1bfd464da4534bc7d25c29b79e664ec6ea588fe66347ff9c4832ff89.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Foundational Works"] -->|add| B["Developmental Works"]
    B --> C["Dictionary learning"]
    C --> D["Transfer learning"]
    D --> E["Asymmetric mapping"]
    E --> F["Top-push constrain [85"]]
    F --> G["Metric learning [79"]]
    G --> H["Label consistent constrain [61"]]
    G --> I["Sparse regularization [86"]]
    G --> J["Graph regularization [60, 89"]]
    G --> K["Label consistent constrain [61"]]
    C --> L["Subspace interpolation [90"]]
    L --> M["Transfer matrix [78"]]
    M --> N["Transfer matrix [83, 87"]]
    N --> O["Subspace interpolation [88"]]
    O --> P["Analysis dictionary"]
    P --> Q["Adaptive geometric structure preservation and graph regularization [81"]]
    P --> R["Analysis-synthetic dictionary"]
    R --> S["Deep dictionary"]
    S --> T["Domain-level adaptation and class-level adaptation"]
    T --> U["Label consistent constrain [84"]]
    C --> V["Dictionary learning"]
    V --> W["Transfer learning"]
```
</details>

Fig. 3. The development of the CDDL algorithms.

Table 3   
A summary of the SCDDL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>PDR [61]</td><td>1. By reconstructing source and target domain data using the dictionary, the alignment between different domains is enhanced, thereby improving the transferability of the dictionary.2. The integration of double reconstruction, label consistency constraints, and classifier learning into a unified optimization framework results in low computational complexity.</td><td>1. The larger size of dictionary can reduce classification accuracy.2. The parameters are sensitive.</td><td>Cross-domain recognition</td></tr><tr><td>XDJDL [81]</td><td>1. Capturing the intrinsic relationship between the source domain (PPG) and the target domain (ECG), enabling more accurate ECG reconstruction.2. Good adaptability and generalization capabilities.</td><td>1. Ignoring the utilization of additional class information may affect the reconstruction of target domain data.2. Dependency on specific datasets.</td><td>Reconstructing ECG signals from PPG signals</td></tr><tr><td>LC-XDJDL [82]</td><td>1. Incorporating label consistency constraints enhances the discriminative ability of the dictionary.2. It possesses good adaptability and generalization capabilities.</td><td>1. It is dependent on the label information.2. The performance is easily affected by noise.</td><td>Reconstructing ECG signals from PPG signals</td></tr></table>

$$
\begin{array}{l} \min _ {D _ {s}, D _ {p}, W} \sum_ {c = 1} ^ {C} \| S _ {c} - D _ {s c} D _ {p c} S _ {c} \| _ {F} ^ {2} + \| S _ {c} - D _ {s c} D _ {p c} T _ {c} \| _ {F} ^ {2} + \lambda \{\| D _ {p c} \overline {{S _ {c}}} \| _ {F} ^ {2} \\ + \left\| D _ {p c} \overline {{T _ {c}}} \right\| _ {F} ^ {2} \} + \alpha \{T r (S _ {c} ^ {T} D _ {p c} ^ {T} E _ {c} D _ {p c} S _ {c}) + T r (T _ {c} ^ {T} D _ {p c} ^ {T} E _ {c} D _ {p c} T _ {c}) \} + \tag {10} \\ \end{array}
$$

$$
\eta \Omega (D _ {p}, W) s. t. \| d _ {i} \| _ {2} ^ {2} \leq 1, d i a g (W _ {c}) = 0, i \in \{1, 2, \dots , h \}, c \in \{1, 2, \dots , C \}
$$

where $E _ { c }$ represents the graph Laplacian matrix for atoms belonging to the ??th class. $D _ { p c }$ represents the analysis dictionary in class $c ,$ and $D _ { s c }$ denotes the synthesis dictionary in class ??. The function $\mathcal { Q } ( D _ { p } , W )$ 号 serves as an adaptive reconstruction weight. The approach suggests adding an extra constraint on $D _ { p c } T _ { c }$ to reduce transitions across various domains, making certain that in every cycle, the coefficients $D _ { p c } T _ { c }$

remain closer to their respective class centers. The quality constraint $d i a g ( W _ { c } ) = 0 ~ \mathrm { f o r } ~ c \in \{ 1 , 2 , \ldots , C \}$ , imposed on ?? is meant to prevent a trivial solution. Additionally, ?? ??(⋅) is the trace of the matrix. $\lambda , \alpha ,$ and ?? are the balancing parameters. The first term represents the error in reconstructing the data itself, and the second term denotes the error in reconstructing the data cross-reconstruction. The third term is a regularization term. The fourth term represents the locality constraint of the ??th class. The LASDL algorithm segments data into smaller blocks to construct structured dictionaries. These dictionaries can capture both shared and unique features across different domains by employing sparse and norm constraints. This approach enhances the representational and generalization abilities of the dictionary, thereby facilitating cross-domain transfer. The LASDL algorithm takes advantage of local adaptability and structured DL strategies to address feature disparities between different domains more effectively and learn enriched feature representations.

Table 4 A summary of the SSCDDL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>DTLC [86]</td><td>1. Employing a multi-layer latent dictionary to extract more effective features.2. Utilizing rank minimization constraints to cluster similar samples across domains, enhancing the transfer effect.</td><td>1. High complexity and significant computational cost.2. Without labeled samples, it is unable to fully learn autonomously.</td><td>Cross-domain recognition</td></tr><tr><td>LASDL [83]</td><td>1. By training class-specific sub-dictionaries, the discrimination ability of the dictionaries can be improved, thereby reducing the distribution differences between domains.2. Utilizing an adaptive geometric structure preservation function helps to retain the local manifold structure of the data and minimizing distribution differences.</td><td>1. It is not able to tackle the unsupervised cross-domain recognition tasks.2. When small shifts between different domains and a strong discriminative ability is required, cross-domain recognition is clumsy.</td><td>Cross-domain recognition</td></tr><tr><td>SS-DDNMF [60]</td><td>1. Effectively addresses feature shift and differences in feature spaces.2. By adopting a category correspondence strategy between source domain and target domain samples, it does not require one-to-one sample correspondence, making it more flexible.</td><td>1. When only a small numbers of labeled samples are available, its performance may degrade.2. High sensitivity to the parameter: the rank (r) of factorization.</td><td>Cross-scene hyperspectral images classification</td></tr><tr><td>TCMDL [84]</td><td>1. By introducing cross-view consistency regularization and sparse regularization, robust feature representations can be effectively learned, thereby enhancing the performance of cross-modal person re-identification.2. Using the top-push ranking constraint makes dictionary learning more suitable for cross-modal person reidentification tasks, enabling better discrimination of subtle differences.</td><td>1. When using Local Maximal Occurrence (LOMO) feature representation, the performance of person re-identification is still limited.2. The lack of an end-to-end processing approach affects the robustness of the input features.</td><td>Cross-modality person re-identification</td></tr><tr><td>TMMLDL [79]</td><td>1. By combining metric learning and DL, it can effectively capture the discriminative information across different domains.2. Utilizing the global structural information and pairwise constraints of training samples, it can reduce the impact of cross-domain distribution differences.3. The collaborative learning approach ensures that both metric learning and DL can achieve optimality.</td><td>1. How to avoid negative transfer has yet to be further studied.2. The optimization process is complex.</td><td>Cross-domain facial expression recognition</td></tr><tr><td>TSSR [85]</td><td>1. Achieving subspace alignment between source and target data, it can effectively improve cross-domain transfer.2. Preserving the local geometric structure of the source and target domain data by using the graph Laplacian regularization term, it can enhance the discriminative ability of the learned dictionary.</td><td>1. High computational cost.2. Its performance on other comprehensive tasks remains to be further verified.</td><td>Cross-corpus facial expression recognition</td></tr></table>

To take advantage of the scarce label information available in the target domain, [60] (2020) introduced a dual dictionary non-negative matrix factorization approach for heterogeneous transfer learning. By acquiring distinct dictionaries for source and target domains, features of varying dimensions are projected into a shared low-dimensional space, effectively bridging the gaps between these feature spaces. [84] (2020) applied the top-push method to constrain dictionary atoms, subsequently building mappings between source and target domains, matching local data distributions, and facilitating cross-domain transfer. Subsequently, [79] (2021) learned feature dictionaries from both source and target domains, using metric learning to boost domain correlation, reduce discrepancies, and improve cross-domain transfer ability. [85] (2023) applied DL for sparse coding of source and target domain data, leveraging sparse representations to extract common features from both, thereby reducing cross-domain distribution differences. Moreover, deep learning has been integrated into the TDL models. [86] (2019) utilized deep Convolutional Neural Network (CNN) to extract features. Subsequently, they employed low-rank coding techniques to encode the features, ensuring that the encoded features possess low rank. This characteristic enhances their compatibility with the data distribution within the specified domain.

The SSCDDL algorithms can use limited label information to enhance the accuracy of feature transfer. In some fields, there may be a shortage of labeled data but an abundance of unlabeled data, making the SSCDDL algorithms more suitable for practical needs. However, implementing the SSCDDL algorithms often involves dealing with complex optimization issues and adjusting parameters, which can make the process more challenging. The summary of the SSCDDL algorithms is shown in Table 4.

# 3.3. The unsupervised cross-domain dictionary learning algorithms

Using the unsupervised DL algorithms for cross-domain transfer focuses on extracting common features across domains from vast unlabeled datasets. Specifically, the UCDDL algorithms represent the data from the source and target domains in a common low-dimensional space, where dictionaries are learned to capture the similarities between the source and target domains. The UCDDL algorithms aid in mapping data from the source domain to the target domain, thus realizing cross-domain transmission. A typical method, such as the Sparse Subspace Correlation Analysis (SSCA) proposed by [87], has its

Table 5 A summary of the UCDDL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>SSCA-SC [87]</td><td>1. Able to handle heterogeneous data.2. Leverage label information to learn more discriminative representations.3. Designed a discriminative and robust classifier adapted to the new data representations, improving classification performance.</td><td>1. During the training process, it needs to be trained for each category, leading to a high computational complexity.2. High sensitivity to the parameter.</td><td>Hyperspectral image classification</td></tr><tr><td>UDDL [88]</td><td>1. Interpolating subspace through dictionary learning helps to better bridge the two domains, improving cross-domain recognition performance.2. By quantitatively measure the differences between two domains, it enables the selection of the optimal source domain to adapt to the target domain, which is particularly important when multiple source domains exist.</td><td>1. High computational complexity.2. When the differences between the source and target domains are large, its effectiveness may be affected.</td><td>1. Face recognition2. Object recognition</td></tr><tr><td>JDMF [89]</td><td>1. The use of JDMF effectively extracts domain-invariant features, reducing cross-domain discrepancies.2. By leveraging the shared latent feature space constructed by JDMF, the CNN model trained on the source domain can be transferred to the target domain, effectively achieving cross-domain transfer.</td><td>1. Negative transfer may impact its performance.2. The uncertainty modeling of the cross-operating condition Remaining Useful Life (RUL) estimation needs further exploration.</td><td>Cross-operating condition RUL estimation</td></tr><tr><td>TDL [90]</td><td>1. By adopting smooth subspace interpolation techniques, the source domain dictionary updates can better adapt to the target domain, effectively addressing the distribution differences that exist between the source and target domain data.2. It realizes cross-domain multi-mode industrial process monitoring and fault isolation, effectively addressing complex industrial scenarios.</td><td>1. In the industrial field, the issue of data dimensionality deserves further research.2. How to transmit process information in a distributed manner is worth studying.</td><td>Cross-domain industrial process detection</td></tr><tr><td>UJSDL [78]</td><td>1. The model can address both cross-view and cross-domain distribution discrepancies simultaneously.2. Through the joint optimization of subspace learning and DL, it can better extract discriminative features across domains.</td><td>1. It requires simultaneous optimization of both the subspace and the dictionary, leading to relatively high computational complexity.2. Negative transfer is a challenge.</td><td>Cross-domain person re-identification</td></tr></table>

objective function as Eq. (11):

$$
\min _ {A _ {s}, A _ {t}, Z _ {s}, Z _ {t}} \| Z _ {s} \| _ {1} + \| Z _ {t} \| _ {1} + \lambda_ {1} \| A _ {s} ^ {T} Y _ {s} - A ^ {T} Y Z _ {s} \| _ {F} ^ {2} + \tag {11}
$$

$$
\lambda_ {2} \parallel A _ {t} ^ {T} Y _ {t} - A ^ {T} Y Z _ {t} \parallel_ {F} ^ {2} - \beta T r (A _ {s} ^ {T} Y _ {s} Y _ {t} ^ {T} A _ {t}), s. t. d i a g (Z) = 0
$$

where $A _ { s }$ and $A _ { t }$ are two different transformation matrices that project the source domain data and target domain data, respectively, into a shared subspace. $A ^ { T } , \ A _ { s } ^ { T } ;$ , and $A _ { t } ^ { T }$ are the transpose matrices of $A , \ A _ { s } ,$ and $A _ { t } ,$ respectively. Specifically, $Z _ { s }$ and $Z _ { t }$ denote the sparse coding coefficient matrices $A _ { s } ^ { T } Y _ { s }$ and $A _ { t } ^ { T } Y _ { t } ,$ respectively. ?? ??(⋅) is the trace of the matrix. ????????(??) denotes the vector that contains the diagonal elements of the matrix $Z . ~ \lambda _ { 1 } > 0 , \beta > 0 ;$ , and $\lambda _ { 2 } > 0$ serve to equilibrate the various terms in the target function. Additionally, the first and third terms are designed to encourage sparsity in the source data representation while minimizing the associated errors in data reconstruction. Similarly, the second and fourth terms aim to achieve the same goals for the target data. The fifth term is a constraint to guarantee the proximity of data from different domains within the shared subspace. The SSCA algorithm adopts a shared-specific class selfexpressive dictionary, which leverages data from each class to train the respective SSCA. The SSCA algorithm ensures that projected data in the same class of the shared subspace exhibit a high correlation, in contrast to the low correlation observed in data from various classes. In addition, its goal is to minimize the distribution discrepancy in the common subspace of the source and target domains.

To capture the similarity between source domain and target domain data, subspace interpolation techniques are an effective way to connect the two domains. [88] (2013) utilized subspace interpolation through DL and sampled multiple intermediate domains along a virtual path between the source and target domains. Each intermediate domain is represented using a dictionary. By learning a set of intermediate domain dictionaries to gradually reduce the reconstruction residual of the target data. Building on subspace interpolation, [90] (2020) adopted a smooth subspace interpolation technique to bridge inter domain features and reduce discrepancies. In order to simultaneously address the differences in cross-view data and cross-domain data distributions. [78] (2018) projected the data from the source domain and the target domain into a shared subspace. In this subspace, a shared dictionary is learned, and domain-invariant features are extracted to reduce crossdomain differences. [89] (2021) employed the Joint Dictionary Matrix Factorization (JDMF) method to project source and target domain data into a common subspace and extract domain-invariant features, thus dampening cross-domain discrepancies.

An outstanding advantage of the UCDDL algorithms is that they do not require labeled training data, allowing for flexible application across various fields, especially when labels are challenging to acquire or expensive. The UCDDL algorithms focus on extracting bottom-level structures and patterns from data, adapting to various data distributions, and better capturing common features across different domains. Yet, by not utilizing label information, they might fail to accurately gain the characteristic representation of the target domain data. Furthermore, the UCDDL algorithms primarily concentrate on the intrinsic structure and patterns within the data and lack specificity for particular labels. The experimental results of unsupervised learning are often more challenging to interpret compared to those of supervised learning. The summary of the UCDDL algorithms is shown in Table 5.

Table 6 Averaged recognition accuracy (%) of the CDDL algorithms. 

<table><tr><td>Model type</td><td>Methods</td><td>Dataset used</td><td>Accuracy (%)</td></tr><tr><td>SCDDL</td><td>PDR [61]</td><td>Office and Caltech256 (CNN Feature)</td><td>94.29</td></tr><tr><td rowspan="5">SSCDDL</td><td>DTLC [86]</td><td>Office-31</td><td>73.93</td></tr><tr><td>LASDL [83]</td><td>Office and Caltech256 (CNN Feature)</td><td>94.33</td></tr><tr><td>SS-DDNMF [60]</td><td>RPaviaU-DPaviaC</td><td>93.30</td></tr><tr><td>TMMLDL [79]</td><td>JAFFE and CK+</td><td>79.21</td></tr><tr><td>TSSR [85]</td><td>JAFFE and CK+ (LBP Feature)</td><td>52.24</td></tr><tr><td rowspan="3">UCDDL</td><td>SSCA-SC [87]</td><td>Urban and Washington DC Mall Area 1</td><td>84.92</td></tr><tr><td>UDDL [88]</td><td>CMU-PIE</td><td>90.4</td></tr><tr><td>UJSDL [78]</td><td>PRID450s (WHOS Feature)</td><td>41.02</td></tr></table>

Note: The experimental results in the table are all sourced from the original literature of each method.

![](images/92942d2c4e1025353f9b8fd6dc6838531f6811a7979c698c9f6db312b1f925c3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Ye et al. [94"]] --> B["2017"]
    C["Zheng et al. [62"]] --> D["2018"]
    E["Yan et al. [95"]] --> F["2019"]
    G["Huang et al. [98"]] --> H["2021"]
    I["Tian et al. [96"]] --> J["2023"]
    K["Yan et al. [97"]] --> L["2017"]
    M["Zhang et al. [91"]] --> N["2019"]
    O["Lei et al. [92"]] --> P["2021"]
    Q["Cai et al. [93"]] --> R["2022"]
    S["Saffari et al. [99"]] --> T["2023"]
    U["Liu et al. [101"]] --> V["2024"]
    W["Kumar et al. [100"]] --> X["2024"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
    style S fill:#f9f,stroke:#333
    style U fill:#f9f,stroke:#333
    style W fill:#f9f,stroke:#333
```
</details>

Fig. 4. Temporal distribution of the DADL algorithms.

# 4. The domain-adaptive dictionary learning algorithms

The DADL algorithms leverage the techniques of DL to solve the adaptive transfer issues between the source and target domains. In the absence of tagged data in the target domain, the DADL algorithms are particularly effective because they play a crucial role in improving transmission performance. The DADL algorithms can extract and utilize features of source and target domains under limited tagged data. This helps to minimize differences between the domains and improves the reliability of the transfer process. According to the usage of class label information from domain data, the DADL algorithms can be divided into three categories: supervised, semi-supervised, and unsupervised DADL algorithms. The averaged recognition accuracy of the DADL algorithms is shown in Table 10.

The temporal distribution of the DADL algorithms is shown in Fig. 4. The timeline includes the development of DADL algorithms in recent years. Each point on the timeline represents the year an algorithm was published, along with the corresponding algorithms introduced in that year. As shown in the figure, a total of 12 DADL algorithms were proposed between 2017 and 2024. Notably, 2023 was the year with the most research activity, with four algorithms proposed during this period. To better describe the evolution process of different DADL algorithms, the development of the DADL algorithms is shown in Fig. 5. The development process consists of two main parts: foundational works and developmental works. In Fig. 5, ‘‘add’’ represents the incorporation of methods. For example, as indicated by the red arrow in the figure, [91] proposed a DADL algorithm that utilizes a synthetic dictionary, MMD, and Laplacian noise matrix.

# 4.1. The supervised domain-adaptive dictionary learning algorithms

The fundamental idea of the SDADL algorithms is to use labeled data to learn the mapping relationship between the source and target domains. The SDADL algorithms enhance DL by optimizing it to align more effectively with data distributions, thereby augmenting model performance across various domains. A key algorithm is the TDL With Mixed Noise (TDL-MN), introduced by [91]. The objective function of the TDL-MN algorithm is defined in Eq. (12):

$$
\min _ {D, Z, \Gamma} \| X - D Z - \Gamma \| _ {F} ^ {2} + T r (Z (\beta \Xi) Z ^ {T}) + \tag {12}
$$

$$
\alpha \parallel Z \parallel_ {1} + \eta \parallel \Gamma \parallel_ {1}, s. t. \parallel d _ {i} \parallel^ {2} \leq c, \forall i = 1, 2, \ldots , h
$$

where ?? is the Laplacian noise matrix. The MMD matrix is denoted as ?? and ???? is the transpose matrix of ??. The model is regularized by two parameters, ?? and ??. ?? controls the sparsity of ?? and ?? governs the Laplacian noise matrix ?? . Additionally, ?? serves as a tuning parameter for the MMD term. The first term denotes the Gaussian noise matrix, which represents the difference between the original signals and the reconstructed signals from the dictionary and coefficients. The second term revolves around minimizing the disparity in data distributions between the source and target domains. The TDL-MN algorithm leverages advanced deep-learning techniques. This innovative approach constructs a comprehensive mixed-noise dictionary, enabling the extraction of robust, domain-agnostic features. Then it harnesses the MMD method to effectively quantify the distribution distance across domains, thereby narrowing the gap between the source and target domains. The TDL-MN algorithm can extract information under complex working conditions and demonstrates stronger robustness in representing complex noise.

In scenarios where the target domain’s labels are unavailable, aiming to enhance the accuracy and robustness of Electroencephalogram (EEG) recognition, [92] (2021) introduced a TDL algorithm. The proposed algorithm utilizes source dictionary regularization within the Reproducing Kernel Hilbert Space (RKHS) subspace. Moreover, it utilizes the kernel methods to map input data into the RKHS, where domain-invariant features of the EEG signals from different domains are learned, thus generating a source dictionary. The learned source dictionary is then regularized to mitigate noise and redundancy interference, promoting domain adaptation transfer.

![](images/0232927c838575c93ad29c5f5a331cb5b6760a01bd8c27e573f942a12575a7ee.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Foundational Works"] -->|add| B["Developmental Works"]
    B --> C["Dictionary learning"]
    C --> D["Transfer learning"]
    D --> E["MMD"]
    E --> F["RKHS subspace learning [92"]]
    E --> G["Laplacian noise matrix [91"]]
    G --> H["Multiple kernel learning [62"]]
    G --> I["Multi-target domain dictionary [96"]]
    E --> J["Transfer matrix"]
    J --> K["Multi-task learning and matrix decomposition [94"]]
    J --> L["Class-specific dictionary and transitional dictionary [95"]]
    L --> M["Transfer matrix"]
    M --> N["Locality information preserving [93"]]
    M --> O["Regularization term [97"]]
    M --> P["CNN and sparse regularization [99"]]
    M --> Q["Category-aware and category contrastive learning [98"]]
    E --> R["Synthetic dictionary"]
    R --> S["Transfer matrix"]
    S --> T["Analysis-synthetic dictionary"]
    T --> U["Deep dictionary"]
    U --> V["Subspace interpolation [100"]]
    V --> W["CNN and sparse regularization [99"]]
    C --> X["Dictionary learning"]
    X --> Y["Transfer learning"]
```
</details>

Fig. 5. The development of the DADL algorithms.

Table 7 A summary of the SDADL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>TDL-MN [91]</td><td>1. Proposed a mixed noise model combining Gaussian and Laplacian distributions, which can better capture the complex noise characteristics in real-world scenarios.2. Employed MMD to reduce the distribution discrepancy between the source domain and the target domain, achieving effective domain adaptation.</td><td>1. The optimization process involves non-convex problems, resulting in high computational complexity.2. Further exploration of the transfer of composite fault features is needed.</td><td>Rolling bearing fault diagnosis</td></tr><tr><td>RKHS [92]</td><td>1. Using dual learning: RKHS learning and RKHS subspace learning, to make the marginal distributions of source domain data and target domain data closer.2. By distributing the target domain data around the source domain with the strongest linear correlation based on MMD, the spatial distribution difference between the source and target domain data of the same category is reduced.</td><td>1. The algorithm has high complexity, making it difficult to meet scenarios with high real-time requirements.2. The MMD criterion can affect model performance in specific cases. Therefore, selecting a better criterion is worth further exploration.</td><td>EEG mental recognition</td></tr></table>

The SDADL algorithms can fully leverage label information present within the target domain to learn dictionaries tailored to specific tasks. The learned transfer dictionaries facilitate the extraction of feature representations that are potentially more congruent with the nuances of the target domain. However, insufficient or low-quality label data can impair the algorithm’s generalization performance. Furthermore, the process of annotating data is inherently labor-intensive and resourceconsuming, contributing to the generally higher cost associated with the deployment of the SDADL algorithms. The summary of the SDADL algorithms is shown in Table 7.

# 4.2. The semi-supervised domain-adaptive dictionary learning algorithms

The SSDADL algorithms leverage both partially labeled and large sets of unlabeled data to train models, effectively adapting to the target domain’s distribution and enhancing the model’s generalization capabilities. A notable algorithm is the Hierarchical Domain-Adaptive Projection Dictionary Pair Learning (HDA-PDPL), put forth by Cai et al. [93], whose objective function is articulated in Eq. (13):

$$
\min _ {D _ {p}, D _ {s}, Z _ {t}, Q _ {t s}, Q _ {s t}} \sum_ {c = 1} ^ {C} \left(\| [ Y _ {s, c} ^ {(m)}, Y _ {t, c} ^ {(m)} ] - D _ {s c} ^ {(m)} [ Z _ {s, c} ^ {(m)}, Z _ {t, c} ^ {(m)} ] \| _ {F} ^ {2} + \right.
$$

$$
\alpha_ {1} \parallel D _ {p c} ^ {(m)} [ Y _ {s, c} ^ {(m)}, Y _ {t, c} ^ {(m)} ] - [ Z _ {s, c} ^ {(m)}, Z _ {t, c} ^ {(m)} ] \parallel_ {F} ^ {2} + \alpha_ {2} \parallel D _ {p c} ^ {(m)} \overline {{Y _ {c} ^ {(m)}}} \parallel_ {F} ^ {2}) \tag {13}
$$

$$
\begin{array}{l} + \beta_ {2} T r (D _ {p} ^ {(m)} Y ^ {(m)}) ^ {T} E ^ {(m)} (D _ {p} ^ {(m)} Y ^ {(m)}) + \beta_ {3} (\parallel Z _ {s} ^ {(m)} - P ^ {(m)} Y _ {t} ^ {(m)} Q _ {t s} ^ {(m)} \parallel_ {F} ^ {2} \\ + \parallel Z _ {t} ^ {(m)} - D _ {p} ^ {(m)} Y _ {s} ^ {(m)} Q _ {s t} ^ {(m)} \parallel_ {F} ^ {2}), s. t. \parallel d _ {j} ^ {(m)} \parallel \leq 1, \forall j = 1, 2, \dots , h \\ \end{array}
$$

Assuming a multi-layer hierarchical architecture, comprising a series of projection pair dictionaries with ?? layers. Where ?? is the layer number. ${ D } _ { p } ^ { ( m ) }$ is defined as the analysis dictionary at the ??th layer. ${ D } _ { p c } ^ { ( m ) }$ $\dot { D _ { s c } ^ { ( m ) } }$ ??(?????? represents the analysis dictionary in class ?? at the ??th layer, and (??) is the graph Laplacian matrix of the ??th layer. Additionally, using the representation coefficients target at the ??th layer, and th $Q _ { s t } ^ { ( m ) }$ from the source domresentation coefficients o the from $\underline { { Q } } _ { t s } ^ { ( m ) }$ ?????? the target domain to the source domain at the ??th layer. $Y _ { c } ^ { ( m ) }$ is the data from the complementary set of $Y _ { c } ^ { ( m ) } . ~ \alpha _ { 1 } , ~ \alpha _ { 2 } , ~ \beta _ { 2 } ,$ , and $\beta _ { 3 }$ ??represent regularization parameters. All items before $\beta _ { 2 }$ in HDA-PDPL represent the domain-adaptive term at the layer ??. The preservation of locality in the projective codes at layer ?? is adjusted by the regularization parameter $\beta _ { 2 }$ , while the regularization parameter $\beta _ { 3 }$ is used to adjust the term addressing the domain-adaptive sparse coding at the same layer.

Table 8   
A summary of the SSDADL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>HDA-PDPL [93]</td><td>1. Adopting a hierarchical framework can better learn the nonlinear characteristics of data.2. Using a local information preservation term, it can capture the discriminative local structure of signals, enhancing the robustness and smoothness of the model.</td><td>1. It involves constructing multi-level subspaces, learning dictionaries, and other processes, resulting in high computational complexity.2. Its application in large-scale datasets is worth exploring.</td><td>EEG classification</td></tr><tr><td>FL-DADL [94]</td><td>1. By performing domain-adaptive at the feature level, directly transforming and adapting the features, the model&#x27;s generalization ability is effectively improved.2. Combining DL and multi-task learning can simultaneously achieve domain adaptation and dimensionality reduction.</td><td>1. The computational complexity is high.2. It is sensitive to the initial dictionary.</td><td>Hyperspectral image classification</td></tr><tr><td>MK-DADP [62]</td><td>1. Using multi-kernel learning can better handle nonlinear data structures.2. By directly optimizing the final recognition task, the algorithm achieves better discriminative performance.</td><td>1. Repeating the calculation of sparse coding in the same manner during both training and testing stages, it may increase the computational complexity.2. Its not able to tackle the unsupervised cross-domain recognition tasks.</td><td>Cross-domain image recognition</td></tr></table>

To establish more robust connections between diverse domains, particularly for EEG classification, the HDA-PDPL algorithm was introduced. The HDA-PDPL algorithm divides EEG signals from both the source and target domains into stratified subspace. At each layer, a shared synthesis-analysis dictionary pair is learned to extract the domain-invariant feature representations through shared synthesis dictionaries. Subsequently, the nonlinear transformation functions are employed to map both the source and target domain data into a common feature space. By harnessing the shared knowledge across different domains, the HDA-PDDL algorithm enhances the accuracy and robustness of classification. Its hierarchical domain-adaptive and dictionary pair learning capabilities effectively tackle the imbalance inherent in EEG data, thereby bolstering the model’s adaptability and stability.

To address the challenge of transfer classification in cross-scene hyperspectral imaging, [94] (2017) introduced the Feature-Level DADL (FL-DADL) algorithm. The FL-DADL algorithm extracts shared spectral features through multitask joint dictionary learning, maps them to a domain-invariant space, and aligns spectral distributions to enhance adaptability. Additionally, the FL-DADL algorithm takes advantage of the limited labeled data from both the source and target scenes to train classifiers in this invariant space, effectively reducing domain disparities. [62] (2019) captured the complex structures of the labeled source domain data and a small set of the labeled target domain data using multi-kernel techniques. Coupled projections are applied to learn the connections between the dictionaries of the source and target domains, thereby reducing domain discrepancies.

The semi-supervised DL labeled target data to guide the learning process. This enhances the performance of the SSDADL algorithms when applied to new domains while reducing reliance on extensive labeled data. However, the efficacy of the SSDADL algorithms relies on the availability and selection of labeled data, which can significantly influence their performance. The summary of the SSDADL algorithms is shown in Table 8.

# 4.3. The unsupervised domain-adaptive dictionary learning algorithms

The UDADL algorithms are designed to learn one or more dictionaries from extensive unlabeled source domain data, aiming to create a shared feature space across domains. By harnessing the learned domain-invariant features, the UDADL algorithms can reducing the distributional shifts between various domains. An example of the UDADL algorithms is the DL-Based Unsupervised Multi Target Domain-Adaptive (DL-UMTDA) algorithm, as introduced by [95]. Its objective function is elaborated in Eq. 14:

$$
\begin{array}{l} \min_{\substack{P_{T}^{f},D^{\prime},\triangle D_{T}^{\prime f},Q_{T}^{f},Q_{S},u_{c,n}^{f}}} \sum_{f = 1}^{F}\big(\frac{1}{2} (\| P_{T}^{f}\|_{F}^{2} + \| \triangle D_{T}^{\prime f}\|_{F}^{2}) \\ + \frac {\lambda_ {1}}{2} \sum_ {c = 1} ^ {C _ {T} ^ {f}} \sum_ {n = 1} ^ {N _ {T} ^ {f}} (u _ {c, n} ^ {f}) ^ {2} \| \theta_ {c} ^ {f} - (P _ {T} ^ {f} X _ {T, n} ^ {f}) \| ^ {2} + \frac {\lambda_ {2}}{2} \| P _ {T} ^ {f} - (D ^ {\prime} \\ + \triangle D _ {T} ^ {' f}) Q _ {T} ^ {f} \| _ {F} ^ {2}) + \frac {\lambda_ {2}}{2} \| P _ {S} - D ^ {\prime} Q _ {S} \| _ {F} ^ {2} + \frac {\lambda_ {3}}{2} \| D ^ {\prime} \| _ {*} + \lambda_ {4} (\| Q _ {S} \| _ {2, 1} + \\ \sum_ {f = 1} ^ {F} \left\| Q _ {T} ^ {f} \right\| _ {2, 1}), s. t. \sum_ {c = 1} ^ {C _ {T} ^ {f}} u _ {c, n} ^ {f} = 1, f \in 1, 2, \dots , F, 0 \leq u _ {c, n} ^ {f} \leq 1 \tag {14} \\ \end{array}
$$

where $X _ { T , n } ^ { f }$ ?? ,?? refers to the ??th training sample in the $f \mathrm { t h }$ target domain. $P _ { T } ^ { f }$ represents its projection matrix for the ??th target domain. $u _ { c , n } ^ { f }$ denotes the clustering membership for the ??th instance to the ??th class in the ??th target domain. For a given target domain, $C _ { T } ^ { f }$ denotes its class number of the ?? th target domain, $N _ { T } ^ { f }$ is number of samples of the ??th target domain, and $\theta _ { c } ^ { f }$ refers to the one-hot coding label of the ??th target domain. Moreover, $\| \cdot \| _ { 2 , 1 }$ is the $\ell _ { 2 1 }$ norm, and $\| \cdot \| _ { * }$ represents the nuclear norm. The DL-UMTDA algorithm is regularized by four parameters, $\lambda _ { 1 } , \lambda _ { 2 } , \lambda _ { 3 }$ and $\lambda _ { 4 }$ . The first and second terms represent clustering and domain-adaptive. The third term represents the individual dictionary to capture individual knowledge of each target domain. The fourth term aims to capture the source domain knowledge,

Table 9   
A summary of the UDADL algorithms. 

<table><tr><td>Methods</td><td>Advantages</td><td>Limitations</td><td>Applications</td></tr><tr><td>CDSDA [96]</td><td>1. Using class-specific dictionaries improves domain adaptation performance.2. Introducing label information into DL the discriminative ability of the dictionaries and the differences in feature distributions.</td><td>1. The influence of spatial distribution information on domain- adaptive has not been considered.2. The computational complexity is high.</td><td>Land-cover classification of aerial images</td></tr><tr><td>DL-UMTDA [95]</td><td>1. By adopting multiple target domains and leveraging the correlations among them, better performance can be achieved.2. Learning individual dictionaries for each target domain helps to preserve the unique characteristics of each target domain.</td><td>1. The computational complexity is sensitive to the number of target domains.2. The lack of spatial local information leads to limitations in the model.</td><td>1. Multi-target domain-adaptive2. Cross-domain face age estimation</td></tr><tr><td>UDADL [97]</td><td>1. Unsupervised learning is adopted, which is robust to data distribution and noise, enabling it to adapt to various complex facial expression recognition scenarios.2. By introducing mutual embedding constraints on the sparse codes of the source domain and the target domain, the domain differences are reduced.</td><td>1. Introducing analysis dictionaries to relax the constraints increases the complexity of the model.2. It is no analytical solution to the optimization problem, and an iterative optimization strategy needs to be adopted to solve it.</td><td>Facial expression recognition</td></tr><tr><td>CaCo [98]</td><td>1. Assigning pseudo class labels to each target domain sample effectively enhances domain-adaptive performance.2. Using class contrastive techniques, it is possible to learn feature representations that are both highly discriminative for classes and invariant across domains.</td><td>1. The presence of inaccurate pseudo-label can impact its effectiveness when applied to the target domain.2. The presence of noise in source domain labels can potentially affect its performance.</td><td>1. Object detection2. Image classification</td></tr><tr><td>SADA [99]</td><td>1. Adopting an adversarial learning approach effectively mitigates domain shift and improves the model&#x27;s generalization ability in the target domain.2. By dynamically adjusting loss weights, domain alignment issues and discriminative loss can be mitigated, enhancing classification performance.</td><td>1. The smaller parameter space may lead to a decrease in its generalization ability.2. The performance improvement is not significant on some tasks.</td><td>Traffic scene classification</td></tr><tr><td>DDL-DA [100]</td><td>1. Adopting deep DL techniques to learn rich data representations from both the source and target domains can enable more effective learning of mappings.2. Using subspace interpolation techniques effectively reduces the discrepancy between domains.</td><td>1. Further increasing the number of dictionary layers does not improve the model performance.2. Its application in more complex scenarios is worth further exploration.</td><td>Machine inspection</td></tr><tr><td>UDA-CA [101]</td><td>1. Effectively addressing the issue of asymmetric class composition between the source and target domains.2. By selectively aligning the common classes between the source and target domains, the negative impact of extra classes can be overcome.</td><td>1. The convergence speed is affected by the differences in data distribution.2. The robustness is affected by the large differences in data distribution.</td><td>Asymmetric drift data for an electronic nose</td></tr></table>

Table 10   
Averaged recognition accuracy (%) of the DADL algorithms. 

<table><tr><td>Model type</td><td>Methods</td><td>Dataset used</td><td>Accuracy(%)</td></tr><tr><td>SDADL</td><td>RKHS [92]</td><td>BCI Competition IV 2a</td><td>67.08</td></tr><tr><td rowspan="3">SSDADL</td><td>HDA-PDPL [93]</td><td>Bonn</td><td>97.98</td></tr><tr><td>DA-FLDA [94]</td><td>Pavia</td><td>95.66</td></tr><tr><td>MK-DADP [62]</td><td>Office and Caltech256</td><td>73.42</td></tr><tr><td rowspan="6">UDADL</td><td>CDSDA [96]</td><td>NWPU-RESISC45, RCI-CB256 and Jinmen</td><td>90.41</td></tr><tr><td>UDADL [97]</td><td>BU-3DFE and CMU Multi-PIE</td><td>70.75</td></tr><tr><td>CaCo [98]</td><td>Office-31</td><td>87.60</td></tr><tr><td>SADA [99]</td><td>Honda Scenes Dataset</td><td>93.20</td></tr><tr><td>DDL-DA [100]</td><td>CWRU and Paderborn</td><td>95.40</td></tr><tr><td>UDA-CA [101]</td><td>Public Drift Dataset</td><td>84.69</td></tr></table>

Note: The experimental results in the table are all sourced from the original literature of each method.

which is to be transferred to the target domain. By employing knowledge distillation and dictionary bridging techniques, the DL-UMTDA algorithm facilitates the process of transferring knowledge from a single source domain to multiple target domains. The DL-UMTDA algorithm is highly practical because it caters to real-world scenarios where test data often originates from a diverse yet interconnected set of domains.

In tackling the issues of unsupervised cross-scene classification, [96] (2019) proposed a domain-adaptive model based on class-specific dictionaries. Initially, the labeled source domain data is used to learn dictionaries specific to each class. These learned dictionaries are then concatenated to form a comprehensive source dictionary. An iterative learning process develops a target dictionary from this, mapping both source and target features into a new feature space to obtain the domain-invariant feature representations. To address the challenge of the unsupervised cross-domain facial expression recognition, [97] (2018) proposed a DADL model. The model connects the source and target domains by constructing a common dictionary, mapping labeled source and unlabeled target data into a sparse high-dimensional space for domain-invariant feature extraction. To better adapt to domain distributions, [102] (2021) introduced a part-aware strategy that decomposed images into multiple segments and conducted feature extraction and alignment for each segment. [98] (2022) utilized data from both the source and target domains to create a hybrid domain. Subsequently, a class-aware and domain-mixed dictionary is constructed by using a subspace-based DL method. By optimizing the category contrastive loss function, samples of the same category are drawn closer in feature space, while those of differing categories are distanced further. Recently, with the widespread application of deep learning algorithms, researchers have utilized deep neural networks to extract domain features. By employing DL for features in both the source and target domains, thereby enhancing the algorithms’ transfer ability [99, 100]. To diminish the disparity between the source and target domains, [99] (2023) incorporated adversarial loss functions, while [100] (2023) applied subspace interpolation techniques, among others. These methods aim to enhance the generalizability of feature representations, making the algorithm better suited to the target domain. For domainadaptive transfer issues involving asymmetric drift data, [101] (2023) proposed a novel domain-adaptive algorithm that integrated DL, CCA, and LPP. The proposed algorithm utilizes source domain data for DL to obtain a source domain dictionary. Then, utilizing the learned source domain dictionary, it maps data from both source and target domains into a feature space, where CCA analysis and feature alignment are conducted. Finally, LPP is applied in the aligned feature space to achieve a low-dimensional representation that preserves the local structure of the data, thus narrowing the differences between the source and target domains.

The UDADL algorithms without the necessity of labeling data from the target domain, providing them a significant advantage in scenarios where labels are scarce or expensive to obtain. The UDADL algorithms are adaptable to various domains and tasks, especially in cases with notable disparities between the source and target domains. Additionally, the UDADL algorithms can uncover hidden patterns and structures within the data, assisting in capturing common features between the source and target domains during the domain-adaptive process. However, they exhibit heightened sensitivity to data distribution, outliers, and noise. The summary of the UDADL algorithms is shown in Table 9.

# 5. Experiments

To demonstrate the performance of the TDL algorithms, we compared some typical CDDL and DADL algorithms with typical DL algorithms, STL algorithms, and DTL algorithms on the CWRU database [103], the Paderborn database [104], the CMU-PIE database [105], the Office database [106] and the Caltech-256 database [107].

Specifically, LC-KSVD [74] and Locality Constrained and Label Embedding (LCLE-DL) [73] are typical DL algorithms. Transform Learning for Unsupervised Domain-Adaptive (TL-UDA) [104], Low-Rank Transfer Subspace Learning (LTSL) [80], and Geodesic Flow Kernel (GFK) [108] all belong to the STL algorithms. The domain-adaptive deep learning algorithm Multi-Kernel Maximum Mean Discrepancy (MK-MMD) [109] is the DTL algorithm. The PDR [61] and LASDL [83] algorithms are both the CDDL algorithms. The DL for Domain-Adaptive (DL-DA) [100], Deep DL for Domain-Adaptive (DDL-DA) [100], and Multi-Kernel Domain-Adaptive Based Discriminative Projections (MK-DADP) [62] are both the DADL algorithms. The speed and robustness analysis of these methods is shown in Table 17.

# 5.1. Dataset introduction

This subsection provides a detailed description of the five datasets.

(1) The Office dataset serves as a prominent benchmark for evaluating visual transfer learning approaches, comprising 31 categories of common office objects, such as laptops and keyboards. The Office dataset includes 3 domains: Amazon (A), Dslr (D), and Webcam (W), each containing 31 categories.

(2) The Caltech-256 dataset, originating from the California Institute of Technology, is selected from the Google Images dataset and manually cleaned of images not fitting their categories. With Caltech-256 (C) serving as a domain, it encompasses 30,607 images across 256 categories, with each category containing at least 80 images.

(3) The CMU-PIE dataset is developed by Carnegie Mellon University. ‘‘PIE’’ stands for pose, illumination, and expression. It includes 41,368 photographs of faces with various poses, lighting conditions, and expressions, taken from 68 individuals.

(4) The Paderborn dataset includes stator current and vibration data recorded at 64 kHz from a test bench comprising a load motor, a test module, a drive motor, and a torque-measuring shaft. Two loading torques (0.7 and 0.1 Nm) and two rotational speeds (900 and 1500 rpm) are recorded for both good and faulty bearings (natural and manmade flaws). Inner race faults (IF) and outer race faults (OF) are two types of bearing failures.

(5) The CWRU dataset comes from Case Western Reserve University (CWRU) and contains vibration signals gathered from a machine’s drive and fan ends. Four distinct load torques (0, 1, 2, and 3 HP), corresponding to 1797, 1772, 1750, and 1730 rpm respectively, are used to collect data at a sampling rate of 12 kHz. Both healthy and defective bearing data are included in the collection. With the use of electric discharge machining, bearing flaws of three distinct sizes (0.007, 0.014, and 0.021 inches) are created, including IF, OF, and ball faults (BF).

# 5.2. Experimental results on the office and caltech-256 datasets

For the problem of cross-domain recognition, a combination of the Office and Caltech-256 datasets is used to compare five algorithms: GFK, MK-DADP, MK-DADP (Hierarchical), PDR, and LASDL. This dataset includes four domains: A, D, W, and C. To create datasets, two different domains are chosen at random, for example, C→A (source domain→target domain), $\mathrm { C \to D , \ \dots , }$ and D→W. Furthermore, two different types of features are used in the compared experiments. One is the 800-bin SURF features, which were introduced by Gong et al. [107]. The other consists of deep features that were extracted using a CNN architecture consisting of three fully connected layers and five convolutional layers from ImageNet [110]. The experimental results are presented in Tables 11, 12, and 13.

As shown in Table 11, the transfer learning algorithm (GFK) is used as a baseline algorithm, with the best recognition accuracy of approximately 80.5%. The optimal recognition accuracy of the TDL algorithm (LC-KSVD) is approximately the same as that of GFK. The TDL algorithm (MK DADP) improves it by about 0.7%. Moreover, the MK DADP (Hierarchical) achieves the best recognition accuracy, which is approximately 5% higher than that of GFK. The experimental results indicate that the TDL algorithms are more performance than those solely relying on transfer learning algorithms. In addition, the MK-DADP (Hierarchical) algorithm employs a hierarchical structure and achieves better performance than the MK-DADP algorithm. This demonstrates that using hierarchically structured dictionaries can more effectively capture transferable features for cross-domain and domain-adaptive.

Table 11 Recognition accuracy (%) for single domain. 

<table><tr><td>Method</td><td>GFK</td><td>LASDL</td><td>MK-DADP</td><td>MK-DADP(Hierarchical)</td></tr><tr><td>C → A</td><td>44.7 ± 3.3</td><td>55.1 ± 2.1</td><td>63.9 ± 1.6</td><td>76.7 ± 1.6</td></tr><tr><td>C → D</td><td>61.0 ± 3.9</td><td>67.7 ± 4.4</td><td>81.1 ± 2.3</td><td>86.6 ± 2.9</td></tr><tr><td>A → C</td><td>37.8 ± 1.6</td><td>40.7 ± 2.1</td><td>45.5 ± 2.3</td><td>60.2 ± 2.1</td></tr><tr><td>A → W</td><td>64.8 ± 3.6</td><td>72.2 ± 4.4</td><td>77.2 ± 3.7</td><td>80.4 ± 3.7</td></tr><tr><td>W → C</td><td>34.4 ± 1.9</td><td>40.4 ± 1.6</td><td>47.0 ± 1.9</td><td>52.8 ± 1.8</td></tr><tr><td>W → A</td><td>46.6 ± 2.8</td><td>53.7 ± 1.8</td><td>65.4 ± 2.2</td><td>77.2 ± 2.5</td></tr><tr><td>D → A</td><td>46.2 ± 2.3</td><td>52.6 ± 2.1</td><td>57.3 ± 2.1</td><td>71.9 ± 2.0</td></tr><tr><td>D → W</td><td>80.5 ± 2.1</td><td>80.5 ± 4.1</td><td>81.2 ± 2.5</td><td>81.6 ± 2.3</td></tr><tr><td>Average</td><td>52.0 ± 2.6</td><td>57.8 ± 2.8</td><td>64.8 ± 2.3</td><td>73.4 ± 2.3</td></tr><tr><td>Gain</td><td>--</td><td>+5.8</td><td>+12.8</td><td>+21.4</td></tr></table>

Note: The data in this table is sourced from the literature [62]. The highest classification accuracy is in bold in Tables 11–16. ‘‘- -’’ represents the baseline method. The ‘‘Gain’’ values mean the enhancement in the average classification accuracy of the method compared to the baseline method’s average classification accuracy. For example, Gain (LASDL) = Average (LASDL) - Average (GFK).

Table 12 Recognition accuracy (%) of different methods on the office and Caltech256 datasets with SURF features. 

<table><tr><td>Method</td><td>GFK</td><td>PDR</td><td>LASDL</td></tr><tr><td>A → D</td><td>57.99 ± 6.35</td><td>58.29 ± 3.49</td><td>66.25 ± 3.08</td></tr><tr><td>A → C</td><td>37.84 ± 1.63</td><td>44.11 ± 1.72</td><td>40.66 ± 2.05</td></tr><tr><td>A → W</td><td>64.84 ± 3.59</td><td>63.12 ± 2.34</td><td>72.16 ± 4.39</td></tr><tr><td>D → A</td><td>46.22 ± 2.32</td><td>51.41 ± 1.50</td><td>52.62 ± 2.05</td></tr><tr><td>D → C</td><td>35.78 ± 1.56</td><td>39.10 ± 1.78</td><td>39.93 ± 1.80</td></tr><tr><td>D → W</td><td>80.49 ± 2.14</td><td>82.02 ± 1.36</td><td>80.49+4.05</td></tr><tr><td>C → A</td><td>44.73 ± 3.29</td><td>54.29+2.06</td><td>55.10 ± 2.14</td></tr><tr><td>C → D</td><td>61.04 ± 3.93</td><td>65.33 ± 3.47</td><td>67.73 ± 4.42</td></tr><tr><td>C → W</td><td>66.51 ± 4.49</td><td>69.04 ± 2.70</td><td>74.56 ± 3.56</td></tr><tr><td>W → A</td><td>46.56 ± 2.84</td><td>50.75 ± 2.14</td><td>53.70 ± 1.78</td></tr><tr><td>W → D</td><td>73.83 ± 4.26</td><td>77.01 ± 3.88</td><td>72.47 ± 4.18</td></tr><tr><td>W → C</td><td>34.43 ± 1.89</td><td>38.87 ± 1.75</td><td>40.35 ± 1.61</td></tr><tr><td>Average</td><td>54.19 ± 3.19</td><td>57.78 ± 2.38</td><td>59.67 ± 2.93</td></tr><tr><td>Gain</td><td>- -</td><td>+3.59</td><td>+5.48</td></tr></table>

Note: The data in this table is sourced from the literature [83]. ‘‘- -’’ represents the baseline method. The ‘‘Gain’’ values mean the enhancement in the average classification accuracy of the method compared to the baseline method’s average classification accuracy. For example, Gain (LASDL) = Average (LASDL) - Average (GFK).

Table 13 Recognition accuracy (%) of different methods on the Office and Caltech256 datasets with DeCAF7 features. 

<table><tr><td>Methods</td><td>GFK</td><td>PDR</td><td>LASDL</td></tr><tr><td>A → D</td><td>95.39 ± 2.43</td><td>98.66 ± 1.21</td><td>98.38 ± 0.86</td></tr><tr><td>A → C</td><td>81.19 ± 1.60</td><td>87.71 ± 0.86</td><td>87.80 ± 1.56</td></tr><tr><td>A → W</td><td>93.90 ± 2.25</td><td>97.23 ± 1.10</td><td>97.60 ± 1.67</td></tr><tr><td>D → A</td><td>89.73 ± 1.00</td><td>93.96 ± 0.58</td><td>93.23 ± 0.88</td></tr><tr><td>D → C</td><td>80.76 ± 1.29</td><td>87.79 ± 0.92</td><td>87.23 ± 1.10</td></tr><tr><td>D → W</td><td>97.79 ± 1.36</td><td>99.10 ± 0.83</td><td>99.48 ± 0.98</td></tr><tr><td>C → A</td><td>90.29 ± 1.01</td><td>92.84 ± 1.16</td><td>92.72 ± 1.22</td></tr><tr><td>C → D</td><td>96.23 ± 2.34</td><td>97.23 ± 1.57</td><td>98.21 ± 1.43</td></tr><tr><td>C → W</td><td>94.67 ± 1.35</td><td>97.51 ± 1.68</td><td>97.60 ± 1.85</td></tr><tr><td>W → A</td><td>88.75 ± 1.63</td><td>92.94 ± 0.88</td><td>93.00 ± 1.32</td></tr><tr><td>W → D</td><td>98.18 ± 1.86</td><td>99.29 ± 0.87</td><td>99.46 ± 0.87</td></tr><tr><td>W → C</td><td>80.46 ± 1.58</td><td>87.26 ± 0.72</td><td>87.20 ± 0.90</td></tr><tr><td>Average</td><td>90.61 ± 1.64</td><td>94.29 ± 1.03</td><td>94.33 ± 1.22</td></tr><tr><td>Gain</td><td>- -</td><td>+3.68</td><td>+3.72</td></tr></table>

Note: The data in this table is sourced from the literature [83]. ‘‘- -’’ represents the baseline method. The ‘‘Gain’’ values mean the enhancement in the average classification accuracy of the method compared to the baseline method’s average classification accuracy. For example, Gain (LASDL) = Average (LASDL) - Average (GFK).

Table 12 displays the experimental results of the GFK, PDR, and LASDL algorithms on the SURF features of the Office and Caltech256 datasets. In Table 12, the transfer learning algorithm (GFK) is used as a baseline algorithm, with average recognition accuracy of approximately 54%. The average recognition accuracy of the TDL algorithm (PDR) is approximately 3% higher than that of GFK. The other TDL algorithm (LASDL) achieves the best recognition accuracy, which is approximately 5% higher than that of GFK. As GFK is an unsupervised domain-adaptive algorithm, it cannot utilize label information from the target domain, resulting in the lowest recognition accuracy. The experimental results not only demonstrate the effectiveness of combining dual reconstruction and structured learning strategies, but also show that the TDL algorithms achieve better recognition accuracy than the transfer learning algorithms.

Table 13 presents the experimental results of the GFK, PDR, and LASDL algorithms on the DeCAF7 features of the Caltech256 and Office datasets. According to Table 13, all of them achieve average recognition accuracy of over 90%. The transfer learning algorithm (GFK) is used as a baseline algorithm, with average recognition accuracy of approximately 90%. The average recognition accuracy of the TDL algorithm (PDR) is approximately 3.6% higher than that of GFK. The other TDL algorithm (LASDL) achieves the best recognition accuracy, which is approximately 3.7% higher than that of GFK.

In all, Tables 11, 12, and 13 demonstrate that the TDL algorithms (PDR and LASDL) achieve better performance than the transfer learning algorithm (GFK). Thus, incorporating DL into the transfer learning algorithms can significantly enhance the recognition accuracy of the transfer learning algorithms to a certain extent.

# 5.3. Experimental results on the CMU-PIE dataset

This section discusses the comparison and analysis of the five algorithms (LC-KSVD, LTSL, LCLE-DL, PDR, and LASDL) addressing the cross-domain recognition problem on the CMU-PIE dataset. In the experiments, the subsets of Pose05 (P1), Pose07 (P2), Pose09 (P3), Pose27 (P4), and Pose29 (P5) are selected to create cross-domain recognition datasets, resulting in 20 dataset combinations, such as P1 with P2, P1 with P3, . . . , P5 with P4. For each dataset, five samples per class are randomly chosen to form the source and target domains. The results of the experimental findings are presented in Table 14.

Table 14 shows the experimental results of the LC-KSVD, LTSL, LCLE-DL, PDR, and LASDL algorithms on the CMU-PIE dataset. In Table 14, the transfer learning algorithm (LTSL) is used as a baseline algorithm, with average recognition accuracy of approximately 78%. The average recognition accuracy of LCLE-DL is 11.45% higher than that of LTSL. The average recognition accuracy of the DL algorithm (LC-KSVD) is 12.93% higher than that of LTSL, and the TDL algorithm (LASDL) improves it by about 15.02%. The other TDL algorithm (PDR) achieves the best classification accuracy, which is approximately 16% higher than that of LTSL. It can be seen that the average recognition accuracy of algorithms based on DL all exceed 90% and demonstrate relative stability. This suggests that, on the CMU-PIE dataset, the TDL algorithms show higher recognition accuracy relative to both the transfer learning algorithm and the DL algorithm. Moreover, the SCDDL algorithm (PDR), given its supervised nature, can effectively utilize label information, resulting in enhanced outcomes.

# 5.4. Experimental results on the CWRU and paderborn datasets

On the CWRU and Paderborn datasets, four unsupervised domainadaptive algorithms (MK-MMD, TL-UDA, DL-UDA, and DDL-DA) are selected to address the domain-adaptive issue in machinery fault diagnosis under varying machine operational settings. The recognition accuracy of different algorithms is evaluated by using Precision (P), Recall (R), F1 score (F1), and Accuracy (Acc). The experimental findings are presented in Tables 15 and 16.

Table 15 presents the mean recognition accuracy derived from five randomly selected training sessions conducted in the source domain (CWRU) to the target domain (Paderborn). In Table 15, the domainadaptive algorithms based on deep learning (MK-MMD) is used as a baseline algorithm, with recognition accuracy of approximately 36%. The recognition accuracy of the DADL methods (DL-UDA) is approximately 48% higher than that of MK-MMD. And the domain-adaptive algorithms based on transfer learning (TL-UDA) improves it by about 56%. The DDL-DA (2-lever (10–10)) and DDL-DA (3-lever (10-10-10)) algorithms are increased by 57% and 59% respectively compared to MK-MMD.

Table 14 Recognition accuracy (%) of different methods on the CMU-PIE dataset. 

<table><tr><td>Methods</td><td>LTSL</td><td>LCLE-DL</td><td>LC-KSVD</td><td>LASDL</td><td>PDR</td></tr><tr><td>P1→P2</td><td>84.51 ± 2.45</td><td>90.77 ± 0.68</td><td>90.78 ± 1.02</td><td>93.27 ± 2.72</td><td>94.20 ± 0.59</td></tr><tr><td>P1→P3</td><td>82.33 ± 2.57</td><td>91.18 ± 0.56</td><td>91.41 ± 0.91</td><td>93.62 ± 2.61</td><td>93.40 ± 0.53</td></tr><tr><td>P1→P4</td><td>85.64 ± 2.39</td><td>91.15 ± 0.79</td><td>92.35 ± 1.21</td><td>95.12 ± 1.95</td><td>96.56 ± 0.45</td></tr><tr><td>P1→P5</td><td>73.46 ± 2.90</td><td>87.69 ± 0.67</td><td>90.69 ± 0.85</td><td>92.03 ± 2.54</td><td>92.18 ± 0.69</td></tr><tr><td>P2→P1</td><td>82.20 ± 2.62</td><td>91.98 ± 0.56</td><td>92.89 ± 0.78</td><td>95.80 ± 1.57</td><td>96.76 ± 0.55</td></tr><tr><td>P2→P3</td><td>81.18 ± 2.44</td><td>91.79 ± 0.83</td><td>90.39 ± 0.89</td><td>92.34 ± 3.46</td><td>94.02 ± 0.56</td></tr><tr><td>P2→P4</td><td>86.41 ± 2.64</td><td>92.94 ± 0.69</td><td>93.18 ± 0.76</td><td>95.49 ± 1.11</td><td>95.61 ± 0.59</td></tr><tr><td>P2→P5</td><td>72.69 ± 2.76</td><td>88.40 ± 0.63</td><td>87.97 ± 0.81</td><td>90.32 ± 3.04</td><td>92.34 ± 0.66</td></tr><tr><td>P3→P1</td><td>74.38 ± 2.52</td><td>92.28 ± 0.54</td><td>92.95 ± 0.65</td><td>96.11 ± 1.45</td><td>96.45 ± 0.71</td></tr><tr><td>P3→P2</td><td>77.48 ± 2.18</td><td>90.16 ± 0.62</td><td>90.93 ± 0.81</td><td>92.15 ± 2.12</td><td>93.17 ± 0.75</td></tr><tr><td>P3→P4</td><td>80.63 ± 2.22</td><td>90.93 ± 0.81</td><td>92.67 ± 0.59</td><td>95.06 ± 1.55</td><td>96.56 ± 0.48</td></tr><tr><td>P3→P5</td><td>66.47 ± 3.31</td><td>87.93 ± 0.72</td><td>91.23 ± 0.97</td><td>92.52 ± 2.90</td><td>92.70 ± 0.60</td></tr><tr><td>P4→P1</td><td>86.18 ± 2.76</td><td>90.25 ± 0.49</td><td>93.01 ± 0.79</td><td>95.98 ± 1.32</td><td>96.92 ± 0.30</td></tr><tr><td>P4→P2</td><td>82.12 ± 2.34</td><td>88.19 ± 0.78</td><td>91.87 ± 0.93</td><td>93.61 ± 2.20</td><td>94.25 ± 0.47</td></tr><tr><td>P4→P3</td><td>80.60 ± 2.23</td><td>90.92 ± 0.80</td><td>92.58 ± 0.86</td><td>92.19 ± 2.74</td><td>95.42 ± 0.96</td></tr><tr><td>P4→P5</td><td>79.82 ± 2.22</td><td>86.85 ± 0.78</td><td>88.62 ± 0.88</td><td>91.21 ± 3.37</td><td>93.52 ± 0.53</td></tr><tr><td>P5→P1</td><td>72.51 ± 2.62</td><td>91.91 ± 0.61</td><td>91.98 ± 0.67</td><td>95.69 ± 1.53</td><td>96.45 ± 0.58</td></tr><tr><td>P5→P2</td><td>74.86 ± 2.64</td><td>88.94 ± 0.45</td><td>92.02 ± 0.45</td><td>93.23 ± 2.14</td><td>93.95 ± 0.63</td></tr><tr><td>P5→P3</td><td>72.14 ± 2.90</td><td>89.01 ± 0.57</td><td>91.80 ± 0.76</td><td>93.01 ± 3.56</td><td>94.25 ± 0.62</td></tr><tr><td>P5→P4</td><td>77.75 ± 2.06</td><td>89.04 ± 0.65</td><td>92.75 ± 0.68</td><td>95.10 ± 1.29</td><td>95.65 ± 0.44</td></tr><tr><td>Average</td><td>78.67 ± 2.54</td><td>90.12 ± 0.66</td><td>91.60 ± 0.81</td><td>93.69 ± 2.26</td><td>94.72 ± 0.58</td></tr><tr><td>Gain</td><td>- -</td><td>+11.45</td><td>+12.93</td><td>+15.02</td><td>+16.05</td></tr></table>

Note: The data in this table is sourced from the literature [61,83]. ‘‘- -’’ represents the baseline method. The ‘‘Gain’’ values mean the enhancement in the average classification accuracy of the method compared to the baseline method’s average classification accuracy. For example, Gain (LASDL) = Average (LASDL) - Average (LTSL).

Table 15 Recognition accuracy (%) for CWRU → Paderborn. 

<table><tr><td>Method</td><td>P</td><td>R</td><td>F1</td><td>Acc</td><td>Acc gain</td></tr><tr><td>MK-MMD</td><td>35.57</td><td>36.25</td><td>33.42</td><td>36.25</td><td>- -</td></tr><tr><td>DL-UDA(1-lever(10))</td><td>82.28</td><td>84.25</td><td>78.95</td><td>84.19</td><td>+47.94</td></tr><tr><td>TL-UDA</td><td>92.73</td><td>92.31</td><td>92.19</td><td>92.23</td><td>+55.98</td></tr><tr><td>DDL-DA(2-lever(10-10))</td><td>94.48</td><td>93.38</td><td>93.19</td><td>93.39</td><td>+57.14</td></tr><tr><td>DDL-DA(3-lever(10-10-10))</td><td>95.58</td><td>95.35</td><td>95.32</td><td>95.40</td><td>+59.15</td></tr></table>

Note: The data in this table is sourced from the literature [100,104]. ‘‘- -’’ represents the baseline method in Tables 15–16. The ‘‘Gain’’ values mean the enhancement in the Acc of the method compared to the baseline method’s Acc in Tables 15–16. For example, Acc gain (TL-UDA) = Acc (TL-UDA) - Acc (MK-MMD).

Table 16 Recognition accuracy (%) for Paderborn → CWRU. 

<table><tr><td>Method</td><td>P</td><td>R</td><td>F1</td><td>Acc</td><td>Acc gain</td></tr><tr><td>MK-MMD</td><td>88.22</td><td>82.73</td><td>81.74</td><td>82.73</td><td>- -</td></tr><tr><td>DL-UDA(1-lever(10))</td><td>89.77</td><td>84.63</td><td>83.49</td><td>86.34</td><td>+3.61</td></tr><tr><td>TL-UDA</td><td>93.16</td><td>90.48</td><td>90.12</td><td>90.51</td><td>+7.78</td></tr><tr><td>DDL-DA(2-lever(10-10))</td><td>93.11</td><td>91.17</td><td>90.71</td><td>91.02</td><td>+8.29</td></tr><tr><td>DDL-DA(3-lever(10-10-10))</td><td>95.48</td><td>93.91</td><td>93.51</td><td>93.83</td><td>+11.1</td></tr></table>

Note: The data in this table is sourced from the literature [100,104].

Table 16 displays the average recognition accuracy from five randomly chosen training sessions carried out in the source domain (Paderborn) to the target domain (CWRU). Similarly, MK-MMD is used as a baseline algorithm, with recognition accuracy of approximately 82%. DL-UDA is approximately 4% higher than that of MK-MMD and TL-UDA improves it by about 8%. The DDL-DA (2-lever(10–10)) and DDL-DA (3- lever(10-10-10)) algorithms are increased by 9% and 11% respectively compared to MK-MMD.

In summary, Tables 15 and 16 suggest that the deep representations can learn the mapping between two domains more effectively than the shallow representations. Additionally, the domain-adaptive algorithms based on transfer learning (TL-UDA) also achieve excellent classification results. In contrast, the domain-adaptive algorithm based on deep learning (MK-MMD) tend to perform less well in comparison to the other algorithms. Overall, the TDL algorithms achieve excellent domain-adaptive performance, highlighting their potential and efficacy in the field of transfer learning.

# 6. Future directions

In this section, five future directions related to TDL are presented.

(1) Deep TDL model based on small-shot learning.

Research Gap: Deep DL has yielded significant results in signal and image processing [43,111]. In the realm of transfer learning, [100] made an initial attempt to introduce deep DL into transfer learning, achieving good performance in machinery fault detection. However, current deep TDL models often rely on large amounts of labeled data, limiting their applicability in data-scare scenarios. There is a lack of methods that integrate small-shot learning mechanisms to overcome these data limitations.

Potential Approaches: Future research could explore combining small-shot learning strategies, such as meta-learning or episodic training, with deep TDL. These techniques can enable the model to generalize effectively from a few labeled examples, thereby improving scalability in domains with limited data availability.

(2) Data augmentation for domain generalization DL.

Research Gap: DL has seen progress in cross-domain and domainadaptive learning [62,90], but DL algorithms have not been applied to domain generalization models yet. Furthermore, DL has also made certain progress in the field of data enhancement [112]. Data augmentation techniques, which are crucial methods for addressing domain generalization, have not been applied to domain generalization DL models. This creates a gap in leveraging augmented datasets to enrich feature representations and improve model robustness.

Potential Approaches: By performing DL on an augmented dataset, one can learn richer and more robust feature representation. For example, manifold-based data augmentation approaches, combined with DL, could be used to synthesize domain-agnostic representations and enhance training diversity. This would improve the model’s generalizability across unseen domains.

(3) Multi-source domain transfer DL for intrusion detection.

Research Gap: Transfer learning algorithms have achieved competitive results in intrusion detection [113,114], but multi-source domain TDL has not been explored in the intrusion detection field. A major challenge is the efficient integration of heterogeneous domain data while addressing domain discrepancies, which are crucial for detecting unknown attacks.

Table 17 Speed and robustness analysis of different methods. 

<table><tr><td>Type</td><td>Methods</td><td>Speed analysis</td><td>Robustness analysis</td></tr><tr><td rowspan="3">STL</td><td>GFK [108]</td><td>High computational complexity.1. Using the least squares and alternating dictionary method of multipliers, if effectively reduces the computational complexity of traditional dictionary learning methods based on spare representation.2. Not using cross-validation parameters reduces computational overhead to a certain extent.</td><td>1. Integrating multiple subspace strategies, enhances domain adaptability and improves the stability of GFK.2. Adaptively subspace selection reduces reliance on subjective choices, thereby enhancing the robustness of GFK.</td></tr><tr><td>TL-UDA [104]</td><td>The computational complexity is lower than dictionary learning.Using the transfer learning framework enables more efficient generation of feature representations.</td><td>1. The algorithm performs well in data-scare scenarios.2. It effectively addresses inter-domain discrepancies, demonstrating excellent domain adaptability.</td></tr><tr><td>LTSL [80]</td><td>Low computational complexity.1. Adopting low-rank constraint to bridge the source domain and the target domain.2. Subspace learning is conducted in a low-dimensional space.</td><td>1. By processing target data that cannot be reconstructed from source data using sparse matrices, the ability to handle outliers is enhanced.2. It performs well in face recognition tasks under varying lighting and expression changes, demonstrating good robustness.</td></tr><tr><td>DTL</td><td>MK-MMD [109]</td><td>The computational complexity is higher compared to STL.1. Its linear complexity in feature learning and domain-adaptive makes it suitable for handling large-scale datasets and reducing computation time.2. Using deep neural networks.</td><td>1. Its independent of the number of target labels and performs well in both unsupervised and semi-supervised domain-adaptive.2. By using MK-MMD as the domain discrepancy measure, it more effectively bridges the domain gap and improves robustness compared to single-kernel MMD.</td></tr><tr><td rowspan="2">DL</td><td>LCLE-DL [73]</td><td>Low computational complexity.1. Adopting a fast iterative strategy enhances algorithm efficiency.2. By extracting intrinsic information through local constraints on atoms, which reduces computational load and accelerating running speed.</td><td>1. By combining local information and label embedding, it better handles noise and outliers, increasing its robustness.2. Through a multi-layer dictionary learning framework, systematically removes redundant and interfering features, further enhancing the model&#x27;s stability and robustness.</td></tr><tr><td>LC-KSVD [74]</td><td>Low computational complexity.1. Using the simple linear predictive classifier during the testing phase, resulting in faster classification speeds.2. Adopting the K-SVD algorithm for optimization is slower than methods that directly utilize training samples.</td><td>1. It maintains high classification accuracy in datasets with multiple categories and large variations, demonstrating good robustness.2. By optimizing the reconstruction and classification error in the objective function, it effectively resists noise and outlier interference.</td></tr><tr><td rowspan="4">TDL</td><td>PDR [61]</td><td>Low computational complexity.1. Employing an efficient optimization method accelerates training speeds.2. The learned dictionary contains fewer atoms, which enhances classification speeds. 3. Its suitable for large-scale datasets.</td><td>1. Using flexible label consistency enhances the algorithm&#x27;s robustness against inconsistent label distributions.2. Low-rank constraints and sparse matrices help filter out noise, further improving robustness against noise and outliers.</td></tr><tr><td>MK-DADP [62]</td><td>High computational complexity.1. Utilizing multi-kernel technology leads to slower computation speeds with large-scale datasets.2. Recomputing the kernel matrix and updating the projection matrix in each iteration is time-consuming.</td><td>1. The algorithm maintains good performance with a limited number of labeled samples.2. It demonstrates good adaptability to data from multiple domains and can be extended to multi-domain problems.</td></tr><tr><td>LASDL [83]</td><td>Low computational complexity.1. Using the least squares and alternating direction method of multipliers, it effectively reduces the computational complexity of traditional dictionary learning methods based on sparse representation. 2. When dealing with large-scale data, it exhibits faster training and testing speeds.</td><td>1. By applying graph regularization constraints enhance the model&#x27;s robustness against noise and outliers.2. Using projection-based dual reconstruction to train class-specific sub-dictionaries effectively reduces inter-domain differences.</td></tr><tr><td>DDL-DA [100]</td><td>High computational complexity.1. For small-scale datasets, the algorithm&#x27;s computation time is reasonable.2. For large-scale datasets, the computation speed is slower. But the multi-layer structure of deep dictionary learning can improve the computation speed to a certain extent.</td><td>1. Using deep dictionary learning and subspace interpolation provides good robustness in cases of missing data.2. Through deep dictionary learning minimizes noise interference and extracts more robust feature representations.</td></tr></table>

Proposed Improvement: Multi-source domain TDL can extract more generalized feature representations by learning from multiple source domain data. Future research can leverage adversarial training or ensemble learning could be utilized to align data distributions across multiple source domains while preserving domain-specific features. This would enable the model to generalize better in intrusion detection

tasks and handle novel attack patterns.

(4) Further exploration of TDL in more complex applications.

Research Gap: TDL has been widely applied in classification and recognition tasks, yet its exploration in more complex applications remains insufficient. For instance, applications such as face generation [115], head motion detection [116], fake face detection [117], and multi-person pose estimation [118] are still under-explored in this context. Current models face difficulties in scaling to these highdimensional and non-stationary data environments.

Proposed Improvement: Future research can explore the integration of TDL with advanced architectures such as generative adversarial networks (GANs) or transformers. These hybrid models have the potential to more effectively handle nonlinear dependencies and extract key features, thereby enhancing scalability and generalization capabilities in complex applications.

# (5) The efficiency and scalability of TDL.

Open problem: With the increasing amount of data and the growing number of domains, maintaining the efficiency and scalability of TDL models has become a critical issue. Traditional DL methods often face significant computational overhead, particularly in multi-domain transfer learning scenarios. The high dimensionality and large-scale data involved in these tasks lead to increased memory and computational requirements, making it challenging to scale these models effectively.

Proposed Improvement: Future research can focus on distributed computing or parallel algorithms for TDL. GPU acceleration or distributed storage can be used to improve training speed and scalability. Additionally, exploring sparse representation techniques can help reduce model complexity and memory usage. This approach can achieve more efficient TDL on large-scale datasets.

# 7. Conclusions

This paper presents a comprehensive review of the current state of the TDL algorithms, encompassing various types such as the unsupervised, semi-supervised, and supervised CDDL algorithms, as well as the unsupervised, semi-supervised, and supervised DADL algorithms. Moreover, we analyze the fundamental principles, strengths, and weaknesses of these methods, aiding readers in better understanding the TDL algorithms. For the cross-domain recognition and domain-adaptive (machinery fault diagnosis), some representative algorithms are selected for experimental comparison. Experimental results show that the TDL algorithms achieve competitive performance on the five wellknown databases. In addition, we also give five future research directions related to TDL. Through this review, readers can become acquainted with the latest advancements in TDL and select appropriate methods to solve practical problems. In summary, the paper provides a comprehensive review of TDL, opening up novel research directions and applications within this burgeoning field.

# CRediT authorship contribution statement

Mengyao Li: Conceptualization. Yang Li: Supervision. Zhengming Li: Writing – review & editing, Validation, Supervision, Funding acquisition.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgments

The work is supported by the Natural Science Foundation of Guangdong Province, China (2022A1515011146), and Special Projects for Key Fields in Higher Education of Guangdong, China (2021ZDZX1042).

# Data availability

Data will be made available on request.

# References

[1] F. Zhuang, Z. Qi, K. Duan, D. Xi, Y. Zhu, H. Zhu, H. Xiong, Q. He, A comprehensive survey on transfer learning, Proc. IEEE 109 (1) (2021) 43–76.   
[2] H. Liang, W. Fu, F. Yi, A survey of recent advances in transfer learning, in: 2019 IEEE 19th International Conference on Communication Technology, IEEE, 2019, pp. 1516–1523.   
[3] T. Cody, P.A. Beling, A systems theory of transfer learning, IEEE Syst. J. 17 (1) (2023) 26–37.   
[4] S. Ganvir, N. Lal, Automatic speaker recognition using transfer learning approach of deep learning models, in: 2021 6th International Conference on Inventive Computation Technologies, IEEE, 2021, pp. 595–601.   
[5] J. Li, S. Qiu, Y.-Y. Shen, C.-L. Liu, H. He, Multisource transfer learning for cross-subject EEG emotion recognition, IEEE Trans. Cybern. 50 (7) (2020) 3281–3293.   
[6] H.E. Kim, A. Cosa-Linan, N. Santhanam, M. Jannesari, M.E. Maros, T. Ganslandt, Transfer learning for medical image classification: A literature review, BMC Med. Imaging 22 (1) (2022) 69.   
[7] F. Shoeleh, M. Asadpour, Skill based transfer learning with domain adaptation for continuous reinforcement learning domains, Appl. Intell. 50 (2) (2020) 502–518.   
[8] X. Li, W. Ji, J. Huang, Local instance-based transfer learning for reinforcement learning, Eng. Appl. Artif. Intell. 133 (2024) 108488.   
[9] S. Mavaddati, Voice-based age, gender, and language recognition based on ResNet deep model and transfer learning in spectro-temporal domain, Neurocomputing 580 (2024) 127429.   
[10] X. Zhang, G. Yu, Y. Jin, F. Qian, An adaptive Gaussian process based manifold transfer learning to expensive dynamic multi-objective optimization, Neurocomputing 538 (2023) 126212.   
[11] S.J. Pan, I.W. Tsang, J.T. Kwok, Q. Yang, Domain adaptation via transfer component analysis, IEEE Trans. Neural Netw. 22 (2) (2011) 199–210.   
[12] A. Jamal, V.P. Namboodiri, D. Deodhare, K. Venkatesh, Deep domain adaptation in action space, BMVC 2 (3) (2018) 5.   
[13] M. Long, Z. Cao, J. Wang, M.I. Jordan, Conditional adversarial domain adaptation, Adv. Neural Inf. Process. Syst. 31 (2018) 1647–1657.   
[14] Y. Ganin, V. Lempitsky, Unsupervised domain adaptation by backpropagation, in: International Conference on Machine Learning, PMLR, 2015, pp. 1180–1189.   
[15] E. Tzeng, J. Hoffman, K. Saenko, T. Darrell, Adversarial discriminative domain adaptation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 7167–7176.   
[16] X. Chen, S. Wang, M. Long, J. Wang, Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation, in: International Conference on Machine Learning, PMLR, 2019, pp. 1081–1090.   
[17] S. Niu, Y. Liu, J. Wang, H. Song, A decade survey of transfer learning (2010–2020), IEEE Trans. Artif. Intell. 1 (2) (2020) 151–166.   
[18] J. Lu, V. Behbood, P. Hao, H. Zuo, S. Xue, G. Zhang, Transfer learning using computational intelligence: A survey, Knowl.-Based Syst. 80 (2015) 14–23.   
[19] X. Yu, J. Wang, Q.-Q. Hong, R. Teku, S.-H. Wang, Y.-D. Zhang, Transfer learning for medical images analyses: A survey, Neurocomputing 489 (2022) 230–254.   
[20] W. Pan, A survey of transfer learning for collaborative recommendation with auxiliary data, Neurocomputing 177 (2016) 447–453.   
[21] Z. Li, Z. Zhang, J. Qin, Z. Zhang, L. Shao, Discriminative fisher embedding dictionary learning algorithm for object recognition, IEEE Trans. Neural Netw. Learn. Syst. 31 (3) (2020) 786–800.   
[22] H. Tang, H. Liu, W. Xiao, N. Sebe, When dictionary learning meets deep learning: Deep dictionary learning and coding network for image recognition with limited data, IEEE Trans. Neural Netw. Learn. Syst. 32 (5) (2020) 2129–2141.   
[23] B. Xu, F. Jiang, Z. Zhu, H. Meng, L. Xu, Adaptive convolutional dictionary learning for denoising seismocardiogram to enhance the classification performance of aortic stenosis, Comput. Biol. Med. 168 (2024) 107763.   
[24] Z. Sun, M. Zhang, H. Sun, J. Li, T. Liu, X. Gao, Multi-modal deep convolutional dictionary learning for image denoising, Neurocomputing 562 (2023) 126918.   
[25] H. Wang, G. Dong, J. Chen, X. Hu, Z. Zhu, A novel dictionary learning named deep and shared dictionary learning for fault diagnosis, Mech. Syst. Signal Process. 182 (2023) 109570.   
[26] X. Liu, J. Li, L. Bo, F. Yang, Feature-oriented unified dictionary learning-based sparse classification for multi-domain fault diagnosis, Signal Process. 221 (2024) 109485.   
[27] Y. Xue, V.K. Lau, Online orthogonal dictionary learning based on Frank–Wolfe method, IEEE Trans. Neural Netw. Learn. Syst. 34 (9) (2021) 5774–5788.   
[28] S. Li, L. Wang, S. Wang, D. Kong, B. Yin, Hierarchical coupled discriminative dictionary learning for zero-shot learning, IEEE Trans. Circuits Syst. Video Technol. 33 (9) (2023) 4973–4984.   
[29] Z. Fan, L. Shi, Q. Liu, Z. Li, Z. Zhang, Discriminative fisher embedding dictionary transfer learning for object recognition, IEEE Trans. Neural Netw. Learn. Syst. 34 (1) (2021) 64–78.   
[30] S. Hawe, M. Kleinsteuber, K. Diepold, Analysis operator learning and its application to image reconstruction, IEEE Trans. Image Process. 22 (6) (2013) 2138–2150.

[31] R. Rubinstein, T. Peleg, M. Elad, Analysis K-SVD: A dictionary-learning algorithm for the analysis sparse model, IEEE Trans. Signal Process. 61 (3) (2013) 661–677.   
[32] J. Wang, Y. Guo, J. Guo, X. Luo, X. Kong, Class-aware analysis dictionary learning for pattern classification, IEEE Signal Process. Lett. 24 (12) (2017) 1822–1826.   
[33] J. Guo, Y. Guo, X. Kong, M. Zhang, R. He, Discriminative analysis dictionary learning, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30, No. 1, 2016.   
[34] W. Tang, A. Panahi, H. Krim, L. Dai, Analysis dictionary learning based classification: Structure for robustness, IEEE Trans. Image Process. 28 (12) (2019) 6035–6046.   
[35] Z. Deng, X. Chen, S. Xie, Y. Xie, H. Zhang, Semi-supervised discriminative projective dictionary pair learning and its application to industrial process, IEEE Trans. Ind. Inform. 19 (3) (2023) 3119–3132.   
[36] Z. Chen, X.-J. Wu, J. Kittler, Relaxed block-diagonal dictionary pair learning with locality constraint for image recognition, IEEE Trans. Neural Netw. Learn. Syst. 33 (8) (2021) 3645–3659.   
[37] H. Yu, Q. Yang, G. Wang, Y. Xie, A novel discriminative dictionary pair learning constrained by ordinal locality for mixed frequency data classification, IEEE Trans. Knowl. Data Eng. 34 (10) (2020) 4572–4585.   
[38] Z. Chen, X.-J. Wu, T. Xu, J. Kittler, Discriminative dictionary pair learning with scale-constrained structured representation for image classification, IEEE Trans. Neural Netw. Learn. Syst. 34 (12) (2022) 10225–10239.   
[39] J. Gou, X. Yuan, L. Du, S. Xia, Z. Yi, Hierarchical graph augmented deep collaborative dictionary learning for classification, IEEE Trans. Intell. Transp. Syst. 23 (12) (2022) 25308–25322.   
[40] A. Majumdar, Energy disaggregation via deep convolutional dictionary learning, IEEE Sensors Lett. (2024).   
[41] S. Tariyal, A. Majumdar, R. Singh, M. Vatsa, Deep dictionary learning, IEEE Access 4 (2016) 10096–10109.   
[42] J. Gou, X. Yuan, B. Yu, J. Yu, Z. Yi, Intra-and inter-class induced discriminative deep dictionary learning for visual recognition, IEEE Trans. Multimed. 25 (2023) 1575–1583.   
[43] J. Gou, X. He, L. Du, B. Yu, W. Chen, Z. Yi, Hierarchical locality-aware deep dictionary learning for classification, IEEE Trans. Multimed. 26 (2023) 447–461.   
[44] I. Tošić, P. Frossard, Dictionary learning, IEEE Signal Process. Mag. 28 (2) (2011) 27–38.   
[45] H. Cheng, Z. Liu, L. Yang, X. Chen, Sparse representation and learning in visual recognition: Theory and applications, Signal Process. 93 (6) (2013) 1408–1425.   
[46] L. Jiao, Y. Yang, F. Liu, S. Yang, B. Hou, The new generation brain-inspired sparse learning: A comprehensive survey, IEEE Trans. Artif. Intell. 3 (6) (2022) 887–907.   
[47] E.M. Eksioglu, O. Bayir, K-svd meets transform learning: Transform k-svd, IEEE Signal Process. Lett. 21 (3) (2014) 347–351.   
[48] W. Zhang, J. Lv, X. Li, D. Zhu, X. Jiang, S. Zhang, Y. Zhao, L. Guo, J. Ye, D. Hu, et al., Experimental comparisons of sparse dictionary learning and independent component analysis for brain network inference from fMRI data, IEEE Trans. Biomed. Eng. 66 (1) (2018) 289–299.   
[49] H. Zhao, S. Ding, X. Li, L. Zhao, ?? Norm independently interpretable regularization based sparse coding for highly correlated data, IEEE Access 7 (2019) 53542–53554.   
[50] X. Zhao, W. Li, M. Zhang, R. Tao, P. Ma, Adaptive iterated shrinkage thresholding-based lp-norm sparse representation for hyperspectral imagery target detection, Remote Sens. 12 (23) (2020) 3991.   
[51] J. Gou, X. Wu, H. Dong, Reduced iteration image reconstruction of incomplete projection CT using regularization strategy through Lp norm dictionary learning, J. X-Ray Sci. Technol. 27 (3) (2019) 559–572.   
[52] W. Zuo, D. Meng, L. Zhang, X. Feng, D. Zhang, A generalized iterated shrinkage algorithm for non-convex sparse coding, in: Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 217–224.   
[53] X. Zong-Ben, G. Hai-Liang, W. Yao, H. Zhang, Representative of L1/2 regularization among Lq (0 < ?? ≤ 1) regularizations: An experimental study based on phase diagram, Acta Automat. Sinica 38 (7) (2012) 1225–1228.   
[54] Z. Li, S. Ding, Y. Li, Z. Yang, S. Xie, W. Chen, Manifold optimization-based analysis dictionary learning with an $\ell _ { 1 / 2 }$ -norm regularizer, Neural Netw. 98 (2018) 212–222.   
[55] C. Ao, B. Qiao, L. Chen, J. Xu, M. Liu, X. Chen, Blade dynamic strain nonintrusive measurement using L1/2-norm regularization and transmissibility, Measurement 190 (2022) 110677.   
[56] B. Bai, X. Li, T. Zhang, D. Lin, Nonconvex L1/2 minimization based compressive sensing approach for duct azimuthal mode detection, AIAA J. 58 (9) (2020) 3932–3946.   
[57] Z. Xu, X. Chang, F. Xu, H. Zhang, $\ell _ { 1 / 2 }$ regularization: A thresholding representation theory and a fast solver, IEEE Trans. Neural Netw. Learn. Syst. 23 (7) (2012) 1013–1027.   
[58] Z. Li, Y. Li, B. Tan, S. Ding, S. Xie, Structured sparse coding with the group log-regularizer for key frame extraction, IEEE/CAA J. Autom. Sin. 9 (10) (2022) 1818–1830.

[59] S. Wu, Y. Yan, H. Tang, J. Qian, J. Zhang, Y. Dong, X.-Y. Jing, Structured discriminative tensor dictionary learning for unsupervised domain adaptation, Neurocomputing 442 (2021) 281–295.   
[60] H. Chen, M. Ye, L. Lei, H. Lu, Y. Qian, Semisupervised dual-dictionary learning for heterogeneous transfer learning on cross-scene hyperspectral images, IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens. 13 (2020) 3164–3178.   
[61] N. Han, J. Wu, X. Fang, S. Teng, G. Zhou, S. Xie, X. Li, Projective double reconstructions based dictionary learning algorithm for cross-domain recognition, IEEE Trans. Image Process. 29 (2020) 9220–9233.   
[62] Y. Zheng, X. Wang, G. Zhang, B. Xiao, F. Xiao, J. Zhang, Multi-kernel coupled projections for domain adaptive dictionary learning, IEEE Trans. Multimed. 21 (9) (2019) 2292–2304.   
[63] S.J. Pan, Q. Yang, A survey on transfer learning, IEEE Trans. Knowl. Data Eng. 22 (10) (2010) 1345–1359.   
[64] W. Lu, H.L. Chieu, J. Löfgren, A general regularization framework for domain adaptation, in: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016, pp. 950–954.   
[65] T. Kim, J. Chai, Cross-domain fault diagnosis of bearings using simple structure model combining with signal processing method, in: 2021 IEEE Asia-Pacific Conference on Computer Science and Data Engineering, IEEE, 2021, pp. 1–6.   
[66] V.F. Arruda, T.M. Paixao, R.F. Berriel, A.F. De Souza, C. Badue, N. Sebe, T. Oliveira-Santos, Cross-domain car detection using unsupervised image-to-image translation: From day to night, in: 2019 International Joint Conference on Neural Networks, IEEE, 2019, pp. 1–8.   
[67] X. Fang, L. Jiang, N. Han, W. Sun, Y. Xu, S. Xie, Cross-domain recognition via projective cross-reconstruction, IEEE Trans. Syst., Man, Cybern. Syst. 52 (12) (2022) 7366–7377.   
[68] Y. Lu, D. Li, W. Wang, Z. Lai, J. Zhou, X. Li, Discriminative invariant alignment for unsupervised domain adaptation, IEEE Trans. Multimed. 24 (2021) 1871–1882.   
[69] H. Guan, M. Liu, Domain adaptation for medical image analysis: A survey, IEEE Trans. Biomed. Eng. 69 (3) (2022) 1173–1185.   
[70] W. Lu, B. Liang, Y. Cheng, D. Meng, J. Yang, T. Zhang, Deep model based domain adaptation for fault diagnosis, IEEE Trans. Ind. Electron. 64 (3) (2017) 2296–2305.   
[71] G. Lu, B. Liu, Y. Xiao, Cross-angle behavior recognition via supervised dictionary learning, in: 2017 13th International Conference on Natural Computation, Fuzzy Systems and Knowledge Discovery, IEEE, 2017, pp. 2296–2300.   
[72] Z. Li, Z. Zhang, J. Qin, S. Li, H. Cai, Low-rank analysis–synthesis dictionary learning with adaptively ordinal locality, Neural Netw. 119 (2019) 93–112.   
[73] Z. Li, Z. Lai, Y. Xu, J. Yang, D. Zhang, A locality-constrained and label embedding dictionary learning algorithm for image classification, IEEE Trans. Neural Netw. Learn. Syst. 28 (2) (2017) 278–293.   
[74] Z. Jiang, Z. Lin, L.S. Davis, Label consistent K-SVD: Learning a discriminative dictionary for recognition, IEEE Trans. Pattern Anal. Mach. Intell. 35 (11) (2013) 2651–2664.   
[75] M. Jian, C. Jung, Semi-supervised bi-dictionary learning for image classification with smooth representation-based label propagation, IEEE Trans. Multimed. 18 (3) (2016) 458–473.   
[76] A.S. Edun, C. LaFlamme, S.R. Kingston, H.V. Tetali, E.J. Benoit, M. Scarpulla, C.M. Furse, J.B. Harley, Finding faults in PV systems: Supervised and unsupervised dictionary learning with SSTDR, IEEE Sens. J. 21 (4) (2020) 4855–4865.   
[77] F. Nie, X. Zhao, R. Wang, X. Li, Z. Li, Fuzzy K-means clustering with discriminative embedding, IEEE Trans. Knowl. Data Eng. 34 (3) (2022) 1221–1230.   
[78] L. Qi, J. Huo, X. Fan, Y. Shi, Y. Gao, Unsupervised joint subspace and dictionary learning for enhanced cross-domain person re-identification, IEEE J. Sel. Top. Sign. Proces. 12 (6) (2018) 1263–1275.   
[79] T. Ni, C. Zhang, X. Gu, Transfer model collaborating metric learning and dictionary learning for cross-domain facial expression recognition, IEEE Trans. Comput. Soc. Syst. 8 (5) (2021) 1213–1222.   
[80] M. Shao, D. Kit, Y. Fu, Generalized transfer subspace learning through low-rank constraint, Int. J. Comput. Vis. 109 (1) (2014) 74–93.   
[81] X. Tian, Q. Zhu, Y. Li, M. Wu, Cross-domain joint dictionary learning for ECG reconstruction from PPG, IEEE Int. Conf. Acoust., Speech Signal Process. (2020) 936–940.   
[82] X. Tian, Q. Zhu, Y. Li, M. Wu, Cross-domain joint dictionary learning for ECG inference from PPG, IEEE Internet Things J. 10 (9) (2023) 8140–8154.   
[83] H. Luo, K. Zhang, S. Luo, J. Li, X. Gao, Locality-adaptive structured dictionary learning for cross-domain recognition, IEEE Trans. Circuits Syst. Video Technol. 32 (4) (2022) 2425–2440.   
[84] P. Zhang, J. Xu, Q. Wu, Y. Huang, J. Zhang, Top-push constrained modalityadaptive dictionary learning for cross-modality person re-identification, IEEE Trans. Circuits Syst. Video Technol. 30 (12) (2020) 4554–4566.   
[85] D. Chen, P. Song, W. Zheng, Learning transferable sparse representations for cross-corpus facial expression recognition, IEEE Trans. Affect. Comput. 14 (2) (2023) 1322–1333.   
[86] Z. Ding, Y. Fu, Deep transfer low-rank coding for cross-domain learning, IEEE Trans. Neural Netw. Learn. Syst. 30 (6) (2019) 1768–1779.

[87] X. Li, L. Zhang, B. Du, L. Zhang, On gleaning knowledge from cross domains by sparse subspace correlation analysis for hyperspectral image classification, IEEE Trans. Geosci. Remote Sens. 57 (6) (2019) 3204–3220.   
[88] J. Ni, Q. Qiu, R. Chellappa, Subspace interpolation via dictionary learning for unsupervised domain adaptation, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2013, pp. 692–699.   
[89] X. Wang, T. Wang, A. Ming, W. Zhang, A. Li, F. Chu, Cross-operating condition degradation knowledge learning for remaining useful life estimation of bearings, IEEE Trans. Instrum. Meas. 70 (2021) 1–11.   
[90] K. Huang, H. Wen, C. Zhou, C. Yang, W. Gui, Transfer dictionary learning method for cross-domain multimode process monitoring and fault isolation, IEEE Trans. Instrum. Meas. 69 (11) (2020) 8713–8724.   
[91] J. Zhang, J. Wu, A novel transfer dictionary learning strategy for rolling bearing fault identification with a mixed noise model, IEEE Trans. Instrum. Meas. 70 (2021) 1–10.   
[92] W. Lei, Z. Ma, S. Liu, Y. Lin, EEG mental recognition based on RKHS learning and source dictionary regularized RKHS subspace learning, IEEE Access 9 (2021) 150545–150559.   
[93] W. Cai, M. Gao, Y. Jiang, X. Gu, X. Ning, P. Qian, T. Ni, Hierarchical domain adaptation projective dictionary pair learning model for EEG classification in IoMT systems, IEEE Trans. Comput. Soc. Syst. 10 (4) (2023) 1559–1567.   
[94] M. Ye, Y. Qian, J. Zhou, Y.Y. Tang, Dictionary learning-based feature-level domain adaptation for cross-scene hyperspectral image classification, IEEE Trans. Geosci. Remote Sens. 55 (3) (2017) 1544–1562.   
[95] Q. Tian, C. Ma, M. Cao, J. Wan, Z. Lei, S. Chen, Unsupervised multitarget domain adaptation with dictionary-bridged knowledge exploitation, IEEE Trans. Neural Netw. Learn. Syst. 35 (3) (2024) 3464–3477.   
[96] L. Yan, R. Zhu, Y. Liu, N. Mo, Class-specific dictionary based semi-supervised domain adaptation for land-cover classification of aerial images, IEEE Int. Geosci. Remote. Sens. Symp. (2019) 720–723.   
[97] K. Yan, W. Zheng, Z. Cui, Y. Zong, T. Zhang, C. Tang, Unsupervised facial expression recognition using domain adaptation based dictionary learning approach, Neurocomputing 319 (2018) 84–91.   
[98] J. Huang, D. Guan, A. Xiao, S. Lu, L. Shao, Category contrast for unsupervised domain adaptation in visual tasks, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 1203–1214.   
[99] M. Saffari, M. Khodayar, S.M.J. Jalali, Sparse adversarial unsupervised domain adaptation with deep dictionary learning for traffic scene classification, IEEE Trans. Emerg. Top. Comput. Intell. 7 (4) (2023) 1139–1150.   
[100] K. Kumar, A. Majumdar, A.A. Kumar, M.G. Chandra, Unsupervised domain adaptation via subspace interpolating deep dictionary learning: A case study in machine inspection, IEEE Int. Conf. Acoust., Speech Signal Process. (2023) 1–5.   
[101] T. Liu, X. Zhu, Q. Wang, Domain adaptation on asymmetric drift data for an electronic nose, IEEE Trans. Instrum. Meas. 72 (2023) 1–11.   
[102] F. Yang, K. Yan, S. Lu, H. Jia, D. Xie, Z. Yu, X. Guo, F. Huang, W. Gao, Partaware progressive unsupervised domain adaptation for person re-identification, IEEE Trans. Multimed. 23 (2020) 1681–1695.   
[103] W.A. Smith, R.B. Randall, Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study, Mech. Syst. Signal Process. 64 (2015) 100–131.   
[104] K. Kumar, A. Majumdar, A.A. Kumar, M.G. Chandra, Transform based subspace interpolation for unsupervised domain adaptation applied to machine inspection, in: 2023 31st European Signal Processing Conference, IEEE, 2023, pp. 1708–1712.   
[105] Y. Xu, X. Fang, J. Wu, X. Li, D. Zhang, Discriminative transfer subspace learning via low-rank and sparse representation, IEEE Trans. Image Process. 25 (2) (2016) 850–863.   
[106] K. Saenko, B. Kulis, M. Fritz, T. Darrell, Adapting visual category models to new domains, in: Proceedings of the 11th European Conference on Computer Vision: Part IV, Springer, 2010, pp. 213–226.   
[107] B. Gong, K. Grauman, F. Sha, Learning kernels for unsupervised domain adaptation with applications to visual object recognition, Int. J. Comput. Vis. 109 (1) (2014) 3–27.   
[108] B. Gong, Y. Shi, F. Sha, K. Grauman, Geodesic flow kernel for unsupervised domain adaptation, in: 2012 IEEE Conference on Computer Vision and Pattern Recognition, 2012, pp. 2066–2073.

[109] M. Long, Y. Cao, J. Wang, M. Jordan, Learning transferable features with deep adaptation networks, in: International Conference on Machine Learning, PMLR, 2015, pp. 97–105.   
[110] A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet classification with deep convolutional neural networks, Commun. ACM 60 (6) (2017) 84–90.   
[111] V. Singhal, A. Majumdar, Reconstructing multi-echo magnetic resonance images via structured deep dictionary learning, Neurocomputing 408 (2020) 135–143.   
[112] T. Feizi, M.H. Moattar, H. Tabatabaee, M2GDL: Multi-manifold guided dictionary learning based oversampling and data validation for highly imbalanced classification problems, Inform. Sci. 682 (2024) 121280.   
[113] J. Wu, H. Dai, Y. Wang, K. Ye, C. Xu, Heterogeneous domain adaptation for IoT intrusion detection: A geometric graph alignment approach, IEEE Internet Things J. 10 (12) (2023) 10764–10777.   
[114] L. Yang, A. Shami, A transfer learning and optimized CNN based intrusion detection system for internet of vehicles, in: ICC 2022-IEEE International Conference on Communications, IEEE, 2022, pp. 2774–2779.   
[115] X. Tu, Y. Zou, J. Zhao, W. Ai, J. Dong, Y. Yao, Z. Wang, G. Guo, Z. Li, W. Liu, et al., Image-to-video generation via 3D facial dynamics, IEEE Trans. Circuits Syst. Video Technol. 32 (4) (2022) 1805–1819.   
[116] Z. Wang, J. Zhao, C. Lu, F. Yang, H. Huang, Y. Guo, et al., Learning to detect head movement in unconstrained remote gaze estimation in the wild, in: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2020, pp. 3443–3452.   
[117] Y. Kim, J.-H. Yoo, K. Choi, A motion and similarity-based fake detection method for biometric face recognition systems, IEEE Trans. Consum. Electron. 57 (2) (2011) 756–762.   
[118] J. Zhao, J. Li, H. Liu, S. Yan, J. Feng, Fine-grained multi-human parsing, Int. J. Comput. Vis. 128 (2020) 2185–2203.

![](images/bd6373039fd475225fd259f0094fa7ab39d5c68a2817b9e662f4bbdff3e63e24.jpg)

<details>
<summary>natural_image</summary>

Portrait photo of a young woman with short dark hair wearing a school uniform (no text or symbols visible)
</details>

Mengyao Li is currently pursuing her master’s degree at Guangdong Polytechnic Normal University. Her current research interests include transfer learning and Artificial intelligence security.

![](images/4fc69e46173af60b34370a174095b80b5a7cdda92fe32cbe2931c930b6948be9.jpg)

<details>
<summary>natural_image</summary>

Portrait of a person wearing glasses against a blue background (no text or symbols visible)
</details>

Yang Li is currently pursuing his master’s degree at Guangdong Polytechnic Normal University. His current research interests include intrusion detection and Artificial intelligence security.

![](images/d88f9adb7f5bea69432706a42e40a28f6d38ac1fd66729f9f865e59ea082b69c.jpg)

<details>
<summary>natural_image</summary>

Portrait photo of a man against a blue background (no text or symbols visible)
</details>

Zhengming Li received M.S and Ph.D. degrees from Southwest University and Harbin Institute of Technology in 2007 and 2017, respectively. Currently, he is a Professor with the School of Cyber Security, Guangdong Polytechnic Normal University. His current research interests include pattern recognition and Artificial intelligence security.