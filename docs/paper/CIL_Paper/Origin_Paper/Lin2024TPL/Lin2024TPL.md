# CLASS INCREMENTAL LEARNING VIA LIKELIHOOD RATIO BASED TASK PREDICTION

Haowei Lin1, Yijia Shao2, Weinan Qian3, Ningxin Pan3, Yiduo Guo3, and Bing Liu4∗

1Institute for Artificial Intelligence, Peking University 2Stanford University

3Wangxuan Institute of Computer Technology, Peking University

4Department of Computer Science, University of Illinois at Chicago

1linhaowei@pku.edu.cn 2shaoyj@stanford.edu

3{ypqwn, 2100017816, yiduo}@stu.pku.edu.cn 4liub@uic.edu

# ABSTRACT

Class incremental learning (CIL) is a challenging setting of continual learning, which learns a series of tasks sequentially. Each task consists of a set of unique classes. The key feature of CIL is that no task identifier (or task-id) is provided at test time. Predicting the task-id for each test sample is a challenging problem. An emerging theory-guided approach (called TIL+OOD) is to train a task-specific model for each task in a shared network for all tasks based on a task-incremental learning (TIL) method to deal with catastrophic forgetting. The model for each task is an out-of-distribution (OOD) detector rather than a conventional classifier. The OOD detector can perform both within-task (in-distribution (IND)) class prediction and OOD detection. The OOD detection capability is the key to task-id prediction during inference. However, this paper argues that using a traditional OOD detector for task-id prediction is sub-optimal because additional information (e.g., the replay data and the learned tasks) available in CIL can be exploited to design a better and principled method for task-id prediction. We call the new method TPL (Task-id Prediction based on Likelihood Ratio). TPL markedly outperforms strong CIL baselines and has negligible catastrophic forgetting.

# 1 INTRODUCTION

Continual learning learns a sequence of tasks, 1, 2, · · · , T , incrementally (Ke & Liu, 2022; De Lange et al., 2021). Each task t consists of a set of classes to be learned. This paper focuses on the challenging CL setting of class-incremental learning (CIL) (Rebuffi et al., 2017). The key challenge of CIL lies in the absence of task-identifier (task-id) in testing. There is another CL setting termed task-incremental learning (TIL), which learns a separate model or classifier for each task. In testing, the task-id is provided for each test sample so that it is classified by the task specific model.

A main assumption of continual learning is that once a task is learned, its training data is no longer accessible. This causes catastrophic forgetting (CF), which refers to performance degradation of previous tasks due to parameter updates in learning each new task (McCloskey & Cohen, 1989). An additional challenge specifically for CIL is inter-task class separation (ICS) (Kim et al., 2022b). That is, when learning a new task, it is hard to establish decision boundaries between the classes of the new task and the classes of the previous tasks without the training data of the previous tasks. Although in replay-based methods (Rebuffi et al., 2017; Kemker & Kanan, 2017; Lopez-Paz & Ranzato, 2017), a small number of training samples can be saved from each task (called the replay data) to help deal with CF and ICS to some extent by jointly training the new task data and the replay data from previous tasks, the effect on CF and ICS is limited as the number of replay samples is very small.

An emerging theoretically justified approach to solving CIL is to combine a TIL technique with an out-of-distribution (OOD) detection method, called the TIL+OOD approach (Kim et al., 2022b). The TIL method learns a model for each task in a shared network. The model for each task is not a traditional classifier but an OOD detector. Note that almost all OOD detection methods can perform two tasks (1) in-distribution (IND) classification and (2) out-of-distribution (OOD) detection (Vaze et al., 2022). At test time, for each test sample, the system first computes a task-id prediction (TP) probability and a within-task prediction (WP) probability (Kim et al., 2022b) (same as IND classification) for each task. The two probabilities are then combined to make the final classification decision, which produces state-of-the-art results (Kim et al., 2022b; 2023). In this approach, WP is usually very accurate because it uses the task-specific model. TP is the key challenge.

There is a related existing approach that first predicts task-id and then predicts the class of the test sample using the task-specific model (Rajasegaran et al., 2020; Abati et al., 2020; Von Oswald et al., 2019). However, what is new is that Kim et al. (2022b) theoretically proved that TP is correlated with OOD detection of each task. Thus, the OOD detection capability of each task model can be used for task-id prediction of each test sample. The previous methods did not realize this and thus performed poorly (Kim et al., 2022b). In Kim et al. (2022b), the authors used the TIL method HAT (Serra et al., 2018) and OOD detection method CSI (Tack et al., 2020). HAT is a parameter isolation method for TIL, which learns a model for each task in a shared network and each task model is protected with learned masks to overcome CF. Each task model is an OOD detector based on CSI.2

Our paper argues that using traditional OOD detectors is not optimal for task-id prediction as they are not designed for CIL and thus do not exploit the information available in CIL for better task-id prediction. By leveraging the information in CIL, we can do much better. A new method for task-id prediction is proposed, which we call TPL (Task-id Prediction based on Likelihood Ratio). It consists of two parts: (1) a new method to train each task model and (2) a novel and principled method for task-id prediction, i.e., to estimate the probability of a test sample x belonging to a task t, i.e., $\mathbf { P } ( t | { \boldsymbol { x } } ) )$ . We formulate the estimation of $\mathbf { P } ( t | { \boldsymbol { x } } )$ as a binary selection problem between two events “x belongs to $t ^ { \ast }$ and “x belongs to tc”. tc is t’s complement with regard to the universal set $U _ { C I L }$ , which consists of all tasks that have been learned, i.e., $U _ { C I L } = \{ 1 , 2 , \bar { \cdot } \bar { \cdot } , T \}$ and $t ^ { c } = U _ { C I L } - \{ t \}$ .

The idea of TPL is analogous to using OOD detection for task-id prediction in the previous work. However, there is a crucial difference. In traditional OOD detection, given a set $U _ { I N D }$ of indistribution classes, we want to estimate the probability that a test sample does not belong to any classes in $U _ { I N D }$ . This means the universal set $U _ { O O D }$ for OOD detection includes all possible classes in the world (except those in $U _ { I N D } )$ , which is at least very large if not infinite in size and we have no data from $U _ { O O D }$ . Then, there is no way we can estimate the distribution of $U _ { O O D }$ . However, we can estimate the distribution of $U _ { C I L }$ based on the saved replay data3 from each task in CIL. This allows us to use the likelihood ratio of $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ to provide a principled solution towards the binary selection problem and consequently to produce the task-id prediction probability $\mathbf { P } ( t | { \boldsymbol { \mathbf { x } } } )$ as analyzed in Sec. 4.1, where $\mathcal { P } _ { t }$ is the distribution of the data in task t and $\mathcal { P } _ { t ^ { c } }$ is the distribution of the data in tc (all other tasks than t), i.e., t’s complement $( t ^ { c } = U _ { C I L } - \{ t \} )$ .

The proposed system (also called TPL) uses the learned masks in the TIL method HAT for overcoming CF but the model for each task within HAT is not a traditional classifier but a model that facilitates task-id prediction (Sec. 3). At test time, given a test sample, the proposed likelihood ratio method is integrated with a logit-based score using an energy function to compute the task-id prediction probability and within-task prediction probability for the test sample to finally predict its class. Our experiments with and without using a pre-trained model show that TPL markedly outperforms strong baselines. With a pre-trained model, TPL has almost no forgetting or performance deterioration. We also found that the current formula for computing the forgetting rate is not appropriate for CIL.

# 2 RELATED WORK

OOD Detection. OOD detection has been studied extensively. Hendrycks & Gimpel (2016) use the maximum softmax probability (MSP) as the OOD score. Some researchers also exploit the logit space (Liang et al., 2017; Liu et al., 2020a; Sun et al., 2021), and the feature space to compute the distance from the test sample to the training data/IND distribution, e.g., Mahalanobis distance (Lee et al., 2018b) and KNN (Sun et al., 2022). Some use real/generated OOD data (Wang et al., 2022d; Liu et al., 2020a; Lee et al., 2018a). Our task-id prediction does not use any existing OOD method.

Continual Learning (CL). Existing CL methods are of four main types. (1) Regularization-based methods address forgetting (CF) by using regularizers in the loss function (Kirkpatrick et al., 2017; Zhu et al., 2021) or orthogonal projection (Zeng et al., 2019) to preserve previous important parameters. The regularizers in DER (Yan et al., 2021) and BEEF (Wang et al., 2022a) are similar to OOD detection but they expand the network for each task and perform markedly poorer than our method. (2) Replay-based methods save a few samples from each task and replay them in training new tasks (Kemker & Kanan, 2017; Lopez-Paz & Ranzato, 2017; Li et al., 2022). However, replaying causes data imbalance (Guo et al., 2023; Xiang & Shlizerman, 2023; Ahn et al., 2021). (3) Parameter isolation methods train a sub-network for each task. HAT (Serra et al., 2018) and SupSup (Wortsman et al., 2020) are two representative methods. This approach is mainly used in task-incremental learning (TIL) and can eliminate CF. (4) TIL+OOD based methods have been discussed in Sec. 1.

Recently, using pre-trained models has become a standard practice for CL in both NLP (Ke et al., 2021a;b; 2023; Shao et al., 2023). and computer vision (CV) (Kim et al., 2022a; Wang et al., 2022e). See the surveys (Ke & Liu, 2022; Wang et al., 2023; De Lange et al., 2021; Hadsell et al., 2020).

Our work is closely related to CIL methods that employ a TIL technique and a task-id predictor. iTAML (Rajasegaran et al., 2020) assumes that each test batch is from a single task and uses the whole batch to detect the task-id. This assumption is unrealistic. CCG (Abati et al., 2020) uses a separate network to predict the task-id. Expert Gate (Aljundi et al., 2017) builds a distinct auto-encoder for each task. HyperNet (Von Oswald et al., 2019) and PR-Ent (Henning et al., 2021) use entropy to predict the task-id. However, these systems perform poorly as they did not realize that OOD detection is the key to task-id prediction (Kim et al., 2022b), which proposed the TIL+OOD approach. Kim et al. (2022b) gave two methods HAT+CSI and SupSup+CSI (Kim et al., 2022b). These two methods do not use a pre-trained model or replay data. The same approach was also taken in MORE (Kim et al., 2022a) and ROW (Kim et al., 2023) but they employ a pre-trained model and replay data in CIL. These methods have established a state-of-the-art performance. We have discussed how our proposed method TPL is different from them in the introduction section.

# 3 OVERVIEW OF THE PROPOSED METHOD

Preliminary. an input space s incremental , a label space ing (CIL) learns a, and a training set $1 , . . . , T$ . Each task t hasrawn i.i.d. from $\chi ^ { ( t ) }$ $\mathcal { y } ^ { ( t ) }$ $\mathcal { D } ^ { ( t ) } = \{ ( \boldsymbol { x } _ { j } ^ { ( t ) } , y _ { j } ^ { ( t ) } ) \} _ { j = 1 } ^ { n ^ { ( t ) } }$ ${ \mathcal { P } } _ { \mathcal { X } ^ { ( t ) } \mathcal { Y } ^ { ( t ) } }$ . The class labels of the tasks are disjoint, $\mathrm { i . e . , } \mathcal { V } ^ { ( i ) } \cap \mathcal { V } ^ { ( k ) } = \emptyset , \forall i \neq k$ . The goal of CIL is to learn a function $f : \cup _ { t = 1 } ^ { T } \chi ^ { ( t ) }  \cup _ { t = 1 } ^ { T } \mathcal { y } ^ { ( t ) }$ to predict the class label of each test sample x.

Kim et al. (2022b) proposed a theory for solving CIL. It decomposes the CIL probability of a test sample x of the j-th class $y _ { j } ^ { ( t ) }$ in task t into two probabilities (as the classes in all tasks are disjoint),

$$
\mathbf {P} (y _ {j} ^ {(t)} | \boldsymbol {x}) = \mathbf {P} (y _ {j} ^ {(t)} | \boldsymbol {x}, t) \mathbf {P} (t | \boldsymbol {x}). \tag {1}
$$

The two probabilities on the right-hand-side (R.H.S) define the CIL probability on the left-hand-side (L.H.S). The first probability on the R.H.S. is the within-task prediction (WP) probability and the second probability on the R.H.S. is the task-id prediction (TP) probability. Existing TIL+OOD methods basically use a traditional OOD detection method to build each task model. The OOD detection model for each task is exploited for estimating both TP and WP probabilities (see Sec. 1).

Overview of the Proposed TPL. This paper focuses on proposing a novel method for estimating task-id prediction probability, i.e., the probability of a test sample x belonging to (or drawing from the distribution of) a task t, i.e., P(t|x) in Sec. 1. The WP probability $\mathbf { P } ( y _ { j } ^ { ( t ) } | \mathbf { x } , t )$ can be obtained directly from the model of each task.

The mask-based method in HAT is used by our method to prevent CF. Briefly, in learning each task, it learns a model for the task and also a set of masks for those important neurons to be used later to prevent the model from being updated by future tasks. In learning a new task, the masks of previous models stop the gradient flow to those masked neurons in back-propagation, which eliminates CF. In the forward pass, all the neurons can be used, so the network is shared by all tasks. We note that our method can also leverage some other TIL methods other than HAT to prevent CF (see Appendix G).

The proposed method TPL is illustrated in Figure 1. It has two techniques for accurate estimation of $\mathbf { P } ( t | \bar { \boldsymbol { x } } )$ , one in training and one in testing (inference).

![](images/012b8a380ba301d7ac57566e1a40df9f5c25a6358a220ccdfcb161e6a90bd7cb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Network Architecture"] --> B["Task 1"]
    A --> C["Task 2"]
    A --> D["..."]
    A --> E["Task t"]
    B --> F["Task-id"]
    C --> G["Task-id"]
    D --> H["Task-id"]
    E --> I["Task-id"]
    F --> J["Adapter (HAT)"]
    G --> K["Adapter (HAT)"]
    H --> L["Multi-Head Attention"]
    I --> M["Embedded Patches"]
    J --> N["L ×"]
    K --> O["Norm"]
    L --> P["+"]
    M --> Q["+"]
    N --> R["MLP"]
    O --> S["MLP"]
    P --> T["Encoder"]
    Q --> U["Encoder"]
    R --> V["Output"]
    S --> W["Output"]
    T --> X["Output"]
    U --> Y["Output"]
    V --> Z["Output"]
    W --> AA["Inference Pipeline"]
    X --> AA
    Y --> AA
    Z --> AB["Logits for task t f(h(x; φ(t)); θ(t))"]
    AB --> AC["Within-task logits"]
    AC --> AD["Softmax"]
    AD --> AE["CIL probability: P(y_j^(t)|x) = P(y_j^(t)|x, t) × P(t|x)"]
    AE --> AF["Feature-based likelihood ratio score h(x; φ^(t))"]
    AF --> AG["S_MLS^(t)(x)"]
    AG --> AH["Max logit score"]
```
</details>

Figure 1: Illustration of the proposed TPL. We use a pre-trained transformer network (in the grey box) (see Sec. 5.1 for the case without using a pre-trained network). The pre-trained network is fixed and only the adapters (Houlsby et al., 2019) inserted into the transformer are trainable to adapt to specific tasks. It is important to note that the adapter (in yellow) used by HAT learns all tasks within the same adapter. The yellow boxes on the left show the progressive changes to the adapter as more tasks are learned.

(1) Training: In the original HAT, each model is a traditional supervised classifier trained with cross-entropy. However, for our purpose of predicting task-id, this is insufficient because it has no consideration of the other classes learned from other tasks. In TPL, each model for a task t is trained using the classes $\mathcal { y } ^ { ( t ) }$ of task t and an extra class (called O, for others) representing the replay buffer data $B u f _ { < t }$ of all the previous tasks. This enables each model to consider not only the new task data but also previous tasks’ data, which facilitates more accurate computation of $\mathbf { P } ( { \dot { t } } | { \mathbf { \mathit { x } } } )$ .

For each task $t ,$ its model consists of a feature extractor $h ( \boldsymbol { x } ; \boldsymbol { \phi } ^ { ( t ) } )$ (partially shared with other tasks based on HAT), and a task-specific classifier $f ( z ; \theta ^ { ( t ) } )$ . When learning task t, the model receives the training data $\mathcal { D } ^ { ( t ) }$ and the replay data $B u f _ { < t }$ (stored in a memory buffer). Then we minimize the loss:

$$
\mathcal {L} \left(\theta^ {(t)}, \phi^ {(t)}\right) = \mathbb {E} _ {\left(\boldsymbol {x}, y\right) \sim \mathcal {D} ^ {(t)} \cup B u f _ {<   t}} \left[ \mathcal {L} _ {C E} \left(f \left(h \left(\boldsymbol {x}; \phi^ {(t)}\right); \theta^ {(t)}\right), y\right) \right] + \mathcal {L} _ {H A T}, \tag {2}
$$

where $\mathcal { L } _ { C E }$ is the cross-entropy loss, $\mathcal { L } _ { H A T }$ is the regularization loss used in HAT (see Appendix G).

(2) Testing (or inference): We follow $\mathrm { e q . }$ (1) to compute the CIL probability. The WP probability $( \mathbf { P } ( y _ { i } ^ { ( t ) } | x , t ) )$ for each test sample is computed through softmax on only the original classes $\mathcal { y } ^ { ( t ) }$ of task $\check { t } ,$ the first term on the right of eq. (3) (also see the top right part in Figure 1). The O class is not used in inference. Note that the probabilities for different tasks can be computed in parallel.

$$
\mathbf {P} (y _ {j} ^ {(t)} | \boldsymbol {x}) = \left[ s o f t m a x \left(f (h (\boldsymbol {x}; \phi^ {(t)}); \theta^ {(t)})\right) \right] _ {j} \cdot \mathbf {P} (t | \boldsymbol {x}) \tag {3}
$$

The class y(tj $y _ { j } ^ { ( t ) }$ with the highest probability will be predicted as the class for test sample $x . ^ { 4 }$ We discuss the proposed method for computing task-id prediction probability $\mathbf { P } ( t | { \boldsymbol { \mathbf { \mathit { x } } } } )$ (see the bottom right part in Figure 1) in the next section. Training will not be discussed any further.

# 4 ESTIMATING TASK-ID PREDICTION PROBABILITY

# 4.1 THEORETICAL ANALYSIS

As noted in Sec. 1, we estimate the TP probability $\mathbf { P } ( t | { \boldsymbol { \mathbf { \mathit { x } } } } )$ by predicting whether a sample x is drawn from the distribution $\mathcal { P } _ { t }$ of task t or drawn from the distribution of t’s complement $t ^ { c }$ , i.e., $\mathcal { P } _ { t ^ { c } }$ . We denote the universal set $U _ { C I I }$ of all tasks (or task-ids) that have been learned, i.e., $U _ { C I L } = \{ 1 , 2 , \cdots , T \}$ and $t ^ { c } = U _ { C I L } - \{ t \}$ . From a frequentist perspective, our objective can be formulated as a binary hypothesis test:

$$
\mathcal {H} _ {0}: \boldsymbol {x} \sim \mathcal {P} _ {t} \quad v. s. \quad \mathcal {H} _ {1}: \boldsymbol {x} \sim \mathcal {P} _ {t ^ {c}}, \tag {4}
$$

Using the Neyman-Pearson lemma (Neyman & Pearson, 1933), we can derive a theorem that demonstrates the principled role of likelihood ratio in this task (the proofs are given in Appendix E):

Theorem 4.1 A test with rejection region R defined as follows is a unique uniformly most powerful (UMP) test for the hypothesis test problem defined in eq. (4):

$$
\mathcal {R} := \{\boldsymbol {x}: p _ {t} (\boldsymbol {x}) / p _ {t ^ {c}} (\boldsymbol {x}) <   \lambda_ {0} \}.
$$

where $\lambda _ { 0 }$ is a threshold that can be chosen to obtain a specified significance level.

Theorem 4.2 The UMP test for hypothesis test defined in eq. (4) maximizes the Area Under the Curve (AUC) of binary classification between $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ .

Theorems 4.1 and 4.2 highlight the importance of detecting samples that do not belong to task t based on low t density $p _ { t } ( { \pmb x } )$ and high tc density $p _ { t ^ { c } } ( { \pmb x } )$ .

Note that in traditional OOD detection, the system has no access to the true OOD distribution $\mathcal { P } _ { t ^ { c } }$ but only $\mathcal { P } _ { t }$ (IND distribution). Some existing methods resort to a proxy distribution $\mathcal { P } _ { t ^ { c } } ^ { p r o x y }$ , such as a uniform distribution (Nalisnick et al., 2018) or an auxiliary data distribution $( \mathrm { L i n } \ \& \ \mathrm { G i } , 2 0 2 3 )$ as the universal sdenoted by $U _ { O O D } ^ { ( t ) }$ the set of all classes in the world and the universal set of all OOD classes for task t is very large if not infinite. This approach can lead to potential risks. For instance, consider a scenario where $\mathcal { P } _ { t ^ { c } } = \mathcal { N } ( 0 , 0 . 0 1 )$ and $\bar { \mathcal { P } } _ { t } ^ { \top } = \mathcal { N } ( 0 , 1 )$ . It is apparent that $p _ { t } ( 0 ) > p _ { t } ( 1 )$ , but 0 is more likely to belongs to $\mathcal { P } _ { t ^ { c } }$ c than 1 as $0 . 1 = p _ { t } ( 0 ) / \dot { p } _ { t ^ { c } } ( \dot { 0 } ) < p _ { t } ( \bar { 1 } \dot { ) } / p _ { t ^ { c } } ( 1 ) = 0 . 1 \cdot \mathrm { e } ^ { 4 9 . 5 }$ . We further show the failure cases in real CIL scenarios in Appendix H.

Good News for CIL. In CIL, the IND distribution $\mathcal { P } _ { t }$ for task t can be interpreted as the marginal distribution ${ \mathcal { P } } _ { \mathcal { X } ^ { ( t ) } }$ , while $\mathcal { P } _ { t ^ { c } }$ corresponds to a mixture distribution $\mathcal { P } _ { \mathcal { X } ( t ^ { c } ) }$ comprising the individual marginal distributions $\{ \mathcal { P } _ { \mathcal { X } ^ { ( t ^ { * } ) } } \} _ { t ^ { * } \neq t }$ (which can be estimated based on the saved replay data), each of which is assigned the equal mixture weight. Consequently, we have the knowledge of $\mathcal { P } _ { t ^ { c } }$ in CIL, thereby offering an opportunity to estimate $\mathcal { P } _ { t ^ { c } }$ to be used to compute task-id prediction $\mathbf { P } ( t | { \boldsymbol { \mathbf { x } } } )$ more accurately. This leads to our design of TPL in the following subsections.

# 4.2 COMPUTING TASK-ID PREDICTION PROBABILITY

We now present the proposed method for computing the task-id prediction probability $\mathbf { P } ( t | { \boldsymbol { x } } )$ , which has three parts: (1) estimating both $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ (as analyzed in Sec. 4.1) and computing the likelihood ratio, (2) integrating the likelihood ratio based score with a logit-based score for further improvement, and (3) applying a softmax function on the scores for all tasks to obtain the task-id prediction probability for each task. The three parts correspond to the bottom right part of Figure 1.

# 4.2.1 ESTIMATING $\mathcal { P } _ { t }$ AND $\mathcal { P } _ { t ^ { c } }$ AND COMPUTING LIKELIHOOD RATIO

Guided by Theorem 4.1, we design a task-id prediction score based on the likelihood ratio $p _ { t } ( { \pmb x } ) / p _ { t ^ { c } } ( { \pmb x } )$ . However, due to the challenges in directly estimating the data distribution within the high-dimensional raw image space, we instead consider estimation in the low-dimensional feature space. Interestingly, many distance-based OOD detection scores can function as density estimators that estimate the IND density $p ( { \pmb x } )$ in the feature space (see Appendix E.4 for justifications). For instance, MD (Mahalanobis Distance) (Lee et al., 2018b) estimates distributions using Gaussian mixture models, while KNN (Sun et al., 2022) uses non-parametric estimation. Our method TPL also uses the two scores to estimate distributions (i.e., $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ in our case).

To connect the normalized probability density with unnormalized task-id prediction scores, we leverage energy-based models (EBMs) to parameterize $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ . Given a test sample ${ \mathbf { \mathit { x } } } ,$ it has density $p _ { t } ( { \pmb x } ) \tilde { \ } = \exp \{ E _ { t } ( { \pmb x } ) \} / Z _ { 1 }$ in $\mathcal { P } _ { t }$ , and density $p _ { t ^ { c } } ( { \pmb x } ) = \exp \{ E _ { t ^ { c } } ( { \pmb x } ) \} / Z _ { 2 }$ in $\mathcal { P } _ { t ^ { c } }$ , where $Z _ { 1 } , Z _ { 2 }$ are normalization constants that ensure the integral of densities $p _ { t } ( { \pmb x } )$ and $p _ { t ^ { c } } ( { \pmb x } )$ equal 1, and $E _ { t } ( \cdot ) , E _ { t ^ { c } } ( \cdot )$ are called energy functions.5 Consequently, we can design a feature-based task-id prediction score using the Likelihood Ratio (LR), which is also shown at the bottom right of Figure 1:

$$
S _ {L R} ^ {(t)} (\boldsymbol {x}) = \log (p _ {t} (\boldsymbol {x}) / p _ {t ^ {c}} (\boldsymbol {x})) = E _ {t} (\boldsymbol {x}) - E _ {t ^ {c}} (\boldsymbol {x}) + \log (Z _ {2} / Z _ {1}). \tag {5}
$$

Since $\log ( Z _ { 2 } / Z _ { 1 } )$ is a constant, it can be omitted in the task-id prediction score definition:

$$
S _ {L R} ^ {(t)} (\boldsymbol {x}) := E _ {t} (\boldsymbol {x}) - E _ {t ^ {c}} (\boldsymbol {x}), \tag {6}
$$

Since the energy functions $E _ { t } ( \cdot )$ and $E _ { t ^ { c } } ( \cdot )$ need not to be normalized, we estimate them with the above scores. We next discuss how to choose specific $E _ { t } ( \cdot )$ and $E _ { t ^ { c } } ( \cdot )$ for eq. (6).

For in-task energy $E _ { t } ( \pmb { x } )$ of a task, we simply adopt an OOD detection score $S _ { M D } ( { \pmb x } )$ , which is the OOD score for MD and is defined as the inverse of the minimum Mahalanobis distance of feature $h ( \boldsymbol { x } ; \boldsymbol { \phi } ^ { ( t ) } )$ to all class centroids. The details of how $S _ { M D } ^ { ( t ) } ( { \pmb x } )$ is computed are given in Appendix F.1.

For out-of-task energy $E _ { t ^ { c } } ( { \pmb x } )$ of a task, we use replay data from other tasks for estimation. Let $B u f _ { t c }$ be the set of buffer/replay data excluding the data of classes in task t. We set $E _ { t ^ { c } } ( { \pmb x } ) =$ $- d _ { K N N } ( { \pmb x } , B u f _ { t ^ { c } } )$ , where $d _ { K N N } ( \pmb { x } , B u f _ { t ^ { c } } )$ is the k-nearest distances of the feature $h ( \pmb { x } ; \phi ) ^ { ( t ) }$ to the set replay of features of the replay $B u f _ { t ^ { \prime } }$ t t c data is small in the feature space. The vanilla KNN score is $B u f _ { t ^ { c } }$ data. If $\dot { d } _ { K N N } ( { \pmb x } , B u f _ { t ^ { c } } )$ is small, it means the distance between x and $S _ { K N N } ^ { ( t ) } ( \pmb { x } ) = - d _ { K N N } ( \pmb { x } , \mathcal { D } ^ { ( t ) } )$ which was originally designed to estimate $p _ { t } ( \pmb { x } )$ using the training set $\mathcal { D } ^ { ( t ) }$ . Here we adopt it to estimate $p _ { t ^ { c } } ( { \pmb x } )$ using the replay data $( B u f _ { t ^ { c } } )$ . Finally, we obtain,

$$
S _ {L R} ^ {(t)} (\boldsymbol {x}) := \underbrace {\alpha \cdot S _ {M D} ^ {(t)} (\boldsymbol {x})} _ {E _ {t} (\boldsymbol {x})} + \underbrace {d _ {K N N} (\boldsymbol {x} , B u f _ {t ^ {c}})} _ {- E _ {t ^ {c}} (\boldsymbol {x})}, \tag {7}
$$

where α is a hyper-parameter to make the two scores comparable. This is a principled task-id prediction score as justified in Sec. 4.1.

Remarks. We can also use some other feature-based estimation methods instead of MD and KNN in $S _ { L R } ( { \pmb x } )$ . The reason why we choose MD to estimate $\mathcal { P } _ { t }$ is that it does not require the task data at test time (but KNN does), and we choose KNN to estimate $\mathcal { P } _ { t } .$ c because the non-parametric estimator KNN is high performing (Yang et al., 2022) and we use only the saved replay data for this. We will conduct an ablation study using different estimation methods for both $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ in Sec. 5.3.

# 4.2.2 COMBINING WITH A LOGIT-BASED SCORE

To further improve the task-id prediction score, we combine the feature-based $S _ { L R }$ score with a logit-based score, which has been shown quite effective in OOD detection (Wang et al., 2022c).

We again develop an energy-based model (EBM) framework for the combination that offers a principled approach to composing different task-id prediction scores. Specifically, to combine the proposed feature-based $S _ { L R } ^ { ( t ) } ( \cdot )$ score with a logit-based score (an energy function) $S _ { l o g i t } ^ { ( t ) } ( \cdot )$ , we can make the composition as:

$$
E _ {\text { composition }} (\boldsymbol {x}) = \log (\exp \{\alpha_ {1} \cdot S _ {\text { logit }} ^ {(t)} (\boldsymbol {x}) \} + \exp \{\alpha_ {2} \cdot S _ {L R} ^ {(t)} (\boldsymbol {x}) \}), \tag {8}
$$

where $\alpha _ { 1 }$ and $\alpha _ { 2 }$ are scaling terms to make different scores comparable. As noted in (Du et al., 2020), the composition emulates an OR gate for energy functions.

To choose a logit-based method for $S _ { l o g i t } ^ { ( t ) } ( \cdot )$ in eq. (8), we opt for the simple yet effective method MLS score $S _ { M L S } ^ { ( t ) } ( { \pmb x } )$ , which is defined as the maximum logit of x (also shown on the right of Figure 1).

Our final score $S _ { T P L } ^ { ( t ) } ( { \pmb x } )$ , which integrates feature-based $S _ { L R } ^ { ( t ) } ( \cdot )$ and the logit-based $S _ { M L S } ^ { ( t ) } ( \cdot )$ scores, uses the composition in Eq. 8:

$$
S _ {T P L} ^ {(t)} (\boldsymbol {x}) = \log \left(\exp \{\beta_ {1} \cdot S _ {M L S} ^ {(t)} (\boldsymbol {x}) \} + \exp \{\beta_ {2} \cdot S _ {M D} ^ {(t)} (\boldsymbol {x}) + d _ {K N N} (\boldsymbol {x}, B u f _ {t _ {c}}) \}\right), \tag {9}
$$

where $\beta _ { 1 }$ and $\beta _ { 2 }$ are scaling terms, which are given by merging α in eq. (7) and $\alpha _ { 1 } , \alpha _ { 2 }$ in eq. (8). Since the scale of $d _ { K N N } ( \cdot )$ is near to 1, we simply choose $\beta _ { 1 }$ and $\beta _ { 2 }$ to be the inverse of empirical means of $S _ { M L S } ^ { ( t ) } ( { \pmb x } )$ and $S _ { M D } ^ { ( t ) } ( { \pmb x } )$ estimated by the training data $\mathcal { D } ^ { ( t ) }$ to make different scores comparable:

$$
\frac {1}{\beta_ {1}} = \frac {1}{| \mathcal {D} ^ {(t)} |} \sum_ {\boldsymbol {x} \in \mathcal {D} ^ {(t)}} S _ {M L S} ^ {(t)} (\boldsymbol {x}), \quad \frac {1}{\beta_ {2}} = \frac {1}{| \mathcal {D} ^ {(t)} |} \sum_ {\boldsymbol {x} \in \mathcal {D} ^ {(t)}} S _ {M D} ^ {(t)} (\boldsymbol {x}) \tag {10}
$$

Remarks. We exploit EBMs, which are known for their flexibility but suffering from intractability. However, we exploit EBMs’ flexibility to derive principled task-id prediction score following Theorem 4.1 and eq. (8), while keeping the tractability via approximation using OOD scores (MD, KNN, MLS) in practice. This makes our proposed TPL maintain both theoretical and empirical soundness.

# 4.3 CONVERTING TASK-ID PREDICTION SCORES TO PROBABILITIES

Although theoretically principled as shown in Sec. 4.1, our final task-id prediction score is still an unnormalized energy function. We convert the task-id prediction scores for all tasks (i.e., $\{ S _ { T P L } ^ { ( t ) } ( \pmb { x } ) \} _ { t = 1 } ^ { T } )$ to normalized probabilities via softmax:

$$
\mathbf {P} (t | \boldsymbol {x}) = \operatorname{softmax} \left(\left[ S _ {T P L} ^ {(1)} (\boldsymbol {x}), S _ {T P L} ^ {(2)} (\boldsymbol {x}), \dots , S _ {T P L} ^ {(T)} (\boldsymbol {x}) \right] / \gamma\right) _ {t}, \tag {11}
$$

where $\gamma$ is a temperature parameter. To encourage confident task-id prediction, we set a low temperature $\gamma = 0 . 0 5$ to produce a low entropy task-id preidction distribution for all our experiments.

# 5 EXPERIMENTS

# 5.1 EXPERIMENTAL SETUP

CIL Baselines. We use 17 baselines, including 11 replay methods: iCaRL (Rebuffi et al., 2017), A-GEM (Chaudhry et al., 2018), EEIL (Castro et al., 2018), GD (Lee et al., 2019), DER++ (Buzzega et al., 2020), HAL (Chaudhry et al., 2021), DER (Yan et al., 2021), FOSTER (Wang et al., 2022b), AFC (Kang et al., 2022), BEEF (Wang et al., 2022a), MORE (Kim et al., 2022a), ROW (Kim et al., 2023), and 6 non-replay methods: HAT (Serra et al., 2018), ADAM (Zhou et al., 2023), OWM (Zeng et al., 2019), PASS (Zhu et al., 2021), SLDA (Hayes & Kanan, 2020), and L2P (Wang et al., 2022e).6 We follow (Kim et al., 2022b) to adapt HAT (which is a TIL method) for CIL and call it $\mathrm { H A T } _ { C I L }$ . Implementation details, network size and running time are given in Appendix I.1.

Datasets. To form a sequence of tasks in CIL experiments, we follow the common CIL setting. We split CIFAR-10 into 5 tasks (2 classes per task) (C10-5T). For CIFAR-100, we conduct two experiments: 10 tasks (10 classes per task) (C100-10T) and 20 tasks (5 classes per task) (C100-20T). For TinyImageNet, we split 200 classes into 5 tasks (40 classes per task) (T-5T) and 10 tasks (20 classes per task) (T-10T). We set the replay buffer size for CIFAR-10 as 200 samples, and CIFAR-100 and TinyImageNet as 2000 samples following Kim et al. (2023). Following the random class order protocol in Rebuffi et al. (2017), we randomly generate five different class orders for each experiment and report the averaged metrics over the 5 random orders. For a fair comparison, the class orderings are kept the same for all systems. Results on a larger dataset are given in Appendix D.1.

Backbone Architectures. We conducted two sets of experiments, one using a pre-trained model and one without using a pre-trained model. Here we focus on using a pre-trained model as that is getting more popular. Following the TIL+OOD works (Kim et al., 2022a; 2023), TPL uses the same DeiT-S/16 model (Touvron et al., 2021) pre-trained using 611 classes of ImageNet after removing 389 classes that are similar or identical to the classes of the experiment data CIFAR and TinyImageNet to prevent information leak (Kim et al., 2022a; 2023). To leverage the pre-trained model while adapting to new knowledge, we insert an adapter module (Houlsby et al., 2019) at each transformer layer except SLDA and L2P.7 The adapter modules, classifiers, and layer norms are trained using HAT while the transformer parameters are fixed to prevent CF. The hidden dimension of adapters is 64 for CIFAR-10, and 128 for CIFAR-100 and TinyImageNet. For completeness, we also report the results of TPL using DeiT-S/16 Pre-trained with the Full ImageNet (called TPLPFI) in the pink rows of Table 1. The results without using a pre-trained model are given in Appendix D.2.

Table 1: CIL ACC (%). “-XT": X number of tasks. The best result in each column is highlighted in bold. The baselines are divided into two groups via the dashed line. The first group contains non-replay methods, and the second group contains replay-based methods. Non-CL (non-continual learning) denotes pooling all tasks together to learn all classes as one task, which gives the performance upper bound for CIL. AIA is the average incremental ACC (%). Last is the ACC after learning the final task. See forgetting rate results in Appendix C.2. The pink rows also show the results of $_ { \mathrm { N o n - C L _ { P F I } } }$ and $\mathrm { T P L } _ { \mathrm { P F I } }$ , which use DeiT Pre-trained with Full ImageNet. 

<table><tr><td rowspan="2"></td><td colspan="2">C10-5T</td><td colspan="2">C100-10T</td><td colspan="2">C100-20T</td><td colspan="2">T-5T</td><td colspan="2">T-10T</td><td colspan="2">Average</td></tr><tr><td>Last</td><td>AIA</td><td>Last</td><td>AIA</td><td>Last</td><td>AIA</td><td>Last</td><td>AIA</td><td>Last</td><td>AIA</td><td>Last</td><td>AIA</td></tr><tr><td>Non-CL</td><td> $95.79^{\pm 0.15}$ </td><td> $97.01^{\pm 0.14}$ </td><td> $82.76^{\pm 0.22}$ </td><td> $87.20^{\pm 0.29}$ </td><td> $82.76^{\pm 0.22}$ </td><td> $87.53^{\pm 0.31}$ </td><td> $72.52^{\pm 0.41}$ </td><td> $77.03^{\pm 0.47}$ </td><td> $72.52^{\pm 0.41}$ </td><td> $77.03^{\pm 0.41}$ </td><td>81.27</td><td>85.16</td></tr><tr><td>OWM</td><td> $41.69^{\pm 6.34}$ </td><td> $56.00^{\pm 3.46}$ </td><td> $21.39^{\pm 3.18}$ </td><td> $40.10^{\pm 1.86}$ </td><td> $16.98^{\pm 4.44}$ </td><td> $32.58^{\pm 1.58}$ </td><td> $24.55^{\pm 2.48}$ </td><td> $45.18^{\pm 0.33}$ </td><td> $17.52^{\pm 3.45}$ </td><td> $35.75^{\pm 2.21}$ </td><td>24.43</td><td>41.92</td></tr><tr><td>ADAM</td><td> $83.92^{\pm 0.51}$ </td><td> $90.33^{\pm 0.42}$ </td><td> $61.21^{\pm 0.36}$ </td><td> $72.55^{\pm 0.41}$ </td><td> $58.99^{\pm 0.61}$ </td><td> $70.89^{\pm 0.51}$ </td><td> $50.11^{\pm 0.46}$ </td><td> $61.85^{\pm 0.51}$ </td><td> $49.68^{\pm 0.40}$ </td><td> $61.44^{\pm 0.44}$ </td><td>60.78</td><td>71.41</td></tr><tr><td>PASS</td><td> $86.21^{\pm 1.10}$ </td><td> $89.03^{\pm 7.13}$ </td><td> $68.90^{\pm 0.94}$ </td><td> $77.01^{\pm 2.44}$ </td><td> $66.77^{\pm 1.18}$ </td><td> $76.42^{\pm 1.23}$ </td><td> $61.03^{\pm 0.38}$ </td><td> $67.12^{\pm 6.26}$ </td><td> $58.34^{\pm 0.42}$ </td><td> $67.33^{\pm 3.63}$ </td><td>68.25</td><td>75.38</td></tr><tr><td> $HAT_{CIL}$ </td><td> $82.40^{\pm 0.12}$ </td><td> $91.06^{\pm 0.36}$ </td><td> $62.91^{\pm 0.24}$ </td><td> $73.99^{\pm 0.86}$ </td><td> $59.54^{\pm 0.41}$ </td><td> $69.12^{\pm 1.06}$ </td><td> $59.22^{\pm 0.10}$ </td><td> $69.38^{\pm 1.14}$ </td><td> $54.03^{\pm 0.21}$ </td><td> $65.63^{\pm 1.64}$ </td><td>63.62</td><td>73.84</td></tr><tr><td>SLDA</td><td> $88.64^{\pm 0.05}$ </td><td> $93.54^{\pm 0.66}$ </td><td> $67.82^{\pm 0.05}$ </td><td> $77.72^{\pm 0.58}$ </td><td> $67.80^{\pm 0.05}$ </td><td> $78.51^{\pm 0.58}$ </td><td> $57.93^{\pm 0.05}$ </td><td> $66.03^{\pm 1.35}$ </td><td> $57.93^{\pm 0.06}$ </td><td> $67.39^{\pm 1.81}$ </td><td>68.02</td><td>76.64</td></tr><tr><td>L2P</td><td> $73.59^{\pm 4.15}$ </td><td> $84.60^{\pm 2.28}$ </td><td> $61.72^{\pm 0.81}$ </td><td> $72.88^{\pm 1.18}$ </td><td> $53.84^{\pm 1.59}$ </td><td> $66.52^{\pm 1.61}$ </td><td> $59.12^{\pm 0.96}$ </td><td> $67.81^{\pm 1.25}$ </td><td> $54.09^{\pm 1.14}$ </td><td> $64.59^{\pm 1.59}$ </td><td>60.47</td><td>71.28</td></tr><tr><td>iCaRL</td><td> $87.55^{\pm 0.99}$ </td><td> $89.74^{\pm 6.63}$ </td><td> $68.90^{\pm 0.47}$ </td><td> $76.50^{\pm 3.56}$ </td><td> $69.15^{\pm 0.99}$ </td><td> $77.06^{\pm 2.36}$ </td><td> $53.13^{\pm 1.04}$ </td><td> $61.36^{\pm 6.21}$ </td><td> $51.88^{\pm 2.36}$ </td><td> $63.56^{\pm 3.08}$ </td><td>66.12</td><td>73.64</td></tr><tr><td>A-GEM</td><td> $56.33^{\pm 7.77}$ </td><td> $68.19^{\pm 3.24}$ </td><td> $25.21^{\pm 4.00}$ </td><td> $43.83^{\pm 0.69}$ </td><td> $21.99^{\pm 4.01}$ </td><td> $35.97^{\pm 1.15}$ </td><td> $30.53^{\pm 3.99}$ </td><td> $49.26^{\pm 0.64}$ </td><td> $21.90^{\pm 5.52}$ </td><td> $39.58^{\pm 3.32}$ </td><td>31.19</td><td>47.37</td></tr><tr><td>EEIL</td><td> $82.34^{\pm 3.13}$ </td><td> $90.50^{\pm 0.72}$ </td><td> $68.08^{\pm 0.51}$ </td><td> $81.10^{\pm 0.37}$ </td><td> $63.79^{\pm 0.66}$ </td><td> $79.54^{\pm 0.69}$ </td><td> $53.34^{\pm 0.54}$ </td><td> $66.63^{\pm 0.40}$ </td><td> $50.38^{\pm 0.97}$ </td><td> $66.54^{\pm 0.61}$ </td><td>63.59</td><td>76.86</td></tr><tr><td>GD</td><td> $89.16^{\pm 0.53}$ </td><td> $94.22^{\pm 0.75}$ </td><td> $64.36^{\pm 0.57}$ </td><td> $80.51^{\pm 0.57}$ </td><td> $60.10^{\pm 0.74}$ </td><td> $78.43^{\pm 0.76}$ </td><td> $53.01^{\pm 0.97}$ </td><td> $67.51^{\pm 0.38}$ </td><td> $42.48^{\pm 2.53}$ </td><td> $63.91^{\pm 0.40}$ </td><td>61.82</td><td>76.92</td></tr><tr><td>DER++</td><td> $84.63^{\pm 2.91}$ </td><td> $89.01^{\pm 6.29}$ </td><td> $69.73^{\pm 0.99}$ </td><td> $80.64^{\pm 2.74}$ </td><td> $70.03^{\pm 1.46}$ </td><td> $81.72^{\pm 1.76}$ </td><td> $55.84^{\pm 2.21}$ </td><td> $66.55^{\pm 3.73}$ </td><td> $54.20^{\pm 3.28}$ </td><td> $67.14^{\pm 1.40}$ </td><td>66.89</td><td>77.01</td></tr><tr><td>HAL</td><td> $84.38^{\pm 2.70}$ </td><td> $87.00^{\pm 7.27}$ </td><td> $67.17^{\pm 1.50}$ </td><td> $77.42^{\pm 2.73}$ </td><td> $67.37^{\pm 1.45}$ </td><td> $77.85^{\pm 1.71}$ </td><td> $52.80^{\pm 2.37}$ </td><td> $65.31^{\pm 3.68}$ </td><td> $55.25^{\pm 3.60}$ </td><td> $64.48^{\pm 1.45}$ </td><td>65.39</td><td>74.41</td></tr><tr><td>DER</td><td> $86.79^{\pm 1.20}$ </td><td> $92.83^{\pm 1.10}$ </td><td> $73.30^{\pm 0.58}$ </td><td> $82.89^{\pm 0.45}$ </td><td> $72.00^{\pm 0.57}$ </td><td> $82.79^{\pm 0.76}$ </td><td> $59.57^{\pm 0.89}$ </td><td> $70.32^{\pm 0.57}$ </td><td> $57.18^{\pm 1.40}$ </td><td> $70.21^{\pm 0.86}$ </td><td>69.77</td><td>79.81</td></tr><tr><td>FOSTER</td><td> $86.09^{\pm 0.38}$ </td><td> $91.54^{\pm 0.65}$ </td><td> $71.69^{\pm 0.24}$ </td><td> $81.16^{\pm 0.39}$ </td><td> $72.91^{\pm 0.45}$ </td><td> $83.02^{\pm 0.86}$ </td><td> $54.44^{\pm 0.28}$ </td><td> $69.95^{\pm 0.28}$ </td><td> $55.70^{\pm 0.40}$ </td><td> $70.00^{\pm 0.26}$ </td><td>68.17</td><td>79.13</td></tr><tr><td>BEEF</td><td> $87.10^{\pm 1.38}$ </td><td> $93.10^{\pm 1.21}$ </td><td> $72.09^{\pm 0.33}$ </td><td> $81.91^{\pm 0.58}$ </td><td> $71.88^{\pm 0.54}$ </td><td> $81.45^{\pm 0.74}$ </td><td> $61.41^{\pm 0.38}$ </td><td> $71.21^{\pm 0.57}$ </td><td> $58.16^{\pm 0.60}$ </td><td> $71.16^{\pm 0.82}$ </td><td>70.13</td><td>79.77</td></tr><tr><td>MORE</td><td> $89.16^{\pm 0.96}$ </td><td> $94.23^{\pm 0.82}$ </td><td> $70.23^{\pm 2.27}$ </td><td> $81.24^{\pm 1.24}$ </td><td> $70.53^{\pm 1.09}$ </td><td> $81.59^{\pm 0.98}$ </td><td> $64.97^{\pm 1.28}$ </td><td> $74.03^{\pm 1.61}$ </td><td> $63.06^{\pm 1.26}$ </td><td> $72.74^{\pm 1.04}$ </td><td>71.59</td><td>80.77</td></tr><tr><td>ROW</td><td> $90.97^{\pm 0.19}$ </td><td> $94.45^{\pm 0.21}$ </td><td> $74.72^{\pm 0.48}$ </td><td> $82.87^{\pm 0.41}$ </td><td> $74.60^{\pm 0.12}$ </td><td> $83.12^{\pm 0.23}$ </td><td> $65.11^{\pm 1.97}$ </td><td> $74.16^{\pm 1.34}$ </td><td> $63.21^{\pm 2.53}$ </td><td> $72.91^{\pm 2.12}$ </td><td>73.72</td><td>81.50</td></tr><tr><td>TPL (ours)</td><td> $92.33^{\pm 0.32}$ </td><td> $95.11^{\pm 0.44}$ </td><td> $76.53^{\pm 0.27}$ </td><td> $84.10^{\pm 0.34}$ </td><td> $76.34^{\pm 0.38}$ </td><td> $84.46^{\pm 0.28}$ </td><td> $68.64^{\pm 0.44}$ </td><td> $76.77^{\pm 0.23}$ </td><td> $67.20^{\pm 0.51}$ </td><td> $75.72^{\pm 0.37}$ </td><td>76.21</td><td>83.23</td></tr><tr><td>Non- $CL_{PFI}$ </td><td> $96.90^{\pm 0.07}$ </td><td> $97.96^{\pm 0.05}$ </td><td> $83.61^{\pm 0.33}$ </td><td> $89.72^{\pm 0.10}$ </td><td> $83.61^{\pm 0.33}$ </td><td> $88.89^{\pm 0.06}$ </td><td> $85.55^{\pm 0.07}$ </td><td> $88.26^{\pm 0.08}$ </td><td> $85.71^{\pm 0.14}$ </td><td> $88.66^{\pm 0.01}$ </td><td>87.08</td><td>90.70</td></tr><tr><td> $TPL_{PFI}$ </td><td> $94.86^{\pm 0.02}$ </td><td> $96.89^{\pm 0.02}$ </td><td> $82.43^{\pm 0.12}$ </td><td> $88.28^{\pm 0.17}$ </td><td> $80.86^{\pm 0.07}$ </td><td> $87.32^{\pm 0.07}$ </td><td> $84.06^{\pm 0.11}$ </td><td> $87.19^{\pm 0.11}$ </td><td> $83.87^{\pm 0.07}$ </td><td> $87.40^{\pm 0.16}$ </td><td>85.22</td><td>89.42</td></tr></table>

Evaluation Metrics. We use threepopular metrics: (1) accuracy after learning the final task (Last in Table 1), (2) average incremental accuracy (AIA in Table 1), and (3) forgetting rate (see Table 6 in Appendix C.2, where we also discuss why the current forgetting rate formula is not appropriate for CIL, but only for TIL. The definitions of all these metrics are given in Appendix C.

# 5.2 RESULTS AND COMPARISONS

Table 1 shows the CIL accuracy (ACC) results. The last two columns give the row averages. Our TPL performs the best in both average incremental ACC (AIA) and ACC after the last task (Last). Based on AIA, TPL’s forgetting (CF) is almost negligible. When the full ImageNet data is used in pre-training (pink rows), TPLPFI has almost no forgetting in both AIA and Last ACC.

Comparison with CIL baselines with pre-training. The best-performing replay-based baseline is ROW, which also follows the TIL+OOD paradigm (Kim et al., 2022b). Since its OOD score is inferior to our principled $S _ { L R } ( { \pmb x } )$ , ROW is greatly outperformed by TPL. The ACC gap between our TPL and the best exemplar-free method PASS is even greater, 68.25% (PASS) vs. 76.21% (TPL) in Last ACC. TPL also markedly outperforms the strong network expansion methods DER, FOSTER, and BEEF.

Without pre-training. The accuracy results after learning the final task without pre-training are given in Table 8 of Appendix D.2. We provide a summary in Table 2 here. As L2P, SLDA, and ADAM are designed specifically for pre-trained backbones, they cannot be adapted to the nonpre-training setting and thus are excluded here. Similar to the observation in Table 1, our TPL achieves the overall best results (with ACC of 57.5%), while DER ranks the second (54.2%).

Table 2: CIL ACC (%) after learning the final task without pre-training (average over the five datasets used in Table 1). The detailed results are shown in Table 8 of Appendix D.2. 

<table><tr><td>OWM</td><td>PASS</td><td>EEIL</td><td>GD</td><td>HAL</td><td>A-GEM</td><td>HAT</td><td>iCaRL</td></tr><tr><td>24.7</td><td>30.6</td><td>46.3</td><td>47.1</td><td>46.4</td><td>45.8</td><td>39.5</td><td>45.6</td></tr><tr><td>DER++</td><td>DER</td><td>FOSTER</td><td>BEEF</td><td>MORE</td><td>ROW</td><td>TPL (ours)</td><td></td></tr><tr><td>46.5</td><td>54.2</td><td>52.2</td><td>53.4</td><td>51.2</td><td>53.1</td><td>57.5</td><td></td></tr></table>

# 5.3 ABLATION STUDY

Performance gain. Figure 2(a) shows the performance gain achieved by adding each proposed technique. Starting from vanilla $\mathrm { H A T } _ { C I L }$ with an average Last ACC of 63.41% over all datasets, the proposed likelihood ratio LR score (HAT+LR) boosts the average Last ACC to 71.25%. Utilizing the OOD detection method MLS (HAT+MLS) only improves the ACC to 68.69%. The final composition of LR and MLS boosted the performance to 76.21%.

![](images/113670655875e8df449ea076147a1732e3872f4b79799268a83b58e18a36d46c.jpg)

<details>
<summary>bar</summary>

| Dataset  | HATcIL | HAT+MLS | HAT+LR | TPL   |
|----------|--------|---------|--------|-------|
| C10-5T   | 0.82   | 0.87    | 0.89   | 0.93  |
| C100-10T | 0.64   | 0.66    | 0.72   | 0.77  |
| C100-20T | 0.60   | 0.64    | 0.70   | 0.77  |
| T-5T     | 0.58   | 0.64    | 0.66   | 0.70  |
| T-10T    | 0.60   | 0.62    | 0.64   | 0.68  |
</details>

(a) Performance Gain

![](images/15ee8dac5cc5506caa8719ec25ffa750fbe9d45179b10754d7db93d02107c379.jpg)

<details>
<summary>heatmap</summary>

| | Constant | MD Etc | KNN |
|---|---|---|---|
| Et KNN Constant | 69.22 | 70.79 | 71.93 |
| Et KNN Constant | 70.73 | 72.22 | 74.01 |
| Et KNN Constant | 71.70 | 73.98 | 76.21 |
</details>

(b) Different $E _ { t }$ v.s. $E _ { t ^ { c } }$

![](images/37c04f7c48d678f3ac803b96d718bb08fb9254a8157e77b7a4b53f279e2fa5e6.jpg)

<details>
<summary>bar</summary>

| Dataset   | TPL (MSP) | TPL (EBO) | TPL (MLS) |
| --------- | --------- | --------- | --------- |
| C10-5T    | 0.87      | 0.91      | 0.92      |
| C100-10T  | 0.73      | 0.76      | 0.77      |
| C100-20T  | 0.72      | 0.76      | 0.76      |
| T-5T      | 0.63      | 0.68      | 0.69      |
| T-10T     | 0.62      | 0.66      | 0.67      |
</details>

(c) Different $E _ { l o g i t }$   
Figure 2: Ablation Studies. Fig (a) illustrates the achieved ACC gain for each of the designed techniques on the five datasets; Fig (b) displays the average ACC results obtained from different choices of $E _ { t }$ and $E _ { t ^ { c } }$ for eq. (7); Fig (c) showcases the results for various selections of $E _ { l o g i t }$ for TPL in eq. (9).

Different $E _ { t }$ v.s. $E _ { t ^ { c } }$ . Recall that the key insight behind the LR score lies in the estimation of likelihood ratio. Figure 2(b) presents the average Last ACC results across 5 datasets, employing various approaches to estimate $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ . In this context, the term Constant refers to the use of a uniform distribution as the distribution of $\mathcal { P } _ { t ^ { c } }$ , where the energy function is a constant mapping. Our TPL approach is equivalent to employing $( E _ { t } = \mathrm { M D } , E _ { t ^ { c } } = \mathrm { K N N } )$ . The results reveal the following: (1) The incorporation of the $\mathcal { P } _ { t ^ { c } }$ distribution estimation is beneficial compared to assuming a uniform distribution. (2) As $\mathcal { P } _ { t ^ { c } }$ can only be estimated using the replay data, the high-performing KNN method outperforms MD. However, since MD can estimate $\mathcal { P } _ { t }$ without task t’s training data during the test phase, it proves to be more effective than KNN when serving as $E _ { t }$ .

Different logit-based scores. Although $S _ { M L S } ( { \pmb x } )$ is used as the logit-based score in Section 4.2.2, alternative logit-based scores can also be considered. In this study, we conduct experiments using 3 popular logit-based scores MSP (Hendrycks & Gimpel, 2016), EBO (Liu et al., 2020a), and MLS (their definitions are given in Appendix F.2). The results presented in Figure 2(c) indicate that EBO and MLS yield comparable results, with average Last ACC of 75.76%, and 76.21% respectively, while MSP has inferior performance with average Last ACC of 71.32%.

Smaller replay buffer sizes. The accuracy after learning the final task with smaller replay buffer sizes are given in Table 9 of Appendix D.3. We provide a summary as Table 3, which shows that when using a smaller replay buffer, the performance drop of TPL is small. The goal of using the replay data in TPL is to compute the likelihood ratio (LR) score, while traditional replay methods focus on preventing forgetting (CF). Note that CF is already addressed by the TIL method HAT in our case. Thus our method TPL is robust with fewer replay samples.

Table 3: ACC (%) after learning the final task (Last) with smaller replay buffer sizes (average over the five datasets in Table 1). The detailed results are shown in Table 9 of Appendix D.3. The replay buffer size is set as 100 for CIFAR-10, and 1000 for CIFAR-100 and TinyImageNet. 

<table><tr><td>iCaRL</td><td>A-GEM</td><td>EEIL</td><td>GD</td><td>DER++</td><td>HAL</td></tr><tr><td>63.60</td><td>31.15</td><td>58.24</td><td>54.39</td><td>62.16</td><td>60.21</td></tr><tr><td>DER</td><td>FOSTER</td><td>BEEF</td><td>MORE</td><td>ROW</td><td>TPL</td></tr><tr><td>68.32</td><td>66.86</td><td>68.94</td><td>71.44</td><td>72.70</td><td>75.56</td></tr></table>

More OOD methods. To understand the effect of OOD detection on CIL, we applied 20 OOD detection methods to CIL and drew some interesting conclusions (see Appendix A). (1) There exists a linear relationship between OOD detection AUC and CIL ACC performances. (2) Different OOD detection methods result in similar TIL (task-incremental learning) ACC when applying HAT.

More pre-trained models (visual encoders). We also study TPL with different pre-trained models in Appendix D.5 (MAE, Dino, ViT and DeiT of different sizes). We found the pre-trained models based on supervised learning outperform self-supervised models in both CIL and TIL.

# 6 CONCLUSION

In this paper, we developed a novel approach for class incremental learning (CIL) via task-id prediction based on likelihood ratio. Recent studies (Kim et al., 2022a;b; 2023) suggested that OOD detection methods can be applied to perform task-id prediction in CIL and thus achieve the state-of-the-art performance. However, we argue that traditional OOD detection is not optimal for CIL as additional information in CIL can be leveraged to design a better and principled method for task-id prediction. Our experimental results show that our TPL outperforms strong baselines and has almost negligible catastrophic forgetting. Limitations of our approach are discussed in Appendix J.

# ACKNOWLEDGEMENTS

We sincerely thank Baizhou Huang of Peking University, Shanda Li of Carnegie Mellon University, and the anonymous reviewers of ICLR 2024 for providing valuable suggestions on this work.

# ETHICS STATEMENT

Since this research involves only classification learning using existing datasets downloaded from the public domain and our algorithms are not for any specific application but for solving the general problem of continual learning, we do not feel there are any possible ethical issues in this research.

# REPRODUCIBILITY STATEMENT

The source code of TPL has been public at https://github.com/linhaowei1/TPL. The proofs of Theorems 4.1 and 4.2 are provided in Appendix E. The training details and dataset details are given in Sec. 5.1 and Appendix I.

# REFERENCES

Davide Abati, Jakub Tomczak, Tijmen Blankevoort, Simone Calderara, Rita Cucchiara, and Babak Ehteshami Bejnordi. Conditional channel gated networks for task-aware continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3931–3940, 2020. 2, 3   
Hongjoon Ahn, Jihwan Kwak, Subin Lim, Hyeonsu Bang, Hyojun Kim, and Taesup Moon. Ssil: Separated softmax for incremental learning. In Proceedings of the IEEE/CVF International conference on computer vision, pp. 844–853, 2021. 3   
Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a network of experts. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3366–3375, 2017. 3   
Jihwan Bang, Hyunseo Koh, Seulki Park, Hwanjun Song, Jung-Woo Ha, and Jonghyun Choi. Online continual learning on a contaminated data stream with blurry task boundaries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9275–9284, 2022. 40   
Abhijit Bendale and Terrance E Boult. Towards open set deep networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1563–1572, 2016. 17   
Prashant Bhat, Bahram Zonooz, and E. Arani. Consistency is the key to further mitigating catastrophic forgetting in continual learning. In CoLLAs, 2022. URL https://api.semanticscholar. org/CorpusID:250425816. 40   
Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020. 7   
Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision (ICCV), 2021. 27   
Francisco M Castro, Manuel J Marín-Jiménez, Nicolás Guil, Cordelia Schmid, and Karteek Alahari. End-to-end incremental learning. In Proceedings of the European conference on computer vision (ECCV), pp. 233–248, 2018. 7   
Arslan Chaudhry, Marc’Aurelio Ranzato, Marcus Rohrbach, and Mohamed Elhoseiny. Efficient lifelong learning with a-gem. arXiv preprint arXiv:1812.00420, 2018. 7   
Arslan Chaudhry, Albert Gordo, Puneet Dokania, Philip Torr, and David Lopez-Paz. Using hindsight to anchor past knowledge in continual learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 6993–7001, 2021. 7

Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah Parisot, Xu Jia, Aleš Leonardis, Gregory Slabaugh, and Tinne Tuytelaars. A continual learning survey: Defying forgetting in classification tasks. IEEE transactions on pattern analysis and machine intelligence, 44(7):3366–3385, 2021. 1, 3   
Xuefeng Du, Zhaoning Wang, Mu Cai, and Yixuan Li. Vos: Learning what you don’t know by virtual outlier synthesis. In International Conference on Learning Representations, 2021. 17   
Yilun Du, Shuang Li, and Igor Mordatch. Compositional visual generation with energy based models. Advances in Neural Information Processing Systems, 33:6637–6647, 2020. 6   
Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016. 17   
Yiduo Guo, Bing Liu, and Dongyan Zhao. Online continual learning through mutual information maximization. In International Conference on Machine Learning, pp. 8109–8126. PMLR, 2022. 40   
Yiduo Guo, Bing Liu, and Dongyan Zhao. Dealing with cross-task class discrimination in online continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11878–11887, 2023. 3   
Raia Hadsell, Dushyant Rao, Andrei A Rusu, and Razvan Pascanu. Embracing change: Continual learning in deep neural networks. Trends in cognitive sciences, 24(12):1028–1040, 2020. 3   
Tyler L Hayes and Christopher Kanan. Lifelong machine learning with deep streaming linear discriminant analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 220–221, 2020. 7   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016. 18, 24   
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross B. Girshick. Masked autoencoders are scalable vision learners. CoRR, abs/2111.06377, 2021. URL https://arxiv. org/abs/2111.06377. 27   
Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. arXiv preprint arXiv:1610.02136, 2016. 2, 9, 17   
Dan Hendrycks, Mantas Mazeika, and Thomas Dietterich. Deep anomaly detection with outlier exposure. arXiv preprint arXiv:1812.04606, 2018. 17   
Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt, and Dawn Song. Scaling out-of-distribution detection for real-world settings. arXiv preprint arXiv:1911.11132, 2019. 17   
Christian Henning, Maria Cervera, Francesco D’Angelo, Johannes Von Oswald, Regina Traber, Benjamin Ehret, Seijin Kobayashi, Benjamin F Grewe, and João Sacramento. Posterior meta-replay for continual learning. Advances in Neural Information Processing Systems, 34:14135–14149, 2021. 3   
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International Conference on Machine Learning, pp. 2790–2799. PMLR, 2019. 4, 7, 18   
Rui Huang, Andrew Geng, and Yixuan Li. On the importance of gradients for detecting distributional shifts in the wild. Advances in Neural Information Processing Systems, 34:677–689, 2021. 17   
Kishaan Jeeveswaran, Prashant Bhat, Bahram Zonooz, and E. Arani. Birt: Bio-inspired replay in vision transformers for continual learning. ArXiv, abs/2305.04769, 2023. URL https: //api.semanticscholar.org/CorpusID:258557568. 40

Minsoo Kang, Jaeyoo Park, and Bohyung Han. Class-incremental learning by knowledge distillation with adaptive feature consolidation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16071–16080, 2022. 7   
Zixuan Ke and Bing Liu. Continual learning of natural language processing tasks: A survey. arXiv preprint arXiv:2211.12701, 2022. 1, 3   
Zixuan Ke, Bing Liu, Nianzu Ma, Hu Xu, and Lei Shu. Achieving forgetting prevention and knowledge transfer in continual learning. Advances in Neural Information Processing Systems, 34: 22443–22456, 2021a. 3   
Zixuan Ke, Hu Xu, and Bing Liu. Adapting bert for continual learning of a sequence of aspect sentiment classification tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4746–4755, 2021b. 3   
Zixuan Ke, Yijia Shao, Haowei Lin, Tatsuya Konishi, Gyuhak Kim, and Bing Liu. Continual pre-training of language models. In The Eleventh International Conference on Learning Representations (ICLR-2023), 2023. 3   
Ronald Kemker and Christopher Kanan. Fearnet: Brain-inspired model for incremental learning. arXiv preprint arXiv:1711.10563, 2017. 1, 3   
Gyuhak Kim, Bing Liu, and Zixuan Ke. A multi-head model for continual learning via out-ofdistribution replay. In Conference on Lifelong Learning Agents, pp. 548–563. PMLR, 2022a. 3, 7, 9, 17, 20, 21, 38   
Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, Zixuan Ke, and Bing Liu. A theoretical study on solving continual learning. In Advances in Neural Information Processing Systems, 2022b. 1, 2, 3, 7, 8, 9, 17, 20, 24, 35   
Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, and Bing Liu. Learnability and algorithm for continual learning. ICML-2023, 2023. 2, 3, 7, 9, 17, 38   
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114 (13):3521–3526, 2017. 3   
Alex Krizhevsky and Geoff Hinton. Convolutional deep belief networks on cifar-10. Unpublished manuscript, 40(7):1–9, 2010. 38   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. 38   
Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7:7, 2015. 38   
Kibok Lee, Kimin Lee, Jinwoo Shin, and Honglak Lee. Overcoming catastrophic forgetting with unlabeled data in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 312–321, 2019. 7   
Kimin Lee, Honglak Lee, Kibok Lee, and Jinwoo Shin. Training confidence-calibrated classifiers for detecting out-of-distribution samples. In International Conference on Learning Representations, 2018a. 2   
Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. Advances in neural information processing systems, 31, 2018b. 2, 5, 17, 31   
Guodun Li, Yuchen Zhai, Qianglong Chen, Xing Gao, Ji Zhang, and Yin Zhang. Continual few-shot intent detection. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 333–343, 2022. 3

Shiyu Liang, Yixuan Li, and Rayadurgam Srikant. Enhancing the reliability of out-of-distribution image detection in neural networks. arXiv preprint arXiv:1706.02690, 2017. 2, 17   
Haowei Lin and Yuntian Gu. Flats: Principled out-of-distribution detection with feature-based likelihood ratio score. ArXiv, abs/2310.05083, 2023. URL https://api.semanticscholar. org/CorpusID:263831173. 5   
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. Advances in Neural Information Processing Systems, 33:21464–21475, 2020a. 2, 9, 17   
Y. Liu, A. A. Liu, Y. Su, B. Schiele, and Q. Sun. Mnemonics training: Multi-class incremental learning without forgetting. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020b. 21   
David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30, 2017. 1, 3, 21   
Arun Mallya and Svetlana Lazebnik. Packnet: Adding multiple tasks to a single network by iterative pruning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 7765–7773, 2018. 35   
Marc Masana, Tinne Tuytelaars, and Joost Van de Weijer. Ternary feature masks: zero-forgetting for task-incremental learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3570–3579, 2021. 35   
Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, volume 24, pp. 109–165. Elsevier, 1989.   
Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don’t know? arXiv preprint arXiv:1810.09136, 2018. 5   
Ibrahima Ndiour, Nilesh Ahuja, and Omesh Tickoo. Out-of-distribution detection with subspace techniques and probabilistic modeling of features. arXiv preprint arXiv:2012.04250, 2020. 17   
Jerzy Neyman and Egon Sharpe Pearson. Ix. on the problem of the most efficient tests of statistical hypotheses. Philosophical Transactions of the Royal Society of London. Series A, Containing Papers of a Mathematical or Physical Character, 231(694-706):289–337, 1933. 5, 28   
Jathushan Rajasegaran, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Mubarak Shah. itaml: An incremental task-agnostic meta-learning approach. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13588–13597, 2020. 2, 3   
Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 2001–2010, 2017. 1, 7   
Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115 (3):211–252, 2015. doi: 10.1007/s11263-015-0816-y. 23   
Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins, Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Razvan Pascanu, and Raia Hadsell. Progressive neural networks. arXiv preprint arXiv:1606.04671, 2016. 35   
Joan Serra, Didac Suris, Marius Miron, and Alexandros Karatzoglou. Overcoming catastrophic forgetting with hard attention to the task. In International Conference on Machine Learning, pp. 4548–4557. PMLR, 2018. 2, 3, 7, 33, 38   
Yijia Shao, Yiduo Guo, Dongyan Zhao, and Bing Liu. Class-incremental learning based on label generation. arXiv preprint arXiv:2306.12619, 2023. 3

Yiyou Sun, Chuan Guo, and Yixuan Li. React: Out-of-distribution detection with rectified activations. Advances in Neural Information Processing Systems, 34:144–157, 2021. 2, 17   
Yiyou Sun, Yifei Ming, Xiaojin Zhu, and Yixuan Li. Out-of-distribution detection with deep nearest neighbors. arXiv preprint arXiv:2204.06507, 2022. 2, 5, 17   
Jihoon Tack, Sangwoo Mo, Jongheon Jeong, and Jinwoo Shin. Csi: Novelty detection via contrastive learning on distributionally shifted instances. Advances in neural information processing systems, 33:11839–11852, 2020. 2   
George R Terrell and David W Scott. Variable kernel density estimation. The Annals of Statistics, pp. 1236–1265, 1992. 36   
Sunil Thulasidasan, Gopinath Chennupati, Jeff A Bilmes, Tanmoy Bhattacharya, and Sarah Michalak. On mixup training: Improved calibration and predictive uncertainty for deep neural networks. Advances in Neural Information Processing Systems, 32, 2019. 17   
Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In International Conference on Machine Learning, pp. 10347–10357. PMLR, 2021. 7, 17   
S Vaze, K Han, A Vedaldi, and A Zisserman. Open-set recognition: A good closed-set classifier is all you need? In International Conference on Learning Representations (ICLR), 2022. 2   
Johannes Von Oswald, Christian Henning, Benjamin F Grewe, and João Sacramento. Continual learning with hypernetworks. arXiv preprint arXiv:1906.00695, 2019. 2, 3   
Fu-Yun Wang, Da-Wei Zhou, Liu Liu, Han-Jia Ye, Yatao Bian, De-Chuan Zhan, and Peilin Zhao. Beef: Bi-compatible class-incremental learning via energy-based expansion and fusion. In The Eleventh International Conference on Learning Representations, 2022a. 3, 7   
Fu-Yun Wang, Da-Wei Zhou, Han-Jia Ye, and De-Chuan Zhan. Foster: Feature boosting and compression for class-incremental learning. In European conference on computer vision, pp. 398–414. Springer, 2022b. 7   
Haoqi Wang, Zhizhong Li, Litong Feng, and Wayne Zhang. Vim: Out-of-distribution with virtuallogit matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4921–4930, 2022c. 6, 17   
Liyuan Wang, Xingxing Zhang, Hang Su, and Jun Zhu. A comprehensive survey of continual learning: Theory, method and application, 2023. 3   
Mengyu Wang, Yijia Shao, Haowei Lin, Wenpeng Hu, and Bing Liu. Cmg: A class-mixed generation approach to out-of-distribution detection. Proceedings of ECML/PKDD-2022, 2022d. 2, 17   
Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 139–149, 2022e. 3, 7   
Hongxin Wei, Renchunzi Xie, Hao Cheng, Lei Feng, Bo An, and Yixuan Li. Mitigating neural network overconfidence with logit normalization. 2022. 17   
Ross Wightman. Pytorch image models. https://github.com/rwightman/ pytorch-image-models, 2019. 27   
Svante Wold, Kim Esbensen, and Paul Geladi. Principal component analysis. Chemometrics and intelligent laboratory systems, 2(1-3):37–52, 1987. 36   
Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu, Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosinski, and Ali Farhadi. Supermasks in superposition. Advances in Neural Information Processing Systems, 33:15173–15184, 2020. 3, 35   
Jinlin Xiang and Eli Shlizerman. Tkil: Tangent kernel optimization for class balanced incremental learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3529–3539, 2023. 3

Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3014–3023, 2021. 3, 7   
Jingkang Yang, Pengyun Wang, Dejian Zou, Zitang Zhou, Kunyuan Ding, Wenxuan Peng, Haoqi Wang, Guangyao Chen, Bo Li, Yiyou Sun, Xuefeng Du, Kaiyang Zhou, Wayne Zhang, Dan Hendrycks, Yixuan Li, and Ziwei Liu. Openood: Benchmarking generalized out-of-distribution detection. 2022. 6, 18   
Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6023–6032, 2019. 17   
Guanxiong Zeng, Yang Chen, Bo Cui, and Shan Yu. Continual learning of context-dependent processing in neural networks. Nature Machine Intelligence, 1(8):364–372, 2019. 3, 7   
Jingyang Zhang, Nathan Inkawhich, Randolph Linderman, Yiran Chen, and Hai Li. Mixture outlier exposure: Towards out-of-distribution detection in fine-grained environments. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5531–5540, 2023. 17   
Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan, and Ziwei Liu. Revisiting class-incremental learning with pre-trained models: Generalizability and adaptivity are all you need. arXiv preprint arXiv:2303.07338, 2023. 7   
Fei Zhu, Xu-Yao Zhang, Chuang Wang, Fei Yin, and Cheng-Lin Liu. Prototype augmentation and self-supervision for incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5871–5880, 2021. 3, 7

# Appendix of TPL

# Table of Contents

A A Comprehensive Study on TIL+OOD based methods 17

A.1 A unified CIL method based on TIL+OOD 17   
A.2 Experimental Setup . . 17   
A.3 Results and Analysis . . . . 18

B Output Calibration 20

C Evaluation Metrics 21

C.1 Definitions of AIA. and Last accuracy 21   
C.2 Rectified Forgetting Rate for CIL 21

D Additional Experimental Results 23

D.1 CIL Experiments on a Larger Dataset 23   
D.2 CIL Experiments without Pre-training . . . 24   
D.3 CIL Experiments on Smaller Replay Buffer Size . . 25   
D.4 Ablation on $\beta _ { 1 }$ and β2 . . . 26   
D.5 Experiments based on More Pre-trained Models . . 27

E Theoretical Justifications 28

E.1 Preliminary 28   
E.2 Proof of Theorem 4.1 28   
E.3 Proof of Theorem 4.2 28   
E.4 Distance-Based OOD Detectors are IND Density Estimators . . . 29

F Additional Details about TPL 31

F.1 Computation of the MD Score . 31   
F.2 Computation of Logit-based Scores 31   
F.3 Pseudo-code 31

G Details of HAT 33

G.1 Training 33   
G.2 Inference . 34   
G.3 Remarks 34

H Visualization of Task Distribution 36

I Implementation Details, Network Size and Running Time 38

I.1 Implementation Details of Baselines 38   
I.2 Hardware and Software 38   
I.3 Computational Budget Analysis . 38

J Limitations 40

# A A COMPREHENSIVE STUDY ON TIL+OOD BASED METHODS

Based on the previous research on CIL that combines TIL method and OOD detection (Kim et al., 2022b;a; 2023), we make a thorough study on this paradigm by benchmarking 20 popular OOD detection methods. This study provides a comprehensive understanding of these methods and draws some interesting conclusions, which we believe are beneficial to the future study of CIL.

# A.1 A UNIFIED CIL METHOD BASED ON TIL+OOD

We first describe how to design a CIL method based on TIL+OOD paradigm. As HAT is already a near-optimal solution for TIL (Kim et al., 2023) which has almost no CF, we choose HAT as the TIL method and only introduce variants in the OOD detection parts.

Categorization of OOD detection methods. Usually, an OOD detection technique consists of two parts: (1) Training a classifier that can better distinguish IND data and OOD data in the feature / logits space; (2) Applying an inference-time OOD score to compute the INDness of the test case x. For training techniques, a group of methods such as LogitNorm Wei et al. (2022), MCDropout Gal & Ghahramani (2016), Mixup Thulasidasan et al. (2019), CutMix Yun et al. (2019) apply data augmentation or confidence regularization to improve OOD detection; another series of methods such as OE Hendrycks et al. (2018), MixOE Zhang et al. (2023), CMG Wang et al. (2022d), VOS Du et al. (2021) use auxiliary OOD data in training. For inference-time techniques, the OOD score can be computed based on feature (the hidden representations) or logits (the unnormalized softmax score). GradNorm Huang et al. (2021), KL-Matching Hendrycks et al. (2019), OpenMax Bendale & Boult (2016), ODIN Liang et al. (2017), MSP Hendrycks & Gimpel (2016), MLS Hendrycks et al. (2019), EBO Liu et al. (2020a), ReAct Sun et al. (2021) are logit-based methods, and Residual Ndiour et al. (2020), KNN Sun et al. (2022), MDS Lee et al. (2018b) are feature-based methods. Recently, VIM (Wang et al., 2022c) is based on both feature and logits. We study the aforementioned 20 OOD detection methods in this section. For simplicity, we use MSP (maximum softmax probability) as the OOD score for the 8 training-time techniques, and adopt standard supervised learning with cross-entropy loss for the inference-time OOD scores.

A unified method for CIL based on TIL+OOD. We introduce a unified CIL method that is based on TIL+OOD paradigm, which can be compatible with any OOD detection techniques. The details are as follows.

We first recall that HAT learns each task by performing two functions jointly in training: (1) learning a model for the task and (2) identifying the neurons that are important for the task and setting masks on them. When a new task is learned, the gradient flow through those masked neurons is blocked in the backward pass, which protects the models of the previous tasks to ensure no forgetting (CF). In the forward pass, no blocking is applied so that the tasks can share a lot of parameters or knowledge. Since (1) can be any supervised learning method, in our case, we replace it with an OOD detection method, which performs both in-distribution (IND) classification and out-of-distribution (OOD) detection.

In testing, we first predict the task-id to which the test instance x belongs, and then perform the IND classification (within task prediction) in the task to obtain the predicted class. Let S(x; t) be the OOD score of x in task t based on task t’s model, the task-id tˆcan be predicted by identifying the task with the highest OOD score:

$$
\hat {t} = \underset {t} {\arg \max} S (\pmb {x}; t)
$$

We use argmax here as OOD score is defined to measure the IND-ness in the literature. Note that OOD score is produced by any OOD detection method.

# A.2 EXPERIMENTAL SETUP

Backbone Architecture. Following the main experiment in our paper, we use DeiT-S/16 (Touvron et al., 2021) that is pre-trained using 611 classes of ImageNet after removing 389 classes that are similar or identical to the classes of the experiment data CIFAR and TinyImageNet to prevent information leak. We insert an adapter module (Houlsby et al., 2019) at each transformer layer. The adapter modules, classifiers and the layer norms are trained using HAT while the transformer parameters are fixed to prevent forgetting in the pre-trained network.

Table 4: Average AUC and ACC results based on pre-trained DeiT on five datasets with five random seeds. Bold and underlined numbers indicate the best and second-best results, respectively. ♡: logit-based OOD scores, ♠: feature-based OOD scores, ♢: training-time techniques. The systems are divided into three categories by dashed lines. The first category includes post-hoc OOD detectors, the second category includes methods that exploit surrogate OOD data, and the last category includes methods that employ special training strategies. 

<table><tr><td rowspan="2">Method</td><td colspan="2">C10-5T</td><td colspan="2">C100-10T</td><td colspan="2">C100-20T</td><td colspan="2">T-5T</td><td colspan="2">T-10T</td><td colspan="2">Average</td></tr><tr><td>OOD</td><td>CIL</td><td>OOD</td><td>CIL</td><td>OOD</td><td>CIL</td><td>OOD</td><td>CIL</td><td>OOD</td><td>CIL</td><td>OOD</td><td>CIL</td></tr><tr><td>GradNorm ♥</td><td> $73.8^{\pm 0.37}$ </td><td> $43.5^{\pm 0.48}$ </td><td> $84.0^{\pm 0.33}$ </td><td> $48.6^{\pm 0.81}$ </td><td> $80.8^{\pm 0.31}$ </td><td> $22.9^{\pm 1.91}$ </td><td> $80.8^{\pm 0.21}$ </td><td> $57.0^{\pm 0.49}$ </td><td> $83.1^{\pm 0.24}$ </td><td> $47.4^{\pm 0.85}$ </td><td>80.5</td><td>43.9</td></tr><tr><td>KL-Matching ♥</td><td> $75.2^{\pm 0.31}$ </td><td> $48.8^{\pm 0.44}$ </td><td> $84.1^{\pm 0.35}$ </td><td> $48.7^{\pm 0.74}$ </td><td> $82.6^{\pm 0.32}$ </td><td> $25.6^{\pm 1.81}$ </td><td> $77.8^{\pm 0.25}$ </td><td> $47.6^{\pm 0.58}$ </td><td> $73.6^{\pm 0.29}$ </td><td> $31.6^{\pm 0.79}$ </td><td>80.7</td><td>40.5</td></tr><tr><td>OpenMax ♥</td><td> $94.1^{\pm 0.14}$ </td><td> $84.1^{\pm 0.29}$ </td><td> $84.7^{\pm 0.26}$ </td><td> $50.9^{\pm 0.62}$ </td><td> $90.6^{\pm 0.24}$ </td><td> $57.9^{\pm 0.47}$ </td><td> $74.1^{\pm 0.19}$ </td><td> $48.1^{\pm 0.24}$ </td><td> $71.7^{\pm 0.09}$ </td><td> $32.0^{\pm 0.33}$ </td><td>83.0</td><td>54.6</td></tr><tr><td>ODIN ♥</td><td> $92.3^{\pm 0.15}$ </td><td> $75.6^{\pm 0.24}$ </td><td> $89.8^{\pm 0.19}$ </td><td> $61.7^{\pm 0.40}$ </td><td> $92.5^{\pm 0.28}$ </td><td> $53.2^{\pm 0.41}$ </td><td> $85.8^{\pm 0.09}$ </td><td> $64.5^{\pm 0.10}$ </td><td> $88.5^{\pm 0.07}$ </td><td> $59.5^{\pm 0.25}$ </td><td>86.4</td><td>62.9</td></tr><tr><td>MSP ♥</td><td> $93.5^{\pm 0.03}$ </td><td> $82.4^{\pm 0.12}$ </td><td> $89.0^{\pm 0.11}$ </td><td> $62.9^{\pm 0.24}$ </td><td> $91.7^{\pm 0.25}$ </td><td> $59.5^{\pm 0.49}$ </td><td> $81.5^{\pm 0.10}$ </td><td> $59.2^{\pm 0.21}$ </td><td> $84.4^{\pm 0.09}$ </td><td> $54.0^{\pm 0.21}$ </td><td>88.0</td><td>63.6</td></tr><tr><td>MLS ♥</td><td> $94.0^{\pm 0.11}$ </td><td> $84.9^{\pm 0.19}$ </td><td> $91.0^{\pm 0.18}$ </td><td> $69.2^{\pm 0.21}$ </td><td> $92.7^{\pm 0.22}$ </td><td> $64.1^{\pm 0.45}$ </td><td> $86.1^{\pm 0.08}$ </td><td> $65.4^{\pm 0.21}$ </td><td> $88.5^{\pm 0.08}$ </td><td> $61.4^{\pm 0.27}$ </td><td>90.4</td><td>69.0</td></tr><tr><td>EBO ♥</td><td> $94.0^{\pm 0.11}$ </td><td> $84.9^{\pm 0.19}$ </td><td> $90.8^{\pm 0.18}$ </td><td> $69.1^{\pm 0.21}$ </td><td> $92.5^{\pm 0.21}$ </td><td> $64.1^{\pm 0.45}$ </td><td> $86.2^{\pm 0.08}$ </td><td> $65.4^{\pm 0.20}$ </td><td> $88.4^{\pm 0.07}$ </td><td> $61.4^{\pm 0.28}$ </td><td>90.4</td><td>69.0</td></tr><tr><td>ReAct ♥</td><td> $94.0^{\pm 0.11}$ </td><td> $84.9^{\pm 0.20}$ </td><td> $90.8^{\pm 0.19}$ </td><td> $69.1^{\pm 0.22}$ </td><td> $92.5^{\pm 0.19}$ </td><td> $64.1^{\pm 0.39}$ </td><td> $86.2^{\pm 0.09}$ </td><td> $65.4^{\pm 0.21}$ </td><td> $88.4^{\pm 0.05}$ </td><td> $61.4^{\pm 0.28}$ </td><td>90.4</td><td>69.0</td></tr><tr><td>KNN ♠</td><td> $92.8^{\pm 0.13}$ </td><td> $76.7^{\pm 0.25}$ </td><td> $85.9^{\pm 0.14}$ </td><td> $61.5^{\pm 0.21}$ </td><td> $90.2^{\pm 0.11}$ </td><td> $54.8^{\pm 0.30}$ </td><td> $74.7^{\pm 0.10}$ </td><td> $49.9^{\pm 0.21}$ </td><td> $79.2^{\pm 0.08}$ </td><td> $44.4^{\pm 0.21}$ </td><td>84.6</td><td>57.5</td></tr><tr><td>Residual ♠</td><td> $92.4^{\pm 0.14}$ </td><td> $83.0^{\pm 0.17}$ </td><td> $85.0^{\pm 0.16}$ </td><td> $64.7^{\pm 0.35}$ </td><td> $89.8^{\pm 0.14}$ </td><td> $61.3^{\pm 0.31}$ </td><td> $78.0^{\pm 0.11}$ </td><td> $53.7^{\pm 0.24}$ </td><td> $81.6^{\pm 0.09}$ </td><td> $51.0^{\pm 0.30}$ </td><td>85.4</td><td>62.8</td></tr><tr><td>MDS ♠</td><td> $92.6^{\pm 0.12}$ </td><td> $85.7^{\pm 0.02}$ </td><td> $86.9^{\pm 0.14}$ </td><td> $69.0^{\pm 0.24}$ </td><td> $91.2^{\pm 0.15}$ </td><td> $65.4^{\pm 0.42}$ </td><td> $82.0^{\pm 0.06}$ </td><td> $60.8^{\pm 0.28}$ </td><td> $84.4^{\pm 0.05}$ </td><td> $56.9^{\pm 0.30}$ </td><td>87.4</td><td>67.6</td></tr><tr><td>VIM ♥♠</td><td> $95.4^{\pm 0.07}$ </td><td> $89.0^{\pm 0.23}$ </td><td> $91.0^{\pm 0.12}$ </td><td> $72.8^{\pm 0.30}$ </td><td> $93.5^{\pm 0.13}$ </td><td> $69.8^{\pm 0.55}$ </td><td> $86.3^{\pm 0.05}$ </td><td> $65.9^{\pm 0.23}$ </td><td> $88.7^{\pm 0.07}$ </td><td> $63.1^{\pm 0.42}$ </td><td>91.0</td><td>72.1</td></tr><tr><td>CMG ◇</td><td> $92.3^{\pm 0.21}$ </td><td> $80.7^{\pm 0.40}$ </td><td> $85.2^{\pm 0.28}$ </td><td> $56.2^{\pm 0.60}$ </td><td> $90.1^{\pm 0.24}$ </td><td> $53.1^{\pm 0.81}$ </td><td> $80.5^{\pm 0.12}$ </td><td> $56.9^{\pm 0.25}$ </td><td> $83.1^{\pm 0.10}$ </td><td> $50.5^{\pm 0.31}$ </td><td>86.2</td><td>59.5</td></tr><tr><td>VOS ◇</td><td> $93.1^{\pm 0.10}$ </td><td> $82.1^{\pm 0.31}$ </td><td> $86.7^{\pm 0.22}$ </td><td> $59.4^{\pm 0.58}$ </td><td> $90.9^{\pm 0.18}$ </td><td> $56.0^{\pm 0.66}$ </td><td> $82.4^{\pm 0.10}$ </td><td> $59.7^{\pm 0.21}$ </td><td> $83.8^{\pm 0.10}$ </td><td> $51.8^{\pm 0.34}$ </td><td>87.8</td><td>61.8</td></tr><tr><td>OE ◇</td><td> $94.5^{\pm 0.10}$ </td><td> $84.3^{\pm 0.24}$ </td><td> $90.9^{\pm 0.15}$ </td><td> $66.7^{\pm 0.31}$ </td><td> $94.0^{\pm 0.14}$ </td><td> $66.2^{\pm 0.40}$ </td><td> $82.4^{\pm 0.08}$ </td><td> $60.5^{\pm 0.19}$ </td><td> $85.7^{\pm 0.10}$ </td><td> $56.4^{\pm 0.33}$ </td><td>89.5</td><td>66.8</td></tr><tr><td>MixOE ◇</td><td> $93.6^{\pm 0.12}$ </td><td> $82.1^{\pm 0.28}$ </td><td> $91.7^{\pm 0.19}$ </td><td> $67.6^{\pm 0.37}$ </td><td> $93.9^{\pm 0.15}$ </td><td> $62.9^{\pm 0.37}$ </td><td> $85.4^{\pm 0.09}$ </td><td> $61.9^{\pm 0.19}$ </td><td> $87.4^{\pm 0.11}$ </td><td> $56.0^{\pm 0.36}$ </td><td>90.4</td><td>66.1</td></tr><tr><td>LogitNorm ◇</td><td> $93.1^{\pm 0.11}$ </td><td> $82.2^{\pm 0.21}$ </td><td> $89.0^{\pm 0.17}$ </td><td> $64.3^{\pm 0.35}$ </td><td> $91.9^{\pm 0.14}$ </td><td> $59.7^{\pm 0.31}$ </td><td> $81.7^{\pm 0.05}$ </td><td> $58.7^{\pm 0.15}$ </td><td> $84.6^{\pm 0.06}$ </td><td> $53.4^{\pm 0.37}$ </td><td>88.1</td><td>63.7</td></tr><tr><td>MCDropout ◇</td><td> $92.4^{\pm 0.17}$ </td><td> $80.7^{\pm 0.27}$ </td><td> $87.8^{\pm 0.19}$ </td><td> $61.7^{\pm 0.38}$ </td><td> $91.3^{\pm 0.16}$ </td><td> $57.5^{\pm 0.44}$ </td><td> $80.9^{\pm 0.08}$ </td><td> $57.5^{\pm 0.19}$ </td><td> $84.0^{\pm 0.08}$ </td><td> $52.2^{\pm 0.32}$ </td><td>87.3</td><td>65.5</td></tr><tr><td>Mixup ◇</td><td> $89.8^{\pm 0.62}$ </td><td> $73.8^{\pm 1.54}$ </td><td> $89.6^{\pm 0.19}$ </td><td> $64.0^{\pm 0.39}$ </td><td> $91.3^{\pm 0.18}$ </td><td> $55.4^{\pm 0.42}$ </td><td> $83.2^{\pm 0.10}$ </td><td> $61.5^{\pm 0.25}$ </td><td> $85.6^{\pm 0.07}$ </td><td> $56.0^{\pm 0.34}$ </td><td>87.9</td><td>66.5</td></tr><tr><td>CutMix ◇</td><td> $90.4^{\pm 0.26}$ </td><td> $69.0^{\pm 0.71}$ </td><td> $89.5^{\pm 0.16}$ </td><td> $62.4^{\pm 0.34}$ </td><td> $90.9^{\pm 0.15}$ </td><td> $50.6^{\pm 0.40}$ </td><td> $82.6^{\pm 0.12}$ </td><td> $61.3^{\pm 0.25}$ </td><td> $85.1^{\pm 0.22}$ </td><td> $55.2^{\pm 0.51}$ </td><td>87.7</td><td>59.7</td></tr></table>

Evaluation Protocol. We compute AUC for OOD detection on each task model. The classes of the task are the IND classes while the classes of all other tasks of the dataset are the OOD classes. The evaluation metric for CIL is accuracy (ACC), which is measured after all tasks are learned. We report the average AUC value over all the tasks in each dataset, and the ACC of each dataset. Note that as KNN needs the training data at test time, in the CIL setting, we can only use the saved replay data of each task for its OOD score computation as the full data of previous tasks are not accessible in CIL. We also compute TIL accuracy in using difference OOD training-time techniques.

# A.3 RESULTS AND ANALYSIS

The experiment results are given in Table 4, which allow us to make some important observations.

(1) OOD detection and CIL performances. The OOD detection results in AUC here have similar trends as those in Yang et al. (2022) except KNN, which was considered as one of the best methods. But it is weak here because, as indicated above, in CIL, KNN can only use the replay data (which is very small) for each task to compute the OOD score. The CIL performances of the top OOD methods are competitive compared to CIL baselines in Table 1.   
(2) Similar TIL Performance. We present the detailed results of TIL accuracy, for the OOD detection baselines as table 5. It is evident from the results that there is minimal variation in the performance of OOD detection methods across different datasets, with average values of 99.2±0.02, 95.7±0.12, 97.6±0.10, 84.3±0.86, 88.1±0.86 for C10-5T, C100-10T, C100-20T, T-5T, and T-10T, respectively. The observation that the majority of OOD methods exhibit negligible impact on the IND classification performance aligns with a previous benchmark study on OOD detection conducted by Yang et al. (2022). Notably, the aforementioned study investigated the finding on ResNet architecture (He et al., 2016) without pre-training, whereas our experiments involved a pre-trained DeiT model. This suggests that the finding extends to various model backbones.   
(3) Linear relationship between OOD AUC and CIL ACC. We plot the relationship between OOD AUC and CIL performances in fig. 3. Interestingly, we see a linear relationship with Pearson correlation coefficients of 0.976, 0.811, 0.941, 0.963, and 0.980 for the 5 datasets, respectively. This finding suggests that improving OOD AUC can bring about a linear improvement of ×1.5 ∼ 3.4 (which is the slopes of the fitted linear function) on CIL ACC. Note that we are not conditioning

Table 5: Average IND classification ACC (TIL accuracy) results of pre-trained DeiT on five datasets with five random seeds. The baselines are divided into three categories as Table 1 in the main text. The first category includes post-hoc detectors, the second category includes methods that exploit surrogate OOD data, and the last category includes methods that employ special training strategies. ♡: logit-based OOD scores, ♠: featurebased OOD scores, ♢: training-time techniques. Note that the post-hoc methods only differ in the OOD score computation, which means they share the same trained model and thus have the same TIL ACC. 

<table><tr><td></td><td>C10-5T</td><td>C100-10T</td><td>C100-20T</td><td>T-5T</td><td>T-10T</td><td>Average</td></tr><tr><td>GradNorm ♥KL-Matching ♥OpenMax ♥ODIN ♥MSP ♥MLS ♥EBO ♥ReAct ♥KNN ♠Residual ♠MDS ♠VIM ♥♠</td><td>99.20</td><td>95.71</td><td>97.50</td><td>84.40</td><td>88.10</td><td>92.98</td></tr><tr><td>CMG ◇</td><td>99.20</td><td>95.70</td><td>97.61</td><td>84.52</td><td>88.30</td><td>93.07</td></tr><tr><td>VOS ◇</td><td>99.20</td><td>95.66</td><td>97.59</td><td>84.51</td><td>88.30</td><td>93.05</td></tr><tr><td>OE ◇</td><td>99.17</td><td>95.73</td><td>97.72</td><td>84.46</td><td>88.64</td><td>93.14</td></tr><tr><td>MixOE ◇</td><td>99.21</td><td>95.82</td><td>97.80</td><td>82.06</td><td>86.53</td><td>92.28</td></tr><tr><td>LogitNorm ◇</td><td>99.20</td><td>95.71</td><td>97.56</td><td>84.55</td><td>87.87</td><td>92.98</td></tr><tr><td>MCDropout ◇</td><td>99.20</td><td>95.60</td><td>97.62</td><td>84.51</td><td>88.23</td><td>93.03</td></tr><tr><td>Mixup ◇</td><td>99.24</td><td>95.57</td><td>97.71</td><td>85.04</td><td>88.47</td><td>93.21</td></tr><tr><td>CutMix ◇</td><td>99.20</td><td>95.40</td><td>97.54</td><td>84.54</td><td>88.40</td><td>93.02</td></tr></table>

![](images/a2707f89526307aa99cad341f69dc526320bb205b6cbb0b03277265deb5aef64.jpg)

<details>
<summary>scatter</summary>

| Group    | OOD Results (AUC) | CIL Results (ACC) |
| -------- | ----------------- | ----------------- |
| C10-5T   | 75                | 45                |
| C10-5T   | 80                | 50                |
| C10-5T   | 85                | 60                |
| C10-5T   | 90                | 70                |
| C10-5T   | 95                | 85                |
| C100-10T | 75                | 40                |
| C100-10T | 80                | 45                |
| C100-10T | 85                | 55                |
| C100-10T | 90                | 65                |
| C100-10T | 95                | 75                |
| C100-20T | 75                | 35                |
| C100-20T | 80                | 25                |
| C100-20T | 85                | 45                |
| C100-20T | 90                | 55                |
| C100-20T | 95                | 65                |
| T-5T     | 75                | 48                |
| T-5T     | 80                | 52                |
| T-5T     | 85                | 60                |
| T-5T     | 90                | 68                |
| T-5T     | 95                | 78                |
| T-10T    | 75                | 32                |
| T-10T    | 80                | 40                |
| T-10T    | 85                | 50                |
| T-10T    | 90                | 60                |
| T-10T    | 95                | 70                |
</details>

Figure 3: The correlation between OOD (AUC) and CIL (ACC) results. Each point denotes the AUC and ACC of one method in table 4 on the same dataset.

on TIL accuracy as (2) above showed the TIL results are similar for different training-time OOD techniques.

# B OUTPUT CALIBRATION

We used the output calibration technique to balance the scales of task-id prediction scores for different tasks, which is motivated by Kim et al. (2022a;b). Even if the task-id prediction of each task-model is perfect, the system can make an incorrect task-id prediction if the magnitudes of the outputs across different tasks are different. As the task-specific modules are trained separately in HAT, it is useful to calibrate the outputs of different task modules.

To ensure that the output values are comparable, we calibrate the outputs by scaling $\sigma _ { 1 } ^ { ( t ) }$ and shifting $\sigma _ { 2 } ^ { ( t ) }$ σ2 parameters for each task. The optimal parameters $\{ ( \sigma _ { 1 } ^ { ( t ) } , \sigma _ { 2 } ^ { ( t ) } ) \} _ { t = 1 } ^ { T } \in \mathbb { R } ^ { 2 T }$ 2 ) } t =1 ∈ (T is the number of tasks) can be found by solving optimization problem using samples in the replay buffer Buf.

Specifically, we minimize the cross-entropy epochs to find optimal calibration parameters $\{ ( \sigma _ { 1 } ^ { ( t ) } , \sigma _ { 2 } ^ { ( t ) } ) \} _ { t = 1 } ^ { T }$ (t) optimizer with batch size 64 for 1001:

$$
\mathcal {L} _ {\text { calibration }} = - \mathbb {E} _ {(\boldsymbol {x}, y) \in B u f} \log p (y | \boldsymbol {x}),
$$

where $p ( y | \mathbf { \boldsymbol { x } } )$ is computed using eq. (3) and calibration parameters:

$$
p (y _ {j} ^ {(t)} | \boldsymbol {x}) = \sigma_ {1} ^ {(t)} \cdot \left[ s o f t m a x \left(f (h (\boldsymbol {x}; \phi^ {(t)}); \theta^ {(t)})\right) \right] _ {j} \cdot S (\boldsymbol {x}; t) + \sigma_ {2} ^ {(t)}
$$

Given the optimal parameters $\{ ( \tilde { \sigma } _ { 1 } ^ { ( t ) } , \tilde { \sigma } _ { 2 } ^ { ( t ) } ) \} _ { t = 1 } ^ { T }$ , we make the final prediction as:

$$
\hat {y} = \operatorname * {a r g   m a x} _ {1 \leq t \leq T, 1 \leq j \leq | \mathcal {Y} _ {t} |} p (y _ {j} ^ {(t)} | \boldsymbol {x})
$$

# C EVALUATION METRICS

# C.1 DEFINITIONS OF AIA. AND LAST ACCURACY

Fllowing (Kim et al., 2022a), we give the formal definitions of average incremental accuracy (AIA in Table 1 in the main text, denoted as $A _ { A I A } )$ and accuracy after learning the final task (Last in Table 1 in the main text, denoted as $A _ { l a s t } )$ . Let the accuracy after learning the task t be:

$$
A ^ {(\leq t)} = \frac {\# c o r r e c t l y c l a s s i f i e d s a m p l e s i n \bigcup_ {k = 1} ^ {t} \mathcal {D} _ {t e s t} ^ {(k)}}{\# s a m p l e s i n \bigcup_ {k = 1} ^ {t} \mathcal {D} _ {t e s t} ^ {(k)}}
$$

Let T be the last task. Then, $A _ { l a s t } = A ^ { ( \leq T ) }$ and $\begin{array} { r } { A _ { A I A } = \frac { 1 } { T } \sum _ { k = 1 } ^ { T } A ^ { ( \leq k ) } } \end{array}$

Here $\mathcal { D } _ { t e s t } ^ { ( k ) }$ denotes the test-set for task k, and # denotes “the number of". To put it simply, $A ^ { ( \leq t ) }$ means the accuracy of all the test data from task 1 to task t.

# C.2 RECTIFIED FORGETTING RATE FOR CIL

Apart from the classification accuracy (ACC), we report another popular CIL evaluation metric average forgetting rate. The popular definition of average forgetting rate is the following

$$
\mathcal {F} ^ {(t)} = \frac {1}{t - 1} \sum_ {i = 1} ^ {t - 1} (A _ {i} ^ {(i)} - A _ {i} ^ {(t)}),
$$

where $A _ { i } ^ { ( t ) }$ is the accuracy of task i’s test set on the CL model after task t is learned (Liu et al., 2020b), which is also referred to as backward transfer in other literature (Lopez-Paz & Ranzato, 2017).

However, this formula is only suitable for TIL but not appropriate for CIL. As the task-id for each test sample is given in testing for TIL, all the test samples from task i will be classified into one of the classes of task i. If there is no forgetting for a TIL model, then $A _ { i } ^ { ( t ) }$ will be equal to $A _ { i } ^ { ( i ) } ( i < t )$ in such a within-task classification, where the number of classes is fixed. But in CIL, the task-id is not provided in testing and we are not doing within-task classification. Even if in the Non-CL setting as more tasks or classes are learned, the classification accuracy will usually decrease for the same test set due to the nature of multi-class classification. That is, $A _ { i } ^ { ( t ) } < A _ { i } ^ { ( i ) } ( i < t )$ is usually true as at task i there are fewer learned classes than at task t. Their difference is not due to forgetting.

Furthermore, as we discussed in Section 1, forgetting (CF) is not the only issue of CIL. Inter-class separation (ICS) is another important one. When considering the performance degradation of each task during continual learning, it is hard to disentangle the effects of these two factors. Our rectified average forgetting rate metric for CIL considers both forgetting and ICS. Two new average forgetting rates for CIL are defined, one for the case where we use the Last accuracy as the evaluation metric and one for the case where we use AIA as the evaluation metric:

$$
\mathcal {F} _ {C I L, L a s t} ^ {(t)} = \frac {1}{t} \sum_ {i = 1} ^ {t} (A _ {i} ^ {(t, N C L)} - A _ {i} ^ {(t)})
$$

$$
\mathcal {F} _ {C I L, A I A} ^ {(t)} = \frac {1}{t} \sum_ {i = 1} ^ {t} \mathcal {F} _ {C I L, L a s t} ^ {(i)},
$$

where NCL means Non- $\cdot \mathrm { C L } , A _ { i } ^ { ( t ) }$ is the accuracy of task i’s test set on the CL model after task t is learned, and $A _ { i } ^ { ( t , N C L ) }$ is the accuracy of task i’s test set on the Non-CL model that learns all tasks from 1 to t. If the test dataset sizes are the same across different tasks, then $\mathcal { F } _ { C I L , L a s t } ^ { ( t ) }$ F (t)CIL, Last is equal to $A _ { l a s t , N C L } - A _ { l a s t } ,$ where $A _ { l a s t }$ is defined in appendix C.1 and $A _ { l a s t , N C L }$ is the $A _ { l a s t }$ of Non-CL. The intuition of introducing NCL performance is to address the loss of accuracy for each task when more tasks are learned.

Table 6: Forgetting rate (%) for CIL on the five datasets of the baselines in Table 1. The lower the rate, the better the method is. 

<table><tr><td rowspan="2">Method</td><td colspan="2">C10-5T</td><td colspan="2">C100-10T</td><td colspan="2">C100-20T</td><td colspan="2">T-5T</td><td colspan="2">T-10T</td><td colspan="2">Average</td></tr><tr><td> $\mathcal{F}_{\text{CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\text{CIL, AIA}}^{(T)}$ </td><td> $\mathcal{F}_{\text{CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\text{CIL, AIA}}^{(T)}$ </td><td> $\mathcal{F}_{\text{CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\text{CIL, AIA}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, AIA}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, AIA}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, Last}}^{(T)}$ </td><td> $\mathcal{F}_{\mathrm {CIL, AIA}}^{(T) }$ </td></tr><tr><td>OWM</td><td>54.10</td><td>41.01</td><td>61.37</td><td>47.10</td><td>65.78</td><td>54.95</td><td>47.97</td><td>31.85</td><td>55.00</td><td>41.28</td><td>56.84</td><td>43.24</td></tr><tr><td>ADAM</td><td>11.87</td><td>6.68</td><td>21.55</td><td>14.65</td><td>23.77</td><td>16.64</td><td>22.41</td><td>15.18</td><td>22.84</td><td>15.59</td><td>20.49</td><td>13.75</td></tr><tr><td>PASS</td><td>9.58</td><td>7.98</td><td>13.86</td><td>10.19</td><td>15.99</td><td>11.11</td><td>11.49</td><td>9.91</td><td>14.18</td><td>9.70</td><td>13.02</td><td>9.78</td></tr><tr><td> $HAT_{CIL}$ </td><td>13.39</td><td>5.95</td><td>19.85</td><td>13.21</td><td>23.22</td><td>18.41</td><td>13.30</td><td>7.65</td><td>18.49</td><td>11.40</td><td>17.65</td><td>11.32</td></tr><tr><td>iCaRL</td><td>39.46</td><td>28.82</td><td>57.55</td><td>43.37</td><td>60.77</td><td>51.56</td><td>41.99</td><td>27.77</td><td>50.62</td><td>37.45</td><td>50.08</td><td>37.79</td></tr><tr><td>A-GEM</td><td>8.24</td><td>7.27</td><td>13.86</td><td>10.70</td><td>13.61</td><td>10.47</td><td>19.39</td><td>15.67</td><td>20.64</td><td>13.47</td><td>15.15</td><td>11.52</td></tr><tr><td>EEIL</td><td>13.45</td><td>6.51</td><td>14.68</td><td>6.10</td><td>18.97</td><td>7.99</td><td>19.18</td><td>10.40</td><td>22.14</td><td>10.49</td><td>17.68</td><td>8.30</td></tr><tr><td>GD</td><td>6.63</td><td>2.79</td><td>18.40</td><td>6.69</td><td>22.66</td><td>9.10</td><td>19.51</td><td>9.52</td><td>30.04</td><td>13.12</td><td>19.45</td><td>8.24</td></tr><tr><td>DER++</td><td>9.00</td><td>4.18</td><td>9.46</td><td>4.31</td><td>10.76</td><td>4.74</td><td>12.95</td><td>6.71</td><td>15.34</td><td>6.82</td><td>11.50</td><td>5.35</td></tr><tr><td>HAL</td><td>11.41</td><td>10.01</td><td>15.59</td><td>9.78</td><td>15.39</td><td>9.68</td><td>19.72</td><td>11.72</td><td>17.27</td><td>12.55</td><td>15.88</td><td>10.75</td></tr><tr><td>DER++</td><td>11.16</td><td>8.00</td><td>13.03</td><td>6.56</td><td>12.73</td><td>5.81</td><td>16.68</td><td>10.48</td><td>18.32</td><td>9.89</td><td>14.38</td><td>8.15</td></tr><tr><td>FOSTER</td><td>9.70</td><td>5.47</td><td>11.07</td><td>6.04</td><td>9.85</td><td>4.51</td><td>18.08</td><td>7.08</td><td>16.82</td><td>7.03</td><td>13.10</td><td>6.03</td></tr><tr><td>BEEF</td><td>8.69</td><td>3.91</td><td>10.67</td><td>5.29</td><td>10.88</td><td>6.08</td><td>11.11</td><td>5.82</td><td>14.36</td><td>5.87</td><td>11.14</td><td>5.39</td></tr><tr><td>MORE</td><td>6.63</td><td>2.78</td><td>12.5</td><td>5.96</td><td>12.2</td><td>5.94</td><td>7.55</td><td>3.00</td><td>9.46</td><td>4.29</td><td>9.68</td><td>4.39</td></tr><tr><td>ROW</td><td>4.82</td><td>2.56</td><td>8.04</td><td>4.33</td><td>8.16</td><td>4.41</td><td>7.41</td><td>2.87</td><td>9.31</td><td>4.12</td><td>7.55</td><td>3.66</td></tr><tr><td>TPL</td><td>3.46</td><td>1.90</td><td>6.23</td><td>3.10</td><td>6.42</td><td>3.07</td><td>3.88</td><td>0.26</td><td>5.32</td><td>1.31</td><td>5.06</td><td>1.93</td></tr><tr><td> $TPL_{PFI}$ </td><td>2.04</td><td>1.07</td><td>1.18</td><td>1.44</td><td>2.75</td><td>1.57</td><td>1.49</td><td>1.07</td><td>1.84</td><td>1.26</td><td>1.86</td><td>1.28</td></tr></table>

Table 6 shows the average forgetting rates of each system based on the new definitions. We clearly observe that TPL and TPLPFI have the lowest average forgetting rates on the five datasets among all systems. SLDA and L2P are not included as they use different architectures and cannot take the Non-CL results in Table 1 in the main text as the upper bounds or NCL results needed in the proposed formulas above.

It is important to note that for both TPL and TPLPFI, the forgetting rate mainly reflects the performance loss due to the ICS problem rather than the traditional catastrophic forgetting (CF) caused by network parameter interference in the incremental learning of different tasks because the TIL method HAT has effectively eliminated CF in TPL and TPLPFI.

# D ADDITIONAL EXPERIMENTAL RESULTS

# D.1 CIL EXPERIMENTS ON A LARGER DATASET

Table 7: The CIL ACC after the final task on ImageNet380-10T. We highlight the best results in bold. 

<table><tr><td> $HAT_{CIL}$ </td><td>ADAM</td><td>SLDA</td><td>PASS</td><td>L2P</td><td>iCaRL</td><td>A-GEM</td><td>EEIL</td></tr><tr><td> $71.20^{\pm0.99}$ </td><td> $62.10^{\pm0.91}$ </td><td> $65.78^{\pm0.05}$ </td><td> $65.27^{\pm1.24}$ </td><td> $47.89^{\pm3.24}$ </td><td> $62.23^{\pm0.66}$ </td><td> $30.38^{\pm10.02}$ </td><td> $63.37^{\pm0.49}$ </td></tr><tr><td>DER++</td><td>HAL</td><td>DER</td><td>FORSTER</td><td>BEEF</td><td>MORE</td><td>ROW</td><td>TPL</td></tr><tr><td> $66.53^{\pm2.36}$ </td><td> $64.83^{\pm2.60}$ </td><td> $69.19^{\pm1.36}$ </td><td> $68.07^{\pm1.88}$ </td><td> $70.07^{\pm1.41}$ </td><td> $72.10^{\pm1.44}$ </td><td> $74.52^{\pm1.38}$ </td><td> $78.49^{\pm0.89}$ </td></tr></table>

To assess the performance of our proposed TPL on large-scale datasets, we use ImageNet-1k (Russakovsky et al., 2015), a widely recognized benchmark dataset frequently examined in the CIL literature. However, due to the nature of our experiments, which involve a DeiT backbone pretrained on 611 ImageNet classes after excluding 389 classes similar to those in CIFAR and TinyImageNet, we cannot directly evaluate our model on the original ImageNet dataset to avoid potential information leak.

To overcome this limitation, we created a new benchmark dataset called ImageNet-380. We randomly selected 380 classes from the remaining 389 classes, excluding those similar to CIFAR and TinyImageNet, from the original set of 1k classes in the full ImageNet dataset. This new dataset consists of approximately 1,300 color images per class. For ImageNet-380, we divided the classes into 10 tasks, with each task comprising 38 classes. We set the replay buffer size to 7600, with 20 samples per class, which is a commonly used number in replay-based methods. We refer to these experiments as ImageNet380-10T. For other training configurations, we kept them consistent with the experiments conducted on T-10T.

The CIL Last ACC achieved after the final task on ImageNet380-10T can be found in Table 7. Notably, the results obtained by TPL still exhibit a significant improvement over the baselines, with a 3.97% higher ACC compared to the best baseline method ROW. The results further provide strong evidence supporting the effectiveness of our proposed TPL system.

# D.2 CIL EXPERIMENTS WITHOUT PRE-TRAINING

Table 8: CIL accuracy (%) after the final task (last) based on ResNet-18 without pre-training over 5 runs with random seeds. “-XT”: X number of tasks. The best result in each column is highlighted in bold. 

<table><tr><td></td><td>C10-5T</td><td>C100-10T</td><td>C100-20T</td><td>T-5T</td><td>T-10T</td><td>Average</td></tr><tr><td>OWM</td><td> $51.8^{\pm 0.05}$ </td><td> $28.9^{\pm 0.60}$ </td><td> $24.1^{\pm 0.26}$ </td><td> $10.0^{\pm 0.55}$ </td><td> $8.6^{\pm 0.42}$ </td><td>24.7</td></tr><tr><td>PASS</td><td> $47.3^{\pm 0.98}$ </td><td> $33.0^{\pm 0.58}$ </td><td> $25.0^{\pm 0.69}$ </td><td> $28.4^{\pm 0.51}$ </td><td> $19.1^{\pm 0.46}$ </td><td>30.6</td></tr><tr><td>EEIL</td><td> $64.5^{\pm 0.93}$ </td><td> $52.3^{\pm 0.83}$ </td><td> $48.0^{\pm 0.44}$ </td><td> $38.2^{\pm 0.54}$ </td><td> $28.7^{\pm 0.87}$ </td><td>46.3</td></tr><tr><td>GD</td><td> $65.5^{\pm 0.94}$ </td><td> $51.4^{\pm 0.83}$ </td><td> $50.3^{\pm 0.88}$ </td><td> $38.9^{\pm 1.01}$ </td><td> $29.5^{\pm 0.68}$ </td><td>47.1</td></tr><tr><td>HAL</td><td> $63.7^{\pm 0.91}$ </td><td> $51.3^{\pm 1.22}$ </td><td> $48.5^{\pm 0.71}$ </td><td> $38.1^{\pm 0.97}$ </td><td> $30.3^{\pm 1.05}$ </td><td>46.4</td></tr><tr><td>A-GEM</td><td> $64.6^{\pm 0.72}$ </td><td> $50.5^{\pm 0.73}$ </td><td> $47.3^{\pm 0.87}$ </td><td> $37.3^{\pm 0.89}$ </td><td> $29.4^{\pm 0.95}$ </td><td>45.8</td></tr><tr><td>HAT $_{CIL}$ </td><td> $62.7^{\pm 1.45}$ </td><td> $41.1^{\pm 0.93}$ </td><td> $25.6^{\pm 0.51}$ </td><td> $38.5^{\pm 1.85}$ </td><td> $29.8^{\pm 0.65}$ </td><td>39.5</td></tr><tr><td>iCaRL</td><td> $63.4^{\pm 1.11}$ </td><td> $51.4^{\pm 0.99}$ </td><td> $47.8^{\pm 0.48}$ </td><td> $37.0^{\pm 0.41}$ </td><td> $28.3^{\pm 0.18}$ </td><td>45.6</td></tr><tr><td>DER++</td><td> $66.0^{\pm 1.20}$ </td><td> $53.7^{\pm 1.20}$ </td><td> $46.6^{\pm 1.44}$ </td><td> $35.8^{\pm 0.77}$ </td><td> $30.5^{\pm 0.47}$ </td><td>46.5</td></tr><tr><td>DER</td><td> $62.1^{\pm 0.97}$ </td><td> $64.5^{\pm 0.85}$ </td><td> $62.5^{\pm 0.76}$ </td><td> $43.6^{\pm 0.77}$ </td><td> $38.3^{\pm 0.82}$ </td><td>54.2</td></tr><tr><td>FOSTER</td><td> $65.4^{\pm 1.05}$ </td><td> $62.5^{\pm 0.84}$ </td><td> $56.3^{\pm 0.71}$ </td><td> $40.5^{\pm 0.92}$ </td><td> $36.4^{\pm 0.85}$ </td><td>52.2</td></tr><tr><td>BEEF</td><td> $67.3^{\pm 1.07}$ </td><td> $60.9^{\pm 0.87}$ </td><td> $56.7^{\pm 0.72}$ </td><td> $44.1^{\pm 0.85}$ </td><td> $37.9^{\pm 0.95}$ </td><td>53.4</td></tr><tr><td>MORE</td><td> $70.6^{\pm 0.74}$ </td><td> $57.5^{\pm 0.68}$ </td><td> $51.3^{\pm 0.89}$ </td><td> $41.2^{\pm 0.81}$ </td><td> $35.4^{\pm 0.72}$ </td><td>51.2</td></tr><tr><td>ROW</td><td> $74.6^{\pm 0.89}$ </td><td> $58.2^{\pm 0.67}$ </td><td> $52.1^{\pm 0.91}$ </td><td> $42.3^{\pm 0.69}$ </td><td> $38.2^{\pm 1.34}$ </td><td>53.1</td></tr><tr><td>TPL</td><td> $78.4^{\pm 0.78}$ </td><td> $62.2^{\pm 0.52}$ </td><td> $55.8^{\pm 0.57}$ </td><td> $48.2^{\pm 0.64}$ </td><td> $42.9^{\pm 0.45}$ </td><td>57.5</td></tr></table>

The experimental setup and results for CIL baselines without pre-training are presented in this section.

Training details. We follow Kim et al. (2022b) to use ResNet-18 (He et al., 2016) for all the datasets (CIFAR-10, CIFAR-100, TinyImageNet) and all the baselines excluding OWM. MORE, ROW, and our TPL are designed for pre-trained models, and we adapt them by applying HAT on ResNet-18. All other baselines adopted ResNet-18 in their original paper. OWM adopts AlexNet as it is hard to apply the method to the ResNet. For the replay-based methods, we also use the same buffer size as specified in Section 5.1. We use the hyper-parameters suggested by their original papers. For MORE, ROW, and our proposed TPL, we follow the hyper-parameters used in HAT for training.

Experimental Results. The results for CIL Last accuracy (ACC) after the final task are shown in Table 8. We can observe that the network-expansion-based approaches (DER, FOSTER, BEEF) and approaches that predict task-id based on TIL+OOD (MORE, ROW) are two competitive groups of CIL baselines. Our proposed TPL achieves the best performance on C10-5T, T-5T, and T-10T, while DER achieves the best on C100-10T and C100-20T. Overall, our proposed TPL achieves the best average accuracy over the 5 datasets (57.5%) while DER ranks the second (with an average ACC of 54.2%).

# D.3 CIL EXPERIMENTS ON SMALLER REPLAY BUFFER SIZE

Table 9: CIL accuracy (%) after the final task (Last) with smaller replay buffer size over 5 runs with random seeds. “-XT”: X number of tasks. The best result in each column is highlighted in bold. The replay buffer size is set to 100 for CIFAR-10, and 1000 for CIFAR-100 and TinyImageNet. The pre-trained model is used. 

<table><tr><td></td><td>C10-5T</td><td>C100-10T</td><td>C100-20T</td><td>T-5T</td><td>T-10T</td><td>Average</td></tr><tr><td>iCaRL</td><td> $86.08^{\pm 1.19}$ </td><td> $66.96^{\pm 2.08}$ </td><td> $68.16^{\pm 0.71}$ </td><td> $47.27^{\pm 3.22}$ </td><td> $49.51^{\pm 1.87}$ </td><td>63.60</td></tr><tr><td>A-GEM</td><td> $56.64^{\pm 4.29}$ </td><td> $23.18^{\pm 2.54}$ </td><td> $20.76^{\pm 2.88}$ </td><td> $31.44^{\pm 3.84}$ </td><td> $23.73^{\pm 6.27}$ </td><td>31.15</td></tr><tr><td>EEIL</td><td> $77.44^{\pm 3.04}$ </td><td> $62.95^{\pm 0.68}$ </td><td> $57.86^{\pm 0.74}$ </td><td> $48.36^{\pm 1.38}$ </td><td> $44.59^{\pm 1.72}$ </td><td>58.24</td></tr><tr><td>GD</td><td> $85.96^{\pm 1.64}$ </td><td> $57.17^{\pm 1.06}$ </td><td> $50.30^{\pm 0.58}$ </td><td> $46.09^{\pm 1.77}$ </td><td> $32.41^{\pm 2.75}$ </td><td>54.39</td></tr><tr><td>DER++</td><td> $80.09^{\pm 3.00}$ </td><td> $64.89^{\pm 2.48}$ </td><td> $65.84^{\pm 1.46}$ </td><td> $50.74^{\pm 2.41}$ </td><td> $49.24^{\pm 5.01}$ </td><td>62.16</td></tr><tr><td>HAL</td><td> $79.16^{\pm 4.56}$ </td><td> $62.65^{\pm 0.83}$ </td><td> $63.96^{\pm 1.49}$ </td><td> $48.17^{\pm 2.94}$ </td><td> $47.11^{\pm 6.00}$ </td><td>60.21</td></tr><tr><td>DER</td><td> $85.11^{\pm 1.44}$ </td><td> $72.31^{\pm 0.78}$ </td><td> $70.25^{\pm 0.98}$ </td><td> $58.07^{\pm 1.40}$ </td><td> $55.85^{\pm 1.23}$ </td><td>68.32</td></tr><tr><td>FOSTER</td><td> $84.99^{\pm 0.89}$ </td><td> $70.25^{\pm 0.58}$ </td><td> $71.14^{\pm 0.76}$ </td><td> $53.35^{\pm 0.54}$ </td><td> $54.58^{\pm 0.85}$ </td><td>66.86</td></tr><tr><td>BEEF</td><td> $86.20^{\pm 1.59}$ </td><td> $70.87^{\pm 2.77}$ </td><td> $70.44^{\pm 1.24}$ </td><td> $60.15^{\pm 0.98}$ </td><td> $57.02^{\pm 0.87}$ </td><td>68.94</td></tr><tr><td>MORE</td><td> $88.13^{\pm 1.16}$ </td><td> $71.69^{\pm 0.11}$ </td><td> $71.29^{\pm 0.55}$ </td><td> $64.17^{\pm 0.77}$ </td><td> $61.90^{\pm 0.90}$ </td><td>71.44</td></tr><tr><td>ROW</td><td> $89.70^{\pm 1.54}$ </td><td> $73.63^{\pm 0.12}$ </td><td> $71.86^{\pm 0.07}$ </td><td> $65.42^{\pm 0.55}$ </td><td> $62.87^{\pm 0.53}$ </td><td>72.70</td></tr><tr><td>TPL</td><td> $91.76^{\pm 0.44}$ </td><td> $75.83^{\pm 0.28}$ </td><td> $75.65^{\pm 0.54}$ </td><td> $68.08^{\pm 0.61}$ </td><td> $66.48^{\pm 0.47}$ </td><td>75.56</td></tr></table>

# D.4 ABLATION ON $\beta _ { 1 }$ AND $\beta _ { 2 }$

Table 10: The CIL accuracy (%) after leraning the final task (Last) of our TPL on C10-5T for different $\beta _ { 1 }$ and $\beta _ { 2 }$ . 

<table><tr><td> $\beta_2$  $\beta_1$ </td><td>15.0</td><td>17.5</td><td>20.0</td><td>22.5</td><td>25.0</td></tr><tr><td>0.5</td><td>91.9</td><td>91.9</td><td>91.8</td><td>92.5</td><td>92.2</td></tr><tr><td>0.6</td><td>92.1</td><td>91.0</td><td>92.2</td><td>91.6</td><td>92.0</td></tr><tr><td>0.7</td><td>91.8</td><td>92.1</td><td>92.3</td><td>92.4</td><td>91.9</td></tr><tr><td>0.8</td><td>92.0</td><td>92.4</td><td>91.7</td><td>91.9</td><td>91.7</td></tr><tr><td>0.9</td><td>92.3</td><td>92.5</td><td>92.0</td><td>91.9</td><td>91.5</td></tr></table>

$\beta _ { 1 }$ and $\beta _ { 2 }$ are two scaling hyper-parameters used in the definition of task-id prediction score $S _ { T P L } ^ { ( t ) } ( { \pmb x } )$ . Table 10 shows an ablation for different ot affect results much. For other hyper-pa $\beta _ { 1 }$ and mete $\beta _ { 2 }$ on C10-5T, which indicates that used in TPL, see Appendix I.1 fo $\beta _ { 1 }$ andmore $\beta _ { 2 }$ details.

# D.5 EXPERIMENTS BASED ON MORE PRE-TRAINED MODELS

Table 11: Last TIL and CIL accuracy results (after the last task is learned) for TPL based on different pre-trained visual encoders or models. 

<table><tr><td rowspan="2">Visual Encoder</td><td rowspan="2">Pre-training</td><td colspan="2">C10-5T</td><td colspan="2">C100-10T</td><td colspan="2">C100-20T</td><td colspan="2">T-5T</td><td colspan="2">T-10T</td><td colspan="2">Average</td></tr><tr><td>TIL</td><td>CIL</td><td>TIL</td><td>CIL</td><td>TIL</td><td>CIL</td><td>TIL</td><td>CIL</td><td>TIL</td><td>CIL</td><td>TIL</td><td>CIL</td></tr><tr><td>DeiT-small-IN661 (TPL)</td><td>supervised</td><td>99.20</td><td>92.33</td><td>95.71</td><td>76.53</td><td>97.50</td><td>76.34</td><td>84.40</td><td>68.64</td><td>88.10</td><td>67.20</td><td>92.98</td><td>76.21</td></tr><tr><td>ViT-tiny</td><td rowspan="4">supervised</td><td>98.80</td><td>91.38</td><td>95.36</td><td>76.79</td><td>97.52</td><td>75.83</td><td>82.37</td><td>71.85</td><td>85.10</td><td>70.18</td><td>91.83</td><td>77.21</td></tr><tr><td>DeiT-tiny</td><td>98.85</td><td>90.79</td><td>94.80</td><td>74.01</td><td>97.67</td><td>73.21</td><td>83.58</td><td>72.46</td><td>86.54</td><td>71.71</td><td>92.29</td><td>76.44</td></tr><tr><td>ViT-small</td><td>99.43</td><td>95.57</td><td>97.51</td><td>84.52</td><td>98.76</td><td>83.94</td><td>89.49</td><td>81.80</td><td>91.30</td><td>80.94</td><td>95.30</td><td>85.35</td></tr><tr><td>DeiT-small (TPL $_{PFI}$ )</td><td>99.24</td><td>94.86</td><td>96.79</td><td>82.43</td><td>97.78</td><td>80.86</td><td>89.78</td><td>84.06</td><td>92.51</td><td>83.87</td><td>95.22</td><td>85.22</td></tr><tr><td>ViT-small-Dino</td><td>self-supervised</td><td>98.75</td><td>87.82</td><td>94.51</td><td>73.83</td><td>96.95</td><td>72.62</td><td>80.77</td><td>68.78</td><td>82.49</td><td>65.97</td><td>90.69</td><td>73.80</td></tr><tr><td>ViT-base-MAE</td><td>self-supervised</td><td>99.09</td><td>88.82</td><td>93.42</td><td>67.47</td><td>96.77</td><td>69.52</td><td>79.58</td><td>65.94</td><td>81.76</td><td>63.10</td><td>90.12</td><td>70.97</td></tr></table>

In this section, we conduct an ablation study on the visual encoder (pre-trained model/network). To prevent data contamination or information leak, TPL uses DeiT-S/16 pre-trained with 611 classes of ImageNet (DeiT-small-IN661) after removing 389 classes that overlap with classes in the continual learning datasets. Here we dismiss this limitation and experiment on more open-sourced pre-trained visual encoders trained using the full ImageNet. The results are reported in Table 11, which also includes the TIL accuracy results if the task-id is provided for each test instance during testing. We show DeiT-small-IN661, which is our TPL, in the first row of the table. It is the backbone used in our main experiments (Table 1) (IN: ImageNet). The other visual encoders are open-sourced in the timm (Wightman, 2019) library. We note that vanilla ViT and DeiT are pre-trained on ImageNet using supervised training, thereby there exists an information leak for CIFAR and Tiny-ImageNet datasets used in continual learning. The details of the models are as follows:

• ViT-tiny: The full model name in timm is “vit\_tiny\_patch16\_224”. It is trained on ImageNet (with additional augmentation and regularization) using supervised learning.   
• DeiT-tiny: The full model name in timm is “deit\_tiny\_patch16\_224”. It is trained on ImageNet (with additional augmentation and regularization) using supervised learning and distillation.   
• ViT-small: The small version of ViT (“vit\_small\_patch16\_224”).   
• DeiT-small: The small version of DeiT (“deit\_small\_patch16\_224”).   
• ViT-small-Dino: The small version of ViT trained with self-supervised DINO method (Caron et al., 2021).   
• ViT-base-MAE: The base version of ViT trained with self-supervised MAE method (He et al., 2021).

Analysis. We found that different pre-trained visual encoders have varied CIL results. Compared to DeiT-small-IN661, the pre-trained small ViT and DeiT that use the full ImageNet to conduct supervised learning have overall better performance, which is not surprising due to the class overlap as we discussed above. For the self-supervised visual encoders (Dino, MAE), the performances are worse than the supervised pre-trained visual encoders. We hypothesize that the variance between different visual encoders is mainly rooted in the learned feature representations during pre-training. It is an interesting future work to design an optimal pre-training strategy for continual learning that does not need supervised data.

# E THEORETICAL JUSTIFICATIONS

# E.1 PRELIMINARY

In this section, we first give some primary definitions and notations of statistical hypothesis testing, rejection region and uniformly most powerful (UMP) test.

Definition 1 (statistical hypothesis testing and rejection region) Consider testing a null hypothesis $H _ { 0 } : \theta \in \Theta _ { 0 }$ against an alternative hypothesis ${ \bar { H } } _ { 1 } : \theta \in { \bar { \Theta _ { 1 } } } ,$ , where $\Theta _ { 0 }$ and $\Theta _ { 1 }$ are subsets of the parameter space Θ and $\begin{array} { r } { \Theta _ { 0 } \cap \Theta _ { 1 } = \varnothing . } \end{array}$ . A test consists of a test statistic $T ( X )$ , which is a function of the data x, and a rejection region ${ \mathcal { R } } ,$ which is a subset of the range of T . If the observed value $t o f \dot { T }$ falls in R, we reject $H _ { 0 }$ .

Type I error occurs when we reject a true null hypothesis $\mathcal { H } _ { \mathrm { 0 } }$ . The probability of making Type I error is usually denoted by α. Type II error occurs when we fail to reject a false null hypothesis $\mathcal { H } _ { \mathrm { 0 } }$ . The level of significance α is the probability we are willing to risk rejecting $\mathcal { H } _ { \mathrm { 0 } }$ when it is true. Typically $\alpha = 0 . 1 , 0 . 0 5 , 0 . 0 1$ are used.

Definition 2 (UMP test) Denote the power function $\beta _ { \mathcal { R } } ( \theta ) = P _ { \theta } ( T ( x ) \in \mathcal { R } )$ , where $P _ { \theta }$ denotes the probability measure when θ is the true parameter. A test with a test statistic T and rejection region R is called a uniformly most powerful (UMP) test at significance level α if it satisfies two conditions:

$\begin{array} { r } { I . \ \operatorname* { s u p } _ { \theta \in \Theta _ { 0 } } \beta _ { \mathcal { R } } ( \theta ) \leq \alpha . } \end{array}$   
2. ∀θ $\mathbf { \beta } \in \Theta _ { 1 } , \beta _ { \mathcal { R } } ( \theta ) \geq \beta _ { R ^ { \prime } } ( \theta )$ for every other test t′ with rejection region R′ satisfying the first condition.

From the definition we see, the UMP test ensures that the probability of Type I error is less than α (with the first condition), while achieves the lowest Type II error (with the second condition). Therefore, UMP is considered an optimal solution in statistical hypothesis testing.

# E.2 PROOF OF THEOREM 4.1

Lemma E.1 (Neyman & Pearson, 1933) Let $\{ X _ { 1 } , X _ { 2 } , . . . , X _ { n } \}$ be a random sample with likelihood function $L ( \theta )$ . The UMP test of the simple hypothesis $H _ { 0 } : \dot { \theta } = \theta _ { 0 }$ against the simple hypothetis $H _ { a } : \theta = \theta _ { a }$ at level α has a rejection region of the form:

$$
\frac {L (\theta_ {0})}{L (\theta_ {a})} <   k
$$

where k is chosen so that the probability of a type I error is α.

Now the proof of Theorem 4.1 is straightforward. From Lemma E.1, the UMP test for Equation (4) in the main text has a rejection region of the form:

$$
\frac {p _ {t} (\pmb {x})}{p _ {t ^ {c}} (\pmb {x})} <   \lambda_ {0}
$$

where $\lambda _ { 0 }$ is chosen so that the probability of a type I error is α.

# E.3 PROOF OF THEOREM 4.2

Note that the AUC is computed as the area under the ROC curve. A ROC curve shows the trade-off between true positive rate (TPR) and false positive rate (FPR) across different decision thresholds. Therefore,

$$
A U C = \int_ {0} ^ {1} (T P R) \mathrm{d} (F P R) \tag {12}
$$

$$
= \int_ {0} ^ {1} (1 - F P R) \mathrm{d} (T P R) \tag {13}
$$

$$
= \int_ {0} ^ {1} \beta_ {\mathcal {R}} (\theta_ {t ^ {c}}) \mathrm{d} (1 - \beta_ {\mathcal {R}} (\theta_ {t})) \tag {14}
$$

$$
= \int_ {0} ^ {1} \beta_ {\mathcal {R}} (\theta_ {t ^ {c}})   \mathrm{d} \beta_ {\mathcal {R}} (\theta_ {t}) \tag {15}
$$

where FPR and TPR are false positive rate and true positive rate. Therefore, an optimal AUC requires UMP test of any given level $\alpha = \beta _ { \mathcal { R } } ( \theta _ { t } )$ except on a null set.

# E.4 DISTANCE-BASED OOD DETECTORS ARE IND DENSITY ESTIMATORS

In this section, we show that $S _ { M D } ( { \pmb x } )$ (MD: Mahalanobis distance) and $S _ { K N N } ( { \pmb x } )$ (KNN: k-nearest neighbor) defined in Equation (16) and Equation (17) are IND (in-distribution) density estimators under different assumptions (We omit the superscript (t) for simplicity):

$$
S _ {M D} (\boldsymbol {x}) = 1 / \min _ {c \in \mathcal {Y}} (\boldsymbol {z} - \boldsymbol {\mu} _ {c}) ^ {T} \boldsymbol {\Sigma} ^ {- 1} (\boldsymbol {z} - \boldsymbol {\mu} _ {c}), \tag {16}
$$

$$
S _ {K N N} (\boldsymbol {x}; \mathcal {D}) = - | | \boldsymbol {z} ^ {*} - k N N (\boldsymbol {z} ^ {*}; \mathcal {D} ^ {*}) | | _ {2}. \tag {17}
$$

In Equation (16), $\pmb { \mu } _ { c }$ is the class centroid for class c and Σ is the global covariance matrix, which are estimated on IND training corpus D. In Equation $( 1 7 ) , | | \cdot | | _ { 2 }$ is Euclidean norm, $z ^ { * } = z / | | z | | _ { 2 }$ denotes the normalized feature z, and $\mathcal { D } ^ { * }$ denotes the set of normalized features from training set D. $k N N ( z ^ { * } ; \mathcal { D } ^ { * } )$ denotes the k-nearest neighbor of $z ^ { * }$ in set $\mathcal { D } ^ { * }$ .

Assume we have a feature encoder $\phi : \mathcal { X }  \mathbb { R } ^ { m }$ , and in training time we empirically observe n IND samples $\left\{ \phi ( \pmb { x } _ { 1 } ) , \phi ( \pmb { x } _ { 2 } ) . . . \phi ( \pmb { x } _ { n } ) \right\}$ .

Analysis of the MD score. Denote Σ to be the covariance matrix of $\phi ( { \pmb x } )$ . The final feature we extract from data x is:

$$
\pmb {z} (\pmb {x}) = A ^ {- 1} \phi (\pmb {x})
$$

where $A A ^ { T } = \Sigma$ . Note that the covariance of z is I.

Given a class label $c ,$ we assume the distribution $z ( x | c )$ follows a Gaussian $\mathcal { N } ( A ^ { - 1 } \mu _ { c } , \mathcal { T } )$ . Immediately we have $\pmb { \mu } _ { c }$ to be the class centroid for class c under the maximum likelihood estimation. We can now clearly address the relation between MD score and IND density $( p ( { \pmb x } ) )$ :

$$
S _ {M D} (\boldsymbol {x}) = 1 / (- 2 \max _ {c \in \mathcal {Y}} (\ln p (\boldsymbol {x} | c)) - m \ln 2 \pi)
$$

Analysis of KNN score. The normalized feature $z ( x ) = \phi ( { \pmb x } ) / | | \phi ( { \pmb x } ) | | _ { 2 }$ is used for OOD detection. The probability density of z can be attained by:

$$
p (\boldsymbol {z}) = \lim _ {r \to 0} \frac {p (\boldsymbol {z} ^ {\prime} \in B (\boldsymbol {z} , r))}{| B (z , r) |}
$$

where $B ( z , r ) = \{ z ^ { \prime } : | | z ^ { \prime } - z | | _ { 2 } \leq r \wedge | | z ^ { \prime } | | = 1 \}$

Assuming each sample $z ( x _ { i } )$ is i.i.d with a probability mass $1 / n ,$ the density can be estimated by KNN distance. Specifically, $r = | | z - k \hat { N } N ( z ) | | _ { 2 } , \bar { p } ( z ^ { \prime } \in \dot { B } ( z , r ) ) = k \bar { / } n$ and $| B ( z , r ) | =$ $\frac { \pi ^ { ( m - 1 ) / 2 } } { \Gamma ( \frac { m - 1 } { 2 } + 1 ) } r ^ { m - 1 } + o ( r ^ { m - 1 } )$ , where Γ is Euler’s gamma function. When n is large and $k / n$ is 2 small, we have the following equations:

$$
p (\pmb {x}) \approx \frac {k \Gamma (\frac {m - 1}{2} + 1)}{\pi^ {(m - 1) / 2} n r ^ {m - 1}}
$$

$$
S _ {K N N} (\pmb {x}) \approx - (\frac {k \Gamma (\frac {m - 1}{2} + 1)}{\pi^ {(m - 1) / 2} n}) ^ {\frac {1}{m - 1}} (p (\pmb {x})) ^ {- \frac {1}{m - 1}}
$$

Recall that the CIL methods based on the TIL+OOD paradigm (i.e., MORE and ROW) use $S _ { M D } ( { \pmb x } )$ to compute the task-prediction probability. As analyzed above, the MD score is in fact IND density estimator, which means $S _ { M D } ( { \pmb x } )$ measures the likelihood of the task distribution $\mathcal { P } _ { t }$ . Therefore, the TIL+OOD methods ignores the likelihood of the distribution of other tasks $( \mathcal { P } _ { t ^ { c } } )$ , which may fail to make the accurate task prediction. We put the detailed analysis in Appendix H.

# F ADDITIONAL DETAILS ABOUT TPL

# F.1 COMPUTATION OF THE MD SCORE

Mahalanobis distance score (MD) is an OOD score function initially proposed by Lee et al. (2018b), which is defined as:

$$
S _ {M D} (\boldsymbol {x}) = 1 / \min _ {c \in \mathcal {Y}} \left(\left(h (\boldsymbol {x}) - \boldsymbol {\mu} _ {c}\right) ^ {T} \boldsymbol {\Sigma} ^ {- 1} (h (\boldsymbol {x}) - \boldsymbol {\mu} _ {c})\right),
$$

where $h ( { \pmb x } )$ is the feature extractor of a tested OOD detection model $\mathcal { M } , \pmb { \mu } _ { c }$ is the centroid for a class c and Σ is the covariance matrix. The estimations of $\pmb { \mu } _ { c }$ and Σ are defined by

$$
\boldsymbol {\mu} _ {c} = \frac {1}{N _ {c}} \sum_ {\boldsymbol {x} \in \mathcal {D} _ {t r a i n} ^ {c}} h (\boldsymbol {x}),
$$

$$
\boldsymbol {\Sigma} = \frac {1}{N} \sum_ {c \in | \mathcal {Y} |} \sum_ {\boldsymbol {x} \in \mathcal {D} _ {t r a i n} ^ {c}} (h (\boldsymbol {x}) - \boldsymbol {\mu} _ {c}) (h (\boldsymbol {x}) - \boldsymbol {\mu} _ {c}) ^ {T},
$$

where $\mathcal { D } _ { t r a i n } ^ { c } : = \{ \pmb { x } : ( \pmb { x } , y ) \in \mathcal { D } _ { t r a i n } , y = c \} , \mathcal { D } _ { t r a i n }$ is the training set, $N$ is the total number of training samples, and $N _ { c }$ is the number of training samples belonging to class c.

In the CIL setting, we have to compute $\pmb { \mu } _ { c }$ and Σ for each task (assuming all classes in the task have the same covariance matrix) with trained task-specific model $\mathcal { M } ^ { ( t ) }$ . Specifically, after training on the t-th task dataset $\mathcal { D } ^ { ( t ) }$ , we compute:

$$
\boldsymbol {\mu} _ {c} ^ {(t)} = \frac {1}{N _ {c}} \sum_ {(\boldsymbol {x}, c) \in \mathcal {D} ^ {(t)}} h (\boldsymbol {x}; \phi^ {(t)}), \quad \forall c \in \mathcal {Y} ^ {(t)} \tag {18}
$$

$$
\boldsymbol {\Sigma} ^ {(t)} = \frac {1}{| \mathcal {D} ^ {(t)} |} \sum_ {c \in | \mathcal {Y} ^ {(t)} |} \sum_ {(\boldsymbol {x}, c) \in \mathcal {D} ^ {(t)}} (h (\boldsymbol {x}; \phi^ {(t)}) - \boldsymbol {\mu} _ {c} ^ {(t)}) (h (\boldsymbol {x}; \phi^ {(t)}) - \boldsymbol {\mu} _ {c} ^ {(t)}) ^ {T} \tag {19}
$$

We put the memory budget analysis of the saved class centroids $\mu _ { c } ^ { ( t ) }$ and co-variance matrices $\pmb { \Sigma } ^ { ( t ) }$ in Appendix I.3.

# F.2 COMPUTATION OF LOGIT-BASED SCORES

In Section 5.3, we compare the performance of TPL with different logit-based scores MSP, EBO, and MLS. They are defined as follows:

$$
S _ {M S P} ^ {(t)} (\boldsymbol {x}) = \max _ {j = 1} ^ {| \mathcal {Y} _ {t} |} \text { softmax } \left(f (h (\boldsymbol {x}; \phi^ {(t)}); \theta^ {(t)})\right) \tag {20}
$$

$$
S _ {E B O} ^ {(t)} (\boldsymbol {x}) = \log \sum_ {j = 1} ^ {| \mathcal {Y} _ {t} |} \left(\exp \left\{f (h (\boldsymbol {x}; \phi^ {(t)}); \theta^ {(t)}) \right\}\right) \tag {21}
$$

$$
S _ {M L S} ^ {(t)} (\boldsymbol {x}) = \max _ {j = 1} ^ {| \mathcal {Y} _ {t} |} \left(f (h (\boldsymbol {x}; \phi^ {(t)}); \theta^ {(t)})\right) \tag {22}
$$

In traditional OOD detection works, they are tested effective in estimating the probability of $\mathbf { \ddot { \mu } } _ { \mathbf { x } }$ belongs to IND classes". Thus we can adopt them to design our TPL method to estimate the probability of “x belongs to task $t "$ .

# F.3 PSEUDO-CODE

To improve reproducibility, we provide the detailed pseudo-code for computing $S _ { T P L } ( \pmb { x } )$ (using Equation (9) in the main text) as Algorithm 1. Then we give the pseudo-code for the CIL training as Algorithm 2 and testing for TPL as Algorithm 3.

Algorithm 1 Compute TPL Score with the t-th Task-specific Model $\mathcal { M } ^ { ( t ) }$

Input: $B u f ^ { \star } \colon$ replay buffer data without classes of task $t ; x \colon$ test sample; t: task-id; $\mathcal { M } ^ { ( t ) } \colon$ : the trained t-th task model $\mathcal { M } ^ { ( t ) }$ with feature extractor $h ( \boldsymbol { x } ; \boldsymbol { \phi } ^ { ( t ) } )$ and classifier $f ( \pmb { x } ; \theta ^ { ( t ) } ) ; \{ \pmb { \mu } _ { c } ^ { ( t ) } \} _ { c \in \mathcal { V } ^ { ( t ) } } :$ : pre-computed class centroids for task $t ; \pmb { \Sigma } ^ { ( t ) }$ : pre-computed covariance matrix for task t; k: KNN hyper-parameter; $1 / \beta _ { 1 } ^ { ( t ) }$ : pre-computed empirical mean of $S _ { M L S } ( \pmb { x } ) ; 1 / \beta _ { 2 } ^ { ( t ) }$ : pre-computed empirical mean of $S _ { M D } ( { \pmb x } )$ .

Return: TPL Score $S _ { T P L } ( \pmb { x } )$   
1: $S_{MLS}(\boldsymbol{x}) \leftarrow \max_{c \in \mathcal{Y}^{(t)}} f(h(\boldsymbol{x}; \phi^{(t)}); \theta^{(t)})_c$ 2: $S_{MD}(\boldsymbol{x}) \leftarrow 1 / (\min_{c \in \mathcal{Y}^{(t)}} ((h(\boldsymbol{x}; \phi^{(t)}) - \boldsymbol{\mu}_c)^T \boldsymbol{\Sigma}^{-1}(h(\boldsymbol{x}; \phi^{(t)}) - \boldsymbol{\mu}_c)))$ 3: $S_{MLS}(\boldsymbol{x}) \leftarrow S_{MLS}(\boldsymbol{x}) * \beta_1^{(t)}$ 4: $S_{MD}(\boldsymbol{x}) \leftarrow S_{MD}(\boldsymbol{x}) * \beta_2^{(t)}$ 5: $z \leftarrow h(\boldsymbol{x}; \phi^{(t)}) / ||h(\boldsymbol{x}; \phi^{(t)})||_2$ 6: for $\hat{x}_i$ in $Buf^\star$ do
7: $z_i \leftarrow h(\hat{x}_i; \phi^{(t)}) / ||h(\hat{x}_i; \phi^{(t)})||_2$ 8: $d_i \leftarrow ||z_i - z||_2$ 9: end for
10: $\{d_{i_j}\}_{j=1}^{|Buf^\star|} \leftarrow sorted(\{d_i\}_{i=1}^{|Buf^\star|})$ 11: $S_{TPL}(\boldsymbol{x}) \leftarrow -\log[\exp\{-S_{MLS}(\boldsymbol{x})\} + \exp\{-S_{MD}(\boldsymbol{x}) - d_{i_k}\}]$

Algorithm 2 CIL Training with TPL   
1: Initialize an empty replay buffer Buf
2: for training data $\mathcal{D}^{(t)}$ of each task do
3:    for each batch $(x_{j}, y_{j}) \subset \mathcal{D}^{(t)} \cup Buf$ , until converge do
4:    Minimize Equation (2) (in the main text) and update the parameters with HAT
5:    end for
6:    Compute $\{\mu_{c}^{(t)}\}_{c \in \mathcal{Y}^{(t)}}$ using Equation (18)
7:    Compute $\Sigma^{(t)}$ using Equation (19)
8:    Compute $1/\beta_{1}^{(t)}, 1/\beta^{(t)}$ using Equation (10)
9:    Update Buf with $\mathcal{D}^{(t)}$ 10: end for
11: Train the calibration parameters $\{(\tilde{\sigma}_{1}^{(t)}, \tilde{\sigma}_{2}^{(t)})\}_{t=1}^{T}$ following Appendix B

Algorithm 3 CIL Testing with TPL   
Input: test sample x
Return: predicted class $\hat{c}$ 1: Compute $S_{TPL}(\boldsymbol{x};t)$ with each task model $\mathcal{M}^{(t)}$ using Algorithm 1.
2: Precision with Equation (3): $\hat{c} = \arg\max_{c,t} \sigma_{1}^{(t)} \cdot \left[ \text{softmax}(f(h(\boldsymbol{x};\phi^{(t)});\theta^{(t)})) \right]_{j} \cdot S(\boldsymbol{x};t) + \sigma_{2}^{(t)}$

# G DETAILS OF HAT

# G.1 TRAINING

For completeness, we briefly describe the hard attention mechanism of HAT (Serra et al., 2018) used in TPL. In learning the task-specific model $\mathcal { M } ^ { ( t ) }$ for each task, TPL at the same time trains a mask for each adapter layer. To protect the shared feature extractor from previous tasks, their masks are used to block those important neurons so that the new task learning will not interfere with the parameters learned for previous tasks. The main idea is to use sigmoid to approximate a 0-1 gate function as hard attention to mask or unmask the information flow to protect parameters learned for each previous task.

The hard attention at layer l and task t is defined as:

$$
\pmb {a} _ {l} ^ {(t)} = s i g m o i d (s \cdot \pmb {e} _ {l} ^ {(t)}),
$$

where s is a temperature scaling term, sigmoid(·) denotes the sigmoid function, and $e _ { l } ^ { ( t ) }$ is a learnable embedding for task t. The attention is element-wise multiplied to the ouptut $h _ { l }$ of layer l as

$$
\boldsymbol {h} _ {l} ^ {\prime} = \boldsymbol {a} _ {l} ^ {(t)} \otimes \boldsymbol {h} _ {l}
$$

The sigmoid function converges to a 0-1 binary gate as s goes to infinity. Since the binary gate is not differentiable, a fairly large s is chosen to achieve a differential pseudo gate function. The pseudo binary value of the attention determines how much information can flow forward and backward between adjacent layers. Denote $\pmb { h } _ { l } = R e L U ( \pmb { W } _ { l } \pmb { h } _ { l - 1 } + \pmb { b } _ { l } )$ , where $R e L U ( \cdot )$ is the rectifier function. For neurons of attention $\pmb { a } _ { l } ^ { ( t ) }$ with zero values, we can freely change the corresponding parameters in $W _ { l }$ and $b _ { l }$ without interfering the output $\pmb { h } _ { l } ^ { \prime } .$ . The neurons with non-zero mask values are necessary to perform the task, and thus need a protection for catastrophic forgetting (CF).

Specifically, during learning task t, we modify the gradients of parameters that are important in performing the previous tasks $1 , 2 , . . . , t - 1$ so they are not interfered. Denote the accumulated mask by

$$
\boldsymbol {a} _ {l} ^ {(<   t)} = \max (\boldsymbol {a} _ {l} ^ {(<   t - 1)}, \boldsymbol {a} _ {l} ^ {(t - 1)}),
$$

where max $\cdot ( \cdot , \cdot )$ is an element-wise maximum and the initial mask ${ \pmb a } _ { l } ^ { ( 0 ) }$ is defined as a zero vector. $\mathbf { \Delta } _ { \mathbf { } \mathbf { } a _ { l } } ( \mathbf { < } t )$ is a collection of mask values at layer l where a neuron has value 1 if it has ever been activated previously. The gradient of parameter $w _ { i j , l }$ is modified as

$$
\nabla w _ {i j, l} ^ {\prime} = (1 - \min (a _ {i, l} ^ {(<   t)}, a _ {j, l - 1} ^ {(<   t)})) \nabla w _ {i j, l},
$$

where $a _ { i , l } ^ { ( < t ) }$ is the i-th unit of $\mathbf { } a _ { l } ^ { ( < t ) }$ . The gradient flow is blocked if both neurons i in the current layer and j in the previous layer have been activated. We apply the mask for all layers of adapters except the last layer. The parameters in last layer do not require protection as they are task-specific parameters.

A regularization is introduced to encourage sparsity in $\pmb { a } _ { l } ^ { ( t ) }$ and parameter sharing with $\mathbf { \boldsymbol { a } } _ { l } ^ { ( < t ) }$ . The capacity of a network depletes when $\mathbf { } a _ { l } ^ { ( < t ) }$ becomes an all-one vector in all layers. Despite a set of new neurons can be added in network at any point in training for more capacity, we utilize resources more efficiently by minimizing the loss:

$$
\mathcal {L} _ {r e g} = \frac {\sum_ {l} \sum_ {i} a _ {i , l} ^ {(t)} (1 - a _ {i , l} ^ {(<   t)})}{\sum_ {l} \sum_ {i} (1 - a _ {i , l} ^ {(<   t)})},
$$

The intuition of this term is to regularize the number of masked neurons. Then the loss of HAT defined as the second term from the R.H.S. of $\operatorname { E q } \left( 7 \right)$ in the main text is:

$$
\mathcal {L} _ {H A T} = \boldsymbol {\mu} \cdot \mathcal {L} _ {r e g},
$$

where $\mu$ is a hyper-parameter to balance the optimization of classification objective and HAT regularization.

# G.2 INFERENCE

In CIL without a task identifier during inference, we are required to forward input data across each task to derive task-specific features $h ^ { ( t ) }$ for each task $t = 1 , 2 , \cdots , T$ . This approach can result in computation overhead, especially with extended task sequences. This section proposes to use parallel computing (PC) to mitigate this by achieving comparable time efficiency to one-pass CL methods, albeit with a trade-off of increased memory usage by a factor of $T$ compared to the one-pass CL methods for latent feature storage, a vector of 384 floating point numbers.

Consider the model M comprising two components: a feature extractor $h ( \cdot )$ and a classifier $f ( \cdot )$ . Unlike the standard model, where both extractor and classifier are universal across tasks, our model uses a shared feature extractor with task-specific classifiers for each task. We analyze the computational costs for each component separately.

For the feature extractor, break it down into L layers. Each layer l involves an affine transformation (with weight $W _ { l }$ and bias $b _ { l } )$ followed by an activation function, specifically $R e L U ( \cdot ) . ^ { 8 }$ In the standard model, computation at each layer l follows:

$$
\boldsymbol {h} _ {l} = \operatorname{ReLU} \left(\boldsymbol {W} _ {l} \boldsymbol {h} _ {l - 1} + \boldsymbol {b} _ {l}\right) \tag {23}
$$

For our model, the computation is extended to:

$$
\boldsymbol {h} _ {l} ^ {(t)} = \boldsymbol {a} _ {l} ^ {(t)} \otimes R e L U (\boldsymbol {W} _ {l} \boldsymbol {h} _ {l - 1} ^ {(t)} + \boldsymbol {b} _ {l}), \quad t = 1, 2, \dots , T \tag {24}
$$

Here, $\pmb { a } _ { l } ^ { ( t ) }$ denotes the stored hard attention at layer l for task t. Comparing the equations, the additional operation inby vectorizing the sets $\{ h _ { l } ^ { ( t ) } \} _ { t = 1 } ^ { T } , \{ h _ { l - 1 } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ ment-, and $\{ \pmb { a } _ { l } ^ { ( t ) } \} _ { t = 1 } ^ { T }$ duct, which is into matrices $H _ { l } , H _ { l - 1 }$ para, and $A _ { l } .$ zable This parallelization achieves near-equivalent time consumption to the standard model, with the trade-off of a T -fold increase in memory usage as our feature $\bar { \pmb { H _ { l } } }$ and $\pmb { H } _ { l - 1 }$ are $T$ times larger compared to $h _ { l }$ and $\boldsymbol { h } _ { l - 1 }$ .

For the classifier, each task employs a simple affine transformation (weight $\mathbf { \mathbf { } } W ^ { ( t ) }$ and bias $\mathbf { \delta } _ { b } ( t ) )$ . The standard model uses $W \in \mathbb { R } ^ { H \times C }$ and $b \in \mathbb { R } ^ { C }$ , where H and $C$ represent the hidden size and class count, respectively. The standard model’s computation is:

$$
\operatorname{logits} = \boldsymbol {W} \boldsymbol {h} _ {L} + \boldsymbol {b} \tag {25}
$$

In contrast, our method involves $T$ task-specific classifiers, each with weight $W ^ { ( t ) } \in \mathbb { R } ^ { H \times \frac { C } { T } }$ and bias ${ \pmb b } ^ { ( t ) } \in \mathbb { R } ^ { \frac { C } { T } }$ . The task-specific logits are calculated as:

$$
\operatorname{logits} ^ {(t)} = \boldsymbol {W} ^ {(t)} \boldsymbol {h} _ {L} ^ {(t)} + \boldsymbol {b} ^ {(t)} \tag {26}
$$

By vectorizing task-specific weights, features, and biases, time consumption is maintained at levels comparable to the standard model, and memory consumption remains unaffected.

In conclusion, our approach achieves similar time efficiency as the standard model in CL inference, at the expense of increased runtime memory for storing task-specific features. This trade-off also allows flexibility between memory usage and time consumption by adjusting the level of parallelism, offering a balance between runtime duration and memory requirements. For example, one can reduce the running time memory by forwarding the input more times to the model with a lower parallelism level each time.

# G.3 REMARKS

It is important to note that our method can also leverage some other TIL methods to prevent CF other than HAT. The reason we chose HAT is to make it easy to compare with previous techniques (e.g.,

MORE and ROW) based on the same setting. For example, we can also exploit SupSup (Wortsman et al., 2020), which has already been applied in Kim et al. (2022b) to construct an effective TIL+OOD method and the empirical performance are similar to that using HAT. Furthermore, our method is also compatible with other architecture-based TIL methods such as PNN (Rusu et al., 2016), PackNet (Mallya & Lazebnik, 2018) or Ternary Masks (Masana et al., 2021).

# H VISUALIZATION OF TASK DISTRIBUTION

The key theoretical analysis behind TPL, theorems 4.1 and 4.2, suggest that an accurate estimate of $\mathcal { P } _ { t ^ { c } }$ (the distribution of the other tasks, i.e., the complement) is important. If we only estimate $\mathcal { P } _ { t }$ (the feature distribution of task t) when doing task prediction and assume $\mathcal { P } _ { t ^ { c } }$ to be a uniform distribution (which was done by existing methods as analyzed in Appendix E.4), there will be potential risks as shown by the toy example on 1D Gaussian in Section 4.1.

Intuitively, the failure happens when $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ have some overlap. A test case $\scriptstyle { \mathbf { \mathscr { x } } } _ { 1 }$ may have higher likelihood in $\mathcal { P } _ { t }$ than another test case $\mathbf { { x } } _ { 2 } ,$ but if it gets even higher likelihood in $\mathcal { P } _ { t ^ { c } }$ , then $\mathbf { \delta x } _ { 1 }$ will be less likely to be drawn from $\mathcal { P } _ { t }$ . In this section, we plot $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ to demonstrate this phenomenon.

Recall that in our CIL scenario, we compute the likelihood ratio for each task to estimate the task prediction probability. For each task t, we compute the likelihood of input data x under the task distribution $\mathcal { P } _ { t }$ , and use the saved replay data from the other tasks to estimate $\mathcal { P } _ { t ^ { c } }$ . Notice that as we do not conduct task prediction in training, all the discussion here is about inference or test time and we have the small among of saved data from all the learned tasks in the memory buffer. Our goal is to analyze whether $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ have some overlap.

Specifically, we draw the feature distribution of the five tasks on C10-5T after all tasks are learned under the pre-training setting. To facilitate an accurate estimation of $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ , we use the whole training set of CIFAR-10 to prepare the extracted feature as it contains more data. The vanilla features are high-dimensional vectors in $\mathbb { R } ^ { 3 8 4 }$ , and we use Principled Component Analysis (PCA) (Wold et al., 1987) to project the vectors into $\mathbb { R } ^ { 2 }$ . We then use Kernel Density Estimation (KDE) (Terrell & Scott, 1992) to visualize the distribution (density) of the data. The figures are shown as follows:

![](images/df1aa806a9b71ed60e7313a710b7aad71f6f4e10cc3046d9767db5a18242d596.jpg)

<details>
<summary>heatmap</summary>

| Category                  | Value |
| ------------------------- | ----- |
| Distribution of Task 1   | 1     |
| Distribution of other tasks | 2     |
</details>

![](images/7e6145823512cc4d94c1f12072becf715dc28601a0f8e21b74ea31e08ff8db7f.jpg)

<details>
<summary>contour</summary>

| Category                  | Value |
| ------------------------- | ----- |
| Distribution of Task 2   | 100   |
| Distribution of other tasks | 100   |
</details>

![](images/f3458765eb6080971a62e601cbfeeb5f3f622bd5a7b36a1c72950a37c1669835.jpg)

<details>
<summary>contour</summary>

| Category                  | Value |
| ------------------------- | ----- |
| Distribution of Task 3   | 100   |
| Distribution of other tasks | 100   |
</details>

![](images/0a92a364ae66f399783eb3a345494d67e0ddba80a0cb1152d7465bbca064c8b8.jpg)

<details>
<summary>heatmap</summary>

| Category                  | Value |
| ------------------------- | ----- |
| Distribution of Task 5   | 100   |
| Distribution of other tasks | 100   |
</details>

![](images/8c2602226cad95a6b4af7ad748401b17b67832db91ba77146ba8f4593c94f41e.jpg)

<details>
<summary>contour</summary>

| Category              | Value |
| --------------------- | ----- |
| Distribution of Task 4 | 4     |
| Distribution of other tasks | 3     |
</details>

Figure 4: Visualization of feature distribution of Task $\textit { t } ( t { = } 1 , 2 , 3 , 4 , 5 )$ data and the other 4 tasks. We use the trained task-specific feature extractor $h ( \boldsymbol { x } ; \boldsymbol { \phi } ^ { ( t ) } )$ to extract features from the the training data that belongs to task t (which represent $\mathcal { P } _ { t } )$ and the training data that belongs to the other 4 tasks (which represent $\mathcal { P } _ { t ^ { c } } )$ .

Notice that the task distributions $\{ \mathcal { P } _ { t } \}$ are bimodal as there are two classes in each task in C10-5T. The most interesting observation is that the distribution of the other tasks $\{ \mathcal { P } _ { t ^ { c } } \}$ have overlap with the task distribution $\{ \mathcal { \hat { P } } _ { t } \}$ . This indicates that the failure may happen as we analyzed above. For example, we draw a demonstrative failure case of the existing TIL+OOD methods (i.e., MORE and ROW) in Task 1 prediction in Figure 5. In this Figure, TIL+OOD methods will compute the task prediction probability solely based on the high likelihood of $\mathcal { P } _ { t } \left( t = 1 \right)$ , while our TPL will consider both high likelihood of $\mathcal { P } _ { t }$ and low likelihood of $\mathcal { P } _ { t ^ { c } }$ . In this case, the red star has higher likelihood in $\mathcal { P } _ { t }$ than the green star (0.9 v.s. 0.4). However, the likelihood ratio between $\mathcal { P } _ { t }$ and $\bar { \mathcal { P } } _ { t ^ { c } }$ of the red star is lower than the green star (3 v.s. 20). Therefore, using likelihood ratio between $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ is crucial in estimating the task prediction probability.

![](images/3083b98b88f315dd7cd46225d4789860ce25f76870813a0fe8bd7f55c6205d5c.jpg)

<details>
<summary>heatmap</summary>

| Category              | Likelihood of Task 1 | Likelihood of the other tasks | Likelihood ratio |
| --------------------- | -------------------- | ------------------------------ | ---------------- |
| Task 1                | 0.4                  | 0.02                           | 20               |
| Other tasks           | 0.9                  | 0.3                            | 3                |
</details>

Figure 5: A failure case of TIL+OOD methods that predict the task based on the likelihood of $\mathcal { P } _ { t }$ (e.g., MORE and ROW). In the figure, the red star has higher likelihood in $\mathcal { P } _ { t } ~ ( t = 1 )$ than the green star. However, the likelihood ratio between $\bar { \mathcal { P } } _ { t }$ and $\mathcal { P } _ { t ^ { c } }$ of the red star is lower than the green star. The correct choice is to accept the green star to be from Task 1 instead of the red star.

# I IMPLEMENTATION DETAILS, NETWORK SIZE AND RUNNING TIME

# I.1 IMPLEMENTATION DETAILS OF BASELINES

Datasets. We use three popular image datasets. (1) CIFAR-10 (Krizhevsky & Hinton, 2010) consists of images of 10 classes with 50,000 / 10,000 training / testing samples. (2) CIFAR-100 (Krizhevsky et al., 2009) consists of images with 50,000 / 10,000 training / testing samples. (3) Tiny-ImageNet (Le & Yang, 2015) has 120,000 images of 200 classes with 500 / 50 images per class for training / testing.

Implementation of CIL baselines (pre-trained setting). For CIL baselines, we follow the experiment setups as reported in their official papers unless additionally explained in Section 5.1. For the regularization hyper-parameter $\mu$ and temperature annealing term s (see Appendix G) used in HAT, we follow the baseline MORE and use $\mu = 0 . 7 5$ and $s = 4 0 0$ for all experiments as recommended in (Serra et al., 2018). For the approaches based on network expansion (DER, BEEF, FOSTER), we expand the network of adapters when using a pre-trained backbone. For ADAM, we choose the ADAM(adapter) version in their original paper, which is the best variant of ADAM. As some of the baselines are proposed to continually learn from scratch, we carefully tune their hyper-parameters to improve the performance to ensure a fair comparison. The implementation details of baselines under non-pre-training setting are shown in Appendix D.2.

Hyper-parameter tuning. Apart from $\beta _ { 1 }$ and $\beta _ { 2 }$ discussed in Section 5.3, the only hyper-parameters used in our method $\mathrm { T P I }$ are $\gamma ,$ which is the temperature parameter for task-id prediction, and k, which is the hyper-parameter of $d _ { K N N } ( \pmb { x } , \pmb { B } \pmb { u } f ^ { \star } )$ in Equation (7). The value of $\gamma$ and k are searched from {0.01, 0.05, 0.10, 0.50, 1.0, 2.0, 5.0, 10.0} and {1, 2, 5, 10, 50, 100}, respectively. We choose $\gamma = 0 . 0 5$ , and $k = 5$ for all the experiments as they achieve the overall best results.

Training Details. To compare with the strongest baseline MORE and ROW, we follow their setup (Kim et al., 2022a; 2023) to set the training epochs as 20, 40, 15, 10 for CIFAR-10, CIFAR-100, T-5T, T-10T respectively. And we follow them to use SGD optimizer, the momentum of 0.9, the batch size of 64, the learning rate of 0.005 for C10-5T, T-5T, T-10T, C100-20T, and 0.001 for C100-10T.

# I.2 HARDWARE AND SOFTWARE

We run all the experiments on NVIDIA GeForce RTX-2080Ti GPU. Our implementations are based on Ubuntu Linux 16.04 with Python 3.6.

# I.3 COMPUTATIONAL BUDGET ANALYSIS

# I.3.1 MEMORY CONSUMPTION

We present the network sizes of the CIL systems (with a pre-trained network) after learning the final task in Table 12.

With the exception of SLDA and L2P, all the CIL methods we studied utilize trainable adapter modules. The transformer backbone consumes 21.6 million parameters, while the adapters require 1.2M for CIFAR-10 and 2.4M for other datasets. In the case of SLDA, only the classifier on top of the fixed pre-trained feature extractor is fine-tuned as it requires a fixed feature extractor for all tasks, while L2P maintains a prompt pool with 32k parameters. Additionally, each method requires some specific elements (e.g., task embedding for HAT), resulting in varying parameter requirements for each method.

As mentioned in Appendix F.1, our method also necessitates the storage of class centroids and covariance matrices. For each class, we save a centroid of dimension 384, resulting in a total of 3.84k, 38.4k, and 76.8k parameters for CIFAR-10, CIFAR-100, and TinyImageNet, respectively. The covariance matrix is saved per task, with a size of $3 8 4 \times 3 8 4$ . Then the parameter count of covariance matrix can be computed by T × 384 × 384 for a dataset with T tasks. Consequently, the total parameter count is 737.3k, 1.5M, 2.9M, 737.3k, and 1.5M for C10-5T, C100-10T, C100-20T, T-5T, and T-10T, respectively. It is worth noting that this consumption is relatively small when compared to certain replay-based methods like iCaRL and HAL, which require a teacher model of the same size as the training model for knowledge distillation.

Table 12: Network size measured in the number of parameters (# parameters) for each method without the memory buffer. 

<table><tr><td></td><td>C10-5T</td><td>C100-10T</td><td>C100-20T</td><td>T-5T</td><td>T-10T</td></tr><tr><td>OWM</td><td>24.1M</td><td>24.4M</td><td>24.7M</td><td>24.3M</td><td>24.4M</td></tr><tr><td>ADAM</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>PASS</td><td>22.9M</td><td>24.2M</td><td>24.2M</td><td>24.3M</td><td>24.4M</td></tr><tr><td> $HAT_{CIL}$ </td><td>24.1M</td><td>24.4M</td><td>24.7M</td><td>24.3M</td><td>24.4M</td></tr><tr><td>SLDA</td><td>21.6M</td><td>21.6M</td><td>21.6M</td><td>21.7M</td><td>21.7M</td></tr><tr><td>L2P</td><td>21.7M</td><td>21.7M</td><td>21.7M</td><td>21.8M</td><td>21.8M</td></tr><tr><td>iCaRL</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>A-GEM</td><td>26.5M</td><td>31.4M</td><td>31.4M</td><td>31.5M</td><td>31.5M</td></tr><tr><td>EEIL</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>GD</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>DER++</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>HAL</td><td>22.9M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td><td>24.1M</td></tr><tr><td>DER</td><td>27.7M</td><td>45.4M</td><td>69.1M</td><td>33.6M</td><td>45.5M</td></tr><tr><td>FOSTER</td><td>28.9M</td><td>46.7M</td><td>74.2M</td><td>35.8M</td><td>48.1M</td></tr><tr><td>BEEF</td><td>30.4M</td><td>48.4M</td><td>82.3M</td><td>37.7M</td><td>50.6M</td></tr><tr><td>MORE</td><td>23.7M</td><td>25.9M</td><td>27.7M</td><td>25.1M</td><td>25.9M</td></tr><tr><td>ROW</td><td>23.7M</td><td>26.0M</td><td>27.8M</td><td>25.2M</td><td>26.0M</td></tr><tr><td>TPL</td><td>23.7M</td><td>25.9M</td><td>27.7M</td><td>25.1M</td><td>25.9M</td></tr></table>

# I.3.2 RUNNING TIME

The computation for our method TPL is very efficient. It involves standard classifier training and likelihood raio score computation, which employed some OOD detection methods. The OOD score computation only involves mean, covariance computation, and KNN search, which are all very efficient with the Python packages $\mathtt { s c i k i t - l e a r n } ^ { \mathtt { j } }$ and $\mathtt { f a i s s } ^ { 1 0 }$ . We give the comparison in running time in Table 13. We use HAT as the base as MORE, ROW and TPL all make use of HAT and MORE and ROW are the strongest baselines.

Table 13: Average running time measured in minutes per task (min/T) for three systems. 

<table><tr><td></td><td>C10-5T</td><td>C100-10T</td><td>C100-20T</td><td>T-5T</td><td>T-10T</td></tr><tr><td> $HAT_{CIL}$ </td><td>17.8 min/T</td><td>17.6 min/T</td><td>9.4 min/T</td><td>28.0 min/T</td><td>9.48 min/T</td></tr><tr><td>MORE</td><td>20.6 min/T</td><td>23.3 min/T</td><td>11.7 min/T</td><td>32.8 min/T</td><td>11.2 min/T</td></tr><tr><td>ROW</td><td>21.8 min/T</td><td>25.2 min/T</td><td>12.6 min/T</td><td>34.1 min/T</td><td>11.9 min/T</td></tr><tr><td>TPL</td><td>20.7 min/T</td><td>23.3 min/T</td><td>11.7 min/T</td><td>32.5 min/T</td><td>11.2 min/T</td></tr></table>

# J LIMITATIONS

Here we discuss the limitations of our proposed method TPL.

First, our TPL method relies on the saved data in the memory buffer like traditional replay-based methods, which may have privacy concerns in some situations and also need extra storage. We will explore how to improve task-id prediction in the CIL setting without saving any previous data in our future work. Second, TPL uses a naive saving strategy that samples task data randomly to put in the memory buffer for simplicity. In our future work, we would also like to consider better buffer saving strategies (Jeeveswaran et al., 2023) and learning algorithm for buffer data (Bhat et al., 2022), which may enable more accurate likelihood ratio computation. Third, this paper focuses on the traditional offline CIL problem set-up. In this mode, all the training data for each task is available upfront when the task arrives and the training can take any number of epochs. Also, the label space ${ \mathcal { V } } ^ { ( t ) }$ of different tasks are disjoint. In online CIL (Guo et al., 2022), the data comes in a stream and there may not be clear boundary between tasks (called Blurry Task Setting (Bang et al., 2022)), where the incoming data labels may overlap across tasks. It is an interesting direction for our future work to explore how to adapt our method to exploit the specific information in this setting.