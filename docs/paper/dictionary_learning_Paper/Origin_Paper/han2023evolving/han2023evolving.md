# Evolving Dictionary Representation for Few-shot Class-incremental Learning

Xuejun Han , Yuhong Guo

Carleton University, Canada

xuejunhan@cmail.carleton.ca, yuhong.guo@carleton.ca

# Abstract

New objects are continuously emerging in the dynamically changing world and a real-world artificial intelligence system should be capable of continual and effectual adaptation to new emerging classes without forgetting old ones. In view of this, in this paper we tackle a challenging and practical continual learning scenario named few-shot class-incremental learning (FSCIL), in which labeled data are given for classes in a base session but very limited labeled instances are available for new incremental classes. To address this problem, we propose a novel and succinct approach by introducing deep dictionary learning which is a hybrid learning architecture that combines dictionary learning and visual representation learning to provide a better space for characterizing different classes. We simultaneously optimize the dictionary and the feature extraction backbone in the base session, while only finetune the dictionary in the incremental session for adaptation to novel classes, which can alleviate the forgetting on base classes compared to finetuning the entire model. To further facilitate future adaptation, we also incorporate multiple pseudo classes into the base session training so that certain space projected by dictionary can be reserved for future new concepts. The extensive experimental results on CIFAR100, miniImageNet and CUB200 validate the effectiveness of our approach compared to other SOTA methods.

# 1 Introduction

Standard machine learning models are normally trained on static datasets and only able to classify the pre-defined categories with identical distributions, which is hardly to adapt to the rapidly changing world where novel objects emerge all the time. Such models are typically fixed upon deployment and therefore lacks the flexibility and compatibility to accommodate new concepts. Moreover, tactless model finetuning on new emerging datasets tends to bring about catastrophic forgetting [McCloskey and Cohen, 1989; Goodfellow et al., 2014] on what has been learned in the past.

![](images/1f58231a5bb5808f2878faebbcf0ea6bd6a15329e84efca8ddbdfc7e6e150ebd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Base session (0) class 1"] --> B["Novel session (1) class 4"]
    C["Base session (0) class 2"] --> D["Novel session (1) class 5"]
    E["Base session (0) class 3"] --> F["Novel session (1) class 5"]
    G["Visual Feature Space visual"] --> H["Final Output"]
```
</details>

Figure 1: Few-shot class-incremental learning. The model is trained on sufficient labeled data from base classes and extended to cover novel classes with limited labeled data. The goal is to accurately classify all classes via a joint classifier. A commonly adopted paradigm to alleviate overfitting to new classes and forgetting on base classes for FSCIL is to freeze the feature extraction backbone after base session and only extend the classifiers’s label set. However by merely doing so the class-wise visual features extracted by backbone for base and novel classes are not well separated, thus leading to performance degradation in novel sessions.

To address this deficiency, the continual or incremental learning [Ring, 1994; Thrun, 1994] was proposed to enable the model to incrementally learn new information from stream format data. There are three fundamental scenarios in the literature: task-incremental, domain-incremental and classincremental learning and each of those has its own challenges [Gido M. van de Ven, 2022]. Being the most realistic setting, class-incremental learning (CIL) has been actively explored in resent works [Rebuffi et al., 2017; Castro et al., 2018; Hou et al., 2019; Wu et al., 2019; Pellegrini et al., 2020; Mai et al., 2021], in which the model is trained on a sequence of disjoint sets of classes and a unified classifier is employed for all the experienced classes. Standard CIL setting assumes that there is sufficient labeled data for new classes, which inevitably impedes its practical applications since it is expensive to collect a large amount of annotated data, not to mention that in some cases the unlabeled ones are also hard to obtain, such as images of rare species. Therefore, how to develop an effective incremental model in the absence of sufficient data for novel classes is worth to investigate.

Recently, few-shot class-incremental learning (FSCIL) has been proposed and attracted much interest from the research community [Tao et al., 2020; Zhang et al., 2021; Cheraghian et al., 2021b; Zhu et al., 2021; Peng et al., 2022; Zhou et al., 2022; Liu et al., 2022] to tackle the problem of data scarcity in CIL. Concretely, the model is initially trained on a large number of annotated data from base classes and then incrementally learns new classes with few-shot labeled data. There are two challenges involved in FSCIL: the forgetting on base classes due to the inaccessibility to previous data, and the overfitting to few-shot data of new classes as a result of data scarcity in novel sessions. An effective strategy to alleviate the issue of overfitting is to freeze the backbone after finishing the base training [Zhang et al., 2021], however this simple scheme relies entirely on base representation learning and lacks the compatibility and extensibility for future adaptation (see Figure 1). [Zhou et al., 2022] enhances the model compatibility by pre-assigning multiple virtual prototypes to the feature space in the base session but no further session specific adaptation is applied to improve the performance over new classes, which results in superior performance on base classes but suboptimal on new ones.

In this paper, we propose a novel and succinct FSCIL method based on deep dictionary learning, which we name by D-FSCIL and show in Figure 2. In this method, the deep feature extractor and the dictionary representation are trained collaboratively in the base session. Meanwhile, we generate a set of synthetic data points in the visual feature space as pseudo classes to enhance the forward compatibility of the dictionary. By optimizing the backbone and dictionary on extended class set, we expect the data representations of different classes to be more separable and the space reserved by pseudo classes to better incorporate new concepts. The dictionary learned on extended categories in the base session will be more generalizable and compatible for future adaptation. Afterwards, the backbone is frozen to alleviate the forgetting and overfitting issues and only the dictionary representation is mildly updated on new classes. Extensive experiments on CIFAR100, miniImageNet and CUB200 demonstrate the superiority of our method over the state-of-the-art.

# 2 Related Work

Few-shot Class-incremental Learning Few-shot classincremental learning as an extension of CIL has attracted a lot of attention recently. [Tao et al., 2020] proposes to employ the neural gas structure to preserve the topology of the feature manifold for different classes. [Cheraghian et al., 2021a] introduces a knowledge distillation algorithm for FSCIL and proposes to use semantic information during training. [Zhang et al., 2021] utilizes a graph model to propagate context information between classifiers for adaptation. [Liu et al., 2022] introduces a data-free replay scheme for generating old samples in FSCIL. [Akyurek ¨ et al., 2022] proposes a simple but effective subspace regularizer for learning in novel sessions. [Chi et al., 2022] proposes a bi-level optimization based on meta-learning to directly optimize the network to learn how to incrementally learn.

![](images/4f1981e757f7b5d3442b53d0e7fcbf6a229db3a711179e51714fe6ddfdb0e0fa.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Base Session (0)"] -->|PC| B["dictionary space"]
    C["Novel Session (1)"] -->|DA| D["dictionary space"]
```
</details>

Figure 2: Illustration of our proposed method D-FSCIL. The 4-point stars represent the synthetic data for pseudo classes and the 8-point stars are class prototypes. PC means including pseudo classes in the base session, i.e. pseudo-class augmented learning; DA is the incremental dictionary adaptation to novel sessions. By including pseudo classes into base session training, the class-wise representations possess higher separability, where the intra-class representations become more compact and more free space is reserved for future novel classes. During the phase of dictionary adaptation, base classes’ prototypes and backbone are frozen and novel classes’ prototypes and dictionary are mildly optimized to better incorporate novel concepts.

Dictionary Learning Dictionary learning has been widely used in the field of image annotation, multi-label learning, zero-shot learning, transfer learning, etc. Typical dictionary learning is to learn a dictionary/projection matrix to project the visual space to another semantic embedding space [Kodirov et al., 2015; Zhang and Saligrama, 2015; Ye and Guo, 2017]. In recent years, deep dictionary learning has been actively explored in image classification, image restoration and predictive phenotyping [Mahdizadehaghdam et al., 2019; Fu et al., 2019; Tang et al., 2020; Zheng et al., 2021]. Different from aforesaid methods, [Zhou et al., 2021] proposes an end-to-end deep dictionary model for multi-label classification by learning collaborative representations in the projected dictionary space instead of sparse ones. In this paper, we adopt a deep dictionary learning architecture to facilitate a better adaptation scheme for FSCIL.

# 3 Problem Setup

FSCIL aims to design a model that can incrementally learn novel classes from few-shot data without forgetting knowledge about old classes. Formally, the model experiences a sequence of labeled datasets $\{ \mathcal { D } _ { 0 } , \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$ with disjoint classes and can only access to one dataset at a time. After done the training at a time, the data is no longer accessible afterwards. In FSCIL, we call $\mathcal { D } _ { 0 } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { 0 } }$ as the training data for base session, in which a large amount of labeled training data from multiple base classes is provided. We denote the number of training examples in the base session as

$N _ { 0 }$ and all base classes as $C _ { 0 } .$ . We use $\mathcal { P } _ { 0 } = \{ \mathbf { p } _ { 1 } ^ { ( 0 ) } , \dots , \mathbf { p } _ { C _ { 0 } } ^ { ( 0 ) } \}$ to denote all class prototypes to be learned for base classes. In the following sessions $\{ \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$ , there is very limited labeled data for each session, which we call novel sessions. In the novel session t, we have the few-shot dataset $\mathcal { D } _ { t } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { C K }$ in a C-way K-shot format —i.e., K instances from each of the C classes.

# 4 Methodology

The key idea of our method D-FSCIL is to learn a prescient and adaptable deep dictionary learning model to tackle FS-CIL. We introduce a hybrid learning architecture by integrating dictionary learning with visual representation learning in Section 4.1. To further facilitate the future adaptation and improve the generalization and compatibility of the dictionary representation, we generate some pseudo classes by mixing up base data points in the visual feature space, which is referred to as pseudo-class augmented learning and presented in Section $4 . 2 .$ . After done the base session, the backbone is frozen which is proven a simple and effective way to overcome forgetting on past knowledge. Thus in our model, only dictionary is updated to adapt to new classes (see Section 4.3). Finally, we provide a detailed description in Section 4.4 for the model training procedure and inference.

# 4.1 Deep Dictionary Learning

Intuitively, the idea of deep dictionary learning is to first take advantage of the deep neural network to embed complex input data into informative and low-level feature embeddings. After that, the dictionary learning is applied to decompose these embeddings into basic patterns and re-represent the data in a high-level pattern based space as projection coefficients, which can be utilized for prediction [Fu et al., 2019]. In this section, we present a deep dictionary learning based classification module for base session training, in which the deep feature extractor and dictionary are trained collaboratively.

We represent the base dataset $\mathcal { D } _ { 0 }$ as a data matrix ${ \bf X } =$ $[ \mathbf { x } _ { 1 } , \ldots , \mathbf { \dot { x } } _ { N _ { 0 } } ] ^ { \top }$ and its extracted visual feature matrix is $\bar { \boldsymbol { \phi } } ( \mathbf { X } ) \in \mathbb { R } ^ { \tilde { N _ { 0 } } \times d }$ . Let $\textbf { M } \in \ \mathbb { R } ^ { m \times d }$ be the dictionary with m atoms in the extracted visual feature space, and ${ \textbf { Z } } =$ $[ \mathbf { z } _ { 1 } , \hdots , \mathbf { z } _ { N _ { 0 } } ] ^ { \top } \ \in \ \mathbb { R } ^ { N _ { 0 } \times m }$ be the coefficient matrix for the $N _ { 0 }$ instances in the base dataset $\mathcal { D } _ { 0 } .$ . Each coefficient vector $\mathbf { z } _ { i }$ is a high-level dictionary-atom based embedding vector for $\phi ( \mathbf { x } _ { i } )$ in the dictionary space. The deep dictionary learning factorizes the visual feature matrix $\phi ( { \bf X } )$ into the product of the dictionary M and the coefficient matrix $\mathbf { Z } ,$ and learns the feature extractor $\phi ,$ the dictionary M and the coefficients Z by minimizing the regularized reconstruction error defined as follows,

$$
\begin{array}{l} L _ {d i c} = \sum_ {i = 1} ^ {N _ {0}} \left(\| \phi (\mathbf {x} _ {i}) - \mathbf {z} _ {i} ^ {\top} \mathbf {M} \| _ {2} ^ {2} + \lambda \| \mathbf {z} _ {\mathbf {i}} \| _ {2} ^ {2}\right) \tag {1} \\ = \| \phi (\mathbf {X}) - \mathbf {Z M} \| _ {F} ^ {2} + \lambda \| \mathbf {Z} \| _ {F} ^ {2} \\ \end{array}
$$

where $\| \cdot \| _ { F }$ is the Frobenius norm and λ is a trade-off parameter. Different from conventional dictionary learning which learns sparse representations in the dictionary space, we adopt the $\ell _ { 2 }$ -norm regularization over Z to enable a closed-form solution for Z given fixed M and $\phi ( \mathbf { X } )$ as follows via the minimization of $L _ { d i c }$ ,

$$
\mathbf {Z} = \phi (\mathbf {X}) \mathbf {M} ^ {\top} \left(\mathbf {M M} ^ {\top} + \lambda \mathbf {I} _ {m}\right) ^ {- 1} \tag {2}
$$

where ${ \mathbf I } _ { m }$ denotes the identity matrix of size m. Henceforward we can treat Z as a function of the dictionary M and the extractor φ.

With the coefficients Z for the base session, we perform prototype based classification by enforcing the coefficient vectors to be close to their corresponding class prototypes and minimizing the following prototype classification loss:

$$
L _ {c l s} = \frac {1}{N _ {0}} \sum_ {i = 1} ^ {N _ {0}} \log \frac {\exp (s (\mathbf {z} _ {i} , \mathbf {p} _ {y _ {i}} ^ {(0)}) / \tau)}{\sum_ {\mathbf {p} \in \mathcal {P} _ {0}} \exp (s (\mathbf {z} _ {i} , \mathbf {p}) / \tau)} \tag {3}
$$

where $s ( \cdot , \cdot )$ denotes the cosine similarity and τ is a temperature parameter. As previously introduced, $\mathcal { P } _ { 0 }$ consists of the set of class prototypes in the base session.

# 4.2 Pseudo-Class Augmented Learning

The dictionary merely learned on the base classes is not particularly prescient on generalization or adaptation to new concepts in future novel sessions. Therefore, we propose to generate some pseudo classes to shed light on the possible future ones. By this way, the representation space for those pseudo classes can be reserved for future incremental classes and the dictionary is expected to be more adaptable and compatible. Specifically, we extend the base dataset and classes by conducting inter-class mixup over the base training instances in the visual feature space and treat the synthetic data generated by each pair of different classes as a new pseudo class.

Mixup [Zhang et al., 2018] is a linear interpolation based data augmentation technique. We adopt this technique in a particular manner to generate sythetic pseudo-classes. Specifically, let $\mathcal { F } _ { 0 } ~ = ~ \{ \mathbf { f } _ { 1 } , \ldots , \mathbf { f } _ { N _ { 0 } } \}$ be the corresponding feature vectors for the base training dataset $\mathcal { D } _ { 0 }$ . To generate the synthetic data for a pseudo-class, we first randomly sample a pair of base classes, say $c _ { 1 }$ and $c _ { 2 } ,$ , and denote the pseudo class generated by class $c _ { 1 }$ and $c _ { 2 }$ as $\tilde { c } _ { 1 }$ . Then, a synthetic feature vector of the pseudo class $\tilde { c } _ { 1 }$ can be generated by mixing up two randomly sampled instance vectors in the feature space from class $c _ { 1 }$ and $c _ { 2 }$ as follows:

$$
\tilde {\mathbf {f}} ^ {(\tilde {c} _ {1})} = \gamma \mathbf {f} ^ {(c _ {1})} + (1 - \gamma) \mathbf {f} ^ {(c _ {2})} \tag {4}
$$

where $\mathbf { f } ^ { ( c _ { 1 } ) }$ and $\mathbf { f } ^ { ( c _ { 2 } ) }$ are feature vectors from ${ \mathcal { F } } _ { 0 } ,$ , belonging to class $c _ { 1 }$ and $c _ { 2 } ,$ respectively. We sample the coefficient $\gamma$ from the range of [0.4, 0.6] to encourage the synthetic data to fall between the two different classes and reduce overlaps with data in existing classes. By mixing up different pairs of feature vectors from class $c _ { 1 }$ and $c _ { 2 }$ , we can generate a set of synthetic feature vectors for the pseudo class $\tilde { c } _ { 1 }$ . We repeat such process $\tilde { C } _ { 0 }$ times by mixing up different pairs of base classes to get $\ddot { C } _ { 0 }$ pseudo classes and their corresponding synthetic data. Suppose the total number of generated mixup instances for the $\tilde { C } _ { 0 }$ pseudo-classes is $\tilde { N } _ { 0 }$ . The set of synthetic mixup data in the visual feature space $\phi ( \cdot )$ can be denoted as $\tilde { \mathcal { F } } = \{ \tilde { \mathbf { f } } _ { 1 } , \hdots , \tilde { \mathbf { f } } _ { \tilde { N } _ { 0 } } \}$ with corresponding labels $\big \{ \tilde { y } _ { 1 } , \dots , \tilde { y } _ { \tilde { N } _ { 0 } } \big \}$ , indicating the pseudo classes.

After obtaining ${ \tilde { \mathcal { F } } } ,$ , its corresponding dictionary based coefficient matrix Z˜ can be calculated as a function of the dictionary M and the extractor φ via Eq. (2) by replacing $\phi ( \mathbf { X } )$ with $\tilde { \mathcal { F } }$ . Let $\tilde { \mathcal { P } } _ { 0 } = \{ \tilde { \bf p } _ { 1 } ^ { ( 0 ) } , \dots , \tilde { \bf p } _ { \tilde { C } _ { 0 } } ^ { ( 0 ) } \}$ denote the set of to-be-C˜0 learned pseudo class prototypes. Accordingly, the classification loss on the synthetic data of pseudo classes is defined as:

$$
\tilde {L} _ {c l s} = \frac {1}{\tilde {N} _ {0}} \sum_ {i = 1} ^ {\tilde {N} _ {0}} \log \frac {\exp (s (\tilde {\mathbf {z}} _ {i} , \tilde {\mathbf {p}} _ {\tilde {y} _ {i}} ^ {(0)}) / \tau)}{\sum_ {\mathbf {p} \in \mathcal {P} _ {0} \cup \tilde {\mathcal {P}} _ {0}} \exp (s (\tilde {\mathbf {z}} _ {i} , \mathbf {p}) / \tau)} \tag {5}
$$

where $\tilde { \mathbf { z } } _ { i }$ is the coefficient vector for $\tilde { \mathbf { f } } _ { i }$ and $\tilde { y } _ { i }$ is its corresponding class label among the pseudo classes.

With the pseudo classes, we then have the following total classification loss for the augmented learning of M, φ, and class prototypes $\mathcal { P } _ { 0 } \cup \tilde { \mathcal { P } } _ { 0 }$ in the base session:

$$
L _ {b a s e} = L _ {c l s} + \eta \tilde {L} _ {c l s}. \tag {6}
$$

Through this loss, the dictionary enhanced by pseudo classes is expected to be more generalizable and capable of effectual adaptation without significantly interfere with the classification of base classes.

# 4.3 Incremental Dictionary Adaptation

In the novel sessions, we freeze the feature extractor φ learned in the base session, and only finetune the dictionary M to allow convenient adaptation to new classes without dramatic visual feature shifting. Formally, in a novel session t, we have a very small labeled dataset $\mathcal { D } _ { t }$ of size CK. We denote all instances as a data matrix $\mathbf { X } _ { t }$ with instance x as the i-th row and its visual feature matrix as $\phi ( \mathbf { X } _ { t } ) \in \mathbb { R } ^ { C \dot { \boldsymbol { K } } \times d }$ . The corresponding coefficient matrix is represented by $\mathbf { Z } _ { t } \in \mathbb { R } ^ { C K }$ ×m, which again can be expressed as a function of M and $\phi ( \mathbf { X } _ { t } )$ via Eq. (2). The set of class prototypes to be learned for the current novel session is denoted as $\mathcal { P } _ { t } = \{ \mathbf { p } _ { 1 } ^ { ( t ) } , \dots , \mathbf { p } _ { C } ^ { ( t ) } \}$ .

To enable the dictionary to incorporate information from the current novel session, we slightly update the dictionary M by minimizing a classification loss on the current training dataset, to push the coefficient vectors $\left\{ \mathbf { z } _ { i } \right\}$ to approach their corresponding prototypes. Specifically, the class prototypes for the current session are first initialized as the mean vectors of the coefficients that belonging to the corresponding classes:

$$
\mathbf {p} _ {c} ^ {(t)} = \frac {1}{n _ {c}} \sum_ {i = 1} ^ {C K} \mathbb {I} _ {[ y _ {i} = c ]} \mathbf {z} _ {i} \tag {7}
$$

where $\mathbf { z } _ { i }$ represents coefficient vector in the i-th row of $\mathbf { Z } _ { t }$ with corresponding class label $y _ { i } .$ , and $n _ { c }$ is the number of instances of class c in $\mathcal { D } _ { t }$ .

Then the dictionary M and the class prototypes $\mathcal { P } _ { t }$ are learned by minimizing the following prototypical classification loss in the novel session t:

$$
\begin{array}{l} L _ {\text { novel }} = \frac {1}{C K} \sum_ {i = 1} ^ {C K} \log \frac {\exp (s (\mathbf {z} _ {i} , \mathbf {p} _ {y _ {i}} ^ {(t)}) / \tau)}{\sum_ {\mathbf {p} \in \cup_ {j \leq t} \mathcal {P} _ {j}} \exp (s (\mathbf {z} _ {i} , \mathbf {p}) / \tau)} \tag {8} \\ + \alpha \| \mathbf {M} - \mathbf {M} _ {0} \| _ {F} ^ {2} \\ \end{array}
$$

Algorithm 1 Training Procedure   
Input: Sequential datasets $D = \{D_{0}, \ldots, D_{T}\}$ Output: Learned feature extractor $\phi$ , dictionary M, prototypes $P_{0}, \ldots, P_{T}$ 1: // Base Session

2: Randomly initialize $\phi$ , M, $P_{0}$ , $\tilde{P}_{0}$ 3: repeat

4: $B \leftarrow$ sample a mini-batch of instances from $D_{0}$ ;

5: $B' \leftarrow$ generate pseudo instances by interclass mixup;

6: Compute coefficients Z and $\tilde{Z}$ for B and $\tilde{B}$ by Eq. (2), respectively;

7: Update $\phi$ , M, $P_{0}$ , $\tilde{P}_{0}$ by minimizing $L_{base}$ in Eq. (6).

8: until maximum iterations reached

9:

10: // Novel Sessions

11: $M_{0} \leftarrow M$ and keep $M_{0}$ fixed

12: for $t = 1, \cdots, T$ do

13: Compute coefficients $Z_{t}$ for $D_{t}$ by Eq. (2);

14: Initialize $P_{t}$ as the means of the coefficient vectors that belonging to the corresponding classes via Eq. (7);

15: repeat

16: Compute coefficients $Z_{t}$ for $D_{t}$ by Eq. (2);

17: Update M, $P_{t}$ by minimizing $L_{novel}$ in Eq. (8)

18: until maximum iterations reached

19: end for

where $y _ { i }$ denotes the class label for $\mathbf { z } _ { i } , \mathbf { M } _ { 0 }$ is the dictionary state produced from the base session, and α is a regularization hyper-parameter that controls the penalty level of dictionary changes. Larger α indicates the higher degree of preserving the dictionary to the base session state and the lower adaptation to new classes. With this regularization term over M , we expect the dictionary representation is only moderately updated to adapt to new classes without severely degrading the performance on the base ones.

# 4.4 Learning Algorithm and Inference

# Learning Algorithm

The proposed deep dictionary learning model D-FSCIL involves the minimization of the reconstruction loss $L _ { d i c }$ for dictionary learning implicitly during the base or novel session training with the classification loss $L _ { b a s e }$ or $L _ { n o v e l } ,$ , as the coefficient matrices can be expressed as functions of the dictionary M and the extractor φ via Eq. (2). Meanwhile, this also allows the dictionary M and the extractor φ to be learned through the prototype learning classification losses $L _ { b a s e }$ and $L _ { n o v e l }$ (φ is fixed for $L _ { n o v e l } )$ .

We describe the detailed model training procedure in Algorithm 1, which jointly learns M, $\phi ,$ and $\mathcal { P } _ { 0 } \cup \tilde { \mathcal { P } } _ { 0 }$ by minimizing $L _ { b a s e }$ in the base session, while jointly learning $\mathcal { P } _ { t }$ and finetuning M in each novel session.

# Inference by Prototypes

Once the learning is finished upon the training set in the session t, we obtain the feature extraction module φ, the dictionary M and prototypes of all seen classes $\cup _ { j \leq t } \mathcal { P } _ { j }$ for prediction. Given a test instance x, its corresponding coefficient vector z is calculated by Eq. 2, and then the label will be assigned to the class with the most similar prototype, namely,

$$
k ^ {*} = \underset {\mathbf {p} _ {k} \in \cup_ {j \leq t} \mathcal {P} _ {j}} {\arg \max} s (\mathbf {z}, \mathbf {p} _ {k}). \tag {9}
$$

# 5 Experiments

We conducted experiments to compare the proposed D-FSCIL with several state-of-the-art FSCIL methods on three benchmark datasets. We start by introducing the dataset statistics and experimental implementation details, and then present comprehensive results and analysis on comparison experiments and ablation studies.

# 5.1 Datasets

We follow the benchmark setting in [Tao et al., 2020] to evaluate the performance of the proposed method on CI-FAR100 [Krizhevsky, 2009], miniImageNet [Russakovsky et al., 2014] and CUB200 [Wah et al., 2011]. CIFAR100 and miniImageNet consist of 60,000 RGB images from 100 classes. The size of images is 32×32 and 84×84, respectively. Each class contains 500 images for training and 100 images for testing. The total 100 classes are first divided into 60 base classes and 40 novel classes, and the 40 novel classes are further formulated into 8 incremental sessions of 5-way 5- shot classification task. CUB200 is a 200-class dataset with 11,788 224×224 RGB images. 100 classes are selected for base session and the rest 100 classes are divided into 10-way 5-shot tasks for 10 novel sessions.

# 5.2 Implementation Details

All compared and proposed methods adopt the same backbone architectures. Following [Tao et al., 2020], we employ ResNet20 [He et al., 2016] for experiments on CIFAR100 and ResNet18 [He et al., 2016] on miniImageNet and CUB200. The model is optimized using SGD with momentum. The batch size is set to be 256 on all datasets. For CIFAR100 and miniImageNet, we train the model for 600 epochs with the learning rate starting from 0.1 and decaying with cosine annealing. On CUB200, the model is trained for 200 epochs and the learning rate starts from 0.005 and decays by a factor of 0.25 every 50 epochs. Data augmentations, such as random crop, random flip and random scale are adopted at base session training time. In the incremental sessions, the model is finetuned for 10 epochs with learning rate 5e-3 on CIFAR100 and miniImageNet and 1 epoch with learning rate 1e-3 on CUB200. For the hyper-parameters, λ is fixed to be 0.1, η is 1e-3 and α is 10 on all datasets. τ is set to be 0.08, 0.05, 0.1 for CIFAR100, miniImageNet and CUB200, respectively. The size of dictionary m is 70 for CIFAR100 and 500 for miniImageNet and CUB200. We set the number of pseudo classes $\tilde { C } _ { 0 }$ as the total number of classes in all the novel sessions.

# 5.3 Evaluation Metrics

Generally, the top-1 class-wise average accuracy for each session is widely employed for evaluation. As [Peng et al., 2022] pointed out, such accuracy is not enough for evaluating the overall performance of FSCIL methods since the number of base classes often occupies a large portion of the total. To make up for this deficiency, the harmonic mean of accuracies on base classes and novel classes is adopted, which is defined as $2 \times A _ { b } \times A _ { n } / ( A _ { b } + A _ { n } )$ , where $A _ { b }$ and $A _ { n }$ are the accuracy for base classes and novel classes in a session, respectively. An ideal FSCIL method should have high performance in both metrics. Thus, in this paper we will use both to evaluate the proposed and compared methods.

# 5.4 Comparison Results

We compare our proposed D-FSCIL with several state-of-theart FSCIL methods including TOPIC [Tao et al., 2020], Zhu et al. [Zhu et al., 2021], CEC [Zhang et al., 2021], Liu et al. [Liu et al., 2022], MetaFSCIL [Chi et al., 2022] and FACT [Zhou et al., 2022], and the main results are reported in Table 1. In this table, the accuracy for each session is calculated on the test data of all seen classes. We also report the average accuracy over all sessions to reveal the general performance for each method. The last column in the table shows the accuracy increase of our method compared to others after learning all sessions. It is worth noting that our method D-FSCIL significantly outperforms all the compared state-of-the-art in each session on three datasets. More specifically, D-FSCIL surpasses the most recent methods Liu et al., MetaFSCIL and FACT in the final session by 2.06%, 2.23% and 0.10% on CIFAR100, 2.80%, 1.82% and 0.52% on miniImageNet, and 5.39%, 5.14% and 0.84% on CUB200, respectively.

In the base session, our method has higher accuracies on all datasets compared to others demonstrating the superb performance of deep dictionary learning for image classification. Next, we will show that the superior performance of D-FSCIL does not merely benefit from the base classes. On the contrary, our proposed method keeps particular better performance on novel classes after learning all sessions. Since CEC has comparable results to the most recent methods Liu et al. and MetaFSCIL with only no more than 1% accuracy difference, we further compare our method D-FSCIL with CEC in terms of the accuracies on base classes and novel classes, and the harmonic accuracy in each session on all datasets. The results are shown in Figure 3, from which we can observe that D-FSCIL achieves much better accuracy for novel classes especially on CIFAR100 and CUB200 while still maintaining the competent performance for base classes. The harmonic accuracy of D-FSCIL also consistently outperforms CEC over all sessions for three datasets except the session 1 on miniImageNet where CEC behaves slightly better.

We further compare our method with FACT [Zhou et al., 2022] on CIFAR100 since both methods benefit from forward compatibility by embracing pseudo classes in the base session. Notably, FACT forbids the model to further update for novel classes to overcome forgetting on base classes, which indeed achieves such effects but loses better incorporation of new classes. From Figure 4, we can see that the good performance of FACT mainly benefits from base classes with less improvements on novel ones, whereas our method can better accommodate novel concepts.

From above observations and analysis, we claim that the proposed method D-FSCIL not only learns the desirable visual feature space and dictionary space for classification in the base session, but also possess the satisfactory capacity to learn and adapt to new classes without forgetting the knowledge of base ones.

Table 1: Comparison with the state-of-the-art methods on CIFAR100, miniImageNet and CUB200 datasets. The results of compared methods are directly cited from the corresponding papers. The best result in each session and the best average accuracy over all sessions are in bold. 

<table><tr><td rowspan="2">Method</td><td colspan="9">Sessions (CIFAR100) w/ ResNet20</td><td rowspan="2">Average Acc</td><td rowspan="2">Final Impro.</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>TOPIC [Tao et al., 2020]</td><td>64.10</td><td>55.88</td><td>47.07</td><td>45.16</td><td>40.11</td><td>36.38</td><td>33.96</td><td>31.55</td><td>29.37</td><td>42.62</td><td>+22.83</td></tr><tr><td>Zhu et al. [Zhu et al., 2021]</td><td>64.10</td><td>65.86</td><td>61.36</td><td>57.34</td><td>53.68</td><td>50.75</td><td>48.58</td><td>45.66</td><td>43.25</td><td>54.51</td><td>+8.95</td></tr><tr><td>CEC [Zhang et al., 2021]</td><td>73.07</td><td>68.88</td><td>65.26</td><td>61.19</td><td>58.09</td><td>55.57</td><td>53.22</td><td>51.34</td><td>49.14</td><td>59.53</td><td>+3.06</td></tr><tr><td>Liu et al. [Liu et al., 2022]</td><td>74.40</td><td>70.20</td><td>66.54</td><td>62.51</td><td>59.71</td><td>56.58</td><td>54.52</td><td>52.39</td><td>50.14</td><td>60.77</td><td>+2.06</td></tr><tr><td>MetaFSCIL [Chi et al., 2022]</td><td>74.50</td><td>70.10</td><td>66.84</td><td>62.77</td><td>59.48</td><td>56.52</td><td>54.36</td><td>52.56</td><td>49.97</td><td>60.79</td><td>+2.23</td></tr><tr><td>FACT [Zhou et al., 2022]</td><td>74.60</td><td>72.09</td><td>67.56</td><td>63.52</td><td>61.38</td><td>58.36</td><td>56.28</td><td>54.24</td><td>52.10</td><td>62.24</td><td>+0.10</td></tr><tr><td>D-FSCIL (ours)</td><td>77.23</td><td>73.11</td><td>69.11</td><td>65.27</td><td>62.39</td><td>59.48</td><td>57.62</td><td>55.24</td><td>52.20</td><td>63.42</td><td></td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="9">Sessions (miniImageNet) w/ ResNet18</td><td rowspan="2">Average Acc</td><td rowspan="2">Final Impro.</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>TOPIC [Tao et al., 2020]</td><td>61.31</td><td>50.09</td><td>45.17</td><td>41.16</td><td>37.48</td><td>35.52</td><td>32.19</td><td>29.46</td><td>24.42</td><td>39.64</td><td>+26.59</td></tr><tr><td>Zhu et al. [Zhu et al., 2021]</td><td>61.45</td><td>63.80</td><td>59.53</td><td>55.53</td><td>52.50</td><td>49.60</td><td>46.69</td><td>43.79</td><td>41.92</td><td>52.75</td><td>+9.09</td></tr><tr><td>CEC [Zhang et al., 2021]</td><td>72.00</td><td>66.83</td><td>62.97</td><td>59.43</td><td>56.70</td><td>53.73</td><td>51.19</td><td>49.24</td><td>47.63</td><td>57.75</td><td>+3.38</td></tr><tr><td>Liu et al. [Liu et al., 2022]</td><td>71.84</td><td>67.12</td><td>63.21</td><td>59.77</td><td>57.01</td><td>53.95</td><td>51.55</td><td>49.52</td><td>48.21</td><td>58.02</td><td>+2.80</td></tr><tr><td>MetaFSCIL [Chi et al., 2022]</td><td>72.04</td><td>67.94</td><td>63.77</td><td>60.29</td><td>57.58</td><td>55.16</td><td>52.90</td><td>50.79</td><td>49.19</td><td>58.85</td><td>+1.82</td></tr><tr><td>FACT [Zhou et al., 2022]</td><td>72.56</td><td>69.63</td><td>66.38</td><td>62.77</td><td>60.60</td><td>57.33</td><td>54.34</td><td>52.16</td><td>50.49</td><td>60.69</td><td>+0.52</td></tr><tr><td>D-FSCIL (ours)</td><td>74.38</td><td>69.76</td><td>66.47</td><td>62.84</td><td>59.77</td><td>56.95</td><td>54.50</td><td>52.32</td><td>51.01</td><td>60.83</td><td></td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="11">Sessions (CUB200) w/ ResNet18</td><td rowspan="2">Average Acc</td><td rowspan="2">Final Impro.</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>TOPIC [Tao et al., 2020]</td><td>68.68</td><td>62.49</td><td>54.81</td><td>49.99</td><td>45.25</td><td>41.40</td><td>38.35</td><td>35.36</td><td>32.22</td><td>28.31</td><td>26.28</td><td>43.92</td><td>+31.50</td></tr><tr><td>Zhu et al. [Zhu et al., 2021]</td><td>68.68</td><td>61.85</td><td>57.43</td><td>52.68</td><td>50.19</td><td>46.88</td><td>44.65</td><td>43.07</td><td>40.17</td><td>39.63</td><td>37.33</td><td>49.32</td><td>+20.45</td></tr><tr><td>CEC [Zhang et al., 2021]</td><td>75.85</td><td>71.94</td><td>68.50</td><td>63.50</td><td>62.43</td><td>58.27</td><td>57.73</td><td>55.81</td><td>54.83</td><td>53.52</td><td>52.28</td><td>61.33</td><td>+5.50</td></tr><tr><td>Liu et al. [Liu et al., 2022]</td><td>75.90</td><td>72.14</td><td>68.64</td><td>63.76</td><td>62.58</td><td>59.11</td><td>57.82</td><td>55.89</td><td>54.92</td><td>53.58</td><td>52.39</td><td>61.52</td><td>+5.39</td></tr><tr><td>MetaFSCIL [Chi et al., 2022]</td><td>75.90</td><td>72.41</td><td>68.78</td><td>64.78</td><td>62.96</td><td>59.99</td><td>58.30</td><td>56.85</td><td>54.78</td><td>53.82</td><td>52.64</td><td>61.92</td><td>+5.14</td></tr><tr><td>FACT [Zhou et al., 2022]</td><td>75.90</td><td>73.23</td><td>70.84</td><td>66.13</td><td>65.56</td><td>62.15</td><td>61.74</td><td>59.83</td><td>58.41</td><td>57.89</td><td>56.94</td><td>64.42</td><td>+0.84</td></tr><tr><td>D-FSCIL (ours)</td><td>78.79</td><td>75.38</td><td>72.05</td><td>67.29</td><td>66.16</td><td>62.42</td><td>62.14</td><td>60.06</td><td>59.25</td><td>58.76</td><td>57.78</td><td>65.46</td><td></td></tr></table>

# 5.5 Ablation Study

After training the model in the base session, the feature extractor is frozen during learning new sessions and only the dictionary is slightly finetuned to adapt to new classes. In Figure 5, we compare the results of our proposed D-FSCIL (red line), and the results by removing each component including pseudo classes in the base session (PC) and dictionary adaptation to novel sessions (DA). We can observe that the deep dictionary learning (DDL) learned by minimizing $L _ { d i c }$ and $L _ { c l s }$ can get the decent performance but comparatively lower than other variants. The pure dictionary learning does not take the future possible changes into account and thus the feature extractor and dictionary lack the ability to further generalize to new emerging classes. As a result, it does not reserve any space for new classes in either visual feature space or dictionary space thereby working comparatively poorly. By including the pseudo classes of which the data is generated by inter-class mixup in the base session, the visual representations of the same base class are expected to be closer and the distance between different base classes are far apart, because the pseudo classes are interposed between base classes. By adding pseudo classes into base session, certain space is reserved for future classes and the dictionary is also trained to map more classes from feature space to dictionary space, therefore, it has higher compatibility to incorporate new classes in novel sessions. The incremental dictionary adaptation procedure allows the dictionary to moderately change to accommodate novel classes without severely degrading the performance of base classes. In particular, the model learned through pseudo classes has already been growable and provident which can better embrace the dictionary adaptation. By combining all the components together, we obtain our model achieving the best results. Thus far, the ablation studies verified the effectiveness of each component in our proposed D-FSCIL.

# Impact of Hyper-parameters

There are five hyper-parameters in D-FSCIL, that are the size m of dictionary elements, the trade-off parameter λ in the reconstruction loss, the temperature parameter τ in classification losses, the trade-off parameter η to balance the losses over base training data and synthetic data in the pseudo incremental learning, and the parameter α in the incremental dictionary adaptation. To evaluate the influence of each hyperparameter to our method, we conducted experiments on CI-

![](images/d69c8c4a76da6c73bfa405022f1ba704d5006558df4dd28c02d75b561b8f91ea.jpg)

<details>
<summary>line</summary>

| Sessions | Accuracy (%) - Series 1 | Accuracy (%) - Series 2 |
| -------- | ------------------------ | ------------------------ |
| 1        | 75                       | 50                       |
| 2        | 73                       | 40                       |
| 3        | 72                       | 35                       |
| 4        | 71                       | 33                       |
| 5        | 70                       | 32                       |
| 6        | 69                       | 33                       |
| 7        | 68                       | 31                       |
| 8        | 67                       | 30                       |
</details>

(a) CIFAR100

![](images/953a92eb514c770712d3724c8729eba743edc492a042ae8675ee66b735c86c35.jpg)

<details>
<summary>line</summary>

| Sessions | CEC-base | D-FSCIL-base | CEC-novel | D-FSCIL-novel |
| -------- | -------- | ------------ | --------- | ------------- |
| 1        | 70       | 73           | 27        | 23            |
| 2        | 70       | 72           | 24        | 25            |
| 3        | 70       | 71           | 22        | 26            |
| 4        | 70       | 70           | 21        | 25            |
| 5        | 70       | 69           | 20        | 24            |
| 6        | 70       | 68           | 18        | 23            |
| 7        | 69       | 67           | 17        | 24            |
| 8        | 69       | 66           | 18        | 25            |
</details>

(b) miniImageNet

![](images/c695ce0cb5f6f814d4ad492d43b66caa515dfe874ea9e404ac95c227b91ccf0b.jpg)

<details>
<summary>line</summary>

| Sessions | Accuracy (%) - Series 1 | Accuracy (%) - Series 2 |
| -------- | ------------------------ | ------------------------ |
| 1        | 75                       | 45                       |
| 2        | 73                       | 42                       |
| 3        | 72                       | 33                       |
| 4        | 71                       | 38                       |
| 5        | 70                       | 33                       |
| 6        | 69                       | 35                       |
| 7        | 68                       | 34                       |
| 8        | 67                       | 33                       |
| 9        | 66                       | 35                       |
| 10       | 65                       | 34                       |
</details>

(c) CUB200

Figure 3: Comparison of CEC (blue) and our method D-FSCIL (red) in terms of the accuracy for base classes and novel classes as well as the harmonic mean in each session on CIFAR100, miniImageNet and CUB200 datasets. Our method D-FSCIL performs particularly well on novel classes while still maintaining satisfactory performance on base classes.   
![](images/cd6f4606576cb3d7de3d62360153db2a07bb65150e144680f08c54231d0bab29.jpg)

<details>
<summary>line</summary>

| Sessions | Accuracy (%) - Blue Line | Accuracy (%) - Red Line |
| -------- | ------------------------ | ----------------------- |
| 1        | 75                       | 50                      |
| 2        | 74                       | 38                      |
| 3        | 73                       | 35                      |
| 4        | 72                       | 33                      |
| 5        | 71                       | 32                      |
| 6        | 70                       | 31                      |
| 7        | 69                       | 30                      |
| 8        | 68                       | 29                      |
</details>

Figure 4: Comparison of FACT (blue) and our method D-FSCIL (red) on CIFAR100. Our method consistently achieves higher harmonic accuracies and better performance on novel classes, while still maintaining satisfactory performance on base classes.

![](images/6431d72ddaf59209b99d7dada9574cbb1aeb2634b406d0ceb49a09bc0e87ea31.jpg)

<details>
<summary>line</summary>

| Sessions | DDL   | DDL + PC | DDL + PC + DA |
| -------- | ----- | -------- | ------------- |
| 4        | 61.5  | 61.8     | 62.0          |
| 5        | 59.0  | 59.2     | 59.5          |
| 6        | 57.0  | 57.2     | 57.8          |
| 7        | 54.5  | 55.0     | 55.2          |
| 8        | 51.0  | 51.5     | 52.0          |
</details>

Figure 5: Ablation study on CIFAR100 to verify the effectiveness of each component. DDL represents our base model of deep dictionary learning; PC means including pseudo classes in the base session training; DA is the dictionary adaptation to novel sessions. We show the results in last five sessions for better curves comparison.

FAR100 and the results in the last session with different values of hyper-parameters are shown in Figure 6. It is noted that when evaluating one hyper-parameter, the values of others are fixed. Firstly, we investigated different values of m from the set of $\{ 1 0 , \bar { 3 0 } , 5 0 , 7 0 , 9 \bar { 0 } \}$ , and we can observe that the performance is relatively poor when m is small like 10. This is due to the fact that the embedding space projected by the dictionary is excessively compressed because of the low dimensionality, which leads to the loss of data information. The reasonable range of m would be around 70. Secondly, the values of λ is selected from $\{ 0 , 0 . 0 1 , 0 . 0 5 , 0 . 1 , 1 \}$ . We can see that no regularization or small values like 0.01 will fail the deep dictionary learning on CIFAR100 and the best result is produced when $\lambda = 0 . 1$ . Thirdly, the temperature parameter $\tau \in \{ 0 . 0 5 , 0 . 0 8 , 0 . 1 , 0 . 1 3 , 0 . 1 5 \}$ and the best result is achieved when $\tau \ = \ 0 . 0 8$ on CIFAR100. Fourthly, $\eta ~ \in ~ [ 0 , 0 . 1 ]$ represents the weight of the classification loss over the generated data of pseudo classes. The larger value of η compels the model to focus more on synthetic data leading to the performance deterioration on base classes. $\eta = 1 e - 3$ achieves the best. Fifthly, $\alpha \in [ 0 , 1 0 0 ]$ determines the degree of preserving the dictionary from drastic changes to overcoming forgetting of base classes. $\alpha = 0$ yields the worst result because the dictionary is greatly changed to adapt to novel classes. However, due to the face that very limited data is available for novel sessions, the dictionary is easily to overfit the small dataset resulting in the performance degradation in both base and novel sessions. Lastly, we evaluate the impact of the number of pseudo classes $\tilde { C } _ { 0 } \in \{ 2 0 , 3 0 , 4 0 , 5 0 , 6 0 \}$ which shows the performance differences are minuscule, thus we default $\tilde { C } _ { 0 }$ as the total number of novel classes.

# 6 Conclusion

In this paper, we proposed a novel and succinct method D-FSCIL by applying deep dictionary learning into the few-shot class-incremental learning (FSCIL) problem. In the base session, the dictionary and feature extraction module are collaboratively learned implicitly through the prototypical classification loss minimization. For a more adaptable and prescient dictionary, we generate some pseudo classes into base training set by conducting inter-class mixup in the visual feature space. The representation space occupied by pseudo classes is expected to be reserved for future novel classes and increase the generalizability. Incremental dictionary adaptation in novel sessions further improves the compatibility with new concepts through dictionary finetuning. Finally extensive experiments have been conducted and the results verify the effectiveness of the proposed D-FSCIL.

![](images/836666f83fb2ea1f81e2fa00dce49da8d79a2ad9a5d47c322d4bb0e6c2dfbcb9.jpg)

<details>
<summary>line</summary>

| m  | Accuracy (%) |
|----|--------------|
| 10 | 46           |
| 30 | 51           |
| 50 | 52           |
| 70 | 52           |
| 90 | 51.5         |
</details>

(a) influence of m

![](images/5b6cbba44bd28e1a87ec34b3c3d804689090e4e63417568e72a8972c3179573d.jpg)

<details>
<summary>line</summary>

| λ    | Accuracy (%) |
| ---- | ------------ |
| 0.0  | 0            |
| 0.01 | 0            |
| 0.05 | 50           |
| 0.1  | 52           |
| 1.0  | 49           |
</details>

(b) influence of λ

![](images/5f85733ddfad12336b9d0fdddb9e3d498c850dd21923cadb4a057e2491d971bf.jpg)

<details>
<summary>line</summary>

| τ    | Accuracy (%) |
| ---- | ------------ |
| 0.05 | 47.5         |
| 0.08 | 52.0         |
| 0.10 | 51.0         |
| 0.13 | 50.5         |
| 0.15 | 48.5         |
</details>

(c) influence of τ

![](images/ccbf46581643143345b776415badbe633cc2b0a98dc5afb9e60e0a3af8dcd935.jpg)

<details>
<summary>line</summary>

| η     | Accuracy (%) |
| ------- | ------------ |
| 0.0   | 51.8         |
| 0.0001 | 51.9         |
| 0.001  | 52.1         |
| 0.01   | 51.3         |
| 0.1   | 49.0         |
</details>

(d) influence of η

![](images/6ca4396ea11b9c1bc3856c4390b1384f815bf885acfead656590e2c29254b9d5.jpg)

<details>
<summary>line</summary>

| α    | Accuracy (%) |
| ---- | ------------ |
| 0.0  | 47.5         |
| 0.1  | 49.2         |
| 1.0  | 51.8         |
| 10.0 | 52.2         |
| 100.0| 51.5         |
</details>

(e) influence of α

![](images/4147a9cee39862b3c3ec04565ad73b190ddc1fe03fd0f171e013f792b4bc1a32.jpg)

<details>
<summary>line</summary>

| C₀ | Accuracy (%) |
| --- | ------------ |
| 20  | 51.7         |
| 30  | 52.2         |
| 40  | 52.2         |
| 50  | 51.8         |
| 60  | 51.8         |
</details>

(f) influence of $\tilde { C } _ { 0 }$   
Figure 6: Impact of hyper-parameters on CIFAR100.

# References

[Akyurek ¨ et al., 2022] Afra Feyza Akyurek, Ekin Aky ¨ urek, ¨ Derry Wijaya, and Jacob Andreas. Subspace regularizers for few-shot class incremental learning. In International Conference on Learning Representations (ICLR), 2022.   
[Castro et al., 2018] Francisco M. Castro, Manuel J. Mar´ın-Jimenez, Nicol ´ as Guil, Cordelia Schmid, and Karteek Ala- ´ hari. End-to-end incremental learning. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss, editors, European Conference on Computer Vision (ECCV), 2018.   
[Cheraghian et al., 2021a] Ali Cheraghian, Shafin Rahman, Pengfei Fang, Soumava Kumar Roy, Lars Petersson, and Mehrtash Harandi. Semantic-aware knowledge distillation for few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2534–2543, June 2021.   
[Cheraghian et al., 2021b] Ali Cheraghian, Shafin Rahman, Sameera Ramasinghe, Pengfei Fang, Christian Simon, Lars Petersson, and Mehrtash Harandi. Synthesized feature based few-shot class-incremental learning on a mixture of subspaces. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.   
[Chi et al., 2022] Zhixiang Chi, Li Gu, Huan Liu, Yang Wang, Yuanhao Yu, and Jin Tang. Metafscil: A metalearning approach for few-shot class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[Fu et al., 2019] Tianfan Fu, Trong Nghia Hoang, Cao Xiao, and Jimeng Sun. Ddl: Deep dictionary learning for predictive phenotyping. In International Joint Conference on Artificial Intelligence (IJCAI), 2019.

[Gido M. van de Ven, 2022] Andreas S. Tolias Gido M. van de Ven, Tinne Tuytelaars. Three types of incremental learning. Nature Machine Intelligence, 2022.   
[Goodfellow et al., 2014] Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgeting in gradientbased neural networks. In International Conference on Learning Representations (ICLR), 2014.   
[He et al., 2016] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2016.   
[Hou et al., 2019] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and Dahua Lin. Learning a unified classifier incrementally via rebalancing. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 831–839, 2019.   
[Kodirov et al., 2015] Elyor Kodirov, Tao Xiang, Zhenyong Fu, and Shaogang Gong. Unsupervised domain adaptation for zero-shot learning. In IEEE International Conference on Computer Vision (ICCV), 2015.   
[Krizhevsky, 2009] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.   
[Liu et al., 2022] Huan Liu, Li Gu, Zhixiang Chi, Yang Wang, Yuanhao Yu, Jun Chen, and Jin Tang. Few-shot class-incremental learning via entropy-regularized datafree replay. In European Conference on Computer Vision (ECCV), 2022.   
[Mahdizadehaghdam et al., 2019] Shahin Mahdizadehaghdam, Ashkan Panahi, Hamid Krim, and Liyi Dai. Deep

dictionary learning: A parametric network approach. IEEE Transactions on Image Processing, 2019.   
[Mai et al., 2021] Zheda Mai, Ruiwen Li, Hyunwoo Kim, and Scott Sanner. Supervised contrastive replay: Revisiting the nearest class mean classifier in online classincremental continual learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPRW), 2021.   
[McCloskey and Cohen, 1989] Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. volume 24 of Psychology of Learning and Motivation, pages 109 – 165. Academic Press, 1989.   
[Pellegrini et al., 2020] Lorenzo Pellegrini, Gabriele Graffieti, Vincenzo Lomonaco, and Davide Maltoni. Latent replay for real-time continual learning. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.   
[Peng et al., 2022] Can Peng, Kun Zhao, Tianren Wang, Meng Li, and Brian C. Lovell. Few-shot class-incremental learning from an open-set perspective. In European Conference on Computer Vision (ECCV), 2022.   
[Rebuffi et al., 2017] S. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert. icarl: Incremental classifier and representation learning. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.   
[Ring, 1994] Mark Bishop Ring. Continual Learning in Reinforcement Environments. PhD thesis, The University of Texas at Austin, 1994.   
[Russakovsky et al., 2014] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 2014.   
[Tang et al., 2020] Hao Tang, Hong Liu, Wei Xiao, and Nicu Sebe. When dictionary learning meets deep learning: Deep dictionary learning and coding network for image recognition with limited data. IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2020.   
[Tao et al., 2020] Xiaoyu Tao, Xiaopeng Hong, Xinyuan Chang, Songlin Dong, Xing Wei, and Yihong Gong. Fewshot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.   
[Thrun, 1994] S. Thrun. A lifelong learning perspective for mobile robot control. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 1994.   
[Wah et al., 2011] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. The caltech-ucsd birds-200-2011 dataset. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011.   
[Wu et al., 2019] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye, Zicheng Liu, Yandong Guo, and Yun Fu.

Large scale incremental learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 374–382, 2019.   
[Ye and Guo, 2017] Meng Ye and Yuhong Guo. Zero-shot classification with discriminative semantic representation learning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.   
[Zhang and Saligrama, 2015] Ziming Zhang and Venkatesh Saligrama. Zero-shot learning via semantic similarity embedding. IEEE International Conference on Computer Vision (ICCV), 2015.   
[Zhang et al., 2018] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. International Conference on Learning Representations (ICLR), 2018.   
[Zhang et al., 2021] Chi Zhang, Nan Song, Guosheng Lin, Yun Zheng, Pan Pan, and Yinghui Xu. Few-shot incremental learning with continually evolved classifiers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.   
[Zheng et al., 2021] Hongyi Zheng, Hongwei Yong, and Lei Zhang. Deep convolutional dictionary learning for image denoising. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.   
[Zhou et al., 2021] Fengtao Zhou, Sheng Huang, and Yun Xing. Deep semantic dictionary learning for multi-label image classification. Proceedings of the AAAI Conference on Artificial Intelligence, 2021.   
[Zhou et al., 2022] Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, and De-Chuan Zhan. Forward compatible few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.   
[Zhu et al., 2021] Kai Zhu, Yang Cao, Wei Zhai, Jie Cheng, and Zheng-Jun Zha. Self-promoted prototype refinement for few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.