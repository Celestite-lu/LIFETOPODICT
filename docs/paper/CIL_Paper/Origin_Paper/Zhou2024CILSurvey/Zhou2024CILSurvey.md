# Class-Incremental Learning: A Survey

Da-Wei Zhou , Qi-Wei Wang , Zhi-Hong Qi, Han-Jia Ye , De-Chuan Zhan , and Ziwei Liu , Member, IEEE

(Survey Paper)

Abstract—Deep models, e.g., CNNs and Vision Transformers, have achieved impressive achievements in many vision tasks in the closed world. However, novel classes emerge from time to time in our ever-changing world, requiring a learning system to acquire new knowledge continually. Class-Incremental Learning (CIL) enables the learner to incorporate the knowledge of new classes incrementally and build a universal classifier among all seen classes. Correspondingly, when directly training the model with new class instances, a fatal problem occurs — the model tends to catastrophically forget the characteristics of former ones, and its performance drastically degrades. There have been numerous efforts to tackle catastrophic forgetting in the machine learning community. In this paper, we survey comprehensively recent advances in class-incremental learning and summarize these methods from several aspects. We also provide a rigorous and unified evaluation of 17 methods in benchmark image classification tasks to find out the characteristics of different algorithms empirically. Furthermore, we notice that the current comparison protocol ignores the influence of memory budget in model storage, which may result in unfair comparison and biased results. Hence, we advocate fair comparison by aligning the memory budget in evaluation, as well as several memory-agnostic performance measures.

Index Terms—Catastrophic forgetting, class-incremental learning, continual learning, lifelong learning.

# I. INTRODUCTION

R ECENT years have witnessed the rapid progress of deeplearning, where neural networks have achieved or even surpassed human-level performances in many fields [1], [2], [3]. The typical training process of a deep network requires

Manuscript received 7 February 2023; revised 31 May 2024; accepted 11 July 2024. Date of publication 16 July 2024; date of current version 5 November 2024. This work was supported in part by National Science and Technology Major Project under Grant 2022ZD0114805, in part by Fundamental Research Funds for the Central Universities under Grant 2024300373, in part by the NSFC under Grant 62376118, Grant 62006112, Grant 62250069, and Grant 61921006, in part by the Collaborative Innovation Center of Novel Software Technology and Industrialization, China Scholarship Council, Ministry of Education, Singapore, under its MOE AcRF Tier 2 under Grant MOET2EP20221- 0012, NTU NAP, and under the RIE2020 Industry Alignment Fund–Industry Collaboration Projects (IAF-ICP) Funding Initiative. Recommended for acceptance by Y. Guo. (Corresponding authors: Han-Jia Ye; Ziwei Liu.)

Da-Wei Zhou, Qi-Wei Wang, Zhi-Hong Qi, Han-Jia Ye, and De-Chuan Zhan are with the School of Artificial Intelligence, National Key Laboratory for Novel Software Technology, Nanjing University, Nanjing 210023, China (e-mail: zhoudw@lamda.nju.edu.cn; wangqiwei@lamda.nju.edu.cn; qizh@lamda.nju.edu.cn; yehj@lamda.nju.edu.cn; zhandc@lamda.nju.edu.cn).

Ziwei Liu is with the S-Lab, College of Computing and Data Science, Nanyang Technological University, Singapore 639798 (e-mail: ziwei.liu@ntu.edu.sg).

The source code is available at https://github.com/zhoudw-zdw/CIL\_ Survey/.

This article has supplementary downloadable material available at https://doi.org/10.1109/TPAMI.2024.3429383, provided by the authors.

Digital Object Identifier 10.1109/TPAMI.2024.3429383

![](images/be3fdeb035bf640e2dd3c253c1faaca63cb484401885fa5861bd380ef2be9057.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Task 1"] --> B["Train"]
    B --> C["CIL Model"]
    D["Task 2"] --> E["Train"]
    E --> F["CIL Model"]
    G["Task 3"] --> H["Train"]
    H --> I["CIL Model"]
    J["Test Set 1"] --> K["Image 1"]
    L["Test Set 2"] --> M["Image 2"]
    N["Test Set 3"] --> O["Image 3"]
```
</details>

Fig. 1. The setting of CIL. Non-overlapping classes arrive sequentially, and the model needs to learn to classify all the classes incrementally. After learning each task, the model is evaluated among all seen classes. An ideal model should perform well in the newly learned classes and remember the former without forgetting.

pre-collected datasets in advance, e.g., large-scale images or texts — the network undergoes the training process of the precollected dataset multiple epochs. However, training data is often with stream format in the open world [4]. These streaming data cannot be held for long due to storage constraints [5] or privacy issues [6], requiring the model to be updated incrementally with only new class instances. Such requirements trigger the prosperity of the Class-Incremental Learning (CIL) field, aiming to continually build a holistic classifier among all seen classes. The fatal problem in CIL is called catastrophic forgetting, i.e., directly optimizing the network with new classes will erase the knowledge of former ones and result in irreversible performance degradation. Hence, how to effectively resist catastrophic forgetting becomes the core problem in building CIL models.

Fig. 1 depicts the typical setting of CIL. Training data emerge sequentially in the stream format. In each timestamp, we can get a new training dataset (denoted as ‘task’ in the figure) and need to update the model with the new classes. For example, the model learns ‘birds’ and ‘dogs’ in the first task, ‘tigers’ and ‘fish’ in the second task, ‘monkeys’ and ‘sheep’ in the third task, etc. Afterward, the model is tested among all seen classes to evaluate whether it has discrimination for them. A good model should strike a balance between depicting the characteristics of new classes and preserving the pattern of formerly learned old classes. This trade-off is also known as the ‘stability-plasticity dilemma’ in neural system [7], where stability denotes the ability to maintain former knowledge and plasticity represents the ability to adapt to new patterns.

![](images/ec8c66e7257db3fe94298ef85ec90636f19df4bf11bf431847f21366d9036cb8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Train"] --> B["Class-Incremental Learning"]
    B --> C["Task-Incremental Learning"]
    C --> D["Domain-Incremental Learning"]
    
    subgraph Train
        E["Image 1"] --> F["Task 1"]
        G["Image 2"] --> H["Task 2"]
        I["Image 3"] --> J["Image 4"]
    end
    
    subgraph Test
        K["Image 1"] --> L["Bird or Dog or Tiger or Fish?"]
    end
    
    subgraph Task-Incremental Learning
        M["Image 1"] --> N["Bird or Dog ?"]
        O["Image 2"] --> P["Tiger or Fish ?"]
    end
    
    subgraph Domain-Incremental Learning
        Q["Image 1"] --> R["Bird or Dog ?"]
        S["Image 2"] --> T["Bird or Dog ?"]
    end
```
</details>

Fig. 2. The setting of Class-Incremental Learning (CIL), Task-Incremental Learning (TIL), and Domain-Incremental Learning (DIL). CIL and TIL share the same training protocol, while TIL is much easier during inference, i.e., only requiring classifying among corresponding label spaces. DIL refers to the data stream with distribution change, where new tasks contain the same classes from different domains, e.g., cartoon and clip-art. The distinction of these scenarios is proposed by [8].

Since the data stream comes continually and requires training a lifelong time, incremental learning is also known as ‘continual learning’ [6] or ‘lifelong learning’ [9]. We interchangeably use these concepts in this paper. Apart from class-incremental learning, there are other fine-grained settings addressing the incremental learning problem, e.g., Task-Incremental Learning (TIL) and Domain-Incremental Learning (DIL) [8]. We show these three protocols in Fig. 2. TIL is a similar setting to CIL, and both of them observe incoming new classes in new tasks. However, the difference lies in the inference stage, where CIL requires the model to differentiate among all classes. By contrast, TIL only requires classifying the instance among the corresponding task space. In other words, it does not require cross-task discrimination ability. Hence, TIL is easier than CIL, which can be seen as a particular case of CIL. On the other hand, DIL concentrates on the scenario with concept drift or distribution change [10], where new tasks contain instances from different domains but with the same label space. In this case, new domains correspond to the images in clip-art format. In this paper, we concentrate on the CIL setting, which is a more challenging scenario in the open world.

There is also research about CIL before the prosperity of deep learning [11]. Typical methods try to solve the catastrophic forgetting problem with traditional machine learning models. However, most of them address the incremental learning within two tasks, i.e., the model is only updated with a single new stage [12], [13]. Furthermore, the rapid development of data collection and processing requires the model to grasp long-term and massive-scale data streams that traditional machine learning models cannot handle. Correspondingly, deep neural networks with powerful representation ability well suit these requirements. As a result, deep learning-based CIL is becoming a hot topic in the machine learning and computer vision community.

There are several related surveys discussing the incremental learning problem. For example, [6] focuses on the taskincremental learning problem and provides a comprehensive survey. [14] is a related survey on the class-incremental learning field, while it only discusses and evaluates the methods till 2020. However, with the rapid development of the CIL field, many great works are emerging day by day, which substantially boost the performance of benchmark settings [15], [16], [17], [18]. On the other hand, with the prosperity of Vision Transformer (ViT) [19] and pre-trained models, a heated discussion about ViT in CIL is attracting the attention of the community. Other surveys either focus on the specific field [20], [21] or lack the performance evolution among state-of-the-arts [8], [22], [23]. Hence, it is urgent to provide an up-to-date survey containing popular methods to speed up the development of the CIL field.

In this paper, we aim for a comprehensive review of classincremental learning methods and divide them into seven categories. We also provide a holistic comparison among different kinds of methods over benchmark datasets, i.e., CIFAR100 [24] and ImageNet100/1000 [25]. On the other hand, we highlight an important factor in CIL model evaluation, i.e., memory budget, and advocate fair comparison among different methods with an aligned budget. Correspondingly, we holistically evaluate the extensibility of CIL models with budget-agnostic measures.

In general, the contribution of this survey can be summarized as follows: 1): We provide a comprehensive survey of CIL, including problem definitions, benchmark datasets, and different families of CIL methods. We organize these algorithms taxonomically (Table I) and chronologically (Fig. 3) to give a holistic overview of state-of-the-art. 2): We provide a rigorous and unified comparison among different methods on several publicly available datasets, including traditional CNN-backed and modern ViT-backed methods. We also discuss the insights and summarize the common rules to inspire future research. 3): To boost real-world applications, CIL models should be deployed not only on high-performance computers but also on edge devices. Therefore, we advocate evaluating different methods holistically by emphasizing the effect of memory budgets. Correspondingly, we provide a comprehensive evaluation of different methods given specific budgets as well as several new performance measures.

# II. PRELIMINARIES

# A. Problem Formulation

Definition 1. Class-Incremental Learning: aims to learn from an evolutive stream with new classes. Assume there is a sequence of B training task $\mathrm { ~ s ~ } ^ { 1 } \left\{ \mathcal { D } ^ { 1 } , \mathcal { D } ^ { 2 } , \ldots \right\} $ without overlapping classes, where $\mathcal { D } ^ { b } = \{ \tilde { ( } \mathbf { x } _ { i } ^ { b } , y _ { i } ^ { b } ) \} _ { i = 1 } ^ { n _ { b } }$ is the b-th incremental step with $n _ { b }$ = (training instances. $\mathbf { x } _ { i } ^ { b } \in \mathbb { R } ^ { D }$ is an instance of class $y _ { i } ^ { b } \in Y _ { b } , Y _ { b }$ is the label space of task $b ,$ where $Y _ { b } \cap Y _ { b ^ { \prime } } = \emptyset$ for $b \neq b ^ { \prime }$ . We can only access data from $\mathcal { D } ^ { b }$ =when training task =b. The ultimate goal of CIL is to continually build a classification model for all classes. In other words, the model should not only acquire the knowledge from the current task $\mathcal { D } ^ { b }$ but also preserve the knowledge from former tasks. After each task, the trained model is evaluated over all seen classes $\mathcal { V } _ { b } = Y _ { 1 } \cup \cdot \cdot \cdot Y _ { b }$ . Formally, CIL aims to fit a model $f ( \mathbf { x } ) : X \to \mathcal { V } _ { b }$ , which minimizes the expected risk:

TABLE I THE TAXONOMY OF CIL 

<table><tr><td colspan="2">Algorithm Category</td><td>Reference</td></tr><tr><td rowspan="2">§ 3.1Data Replay</td><td>Direct Replay</td><td>[26], [33], [34], [35], [36], [37], [38], [39], [40]</td></tr><tr><td>Generative Replay</td><td>[41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52]</td></tr><tr><td colspan="2">§ 3.2: Data Regularization</td><td>[34], [53], [54], [55], [56], [57], [58]</td></tr><tr><td rowspan="3">§ 3.3Dynamic Networks</td><td>Neuron Expansion</td><td>[45], [59], [60], [61]</td></tr><tr><td>Backbone Expansion</td><td>[15], [16], [17], [62], [63], [64], [65], [66], [67], [68]</td></tr><tr><td>Prompt Expansion</td><td>[18], [69], [70], [71], [72]</td></tr><tr><td colspan="2">§ 3.4: Parameter Regularization</td><td>[33], [73], [74], [75], [76], [77], [78], [79], [80]</td></tr><tr><td rowspan="3">§ 3.5Knowledge Distillation</td><td>Logit Distillation</td><td>[81], [82], [83], [84], [85], [86], [87], [88]</td></tr><tr><td>Feature Distillation</td><td>[89], [90], [91], [92], [93], [94], [95], [96], [97], [98]</td></tr><tr><td>Relational Distillation</td><td>[27], [99], [100], [101], [102], [103]</td></tr><tr><td rowspan="3">§ 3.6Model Rectify</td><td>Feature Rectify</td><td>[104], [105], [106], [107], [108], [109]</td></tr><tr><td>Logit Rectify</td><td>[83], [89], [110], [111]</td></tr><tr><td>Weight Rectify</td><td>[112], [113], [114]</td></tr><tr><td colspan="2">§ 3.7: Template-Based Classification</td><td>[37], [82], [104], [115], [116], [117], [118], [119], [120], [121], [122], [123]</td></tr></table>

The shade color in the last column denotes the subcategory,which is consistent with Figure 3.

![](images/b51db1eefe541bee74bdf1bfd3080b85963cc57fad08854c413366036d67c36a.jpg)  
Fig. 3. The roadmap of class-incremental learning. We organize representative methods chronologically to show the concentration at different stages. Different colors of these methods denote the sub-categories in Table I. Knowledge distillation and data replay dominated the research before 2021, while model rectify and dynamic networks became popular after 2021.

$$
f ^ {*} = \underset {f \in \mathcal {H}} {\operatorname{argmin}} \mathbb {E} _ {(\mathbf {x}, y) \sim \mathcal {D} _ {t} ^ {1} \cup \dots \mathcal {D} _ {t} ^ {b}} \mathbb {I} (y \neq f (\mathbf {x}))  , \tag {1}
$$

where H is the hypothesis space, I · is the indicator function ( )which outputs 1 if the expression holds and 0 otherwise. $\mathcal { D } _ { t } ^ { b }$ denotes the data distribution of task b. A good CIL model satisfying (1) has discriminability among all classes, which not only works well on new classes but also preserves the knowledge of former ones.

Class Overlapping: Typical CIL setting assumes $Y _ { b } \cap Y _ { b ^ { \prime } }$ ∅ for $b \neq b ^ { \prime } , \mathrm { i . e . }$ , there are no overlapping classes in different tasks. However, in the real world, it is common to observe the old classes emerging in new tasks. When $Y _ { b } \cap Y _ { b ^ { \prime } } \neq \emptyset$ , the setting =is called blurry class-incremental learning (Blurry CIL) [26]. It enables the model to revisit former instances in the later stage, which weakens the learning difficulty.

Number of Instances Typical CIL setting assumes the training data of each class is balanced and many-shot (e.g., hundreds or thousands). However, the data collection may face challenges in the real world, e.g., we can only collect a limited number of training instances for rare birds. When the training instances of new classes are limited (e.g., 5-shot per class), the setting is called Few-Shot Class-Incremental Learning (FSCIL) [27]. When the training instances are highly imbalanced and long-tailed, the setting is called Long-Tailed Class-Incremental Learning (LT-CIL) [28].

Online CIL: Although data comes with stream format, the model can conduct multi-epoch training with each task, i.e., offline training within each task. There are some works addressing fully online (one-pass) CIL, where each batch can be processed once and then dropped [29]. It is a specific case of the current setting, and we concentrate on the generalized CIL setting in this paper.

In the following discussions, we decompose the CIL model into the embedding module and linear layers, $\operatorname { i . e . , } \ f ( \mathbf { x } )$ $\mathbf { \Phi } = W ^ { \top } \phi ( \mathbf { x } ) , ^ { 2 }$ 2 where $\mathbf { \bar { \phi } } ( \cdot ) : \mathbb { R } ^ { D }  \mathbb { R } ^ { d } , W \in \mathbb { R } ^ { \bar { d } \times | \mathcal { V } _ { b } | }$ ( )|. The linear = ( ) ( ) :layer can be further decomposed into the combination of classifiers: $W = [ { \pmb w } _ { 1 } , { \pmb w } _ { 2 } , \ldots , { \pmb w } _ { | \mathcal { V } _ { b } | } ]$ , where $\boldsymbol { w } _ { k } \in \mathbb { R } ^ { d } , | \cdot$ | denotes = [ ]the size of the set. The logits are then passed to the Softmax activation for further optimization, i.e., the output probability on class k is denoted as:

$$
\mathcal {S} _ {k} (f (\mathbf {x})) = \frac {\exp^ {\boldsymbol {w} _ {k} ^ {\top} \phi (\mathbf {x}) / \tau}}{\sum_ {j = 1} ^ {| \mathcal {Y} _ {b} |} \exp^ {\boldsymbol {w} _ {j} ^ {\top} \phi (\mathbf {x}) / \tau}}, \tag {2}
$$

where τ is the temperature parameter.

Backbones: As defined in (2), the predictions are derived by feeding instance x into the embedding function and linear layer. The embedding function is designed to project input instances into the embedding space to reflect its semantic information. Hence, ideal CIL algorithms should work for any type of backbones, e.g., multi-layer perceptron (MLP) [30], convolutional neural network (CNN) [31], and Vision Transformer (ViT) [19]. Specifically, we often treat the whole image as the input of MLP and CNN and utilize the final product as the embedding. By contrast, ViT transforms the image into a sequence of patches for patch features. These patches are then fed forward to the self-attention modules [32] and MLP layer to get the contextualized information. Typical ViT appends an extra token (i.e., [CLS] token) to the set of patches and utilizes the final representation of [CLS] token as the embedding. Since ViT relies on the self-attention mechanism to relate and adjust patch-wise features, it is intuitive to influence the embedding context by adding task-specific tokens as the input. The basic difference between these backbones triggers different tuning algorithms in CIL, which will be further discussed in Section III-C.

Baseline in CIL: In CIL, the typical baseline is sequential finetuning the model with the current dataset $\mathcal { D } ^ { b }$ (denoted as ‘Finetune’), whose loss function can be denoted as:

$$
\mathcal {L} = \sum_ {(\mathbf {x}, y) \in \mathcal {D} ^ {b}} \ell (f (\mathbf {x}), y), \tag {3}
$$

where $\ell ( \cdot , \cdot )$ measures the discrepancy between inputs, e.g., cross-entropy loss. Finetune is known as the baseline method for CIL since it only concentrates on learning the new concepts in the current task. Consequently, it suffers severe forgetting since the model pays no attention to former ones.

# B. Exemplars and Exemplar Set

As defined in Definition 1, in each incremental task, the model can only access the current dataset $\mathcal { D } ^ { b }$ . This helps to preserve user privacy and release the storage burden. However, this restriction is relaxed in some cases, and the model can keep a relatively small set, namely exemplar set, to reserve the representative instances from former tasks.

Definition 2: Exemplar Set3 is an extra collection of instances from former tasks $\mathcal { E } = \{ ( \mathbf { x } _ { j } , y _ { j } ) \} _ { j = 1 } ^ { M } , y _ { j } \in \mathcal { V } _ { b - 1 }$ . With the help of the exemplar set, the model can utilize $\mathcal { E } \cup \mathcal { D } ^ { b }$ for the update within each task. The model manages the exemplar set after the training process of each task.

Exemplar Set Management: Since the data stream is evolving, there are two main strategies to manage the exemplar set in CIL [89]. The first way is to keep a fixed number of exemplars per class, e.g., R per class. Under such circumstances, the size of the exemplar set will grow as the data stream evolves — the model keeps $R | \mathcal { V } _ { b }$ | after the b-th task. This will result in a linearly growing memory budget, which is inapplicable in real-world learning systems. To this end, another strategy advocates saving a fixed number of exemplars, e.g., M. The model keeps $\big [ \frac { M } { | \mathcal { V } _ { b } | } \big ]$ instances per class, where · denotes floor function. It helps [ ]to keep a fixed size of exemplars in the memory and release the storage burden. In this paper, we use the second strategy to organize the exemplar set.

Exemplar Selection: Exemplars are representative instances of each class, which need to be selected from the entire training set. An intuitive way to choose the exemplars is random sampling, which results in diverse instances. By contrast, a commonly used strategy is called herding [82], [124], aiming to select the most representative ones of each class. Given the instance set $X = \{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { n } \}$ from class $y ,$ herding first =calculates the class center with current embedding $\textstyle { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \phi ( \mathbf { x } _ { i } )$ ( ) . Afterward, it iteratively appends instances into $\phi ( \cdot ) \colon \mu _ { y } \gets$

the exemplar set:

$$
\boldsymbol {p} _ {k} \leftarrow \underset {\mathbf {x} \in X} {\operatorname{argmin}} \left\| \boldsymbol {\mu} _ {y} - \frac {1}{k} \left[ \phi (\mathbf {x}) + \sum_ {j = 1} ^ {k - 1} \phi (\boldsymbol {p} _ {j}) \right] \right\|, \tag {4}
$$

until k reaches the memory bound of each class $( \big [ \frac { M } { | \mathcal { V } _ { b } | } \big ] )$ . The exemplar set of class y is the concatenation of $\{ p _ { 1 } , p _ { 2 } , \cdot \cdot \cdot p _ { [ \frac { M } { | \mathcal { y } _ { h } | } ] } \}$ . Equation (4) ensures that the average feature vector over all exemplars selected thus far is closest to the class mean. Since the class center can be seen as the most representative pattern of each class, selecting exemplars near the center also enhances the representativeness. Herding is now a commonly used strategy to select exemplars in CIL, and we also adopt it in this paper.

# III. CLASS-INCREMENTAL LEARNING: TAXONOMY

There are numerous works addressing CIL in recent years, raising a heated discussion among machine learning and computer vision society. We organize these methods taxonomically from seven aspects, as shown in Table I. Specifically, data replay and data regularization concentrate on solving CIL with exemplars, either by revisiting former instances or using them as indicators to regularize model updating. Dynamic networks expand the network structure for stronger representation ability, while parameter regularization-based methods regularize the model parameters to prevent them from drifting away to resist forgetting. Moreover, knowledge distillation builds the mapping between incremental models to resist forgetting, and model rectification aims to reduce the biased prediction of incremental learners. Template-based classification aims to transform inference into query-template matching. We list the representative methods chronologically in Fig. 3 to show the research focus of different periods.

Note that the classification rules of these categories are based on the ‘key point’ (or the special focus) of each algorithm. With the rapid development of class-incremental learning, some techniques are becoming the well-acknowledged baseline to be shared among a variety of algorithms. Hence, these seven categories are not mutually exclusive, and there is no strict boundary between them. We organize them into several categories to enhance a holistic understanding of CIL from a specific perspective. In the following sections, we will discuss CIL methods from these aspects.

# A. Data Replay

‘Replay’ is important in human cognition system [125], [126], [127] — a student facing final exams shall go over the textbooks to recall former memory and knowledge. This phenomenon can also be extended to the network training process, where a network can overcome catastrophic forgetting by revisiting former exemplars. Correspondingly, an intuitive way [82], [128] is to save an extra exemplar set E (as defined in Definition 2) and include it into the model updating process:

$$
\mathcal {L} = \sum_ {(\mathbf {x}, y) \in (\mathcal {D} ^ {b} \cup \mathcal {E})} \ell (f (\mathbf {x}), y). \tag {5}
$$

Comparing (5) to (3), we can find that exemplars are concatenated to the training set for updating, enabling the retrospective review of former knowledge when learning new concepts.

Hot to Construct the Exemplar Set? Utilizing the exemplar set for rehearsal is simple yet effective, which leads to numerous following works. The exemplar sampling process is also similar to the active learning protocol, and some works propose corresponding sampling measures to select the informative ones. For example, [33] suggests sampling exemplars with high prediction entropy and near the decision boundary. The model will achieve higher generalization ability by replaying these ‘hard’ exemplars. [26] proposes estimating exemplars’ uncertainty via data augmentation. They choose instances with large prediction diversity by aggregating the predictions of multiple augmented instances. Similarly, [34] proposes to sample exemplars with a greedy strategy in online incremental learning. It proves that exemplar selection is equivalent to maximizing the diversity of exemplars with parameters gradient as the feature. Without explicit task boundaries, [35] explores the reservoir sampling process to ensure the exemplars are i.i.d. sampled. [36] formulates the replay process into a bi-level optimization and keeps intact predictions on some anchor points of past tasks. [37] introduces data replay in prototypical network [129], and utilizes the exemplars as pseudo-prototypes for embedding evaluation. Mnemonics [40] proposes a way to parameterize exemplars and optimize them in a meta-learning manner. The framework is trained through bi-level optimizations, i.e., model-level and exemplar-level, which can be combined with various CIL algorithms.

Memory-efficient memory: Since exemplars are raw images, directly saving a set of instances may consume enormous memory costs. To this end, several works are proposed to build a memory-efficient replay buffer [17]. [38] argues that extracted features have lower dimensions than raw images and proposes saving features in the exemplar set to release the burden. Similarly, [39] proposes keeping low-fidelity images instead of raw ones. However, since the distributions of extracted features and low-fidelity images may differ from the raw images, an extra adaptation process is needed for these methods, adding to the algorithm’s complexity.

Generative replay The above-mentioned methods achieve competitive performance by replaying former instances in the memory. Apart from directly saving instances in the exemplar set, generative models show the potential to model the distribution and generate instances [153], [154], which have also been applied to class-incremental learning. We denote the aforementioned methods directly saving instances for replay as ‘direct replay,’ and the methods utilizing generative models as ‘generative replay.’

There often exist two models in generative replay-based CIL, i.e., the generative model for data generation and the classification model for prediction. GR [41] first proposes to utilize the generative adversarial network (GAN) [153] in CIL. In each updating process, it utilizes the GAN to generate the instances from former classes and then updates GAN and classification model with both old and new classes. ESGR [42] extends GR by saving extra exemplars. It also proposes to train a separate GAN for each incremental task, which does not require updating GAN incrementally. [43] extends GR by introducing the dynamic parameter generator for model adaptation at test time. FearNet [44] explores brain-inspired CIL and uses a dual-memory system in which new memories are consolidated from a network for recent memories. Recently, [45], [46], [47] explore the application of conditional GAN [155] in CIL, and [48] adopts variational auto-encoders (VAE) [154] to model data distribution. Similarly, [49], [50] model each class into a Gaussian distribution and sample instances directly from the class center. With the prosperity of diffusion models [156], recent works also consider using diffusion models as the data generator for high-quality samples. Correspondingly, SDDR [51] studies the use of a pre-trained diffusion model as a complementary source of data. Similarly, DDGR [52] adopts a diffusion model as the generator and calculates an instruction operator through the classifier to instruct the generation of samples. There are also works [157], [158] on solving the catastrophic forgetting of these diffusion models.

Discussions: Direct replay is a simple yet effective strategy, which has been widely applied to camera localization [159], semantic segmentation [160], video classification [161], and action recognition [91]. Since it directly optimizes the loss over old exemplars, it is found to help continual learners stay in the low-loss region of prior tasks during the optimization strategy [162]. However, since the exemplar set only saves a tiny portion of the training set, data replay may suffer the overfitting problem and weaken the generalizability [53], [162]. Considering repeated optimization on a small pool of data inevitably leads to tight and unstable decision boundaries, several works tackle this problem by constraining the model’s layerwise Lipschitz constants with regard to exemplars [163] or enlarging representational variations to alleviate representation collapse [164]. Besides, the data-imbalance problem [165] also occurs due to the gap between the few-shot exemplars and the many-shot training set. Iteratively optimizing the imbalanced training set introduces extra bias in the classifier, and several works address this problem with balanced sampling [16], [110]. Finally, since direct replay requires saving exemplars of former classes, it will face privacy issues when the raw data contains face images or resource deficiency when the raw data contains images of rare animals [27]. Under such circumstances, algorithms should be designed to learn without exemplars [49] or using privacy-friendly strategies like feature replay [50].

On the other hand, the performance of generative replay methods relies on the quality of generated data. They are found to work well on simple datasets [166], [167] while failing in complex, large-scale inputs [168]. To tackle this problem, some works find generating features is much easier than generating raw images in terms of computational complexity and semantic information. Hence, they either utilize conditional GAN to generate features [169] or VAE to model the internal representations [166]. Additionally, recent advances in diffusion models reveal a promising way to generate instances with pre-trained diffusion models [51], while utilizing pre-trained models leads to an unfair comparison to other methods without extra information. Furthermore, when sequentially updating the generative model, the catastrophic forgetting phenomena can also be observed on these generative models [157], [158]. As a result, algorithms should be designed to address catastrophic forgetting in two aspects (i.e., classifier aspect and generative model aspect) when using generative replay.

# B. Data Regularization

Apart from directly replaying former data, another group of works tries to regularize the model with former data and control the optimization direction. Since learning new classes will result in the catastrophic forgetting of old ones, the intuitive idea is to ensure that optimizing the model for new classes will not hurt former ones. GEM [53] aims to find the model that satisfies:

$$
f ^ {*} = \underset {f \in \mathcal {H}} {\operatorname{argmin}} \sum_ {(\mathbf {x}, y) \in \mathcal {D} ^ {b}} \ell (f (\mathbf {x}), y)
$$

$$
\text { s.t. } \sum_ {(\mathbf {x} _ {j}, y _ {j}) \in \mathcal {E}} \ell (f (\mathbf {x} _ {j}), y _ {j}) \leq \sum_ {(\mathbf {x} _ {j}, y _ {j}) \in \mathcal {E}} \ell (f ^ {b - 1} (\mathbf {x} _ {j}), y _ {j}), \tag {6}
$$

where $f ^ { b - 1 }$ stands for the incremental model after training the last task $\mathcal { D } ^ { b - 1 }$ . Equation (6) optimizes the model with a restriction, which requires the loss calculated with the exemplar set not to exceed the former model. Since exemplars are representative instances from former classes, GEM strikes a balance between learning new classes and preserving former knowledge. Furthermore, it transforms the constraints in (6) into:

$$
\langle g, g _ {o l d} \rangle := \left\langle \frac {\partial \ell (f (\mathbf {x}) , y)}{\partial \theta}, \frac {\partial \ell (f , \mathcal {E})}{\partial \theta} \right\rangle \geq 0, \tag {7}
$$

where $g , g _ { o l d }$ denotes the gradients of the current updating step and exemplar set, respectively. Equation (7) requires the angle between gradients to be acute. If all the inequality constraints are satisfied, then the proposed parameter update is unlikely to increase the loss of previous tasks. However, if violations occur, GEM proposes to project the gradients g to the closest gradient g satisfying the constraints. GEM further transforms the optimization into a Quadratic Program (QP) problem. However, since the regularization in (7) is defined among all exemplars, it requires calculating loss among exemplar sets and solving the QP problem in each optimization step. Hence, optimizing GEM is very time-consuming. To this end, A-GEM [54] is proposed to speed up the optimization by relaxing the constraints in (7) into a random batch. A similar idea is also adopted in [34].

There are other methods to address the regularization problem with exemplars. For example, Adam-NSCL [55] proposes to sequentially optimize network parameters by projecting the candidate parameter update into the approximate null space of all previous tasks. OWM [56] only allows weight modification in the direction orthogonal to the subspace spanned by all previously learned inputs. LOGD [57] further decomposes the gradients into shared and task-specific ones. In model updating, the gradient should be close to the gradient of the new task, consistent with the gradients shared by all old tasks, and orthogonal to the space spanned by the gradients specific to the old tasks.

Discussions: Data regularization methods utilize the exemplar set in another manner, i.e., treating the loss of them as the indicator of forgetting. They assume that the loss on exemplars is consistent with the prior tasks and align the parameter updates with the direction of the exemplar set. Hence, previous knowledge can be preserved for these exemplars due to the aligned gradient direction. However, these assumptions may not stand in some cases [54], which results in poor performance. To this end, [170] gets rid of the requirement of exemplars and manually projects the gradient direction to be orthogonal with previous ones. On the other hand, some works assume the updating rule can be meta-learned [171] from a series of related tasks. MER [58] regularizes the objective of data replay so that gradients on incoming examples are more likely to have transferability and less likely to have interference with respect to past examples. iTAML [172] separates the generic feature extraction module from the task-specific classifier, thereby minimizing interference and promoting a shared feature space among tasks. [173] extends data replay with adversarial attack and meta-learns an adaptive fusion module to help allocate capacity to the knowledge of different difficulties.

Moreover, since data regularization and data replay both need to save previous data in the memory, similar problems will also occur in data regularization-based methods, e.g., overfitting, generalization issues [53], [162], and privacy concerns [149]. Consequently, it would be interesting to design privacy-friendly algorithms to build the regularization term with intermediate products for real-world applications.

# C. Dynamic Networks

Deep neural networks are proven to produce task-specific features [174]. For example, when the training dataset contains ‘cars,’ the model tends to depict the wheels and windows. However, if the model is updated with new classes containing ‘cats,’ the features would be adapted for beards and stripes. Since the capacity of a model is limited, adapting to new features will result in the overwriting of old ones and forgetting [59]. Hence, utilizing the extracted features for beards and strides is inefficient for recognizing a car. To this end, dynamic networks are designed to dynamically adjust the model’s representation ability to fit the evolving data stream. There are several ways to expand representation ability, and we divide them into three sub-groups, i.e., neuron expansion, backbone expansion, and prompt expansion.

1) Neuron Expansion: Early works focus on neuron expansion adding neurons when the representation ability is insufficient to capture new classes. DEN [59] formulates the adjusting process into selection, expansion, duplication, and elimination. Facing a new incremental task, the model first selectively retrains the neurons that are relevant to this task. If the retrained loss is still above some threshold, DEN considers expanding new neurons top-down and eliminating the useless ones with groupsparsity regularization. Afterward, it calculates the neuron-wise drift and duplicates neurons that drift too much from the original values. Apart from heuristically expanding and shrinking the network structure, RCL [60] formulates the network expansion process into a reinforcement learning problem and searches for the best neural architecture for each incoming task. Similarly, Neural Architecture Search (NAS) [175] is also adopted to find the optimal structure for each of the sequential tasks [61].

2) Backbone Expansion: Expanding neurons shows competitive results with expandable representation. Correspondingly, several works try to duplicate the backbone network for stronger representation ability. PNN [62] proposes learning a new backbone for each new task and fixing the former in incremental learning. It also adds layer-wise connections between old and new models to reuse former features. Expert Gate [63] also expands the backbone per incremental task while it requires learning an extra gate to map the instance to the most suitable pathway during inference. To release the expansion cost, P&C [64] suggests a progression-compression protocol — it first expands the network to learn representative representations. Afterward, a compression process is conducted to control the total budget. AANets [66] partially expands stable and plastic blocks and aggregates their predictions to enhance model’s representation ability. [65], [67] maintain a dual-branch network for class-incremental learning, one for fast adaptation and one for slow adaption. Recently, DER [15] has been proposed to address the CIL problem. Similar to PNN, it expands a new backbone when facing new tasks and aggregates the features with a larger FC layer. Take the second incremental task for an example, where the model output is aggregated as:

$$
f (\mathbf {x}) = W _ {n e w} ^ {\top} [ \phi_ {o l d} (\mathbf {x}), \phi_ {n e w} (\mathbf {x}) ], \tag {8}
$$

where $\phi _ { o l d }$ is the former backbone, and $\phi _ { n e w }$ is the newly initialized backbone. ·, · denotes feature aggregation, and $W _ { n e w } \in \mathbb { R } ^ { 2 d \times | y _ { b } | }$ [ ]| is the newly initialized FC layer. In the model updating process, the old backbone is frozen to maintain former knowledge:

$$
\mathcal {L} = \sum_ {k = 1} ^ {| \mathcal {Y} _ {b} |} - \mathbb {I} (y = k) \log \mathcal {S} _ {k} (W _ {n e w} ^ {\top} [ \bar {\phi} _ {o l d} (\mathbf {x}), \phi_ {n e w} (\mathbf {x}) ]), \tag {9}
$$

where $\bar { \phi } _ { o l d } ( \mathbf { x } )$ denotes the old backbone is frozen. DER also ( )adopts an auxiliary loss to differentiate between old and new classes. Equation (9) depicts a way to continually expand the model with new features. Under such circumstances, if the old backbone is optimized with ‘cars,’ the features extracted by $\phi _ { o l d }$ are then representative of wheels and windows. The new backbone trained with ‘cats’ is responsible for extracting beards and stripes. Since the old backbone is frozen in later stages, learning new classes will not overwrite the features of old ones, and forgetting is alleviated. Fig. 4 (left) depicts the model evolution of DER.

However, saving a backbone per task requires numerous memory size in DER, and many works are proposed to obtain expandable features with a limited memory budget. FOSTER [16] formulates the learning process in (9) as a feature-boosting [176] problem. It argues that not all expanded features are needed for incremental learning and need to be integrated to reduce redundancy. For example, suppose old classes contain ‘tigers,’ and new classes contain ‘zebras.’ In that case, the stripe will be a useful feature that both old and new backbones could extract. Under such circumstances, forcing the new backbone to extract the same features is less effective for recognition. Hence, FOSTER adds an extra model compression process by

![](images/9072b55df8bc9f549428673d14841972d35dca31e28c96fb6ab90a891c12c7cf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph DER
        A["Task 3"]
        B["Task 2"]
        C["Task 1"]
    end

    subgraph FOSTER
        D["Distill"]
        E["Distill"]
    end

    subgraph MEMO
        F["Specialized Blocks: New, Old"] --> G["Model Inherit"]
    end

    A --> D
    B --> E
    C --> F
    D --> H["Block (New)"]
    D --> I["Block (Old)"]
    E --> J["Specialized Blocks: Generalized Blocks"]
```
</details>

Fig. 4. Illustration of network structure evolving in backbone expansion. Left: DER expands a new backbone per incremental task. Middle: FOSTER adds an extra model compression stage, which maintains limited model storage. Right: MEMO decouples the network structure and only expands specialized blocks.

knowledge distillation [177]:

$$
\min _ {f _ {s} (\mathbf {x})} \mathrm{KL} \left(\mathcal {S} \left(f _ {t} (\mathbf {x})\right) \| \mathcal {S} \left(f _ {s} (\mathbf {x})\right)\right). \tag {10}
$$

Equation (10) aims to find the student model $f _ { s }$ that has the same discrimination ability as the teacher model $f _ { t }$ by minimizing the discrepancy between them. The teacher is the frozen expanded model with two backbones: $f _ { t } ( \mathbf { x } ) = W _ { n e w } ^ { \top } [ \phi _ { o l d } ( \mathbf { x } ) , \phi _ { n e w } ( \mathbf { x } ) ]$ , ( ) =and the student is the newly initialized model $f _ { s } ( \mathbf { x } ) = W ^ { \top } \phi ( \mathbf { x } )$ . ( ) = ( )Hence, the number of backbones is consistently limited to a single one, and the memory budget will not suffer catastrophic expansion. Fig. 4 (middle) depicts the model evolution of FOS-TER.

MEMO [17] addresses the memory problem in CIL, aiming to enable model expansion with the least budget cost. It finds that in CIL, shallow layers of different models are similar, while deep layers are diverse. In other words, shallow layers are more generalizable, while deep layers are specific to the task, making expanding shallow layers less memory-efficient for CIL. Hence, MEMO proposes to decouple the backbone at middle layers: $\phi ( \mathbf { x } ) = \phi _ { s } ( \phi _ { g } ( \mathbf { x } ) )$ , where specialized block $\phi _ { s }$ corresponds to ( ) = ( ( ))the deep layers in the network, while generalized block $\phi _ { g }$ corresponds to the rest shallow layers. Compared to DER, MEMO only expands specialized blocks $\phi _ { s }$ , and transforms (9) into:

$$
\sum_ {k = 1} ^ {| \mathcal {Y} _ {b} |} - \mathbb {I} (y = k) \log \mathcal {S} _ {k} (W _ {n e w} ^ {\top} [ \phi_ {s o l d} (\phi_ {g} (\mathbf {x})), \phi_ {s n e w} (\phi_ {g} (\mathbf {x})) ]),
$$

which indicates that task-specific deep layers can be built for each task upon the shared shallow layers $\phi _ { g } ( \mathbf { x } )$ . Fig. 4 (right) depicts the model evolution of MEMO.

3) Prompt Expansion: Recently, Vision Transformer (ViT) [19] has attracted the attention of the computer vision community, and many works tend to design CIL learners using ViT as the backbone. DyTox [18] is the first work to explore ViT in CIL, which finds that model expansion in ViT is much easier than in convolutional networks. In DyTox, only task tokens are expanded for each new task, which requires much less memory than saving the whole backbone. Similarly, L2P [69] and DualPrompt [70] explore how to build CIL learners with pre-trained ViT. They borrow ideas from Visual Prompt Tuning (VPT) [178] to incrementally finetune the model with prompts. In L2P, the pre-trained ViT is frozen during the learning process, and the model only optimizes the prompts to fit new patterns. The prompt pool is defined as: $\mathbf { P } = \{ P _ { 1 } , P _ { 2 } , \ldots , P _ { M } \}$ , where =M is the total number of prompts, $P _ { i } \in \mathbb { R } ^ { L _ { p } \times d }$ is a single prompt with token length $L _ { p }$ and the same embedding size d as the instance embedding φ x . The prompts are organized ( )as key-value pairs — each instance selects the most similar prompts in the prompt pool via KNN search. It obtains instance-specific predictions by adapting the input embeddings as: $\begin{array} { r } { \mathbf { x } _ { p } = [ P _ { s _ { 1 } } ; \cdot \cdot \cdot ; P _ { s _ { N } } ; \phi ( \mathbf { x } ) ] , ~ 1 \le N \le M , } \end{array}$ where $P _ { s _ { j } }$ are = [ ; ; ; ( )] 1the selected instance-specific prompts. The adapted embeddings are then fed into the self-attention layers [32] to obtain instance-specific representations. CODA-Prompt [71] extends the prompt search with the attention mechanism. Apart from pre-trained ViT, S-Prompt [72] utilizes the pre-trained languagevision model CLIP [179] for CIL, which simultaneously learns language prompts and visual prompts to boost representative embeddings.

Discussions: Learning dynamic networks, especially backbone expansion methods, has achieved state-of-the-art performance in recent years [15], [16], [17], [68]. However, it often requires expandable memory budgets, which is unsuitable for incremental learning on edge devices. To tackle this problem, further model compression [16], decoupling [17], and pruning can be adopted to alleviate the memory budget. Additionally, training DER requires an individual backbone for each task and aggregates all historical backbones as the feature extractor. It implicitly results in an unfair comparison to other methods with a single backbone [17]. In this paper, we systematically investigate the fair comparison protocol between these dynamic networks and others in Section IV-D and IV-E. Additionally, expanding backbones ignores the semantic information across tasks, e.g., when the old task contains ‘tigers’ and the new task contains ‘zebras,’ features like ‘stripes’ will be extracted by multiple backbones, resulting in feature redundancy. Hence, analyzing the semantic relationship across tasks [88] can help detect the feature redundancy, and contrastive learning [180] can be adopted for generalizable features.

On the other hand, most prompt expansion methods rely on the pre-trained models as initialization. Without such generalizable backbones, lightweight model updating with prompts often fails [181]. However, a pre-trained model is not always available for some specific downstream tasks, e.g., face recognition and speech recognition. Therefore, how to get rid of the dependence on pre-trained models is essential for these methods in real-world applications. Additionally, these prompt expansion-based methods tend to select instance-specific prompts based on a batch of instances. This requirement also needs to be satisfied during inference for accurate prompt retrieval [69], which implicitly results in an unfair comparison. Since a batch of instances is utilized to get the prompt, the context among instances becomes available, which is against the common sense of i.i.d. testing in machine learning.

Apart from these groups, there are works addressing network masks to divide a large network into sub-networks for each task [132], [135], [136]. However, deciding the activation of a specific sub-network requires the task identifier or learning extra task classifiers. On the other hand, several works [152], [182] propose to design specific modules for incremental new tasks, e.g., adapters [183]. However, manually handcrafting these modules requires heuristic designs or task-specific priors.

# D. Parameter Regularization

Dynamic networks seek to adjust model capacity with data evolves. However, if the model structure is fixed and unchangeable, how can we adjust the plasticity to resist catastrophic forgetting? Parameter regularization methods consider that the contribution of each parameter to the task is not equal. Hence, they seek to evaluate each parameter’s importance to the network and keep the important ones static to maintain former knowledge.

Typical works estimate a distribution over the model parameters and use it as the prior when learning new tasks. Due to the large amounts of parameters, the estimation process often assumes them to be independent. EWC [73] is the first work addressing parameter regularization. It maintains an importance matrix with the same scale of the network, i.e., . Denote the k-th model parameters as $\theta _ { k }$ , the importance of $\theta _ { k }$ Ωis represented by $\Omega _ { k } \geq 0$ (the larger $\Omega _ { k }$ indicates $\theta _ { k }$ is more important). Apart Ω 0 Ωfrom the training loss in (3) to learn new classes, EWC builds an additional regularization term to remember old ones:

$$
\mathcal {L} = \ell (f (\mathbf {x}), y) + \frac {1}{2} \lambda \sum_ {k} \Omega_ {k} (\theta_ {k} ^ {b - 1} - \theta_ {k}) ^ {2}. \tag {11}
$$

The parameter-wise regularization term is calculated based on two parts. θb−1k . He $\theta _ { k } ^ { b - 1 }$ k-th parameter after learning last taskrepresents the parameter drift from $\mathcal { D } ^ { b - \bar { 1 } }$ $( \theta _ { k } ^ { b - 1 } - \theta _ { k } ) ^ { 2 }$ (the last stage, and $\Omega _ { k }$ )weighs it to ensure important parameters do not shift away from the last stage. Since the model at the last stage represents the ‘old’ knowledge, consolidating important parameters can prevent the knowledge from being forgotten.

Equation (11) depicts a way to penalize essential parameters, and there are different ways to calculate the importance matrix . In EWC, Fisher information matrix [184] is adopted to Ωestimate . However, the importance calculation in EWC is Ωconducted at the end of each task, which ignores the optimization dynamics along the model training trajectory. To this end, SI [74] proposes to estimate  in an online manner and weigh the Ωimportance via its contribution to loss decay. RWalk [33] combines these importance estimation techniques. [75], [76] resort to an extra unlabeled dataset for online evaluation. IMM [77] finds a maximum of the mixture of Gaussian posteriors with the estimated Fisher information matrix. IADM [78] and CE-IDM [79] analyze the capacity and sustainability of different layers and find that different layers have different characteristics in CIL. In detail, shallow layers converge faster but have limited representation ability. By contrast, deep layers converge slowly while having powerful discrimination abilities. Hence, IADM augments EWC with an ensemble of different layers and learns layer-wise importance matrix in an online manner. K-FAC [80] extends the Fisher information matrix approximation with the Kronecker factorization technique.

Discussions: Although parameter and data regularization (Section III-B) both exert regularization terms to resist forgetting, their basic idea differs substantially. Specifically, data regularization relies on the exemplar set to direct the optimization direction, while parameter regularization is based on the parameter-wise importance to construct the regularization term.

As shown in Fig. 3, parameter regularization methods have attracted the attention of the community in the early years. [185] shows that despite stemming from very different motivations, both SI [74] and MAS [75] approximate the square root of the Fisher Information, with the Fisher being the theoretically justified basis of EWC. However, estimating parameter importance requires saving the matrix with the same scale as the backbone. It faces the same risk as dynamic networks in that the memory budget is linearly increasing when learning more and more tasks.

On the other hand, the importance matrix may conflict at different incremental stages [88], making it hard to optimize the model and achieve poor performance on new tasks. Hence, although these works achieve competitive results in task-incremental learning, many works [8] find that parameter regularization-based methods perform poorly in the classincremental learning scenario. To this end, some works try to alleviate the intransigence by learning a new backbone and consolidating them into a single one [64], [77]. In that case, the learning of new tasks will not be affected by the regularization term, and the parameter importance will only be considered during the consolidation process, enabling the model to be fully fitted to new tasks.

# E. Knowledge Distillation

Training data is evolving in the learning process, requiring tuning the model sequentially. We can denote the model after the previous stage $f ^ { b - 1 }$ as the ‘old model’ and the current updating model f a s the ‘new model.’ Assuming the old model is a good classifier for all the seen classes in $\mathcal { D } _ { b - 1 }$ , how can we utilize it to resist forgetting in the new model? To enable the old model to assist the new model, an intuitive way utilizes the concept proposed in [177], i.e., knowledge distillation (KD). KD enables the knowledge transfer from a teacher model to the student model, with which we can teach the new model not to forget. There are several ways to build the distillation relationship, and we divide these KD-based methods into three subgroups, i.e., logit distillation, feature distillation, and relational distillation.

LwF [81] is the first success to apply knowledge distillation into CIL. Similar to (11), it builds the regularization term via knowledge distillation to resist forgetting:

$$
\mathcal {L} = \underbrace {\ell (f (\mathbf {x}) , y)} _ {\text { Learning   New   Classes }} + \underbrace {\sum_ {k = 1} ^ {| \mathcal {Y} _ {b - 1} |} - \mathcal {S} _ {k} \left(f ^ {b - 1} (\mathbf {x})\right) \log \mathcal {S} _ {k} (f (\mathbf {x}))} _ {\text { Remembering   Old   Classes }}, \tag {12}
$$

where the old model $f ^ { b - 1 }$ is frozen during updating. The regularization term builds the mapping between the old and new models by forcing the predicted probability among old classes to be the same. Given a specific input x, the output probability of the k-th class reveals the semantic similarity of the input to this class. Hence, (12) forces the semantic relationship of the old and new models to be the same and resists forgetting. iCaRL [82] extends LwF with the exemplar set, which helps to further recall former knowledge during incremental learning. Additionally, it drops the fully-connected layers and follows [186] to utilize nearest-mean-of-exemplars during inference. Equation (12) strikes a trade-off between old and new classes, where the former part aims to learn new classes and the latter one maintains old knowledge. Since the number of old and new classes may differ in different incremental stages, BiC [83] extends Equation (12) by introducing a dynamic trade-off term:

$$
\mathcal {L} = (1 - \lambda) \ell (f (\mathbf {x}), y) + \lambda \sum_ {k = 1} ^ {| \mathcal {Y} _ {b - 1} |} - \mathcal {S} _ {k} (f ^ {b - 1} (\mathbf {x})) \log \mathcal {S} _ {k} (f (\mathbf {x})),
$$

where $\begin{array} { r } { \lambda = \frac { | \mathcal { V } _ { b - 1 } | } { | \mathcal { V } _ { b } | } } \end{array}$ denotes the proportion of old classes among =all classes. It increases as incremental tasks evolve, indicating that the model pays more attention to old ones.

LwF inspires the community to build the mapping between models, making knowledge distillation a useful tool in CIL. D+R [84] suggests changing the first part in (12) into a distillation loss by training an extra expert model. GD [85] proposes to select wild data for model distillation and designs a confidence-based sampling method to effectively leverage external data. Similarly, DMC [86] proposes to train a new model in each incremental stage and then compress them into a single model via an extra unlabeled dataset. Finally, if no additional data is available for knowledge distillation, ABD [87] proposes distilling synthetic data for incremental learning. These methods only concentrate on utilizing the old model to help resist forgetting in the new model. However, COIL [88] suggests conducting bidirectional distillation with co-transport, where semantic relationships between old and new models are both utilized.

Apart from distilling the logits, some works propose to distill the intermediate product of deep models, e.g., extracted features. UCIR [89] replaces the regularization term in (12) into:

$$
\mathcal {L} = \ell (f (\mathbf {x}), y) + \left(1 - \left\langle \frac {\phi^ {b - 1} (\mathbf {x})}{\| \phi^ {b - 1} (\mathbf {x}) \|}, \frac {\phi (\mathbf {x})}{\| \phi (\mathbf {x}) \|} \right\rangle\right). \tag {13}
$$

Equation (13) forces the features extracted by the new embedding module to be the same as the old one, which is a stronger regularization than (12). Several works follow it to utilize feature distillation in CIL [90], [91], [97], [98], while others address distilling other products. LwM [92] suggests penalizing the changes in classifiers’ attention maps to resist forgetting. AFC [93] conducts the distillation considering the importance of different feature maps. PODNet [94] minimizes the difference of the pooled intermediate features in the height and width directions instead of performing element-wise distillations. CVIC [187] decouples the distillation term into spatial and temporal features for video classification. DDE [95] distills the causal effect from the old training to preserve the old knowledge. GeoDL [96] conducts distillation based on the projection of two sets of features from old and new models.

![](images/e39afcc99c8bf46a53b3f53c8858d151317324a2ccbd707945cea8a11142c37e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Logit Distillation"] --> B["Embedding (Old)"]
    A --> C["Embedding (New)"]
    B --> D["Mapping"]
    C --> D
    D --> E["Instance ( "]
    D --> F["Instance Group"]
    E --> G["Classifier"]
    F --> G
    G --> H["Logits"]
    G --> I["Features"]
    J["Feature Distillation"] --> K["Embedding (Old)"]
    J --> L["Embedding (New)"]
    K --> M["Features"]
    L --> M
    M --> N["Feature Space"]
    N --> O["Mapping"]
    O --> P["Relational Distillation"]
    P --> Q["Embedding (Old)"]
    P --> R["Embedding (New)"]
    Q --> S["Feature Space"]
    R --> S
    S --> T["Feature Space"]
```
</details>

Fig. 5. Illustration of knowledge distillation in CIL. Left: Logit distillation aligns the model outputs to make the old and new models share the same semantic relationship. Middle: Feature distillation aligns the features produced by the old and new models to ensure the new model does not forget old features. Right: Relational distillation resorts to structural inputs, e.g., triples, and aligns the input relationship of the old and new model.

However, both logit and feature distillation address the instance-wise mapping between old and new models. To reveal the structural information in model distillation, several works suggest conducting relational knowledge distillation [188]. The differences between these groups of knowledge distillation are shown in Fig. 5.

To conduct relational distillation, a group of instances needs to be extracted, e.g., triplets. We denote the extracted triplets as $\left\{ \mathbf { x } _ { i } , \mathbf { x } _ { j } , \mathbf { x } _ { k } \right\}$ , where $\mathbf { x } _ { i }$ is called the anchor. In a triplet, the target neighbor $\mathbf { x } _ { j }$ is similar to the anchor $\mathbf { x } _ { i }$ with the same class, while the impostor $\mathbf { x } _ { k }$ is dissimilar to $\mathbf { x } _ { i }$ (usually from different classes). R-DFCIL [99] suggests mapping the angle among triplets:

$$
\sum_ {\left\{\mathbf {x} _ {i}, \mathbf {x} _ {j}, \mathbf {x} _ {k} \right\} \in \mathcal {D} ^ {b}} \| \cos \angle \mathbf {t} _ {i} \mathbf {t} _ {j} \mathbf {t} _ {k} - \cos \angle \mathbf {s} _ {i} \mathbf {s} _ {j} \mathbf {s} _ {k} \|, \tag {14}
$$

where $\mathbf { t } _ { m } = \phi ^ { b - 1 } ( \mathbf { x } _ { m } )$ is the representation in the old model’s = (embedding space, and $\mathbf { s } _ { m } = \phi ( \mathbf { x } _ { m } )$ denotes the representation = ( )in the current model. The cosine value is calculated in the corresponding embedding space. Eq. (14) provides a way to encode the old model’s structural information into the new model and align the feature space softly. ERL [100] extends this regularization into few-shot CIL scenarios. TPCIL [101] models the relationship with the elastic Hebbian graph and penalizes the changing of the topological relations between vertices. TOPIC [27] further explores neural gas network to model the class-wise relationship. Apart from the triplet relationship, MBP [102] extends the regularization to the instance neighborhood, requiring the old and new models to have the same distance ranking in the neighborhood.

Discussions: Knowledge distillation is a general idea to build the mapping between a set of methods, which has been widely adopted in class-incremental learning with many formats (e.g., logits, features, relationships). Due to their flexibility, knowledge distillation-based methods have also been widely applied to various incremental learning tasks, e.g., semantic segmentation [189], person re-identification [190], human action recognition [91], and federated learning [149]. Since a set of models exists in CIL, it is intuitive to build the student-teacher mapping in CIL, making knowledge distillation an essential solution for most works.

However, since the knowledge distillation term aims to strike a balance between learning the new and remembering the old, it is hard to control the precise trade-off term between plasticity and stability. Specifically, giving more importance to the knowledge distillation term will harm the plasticity of learning new tasks, while giving lower importance will cause catastrophic forgetting or feature overwriting. Compared to dynamic networks, knowledge distillation-based methods lack the ability to learn more informative features as data evolves. [17] compares knowledge distillation-based and dynamic network-based methods and finds that knowledge distillation-based methods and dynamic networks have their advantages given different memory budgets. Specifically, knowledge distillation methods show stronger performance given limited memory, while dynamic networks require an adequate memory budget to perform competitively. Besides, feature/relational distillation only regularizes the extracted features to be similar, thus regularizing the embedding function to resist forgetting. However, due to the data characteristics of CIL, the classifier layer also gets biased and forgets former knowledge and knowledge distillation-based methods cannot handle such challenges.

# F. Model Rectify

Assuming we can get all the training datasets at once and shuffle them for training with multiple epochs, the model will not suffer any forgetting and will perform well among all classes. Such a protocol is known as the upper bound of class-incremental learning, denoted as ‘Oracle.’ However, since the models trained with incremental data suffer catastrophic forgetting, several methods try to find the abnormal behaviors in CIL models and rectify them like the oracle model. These abnormal behaviors include the output logits, classifier weights, and feature embedding.

The first method addressing the bias of the CIL model is UCIR [89]. It finds that the weight norm of new classes is significantly larger than old ones, and the model tends to predict instances as the new classes with larger weights. Hence, UCIR proposes to utilize a cosine classifier to avoid the influence of biased classifiers: $\begin{array} { r } { f ( \mathbf { x } ) = \frac { W } { \| W \| } ^ { \top } \frac { \phi ( \mathbf { x } ) } { \| \phi ( \mathbf { x } ) \| } } \end{array}$ . Thus, the weight norm ( ) =will not influence model predictions in incremental learning. WA [112] further normalizes the weight after every optimization step. It also introduces weight clipping to ensure the predicted probability is proportionate to classifier weights. SS-IL [113] explains the reason for weight drifting, which is caused by the imbalance phenomena between old and new instances. Since the number of new class instances is much more than that of old ones, optimizing the model with cross-entropy loss will increase the weight of new classes and decrease old ones. Hence, SS-IL suggests separated softmax operation and task-wise knowledge distillation to alleviate the influence of imbalanced data. RPC [114] claims that the classifiers of all classes can be pre-allocated and fixed. This makes it impossible for classifiers to be biased toward new classes.

On the other hand, several works find that the predicted logits of new classes are much larger than old ones. E2E [110] proposes to finetune the fully-connected layers with a balanced dataset after each stage. Furthermore, BiC [83] proposes to attach an extra rectification layer to adjust the predictions. The extra layer only have two parameters, i.e., re-scale parameter α and bias parameter $\beta ,$ and the rectified output for the k-th class is denoted as:

$$
\hat {f} (\mathbf {x}) _ {k} = \left\{ \begin{array}{l l} \alpha \boldsymbol {w} _ {k} ^ {\top} \phi (\mathbf {x}) + \beta , & k \in Y _ {b} \\ \boldsymbol {w} _ {k} ^ {\top} \phi (\mathbf {x}), & \text { otherwise } \end{array} \right.. \tag {15}
$$

Only the logits for new classes $( k \in Y _ { b } )$ are rectified after each incremental task. BiC separates an extra validation set from the exemplar set, i.e., $\mathcal { E } = \mathcal { E } _ { t r a i n } \cup \mathcal { E } _ { v a l }$ , and uses the validation =set to tune the rectification layer. On the other hand, IL2M [111] suggests re-scaling the outputs with historical statistics. Suppose an instance is predicted as a new class. In that case, the logits will be re-scaled to ensure the predictions of old and new classes follow the same distribution.

Lastly, since the embedding module is sequentially updated in CIL, some works try to rectify the biased representations of incremental models. For example, SDC [105] utilizes a nearest-mean-of-exemplars classifier, which calculates the class centers and assigns instances to the nearest class center. However, since the embedding is incrementally updated, the class centers calculated in the former stage may suffer a drift in the next stage, making the classification results unreliable. Since old class instances are not available in the current stage, SDC aims to calibrate the class centers of old classes with the drift of new classes. CwD [106] analyzes the differences of embeddings among the CIL model and the oracle model and finds that the embeddings of the oracle model scatter more uniformly. It aims to make the CIL model similar to the oracle by enforcing the eigenvalues to be close. ConFiT [107] relieves the feature drift from the middle layers. Other works address the model weight rectification. CCLL [108] aims to calibrate the activation maps of old models during incremental learning. RKR [109] proposes to rectify the convolutional weights of old models when learning new tasks. FACT [104] depicts a new training paradigm for CIL, namely forward compatible training. Since the embedding space is endlessly adjusted for new classes, FACT proposes to pre-assign the embedding space for new classes to relieve the burden of embedding tuning.

Discussions: Model rectify-based methods aim to reduce the inductive bias in the CIL model and align it to the oracle model. This line of work helps to understand the inherent factor of catastrophic forgetting. Apart from the rectification methods listed in this section, [113] addresses that the bias of the CIL model is from the imbalanced data stream. [180] finds that the embedding trained with contrastive loss suffers less forgetting than cross-entropy loss. [191] finds that the batch normalization layers [192] are biased in CIL and proposes re-normalizing the layer outputs. [193] finds that vision transformers gradually lose the locality information when incrementally updated and proposes to insert the prior information about locality into the self-attention process. These works often treat the oracle model as the example, and design training techniques to reflect oracle model’s characteristics.

Apart from mimicking the oracle model, there are also works addressing the forward compatibility [104] and flat loss landscape. Since the ultimate goal of CIL is to find a flat minimum in loss landscape among all tasks, several works aim to achieve this goal during the first stage [194], [195]. It is worth exploring other factors in catastrophic forgetting and corresponding solutions in the future. On the other hand, the oracle model is obtained via joint training of all data via supervised loss. Other task-agnostic features could also help build a holistic classifier, e.g., via contrastive learning [180], which the oracle model does not possess.

# G. Template-Based Classification

Lastly, we discuss template-based classification, which is widely adopted in CIL. If we can build a ‘template’ for each class, the classification can be done by matching the query instance to the most similar template. A popular approach is to utilize class prototypes [129] as the template, which is rooted in cognitive science [196]. The prototype in deep neural networks is often defined as the average vector in the embedding space. For example, we can utilize the current embedding function φ · to extract the prototype of the i-th class:

$$
\boldsymbol {p} _ {i} = \frac {1}{N} \sum_ {j = 1} ^ {| \mathcal {D} ^ {b} |} \mathbb {I} (y _ {j} = i) \phi (\mathbf {x} _ {j}), \tag {16}
$$

where N is the instance number of class i. In (16), the class prototype is calculated via the class center in the embedding space. Hence, we can make inference without relying on the fully connected layer by matching an instance to the nearest prototype:

$$
y ^ {*} = \underset {y = 1, \dots , \left| \mathcal {Y} _ {b} \right|} {\operatorname{argmin}} \left\| \phi (\mathbf {x}) - \boldsymbol {p} _ {y} \right\|. \tag {17}
$$

As discussed in model rectification-based methods, sequentially updating the incremental model will result in bias in the fully-connected layer. Correspondingly, iCaRL [82] suggests conducting inferences via (17), which is also known as nearestclass-mean classifier [186]. Since inference is conducted by instance-prototype matching in the same embedding space, the bias among different stages can be alleviated. However, utilizing prototype-based inference also faces another challenge, i.e., the embedding mismatch between stages. Since the embedding function keeps changing among different stages, the prototypes of former stages may be incompatible with the query embedding of later stages. This phenomenon is also known as the semantic drift [105]. To fill in this gap, a naive solution is to utilize the set of exemplars and re-calculate class prototypes after each stage [37], [82]. Since the exemplar set $\bar { \mathcal { E } }$ contains instances of former classes, re-calculating all class prototypes after each incremental stage ensures the compatibility between prototypes and the latest embedding function. However, when exemplars are unavailable, specific algorithms need to be designed to compensate for the semantic drift.

Prototype-based Inference without Exemplars: When exemplars are unavailable, there are two main solutions to maintain a prototype classifier. A naive solution is to freeze the embedding function after the first incremental stage [104], [116], which forces the prototypes of different stages to be compatible. Such a learning process assumes the embedding function trained with first-stage data is generalizable enough for future tasks, relying on the vast number of training instances in $\mathcal { D } ^ { 1 }$ . As a result, researchers tend to design suitable training techniques with $\mathcal { D } ^ { 1 }$ to obtain a generalizable feature space for future tasks. [49], [120] draw inspiration from contrastive learning and design pre-text tasks to enhance the representation learned by the first stage. CEC [115] meta-learns a graph model to adjust new class prototypes with known ones, which propagates context information between classifiers for adaptation. LIMIT [121] finds that using prototypes extracted by the first stage backbone tends to predict instances into classes of the first stage. It proposes calibrating the prediction logits by meta-learning a transformer block between old and new classes. TEEN [119] systematically analyzes the performance gap between old and new classes and finds that the prototypical network forces the model to predict new classes into the most similar old class. Hence, it suggests pushing the prototypes of new classes to old classes for a calibrated decision boundary. Moreover, [104] aims to enhance the model’s forward compatibility by reserving the embedding space for new classes so that new classes can be inserted into the embedding space without harming existing ones. It allocates ‘virtual prototypes’ for new classes and explicitly reserves the embedding space for them using a bimodal target label and manifold mixup [197] to generate new class instances.

When we have a pre-trained embedding function as initialization, ADAM [116] finds that using a prototype-based classifier easily beats the state-of-the-art prompt-based methods [69], [70]. However, although the pre-trained model possesses generalizable features, it still lacks task-specific information on incremental datasets. Hence, it designs the ‘adapt and merge’ protocol to unify the generalizability of the pre-trained model and the adaptivity of downstream tasks. With the pre-trained embedding, it finetunes it with the first stage dataset $\bar { \mathcal { D } } ^ { 1 }$ , and concatenates the pre-trained and adapted embeddings for prototype extraction. Based on ADAM, RanPAC [117] designs random projection to project the concatenated features into a high-dimensional space, within which classes are separated more clearly. It also incrementally updates a Mahalanobis distance-based classifier, which shows stronger performance than cosine distance between prototypes and query embedding. FeCAM [118] also finds the inadequacy of the cosine classifier and proposes using a Bayesian classifier instead.

Apart from freezing the embedding, some works try to compensate for the semantic drift among different stages. Since prototypes will drift with the ever-changing embedding functions, they aim to estimate such drift to estimate the prototypes in the latest embedding space. SDC [105] aims to measure the prototype drift between different stages and utilizes the weighted combination of current stage data for reference. ZSTCI [122] achieves this goal by mapping prototypes of different stages into the same embedding space and designing a prototype alignment loss across stages.

Finally, another line of work considers generative classification [123]. Different from estimating the class prototype as a template, the template for each class is a generative model. Hence, the inference process can be measured via the likelihood of the query instance under such a generative model. However, it requires more calculation budgets for generative templates during inference than prototype-based methods.

Discussions: There are two advantages of template-based classification. First, when the exemplar set is available, using template-based classification enables query-prototype matching in the embedding space. Since the classifier will be biased after incremental learning, utilizing such a matching target alleviates the inductive bias during inference [82]. Second, when the pre-trained model (or model trained with large base classes) is available as initialization, the feature representations are generalizable and can be transferred to downstream tasks. Hence, freezing the embedding and using template-based classification can take full use of the generalizable features, and such a non-incremental learner will not suffer forgetting due to the frozen backbone [116], [117], [118].

However, there are also some drawbacks. When the exemplar set is not available, re-calculating the prototypes is impossible, and it requires a complex adjusting process to overcome the semantic drift [105], [122]. Second, when freezing the backbone and taking a template-based classifier, the model sacrifices its adaptivity for downstream tasks. When there are significant domain gaps between the pre-trained model and the downstream data [198], [199], template-based classification shall fail due to the incapability to extract generalizable features. In that case, continually adjusting the backbone could be more suitable to extract task-specific features. Finally, we can use an energybased model [200] to compute for each class an energy value rather than a likelihood [201] to alleviate the high cost of the generative model.

# IV. EXPERIMENTAL EVALUATION

In this section, we conduct comprehensive experiments to evaluate the performance of different kinds of CIL methods with benchmark datasets. We first introduce the benchmark experimental setting, dataset split, evaluation protocol, and implementation details. Afterward, we aim to compare these methods from three aspects:

- How do these methods perform on benchmark datasets?   
- Are they fairly compared? How to fairly compare them?   
- How to evaluate them with memory-agnostic measure?

Specifically, Section IV-B and IV-C answer the first question, and Section IV-D answers the second question. We provide the holistic performance measures in Section IV-E to answer the third question and summarize the results in Section IV-F. We report more results, measures, and visualizations in the supplementary material.

# A. Experimental Settings

1) Benchmark Datasets: iCaRL [82] first formulates the comparison protocol of class-incremental learning, which was widely followed and compared in other works. It suggests using CIFAR100 [24] and ImageNet100/1000 [202] for evaluation. CIFAR100 contains 100 classes with 60,000 images, in which 50,000 are training instances, and 10,000 are testing ones, with 100 images per class. Each image is represented by $3 2 \times 3 2$ pixels. ImageNet1000 is a large-scale dataset with 1,000 classes, with about 1.28 million images for training and 50,000 for validation. ImageNet100 is the subset of ImageNet1000 containing 100 classes [82]. These classes are selected from the first 100 classes after a random shuffle. Some works [27], [72], [73], [113] utilize MNIST [203], CUB200 [204] and miniImageNet [202] for evaluation, while the aforementioned datasets are the most widely adopted in the current CIL community, and we choose them for model evaluation.

Dataset Split: Following the protocol defined in [82], all classes are first shuffled by Numpy random seed 1993. Afterward, there are two different ways to split the classes into incremental stages:

Train from scratch (TFS): splits all classes equally into each incremental stage. For example, if there are B stages and C classes in total, each incremental task contains $C / B$ classes for training.   
Train from half (TFH): splits half of the total classes as the first incremental task and equally assigns the rest classes into the following stages. Specifically, it assigns $C / 2$ classes to the first task and $C / 2 ( B - 1 )$ classes to the 2rest tasks.

Both of these settings are widely adopted in the current CIL community [83], [89]. Hence, we unify these settings as $\mathbf { \cdot B a s e - } m , \mathbf { I n c - } n ^ { \circ }$ , where m stands for the number of classes in the first stage, and n stands for the number of classes in each incremental task. $m = 0$ stands for the TFS protocol. We use = 0the same training splits for every compared method for a fair comparison. The testing set is the same as the original one for holistic evaluation.

2) Evaluation Metrics: There are several metrics to evaluate the CIL model. We denote the Top-1 accuracy after the b-th task as ${ \mathcal { A } } _ { b } ,$ and higher $\mathcal { A } _ { b }$ indicates a better prediction accuracy. Since the CIL model is continually updated, the accuracy often decays with more tasks incorporated. Hence, the accuracy after the last stage $( \mathcal { A } _ { B } )$ is a proper metric for measuring the overall accuracy among all classes.

However, only comparing the final accuracy ignores the performance evolution along the learning trajectory. Hence, another metric denoted as ‘average accuracy’ considers the performance of all incremental stages: $\begin{array} { r } { \bar { \mathcal { A } } = \frac { 1 } { B } \dot { \sum } _ { b = 1 } ^ { B } \mathcal { A } _ { b } } \end{array}$ . A higher average =accuracy denotes a stronger performance along the incremental stages. Apart from these measures, we also consider forgetting and intransigence [33] measures in the supplementary.

3) Implementation Details: Selected methods: In the comparison, we aim to contain all kinds of methods in Table I. We systematically choose 17 methods, including: Replay [128], RMM [144] (data replay), GEM [53] (data regularization), EWC [73] (parameter regularization), AANets [66], FOS-TER [16], MEMO [17], DER [15], DyTox [18], L2P [69] (dynamic networks), LwF [81], iCaRL [82], PODNET [94],

Coil [88] (knowledge distillation), WA [112], BiC [83] (model rectify). We also report the baseline method ‘Finetune,’ which updates the model with (3).

The choice of these methods follows the development timeline of CIL, which is also the way we introduce these works. Additionally, it not only contains all seven aspects of CIL algorithms in our taxonomy but also includes early works (e.g., EWC, LwF, iCaRL) and recent state-of-the-art (DER, MEMO, L2P). The choice also gives consideration to CNN-based methods, ViT-based methods (DyTox), and even pre-trained ViT-based methods (L2P). Specifically, RMM is a specific technique to efficiently organize the memory budget, which can be orthogonally combined with other methods, and we combine it with FOSTER, denoted as FOSTER+RMM. Similarly, we combine AANets with LUCIR [89], denoted as LUCIR+AANets. L2P requires pre-trained ViT as the backbone model, while others are trained from scratch.

Training details: We implement the above methods with PyTorch [205] and PyCIL [206]. Specifically, we use the same network backbone for all CNN-based compared methods, i.e., ResNet32 [207] for CIFAR100 and ResNet18 for ImageNet. We use SGD with an initial learning rate of 0.1 and momentum of 0.9. The training epoch is set to 170 for all datasets with a batch size of 128. The learning rate suffers a decay of 0.1 at 80 and 120 epochs. For ViT-based methods like DyTox and L2P, we follow the original implementation and use ConViT [208] for DyTox and pre-trained ViT-B/16 [19] for L2P. The optimization parameters of them are set according to the original paper since ViT has a different optimization preference to CNN. We follow the original paper to set the algorithm-specific parameters, e.g., splitting 10% exemplars from the exemplar set as validation for BiC, setting the temperature τ to 5 and using a 10 epochs warmup for DER, using $\ell _ { 2 }$ norm to normalize the fully-connected layers in WA. For EWC, the λ parameter is done via a grid search among { , 1, 2, 3, 4}, and we find $1 0 ^ { 3 }$ leads to 1 10its best performance.

Research has shown that a good starting point (i.e., the first stage accuracy $\boldsymbol { \mathcal { A } } _ { 1 } )$ implies better transferability and will suffer less forgetting [110]. It must be noted that the performance gap of the first stage should be eliminated so as not to affect the forgetting evaluation. Hence, to make the methods share the same starting point, we utilize the same training strategy, data augmentation, and hyperparameters. We use basic data augmentation, e.g., random crop, horizontal flip, and color jitter for CIFAR100 and ImageNet.

It must be noted that Finetune, EWC, LwF, and L2P are exemplar-free methods, and we do not use any exemplar set for them. For other methods, we follow the benchmark setting to set the number of exemplars to 2,000 for CIFAR100 and ImageNet100 and 20,000 for ImageNet1000. These exemplars are equally sampled from each seen class via the herding [124] algorithm in Definition 2.

# B. Comparison on Small-Scale Datasets

This section includes the comparison on the small-scale dataset, i.e., CIFAR100. We compare these methods under TFS and TFH settings with four data splits and report the incremental performance in Fig. 6 (top). We summarize the average and final performance, the number of parameters in Table II.

TABLE II AVERAGE AND LAST ACCURACY PERFORMANCE COMPARISON ON CIFAR100 

<table><tr><td rowspan="2">Method</td><td colspan="3">Base0 Inc5</td><td colspan="3">Base0 Inc10</td></tr><tr><td>#P</td><td> $\bar{A}$ </td><td> $A_B$ </td><td>#P</td><td> $\bar{A}$ </td><td> $A_B$ </td></tr><tr><td>Finetune</td><td>0.46</td><td>17.59</td><td>4.83</td><td>0.46</td><td>26.25</td><td>9.09</td></tr><tr><td>EWC</td><td>0.46</td><td>18.42</td><td>5.58</td><td>0.46</td><td>29.73</td><td>12.44</td></tr><tr><td>LwF</td><td>0.46</td><td>30.93</td><td>12.60</td><td>0.46</td><td>43.56</td><td>23.25</td></tr><tr><td>GEM</td><td>0.46</td><td>31.73</td><td>19.48</td><td>0.46</td><td>40.18</td><td>23.03</td></tr><tr><td>Replay</td><td>0.46</td><td>58.20</td><td>38.69</td><td>0.46</td><td>59.31</td><td>41.01</td></tr><tr><td>RMM</td><td>0.46</td><td>65.72</td><td>51.10</td><td>0.46</td><td>68.54</td><td>56.64</td></tr><tr><td>iCaRL</td><td>0.46</td><td>63.51</td><td>45.12</td><td>0.46</td><td>64.42</td><td>49.52</td></tr><tr><td>PODNet</td><td>0.46</td><td>47.88</td><td>27.99</td><td>0.46</td><td>55.22</td><td>36.78</td></tr><tr><td>Coil</td><td>0.46</td><td>57.68</td><td>34.33</td><td>0.46</td><td>60.27</td><td>39.85</td></tr><tr><td>WA</td><td>0.46</td><td>64.65</td><td>48.46</td><td>0.46</td><td>67.09</td><td>52.30</td></tr><tr><td>BiC</td><td>0.46</td><td>62.38</td><td>43.08</td><td>0.46</td><td>65.08</td><td>50.79</td></tr><tr><td>FOSTER</td><td>0.46</td><td>63.38</td><td>49.42</td><td>0.46</td><td>66.49</td><td>53.21</td></tr><tr><td>AANets</td><td>0.99</td><td>59.34</td><td>42.42</td><td>0.99</td><td>61.73</td><td>45.53</td></tr><tr><td>DER</td><td>9.27</td><td>67.99</td><td>53.95</td><td>4.60</td><td>69.74</td><td>58.59</td></tr><tr><td>MEMO</td><td>7.14</td><td>68.10</td><td>54.23</td><td>3.62</td><td>70.20</td><td>58.49</td></tr><tr><td>DyTox</td><td>10.7</td><td>68.06</td><td>52.23</td><td>10.7</td><td>71.07</td><td>58.72</td></tr><tr><td>L2P</td><td>85.7</td><td>84.00</td><td>78.96</td><td>85.7</td><td>89.35</td><td>83.39</td></tr></table>

“#P' represents the number of parameters (million).

As we can infer from these figures, finetune shows the worst performance among all settings, verifying the fact that the model will suffer forgetting when sequentially learning new concepts. Regularizing the parameters, i.e., EWC, shows a negligible improvement over finetune. By contrast, LwF adds knowledge distillation loss to resist forgetting, which substantially improves the performance. When the exemplar set is available, directly replaying them during model updating can further enhance the performance by a substantial margin. Coil and iCaRL combine the exemplar replay and knowledge distillation and further obtain a performance boost than the vanilla replay. Model rectify methods, i.e., BiC and WA, rectify the bias in iCaRL and further improve the accuracy. However, recent methods based on dynamic networks (i.e., DER, FOSTER, MEMO, and DyTox) show competitive results with the help of multiple backbones. It indicates that saving more backbones can substantially help the model overcome forgetting. When it comes to pre-trained models, L2P obtains the best performance among all methods. However, other methods are trained from scratch, while L2P relies on the ViT pre-trained on ImageNet-21 K, making it unfair to directly compare these two lines of methods.

On the other hand, we can infer from different settings that TFH requires more stability than TFS. Since the evaluation is based on the accuracy among all classes, remembering old classes becomes more critical when there is a large group of base classes.

# C. Comparison on Large-Scale Datasets

In this section, we evaluate different methods on the largescale dataset, i.e., ImageNet100 and ImageNet1000. We compare these methods under the TFS and TFH settings and report the incremental performance (top-1 accuracy) of different methods in Fig. 6 (bottom). We report the top-5 accuracy and summarize the average and final performance, the number of parameters in the supplementary. Since GEM requires saving a large-scale matrix for solving the QP problem, it cannot be conducted with the ImageNet dataset. On the other hand, since the ViT in L2P is pre-trained on ImageNet-21 K, incrementally training it on ImageNet is meaningless, and we do not report its results.

![](images/4d45c8f3465f7bff42df155c7b6aff2b10f01b3eeeb85fc2ca037c2ff3e1b20c.jpg)

<details>
<summary>text_image</summary>

Finetune
RMM
EWC
DER
L2P
MEMO
iCaRL
Coil
WA
Replay
GEM
AANets
DyTox
FOSTER
LwF
PODNet
BiC
CNN-Oracle
</details>

![](images/f4b1bc9d85267ee0bc27a39f7d91a63a496b209ea850cf38c5e79019d59343af.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 5                 | 100          |
| 20                | 80           |
| 40                | 60           |
| 60                | 40           |
| 80                | 30           |
| 100               | 20           |
</details>

![](images/19e02302aae1b04b813d3a275569332c3514b3e9bab443011dbaa3565bd2d62e.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) - Series 1 | Accuracy (%) - Series 2 | Accuracy (%) - Series 3 | Accuracy (%) - Series 4 | Accuracy (%) - Series 5 | Accuracy (%) - Series 6 | Accuracy (%) - Series 7 | Accuracy (%) - Series 8 | Accuracy (%) - Series 9 | Accuracy (%) - Series 10 |
| ----------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------ | ------------------------- |
| 10                | 100                      | 100                      | 100                      | 100                      | 100                      | 100                      | 100                      | 100                      | 100                      | 100                       |
| 20                | 95                       | 92                       | 90                       | 88                       | 85                       | 82                       | 80                       | 78                       | 75                       | 72                        |
| 40                | 90                       | 85                       | 82                       | 78                       | 75                       | 72                       | 70                       | 68                       | 65                       | 62                        |
| 60                | 85                       | 80                       | 75                       | 72                       | 70                       | 68                       | 65                       | 62                       | 60                       | 58                        |
| 80                | 80                       | 75                       | 70                       | 68                       | 65                       | 62                       | 60                       | 58                       | 55                       | 52                        |
| 100               | 75                       | 70                       | 65                       | 62                       | 60                       | 58                       | 55                       | 52                       | 50                       | 48                        |
</details>

![](images/6d694abb179909d918700ce61f5e7296ddd0baa9ce4265eef55cbc9cbd76647b.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 50                | 80           |
| 60                | 70           |
| 70                | 65           |
| 80                | 60           |
| 90                | 55           |
| 100               | 50           |
</details>

![](images/b64377551d20efdd227a7e12603f543a11001394586f13e476379e208196bbf4.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 50                | 80           |
| 75                | 60           |
| 100               | 40           |
</details>

(a) CIFAR100 Base0 Inc5   
![](images/c69259ae74926a793f3cf18effcb4fde78649a915fb2706d637171fb92d57cb9.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 5                 | 100          |
| 20                | 80           |
| 60                | 60           |
| 80                | 50           |
| 100               | 40           |
</details>

(e)ImageNet100 BaseO Inc5

(b) CIFAR100 Base0 Inc10   
![](images/59acebd6dff4fe3b1ecc7f3a4969ce0daafb48eb4f47e187d8ea70381d1fc446.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 50                | 85           |
| 60                | 15           |
| 70                | 45           |
| 80                | 35           |
| 90                | 30           |
| 100               | 25           |
</details>

(f) ImageNet100 Base50 Inc10

(c) CIFAR100 Base50 Inc10   
![](images/ee5b1bd1e30840582700e485472091c4470f7d2593165472173567bd484a1def.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 100               | 85           |
| 200               | 75           |
| 400               | 65           |
| 600               | 55           |
| 800               | 45           |
| 1000              | 35           |
</details>

(g) ImageNet1000 Base0 Inc100

(d) CIFAR100 Base50 Inc25   
![](images/543775f658a32e2ad1e87679ad21002d283284488c26117125ff6417164c08ee.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 500               | 80           |
| 600               | 70           |
| 700               | 60           |
| 800               | 50           |
| 900               | 40           |
| 1000              | 30           |
</details>

(h） ImageNet1000 Base500 Inc100

Fig. 6. Incremental performance of different methods on CIFAR100 (a)–(d), ImageNet100 (e)–(f), and ImageNet1000 (g)–(h). Legends are shown at the top of this figure, and we report the results of more settings in the supplementary.   
![](images/2610e02cf0729c55a2cb03ec5fd6dfce9191a3b79160c565648067fc1159164c.jpg)  
(a) CIFAR100 Base0 Inc5

![](images/259d8beae5213df81d3cb6e9db1fe13df8f1f8267e6c343968bf822328b09ca6.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) - Line 1 | Accuracy (%) - Line 2 | Accuracy (%) - Line 3 |
| ----------------- | --------------------- | --------------------- | --------------------- |
| 10                | 90                    | 90                    | 90                    |
| 20                | 75                    | 80                    | 40                    |
| 40                | 65                    | 75                    | 25                    |
| 60                | 60                    | 70                    | 20                    |
| 80                | 55                    | 65                    | 15                    |
| 100               | 50                    | 60                    | 10                    |
</details>

(b) CIFAR100 Base0 Inc10

![](images/61d4dcf126784f34b4db294cf49de82466fef51bbd1151fd317f5bd2386bb279.jpg)

<details>
<summary>line</summary>

| Number of Classes | Accuracy (%) |
| ----------------- | ------------ |
| 50                | 85           |
| 60                | 75           |
| 70                | 65           |
| 80                | 60           |
| 90                | 55           |
| 100               | 50           |
</details>

(c) ImageNet100 Base50 Inc5

![](images/d216bbb53a6caea64b4fc3fda3b55a58df18b918c1fe9a8af88fb2ae76a862d7.jpg)  
(d) ImageNet100 BaseO Inc10   
Fig. 7. Incremental accuracy of different methods with aligned memory cost. Legends are shown in (a) and (d).

As we can infer from these figures, most methods share the same trend as CIFAR100. Replay acts as a strong baseline in both small-scale and large-scale inputs, verifying the effectiveness of exemplars in incremental learning. Dynamic networks consistently show the best performance among all settings, outperforming other methods by a substantial margin in the benchmark comparison.

# D. Memory-Aligned Comparison

Former experimental evaluations indicate dynamic networks show the best performance among all methods. However, are these methods fairly compared? We argue that these dynamic networks implicitly introduce an extra memory budget, namely model buffer for keeping old models. The additional buffer results in an unfair comparison to those methods without storing models. Taking Table II for an example, the number of parameters in DER is ten times that of iCaRL in the CIFAR100 Base0 Inc10 setting. We visualize the memory cost of different methods in Fig. 8(a) and find that dynamic networks obtain better performance at the expense of more memory budgets. It makes directly comparing different kinds of methods unfair.

![](images/0ad8764bfc4aa38cddedf4f36dfa7d846a20674557ae26eb9a2986769076f0f2.jpg)

<details>
<summary>bar_stacked</summary>

| Component | Exemplar Size (MB) | Model Size (MB) |
| :--- | :--- | :--- |
| Finance | 1 | 2 |
| LWC | 1 | 2 |
| LwF | 1 | 2 |
| GCD | 3 | 2 |
| Repdy | 4 | 2 |
| RBM | 4 | 2 |
| LcRL | 4 | 2 |
| PDDNet | 4 | 2 |
| Coil | 4 | 2 |
| FDSTER | 4 | 2 |
| BIC | 4 | 2 |
| WA | 5 | 6 |
| DPAR | 6 | 18 |
| MEMO | 7 | 30 |
| DYto | 8 | 48 |
</details>

(a) Memory size of Figure 6(b)

![](images/ecd539f1ad6a0f9fad395a463b25303ee333a3ca874094d835a6132344172a7f.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Exemplar Size (MB) | Model Size (MB) |
| :--- | :--- | :--- |
| GEM | 20 | 1 |
| BapNet | 20 | 1 |
| C-CRL | 20 | 1 |
| PODNet | 20 | 1 |
| Coil | 20 | 1 |
| WA | 20 | 1 |
| B/C | 20 | 1 |
| FOSTDR | 20 | 1 |
| DFR | 8 | 35 |
| MSMD | 14 | 16 |
</details>

(b) Memory size of Figure 7(b)   
Fig. 8. Memory size of different comparison protocols. Dark bars denote the budget for exemplars, and red bars represent the budget for keeping the model. Different methods should be aligned to the same budget for a fair comparison, as shown in (b).

TABLE III PERFORMANCE COMPARISON ON CIFAR100 WITH ALIGNED MEMORY COST 

<table><tr><td rowspan="2">Method</td><td colspan="5">CIFAR100 Base0 Inc10</td></tr><tr><td>#P</td><td>#E</td><td>MS</td><td> $\bar{A}$ </td><td> $A_B$ </td></tr><tr><td>GEM</td><td>0.46</td><td>7431</td><td>23.5</td><td>27.03</td><td>10.72</td></tr><tr><td>Replay</td><td>0.46</td><td>7431</td><td>23.5</td><td>69.97</td><td>55.61</td></tr><tr><td>iCaRL</td><td>0.46</td><td>7431</td><td>23.5</td><td>70.94</td><td>58.52</td></tr><tr><td>PODNet</td><td>0.46</td><td>7431</td><td>23.5</td><td>60.80</td><td>45.38</td></tr><tr><td>Coil</td><td>0.46</td><td>7431</td><td>23.5</td><td>70.69</td><td>54.40</td></tr><tr><td>WA</td><td>0.46</td><td>7431</td><td>23.5</td><td>69.55</td><td>59.26</td></tr><tr><td>BiC</td><td>0.46</td><td>7431</td><td>23.5</td><td>70.69</td><td>59.60</td></tr><tr><td>FOSTER</td><td>0.46</td><td>7431</td><td>23.5</td><td>72.28</td><td>59.39</td></tr><tr><td>DER</td><td>4.60</td><td>2000</td><td>23.5</td><td>71.47</td><td>60.26</td></tr><tr><td>MEMO</td><td>3.62</td><td>3312</td><td>23.5</td><td>72.37</td><td>61.98</td></tr></table>

‘#P'represents the number of parameters (million).“#E' denotes the number of exemplars,and ‘MS'denotes the memory size (MB).

In this section, we follow [17] for a fair comparison among different methods with different memory budgets. For those methods with different memory costs, we need to align the performance measure at the same memory scale for a fair comparison. For example, a ResNet32 model costs 463,504 parameters (float), while a CIFAR image requires $3 \times 3 2 \times 3 2$ 3 32 32integer numbers (int). Hence, the budget for saving a backbone is equal to saving 463,504 floats × bytes/float ÷ ×  × 4 (3 32 32)bytes/image ≈  instances for CIFAR. A fair comparison 603between different methods can be made by equipping the other methods with more exemplars. Since DER requires saving all the backbones from history, we align the memory cost of other methods to DER with extra exemplars. We visualize the memory budget of different methods under the current protocol in Fig. 8(b), where the total budgets of different methods are aligned to the same scale.

We report the results under the current protocol in Fig. 7 and the detailed performance in Table III. Since Finetune, LwF, and EWC cannot be combined with the exemplar set, we do not report the results of these methods. Similarly, DyTox and L2P utilize different kinds of backbones, and we do not report their results. As we can infer from these results, the gap between dynamic networks and other methods is no longer large under fair comparison. For example, DER outperforms iCaRL by 9.07% in terms of the final accuracy in the benchmark comparison of CIFAR100 Base0 Inc10, while the gap decreases to 1.74% under the current protocol.

# E. Memory-Agnostic Measure

Section IV-D enables fair comparison among different methods by aligning the memory cost. However, the comparison is made by aligning the memory cost of other methods to DER. In contrast, open-world applications require conducting classincremental learning in various scenarios, i.e., high-performance computers and edge devices are both essential. Hence, it requires a memory-agnostic measure for CIL to measure the model’s extendability given any memory budget.

To this end, we can set several ‘comparison budgets’ and align the memory cost of different methods to them. The budget list starts from a small value and incrementally enlarges, containing the requirement of different scale models. In this setting, we set the budget of the start point to a single backbone and gradually increase it to the budget cost of DER. For those algorithms without extra model storage, we equip more exemplars to meet the selected budget. However, when the selected budget is smaller than the required size, dynamic networks (i.e., DER and MEMO) cannot be deployed with the benchmark backbone. Therefore, we choose smaller backbones with fewer parameters for alignment. Hence, we can measure the performance of different methods at different memory scales and draw the performance-memory curve, as shown in Fig. 9. The X-coordinate corresponds to the memory cost, and the Y-coordinate indicates the average performance A (or last performance AB). We suggest the area under the performance-memory curve (AUC) since the curve of each method indicates the dynamic ability with the change of model size. We calculate AUC-A and AUC-L, standing for the AUC under the average performance-memory and last performance-memory curves.

![](images/b932c9e8cca3369274575f9789aca81b798303905f348d148fa6da76910b277c.jpg)

<details>
<summary>line</summary>

| Memory Size (MB) | Coil  | BiC   | Replay | FOSTER | DER   | iCaRL | GEM   | WA    | MEMO  | PODNet |
| ---------------- | ----- | ----- | ------ | ------ | ----- | ----- | ----- | ----- | ----- | ------ |
| 7.6              | 60.0  | 60.0  | 55.0   | 65.0   | 65.0  | 65.0  | 28.0  | 65.0  | 55.0  | 60.0   |
| 12.4             | 65.0  | 65.0  | 68.0   | 68.0   | 68.0  | 68.0  | 28.0  | 68.0  | 70.0  | 65.0   |
| 16.0             | 68.0  | 68.0  | 70.0   | 70.0   | 70.0  | 70.0  | 28.0  | 70.0  | 72.0  | 68.0   |
| 19.8             | 70.0  | 70.0  | 72.0   | 72.0   | 72.0  | 72.0  | 28.0  | 72.0  | 73.0  | 70.0   |
| 23.5             | 72.0  | 72.0  | 73.0   | 73.0   | 73.0  | 73.0  | 28.0  | 73.0  | 74.0  | 72.0   |
</details>

(a) CIFAR100 Base0 Inc10

![](images/f69aff81f5880ca1ad81e5f0d5f6f1c48813a5c85d84023dc1c9438e711f91e1.jpg)

<details>
<summary>line</summary>

| Memory Size (MB) | Coil  | BiC   | Replay | FOSTER | DER   | iCaRL | PODNet | WA    | MEMO  |
| ---------------- | ----- | ----- | ------ | ------ | ----- | ----- | ------ | ----- | ----- |
| 329              | 65.0  | 45.0  | 45.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 65.0  |
| 493              | 65.0  | 55.0  | 55.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 70.0  |
| 755              | 65.0  | 65.0  | 65.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 75.0  |
| 872              | 65.0  | 70.0  | 70.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 75.0  |
| 1180             | 65.0  | 75.0  | 75.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 78.0  |
| 1273             | 65.0  | 75.0  | 75.0   | 65.0   | 65.0  | 65.0  | 65.0   | 65.0  | 78.0  |
</details>

(b) ImageNet100 Base50 Inc5   
Fig. 9. Average performance-memory curve of different methods with different datasets. Dynamic networks perform better with large budgets, while other methods dominate small ones.

TABLE IV MEMORY-AGNOSTIC PERFORMANCE MEASURES FOR CIL 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR100 Base0 Inc10</td><td colspan="2">ImageNet100 Base50 Inc5</td></tr><tr><td>AUC-A</td><td>AUC-L</td><td>AUC-A</td><td>AUC-L</td></tr><tr><td>GEM</td><td>4.31</td><td>1.70</td><td>-</td><td>-</td></tr><tr><td>Replay</td><td>10.49</td><td>8.02</td><td>553.6</td><td>470.1</td></tr><tr><td>iCaRL</td><td>10.81</td><td>8.64</td><td>607.1</td><td>527.5</td></tr><tr><td>PODNet</td><td>9.42</td><td>6.80</td><td>701.8</td><td>624.9</td></tr><tr><td>Coil</td><td>10.60</td><td>7.82</td><td>601.9</td><td>486.5</td></tr><tr><td>WA</td><td>10.80</td><td>8.92</td><td>666.0</td><td>581.7</td></tr><tr><td>BiC</td><td>10.73</td><td>8.30</td><td>592.7</td><td>474.2</td></tr><tr><td>FOSTER</td><td>11.12</td><td>9.03</td><td>638.7</td><td>566.3</td></tr><tr><td>DER</td><td>10.74</td><td>8.95</td><td>699.0</td><td>639.1</td></tr><tr><td>MEMO</td><td>10.85</td><td>9.03</td><td>713.0</td><td>654.6</td></tr></table>

AUC depicts the dynamic ability with the change of memory size.

As we can infer from Table IV, there are two main conclusions. First, there exists an intersection between dynamic networks and other methods, where other methods perform better given a small memory size. In contrast, dynamic networks perform better given a large memory size. In other words, there is no free lunch in CIL, and different kinds of methods have their dominant domains. Second, the AUC measure provides a holistic way to measure the extendability of different methods given various memory budgets. FOSTER and MEMO show competitive results regarding the AUC-A/L measure, implying their stronger extendability. It would be interesting to introduce it as the performance measure and design CIL methods for real-world applications. We report implementation details and the performance of each budget point in the supplementary.

# F. Discussions About the Comparison

With the above experimental evaluations from three aspects, we have the following conclusions: 1) Equipping the model with exemplars is a simple and effective way to resist forgetting in CIL models. 2) Knowledge distillation performs better than parameter regularization in resisting forgetting with the same cost. 3) Model rectification can further boost the performance of other CIL models in a plug-and-play manner. 4) Pre-trained models can ease the burden of incremental learning and show very strong performance. However, since the features of pre-trained models are already available and do not need to be incrementally learned, comparing pre-trained models to other methods may be unfair. 5) Dynamic networks show the best performance in the evaluation at the cost of extra memory budgets. However, when changing the extra budgets into equal size of exemplars, the performance gap becomes smaller. 6) AUC-A and AUC-L provide a way to evaluate different CIL methods in a memory-agnostic manner, which can help select the method with extendability given any budget.

# V. FUTURE DIRECTIONS

In this section, we discuss the possible future directions of the class-incremental learning field.

CIL With Complex Inputs: In the real world, data is often with complex format, e.g., few-shot [27], imbalanced [28], weak supervised [180], [209], multi modal [179], concept drift [10], novel classes [210] etc. CIL methods should be able to handle these real-world scenarios for better generalizability. Specifically, [27], [104], [115], [121] address the few-shot CIL problem, where the model is required to be adapted to incoming few-shot classes. [28], [141] propose to tackle the long-tailed CIL problem, where the head classes are easy to collect with adequate instances while tail classes are scarce. Recently, with the prosperity of CLIP [179], incrementally training the language vision models to handle multi-modal data streams [211] is becoming popular. Since the labeling cost is always expensive in the real world, there are some works addressing training CIL models in a semi-supervised or unsupervised manner [180]. [148] proposes a unified framework to handle CIL with concept drift. If the test dataset contains instances from unknown classes, open-set recognition [210] and novel class discovery [212] can equip the model with the detection ability, which refers to the open-world recognition problem [213]. Lastly, real-world data may emerge hierarchically, and the labels could evolve from coarse-grained to fine-grained. CIL with refined concepts [214] is also vital to building real-world learning systems.

CIL With General Data Stream: Current CIL methods have a set of restrictions on the data stream, e.g., saving exemplars for rehearsal, undergoing multi-epoch offline training within an incremental task, etc. Future CIL algorithms should be able to conduct fully online training [215] without requiring the task boundaries [216]. On the other hand, saving exemplars from the history may violate user privacy in some cases. To this end, exemplar-free CIL [50], [150], [217] should be conducted to enable the model to be adapted without the help of exemplars. Lastly, most CIL methods rely on the number of base classes to define hyper-parameters in model optimization, where more base classes require larger stability and fewer base classes require larger plasticity. Therefore, designing the algorithm to handle CIL problems given any base classes [218] is also essential to the real world.

CIL With Any Memory/Computational Budgets: Dynamic networks have obtained impressive performance in recent years [16]. However, most of them require an extra memory budget to save the external backbones. In the real world, a good CIL algorithm should handle different budget restrictions. For example, an algorithm should be able to learn with high-performance computers or with edge devices (e.g., smartphones), and both scenarios are essential. Hence, deploying and comparing CIL models in the real world should take the memory budgets into consideration. The performance measures of AUC-A/L [17] are proper solutions to holistically compare different methods given any memory budgets. On the other hand, future CIL methods are also encouraged to handle specific learning scenarios, e.g., [219] addresses training CIL systems under resource-limited scenarios. Another important characteristic is the computational budget [220]. For realistic scenarios with high-throughput streams, computational bottlenecks impose implicit constraints on learning from past samples that can be too many to be revisited during training. In that case, we can only update the model with limited iterations and cannot tackle all the data. Hence, designing computational-efficient algorithms in real-world applications remains a promising direction for relating CIL to realistic scenarios.

CIL With Pre-Trained Models: Recently, pre-trained models have shown to work competitively with their strong transferability, especially for ViT-based methods [69], [70], [71], [72]. The pre-trained models provide generalizable features for the downstream tasks, enabling the incremental model to adjust with minimal cost [178]. CIL with pre-trained models is a proper way to handle real-world incremental applications from an excellent starting point. However, since the final target for incremental learning is to build a generalizable feature continually, some will argue that pre-trained models weaken the difficulty of incremental learning. From this perspective, designing proper algorithms to train the CIL model from scratch is more challenging. On the other hand, pre-trained language-vision models [179] have shown powerful generalizability in recent years, and exploring the ensemble of various pre-trained models is also an exciting topic.

Analyzing the Reason Behind Catastrophic Forgetting: Model rectification-based methods aim to reduce the inductive bias in the CIL model. Recently, more works have tried to analyze the reason for forgetting in CIL. [191] suggests that batch normalization layers are biased when sequentially trained, which triggers the different activation of old and new classes. [217] finds that representations with large eigenvalues transfer better and suffer less forgetting, and [215] shows that eigenvalues can be enlarged by maximizing the mutual information between old and new features. [221] theoretically decomposes the CIL problem into within-task and task-id predictions. It further proves that good within-task and task-id predictions are necessary and sufficient for good CIL performances. [222] finds that applying data replay causes the newly added classes’ representations to overlap significantly with the previous classes, leading to highly disruptive parameter updates. [195] empirically shows that the amount of forgetting correlates with the geometrical properties of the convergent points. It would be interesting to explore more reasons for catastrophic forgetting theoretically and empirically in the future.

# VI. CONCLUSION

Real-world applications often face streaming data, with which the model should be incrementally updated without catastrophic forgetting. In this paper, we provide a comprehensive survey about class-incremental learning by dividing them into seven categories taxonomically and chronologically. Additionally, we provide a holistic comparison among different methods on several publicly available datasets. With these results, we discuss the insights and summarize the common rules to inspire future research. Finally, we highlight an important factor in CIL comparison, namely memory budget, and advocate evaluating different methods holistically by emphasizing the effect of memory budgets. We provide a comprehensive evaluation of different methods given specific budgets as well as some new performance measures. We expect this survey to provide an effective way to understand current state-of-the-art and speed up the development of the CIL field.

# REFERENCES

[1] D. Silver et al., “Mastering the game of Go with deep neural networks and tree search,” Nature, vol. 529, no. 7587, pp. 484–489, 2016.   
[2] J. Jumper et al., “Highly accurate protein structure prediction with alphafold,” Nature, vol. 596, no. 7873, pp. 583–589, 2021.   
[3] Q. Feng and S. Chen, “Learning multi-tasks with inconsistent labels by using auxiliary big task,” Front. Comput. Sci., vol. 17, no. 5, 2023, Art. no. 175342.   
[4] H. M. Gomes, J. P. Barddal, F. Enembreck, and A. Bifet, “A survey on ensemble learning for data stream classification,” ACM Comput. Surv., vol. 50, no. 2, pp. 1–36, 2017.   
[5] G. Krempl et al., “Open challenges for data stream mining research,” in Proc. Int. Conf. Knowl. Discov. Data Mining, 2014, pp. 1–10.   
[6] M. De Lange et al., “A continual learning survey: Defying forgetting in classification tasks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 7, pp. 3366–3385, Jul. 2022.   
[7] S. T. Grossberg, Studies of Mind and Brain: Neural Principles of Learning, Perception, Development, Cognition, and Motor Control, vol. 70, Berlin, Germany: Springer Science & Business Media, 2012.   
[8] G. M. van de Ven, T. Tuytelaars, and A. S. Tolias, “Three types of incremental learning,” Nat. Mach. Intell., vol. 4, pp. 1185–1197, 2022.   
[9] Z. Chen and B. Liu, “Lifelong machine learning,” Synth. Lectures Artif. Intell. Mach. Learn., vol. 12, no. 3, pp. 1–207, 2018.   
[10] J. Lu, A. Liu, F. Dong, F. Gu, J. Gama, and G. Zhang, “Learning under concept drift: A review,” IEEE Trans. Knowl. Data Eng., vol. 31, no. 12, pp. 2346–2363, Dec. 2019.   
[11] Z.-H. Zhou, J. Wu, and W. Tang, “Ensembling neural networks: Many could be better than all,” Artif. Intell., vol. 137, no. 1-2, pp. 239–263, 2002.   
[12] I. Kuzborskij, F. Orabona, and B. Caputo, “From n to n+ 1: Multiclass transfer incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2013, pp. 3358–3365.   
[13] Q. Da, Y. Yu, and Z.-H. Zhou, “Learning with augmented class by exploiting unlabeled data,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2014, pp. 1760–1766.

[14] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. Van De Weijer, “Class-incremental learning: Survey and performance evaluation on image classification,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 5, pp. 5513–5533, May 2023.   
[15] S. Yan, J. Xie, and X. He, “DER: Dynamically expandable representation for class incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 3014–3023.   
[16] F.-Y. Wang, D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Foster: Feature boosting and compression for class-incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 398–414.   
[17] D.-W. Zhou, Q.-W. Wang, H.-J. Ye, and D.-C. Zhan, “A model or 603 exemplars: Towards memory-efficient class-incremental learning,” in Proc. Int. Conf. Learn. Representations, 2023.   
[18] A. Douillard, A. Ramé, G. Couairon, and M. Cord, “DyTox: Transformers for continual learning with dynamic token expansion,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 9285–9295.   
[19] A. Dosovitskiy et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” in Proc. Int. Conf. Learn. Representations, 2020.   
[20] Z. Mai, R. Li, J. Jeong, D. Quispe, H. Kim, and S. Sanner, “Online continual learning in image classification: An empirical survey,” Neurocomputing, vol. 469, pp. 28–51, 2022.   
[21] M. M. Biesialska, K. Biesialska, and M. R. Costa-Juss à, “Continual lifelong learning in natural language processing: A survey,” in Proc. Int. Conf. Comput. Linguistics, 2020, pp. 6523–6541.   
[22] G. I. Parisi, R. Kemker, J. L. Part, C. Kanan, and S. Wermter, “Continual lifelong learning with neural networks: A review,” Neural Netw., vol. 113, pp. 54–71, 2019.   
[23] E. Belouadah, A. Popescu, and I. Kanellos, “A comprehensive study of class incremental learning algorithms for visual tasks,” Neural Netw., vol. 135, pp. 38–54, 2021.   
[24] A. Krizhevsky and G. Hinton, “Learning multiple layers of features from tiny images,” Master’s thesis, Univ. Tront, 2009.   
[25] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2009, pp. 248–255.   
[26] J. Bang, H. Kim, Y. J. Yoo, J.-W. Ha, and J. Choi, “Rainbow memory: Continual learning with a memory of diverse samples,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 8218–8227.   
[27] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, and Y. Gong, “Few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 12183–12192.   
[28] X. Liu, Y.-S. Hu, X.-S. Cao, A. D. Bagdanov, K. Li, and M.-M. Cheng, “Long-tailed class incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 495–512.   
[29] Z. Mai, R. Li, H. Kim, and S. Sanner, “Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 3589–3599.   
[30] R. Kruse, S. Mostaghim, C. Borgelt, C. Braune, and M. Steinbrecher, “Multi-layer perceptrons,” in Computational Intelligence: A Methodological Introduction. Berlin, Germany: Springer, 2022, pp. 53–124.   
[31] Y. LeCun, K. Kavukcuoglu, and C. Farabet, “Convolutional networks and applications in vision,” in Proc. IEEE Int. Symp. Circuits Syst., 2010, pp. 253–256.   
[32] A. Vaswani et al., “Attention is all you need,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 5998–6008.   
[33] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. HS Torr, “Riemannian walk for incremental learning: Understanding forgetting and intransigence,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 532–547.   
[34] R. Aljundi, M. Lin, B. Goujaud, and Y. Bengio, “Gradient based sample selection for online continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2019, pp. 11816–11825.   
[35] D. Isele and A. Cosgun, “Selective experience replay for lifelong learning,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2018, pp. 3302–3309.   
[36] A. Chaudhry, A. Gordo, P. K. Dokania, P. Torr, and D. Lopez-Paz, “Using hindsight to anchor past knowledge in continual learning,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2020, pp. 6993–7001.   
[37] M. De Lange and T. Tuytelaars, “Continual prototype evolution: Learning online from non-stationary data streams,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 8250–8259.   
[38] A. Iscen, J. Zhang, S. Lazebnik, and C. Schmid, “Memory-efficient incremental learning through feature adaptation,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 699–715.

[39] H. Zhao, H. Wang, Y. Fu, F. Wu, and X. Li, “Memory-efficient classincremental learning for image classification,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 10, pp. 5966–5977, Oct. 2022.   
[40] Y. Liu, Y. Su, A.-A. Liu, B. Schiele, and Q. Sun, “Mnemonics training: Multi-class incremental learning without forgetting,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 12245–12254.   
[41] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning with deep generative replay,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 2990–2999.   
[42] C. He, R. Wang, S. Shan, and X. Chen, “Exemplar-supported generative reproduction for class incremental learning,” in Proc. Brit. Mach. Vis. Conf., 2018, Art. no. 98.   
[43] W. Hu et al., “Overcoming catastrophic forgetting for continual learning via model adaptation,” in Proc. Int. Conf. Learn. Representations, 2019.   
[44] R. Kemker and C. Kanan, “FearNet: Brain-inspired model for incremental learning,” in Proc. Int. Conf. Learn. Representations, 2018.   
[45] O. Ostapenko, M. Puscas, T. Klein, P. Jahnichen, and M. Nabi, “Learning to remember: A synaptic plasticity driven framework for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 11321–11329.   
[46] Y. Xiang, Y. Fu, P. Ji, and H. Huang, “Incremental learning using conditional adversarial networks,” in Proc. Int. Conf. Comput. Vis., 2019, pp. 6619–6628.   
[47] L. Wang, K. Yang, C. Li, L. Hong, Z. Li, and J. Zhu, “ORDisCo: Effective and efficient usage of incremental unlabeled data for semi-supervised continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 5383–5392.   
[48] J. Jiang, E. Cetin, and O. Celiktutan, “IB-DRR-incremental learning with information-back discrete representation replay,” in Proc. Conf. Comput. Vis. Pattern Recognit. Workshop, 2021, pp. 3533–3542.   
[49] F. Zhu, X.-Y. Zhang, C. Wang, F. Yin, and C.-L. Liu, “Prototype augmentation and self-supervision for incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 5871–5880.   
[50] G. Petit, A. Popescu, H. Schindler, D. Picard, and B. Delezoide, “FeTrIL: Feature translation for exemplar-free class-incremental learning,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2023, pp. 3911–3920.   
[51] Q. Jodelet, X. Liu, Y. J. Phua, and T. Murata, “Class-incremental learning using diffusion model for distillation and replay,” in Proc. Int. Conf. Comput. Vis. Workshops, 2023, pp. 3425–3433.   
[52] R. Gao and W. Liu, “DDGR: Continual learning with deep diffusionbased generative replay,” in Proc. Int. Conf. Mach. Learn., 2023, pp. 10744–10763.   
[53] D. Lopez-Paz and M.’ A. Ranzato, “Gradient episodic memory for continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 6467–6476.   
[54] A. Chaudhry, M. ’A. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with A-GEM,” in Proc. Int. Conf. Learn. Representations, 2018.   
[55] S. Wang, X. Li, J. Sun, and Z. Xu, “Training networks in null space of feature covariance for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 184–193.   
[56] G. Zeng, Y. Chen, B. Cui, and S. Yu, “Continual learning of contextdependent processing in neural networks,” Nat. Mach. Intell., vol. 1, no. 8, pp. 364–372, 2019.   
[57] S. Tang, D. Chen, J. Zhu, S. Yu, and W. Ouyang, “Layerwise optimization by gradient decomposition for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 9634–9643.   
[58] M. Riemer et al., “Learning to learn without forgetting by maximizing transfer and minimizing interference,” in Proc. Int. Conf. Learn. Representations, 2018.   
[59] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks,” in Proc. Int. Conf. Learn. Representations, 2018.   
[60] J. Xu and Z. Zhu, “Reinforced continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2018, pp. 899–908.   
[61] X. Li, Y. Zhou, T. Wu, R. Socher, and C. Xiong, “Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 3925–3934.   
[62] A. A. Rusu et al., “Progressive neural networks,” 2016, arXiv: 1606.04671.   
[63] R. Aljundi, P. Chakravarty, and T. Tuytelaars, “Expert gate: Lifelong learning with a network of experts,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2017, pp. 3366–3375.   
[64] J. Schwarz et al., “Progress & compress: A scalable framework for continual learning,” in Proc. Int. Conf. Mach. Learn., 2018, pp. 4528–4537.

[65] H. Zhao, Y. Fu, M. Kang, Q. Tian, F. Wu, and X. Li, “MgSvF: Multi-grained slow versus fast framework for few-shot class-incremental learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 3, pp. 1576–1588, Mar. 2024.   
[66] Y. Liu, B. Schiele, and Q. Sun, “Adaptive aggregation networks for classincremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 2544–2553.   
[67] Q. Pham, C. Liu, and S. Hoi, “DualNet: Continual learning, fast and slow,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 16131–16144.   
[68] F.-Y. Wang et al., “Beef: Bi-compatible class-incremental learning via energy-based expansion and fusion,” in Proc. Int. Conf. Learn. Representations, 2023.   
[69] Z. Wang et al., “Learning to prompt for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 139–149.   
[70] Z. Wang et al., “DualPrompt: Complementary prompting for rehearsalfree continual learning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 631–648.   
[71] J. S. Smith et al., “Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 11909–11919.   
[72] Y. Wang, Z. Huang, and X. Hong, “S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 5682–5695.   
[73] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proc. Nat. Acad. Sci. USA, vol. 114, no. 13, pp. 3521–3526, 2017.   
[74] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in Proc. Int. Conf. Mach. Learn., 2017, pp. 3987–3995.   
[75] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 139–154.   
[76] R. Aljundi, K. Kelchtermans, and T. Tuytelaars, “Task-free continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 11254–11263.   
[77] S.-W. Lee, J.-H. Kim, J. Jun, J.-W. Ha, and B.-T. Zhang, “Overcoming catastrophic forgetting by incremental moment matching,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 4652–4662.   
[78] Y. Yang, D.-W. Zhou, D.-C. Zhan, H. Xiong, and Y. Jiang, “Adaptive deep models for incremental learning: Considering capacity scalability and sustainability,” in Proc. Int. Conf. Knowl. Discov. Data Mining, 2019, pp. 74–82.   
[79] Y. Yang, D.-W. Zhou, D.-C. Zhan, H. Xiong, Y. Jiang, and J. Yang, “Costeffective incremental deep model: Matching model capacity with the least sampling,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 4, pp. 3575–3588, Apr. 2023.   
[80] J. Lee, H. G. Hong, D. Joo, and J. Kim, “Continual learning with extended kronecker-factored approximate curvature,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 9001–9010.   
[81] Z. Li and D. Hoiem, “Learning without forgetting,” in Proc. Eur. Conf. Comput. Vis., 2016, pp. 614–629.   
[82] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “iCaRL: Incremental classifier and representation learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2017, pp. 2001–2010.   
[83] Y. Wu et al., “Large scale incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 374–382.   
[84] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Lifelong learning via progressive distillation and retrospection,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 437–452.   
[85] K. Lee, K. Lee, J. Shin, and H. Lee, “Overcoming catastrophic forgetting with unlabeled data in the wild,” in Proc. Int. Conf. Comput. Vis., 2019, pp. 312–321.   
[86] J. Zhang et al., “Class-incremental learning via deep model consolidation,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis., 2020, pp. 1131–1140.   
[87] J. Smith, Y.-C. Hsu, J. Balloch, Y. Shen, H. Jin, and Z. Kira, “Always be dreaming: A new approach for data-free class-incremental learning,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 9374–9384.   
[88] D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Co-transport for classincremental learning,” in Proc. 29th ACM Int. Conf. Multimedia, 2021, pp. 1645–1654.   
[89] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, “Learning a unified classifier incrementally via rebalancing,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 831–839.

[90] Y. Lu, M. Wang, and W. Deng, “Augmented geometric distillation for data-free incremental person reid,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 7329–7338.   
[91] J. Park, M. Kang, and B. Han, “Class-incremental learning for action recognition in videos,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 13698–13707.   
[92] P. Dhar, R. V. Singh, K.-C. Peng, Z. Wu, and R. Chellappa, “Learning without memorizing,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 5138–5146.   
[93] M. Kang, J. Park, and B. Han, “Class-incremental learning by knowledge distillation with adaptive feature consolidation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 16071–16080.   
[94] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “Podnet: Pooled outputs distillation for small-tasks incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 86–102.   
[95] X. Hu, K. Tang, C. Miao, X.-S. Hua, and H. Zhang, “Distilling causal effect of data in class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 3957–3966.   
[96] C. Simon, P. Koniusz, and M. Harandi, “On learning the geodesic path for incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 1591–1600.   
[97] H. Jung, J. Ju, M. Jung, and J. Kim, “Less-forgetful learning for domain expansion in deep neural networks,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2018, pp. 3358–3365.   
[98] D. Li, S. Tasci, S. Ghosh, J. Zhu, J. Zhang, and L. Heck, “RILOD: Near real-time incremental learning for object detection at the edge,” in Proc. 4th ACM/IEEE Symp. Edge Comput., 2019, pp. 113–126.   
[99] Q. Gao, C. Zhao, B. Ghanem, and J. Zhang, “R-DFCIL: Relation-guided representation learning for data-free class incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 423–439.   
[100] S. Dong, X. Hong, X. Tao, X. Chang, X. Wei, and Y. Gong, “Few-shot class-incremental learning via relation knowledge distillation,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2021, pp. 1255–1263.   
[101] X. Tao, X. Chang, X. Hong, X. Wei, and Y. Gong, “Topology-preserving class-incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 254–270.   
[102] Y. Liu, X. Hong, X. Tao, S. Dong, J. Shi, and Y. Gong, “Model behavior preserving for class-incremental learning,” IEEE Trans. Neural Netw. Learn. Syst., vol. 34, no. 10, pp. 7529–7540, Oct. 2023.   
[103] N. Asadi, M. R. Davari, S. Mudur, R. Aljundi, and E. Belilovsky, “Prototype-sample relation distillation: Towards replay-free continual learning,” in Proc. Int. Conf. Mach. Learn., 2023, pp. 1093–1106.   
[104] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, L. Ma, S. Pu, and D.-C. Zhan, “Forward compatible few-shot class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 9046–9056.   
[105] L. Yu et al., “Semantic drift compensation for class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 6982–6991.   
[106] Y. Shi et al., “Mimicking the oracle: An initial phase decorrelation approach for class incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 16722–16731.   
[107] S. Jie, Z.-H. Deng, and Z. Li, “Alleviating representational shift for continual fine-tuning,” in Proc. Conf. Comput. Vis. Pattern Recognit. Workshop, 2022, pp. 3810–3819.   
[108] P. Singh, V. K. Verma, P. Mazumder, L. Carin, and P. Rai, “Calibrating CNNs for lifelong learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2020, pp. 15579–15590.   
[109] P. Singh, P. Mazumder, P. Rai, and V. P. Namboodiri, “Rectification-based knowledge retention for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 15282–15291.   
[110] F. M. Castro, M. J. Marín-Jiménez, N. Guil, C. Schmid, and K. Alahari, “End-to-end incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2018, pp. 233–248.   
[111] E. Belouadah and A. Popescu, “IL2M: Class incremental learning with dual memory,” in Proc. Int. Conf. Comput. Vis., 2019, pp. 583–592.   
[112] B. Zhao, X. Xiao, G. Gan, B. Zhang, and S.-T. Xia, “Maintaining discrimination and fairness in class incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 13208–13217.   
[113] H. Ahn, J. Kwak, S. Lim, H. Bang, H. Kim, and T. Moon, “SS-IL: Separated softmax for incremental learning,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 844–853.   
[114] F. Pernici, M. Bruni, C. Baecchi, F. Turchini, and A. D. Bimbo, “Classincremental learning with pre-allocated fixed classifiers,” in Proc. Int. Conf. Pattern Recognit., 2021, pp. 6259–6266.

[115] C. Zhang, N. Song, G. Lin, Y. Zheng, P. Pan, and Y. Xu, “Few-shot incremental learning with continually evolved classifiers,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 12455–12464.   
[116] D.-W. Zhou, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Revisiting classincremental learning with pre-trained models: Generalizability and adaptivity are all you need,” 2023, arXiv:2303.07338.   
[117] M. D. McDonnell, D. Gong, A. Parveneh, E. Abbasnejad, and A. van den Hengel, “RanPAC: Random projections and pre-trained models for continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2023, pp. 12022–12053.   
[118] D. Goswami, Y. Liu, B. Twardowski, and J. van de Weijer, “FeCAM: Exploiting the heterogeneity of class distributions in exemplar-free continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2023, pp. 6582–6595.   
[119] Q.-W. Wang, D.-W. Zhou, Y.-K. Zhang, D.-C. Zhan, and H.-J. Ye, “Fewshot class-incremental learning via training-free prototype calibration,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2023, pp. 15060–15076.   
[120] W. Shi and M. Ye, “Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 1772–1781.   
[121] D.-W. Zhou, H.-J. Ye, L. Ma, D. Xie, S. Pu, and D.-C. Zhan, “Few-shot class-incremental learning by sampling multi-phase tasks,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 11, pp. 12816–12831, Nov. 2023.   
[122] K. Wei, C. Deng, X. Yang, and M. Li, “Incremental embedding learning via zero-shot translation,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2021, pp. 10254–10262.   
[123] G. M. V. D. Ven, Z. Li, and A. S. Tolias, “Class-incremental learning with generative classifiers,” in Proc. Conf. Comput. Vis. Pattern Recognit. Workshop, 2021, pp. 3611–3620.   
[124] M. Welling, “Herding dynamical weights to learn,” in Proc. Int. Conf. Mach. Learn., 2009, pp. 1121–1128.   
[125] M. A. Wilson and B. L. McNaughton, “Reactivation of hippocampal ensemble memories during sleep,” Science, vol. 265, no. 5172, pp. 676–679, 1994.   
[126] A. Tambini and L. Davachi, “Persistence of hippocampal multivoxel patterns into postencoding rest is related to memory,” Proc. Nat. Acad. Sci. USA, vol. 110, no. 48, pp. 19591–19596, 2013.   
[127] D. N. Barry and B. C. Love, “A neural network account of memory replay and knowledge consolidation,” Cereb. Cortex, vol. 33, no. 1, pp. 83–95, 2023.   
[128] R. Ratcliff, “Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions,” Psychol. Rev., vol. 97, no. 2, pp. 285, 1990.   
[129] J. Snell, K. Swersky, and R. Zemel, “Prototypical networks for fewshot learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 4080–4090.   
[130] A. Rannen, R. Aljundi, M. B. Blaschko, and T. Tuytelaars, “Encoder based lifelong learning,” in Proc. Int. Conf. Comput. Vis., 2017, pp. 1320–1328.   
[131] A. Mallya and S. Lazebnik, “PackNet: Adding multiple tasks to a single network by iterative pruning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2018, pp. 7765–7773.   
[132] J. Rajasegaran, M. Hayat, S. Khan, F. S. Khan, and L. Shao, “Random path selection for incremental learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2019, pp. 12669–12679.   
[133] D. Park, S. Hong, B. Han, and K. M. Lee, “Continual learning by asymmetric loss approximation with single-side overestimation,” in Proc. Int. Conf. Comput. Vis., 2019, pp. 3335–3344.   
[134] M. Rostami, S. Kolouri, and P. K. Pilly, “Complementary learning for overcoming catastrophic forgetting using experience replay,” in Proc. Int. Joint Conf. Artif. Intell., 2019, pp. 3339–3345.   
[135] C.-Y. Hung, C.-H. Tu, C.-E. Wu, C.-H. Chen, Y.-M. Chan, and C.-S. Chen, “Compacting, picking and growing for unforgetting continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2019, pp. 13669–13679.   
[136] D. Abati, J. Tomczak, T. Blankevoort, S. Calderara, R. Cucchiara, and B. E. Bejnordi, “Conditional channel gated networks for task-aware continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 3931–3940.   
[137] K. J. Joseph, J. Rajasegaran, S. Khan, F. S. Khan, and V. N. Balasubramanian, “Incremental object detection via meta-learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 12, pp. 9209–9216, Dec. 2022.   
[138] J. He, R. Mao, Z. Shao, and F. Zhu, “Incremental learning in online scenario,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 13926–13935.

[139] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. Calderara, “Dark experience for general continual learning: A strong, simple baseline,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2020, pp. 15920–15930.   
[140] A. Prabhu, P. HS Torr, and P. K. Dokania, “GDumb: A simple approach that questions our progress in continual learning,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 524–540.   
[141] C. D. Kim, J. Jeong, and G. Kim, “Imbalanced continual learning with partitioning reservoir sampling,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 411–428.   
[142] T. L. Hayes and C. Kanan, “Lifelong machine learning with deep streaming linear discriminant analysis,” in Proc. Conf. Comput. Vis. Pattern Recognit. Workshop, 2020, pp. 220–221.   
[143] Y. Liu, S. Parisot, G. Slabaugh, X. Jia, A. Leonardis, and T. Tuytelaars, “More classifiers, less forgetting: A generic multi-classifier paradigm for incremental learning,” in Proc. Eur. Conf. Comput. Vis., 2020, pp. 699–716.   
[144] Y. Liu, B. Schiele, and Q. Sun, “RMM: Reinforced memory management for class-incremental learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 3478–3490.   
[145] H. Yin et al., “Mitigating forgetting in online continual learning with neuron calibration,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 10260–10272.   
[146] R. Tiwari, K. Killamsetty, R. Iyer, and P. Shenoy, “GCR: Gradient coreset based replay buffer selection for continual learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 99–108.   
[147] K. J. Joseph, S. Khan, F. S. Khan, R. M. Anwer, and V. N. Balasubramanian, “Energy-based latent aligner for incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 7452–7461.   
[148] J. Xie, S. Yan, and X. He, “General incremental learning with domainaware categorical representations,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 14351–14360.   
[149] J. Dong et al., “Federated class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 10164–10173.   
[150] K. Zhu, W. Zhai, Y. Cao, J. Luo, and Z.-J. Zha, “Self-sustaining representation expansion for non-exemplar class-incremental learning,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 9296–9305.   
[151] T.-Y. Wu et al., “Class-incremental learning with strong pre-trained models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 9601–9610.   
[152] B. Ermis, G. Zappella, M. Wistuba, A. Rawal, and C. Archambeau, “Memory efficient continual learning with transformers,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 10629–10642.   
[153] I. Goodfellow et al., “Generative adversarial nets,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2014, pp. 2672–2680.   
[154] D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” in Proc. Int. Conf. Learn. Representations, 2014.   
[155] M. Mirza and S. Osindero, “Conditional generative adversarial nets,” 2014, arXiv:1411.1784.   
[156] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2020, pp. 6840–6851.   
[157] M. Zaj ˛ac et al., “Exploring continual learning of diffusion models,” 2023, arXiv:2303.15342.   
[158] J. S. Smith et al., “Continual diffusion: Continual customization of textto-image diffusion with C-LoRA,” 2023, arXiv:2304.06027.   
[159] L. Korycki and B. Krawczyk, “Class-incremental experience replay for continual learning under concept drift,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 3649–3658.   
[160] A. Maracani, U. Michieli, M. Toldo, and P. Zanuttigh, “Recall: Replaybased continual learning in semantic segmentation,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 7026–7035.   
[161] A. Villa, K. Alhamoud, V. Escorcia, F. Caba, J. L. Alcázar, and B. Ghanem, “vclimb: A novel video class incremental learning benchmark,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2022, pp. 19035–19044.   
[162] E. Verwimp, M. D. Lange, and T. Tuytelaars, “Rehearsal revealed: The limits and merits of revisiting samples in continual learning,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 9385–9394.   
[163] L. Bonicelli, M. Boschini, A. Porrello, C. Spampinato, and S. Calderara, “On the effectiveness of Lipschitz-driven rehearsal in continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 31886–31901.   
[164] L. Yu, T. Hu, L. Hong, Z. Liu, A. Weller, and W. Liu, “Continual learning by modeling intra-class variation,” 2022, arXiv:2210.05398.   
[165] Y. Zhang, B. Kang, B. Hooi, S. Yan, and J. Feng, “Deep long-tailed learning: A survey,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 9, pp. 10795–10816, Sep. 2023.

[166] G. M. Van de Ven, H. T. Siegelmann, and A. S. Tolias, “Brain-inspired replay for continual learning with artificial neural networks,” Nat. Commun., vol. 11, no. 1, pp. 4069, 2020.   
[167] L. Wang, B. Lei, Q. Li, H. Su, J. Zhu, and Y. Zhong, “Triple-memory networks: A brain-inspired method for continual learning,” IEEE Trans. Neural Netw. Learn. Syst., vol. 33, no. 5, pp. 1925–1934, May 2022.   
[168] R. Yoshihashi, W. Shao, R. Kawakami, S. You, M. Iida, and T. Naemura, “Classification-reconstruction learning for open-set recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 4016– 4025.   
[169] X. Liu et al., “Generative feature replay for class-incremental learning,” in Proc. Conf. Comput. Vis. Pattern Recognit. Workshop, 2020, pp. 226–227.   
[170] Y. Guo, W. Hu, D. Zhao, and B. Liu, “Adaptive orthogonal projection for batch and online continual learning,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2022, pp. 6783–6791.   
[171] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in Proc. Int. Conf. Mach. Learn., 2017, pp. 1126–1135.   
[172] J. Rajasegaran, S. Khan, M. Hayat, F. S. Khan, and M. Shah, “iTAML: An incremental task-agnostic meta-learning approach,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2020, pp. 13588–13597.   
[173] R. Wang, Y. Bao, B. Zhang, J. Liu, W. Zhu, and G. Guo, “Anti-retroactive interference for lifelong learning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 163–178.   
[174] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.   
[175] T. Elsken, J. H. Metzen, and F. Hutter, “Neural architecture search: A survey,” J. Mach. Learn. Res., vol. 20, no. 1, pp. 1997–2017, 2019.   
[176] Z.-H. Zhou, Ensemble Methods: Foundations and Algorithms. Boca Raton, FL, USA: CRC Press, 2012.   
[177] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” 2015, arXiv:1503.02531.   
[178] M. Jia et al., “Visual prompt tuning,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 709–727.   
[179] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 8748–8763.   
[180] H. Cha, J. Lee, and J. Shin, “Co2L: Contrastive continual learning,” in Proc. Int. Conf. Comput. Vis., 2021, pp. 9516–9525.   
[181] Y.-M. Tang, Y.-X. Peng, and W.-S. Zheng, “When prompt-based incremental learning does not meet strong pretraining,” in Proc. Int. Conf. Comput. Vis., 2023, pp. 1706–1716.   
[182] O. Ostapenko, P. Rodriguez, M. Caccia, and L. Charlin, “Continual learning via local module composition,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 30298–30312.   
[183] S.-A. Rebuffi, H. Bilen, and A. Vedaldi, “Learning multiple visual domains with residual adapters,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2017, pp. 506–516.   
[184] L. L. Cam, Asymptotic Methods in Statistical Decision Theory, Berlin, Germany: Springer Science & B. Media, 2012.   
[185] F. Benzing, “Unifying importance based regularisation methods for continual learning,” in Proc. Int. Conf. Artif. Intell. Statist., 2022, pp. 2372– 2396.   
[186] T. Mensink, J. Verbeek, F. Perronnin, and G. Csurka, “Distance-based image classification: Generalizing to new classes at near-zero cost,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 35, no. 11, pp. 2624–2637, Nov. 2013.   
[187] H. Zhao, X. Qin, S. Su, Y. Fu, Z. Lin, and X. Li, “When video classification meets incremental classes,” in Proc. 29th ACM Int. Conf. Multimedia, 2021, pp. 880–889.   
[188] W. Park, D. Kim, Y. Lu, and M. Cho, “Relational knowledge distillation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2019, pp. 3967– 3976.   
[189] G. Yang et al., “Uncertainty-aware contrastive distillation for incremental semantic segmentation,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 2, pp. 2567–2581, Feb. 2023.   
[190] N. Pu, W. Chen, Y. Liu, E. M. Bakker, and M. S. Lew, “Lifelong person reidentification via adaptive knowledge accumulation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 7901–7910.   
[191] Q. Pham, C. Liu, and H. Steven, “Continual normalization: Rethinking batch normalization for online continual learning,” in Proc. Int. Conf. Learn. Representations, 2022.   
[192] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in Proc. Int. Conf. Mach. Learn., 2015, pp. 448–456.

[193] B. Zheng, D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Preserving locality in vision transformers for class incremental learning,” in Proc. IEEE Int. Conf. Multimedia Expo, 2023, pp. 1157–1162.   
[194] G. Shi, J. Chen, W. Zhang, L.-M. Zhan, and X.-M. Wu, “Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 6747–6761.   
[195] S. I. Mirzadeh, M. Farajtabar, R. Pascanu, and H. Ghasemzadeh, “Understanding the role of training regimes in continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2020, pp. 7308–7320.   
[196] R. M. Nosofsky, “Attention, similarity, and the identification– categorization relationship,” J. Exp. Psychol., Gen., vol. 115, no. 1, 1986, Art. no. 39.   
[197] V. Verma et al., “Manifold mixup: Better representations by interpolating hidden states,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 6438–6447.   
[198] D. Hendrycks, K. Zhao, S. Basart, J. Steinhardt, and D. Song, “Natural adversarial examples,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 15262–15271.   
[199] A. Alfassy et al., “Feta: Towards specializing foundational models for expert task applications,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 29873–29888.   
[200] Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato, and F. Huang, “A tutorial on energy-based learning,” Predicting Struct. Data, vol. 1, pp. 1–71, 2006.   
[201] S. Li, Y. Du, G. van de Ven, and I. Mordatch, “Energy-based models for continual learning,” in Proc. Conf. Lifelong Learn. Agents, 2022, pp. 1–22.   
[202] O. Russakovsky et al., “Imagenet large scale visual recognition challenge,” Int. J. Comput. Vis., vol. 115, no. 3, pp. 211–252, 2015.   
[203] Y. LeCun, C. Cortes, and C. J. Burges, “Mnist handwritten digit database,” AT&T Labs, vol. 2, 2010, Art. no. 18. [Online]. Available: http://yann. lecun.com/exdb/mnist   
[204] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, “The Caltech-UCSD Birds-200–2011 dataset,” California Institute of Technology, Tech. Rep. CNS-TR-2011-001, 2011.   
[205] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2019, pp. 8026–8037.   
[206] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, and D.-C. Zhan, “Pycil: A Python toolbox for class-incremental learning,” Sci. China Inf. Sci., vol. 66, no. 9, pp. 197101–197102, 2023.   
[207] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2015, pp. 770–778.   
[208] S. d’Ascoli, H. Touvron, M. L. Leavitt, A. S. Morcos, G. Biroli, and L. Sagun, “ConViT: Improving vision transformers with soft convolutional inductive biases,” in Proc. Int. Conf. Mach. Learn., 2021, pp. 2286–2296.   
[209] Y. Zhong, J.-H. Pan, H. Li, and W.-S. Zheng, “Weakly supervised action anticipation without object annotations,” Front. Comput. Sci., vol. 17, no. 2, 2023, Art. no. 172313.   
[210] D.-W. Zhou, H.-J. Ye, and D.-C. Zhan, “Learning placeholders for open-set recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 4401–4410.   
[211] S. Yan et al., “Generative negative text replay for continual visionlanguage pretraining,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 22–38.   
[212] A. Rios, N. Ahuja, I. Ndiour, U. Genc, L. Itti, and O. Tickoo, “incDFM: Incremental deep feature modeling for continual novelty detection,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 588–604.   
[213] A. Bendale and T. Boult, “Towards open world recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2015, pp. 1893–1902.   
[214] M. Abdelsalam, M. Faramarzi, S. Sodhani, and S. Chandar, “IIRC: Incremental implicitly-refined classification,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2021, pp. 11038–11047.   
[215] Y. Guo, B. Liu, and D. Zhao, “Online continual learning through mutual information maximization,” in Proc. Int. Conf. Mach. Learn., 2022, pp. 8109–8126.   
[216] J. Pourcel, N.-S. Vu, and R. M. French, “Online task-free continual learning with dynamic sparse distributed memory,” in Proc. Eur. Conf. Comput. Vis., 2022, pp. 739–756.   
[217] F. Zhu, Z. Cheng, X.-Y. Zhang, and C.-L. Liu, “Class-incremental learning via dual augmentation,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2021, pp. 14306–14318.   
[218] Y. Liu, Y. Li, B. Schiele, and Q. Sun, “Online hyperparameter optimization for class-incremental learning,” in Proc. Conf. Assoc. Advance. Artif. Intell., 2023, pp. 8906–8913.

[219] Z. Wang et al., “SparCL: Sparse continual learning on the edge,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 20366–20380.   
[220] A. Prabhu et al., “Computationally budgeted continual learning: What does matter?,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2023, pp. 3698–3707.   
[221] G. Kim, C. Xiao, T. Konishi, Z. Ke, and B. Liu, “A theoretical study on solving continual learning,” in Proc. Int. Conf. Neural Inf. Process. Syst., 2022, pp. 5065–5079.   
[222] L. Caccia, R. Aljundi, N. Asadi, T. Tuytelaars, J. Pineau, and E. Belilovsky, “New insights on reducing abrupt representation change in online continual learning,” in Proc. Int. Conf. Learn. Representations, 2022.

![](images/c4a4adb990afb307589003cc55dd688dac6f3d44135b4300b2f418034d501c62.jpg)

<details>
<summary>natural_image</summary>

Portrait of a young man wearing glasses and a suit (no text or symbols visible)
</details>

Da-Wei Zhou received the PhD degree in computer science from Nanjing University, China, in 2024. He is currently an associate researcher with the National Key Lab for Novel Software Technology, the School of Artificial Intelligence, Nanjing University. His research interests lie primarily in machine learning and computer vision. He has published extensively on top-tier conferences and journals in relevant fields, including NeurlPS, ICLR, ICML, CVPR, ECCV, IEEE Transactions on Pattern Analysis and Machine Intelligence, etc. He serves as Program Committee

member in leading conferences, such as ICML, NeurIPS, ICLR, CVPR, etc, and was recognized as NeurIPS’s top reviewer and CVPR’s outstanding reviewer.

![](images/7a5c506087edfd8815a7e5e8c244ee9438c5b7fdc1fef3ec320a81e8eca39bb2.jpg)

<details>
<summary>natural_image</summary>

Portrait of a young man wearing glasses and a gray shirt (no text or symbols visible)
</details>

Qi-Wei Wang is currently working toward the MSc degree with the National Key Lab for Novel Software Technology, School of Artificial Intelligence, Nanjing University, China.

![](images/a0a8258b00b0f545ec62feba0fc0a412a5b8b200f7e0d1fe004129305b52c44b.jpg)

<details>
<summary>natural_image</summary>

Portrait of a person wearing glasses (no text or symbols visible)
</details>

Zhi-Hong Qi is currently working toward the MSc degree with the National Key Lab for Novel Software Technology, School of Artificial Intelligence, Nanjing University, China.

![](images/b4b6eaea8a0f49817aa352d76b3312cb6c04d920aa5ec64a9a7428fc614b94a5.jpg)

<details>
<summary>natural_image</summary>

Portrait of a man wearing glasses and a collared shirt (no text or symbols visible)
</details>

Han-Jia Ye received the PhD degree in computer science from Nanjing University, China, in 2019. He joined the School of Artificial Intelligence at Nanjing University as a faculty member in the same year and currently holds the position of associate professor. His research focuses primarily on machine learning, with interests in representation learning, model reuse, and meta-learning. He has served as the Tutorial Co-Chair for SDM 2023. Additionally, he participates as area chairs in conferences, such as ICML, NeurIPS, and CVPR, and others.

![](images/f520b1caf82cc8494b2b6fd30c9bb67be2e82f0f95dd4032f65c6ea755a874fe.jpg)

<details>
<summary>natural_image</summary>

Portrait photo of a smiling man in a collared shirt (no text or symbols visible)
</details>

De-Chuan Zhan received the PhD degree in computer science, Nanjing University, China in 2010. In the same year, he became a faculty member in the Department of Computer Science and Technology at Nanjing University, China. He is currently a professor with the School of Artificial Intelligence at Nanjing University. His research interests are mainly in machine learning, data mining, and mobile intelligence. He has published more than 100 papers in leading international journal/conferences. He serves as an editorial board member of IDA and IJAPR, and serves as SPC/PC in leading conferences, such as IJCAI, AAAI, ICML, NeurIPS, etc.

![](images/ee5c5bfed1be1a4684556d6f5ee82de5c28bffbe7266b4a30aa6963d2b916eae.jpg)

<details>
<summary>natural_image</summary>

Portrait of a young man in a collared shirt (no text or symbols visible)
</details>

Ziwei Liu (Member, IEEE) is currently a nanyang assistant professor with Nanyang Technological University, Singapore. His research interests include computer vision machine learning and computer graphics. He has published extensively on top-tier conferences and journals in relevant fields, including CVPR, ICCV, ECCV, NeurlPS, ICLR, ICML, IEEE Transactions on Pattern Analysis and Machine Intelligence, ACM Transactions on Graphics and Nature - Machine Intelligence. He is the recipient of ICCV Young Researcher Award, HKSTP Best Paper Award, CVPR Best Paper Award Candidate, ICBS Frontiers of Science Award and MIT Technology Review Innovators under 35 Asia Pacific. He serves as an area chair of CVPR, ICCV, ECCV, NeurlPS, and ICLR, as well as an associate editor of International Journal of Computer Vision.