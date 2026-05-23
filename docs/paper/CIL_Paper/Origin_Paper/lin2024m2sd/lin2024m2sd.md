# M2SD:Multiple Mixing Self-Distillation for Few-Shot Class-Incremental Learning

Jinhao Lin1, 2, Ziheng Wu2, Weifeng Lin1, 2, Jun Huang2, RongHua Luo1\*

1 South China University of Technology

2 Alibaba Group

csljh jasper@mail.scut.edu.cn, zhoulou.wzh@alibaba-inc.com, eelinweifeng@mail.scut.edu.cn,

huangjun.hj@alibaba-inc.com, rhluo@scut.edu.cn

# Abstract

Few-shot Class-incremental learning (FSCIL) is a challenging task in machine learning that aims to recognize new classes from a limited number of instances while preserving the ability to classify previously learned classes without retraining the entire model. This presents challenges in updating the model with new classes using limited training data, particularly in balancing acquiring new knowledge while retaining the old. We propose a novel method named Multiple Mxing Self-Distillation (M2SD) during the training phase to address these issues. Specifcally, we propose a dual-branch structure that facilitates the expansion of the entire feature space to accommodate new classes. Furthermore, we introduce a feature enhancement component that can pass additional enhanced information back to the base network by self-distillation, resulting in improved classifcation performance upon adding new classes. After training, we discard both structures, leaving only the primary network to classify new class instances. Extensive experiments demonstrate that our approach achieves superior performance over previous state-of-the-art methods.

# Introduction

Deep learning has had a signifcant impact on computer vision, particularly in image classifcation (Krizhevsky, Sutskever, and Hinton 2017; Simonyan and Zisserman 2014; Ren et al. 2015). Currently, most computer vision scenarios, including classifcation, rely on data-driven discriminative tasks, requiring obtaining the specifc task data distribution and designing targeted loss functions before model training to achieve better results. Even though numerous publicly available datasets have produced excellent results in various scenarios, the classifcation task suffers from a signifcant limitation: it often performs poorly when new classes are added, particularly when the amount of data in these new classes is insuffcient and uneven (Ren et al. 2019; Zhu et al. 2021b).

In many real-world scenarios, the situation may worsen when new classes have only a few instances. To address this challenge, researchers have introduced Few-shot Learning(FSL)(Chen et al. 2019) to Class Incremental Learn-

![](images/dd071c345f55d54df2a07fe82c266433e97474eb81b778e9aa71cefdf58d798d.jpg)

<details>
<summary>text_image</summary>

Others
Base session
Ours
Few-shot
Class incremental
sessions
</details>

Figure 1: This is a schematic diagram of the boosting of our method in FSCIL. Our method can signifcantly reduce the intra-class distance, boost the inter-class distance, and have better spacing for the added classes. It is refected in the accuracy results and the visual distribution of the fnal features, detailed analysis in Table 1 and Figure 4.

ing(CIL)(Wu et al. 2019) and proposed Few-shot Class Incremental Learning (FSCIL)(Ayub and Wagner 2020). One of the signifcant challenges FSCIL faces is catastrophic forgetting. Introducing new classes to predefned models for a classifcation task by fne-tuning can potentially lead to an overemphasis on the new classes, possibly resulting in a decline in performance for the existing classes(Kirkpatrick et al. 2017).

To address this problem, various methods(Shi et al. 2021) incorporate more robust regularization with the objective function during incremental sessions, resulting in a minor change in the parameters of the new model relative to the original one. Unlike the regularization methods, the idea behind FACT(Zhou et al. 2022) is to prepare for the arrival of new data in advance, which is akin to forward compatibility in software updates. It is more concerned with obtaining a feature space in the base session of FSCIL that is suitable for subsequent incremental learning sessions.

We are inspired by the idea of FACT and propose a method called Multiple Mixing Self-Distillation(M2SD) for FSCIL, which prepares the feature space for incremental sessions and emphasizes feature extensibility and inclusiveness in later incremental sessions, showed in Figure 1.

To prepare the feature space for subsequent incremental sessions, M2SD utilizes multi-scale feature extraction and fusion; the specifc structure can be seen in Figure 2. We extract features at different scales for each instance and fuse them to capture the characteristics of multiple dimensions of the instances fully. By comparing the fusion of these features with a single feature, as in previous work(Ji et al. 2021), the feature module can better understand various aspects of the instance, leading to greater inclusiveness. For expansion, inspired by using virtual classes in FACT, we propose a dualbranch virtual class to enhance further the extensibility of the feature extraction module(Xu and Liu 2019). It becomes aware of potential expansion possibilities by optimizing the dual-branch virtual class and allowing the feature module to see the unknown in advance. It tries to reserve feature space for new classes in the future.

Given that the entire training phase is divided into base and incremental sessions, and referring to CEC(Zhang et al. 2021) to address the issue of imbalanced data between old and new classes, we decouple the model into feature and classifer learning, which can be seen by Figure 2. This structure is used in many CEC follow-up works(Zhou et al. 2022; Song et al. 2023). The parameters are frozen once the feature module is learned using suffcient data from the base session. The feature module functions solely as a feature extractor during incremental learning, and its parameters are no longer updated. The classifer is the only part of the model updated upon the arrival of new classes.

The three main contributions of our paper:

• To our knowledge, we are the frst to introduce selfdistillation to FSCIL. Our method improves model feature discriminative and classifcation accuracy.   
• We propose a two-branch virtual inter-class distillation. Based on the successful use of mixup(Zhang et al. 2017) to construct virtual classes, we have successfully introduced the virtual class constructed by CutMix(Yun et al. 2019), which was negative for FSCIL tasks.   
• Experiments on CIFAR100, CUB200, and miniImageNet demonstrate that our method outperforms the baseline and achieves new SOTA results.

# Related Work

# Few-Shot Learning(FSL)

Few-Shot Learning (FSL) refers to the task of training models on small datasets that typically have only a few instances per class. The goal of it is to overcome the challenge of overftting in traditional deep learning methods, which require large amounts of data to train. Extensive related work in this feld includes metric-based methods(Snell, Swersky, and Zemel 2017a) such as prototype-based models, and optimization-based methods(Li et al. 2019) like gradient-based optimization. Another popular approach is to use meta-learning(Snell, Swersky, and Zemel 2017b), where the model learns how to learn from few-shot instances by training on various tasks. Recent work(Tian et al. 2020; Zhang et al. 2023b) demonstrates that a good feature space is essential for FSL, and further enhancements can be achieved by incorporating self-distillation. Inspired by its fndings, we applied self-distillation to FSCIL to obtain a high-quality feature space.

# Class-Incremental Learning(CIL)

Class-Incremental Learning (CIL) refers to the problem of continuously learning new classes in a lifelong learning scenario, where the data distribution of the new classes may differ from the previously learned classes. Various approaches have been proposed in the literature, including regularization-based methods(Li and Hoiem 2017), memory-based methods(Lopez-Paz and Ranzato 2017; Gidaris and Komodakis 2018), and architecture-based methods(Rusu et al. 2016). In particular, IL2A(Zhu et al. 2021a) highlights the critical role of representation learning in CIL. Integrating data augmentation and knowledge distillation used in IL2A inspires our research.

# Few-Shot Class-Incremental Learning(FSCIL)

Few-shot Class-incremental learning(FSCIL) builds upon the foundations of FSL and CIL(Akyurek et al. 2021). As ¨ a result, the category(Kim and Choi 2021) of FSCIL methods is not fundamentally different from them. Prior works in this area include CEC, which decouples the model into two parts, representation learning, and classifer learning, and it is well-suited for FSCIL. FACT addresses the FS-CIL challenge by constructing virtual classes using manifold mixup(Verma et al. 2019), which improves the quality of the trained feature space for the incremental sessions. Both of these ideas inspired our method.

# Knowledge Distillation

Knowledge distillation(Hinton, Vinyals, and Dean 2015) is a method that involves transferring knowledge from a pretrained model, known as the teacher model, to a new model, known as the student model, to improve student performance. The student model may have a different architecture or be smaller than the teacher model, but they do not need to be identical. Self-distillation(Lee, Hwang, and Shin 2020; Zhang et al. 2019) is a specifc case of knowledge distillation in which a single model generates soft targets that are then used to train another identical model. In this paper, we are the frst to incorporate self-distillation into FSCIL. It can facilitate the model to learn that the feature space will be enriched and that knowledge can be internalized, ultimately leading to enhanced model generalization. So we design a Multiple Mixing Self-Distillation(M2SD) into FSCIL to get a feature space more suitable for FSCIL.

# Problem Set-Up

FSCIL comprises multiple learning sessions that are completed sequentially. During each session, the model is extended to a new subset of the dataset, while the training subsets from previous sessions are inaccessible. The evaluation of the FSCIL method in each session considers all classes learned in previous and current sessions. In summary, the overall training process is divided into a base session and multiple incremental sessions.

![](images/da62b85e58ae880d608b9f9df661c0f92e0b2b8bb906f3668d074eab7bd81cbe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Base Session
        A1["Res Block"] --> B1["Softmax"]
        B1 --> C1["Linear"]
        C1 --> D1["p(y|x)"]
    end

    subgraph Incremental Sessions
        E1["Res Block"] --> F1["Linear - new classifier"]
        F1 --> G1["Classifier updating"]
    end

    subgraph Stage 1 Representation Pre-training Stage
        H1["Res Block"] --> I1["Softmax"]
        I1 --> J1["Linear"]
        J1 --> K1["p(y|x_virtual2)"]
    end

    subgraph Stage 2 Multiple Mixing Self-Distillation (M2SD)
        L1["Res Block"] --> M1["Softmax"]
        M1 --> N1["p(y|x_virtual1)"]
        N1 --> O1["D_KL(P_virtual1||P_virtual2)"]
        P1["Res Block"] --> Q1["Softmax"]
        Q1 --> R1["p(y|x_virtual2)"]
        S1["Res Block"] --> T1["Softmax"]
        T1 --> U1["D_KL(P_ori||P_enhanced)"]
        V1["MHSA/CA"] --> W1["Linear"]
        X1["MHSA/CA"] --> Y1["Linear"]
        Z1["MHSA/CA"] --> AA["Linear"]
        AB["MHSA/CA"] --> AC["Linear"]
        AD["MHSA/CA"] --> AE["Linear"]
        AF["MHSA/CA"] --> AG["Linear"]
    end

    subgraph Stage 3 Classifier Updating
        AH["Res Block"] --> AI["Linear - new classifier"]
        AJ["Res Block"] --> AK["Linear - new classifier"]
        AL["Res Block"] --> AM["Linear - new classifier"]
    end

    B1 --> M1
    M1 --> N1
    N1 --> O1
    O1 --> P1
    P1 --> Q1
    Q1 --> R1
    R1 --> S1
    S1 --> T1
    T1 --> U1
    U1 --> V1
    V1 --> W1
    W1 --> X1
    X1 --> Y1
    Y1 --> Z1
    Z1 --> AA
    AA --> AB
    AB --> AC
    AC --> AD
    AD --> AE
    AE --> AF
```
</details>

Figure 2: Our methodology framework is divided into two main parts: base session and incremental sessions. Base session are divided into two stages. One is the general model pre-training stage, and the other is the stage of M2SD, which is composed of two self-distillation modules. Incremental sessions have only one stage, which is classifer updating.

Base Session: At this session, the model receives sufng data from the training set and is then evaluated on the test $\begin{array} { r l } { \mathcal { D } _ { t r a i n } ^ { 0 } } & { { } = } \end{array}$ (xi, yi)ntrai=1 $\left( x _ { i } , y _ { i } \right) _ { i = 1 } ^ { n _ { t r a i n } }$ in $\mathcal { D } _ { t e s t } ^ { 0 } =$ $\left( x _ { i } , y _ { i } \right) _ { i = 1 } ^ { n _ { t e s t } } , x _ { i }$ test  is an instance in the training dataset, and its corresponding label $y _ { i } \in Y _ { 0 } . Y _ { 0 }$ is the label space of $\mathcal { D } ^ { 0 }$ .

Incremental Sessions: New classes arrive incrementally with insuffcient instances at these sessions. A series of datasets $\{ \mathcal { D } ^ { 1 } , \mathcal { D } ^ { 2 } , \ldots , \mathcal { D } ^ { B } \}$ appearing in sequence. B means there are a total of B incremental sessions. $\mathcal { D } ^ { b } ~ =$ $( x _ { i } , y _ { i } ) _ { i = 1 } ^ { N K } , y _ { i } \in Y _ { b }$ which is the label space of session b and $Y _ { b } \cap Y _ { b ^ { \prime } } = \emptyset$ for $b \neq b ^ { \prime }$ . At session b, only dataset $\mathcal { D } ^ { b }$ can be reached. The subset is represented as an N-way, K-shot to form, which denotes that each session dataset is wrapped in N classes, each class with K instances. After training the dataset for a new session, the model should test the model with classes from this session and all previous sessions.

# Method

In this section, we present our framework to solve the FSCIL problem. Our method consists of three stages: the frst two are performed in the base session, and the last stage for the incremental sessions. The central part of our method is in Stage2. In the optimal setting, it accounts for the latter 80% epochs of the base session training, and it can be explained in subsequent ablation experiments, shown in Table 3.

# Representation Pre-training Stage

This stage begins the base session, and the methodology follows general deep learning. Given the abundance of classes $\{ y _ { 1 } , . . . , y _ { i } \} \in \mathsf { \bar { Y } } _ { 0 }$ , and corresponding instances

$\{ x _ { 1 } , . . . , x _ { i } \} \in X _ { 0 }$ from the dataset $\mathcal { D } ^ { 0 }$ . This stage focuses on enabling the model to quickly assimilate the dataset’s characteristics and acquire the most suitable feature space for the next stage. Our model is $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ with a backbone $\phi ( x )$ and a linear classifer W .

$$
f _ {\theta} (x) = W ^ {T} \phi (x) \tag {1}
$$

$$
J (\theta) = \frac {1}{m} \sum_ {i = 1} ^ {m} L _ {C E} (y _ {i}, W ^ {T} \phi (x _ {i})) \tag {2}
$$

where θ represents $\phi ( x )$ and W parameters. m is the number of training samples, $x _ { i }$ is the i-th input instance from $X _ { 0 } ,$ y i is the corresponding label from base session label space $Y _ { 0 } .$

This stage is crucial in enabling successful self-distillation operations in the subsequent stages. Without a suitable pretrained backbone network $f _ { \theta } ( x )$ , models initialized with random parameters can only impart random knowledge, rendering them ineffective in supporting successful distillation. We will demonstrate the importance of this stage and its impact on the results in the subsequent ablation experiments, shown in Table 3.

# Multi Branch Virtual Classes Mixing Distillation

SAVC(Song et al. 2023), FACT(Zhou et al. 2022), and IL2A(Zhu et al. 2021a) have both improved future predictability by using virtual classes. Based on these works, we have made specifc enhancements to strengthen further the expanding effect of this virtual class on the inter-class distance within the feature space. By introducing multiinstance, we construct virtual classes from different ensemble perspectives, enabling the model to learn virtual classes that can keep feature space for future insertion of new classes.

For our multi-branch, we utilize two different methods: mixup(Zhang et al. 2017) and CutMix(Yun et al. 2019).

<table><tr><td rowspan="2">Method</td><td colspan="9">Accuracy in each session (%)</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Finetune</td><td>61.31</td><td>27.22</td><td>16.37</td><td>6.08</td><td>2.54</td><td>1.56</td><td>1.93</td><td>2.60</td><td>1.40</td></tr><tr><td>iCarl(Rebuffi et al. 2017)</td><td>61.31</td><td>46.32</td><td>42.94</td><td>37.63</td><td>30.49</td><td>24.00</td><td>20.89</td><td>18.80</td><td>17.21</td></tr><tr><td>NCM(Hou et al. 2019)</td><td>61.31</td><td>47.80</td><td>39.31</td><td>31.91</td><td>25.68</td><td>21.35</td><td>18.67</td><td>17.24</td><td>14.17</td></tr><tr><td>TOPIC(Tao et al. 2020)</td><td>61.31</td><td>50.09</td><td>45.17</td><td>41.16</td><td>37.48</td><td>35.52</td><td>32.19</td><td>29.46</td><td>24.42</td></tr><tr><td>CEC†(Zhang et al. 2021)</td><td>72.23</td><td>66.86</td><td>63.19</td><td>60.00</td><td>57.20</td><td>54.31</td><td>51.62</td><td>49.92</td><td>48.28</td></tr><tr><td>ERDFR(Liu et al. 2022)</td><td>71.84</td><td>67.12</td><td>63.21</td><td>59.77</td><td>57.01</td><td>53.95</td><td>51.55</td><td>49.52</td><td>48.21</td></tr><tr><td>MetaFSCIL(Chi et al. 2022)</td><td>72.04</td><td>67.94</td><td>63.77</td><td>60.29</td><td>57.58</td><td>55.16</td><td>52.90</td><td>50.79</td><td>49.19</td></tr><tr><td>Fact†(Zhou et al. 2022)</td><td>76.30</td><td>71.00</td><td>66.61</td><td>62.84</td><td>59.41</td><td>56.06</td><td>53.10</td><td>51.06</td><td>49.31</td></tr><tr><td>ALICE(Peng et al. 2022)</td><td>80.60</td><td>70.60</td><td>67.40</td><td>64.50</td><td>62.50</td><td>60.00</td><td>57.80</td><td>56.80</td><td>55.70</td></tr><tr><td>SSFE-Net(Pan et al. 2023)</td><td>72.06</td><td>66.17</td><td>62.25</td><td>59.74</td><td>56.36</td><td>53.85</td><td>51.96</td><td>49.55</td><td>47.73</td></tr><tr><td>WaRP(Kim et al. 2022)</td><td>72.99</td><td>68.10</td><td>64.31</td><td>61.30</td><td>58.64</td><td>56.08</td><td>53.40</td><td>51.72</td><td>50.65</td></tr><tr><td>CABD(Zhao et al. 2023)</td><td>74.65</td><td>70.43</td><td>66.29</td><td>62.77</td><td>60.75</td><td>57.24</td><td>54.79</td><td>53.65</td><td>52.22</td></tr><tr><td>GKEAL(Zhuang et al. 2023)</td><td>73.59</td><td>68.90</td><td>65.33</td><td>62.29</td><td>59.39</td><td>56.70</td><td>54.20</td><td>52.59</td><td>51.31</td></tr><tr><td>SAVC†(Song et al. 2023)</td><td>81.02</td><td>75.59</td><td>71.36</td><td>67.48</td><td>64.40</td><td>60.86</td><td>57.97</td><td>56.10</td><td>54.59</td></tr><tr><td>Ours</td><td>82.11</td><td>79.92</td><td>75.44</td><td>71.31</td><td>68.29</td><td>64.32</td><td>61.13</td><td>58.64</td><td>56.51</td></tr></table>

Table 1: This table mainly shows the performance of our method and previous work on each session on the miniImageNet. Three of these methods†are based on our replication implementation, and the others are taken from their articles. For the other two datasets, refer to the Appendix for specifc results.

![](images/5839e65d50f1d4e1cfac4522bea1cb9d02adf3f6e2aa12a36f99cdddb1be8120.jpg)  
Finetune   
ERDFR

Figure 3: The comparison is made with the previous state-of-the-art works on three mainstream benchmark datasets in CI-FAR100, CUB200, and miniImageNet. Our method outperforms all current methods for all sessions and achieves signifcant performance improvement in the average case. Detailed fgures are in Table 1 and the Appendix.

Mixup generates virtual instances by linearly interpolating pairs of instances, focusing on producing greater diversity, which is crucial for virtual classes. In contrast, CutMix combines pairs of images by cutting and pasting, concentrating on generating more realistic instances. Since the new classes in the incremental sessions are real, the virtual classes need to be authentic. The diverse and realistic branches are distilled based on the characteristics of the newly arrived class instances in the incremental sessions. We must admit that not every virtual instance matches our analysis above, adding random factors to the method and making it more robust.

This stage is still part of the base session, so the dataset used is $D _ { 0 }$ . During the same mini-batch $\{ x _ { 1 } , . . . , x _ { b } \}$ , b indicates the size of the mini-batch. We randomly interpolate two different classes of instances $( x _ { i } , x _ { j } )$ , corresponding $y _ { i } \neq y _ { j }$ to generate two instances of two virtual classes

denoted as $x _ { I } ^ { v i r t u a l }$ and $x _ { I I } ^ { v i r t u a l }$

$$
x _ {I} ^ {\text { virtual }} = \operatorname{mixup} (x _ {i}, x _ {j}) = \lambda x _ {i} + (1 - \lambda) x _ {j} \tag {3}
$$

$$
x _ {I I} ^ {\text { virtual }} = \operatorname{CutMix} \left(x _ {i}, x _ {j}\right) = M \odot x _ {i} + (1 - M) \odot x _ {j} \tag {4}
$$

$$
\lambda \in \beta (\alpha , \alpha) \tag {5}
$$

where two λ is the same, a random number of interpolation coeffcients sampled from Beta distribution. $\bar { M _ { \mathrm { ~ \scriptsize ~ \in ~ } } }$ $\{ 0 , 1 \} ^ { W \times H }$ is a binary mask indicating the patch’s location cropped from the two instances. In order to get the CutMix mask, frst determine the coordinates of the bounding box $( r _ { x } , r _ { y } , r _ { w } , r _ { h } )$ . Unif refers to Uniform Distribution.

$$
r _ {x} \sim \operatorname{Unif} (0 + r _ {w} / 2, W - r _ {w} / 2), \quad r _ {w} = W \sqrt {1 - \lambda}
$$

$$
r _ {y} \sim \operatorname{Unif} (0 + r _ {h} / 2, H - r _ {h} / 2), \quad r _ {h} = H \sqrt {1 - \lambda} \tag {6}
$$

The corresponding labels of the virtual classes are $y _ { I } ^ { v i r t u a l }$ and $y _ { I I } ^ { v i r t u a l }$ .

$$
y _ {I} ^ {\text { virtual }} = y _ {I I} ^ {\text { virtual }} = \lambda y _ {i} + (1 - \lambda) y _ {j} \tag {7}
$$

Similar to other works(Song et al. 2023; Zhou et al. 2022) that use virtual classes, we force $f _ { \theta } ( x ^ { \mathrm { v i r t u a l } } )$ to ft the distribution of virtual classes with equation(8). The virtual class information is not directly used in the subsequent incremental sessions. However, it encourages the $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ to adapt to types that have never been seen before, a characteristic of incremental sessions.

$$
\mathcal {L} _ {c e} ^ {v} = \sum \mathcal {L} _ {c e} (y _ {i} ^ {v i r t u a l}, f _ {\theta} (x _ {i} ^ {v i r t u a l})), i = I, I I \tag {8}
$$

It is also required that the distribution obtained from the two virtual classes is consistent, which makes $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ stable and consistent in the virtual classes space composed of multi-mixture enhancements. So Kullback–Leibler divergence (Kullback and Leibler 1951) is used here.

$$
\begin{array}{r l} \mathcal {L} _ {K L} ^ {v} & = \mathcal {D} _ {K L} \left(f _ {\theta} (x _ {I} ^ {v i r t u a l}) | | (f _ {\theta} (x _ {I I} ^ {v i r t u a l})) \right. \\ & \quad + \mathcal {D} _ {K L} \left(f _ {\theta} (x _ {I} ^ {v i r t u a l}) | | (f _ {\theta} (x _ {I I} ^ {v i r t u a l})) \right. \end{array} \tag {9}
$$

$$
+ \mathcal {D} _ {K L} \left(f _ {\theta} (x _ {I I} ^ {v i r t u a l}) | | (f _ {\theta} (x _ {I} ^ {v i r t u a l})) \right.
$$

# Self-Distillation with Attention Enhancement

In order to fuse features from multiple phases of the network and obtain more discriminative features(Zhang et al. 2023a), we designed a feature enhancement module $\bar { F } E ( x )$ to perform self-distillation on the network.

$$
F E (x) = F P N (E N (\phi (x))) = F P N (\phi^ {\prime} (x))) \tag {10}
$$

The frst step is to use attention $E N ( \cdot )$ to augment the feature obtained from $\phi ( x )$ and obtain high-quality features $\phi ^ { \prime } ( x )$ . Then multi-scale feature fusion is introduced to obtain more semantic information by FPN from the attentionenhanced virtual class features. Finally, we apply a selfdistillation module that uses the enhanced features to align the original network features.

Attention Enhanced: To fully use the feature information of inputs, we use the attention mechanism to enhance the feature of each block in the $\phi ( x )$ , which also helps to reduce irrelevant information.

$$
\phi (x) = [ \phi_ {1} (x), \phi_ {2} (x), \phi_ {3} (x), \phi_ {4} (x) ] \tag {11}
$$

Each block $\phi _ { i } ( x )$ is enhanced by an attention mechanism. Multi-Head Self-Attention(MHSA)(Vaswani et al. 2017) is used for the frst and last layer, and Coordinate Attention(CA)(Hou, Zhou, and Feng 2021) is used for the second and third. Using MHSA in the frst block can provide a strong foundation for later blocks to build upon, which is particularly useful when the input data is complex. This is the biggest characteristic of virtual classes. In the second and third blocks, the CA selectively attends to different spatial locations within the feature maps, which virtual classes lack. Finally, in the fourth block, the network typically reverts to using MHSA, which can further refne and integrate the feature learned in the previous blocks. After attention enhancing, the feature $\phi _ { i } ^ { \prime } ( x )$ , i denotes the i-th block.

$$
\phi_ {i} ^ {\prime} (x) = E N (\phi_ {i} (x)) = M H S A / C A (\phi_ {i} (x)) \tag {12}
$$

Feature Fusion: To align the above feature after attention enhancement and leverage multi-scale information of virtual class features, we use a BiFPN(Tan, Pang, and Le 2019) structure to fuse them.

$$
T _ {i} = F P N (\phi_ {i} ^ {\prime} (x)) \tag {13}
$$

$T _ { i }$ is attention-enhanced refned feature-map, i denotes the i-th block. Then, a temporary classifer $W ^ { ' }$ is used to predict the result.

$$
\hat {p} = W ^ {\prime} (T _ {\text { last }}) \tag {14}
$$

$\hat { p }$ is treated as a soft label.

Attention Transfer Loss: Finally, we introduce an Attention Transfer Loss(Zagoruyko and Komodakis 2016) to distillate the information between the original activation’s attention map and the multi-scale attention-enhanced activation’s attention map generated by $F E ( \cdot )$ .

$$
\mathcal {L} _ {\mathrm{AT}} = D _ {K L} \left(\hat {p}, f _ {\theta} (x)\right) + \sum_ {i = 1} ^ {N} \| \operatorname{at} \left(T _ {i} ^ {\prime}\right) - \operatorname{at} \left(\phi_ {i} (x)\right) \| _ {2} ^ {2} \tag {15}
$$

where at(·) is the attention map computed from the input feature. Note that the loss function consists of two terms: the frst is the KL divergence between model output $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } )$ and soft label pˆ from feature fusion. The second term measures the difference between the attention maps of Ti and $\phi _ { i } ^ { \mathrm { v i r t u a l } }$ . N is the number of attention maps. Finally, the loss function for our second stage is

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{ce}} ^ {\mathrm{v}} + \mathcal {L} _ {\mathrm{KL}} ^ {\mathrm{v}} + \mathcal {L} _ {\mathrm{AT}} \tag {16}
$$

# Classifer Updating Stage

This stage focuses on the incremental learning process. Similar to CEC, our method involves freezing the backbone network $\theta _ { \phi }$ after two stages of training in the base session, removing the $F E$ structure, and only updating the classifer parameter W during subsequent incremental sessions. We consider a total of $\bar { ( 1 + B ) }$ sessions, where 1 denotes the base session and B denotes the B incremental sessions. To represent each class in these datasets, we average the data over all the instances corresponding to that class. We update the classifer by appending these class features to the weight matrix, i.e., $\tilde { W _ { \mathrm { n e w } } } \stackrel {  } { = } [ W _ { \mathrm { n e w } } ; \boldsymbol { w } _ { i } , i \in Y ^ { b } ]$ ].

$$
\boldsymbol {w} _ {i} = \frac {1}{K} \sum_ {j = 1} ^ {| \mathcal {P} ^ {b} |} \mathbb {I} (y _ {j} = i) \phi_ {f _ {\theta}} (x _ {j}) \tag {17}
$$

$( x _ { j } , y _ { j } ) \in \mathcal { D } ^ { b } , \mathcal { D } ^ { b }$ is the training dataset $\mathcal { D } ^ { k } , k \in \mathbf { \Omega }$ $\{ 0 , 1 , . . . , B \}$ that contains instances belonging to class i. $\left| \dot { \mathcal { D } } ^ { b } \right|$ is the number of instances in the $\mathcal { D } ^ { b } . \mathbb { I } \left( y _ { j } = i \right)$ is an indicator function that is equal to 1 if the label of the j-th instance is i, and 0 otherwise. K is a normalization constant ensuring the prototype has the same scale as the representation vectors.

# Experiment

In this section, we evaluate our method performance on three mainstream benchmark datasets: Caltech-UCSD Birds-200- 2011 (CUB200) (Wah et al. 2011), CIFAR100 (Krizhevsky, Hinton et al. 2009), and miniImageNet (Russakovsky et al. 2015), the results are shown in Figure 3.

<table><tr><td>Metric</td><td>Method</td><td>Average value</td></tr><tr><td rowspan="2">Intra-class distance ↓</td><td>Baseline</td><td>1.35</td></tr><tr><td>M2SD</td><td>0.98(-27%)</td></tr><tr><td rowspan="2">Inter-class distance ↑</td><td>Baseline</td><td>6.49</td></tr><tr><td>M2SD</td><td>7.92(+22%)</td></tr></table>

Table 2: Based on feature vectors, our method reduces the average intra-class distance by 27% and increases the average inter-class distance by 22% compared to the baseline.

# Implementation Details

Dataset: Following the guidance of CEC, we divide each dataset into base and incremental sessions. For CUB200, the base session comprises 100 classes, and each of the 10 incremental sessions comprises 100 classes. Each incremental session consists of 10 classes, with 5 instances per class. For CIFAR100 and miniImageNet, the base session comprises 60 classes, and each of the 8 incremental sessions comprises 40 classes. Each incremental session consists of 5 classes, with 5 instances per class.

Training Details: Our implementation is based on Py-Torch, and the choice of backbone model is determined according to TOPIC(Tao et al. 2020): ResNet-20(He et al. 2016) for CIFAR100, ResNet-18 for CUB200 and miniImageNet. The same general data augmentation methods as other methods are used. We employ stochastic gradient descent (SGD) with momentum for optimization with a learning rate 1e-3 and a batch size of 256 on a 4xA100 GPU.

# Benchmark Comparison

We mainly compare knowledge distillation-based methods and other SOTA methods. Our results on the three datasets are presented in Figure 3, showing that our method outperforms the current SOTA methods. Specifcally, on the CUB200 our method achieves an average improvement of more than 2.0% over each stage compared to them. On the CIFAR100, the average improvement is more than 2.1%. On the miniImageNet, our method performs the best, which outperforms the SOTA method by more than 3.2% on average. Compared with the two SOTA methods based on knowledge distillation of SSFE-Net and CABD, our method is still very advantageous regarding accuracy across all sessions. Compared with them, our method improves by 7.21% and 8.94% per session on average.

# Numerical Analysis

The effectiveness and characteristics of our method are refected through the following numerical and visual analysis.

Visualization Increment: We used t-SNE(Van der Maaten and Hinton 2008) to visualize the change in the decision boundary on CIFAR100. Five classes were randomly selected from the base session, and another fve classes were randomly selected from the incremental sessions. Figure 4a is the incremental result of the baseline method. The baseline method uses cross-entropy loss to train the entire base session, and then the incremental session uses our same classifer update strategy. Figure 4b is our method, and it can be seen that the added classes are more naturally entered into the feature space than the baseline method. Compared with the baseline, our method reduces the average intra-class distance by 27% and increases the average inter-class distance by 22%. The intra-class distance is the average distance from the same class of data to the centroid of the class. The interclass distance is the minimum value between a class centroid and other class center centroids, shown in Table 2.

![](images/fa9e5a8d0cfa4189d8787a57ec0e5837def05918acd0ac96ad0f6d2ad895e187.jpg)

![](images/b336fc397315a4d14a121b4eb541bbebdb0e3b26076f627a14a01c71f45ef17d.jpg)

<details>
<summary>scatter</summary>

| x  | y  | Class   |
|----|----|---------|
| 5  | 30 | Class 0 |
| 10 | 15 | Class 1 |
| 15 | 10 | Class 2 |
| 20 | 25 | Class 3 |
| 25 | 20 | Class 4 |
| 30 | 15 | Class 5 |
| 35 | 10 | Class 6 |
| 40 | 5  | Class 7 |
| 45 | 0  | Class 8 |
| 50 | -5 | Class 9 |
</details>

(a)baseline

![](images/df66e76bb8acadc57523142fb4eeb3782da754a81f1345ee5bce98b1bd2acf95.jpg)

<details>
<summary>scatter</summary>

| x | y | Class   |
|---|---|---------|
| 0 | 5 | Class 0 |
| 0 | 6 | Class 0 |
| 0 | 7 | Class 0 |
| 1 | 10 | Class 1 |
| 1 | 12 | Class 1 |
| 1 | 15 | Class 1 |
| 2 | 18 | Class 2 |
| 2 | 19 | Class 2 |
| 2 | 20 | Class 2 |
| 3 | 25 | Class 3 |
| 3 | 28 | Class 3 |
| 3 | 30 | Class 3 |
| 4 | 30 | Class 4 |
| 4 | 32 | Class 4 |
| 4 | 35 | Class 4 |
| 5 | 35 | Class 4 |
| 5 | 38 | Class 4 |
| 5 | 40 | Class 4 |
| 6 | 40 | Class 4 |
| 6 | 42 | Class 4 |
| 6 | 45 | Class 4 |
| 7 | 45 | Class 4 |
| 7 | 48 | Class 4 |
| 7 | 50 | Class 4 |
| 8 | 50 | Class 4 |
| 8 | 52 | Class 4 |
| 8 | 55 | Class 4 |
| 9 | 55 | Class 4 |
| 9 | 58 | Class 4 |
| 9 | 60 | Class 4 |
| 10| 60 | Class 4 |
| 10| 62 | Class 4 |
| 10| 65 | Class 4 |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... | ... | ...     |
| ... = [0,1]    | ...       | ...     |
| ... = [1,2]    | ...       | ...     |
| ... = [2,3]    | ...       | ...     |
| ... = [3,4]    | ...       | ...     |
| ... = [4,5]    | ...       | ...     |
| ... = [5,6]    | ...       | ...     |
| ... = [6,7]    | ...       | ...     |
| ... = [7,8]    | ...       | ...     |
| ... = [8,9]    | ...       | ...     |
| ... = [9,10]   | ...       | ...     |
| ... = [10,11]   | ...       | ...     |
| ... = [11,12]   | ...       | ...     |
| ... = [12,13]   | ...       | ...     |
| ... = [13,14]   | ...       | ...     |
| ... = [14,15]   | ...       | ...     |
| ... = [15,16]   | ...       | ...     |
| ... = [16,17]   | ...       | ...     |
| ... = [17,18]   | ...       | ...     |
| ... = [18,19]   | ...       | ...     |
| ... = [19,20]   | ...       | ...     |
| ... = [20,21]   | ...       | ...     |
| ... = [21,22]   | ...       | ...     |
| ... = [22,23]   | ...       | ...     |
| ... = [23,24]   | ...       | ...     |
| ... = [24,25]   | ...       | ...     |
| ... = [25,26]   | ...       | ...     |
| ... = [26,27]   | ...       | ...     |
| ... = [27,28]   | ...       | ...     |
| ... = [28,29]   | ...       | ...     |
| ... = [29,30]   | ...       | ...     |
| ... = [30,31]   | ...       | ...     |
| ... = [31,32]   | ...       | ...     |
| ... = [32,33]   | ...       | ...     |
| ... = [33,34]   | ...       | ...     |
| ... = [34,35]   | ...       | ...     |
| ... = [35,36]   | ...       | ...     |
| ... = [36,37]   | ...       | ...     |
| ... = [37,38]   | ...       | ...     |
| ... = [38,39]   | ...       | ...     |
| ... = [39,40]   | ...       | ...     |
| ... = [40,41]   | ...       | ...     |
| ... = [41,42]   | ...       | ...     |
| ... = [42,43]   | ...       | ...     |
| ... = [43,44]   | ...       | ...     |
| ... = [44,45]   | ...       | ...     |
| ... = [45,46]   | ...       | ...     |
| ... = [46,47]   | ...       | ...     |
| ... = [47,48]   | ...       | ...     |
| ... = [48,49]   | ...       | ...     |
| ... = [49,50]   | ...       | ...     |
| ..= (10)      ..= (1)    ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;...= (1)      ;-          .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .         .        <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >           <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >             <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >            <          >              <           max>             |
The data is generated using a random seed for each class. The values are estimated based on the number of samples from a base session. The values are estimated based on the number of samples from a base session. The values are estimated based on the number of samples from a base session. The values are estimated based on the number of samples from a base session. The values are estimated based on the number of samples from a base session. The values are estimated based on the number of samples from a base session.
</details>

![](images/beef17796216876558537b7355011bd6fdec9786d2f71040ee40fdbd3df61206.jpg)

<details>
<summary>scatter</summary>

| x  | y  | Class   |
|----|----|---------|
| 0  | 35 | Class 0 |
| 1  | 30 | Class 1 |
| 2  | 25 | Class 2 |
| 3  | 20 | Class 3 |
| 4  | 15 | Class 4 |
| 5  | 10 | Class 5 |
| 6  | 5  | Class 6 |
| 7  | 0  | Class 7 |
| 8  | -5 | Class 8 |
| 9  | -10| Class 9 |
</details>

"(b) ours   
Figure 4: The distance coordinate scales of the two t-SNE are the same. Part (a) is the incremental result of the baseline method. The baseline method uses cross-entropy loss to train the entire base session, and then the incremental session uses our same classifer update strategy. Part (b) is the incremental result of our method.

Confusion Matrix: We plot the confusion matrix generated by training the model without(Figure 5a) and with our method(Figure 5b) on CIFAR100. As we can see, our diagonal is brighter, whether the base class is in the early stage or the incremental class in the later stage.

Nways-Kshots: We set the number of ways N and shots K in {1,5,10,15,20}. A total of 25 experiments are performed on the CUB200 and record the classifcation results of the last incremental session, shown in Figure 5c. It can be seen that the classifer update is insensitive to N for the same K. It is evident from the function of classifer update. When the same N , the larger K classifcation, the better the result, but the growth gradually decreases.

# Ablation Study

The ablation experiment was set up in four parts, as shown in Table 3. The experiments were conducted on the CUB200 with other confguration parameters unchanged.

Division of Base Session: The primary purpose of the experiments here is to demonstrate the necessity of the pretrain stage and its trade-off with our method in the base session.

<table><tr><td>ablation</td><td colspan="2">situation</td><td colspan="11">Accuracy in each session (%)</td></tr><tr><td rowspan="6">ratio</td><td>stage1</td><td>stage2</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>0</td><td>100%</td><td>81.19</td><td>75.59</td><td>71.48</td><td>65.82</td><td>64.28</td><td>60.38</td><td>59.35</td><td>57.33</td><td>55.63</td><td>53.89</td><td>52.37</td></tr><tr><td>50%</td><td>50%</td><td>81.59</td><td>77.19</td><td>74.08</td><td>69.05</td><td>68.90</td><td>65.78</td><td>65.01</td><td>64.20</td><td>62.13</td><td>61.83</td><td>60.73</td></tr><tr><td>80%</td><td>20%</td><td>81.87</td><td>77.46</td><td>74.25</td><td>69.00</td><td>68.97</td><td>66.03</td><td>64.90</td><td>63.81</td><td>62.54</td><td>61.81</td><td>60.63</td></tr><tr><td>100%</td><td>0</td><td>73.38</td><td>66.90</td><td>63.55</td><td>58.15</td><td>57.56</td><td>55.47</td><td>54.40</td><td>52.51</td><td>50.43</td><td>49.56</td><td>48.35</td></tr><tr><td>20%</td><td>80%</td><td>81.49</td><td>76.67</td><td>73.58</td><td>68.77</td><td>68.73</td><td>65.78</td><td>64.73</td><td>64.03</td><td>62.70</td><td>62.09</td><td>60.96</td></tr><tr><td rowspan="6">method</td><td>virtual1</td><td>virtual2</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>mixup</td><td>-</td><td>81.30</td><td>76.10</td><td>73.30</td><td>68.03</td><td>68.44</td><td>65.30</td><td>63.87</td><td>63.34</td><td>61.50</td><td>61.19</td><td>60.19</td></tr><tr><td>CutMix</td><td>-</td><td>81.19</td><td>75.59</td><td>71.48</td><td>65.82</td><td>64.28</td><td>60.38</td><td>59.35</td><td>57.33</td><td>55.63</td><td>53.89</td><td>52.37</td></tr><tr><td>mixup</td><td>mixup</td><td>81.66</td><td>76.69</td><td>74.01</td><td>69.34</td><td>68.25</td><td>65.19</td><td>63.85</td><td>62.51</td><td>61.64</td><td>60.56</td><td>59.49</td></tr><tr><td>CutMix</td><td>CutMix</td><td>81.21</td><td>76.35</td><td>73.41</td><td>68.68</td><td>68.08</td><td>65.10</td><td>64.15</td><td>63.33</td><td>61.65</td><td>60.94</td><td>59.88</td></tr><tr><td>mixup</td><td>CutMix</td><td>81.49</td><td>76.67</td><td>73.58</td><td>68.77</td><td>68.73</td><td>65.78</td><td>64.73</td><td>64.03</td><td>62.70</td><td>62.09</td><td>60.96</td></tr><tr><td rowspan="4">FE</td><td>FE</td><td>dual</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>✓</td><td></td><td>80.88</td><td>75.78</td><td>72.19</td><td>67.55</td><td>67.32</td><td>64.22</td><td>63.51</td><td>62.73</td><td>61.81</td><td>60.87</td><td>60.02</td></tr><tr><td></td><td>✓</td><td>81.25</td><td>76.25</td><td>72.92</td><td>68.51</td><td>68.12</td><td>64.78</td><td>63.68</td><td>62.93</td><td>61.70</td><td>60.85</td><td>59.84</td></tr><tr><td>✓</td><td>✓</td><td>81.49</td><td>76.67</td><td>73.58</td><td>68.77</td><td>68.73</td><td>65.78</td><td>64.73</td><td>64.03</td><td>62.70</td><td>62.09</td><td>60.96</td></tr><tr><td rowspan="3">Classifier</td><td colspan="2">updating</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td></td><td></td><td>81.49</td><td>72.81</td><td>66.57</td><td>61.32</td><td>56.83</td><td>53.00</td><td>49.57</td><td>46.60</td><td>43.96</td><td>42.36</td><td>40.17</td></tr><tr><td>✓</td><td></td><td>81.49</td><td>76.67</td><td>73.58</td><td>68.77</td><td>68.73</td><td>65.78</td><td>64.73</td><td>64.03</td><td>62.70</td><td>62.09</td><td>60.96</td></tr></table>

Table 3: Ablation study on CUB200. The frst column shows the different methods used. The second column provides specifc details about the settings used for each method. The last row of each sub-table indicates the best-performing implementation selected in this study.

![](images/46314838b0c2b9d901e9fe2e85b0f6997f2bd5750d4c83a5a786c1ba51ab5a7c.jpg)  
Figure 5: Part(a) and Part(b) shows that the effectiveness and accuracy of our method are improved. Part(c) primarily shows the impact of classifer updating, which is not sensitive to the incremental increase in the number of classes.

We have set the ratio of pre-training and our method based on epochs. As shown in the frst part of Table 3, it is clear that our method directly leads to the worst results without pre-training the model. The average performance is 0.5% lower in the base session and 8% lower in the fnal session. It can also be seen that the larger the ratio, the better the performance in the early stages, while the later performance gets worse. To average the overall performance, we choose the ratio of 20%:80%.

Multiple Mixing Self-Distillation: As shown in the frst part of Table 3, when stage1 ratio is 100%, it is equivalent

to not using our self-distillation method. Comparing with our fnal results, we can see that our method has an average increase of 10% per session. It can refect the effectiveness of our self-distillation method.

Multi Branch Virtual Classes Mixing Distillation: As shown in the second part of Table 3, the ablation experiments for this aimed to investigate the necessity of dual branches and the choice of virtual class construction method for each branch. Using mixup alone to construct virtual classes yields good results. However, when using CutMix alone, the last session decreased by 8.4% compared to our method. The improvement in the base session through virtual classes constructed by CutMix is similar to the mixup. However, Cut-Mix drops much faster in the incremental sessions than in the mixup. It shows that the feature space obtained by Cut-Mix training is unsuitable for FSCIL. However, combining CutMix and mixup with distillation can further improve the gain of virtual class features brought by mixup to FSCIL while avoiding the negative impact of CutMix.

We also compared dual branches using the same virtual class construction method. The experimental involved sampling different lambda values and randomly selecting two different classes. It was found that no other combination is better than ours.

Self-Distillaton with Attention Enhancement: The results in the third part of Table 3 showed that the dualbranch the virtual class structure provided an average boost of 1.15% per session, while the feature enhancement structure provided an average boost of 0.79% per session. The attention enhancement module demonstrated the most signifcant improvement between the two experiments.

Classifer Learning: In the fourth part of the Table 3, we can see that if we do not empty the original W parameters directly into the incremental session classifcation, there is a signifcant impact on the subsequent classifcation effect. From session 1 lags behind 3.86%, the lagging magnitude increases until the last session lags behind 20.79%, with an average drop of 13.49% per session. It can be seen that the use of classifer updates is essential.

# Conclusion

To solve the FSCIL problem, we frst employ dual-branch virtual class distillation to expand the feature space. This expansion enables it to accommodate both current and future classes, which we have verifed through numerical and visual methods. Furthermore, we employ feature enhancement and self-distillation to fully utilize the virtual class feature and enhance the compatibility of the feature space. These methods allow us to obtain a feature space suitable for the FSCIL. Experimental results on three datasets demonstrate that our approach leads to signifcant session accuracy gains, indicating the effectiveness of our M2SD.

# Acknowledgments

This work was supported by Alibaba Group through Alibaba Research Intern Program. We thank the support from the Alibaba-South China University of Technology Joint Graduate Education Program. This work was also partially supported by Guangdong Artifcial Intelligence and Digital Economy Laboratory (Guangzhou).

# References

Akyurek, A. F.; Aky ¨ urek, E.; Wijaya, D. T.; and Andreas, J. ¨ 2021. Subspace regularizers for few-shot class incremental learning. arXiv preprint arXiv:2110.07059.   
Ayub, A.; and Wagner, A. R. 2020. Cognitively-inspired model for incremental learning using a few examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 222–223.   
Chen, W.-Y.; Liu, Y.-C.; Kira, Z.; Wang, Y.-C. F.; and Huang, J.-B. 2019. A closer look at few-shot classifcation. arXiv preprint arXiv:1904.04232.   
Chi, Z.; Gu, L.; Liu, H.; Wang, Y.; Yu, Y.; and Tang, J. 2022. MetaFSCIL: a meta-learning approach for few-shot class incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 14166–14175.   
Gidaris, S.; and Komodakis, N. 2018. Dynamic few-shot visual learning without forgetting. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4367–4375.   
He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778.

Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.   
Hou, Q.; Zhou, D.; and Feng, J. 2021. Coordinate Attention for Effcient Mobile Network Design. Cornell University - arXiv.   
Hou, S.; Pan, X.; Loy, C. C.; Wang, Z.; and Lin, D. 2019. Learning a unifed classifer incrementally via rebalancing. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, 831–839.   
Ji, M.; Shin, S.; Hwang, S.; Park, G.; and Moon, I.-C. 2021. Refne myself by teaching myself: Feature refnement via self-knowledge distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10664–10673.   
Kim, D.-Y.; Han, D.-J.; Seo, J.; and Moon, J. 2022. Warping the space: Weight space rotation for class-incremental fewshot learning. In The Eleventh International Conference on Learning Representations.   
Kim, J.-Y.; and Choi, D.-W. 2021. Split-and-bridge: Adaptable class incremental learning within a single neural network. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 35, 8137–8145.   
Kirkpatrick, J.; Pascanu, R.; Rabinowitz, N.; Veness, J.; Desjardins, G.; Rusu, A. A.; Milan, K.; Quan, J.; Ramalho, T.; Grabska-Barwinska, A.; et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13): 3521–3526.   
Krizhevsky, A.; Hinton, G.; et al. 2009. Learning multiple layers of features from tiny images.   
Krizhevsky, A.; Sutskever, I.; and Hinton, G. E. 2017. Imagenet classifcation with deep convolutional neural networks. Communications of the ACM, 60(6): 84–90.   
Kullback, S.; and Leibler, R. A. 1951. On information and suffciency. The annals of mathematical statistics, 22(1): 79–86.   
Lee, H.; Hwang, S. J.; and Shin, J. 2020. Self-supervised label augmentation via input transformations. In International Conference on Machine Learning, 5714–5724. PMLR.   
Li, A.; Luo, T.; Xiang, T.; Huang, W.; and Wang, L. 2019. Few-shot learning with global class representations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 9715–9724.   
Li, Z.; and Hoiem, D. 2017. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12): 2935–2947.   
Liu, H.; Gu, L.; Chi, Z.; Wang, Y.; Yu, Y.; Chen, J.; and Tang, J. 2022. Few-Shot Class-Incremental Learning via Entropy-Regularized Data-Free Replay.   
Lopez-Paz, D.; and Ranzato, M. 2017. Gradient episodic memory for continual learning. Advances in neural information processing systems, 30.   
Pan, Z.; Yu, X.; Zhang, M.; and Gao, Y. 2023. SSFE-Net: Self-Supervised Feature Enhancement for Ultra-Fine-Grained Few-Shot Class Incremental Learning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 6275–6284.

Peng, C.; Zhao, K.; Wang, T.; Li, M.; and Lovell, B. C. 2022. Few-Shot Class-Incremental Learning from an Open-Set Perspective. Springer, Cham.   
Rebuff, S.-A.; Kolesnikov, A.; Sperl, G.; and Lampert, C. H. 2017. icarl: Incremental classifer and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2001–2010.   
Ren, M.; Liao, R.; Fetaya, E.; and Zemel, R. 2019. Incremental few-shot learning with attention attractor networks. Advances in neural information processing systems, 32.   
Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster R-CNN: Towards real-time object detection with region proposal networks. Advances in neural information processing systems, 28.   
Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; et al. 2015. Imagenet large scale visual recognition challenge. International journal of computer vision, 115: 211– 252.   
Rusu, A. A.; Rabinowitz, N. C.; Desjardins, G.; Soyer, H.; Kirkpatrick, J.; Kavukcuoglu, K.; Pascanu, R.; and Hadsell, R. 2016. Progressive neural networks. arXiv preprint arXiv:1606.04671.   
Shi, G.; Chen, J.; Zhang, W.; Zhan, L.-M.; and Wu, X.-M. 2021. Overcoming catastrophic forgetting in incremental few-shot learning by fnding fat minima. Advances in Neural Information Processing Systems, 34: 6747–6761.   
Simonyan, K.; and Zisserman, A. 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.   
Snell, J.; Swersky, K.; and Zemel, R. 2017a. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30.   
Snell, J.; Swersky, K.; and Zemel, R. 2017b. Prototypical networks for few-shot learning. Advances in neural information processing systems, 30.   
Song, Z.; Zhao, Y.; Shi, Y.; Peng, P.; Yuan, L.; and Tian, Y. 2023. Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 24183–24192.   
Tan, M.; Pang, R.; and Le, Q. V. 2019. EffcientDet: Scalable and Effcient Object Detection. Cornell University - arXiv.   
Tao, X.; Hong, X.; Chang, X.; Dong, S.; Wei, X.; and Gong, Y. 2020. Few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12183–12192.   
Tian, Y.; Wang, Y.; Krishnan, D.; Tenenbaum, J. B.; and Isola, P. 2020. Rethinking few-shot image classifcation: a good embedding is all you need? In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23– 28, 2020, Proceedings, Part XIV 16, 266–282. Springer.   
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. Journal of machine learning research, 9(11).

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, L.; and Polosukhin, I. 2017. Attention is All you Need. Neural Information Processing Systems.   
Verma, V.; Lamb, A.; Beckham, C.; Najaf, A.; Mitliagkas, I.; Lopez-Paz, D.; and Bengio, Y. 2019. Manifold mixup: Better representations by interpolating hidden states. In International conference on machine learning, 6438–6447. PMLR.   
Wah, C.; Branson, S.; Welinder, P.; Perona, P.; and Belongie, S. 2011. The caltech-ucsd birds-200-2011 dataset.   
Wu, Y.; Chen, Y.; Wang, L.; Ye, Y.; Liu, Z.; Guo, Y.; and Fu, Y. 2019. Large scale incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 374–382.   
Xu, T.-B.; and Liu, C.-L. 2019. Data-distortion guided selfdistillation for deep neural networks. In Proceedings of the AAAI Conference on Artifcial Intelligence, volume 33, 5565–5572.   
Yun, S.; Han, D.; Oh, S. J.; Chun, S.; Choe, J.; and Yoo, Y. 2019. CutMix: Regularization Strategy to Train Strong Classifers with Localizable Features. Cornell University - arXiv.   
Zagoruyko, S.; and Komodakis, N. 2016. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928.   
Zhang, C.; Song, N.; Lin, G.; Zheng, Y.; Pan, P.; and Xu, Y. 2021. Few-shot incremental learning with continually evolved classifers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 12455– 12464.   
Zhang, H.; Cisse, M.; Dauphin, Y. N.; and Lopez-Paz, D. 2017. mixup: Beyond Empirical Risk Minimization. Cornell University - arXiv.   
Zhang, J.; Gao, L.; Hao, B.; Huang, H.; Song, J.; and Shen, H. 2023a. From Global to Local: Multi-scale Outof-distribution Detection. IEEE Transactions on Image Processing.   
Zhang, J.; Gao, L.; Luo, X.; Shen, H.; and Song, J. 2023b. DETA: Denoised Task Adaptation for Few-Shot Learning. arXiv preprint arXiv:2303.06315.   
Zhang, L.; Song, J.; Gao, A.; Chen, J.; Bao, C.; and Ma, K. 2019. Be your own teacher: Improve the performance of convolutional neural networks via self distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 3713–3722.   
Zhao, L.; Lu, J.; Xu, Y.; Cheng, Z.; Guo, D.; Niu, Y.; and Fang, X. 2023. Few-Shot Class-Incremental Learning via Class-Aware Bilateral Distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 11838–11847.   
Zhou, D.-W.; Wang, F.-Y.; Ye, H.-J.; Ma, L.; Pu, S.; and Zhan, D.-C. 2022. Forward compatible few-shot classincremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9046– 9056.

Zhu, F.; Cheng, Z.; Zhang, X.-Y.; and Liu, C.-l. 2021a. Class-incremental learning via dual augmentation. Advances in Neural Information Processing Systems, 34: 14306–14318.   
Zhu, K.; Cao, Y.; Zhai, W.; Cheng, J.; and Zha, Z.-J. 2021b. Self-promoted prototype refnement for few-shot class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 6801–6810.   
Zhuang, H.; Weng, Z.; He, R.; Lin, Z.; and Zeng, Z. 2023. GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 7746–7755.