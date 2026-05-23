# Technical Report for CVPR 2024 5th CLVISION Challenge: Multi-Level Knowledge Distillation and Dynamic Self-Supervised Learning for Continual Learning.

Taeheon Kim1,2 San Kim1,2 Minhyuk Seo1,2 Dongjae Jeon1,2 Wonje Jeung1,2 Jonghyun Choi2

1Yonsei University 2Seoul National University

{thkim0305,nasmik419,dbd0508,dongjae0324,specific0924}@yonsei.ac.kr

jonghyunchoi@snu.ac.kr

# Abstract

Class-incremental with repetition (CIR), where previously trained classes repeatedly introduced in future tasks, is a more realistic scenario than the traditional class incremental setup, which assumes that each task contains unseen classes. CIR assumes that we can easily access abundant unlabeled data from external sources, such as the Internet. Therefore, we propose two components that efficiently use the unlabeled data to ensure the high stability and the plasticity of models trained in CIR setup. First, we introduce multi-level knowledge distillation (MLKD) that distills knowledge from multiple previous models across multiple perspectives, including features and logits, so the model can maintain much various previous knowledge. Moreover, we implement dynamic self-supervised loss (SSL) to utilize the unlabeled data that accelerates the learning of new classes, while dynamic weighting of SSL keeps the focus of training to the primary task. Both of our proposed components significantly improve the performance in CIR setup, achieving 2nd place in the CVPR 5th CLVISION Challenge1.

# 1. Introduction

Class incremental learning is a setup where the model learns new classes while preventing forgetting of the previously seen classes. Traditionally, CIL setups assume that new classes are introduced with each new task and that a fixed number of new classes are added per task. However, in a real-world scenario, a model may encounter both new and previously seen classes. To bridge the gap between these scenarios and the existing CIL setup, Hemati et al. [12] proposed Class-Incremental Learning with Repetition (CIR), a more flexible framework where the model encounters a mixture of old and new classes. This setup aligns with concepts from other learning paradigms, such as the non-IID assumption in federated learning[14] and the blurry task boundaries in online continual learning [16], challenging the applicability of methods that rely on stricter assumptions.

Rehearsal-based methods [3, 23, 24, 26] are among the most widely studied approaches in CIL and CIR due to their effectiveness in mitigating forgetting of previous tasks, i.e., catastrophic forgetting [20]. These methods typically store a small subset [23] or even the full dataset [22] of previous task exemplars in a memory buffer and replay them during training with new data. However, practical constraints such as data privacy concerns [27] and storage limitations can make rehearsal-based methods infeasible in real-world scenarios. This leads to the development of rehearsal-free continual learning approaches [8, 18, 25] that aim to maintain model performance without relying on stored exemplars.

A promising variant within the rehearsal-free category involves leveraging open-source external data that can be easily accessed online. These data can be downloaded, utilized during training, and discarded afterwards, effectively circumventing the challenges related to storing sensitive or large amounts of data. Such an approach offers greater flexibility in model training and presents a valuable opportunity to enhance both the stability and plasticity of models in CIR. Therefore, developing strategies that maximize the benefits of open-source data becomes crucial in advancing the effectiveness of continual learning systems.

In this paper, we propose two components that utilize the external unlabeled data to effectively promote the stability and plasticity of the model. To encourage stability, we introduce multi-level knowledge distillation (MLKD), which extracts learned knowledge from multiple previous models from various perspectives (e.g., features, logits) to compensate for the data distribution shift from the training data to the external unlabeled data. To enhance plasticity and facilitate faster learning, we use the unlabeled dataset to help the model extracting general features from the input, enabling it to identify relevant features from unseen classes and aiding in the classification of new classes. Thus, we apply dynamic self-supervised loss (SSL) in the continual training of the model.

Our multi-level knowledge distillation and dynamic selfsupervised loss, built upon the baseline of local crossentropy and feature replay, show significant performance improvements in the image classification task on the ImageNet-1K [7] subset. Our method effectively utilizes unlabeled data to preserve knowledge, achieving 2nd place in the CVPR 2024 5th CLVISION Challenge.

# 2. Methodology

We define the problem statement of class-incremental learning with repetition (CIR) utilizing an unlabeled data. We then explain how we address the plasticity and stability of the model through multi-level knowledge distillation (MLKD) and dynamic self-supervised learning (SSL). The overview of our method is illustrated in Figure 1.

# 2.1. Problem Statement

Let $\mathcal { D } _ { l } ~ = ~ \{ ( x _ { i } , y _ { i } ) ~ \vert ~ x _ { i } ~ \in ~ X _ { l } , y _ { i } ~ \in ~ Y _ { l } \} _ { i = 1 } ^ { N }$ denote the labeled dataset, which consists of N samples, each represented by an image-label pair $( x _ { i } , y _ { i } )$ from an image set $X _ { l }$ and a label space $Y _ { l } = \{ 1 , 2 , . . . , C \}$ , and let $\mathcal { D } _ { u } = \{ x _ { i } ^ { \prime } \mid $ $x _ { i } ^ { \prime } \in X _ { u } \} _ { i = 1 } ^ { N ^ { \prime } }$ denote the unlabeled dataset with $N ^ { \prime }$ samples $x _ { i } ^ { \prime } .$ . There exist a small memory buffer M that can store relevant information about the previously learned tasks.

The objective of CIR is to maximize final accuracy in the image classification task on the test set with the data from the label space $Y _ { l } .$ .

In this paper, we denote the feature as the output of the feature extractor part of the model (i.e. ResNet encoder), and the logit as the output of the prediction head of the model.

# 2.2. Multi-Level Knowledge Distillation (MLKD)

To alleviate catastrophic forgetting in CIR, it is crucial to distill knowledge from previous models. Extending from common knowledge distillation (KD) in continual learning [16, 17], our MLKD involves distilling knowledge from multiple perspectives across multiple previous models, utilizing both feature-level KD and logit-level KD.

For feature KD, we apply L2 loss to distill features from previous models. This helps ensure that the representations learned by the model remain consistent over time, preserving the knowledge embedded in these features. We linearly increase the weight of feature KD loss over tasks, as features from early stages may not contain much useful information and could hinder the learning of new knowledge.

We also apply KD using logits from previous models. However, naively distilling by instance-wise comparison is harmful, as the logits produced by the unlabeled data do not accurately reflect the knowledge of trained data due to data distribution discrepancies. To account for this distribution shift, we introduce a sophisticated KD mechanism, using Gram matrices to capture the correlations within batches and across classes [13], as follows:

$$
\begin{array}{l} L _ {\text {LogitKD}} = \frac {1}{B} \sum_ {k = 1} ^ {K} \left| \left| G _ {\text {curr}} ^ {k} - G _ {\text {prev}} ^ {k} \right| \right| ^ {2} \tag {1} \\ + \frac {1}{C} \sum_ {k = 1} ^ {K} | | M _ {c u r r} ^ {k} - M _ {p r e v} ^ {k} | | ^ {2}. \\ \end{array}
$$

The first term distills the correlation of instances within the batch, and the second term distills the correlation of classes. $G$ and M are Gram matrices, where for a logit batch $l \in$ $\mathbb { R } ^ { B \times C }$ with batch size B and the number of classes $C , G =$ $l \cdot l ^ { T }$ and $M = l ^ { T } \cdot l$ .

To extract a richer set of knowledge, we keep multiple previous models and mix their outputs for distillation. We maintain the trained model at the end of each task, and we select features and logits based on the confidence of the outputs. This multi-model approach enhances the robustness of knowledge transfer, compensating for the data distribution shift. Moreover, we gradually update all previous models through exponential moving average (EMA) with the current model, reflecting new knowledge in the distillation process and promoting the model’s plasticity.

# 2.3. Dynamic Self-Supervised Learning

In addition to knowledge distillation, we incorporate selfsupervised loss using the unlabeled dataset to foster the model’s plasticity, enabling it to adapt to new concepts effectively. Self-supervised learning (SSL) encourages the model to extract general features that are useful across various tasks. This generalization capability is crucial for learning new concepts without significantly altering existing knowledge.

However, a naive application of SSL can degrade the performance of the primary task (e.g., image classification), as SSL does not directly align with the target task objectives. To balance the influence of SSL on the model’s learning process, we apply dynamic weighting to SSL, as follows:

$$
L (X _ {l}, X _ {u}) = (1 - 0. 1 \cdot \alpha) L _ {A C E} (X _ {l}) + \alpha L _ {S S L} (X _ {u}), \tag {2}
$$

where $\alpha = c \omega ^ { t }$ , where c and ω are hyperparameters, and t is a task number. ω is set to less than 1, causing α to decreases over the tasks. This ensures that the model remains focused on the primary task $( i . e .$ , image classification) while still benefiting from the generalization effects of SSL.

![](images/26503d156a5f7d5ac44f51bb8a650d9897dfa50add564f45fb758c9187626613.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image"] --> B["Feature Extractor"]
    B --> C["Feature"]
    C --> D["FC"]
    D --> E["Logit"]
    F["Previous Feature Extractors"] --> G["FC"]
    G --> H["FC"]
    H --> I["Logit"]
    J["EMA"] -.-> B
```
</details>

![](images/3d9d1a851fc8ab9e53bcb93abbde6179204eb180f2e9279ef28e632adb72e15f.jpg)  
Figure 1. Overview of our method.

# 2.4. Final Loss

We build our two components on a baseline commonly used in continual learning. Our final loss is as follows:

$$
\begin{array}{l} L _ {f i n a l} (X _ {l}, X _ {u}, M) = \\ (1 - 0. 1 \cdot \alpha) L _ {A C E} (X _ {l}, M) \\ + \alpha L _ {S S L} (X _ {u}) \\ + \gamma L _ {L C} (X _ {l}) + \eta L _ {d e r} (M) \\ + \beta L _ {f e a t u r e K D} (X _ {u}) \\ + \delta L _ {l o g i t K D} (X _ {u}), \\ \end{array}
$$

where $L _ { A C E }$ is an asynchronous cross-entropy loss [4] that compute loss only on classes in the batch, $L _ { L C }$ is a logit constraint loss introduced in XDER [2] to help discriminate the classes from different tasks, and we store features and logits of samples into a memory buffer M for replay [3].

# 3. Experiments

# 3.1. Challenge Settings

The challenge assumes a continual learning setup with a stream of 50 experiences. The dataset is composed of a subset of ImageNet-1K [7], containing 130 classes related to birds, sea animals, and bugs. The labeled data stream contains data from 100 classes. For each experience introduced sequentially, 500 labeled data samples and 1000 unlabeled data samples are provided. The unlabeled data includes: 1) images of the same classes as the labeled data in the same experience, 2) images from within the 100 classes of the labeled data stream, and 3) random images from the whole dataset, depending on the challenge scenarios. The model architecture is ResNet-18 [11], with constraints of 8,000 MB GPU memory and a time limit of 600 minutes.

<table><tr><td>Method</td><td>Final Acc (%)</td></tr><tr><td>Fine-tuning</td><td>5.82</td></tr><tr><td>Baseline</td><td>19.54</td></tr><tr><td>Baseline + Dynamic SSL</td><td>23.82</td></tr><tr><td>Baseline + MLKD</td><td>39.82</td></tr><tr><td>Baseline + Dynamic SSL + MLKD (Ours)</td><td>42.00</td></tr></table>

Table 1. Experiment results.

The memory buffer is limited to storing only 200 exemplars, each of size less than 1,024 floating points.

# 3.2. Implementation Details

We implemented our method using PyTorch [21] and Avalanche [19] libraries. We used Adam optimizer [15] with a learning rate of 4e-4. The batch size for labeled and unlabeled data was set to 64 and 100, respectively. We chose rotation prediction [10] as the self-supervised loss. The parameters were set as follows: $c = 0 . 5 , \omega = 0 . 9 5$ , γ = 0.1, η = 0.4, β = 0.002t, and $\delta = 0 . 1$ . For inference, we utilized model ensemble, averaging the predictions of the current model and one previous model. All experiments were conducted on a single RTX 2080Ti GPU.

# 3.3. Experiment Results

The experiment results are presented in Table 1, showing the performance of different methods on the final accuracy. Our proposed method, combining multi-level knowledge distillation (MLKD) and dynamic self-supervised learning (SSL), achieves the highest accuracy of 42.00%. The naive fine-tuning (FT) method results in the lowest accuracy at 5.82%, indicating significant catastrophic forgetting. The baseline method improves the accuracy to 19.54%, while adding dynamic SSL to the baseline further increases it to 23.82%. Incorporating MLKD significantly boosts the accuracy to 39.82%, demonstrating the effectiveness of our approach in preserving knowledge. Finally, our full method, which combines both dynamic SSL and MLKD, achieves the best performance.

<table><tr><td>FKD</td><td>EMA</td><td>CLKD</td><td>MPM</td><td>Final Acc (%)</td></tr><tr><td>√</td><td></td><td></td><td></td><td>39.40</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>40.84</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>41.08</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>42.00</td></tr></table>

Table 2. Ablation study of MLKD. FKD, EMA, CLKD, and MPM refer to Feature KD, Exponential Moving Average, Correlationbased Logit KD, and Multiple Previous Models, respectively.

<table><tr><td>Number of previous models</td><td>Final Acc (%)</td></tr><tr><td>1 previous model</td><td>40.34</td></tr><tr><td>2 previous models</td><td>41.56</td></tr><tr><td>3 previous models</td><td>42.00</td></tr><tr><td>4 previous models</td><td>41.84</td></tr></table>

Table 3. The number of previous models used in MLKD.

The detailed ablation study of the components of MLKD is shown in Table 2. Each component’s contribution is analyzed by progressively adding them to the feature knowledge distillation (FKD) baseline. When only FKD is applied, the accuracy is 39.40%. Adding exponential moving average (EMA) to FKD increases the accuracy to 40.84%. Including correlation-based logit KD (CLKD) further improves the accuracy to 41.08%. Finally, combining all components, including multiple previous models (MPM), achieves the highest accuracy of 42.00%, confirming the synergistic effect of these components.

Table 3 explores the impact of the number of previous models used in MLKD. More previous models lead to better performance, but require more training time due to more number of model forwarding and EMA updating. Using only one previous model results in an accuracy of 40.34%. As we increase the number of previous models to two and three, the accuracy improves to 41.56% and 42.00%, respectively. However, adding a fourth previous model slightly decreases the accuracy to 41.84%, suggesting that three previous models provide the optimal balance between retaining useful knowledge and avoiding redundancy with reasonable training time.

Table 4 compares the performance with and without dynamic weighting applied to the self-supervised loss. Without dynamic weighting, the model achieves an accuracy of 40.76%. When dynamic weighting is applied, the accu-

<table><tr><td>Method</td><td>Final Acc (%)</td></tr><tr><td>Without dynamic weighting</td><td>40.76</td></tr><tr><td>With dynamic weighting</td><td>42.00</td></tr></table>

Table 4. The performance difference with and without dynamic weighting to self-supervised loss.

<table><tr><td>SSL Method</td><td>Final Acc (%)</td></tr><tr><td>Rotation Prediction [10]</td><td>42.00</td></tr><tr><td>SimCLR [5]</td><td>39.10</td></tr><tr><td>SimSiam [6]</td><td>39.46</td></tr><tr><td>VICReg [1]</td><td>36.18</td></tr><tr><td>VICReg-ctr [9]</td><td>37.26</td></tr></table>

Table 5. The performance difference with the choice of selfsupervised loss.

racy improves to 42.00%, demonstrating that dynamically adjusting the influence of SSL helps maintain the model’s focus on the primary task while still benefiting from the generalization effects of SSL.

We also experimented with various self-supervised losses other than rotation prediction, as presented in Table 5. Other recent state-of-the-art SSL methods perform worse than simple rotation prediction. We assume this is because these methods require a large batch size and a long training time to be effective. Thus, complying with the challenge‘s restriction leads to sub-optimal performance when using these contrastive-based SSL methods.

# 4. Conclusion

We propose multi-level knowledge distillation and adaptive self-supervised loss to promote stability and plasticity to utilize external unlabeled data instead of storing labeled data that the risk of privacy issues exist. These two approaches that are for maintaining the previous knowledge in the previously trained models and for accelerating the learning of new classes improve the existing continual learning baseline. Various ablation studies are conducted to demonstrate the effectiveness of our method in class incremental with repetition, and we rank 2nd place on CVPR 5th CLVI-SION Challenge.

# 5. Acknowledgment

We acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputers MareNostrum5 at BSC, Spain; LEONARDO at CINECA, Italy; VEGA at IZUM, Slovenia; Karolina at IT4Innovations, Czech Republic; MeluXina at LuxProvide, Luxembourg; Discoverer at Sofia Tech Park, Bulgaria; and Deucalion at Minho Advanced Computing Centre, Portugal, under project IDs EHPC-DEV-2025D08-065 and EHPC-DEV-2025D08-088, through EuroHPC Development Access calls.

# References

[1] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for selfsupervised learning. arXiv preprint arXiv:2105.04906, 2021. 4   
[2] Matteo Boschini, Lorenzo Bonicelli, Pietro Buzzega, Angelo Porrello, and Simone Calderara. Class-incremental continual learning into the extended der-verse. IEEE transactions on pattern analysis and machine intelligence, 45(5):5497–5512, 2022. 3   
[3] Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. Advances in neural information processing systems, 33:15920–15930, 2020. 1, 3   
[4] Lucas Caccia, Rahaf Aljundi, Nader Asadi, Tinne Tuytelaars, Joelle Pineau, and Eugene Belilovsky. New insights on reducing abrupt representation change in online continual learning. arXiv preprint arXiv:2104.05025, 2021. 3   
[5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR, 2020. 4   
[6] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758, 2021. 4   
[7] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 2, 3   
[8] Qiankun Gao, Chen Zhao, Bernard Ghanem, and Jian Zhang. R-dfcil: Relation-guided representation learning for datafree class incremental learning. In European Conference on Computer Vision, pages 423–439. Springer, 2022. 1   
[9] Quentin Garrido, Yubei Chen, Adrien Bardes, Laurent Najman, and Yann Lecun. On the duality between contrastive and non-contrastive self-supervised learning. arXiv preprint arXiv:2206.02574, 2022. 4   
[10] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018. 3, 4   
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 3   
[12] Hamed Hemati, Andrea Cossu, Antonio Carta, Julio Hurtado, Lorenzo Pellegrini, Davide Bacciu, Vincenzo Lomonaco, and Damian Borth. Class-incremental learning with repetition. In Conference on Lifelong Learning Agents, pages 437–455. PMLR, 2023. 1   
[13] Ying Jin, Jiaqi Wang, and Dahua Lin. Multi-level logit distillation. In Proceedings of the IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition, pages 24276–24285, 2023. 2   
[14] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International conference on machine learning, pages 5132–5143. PMLR, 2020. 1   
[15] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 3   
[16] Hyunseo Koh, Minhyuk Seo, Jihwan Bang, Hwanjun Song, Deokki Hong, Seulki Park, Jung-Woo Ha, and Jonghyun Choi. Online boundary-free continual learning by scheduled data prior. In The Eleventh International Conference on Learning Representations, 2022. 1, 2   
[17] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017. 2   
[18] Huan Liu, Li Gu, Zhixiang Chi, Yang Wang, Yuanhao Yu, Jun Chen, and Jin Tang. Few-shot class-incremental learning via entropy-regularized data-free replay. In European Conference on Computer Vision, pages 146–162. Springer, 2022. 1   
[19] Vincenzo Lomonaco, Lorenzo Pellegrini, Andrea Cossu, Antonio Carta, Gabriele Graffieti, Tyler L. Hayes, Matthias De Lange, Marc Masana, Jary Pomponi, Gido van de Ven, Martin Mundt, Qi She, Keiland Cooper, Jeremy Forest, Eden Belouadah, Simone Calderara, German I. Parisi, Fabio Cuzzolin, Andreas Tolias, Simone Scardapane, Luca Antiga, Subutai Amhad, Adrian Popescu, Christopher Kanan, Joost van de Weijer, Tinne Tuytelaars, Davide Bacciu, and Davide Maltoni. Avalanche: an end-to-end library for continual learning. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 2021. 3   
[20] Michael McCloskey and Neal J Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology of learning and motivation, pages 109–165. Elsevier, 1989. 1   
[21] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. 2017. 3   
[22] Ameya Prabhu, Zhipeng Cai, Puneet Dokania, Philip Torr, Vladlen Koltun, and Ozan Sener. Online continual learning without the storage constraint. arXiv preprint arXiv:2305.09253, 2023. 1   
[23] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017. 1   
[24] Minhyuk Seo, Hyunseo Koh, Wonje Jeung, Minjae Lee, San Kim, Hankook Lee, Sungjun Cho, Sungik Choi, Hyunwoo Kim, and Jonghyun Choi. Learning equi-angular representations for online continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23933–23942, 2024. 1

[25] James Smith, Yen-Chang Hsu, Jonathan Balloch, Yilin Shen, Hongxia Jin, and Zsolt Kira. Always be dreaming: A new approach for data-free class-incremental learning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9374–9384, 2021. 1   
[26] Jaehong Yoon, Divyam Madaan, Eunho Yang, and Sung Ju Hwang. Online coreset selection for rehearsal-based continual learning. arXiv preprint arXiv:2106.01085, 2021. 1   
[27] Dawen Zhang, Boming Xia, Yue Liu, Xiwei Xu, Thong Hoang, Zhenchang Xing, Mark Staples, Qinghua Lu, and Liming Zhu. Tag your fish in the broken net: A responsible web framework for protecting online privacy and copyright. arXiv preprint arXiv:2310.07915, 2023. 1