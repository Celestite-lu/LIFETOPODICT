# Leveraging the Sequential Nature of Language for Interpretability

Usha Bhalla \* 1 2 Alex Oesterling \* 1 Claudio Mayrink Verdun 1 Flavio P. Calmon 1 Himabindu Lakkaraju 1

# Abstract

Interpretability strives to discover the concepts learned and represented by models, frequently with unsupervised learning methods such as dictionary learning. While such methods allow for flexibility and applicability to a wide variety of domains, recent implementations such as sparse autoencoders (SAEs) discard valuable modalityspecific information and priors that we have on the structure of different data modalities. In this work, we argue that the temporal dimension of language is a rich feature source that can be leveraged by dictionary learning methods in a self-supervised manner, allowing for better learning and disentanglement of semantic and syntactical features represented by language models. We propose a data-generating process for such features, which informs a novel approach to train Temporal SAEs that can extract semantic concepts from natural language. We experimentally verify that accounting for the temporal structure of language improves SAEs’ ability to capture semantic features in text data with minimal loss in performance.

# 1. Introduction

The field of machine learning interpretability seeks to understand what information models represent and encode – often with goals such as being able to rigorously audit models, control and steer them, or learn something new about the data itself or the functions and algorithms the model has learned. In cases where one has a specific concept of interest that they are trying to understand or control, they can leverage supervised interpretability methods such as probing (Kohn ¨ , 2016; Alain & Bengio, 2016; Belinkov, 2022) or steering (Subramani et al., 2022); however, many other cases exist where we may want to discover concepts in an unsupervised manner, to better understand and learn

\*Equal contribution 1Harvard University 2Kempner Institute. Correspondence to: Usha Bhalla <usha bhalla@g.harvard.edu>, Alex Oesterling <aoesterling@g.harvard.edu>.

ICML 2025 Workshop on Assessing World Models. Copyright 2025 by the author(s).

about the variety of features that models rely on. Dictionary learning (Dumitrescu & Irofti, 2018; Bricken et al., 2023), and more specifically, sparse autoencoders (Ng et al., 2011; Makhzani & Frey, 2013), have emerged as the primary method to perform such unsupervised concept discovery, with prior works finding they discover, human-interpretable and sometimes even unknown features that can be used to steer models. In fact, SAEs have been applied across a variety of modalities: language, vision, x-rays (Abdulaal et al., 2024), protein sequencing (Garcia & Ansuini, 2025), time-series data (Wu et al., 2019), and more, with almost no changes or adaptations to the architecture for these different data types. However, this standard, out-of-the-box application across modalities assumes no underlying structure of the data. While this lack of supervision allows for high flexibility, it forgoes any priors and inductive biases we have about the data and the concepts we hope that SAEs will recover.

In this work, we focus on natural language and demonstrate how we can leverage intuitions and priors related to its temporal behavior to improve SAEs. To do so, we first construct a potential data-generating process (DGP) for language data that accounts for its sequential nature. In particular, we propose that high-level, abstract features should be smooth over time, and that low-level, syntactic features should be independent from the high-level features. We further expect that model representations encode a mixture of features relating to the semantics of the text, the prior context, and the syntactic requirements for the next-token generation task. We then propose a novel dictionary learning procedure that accounts for these assumptions and validate its efficacy through explorations of the ability of SAEs to recover the semantic, sequential, and syntactic features of the data.

Our contributions are the following:

1. We describe a simple data-generating process to model sequential structure in language.

2. We propose a novel loss function to ensure temporal consistency in SAE features.

3. We demonstrate empirically that Temporal SAEs exhibit better semantic structure while maintaining competitive performance to state-of-the-art SAEs.

# 2. Related Work

Sparse Autoencoders. In recent years, SAEs have emerged as a popular mechanistic interpretability technique for self-supervised concept discovery. They aim to explain models by decomposing intermediate model activations into sparse, human-interpretable feature spaces. While they were initially promising for addressing the problem of polysemanticity, where a single neuron in a model can represent multiple features (Elhage et al., 2022), in practice they have been shown to create new problems, such as feature splitting and absorption (Chanin et al., 2024), where features are split across multiple features or absorbed into less interpretable sub-features. To address these subsequent issues, methods such as Matryoshka SAEs (Bussmann et al., 2025) and transcoders (Paulo et al., 2025) have been proposed, which learn hierarchical and causal features. Recent work has also proposed learning dictionary features that are constrained to the data manifold (Fel et al., 2025) and reflect intuition about the geometry of model latent spaces (Hindupur et al., 2025), allowing for the recovery of heterogeneous concepts. However, all of these works assume a fully unsupervised objective for learning SAEs, treating each token in the training data as i.i.d., without acknowledging the temporal aspect of language and other sequential modalities.

Cognitive and Computational Models of Language. Literature in cognitive science has long studied the difference between syntactical and semantic content in language, with empirical evidence of different developmental trends for the two during human language acquisition (Brown, 1973) as well as differences in patterns of brain activity for both (Neville et al., 1992). Statistical methods subsequently demonstrated the ability to discover semantic content from language via topic modeling such as with Latent Dirichlet Allocation (Blei et al., 2003) and syntactic content with distributional methods (Redington et al., 1998) and Hidden Markov Models (Manning & Schutze, 1999). (Griffiths et al., 2004) combine computational models of semantics and syntax into HMM-LDA to model both simultaneously. Importantly, they argue that semantics in language exhibit long-range behavior, with different words or sentences in the same document having similar semantic content, whereas syntax is mostly dependent on short-range interactions. This perspective informs our model of the data-generating process of model latents and our training approach for inducing temporal consistency in SAEs.

# 3. Data-Generating Process

We formalize our data-generating process as such. Consider a speaker who is producing language, or a sequence of tokens $\tau _ { 1 } , . . . , \tau _ { T }$ . When the speaker produces each token $\tau _ { t } .$ they take into account many factors — their intent in speaking, the prior context of the token (i.e. what has already been said), syntactic requirements, and other implicit features corresponding to speaker idiosyncrasies (such as their accent, their method of language production, or linguistic style). These factors can be modeled as latent variables that control the language generation process, and they can be generally categorized into two types: variables that encode high-level or global information, $\mathbf { h } _ { t } ,$ and variables that encode lowlevel or local information ${ \bf l } _ { t }$ . High-level variables can be thought of as features that are invariant to the specific token, such as those capturing semantics and intent. Conversely, low-level information pertains to the specific timestep or token being produced, such as a word’s grammatical gender.

We model the speaker’s language production process as a randomized function mapping the context and these latent variables to the next token

$$
\tau_ {t} = \phi (\tau^ {t - 1}, \mathbf {h} _ {t}, \mathbf {l} _ {t}),
$$

where $\tau ^ { t - 1 }$ represents the previously-uttered tokens $\tau _ { 1 } , . . . , \tau _ { t - 1 }$ . Given a language model M, we pass tokens $\tau ^ { T }$ into M which produces latent vectors $\{ \mathbf { x } _ { t } ^ { L } \} _ { t = 1 } ^ { \bar { T } } \in \mathbb { R } ^ { d }$ at layer $L .$ For simplicity, we analyze a single layer and drop the L superscript.

Our goal is to recover $M \mathrm { { s } }$ encoding of the data-generating latent variables by decomposing its representations into interpretable features corresponding to $\mathbf { h } _ { t } , \mathbf { l } _ { t }$ . We ground our DGP in computational cognitive science and linguistics literature, where statistical models of language found success with similar assumptions (Griffiths et al., 2004). We specify the assumptions of our DGP, specifically regarding the nature of $\mathbf { h } _ { t } , \mathbf { l } _ { t }$ , below.

1. $\mathbf { h } _ { t }$ is time invariant, meaning two tokens $\mathbf { x } _ { t } , \mathbf { x } _ { t ^ { \prime } }$ sampled from the same sequence should have similar latents $\mathbf { h } _ { t } \approx \mathbf { h } _ { t ^ { \prime } }$ . Essentially, for a given sequence of text, the semantics should be relatively constant over time.   
2. In a language model, $\mathbf { h } _ { t }$ and ${ \bf l } _ { t }$ are represented additively: we can decompose $\mathbf { x } _ { t }$ as $\mathbf { x } _ { t } ^ { \mathbf { h } } + \mathbf { x } _ { t } ^ { \mathbf { l } }$ where $\mathbf { x } _ { t } ^ { \mathbf { h } }$ captures the high level features used to generate $\mathbf { x } _ { t }$ and similarly for $\mathbf { x } _ { t } ^ { \overline { { 1 } } }$ .

# 4. Temporal Sparse Autoencoders

We propose a modification to existing SAE architectures that leverages the sequential nature of language data as follows. Sparse autoencoders are comprised of an encoder, decoder, and nonlinear activation function. We decompose the encoders and decoders into two parts each: one that represents the high-level features $\mathbf { h } _ { t }$ present in the input data $\mathbf { x } _ { t }$ and one that represents the low-level features $\mathbf { l } _ { t } .$ .

We partition the SAE feature space into high level and low level features. Without loss of generality we assume the first h indices are our high level features and the last $m - h$ indices are our low level features, where m is the number of features in the SAE. The SAE architecture can be defined as the following, taking in input $\mathbf { x } _ { t } \in \mathbb { R } ^ { d }$ :

$$
\mathbf {f} (\mathbf {x} _ {t}) = \sigma (\mathbf {W} ^ {\mathrm{enc}} \mathbf {x} _ {t} + \mathbf {b} ^ {\mathrm{enc}}),
$$

$$
\hat {x} (\mathbf {f}) = \mathbf {W} ^ {\mathrm{dec}} \mathbf {f} (\mathbf {x} _ {t}) + \mathbf {b} ^ {\mathrm{dec}}.
$$

Here, $\mathbf { W } ^ { \mathrm { e n c } } \in \mathbb { R } ^ { m \times d }$ is the encoder matrix, and $\mathbf { W } ^ { \mathrm { d e c } } \in$ $\mathbb { R } ^ { d \times m }$ is the decoder comprised of high-level features $\mathbf { W } _ { 0 : h } ^ { \mathrm { d e c } } \in \mathbb { R } ^ { d \times h }$ and low-level features $\mathbf { W } _ { h : m } ^ { \mathrm { { d e c } } } \in \mathbb { R } ^ { d \times ( m - h ) }$ h:m such that their concatenation equals ${ \mathbf W } ^ { \mathrm { d e c } } . ~ { \mathbf b } ^ { \mathrm { e n c } } ~ \in ~ \mathbb { R } ^ { d }$ and $\mathbf { b } ^ { \mathrm { d e c } } \in \mathbb { R } ^ { d }$ are the encoder and decoder bias respectively. We define the following loss function, where the high-level features $\mathbf { f } _ { 0 : h } ( \mathbf { x } _ { t } )$ should reconstruct the input and the low-level features $\mathbf { f } _ { h : m } ( \mathbf { x } _ { t } )$ should reconstruct the residual, similar to the Matryoshka SAE objective in (Bussmann et al., 2025).

$$
\mathcal {L} (\mathbf {x} _ {t}) = \mathcal {L} _ {H} + \mathcal {L} _ {L} + \alpha \mathcal {L} _ {\text { contr }},
$$

$$
\mathcal {L} _ {H} = \| \mathbf {x} _ {t} - \mathbf {W} _ {0: h} ^ {\mathrm{dec}} \mathbf {f} _ {0: h} (\mathbf {x} _ {t}) + \mathbf {b} ^ {\mathrm{dec}} \| _ {2} ^ {2},
$$

$$
\mathcal {L} _ {L} = \| \mathbf {x} _ {t} - \mathbf {W} ^ {\mathrm{dec}} \mathbf {f} (\mathbf {x} _ {t}) + \mathbf {b} ^ {\mathrm{dec}} \| _ {2} ^ {2}.
$$

We then add a training objective that encourages $\mathbf { W } _ { 0 : h } ^ { \mathrm { e n c } }$ to learn features that respect our assumptions about ht. We do this by adding a contrastive term to the loss function that encourages ${ \bf W } _ { 0 : h } ^ { \mathrm { e n c } } { \bf x } _ { t }$ to be similar to ${ \bf W } _ { 0 : h } ^ { \mathrm { e n c } } { \bf x } _ { t - 1 }$ , as we expect high-level features to be similar for two tokens from the same sequence, especially for two adjacent tokens. Let $\mathbf { z } _ { t }$ be the high-level features $\mathbf { f } _ { 0 : h } ( \mathbf { x } _ { t } )$ , and let $s ( \mathbf { x } , \mathbf { y } )$ be the cosine similarity between vectors x and y in the same latent space. Our contrastive loss at temperature λ is defined as

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{contr}} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {e ^ {s (\mathbf {z} _ {t} ^ {(i)} , \mathbf {z} _ {t - 1} ^ {(i)}) / \lambda}}{\sum_ {j = 1} ^ {N} e ^ {s (\mathbf {z} _ {t} ^ {(i)} , \mathbf {z} _ {t - 1} ^ {(j)}) / \lambda}} \\ - \frac {1}{N} \sum_ {j = 1} ^ {N} \log \frac {e ^ {s (\mathbf {z} _ {t - 1} ^ {(j)} , \mathbf {z} _ {t} ^ {(j)}) / \lambda}}{\sum_ {i = 1} ^ {N} e ^ {s (\mathbf {z} _ {t - 1} ^ {(i)} , \mathbf {z} _ {t} ^ {(j)}) / \lambda}}, \\ \end{array}
$$

where N is our batch size an d z(i) $\mathbf { z } _ { t } ^ { ( i ) }$ is the ith latent vector in the batch. In practice, we load activations in pairs $\mathbf { x } _ { t } , \mathbf { x } _ { t - 1 }$ and shuffle the pairs to get diversity in each batch. We additionally explore sampling the second token uniformly over past tokens $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { t - 1 }$ to encourage long range semantic consistency (Appendix A.1.1). Intuitively, we expect that for high-level, abstract, and semantic features, tokens from the same sequence should have more similar activations than those sampled from two sequences randomly. Thus, we constrain the high-level dictionary to represent these as close in SAE feature space. For low-level, token-specific features, we do not have the same expectation, but by nature of fitting the residual left over by the high-level component of the network, our loss naturally encourages the remaining latents to capture higher-variance features over a sequence.

Table 1. Performance metrics for BatchTopK, Matryoshka, and Temporal SAEs for Pythia-160m and Gemma2-2b. 

<table><tr><td></td><td>SAE Model</td><td>FVU</td><td>% Dead</td><td> $\mathcal{L}_{L_2}$ </td><td>Cos. Sim.</td></tr><tr><td rowspan="3">Pythia</td><td>Temporal</td><td>0.09</td><td>34%</td><td>10.4</td><td>0.89</td></tr><tr><td>Matryoshka</td><td>0.07</td><td>44%</td><td>9.2</td><td>0.91</td></tr><tr><td>BatchTopK</td><td>0.07</td><td>62%</td><td>9.3</td><td>0.91</td></tr><tr><td rowspan="3">Gemma</td><td>Temporal</td><td>0.41</td><td>1%</td><td>131</td><td>0.86</td></tr><tr><td>Matryoshka</td><td>0.38</td><td>1%</td><td>119</td><td>0.88</td></tr><tr><td>BatchTopK</td><td>0.38</td><td>1%</td><td>122</td><td>0.88</td></tr></table>

# 5. Experiments

We conduct experiments on Pythia 160M (Biderman et al., 2023) and Gemma2-2b (Team et al., 2024) to understand whether Temporal SAEs learn clear high- and low-level features, and whether these features capture the types of concepts we expect. We compare against baselines of Batch-TopK SAEs (Bussmann et al., 2024) and Matryoshka SAEs (Bussmann et al., 2025).

# 5.1. Implementation Details

We train and evaluate SAEs on the residual streams of Gemma2-2b (layer 18, 16k features) and Pythia-160m (layer 8, 32k features) on the RedPajama-Data-1T-Sample dataset (Weber et al., 2024). When training Temporal and Matryoshka SAEs, we split the dictionaries into two chunks, with the first being 20% of features for Pythia and 10% for Gemma. We only apply our temporal loss to the first chunk. All models were trained with the BatchTopK activation function (Bussmann et al., 2024) and auxiliary loss from (Gao et al., 2024), which encourages dead latents to reconstruct the error left from existing reconstructions. SAEs for Pythia were trained with a k−sparsity of 20 and Gemma SAEs were trained with $k = 4 0$ . We compare our Temporal SAEs to baselines of Matryoshka and regular BatchTopK SAEs.

# 5.2. SAE Evaluation Metrics

We evaluate all SAEs on a 128k token sample from the Pile (Gao et al., 2020) on the following standard metrics to ensure that temporal consistency does not significantly degrade reconstruction quality.

Fraction of variance unexplained (FVU) measures the proportion of the variance of the model latent, x, that the SAE reconstruction, ˆx, fails to account for as $1 - ( \mathrm { V a r } ( \mathbf { x } -$ $\hat { \mathbf { x } } ) / \mathrm { V a r } ( \mathbf { x } ) )$ , where Var is the empirical variance. $\mathbf { L } _ { 2 }$ loss measures the mean-squared error of the reconstruction, or $\| \mathbf { x } - \hat { \mathbf { x } } \| _ { 2 } ^ { 2 }$ . Cosine similarity computes the similarity of the reconstruction and SAE input as $\langle \mathbf { x } , \hat { \mathbf { x } } \rangle / ( \| \mathbf { x } \| _ { 2 } \| \hat { \mathbf { x } } \| _ { 2 } )$ . Finally, the percentage of dead latents counts the percentage of SAE latent features that did not activate for any token in the evaluation dataset. We find that Temporal SAEs result in a minimal loss in reconstruction metrics compared to both regular BatchTopK and Matryoshka SAEs across both models, and in some cases improve upon baselines by removing the number of dead features (Table 1). However, this slight loss in reconstruction comes at a significant improvement in the representation of semantic and contextual information.

![](images/dc89861b1197249db8cecc81dee478f4bbc26df6bfdbae13acbd4abf0d407e01.jpg)

<details>
<summary>scatter</summary>

| Model        | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 | Label 6 | Label 7 | Label 8 | Label 9 | Label 10 |
| ------------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| Temporal SAE | 120     | 85      | 70      | 65      | 55      | 45      | 35      | 25      | 15      | 10       |
| Matryoshka SAE| 110     | 75      | 60      | 55      | 45      | 35      | 25      | 15      | 10      | 5        |
</details>

![](images/f71fe65cbdd1b5764caf16d0ef0ef30798d656980f940a183cd3f13d28da2882.jpg)

![](images/2b4cd2cca46862e6f8c4401815b133330ee3d7559768a17c4beee3479d956ce7.jpg)

<details>
<summary>scatter</summary>

| Label Type | X Coordinate | Y Coordinate |
|------------|--------------|--------------|
| Top Left   | [various]    | [various]    |
| Top Right  | [various]    | [various]    |
</details>

Figure 1. TSNE plots of the SAE feature activation for Temporal (top row) and Matryoshka (bottom row) SAEs. We label points by their semantic category (left), the context or sequence the token comes from (middle), and the part of speech of the token (right). We find that Temporal SAEs learn a balance of both semantic and syntactic information, whereas Matryoshka SAEs prioritize only syntactic information at the cost of all semantics.

# 5.3. Temporal Consistency Evaluation

We next explore the benefits of our temporal loss and find that Temporal SAEs better capture and differentiate semantic features of the data. We use high-level semantic labels from two datasets: finefineweb and wikipedia. Examples of these labels for finefineweb are {“literature”, “food”, “drama and film”, “mathematics”, “medical”} and examples from wikipedia are {“Geography”, “History”, “Knowledge”, “People”, “Religion”}. We use a part-ofspeech tagger to construct syntactic labels for tokens, and also group tokens by sequence or article to understand how Temporal SAEs treat tokens from the same context.

We first conduct a qualitative evaluation by applying TSNE (Van der Maaten & Hinton, 2008) to visualize the concepts represented by Temporal SAEs and a Matryoshka SAE baseline. Language models are known to be good at tracking and recovering data over long contexts and produce semantically relevant and syntactically correct outputs, so we expect that SAEs should be able to recover all of these types of features. In Figure 1 for finefineweb, we see that Matryoshka SAEs only represent features corresponding to syntactic concepts, as we can only see clear clusters forming in the plot labeled with part-of-speech tags. In contrast, the plots

<table><tr><td colspan="2"></td><td>Semantics</td><td>Syntax</td><td>Context</td></tr><tr><td rowspan="4">Pythia</td><td>Base Model</td><td>0.703</td><td>0.865</td><td>0.835</td></tr><tr><td>Temporal SAE</td><td>0.647</td><td>0.796</td><td>0.799</td></tr><tr><td>Matryoshka SAE</td><td>0.576</td><td>0.848</td><td>0.430</td></tr><tr><td>BatchTopK SAE</td><td>0.587</td><td>0.847</td><td>0.654</td></tr><tr><td rowspan="4">Gemma</td><td>Base Model</td><td>0.720</td><td>0.875</td><td>0.974</td></tr><tr><td>Temporal SAE</td><td>0.682</td><td>0.857</td><td>0.964</td></tr><tr><td>Matryoshka SAE</td><td>0.633</td><td>0.877</td><td>0.895</td></tr><tr><td>BatchTopK SAE</td><td>0.648</td><td>0.876</td><td>0.918</td></tr></table>

Table 2. Performance on semantic, syntactic, and context identification tasks for FineFineWeb. Temporal SAEs outperform baselines for capturing semantic and contextual information.

<table><tr><td colspan="2"></td><td>Semantics</td><td>Syntax</td><td>Context</td></tr><tr><td rowspan="4">Pythia</td><td>Base Model</td><td>0.718</td><td>0.832</td><td>0.869</td></tr><tr><td>Temporal SAE</td><td>0.633</td><td>0.770</td><td>0.731</td></tr><tr><td>Matryoshka SAE</td><td>0.568</td><td>0.824</td><td>0.582</td></tr><tr><td>BatchTopK SAE</td><td>0.559</td><td>0.815</td><td>0.605</td></tr><tr><td rowspan="4">Gemma</td><td>Base Model</td><td>0.939</td><td>0.873</td><td>0.927</td></tr><tr><td>Temporal SAE</td><td>0.896</td><td>0.865</td><td>0.882</td></tr><tr><td>Matryoshka SAE</td><td>0.882</td><td>0.877</td><td>0.817</td></tr><tr><td>BatchTopK SAE</td><td>0.896</td><td>0.875</td><td>0.846</td></tr></table>

Table 3. Performance on semantic, syntactic, and context identification tasks for Wikipedia data. Temporal SAEs outperform baselines for capturing semantic and contextual information.

colored by semantic category and context show no clear patterns. Temporal SAEs, on the other hand, learn a mix of all three types of features, with clusters forming in all three plots. Intuitively, clusters are most clear when labeled by context, as this is exactly what our proposed contrastive objective optimizes for. Results are qualitativaly very similar for wikipedia, as seen in Appendix Figure 2.

We validate these results by training probes on the SAE decompositions to predict these labels, and compare with a baseline of probing the base language model latents directly. This baseline represents how much information is natively represented in the language model; our goal is to preserve this information while transforming to a more interpretable space. Table 2 demonstrates that Temporal SAEs greatly outperform baseline methods on finefineweb in terms of capturing semantics and context with gains of 10% and 25% respectively for Pythia and 5% and 7% respectively for Gemma. On wikipedia (Table 3), we observe a similar trend. Additionally, Temporal SAEs preserve syntactical information to a similar level as the base language model. We emphasize that these SAE methods are all trained in an unsupervised manner on the same dataset, and Temporal SAEs only vary in the inclusion of a temporal consistency loss term to capture our intuition about the structure of language.

# 6. Conclusion

In this work, we advocate for the incorporation of priors on the data modality into the development of unsupervised methods for interpretability. We focus on the domain of natural language, leveraging the insight that semantic content remains consistent across words and sentences. We then propose a novel loss function for training SAEs such that a subset of features behave smoothly over time, better extracting semantic features from data. Through experiments, we demonstrate that the features learned by Temporal SAEs are more semantically structured, with minimal loss to reconstruction performance.

# 7. Acknowledgements

This work is supported in part by the National Science Foundation under grants CIF 2312667, FAI 2040880, CIF 2231707, IIS-2008461, IIS-2040989, IIS-2238714, the AI2050 Early Career Fellowship by Schmidt Sciences, and faculty research awards from Google, OpenAI, Adobe, JP-Morgan, Harvard Data Science Initiative, and the Digital, Data, and Design (Dˆ3) Institute at Harvard. UB is funded by the Kempner Institute Graduate Research Fellowship. AO is supported by the National Science Foundation Graduate Research Fellowship under Grant No. DGE-2140743.

# References

Abdulaal, A., Fry, H., Montana-Brown, N., Ijishakin, A., ˜ Gao, J., Hyland, S., Alexander, D. C., and Castro, D. C. An x-ray is worth 15 features: Sparse autoencoders for interpretable radiology report generation. arXiv preprint arXiv:2410.03334, 2024.

Alain, G. and Bengio, Y. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.   
Belinkov, Y. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.   
Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.   
Blei, D. M., Ng, A. Y., and Jordan, M. I. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan): 993–1022, 2003.   
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformercircuits.pub/2023/monosemantic-features/index.html.   
Brown, R. A first language: The early stages. Harvard University Press, 1973.   
Bussmann, B., Leask, P., and Nanda, N. Batchtopk sparse autoencoders. arXiv preprint arXiv:2412.06410, 2024.   
Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. Learning multi-level features with matryoshka sparse autoencoders. arXiv preprint arXiv:2503.17547, 2025.   
Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., and Bloom, J. A is for absorption: Studying feature splitting and absorption in sparse autoencoders. arXiv preprint arXiv:2409.14507, 2024.   
Dumitrescu, B. and Irofti, P. Dictionary learning algorithms and applications. Springer, 2018.   
Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.   
Fel, T., Lubana, E. S., Prince, J. S., Kowal, M., Boutin, V., Papadimitriou, I., Wang, B., Wattenberg, M., Ba, D., and Konkle, T. Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models. arXiv preprint arXiv:2502.12892, 2025.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.   
Garcia, E. N. V. and Ansuini, A. Interpreting and steering protein language models through sparse autoencoders. arXiv preprint arXiv:2502.09135, 2025.   
Griffiths, T., Steyvers, M., Blei, D., and Tenenbaum, J. Integrating topics and syntax. Advances in neural information processing systems, 17, 2004.   
Hindupur, S. S. R., Lubana, E. S., Fel, T., and Ba, D. Projecting assumptions: The duality between sparse autoencoders and concept geometry. arXiv preprint arXiv:2503.01822, 2025.   
Kohn, A. Evaluating embeddings using syntax-based clas-¨ sification tasks as a proxy for parser performance. In Proceedings of the 1st Workshop on Evaluating Vector-Space Representations for NLP, pp. 67–71, 2016.   
Makhzani, A. and Frey, B. K-sparse autoencoders. arXiv preprint arXiv:1312.5663, 2013.   
Manning, C. and Schutze, H. Foundations of statistical natural language processing. MIT press, 1999.   
Neville, H. J., Mills, D. L., and Lawson, D. S. Fractionating language: Different neural subsystems with different sensitive periods. Cerebral cortex, 2(3):244–258, 1992.   
Ng, A. et al. Sparse autoencoder. CS294A Lecture notes, 72 (2011):1–19, 2011.   
Paulo, G., Shabalin, S., and Belrose, N. Transcoders beat sparse autoencoders for interpretability. arXiv preprint arXiv:2501.18823, 2025.   
Redington, M., Chater, N., and Finch, S. Distributional information: A powerful cue for acquiring syntactic categories. Cognitive science, 22(4):425–469, 1998.   
Subramani, N., Suresh, N., and Peters, M. E. Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 566–581, 2022.   
Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahriari, B., Rame, A., et al. Gemma 2: Improving open ´ language models at a practical size. arXiv preprint arXiv:2408.00118, 2024.

Van der Maaten, L. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.   
Weber, M., Fu, D. Y., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., Lyu, X., Nguyen, H., Yao, X., Adams, V., Athiwaratkun, B., Chalamala, R., Chen, K., Ryabinin, M., Dao, T., Liang, P., Re, C., Rish, I., and Zhang, C. ´ Redpajama: an open dataset for training large language models. NeurIPS Datasets and Benchmarks Track, 2024.   
Wu, K., Liu, J., Liu, P., and Yang, S. Time series prediction using sparse autoencoder and high-order fuzzy cognitive maps. IEEE transactions on fuzzy systems, 28(12):3110– 3121, 2019.

# A. Appendix

# A.1. Additional Results.

# A.1.1. TRAINING ON RANDOM PREVIOUS TOKEN.

In addition to training with a constrastive loss on the previous token $\mathbf { x } _ { t - 1 }$ , we also explore training on tokens sampled uniformly at random from all past tokens $\mathbf { x } _ { 1 } , . . . , \mathbf { x } _ { t - 1 }$ . In Table 4 we show results on Pythia when training on a random past or the t − 1th token. Random sampling helps recover lost performance in terms of syntax representation. We hypothesize this is because syntax is extremely predictable between two adjacent tokens $\mathbf { x } _ { t - 1 }$ and $\mathbf { x } _ { t } ,$ and random sampling breaks this predictability and requires the SAE to actually represent syntactic features.

<table><tr><td></td><td>Semantics</td><td>Syntax</td><td>Context</td></tr><tr><td>Base Model</td><td>0.703</td><td>0.865</td><td>0.835</td></tr><tr><td>Temporal SAE (t-1)</td><td>0.647</td><td>0.796</td><td>0.799</td></tr><tr><td>Temporal SAE (random)</td><td>0.687</td><td>0.839</td><td>0.830</td></tr><tr><td>Matryoshka SAE</td><td>0.576</td><td>0.848</td><td>0.430</td></tr><tr><td>BatchTopK SAE</td><td>0.587</td><td>0.847</td><td>0.654</td></tr></table>

Table 4. Table 2 from main paper with additional results for Temporal SAE trained on randomly-sampled past token.

# A.1.2. TSNE PLOTS FOR WIKIPEDIA.

In Figure 2, we report TSNE plots on the wikipedia dataset. Similar to finefineweb, we observe that Temporal SAEs contain more semantic structure than baselines.

![](images/d663ad6bacfd05e5686248d461f221862c95053163f2d2ddbe25d6c112ba372d.jpg)  
Figure 2. TSNE plots of the SAE feature activation for all SAE architectures and the LLM Latent Baseline (bottom). We label points by their semantic category (left), the context or sequence the token comes from (middle), and the part of speech of the token (right). We find that Temporal SAEs learn a balance of both semantic and syntactic information, whereas other SAEs prioritize only syntactic information at the cost of all semantics.

![](images/73102bb1528564b5ffba4b400195546df5bf686a64a9df8ce6173a9bef11e5c2.jpg)

<details>
<summary>line</summary>

| Subspace Dimensionality | Accuracy (Blue) | Accuracy (Orange) | Accuracy (Red) | Accuracy (Green) |
| ----------------------- | --------------- | ----------------- | -------------- | ---------------- |
| 2^5                     | 0.60            | 0.68              | 0.56           | 0.55             |
| 2^6                     | 0.65            | 0.69              | 0.58           | 0.57             |
| 2^7                     | 0.70            | 0.69              | 0.61           | 0.60             |
| 2^8                     | 0.71            | 0.69              | 0.62           | 0.61             |
| 2^9                     | 0.72            | 0.69              | 0.63           | 0.61             |
| 2^10                    | 0.72            | 0.69              | 0.62           | 0.61             |
| 2^11                    | 0.72            | 0.69              | 0.61           | 0.60             |
</details>

Pythia 160m Temporal SAE Matryoshka SAE BatchTopK SAE Baseline Accuracy

![](images/60e7ee635bf2bc4c3d11339c02ae42aac0184e1e8e1934ab52f23cd3dea85ef7.jpg)

<details>
<summary>line</summary>

| Subspace Dimensionality | Blue Line | Green Line | Red Line | Orange Line |
| ----------------------- | --------- | ---------- | -------- | ----------- |
| 2^5                     | 0.78      | 0.68       | 0.67     | 0.50        |
| 2^6                     | 0.81      | 0.73       | 0.72     | 0.60        |
| 2^7                     | 0.84      | 0.78       | 0.77     | 0.74        |
| 2^8                     | 0.85      | 0.81       | 0.80     | 0.78        |
| 2^9                     | 0.86      | 0.83       | 0.82     | 0.80        |
| 2^10                    | 0.86      | 0.84       | 0.83     | 0.81        |
| 2^11                    | 0.86      | 0.85       | 0.84     | 0.82        |
</details>

![](images/17769722a513f01b89b9897084539aeb8f0555609f41595c885c6b57b60dbd0e.jpg)

<details>
<summary>line</summary>

| Subspace Dimensionality | Blue Line | Orange Line | Red Line | Green Line |
| ----------------------- | --------- | ----------- | -------- | ---------- |
| 2^5                     | 0.45      | 0.60        | 0.18     | 0.19       |
| 2^6                     | 0.60      | 0.65        | 0.25     | 0.26       |
| 2^7                     | 0.75      | 0.70        | 0.35     | 0.34       |
| 2^8                     | 0.80      | 0.72        | 0.40     | 0.38       |
| 2^9                     | 0.82      | 0.73        | 0.45     | 0.43       |
| 2^10                    | 0.83      | 0.72        | 0.45     | 0.44       |
| 2^11                    | 0.83      | 0.71        | 0.45     | 0.44       |
</details>

Figure 3. Probe performance on Semantics, Syntax, and Context after projecting via PCA to varying subspace dimensionalities. Note that Pythia’s latent space has a maximum projected of $2 ^ { 9 } = 5 1 2$ as its original dimensionality is 768.

# A.1.3. PROBING WITH VARYING FEATURE SUBSPACE SIZES.

We explore how semantic, syntactic, and contexual information are distributed in the representations of Pythia and its SAEs. An interpretable monosemantic representation should use a few (and ideally one) neuron to represent a feature, and we test this by projecting the SAE representations into a lower-dimensional subspace and measure the drop in performance of probing. Additionally, we apply the same projection analysis to the latent representation of the base model, and under the hypothesis that the language model neurons are highly polysemantic and represent features in superposition, the performance drop of the base model should be relatively larger than that of an SAE.

We project into the optimal lower-dimensional subspace that contributes most to the variance of our data distribution using PCA. Specifically, for each dataset, we compute a PCA of the data matrix and then retain only the top k components for varying amounts of k. We then train probes with $\ell _ { 2 }$ regularization on the base dataset and the filtered dataset for each level of k and report test accuracy (on an 80%-20% train-test split) in Figure 3. We observe that the probe accuracy on the semantic task increases more rapidly for the base model than for any of the SAEs, validating the superposition hypothesis: signal for the semantic task is distributed across more features, so the baseline model needs a relatively higher set of principal components to recover its full-latent performance. We observe a similar trend for the context task. Finally, for the syntax task, we observe that Temporal SAE performance is close-behind the base model and baseline SAEs, although it also has a sharp increase in accuracy (indicating a more distributed syntax representation) than baselines.