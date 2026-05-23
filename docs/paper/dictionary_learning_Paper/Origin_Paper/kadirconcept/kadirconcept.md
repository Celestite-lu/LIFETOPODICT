# CONCEPT-GUIDED DICTIONARY LEARNING FOR INTER-PRETABLE CONCEPT EXTRACTION AND ATTRIBUTION IN LARGE VISION–LANGUAGE MODELS

Anonymous authors

Paper under double-blind review

# ABSTRACT

Autoregressive large vision–language models (LVLMs) generate text sequentially, conditioning each token on evolving multimodal states. This makes it difficult to assess whether predictions are grounded in visual concepts or instead reflect hallucination or bias. Existing concept-discovery approaches, such as TCAV, CRAFT, and CLIP-Dissect, are designed for encoder-only or contrastive models. At the same time, recent LVLM methods such as CoX-LMM depend on labeled concepts and simplified settings, limiting scalability.

We propose Concept-Guided Dictionary Learning (CGDL), a weakly supervised and scalable framework for discovering multimodal concept vectors in autoregressive LVLMs. CGDL first prompts the model to identify textual concepts in a dataset. For each concept, it constructs positive and negative patch sets from SAM-generated foreground crops and randomized background patches. Next, we apply a binary prompt to extract hidden activations for positive and negative patches. A contrastive dictionary learning stage then disentangles concept-aligned activations from residual noise, yielding sparse, monosemantic vectors that reveal semantically aligned visual–textual interactions and enable faithful attribution of predictions to visual evidence.

On ImageNet-1k and MSCOCO, CGDL outperforms recent interpretability methods with up to 4% higher sparsity, 11% greater stability, 17% lower overlap, and strong attribution faithfulness, while scaling efficiently to large concept vocabularies. These results advance concept-based interpretability for LVLMs and provide a practical step toward transparent multimodal reasoning.

# 1 INTRODUCTION

Large vision–language models (LVLMs) generate text autoregressively, conditioning each token on evolving multimodal states. This enables rich, context-sensitive prediction but raises unique interpretability challenges: predictions may stem from spurious correlations or hallucinations rather than visual evidence. Existing methods, such as saliency maps or training-based grounding, localize objects but do not reveal the higher-level concepts driving model decisions. Concept-based explanations offer more conceptual insights, yet most methods assume static feature spaces. Approaches like TCAV Kim et al. (2018), CRAFT Fel et al. (2023b), CLIP-Dissect Oikarinen & Weng (2023), and Holistic Fel et al. (2023a) cannot extend to autoregressive models and are restricted to single-object settings, while CoX-LMM Parekh et al. (2024) additionally requires a labeled concept token, further limiting scalability (Table 1).

We propose Concept-Guided Dictionary Learning (CGDL), a weakly supervised framework for scalable concept discovery in autoregressive LVLMs. We use native vision–language models (e.g., Qwen2.5-VL and Gemma-3n Team (2025b;a)), rather than models that use a frozen LLM adapted via a bridging mechanism Vallaeys et al. (2024), because this setting lets us trace how image concepts propagate through an autoregressively trained VLM all the way to the final layer. CGDL finds candidate concepts and treats each one as a one-vs-all representation learning problem with a two-basis decomposition. It constructs positive and negative patch sets using segmentation and random cropping, prompts the model with a binary question to identify whether the concept exists in each crop, and extracts the model’s hidden activations for both positive and negative patches. This formulation forces the dictionary to disentangle activations from residual noise (negative patches), yielding sparse, non-overlapping vectors that faithfully align visual and textual modalities.

Table 1: Comparison of concept-based interpretability methods. Only CGDL supports weakly supervised, scalable discovery in autoregressive LVLMs; most prior methods target encoder-only image-encoder (IE) models and single-object concepts, and they cannot perform text grounding (TG). 

<table><tr><td>Method</td><td>Type</td><td>TG</td><td>AutoReg</td><td>Limitation/ tage</td><td>Advantage</td></tr><tr><td>TCAV Kim et al. (2018)</td><td>IE</td><td>✕</td><td>✕</td><td>Supervised</td><td></td></tr><tr><td>ACE Ghorbani et al. (2019)</td><td>IE</td><td>✕</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>CRAFT Fel et al. (2023b)</td><td>IE</td><td>✕</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>EAC Sun et al. (2023)</td><td>IE</td><td>✕</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>CLIP-Dissect Oikarinen &amp; Weng (2023)</td><td>IE</td><td>√</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>Holistic Fel et al. (2023a)</td><td>IE</td><td>✕</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>CoX-LMM Parekh et al. (2024)</td><td>AutoReg</td><td>√</td><td>√</td><td>Single object</td><td></td></tr><tr><td>MCD Grobrügge et al. (2025)</td><td>IE</td><td>✕</td><td>✕</td><td>Single object</td><td></td></tr><tr><td>CGDL (Ours)</td><td>AutoReg</td><td>√</td><td>√</td><td>/ Unlimited-object</td><td></td></tr></table>

While most concept extraction methods rely on similar dictionary learning formulations, prior work has shown that the quality of discovered concepts depends more on how the data is presented to the dictionary than on the specific factorization algorithm Grobrugge et al. (2025); Sun et al. (2023). ¨ CGDL is not merely Semi-NMF Fel et al. (2023a) with preprocessing; rather, it provides a general, model-agnostic framework that reframes concept discovery as a contrastive one-vs-all decomposition. The two-basis design prevents atom collapse in large vocabularies, allowing CGDL to scale robustly to thousands of concepts Kim & Park (2008). By combining weakly supervised concept bags, weak localization, residual contrast, and two-basis factorization, CGDL achieves monosemantic, multimodal vectors at scale—something previous dictionary learning methods for LVLMs have not demonstrated.

# Our contributions are:

• We introduce CGDL, a weakly supervised framework for scalable concept discovery in autoregressive LVLMs, overcoming the single-object limitation of prior methods.   
• We leverage the Segment Anything Model (SAM) Kirillov et al. (2023) for weak concept localization, providing high-quality positive/negative patch sets.   
• We propose a contrastive residual extraction scheme that enforces monosemanticity by separating concept vs. no-concept activations.   
• We show that CGDL yields faithful multimodal attribution, bridging visual and textual modalities using extensive experiments on ImageNet-1k and MSCOCO, demonstrating superior sparsity, stability, and faithfulness over prior methods, scaling to 1k concepts with improved attribution quality (up to +9% CLIPScore (Radford et al., 2021a), +4% BERTScore (Devlin et al., 2019)).

Together, these advances provide a practical step toward transparent multimodal reasoning, bridging the gap between autoregressive generation and human-understandable concept-based explanations.

# 2 RELATED WORK

A central goal in interpretability is to uncover how internal representations encode semantically meaningful concepts. Early work introduced Concept Activation Vectors (CAVs) (Kim et al., 2018), which quantify model sensitivity to human-defined concepts but require curated examples, limiting scalability. Subsequent extensions sought to automate discovery (Ghorbani et al., 2019), yet dependence on segmentation quality hindered robustness.

Alternative approaches use saliency maps (Selvaraju et al., 2017; Strumbelj & Kononenko, 2014) or training-based grounding (Kang et al., 2024; Zhang et al., 2023; Ma et al., 2024) to highlight where evidence lies. While effective for localization, these methods do not articulate what abstract concepts the model internally represents, thereby complementing, but not replacing, concept-based explanations. To reduce supervision, some methods decompose hidden states into interpretable factors. NMF (Liu et al., 2025), dictionary learning with prototypes (CRAFT) (Fel et al., 2023b), PCA, UMAP, and sparse autoencoders (Pach et al., 2025) have shown promise in vision-only settings, with unifying benchmarks (Fel et al., 2023a). CLIP-based concept attribution (Oikarinen & Weng, 2023; Dreyer et al., 2025) further advanced automated discovery, but these methods remain restricted to vision encoders.

Moreover, concept bottleneck models (CBMs) explain vision classifiers or image encoders by using an LVLM to extract textual concepts from images and then training a linear proxy on these concepts to predict the model output (Oikarinen et al., 2023; Yang et al., 2023). We also use an LVLM to first identify which concepts are present in the dataset, but we do not train any proxy model. Instead, the discovered concepts guide the extraction of concept vectors from the same LVLM’s hidden states for post hoc explanation of that LVLM.

Feature finding (Pach et al., 2025; Zhang et al., 2025) for understanding a model’s knowledge and for model steering using sparse autoencoders is another promising direction for concept-level explanation. However, these methods integrate a sparse autoencoder (SAE) inside the model, which changes the architecture and can alter its behavior, making them less suitable for post hoc local explanation. While concept extraction overlaps with mechanistic interpretability (Templeton, 2024; Pach et al., 2025; Elhage et al., 2022), our focus is on single-layer monosemantic concept discovery in autoregressive LVLMs rather than neuron- or circuit-level analysis. Alternative approaches such as attention maps (Jain & Wallace, 2019), causal tracing (Meng et al., 2023), and linear probes (Alain & Bengio, 2018) provide useful insights but suffer from weak causal grounding, poor scalability, or reliance on labeled supervision.

Autoregressive LVLMs present new challenges: activations evolve across time steps, residual streams encode multiple dependencies, and concepts rarely align with a single hidden state (Templeton, 2024). CoX-LMM (Parekh et al., 2024) adapted Semi-NMF (SNMF) (Trigeorgis et al., 2014) to LVLM activations but struggles with (i) reliance on tokenized object names, (ii) limited support for multi-token concepts, (iii) background noise from full-image extraction, and (iv) persistent polysemanticity in residual streams. While methods such as TCAV, CRAFT, and CLIP-Dissect pioneered concept-based interpretability, they target static encoders and cannot be applied to LVLMs. CoX-LMM incorporates many of these ideas into a dictionary learning framework for autoregressive models and thus serves as the most representative baseline. We therefore compare CGDL against CoX-LMM using its strongest dictionary learning variants (SNMF, SAE) to ensure a fair comparison.

Unlike CoX-LMM and other concept extraction methods that rely on labeled tokens and often yield polysemantic vectors and assume single-object settings, we propose Concept-Guided Dictionary Learning (CGDL). CGDL introduces contrastive residual extraction with spatially localized image crops and candidate textual concepts, enforcing a clean separation between concept and noise. This produces faithful, monosemantic concept vectors. To the best of our knowledge, CGDL is the first framework to scale weakly supervised concept discovery from single-object to multi-object settings.

# 3 PRELIMINARIES

Recent work on concept vector extraction (Ghorbani et al., 2019; Sun et al., 2023; Fel et al., 2023a; Parekh et al., 2024) shows that many of these methods can be framed as dictionary learning, where activations are approximated by a small set of interpretable vectors. Formally, given activations $S \in \mathbb { R } ^ { n \times d }$ with n samples and d-dimensional features, we posit K latent concepts and solve

$$
\underset {U \in \mathbb {R} ^ {d \times K}, V \in \mathbb {R} ^ {n \times K}} {\arg \min} \| S - V U ^ {\top} \| _ {F} ^ {2},
$$

where $U = [ u _ { 1 } , \dotsc , u _ { K } ]$ are the concept vectors (CAVs) and $V = [ v _ { 1 } ^ { \top } ; \ldots ; v _ { n } ^ { \top } ]$ ] are the activations, with row $v _ { i }$ giving concept coordinates of sample i.

This factorization unifies prior approaches via constraints on (U, V ):

$$
\left\{ \begin{array}{l l} v _ {i} \in \{e _ {1}, \ldots , e _ {K} \}, & \text { K - Means   (ACE)   Ghorbani   et   al.   (2019) }, \\ U ^ {\top} U = I, & \text { PCA   Graziani   et   al.   (2023) }, \\ S \geq 0, U \geq 0, V \geq 0, & \text { NMF   (CRAFT   Fel   et   al.   (2023a;b) }, \\ U \text { free }, V \geq 0, & \text { Semi - NMF   (SNMF)   Trigeorgis   et   al.   (2014) }; \text { Parekh   et   al.   (2024) }, \\ V = \psi (S), \| v _ {i} \| _ {0} \leq s, & \text { Sparse   Autoencoder   (SAE)   Templeton   (2024) }; \text { Pach   et   al.   (2025) }. \end{array} \right.
$$

Columns of U are concept vectors (CAVs), rows of V are per-sample activations. Special cases include PCA (orthogonal bases), NMF (nonnegative factors), SNMF (mixed-sign bases with nonnegative activations), K-Means (one-hot codes), and SAE (encoder-decoder with sparsity).

# 4 METHOD

We begin by formalizing LVLMs as black-box systems with accessible intermediate activations, taking multimodal inputs (text, image) and producing corresponding outputs (text/ set of tokens).

# 4.1 POSITIVE AND NEGATIVE CONCEPTS

We address the limitations of single-object dependence and polysemantic behavior in CoX-LMM by introducing concept-example bags—collections of image patches that serve as positive (conceptpresent) or negative (concept-absent) instances for each automatically discovered concept $c _ { k } \in$ $\{ c _ { 1 } , \ldots , c _ { K } \}$ .

Given an unlabeled image set $\mathcal { T } = \{ I _ { k } \} _ { k = 1 } ^ { N }$ , the LVLM f predicts candidate concepts for each image using a structured prompt (See Appendix B for details): $C ( I _ { k } ) = f ( I _ { k } , \mathrm { p r o m i p t } ) \subseteq \mathcal { V }$ and the global vocabulary is

$$
\mathcal {C} = \bigcup_ {k = 1} ^ {N} C (I _ {k}).
$$

For each $c \in { \mathcal { C } } .$ , we collect supporting images $\Phi ( c ) = \{ I _ { k } \ | \ c \in C ( I _ { k } ) \}$ }. Using SAM Kirillov et al. (2023), a patch operator $\mathcal { P } ( I _ { k } , c ) \in$ {randomcrop ${ ( I _ { k } ) }$ , sam $\left( I _ { k } , c \right) \}$ localizes the region associated with c. The recent visually grounded CBM method Prasse et al. (2025) uses SAM Kirillov et al. (2023) for segmentation-based concept crops to train and explain CBM outputs. In contrast, we leverage SAM to construct concepts for LVLMs and explain LVLM behavior post hoc. Because such patches may include background context, the resulting concept-example bag is a mixture of positives and negatives:

$$
\mathcal {B} (c) = \left\{\mathcal {P} \left(I _ {k}, c\right): I _ {k} \in \Phi (c) \right\} = \mathcal {B} ^ {+} (c) \cup \mathcal {B} ^ {-} (c).
$$

Unlike prior approaches that rely on annotated single-object images, this formulation is weakly supervised and scales to multi-object datasets. For example, the concept “stripes” may emerge from zebras, tigers, or cats, without requiring manual concept labels.

# 4.2 CONCEPT-GUIDED DICTIONARY LEARNING

According to Fel et al. (2023a), most prior concept-expansion methods rely on dictionary learning for concept extraction; CoX-LMM is no exception. The key difference lies in the data passed to the dictionary learning algorithm, which critically affects concept quality Grobrugge et al. (2025); Sun ¨ et al. (2023).

Unlike prior approaches for concept extraction in LVLMs that rely on open-ended token generation and single-token analysis, our method restricts token generation to reduce noise and entanglement. Open-ended generation makes a single token’s hidden representation noisy, since each token is influenced by the entire sequence. This leads to overlapping concepts Templeton (2024) and prevents dictionary elements from representing clean, disentangled semantics. Multi-object images further exacerbate this effect, as activations mix features from different objects.

We address these issues with Concept-Guidance, a contrastive residual extraction scheme. The model is prompted to output either the target concept $c _ { k }$ or $\mathrm { N o } { - c _ { k } }$ , ensuring that activations are aligned with a single concept and enforcing monosemanticity. This binary design yields cleaner, concept-focused residuals.

$$
\begin{array}{l} \text { prompt } _ {c g} = ` ` \text { Does   the   image   contain } c _ {k}? \text { If   yes,   output } \\ c _ {k}; \text { otherwise,   output   No- } c _ {k}. ^ {\prime \prime} \end{array}
$$

For each concept bag $B ( c _ { k } )$ , we query the model with the above prompt and collect residual activations. Following Koh et al. (2020); Alam et al. (2025), we decompose the LVLM into an embedding function g (vision encoder, bridging, decoder attention) and an output function h (projection and softmax). Based on Fel et al. (2023a); Parekh et al. (2024), we analyze the penultimate residual layer (language model.norm). For an image crop $x ^ { c _ { k } } \in B ( c _ { k } )$ and cached prefix $\hat { y } _ { < t }$ , the embedding function produces

$$
a _ {t} ^ {(l)} = g ^ {(l)} (x ^ {c _ {k}}, \text { prompt } _ {c g}, \hat {y} _ {<   t}) \in \mathbb {R} ^ {p}, \tag {1}
$$

where $p$ is the residual dimension Geva et al. (2020). The output function then predicts

$$
\hat {y} _ {t} = h ^ {(l)} (a _ {t} ^ {(l)}). \tag {2}
$$

According to Geva et al. (2020), we can average residuals across tokens in each response to obtain a concept-level embedding without losing semantic meaning.

$$
s _ {m} = \frac {1}{T _ {m}} \sum_ {t = 1} ^ {T _ {m}} a _ {t} ^ {(l)} \in \mathbb {R} ^ {p}, \tag {3}
$$

where $T _ { m }$ is the number of generated tokens for sample m. Collecting M such samples yields

$$
S = \left[ s _ {1}, \dots , s _ {M} \right] \in \mathbb {R} ^ {M \times p}, \tag {4}
$$

which serves as input for concept extraction. Unlike approaches that rely solely on token embeddings, this formulation captures information from multi-token concepts (e.g., hot dog) rather than splitting them into isolated tokens (hot, dog).

Dictionary learning then decomposes S into concept and negation bases:

$$
S \approx V U ^ {T}, \quad V \in \mathbb {R} ^ {M \times 2}, U \in \mathbb {R} ^ {p \times 2}.
$$

Here, $U$ contains basis vectors for $c _ { k }$ and $\mathrm { N o - } c _ { k }$ , while $V$ captures sample activations. When restricting each bag to two bases (concept vs. negation), the per-bag cost reduces to $\mathcal { O } ( M _ { c } p + M _ { c } ^ { 2 } )$ , where $\hat { M _ { c } }$ is the number of samples in $B ( c _ { k } )$ . Over K concepts, the total complexity scales as $\mathcal { O } ( K ( M _ { c } p + M _ { c } ^ { 2 } ) )$ , which remains efficient for large concept sets, consistent with the per-iteration complexity of Semi-NMF Kim & Park (2008); Ding et al. (2010). This contrasts with full NMF, where larger dictionary sizes $( K \gg 2 )$ lead to cubic dependence on $K ,$ , and with Sparse Autoencoders (SAE), where training requires backpropagation through millions of parameters. By reducing each bag to two bases, our approach avoids interference among dictionary atoms and remains scalable for LVLM interpretability.

This formulation contrasts each concept against all others (akin to one-vs-all classification), capturing inter-concept relations without requiring oversized dictionaries. Unlike prior methods that entangle concepts across long sequences, Concept-Guidance yields disentangled, monosemantic residuals suitable for large-scale LVLM interpretability.

# 4.3 POSITIVE CONCEPT VECTOR IDENTIFICATION

Inspired by the visual grounding in Parekh et al. (2024), for each feature $u \in U$ from the dictionary decomposition, we extract the maximum activated crops (MAC)—the top-αMAC image patches that most strongly activate it, i.e., the highest values in the k-th column of the activation matrix $V { : }$

$$
X _ {k, \mathrm{MAC}} = \left\{i \mid v _ {i} ^ {(k)} \text {   is   among   the   top- } \alpha_ {\mathrm{MAC}} \text {   values   of   } v ^ {(k)} \right\}.
$$

Since each concept bag contains both $c _ { k }$ (positive) and $\mathrm { N o } { - c _ { k } }$ (negative) samples, we select the basis as

$$
k ^ {*} = \arg \max _ {j \in \{0, 1 \}} \big | \big \{i \in X _ {k, \mathrm{MAC}} \mid \hat {y} _ {i} = c _ {k} \big \} \big |,
$$

i.e., the one aligned with the majority of positive outputs. We then define $u _ { k ^ { * } } \in \mathbb { R } ^ { p }$ as the concept vector and $X _ { k ^ { * } , \mathrm { M A C } }$ as its supporting crops. Repeating this across all concept bags yields a dictionary $U = [ u _ { 1 } , \dots , u _ { K } ] \in \mathbb { R } ^ { p \times K }$ with visual groundings $X _ { \mathrm { M A C } } = \{ X _ { 1 , \mathrm { M A C } } , \dots , X _ { K , \mathrm { M A C } } \}$ .

For textual grounding, we use the LVLM’s output function directly. Recall that for a residual activation $a _ { t } ^ { ( \bar { l } ) }$ , the model predicts the next token as

$$
y _ {t} = h ^ {(l)} (a _ {t} ^ {(l)}). \tag {2}
$$

Analogously, for each concept feature uk, we compute its token distribution by applying the same output function:

$$
q _ {k} = h ^ {(l)} (u _ {k}) \in \mathbb {R} ^ {| V |}
$$

where |V | denotes the size of the model’s output vocabulary (i.e., the number of distinct tokens in the LVLM). We then select the top-τ tokens $( \mathrm { e } . \mathrm { g } . , 5 0 )$ with the highest scores, remove stopwords and noise, and define the resulting set as $X _ { k , \mathrm { M A T } }$ for concept $u _ { k }$ . Collecting across all concepts yields $X _ { \mathrm { M A T } } = \{ X _ { 1 , \mathrm { M A T } } , \dots , X _ { K , \mathrm { M A T } } \}$ .

# 4.4 CONCEPT ATTRIBUTION AND MULTI-MODAL ALIGNMENT

By taking motivation from Kim et al. (2018), we project residual activations at layer l onto the learned concept subspace $\widehat { U } = [ u _ { 1 } , \dots , u _ { K } ] \in \mathbb { R } ^ { p \times K }$ . At the token level, concept scores are

$$
\alpha_ {j} ^ {(t)} = \cos \_ s i m (u _ {j}, a _ {t} ^ {(l)}),
$$

while at the phrase level, we use the mean-pooled activation $\begin{array} { r } { s _ { m } = \frac { 1 } { T _ { m } } \sum _ { t = 1 } ^ { T _ { m } } a _ { t } ^ { ( l ) } } \end{array}$ to compute

$$
\alpha_ {m, j} = \cos \_ s i m (u _ {j}, s _ {m}).
$$

Thus $a _ { t } ^ { ( l ) }$ provides fine-grained token attribution, whereas $s _ { m }$ captures phrase/sentence-level semantics. The most activated concept is

$$
k ^ {*} = \arg \max _ {j} \alpha_ {m, j},
$$

and its groundings $X _ { \mathrm { M A C } } [ k ^ { * } ]$ (visual) and $X _ { \mathrm { M A T } } [ k ^ { * } ]$ (textual) provide multimodal explanations.

In the text-only mode, this procedure further allows us to assess whether purely textual prompts activate the same feature directions as multi-modal inputs, thereby quantifying multi-modal alignment. This shared projection space links visual and textual semantics, supporting faithful cross-modal interpretability.

# 5 EXPERIMENTS

# 5.1 MODEL AND DATA

We evaluate three recent instruction-tuned LVLMs—Qwen2-VL-7B Wang et al. (2024), Qwen2.5- VL-7B Team (2025b), and Gemma-3n-E4B Team (2025a)—keeping all models frozen to ensure post hoc interpretability. Results for the two Qwen models are reported in Appendix E.

For concept learning, we collect 300 examples per class from ImageNet and MSCOCO, extracting features from the penultimate norm layer, shown to yield high-quality embeddings (Parekh et al., 2024; Kim et al., 2018). Evaluation uses a disjoint set of 50 images per class from the validation splits of ImageNet (1,000 classes) and MSCOCO (10 randomly selected objects).

Unlike CoX-LMM, which requires ImageNet labels and MSCOCO caption tokens during extraction, CGDL is weakly supervised: we collect a large pool of concept examples and retain the top 1,000 most frequent concepts for ImageNet and the top 10 for MSCOCO, with each concept bag capped at 1,600 cropped patches. This setup allows us to study scalability across very different concept set sizes, while keeping comparisons fair against CoX-LMM, which requires ground-truth labels for multiple concepts. Images are resized to 500 pixels in width and cropped into 200 × 200 windows with 0.2 overlap, yielding on average 5–6 crops per image. This makes the effective number of samples per bag comparable to the 300 full images used in CoX-LMM.

We compare CGDL against CoX-LMM Parekh et al. (2024) using two dictionary learning methods: Sparse Autoencoders (SAE) Pach et al. (2025) and Semi-Nonnegative Matrix Factorization (SNMF) Trigeorgis et al. (2014). We also include a simple baseline, SIMPLE, which tests whether residuals with the highest l2-norm align with concepts.

Ground-truth concepts are defined by image class labels, with correctness measured by top-1 alignment. For faithfulness testing, we evaluate on the same 10 MSCOCO objects.

# 5.2 EVALUATION METRICS

Concept discovery can be framed as a special case of dictionary learning. Following the evaluation protocol of Fel et al. (2023a), we first assess the quality of discovered features using three standard metrics: Sparsity (↑), Stability (↓), and Overlap (↓). Based on these results, SNMF emerges as the most suitable method for downstream evaluation.

Scalability is evaluated with both 10 and 1,000 concepts, demonstrating that purity and uniqueness are maintained as the number of concepts grows. Attribution quality Parekh et al. (2024) is measured using CLIPScore and BERTScore, which are standard metrics for text–image and text–text alignment. CLIP Radford et al. (2021b) provides cross-modal contrastive alignment, while BERT Devlin et al. (2019) captures contextual semantics through bidirectional encoding.

Faithfulness is assessed using concept insertion and concept deletion curves Fel et al. (2023a); Kadir et al. (2023), which quantify performance shifts as important concepts are progressively inserted or removed. We report results across the top-1, top-2, and top-3 concepts ranked by their influence on model output.

Finally, qualitative analyses highlight representative extracted concepts and their textual groundings, and illustrate their application to binary classification and text-to-concept alignment.

# 5.3 RESULTS

Table 2 reports quantitative results for concept vectors extracted from Gemma-3n-E4B on ImageNet and MSCOCO. We compare CGDL with CoX-LMM across three dictionary learning settings (SNMF, SAE, SIMPLE). CGDL–SNMF consistently achieves the best performance, with the highest sparsity, lowest stability, and lowest overlap. SAE also benefits from concept guidance, while SIMPLE remains weak with low sparsity and high overlap. These results highlight that concept guidance substantially improves the quality of learned concept vectors, particularly under SNMF Fel et al. (2023a); Parekh et al. (2024).

<table><tr><td rowspan="2">Method</td><td rowspan="2">Dictionary Learning</td><td colspan="3">ImageNet</td><td colspan="3">MSCOCO</td></tr><tr><td>Spars. ↑</td><td>Stab. ↓</td><td>Overlap ↓</td><td>Spars. ↑</td><td>Stab. ↓</td><td>Overlap ↓</td></tr><tr><td rowspan="3">CoX-LMM</td><td>SNMF</td><td>0.96</td><td>0.13</td><td>0.25</td><td>1.00</td><td>0.02</td><td>0.16</td></tr><tr><td>SAE</td><td>0.84</td><td>0.21</td><td>0.27</td><td>0.97</td><td>0.17</td><td>0.28</td></tr><tr><td>SIMPLE</td><td>0.07</td><td>0.79</td><td>0.68</td><td>0.63</td><td>0.91</td><td>0.84</td></tr><tr><td rowspan="3">CGDL</td><td>SNMF</td><td>1.00</td><td>0.02</td><td>0.08</td><td>1.00</td><td>0.00</td><td>0.06</td></tr><tr><td>SAE</td><td>0.90</td><td>0.16</td><td>0.10</td><td>1.00</td><td>0.05</td><td>0.16</td></tr><tr><td>SIMPLE</td><td>0.52</td><td>0.64</td><td>0.67</td><td>0.80</td><td>0.49</td><td>0.54</td></tr></table>

Table 2: Concept vector evaluation on ImageNet and MSCOCO. Higher sparsity is better; lower stability and overlap are better. CGDL–SNMF yields the best results across both datasets.

Table 3 presents attribution results for Gemma-3n-E4B on ImageNet and MSCOCO. We evaluate two aspects of alignment: (i) BERTScore, which measures semantic correspondence between topactivated concept groundings and ground-truth labels, and (ii) CLIPScore, which assesses multimodal consistency between concept groundings and input images. We compare our method (CGDL) against CoX-LMM under three settings: Text-only, where activations are probed via class names; Imageonly, where short visual descriptions are used; and Combined, where the model predicts concept presence (ck vs. UNK). Random baselines correspond to assigning concepts uniformly at random, which yields nearly constant values due to the data distribution. Across datasets and metrics, CGDL with SNMF consistently achieves stronger alignment and outperforms CoX-LMM in all modalities, despite requiring substantially less supervision. This advantage holds across both small-scale (10 concepts, MSCOCO) and large-scale (1,000 concepts, ImageNet) evaluations.

We evaluate the concept attribution ranking using established faithfulness metrics Fel et al. (2023a), namely concept deletion (C-Deletion) and concept insertion (C-Insertion), on the MSCOCO validation set. For each image, the model is prompted to classify the input, and the top-3 activated concepts are identified from the residual embeddings (Sec. 4.4). Attribution faithfulness is then measured as follows:

1. C-Deletion: progressively set to zero the coordinates corresponding to the most influential concept directions, ranked by gradient magnitude with respect to the highest-probability token, and record the drop in output probability.   
2. C-Insertion: start from a zero vector and gradually add concept coordinates in the same order, recording the corresponding probability increase.

Scores are averaged across tokens for each image, then aggregated over the validation set. Results are summarized in Figure 1.

Table 3: Comparison of CGDL and CoX-LMM on CLIPScore and BERTScore across datasets and concept types. Higher metric values indicate better alignment. 

<table><tr><td>Method</td><td>Dataset</td><td>Concept</td><td>Metric</td><td>Random</td><td>Text-only</td><td>Image-only</td><td>Combined</td></tr><tr><td rowspan="4">CGDL</td><td rowspan="2">ImageNet</td><td rowspan="2">1,000</td><td>CLIPScore</td><td> $0.52 \pm 0.04$ </td><td>-</td><td> $0.62 \pm 0.08$ </td><td> $0.67 \pm 0.09$ </td></tr><tr><td>BERTScore</td><td> $0.71 \pm 0.06$ </td><td> $0.78 \pm 0.07$ </td><td> $0.84 \pm 0.08$ </td><td> $0.86 \pm 0.10$ </td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CLIPScore</td><td> $0.48 \pm 0.04$ </td><td>-</td><td> $0.60 \pm 0.06$ </td><td> $0.64 \pm 0.08$ </td></tr><tr><td>BERTScore</td><td> $0.71 \pm 0.01$ </td><td> $0.89 \pm 0.06$ </td><td> $0.91 \pm 0.05$ </td><td> $0.93 \pm 0.07$ </td></tr><tr><td rowspan="4">CoX-LMM</td><td rowspan="2">ImageNet</td><td rowspan="2">1,000</td><td>CLIPScore</td><td> $0.49 \pm 0.04$ </td><td>-</td><td> $0.57 \pm 0.03$ </td><td> $0.58 \pm 0.05$ </td></tr><tr><td>BERTScore</td><td> $0.71 \pm 0.00$ </td><td> $0.74 \pm 0.08$ </td><td> $0.75 \pm 0.06$ </td><td> $0.82 \pm 0.09$ </td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CLIPScore</td><td> $0.51 \pm 0.04$ </td><td>-</td><td> $0.57 \pm 0.10$ </td><td> $0.55 \pm 0.05$ </td></tr><tr><td>BERTScore</td><td> $0.71 \pm 0.01$ </td><td> $0.83 \pm 0.01$ </td><td> $0.79 \pm 0.09$ </td><td> $0.73 \pm 0.11$ </td></tr></table>

CoX-LMM   
![](images/3902538eda0de0cd54e8fc7929ddd96c3de0bf0214ed6170620ce9e3c2a300ed.jpg)

![](images/444d1b3616eb9b119c3b72fbce9ba000e4eaef08eb1579cb327b98a6f9239ede.jpg)

CGDL   
![](images/77cef69fe070e5091848631ffc36ad6552413ee79dcc5bfe3464688708eb1afa.jpg)

![](images/e21ca2b24b49ceda9f3570691cd6518c65bdca6c103d6e4404549d97f50dfd0d.jpg)  
Figure 1: Faithfulness comparison of CoX-LMM and CGDL using concept deletion and insertion. CoX-LMM yields relatively flatter curves with weak separation across ranks, whereas CGDL preserves a clear order (Top-1 >Top-2 >Top-3), with sharper degradation under deletion and stronger recovery under insertion. This shows that CGDL produces more faithful and discriminative concept rankings.

Table 4: Mean ± std. of CLIPScore across binary prompts on MSCOCO-10. Abbreviations: $Q { - } 2 =$ Qwen-2, Q-2.5 = Qwen2.5, G-3n = Gemma-3n. 

<table><tr><td>Prompt</td><td>Q-2</td><td>Q-2.5</td><td>G-3n</td><td>Prompt</td><td>Q-2</td><td>Q-2.5</td><td>G-3n</td></tr><tr><td>P1</td><td>0.57±0.10</td><td>0.65±0.13</td><td>0.62±0.08</td><td>P3</td><td>0.57±0.14</td><td>0.62±0.07</td><td>0.62±0.06</td></tr><tr><td>P2</td><td>0.58±0.07</td><td>0.63±0.11</td><td>0.64±0.08</td><td>P4</td><td>0.59±0.11</td><td>0.63±0.08</td><td>0.64±0.09</td></tr></table>

Figure 2 compares ImageNet concepts extracted by CoX-LMM (left) and CGDL (right). CGDL yields fine-grained, monosemantic representations (e.g., fur, stripes), whereas CoX-LMM produces entangled groundings that mix semantics (e.g., tiger conflated with lion or multiple animals). Figure 3 shows attribution in three settings: (i) binary classification, (ii) open-ended classification, and (iii) text–image alignment. In each case, attribution is explained by retrieving the nearest concept examples to the residual activation, demonstrating robust multimodal alignment.

# 6 ABLATION STUDY

We test multiple variants of the contrastive prompt template (see AppendixB for details). Table 4 shows only minor fluctuations in CLIPScore, indicating robustness to phrasing. Qwen-2 exhibits slightly higher variance than Qwen2.5 and Gemma-3n.

Furthermore, we ablate the SNMF sparsity weight α and dictionary sizes $K > 2$ and find that attribution results remain stable, while CLIP scores decrease for large K. Hence, we select $\alpha = 2 0$ and $K = 2$ , which we find sufficient for attribution. Full results are reported in Appendix F. We also compare SAM with a simpler random-cropping baseline. SAM yields slightly higher BERTScore and CLIPScore, as its segmentation better localizes concepts in raw images. Detailed numerical results are reported in Appendix F, Table 10, and qualitative examples are shown in Figures 7 and 5.

# CONCLUSION

We introduced Concept-Guided Dictionary Learning (CGDL), a weakly supervised framework that enforces monosemanticity and grounds concepts directly within LVLMs, yielding faithful multimodal alignment. CGDL is flexible, efficient, and improves concept quality across dictionary learning methods. Limitations include the lack of hierarchical organization and the requirement that models understand basic language instructions—though this holds for most modern LVLMs. Future work will extend CGDL to capture hierarchical concepts and to evaluate beyond LVLMs. CGDL relies on the target LVLM being able to follow prompts of similar difficulty to those in Appendix B, so its applicability is tied to the LVLM’s prompt-understanding ability.

![](images/46389b4f3d54774a0668948277f70b70e843eb81ce67201e628899b6b46d9200.jpg)

<details>
<summary>text_image</summary>

tiger
kitten
sleeping
tab
kitty
pepper
canine
chili
sandwich
sausage
animal
dog
horse
black
rod
furry
wet
mole
small
wild
resting
striped
lion
tiger
hot
dog
dog
fur
stripe
</details>

Figure 2: Qualitative examples of concept representations extracted from ImageNet. CoX-LMM (left) concepts are often grounded to multiple unrelated or overlapping tokens (e.g., “canine” linked to “hot dog”), reflecting polysemantic vectors. In several cases, concepts mix distinct animals: for example, tiger grounds across multiple concepts, while lion is misrepresented as tiger. In contrast, CGDL (right) discovers fine-grained and monosemantic concepts (e.g., fur, dog, stripes).   
![](images/609b15d2a7eccba991eab6ab2de7dfb154edeb652ce7c27d6e4dd5a520286eb2.jpg)

<details>
<summary>bar</summary>

| Prediction | Concept | Prompt: Identify the object in the image as either a [concept] or a 'thing'. LVLM Output: rabbit | Prompt: Describe the image with one or two phrases. LVLM Output: rabbit | No Image | Concept = rabbit | Prompt: Write the word: [concept] LVLM Output: rabbit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| rabbit | Projection Score | -500 | 0 | 500 | -500 | 0 |
| cat | Projection Score | -500 | 0 | 500 | -500 | 0 |
| (a) | Projection Score | -500 | 0 | 500 | -500 | 0 |
| (b) | Projection Score | -500 | 0 | 500 | -500 | 0 |
| (c) | Projection Score | -500 | 0 | 500 | -500 | 0 |
</details>

Figure 3: Attribution-based explanation using concepts (a) Aligned image-text concepts yield strong feature attribution. (b) An image-only input without concept information still triggers a relevant feature. (c) Text alone activates semantically meaningful features, demonstrating robust multimodal alignment. For token attribution with concepts from the same object, different objects, or abstract concepts, we provide more examples in Appendix E.3, E.5, and E.4. We find that our concepts can attribute a model’s output both to abstract concepts and to objects from similar categories.

Reproducibility Statement We release a reproducible pipeline that requires only a Hugging Face model card, an access token, and a dataset directory with train/val splits. CGDL and CoX-LMM can be run via scripts/run full pipeline.sh and scripts/run full pipeline dl.sh, respectively. Code and configs are shared anonymously at https://anonymous.4open.science/r/xl-vlms-30C1, with installation and usage detailed in the README. All experiments used a single NVIDIA RTX 3090 (24GB) GPU with fixed random seeds.

Ethics Statement. This work poses no direct risks beyond standard interpretability concerns. Our method may reveal biased concepts, which should be handled responsibly. We used an LLM for minor editing; all scientific contributions are our own.

# REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2018. URL https://arxiv.org/abs/1610.01644.   
Hasan Md Tusfiqur Alam, Devansh Srivastav, Md Abdul Kadir, and Daniel Sonntag. Towards interpretable radiology report generation via&nbsp;concept bottlenecks using a&nbsp;multi-agentic rag. In Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part III, pp. 201–209, Berlin, Heidelberg, 2025. Springer-Verlag. ISBN 978-3-031-88713-0. doi: 10.1007/978-3-031-88714-7 18. URL https://doi.org/10.1007/978-3-031-88714-7\_18.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, 2019.   
Chris H.Q. Ding, Tao Li, and Michael I. Jordan. Convex and semi-nonnegative matrix factorizations. IEEE Transactions on Pattern Analysis and Machine Intelligence, 32(1):45–55, 2010.   
Maximilian Dreyer, Jim Berend, Tobias Labarta, Johanna Vielhaben, Thomas Wiegand, Sebastian Lapuschkin, and Wojciech Samek. Mechanistic understanding and validation of large ai models with semanticlens, 2025. URL https://arxiv.org/abs/2501.05398.   
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. URL https://arxiv.org/abs/2209.10652.   
Thomas Fel, Victor Boutin, Mazda Moayeri, Remi Cad ´ ene, Louis Bethune, L \` eo and ´ eol, Mathieu ´ Chalvidal, and Thomas Serre. A holistic approach to unifying automatic concept extraction and concept importance estimation, 2023a. URL https://arxiv.org/abs/2306.07304.   
Thomas Fel, Agustin Picard, Louis Bethune, Thibaut Boissin, David Vigouroux, Julien Colin, R ´ emi ´ Cadene, and Thomas Serre. Craft: Concept recursive activation factorization for explainability. In \` Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2711–2721, June 2023b.   
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. CoRR, abs/2012.14913, 2020. URL https://arxiv.org/abs/2012. 14913.   
Amirata Ghorbani, James Wexler, James Zou, and Been Kim. Towards automatic concept-based explanations. Curran Associates Inc., Red Hook, NY, USA, 2019.   
Mara Graziani, An phi Nguyen, Laura O’Mahony, Henning Muller, and Vincent Andrearczyk. ¨ Concept discovery and dataset exploration with singular value decomposition. In ICLR 2023 Workshop on Pitfalls of limited data and computation for Trustworthy ML, 2023. URL https: //openreview.net/forum?id=iOlYmD1PtC8.   
Arne Grobrugge, Niklas K ¨ uhl, Gerhard Satzger, and Philipp Spitzer. Towards human-understandable ¨ multi-dimensional concept discovery, 2025. URL https://arxiv.org/abs/2503. 18629.   
Sarthak Jain and Byron C. Wallace. Attention is not Explanation. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3543–3556, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1357. URL https://aclanthology. org/N19-1357/.   
Md Abdul Kadir, Amir Mosavi, and Daniel Sonntag. Evaluation metrics for xai: A review, taxonomy, and practical applications. In 2023 IEEE 27th International Conference on Intelligent Engineering Systems (INES), pp. 000111–000124, 2023. doi: 10.1109/INES59282.2023.10297629.

Weitai Kang, Gaowen Liu, Mubarak Shah, and Yan Yan. Segvg: Transferring object bounding box to segmentation for visual grounding, 2024. URL https://arxiv.org/abs/2407.03200.   
Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav), 2018. URL https://arxiv.org/abs/1711.11279.   
Hyunsoo Kim and Haesun Park. Nonnegative matrix factorization based on alternating nonnegativity constrained least squares and active set method. SIAM Journal on Matrix Analysis and Applications, 30(2):713–730, 2008.   
Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. ´ Segment anything, 2023. URL https://arxiv.org/abs/2304.02643.   
Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International conference on machine learning, pp. 5338–5348. PMLR, 2020.   
Jingjing Liu, Nian Wu, Xianchao Xiu, and Jianhua Zhang. Robust orthogonal nmf with label propagation for image clustering. arXiv preprint arXiv:2504.21472, 2025.   
Chuofan Ma, Yi Jiang, Jiannan Wu, Zehuan Yuan, and Xiaojuan Qi. Groma: Localized visual tokenization for grounding multimodal large language models, 2024. URL https://arxiv. org/abs/2404.13013.   
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt, 2023. URL https://arxiv.org/abs/2202.05262.   
Tuomas Oikarinen and Tsui-Wei Weng. Clip-dissect: Automatic description of neuron representations in deep vision networks, 2023. URL https://arxiv.org/abs/2204.10965.   
Tuomas Oikarinen, Subhro Das, Lam M. Nguyen, and Tsui-Wei Weng. Label-free concept bottleneck models, 2023. URL https://arxiv.org/abs/2304.06129.   
Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. arXiv preprint arXiv:2504.02821, 2025.   
Jayneel Parekh, Pegah Khayatan, Mustafa Shukor, Alasdair Newson, and Matthieu Cord. A conceptbased explainability framework for large multimodal models. arXiv preprint arXiv:2406.08074, 2024.   
Katharina Prasse, Patrick Knab, Sascha Marton, Christian Bartelt, and Margret Keuper. DCBM: Data-efficient visual concept bottleneck models. In International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=BdO4R6XxUH.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. CoRR, abs/2103.00020, 2021a. URL https://arxiv.org/abs/2103.00020.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021b.   
Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618–626, 2017.

Erik Strumbelj and Igor Kononenko. Explaining prediction models and individual predictions with feature contributions. Knowledge and Information Systems, 41:647–665, 2014. URL https: //api.semanticscholar.org/CorpusID:2449098.   
Ao Sun, Pingchuan Ma, Yuanyuan Yuan, and Shuai Wang. Explain any concept: Segment anything meets concept-based explanation, 2023. URL https://arxiv.org/abs/2305.10289.   
Gemma Team. Gemma 3n. 2025a. URL https://ai.google.dev/gemma/docs/ gemma-3n.   
Qwen Team. Qwen2.5-vl, January 2025b. URL https://qwenlm.github.io/blog/ qwen2.5-vl/.   
Adly Templeton. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Anthropic, 2024.   
George Trigeorgis, Konstantinos Bousmalis, Stefanos Zafeiriou, and Bjorn W. Schuller. A deep semi- ¨ nmf model for learning hidden representations. In Proceedings of the 31st International Conference on International Conference on Machine Learning - Volume 32, ICML’14, pp. II–1692–II–1700. JMLR.org, 2014.   
Theophane Vallaeys, Mustafa Shukor, Matthieu Cord, and Jakob Verbeek. Improved baselines for ´ data-efficient perceptual augmentation of llms. In European Conference on Computer Vision, pp. 369–387. Springer, 2024.   
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.   
Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, and Mark Yatskar. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification, 2023. URL https://arxiv.org/abs/2211.11158.   
Hao Zhang, Hongyang Li, Feng Li, Tianhe Ren, Xueyan Zou, Shilong Liu, Shijia Huang, Jianfeng Gao, Lei Zhang, Chunyuan Li, and Jianwei Yang. Llava-grounding: Grounded visual chat with large multimodal models, 2023. URL https://arxiv.org/abs/2312.02949.   
Kaichen Zhang, Yifei Shen, Bo Li, and Ziwei Liu. Large multi-modal models can interpret features in large multi-modal models, 2025. URL https://arxiv.org/abs/2411.14982.

# Appendix

# A MONOSEMANTIC VS. POLYSEMANTIC REPRESENTATIONS

A central challenge in interpreting large vision–language models (LVLMs) lies in superposition and feature entanglement in high-dimensional residual streams (Elhage et al., 2022). Here, features can be understood as vector directions in activation space that encode candidate concepts. Ideally, such vectors should be monosemantic-each aligned with a single interpretable concept. In practice, however, LVLMs often learn polysemantic vectors, where multiple, semantically unrelated concepts activate a single direction.

For instance, in Fig. 4(a), a feature $f _ { 1 }$ responds to both “cat” and “chair.” Such overlap can arise when these concepts frequently co-occur in training data, leading the model to conflate them. When $f _ { 1 }$ activated, it is therefore ambiguous whether the cause was the presence of a cat, a chair, or both. This ambiguity breaks the one-to-one mapping between features and concepts, making attribution unreliable. In this paper, we investigate how to extract monosemantic concept vectors from the polysemantic features that neurons of a model fire during prediction.

Monosemantic features (Fig. 4b) provide a clean relation to concepts: e.g., A concept vector is an approximation of monosemantic features.

![](images/f1f1aa7c8832962b815de770b0b37b848fa36a829becb50807ce1aef1c3a4855.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a)
        f1["f_1"] --> cat["cat"]
        f2["f_2"] --> human["human"]
        f3["f_3"] --> table["table"]
        f4["f_4"] --> chair["chair"]
    end
    subgraph (b)
        f1f["f_1"] --> catf["cat"]
        f2f["f_2"] --> rabbit["rabbit"]
        f3f["f_3"] --> tablef["table"]
        f4f["f_4"] --> humanf["human"]
    end
```
</details>

Figure 4: Comparison between prior concept decomposition methods and our proposed approach. (a) Previous methods (e.g., Parekh et al. (2024)) often produce polysemantic features $\left( \mathrm { e . g . , } f _ { 3 } \right.$ activates for “chair” and “cat”). (b) Our method encourages monosemantic features $( \mathrm { e . g . } , f _ { 1 } \mathrm { f o r \ ^ { \ast } c a t , ^ { \ast } } $ $f _ { 3 }$ for “table”).

Apart from a limited number of studies Templeton (2024); Pach et al. (2025), most existing models do not offer disentangled representations, and tools for analyzing and extracting monosemantic features remain scarce. This paper addresses this gap through CGDL.

# B PROMPTS

Concept generation prompt. We use the following instruction to extract candidate concept text from images: “Identify every visible object, item, concept, and pattern in the image. Output only a single-word, comma-separated list. No explanations or sentences.” This prompt generates concept tokens directly from the dataset without requiring manual labels, thereby creating a concept-to-image mapping.

# Concept-Guidance prompt

• P1: Detect whether the image contains $c _ { k }$ . If yes, return $c _ { k } ;$ otherwise, return UNK.   
• P2: Does the image contain $c _ { k } ?$ If yes, output $c _ { k } )$ ; otherwise, output $\mathrm { N o - } c _ { k }$ .

• P3: Is there a clear instance of $c _ { k }$ in this image? Reply with $c _ { k }$ or thing, nothing else.   
• P4: Recognize whether the concept $c _ { k }$ is present in the picture. Use only $c _ { k }$ or UNK as your answer.

# C MODELS

# C.1 GEMMA-3N E4B-IT TEAM (2025A)

Gemma-3n E4B-IT (4-billion-parameter model) is trained using a nested subnetwork approach based on the Matryoshka Transformer (MatFormer) architecture. Each Transformer layer supports multiple capacity levels, implemented as top-left submatrices of full-size weight tensors. The model is instruction-tuned on a mixture of multilingual and multimodal data, including text, images, audio, and video inputs. It is trained autoregressively to predict the next token, with a maximum context length of 32k tokens.

# C.2 QWEN2.5-VL-7B-INSTRUCTTEAM (2025B) AND QWEN2-VL-7B-INSTRUCT WANG ET AL. (2024)

Qwen2.5-VL-7B-Instruct is a 7-billion-parameter multimodal instruction-tuned language model designed for vision-language tasks. It accepts both text and image inputs and generates text outputs. The model supports a maximum context length of 8192 tokens, enabling it to handle long conversational and reasoning scenarios. Training is performed via supervised instruction tuning on paired text-image datasets. The model is optimized with an autoregressive next-token prediction objective using cross-entropy loss, conditioning on both textual and visual contexts. Large-scale distributed training with mixed precision improves efficiency. This approach enhances the model’s instruction-following capabilities and generalization to diverse vision-language tasks.

# D DATASET DESCRIPTIONS

We evaluate our models on four datasets: ImageNet, MSCOCO (10 classes), CIFAR100, and DTD (Describable Textures Dataset).

• ImageNet: A large-scale visual dataset with 1000 object categories and high-resolution images, commonly used for image classification tasks.   
• MSCOCO (10 classes): A subset of the MSCOCO dataset with 10 object categories, featuring complex scenes and multiple annotated objects per image.

<table><tr><td>Dataset</td><td>Total Classes</td><td>Train Samples/Class</td><td>Val Samples/Class</td></tr><tr><td>ImageNet</td><td>1000</td><td>300</td><td>50</td></tr><tr><td>MSCOCO (10)</td><td>10</td><td>300</td><td>50</td></tr><tr><td>CIFAR100</td><td>100</td><td>300</td><td>100</td></tr><tr><td>DTD</td><td>47</td><td>95</td><td>24</td></tr></table>

Table 5: Overview of datasets used for training and validation, including the number of classes and samples per class.

# E RESULTS

# E.1 QWEN 2.0-VL-7B

Table 6 reports the performance of CGDL and CoX-LMM on four benchmark datasets. We evaluate alignment using CLIPScore (CS) and BERTScore (BS), where higher is better.

Across all datasets, CGDL consistently outperforms CoX-LMM. Notably, on MSCOCO, CGDL achieves the strongest gains: BS improves from 0.82 (CoX-LMM) to 0.94, and CS improves from

Table 6: Comparison of CGDL and CoX-LMM across datasets. Metrics: CS = CLIPScore, BS = BERTScore. Higher is better. The best non-random results are bolded. 

<table><tr><td>Method</td><td>Dataset</td><td>#C</td><td>Metric</td><td>Rand</td><td>Text</td><td>Img</td><td>Comb</td></tr><tr><td rowspan="8">CGDL</td><td rowspan="2">ImageNet</td><td rowspan="2">1k</td><td>CS</td><td>0.53 ± 0.02</td><td>-</td><td>0.62 ± 0.07</td><td>0.63 ± 0.08</td></tr><tr><td>BS</td><td>0.80 ± 0.05</td><td>0.86 ± 0.06</td><td>0.88 ± 0.08</td><td>0.86 ± 0.09</td></tr><tr><td rowspan="2">CIFAR100</td><td rowspan="2">100</td><td>CS</td><td>0.51 ± 0.02</td><td>-</td><td>0.63 ± 0.04</td><td>0.63 ± 0.05</td></tr><tr><td>BS</td><td>0.83 ± 0.08</td><td>0.86 ± 0.07</td><td>0.87 ± 0.08</td><td>0.91 ± 0.08</td></tr><tr><td rowspan="2">DTD</td><td rowspan="2">47</td><td>CS</td><td>0.53 ± 0.06</td><td>-</td><td>0.63 ± 0.05</td><td>0.62 ± 0.05</td></tr><tr><td>BS</td><td>0.74 ± 0.04</td><td>0.84 ± 0.06</td><td>0.83 ± 0.07</td><td>0.87 ± 0.06</td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CS</td><td>0.52 ± 0.03</td><td>-</td><td>0.64 ± 0.06</td><td>0.62 ± 0.06</td></tr><tr><td>BS</td><td>0.82 ± 0.02</td><td>0.88 ± 0.06</td><td>0.89 ± 0.05</td><td>0.94 ± 0.08</td></tr><tr><td rowspan="8">CoX-LMM</td><td rowspan="2">ImageNet</td><td rowspan="2">1k</td><td>CS</td><td>0.53 ± 0.03</td><td>-</td><td>0.54 ± 0.03</td><td>0.53 ± 0.05</td></tr><tr><td>BS</td><td>0.82 ± 0.04</td><td>0.82 ± 0.03</td><td>0.84 ± 0.04</td><td>0.86 ± 0.09</td></tr><tr><td rowspan="2">CIFAR100</td><td rowspan="2">100</td><td>CS</td><td>0.53 ± 0.04</td><td>-</td><td>0.53 ± 0.06</td><td>0.53 ± 0.05</td></tr><tr><td>BS</td><td>0.78 ± 0.06</td><td>0.84 ± 0.06</td><td>0.80 ± 0.03</td><td>0.73 ± 0.07</td></tr><tr><td rowspan="2">DTD</td><td rowspan="2">47</td><td>CS</td><td>0.51 ± 0.05</td><td>-</td><td>0.54 ± 0.05</td><td>0.52 ± 0.05</td></tr><tr><td>BS</td><td>0.80 ± 0.07</td><td>0.84 ± 0.06</td><td>0.83 ± 0.07</td><td>0.77 ± 0.07</td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CS</td><td>0.53 ± 0.03</td><td>-</td><td>0.57 ± 0.04</td><td>0.58 ± 0.06</td></tr><tr><td>BS</td><td>0.82 ± 0.04</td><td>0.83 ± 0.01</td><td>0.83 ± 0.05</td><td>0.82 ± 0.03</td></tr></table>

0.58 to 0.64. Similarly, on CIFAR100, the BS of CoX-LMM drops to 0.73 in the combined setting, while CGDL achieves a significantly higher 0.91. These results highlight the robustness of CGDL in both low- and high-concept regimes.

# E.2 QWEN 2.5-VL-7B

Table 7 shows results for the Qwen2.5 backbone. Again, CGDL achieves the strongest improvements across all datasets. In particular, on MSCOCO, CGDL improves BS from 0.90 to 0.95 in the combined setting. On CIFAR100, CGDL reaches 0.92, compared to 0.87 with CoX-LMM. These consistent gains indicate that CGDL scales effectively from small (10 concepts) to large-scale (1k concepts) benchmarks.

# E.3 POSTHOC CONCEPT EXPLANATIONS FOR LVLMS

Unlike the existing methods, which can’t provide any token-level posthoc explanation, our method provides token-level explanations in autoregressive Large Vision-Language Models (LVLMs). We present qualitative examples in Figure 7 5 and 8 for Qwen2.5-VL-7B. Figure 5 results belong to the experiemnt wsing SAM as a localizer mentioned in 4.1 , while Figure 7 and 8 present the concept attrribution example using randdom cropping localizer during concept. Each example uses a structured 2 × 2 image grid that intentionally makes the prediction task more challenging while encouraging the model to produce structured, multi-object descriptions.

On the left of each example, we show the input image (the 2 × 2 grid) together with the prompt used for generation. Directly below, we display the LVLM’s textual output produced for this visual input.

On the right, we visualize token-wise concept activations. For each generated token, we extract its penultimate-layer embedding and compute its cosine similarity to all learned concept vectors. For token-to-word mapping, we consider only the first produced token embedding for that position when calculating the cosine distance to the concept vectors. We then identify the top-2 most activated concepts (from left to right) corresponding to that token.

The right-hand panel contains:

• a concept grid showing, for each token, the two highest-scoring concepts;

<table><tr><td>Method</td><td>Dataset</td><td>#C</td><td>Metric</td><td>Rand</td><td>Text</td><td>Img</td><td>Comb</td></tr><tr><td rowspan="8">CGDL</td><td rowspan="2">ImageNet</td><td rowspan="2">1k</td><td>CS</td><td>0.51 ± 0.06</td><td>-</td><td>0.63 ± 0.06</td><td>0.64 ± 0.05</td></tr><tr><td>BS</td><td>0.82 ± 0.04</td><td>0.87 ± 0.01</td><td>0.88 ± 0.07</td><td>0.87 ± 0.07</td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CS</td><td>0.53 ± 0.05</td><td>-</td><td>0.64 ± 0.07</td><td>0.64 ± 0.08</td></tr><tr><td>BS</td><td>0.83 ± 0.06</td><td>0.89 ± 0.07</td><td>0.90 ± 0.06</td><td>0.95 ± 0.08</td></tr><tr><td rowspan="2">CIFAR100</td><td rowspan="2">100</td><td>CS</td><td>0.50 ± 0.07</td><td>-</td><td>0.62 ± 0.05</td><td>0.64 ± 0.06</td></tr><tr><td>BS</td><td>0.80 ± 0.08</td><td>0.88 ± 0.06</td><td>0.88 ± 0.09</td><td>0.92 ± 0.07</td></tr><tr><td rowspan="2">DTD</td><td rowspan="2">47</td><td>CS</td><td>0.51 ± 0.06</td><td>-</td><td>0.63 ± 0.08</td><td>0.63 ± 0.07</td></tr><tr><td>BS</td><td>0.79 ± 0.07</td><td>0.87 ± 0.07</td><td>0.85 ± 0.09</td><td>0.89 ± 0.08</td></tr><tr><td rowspan="8">CoX-LMM</td><td rowspan="2">ImageNet</td><td rowspan="2">1k</td><td>CS</td><td>0.51 ± 0.05</td><td>-</td><td>0.56 ± 0.04</td><td>0.55 ± 0.06</td></tr><tr><td>BS</td><td>0.81 ± 0.06</td><td>0.83 ± 0.04</td><td>0.85 ± 0.06</td><td>0.86 ± 0.03</td></tr><tr><td rowspan="2">CIFAR100</td><td rowspan="2">100</td><td>CS</td><td>0.53 ± 0.07</td><td>-</td><td>0.58 ± 0.06</td><td>0.57 ± 0.04</td></tr><tr><td>BS</td><td>0.78 ± 0.07</td><td>0.84 ± 0.06</td><td>0.85 ± 0.05</td><td>0.87 ± 0.06</td></tr><tr><td rowspan="2">MSCOCO</td><td rowspan="2">10</td><td>CS</td><td>0.52 ± 0.03</td><td>-</td><td>0.60 ± 0.03</td><td>0.60 ± 0.02</td></tr><tr><td>BS</td><td>0.81 ± 0.04</td><td>0.88 ± 0.04</td><td>0.90 ± 0.04</td><td>0.90 ± 0.07</td></tr><tr><td rowspan="2">DTD</td><td rowspan="2">47</td><td>CS</td><td>0.53 ± 0.03</td><td>-</td><td>0.56 ± 0.03</td><td>0.56 ± 0.06</td></tr><tr><td>BS</td><td>0.81 ± 0.04</td><td>0.83 ± 0.05</td><td>0.82 ± 0.06</td><td>0.88 ± 0.05</td></tr></table>

Table 7: Comparison of CGDL and CoX-LMM across datasets using CLIPScore (CS) and BERTScore (BS). Best non-random results are bolded.

• a concept bank in the form of a row of thumbnails, starts with a similarity bar where it shows the cosine similarity between the token embedding and the corresponding concept vector;   
• a short textual grounding label above each concept bank, summarizing the semantic meaning of the discovered concept.

We find that SAM and the random localizer perform similarly in attribution ranking: both methods correctly attribute the concept images. The main difference is that SAM yields better visualizations for concepts such as hot-dog, beaver, and bear in the concept bank. Moreover, in Figure 6, we show qualitative examples of explanations produced by the baseline CoX-LMM to highlight its limitations and how our method improves the ranking of similar concepts. While the CoX-LMM concept vector often shows high cosine similarity with the token activations, the corresponding concept bank contains many different object types. This indicates that multiple concept signals are entangled in a single vector, i.e., the concept vectors are polysemantic. Such explanations are less useful because we cannot reliably map the model output to a specific, human-interpretable concept. In addition, some textual concepts associated with the concept bank are not clearly related to the underlying visual patterns.

# E.4 CONCEPT VECTORS ARE GENERALIZABLE TO RELATED OBJECTS

To study the robustness and generalization of our concept-based explanations, we also analyze cases where the object-specific concept is missing in the concept dictionary. In Figure 9, we provide qualitative examples showing how the LVLM aligns its predictions with the closest available concepts, even when the extracted concept set belongs to different object classes than the input images.

As in the previous section, E.3, each example contains a structured 2 × 2 input grid. On the left side of each example, we show the input image, the prompt, and the LVLM’s generated output. Here, the learned concepts originate from different object categories than the input images. Despite this mismatch, the LVLM often activates semantically related concepts whose attributes partially overlap with the visual content (e.g., “stripe-like patterns”, “ texture”, “fur-like appearance”).

Prompt: What are the main objects in each   
![](images/e2548690221181d9b7fd09a81871ed498ac2d1868b5dbeef265da14668b27e35.jpg)

<details>
<summary>natural_image</summary>

Collage of four animal photos: a tiger in a chain-link fence, a cat sleeping on a surface, a horse-drawn carriage with people, and a young dog outdoors (no text or symbols visible)
</details>

LVLM Prediction: , , , tiger cat horse cart dog

![](images/afe0d179c613c3c9c069eb5ab5219976c6f5e9f5e03cc8aa38c8b9b7a20aafc6.jpg)

<details>
<summary>bar</summary>

| Category | Subcategory | Value |
|---|---|---|
| 'tiger' | tiger | 0.352 |
| 'tiger' | zel, za, bra | 0.251 |
| 'cat' | tiger | 0.103 |
| 'cat' | category, kat, dog | 0.103 |
| 'horse cart' | animal, horse | 0.110 |
| 'horse cart' | hot, dog | 0.065 |
| 'dog' | cat, animal, photo, dogs | 0.078 |
| 'dog' | hot, dog | 0.074 |
</details>

Prompt: What are the main objects in each grid?   
![](images/3a6d4b94eef980973de971e17f3cb68fd415176faf87bd8e639de9185bfb8305.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature scenes: a dog wearing a vest, a rabbit sitting on grass, a sandwich with food, and a zebra in a dark jacket (no text or symbols visible)
</details>

LVLM Prediction: , , , dog hot dog rabbit zebra

![](images/ae88ab987caf64988032e52220fd2827eb9b67f9bc8d4584308fc9179fe52a0c.jpg)

<details>
<summary>bar</summary>

| Category | Image Count | Value |
|---|---|---|
| "dog" | 0.238 | 1.0 |
| "hot dog" | 0.311 | 1.0 |
| "rabbit" | 0.197 | 1.0 |
| "zebra" | 0.245 | 1.0 |
| cat, animal, photo, dogs | 0.190 | 1.0 |
| hot, dog | 0.133 | 1.0 |
| rabbit | 0.070 | 1.0 |
| zel, za, bra | 0.140 | 1.0 |
| tiger | 0.140 | 1.0 |
</details>

Prompt: What are the main objects in each grid?   
![](images/35c7fd6c2f9dd457d52adec0966fd2d9a2477830dbe5ef76b67283639c8c73d8.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature scenes: a tiger in snowy weather, a dog walking on snow, a water droplet floating, and a row of horse-drawn buildings under a clear sky.
</details>

LVLM Prediction: , , , tiger beaver dog horse

![](images/404ef306e0cb8e2131c0f373aaa7b2630de99c128100aae67d8e1b27de241b4b.jpg)

<details>
<summary>bar</summary>

| Category | 'tiger' | 'beaver', 'bear', 'zez', 'dog', 'horse' |
|---|---|---|
| tiger | 0.349 | 0.351 |
| beer, brown, bearer, animal, baby | 0.244 | 0.131 |
| zel, za, bra | 0.131 | 0.131 |
| animal, horse | 0.110 | 0.046 |
| zel, za, bra | 0.046 | 0.046 |
</details>

Figure 5: Left: input 2 × 2 grid, prompt, and LVLM output. Right: token-wise top-2 concept activations, cosine-similarity bars, and textual grounding and visual grounding. Example using SAM as an object localizer during concept extraction.

# E.5 EXPLANATION WITH ABSTRACT CONCEPTS

We explored how abstract concepts relate to an LVLM’s outputs using ImageNet validation examples. In Figure 10, we present three cases. We found that our explanation method captures clear relationships between predicted tokens and related abstract concepts. For instance, macaw shows high similarity to colorful concepts; ladybug and spotted dog (Dalmatian) show high similarity to polka-dot; and jellyfish shows high similarity to skin and soft concepts. These results shed light on how the model may internally perform abstract reasoning when predicting a token.

Prompt: What are the main objects in each grid?   
![](images/14c4ecb0ea4cdf264c8c2c978bc107e1af31a79c544ac0237a0fa4100b59fdc0.jpg)

<details>
<summary>natural_image</summary>

Four-panel collage showing animals: a squirrel, a baby kitten, a tiger, and a zebra with its feet (no text or symbols)
</details>

LVLM Prediction: , , , squirrel kitty tiger zebra

![](images/a989d85a13f3edc0e2bdb632c26c7bd1e86e700e8056f643b7526a5f23570e8f.jpg)

Prompt: What are the main objects in each grid?   
![](images/f519beb2892b181f4234d6937c6dd4a8d405e47efcca942d9be62984a8e83a8f.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature scenes: a hand holding a stuffed food item, a zebra in a grassy field, two bears walking on grass, and a small rabbit emerging from water.
</details>

LVLM Prediction: , , , hotdog zebra bears rabbit

![](images/067ac6c7980f487be2fe688907d856533f6b3d514c6aa385c8691ffdc01724a0.jpg)

<details>
<summary>heatmap</summary>

| Category | 'hotdog' | 'zebra' | 'bears' | 'rabbit' |
|---|---|---|---|---|
| man, bearer, baby, mon, bo | 0.467 | 0.383 | 0.225 | 0.327 |
| animal, polar, grass, coastal, land | 0.254 | 0.305 | 0.218 | 0.322 |
The image contains two rows of visual images (image types) and one row of corresponding numerical values (e.g., 0.467 for 'hotdog'). The text in the top row is 'animal, polar, grass, coastal, land'. The bottom row contains 'woodland, animal, forest, hare, sandy' and 'animal, dirt, polar, grass, sign'. All images are labeled with the same axis names.
</details>

Prompt: What are the main objects in each grid?   
![](images/0aef6f1305b517d1f8a6fe6d763730f5079f604ac590137138b0589534b2a19d.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing animals: a brown bear, a purple beaver, a black dog, and a gray cat sleeping on a surface (no text or symbols)
</details>

LVLM Prediction: , , , bear dog dog cat

![](images/a1d4181f4d319dbea5a3d5ed20a0185658ccc95fda2e5947eb4c00b4160a5d07.jpg)  
Figure 6: Left: input 2 × 2 image grid, prompt, and LVLM output. Right: token-wise top-2 concept activations, cosine-similarity bars, and textual/visual grounding obtained with the baseline CoX-LMM. SAM is used as an object localizer during concept extraction.

# E.6 GROUNDING LIMITATION

Although our concept-based attribution method generally provides coherent visual and textual grounding, we observe an important failure case when analyzing images containing a beaver property. Figure 11 illustrates this phenomenon.

Shifted textual grounding in some examples. Textual grounding is sometimes slightly shifted (Figure 11) or offset from the intended semantic meaning (e.g., the ”cat” concept is grounded as ”dog,” ”category,” and ”kat,” and for ”beaver,” it shifted to the tokens ”based” and ”prediction”), while the visual grounding is consistent with the image of a ”cat” and ”beaver.”

This discrepancy suggests that while the concept vectors are visually stable and reliably activated across different images, the mapping from concept vectors to text tokens remains sensitive to local variations in the LVLM’s decoder distribution. As a result, text grounding may deviate slightly even when image grounding is fully correct. Overall, however, the image-side concept activations in our concept-based examples remain remarkably uniform and consistent across all inputs.

![](images/735235c5010d0bc21733ac9bc939889204144897643e6e382edd28e3b749bd59.jpg)  
Figure 7: Example using random cropping as an object localizer during concept extraction.

# F ABLATION STUDY

Dictionary size ablation on Gemma-3n. We perform an ablation on the dictionary size K (number of atoms) (Table 8) using Gemma-3n. Dictionaries are learned on ImageNet training data and evaluated on the ImageNet validation split from the same five object classes (chosen to reduce computation). For each K, we measure CLIPScore and BERT scores for image–text alignment and BERTScore for semantic similarity of the concept phrases.

SNMF α ablation on Gemma-3n. We also ablate the SNMF sparsity weight α while keeping the dictionary size fixed. Larger α promotes sparser and more selective atoms, while smaller α yields denser activations. We train on ImageNet training images from the same five classes and evaluate on the corresponding validation split, reporting CLIPScore and BERTScore for each α. As shown in

Prompt: What are the main objects in each grid?   
![](images/aa23a57d44da5dd1aec5ee119b7d4164360636983889ba81cd509a371b0b00c5.jpg)

<details>
<summary>natural_image</summary>

Four-panel collage showing animals: a sandwich, a bear, a rabbit, and a tiger, all with no visible text or symbols.
</details>

LVLM Prediction: , , , hotdog bear rabbit tiger

![](images/fc17a9104a70f6383aa2aebdcc2f6e3b4d8a3da5860ee032cf68d1599660f5d6.jpg)

<details>
<summary>bar</summary>

| Category | Item | Value |
|---|---|---|
| "hotdog" | dog, hot | 0.286 |
| "hotdog" | dogs, photo, animal, cat | 0.204 |
| "bear" | beer, ted, bearer, baby, animal | 0.147 |
| "bear" | tiger | 0.129 |
| "rabbit" | rabbit | 0.231 |
| "rabbit" | animal, horse | 0.117 |
| "tiger" | tiger | 0.246 |
| "tiger" | za, zel, bra, zee | 0.137 |
</details>

Prompt: What are the main objects in each grid?   
![](images/9c7a7f48512e2b6cd03d3d80eedb03714c81ee3232fd3e962c94a4c8310e6ac7.jpg)

<details>
<summary>natural_image</summary>

Collage of four animal photos: a horse-drawn carriage, a bear with water, a German Shepherd dog, and a rabbit on grass (no text or symbols)
</details>

LVLM Prediction: , , , horse grizzlybears dog rabbit

![](images/dbb590ef0f68387c49c3bb888cec1baff256ecf3946a7de9d67d179665ce5c56.jpg)

<details>
<summary>bar</summary>

| Category | Animal | Zebra | Zebra zebra |
| :--- | :--- | :--- | :--- |
| "horse" | 0.292 | 0.205 | 0.205 |
| "grizzlybears" | 0.137 | 0.109 | 0.109 |
| "dog" | 0.130 | 0.129 | 0.129 |
| "rabbit" | 0.181 | 0.044 | 0.044 |
</details>

Prompt: What are the main objects in each grid?   
![](images/fab03da2d35e4f1b72478d45f495311535dfa43f68a1adc8fd0da62d46ef45d0.jpg)

<details>
<summary>natural_image</summary>

Four-panel image collage showing different animal animals: squirrel, baby kitten, tiger, and zebra (no text or symbols)
</details>

LVLM Prediction: , , , squirrel kitten tiger zebra

![](images/cb3af03462deaa1b3e23beca1f5873c3177ba876615ad08230a2d5c97a451b68.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| "squirrel" | 0.225 |
| "kitten" | 0.118 |
| "tiger" | 0.280 |
| "zebra" | 0.353 |
| "rabbit" | 0.198 |
| "dog, category, kat" | 0.094 |
| "tiger" | 0.101 |
| "za, zel, bra, zee" | 0.066 |
The image displays a grid of small images representing different animals and pet types, with some images in the top row showing the same visual content (e.g., cats, zebrafish, zebra). The text labels above each bar indicate the category name.
</details>

Figure 8: More examples using random localizer

Table 9, increasing α gives only a slight improvement and the overall differences are small. Based on this study, we use α = 20 in all main experiments.

SAM vs. non-SAM localization ablation. We further ablate the image localization step by comparing SAM-based region proposals with a non-SAM baseline (random/local crops), while keeping the LVLM (Gemma-3n) and dictionary settings fixed. Both variants are trained on ImageNet training images and evaluated on the validation split of the same five classes. We report CLIPScore and BERTScore to quantify concept quality. As shown in Fig. ??, the two approaches achieve similar quantitative performance, but SAM produces cleaner and more visually coherent concept exemplars for some classes, improving qualitative interpretability.

Layer Ablation We extract concept vectors from different normalization layers of Gemma-3n and report their BERTScore and CLIPScore in Figure 12. We observe that the CLIP scores follow the same trend as in prior work: they are generally higher for deeper layers (Parekh et al., 2024). This is expected, since deeper layers contain more global image features, and CLIPScore primarily measures global image–text similarity rather than fine-grained local details.

![](images/18e4f72b1a6c0dddaaf457e15fc415c1e9103553d3e4e3d388857580c9027b09.jpg)  
Figure 9: Concept-Mismatch Analysis. When the exact object’s concept does not exist in the dictionary, the LVLM aligns its token embeddings with the most semantically related available concepts.

In contrast, the BERT scores do not increase monotonically with depth. Instead, they are high for some layers and low for others. This is reasonable because BERTScore only compares the text descriptions of the concepts. Text is discrete and does not decompose into “low-level” vs. “high-level” visual features in the same way as image representations, so BERTScore is largely agnostic to layer depth. A high BERTScore for a layer indicates that its concept vectors yield coherent and semantically rich textual concepts, whereas a low BERTScore suggests that the corresponding layer provides a poorer representation of the underlying concepts.

Prompt: What are the main objects in each grid?   
![](images/c9d390c64905fc086c402afca460a70fd57c7fa3360946088345272191b67dc5.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature photos: a colorful macaque perched on a branch, a blue circular object with Earth-like surface, a ladybug perched on a branch, and underwater gear (no text or symbols)
</details>

LVLM Prediction: , , , parrot jellyfish ladybug catfish   
![](images/abd979b693269a61be3d6def50d7a2b35733f331ea61745e8ef35943da6b37fa.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| "parrot" | 0.303 |
| skinned | 0.215 |
| soft | 0.191 |
| dotted, aka, polka | 0.276 |
| blackjack | 0.264 |
| catfish | 0.281 |
The bar chart displays the performance scores for each category under different image conditions. The color-coded images represent the visual data for each category.
</details>

Prompt: What are the main objects in each grid?   
![](images/ca3fa259dcfb4c26f13844ea7917155c75b5aa163266cf725c6349480bf40136.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a colorful parrot, a Dantler dog walking, and an aquarium with aquatic plants (no text or symbols)
</details>

LVLM Prediction: , , , parrot dalmatian shelves octopus   
![](images/60de1ac28a886279f60d5baa455cbc5de8a01009870da9616d16c2bc0b5d76e5.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| "parrot" | 0.327 |
| dalmatian* | 0.245 |
| shelves* | 0.171 |
| octopus* | 0.301 |
| colored, colorful, coll, rainbow | 0.80 |
| organic, based | 0.282 |
| dotted, aka, polka | 0.80 |
| red, led, ted, patterned | 0.230 |
| blackjack | 0.80 |
| blackjack | 0.80 |
| red, text | 0.296 |
</details>

Prompt: What are the main objects in each grid?   
![](images/afb8644c4304e60d917107ca160b140888925f23d82f93d53389634d4fe754f9.jpg)

<details>
<summary>natural_image</summary>

Collage of four nature scenes: a jellyfish in blue water, a potted goat, a parrot with colorful wings, and a bird perched on rocks (no text or symbols)
</details>

LVLM Prediction: , , , jellyfish fish dalmatian dog macaws   
![](images/93b13dd75c847ca0a69a13f05f97f19440c719993c3b90960ed6c5c243989264.jpg)

<details>
<summary>bar</summary>

| Category | Color/Pattern | Value |
|---|---|---|
| "jellyfish" | colored, colorful, coll, rainbow | 0.329 |
| "jellyfish" | striped, inter, stride | 0.327 |
| "fish" | red, led, ted, patterned | 0.213 |
| "fish" | red, text | 0.212 |
| "dalmatian dog" | dotted, aka, polka | 0.216 |
| "dalmatian dog" | red, led, ted, patterned | 0.194 |
| "macaws" | colored, colorful, coll, rainbow | 0.217 |
| "macaws" | multicolored, gold, mixed, rainbow | 0.212 |
</details>

Figure 10: Examples: Top-2 concept activations reveal partial semantic similarity between token prediction and related abstract concepts.

# G COMPUTATIONAL COST ANALYSIS

Setup. We compare the computational cost of CGDL and the CoX-LMM baseline under the same hardware: GPU: NVIDIA RTX 3090 (24GB), CPU: AMD Ryzen 9 5950X (16 cores). On this setup, extracting concepts for a fixed set of 10 objects takes 421 s for CGDL and 4375 s for CoX-LMM, assuming that the image–concept assignment (concept bag) is already available on the GPU.

Summary. Table 11 reports the estimated training and inference cost. CGDL requires substantially fewer FLOPs than CoX-LMM during concept learning (∼ 40% reduction in GPU FLOPs and ∼ 4× fewer CPU operations), while the per-image inference cost is identical between the two methods.

Decomposition of CGDL FLOPs. We approximate the LVLM cost using a Gemma-3n-4B backbone with $M = 4 \times 1 0 ^ { 9 }$ parameters and assume a per-token cost of ≈ 2M ≈ $8 \times 1 0 ^ { 9 }$ FLOPs.

![](images/e960d36cb01c56fb8b0e55102a3ee1d82bf2bf2907c89fd5268e56973fd351cf.jpg)

<details>
<summary>bar</summary>

| Category | Value |
| -------- | ----- |
| dog, hot | 0.269 |
| dog, photo, animal, cat | 0.205 |
| dog, category, kat | 0.137 |
| tiger | 0.101 |
| rabbit | 0.116 |
| beer, ted, bearer, baby, animal | 0.106 |
| base, prediction | 0.303 |
</details>

Figure 11: Failure Case: LVLM Hallucination, Text Grounding Incorrect of CGDL. LVLM predicts ”bears’” image as beavers. The concept explanation relates more to a rabbit than a bear. This means that the model knows bears, but it just hallucinated due to the beaver-like appearance. Secondly, even though the textual grounding for ”beaver” is incorrect in the last concept bank in the middle, image grounding is correct, and activation correctly responds to the beaver-like region (body on water). The textual grounding of the concept is incorrectly shifted due to interference (possibly “water-based”), demonstrating a misalignment between visual and textual grounding.

<table><tr><td>K</td><td>BERT@1↑</td><td>BERT@2↑</td><td>BERT@3↑</td><td>CLIP@1↑</td><td>CLIP@2↑</td><td>CLIP@3↑</td></tr><tr><td>2</td><td>0.881±0.013</td><td>0.884±0.014</td><td>0.885±0.014</td><td>0.616±0.044</td><td>0.638±0.041</td><td>0.652±0.043</td></tr><tr><td>10</td><td>0.881±0.013</td><td>0.884±0.013</td><td>0.885±0.013</td><td>0.603±0.047</td><td>0.629±0.047</td><td>0.643±0.048</td></tr><tr><td>30</td><td>0.884±0.040</td><td>0.891±0.041</td><td>0.892±0.041</td><td>0.475±0.033</td><td>0.513±0.026</td><td>0.532±0.026</td></tr><tr><td>50</td><td>0.885±0.040</td><td>0.891±0.041</td><td>0.892±0.041</td><td>0.509±0.028</td><td>0.532±0.028</td><td>0.552±0.030</td></tr><tr><td>100</td><td>0.887±0.041</td><td>0.891±0.041</td><td>0.892±0.041</td><td>0.488±0.035</td><td>0.512±0.031</td><td>0.525±0.032</td></tr></table>

Table 8: Ablation over the number of concept atoms K in the dictionary (five sampled settings). BERTScore stays roughly constant around 0.88, while CLIPScore generally decreases as K increases. Values are mean±std over five ImageNet classes.

<table><tr><td rowspan="2">α</td><td colspan="3">BERTScore ↑</td><td colspan="3">CLIPScore ↑</td></tr><tr><td>@1</td><td>@2</td><td>@3</td><td>@1</td><td>@2</td><td>@3</td></tr><tr><td>0</td><td>0.881 ± 0.013</td><td>0.884 ± 0.013</td><td>0.885 ± 0.013</td><td>0.610 ± 0.039</td><td>0.634 ± 0.036</td><td>0.646 ± 0.037</td></tr><tr><td>20</td><td>0.881 ± 0.013</td><td>0.884 ± 0.014</td><td>0.885 ± 0.014</td><td>0.616 ± 0.044</td><td>0.638 ± 0.041</td><td>0.652 ± 0.043</td></tr><tr><td>100</td><td>0.881 ± 0.013</td><td>0.884 ± 0.014</td><td>0.885 ± 0.014</td><td>0.615 ± 0.043</td><td>0.638 ± 0.040</td><td>0.652 ± 0.042</td></tr><tr><td>150</td><td>0.881 ± 0.013</td><td>0.884 ± 0.014</td><td>0.885 ± 0.013</td><td>0.617 ± 0.043</td><td>0.639 ± 0.038</td><td>0.653 ± 0.039</td></tr><tr><td>200</td><td>0.881 ± 0.013</td><td>0.883 ± 0.013</td><td>0.884 ± 0.014</td><td>0.621 ± 0.039</td><td>0.644 ± 0.039</td><td>0.659 ± 0.042</td></tr></table>

Table 9: SNMF sparsity weight α ablation on Gemma-3n (5 ImageNet classes). Values are mean ± std.

Let T denote the total token length of the multimodal sequence (image tokens, prompt tokens, and generated tokens).

(1) CONCEPT-BAG CREATION (SEC. 4.1). We create concept bags from $N _ { \mathrm { i m g } } = 3 0 0 0$ images (300 images per category) with token length $T = 1 9 6 + 4 0 + 2 0 0 = 4 3 6$ per sample. The total cost for this stage is

$$
\mathrm{FLOPs} _ {\text {imgs}} \approx (8 \times 1 0 ^ {9}) \cdot T \cdot N _ {\text {img}} \approx 1. 0 5 \times 1 0 ^ {1 6}.
$$

(2) SEGMENTATION COST (SAM) (SEC. 4.1). From these images we obtain 16,000 object-centric crops using SAM. Approximating the SAM forward cost as $\approx 1 . 1 \times 1 0 ^ { 1 2 }$ FLOPs per image, we obtain

$$
\mathrm{FLOPs} _ {\mathrm{SAM}} \approx (1. 1 \times 1 0 ^ {1 2}) \cdot 3 0 0 0 \approx 3. 3 \times 1 0 ^ {1 5}.
$$

<table><tr><td rowspan="2">Model</td><td colspan="2">SAM</td><td colspan="2">Random</td></tr><tr><td>BERTScore@1↑</td><td>CLIPScore@1↑</td><td>BERTScore@1↑</td><td>CLIPScore@1↑</td></tr><tr><td>Gemma-3n</td><td>0.93 ± 0.04</td><td>0.61 ± 0.02</td><td>0.88 ± 0.01</td><td>0.59 ± 0.04</td></tr><tr><td>Qwen-2.5</td><td>0.92 ± 0.03</td><td>0.67 ± 0.03</td><td>0.91 ± 0.06</td><td>0.66 ± 0.03</td></tr><tr><td>Qwen-2.0</td><td>0.93 ± 0.04</td><td>0.63 ± 0.04</td><td>0.91 ± 0.06</td><td>0.61 ± 0.04</td></tr></table>

Table 10: SAM vs. Random only localization ablation across three LVLM backbones. Values are mean ± std over five ImageNet classes. We notice that SAM improves both CLIP and BERT compared to random cropping for localization.

![](images/8c567eeab702aeb5e281d34d110828e218c44280c05d1e991a5277b9011df07c.jpg)

<details>
<summary>line</summary>

| Norm Layer | BERT@1 mean | BERT@2 mean | BERT@3 mean |
| ---------- | ----------- | ----------- | ----------- |
| 0          | 0.9         | 0.9         | 0.9         |
| 5          | 0.7         | 0.8         | 0.8         |
| 10         | 0.6         | 0.8         | 0.8         |
| 15         | 0.9         | 0.9         | 0.9         |
| 20         | 0.3         | 0.9         | 0.9         |
| 25         | 0.1         | 0.9         | 0.9         |
| 30         | 0.9         | 0.9         | 0.9         |
| 35         | 0.9         | 0.9         | 0.9         |
</details>

(a) BERTScore across norm layers.

![](images/f33d9beea9d63aa1bda520718f5b275819ef98ef2b26f9019a886ca01c862f48.jpg)

<details>
<summary>line</summary>

| Norm Layer | CLIP@1 mean | CLIP@2 mean | CLIP@3 mean |
| ---------- | ----------- | ----------- | ----------- |
| 0          | 0.50        | 0.50        | 0.50        |
| 5          | 0.50        | 0.50        | 0.50        |
| 10         | 0.50        | 0.50        | 0.50        |
| 15         | 0.48        | 0.48        | 0.48        |
| 20         | 0.50        | 0.50        | 0.50        |
| 25         | 0.50        | 0.50        | 0.50        |
| 30         | 0.50        | 0.50        | 0.50        |
| 35         | 0.67        | 0.67        | 0.67        |
</details>

(b) CLIPScore across norm layers.   
Figure 12: Layer-wise ablation on Gemma-3n. (a) BERTScore of concept phrases extracted from different norm layers. (b) CLIPScore between visual concepts and their text descriptions.

(3) RESIDUAL-STREAM EXTRACTION FOR CONCEPT BAGS (SEC. 4.2). For dictionary learning we use $N _ { \mathrm { r e s } } ~ = ~ 1 6 { , } 0 0 0$ residual samples (1600 crops per concept, 10 concepts). During binary prompting, the token length is $T _ { \mathrm { r e s } } \approx 1 9 6 + 2 1 + 1 0 = 2 2 7$ . The LVLM FLOPs for residual extraction are

$$
\mathrm{FLOPs} _ {\text { res }} \approx (8 \times 1 0 ^ {9}) \cdot T _ {\text { res }} \cdot N _ {\text { res }} \approx 2. 9 1 \times 1 0 ^ {1 6}.
$$

(4) SNMF DICTIONARY LEARNING ON CPU (SEC. 4.2). We factorize the residual activations with sparse NMF using N = 1600 samples, feature dimension D = 2048, dictionary size K = 2, 5000 iterations, and 10 concepts:

$$
\operatorname{Ops} _ {\text {CGDL,SNMF}} \approx N \cdot D \cdot K \cdot \text {iters} \cdot \text {Concepts} = 1 6 0 0 \cdot 2 0 4 8 \cdot 2 \cdot 5 0 0 0 \cdot 1 0 \approx 3. 2 8 \times 1 0 ^ {1 1} \text {CPU ops}.
$$

Total CGDL cost. Summing the LVLM FLOPs across stages yields

$$
\mathrm{FLOPs} _ {\mathrm{CGDL,total}} \approx 1. 0 5 \times 1 0 ^ {1 6} + 3. 3 \times 1 0 ^ {1 5} + 2. 9 1 \times 1 0 ^ {1 6} \approx 4. 2 8 \times 1 0 ^ {1 6}.
$$

# Decomposition of CoX-LMM FLOPs.

(1) CAPTION-LEVEL SEARCH (CPU). CoX-LMM first searches the MSCOCO captions to find images containing the target objects. This involves scanning ≈ 120,000 captions, each of length ≈ 400 characters, which leads to

$$
\mathrm{Ops} _ {\mathrm{CoX}, \text { search }} \approx 1 2 0, 0 0 0 \times 4 0 0 = 4. 8 \times 1 0 ^ {7} \text {   character   comparisons. }
$$

This cost is negligible compared to the LVLM forward passes.

(2) RESIDUAL-STREAM EXTRACTION. CoX-LMM extracts residual activations from 35,000 images (≈ 3500 images per object for 10 objects). The token length is $T _ { \mathrm { r e s } } \approx 1 9 6 + 5 0 + 1 1 = 2 5 7 ,$ , leading to

$$
\mathrm{FLOPs} _ {\mathrm{CoX,res}} \approx (8 \times 1 0 ^ {9}) \cdot T _ {\mathrm{res}} \cdot 3 5, 0 0 0 \approx 7. 2 0 \times 1 0 ^ {1 6}.
$$

<table><tr><td>Category</td><td>CGDL</td><td>CoX-LMM</td><td>Comment</td></tr><tr><td>Training GPU FLOPs</td><td> $\approx 4.28 \times 10^{16}$ </td><td> $\approx 7.20 \times 10^{16}$ </td><td>CGDL  $\approx 40\%$  cheaper</td></tr><tr><td>Training CPU ops (dictionary learning)</td><td> $\approx 3.28 \times 10^{11}$ </td><td> $\approx 1.43 \times 10^{11}$ </td><td>CoX-LMM  $\approx 4 \times$  higher</td></tr><tr><td>Inference FLOPs / image</td><td> $\approx 1.6 \times 10^{11}$ </td><td> $\approx 1.6 \times 10^{11}$ </td><td>Cosine cost negligible</td></tr></table>

Table 11: Training and inference cost of CGDL vs. CoX-LMM.

(3) DICTIONARY LEARNING ON CPU. CoX-LMM uses SNMF with N = 35,000 samples, $D = 2 0 4 8 , K = 1 0$ concepts, and 200 iterations:

$$
\mathrm{Ops} _ {\mathrm{CoX}, \mathrm{SNMF}} \approx N \cdot D \cdot K \cdot \text {iters} = 3 5, 0 0 0 \cdot 2 0 4 8 \cdot 1 0 \cdot 2 0 0 \approx 1. 4 3 \times 1 0 ^ {1 1} \mathrm{CPUops}.
$$

Total CoX-LMM cost. The dominant cost for CoX-LMM is the LVLM residual extraction:

$$
\mathrm{FLOPs} _ {\mathrm{CoX,total}} \approx 7. 2 0 \times 1 0 ^ {1 6}.
$$

The additional CPU cost from caption search and SNMF is small compared to the GPU FLOPs.

Inference-time cost. At inference time, both CGDL and CoX-LMM reuse the same LVLM backbone. For a single image, we approximate the LVLM cost as $\approx 1 . 6 \times 1 0 ^ { 1 1 } \mathrm { F L O P s }$ , and the extra cosine similarity operations between residual activations and concept vectors are negligible. Therefore, the per-image inference cost is effectively identical for both methods.