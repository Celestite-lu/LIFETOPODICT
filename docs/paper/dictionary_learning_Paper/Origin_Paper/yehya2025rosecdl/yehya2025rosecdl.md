# RoseCDL: Robust and Scalable Convolutional Dictionary Learning for Rare-event and Anomaly Detection

Jad Yehya1,∗

Mansour Benbakoura1,∗

Cédric Allain1

Benoît Malézieux1

# Matthieu Kowalski2

# Thomas Moreau1

1Université Paris-Saclay, Inria, CEA, Palaiseau, 91120, France.   
2Université Paris-Saclay, Inria, CNRS, Laboratoire Interdisciplinaire des Sciences du Numériques, Gif-sur-Yvette, France.   
∗Equal Contributions.

# Abstract

Detecting rare events and anomalies in largescale signals is essential in fields such as astronomy, physical simulations, and biomedical science. In many cases, this problem naturally decomposes into identifying common local patterns and detecting deviations that correspond to anomalies. Convolutional Dictionary Learning (CDL) is a powerful tool for modeling local structures, but its adoption for this task has been limited by computational demands and sensitivity to outliers. We introduce RoseCDL, a novel CDL algorithm designed for robust and scalable modeling of signal pattern distribution. RoseCDL leverages stochastic windowing for efficient training and incorporates inline outlier detection to enhance robustness. This enables unsupervised identification of anomalous and rare patterns in long signals based on the local reconstruction loss. Experiments on real-world datasets show that RoseCDL delivers improved detection accuracy and computational efficiency, making CDL practical for challenging detection tasks in large-scale signal analysis.

# 1 INTRODUCTION

Identifying recurring patterns and anomalies in a signal is a crucial task in many scientific fields, from finding

Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS) 2026, Tangier, Morocco. PMLR: Volume 300. Copyright 2026 by the author(s).

QRS complex – a.k.a. heartbeats – in ECG (Luz et al., 2016) to detecting blood-cells in biological images (Yellin et al., 2017) or particular celestial objects in astronomical images (Giavalisco and the GOODS Teams, 2004). In large-scale settings, this process must be automated. Supervised methods have been developed to address these tasks, often relying on large annotated datasets and deep learning models (Murat et al., 2020; Choudhary et al., 2024; Cornu et al., 2024). However, obtaining labeled data can be costly, time-consuming, or even infeasible, especially when the patterns of interest are rare or hard to characterize. This motivates the development of unsupervised methods capable of automatically uncovering patterns distribution in large datasets.

Convolutional Dictionary Learning (CDL; Grosse et al., 2007) is a powerful method for modeling local structures in signals, with applications in audio, neuroscience, and image processing (Dupré la Tour et al., 2018; Papyan et al., 2017). Its core principle is to represent an observed signal as the convolution of a learned dictionary of patterns with a sparse activation vector. By aggregating over many signals, CDL learns a dictionary that captures recurring local structures, making it particularly suitable for unsupervised pattern discovery. Despite its promise, the use of CDL in practice remains limited. Two challenges, in particular, hinder its applicability to large, noisy datasets.

Scalability. Learning with CDL is computationally expensive, primarily due to the sparse coding phase. Considerable effort has been devoted to improving scalability, including deterministic optimizations (Wohlberg, 2015), local-window methods that partition the signal (Grosse et al., 2007; Papyan et al., 2017; Moreau and Gramfort, 2020; Dragoni et al., 2022), and approximate techniques such as learned sparse coding and algorithm unrolling (Gregor and LeCun, 2010; Tang et al., 2021; Tolooshams et al., 2018; Tolooshams and Ba, 2022; Malézieux et al., 2022). Online algorithms (Mairal et al., 2010; Mensch et al., 2016; Liu et al., 2017; Zeng et al., 2019) further reduce computational cost by updating the dictionary on subsets of the data. However, existing approaches either struggle to generalize across large datasets or achieve only limited computational gains.

Robustness to anomalies and rare events. CDL is highly sensitive to anomalies: because outliers are sparse but often high-magnitude, the algorithm may mistakenly interpret them as meaningful patterns. Theoretical work in classical dictionary learning has formalized this risk (Gribonval et al., 2015), and a few practical remedies have been proposed, such as Elastic Net regularization (Mairal et al., 2010) or heavy-tailed fitting terms (Jas et al., 2017). Yet these methods either increase computational cost or only partially address robustness. Related research in outlier detection (OD) has emphasized reconstruction-based frameworks, where models are trained to reconstruct normal data while failing on anomalies (Ruff et al., 2021; Schmidl et al., 2022). While the majority of reconstructionbased OD methods are semi-supervised, fully unsupervised methods perform competitively (Darban et al., 2024), and similar principles are used for rare-event detection (Shyalika et al., 2024). This idea also resonates with classical robust regression, where corrupted observations are trimmed based on residual magnitude. The Least Trimmed Squares (LTS) estimator (Rousseeuw, 1984; Rousseeuw and Leroy, 2005) and its extensions, including Sparse LTS (Alfons et al., 2013) demonstrate the effectiveness of trimming in high-dimensional and structured settings.

Our contributions. This paper introduces a novel CDL algorithm called RObust and ScalablE CDL (RoseCDL) that addresses these limitations. To ensure scalability, we propose a stochastic windowing approach, where the sparse coding problem is solved approximately on a small, randomly sampled data windows, and the dictionary is updated using the results of local computations. To ensure robustness, we integrate an inline outlier detection mechanism based on local reconstruction error, which discards patches poorly explained by the learned dictionary. The same reconstruction-error criterion can be applied at test time to identify anomalies and rare events, providing a practical unsupervised detection tool alongside representation learning. The resulting algorithm learns dictionaries that capture the local structure of data, remains robust to artifacts, scales to large datasets, and naturally doubles as an unsupervised anomaly detection method. More broadly, it reframes CDL from a reconstruction problem to the estimation of the patch distribution underlying a signal, enabling both robust representation and principled rare-event and anomaly detection. We demonstrate its effectiveness on both synthetic and real-world data.

# 2 FINDING COMMON AND RARE PATTERNS IN SIGNALS: THE RoseCDL ALGORITHM

Let $\mathbf { x } \in \mathbb { R } ^ { T }$ be a univariate signal with length T . Convolutional Dictionary Learning (CDL) consists of finding a dictionary: $D = ( \mathbf { d } _ { k } ) _ { k \in  { [ [ 1 , K ] ] } } \in  { \mathbb { R } } ^ { K \times L }$ of K patterns of length $L \ll T ,$ J K and corresponding activation vectors $Z = \mathsf { \bar { \rho } } ( \mathbf { z } _ { k } ) _ { k \in [ [ 1 , K ] ] } \in \mathbb { R } ^ { K \times ( T - L + 1 ) }$ , that minimize the J Kdistance between x and $\begin{array} { r } { \hat { \mathbf { x } } = D * Z = \sum _ { k } \mathbf { d } _ { k } * \mathbf { z } _ { k } } \end{array}$ , where ∗ denotes the convolution.

This is achieved by solving the following optimization problem:

$$
\min _ {D, Z} F (D, Z; \mathbf {x}) = \underbrace {\frac {1}{2} \| \mathbf {x} - D * Z \| _ {2} ^ {2}} _ {f _ {\mathbf {x}} (Z)} + \lambda \| Z \| _ {1}, \tag {1}
$$

$$
\text { s.t. } \quad \| \mathbf {d} _ {k} \| _ {2} ^ {2} \leq 1, \quad \forall k \in [   [ 1, K ]   ].
$$

where $\begin{array} { r } { \| Z \| _ { 1 } = \sum _ { k } \| \mathbf { z } _ { k } \| _ { 1 } } \end{array}$ . Fig. 1 illustrates the CDL model for an univariate signal. This problem can easily be extended to multivariate and multidimensional signals such as images by adapting the convolution operator to work with tensors.

When working with large signal databases, the primary interest is not finding the best reconstruction of the signal used to learn the dictionary but instead finding the patterns that best model the population. Therefore, this paper focuses on characterizing the distribution of the signals x, with the following optimization problem:

$$
\min _ {D} \mathbb {E} _ {\mathbf {x}} \left[ \min _ {Z} F (D, Z; \mathbf {x}) \right], \tag {2}
$$

$$
\text { s.t. } \quad \| \mathbf {d} _ {k} \| _ {2} ^ {2} \leq 1, \forall k \in [ [ 1, K ] ].
$$

While this formulation departs from classical optimization-based literature for CDL (Wohlberg, 2015), it is often used with deep CDL approaches to learn denoisers that generalize to unseen images using super learning losses (Scetbon et al., 2021; Zheng et al., 2021; Deng et al., 2023). Here, we consider the unsupervised loss F to learn the dictionary of event patterns directly from the signals.

For both formulation (1) and (2), classical solvers rely on an iterative optimization algorithm to solve the sparse coding problem in Z for fixed D, and alternate it with a dictionary update step. The sparse coding step is typically solved using iterated soft-thresholding iterations (e.g., FISTA, Beck and Teboulle, 2009; Chambolle and Dossal, 2015), which reads:

![](images/5243f071d76716779fb360f1cfdd5e57d659851f56810537038845b09477f554.jpg)

<details>
<summary>line</summary>

| Time Point | Signal X Value | Atom D Value |
| ---------- | -------------- | ------------ |
| t₁         | ~0.5           | ~0.5         |
| t₂         | ~0.5           | ~0.5         |
| t₃         | ~0.5           | ~0.5         |
| t₄         | ~0.5           | ~0.5         |
</details>

Figure 1: Schematic operation of the CDL 1D univariate signal, adapated from the dicodile package example1. The output of the CDL is composed of a set of atoms (here 1) alongside their respective activations.

$$
t _ {0} = 1
$$

$$
Z _ {0} = Y _ {0} \in \mathbb {R} ^ {K \times T - L + 1}; t _ {0} = 1
$$

$$
Z _ {k + 1} = \operatorname{St} (Y _ {k} - \nu \nabla_ {Z} f _ {\mathbf {x}} (Z), \nu \lambda) \tag {3}
$$

$$
t _ {k + 1} = \frac {1}{2} (1 + \sqrt {1 + 4 t _ {k} ^ {2}})
$$

$$
Y _ {k + 1} = Z _ {k} + \frac {t _ {k} - 1}{t _ {k + 1}} (Z _ {k + 1} - Z _ {k})
$$

with St the soft-thresholding operator $\begin{array} { r l } { \mathrm { S t } ( Z ) } & { { } = } \end{array}$ sign $( Z ) ( | Z | - \lambda ) _ { + }$ and ν the step size. The dictionary update may rely on projected gradient descent constrained to the unit $\ell _ { 2 } { \mathrm { - b a l l } }$ . However, these methods become computationally expensive on long signals or large datasets, as both the sparse codes and gradients must be computed over all the signals. Online methods mitigate the need to consider all signals, but at the expense of increased memory usage and still require solving the sparse coding problem on full signals. This motivates using stochastic or localized approximations, especially in settings where robustness to artifacts or outliers is required.

# 2.1 Stochastic windowing

Due to the convolutional structure of the CDL model, points in the far-apart signal are only weakly dependent Moreau and Gramfort (2020). Indeed, the value of the sparse code at time t is seldom impacted by the values at time $t + s ,$ , with s larger than the size of the dictionary L. We will therefore perform windowing spanning across all channels synchronously.

By considering long enough windows of the signal, the problem (2) can be approximated by

$$
\min _ {D} \mathbb {E} _ {\tau} \left[ \min _ {Z} F (D, Z; \mathbf {x} _ {\tau}) \right], \tag {4}
$$

$$
\text { s.t. } \quad \| \mathbf {d} _ {k} \| _ {2} ^ {2} \leq 1, \forall k \in [   [ 1, K ]   ],
$$

where $\mathbf { x } _ { \tau }$ are windows of the signal, starting at $\tau \ \in \ [ [ 1 , T - W _ { \mathrm { w i n } } + 1 ] ]$ and of size $W _ { \mathrm { w i n } }$ such that $L \leq W _ { \mathrm { w i n } } \ll T$ K. This formulation is only approximate as it produces inexact sparse code for the windows, due to border effect. But with the weak spatial dependence of the model, if the window is large and the activation are sparse, these effects are negligible. Moreover, by considering all windows with overlap, we limit the impact of specific border effects in the algorithm. To minimize (4), we propose RoseCDL, a stochastic gradient descent algorithm aimed to characterize the distribution of patches in the signal.

We process as follows: At each step of the CDL outer problem, we begin by sampling $N _ { W }$ windows $( \mathbf { x } _ { w } ) _ { 1 \leq w \leq N _ { W } }$ from a uniform distribution. The windows are selected with overlap to limit the bias due to border effects. We then compute an approximate sparse code $Z _ { w } ^ { \star }$ for each subproblem by minimizing the sparse coding over the window. Following Malézieux et al. (2022); Tolooshams and Ba (2022), whose results suggested that optimizing over D did not require precise sparse coding at each step, we use an approximation $\bar { Z } ^ { N _ { \mathrm { F I S T A } } }$ of $Z ^ { \star } ( D ; { \bf x } )$ , where $Z ^ { N _ { \mathrm { F I S T A } } }$ is given by NFISTA iterations of the FISTA algorithm (3).

Using these per-window approximated activation vectors, we can then update the dictionary D. Due to the stochastic nature of the algorithm, we depart from traditional alternate minimization strategies and perform only one gradient step. This single-step update strategy is further justified by the inherent noise in the gradient estimate, which arises from both the stochastic sampling of windows and the use of approximate sparse codes. In such settings, performing multiple gradient steps per batch does not significantly reduce the update variance and may even amplify the impact of the approximation error in the sparse coding stage. This strategy is in-line with the deep CDL algorithm, but we do not backpropagate the gradient through the sparse code approximation $Z ^ { N _ { \mathrm { F I S T A } } }$ , as it leads to unstable Jacobian estimation for the original problem (4), as described in Malézieux et al. (2022). To stabilize the learning and reduce the number of parameters, we compute the optimal stepsize of the dictionary update with the Stochastic Line Search algorithm (SLS; Vaswani et al., 2019). This algorithm can be efficiently implemented using deep learning frameworks and can leverage GPU acceleration. The complete procedure is summarized in Alg. 1. We note that when trimming is ignored and the sparse codes are solved exactly, RoseCDL fits within the SGD setting, with stochasticity arising from the sampled windows. In this setting, convergence properties are well established (Bottou et al., 2018)

Algorithm 1 CDL with stochastic windowing.   
input X, $N_{iter}$ , $N_{W}$ , $N_{FISTA}$ Initialize $D^{(0)}$ for $0 \leq i \leq N_{iter} - 1$ do
    Sample $N_{W}$ windows in the dataset: $(X_{w})_{w \in [1, N_{w}]}$ for $1 \leq w \leq N_{W}$ do
    Compute the approximate sparse code $Z_{w}^{N_{FISTA}} \approx Z_{w}^{\star}(D^{(i)}; X_{w})$ Compute an outlier mask (cf. Sec. 2.2)
    Compute the loss F and its gradient $\nabla_{D}F$ outside the outlier mask
    end for
    Compute best step size $\alpha_{i}$ with SLS $D^{(i+1)} \leftarrow D^{(i)} - \alpha_{i}\nabla_{D}\sum_{w} F_{w}(D^{(i)}, Z_{w}^{N_{FISTA}}; X_{w})$ end for
output $D^{(N_{iter})}$

# 2.2 Inline outlier detection

The second component of RoseCDL is its inline outlier detection framework. Let x be a signal of the form:

$$
\mathbf {x} = \mathbf {d} _ {a} * \mathbf {z} _ {a} + \mathbf {d} _ {b} * \mathbf {z} _ {b} + \mathbf {n}, \tag {5}
$$

where ${ \bf d } _ { a }$ is a common pattern in the signal, $\mathbf { d } _ { b }$ is a rare pattern, and n is an abnormal pollution (e.g., artifacts, sudden spikes in the signal). Traditional CDL algorithms are likely to struggle to recover ${ \bf d } _ { a }$ for two main reasons. First, artifacts in n often have a significant variance, which can distract the CDL from the significant signal. Second, even when the anomalies in n are well discarded, the rare pattern $\mathbf { d } _ { b }$ acts as a pollution that prevents the algorithm from learning the atom ${ \bf d } _ { a }$ . While effects of n and $\mathbf { d } _ { b }$ are often discarded through preprocessing Dupré la Tour et al. (2018), it is often very difficult to use this reliably on large population datasets.

The intuition behind our approach is that if ${ \bf d } _ { a }$ is sufficiently represented in x, then most of the patches of x should contain information relative to this pattern. Consequently, these patches should be the best reconstructed ones. In comparison, the patches containing non-zero values of n are expected to have a high reconstruction error due to the unpredictable nature of anomalies and artifacts. Finally, given a dictionary d that is more correlated with ${ \bf d } _ { a }$ than db, the patches containing chunks of $\mathbf { d } _ { b }$ are expected to have a higher reconstruction error than those with ${ \bf d } _ { a }$ .

To summarize, the distribution of patch reconstruction errors $F ( D , Z ; { \bf x } _ { \tau } )$ τ is expected to have three modes:

1. The main, low-value mode corresponding to chunks of ${ \bf x } _ { a } = { \bf d } _ { a } * { \bf z } _ { a }$ ,   
2. A secondary mode with a slightly higher value corresponding to chunks of $\mathbf { x } _ { b } = \mathbf { d } _ { b } * \mathbf { z } _ { b }$ ,   
3. A high-value mode corresponding to chunks of n.

Consequently, the reconstruction error of a patch can be used as an indicator to determine whether it contains information about the relevant signal. We leverage this intuition to inflect the learning trajectory of d towards $\mathbf { d } _ { a } .$ We define the set $\mathcal { P } _ { \beta } = \{ \tau | F ( D , Z ; \mathbf { x } _ { \tau } ) < \beta \}$ , where $\beta \in [ 0 , 1 ]$ is a threshold selecting the proportion of outliers. With a well chosen ${ \boldsymbol { \beta } } , \mathcal { P } _ { \beta }$ indicates the patches that coincide with realizations of ${ \bf \delta x } _ { a }$ . Therefore, to make the CDL converge towards $\mathbf { d } _ { a } .$ , we change the objective value for the dictionary updates for its trimmed version:

$$
\widetilde {F} (Z, D; \mathbf {x}) = \frac {1}{W _ {\mathrm{patch}}} \sum_ {\tau \in \mathcal {P} _ {\beta}} F (Z, D; \mathbf {x} _ {\tau}). \tag {6}
$$

Using this trimmed loss, we can show that in simple settings, the RoseCDL algorithm is more robust to the presence of outliers.

Proposition 2.1 (Stability of the common pattern). Consider a population of signals X composed of two patterns ${ \bf d } _ { a }$ and $\mathbf { d } _ { b }$ , with activations such that the patterns do not overlap in the signal, and corrupted by an additive Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \sigma I d )$ . Introduce $c = \mathbf { d } _ { a } ^ { \top }$ db and $\rho$ the proportion of rare-event pattern $\mathbf { d } _ { b }$ activations in the population. Then, in the noiseless setting,

i. ${ \bf d } _ { a }$ is a fixed point of the classical CDL algorithm $f o r \ K = 1 \ i f c \leq \lambda$

![](images/351264576d44c17026152d9bd500fa0d11b9ae57e5210960248ebf23869334e6.jpg)

<details>
<summary>line</summary>

| Time (s.) | Raw signal | Outlier Mask | Error |
| --------- | ---------- | ------------ | ----- |
| 2000      | ~0.0       | ~0.0         | ~0.0  |
| 2500      | ~0.0       | ~0.0         | ~0.0  |
| 3000      | ~0.0       | ~0.0         | ~0.0  |
| 3500      | ~0.0       | ~0.0         | ~0.0  |
| 4000      | ~-1.0      | ~1.0         | ~0.1  |
| 4500      | ~-1.0      | ~1.0         | ~0.1  |
| 5000      | ~-1.0      | ~1.0         | ~0.1  |
</details>

Figure 2: Raw signal X, reconstruction error, threshold and learned outlier mask on subject a02 (minute 56) of Physionet Apnea-ECG data set. Detection method is based on modified z-score (MAD), with $\alpha = 3 . 5$ . The method correctly identifies outliers blocks.

ii. db is a fixed point of RoseCDL algorithm for $K = 1 \ i f c \leq \lambda$ or the RoseCDL algorithm is used with an outlier threshold trimming a proportion of windows greater than $\rho .$

The proof is deferred in Sec. A. This theoretical analysis is intentionally illustrative: its purpose is to provide intuition on why trimming improves robustness, rather than to serve as a full convergence guarantee for the algorithm. Nevertheless, this demonstrates that even in ideal conditions (no overlap and starting with the right pattern), classical CDL fails to recover the common pattern due to interference from rare events, while RoseCDL corrects this via trimming. However, a critical design choice is the selection of the statistic $\beta$ used as the threshold, discussed below.

Threshold selection. From the distribution of patch reconstruction errors $( \pmb { \varepsilon } _ { \tau } ) _ { \tau } \ = \ ( F ( D , Z ; { \bf x } _ { \tau } ) ) _ { \tau }$ , one needs to compute a threshold $\beta$ which separates the outlier patches in x from the normal ones. The goal is to detect extreme points in the distribution, which are too large compared to the population of patch errors. The outlier detection literature provides three main methods relevant in this case:

1. The method of quantiles, where $\beta = Q _ { \varepsilon , ( 1 - \alpha ) }$ is the quantile of order (1 − α) of the set ε,   
2. The z-score method (Iglewicz and Hoaglin, 1993), where each error $\varepsilon _ { \tau }$ has an associated score $z _ { \tau } = ( \varepsilon _ { \tau } - \mu _ { \varepsilon } ) / \sigma _ { \varepsilon }$ , with $\mu _ { \varepsilon }$ and $\sigma _ { \varepsilon }$ denoting respectively the mean and standard deviation of the distribution of errors, and where outliers are defined as observations such that $| z _ { \tau } | > \alpha$ , generally α = 2 or 3, thus having $\beta = \mu _ { \epsilon } + \alpha \ \sigma _ { \epsilon }$ ,   
3. The modified z-score (MAD; Iglewicz and Hoaglin 1993), where, similarly as the z-score, each error point is associated with a score based

on the median, and the resulting threshold is $\beta = \mathrm { M e d } _ { \epsilon } + ( \alpha$ Madε)/0.6745, with $\mathrm { { M a d } } _ { \varepsilon } ~ =$ Med $\left( | \varepsilon - \mathrm { M e d } _ { \varepsilon } | \right)$ , where Med denotes the median operator, and generally $\alpha = 3 . 5$ . Here, $\alpha = 3 . 5$ is not a tunable hyperparameter but a standard choice in robust statistics (Iglewicz and Hoaglin, 1993). The threshold is computed adaptively from the empirical distribution of reconstruction errors, making it data-driven, statistically grounded, and independent of any contamination-rate assumptions. It is not inflated by extreme reconstruction errors caused by outliers, which prevents it from being excessively high, which would reduce detection sensitivity.

These methods are initially bilateral, but only the upper bound is considered in this work because we aim to detect outliers with large reconstruction errors. $\mathrm { F i g . }$ 2 illustrates the outlier detection method on real ECG data, with the outliers mask computed with the reconstruction error and the threshold.

Role of the outlier mask for rare-event detection. The inline outlier detection module is a central component of our algorithm. It enables the computation of an outlier mask during training, serving two key purposes: (1) it excludes identified outliers from the loss function, thereby improving the robustness of dictionary learning; (2) it enables the unsupervised detection of rare events in the signal by interpreting the outlier mask as a detection map. Indeed, provided a signal x in the form of $\operatorname { E q . }$ (5), we have shown that RoseCDL is able to extract the contribution of the common pattern ${ \bf d } _ { a }$ . Consequently, the corresponding outlier mask can be used to select the residual signal $\mathbf { x } ^ { \prime } = \mathbf { d } _ { b } { * } \mathbf { z } _ { b } + \mathbf { n }$ . Running another instance of RoseCDL on $\mathbf { x } ^ { \prime }$ then allows one to recover the pattern $\mathbf { d } _ { b }$ and localize its occurrences $\mathbf { z } _ { b } .$ . In multivariate time series, RoseCDL computes reconstruction errors on multichannel patches, so even small deviations on individual channels can accumulate into a global deviation that exceeds the computed threshold. In practice, multivariate anomalies are detected whenever their combined effect sufficiently alters the local motif.

![](images/9a28e5bf35a698678a882b4ee3b6a62d252fbab1d1b1fd7d589a87527e7c3fa3.jpg)  
Figure 3: Comparison of optimization runtime for RoseCDL, AlphaCSC, Sporco, and DeepCDL in 1D and 2D settings, highlighting the superior scalability and convergence speed of RoseCDL. The runtime plots show the evolution of test loss over time. The third subplot reports the dictionary recovery score at convergence for 2D data. As AlphaCSC cannot be applied to 2D data, only results for Sporco, DeepCDL, and RoseCDL are shown.

# 3 NUMERICAL EXPERIMENTS

In this section, we present numerical experiments that demonstrate the scalability and robustness of RoseCDL, as well as its ability to detect anomalies in temporal (1D) and spatial (2D) signals, with an arbitrary number of channels, on simulated and realworld data. We implemented the algorithm using the Pytorch framework (Paszke et al., 2017). 2 The experiments fall into three categories, each targeting a different aspect: scalability, robustness to outliers, and anomaly detection.

# 3.1 Scalability

We performed a comprehensive comparison between RoseCDL and three state-of-the-art CDL methods: AlphaCSC (Dupré la Tour et al., 2018), Sporco (Wohlberg, 2017), and DeepCDL, a variant of Tolooshams and Ba (2022). DeepCDL represents the unrolled variant of RoseCDL where windows are not stochastically sampled and gradients account for the Jacobian of the sparse code computed through backpropagation. In contrast, RoseCDL employs an alternating minimization scheme as explained in Sec. 2.

We evaluate each method’s computational cost and runtime on two large-scale datasets. The one-dimensional (1D) experiment is performed on 20 synthetic multivariate signals containing 50,000 time samples with two channels (see Sec. C for details). The two-dimensional (2D) experiment is conducted on a semi-synthetic dataset of images of 2000 × 2000 pixels. The cost is evaluated as the value of the objective function $F ( D , Z ^ { \star } ( D ) ; { \bf x } )$ at each iteration on a separate test set, with $Z ^ { \star }$ computed to convergence, assessing the capacity of the solver to minimize Eq. (2). To evaluate dictionary recovery, we use the metric proposed by Moreau and Gramfort (2020), which aligns true and estimated atoms by convolutional cosine similarity (see Sec. D).

The empirical results shown in the left and middle panels of Fig. 3 demonstrate that RoseCDL exhibits superior scalability attributable to two architectural advantages shared with our implementation of Deep-CDL: (i) GPU-optimized training yields substantial speedups when properly leveraged (ii) Fast Fourier Transform-based convolutions significantly reduce the computation time for large kernels compared to standard spatial convolutions. The distinction from Deep-CDL arises primarily from optimization strategy differences: DeepCDL’s unrolled architecture requires gradient computation over substantially larger parameter sets, increasing computational overhead compared to RoseCDL’s alternating minimization approach.

The right panel of Fig. 3 demonstrates that RoseCDL matches the dictionary recovery performance of Deep-CDL and substantially better than Sporco, validating both computational efficiency and solution quality.

It is important to note that AlphaCSC does not support two-dimensional data and was excluded from imagebased experiments. Despite this limitation, RoseCDL achieves comparable or superior test costs across all evaluated configurations.

![](images/b61b0a7df780bb62fd4cdf8feffaca4ead3f3e58ea3ee2dc70f39d1c69b03a7c.jpg)  
Figure 4: Comparison between RoseCDL without inline outlier detection (left panel) and with it (middle panel). Evolution of median recovery scores for RoseCDL on the ROSE+Z data set across 20 independent trials. Shaded regions represent the range between 25th and 75th percentiles. Right panel: illustration of recovered atoms for different recovery scores.

To ensure that the results above are not artifacts of parameter choices, we evaluated RoseCDL under different setups and compared it to AlphaCSC, assessing both computation acceleration and potential trade-offs in solution quality. Varying the window size from 10 to 100 times the atom length produced only minor differences, with performance more strongly influenced by the choice of optimizer. Using Adam resulted in a 7–8× speedup while keeping the error within +4% of AlphaCSC’s, whereas SLS achieved smaller speedups (4.5–5.5×) but reduced the error compared to AlphaCSC. When varying the signal length, RoseCDL exhibited sublinear scaling and remained usable on signal lengths where AlphaCSC became prohibitive. These experiments are detailed in Appendix B.1.

# 3.2 Robustness to outliers

In this subsection, we study the role of the inline outlier detection module in enabling RoseCDL to reconstruct common patterns in the presence of anomalies and outlier patterns. First, an example of RoseCDL mechanism on the Physionet Apnea-ECG (Penzel et al., 2000) is shown in Fig. 2 highlighting parts of the data that are discarded during training when using the inline outlier module as well as the outlier mask produced. This shows that the reconstruction error highlights part of the signal that significantly differ from the rest, allowing to identify anomalies in the data. This also stabilizes the learning, as shown in Sec. B.3 which presents a side-by-side visualization of atoms learned with and without inline outlier detection, demonstrating the clear benefits of this module.

To better quantify this effect, we constructed the ROSE+Z dataset, a semi-synthetic corpus of images generated from 5,000 characters drawn from four letters (R, O, S, and E) together with spaces, mimicking text-like documents. A small proportion of the letter Z was added to introduce rare events. Using this dataset, we assess how RoseCDL recovers the underlying patterns under varying levels of contamination. For this experiment, we implemented the inline outlier detection module with the MAD method described in Sec. 2.2, as preliminary comparisons against the quantile and zscore methods consistently favored MAD (see Sec. B.2). We then evaluated RoseCDL with and without inline outlier detection across 20 independent trials on the ROSE+Z dataset. Fig. 4 shows the recovery score for both the common letters ROSE and the outlier letter Z depending on the outlier frequency.

The inline outlier detection stabilizes training, improves the quality of the learned dictionary, and reduces variability across trials by preventing rare events from contaminating the learning process. While without inline outlier detection, the recovery slowly decreases, as the outlier gets mixed in with the common patterns, the inline mechanism allows for a sharper transition once the outlier frequency becomes too large and cannot be discarded anymore. A key parameter of RoseCDL is the regularization coefficient λ, introduced in Sec. 2, which controls the sparsity of the learned activation vector. We express λ as a fraction of the maximum regularization value, $\lambda _ { m a x } ,$ corresponding to the smallest regularization value for which the activation vector is entirely zero. Our insights indicate that the best dictionary recovery is achieved when setting $\lambda = 0 . 1 \lambda _ { m a x } ,$ effectively balancing sparsity and reconstruction accuracy. Qualitative and quantitative results are given in Sec. B.4. The inline outlier detection also improves the algorithm’s robustness with respect to this choice of the regularization parameter λ.

<table><tr><td></td><td>Dim.</td><td>Dataset</td><td>RoseCDL</td><td>AB</td><td>DAGMM</td><td>MP</td><td>AT</td><td>TimesFM</td><td>TimesNet</td></tr><tr><td rowspan="3">Pattern</td><td>U</td><td>ECG</td><td>0.534</td><td>0.099</td><td>0.077</td><td> $\underline{0.163}$ </td><td>0.095</td><td>0.150</td><td>0.151</td></tr><tr><td>U</td><td>SVDB</td><td>0.431</td><td>0.217</td><td>-</td><td> $\underline{0.224}$ </td><td>0.106</td><td>0.119</td><td>0.121</td></tr><tr><td>M</td><td>Simulated</td><td>0.986</td><td> $\underline{0.704}$ </td><td>0.644</td><td>-</td><td>0.624</td><td>0.652</td><td>0.590</td></tr><tr><td rowspan="5">No Pattern</td><td>U</td><td>Daphnet</td><td>0.111</td><td> $\underline{0.186}$ </td><td>-</td><td>0.101</td><td>0.060</td><td>0.048</td><td>0.289</td></tr><tr><td>U</td><td>MITDB</td><td>0.155</td><td> $\underline{0.186}$ </td><td>-</td><td>0.250</td><td>0.135</td><td>0.136</td><td>0.152</td></tr><tr><td>U</td><td>Dodgers</td><td>0.238</td><td>0.078</td><td>0.102</td><td> $\underline{0.199}$ </td><td>0.140</td><td>0.087</td><td>0.126</td></tr><tr><td>M</td><td>MSL</td><td>0.136</td><td>0.171</td><td>0.088</td><td>-</td><td>0.053</td><td> $\underline{0.136}$ </td><td>0.111</td></tr><tr><td>M</td><td>SMAP</td><td>0.111</td><td>0.136</td><td>-</td><td>-</td><td>0.564</td><td>0.107</td><td> $\underline{0.140}$ </td></tr><tr><td colspan="3">Avg. Runtime (s)</td><td>23.47</td><td>2594.71</td><td>23221.20</td><td> $\underline{687.12}$ </td><td>11715.00</td><td>3986.07</td><td>1916.37</td></tr><tr><td colspan="3">Avg. Pattern</td><td> $0.65 \pm 0.30$ </td><td> $0.34 \pm 0.32$ </td><td> $\underline{0.36 \pm 0.40}$ </td><td> $0.19 \pm 0.04$ </td><td> $0.28 \pm 0.30$ </td><td> $0.31 \pm 0.30$ </td><td> $0.29 \pm 0.26$ </td></tr><tr><td colspan="3">Avg. No Pattern</td><td> $0.15 \pm 0.05$ </td><td> $0.15 \pm 0.05$ </td><td> $0.09 \pm 0.01$ </td><td> $\underline{0.18 \pm 0.08}$ </td><td> $\underline{0.19 \pm 0.21}$ </td><td> $0.10 \pm 0.04$ </td><td> $0.16 \pm 0.07$ </td></tr></table>

Table 1: Comparison of AUC-PR of RoseCDL with state-of-the-art anomaly detection methods on both pattern (P) and non-pattern (NP) datasets, spanning univariate (U) and multivariate (M) settings. RoseCDL achieves the strongest performance on pattern anomaly datasets, its target regime, while remaining competitive on non-pattern data. Notably, RoseCDL delivers these results with a runtime more than two orders of magnitude faster than all deep baselines, and unlike Matrix Profile (MP), it scales seamlessly to multivariate inputs. Bold values indicate the best score in each row, underlined the second best. (AB : AnomalyBERT, AT : AnomalyTransformer).

# 3.3 Anomaly detection

We evaluated the anomaly detection capability of RoseCDL on diverse datasets from the TSB-UAD and TSB-AD benchmarks (Paparrizos et al., 2022; Liu and Paparrizos, 2024), covering both univariate and multivariate cases. Since RoseCDL targets pattern anomalies, we included datasets both well- and poorlyaligned with this inductive bias to assess its strengths and limitations.

For baselines, we compared against representative unsupervised methods across two major families identified by Darban et al. (2024): forecasting-based and reconstruction-based. Specifically, we selected competitive examples such as the Anomaly Transformer (AT; Xu et al., 2022) and TimesNet (Wu et al., 2023), which respectively exemplify reconstruction-based and forecasting-based approaches. To further broaden the comparison, we included the classical Matrix Profile method (MP; Yeh et al., 2016), representing a strong shallow baseline, as well as the recently proposed foundation model TimesFM (Das et al., 2024), which extends beyond the scope of recent surveys such as Ruff et al. (2021) and Darban et al. (2024).

As summarized in Tab. 1, RoseCDL consistently achieves competitive or superior performance in terms of AUC-PR on datasets with pattern anomalies. This validates its central design principle of leveraging local structures. On datasets lacking strong patterns (e.g., Daphnet, Dodgers, MSL, SMAP) or where anomalies arise outside the pattern dimension (e.g., MITDB), performance drops but remains comparable to baselines, highlighting both the scope and the boundaries of the proposed method. More precisely, anomalies spanning several atom lengths can still be detected, but anomalies affecting long-range temporal dynamics or global drifts without disrupting local shape are less aligned with CDL assumptions, which explains the performance gap on datasets like MITDB where anomalies primarily affect heartbeat timing rather than morphology.

Examples of both missed and detected anomalies on various datasets are given in Sec. B.5. Beyond accuracy, RoseCDL is over two orders of magnitude faster than competing methods, enabling scalability to large collections of time series.

These results demonstrate that RoseCDL can extract meaningful patterns from corrupted real-world data without explicit preprocessing or prior knowledge of the outlier proportion. This is made possible by the inline outlier detection module, which enhances both training stability and robustness to hyperparameter choices.

# 4 CONCLUSION

In this study, we introduce RoseCDL, a robust and scalable approach to Convolutional Dictionary Learning (CDL). While CDL has traditionally focused on reconstructing signals, we reframe it as estimating the distribution of local patches. This approach supports the two main features of RoseCDL: stochastic windowing for scalability and inline outlier detection for robustness to anomalies. Together, these components make RoseCDL suitable not only for large-scale pattern discovery but also for unsupervised anomaly and rare-event detection. Concretely, the method first models the common patterns in a signal, removes them, and then learns a dictionary on the residuals. This yields a robust representation and an outlier mask, while also uncovering the distinctive patterns that occur specifically within anomalous regions.

A central concept of this paper is reframing CDL as a tool to characterize the distribution of patterns in the signal. Beyond improving scalability and robustness, this perspective also provides a principled foundation for anomaly and rare-event detection: local reconstruction errors naturally reflect deviations from the learned patch distribution.

Limitations. Like all CDL-based methods, our approach relies on a non-convex objective. Nevertheless, the optimization scheme builds on well-established principles, for which convergence to meaningful local minima has been observed and studied in prior work (Gribonval et al., 2015; Malézieux et al., 2022; Tolooshams and Ba, 2022). Our trimming strategy intentionally excludes poorly reconstructed patches during training, which introduces bias but is consistent with our design goal of separating atypical regions rather than fitting them. While this mechanism confers robustness to outliers, it is conservative by nature: many segments may be flagged as atypical. Consequently, RoseCDL is better suited for robust representation learning and rare event detection than for high-precision anomaly scoring.

# Acknowledgement

This project was supported by the French National Research Agency (ANR) through the BenchArk project (ANR-24-IAS2-0003) and EBUL project (ANR- 23- CE23-0001). Mansour Benbakoura was supported from a national grant attributed to the ExaDoST project of the NumPEx PEPR program, under the reference ANR-22-EXNU-0004. This work was performed using HPC resources from GENCI–IDRIS (Grant 2025- AD011015308R1).

# References

A. Alfons, C. Croux, and S. Gelper. Sparse least trimmed squares regression for analyzing highdimensional large data sets. The Annals of Applied Statistics, pages 226–248, 2013. 2

A. Beck and M. Teboulle. A fast iterative shrinkagethresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183–202, 2009. 3   
L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. SIAM Review, 60(2):223–311, 2018. 4   
A. Chambolle and C. Dossal. On the convergence of the iterates of the “fast iterative shrinkage/thresholding algorithm”. Journal of Optimization Theory and Applications, 166:968–982, 2015. 3   
S. Choudhary, S. Kumar, P. S. Siddhaarth, and G. Charitasri. Transforming blood cell detection and classification with advanced deep learning models: A comparative study. arXiv:2410.15670, 2024. 1   
D. Cornu, P. Salomé, B. Semelin, A. Marchal, J. Freundlich, S. Aicardi, X. Lu, G. Sainton, F. Mertens, F. Combes, and C. Tasse. YOLO-CIANNA: Galaxy detection with deep learning in radio data - I. A new YOLO-inspired source detection method applied to the SKAO SDC1. Astronomy & Astrophysics, 690: A211, 2024. 1   
D. F. Crouse. On implementing 2D rectangular assignment algorithms. IEEE Transactions on Aerospace and Electronic Systems, 52(4):1679–1696, 2016. 20   
Z. Z. Darban, G. I. Webb, S. Pan, C. Aggarwal, and M. Salehi. Deep learning for time series anomaly detection: A survey. ACM Computing Surveys, 57(1): 1–42, 2024. 2, 8   
A. Das, W. Kong, R. Sen, and Y. Zhou. A decoderonly foundation model for time-series forecasting. In Proceedings of the 41st International Conference on International Conference on Machine Learning (ICML), 2024. 8   
X. Deng, J. Xu, F. Gao, X. Sun, and M. Xu. DeepM2CDL: Deep multi-scale multi-modal convolutional dictionary learning network. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(5):2770–2787, 2023. 2   
L. Dragoni, R. Flamary, K. Lounici, and P. Reynaud-Bouret. Sliding window strategy for convolutional spike sorting with LASSO: Algorithm, theoretical guarantees and complexity. Acta Applicandae Mathematicae, 179(1), 2022. 1   
T. Dupré la Tour, T. Moreau, M. Jas, and A. Gramfort. Multivariate convolutional sparse coding for electromagnetic brain signals. In Advances in Neural Information Processing Systems (NeurIPS), volume 31, pages 3292–3302, 2018. 1, 4, 6

M. Giavalisco and the GOODS Teams. The great observatories origins deep survey: Initial results from optical and near-infrared imaging. The Astrophysical Journal Letters, 600(2):L93, 2004. 1   
K. Gregor and Y. LeCun. Learning fast approximations of sparse coding. In Proceedings of the 27th International Conference on International Conference on Machine Learning (ICML), page 399–406. Omnipress, 2010. 2   
R. Gribonval, R. Jenatton, and F. Bach. Sparse and spurious: Dictionary learning with noise and outliers. IEEE Transactions on Information Theory, 61(11): 6298–6319, 2015. 2, 9   
R. Grosse, R. Raina, H. Kwong, and A. Y. Ng. Shiftinvariant sparse coding for audio classification. Cortex, 8:9, 2007. 1   
B. Iglewicz and D. C. Hoaglin. Chapter 3: Outlier labeling. In How to Detect and Handle Outliers, volume 16 of The ASQC Basic References in Quality Control: Statistical Techniques. ASQC Quality Press, 1993. 5   
M. Jas, T. D. La Tour, U. Şimşekli, and A. Gramfort. Learning the morphology of brain signals using alphastable convolutional sparse coding. In Advances in Neural Information Processing Systems (NeurIPS), pages 1099–1108, 2017. 2   
J. Liu, C. Garcia-Cardona, B. Wohlberg, and W. Yin. Online convolutional dictionary learning. In Proceedings of the International Conference on Image Processing (ICIP), pages 1707–1711, 2017. 2   
Q. Liu and J. Paparrizos. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, 2024. 8   
E. J. d. S. Luz, W. R. Schwartz, G. Cámara-Chávez, and D. Menotti. ECG-based heartbeat classification for arrhythmia detection: A survey. Computer Methods and Programs in Biomedicine, 127:144–164, 2016. 1   
J. Mairal, F. Bach, J. Ponce, and G. Sapiro. Online learning for matrix factorization and sparse coding. Journal of Machine Learning Research, 11(1), 2010. 2   
B. Malézieux, T. Moreau, and M. Kowalski. Understanding approximate and unrolled dictionary learning for pattern recovery. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 2, 3, 4, 9

A. Mensch, J. Mairal, B. Thirion, and G. Varoquaux. Dictionary learning for massive matrix factorization. In Proceedings of the 33rd International Conference on Machine Learning (ICML), volume 48, pages 1737– 1746. PMLR, 2016. 2   
T. Moreau and A. Gramfort. DiCoDiLe: Distributed convolutional dictionary learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020. 1, 3, 6, 19   
F. Murat, O. Yildirim, M. Talo, U. B. Baloglu, Y. Demir, and U. R. Acharya. Application of deep learning techniques for heartbeats detection using ECG signals-analysis and review. Computers in Biology and Medicine, 120:103726, 2020. 1   
J. Paparrizos, Y. Kang, P. Boniol, R. S. Tsay, T. Palpanas, and M. J. Franklin. Tsb-uad: an end-to-end benchmark suite for univariate time-series anomaly detection. Proceedings of the VLDB Endowment, 15 (8):1697–1711, 2022. 8   
V. Papyan, Y. Romano, J. Sulam, and M. Elad. Convolutional dictionary learning via local processing. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), Oct 2017. 1   
A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in PyTorch. In Advances in Neural Information Processing Systems (NeurIPS), 2017. 6   
T. Penzel, G. B. Moody, R. G. Mark, A. L. Goldberger, and J. H. Peter. The Apnea-ECG database. Computers in Cardiology, 27:255–258, 2000. 7, 16   
P. J. Rousseeuw. Least median of squares regression. Journal of the American Statistical Association, 79 (388):871–880, 1984. 2   
P. J. Rousseeuw and A. M. Leroy. Robust Regression and Outlier Detection. John Wiley & Sons, 2005. 2   
L. Ruff, J. R. Kauffmann, R. A. Vandermeulen, G. Montavon, W. Samek, M. Kloft, T. G. Dietterich, and K.-R. Muller. A unifying review of deep and shallow anomaly detection. Proceedings of the IEEE, 109(5):756–795, 2021. 2, 8   
M. Scetbon, M. Elad, and P. Milanfar. Deep K-SVD denoising. IEEE Transactions on Image Processing, 30:5944–5955, 2021. 2   
S. Schmidl, P. Wenig, and T. Papenbrock. Anomaly detection in time series: a comprehensive evaluation. Proceedings of the VLDB Endowment, 15(9):1779– 1797, 2022. 2

C. Shyalika, R. Wickramarachchi, and A. P. Sheth. A comprehensive survey on rare event prediction. ACM Computing Surveys, 57(3), 2024. 2   
H. Tang, H. Liu, W. Xiao, and N. Sebe. When dictionary learning meets deep learning: Deep dictionary learning and coding network for image recognition with limited data. IEEE Transactions on Neural Networks and Learning Systems, 32(5):2129–2141, 2021. 2   
B. Tolooshams and D. E. Ba. Stable and interpretable unrolled dictionary learning. Transactions on Machine Learning Research (TMLR), 2022. 2, 3, 6, 9   
B. Tolooshams, S. Dey, and D. Ba. Scalable convolutional dictionary learning with constrained recurrent sparse auto-encoders. In Proceedings of the 28th IEEE International Workshop on Machine Learning for Signal Processing (MLSP), pages 1–6, 2018. 2   
S. Vaswani, A. Mishkin, I. Laradji, M. Schmidt, G. Gidel, and S. Lacoste-Julien. Painless stochastic gradient: Interpolation, line-search, and convergence rates. In Advances in Neural Information Processing Systems (NeurIPS), volume 32, pages 3732–3745, 2019. 4   
B. Wohlberg. Efficient algorithms for convolutional sparse representations. IEEE Transactions on Image Processing, 25(1):301–315, 2015. 1, 2   
B. Wohlberg. SPORCO: A Python package for standard and convolutional sparse representations. In Proceedings of the 15th Python in Science Conference, pages 1–8, 2017. 6   
H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 8   
J. Xu, H. Wu, J. Wang, and M. Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 8   
C.-C. M. Yeh, Y. Zhu, L. Ulanova, N. Begum, Y. Ding, H. A. Dau, D. F. Silva, A. Mueen, and E. Keogh. Matrix profile i: All pairs similarity joins for time series: A unifying view that includes motifs, discords and shapelets. In Proceedings of the 16th International Conference on Data Mining (ICDM), pages 1317–1322, 2016. 8   
F. Yellin, B. D. Haeffele, and R. Vidal. Blood cell detection and counting in holographic lens-free imaging by convolutional sparse dictionary learning and

coding. In Proceedings of the IEEE International Symposium on Biomedical Imaging (ISBI), pages 650–653, 2017. 1

Y. Zeng, J. Chen, and G.-B. Huang. Slice-based online convolutional dictionary learning. IEEE Transactions on Cybernetics, pages 1–14, 2019. 2

H. Zheng, H. Yong, and L. Zhang. Deep convolutional dictionary learning for image denoising. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 630–641, 2021. 2

# Checklist

1. For all models and algorithms presented, check if you include:   
(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes   
(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. Yes   
(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. Yes   
2. For any theoretical claim, check if you include:   
(a) Statements of the full set of assumptions of all theoretical results. Yes   
(b) Complete proofs of all theoretical results. Yes   
(c) Clear explanations of any assumptions. Yes   
3. For all figures and tables that present empirical results, check if you include:   
(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Yes   
(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). Yes   
(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Yes   
(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). Yes   
4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include: (a) Citations of the creator If your work uses existing assets. Yes

(b) The license information of the assets, if applicable. Not Applicable   
(c) New assets either in the supplemental material or as a URL, if applicable. Yes   
(d) Information about consent from data providers/curators. Yes   
(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Not Applicable

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. Not Applicable   
(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. Not Applicable   
(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. Not Applicable

# A ANALYTICAL STUDY

Proposition 2.1 (Stability of the common pattern). Consider a population of signals X composed of two patterns ${ \bf d } _ { a }$ and $\mathbf { d } _ { b } ,$ , with activations such that the patterns do not overlap in the signal, and corrupted by an additive Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \sigma I d )$ . Introduce $c = \mathbf { d } _ { a } ^ { \top } \mathbf { d } _ { b }$ and $\rho$ the proportion of rare-event pattern $\mathbf { d } _ { b }$ activations in the population. Then, in the noiseless setting,

i. ${ \bf d } _ { a }$ is a fixed point of the classical CDL algorithm for K = 1 if $c \leq \lambda$   
ii. $\mathbf { d } _ { b }$ is a fixed point of RoseCDL algorithm for $K = 1 \ i f c \leq \lambda$ or the RoseCDL algorithm is used with an outlier threshold trimming a proportion of windows greater than $\rho .$ .

Proof. We consider a dictionary $D \in \mathbb { R } ^ { 1 \times L }$ with a single atom d. As we consider signals composed of patterns with no overlap, we can separate each segment and we have a population of signals $X = z d _ { i } + \epsilon ,$ with $z \in \mathbb { R }$ , $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ and $d _ { i } = \mathbf { d } _ { a }$ with probability $1 - \rho$ and $\mathbf { d } _ { b }$ with probability $\rho ,$ with $\lVert \mathbf { d } _ { a } \rVert _ { 2 } = \lVert \mathbf { d } _ { b } \rVert _ { 2 } = 1$ . We consider all atoms $\mathbf { d } , \mathbf { d } _ { a } , \mathbf { d } _ { b }$ to be unit norm. Wlog, we can consider $z = 1 ,$ , as this amounts to rescaling the value of $\lambda _ { m a x } ,$ and we consider that $c _ { a } = \mathbf { d } ^ { \top } \mathbf { d } _ { a }$ and $c _ { b } = \mathbf { d } ^ { \top } \mathbf { d } _ { b }$ are positive, as we can consider −d otherwise. We also consider that the noise level is small enough such that $\sigma ^ { 2 } < c _ { j }$ .

This model is a simplified model in which we have a population of signals where we want to identify the pattern of an event ${ \bf d } _ { a }$ from the pattern of a rare event $\mathbf { d } _ { b }$ .

In this setting, if we further have that the auto-correlation of d with ${ \bf d } _ { a }$ and $\mathbf { d } _ { b }$ is maximal when they are aligned, then the sparse coding of a signal X can be computed with the following formula:

$$
z ^ {*} (X, \mathbf {d}) = \left\{ \begin{array}{l l} 0 & \text { if } c + \epsilon^ {\top} \mathbf {d} \leq \lambda \\ c + \epsilon^ {\top} \mathbf {d} - \lambda & \text { otherwise } \end{array} \right. \tag {7}
$$

with $c = \mathbf { d } ^ { \top } d _ { i }$ , which has value $c _ { a }$ with probability $1 - \rho$ and $c _ { b }$ otherwise.

We can compute the loss value for this $z ^ { * } ( X , \mathbf { d } )$ for X where $z ^ { * }$ is non-zero:

$$
\begin{array}{l} F (\mathbf {d}, z ^ {*}; X) = \frac {1}{2} \| X - z ^ {*} \mathbf {d} \| _ {2} ^ {2} + \lambda \| z ^ {*} \| _ {1} (8) \\ = \frac {1}{2} \left(\| X \| _ {2} ^ {2} - 2 (c + \epsilon^ {\top} \mathbf {d} - \lambda) (c + \epsilon^ {\top} \mathbf {d}) + \| (c + \epsilon^ {\top} \mathbf {d} - \lambda) \mathbf {d} \| _ {2} ^ {2}\right) + \lambda | c + \epsilon^ {\top} \mathbf {d} - \lambda | (9) \\ = \frac {1}{2} \left(\| X \| _ {2} ^ {2} - 2 (c + \epsilon^ {\top} \mathbf {d} - \lambda) (c + \epsilon^ {\top} \mathbf {d}) + (c + \epsilon^ {\top} \mathbf {d} - \lambda) ^ {2} + 2 \lambda (c + \epsilon^ {\top} \mathbf {d} - \lambda)\right) (10) \\ = \frac {1}{2} \left(\| X \| _ {2} ^ {2} - 2 (c + \epsilon^ {\top} \mathbf {d} - \lambda) (c + \epsilon^ {\top} \mathbf {d} - \lambda) + (c + \epsilon^ {\top} \mathbf {d} - \lambda) ^ {2}\right) (11) \\ = \frac {1}{2} \left(\| X \| _ {2} ^ {2} - (c + \epsilon^ {\top} \mathbf {d} - \lambda) ^ {2}\right) (12) \\ = \frac {1}{2} \left(\| d _ {i} \| _ {2} ^ {2} - (c - \lambda) ^ {2} + \| \epsilon \| _ {2} ^ {2} - (\epsilon^ {\top} \mathbf {d}) ^ {2} - 2 (1 - (c - \lambda)) \epsilon^ {\top} \mathbf {d}\right) (13) \\ \end{array}
$$

Taking the expectation over the noise yields:

$$
\mathbb {E} _ {\epsilon} \left[ F (\mathbf {d}, z ^ {*}; X) \right] = \frac {1}{2} (1 - (c - \lambda) ^ {2} + (L - 1) \sigma^ {2}) \tag {15}
$$

For c between λ and 1, this function is decreasing in $c ,$ meaning that for two samples constructed with ${ \bf d } _ { a }$ and $\mathbf { d } _ { b } .$ if the correlation $c _ { 0 } = \mathbf { d } ^ { \top } \mathbf { d } _ { a }$ is larger than the correlation $c _ { b } = \mathbf { d } ^ { \top } \mathbf { d } _ { b }$ , then the reconstruction loss for sample 0 is smaller in expectation than the reconstruction loss for a sample 1.

We can also compute the gradient of this function with respect to d. Note that with the KKT condition defining $z ^ { * }$ , we have that the $\nabla _ { z } F ( \mathbf { d } , z * ; X ) = 0$ , and thus we do not need to compute the Jacobian of $z ^ { * }$ when computing the derivative of F with respect to d. The gradient reads:

$$
\nabla_ {\mathbf {d}} F (\mathbf {d}, z ^ {*}; X) = z ^ {*} \left(z ^ {*} \mathbf {d} - X\right) \tag {16}
$$

$$
= \left(z ^ {*}\right) ^ {2} \mathbf {d} - z ^ {*} X \tag {17}
$$

$$
= (c + \epsilon^ {\top} \mathbf {d} - \lambda) ^ {2} \mathbf {d} - (c + \epsilon^ {\top} \mathbf {d} - \lambda) (d _ {i} + \epsilon) \tag {18}
$$

Taking the expectation over the noise yields:

$$
\mathbb {E} _ {\epsilon} \left[ \nabla_ {\mathbf {d}} F (\mathbf {d}, z ^ {*}; X) \right] = ((c - \lambda) ^ {2} + \sigma^ {2}) \mathbf {d} - (c - \lambda) d _ {i} + \underbrace {\mathbb {E} _ {\epsilon} \left[ \epsilon^ {\top} \mathbf {d} \epsilon \right]} _ {\sigma^ {2} \mathbf {d}} \tag {19}
$$

$$
= ((c - \lambda) ^ {2} + 2 \sigma^ {2}) \mathbf {d} - (c - \lambda) d _ {i} \tag {20}
$$

This yields

$$
\mathbb {E} [ \nabla_ {\mathbf {d}} F (\mathbf {d}, z ^ {*}; X) ] = \left((1 - \rho) (c _ {a} - \lambda) ^ {2} + \rho (c _ {b} - \lambda) ^ {2} + 2 \sigma^ {2}\right) \mathbf {d} - (1 - \rho) (c _ {a} - \lambda) \mathbf {d} _ {a} - \rho (c _ {b} - \lambda) \mathbf {d} _ {b}
$$

In the noiseless case, if $\mathbf { d } = \mathbf { d } _ { a } .$ , and $\lambda \leq c _ { b } = ( \mathbf { d } _ { b } ) ^ { \top } \mathbf { d } _ { a } < 1$ , with the classical algorithm, the expected gradient reads

$$
\mathbb {E} _ {X} \left[ \nabla_ {\mathbf {d}} F \left(\mathbf {d} _ {a}, z ^ {*}; X\right) \right] = - (1 - \rho) \lambda (1 - \lambda) \mathbf {d} _ {a} + \rho \left((c _ {a} - \lambda) ^ {2} \mathbf {d} _ {a} - (c _ {b} - \lambda) \mathbf {d} _ {b}\right) \tag {21}
$$

$$
= \left(\rho (c _ {a} - \lambda) ^ {2} - (1 - \rho) \lambda (1 - \lambda)\right) \mathbf {d} _ {a} - \rho (c _ {b} - \lambda) \mathbf {d} _ {b} \tag {22}
$$

This gradient is not colinear with $\mathbf { d } _ { a } .$ showing that ${ \bf d } _ { a }$ is not a fixed point of the projected gradient descent algorithm in this context. Even in a noiseless and very simple setting, the ${ \bf d } _ { a }$ is not a solution of the Classical CDL algorithm.

In contrast, when using the least trimmed square procedure with a trimming threshold rejecting a proportion $\rho$ of the samples, we can show that ${ \bf d } _ { a }$ is a fixed point in the noiseless setting. As seen in (15), the loss for samples X associated with $\mathbf { d } _ { b }$ is smaller than the loss for samples associated with $\mathbf { d } _ { a } .$ and therefore rejecting ρ samples from the gradient computation leads to:

$$
\mathbb {E} _ {X} \left[ \nabla_ {\mathbf {d}} F (\mathbf {d} _ {a}, z ^ {*}; X) \right] = - (1 - \rho) \lambda (1 - \lambda) \mathbf {d} _ {a} \tag {23}
$$

as the gradient is colinear with ${ \bf d } _ { a }$ , thus ${ \bf d } _ { a }$ is a fixed point of the projected gradient descent and of the learning procedure. □

# B ADDITIONAL EXPERIMENTS

In this section, we present two numerical experiments intended to characterize the behavior of RoseCDL. We first discuss the influence of the regularization parameter λ on the ability of RoseCDL to learn the atoms. Then, we highlight the importance of the inline outlier detection module in making the CDL robust to outliers.

# B.1 Scalability

To thoroughly assess scalability, we conduct additional experiments varying both window sizes and signal lengths.

# B.1.1 Window size

For window size analysis on 1D signals with T = 100,000 and λ = 0.8, we evaluate both Adam and SLS optimizers across window sizes ranging from 10L to 100L. To ensure full GPU utilization, we maintain a constant product window size × batch size. As shown in Tab. B.1, RoseCDL consistently achieves validation losses within 4% of AlphaCSC performance across all configurations, while maintaining runtimes of 12–22% relative to AlphaCSC, corresponding to approximately 5× speedup.

<table><tr><td rowspan="2">Optimizer Window size</td><td colspan="4">Adam</td><td colspan="4">SLS</td></tr><tr><td>10L</td><td>20L</td><td>50L</td><td>100L</td><td>10L</td><td>20L</td><td>50L</td><td>100L</td></tr><tr><td>Validation loss</td><td>+3.0%</td><td>+3.8%</td><td>+2.7%</td><td>+2.7%</td><td>-0.2%</td><td>-0.2%</td><td>-0.2%</td><td>-0.4%</td></tr><tr><td>Runtime</td><td>15.4%</td><td>12.3%</td><td>12.5%</td><td>12.7%</td><td>21.7%</td><td>21.2%</td><td>16.3%</td><td>17.6%</td></tr></table>

Table B.1: Comparison of Adam and SLS with different window sizes. Validation loss expressed relatively to AlphaCSC’s. Runtime expressed in proportion to AlphaCSC’s.

Notably, the SLS optimizer demonstrates superior convergence properties with validation losses within 0.4% of AlphaCSC, though with slightly increased runtime compared to Adam. The results validate that border effects do not hinder convergence with our stochastic windowing approach, even for small window sizes (as low as 10L).

# B.1.2 Signal length

For signal length scalability, we evaluate performance on signals ranging from 10k to 1M time samples. As detailed in Tab. B.2, RoseCDL demonstrates sublinear scaling, with runtimes growing from 15.2 s (10k samples) to 202.8 s (1M samples), while AlphaCSC becomes computationally prohibitive beyond 100k samples. This superior scalability enables RoseCDL to process signals substantially larger than existing full-signal sparse coding methods.

<table><tr><td>T</td><td>10k</td><td>30k</td><td>100k</td><td>300k</td><td>1M</td></tr><tr><td>Runtime RoseCDL (s)</td><td>15.2</td><td>16.4</td><td>32.6</td><td>68.0</td><td>202.8</td></tr><tr><td>Runtime AlphaCSC (s)</td><td>67.5</td><td>102.8</td><td>198.5</td><td>N/A</td><td>N/A</td></tr></table>

Table B.2: Signal length scaling: runtime comparison between RoseCDL and AlphaCSC for varying signal sizes T .

# B.2 Choice of the outlier detection method

To evaluate the performance of RoseCDL on 2D data, we constructed a semi-synthetic dataset of images by generating 5000 characters sampled from a set of four letters (R, O, S, and E) along with spaces. These images emulate text-like documents composed of words formed from the selected characters. We added the letter Z in a small proportion to introduce rare events. Experiments were conducted with a 10% contamination rate. Consequently, for the inline outlier detection methods, we implemented the quantile detection method with α value: 10 %, alongside with MAD and z-score methods as described in Sec. 2.2. To assess the efficiency of the inline outlier module on this dataset, we compared it to RoseCDL without the inline outlier module, where we computed the threshold by computing the reconstruction error after the learning part. We show the detection over the epochs using different outlier detection algorithms by computing the F1 score between the inline model mask for the during procedure and the mask after the reconstruction by a RoseCDL model without an inline outlier method for the after procedure in Fig. B.1. The inline outlier detection module, particularly the MAD method, works by discarding the rare events from the dictionary learning part. These rare events are reconstructed with lower fidelity compared to the more prevalent patterns. This selective degradation results in higher reconstruction errors that are sharply localized at the rare events’ positions and improves their detection. In opposition to that, in the absence of this module, rare events are still reconstructed less accurately than common patterns; however, the contrast in reconstruction quality is less distinct. This diminishes the precision of rare event detection and ultimately leads to a lower F1 score.

![](images/71ab6175b11526ec8edd45fb8104e4e7f9c78c64533719933a8ac3e67a53b409.jpg)

<details>
<summary>line</summary>

| Epoch | MAD (3.5) | Quant. (0.1) | Z-score (1.5) |
|-------|-----------|--------------|---------------|
| 0     | 0.0       | 0.1          | 0.2           |
| 5     | 0.6       | 0.45         | 0.5           |
| 10    | 0.75      | 0.48         | 0.6           |
| 15    | 0.76      | 0.49         | 0.62          |
| 20    | 0.77      | 0.5          | 0.63          |
| 25    | 0.77      | 0.5          | 0.63          |
| 27    | 0.77      | 0.5          | 0.63          |
</details>

Figure B.1: Comparison of the F1 score evolution of different methods over the epochs on rare event detection task on the (R, O, S, E, and Z).

# B.3 Inline outlier detection on real-world data

![](images/66f18bb64e3bf7dd16ad36194eb6a0eda60b3ec93448f613a4bcbe472d875c1b.jpg)  
Figure B.2: Learned atoms with and without outliers detection method, on 10 bad trials of subject a02 of dataset Physionet Apnea-ECG.

We evaluate our approach through two complementary experiments. First, we consider the Physionet Apnea-ECG dataset (Penzel et al., 2000) without any preprocessing, thereby preserving the raw variability of the signals. From a 10 min ECG segment containing blocks of anomalies, we learned a three-atom dictionary (1 s each) to assess robustness against corrupted data (Fig. 2). Without outlier detection, the model converges to noise-like patterns, whereas the inline anomaly detection mechanism suppresses high-variance anomalies and enables recovery of meaningful ECG atoms (Fig. B.2). While careful parameter tuning can sometimes yield similar results, it remains unreliable and computationally costly.

# B.4 Influence of the regularization parameter

A key parameter of RoseCDL is the regularization coefficient λ, introduced in Sec. 2, which controls the sparsity of the learned activation vector. We express λ as a fraction of the maximum regularization value, $\lambda _ { \operatorname* { m a x } } ,$ corresponding to the smallest regularization value for which the activation vector is entirely zero. Note that when using the trimmed objective, $\lambda _ { \mathrm { m a x } }$ computation is adapted to account for the modified loss. To analyze the effect of regularization on dictionary recovery, we compare recovery scores across different values of λ using synthetic data. The results, presented in Fig. B.3, demonstrate how varying λ influences the quality of the recovered dictionary. Our findings indicate that the best dictionary recovery is achieved when setting $\lambda = 0 . 1 \lambda _ { \operatorname* { m a x } }$ , effectively balancing sparsity and reconstruction accuracy.

![](images/7cbda754a0db62a23b92be4ed60f5c66436619b6de459f831d77d17234a24ef7.jpg)

<details>
<summary>line</summary>

| Epoch | λ=0.11 | λ=0.07 | λ=0.03 | λ=0.09 | λ=0.05 | λ=0.01 |
|-------|--------|--------|--------|--------|--------|--------|
| 0     | 0.2    | 0.2    | 0.2    | 0.2    | 0.2    | 0.2    |
| 5     | 0.95   | 0.85   | 0.65   | 0.9    | 0.7    | 0.5    |
| 10    | 0.98   | 0.9    | 0.75   | 0.95   | 0.8    | 0.6    |
| 15    | 0.99   | 0.95   | 0.8    | 0.98   | 0.85   | 0.65   |
| 20    | 0.995  | 0.97   | 0.82   | 0.99   | 0.87   | 0.67   |
| 25    | 0.998  | 0.98   | 0.83   | 0.995  | 0.88   | 0.68   |
| 30    | 1.0    | 0.99   | 0.84   | 1.0    | 0.89   | 0.69   |
</details>

![](images/1597f3fd503d204f7f51ccf71c6f1924f17d1a34c872942d0fc64b6adc5ac2b0.jpg)  
Figure B.3: (a) Impact of the regularization on the recovery score over the epochs. (b) Atoms recovered at the end of training for different regularization values.

# B.5 Anomalies : Missed and detected examples

![](images/02d10684b2d41d759c162232d4c8587e4189ee54780b996af0a54a40798cc2a9.jpg)

<details>
<summary>line</summary>

| Time   | Value |
| ------ | ----- |
| 10800  | -0.7  |
| 10900  | 1.2   |
| 11000  | -0.9  |
| 11100  | 1.3   |
| 11200  | -0.4  |
| 11300  | 1.5   |
| 11400  | -0.3  |
| 11500  | 1.4   |
| 11600  | -0.8  |
</details>

![](images/65d1d285931966d8a400ebba01db0a1ec47074614e581b6808a8cc5311912699.jpg)

<details>
<summary>line</summary>

| Time    | Value |
| ------- | ----- |
| 152600  | -0.5  |
| 152800  | -0.2  |
| 153000  | 0.1   |
| 153200  | -0.3  |
| 153400  | -0.7  |
</details>

Figure B.4: Examples of missed and detected anomalies on the MITDB Arythmia dataset.

![](images/75b89d8d829d69be7b4aed9daf739e8c939386f00a8223226f25ad3d44045546.jpg)

<details>
<summary>line</summary>

| Time | Value |
|------|-------|
| 0    | ~0.5  |
| 100  | ~-2.5 |
| 200  | ~-1.5 |
| 300  | ~-3.5 |
| 400  | ~-2.5 |
| 500  | ~-1.5 |
| 600  | ~-4.5 |
| 700  | ~-1.5 |
</details>

![](images/4541d0d1c653c00570fbacf42e5b5666dab47cd1a348ac48dd6b24617ddd7d65.jpg)

<details>
<summary>line</summary>

| Time   | Value |
|--------|-------|
| 50600  | ~0    |
| 50800  | ~-2   |
| 51000  | ~-4   |
| 51200  | ~-2   |
| 51400  | ~-1   |
</details>

![](images/8a73d2bf45a21b4444092c4b724630ca755c39bf3fd8a2ce446bf530e4f152f3.jpg)

<details>
<summary>line</summary>

| Time  | Value |
|-------|-------|
| 2800  | ~0    |
| 3000  | ~-2   |
| 3200  | ~2    |
| 3400  | ~-2   |
| 3600  | ~-2   |
| 3800  | ~-2   |
</details>

![](images/0a143c1e331bfcd37b93b145b670d559b9fd64bf19122d36a0ca70f666835906.jpg)

<details>
<summary>line</summary>

| Time | True Anomaly | Predicted Anomaly |
|------|--------------|-------------------|
| 0    | 0            | 0                 |
| 100  | 0            | 0                 |
| 200  | 0            | 0                 |
| 300  | 0            | 0                 |
| 400  | -2           | 0                 |
| 500  | 0            | 0                 |
| 600  | 0            | 0                 |
| 700  | 0            | 0                 |
| 800  | 0            | 0                 |
| 900  | 0            | 0                 |
</details>

![](images/0ea4794e8e121d780199d4fd09acefca3dcf47ac4d41e4bd03e1c952b9fd84c1.jpg)

<details>
<summary>line</summary>

| Time | Value |
|------|-------|
| 600  | ~5    |
| 700  | ~40   |
| 800  | ~60   |
| 900  | ~20   |
| 1000 | ~50   |
| 1100 | ~30   |
| 1200 | ~5    |
| 1300 | ~45   |
| 1400 | ~20   |
| 1500 | ~5    |
| 1600 | ~30   |
</details>

![](images/e6ab57e5716f347df99f53cfebedd5492a9126d862bc8b88a2abefc52fbea6b4.jpg)

<details>
<summary>line</summary>

| Time   | True Anomaly | Predicted Anomaly |
|--------|--------------|-------------------|
| 27000  | 0            | 0                 |
| 27200  | 50           | 0                 |
| 27400  | 0            | 0                 |
| 27600  | 50           | 0                 |
| 27800  | 50           | 0                 |
</details>

![](images/a638f2c9d4cd22441ad2dc9150538ae4dd12fa962d2e8a1fb1883a1cdf1e8138.jpg)

<details>
<summary>line</summary>

| Time   | Value |
| ------ | ----- |
| 20600  | ~0    |
| 20800  | ~0    |
| 21000  | ~0    |
| 21200  | ~0    |
| 21400  | ~0    |
</details>

![](images/19b6b6be8505e0884f693fb925f9fe7fef280155ad4404edc9086abbc48e9789.jpg)

<details>
<summary>line</summary>

| Time   | Value |
| ------ | ----- |
| 21600  | -500  |
| 21800  | 200   |
| 22000  | -300  |
| 22200  | 1500  |
| 22400  | -800  |
| 22600  | 100   |
</details>

Figure B.5: Examples of missed (left) and detected (right) anomalies across datasets: MITDB, SVDB, ECG, Dodgers, and Daphnet.

# C DATA SIMULATION

The synthetic multivariate 1D signals $X \in \mathbb { R } ^ { P \times T }$ used in Sect. 3 are generated from a dictionary $D \in \mathbb { R } ^ { K \times P \times L }$ a sparse activation vector $Z \in \mathbb { R } ^ { \breve { K } \times ( T - L + 1 ) }$ , and a random Gaussian noise $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ as $X = D * Z + \varepsilon$ . In this definition,

• P is the number of channels,   
• T is the length of the signal,   
• K is the number of atoms,   
• L is the length of the atoms.

In the experiments conducted in Sect. 3, we generated signals of length T = 50 000 with P = 2 channels from dictionaries with K = 2 atoms of length L = 64. The atoms were generated from sine and gaussian waveforms, as illustrated in Fig. C.1. The activations Z were randomly generated sparse Dirac combs with sparsity 0.4 % and the noise level was set to $\sigma = 0 . 1$ .

![](images/04c3c22edef51a3f77d0f1bc8f0a5f364de8ed1ecac86a201de7be5212760ecb.jpg)

<details>
<summary>line</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.0   |
| 1    | 0.05  |
| 2    | 0.1   |
| 3    | 0.08  |
| 4    | 0.03  |
| 5    | -0.02 |
| 6    | -0.07 |
| 7    | -0.12 |
| 8    | -0.08 |
| 9    | 0.0   |
</details>

![](images/e0a61a905060301a79a9f2b5a526fb8aae559f8a3eb92eb6b99a239c397ed336.jpg)

<details>
<summary>line</summary>

| Point | Value |
|-------|-------|
| 1     | 0.0   |
| 2     | 0.5   |
| 3     | 1.0   |
| 4     | 0.5   |
| 5     | 0.0   |
</details>

![](images/6a622e4cb0e070a81b3a71e2a5bbe89bd1475a4a6a06e388ede96e9058e45e28.jpg)

<details>
<summary>line</summary>

| Time | Value |
|------|-------|
| 0    | 0.0   |
| 10   | 0.0   |
| 20   | 0.1   |
| 30   | 0.0   |
| 40   | 0.1   |
| 50   | 0.0   |
| 60   | 0.0   |
</details>

![](images/9e40677895c3ef9d1cd889352187b8753d1676eb702cadcf2ea3cbc5a8e54530.jpg)

<details>
<summary>line</summary>

| Time | Value |
|------|-------|
| 0    | 0     |
| 10   | 2     |
| 20   | -2    |
| 30   | 2     |
| 40   | 4     |
| 50   | -2    |
| 60   | 0     |
</details>

Figure C.1: True dictionary in experiments on synthetic data.

# D DICTIONARY EVALUATION

In our methodology, we evaluate the effectiveness of a learned dictionary, denoted as $\widehat { \mathbf { D } } \in \mathbb { R } ^ { K ^ { \prime } \times P \times L ^ { \prime } }$ , by comparing it against a set of true dictionary patterns, represented as $\mathbf { D } \in \mathbb { R } ^ { K \times P \times L }$ band computing a “recovery score”, using the convolutional cosine similarity following optimal assignment, as defined by Moreau and Gramfort (2020). The learned dictionary and the true patterns are structured as three-dimensional arrays, where dimensions correspond to the number of atoms, channels, and atoms’ duration. The learned dictionary may differ from the true dictionary in terms of the number of atoms and the length of time atoms, typically featuring more atoms and extended durations.

The evaluation process involves a computational step known as multi-channel correlation. In this step, each atom of the learned dictionary is systematically compared with each pattern in the true dictionary. This comparison is carried out channel by channel, aggregating the results to capture the overall similarity between the dictionary atom and the pattern.

After performing these comparisons for all combinations of atoms and patterns, we create a matrix that represents the correlation strengths between each pair. To objectively assess the quality of the learned dictionary, we use an optimization technique called the Hungarian algorithm. This algorithm finds the best possible “matching” between the learned dictionary atoms and the true patterns, aiming to maximize the overall correlation.

The final score, which quantifies the performance of the learned dictionary, is derived by averaging the values of these optimal matchings. This score is scaled between 0 and 1, where 1 represents the best possible performance. A higher score indicates that the learned dictionary more accurately represents the true dictionary patterns, providing a measure of its quality and effectiveness in capturing the essential features of the data.

Mathematically, the recovery score between the dictionaries $\hat { \bf D }$ and D can be expressed as follow:

$$
\text { score } = \frac {1}{K} \sum_ {i = 1} ^ {K} C _ {i, j ^ {*} (i)} , \tag {24}
$$

where $j ^ { * } ( i ) , i = 1 , \ldots , K$ denote the results of the linear sum assignment problem $( \mathrm { C r o u s e } , 2 0 1 6 ) ^ { 3 }$ on correlation matrix $C : = \mathrm { C o r r } \left( \mathbf { D } , \widehat { \mathbf { D } } \right) \in \mathbb { R } ^ { K \times K ^ { \prime } }$ , with $\forall i \in \mathbb { I } ^ { 1 , K } \mathbb { I } , \forall j \in \mathbb { I } ^ { 1 , K ^ { \prime } } \mathbb { I }$ ,

$$
C _ {i, j} = \max _ {l = 1, \dots , L + L ^ {\prime} - 1} \operatorname{Corr} _ {2 \mathrm{D}} \left(D _ {i}, \widehat {D} _ {j}\right) [ l ] \in \mathbb {R}, \tag {25}
$$

where $D _ { i } \in \mathbb { R } ^ { P \times L }$ and $\widehat { D } _ { j } \in \mathbb { R } ^ { P \times L ^ { \prime } }$ . The multivariate “2D” correlation between the two matrices D and $\widehat { D }$ is defined as follow:

$$
\operatorname{Corr} _ {2 \mathrm{D}} (D, \widehat {D}) = \sum_ {p = 1} ^ {P} \operatorname{Corr} _ {1 \mathrm{D}} \left(d _ {p}, \hat {d} _ {p}\right) \in \mathbb {R} ^ {L + L ^ {\prime} - 1}, \tag {26}
$$

where $d _ { p } \in \mathbb { R } ^ { L }$ and $\hat { d } _ { p } \in \mathbb { R } ^ { L ^ { \prime } }$ . The 1D $^ { 6 6 } f u l l ^ { , 9 }$ correlation between the two vectors d and $\hat { d }$ is defined as follow, $\forall t \in \mathbb { [ 1 , \bar { \mu } + \mu ^ { \prime } - 1 ] }$ :

$$
\operatorname{Corr} _ {1 \mathrm{D}} (d, \hat {d}) [ t ] = (d * \hat {d}) [ t - T + 1 ] = \sum_ {l = 1} ^ {L} d [ l ] \hat {d} [ l - t + T ] \in \mathbb {R}, \tag {27}
$$

where $T : = \operatorname* { m a x } \left( L , L ^ { \prime } \right)$ .

# E EXPERIMENTS SETUP

<table><tr><td>Experiment</td><td>Data size</td><td>Number of runs</td><td>Hardware</td></tr><tr><td>Runtime 1D</td><td>20 signals, 50,000 data points, 2 channels</td><td>50</td><td>GPU NVIDIA A40, 30 CPUs</td></tr><tr><td>Runtime 2D</td><td>2000×2000 grayscale images</td><td>20</td><td>GPU NVIDIA A40</td></tr><tr><td>Regularization impact 1D</td><td>10 signals, 30,000 data points, 1 channel</td><td>10</td><td>GPU NVIDIA A40</td></tr><tr><td>Regularization impact 2D</td><td>2000×2000 grayscale images</td><td>10</td><td>GPU NVIDIA A40</td></tr><tr><td>Inline vs After</td><td>2D image data</td><td>10</td><td>GPU NVIDIA A40</td></tr><tr><td>Physionet</td><td>10 trials of subject a02</td><td>10</td><td>GPU NVIDIA A40 and CPUs</td></tr></table>