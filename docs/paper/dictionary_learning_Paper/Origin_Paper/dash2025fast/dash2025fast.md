# Fast Structured Orthogonal Dictionary Learning using Householder Reflections

Anirudh Dash

Department of Electrical Engineering

Indian Institute of Technology, Hyderabad

Aditya Siripuram

Department of Electrical Engineering

Indian Institute of Technology, Hyderabad

Abstract—In this paper, we propose and investigate algorithms for the structured orthogonal dictionary learning problem. First, we investigate the case when the dictionary is a Householder matrix. We give sample complexity results and show theoretically guaranteed approximate recovery (in the $l _ { \infty }$ sense) with optimal computational complexity. We then attempt to generalize these techniques when the dictionary is a product of a few Householder matrices. We numerically validate these techniques in the sample-limited setting to show performance similar to or better than existing techniques while having much improved computational complexity.

Index Terms—Fast dictionary learning, Householder matrices, optimal computational complexity, orthogonal dictionary, sample-limited setting

# I. INTRODUCTION

The orthogonal dictionary learning problem is posed as follows: Given a matrix $\mathbf { Y } \in \mathbb { R } ^ { n \times p }$ , can we find an orthogonal matrix V ∈ $\mathbb { R } ^ { n \times n }$ and a coefficient matrix $\mathbf { X } \in \mathbb { R } ^ { n \times p }$ such that Y = VX? Variants of this problem appear in standard sparse signal processing literature [1] and signal processing-based graph learning approaches [2], [3]. Prior work in [4], [5] developed fast dictionary learning approaches assuming additional structure on the orthogonal matrix. The goal of this work is to build on this line of investigation to obtain recovery guarantees on V and X and sample complexity bounds under strong structural assumptions on the orthogonal matrix. We then attempt to extend some algebraic ideas from the solution to the case when some of the structural assumptions on V are relaxed.

The standard unstructured dictionary learning problem (Y = DX) has been well investigated in literature. We refer to [6], [7], [8], [9], [10] as a few references. The case when the dictionary is orthogonal is also well investigated: algorithms based on alternate minimization have been proposed [11], [12]. Theoretical results pertinent to the above problem are usually of two kinds: proving the validity of proposed algorithms and identifying fundamental conditions (i.e., sample complexity or the number of columns p required) for any algorithm to recover the factors V and X.

This work focuses on the problem of orthogonal dictionary learning and is motivated by the following observations:

1) Some applications, for example, graph learning, place additional structural assumptions on the orthogonal dictionary: for e.g. in graph learning, the orthogonal matrix V is known to be an eigenvector matrix of a suitable graph.   
2) Even for unstructured orthogonal dictionary learning, attempts have been made to speed up the dictionary computation by approximating the dictionary as a structured orthogonal matrix [4], [5]. Most of the existing work is on unstructured orthogonal dictionary learning [13], while work on structured orthogonal matrices in [4], [5] doesn’t have sample complexity results.   
3) Most of the existing techniques are iterative and are sensitive to initialization [4], [5], [14].

We start the above investigation by assuming that the orthogonal matrix is a Householder matrix, similar to [4]. We note that every orthogonal matrix can be expressed as a product of Householder matrices [15], [16], thus allowing for the development of a new procedure to solve the orthogonal dictionary factorization problem.

In this paper, we first analyze sample complexity for Householder matrices. By imposing a statistical model on the coefficient matrix X, we show that recovery is possible with only $\Omega ( \log n )$ columns in Y in the $l _ { \infty }$ sense. The algorithm proposed utilizes the statistics of Y and is a non-iterative approach with theoretical guarantees for recovery. The computational complexity in learning the dictionary is O(np), which is substantially smaller than previous methods such as [4], [17]. We then generalize these ideas to a product of multiple Householder matrices.

# II. PROBLEM FORMULATION AND RESULT SUMMARY

Consider the setup of the unstructured orthogonal dictionary learning problem $\mathbf { Y } = \mathbf { V } \mathbf { X }$ , where $\textbf { Y } ~ \in ~ \mathbb { R } ^ { n \times p }$ is the data matrix, $\textbf { X } \in \mathbb { R } ^ { n \times p }$ is an (unknown) sparse representation matrix and $\mathbf { V } = \mathbf { H _ { 1 } } \mathbf { H _ { 2 } } \dots \mathbf { H _ { m } } \in \mathbb { R } ^ { n \times n }$ is a product of m Householder matrices $\mathbf { H _ { 1 } } , \mathbf { H _ { 2 } } , \dots , \mathbf { H _ { m } } ;$ with $\mathbf { H } _ { \mathbf { i } } = \mathbf { I } - \mathbf { 2 } \mathbf { u } _ { \mathbf { i } } \mathbf { u } _ { \mathbf { i } } ^ { \mathsf { T } }$ , where ui are (unknown) unit-norm vectors. Given the data matrix Y, we want to estimate V and X.

We refer by $u _ { i }$ the entries of the vector u, and denote by $\lvert \lvert \mathbf { u } \rvert \rvert _ { \infty } = \operatorname* { m a x } \lvert u _ { i } \rvert .$ , the infinity-norm of u. We denote by $| | \mathbf { A } | | _ { F }$ (with $| | \mathbf { A } | | _ { F } ^ { 2 } = \operatorname { T r a c e } ( \mathbf { A } ^ { \top } \mathbf { A } ) )$ the Frobenius norm of matrix A. We refer by $X _ { i j }$ the entries of matrix X.

We use the following sparsity model on X: the support is drawn from an iid Bernoulli distribution with parameter θ:

$$
X _ {i j} \neq 0 \text {   w.p.   } \theta , 0 \text {   w.p.   } 1 - \theta . \tag {1}
$$

All entries on the support are drawn from an i.i.d Uniform distribution in the range [1, 2]. Let µ be the mean of this distribution. We assume that θ and $\mu$ are known. We attempt to investigate the following questions in this work:

1) How many columns p in Y are required to estimate V to reasonable accuracy?   
2) How does the recovery of V depend on the sparsity θ in X and how robust is this recovery to errors in Y ?   
3) What is the computational complexity of this estimate?

In section IV, we analyze the case when $m \ = \ 1$ . We show that with $p = \Omega ( \log n ) ^ { 1 }$ columns in Y, it is possible to recover the underlying vector u accurately in the $l _ { \infty }$ sense with $O ( n p )$ computations non-iteratively (as opposed to $O ( n ^ { 2 } p )$ per iteration with a standard Procrustes based solution (see Section III)). Building on the ideas from Section IV, we propose algorithms for the case $m > 1$ in Section V. We demonstrate with numerical experiments that the proposed algorithm improves both approximation error and

1Note that we say $f ( n ) = \Omega ( g ( n ) ) { \mathrm { ~ i f ~ } } | f ( n ) | \geq C | g ( n ) |$ | for some constant C for all n large enough.

computational performance compared to existing solutions when the number of columns p is low.

We also use the following: if H is an $n \times n$ Householder matrix, computing Hx for a vector x costs $O ( n )$ arithmetic operations, as opposed to $O ( n ^ { 2 } )$ for an arbitrary matrix.

# III. PRIOR WORK

The standard orthogonal dictionary learning problem is formulated as follows (given a dataset $\mathbf { Y } \in \mathbb { R } ^ { n \times p }$ and a fixed sparsity level of s): arg mi $\mathbf { \mathsf { 1 } } _ { \mathbf { V } , \mathbf { X } } \{ \| \mathbf { Y } - \mathbf { V } \mathbf { X } \| _ { F } ^ { 2 } : \mathbf { V } ^ { \mathsf { T } } \mathbf { V } = \mathbf { I } , \quad \| \mathbf { x } _ { i } \| _ { 0 } \leq s , 1 \leq i \leq n \}$ , where ∥xi∥0 is the number of non-zero entries in the $i ^ { \mathsf { t h } }$ column of X.

Solving this involves a standard alternating minimization solution [11]: When V is fixed, the estimate $\hat { \mathbf X }$ is updated by thresholding the product $\mathbf { V } ^ { \mathsf { T } } \mathbf { Y }$ . When X is fixed, the estimate $\hat { \textbf { V } }$ is updated via Orthogonal Procrustes $[ 1 8 ] ~ ( { \hat { \mathbf { V } } } = \mathbf { U } \mathbf { W } ^ { \mathsf { T } }$ , given the Singular Value Decomposition $\mathbf { Y } \mathbf { X } ^ { \mathsf { T } } = \mathbf { U } \boldsymbol { \Sigma } \mathbf { W } ^ { \mathsf { T } } )$ . These updates are done iteratively till convergence.

However, the technique above is computationally expensive due to an SVD (of $\mathbf { Y X } ^ { \top } )$ in each iteration (thus costing $O ( n ^ { 2 } \operatorname* { m a x } \{ n , p \} )$ per iteration). Note that non-orthogonal dictionary learning techniques [19], [20] have similar computational complexity while having better representational performance. Consequently, [4], [5] provide fast orthogonal transform learning techniques by assuming some additional structure on the orthogonal dictionary, leading to improved computational performance. The work in [4] assumes the orthogonal dictionary to be a product of $O ( \log n )$ Householder reflections and provides an iterative algorithm to estimate the dictionary. The work in [5] generalizes this approach using Givens rotations.

This work builds on the prior work by investigating the sample complexity (number of columns required) and robustness under statistical assumptions on the sparse representation X. We completely analyze the case when $m = 1$ using concentration inequalities and propose algorithms for the general case that improves on prior work under these statistical assumptions in the sample limited case (i.e., $p < n )$ . Due to the statistical assumptions, our approach also has the advantage of being non-iterative and non-spectral, as opposed to prior work.

# IV. RECOVERY FOR THE STRUCTURED ORTHOGONAL

# (HOUSEHOLDER) DICTIONARY

In this section, we give sample complexity results for the case when $\mathbf { Y } = \mathbf { H } \mathbf { X }$ , the matrix H is Householder $( \mathbf { H } = \mathbf { I } - \mathbf { 2 u u } ^ { \mathsf { T } } )$ and the sparse representation X follows the statistical model described in (1). We see that using the first-order statistical properties of the induced distribution on Y is sufficient to estimate H to high accuracy.

Theorem 1. (Householder Recovery) Consider $\mathbf { Y } = \mathbf { H } \mathbf { X }$ and the model described in (1) for X. Suppose

1) the unit vector u defining the Householder matrix H satisfies $c = \textstyle \sum u _ { i } = \Omega ( n ^ { \alpha } ) f o r \alpha > 1 / 4 ,$   
2) $u _ { i } c \ i s$ bounded for all i, and   
3) the number of samples/columns $p > C \log n / \theta ^ { 2 } \mu ^ { 2 } ,$ with constant C large enough;

then u can be recovered (up to sign) with the following recovery guarantee: $\mathbb { P } \left( \| \mathbf { u } - \hat { \mathbf { u } } \| _ { \infty } > t \right) \leq O \left( \left( 1 / n \right) ^ { t ^ { 2 } O \left( 1 \right) } \right)$ .

We get $\mathbb { P } \left( \| \mathbf { u } - \hat { \mathbf { u } } \| _ { \infty } > t \right) \to 0 ,$ . The estimate uˆ is computed via $A l -$ gorithm 1, and the computational complexity involved in calculating ˆu is $O ( n p )$ .

Algorithm 1 Finding H, X for $\mathbf { \overline { { Y } } } = \mathbf { H } \mathbf { X }$   
Input: $\mathbf { \overline { { Y } } } , \theta , \mu$   
Output: H, X   
1: Set $c^{2}=\left(n-\sum_{i=1}^{n}\sum_{j=1}^{p}Y_{ij}/p\theta\mu\right)/2$ , with $c\geq0$ 2: set $u_{i}=\left(1-\sum_{j=1}^{p}Y_{ij}/p\theta\mu\right)/2c$ , for i=1:n
3: Set $H = I - 2u u^{T}$ 4: Set $X' = H^{T}Y$ 5: Set $X = HT_{\zeta}(X')$

Note that once H is obtained, we estimate the sparse representation X by computing $\mathbf { H } \mathbf { Y } ( = \mathbf { H } ^ { \mathsf { T } } \mathbf { Y } )$ and thresholding entry-wise. Note that this operation can be performed in $O ( n p )$ arithmetic operations (since $\mathbf { H } \mathbf { Y } = \mathbf { Y } - 2 \mathbf { u } \mathbf { u } ^ { \mathsf { T } } \mathbf { Y }$ can be computed in $O ( n p ) )$ ). Thus both H and X are estimated in $O ( n p )$ . We provide numerical simulations for this algorithm in Section VI.

Remark: $H T _ { \zeta } ( \cdot )$ is the hard threshold operator (i.e., $H T _ { \zeta } ( x ) =$ $x \cdot \mathbb { I } ( | x | \geq \zeta ) )$ . The value ζ is chosen heuristically. Furthermore, we are implicitly using the following: if u is a solution, then −u is also a solution, as both produce the same Householder matrix.

# V. RECOVERY FOR THE GENERAL ORTHOGONAL DICTIONARY

We move to the case when the orthogonal dictionary $\mathbf { V } = \mathbf { H _ { 1 } H _ { 2 } } \dots \mathbf { H _ { m } }$ is a product of $m > 1$ Householder reflectors. Prior work in [4] proposes an alternating iterative technique to estimate the Householder matrices. Following up from Section IV, we propose a sequential update strategy to estimate $\mathbf { H _ { i } }$ . Unlike the earlier work, this update is non-iterative (involves a fixed m number of steps). Before discussing the algorithm, we note the following fundamental limitation to recovering the Householder matrices $\mathbf { H _ { i } }$ in this setup.

Lemma 1. For any Householder matrix H, there exist Householder matrices $\mathbf { H _ { 1 } } , \mathbf { H _ { 2 } } , \mathbf { H _ { 3 } }$ different from H such that $\mathbf { H H _ { 1 } } = \mathbf { H _ { 2 } H _ { 3 } } .$ .

Since the Householder matrices cannot be uniquely identified, we consider the error metric in this setup as $| | \mathbf { V } - \hat { \mathbf { V } } | | _ { F }$ (as opposed to an error in the individual ui).

We first note the following modification of Algorithm 1. Suppose Q is an orthogonal matrix, and let $\mathbf { Y } = \mathbf { H Q X }$ where X satisfies the statistical model from (1). If Q is known, then H can be recovered from Y in a very similar fashion to the approach described in the proof of Theorem 1. We skip the steps due to space constraints, and refer to Section VII-C for additional details. The modified algorithm is summarized in Algorithm 2.

Algorithm 2 Finding H, X for Y = HQX   
Input: ${ \overline { { \mathbf { Y } , \mathbf { Q } , \theta , \mu } } }$   
Output: H, X   
1: set $s_{i} = \sum_{j=1}^{n} Q_{ij}$ , for i = 1 : n
2: set $k_{i} = \left( s_{i} - \sum_{j=1}^{p} Y_{ij} / p \theta \mu \right) / 2$ for i = 1 : n
3: set $u_{i} = k_{i} / \left( \sum_{m=1}^{n} k_{m} s_{m} \right)^{1/2}$ for i = 1 : n
4: Set H = I - 2uu $^{\top}$ 5: Set X' = (HQ) $^{\top}$ Y
6: Set X = HT $_{\zeta}$ (X')

Following this, we use a sequential strategy to update the Hi. We consider a general case initialization to elaborate upon our algorithm. In step i, we note that

$$
Y = \underbrace {\mathbf {H} _ {1} \mathbf {H} _ {2} \ldots \mathbf {H} _ {\mathrm{i-1}}} _ {W} \mathbf {H} _ {\mathrm{i}} \underbrace {\mathbf {H} _ {\mathrm{i+1}} \mathbf {H} _ {\mathrm{i+2}} \ldots \mathbf {H} _ {\mathrm{m}}} _ {Q} \mathbf {X},
$$

so that $\mathbf { W } ^ { \mathsf { T } } \mathbf { Y } = \mathbf { H } _ { \mathbf { i } } \mathbf { Q } \mathbf { X }$ . Applying Algorithm 2 to $\mathbf { W } ^ { \mathsf { T } } \mathbf { Y }$ gives us Hi. We do this for $i = 1 , 2 , \dots , m$ to obtain estimates for $\mathbf { H _ { i } } .$ , and then set the estimate Vˆ as the product of the obtained estimates for Hi. This is outlined in Algorithm 3. Computationally, the algorithm requires updating the data matrix Y for each $i ~ = ~ 1 ~ : ~ m$ , and each update requires multiplying with a Householder matrix, costing $O ( n p )$ for each i. Note that we only need the sum of the entries of Q for each row; the sequential precomputation of these vectors Q1 for all $i = 1 : m$ costs $O ( m n )$ . Finally, Algorithm 2 costs $O ( n p )$ and must be repeated at each update step. Thus, the overall computational complexity of Algorithm 3 is O(nmp).

The spectral technique [4] involves computing $\mathbf { X Y } ^ { \mathsf { T } }$ (which costs $O ( n ^ { 2 } p ) )$ followed by finding the eigenvectors of an n × n matrix for each step (i.e. m times). This process is repeated for a certain number of iterations till convergence. In contrast, the proposed approach is non-iterative and costs O(npm) (i.e., the complexity scales linearly with n). This reduction is achieved due to statistical assumptions on the sparse representation X.

Algorithm 3 Finding V for $\mathbf { \overline { { Y } } } = \mathbf { V } \mathbf { X }$   
Input: Y, m, θ, μ
Output: V
1: Initialize $H_{i}$ 2: Set $Z[m+1] = 1$ 3: For i = m to 1, set $Z[i] = H_{i}Z[i+1]$ 4: for For i = 1 to m do
5: Find $H_{i}$ using Algorithm 2, using Z[i] as Q1
6: Update Y as $H_{i}^{T}Y$ 7: end for
8: Set $V = \prod_{i} H_{i}$

# VI. SIMULATIONS

In this section, we show some numerical results on the approximation error and robustness of the proposed algorithm.

# A. Data Generation

The ground truth Householder matrices $\mathbf { H } _ { \mathbf { i } } = \mathbf { I } - \mathbf { 2 } \mathbf { u } _ { \mathbf { i } } \mathbf { u } _ { \mathbf { i } } ^ { \mathsf { T } }$ are generated by selecting each entry of ui randomly (i.i.d Gaussian/Uniform) and then normalizing the obtained vector. The support of X is generated by using i.i.d Bernoulli entries for each entry $X _ { i j }$ with parameter θ. The non-zero entries are filled using i.i.d Uniform samples in the range [1, 2]. With Hi and X, the data matrix Y is computed as $\mathbf { Y } = \mathbf { H _ { 1 } H _ { 2 } } \dots \mathbf { H _ { m } } \mathbf { X } + \mathbf { N }$ where N is a noise matrix with i.i.d zero mean Gaussian entries. In most experiments, the number of rows n is set to $n = 1 0 0 0$ , and the number of columns p varies from 2 to 18.

# B. Results for the case m = 1

In Fig 3, we plot the $l _ { \infty }$ error in u (on the y-axis) with varying number of columns in Y; for different sparsity regimes. As we can see, the error decreases with an increase in the number of columns, as expected. The error when θ is lower is slightly higher than the corresponding error for larger values of θ, which is consistent with Theorem 1. Figure 4 shows the average per entry error in (Frobenius norm sense) X, with varying number of columns. Finally, in Figure 5, we plot the estimation error under different SNR2 regimes. As is evident from the results, the algorithm is relatively robust to noise.

2The Signal-to-Noise Ratio (SNR) in decibels (dB) is given by: $\mathrm { S N R } _ { \mathrm { d B } } =$ $1 0 \log _ { 1 0 } \left( \mathrm { \bar { / } { { P } _ { \mathrm { { s i g n a l } } } } } / { { P } _ { \mathrm { { n o i s e } } } } \right)$ where $P _ { \mathrm { s i g n a l } }$ and Pnoise represent the signal and noise power, respectively.

# C. Results for the general case

Next, we provide results for orthogonal matrix recovery in a sample-limited setup. The orthogonal matrix V is generated as a product of m Householder matrices $\mathbf { H _ { 1 } } , \mathbf { H _ { 2 } } , \cdot \cdot \cdot \mathbf { H _ { m } } .$ (where the Householder vector ui, $i \in \{ 1 , 2 , \cdots m \}$ is generated by choosing each entry randomly and then normalizing the vector) i.e., $\mathbf { V } = \mathbf { H _ { 1 } } \mathbf { H _ { 2 } } \cdot \cdot \cdot \mathbf { H _ { m } }$ .

Although many initializations work well, we initialize $\mathbf { H _ { 1 } } , \mathbf { H _ { 2 } } , \ldots$ to I for our experiments. In Fig 1, we plot the Frobenius norm error in V for our method, the algorithm proposed in [4], and the solution to the Orthogonal Procrustes problem (for the Orthogonal Procrustes solution, we assume we know X exactly: so this is the best case performance of Procrustes) for a varying number of Householders, with $p = 2 0 .$ . In Fig 2, we repeat the above experiment for varying columns for a fixed $m \ = \ 1 0$ . In this case, we used $n \ = \ 2 0 0 .$ , as opposed to the other experiments, since we increased p to a relatively larger value of 200. As the plots show, our method performs significantly better than the best-case Procrustes solution (i.e., with known X) when we have very few samples and does slightly better than the method proposed in [4] but with much better computational complexity.

![](images/2972e1487032dbdb0a83c8cb838a61fb1594cfb76bb0eaef6e1330240bcbcc33.jpg)

<details>
<summary>line</summary>

| Number of shareholder matrices (m) | Error in V (our method) for p=20 | Error in V (Orthogonal Procrustes) for p=20 | Error in V H₀DLA for p = 20 |
| ---------------------------------- | --------------------------------- | ------------------------------------------ | ---------------------------- |
| 2                                  | 4.0                               | 43.0                                       | 7.5                          |
| 3                                  | 4.5                               | 43.0                                       | 7.5                          |
| 4                                  | 4.5                               | 43.0                                       | 7.5                          |
| 5                                  | 5.0                               | 43.0                                       | 8.0                          |
| 6                                  | 5.0                               | 43.0                                       | 8.0                          |
| 7                                  | 5.5                               | 43.0                                       | 8.5                          |
| 8                                  | 6.0                               | 43.0                                       | 8.5                          |
| 9                                  | 6.5                               | 43.0                                       | 9.0                          |
| 10                                 | 7.0                               | 43.0                                       | 9.0                          |
</details>

Fig. 1: Frobenius norm error in the estimated orthogonal dictionary for a varying number of Householder matrices (n=1000; p=20)

![](images/539c4d8c4088527ef9c3631c492f8b206d785359160d62d04f8918c4df3a3993.jpg)

<details>
<summary>line</summary>

| Number of columns in X (p) | Error in V (our method) for m=10 | Error in V (Orthogonal Procrustes) for m=10 | Error in V HmDLA for m=10 |
| -------------------------- | ---------------------------------- | ------------------------------------------- | -------------------------- |
| 25                         | 6.0                                | 19.5                                        | 8.5                        |
| 50                         | 6.0                                | 17.5                                        | 8.5                        |
| 100                        | 6.0                                | 14.5                                        | 8.5                        |
| 150                        | 6.0                                | 10.0                                        | 8.5                        |
| 200                        | 6.0                                | 0.0                                         | 8.5                        |
</details>

Fig. 2: Frobenius norm error in the estimated orthogonal dictionary for a varying number of columns $( n = 1 0 0 0 ; m = 1 0 )$

# VII. PROOFS AND OTHER DETAILS

# A. Proof of Theorem 1

Given that $\begin{array} { r } { c = \sum u _ { i } = \Omega ( n ^ { 1 / 4 } ) } \end{array}$ we assume without loss of generality that $c > 0 .$ First, we note that due to the Householder structure of the matrix H, we have $\begin{array} { r c l } { { Y _ { i j } } } & { { = } } & { { \sum _ { k = 1 } ^ { n } H _ { i k } X _ { k j } } } \end{array} =$ $\begin{array} { r } { \sum _ { k = 1 } ^ { n } ( \delta _ { i k } - 2 u _ { i } u _ { k } ) X _ { k j } } \end{array}$ . Taking expectations,

$$
\mathbb {E} [ Y _ {i j} ] = \theta \mu \sum_ {k = 1} ^ {n} (\delta_ {i k} - 2 u _ {i} u _ {k}) = \theta \mu (1 - 2 u _ {i} c), \tag {2}
$$

![](images/cdb22409f020f1ad64bcd1159f0eb9f2b9006958d8a4a02db8e2eed172beff06.jpg)

<details>
<summary>line</summary>

| p  | θ = 0.01 | θ = 0.05 | θ = 0.1 | θ = 0.4 | θ = 0.7 | θ = 1 |
|----|----------|----------|---------|---------|---------|-------|
| 2  | 0.45     | 0.18     | 0.13    | 0.05    | 0.02    | 0.02  |
| 4  | 0.28     | 0.11     | 0.09    | 0.04    | 0.02    | 0.02  |
| 6  | 0.23     | 0.11     | 0.07    | 0.03    | 0.02    | 0.02  |
| 8  | 0.20     | 0.10     | 0.06    | 0.03    | 0.02    | 0.02  |
| 10 | 0.20     | 0.11     | 0.05    | 0.03    | 0.02    | 0.02  |
| 12 | 0.17     | 0.08     | 0.05    | 0.03    | 0.02    | 0.02  |
| 14 | 0.17     | 0.07     | 0.05    | 0.03    | 0.02    | 0.02  |
| 16 | 0.16     | 0.06     | 0.04    | 0.03    | 0.02    | 0.02  |
| 18 | 0.15     | 0.08     | 0.04    | 0.03    | 0.02    | 0.02  |
</details>

Fig. 3: $l _ { \infty }$ norm error in u for a varying number of columns and different sparsity levels (θ) for Y = HX (n = 1000)

![](images/dabcbc1a28202b9af87a434752c24ce5ee21261f33b58f49de4a4626a5c747fd.jpg)

<details>
<summary>line</summary>

| p  | θ = 0.01 | θ = 0.05 | θ = 0.1 | θ = 0.4 | θ = 0.7 | θ = 1   |
|----|----------|----------|---------|---------|---------|---------|
| 2  | 0.007    | 0.016    | 0.023   | 0.043   | 0.055   | 0.062   |
| 4  | 0.005    | 0.011    | 0.012   | 0.022   | 0.028   | 0.033   |
| 6  | 0.004    | 0.008    | 0.009   | 0.015   | 0.021   | 0.023   |
| 8  | 0.003    | 0.006    | 0.007   | 0.012   | 0.016   | 0.017   |
| 10 | 0.003    | 0.005    | 0.006   | 0.011   | 0.014   | 0.015   |
| 12 | 0.003    | 0.004    | 0.005   | 0.010   | 0.012   | 0.013   |
| 14 | 0.003    | 0.004    | 0.005   | 0.010   | 0.011   | 0.012   |
| 16 | 0.003    | 0.004    | 0.005   | 0.010   | 0.011   | 0.011   |
| 18 | 0.003    | 0.004    | 0.005   | 0.010   | 0.011   | 0.011   |
</details>

Fig. 4: Frobenius norm error per entry in X for a varying number of columns and varying sparsity levels (θ) for $\mathbf { Y } = \mathbf { H } \mathbf { X } \ ( n = 1 0 0 0 )$

so that $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { p } \mathbb { E } ( Y _ { i j } ) = \theta \mu ( n - 2 c ^ { 2 } ) p . } \end{array}$ . We estimate this expectation empirically, estimating $c = \sum u _ { i }$ first, and consequently $u _ { i } .$ . Following (2), the estimate for c is

$$
\hat {c} ^ {2} = \left(1 - \frac {\sum_ {i = 1} ^ {n} \sum_ {j = 1} ^ {p} Y _ {i j}}{n p \theta \mu}\right) \frac {n}{2}, \quad \hat {c} \geq 0.
$$

We note that $\begin{array} { r } { \sum _ { i } \sum _ { j } Y _ { i j } \ = \ \sum _ { i } \sum _ { j } \sum _ { k } H _ { i k } X _ { k j } } \end{array}$ is a weighted sum of independent random variables, so we use the Hoeffding’s inequality [21] to bound the error in the estimate of c:

$$
\mathbb {P} \left(\left| c - \hat {c} \right| > t\right) \leq \mathbb {P} \left(\left| c ^ {2} - \hat {c} ^ {2} \right| > t\right) \leq 2 \exp \left(- 8 t ^ {2} \theta^ {2} \mu^ {2} p\right), \tag {3}
$$

So $\hat { c } = c { + } O ( 1 )$ w.h $\cdot \mathrm { p } ^ { 3 }$ . We skip the algebra due to space constraints. However, this follows from a direct application of Hoeffding’s inequality. Following the estimate of c from above, using (2) we

3We say that an event $A _ { n }$ holds with high probability (w.h.p) if $P ( A _ { n } ) \geq$ $1 - 1 / n ^ { \dot { \alpha } }$ for some $\alpha > 0$

![](images/3912237769bc54a68335e3e2f57154a3a49262a3b6a395f9685337ca3dc23841.jpg)

<details>
<summary>line</summary>

| p  | SNR 36.01403 | SNR 26.08859 | SNR 16.13284 | SNR 6.93217 |
|----|--------------|--------------|--------------|-------------|
| 2  | 0.068        | 0.072        | 0.078        | 0.090       |
| 4  | 0.050        | 0.055        | 0.058        | 0.082       |
| 6  | 0.038        | 0.040        | 0.048        | 0.052       |
| 8  | 0.025        | 0.032        | 0.035        | 0.048       |
| 10 | 0.022        | 0.028        | 0.030        | 0.038       |
| 12 | 0.020        | 0.025        | 0.027        | 0.032       |
</details>

Fig. 5: $l _ { \infty }$ norm error in u for a varying number of columns and different sparsity levels (θ) for $\mathbf { Y } = \mathbf { H } \mathbf { X }$ under noisy conditions (SNR is in dB) $( n = 1 0 0 0 )$

estimate the entries of the ground truth vector u generating the Householder matrix as

$$
\hat {u} _ {i} = \frac {1}{2 \hat {c}} \left(1 - \sum_ {j = 1} ^ {p} Y _ {i j} / p \theta \mu\right) := \frac {\hat {y}}{2 \hat {c}}.
$$

Note that the error in the estimate $\hat { u } _ { i }$ has two components - due to the error in cˆ and the deviation in the empirical mean above. Reapplying the Hoeffding’s inequality to $\begin{array} { r } { \sum _ { j } Y _ { i j } = \sum _ { j } \sum _ { k } H _ { i k } X _ { k j } } \end{array}$ , we obtain the error in the empirical mean as

$$
\mathbb {P} \left[ | \hat {y} / 2 c - u _ {i} | \geq t \right] \leq 2 \exp \left(- 8 t ^ {2} c ^ {2} \theta^ {2} \mu^ {2} p\right). \tag {4}
$$

Now we have

$$
\begin{array}{l} \mathbb {P} \left[ | \hat {u} _ {i} - u _ {i} | \geq t \right] = \mathbb {P} \left[ | \hat {y} / 2 \hat {c} - u _ {i} | \geq t \right] \\ \leq \mathbb {P} \left[ | \hat {y} / 2 \hat {c} - \hat {y} / 2 c | \geq t / 2 \right] + \mathbb {P} \left[ | \hat {y} / 2 c - u _ {i} | \geq t / 2 \right], \\ \leq \mathbb {P} \left[ | \hat {y} (c - \hat {c}) / c \hat {c} | \geq t \right] + 2 \exp \left(- 2 t ^ {2} c ^ {2} \theta^ {2} \mu^ {2} p\right). \\ \end{array}
$$

Note that from $( 4 ) , \hat { y } = 2 u _ { i } c + O ( 1 )$ w.h.p, and from (3) and the hypothesis of the theorem, ccˆ is $\Omega \dot { ( n ^ { 1 / 2 } ) }$ . Therefore $\hat { y } / c \hat { c }$ is O(1) w.h.p, and a repeat application of (3) gives us

$$
\begin{array}{l} \mathbb {P} \left[ | \hat {u} _ {i} - u _ {i} | \geq t \right] \\ \leq 4 \exp (- 8 \theta^ {2} \mu^ {2} t ^ {2} O (1) p) (1 - 1 / n ^ {\alpha}) \text {   for   some   } \alpha > 0 \\ \leq 4 \left(\frac {1}{n}\right) ^ {8 C t ^ {2}} \left(1 - \frac {1}{n ^ {\alpha}}\right) \\ \end{array}
$$

By union bound over all the $u _ { i } \mathrm { ^ s } ,$ , the theorem follows.

We show an equivalent method to recover u. Define

$$
k _ {i} = \left(1 - \sum_ {j = 1} ^ {p} Y _ {i j} / p \theta \mu\right) / 2 = u _ {i} \left(\sum_ {z = 1} ^ {n} u _ {z}\right) = u _ {i} c, \tag {5}
$$

From this, we have $\begin{array} { c c l } { k _ { m } u _ { i } } & { = } & { k _ { i } u _ { m } } \end{array}$ . Thus, we have $\begin{array} { r l } { k _ { i } ^ { 2 } } & { { } = } \end{array}$ $\begin{array} { r } { u _ { i } ^ { 2 } \left( \sum _ { m = 1 } ^ { n } k _ { m } \right) } \end{array}$ . Using the unit norm property of u, we obtain an estimate of $u _ { i }$ using $k _ { 1 } , k _ { 2 } , \ldots$ . The computational complexity involved in calculating these estimates (using either approach) is $O ( n p )$ . For other details, please refer [22].

# B. Proof of Lemma 1

Let the Householder vectors corresponding to the matrices $\mathbf { H } , \mathbf { H _ { 1 } } , \mathbf { H _ { 2 } } , \mathbf { H _ { 3 } }$ be u, u1, u2, u3, respectively. Given u, we choose u1 such that u1 is orthogonal to u. Then we have

$$
\mathbf {H} \mathbf {H} _ {\mathbf {1}} = (\mathbf {I} - 2 \mathbf {u u} ^ {\top}) (\mathbf {I} - 2 \mathbf {u _ {1}} \mathbf {u _ {1}} ^ {\top}) = \mathbf {I} - 2 \mathbf {u u} ^ {\top} - 2 \mathbf {u _ {1}} \mathbf {u _ {1}} ^ {\top}.
$$

Now we set $\mathbf { u _ { 2 } } = ( \mathbf { u } + \mathbf { u _ { 1 } } ) / \sqrt { 2 }$ and $\mathbf { u _ { 3 } } = ( \mathbf { u } - \mathbf { u _ { 1 } } ) / \sqrt { 2 } ,$ , so that $\mathbf { u } _ { 1 }$ and u2 are unit norm and orthogonal. We can now verify with basic algebra that

$$
\mathbf {H} _ {2} \mathbf {H} _ {3} = \mathbf {I} - 2 \mathbf {u} _ {2} \mathbf {u} _ {2} ^ {\top} - 2 \mathbf {u} _ {3} \mathbf {u} _ {3} ^ {\top} = \mathbf {I} - 2 \mathbf {u u} ^ {\top} - 2 \mathbf {u} _ {1} \mathbf {u} _ {1} ^ {\top} = \mathbf {H H} _ {1}.
$$

# C. Details on Algorithm 2

The key difference in Algorithm 2 is how the entries of the matrix Q enter the computation of the estimates. We set $\begin{array} { r } { s _ { i } ~ = ~ \sum _ { k = 1 } ^ { n } Q _ { i k } } \end{array}$ as the entries of $\mathbf { s } = \mathbf { Q 1 }$ , so that (2) modifies to $\begin{array} { r l } { \mathbb { E } [ Y _ { i j } ] { \bf \Xi } } & { { } = \mathbf { \Xi } \theta \mu ( s _ { i } \mathrm { ~ - ~ } 2 u _ { i } \mathbf { u } ^ { \mathsf { T } } \mathbf { s } ) } \end{array}$ . Accordingly, (5) modifies to $\begin{array} { l l l l } { k _ { i } } & { = } & { \left( s _ { i } - \sum _ { j = 1 } ^ { p } Y _ { i j } / p \theta \mu \right) / 2 } & { = } & { u _ { i } \mathbf { u } ^ { \mathsf { T } } \mathbf { s } } \end{array}$ . Thus, we have $\begin{array} { r } { k _ { i } ^ { 2 } ~ = ~ u _ { i } ^ { 2 } \big ( et { } { ' } \sum _ { m = 1 } ^ { n } s _ { m } k _ { m } \big ) } \end{array}$ , from which we obtain an estimate of ui using $k _ { 1 } , k _ { 2 } , \ldots$ This approach reduces to that of Algorithm 1 when $\mathbf Q = \mathbf I$ (since $s _ { i } = 1$ for all i). Note that $u _ { i }$ follows the sign of $k _ { i }$ and $\textstyle \sum _ { m = 1 } ^ { n } k _ { m } s _ { m } \neq 0$ is a necessary condition for the above to work. The proof of the theoretical guarantee for this approach is very similar in structure to that described in Theorem 1.

# REFERENCES

[1] Michael Elad, Sparse and redundant representations: from theory to applications in signal and image processing, Springer Science & Business Media, 2010.   
[2] Xiaowen Dong, Dorina Thanou, Michael Rabbat, and Pascal Frossard, “Learning graphs from data: A signal representation perspective,” IEEE Signal Processing Magazine, vol. 36, no. 3, pp. 44–63, 2019.   
[3] Dorina Thanou, David I Shuman, and Pascal Frossard, “Learning parametric dictionaries for signals on graphs,” IEEE Transactions on Signal Processing, vol. 62, no. 15, pp. 3849–3862, 2014.   
[4] Cristian Rusu, Nuria Gonzalez-Prelcic, and Robert W Heath, “Fast ´ orthonormal sparsifying transforms based on householder reflectors,” IEEE Transactions on Signal Processing, vol. 64, no. 24, pp. 6589– 6599, 2016.   
[5] Cristian Rusu and John Thompson, “Learning fast sparsifying transforms,” IEEE Transactions on Signal Processing, vol. 65, no. 16, pp. 4367–4378, 2017.   
[6] Bruno A Olshausen and David J Field, “Sparse coding with an overcomplete basis set: A strategy employed by v1?,” Vision research, vol. 37, no. 23, pp. 3311–3325, 1997.   
[7] Kjersti Engan, Sven Ole Aase, and J Hakon Husoy, “Method of optimal directions for frame design,” in 1999 IEEE International Conference on Acoustics, Speech, and Signal Processing. Proceedings. ICASSP99 (Cat. No. 99CH36258). IEEE, 1999, vol. 5, pp. 2443–2446.   
[8] Michal Aharon, Michael Elad, and Alfred Bruckstein, “K-svd: An algorithm for designing overcomplete dictionaries for sparse representation,” IEEE Transactions on signal processing, vol. 54, no. 11, pp. 4311–4322, 2006.   
[9] Julien Mairal, Francis Bach, Jean Ponce, and Guillermo Sapiro, “Online dictionary learning for sparse coding,” in Proceedings of the 26th annual international conference on machine learning, 2009, pp. 689–696.   
[10] Ju Sun, Qing Qu, and John Wright, “Complete dictionary recovery over the sphere,” in 2015 International Conference on Sampling Theory and Applications (SampTA). IEEE, 2015, pp. 407–410.   
[11] Sylvain Lesage, Remi Gribonval, Fr´ ed´ eric Bimbot, and Laurent Be-´ naroya, “Learning unions of orthonormal bases with thresholded singular value decomposition,” in Proceedings.(ICASSP’05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005. IEEE, 2005, vol. 5, pp. v–293.   
[12] Chenglong Bao, Jian-Feng Cai, and Hui Ji, “Fast sparsity-based orthogonal dictionary learning for image restoration,” in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 3384–3391.   
[13] Ke-Lin Du, MNS Swamy, Zhang-Quan Wang, and Wai Ho Mow, “Matrix factorization techniques in machine learning, signal processing, and statistics,” Mathematics, vol. 11, no. 12, pp. 2674, 2023.   
[14] Geyu Liang, Gavin Zhang, Salar Fattahi, and Richard Y Zhang, “Simple alternating minimization provably solves complete dictionary learning,” arXiv preprint arXiv:2210.12816, 2022.   
[15] Gene H Golub and Charles F Van Loan, Matrix computations, JHU press, 2013.   
[16] Frank Uhlig, “Constructive ways for generating (generalized) real orthogonal matrices as products of (generalized) symmetries,” Linear Algebra and its Applications, vol. 332, pp. 459–467, 2001.   
[17] Yuexiang Zhai, Zitong Yang, Zhenyu Liao, John Wright, and Yi Ma, “Complete dictionary learning via l4-norm maximization over the orthogonal group,” Journal of Machine Learning Research, vol. 21, no. 165, pp. 1–68, 2020.   
[18] Jos MF Ten Berge, “Orthogonal procrustes rotation for two or more matrices,” Psychometrika, vol. 42, pp. 267–276, 1977.   
[19] Joel A Tropp, “Greed is good: Algorithmic results for sparse approximation,” IEEE Transactions on Information theory, vol. 50, no. 10, pp. 2231–2242, 2004.   
[20] Joel A Tropp, “Just relax: Convex programming methods for subset selection and sparse approximation,” ICES report, vol. 404, 2004.   
[21] Wassily Hoeffding, “Probability inequalities for sums of bounded random variables,” The collected works of Wassily Hoeffding, pp. 409– 426, 1994.   
[22] Anirudh Dash and Aditya Siripuram, “Fast structured orthogonal dictionary learning using householder reflections,” arXiv preprint arXiv:2409.09138, 2024.