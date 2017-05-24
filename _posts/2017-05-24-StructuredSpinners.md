---
layout: post
title: Structured Spinners for Fast Machine Learning
categories: general
author: Tim Wang
excerpt_separator: <!--more-->
comments: true
---

It's been a few weeks since Jalaj introduced these two papers in
reading group, but better late than never! The first paper is
“Structured adaptive and random spinners for fast machine learning
computations” by Mariusz et al.[1] and the second is “Bayesian
Optimization in a Billion Dimensions via Random Embeddings” by Ziyu et
al. [2].  This note mainly focuses on the first paper, which
introduced a structured matrices called Structure Spinners for fast
computation and provided theoretical guarantees for the effectiveness
of this structured transform. We also briefly summarised the second
paper, which introduced a random embedding method that makes Bayesian
optimisation techniques applicable to high dimensional data that has
slow effective dimensions.


<!--more-->

_The figures below are taken from the aforementioned papers._

Background
-----------

A plethora machine learning methods use matrix to transform input data
before passing to a possibly highly nonlinear function. However, the
computation of projection takes $$\Theta(mn \vert X \vert)$$ time,
where $$m \times n$$ is the size of the matrix and $$\vert X\vert $$
is the number of samples. It is also costly to store the projection
matrix frequently. A nature idea is to use some structured matrix
instead of any $$m \times n$$ matrices. In this case, projections can
be reduced to $$O(n \log(m))$$ time and matrices can be easily stored.

Lots of closely related structured matrices were previously studied in
the context of Johnson-Lindenstrauss Transform (JLT), which searches
for an embedding that approximately preserves the Euclidean
distance. The JLT Lemma is as follows:

Given $$0 \lt \epsilon \lt 1$$, a set $$X$$ of $$N$$ points in $$R^{n}$$, let $$k \ge \frac{24}{3\epsilon^2-2\epsilon^3} \log(N)$$, then there is a linear map $$f:R^{n}\to R^{k}$$ such that for any $$x_i, x_j$$, we have 

$$(1-\epsilon)\| x_i - x_j\| \le \| f(x_i) - f(x_j)\| \le (1+\epsilon)\| x_i - x_j\|$$

Another similar problem we covered is the Fast Food Transform method
by Quoc et al.[3], which focus on the problem of fast kernel
approximation. The matrices used in fast food transform take the form
$$SHG\Pi HB$$, where $$S, H, G, \Pi, B$$ serve as scaling matrix,
Hadamard matrix, Gaussian scaling matrix, permutation and binary
scaling matrix, respectively.


One important structure used in this paper is the $$HD_3HD_2HD_1$$
structured matrix used by Wang et al [4]. The $$D_i$$s are either
random diagonal $$\pm 1$$ matrix in random setting or adaptive
diagonal matrices in adaptive setting. $$H$$ is a Hadamard matrix: a
square matrix whose entries are $$\pm 1$$ and rows are mutually
orthogonal. It can be constructed as

$$H_{2^k} = \left(\begin{matrix}
H_{2^{k-1}} \ \ \    H_{2^{k-1}}\\
H_{2^{k-1}} \  -H_{2^{k-1}}
\end{matrix}\right)$$

Problem
-----------
Let $$A_{G}$$ be a machine learning algorithm applied to dataset $$X \in R^n$$. $$A_{G}$$ consists of functions $$f_1, \cdots, f_s$$, where each $$f_i$$, associated with a matrix $$G_i \in G$$ is a function of random vector 

$$q_{f_i} = ((G_ix^1)^\mathsf{T}, \ldots, (G_ix^{d_i})^\mathsf{T})^\mathsf{T} \in R^{d_im}$$

Here, $$x^1, \cdots, x^{d_i}$$ stands for some fixed basis for a linear space $$L_i$$.

In the random setting, $$G_i$$ is a Gaussian with independent entries
taken from $$N(0,1)$$. In the adaptive setting, $$G_i$$ is a learned
matrix. Now we replace $$G_i$$ with structured matrix and denote the
new function as $$f_i'$$. We want to show that for structured matrices
in our choice, $$f_i'$$s resemble $$f_i$$s distribution-wise by
showing that the probability of the vectors belonging to a convex set
is similar in both cases. Namely, with high probability, for any
convex set $$S$$, $$ \vert P[f_i(q_i) \in S] - P[f_i'(q_{f_i'}) \in S]
\vert $$ is small.


Structured Spinners 
-----------

Here we introduce the structured spinners used in the paper:

A structure spinner is matrices as a product of three main structured blocks 
$$
G_{struct} = B_3B_2B_1
$$
with conditions:
1. $$B_1$$ and $$B_2B_1$$ are $$(\delta(n), p(n))$$-balanced isometries.
2. ($$B_2, B_3$$) is $$(K, \Lambda_{F}, \Lambda_{2})$$-random.




There are some terms used in the above definition, we list them below:

**$$(\delta(n), p(n))$$-balanced matrices**

A random matrix $$M \in R^{n \times m}$$ is $$(\delta(n), p(n))$$-balanced if for every unit vector $$x$$, we have $$P[\left \| Mx\right \|_{\infty} > \frac{\delta(n)}{n}] \le p(n)$$

**$$(\Delta_{F}, \Delta_{2})$$-smooth sets**

A deterministic set of matrices $$W^1, \cdots, W^n \in R^{k \times n}$$ is $$(\Delta_{F}, \Delta_{2})$$-smooth if :
	
* each column of the matrix has the same 2-norm
* for $$i \ne j$$ and any $$l$$, $$(W_l^i)^T\cdot W_l^j = 0$$
* max$$_{i, j} \left \| (W^j)^T W^i \right \|_{F} \le \Lambda_{F}$$
* max$$_{i, j} \left \| (W^j)^T W^i \right \|_{2} \le \Lambda_{2}$$

**$$(K, \Lambda_{F}, \Lambda_{2})$$-randomness**

A pair of matrices $$(Y, Z) \in R^{n \times n} \times R^{n \times n}$$ is $$K, \Lambda_{F}, \Lambda_{2})$$-random if there exists $$r \in R^{k}$$, and a set of linear isometries $$\phi = \phi_{1}, \cdots, \phi_{n}$$, where $$\phi_i: R^n\to R^k$$, such that:


* r is either a random Rademacher vector (iid from $$\pm 1$$) or a Gaussian with identity covariance. 
* for every $$x \in R^n$$ the $$j^{th}$$ element of $$Zx$$ is $$r^T \cdot \phi_j(x)$$
* there exists a set of iid sub-Gaussian random variables $${\rho_1, \cdots, \rho_n}$$ with mean 0, same second moments and sub-Gaussian norm at most $$K$$ and a $$(\Lambda_F, \Lambda_2)$$-smooth set of matrices $$\{W^i\}_{i = 1, \cdots, n}$$ such that for every $$x$$, we have $$\phi_i(Yx) = W^i(\rho_1x_1, \cdots, \rho_nx_n)^T$$. 


There are three blocks in the structured matrix, which play different roles.
* $$B_1$$ makes vectors “balanced”, so each dimension have the similar $$L_2$$ norm. 
* $$B_2$$ makes vectors “close to orthogonal” in random setting, or “independent” in adaptive setting.  
* $$B_3$$ defines the capacity of the entire structured transform. 

A pictorial explanation can been seen here:
![ABC]({{site.base_url}}/img/2017-05-13/structured_spinners_1.png)

Theory and sketch of proof
-----------

**random setting**

With $$A_G, G, f_i$$ as described above, if we replace unstructured Gaussian matrices $$G$$ with stacking structured spinners, then with probability $$p_{succ}$$ with random choice of $$B_1, B_2$$, we have 

$$
P[f_i(q_i) \in S] - P[f_i'(q_{f_i'}) \in S]  \le b\eta
$$

with respect to random choice of $$B_3$$. $$S$$ is a union of at most $$b$$ pairwise disjoint convex sets. When $$\eta = \frac{\delta^3(n)}{n^{2/5}}$$ and $$\delta(n), p(n), K, \Lambda_f, \Lambda_2$$ are described as above, and $$\epsilon = o_{md}(1)$$, we have $$p_{succ}$$ is at least 

$$
1 - 2p(n)d-2 \binom{md}{2} \exp\{-\Omega(\min(\frac{\epsilon^2n^2}{K^4\Lambda_F^2\delta^4(n)}, \frac{\epsilon n}{K^2\Lambda_2\delta^2(n)}))\}
$$



In the special case of optimal LSH setting where 


$$
G_{struct} = \sqrt{n}HD_3HD_2HD_1
$$


we have 

$$
\delta(n) = \log(n), p(n) = 2ne^{\frac{\log^2(n)}{8}},
K = 1, \Lambda_F = O(\sqrt{n}), \Lambda_2 = O(1).
$$



**adaptive setting**

Consider a matrix $$M \in R^{m \times n}$$ as the weights in a unstructured neural network, then with probability at least $$p_{succ}$$ with respect to random choices of $$B_1$$ and $$B_2$$, there exists a vector $$r$$ defining $$B_3$$ such that the structured spinner equals to $$M$$ on $$L$$. Here, $$p_{succ}$$ has the same form as in the random setting while $$\epsilon = \frac{1}{md}$$.


For the proof of the random case, we use the Berry-Esseen type Central Limit Theorem by showing that $$q_{f_i} \sim N(0, I)$$ and $$q_{f_i'} \sim N(0, I+E)$$ with high probability. $$E$$ is a small error matrix. The first part is straightforward. As $$G $$ is iid from $$N(0, 1)$$, we have:

$$\mathbb{E}((G_ix^1)^T(G_ix^2)) = (x^1)^TE(G_i^TG_i)x_2 = (x^1)^TIx_2 = 0$$

$$\mathbb{E}((G_k^ix^1)^T(G_j^ix^1)) = (x^1)^T (G_k^i)^TG_j^i x^1 = 0$$

The other part is nontrivial and all the details can be found in the Appendix in [1]. 


<!--
Experiment
-----------

We found the theoretical contributions of this paper to be more
interesting than the experiments. Instead of going through experiment
with detail, we just paste the runtime and test error for the MLP. We
found it unclear whether the result with large $$h$$ comes from
effectiveness of structured matrix or overparameterization of
unstructured one.
![ABC]({{site.base_url}}/img/2017-05-13/structured_spinners_2.png)
-->

# Bayesian Optimisation via Random Embeddings


As we did not go deeply with this paper, I will summarise the key idea and main algorithm proposed by this paper. 

We focus on the optimization problem 

$$
x^* = \arg\max\limits_{x \in X} f(x)
$$

where $$f: X \to R$$ be a function on compact set $$X \in R^D$$.

Bayesian optimisation uses a prior distribution that captures our
beliefs about the behaviour of $$f$$ and an acquisition function that
quantifies the expected value (in terms of, say, expected improvement
over the current optimum) of learning the value of $$f(x)$$ for each
$$x \in X$$. However, Bayesian optimisation is challenging in high
dimensions, as the set of possible inputs grows exponentially. This
paper proposed a Random Embedding Bayesian Optimisation (REMBO)
algorithm that first project the inputs into a low dimensional space and
then solves the optimisation in this low dimensional space.

The algorithm relies on following theorem: 

Given a function $$f: R^D \to R$$ with effective dimensionality
$$d_e$$ and a random matrix $$A \in R^{D \times d}$$ with entries iid
from $$N(0,1)$$ and $$d \gt d_e$$. Then with probability $$1$$, for any
$$x$$, there exists a $$y \in R^d$$ such that $$f(x) = f(Ay)$$.

$$d_e$$ is the effective dimensionality of $$f$$ if there exists a linear subspace $$T$$ such that $$f(x) = f(x^{\perp})$$ for any $$x \in R^D$$.


This means that, when the effective dimensionality of inputs is small,
we can always project inputs into a small space, where Bayesian
optimisation is applicable, and the optimal result we achieve is also
optimal for the original problem. The full algorithm is shown below:
![ABC]({{site.base_url}}/img/2017-05-13/REMBO_1.png)



### References

[1] Bojarski, Mariusz, et al. "Structured adaptive and random spinners for fast machine learning computations." arXiv preprint arXiv:1610.06209 (2016). [link](https://arxiv.org/pdf/1610.06209.pdf)

[2] Wang, Ziyu, et al. "Bayesian optimization in a billion dimensions via random embeddings." Journal of Artificial Intelligence Research 55 (2016): 361-387. 
[link](http://www.jair.org/media/4806/live-4806-9131-jair.pdf)

[3] Le, Quoc, et al. "Fastfood-approximating kernel expansions in loglinear time." Proceedings of the international conference on machine learning. (2013).[link](https://www.robots.ox.ac.uk/~vgg/rg/papers/le__icml2013__fastfood.pdf)

[4] Wang, Ziyu, et al. "Bayesian optimization in a billion dimensions via random embeddings." Journal of Artificial Intelligence Research 55 (2016): 361-387.[link](http://www.jair.org/media/4806/live-4806-9131-jair.pdf)

