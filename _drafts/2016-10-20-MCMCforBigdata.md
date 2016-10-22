---
layout: post
title: "MCMC for Big Data"
categories: general
author: Ghazal Fazelnia
excerpt_separator: <!--more-->
comments: true
---

This week we read a paper on Scalable Exact algorithm for Bayesian inference for Big Data.
It introduces a Monte Carlo algorithm based on a Markov process whose quasi-stationary 
distribution coincides with distribution of interest. Authors show theoretical guarantee 
for recovering the correct limiting target distribution using proposed algorithm. They argue that this 
methodology is practical for big data in which they use a subsampling technique with sub-linear
iterative cost as a function of data size.

<!--more-->

The problems with current methods like Metropolis Hastings especially for big data are:
* They need $$O(n)$$ calculation at each step for Reject/Accept decisions
* They need $$O(n)$$ new variables when augmenting data

Hence, modifications are needed when working with big data.
Some of the previous solutions include:

* Divide-and-conquer: The weakness of this method is that the recombination of the separately conducted inference is inexact.
* Subsampling: Although cheaper, it still requires subset of data of size $$O(n)$$.
* Approximate subsampling: Usually the approximation is not exact, and it still needs $$O(n)$$ subsample sizes. It could also be 
hard to quantify the bias caused by approximations and time-discretization of the dynamics.

### Model and Theoretical Results
The authors model the problem as a diffusion $$X:R \to R^d $$ over an infinite-time horizon.

$$
dX_t = \beta(X_t) dt + \Lambda ^{1/2} dW_t
$$

For simplicity of notations, I will assume that $$\Lambda$$ is identity matrix (keeping in mind that this model works for general 
$$\Lambda$$). They specifically use the 'Langevin' diffusion with $$\nu (x) $$ as its invariant distribution as follow:

$$
\beta(X_t) =1/2 \nabla log \nu (X_t)
$$

Transition densities of this model are:

$$
p_T (x,y) \propto w_T (x,y) \nu^{1/2}(y) E_{w^{x,y}}[exp(-\int_{0}^{T}\Phi(X_s)ds)]
$$

where $$w_T(x,y)$$ is a Brownian motion (BM) with start x and end y. Moreover,

$$
\Phi(u) = \parallel \beta(u) \parallel^2 + div \beta(u)/2 
$$

This transition density could be interpreted that we accept with probability of the expected value result.
Since this transition probability is not computable due to $$\nu$$, they introduce a killed/stopped BM (KBM). The transition
density of this KBM is:

$$
k_T (x,y) \propto w_T (x,y) E_{w^{x,y}}[exp(-\int_{0}^{T}\Phi(X_s)ds)]
$$

In the first theorem it is shown that if we choose $$\nu$$ to be $$\pi^2$$ where $$\pi$$ is the probability of interest, then 
$$\lim_{T\to\infty} k_T(x,.) \to \pi(.)$$. This means that by simulating from the KBM, we can approximate $$\pi$$. Therefor, 
we need to have an unbiased estimate of $$E_{w^{x,y}}[exp(-\int_{0}^{T}\Phi(X_s)ds)]$$.
The authors introduce Path-space Rejection Samplers (PRS) which is a class of rejection samplers operating on diffusion path-space over a 
finite time horizon. In this method, appropriately weighted finite dimensional subsets of sample paths are drawn from some target
measure (in this case $$K_T^x$$), by means of simulating from a tractable equivalent measure with respect to the target has
bounded Randon-Nikodym derivatives. This strategy is used when a point-wise evaluation of a target distribution is not possible,
however, the target distribution is bounded by a positive coefficient of a proposal distribution. In that case, we can approximate
the target distribution using the proposal distribution. 

The Monte Carlo algorithm that they focus on in this paper is an importance sampling variant of the KBM , in which Brownian motion 
with path-space information is simulated and weighted continuously in time, with evaluations of the trajectory occurring 
at exponentially distributed times. They call this approach Importance Sampling Killed Brownian motion (IS-KBM) algorithm.
The advantage of this approach over KBM is that we can recover for any $$ t \in [0, \infty)$$ a weighted trajectory by halting 
the algorithm.

By working on the math and using multiple reparametrizations, they show that:

$$
exp(-\int_{0}^{T}(\Phi(X_s)-L)ds) = E_{\kappa}[E_{\eta_{1:\kappa}}[\prod_{j=1}^{\kappa} (U-\Phi(X_{\eta_j}))/(U-L)]]
$$

where $$U$$ and $$L$$ are upper and lower bounds for $$\Phi$$, respectively.
$$\kappa$$ has a Poisson distribution with parameter $$T(U-L)$$, and $$\eta$$ is uniformly selected from interval $$[0,T]$$.

It could be interpreted that we need to simulate the time until the process hits one of the bounds for the bridge BM.
Then, by simulating a Poisson whose rate is proportional to that time interval, 
we will get the number of data points that need to be examined in that interval. Last, by picking
uniformly points in that interval and evaluating $$\Phi$$ at them, we will get the unbiased estimate that we are after. 

Note that we evaluate the exponential term only at $$\kappa$$ data points instead of all. 
The following figure shows an example for layers simulated for a Brownian motion sample path (left), and Brownian motion sample path
simulated as finite collection of event times (right).

![Figure1]({{site.base_url}}/img/fig1.png)

The following figure shows the evaluated $$\Phi$$ function at those points (left) as well as un-normalized importance weight process
of sample path, comprising of exponential growth between event times and discrete times at event times.

![Figure2]({{site.base_url}}/img/fig2.png)

Although the IS-KBM framework is conceptually appealing, in order to normalize we require a number of trajectories, noting that the
variance of the importance sampling weights of these trajectories will increase with time. This means that one sample could have weight 
1 while others have weight 0. They address this issue using 
Sequential Monte Carlo (SMC) method in which importance sampling and resampling techniques are combined 
in order to approximate a sequence of distributions of interest. Hence, they resample at several times (the number of which would be 
a user chosen threshold). They call this approach 'Quasi-Stationary Monte Carlo'.

One unsolved problem here is evaluation of $$\Phi(.)$$ which causes O(n) bottleneck within this estimator. They suggest to use 
$$\widetilde{\Phi(.)_A}$$, $$\widetilde{U_X}$$ and $$\widetilde{L_X}$$ for an auxiliary random variable A (i.e., uniform policy). 
One issue with this is that initialization of x needs to be within $$O(n^{-1/2})$$ of the true model. 

### Discussion
At the end of the reading session, we discussed few points regarding this paper:

* Theoretical methods and analysis offered in this paper are very interesting. 
* Altough there is a thorough state-of-the-art review, there is no comparison of this method to other methods such as 
Bouncy Particle Sampler (BPS). 
* The results in experimental sections are all on low-dimensional data sets. We know that SMC has issues with high-dimensional data, 
and its complexity grows exponentially with dimensionality of data. It would have been better if the authors had considered some 
high-dimensional examples.
* Additionally, the initializations of the experiments look too close to the true model, which might not be easy in general.
