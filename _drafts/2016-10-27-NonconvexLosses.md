---
layout: post
title: "Nonconvex Loss Functions for Classifiers and Deep Networks"
categories: general
author: Keyon Vafa
excerpt_separator: <!--more-->
comments: true
---

Two papers this week focused on appealing and unintuitive qualities of using stochastic gradients to
optimize non-convex loss functions. 
In The Landscape of Empirical Risk for Non-convex Losses by Mei et. al, 2016 [1], the authors
show that while empirical risk for squared loss is non-convex for linear classifiers, there are
numerous desirable qualities once we reach a certain sample size, namely exponentially fast
convergence to a local minimum that is also the global minimum. In Deep Learning without Poor
Local Minima by Kawaguchi (2016) [2], by reducing deep linear networks to deep nonlinear networks,
the author shows that, among other things, every unique minimum is a global minimum and every non-minimum 
critical point is a saddle point.

<!--more-->

# The Landscape of Empirical Risk for Non-convex Losses

While non-convex optimization has historically been associated with NP-hardness, there are a plethora of 
non-convex functions that can be optimized more easily by taking advantage of some special structural
properties. This paper focuses on the specific case of noisy linear classification, and shows that under
certain assumptions, the empirical risk, which is non-convex, has a unique local minimum, which gradient 
descent approaches exponentially fast.

### Setup

* Data is $$(x_i, y_i) \in \mathcal{R}^d \times \{0, 1\}$$ i.i.d.
* $$P(y_i = 1|x_i) = \sigma(\langle w_0, x\rangle)$$ for some function $$\sigma(\cdot)$$, which is known. 
* Loss function is _empirical risk_ $$\hat R_n(w) = \frac{1}{n}\sum_{i=1}^n (y_i - \sigma(\langle w, x_i\rangle))^2$$.

We note that the empirical risk is non-convex. However, the authors argue that non-convex losses can be 
beneficial for this task because they are unbounded, they have a smaler number of support vectors, and 
models such as neural networks have been shown to "work better."

The _true risk_, $$R(w) = E((y-\sigma(\langle w, x\rangle))^2)$$ is convex in terms of $$w$$, and the 
authors show that the empirical risk function ends up sharing many of these nice properties. The following graph shows
three possible scenarios for the empirical risk. On the left is a function with many local minima, and in the center 
is a loss with local minima that are close to the risk global minimum. Niether of these are ideal, and we will see 
for sufficient sample data, the function behaves like the graph on the right with a local minimum that is the risk 
global minimum, verifying the effectiveness of stochastic gradient descent. 

[INCLUDE IMAGE]

We make the following assumptions: 

1 $\sigma$ is three times differentiable, where $$\sigma'(z) > 0$$ and the first three derivatives are bounded.
2 The feature vector X is **sub-Gaussian**, i.e. for all $$\lambda \in \mathbb{R}^D$$, $$E(e^{\langle \lambda, X \rangle})
\leq e^{\frac{\tau^2 \|\lambda\|_2^2}{2}}$$.
3 X spans all directions in $$R^d$$ so $$E(XX^T) \succeq c\tau^2 I_{d \times d}$$ for some $$0 < c < 1$$.

### Results

The main result is **Theorem 1** which has three parts. If $$n \geq Cd \log d$$,

1. $$\hat R_n(w)$$ has a unique local optimizer, which is also global. This impleis no other critical points exist.
2. Gradient descent converges exponentially fast.
3. The global optimum of the empirical risk $$\hat w_n$$ converges to the true minimum $$w_0$$: $$\|hat w_n - w_0\|_2
\leq C \sqrt{(d\log n)/n}$$. 

Looking at $$n \geq Cd \log d$$, $$n$ is roughly the same scale as $$d$$, and becuase $$n$$ cannot be less than $$d$$, 
these results are ideal. 

The authors verify these results by running experiments with simulated data and various conditions for $$n$$. These results 
verify that once $$$n$$ crosses some critical threshold, the convergence results change dramatically. 

### Proof Ideas

We discuss a few theorems the authors develop to construct their proof (which is too technical to include in this writeup). The 
meat of their argument lies in **Theorem 2**, which conerns the true risk $$R(w)$$. Summarizing these results,

* The true risk $$R(w)$$ has a unique minimizer $$w_0$$.
* The Hessian has bounds, where in a small ball around the the global optimum, we have a strongly convex function 
(i.e. we converge very quickly):

$$
\inf_{w_0, \epsilon_0} \lambda_{\min} (\nabla^2 R(w)) \geq \kappa_0
$$

* The gradient has bounds, such that when we are outside of a ball, the graidents are non-zero:

$$
\inf_{w \in B(0, B_0) \set minus B(W_0, \epsilon_0)} \| \nabla R(w) \|_2 \geq L_0
$$
Namely, this implies our gradients point toward the optimum.

The remaining theorems bound the probabilities that the gradient and Hessian of empirical risk differ from 
the true risk by constant. Combining these probabilities with Theorem 2, the authors conclude that the desirable qualities 
are recovered with a high probability. 

### Discussion
We were curious how the proofs generalize to functions outside of squared loss, since in real life squared-loss 
isn't the most frequently used function for classification.

Overall, we thought this was a very well-written paper that did a good job of stating the high-level at the beginning, 
while exploring the technical details throughout. 



Use the excerpt_separator to mark the end of the short intro
(that's all that show's up on the homepage)

# This is a first level heading

## This is a second level heading

## This is a third level heading

> This is a
> blockquote
> with
>
> two paragraphs

This is a list:
* A
* B
* C

If your list items span multiple paragraphs, intend the items with three spaces.
Here the list is enumerated.

1.   This is the first sentence.

     This is the second sentence.

2.   And so on...

3.   And so forth...

This is **bold text**.
This is _italic text_.
This is ~~strikeout text~~.

This is an inline equation: $$a^2 + b^2 = c^2$$. You have to use two
dollar signs to open and close, not one like in Latex.
To have a centered equation, write it as a pararaph that starts and
ends with the two dollar signs:

$$
p(\theta \, | \, y) \propto p(\theta) \, 
p(y \, | \, \theta).
$$

I don't think you can do align blocks yet.

This is `inline  code`. 
Code blocks are intended paragraphs with four spaces:

```python
F = lambda n: ((1+np.sqrt(5))**n - (1-np.sqrt(5))**n) / (2**n * np.sqrt(5))
```
This is a figure. Note that `site.base_url` refers to the homepage.
In this case, `abc.png` is located in the `img` folder under root.

![ABC]({{site.base_url}}/img/abc.png)

### References
I've just been copying and pastying references as follows: 

[1] Meeds, Edward, Robert Leenders, and Max Welling. "Hamiltonian ABC." _arXiv preprint arXiv:1503.01916_ (2015). [link](http://arxiv.org/pdf/1503.01916)
...

### Footnotes
Here's my trick for footnotes. You can write HTML inside markdown, so I just create a
div with the id footnotes and then add a link[<sup>1</sup>](#footnotes)

<div id="footnotes"></div>
1. like this.
2. ...

