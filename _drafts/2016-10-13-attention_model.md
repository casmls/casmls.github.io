---
layout: post
title:  "Attention Model in Caption Generation and Image Generation"
categories: general
author: Liping Liu and Patrick Stinson 
excerpt_separator: <!--more-->
comments: true
---

We read two papers related to attention model on last Thursday: “DRAW: A recurrent neural network for image generation” and “Show, attend and tell: Neural image caption generation with visual attention”. The first paper generates images from the same distribution as the input images while the second paper generates captions for images. By exploiting the attention model, both model focus on a region of the image at every step and output the data in steps. To model the the sequential generative process, both model have used LSTM as one layer of the network. 

<!--more-->

**1. Summary of the DRAW paper **

Two ideas of the paper. The generation of an image should be refined in multiple steps. In the multiple steps of refinement, the model should only focuses its attention on as small region of the image. 

Model representation of the draw model

Roughly speaking, the model is a combination of LSTM and auto-encoder.

![planar flow](/img/draw_model_representation.png)

At each time step, data passes vertically from the bottom to the top. 

Input: $$x$$ and $$x - \sigma(c_{t-1})$$, (the latter one is the residual of the current fitting)
Read: Attention model is put here: an $$N\times N$$ Gaussian filter is positioned over the image. Five parameters are used to control the 
position, area, effective resolution, and intensity of the attention. The parameters are controled by the decode output at the last step. 
Encoding: The data read in by the attention layer is encoded 
Code: the distribution of the code $$z_t$$ should be similar to a latent prior $$P(Z_t)$$
Decoding: The code z is decoded though the generative process
Write: the decode output set the attention model, and the data is written through filters of the attention model. The usage of the filters here is in the reverse order of that in the read step. 


Over time steps, data are related in the following ways. 

First, the input include the residual of last step's fitting result $$x - \sigma(c_{t-1})$$.
Second, the filters of the attention model is controlled by last step's decoding output
Third, last step's decode output is also feed into the current step's encoding module. 
Fourth, the sum of the write-out data are the final reconstructed result.

The training objective is minimize the reconstructive loss and the KL divergence of the latent prior to the encoding distribution 
$$L = \langle L^x + L^z \rangle_{z\sim Q(Z | h^{enc})}$$. 


Some results: 
 
![Fig. 1 in the paper](/img/draw_result.png)

(Also check animated results from Eric Jang's blog [link](http://blog.evjang.com/2016/06/understanding-and-implementing.html))


**2. Discussion:**

The paper shows interesting results in several tasks. In the image generation task, the generation process does exhibit some behavior progressive refinement, though still different with the way we humans draw digits. The model improves the predictive log-likelihood. In the task of identifying numbers in cluttered images, the model often correctly focus its attention to the numbers.  


# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention 

**1. Paper summary:**

The argument of the paper is that the attention to an image changes as the caption generation for the image proceeds in steps.

The model structure is also LSTM. The input of the model are feature vectors ${a_i}$ extracted from image regions $\{i\}$ by CNN. The fitting target are embeded vectors of words in training captions. The attention model, being put in between the input and the LSTM units, selects the feature vectors of only some regions of the image. 

The attention model $f_{att}$ calculates a set of probabilities for current locations with a multilayer perceptron conditioned on previous hidden state.  

$$e_{ti} = f_{att}(h_{t-1}, a_i)$$
$$\alpha_{ti} = \frac{exp(e_{ti})}{\sum_k exp(e_{tk})}$$


The context vector z_{t} (input to the LSTM) is calculated with either hard attention or soft attention.

Hard attention

The attention location at every step is a categrical r.v. and thus is a hidden variable. 

$s_t ~ Categorical(\alpha_t)$  (s_t is an indicator one-hot variable)
$z_t = \sum_{i} s_{ti} a_i $

The model maximize the variational lower bound of the log-likelihood, which is optimized by stochastic gradient method. Moving average and other techniques are used to control the variance of gradients. 

Soft attention

$z_t = \sum_{i} \alpha_{ti} a_i $

The input is a weighted sum of feature vectors at different locations. The model can be optimized by back-propagation. 

Extra constraint: every location should be focused on at least once during the generation process. 

The paper shows significant improvement of BLEU score over previous methods on several caption generation tasks. In example images, a lot of attention locations can be well explained. Figure 5 of the paper shows several good examples. 

![Fig. 5 in the paper](/img/attention_location.png)




## References
[1] Gregor, Karol, et al. “DRAW: A recurrent neural network for image generation.” arXiv preprint arXiv:1502.04623 (2015). [link](http://arxiv.org/pdf/1502.04623.pdf)

[2] Xu, Kelvin, et al. “Show, attend and tell: Neural image caption generation with visual attention.” arXiv preprint arXiv:1502.03044 2.3 (2015). [link](http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf)

[3] Eslami, S. M., et al. “Attend, Infer, Repeat: Fast Scene Understanding with Generative Models.” arXiv preprint arXiv:1603.08575 (2016). [link](http://arxiv.org/pdf/1603.08575.pdf)

[4] Mnih, Volodymyr, Nicolas Heess, and Alex Graves. “Recurrent models of visual attention.” Advances in Neural Information Processing Systems. 2014. [link](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)


