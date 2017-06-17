---
layout: post
comments: true
title:  "Testing Latext"
excerpt: "Testing latext"
date:   2016-05-31 11:00:00
mathjax: true
---

<!-- 
<svg width="800" height="200">
	<rect width="800" height="200" style="fill:rgb(98,51,20)" />
	<rect width="20" height="50" x="20" y="100" style="fill:rgb(189,106,53)" />
	<rect width="20" height="50" x="760" y="30" style="fill:rgb(77,175,75)" />
	<rect width="10" height="10" x="400" y="60" style="fill:rgb(225,229,224)" />
</svg>
 -->




In an implementation we would enter gradient of 1.0 on the log probability of UP and run backprop to compute the gradient vector \\(\nabla_{W} \log p(y=UP \mid x) \\). This gradient would tell us how we should change every one of our million parameters to make the network slightly more likely to predict UP. 

**Policy Gradients**. 



**Update: December 9, 2016 - alternative view**. In my explanation above I use the terms such as "fill in the gradient and backprop", which I realize is a special kind of thinking if you're used to writing your own backprop code, or using Torch where the gradients are explicit and open for tinkering. However, if you're used to Theano or TensorFlow you might be a little perplexed because the code is oranized around specifying a loss function and the backprop is fully automatic and hard to tinker with. In this case, the following alternative view might be more intuitive. In vanilla supervised learning the objective is to maximize \\( \sum\_i \log p(y\_i \mid x\_i) \\) where \\(x\_i, y\_i \\) are training examples (such as images and their labels). Policy gradients is exactly the same as supervised learning with two minor differences: 1) We don't have the correct labels \\(y\_i\\) so as a "fake label" we substitute the action we happened to sample from the policy when it saw \\(x\_i\\), and 2) We modulate the loss for each example multiplicatively based on the eventual outcome, since we want to increase the log probability for actions that worked and decrease it for those that didn't. So in summary our loss now looks like \\( \sum\_i A\_i \log p(y\_i \mid x\_i) \\), where \\(y\_i\\) is the action we happened to sample and \\(A_i\\) is a number that we call an **advantage**. In the case of Pong, for example, \\(A\_i\\) could be 1.0 if we eventually won in the episode that contained \\(x\_i\\) and -1.0 if we lost. This will ensure that we maximize the log probability of actions that led to good outcome and minimize the log probability of those that didn't. So reinforcement learning is exactly like supervised learning, but on a continuously changing dataset (the episodes), scaled by the advantage, and we only want to do one (or very few) updates based on each sampled dataset.

**More general advantage functions**. I also promised a bit more discussion of the returns. So far we have judged the *goodness* of every individual action based on whether or not we win the game. In a more general RL setting we would receive some reward \\(r_t\\) at every time step. One common choice is to use a discounted reward, so the "eventual reward" in the diagram above would become \\( R\_t = \sum\_{k=0}^{\infty} \gamma^k r\_{t+k} \\), where \\(\gamma\\) is a number between 0 and 1 called a discount factor (e.g. 0.99). The expression states that the strength with which we encourage a sampled action is the weighted sum of all rewards afterwards, but later rewards are exponentially less important. In practice it can can also be important to normalize these. For example, suppose we compute \\(R_t\\) for all of the 20,000 actions in the batch of 100 Pong game rollouts above. One good idea is to "standardize" these returns (e.g. subtract mean, divide by standard deviation) before we plug them into backprop. This way we're always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator. A more in-depth exploration can be found [here](http://arxiv.org/abs/1506.02438).

**Deriving Policy Gradients**. I'd like to also give a sketch of where Policy Gradients come from mathematically. Policy Gradients are a special case of a more general *score function gradient estimator*. The general case is that when we have an expression of the form \\(E_{x \sim p(x \mid \theta)} [f(x)] \\) - i.e. the expectation of some scalar valued score function \\(f(x)\\) under some probability distribution \\(p(x;\theta)\\) parameterized by some \\(\theta\\). Hint hint, \\(f(x)\\) will become our reward function (or advantage function more generally) and \\(p(x)\\) will be our policy network, which is really a model for \\(p(a \mid I)\\), giving a distribution over actions for any image \\(I\\). Then we are interested in finding how we should shift the distribution (through its parameters \\(\theta\\)) to increase the scores of its samples, as judged by \\(f\\) (i.e. how do we change the network's parameters so that action samples get higher rewards). We have that:

$$
\begin{align}
\nabla_{\theta} E_x[f(x)] &= \nabla_{\theta} \sum_x p(x) f(x) & \text{definition of expectation} \\
& = \sum_x \nabla_{\theta} p(x) f(x) & \text{swap sum and gradient} \\
& = \sum_x p(x) \frac{\nabla_{\theta} p(x)}{p(x)} f(x) & \text{both multiply and divide by } p(x) \\
& = \sum_x p(x) \nabla_{\theta} \log p(x) f(x) & \text{use the fact that } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
& = E_x[f(x) \nabla_{\theta} \log p(x) ] & \text{definition of expectation}
\end{align}
$$

To put this in English, we have some distribution \\(p(x;\theta)\\) (I used shorthand \\(p(x)\\) to reduce clutter) that we can sample from (e.g. this could be a gaussian). For each sample we can also evaluate the score function \\(f\\) which takes the sample and gives us some scalar-valued score. This equation is telling us how we should shift the distribution (through its parameters \\(\theta\\)) if we wanted its samples to achieve higher scores, as judged by \\(f\\). In particular, it says that look: draw some samples \\(x\\), evaluate their scores \\(f(x)\\), and for each \\(x\\) also evaluate the second term \\( \nabla\_{\theta} \log p(x;\theta) \\). What is this second term? It's a vector - the gradient that's giving us the direction in the parameter space that would lead to increase of the probability assigned to an \\(x\\). In other words if we were to nudge \\(\theta\\) in the direction of \\( \nabla\_{\theta} \log p(x;\theta) \\) we would see the new probability assigned to some \\(x\\) slightly increase. If you look back at the formula, it's telling us that we should take this direction and multiply onto it the scalar-valued score \\(f(x)\\). This will make it so that samples that have a higher score will "tug" on the probability density stronger than the samples that have lower score, so if we were to do an update based on several samples from \\(p\\) the probability density would shift around in the direction of higher scores, making highly-scoring samples more likely.


### Conclusions
Finally, if no supervised data is provided by humans it can also be in some cases computed with expensive optimization techniques, e.g. by [trajectory optimization](http://people.eecs.berkeley.edu/~igor.mordatch/policy/index.html) in a known dynamics model (such as \\(F=ma\\) in a physical simulator), or in cases where one learns an approximate local dynamics model (as seen in very promising framework of [Guided Policy Search](http://arxiv.org/abs/1504.00702)).

