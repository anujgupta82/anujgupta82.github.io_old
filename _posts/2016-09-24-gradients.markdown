---
layout: post
comments: true
title:  "Computing Gradients"
excerpt: "Gradients behind training Neural Nets"
date:   2016-09-24 15:40:00
mathjax: true
---


### Introduction

Training neural nets ia all about computing gradients. In this post we will systmatically see the maths that goes into computing these gradients. The complexity of calculations depends on 3 things: 

1. Depth of the network
2. Number of training examples (1 or more)
3. Number of input nodes (1=scalar, more=vector)

Through out this post we assume:
1. There is no bias term.
2. All activations are (sigmoid)[https://www.quora.com/What-is-the-sigmoid-function-and-what-is-its-use-in-machine-learnings-neural-networks]
3. Number of input nodes (1=scalar, more=vector)
4. `.` is matrix multiplication, `*` is element wise product and ` $$\times$$ ` $$\times$$ is normal multiplication. 


We will start with the simplest case and increase the complexity gradually. 

#### **1 layer network, 1 input (scalar)**

Consider a simplest version of a neural net - 1 layer, 1 input node (scalar)

<div class="imgcap">
<img src="/assets/gradients/NN_1.png">
<div class="thecap">simple neural net</div>
</div>

Input is (x,y) : x, y both are scalars. In matrix form (just becuase later on every thing will be a matric), they are $$[X]_{\scriptscriptstyle 1\times 1}$$ and $$[y]_{\scriptscriptstyle 1\times 1}$$. Let W be weight matrix. In this case its $$[y]_{\scriptscriptstyle 1\times 1}$$


Predicted output ($$ \hat{y} $$) &= $$\frac{1}{1 + e^{-[X] . [W]}} \label{ref0} \tag{0}$$

Let loss ($$ L $$) = $$\frac{1}{2} (y - \hat{y})^{2} $$

Let's compute gradients, $$\nabla_{\theta} W = \frac{\partial L}{\partial W} $$

<!---
\begin{equation}
  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial W} 
  \tag{1}
  \frac{\partial L}{\partial \hat{y}} &= \frac{1}{2} \times 2 \times (y - \hat{y})^{1} \times (-1) 
  \tag{a}
\end{equation}
-->

$$
\begin{align}
\frac{\partial L}{\partial W} & = \frac{\partial L}{\partial \hat{y}} \times \frac{\partial \hat{y}}{\partial W} \label{ref1} \tag{1}\\
\frac{\partial L}{\partial \hat{y}} &= \frac{1}{2} \times 2 \times (y - \hat{y})^{1} \times (-1) \label{ref2} \tag{2}\\
\frac{\partial \hat{y}}{\partial W} &= \big{(} \frac{1}{1 + e^{-[X] . [W]}} \big{)} \times \big{(}1- \frac{1}{1 + e^{-[X] . [W]}} \big{)} \times x \\
& = \hat{y} \times (1- \hat{y}) \times x \dots && \text{using \eqref{ref0}} \label{ref3} \tag{3}\\
\end{align}
$$

Substituting \eqref{ref2} & \eqref{ref3} in \eqref{ref1}, we get 

$$
\begin{align}
\frac{\partial L}{\partial W} &= \big{(} (-1) \times (y - \hat{y}) \big{)} \times \big{(} \hat{y} \times (1- \hat{y}) \times -x \big{)}\\ \\
&= (y - \hat{y}) \times \hat{y} \times (1- \hat{y}) \times x \label{ref4} \tag{4} \\
\end{align}
$$

Let,  
$$ 
\begin{align}
\Delta l_{1} = (y - \hat{y}) \times \hat{y} \times (1- \hat{y}) \label{ref5} \tag{5} \\
\end{align}
$$

Then, eq \eqref{ref1} reduces to:	
$$ 
\begin{align}
\frac{\partial L}{\partial W} = \Delta l_{1} \times x \\
& = [X^{T}] . \Delta l_{1} \label{ref6} \tag{6} \\
\end{align}
$$











#### __1 layer network, 1 input (vector)__

#### __1 layer network, multiple inputs (each is a vector)__



#### 2 layer network, 1 input (vector)

$$A_{\scriptscriptstyle p\times k}x_{\scriptscriptstyle k\times 1}=b_{\scriptscriptstyle p\times 1}$$

#### 2 layer network, multiple inputs (vector)


<!---
Deriving Policy Gradients. I'd like to also give a sketch of where Policy Gradients come from mathematically. Policy Gradients are a special case of a more general score function gradient estimator. The general case is that when we have an expression of the form \(E_{x \sim p(x \mid \theta)} [f(x)] \) - i.e. the expectation of some scalar valued score function \(f(x)\) under some probability distribution \(p(x;\theta)\) parameterized by some \(\theta\). Hint hint, \(f(x)\) will become our reward function (or advantage function more generally) and \(p(x)\) will be our policy network, which is really a model for \(p(a \mid I)\), giving a distribution over actions for any image \(I\). Then we are interested in finding how we should shift the distribution (through its parameters \(\theta\)) to increase the scores of its samples, as judged by \(f\) (i.e. how do we change the network's parameters so that action samples get higher rewards). We have that:

$$
\begin{align}
\nabla_{\theta} E_x[f(x)] &= \nabla_{\theta} \sum_x p(x) f(x) & \text{definition of expectation} \\
& = \sum_x \nabla_{\theta} p(x) f(x) & \text{swap sum and gradient} \\
& = \sum_x p(x) \frac{\nabla_{\theta} p(x)}{p(x)} f(x) & \text{both multiply and divide by } p(x) \\
& = \sum_x p(x) \nabla_{\theta} \log p(x) f(x) & \text{use the fact that } \nabla_{\theta} \log(z) = \frac{1}{z} \nabla_{\theta} z \\
& = E_x[f(x) \nabla_{\theta} \log p(x) ] & \text{definition of expectation}
\end{align}
$$
-->






    
