---
layout: post
comments: true
title:  "Gradients for Neural Nets"
excerpt: "Computing Gradients that go into training Neural Nets"
date:   2016-09-24 15:40:00
mathjax: true
---


### Introduction

Training neural nets ia all about [computing gradients](http://deeplearning.stanford.edu/wiki/index.php/Deriving_gradients_using_the_backpropagation_idea). In this post we will systmatically see the maths that goes into computing these gradients. The complexity of calculations depends on 3 things: 

1. Depth of the network
2. Number of training examples (1 or more)
3. Number of input nodes (1=scalar, more=vector)

Through out this post we assume:
1. There is no bias term.
2. All activations are [sigmoid](https://www.quora.com/What-is-the-sigmoid-function-and-what-is-its-use-in-machine-learnings-neural-networks)
3. Number of input nodes (1=scalar, more=vector)
4. `.` is matrix multiplication, `*` is element wise product and ` $$\times$$ ` `X` is normal multiplication. 


We will start with the simplest case and increase the complexity gradually. 

#### **1 layer network, 1 input (scalar)**

Consider a simplest version of a neural net - 1 layer, 1 input node (scalar)

<div class="imgcap">
<img src="/assets/gradients/NN_1_1.jpeg" height="300" width="350">
<div class="thecap">simple neural net</div>
</div>

Input is (x,y) : x, y both are scalars. (Later on every thing will be a matrix, so just to be using same notaion. We will abuse the notation to express scalars as matrix of dimension 1 \\(\times\\) 1). Thus, in matrix form x,y are $$[X]_{\scriptscriptstyle 1\times 1}$$ and $$[y]_{\scriptscriptstyle 1\times 1}$$. Let W be weight matrix. In this case its $$[W]_{\scriptscriptstyle 1\times 1}$$

<!--
**Deriving Policy Gradients**. I'd like to also give a sketch of where Policy Gradients come from mathematically. Policy Gradients are a special case of a more general *score function gradient estimator*. The general case is that when we have an expression of the form \\(E_{x \sim p(x \mid \theta)} [f(x)] \\) - i.e. the expectation of some scalar valued score function \\(f(x)\\) under some probability distribution \\(p(x;\theta)\\) parameterized by some \\(\theta\\). Hint hint, \\(f(x)\\) will become our reward function (or advantage function more generally) and \\(p(x)\\) will be our policy network, which is really a model for \\(p(a \mid I)\\), giving a distribution over actions for any image \\(I\\). Then we are interested in finding how we should shift the distribution (through its parameters \\(\theta\\)) to increase the scores of its samples, as judged by \\(f\\) (i.e. how do we change the network's parameters so that action samples get higher rewards). We have that:
-->


Let \\( \hat{y} \\) be the predicted output. Then, $$ \hat{y} = \sigma (Wx) = \frac{1}{1 + e^{-[X] . [W]}} \label{ref0} \tag{0}$$

Let loss be squared error loss. For ease of maths we take \\( \frac{1}{2} \\) of it. $$ L  = \frac{1}{2} (y - \hat{y})^{2} $$

Let's compute gradients, $$\nabla_{W} L = \frac{\partial L}{\partial W} $$

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
\frac{\partial \hat{y}}{\partial W} &= \big{(} \frac{1}{1 + e^{-[X] . [W]}} \big{)} \times \big{(}1- \frac{1}{1 + e^{-[X] . [W]}} \big{)} * X \\
& = \sigma (Wx) \times (1- \sigma (Wx)) * -X \dots && \text{using \eqref{ref0}} \label{ref3} \tag{3}\\
& = \hat{y} \times (1- \hat{y}) * -X \dots && \text{using \eqref{ref0}} \label{ref33}\\
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

\begin{align}
\Delta l_{1} = (y - \hat{y}) \times \hat{y} \times (1- \hat{y}) \label{ref5} \tag{5} \\
\end{align}


Then, eq \eqref{ref1} reduces to:	
$$ 
\begin{align}
\frac{\partial L}{\partial W} &= \Delta l_{1} \times x \\
& = \Delta l_{1} * X \\
& = [X^{T}] . \Delta l_{1} \label{ref6} \tag{6} \\
\end{align}
$$



#### __1 layer network, 1 input (vector)__

Our neural net still has 1 layer, but input now is a vector. 

<div class="imgcap">
<img src="/assets/gradients/NN_2_2.jpeg" height="300" width="350">
<div class="thecap">Neural net with 1 layer, but input is vector</div>
</div>

Input is \\((\vec{X},y)\\) : \\(\vec{X}\\) is a vector, while y is a scalars. 

\\(X = [x^1 ~~x^2 ~~x^3]\\)		&nbsp; &nbsp; &nbsp; \\(x^i = i^{th}\\) component of \\(\vec{X}\\).
	


Thus, in matrix form x,y are $$[X]_{\scriptscriptstyle 1\times 3}$$ and $$[y]_{\scriptscriptstyle 1\times 1}$$. W, weight matrix is $$[W]_{\scriptscriptstyle 3 \times 1}$$

$$
\begin{equation}
     W=\begin{bmatrix}
         w_{1} \\
         w_{2} \\
         w_{3} \\
         \end{bmatrix}
\end{equation}
$$

\\( \hat{y} \\) is predicted output. In matrix format, \\([\hat{y}]_{\scriptscriptstyle 1\times 1}\\)
\\( \hat{y} = \sigma ([X] . [W]) = \frac{1}{1 + e^{-[X] . [W]}} \label{ref11} \tag{11} \\)



Like before, we will use half of squared error loss. $$ L  = \frac{1}{2} (y - \hat{y})^{2} $$

Let's compute gradients. 

$$
\begin{equation}
\nabla_{W} L = \frac{\partial L}{\partial W} \\
\nabla_{W} L = \begin{bmatrix}
     \frac{\partial L}{\partial w_{1}} \\
     \frac{\partial L}{\partial w_{2}} \\
     \frac{\partial L}{\partial w_{3}} \\
     \end{bmatrix}
\end{equation}
$$


So, lets compute \\( \frac{\partial L}{\partial w_{1}} \\)

$$
\begin{align}
\frac{\partial L}{\partial w_1} &= \frac{\partial L}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial w_1} \label{ref12} \tag{12} \\
\frac{\partial L}{\partial \hat{y}} &= \frac{1}{2} \times 2 \times (y - \hat{y})^{1} \times (-1) \label{ref13} \tag{13} \\
\frac{\partial \hat{y}}{\partial w_1} &= \big{(} \frac{1}{1 + e^{-[X] . [W]}} \big{)} \times \big{(}1- \frac{1}{1 + e^{-[X] . [W]}} \big{)} * x_1 \label{ref14} \tag{14}\\
& = \sigma ([X] . [W]) \times (1- \sigma ([X] . [W])) * -x_1 \dots & \text{using \eqref{ref11}} & \label{ref15} \tag{15}\\
& = \hat{y} \times (1- \hat{y}) * -x_1 \dots & \text{using \eqref{ref11}} & \label{ref16} \tag{16}\\
\end{align}
$$

Substituting \eqref{ref13} & \eqref{ref16} in \eqref{ref12}, we get 

$$
\begin{align}
\frac{\partial L}{\partial w_1} &= \big{(} (-1) \times (y - \hat{y}) \big{)} \times \big{(} \hat{y} \times (1- \hat{y}) \times -x_1 \big{)} \\
& = (y - \hat{y}) \times \hat{y} \times (1- \hat{y}) \times x_1
\end{align}
$$

Thus, in general:
$$
\begin{align}
\frac{\partial L}{\partial w_1} &= (y - \hat{y}) \times \hat{y} \times (1- \hat{y}) \times x_1
\end{align}
$$


#### __1 layer network, multiple inputs (each is a vector)__



#### __2 layer network, 1 input (vector)__

$$A_{\scriptscriptstyle p\times k}x_{\scriptscriptstyle k\times 1}=b_{\scriptscriptstyle p\times 1}$$

#### __2 layer network, multiple inputs (vector)__


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






    
