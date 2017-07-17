---
layout: post
comments: true
title:  "Gradients - summary"
excerpt: "Take home on Computing Gradients that go into training Neural Nets"
date:   2016-09-14 01:23:00
mathjax: true
---


### Generalization

In this post, based on our conclusions in last post, we will try and generalise a strategy to compute gradients for arbit networks, as shown in figure below:

<div class="imgcap">
<img src="/assets/gradients/NN_generic.jpeg" height="300" width="350">
<div class="thecap">simple neural net</div>
</div>

Imagine we have a network with 1 input layer \\(L_0\\), 1 output layer \\(L_3\\) and 2 hidden layers \\(L_1\\), \\(L_2\\) respectively. Further, let \\(l_i\\) be output of layer \\(L_i\\). Also, by design, \\(l_1 = X\\) [input] and \\(l_3 = \hat{y}\\) [output]





<!--
{% include button.html button_name="Next" button_class="primary" %}
-->
