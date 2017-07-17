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

Imagine we have a (Feed forward) network with 1 input layer \\(L_0\\), 1 output layer \\(L_3\\) and 2 hidden layers \\(L_1\\), \\(L_2\\) respectively. Further, let \\(l_i\\) be output of layer \\(L_i\\). Also, by design, \\(l_1 = X\\) [input] and \\(l_3 = \hat{y}\\) [output]. Let \\(W_{ij}\\) be weights between layers \\(L_i\\) and \\(L_j\\). We have 3 weight matrices - \\(W_{01}\\), \\(W_{12}\\) and \\(W_{23}\\). 


| Tables   |      Are      |  Cool |
|----------|:-------------:|------:|
| col 1 is |  left-aligned | $1600 |
| col 2 is |    centered   |   $12 |
| col 3 is | right-aligned |    $1 |


Name | Lunch order | Spicy      | Owes
------- | ---------------- | ---------- | ---------:
Joan  | saag paneer | medium | $11
Sally  | vindaloo        | mild       | $14
Erin   | lamb madras | HOT      | $5


<!--
{% include button.html button_name="Next" button_class="primary" %}
-->
