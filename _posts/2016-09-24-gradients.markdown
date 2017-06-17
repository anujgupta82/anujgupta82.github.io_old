---
layout: post
comments: true
title:  "Computing Gradients"
excerpt: "Gradients behind training Neural Nets"
date:   2016-09-24 15:40:00
mathjax: false
---

### Introduction

Training neural nets ia all about computing gradients. In this post we will systmatically see the maths that goes into computing these gradients. The complexity of calculations depends on 3 things: 

1. Depth of the network
2. Number of training examples (1 or more)
3. Number of input nodes (1=scalar, more=vector)

Through out this post we assume:
1. There is no bias term.
2. All activations are (sigmoid)[]
3. Number of input nodes (1=scalar, more=vector)


We will start with the simplest case and increase the complexity gradually. 

#### 1 layer network, 1 input (scalar)

Consider a simplest version of a neural net - 1 layer, 1 input node

    
