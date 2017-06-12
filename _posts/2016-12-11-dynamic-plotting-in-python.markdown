---
layout: post
comments: true
title:  "Dynamic Plotting in Python"
excerpt: "Plot a bunch values that keep on changing"
date:   2016-12-11 18:42:00
mathjax: false
---

In my work I am often want to visualize how the data or some aspect of model changes. This requires to have plots/graph which get Dynamically or Live updates - I have a bunch of values. One or more of these values keep on changing with time. I want to visualize these changes using plots. 

When I looking for a solution, I did not come across an elegent one. So tried piecing together a solution myself.

For simplicity, imagine we have M data points with a set of initial values. Further, these vaues get updated/changed, so we have a set pf new M values. This happens N times. This can be captured via a matrix 2D matrix A of (m x n) dimension where each row is a set of values. 

We record the initial set of values, Then these values get updated, then the update happens again, and again. Row 1 stores the initial values, row 2 stores the subsequent updated values, row 3 stores values after next update, so on and so forth. In esence, the matrix A stores the entire history of values. 

Now we will like to plot these values. Where the graph starts with values in first row, then gets updated to values in 2nd row, then gets updated tp values in 3rd row, so on and so forth. [This notebook](https://github.com/anujgupta82/Musings/blob/master/Dynamic%20or%20Live%20update%20of%20a%20Plot.ipynb) is about how to plot such graphs


[This is especially useful in machine learning, where you want to visualize how a particular property of model/data evolves with time (training) ]

Our matrix is of MxN dimensions, initialised randonly. 







<div class="imgcap">
<img src="/assets/ml_models_2/image_1.png">
<div class="thecap">gunicorn server up and running</div>
</div>

Lets hit the server with a POST request. Open another terminal window and type in:

```python
curl -i -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/hi
```




All the code in the snippets above can be found in the following [jupyter notebook](https://github.com/anujgupta82/Musings/blob/master/Dynamic%20or%20Live%20update%20of%20a%20Plot.ipynb). If you face any issue with the code, please open a "[New issue](https://github.com/anujgupta82/Musings/issues)" on Github. 
