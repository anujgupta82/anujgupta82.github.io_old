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








### Falcon

In the last [post](https://anujgupta82.github.io/2016/11/04/deploying-ml-models-part-1/), we used flask to deploy our ML model. Today, we will be exploring a different framework to achieve the same - Falcon. Falcon is pretty simple to work with. Lets get coding :

#### Install Falcon

Load your virtual environment and execute

```python
pip install falcon
```

I used python 2.7 and falcon==1.1.0. This will get falcon installed.

#### Bare bones Example

The code consists if 2 files - [app.py](https://github.com/anujgupta82/Musings/blob/master/falcon/app.py) [contains app structure and end points] and [functionality.py](https://github.com/anujgupta82/Musings/blob/master/falcon/functionality.py) [contains the code to support the functionality].

__app.py__

```python
import functionality

api = application = falcon.API()

hello_world = functionality.hello_world()

api.add_route('/hi', hello_world)
```

All this code does is:

imports falcon package and create a basic application; access hello_world object from functionality.py and create an end point 'hi' where when data is sent via POST request, an appropriate function of hello_world object is invoked.

__functionality.py__

```python
import falcon
import json

class hello_world(object):
  def __init__(self):
    print "init"

  def on_post(self, req, resp):
    print "post: hello_world"
    result = {}

    result['msg'] = "hello world"
    resp.status = falcon.HTTP_200
    resp.body = json.dumps(result, encoding='utf-8')
```

Here codes does following: creates a class hello_world with on_post() - it takes a request and response object. Consume the request object and set the attributes of the response object. Let's get the code running.

#### Show time:
Prerequisite: You need to install gunicorn server. load you virtual environment and execute:

```python
pip install gunicorn
```

I used python 2.7 and gunicorn==19.6.0

#### Run the following steps in terminal:

```python
1. Use cd to get pwd to path where app.py is saved.
2. gunicorn app
```

This should get the gunicorn server up and running :

<div class="imgcap">
<img src="/assets/ml_models_2/image_1.png">
<div class="thecap">gunicorn server up and running</div>
</div>

Lets hit the server with a POST request. Open another terminal window and type in:

```python
curl -i -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/hi
```

On the client terminal, you should see 200 ok followed by a json containing our hello world message.

<div class="imgcap">
<img src="/assets/ml_models_2/image_2.png">
<div class="thecap">Command and response on Client side</div>
</div>

On the server terminal you should see our print statement in on_post().

<div class="imgcap">
<img src="/assets/ml_models_2/image_3.png">
<div class="thecap">Server Side</div>
</div>

Congrats, your first application in falcon is up and working !

Wanna read more on falcone ? Head to [falcon’s page](http://falcon.readthedocs.io/en/stable/user/tutorial.html).

### FALCON VS FLASK:

I am told falcon is light weight. I am not an expert on this topic, here are a couple of links if you wish to take a deep dive into Flask vs Falcon: [Python web framworks benchmarking](http://klen.github.io/py-frameworks-bench/), [reddit thread](https://www.reddit.com/r/Python/comments/4rq19c/your_choice_of_rest_framework_flask_vs_falcon/), [hacker news](https://news.ycombinator.com/item?id=8835776), [slant](https://www.slant.co/versus/1398/1744/~flask_vs_falcon).


We have a model that is up and running as service. Now you can go ahead and add more functionality like resetting the model, training the model if not already trained, and lot more.

If you face any issue with the code, please open a "[New issue](https://github.com/anujgupta82/Musings/issues)" on Github. All the above code files are available [here](https://github.com/anujgupta82/Musings/tree/master/falcon).