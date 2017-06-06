---
layout: post
comments: true
title:  "Deploying ML models - Part 1"
excerpt: "Deploy ML model using Flask"
date:   2016-11-04 15:40:00
mathjax: false
---

### Flask

Flask is a lightweight Python web framework to create microservices. Wanna read [more](https://code.tutsplus.com/tutorials/an-introduction-to-pythons-flask-framework--net-28822) ? I am a ML guy, and this sounds complex :-( Rather than reading lets code a simple one quickly !

#### Install Flask

```python
pip install flask
```

I used python 2.7 and Flask==0.11.1

#### Bare bones Example

Open am editor and copy paste code from my [git repo](https://github.com/anujgupta82/Musings/blob/master/flask/simple_app.py)

```python
from flask import Flask

app = Flask(__name__)

@app.route('/1')   # path to resource on server
def index_1():        # action to take
  return "Hello_world 1"

@app.route('/2')
def index_2():
  return "Hello_world 2"

if __name__ == '__main__':
  app.run(debug=True)
```

To run this:

    1. Save it as simple_app.py    
    2. Install Flask in your virtual environment [pip install Flask]     
    3. Open terminal, go to the directory where app.py is saved. Run following two commands     

```python
export FLASK_APP=simple_app.py
flask run
```

This should have the flask server up and running on http://127.0.0.1:5000

<div class="imgcap">
<img src="/assets/ml_models_1/image_1.png">
<div class="thecap">Flask Server up and running</div>
</div>



If you see the code carefully it says - we have 2 resources with relative URIs as '/1' and '/2'. Lets access them. Go to browser and type http://127.0.0.1:5000/1

This should fire the function index_1() function and give the following output on the command prompt

<div class="imgcap">
<img src="/assets/ml_models_1/image_2.png">
<div class="thecap">Output from function index_1()</div>
</div>

Like wise http://127.0.0.1:5000/2 should work. This is a simple flask application. (Oh yeah! this sounds easy, lets move on)

#### REST (in peace)

There are a couple of terms that are part and parcel on micro services. Lets quickly  understand something about them.

    1) API : Application Program Interface - set of routines, protocols, and tools for building software applications.     

    2) API Endpoint :It's one end of a communication channel, so often this would be represented as the URL of a server or service. In our example "http://127.0.0.1:5000/1"      
    
    3) REST :underlying architectural principle of the web. 
    Read these awesome [stackoverflow answer](https://stackoverflow.com/questions/671118/what-exactly-is-restful-programming/671132#671132) and this brilliant [post](http://web.archive.org/web/20130116005443/http://tomayko.com/writings/rest-to-my-wife) from Ryan Tomayko and this [post](https://martinfowler.com/articles/richardsonMaturityModel.html) from Martin Fowler to understand the same.     

In nutshell, you need to have - GET, POST, PUT, DELETE.

Lets add this to our [code](https://github.com/anujgupta82/Musings/blob/master/flask/RESTful_app.py). To see this in action, run the server (like previously), go to terminal and type

```python
curl -i http://localhost:5000/tasks
```

or 

```python
curl -i -X GET http://localhost:5000/tasks
```

Both the commands will give same output:

<div class="imgcap">
<img src="/assets/ml_models_1/image_3.png">
<div class="thecap">GET request</div>
</div>


Your server terminal will show "200" (success) for both the requests.

<div class="imgcap">
<img src="/assets/ml_models_1/image_4.png">
<div class="thecap">200 – success</div>
</div>

#### **RESTful App**

Lets add other parts of RESTful to out code. Here it is. To see this in action, run the server (like previously), go to terminal and type:

 	1) Get All tasks:    

```python
curl -i http://localhost:5000/tasks/
```         

 	2) Get a specific task:

```python
curl -i http://localhost:5000/tasks/2
```    

<div class="imgcap">
<img src="/assets/ml_models_1/image_5.png">
<div class="thecap">Get task with id=2</div>
</div>    

Since there is no task with id=4, lets see what happens when we try to get that:    

```python
curl -i http://localhost:5000/tasks/4
```

<div class="imgcap">
<img src="/assets/ml_models_1/image_6.png">
<div class="thecap">Error. Task with id=4 does not exists</div>
</div>


3) Add a task:     

```python
curl -i -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}' http://localhost:5000/tasks/
```     

4) Update a specific task:     

```python
curl -i -H "Content-Type: application/json" -X PUT -d '{"done":true}' http://localhost:5000/tasks/2
``` 
<div class="imgcap">
<img src="/assets/ml_models_1/image_7.png">
<div class="thecap">Task 2's status is updated ['done':true]</div>
</div>     

5) Delete a specific task:     

```python
curl -i -X DELETE http://localhost:5000/tasks/2
```     
<div class="imgcap">
<img src="/assets/ml_models_1/image_8.png">
<div class="thecap">task with id=2 successfully deleted.</div>
</div>

Source: Large part of the code comes from Miguel Grinberg awesome [post](https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask).      

### ML APP

Its all great until now, what about the core issue – ML model as microservice ? Here we go:

```python
from flask import Flask, jsonify, request, abort
import pickle
import numpy as np

app = Flask(__name__)
model_file_path = "./../models/final_model.pkl"
model = None

def get_prediction(X):
  X_i = X.reshape(1, 2)
  print "X_i.shape"
  print X_i.shape#load the model if not already done
  
  global model
  if model == None:
    model = pickle.load(open(model_file_path, 'rb'))#make prediction and return the same
    prediction = model.predict(X_i)[0]

return prediction

@app.route('/predict', methods=['POST'])

def predict():
  if not request.json or not 'X1' in request.json or not 'X2' in request.json:
    abort(404)

  X1 = request.json['X1']
  X2 = request.json['X2']

  X1_ = np.float64(X1)
  X2_ = np.float64(X2)
  X = np.array([X1_, X2_])
  prediction = get_prediction(X)

  return jsonify({'prediction':prediction})

if __name__ == '__main__':
  app.run(debug=True)
``` 

To run this code, get the server up and running and then fire the command below from another terminal.

```python
curl -i -H "Content-Type: application/json" -X POST -d '{"X1":"61.1", "X2":"17.3"}' http://localhost:5000/predict
``` 

Complete code is [here](https://github.com/anujgupta82/Musings/blob/master/flask/ML_app.py). It has lot of print statements. You should see these statements printing the relevant stuff on the server terminal.

<div class="imgcap">
<img src="/assets/ml_models_1/image_9.png">
<div class="thecap">Model Prediction </div>
</div>

<div class="imgcap">
<img src="/assets/ml_models_1/image_10.png">
</div>

Lets understand some key aspects of this code. Here, predict() handles any POST request coming on /predict on the server. We extract the payload – components of input vector – these cannot be sent from client as numpy arrays. These components come in as unicode strings. Hence, we transform them explicitly into numpy.float64 and then make a numpy array on the server side. There is an elegant way to do it, which we will see in short while.

Once we have the payload in right format, we invoke the function – get_prediction(). Its main job is load the model into memory if not already loaded, and fire the model on input for getting the prediction.

We have a model that is up and running as service. Now you can go ahead and add more functionality like resetting the model, training the model if not already trained, and lot more.

If you face any issue with the code, please open a "[New issue](https://github.com/anujgupta82/Musings/issues)" on Github. All the above 4 code files are available [here](https://github.com/anujgupta82/Musings/tree/master/flask).