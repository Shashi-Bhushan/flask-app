This is a demonstration of a Flask app, intended to load an SVM model trained on MNIST dataset.
It takes the index via a POST call and returns the predicted number for the entry in Training data at the index.

First run trainandsave.py to train the model. I've trained it on a minimal dataset for demonstration purpose only. 
Advice to run on full training dataset for accurate predictions.

Then, run app.py, it loads ML Mode in the Flask app.

Sample curl command to Flask is for the 

curl -X POST \
  http://localhost:5000/mnist_classify \
  -H 'cache-control: no-cache' \
  -H 'content-type: application/json' \
  -H 'postman-token: 7f8c7c4c-3dbc-b5a9-abeb-1adff331c59e' \
  -d '{"index": "180"}'