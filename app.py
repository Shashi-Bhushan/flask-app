from flask import Response
from flask import Flask, render_template, request
from sklearn.datasets import fetch_openml
import pickle
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


mnist = fetch_openml('mnist_784', version=1, data_home='./')
data = mnist.data
target = mnist.target
svm_classifier = pickle.load(open('svm.sav', 'rb'))


@app.route('/mnist_classify', methods=['POST'])
def transform():
    if request.headers['Content-Type'] == 'application/json':
        # Get the Query Param
        query_index = int(request.json.get("index"))

        # use SVM to Predict the Data from the index. Valid values are 0 to size of data.
        results = svm_classifier.predict([data[query_index]])

        # Dump predicted value in result
        js = json.dumps({"result": results[0]})

        resp = Response(js, status=200, mimetype='application/json')
        return resp


if __name__ == '__main__':
    app.run(debug=True)
