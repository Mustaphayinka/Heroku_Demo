from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    print(">>>Got here")
    model = pickle.load(open('model.pkl', 'rb'))
    print(">>>Was able to load model")
    text_query = request.form['textquery']
    print(">>>Was able to query form", text_query)
    print('dir:',dir(model))
    print('type',type(model))
    prediction = model.predict(text_query)
    print('prediction1',prediction)
    prediction = model.predict([text_query])
    print('prediction:',prediction)
    prediction_proba = model.predict_proba([text_query])
    probability = np.max(prediction_proba)
    print('Final command, prob:',probability)
    return render_template('result.html', prediction = prediction, pred = probability)




if __name__ == '__main__':
    app.run(debug = True)
