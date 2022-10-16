#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model_diabetes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diabetes', methods=['POST', 'GET'])
def rdiabetes():
    return render_template('diabetes.html')
	
@app.route('/diabetes.html', methods=['POST', 'GET'])
def diabetes():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        pred = "are likely to have DIABETES"
    elif prediction == 0:
        pred = "don't worry! You are not likely to have DIABETES"
    output = pred
    return render_template('diabetes.html', prediction_text='You {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)