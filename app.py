from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(review_text= request.form.get('review_text'))

        pred_df = data.get_data_as_data_frame()

        logging.info("DataFrame for Prediction:")
        logging.info(str(data))
        logging.info("Before Prediction")

        predict_pipeline = PredictPipeline()
        logging.info("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        logging.info("After Prediction")

        return render_template('home.html', results= results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5003)        


