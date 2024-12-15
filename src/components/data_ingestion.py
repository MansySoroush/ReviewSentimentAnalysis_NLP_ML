import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.configs.configurations import DataIngestionConfig
from src.utils import evaluate_w2v_model, save_object

from nltk import sent_tokenize
from gensim.utils import simple_preprocess

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebooks/data/cleaned_reviews.csv')
            logging.info('Read the dataset as data-frame')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Extract corpus from data-frame")
            corpus = self.extract_corpus(df)

            logging.info("Hyperparameter tuning to find best W2V estimator")
            w2v_param = self.get_w2v_param()

            w2v_report = evaluate_w2v_model(w2v_param= w2v_param, corpus= corpus)

            logging.info(f"Best W2V Parameters: {w2v_report['best_param']}")

            best_w2v_model_estimator = w2v_report['best_estimator']

            # Save the best W2V Model
            save_object(self.ingestion_config.w2v_model_estimator_path, obj= best_w2v_model_estimator)

            w2v_model = best_w2v_model_estimator.model

            logging.info("Ingestion of the data is completed")

            return(self.ingestion_config.raw_data_path, self.ingestion_config.w2v_model_estimator_path, corpus)
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def extract_corpus(self, data_set):
        reviews = data_set['reviewText'].to_numpy()
        corpus=[]
        for sent1 in reviews:
            sent_token = sent_tokenize(sent1)
            for sent2 in sent_token:
                corpus.append(simple_preprocess(sent2))
        return corpus
    
    def get_w2v_param(self):
        param_dict = {
            'vector_size': [50, 100, 150, 300],
            'window': [3, 5, 7],
            'sg': [0, 1],  # 0 = CBOW, 1 = Skip-gram
            'min_count': [1, 3, 5, 10],
            'epochs': [10, 20, 30],
            'alpha': np.linspace(0.01, 0.1, 5),
        }
        return param_dict
    
if __name__ == "__main__":
    obj=DataIngestion()
    raw_data, w2v_estimator, corpus = obj.initiate_data_ingestion()



