import sys
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, extract_corpus
from src.configs.configurations import PredictPipelineConfig

class CustomData:
    def __init__(self, review_text: str):
        self.review_text = review_text

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "reviewText": [self.review_text]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        
    def __str__(self):
        return f"Review Text={self.review_text}"


class PredictPipeline:
    def __init__(self):
        self.predict_pipeline_config = PredictPipelineConfig()

    def predict(self,features):
        try:
            logging.info("Before Loading")

            model = load_object(file_path=self.predict_pipeline_config.model_path)
            y_preprocessor = load_object(file_path=self.predict_pipeline_config.Y_preprocessor_obj_file_path)
            w2v_model_estimator = load_object(file_path= self.predict_pipeline_config.w2v_model_estimator_path)
        
            logging.info("Extract corpus from features & Convert to vectors")
            corpus = extract_corpus(features)

            # Convert corpus to numerical vectors using Word2Vec
            X = w2v_model_estimator.transform(corpus)

            logging.info("After Vectorization")

            y_pred = model.predict(X)
            logging.info(f"y_pred: {y_pred}")
            
            return y_pred
        
        except Exception as e:
            raise CustomException(e,sys)
