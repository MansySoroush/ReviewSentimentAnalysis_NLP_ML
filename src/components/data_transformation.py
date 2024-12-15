import sys
import numpy as np 
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.configs.configurations import DataTransformationConfig
from src.utils import save_object, load_object

from src.components.data_ingestion import DataIngestion


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation      
        '''
        try:
            # Initialize the label encoder
            label_encoder = LabelEncoder()

            return label_encoder
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, raw_data_path, w2v_estimator_path, corpus):
        try:
            df = pd.read_csv(raw_data_path)

            logging.info("Read raw data completed")
            logging.info("Obtaining preprocessing object and making ready X & y to split")

            target_column_name = "rating"

            # Dependent feature
            y_preprocessing_obj = self.get_data_transformer_object()
            df[target_column_name] = y_preprocessing_obj.fit_transform(df[target_column_name])

            custom_labels = self.get_custom_target_labels()

            df[target_column_name] = df[target_column_name].map(custom_labels)
            y = df[target_column_name]

            # Independent feature
            #--------------------
            w2v_model_estimator = load_object(file_path= w2v_estimator_path)

            # Transform reviews into vectors using the best Word2Vec model
            X = w2v_model_estimator.transform(corpus)

            logging.info("Train test split initiated")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            logging.info(f"Training data size: {len(X_train)}")
            logging.info(f"Test data size: {len(X_test)}")

            # Convert X_train and X_test to DataFrames for compatibility
            X_train_df = pd.DataFrame(X_train, columns=[f'vector_{i}' for i in range(X_train.shape[1])])
            X_test_df = pd.DataFrame(X_test, columns=[f'vector_{i}' for i in range(X_test.shape[1])])

            # Ensure y_train and y_test are pandas Series (if not already)
            y_train_df = y_train.reset_index(drop=True)
            y_test_df = y_test.reset_index(drop=True)

            # Concatenate X and y for train and test datasets
            train_set = pd.concat([X_train_df, y_train_df], axis=1)
            test_set = pd.concat([X_test_df, y_test_df], axis=1)

            # Save the train and test datasets to CSV files
            train_set.to_csv(self.data_transformation_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_transformation_config.test_data_path, index=False, header=True)

            logging.info("Train and test datasets saved.")

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.Y_preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)

    def get_custom_target_labels(self):
        custom_labels = {1: 'Positive', 0: 'Negative'}
        return custom_labels
    

if __name__=="__main__":
    obj = DataIngestion()
    raw_data, w2v_estimator, corpus = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(raw_data, w2v_estimator, corpus)
