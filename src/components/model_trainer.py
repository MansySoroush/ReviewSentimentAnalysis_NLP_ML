import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, calculate_score
from src.configs.configurations import ModelTrainerConfig

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Evaluating various models to find the best trained model.")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = self.get_models_to_train()

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train,
                                                X_test = X_test, y_test = y_test,
                                                models = models)
            
            logging.info("Evaluating models completed.")

            # Rank models by combined score
            scores = {
                name: calculate_score(model_info["test_metrics"]) for name, model_info in model_report.items()
            }

            best_model_name = max(scores, key=scores.get)
            best_model_score = scores[best_model_name]

            threshold_score = calculate_score(self.model_trainer_config.thresholds)
            logging.info(f"threshold_score: {threshold_score}")
            logging.info(f"best_model_score: {best_model_score}")

            if best_model_score < threshold_score:
                raise CustomException("No model met the performance threshold")

            # Retrieve the best model
            best_model = self.get_model_by_name(model_name= best_model_name, models= models)

            if best_model == None:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"------------Results-----------")
            logging.info(f"Selected Model: {best_model_name}")
            logging.info(f"Best Params: {model_report[best_model_name]['params']}")
            logging.info(f"Test Metrics: {model_report[best_model_name]['test_metrics']}")
            logging.info(f"Score: {best_model_score}")
            logging.info(f"------------------------------")

            # Save and return the best model
            save_object(self.model_trainer_config.trained_model_file_path, obj= best_model)
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)

            logging.info(f"Final Accuracy: {accuracy}")

            return accuracy           
        except Exception as e:
            raise CustomException(e,sys)
        
    def get_models_to_train(self):
        log_reg_params = {
            "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            "penalty": ['none', 'l1', 'l2', '‘elasticnet’'],
            "C": [100, 10, 1.0, 0.1, 0.01]
        }

        ridge_params = {
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }

        dec_tree_params = {
            'criterion':['gini','entropy', 'log_loss'],
            'splitter':['best','random'],
            'max_depth':[1,2,3,4,5],
            'max_features':['auto','sqrt','log2']
        }

        rf_params = {
            "max_depth": [5, 8, 15, None, 10],
            "max_features": ['auto', 'sqrt', 'log2'],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators": [100, 200, 500, 1000]
        }

        gradient_params={
            "loss": ['log_loss','deviance','exponential'],
            "criterion": ['friedman_mse','squared_error','mse'],
            "min_samples_split": [2, 8, 15, 20],
            "n_estimators": [100, 200, 500],
            "max_depth": [5, 8, 15, None, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }

        adaboost_params = {
            'n_estimators' : [50, 70, 90, 120, 180, 200],
            'learning_rate' : [0.001, 0.01, 0.1, 1, 10],
            'algorithm':['SAMME','SAMME.R']
            }

        svc_params = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'degree': [2, 3, 4],
            'kernel': ['rbf','linear','poly','sigmoid']
        }

        knb_params = {
            'n_neighbors': range(1, 21, 2),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        models = [
            ("Logistic", LogisticRegression(), log_reg_params),
            ("Ridge", RidgeClassifier(), ridge_params),
            ("Decision Tree", DecisionTreeClassifier(), dec_tree_params),
            ("Random Forest", RandomForestClassifier(), rf_params),
            ("Gradient Boost", GradientBoostingClassifier(), gradient_params),
            ("Ada-boost", AdaBoostClassifier(), adaboost_params),
            ("SVC", SVC(kernel='linear'), svc_params),
            ("K-Neighbors", KNeighborsClassifier(), knb_params)
        ]

        return models

    def get_model_by_name(self, model_name, models):
        for name, model, params in models:
            if name == model_name:
                return model
            
        return None



if __name__=="__main__":
    obj = DataIngestion()
    raw_data, w2v_estimator, corpus = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(raw_data, w2v_estimator, corpus)

    model_trainer = ModelTrainer()
    accuracy = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Accuracy Score of the Trained Model is: {accuracy}")
    

