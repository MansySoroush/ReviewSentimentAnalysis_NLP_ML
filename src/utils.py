import os
import sys
import dill
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.w2v_estimator import Word2VecEstimator
from src.configs.configurations import WeightScoreConfig
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score 

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def custom_w2v_score(estimator, X, y=None):
    '''
    This function is responsible for scoring word similarity task
    '''
    # Example: Compute cosine similarity for specific word pairs
    word_pairs = [("king", "queen"), ("man", "woman")]
    similarities = [estimator.model.wv.similarity(w1, w2) for w1, w2 in word_pairs]
    return np.mean(similarities)

def evaluate_w2v_model(w2v_param, corpus):
    try:
        report = {}

        word2vec_estimator = Word2VecEstimator()
        random_search = RandomizedSearchCV(
            estimator= word2vec_estimator,
            param_distributions= w2v_param,
            n_iter= 30,  # Number of random samples
            scoring= make_scorer(custom_w2v_score, greater_is_better= True),
            cv=3,  # Cross-validation folds
            verbose=1,
            n_jobs=-1
        )

        # Run hyperparameter tuning
        random_search.fit(corpus)
        best_w2v_estimator = random_search.best_estimator_

        report = {
            "best_param": random_search.best_params_,
            "best_score": random_search.best_score_,
            "best_estimator": best_w2v_estimator
        }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for name, model, params in models:
            logging.info(f"Training {name} Model...")
            search_cv = RandomizedSearchCV(estimator=model,
                                        param_distributions=params,
                                        n_iter=50,
                                        cv=3,
                                        verbose=2,
                                        random_state=42,
                                        n_jobs=-1)
            search_cv.fit(X_train, y_train)

            # Update model with best parameters
            model.set_params(**search_cv.best_params_)
            model.fit(X_train,y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate performance (Training set & Test set)
            train_metrics = evaluate_model(y_train, y_train_pred)
            test_metrics = evaluate_model(y_test, y_test_pred)

            # Add to report
            report[name] = {
                "params": search_cv.best_params_,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(true, predicted):
    metrics = {
        "accuracy": accuracy_score(true, predicted),
        "f1_score": f1_score(true, predicted, average='weighted'),
        "precision": precision_score(true, predicted, average='weighted'),
        "recall": recall_score(true, predicted, average='weighted'),
    }
    return metrics

def calculate_score(metrics, weights=None):
    if weights is None:
        weights = { 
            "accuracy": WeightScoreConfig.accuracy_weight, 
            "f1_score": WeightScoreConfig.f1_score_weight, 
            "precision": WeightScoreConfig.precision_weight, 
            "recall": WeightScoreConfig.recall_weight
        }
    
    score = 0
    for metric, weight in weights.items():
        score += metrics[metric] * weight
    return score

def meets_thresholds(metrics, thresholds):
    return all(metrics[metric] >= threshold for metric, threshold in thresholds.items())

