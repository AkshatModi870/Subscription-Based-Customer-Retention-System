# utils.py
import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src_sbcrs.exception import CustomException
from src_sbcrs.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def get_custom_scorer(recall_weight=50):
    """
    Creates an F-beta scorer where beta is derived from the desired recall weight.
    recall_weight : int between 0 and 100
    """
    try:
        precision_weight = 100 - recall_weight
        beta = (recall_weight / precision_weight) ** 0.5
        scorer = make_scorer(fbeta_score, beta=beta)
        return scorer, beta
    except Exception as e:
        raise CustomException(e, sys)

def get_param_grids(scale_pos_weight=1):
    """
    Returns a dictionary of models and their corresponding hyperparameter grids.
    """
    try:
        models = {
            "Logistic Regression": {
                'model': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=53),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            "Random Forest": {
                'model': RandomForestClassifier(class_weight='balanced', random_state=53, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            "XGBoost": {
                'model': XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=53, verbosity=0,
                                       use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 1.0],
                    'colsample_bytree': [0.7, 0.8, 1.0]
                }
            },
            "LightGBM": {
                'model': LGBMClassifier(class_weight='balanced', random_state=53, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.7, 0.8, 1.0]
                }
            },
            "CatBoost": {
                'model': CatBoostClassifier(auto_class_weights='Balanced', random_seed=53, verbose=0),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'l2_leaf_reg': [1, 3, 5]
                }
            },
            "AdaBoost": {
                'model': AdaBoostClassifier(random_state=53),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                }
            },
            "SVM (RBF)": {
                'model': SVC(class_weight='balanced', probability=True, random_state=53),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf']
                }
            },
            "K-Neighbors": {
                'model': KNeighborsClassifier(weights='distance'),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'p': [1, 2]
                }
            }
        }
        return models
    except Exception as e:
        raise CustomException(e, sys)

def tune_all_models(X_train, y_train, original_y_train=None, recall_weight=50, cv=5, n_iter=20):
    """
    Performs hyperparameter tuning for all models defined in get_param_grids.
    Returns a dictionary with best estimators and a DataFrame of results.
    """
    try:
        # Compute scale_pos_weight from original labels (before resampling)
        if original_y_train is not None:
            neg, pos = np.bincount(original_y_train.astype(int))
            scale_pos_weight = neg / pos
        else:
            scale_pos_weight = 1
            logging.info("No original_y_train provided; using default scale_pos_weight=1 for XGBoost")

        models_dict = get_param_grids(scale_pos_weight=scale_pos_weight)
        scorer, beta = get_custom_scorer(recall_weight)

        tuned_results = []
        best_estimators = {}

        logging.info(f"Starting hyperparameter tuning with custom F-beta scorer (beta={beta:.3f})")

        for name, config in models_dict.items():
            logging.info(f"Tuning {name}...")
            model = config['model']
            params = config['params']

            if name in ["XGBoost", "LightGBM", "Random Forest", "CatBoost"]:
                search = RandomizedSearchCV(
                    model, params, n_iter=n_iter, scoring=scorer,
                    cv=cv, random_state=53, n_jobs=-1, verbose=0
                )
            else:
                search = GridSearchCV(
                    model, params, scoring=scorer,
                    cv=cv, n_jobs=-1, verbose=0
                )

            search.fit(X_train, y_train)
            best_estimators[name] = search.best_estimator_

            y_pred = search.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            prec = precision_score(y_train, y_pred)
            rec = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)

            tuned_results.append({
                'Model': name,
                'Best Params': str(search.best_params_),
                'CV Score': search.best_score_,
                'Train Accuracy': acc,
                'Train Precision': prec,
                'Train Recall': rec,
                'Train F1': f1
            })

        results_df = pd.DataFrame(tuned_results).sort_values('Train Recall', ascending=False)
        logging.info("Hyperparameter tuning completed.")
        return best_estimators, results_df

    except Exception as e:
        raise CustomException(e, sys)