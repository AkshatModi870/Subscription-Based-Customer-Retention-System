import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
from sklearn.feature_selection import SelectFromModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import StackingClassifier

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from src_sbcrs.exception import CustomException
from src_sbcrs.logger import logging
from src_sbcrs.utils import save_object, tune_all_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    feature_selector_path = os.path.join('artifacts', 'feature_selector.pkl')
    threshold_info_path = os.path.join('artifacts', 'threshold.txt')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Map target to 0/1 if necessary (should already be done in transformation)
            if y_train.dtype == object:
                y_train = y_train.map({'Yes': 1, 'No': 0})
                y_test = y_test.map({'Yes': 1, 'No': 0})

            original_y_train = y_train.copy()

            # 1. Feature Selection (Top 12 features using Random Forest)
            
            logging.info("Performing feature selection (top 12 features)")
            selector_rf = RandomForestClassifier(n_estimators=100, random_state=53, n_jobs=-1)
            selector_rf.fit(X_train, y_train)

            # Select top 12 features
            importances = selector_rf.feature_importances_
            indices = np.argsort(importances)[-12:][::-1]
            X_train_selected = X_train[:, indices]
            X_test_selected = X_test[:, indices]

            logging.info(f"Selected feature indices: {indices.tolist()}")

            # Save feature selector (indices) for later use
            save_object(
                file_path=self.model_trainer_config.feature_selector_path,
                obj=indices
            )

            # 2. Handle Class Imbalance with SMOTE + RandomOverSampler

            logging.info("Applying SMOTE + RandomOverSampler")
            smote_pipeline = ImbPipeline([
                ('smote', SMOTE(sampling_strategy=0.5, random_state=53)),
                ('over', RandomOverSampler(sampling_strategy=1, random_state=53))
            ])
            X_resampled, y_resampled = smote_pipeline.fit_resample(X_train_selected, y_train)
            logging.info(f"Resampled dataset shape: {X_resampled.shape}, class distribution: {np.bincount(y_resampled.astype(int))}")

            # 3. Hyperparameter Tuning of Base Models

            RECALL_WEIGHT = 50   # Adjust as needed (same as notebook)
            best_estimators, tuning_results = tune_all_models(
                X_resampled, y_resampled,
                original_y_train=original_y_train,
                recall_weight=RECALL_WEIGHT,
                cv=5,
                n_iter=20
            )

            # Log tuning summary
            logging.info("Tuning Results (sorted by Recall):\n" + tuning_results.to_string())

            # 4. Build Stacking Ensemble

            # Select top base models for stacking (e.g., top 3 by recall from tuning_results)
            base_model_names = ['Logistic Regression', 'AdaBoost', 'SVM (RBF)']
            base_models = [(name, best_estimators[name]) for name in base_model_names]
            logging.info(f"Base models for stacking: {base_model_names}")

            meta_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=53)
            stack = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                cv=5,
                stack_method='predict_proba'
            )

            logging.info("Training stacking classifier on resampled data")
            stack.fit(X_resampled, y_resampled)

            # 5. Evaluate with Manual Threshold

            CHOSEN_THRESHOLD = 0.35   # You may tune this further
            y_proba = stack.predict_proba(X_test_selected)[:, 1]
            y_pred_manual = (y_proba >= CHOSEN_THRESHOLD).astype(int)

            logging.info(f"Evaluation on test set with threshold = {CHOSEN_THRESHOLD}")
            report = classification_report(y_test, y_pred_manual, target_names=['No Churn', 'Churn'])
            logging.info("\n" + report)

            # Log confusion matrix counts
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_manual).ravel()
            logging.info(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")

            # Save the chosen threshold
            with open(self.model_trainer_config.threshold_info_path, 'w') as f:
                f.write(f"threshold={CHOSEN_THRESHOLD}\n")
                f.write(f"recall_weight={RECALL_WEIGHT}\n")

            # 6. Save Final Model Artifacts

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=stack
            )
            logging.info(f"Stacking model saved at {self.model_trainer_config.trained_model_file_path}")

            # Also print results to console (optional)
            print("MODEL TRAINING COMPLETED")
            print("-"*60)
            print(f"Selected feature indices: {indices.tolist()}")
            print(f"Top base models: {[name for name, _ in base_models]}")
            print(f"Threshold used: {CHOSEN_THRESHOLD}")
            print("\nClassification Report on Test Set:\n")
            print(report)

            return (
                stack,
                indices,
                CHOSEN_THRESHOLD
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    # Load the preprocessed arrays from artifacts
    train_path = os.path.join('artifacts', 'train_arr.npy')
    test_path = os.path.join('artifacts', 'test_arr.npy')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Preprocessed arrays not found. Run data_ingestion.py first.")
        sys.exit(1)

    train_arr = np.load(train_path, allow_pickle=True)
    test_arr = np.load(test_path, allow_pickle=True)

    trainer = ModelTrainer()
    model, indices, threshold = trainer.initiate_model_trainer(train_arr, test_arr)