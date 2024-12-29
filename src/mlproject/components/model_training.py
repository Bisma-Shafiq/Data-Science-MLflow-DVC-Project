import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Any
from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
from src.mlproject.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("dataset_source", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate evaluation metrics for model performance."""
        try:
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            return rmse, mae, r2
        except Exception as e:
            logging.error(f"Error in calculating metrics: {str(e)}")
            raise CustomException(e, sys)

    def get_model_params(self) -> Tuple[Dict, Dict]:
        """Define models and their hyperparameters."""
        models = {
            'Linear Regression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }

        params = {
            "Linear Regression": {},
            "Lasso": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'max_iter': [100, 500, 1000]
            },
            "Ridge": {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "ElasticNet": {
                'alpha': [0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.9]
            },
            "Decision Tree": {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "SVR": {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            "KNN": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }
        }
        return models, params

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        try:
            logging.info("Starting model training process")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models, params = self.get_model_params()
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            logging.info(f"Best performing model: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            logging.info(f"Model training completed. R2 Score: {r2_square}")
            return r2_square

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
