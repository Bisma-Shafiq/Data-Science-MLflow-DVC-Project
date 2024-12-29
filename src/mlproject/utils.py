# utils.py
import os
import sys
from typing import Dict, Any
import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import pymysql
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.mlproject.exception_handling import CustomException
from src.mlproject.logger import logging

load_dotenv()

class DatabaseConfig:
    host = os.getenv("host")
    user = os.getenv("user")
    password = os.getenv("password")
    db = os.getenv('db')

def read_sql_data() -> pd.DataFrame:
    """Read data from SQL database."""
    logging.info("Reading SQL database started")
    try:
        config = DatabaseConfig()
        mydb = pymysql.connect(
            host=config.host,
            user=config.user,
            password=config.password,
            db=config.db
        )
        logging.info("Database connection established successfully")
        df = pd.read_sql_query('SELECT * FROM students', mydb)
        logging.info(f"Data read successfully. Shape: {df.shape}")
        return df

    except Exception as ex:
        logging.error(f"Error in reading SQL data: {str(ex)}")
        raise CustomException(ex, sys)
    finally:
        if 'mydb' in locals():
            mydb.close()
            logging.info("Database connection closed")

def save_object(file_path: str, obj: Any) -> None:
    """Save a Python object to a file using pickle."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error in saving object: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict,
    params: Dict
) -> Dict:
    """Evaluate multiple models using grid search and return their performance metrics."""
    try:
        report = {}
        
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            
            # Get parameters for current model
            para = params.get(model_name, {})
            
            if para:
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                
                model.set_params(**gs.best_params_)
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
            
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
            logging.info(f"{model_name} - Train Score: {train_model_score}, Test Score: {test_model_score}")
        
        return report

    except Exception as e:
        logging.error("Error in model evaluation")
        raise CustomException(e, sys)