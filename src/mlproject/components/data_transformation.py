import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException

@dataclass

class DataTransformationConfig:
    preprocessor_file_path= os.path.join('dataset_source','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        # responsible for data transformation
        try:
            preprocessor = self.load_preprocessor()
            return preprocessor
        
        except Exception as e:
            logging.error(f"Error in get_data_transformation: {str(e)}")
            raise CustomException(f"Error in get_data_transformation: {str(e)}")

