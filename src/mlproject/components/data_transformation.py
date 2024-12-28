import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.mlproject.utils import save_object
from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('dataset_source', 'preprocessor.pkl')

def label_encode_column(data):
    """
    Encodes categorical columns using LabelEncoder.
    Handles both pandas DataFrame and NumPy array inputs.
    """
    try:
        if isinstance(data, np.ndarray):
            # Assuming input is already split column-wise in a NumPy array
            return np.array([LabelEncoder().fit_transform(data[:, col]) for col in range(data.shape[1])]).T
        elif hasattr(data, "columns"):  # pandas DataFrame
            return np.array([LabelEncoder().fit_transform(data[col]) for col in data.columns]).T
        else:
            raise ValueError("Unsupported data type for label encoding")
    except Exception as e:
        raise CustomException(e, sys)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        """
        Prepares the preprocessing pipelines for numerical and categorical data.
        """
        try:
            numerical_columns = ['Year', 'owner']
            categorial_columns = ['car_name', 'fuel_type', 'transmission']

            # Pipeline for numerical data
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scalar', StandardScaler())
            ])

            # Pipeline for categorical data
            categorial_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('label_encoder', FunctionTransformer(label_encode_column, validate=False)),  # Updated function
                ('scalar', StandardScaler(with_mean=False))
            ])

            logging.info(f'Categorical columns: {categorial_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            # Combined preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical_pipeline', numerical_pipeline, numerical_columns),
                    ('categorical_pipeline', categorial_pipeline, categorial_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformation: {str(e)}")
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        """
        Applies preprocessing to training and testing datasets and saves the preprocessor object.
        """
        try:
            # Load datasets
            train_dataset = pd.read_csv(train_path)
            test_dataset = pd.read_csv(test_path)

            logging.info("Loaded train and test datasets")

            preprocessing_obj = self.get_data_transformation()

            target_column = "Selling Price"
            input_features_train = train_dataset.drop(columns=[target_column])
            target_features_train = train_dataset[target_column]

            input_features_test = test_dataset.drop(columns=[target_column])
            target_features_test = test_dataset[target_column]

            # Apply transformations
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_test)

            # Combine features and target
            train_arr = np.c_[input_features_train_arr, target_features_train.values]
            test_arr = np.c_[input_features_test_arr, target_features_test.values]

            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessing and saving completed")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_file_path

        except Exception as e:
            logging.error("Error in data transformation")
            raise CustomException(e, sys)
