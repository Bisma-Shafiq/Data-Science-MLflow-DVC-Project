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
            numerical_columns=['Year', 'owner', 'Selling Price']
            categorial_columns=['car_name', 'fuel_type', 'transmission']

            numerical_pipeline= Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scalar',StandardScaler())
            ])

            categorial_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('label_encoder',LabelEncoder()),
                ('scalar',StandardScaler(with_mean=False))
            ])

            logging.info(f'Categorical columns:{categorial_columns}')
            logging.info(f"Numerical columns:{numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline',numerical_pipeline,numerical_columns),
                    ('categorical_pipeline',categorial_pipeline,categorial_columns)
                ]
            )        
            return preprocessor



        except Exception as e:
            logging.error(f"Error in get_data_transformation: {str(e)}")
            raise CustomException(f"Error in get_data_transformation: {str(e)}")


def initiate_data_transformation(self,train_path,test_path):
    try:
        train_dataset=pd.read_csv(train_path)
        test_dataset=pd.read_csv(test_path)


        logging.info("Reading the train test file")

        preprocessing_obj = self.get_data_transformation_object()

        target_columns = 'Selling Price'
        numerical_columns = ['Year', 'owner']
    # divide data to independent dependent 
        input_features_train_dataset = train_dataset.drop(columns=[target_columns],axis=1)
        target_features_train_dataset= train_dataset[target_columns]

        # divide data to independent dependent 
        input_features_test_dataset = test_dataset.drop(columns=[target_columns],axis=1)
        target_features_test_dataset= test_dataset[target_columns]

        logging.info('Applying Preprocessing on training and testing dataframe')

        input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_dataset)
        input_features_test_arr = preprocessing_obj.transform(input_features_test_dataset)

        train_arr = np.c_[
            input_features_train_arr, np.array(target_features_train_dataset)
        ]


        logging.info(f"Saved preprocessing object")

        save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

        return (

                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

    except Exception as e:
        raise CustomException(sys,e)