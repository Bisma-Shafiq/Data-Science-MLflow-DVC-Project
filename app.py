from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.data_transformation import DataTransformationConfig

import sys


if __name__ == "__main__":
    logging.info("Hello, world!")

    try:
        #a=1/0
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr=data_transformation.initiate_data_transformation(train_data_path,train_data_path)
        

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e,sys)