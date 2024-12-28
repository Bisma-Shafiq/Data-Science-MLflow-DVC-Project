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
# data ingestion

        #data_ingestion_config=DataIngestionConfig()
        #data_ingestion=DataIngestion()
        #data_ingestion.initiate_data_ingestion()
# data transfromation
        #data_transformation_config=DataIngestionConfig()
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation()

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e,sys)