from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
from src.mlproject.components.data_ingestion import DataIngestion , DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation , DataTransformationConfig
import sys

if __name__ == "__main__":
    try:
        logging.info("Pipeline started")

        # Data ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error("Pipeline failed")
        raise CustomException(e, sys)
