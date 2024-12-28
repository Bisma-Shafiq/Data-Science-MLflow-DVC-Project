from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
from src.mlproject.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
from src.mlproject.components.model_training import ModelTrainer,ModelTrainingConfig

if __name__ == "__main__":
    try:
        logging.info("Pipeline started")

        # Data ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        # Model training
        model_training = ModelTrainer()
        r2_score = model_training.initiate_model_trainer(train_arr, test_arr)
        print(f"Model training completed successfully. R2 Score: {r2_score}")

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error("Pipeline failed")
        raise CustomException(e)
