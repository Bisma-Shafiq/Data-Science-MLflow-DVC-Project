import sys
from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_training import ModelTrainer

def run_training_pipeline():
    """Execute the complete training pipeline."""
    try:
        logging.info("Starting the training pipeline")
        
        # Data Ingestion
        logging.info("Initiating data ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # Data Transformation
        logging.info("Initiating data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path
        )
        
        # Model Training
        logging.info("Initiating model training")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"Training pipeline completed successfully. Final R2 Score: {r2_score}")
        return r2_score
        
    except Exception as e:
        logging.error("Error in training pipeline")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        sys.exit(1)