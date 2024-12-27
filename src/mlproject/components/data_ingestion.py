from src.mlproject.exception_handling import CustomException
from src.mlproject.logger import logging
import os
import pandas as pd
import numpy as np
import sys

from dataclasses import dataclass

@dataclass

class DataIngestionConfig:
    """
    This class is used to ingest data from the source
    """
    train_data_path:str = os.path.join("dataset_source","train.csv")
    test_data_path:str = os.path.join("dataset_source","test.csv")
    raw_data_path:str = os.path.join("dataset_source","raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # data ingestion from mysql
    def initiate_data_ingestion(self):
        try:
            logging.info("Data ingestion started from mysql")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            
        except Exception as e:
            logging.info("An error occurred")
            raise CustomException(e,sys)  

