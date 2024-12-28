from src.mlproject.exception_handling import CustomException
from src.mlproject.logger import logging
import os
import pandas as pd
import numpy as np
import sys
from dataclasses import dataclass
from src.mlproject.utils import sql_data_read
from sklearn.model_selection import train_test_split
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
            # read data from mysql
            df = pd.read_csv(os.path.join('ML_Project/data','raw_data.csv'))

            logging.info("Data ingestion reading Completed from mysql")
            
            # make directory if not exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # convert data to csv
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # train test split
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)        
            
            logging.info("Data ingestion writing Completed")
        
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        except Exception as e:
            logging.info("An error occurred")
            raise CustomException(e,sys)  

