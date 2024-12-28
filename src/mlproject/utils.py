import os
import logging
import pandas as pd
import sys
from src.mlproject.exception_handling import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import pymysql

import numpy as np

import pickle
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
database = os.getenv("database")

def sql_data_read():
    logging.info("Data Read started from mysql")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        logging.info("Connection established")
        df = pd.read_sql_query('Select * from car_price',mydb)
        print(df.head())

        return df

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e,sys)


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open (file_path,'wb') as file_obj:
            pickle.dump(obj, file_obj)

            
    except Exception as e:
        raise CustomException (e,sys)
