import os
import logging
import pandas as pd
import sys
from src.mlproject.exception_handling import CustomException
from src.mlproject.logger import logging
from dotenv import load_dotenv
import pymysql
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

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e,sys)
