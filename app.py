from src.mlproject.logger import logging
from src.mlproject.exception_handling import CustomException
import sys



if __name__ == "__main__":
    logging.info("Hello, world!")

    try:
        a=1/0

    except Exception as e:
        logging.info("An error occurred")
        raise CustomException(e,sys)