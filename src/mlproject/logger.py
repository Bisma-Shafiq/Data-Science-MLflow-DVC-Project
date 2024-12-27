import os
import logging
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
# lof file path
log_path = os.path.join(os.getcwd(),"logs", LOG_FILE)
# make sure the directory exists
os.makedirs(log_path,exist_ok=True)
# complete log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    )

