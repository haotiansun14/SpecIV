import os
import logging
from logging import FileHandler, Formatter
from datetime import datetime

# Initial log directory
BASE_DIR = f"logs"

def get_base_dir():
    return BASE_DIR

def update_base_dir(new_base_dir):
    global BASE_DIR
    BASE_DIR = f"logs/{new_base_dir}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    os.makedirs(BASE_DIR, exist_ok=True)

def update_log_name(round):
    # Initialize loggers
    logger = logging.getLogger("init")
    logger.setLevel(logging.DEBUG) 

    # Remove old file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, FileHandler):
            logger.removeHandler(handler)
        
    # Create a new file handler with the updated directory
    file_name = f"round_{round}.log"
    new_file_handler = FileHandler(f"{BASE_DIR}/{file_name}")
    new_file_handler.setFormatter(Formatter('%(message)s'))
    logger.addHandler(new_file_handler)

logger = logging.getLogger("init")
logger.setLevel(logging.DEBUG)
