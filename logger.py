import logging
import json
import os

def setup_logger(name, level=logging.INFO):
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    file_handler = logging.FileHandler(f'logs/{name}.log')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def log_with_data(logger, level, message, data):
    """Log a message with structured data."""
    if isinstance(data, dict):
        logger.log(level, f"{message} - {json.dumps(data, indent=2)}")
    else:
        logger.log(level, f"{message} - {data}") 