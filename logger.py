import logging
import json
import os

class LogColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            record.msg = f"{LogColors.OKGREEN}{record.msg}{LogColors.ENDC}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{LogColors.WARNING}{record.msg}{LogColors.ENDC}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{LogColors.FAIL}{record.msg}{LogColors.ENDC}"
        return super().format(record)

def setup_logger(name, level=logging.INFO):
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = ColoredFormatter(
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