import logging
import logging.handlers

def get_logger(name=__file__):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("./train.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)-8s] %(asctime)s| %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

