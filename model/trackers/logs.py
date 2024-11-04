import logging
import os
import sys
RANK = int(os.getenv('RANK', -1))
LOGGING_NAME ="tracks"
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'
def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support."""
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR  # rank in world for Multi-GPU trainings
    # Create and configure the StreamHandler
    formatter = logging.Formatter('%(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)