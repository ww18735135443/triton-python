import logging
import logging.handlers


def get_logger(name='root'):
    logging.basicConfig(filename=name, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    handler = logging.StreamHandler()
    filer = logging.handlers.RotatingFileHandler(name, maxBytes=10485760, backupCount=20, encoding="utf-8")
    # handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(filer)
    return logger