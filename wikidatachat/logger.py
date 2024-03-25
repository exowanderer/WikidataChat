import logging
import sys


def get_logger(name):
    # Create a logger
    # Source: https://docs.python.org/3/howto/logging.html
    logging.basicConfig(
        filename='wdchat_api.log',
        encoding='utf-8',
        level=logging.DEBUG
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Source: stackoverflow.com/questions/14058453/
    #   making-python-loggers-output-all-messages-
    #   to-stdout-in-addition-to-log-file

    # Create console handler and set level to debug
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
