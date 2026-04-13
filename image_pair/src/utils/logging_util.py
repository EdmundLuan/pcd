"""Logger utility module."""

"""
Usage:
    logger = LoggingUtils.configure_logger(log_name=__name__)
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
"""


import logging
from termcolor import colored
from logging.handlers import RotatingFileHandler


class ColoredFormatter(logging.Formatter):
    """Custom formatter to color log messages based on level."""
    
    def format(self, record):
        level_colors = {
            'DEBUG'    : 'cyan',
            'INFO'     : 'white',
            'WARNING'  : 'yellow',
            'ERROR'    : 'red',
            'CRITICAL' : 'magenta'
        }
        
        color = level_colors.get(record.levelname)
        if color:
            record.msg = colored(str(record.msg), color)
        return super().format(record)


class LoggingUtils:
    """Logging related helper functions"""
    
    @staticmethod
    def configure_logger(log_name="my_logger"):
        """General Logging Configuration"""
        # Get a named logger; avoid using None to prevent issues with the root logger.
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent messages from propagating to the root logger
        logger.propagate = False
        
        # Add a handler only if one does not exist already
        if not logger.handlers:
            # Console Handler (useful when the application is hosted in environments like Docker)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger


if __name__ == '__main__':
    logger = LoggingUtils.configure_logger()
    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
