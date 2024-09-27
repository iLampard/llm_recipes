import logging
from pathlib import Path
import os
import sys
from datetime import datetime
from transformers import logging as transformers_logging
import torch.distributed as dist

# -------- log setting ---------
DEFAULT_LOGGER = "easyllm_kit.logger"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s - %(filename)s[pid:%(process)d;line:%(lineno)d:%(funcName)s]' \
             ' - %(levelname)s: %(message)s'
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


DEFAULT_FORMATTER = CustomFormatter()
_ch = logging.StreamHandler(stream=sys.stdout)
_ch.setFormatter(DEFAULT_FORMATTER)
_DEFAULT_HANDLERS = [_ch]
_LOGGER_CACHE = {}  # type: typing.Dict[str, logging.Logger]


class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Prevent logging empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def get_logger(name=DEFAULT_LOGGER, level="INFO", update=False, log_dir=None):
    # Check if we're in a distributed environment and not the main process
    if dist.is_initialized() and dist.get_rank() != 0:
        # For non-main processes, return a logger that doesn't do anything
        return logging.getLogger('dummy')

    if not log_dir:
        # Get the current file's path
        current_file_path = Path(__file__)

        # Create a directory for logs if it doesn't exist
        log_dir = os.path.join(current_file_path.parent, 'logs')
        os.makedirs(log_dir, exist_ok=True)

    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler that saves to "{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_name = datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    file_handler.setFormatter(DEFAULT_FORMATTER)

    # Clear existing handlers and set new handlers
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(_ch)  # Keep the console handler

    logger.propagate = False
    _LOGGER_CACHE[name] = logger  # Cache the logger for future use

    transformers_logger = transformers_logging.get_logger()
    transformers_logger.handlers = []

    # Set up the file handler for transformers logger
    transformers_logger.addHandler(_DEFAULT_HANDLERS[0])  # Adding console handler
    transformers_logger.addHandler(file_handler)  # Adding file handler
    transformers_logger.setLevel(logging.DEBUG)

    # Now, use your custom logger to log messages from your transformers operations
    logger.info("Custom logging is set up for the Transformers library.")

    # Redirect stdout and stderr to the logger
    sys.stdout = StreamToLogger(logger, level=logging.INFO)

    # Handle unhandled exceptions
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = exception_handler

    return logger
