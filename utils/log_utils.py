import logging
import os
import sys
import typing

import yaml

# -------- log setting ---------
DEFAULT_LOGGER = "easyllm.logger"


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


def get_logger(name, level="INFO", handlers=None, update=False):
    if name in _LOGGER_CACHE and not update:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = handlers or _DEFAULT_HANDLERS
    logger.propagate = False
    return logger


def save_yaml_config(save_dir, config):
    """A function that saves a dict of config to yaml format file.

    Args:
        save_dir (str): the path to save config file.
        config (dict): the target config object.
    """
    prt_dir = os.path.dirname(save_dir)

    from collections import OrderedDict
    # add yaml representer for different type
    yaml.add_representer(
        OrderedDict,
        lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items())
    )

    if prt_dir != '' and not os.path.exists(prt_dir):
        os.makedirs(prt_dir)

    with open(save_dir, 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    return


def load_yaml_config(config_dir):
    """ Load yaml config file from disk.

    Args:
        config_dir: str or Path
            The path of the config file.

    Returns:
        Config: dict.
    """
    with open(config_dir) as config_file:
        # load configs
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    return config


# -------------------------- Singleton Object --------------------------
default_logger = get_logger(DEFAULT_LOGGER)
