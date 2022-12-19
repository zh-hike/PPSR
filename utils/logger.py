import logging
from paddle import distributed as dist
import os


_logger = None


def log_at_train0(fun):

    def warpper(*args, **kargs):
        if dist.get_rank() == 0:
            fun(*args, **kargs)

    return warpper


@log_at_train0
def init_logger(name="ppsr", logger_file='./output/train.log'):
    _dir = os.path.dirname(logger_file)
    os.makedirs(_dir, exist_ok=True)
    global _logger
    _logger = logging.getLogger(name)
    _logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(logger_file, 'a', encoding="UTF-8")
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

@log_at_train0
def info(fmt):
    _logger.info(fmt)

@log_at_train0
def debug(fmt):
    _logger.debug(fmt)

@log_at_train0
def warning(fmt):
    _logger.warning(fmt)

@log_at_train0
def error(fmt):
    _logger.error(fmt)

def print_config(cfg):
    info(cfg)
