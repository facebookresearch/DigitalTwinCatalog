# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import functools
import logging
import sys

from .utils import _suppress_print, is_main_process


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename: str, buffering=1024):
    io = open(filename, "a", buffering=buffering)
    atexit.register(io.close)
    return io


def setup_logging(save_path, mode: str = "a", buffering=1024) -> None:
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    if is_main_process():
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        _suppress_print()
        return

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    if is_main_process():
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(print_plain_formatter)
        logger.addHandler(ch)

    if save_path is not None and is_main_process():
        fh = logging.StreamHandler(_cached_log_stream(save_path, buffering=buffering))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def get_default_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    if len(root_logger.handlers) == 0:
        print_plain_formatter = logging.Formatter(
            "[%(asctime)s]: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(print_plain_formatter)
        logger.addHandler(ch)

    return logger
