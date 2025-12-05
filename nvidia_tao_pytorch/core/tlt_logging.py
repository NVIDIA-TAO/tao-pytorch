# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common utilities useful for logging."""

import logging as _logging
import os

from random import randint
from omegaconf import OmegaConf


# Mapping of string log levels to logging constants
LOG_LEVEL_MAPPING = {
    'DEBUG': _logging.DEBUG,
    'INFO': _logging.INFO,
    'WARNING': _logging.WARNING,
    'WARN': _logging.WARNING,
    'ERROR': _logging.ERROR,
    'CRITICAL': _logging.CRITICAL,
    'FATAL': _logging.CRITICAL,
}


def get_logging_level():
    """Get logging level from environment variable.

    Reads TAO_LOGGING_LEVEL environment variable and returns the corresponding
    logging level. Defaults to INFO if not set or invalid.

    Returns:
        int: Python logging level constant (e.g., logging.INFO)
    """
    level_str = os.getenv('TAO_LOGGING_LEVEL', 'INFO').upper()
    level = LOG_LEVEL_MAPPING.get(level_str, _logging.INFO)
    return level


class MessageFormatter(_logging.Formatter):
    """Formatter that supports colored logs."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        _logging.DEBUG: grey + fmt + reset,
        _logging.INFO: grey + fmt + reset,
        _logging.WARNING: yellow + fmt + reset,
        _logging.ERROR: red + fmt + reset,
        _logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        """Format the log message."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = _logging.Formatter(log_fmt)
        return formatter.format(record)


# Get logging level from environment variable
_log_level = get_logging_level()

logger = _logging.getLogger('TAO Toolkit')
logger.setLevel(_log_level)
ch = _logging.StreamHandler()
ch.setLevel(_log_level)
ch.setFormatter(MessageFormatter())
logger.addHandler(ch)
logging = logger


class StatusLoggerHandler(_logging.Handler):
    """Handler that forwards standard logging to StatusLogger.

    This handler bridges the standard Python logging system with the
    TAO StatusLogger, allowing log messages to be written to both
    console and status log files simultaneously.

    The handler fetches the current status logger on each emit, so it
    automatically uses the most recently set StatusLogger instance.
    """

    def __init__(self):
        """Initialize the StatusLoggerHandler."""
        super().__init__()
        self._get_status_logger = None
        self._Verbosity = None
        self._Status = None

    def emit(self, record):
        """Forward log records to the status logger.

        Args:
            record (logging.LogRecord): The log record to emit.
        """
        try:
            # Lazy import to avoid circular dependencies
            if self._get_status_logger is None:
                from nvidia_tao_pytorch.core.loggers.api_logging import (  # pylint: disable=C0415
                    get_status_logger, Verbosity, Status
                )
                self._get_status_logger = get_status_logger
                self._Verbosity = Verbosity
                self._Status = Status

            # Get the current status logger (may change if set_status_logger is called again)
            status_logger = self._get_status_logger()

            # Map Python logging levels to StatusLogger verbosity
            level_map = {
                _logging.DEBUG: self._Verbosity.DEBUG,
                _logging.INFO: self._Verbosity.INFO,
                _logging.WARNING: self._Verbosity.WARNING,
                _logging.ERROR: self._Verbosity.ERROR,
                _logging.CRITICAL: self._Verbosity.CRITICAL
            }

            verbosity_level = level_map.get(record.levelno, self._Verbosity.INFO)

            # Determine status based on log level
            if record.levelno >= _logging.ERROR:
                status_level = self._Status.FAILURE
            else:
                status_level = self._Status.RUNNING

            # Format the message
            message = self.format(record)

            # Write to status logger
            status_logger.write(
                data={},
                status_level=status_level,
                verbosity_level=verbosity_level,
                message=message
            )
        except Exception:
            self.handleError(record)


def enable_dual_logging():
    """Enable logging to both console and status logger.

    This function adds a StatusLoggerHandler to the TAO Toolkit logger,
    which forwards all log messages to the status logger in addition to
    the standard console output.

    Note: This is automatically called by set_status_logger(), so you typically
    don't need to call it manually. It's safe to call multiple times - if the
    handler is already added, this function silently returns.

    Example:
        >>> from nvidia_tao_pytorch.core.loggers.api_logging import StatusLogger, set_status_logger
        >>> from nvidia_tao_pytorch.core.tlt_logging import logging
        >>>
        >>> # Setup status logger (automatically enables dual logging)
        >>> status_logger = StatusLogger(filename="status.json")
        >>> set_status_logger(status_logger)
        >>>
        >>> # Now all logging calls automatically go through both systems
        >>> logging.info("This goes to both console and status file!")
    """
    # Check if handler is already added (safe for multiple calls)
    for handler in logger.handlers:
        if isinstance(handler, StatusLoggerHandler):
            # Handler already exists, silently return
            return

    # Add the handler only if it doesn't exist
    status_handler = StatusLoggerHandler()
    status_handler.setLevel(get_logging_level())
    logger.addHandler(status_handler)


def obfuscate_logs(cfg):
    """Function obfuscates encryption key if exposed/present in args.

    Args:
        cfg(OmegaConf.DictConfig): Function to obfuscate key from the log.
    """
    # First obfuscate what is being shown as configuration.
    config = OmegaConf.to_container(cfg)
    if "encryption_key" in config.keys():
        config["encryption_key"] = '*' * randint(3, 10)

    # Show the experiment configuration.
    logging.info(f'Experiment configuration:\n{OmegaConf.to_yaml(config)}')


def remove_logs(log_dir):
    """Function removes the cmd-args and git-info log files from log_dir.

    Args:
        log_dir(str): Path to the results directory containing the logs.
    """
    log_files = ["cmd-args.log", "git-info.log"]
    for log in log_files:
        logfile = os.path.join(log_dir, log)
        if os.path.exists(logfile):
            os.remove(logfile)
