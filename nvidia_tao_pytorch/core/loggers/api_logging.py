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

"""Logger class for TAO Toolkit models."""

import atexit
from datetime import datetime
import json
import logging
import os

from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import status_callback

from torch import distributed as torch_distributed
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

logger = logging.getLogger(__name__)


class Verbosity():
    """Verbosity levels."""

    DISABLE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# Defining a log level to name dictionary.
log_level_to_name = {
    Verbosity.DISABLE: "DISABLE",
    Verbosity.DEBUG: 'DEBUG',
    Verbosity.INFO: 'INFO',
    Verbosity.WARNING: 'WARNING',
    Verbosity.ERROR: 'ERROR',
    Verbosity.CRITICAL: 'CRITICAL'
}


class Status():
    """Status levels."""

    SUCCESS = 0
    FAILURE = 1
    STARTED = 2
    RUNNING = 3
    SKIPPED = 4


status_level_to_name = {
    Status.SUCCESS: 'SUCCESS',
    Status.FAILURE: 'FAILURE',
    Status.STARTED: 'STARTED',
    Status.RUNNING: 'RUNNING',
    Status.SKIPPED: 'SKIPPED'
}


class BaseLogger(object):
    """File logger class."""

    def __init__(self, verbosity=Verbosity.INFO):
        """Base logger class.

        Args:
            verbsority (int): Logging level

        """
        self.verbosity = verbosity
        self.categorical = {}
        self.graphical = {}
        self.kpi = {}

    @property
    def date(self):
        """Get date from the status.

        Returns:
            Formatted string containing mm/dd/yyyy.
        """
        date_time = datetime.now()
        date_object = date_time.date()
        return "{}/{}/{}".format(
            date_object.month,
            date_object.day,
            date_object.year
        )

    @property
    def time(self):
        """Get date from the status.

        Returns:
            Formatted string with time in hh:mm:ss
        """
        date_time = datetime.now()
        time_object = date_time.time()
        return "{}:{}:{}".format(
            time_object.hour,
            time_object.minute,
            time_object.second
        )

    @property
    def categorical(self):
        """Property getter for categorical data to be logged."""
        return self._categorical

    @categorical.setter
    def categorical(self, value: dict):
        """Set categorical data to be logged."""
        self._categorical = value

    @property
    def graphical(self):
        """Property getter for graphical data to be logged."""
        return self._graphical

    @graphical.setter
    def graphical(self, value: dict):
        """Set graphical data to be logged."""
        self._graphical = value

    @property
    def kpi(self):
        """Set KPI data."""
        return self._kpi

    @kpi.setter
    def kpi(self, value: dict):
        """Set KPI data."""
        self._kpi = value

    @rank_zero_only
    def flush(self):
        """Flush the logger."""
        pass

    def format_data(self, data: dict):
        """Format the data.

        Args:
            data(dict): Dictionary data to be formatted to a json string.

        Returns
            data_string (str): Recursively formatted string.
        """
        return json.dumps(data)

    @rank_zero_only
    def log(self, level, string):
        """Log the data string.

        This method is implemented only for rank 0 process in a multiGPU
        session.

        Args:
            level (int): Log level requested.
            string (string): Message to be written.
        """
        # Base class will not flush data to the
        # terminal.
        pass

    @rank_zero_only
    def write(self, data=None,
              status_level=Status.RUNNING,
              verbosity_level=Verbosity.INFO,
              message=None):
        """Write data out to the log file.

        Args:
            data (dict): Dictionary of data to be written out.
            status_level (nvidia_tao_pytorch.core.loggers.api_logging.Status): Current status of the
                process being logged. DEFAULT=Status.RUNNING
            verbosity level (nvidia_tao_pytorch.core.loggers.api_logging.Vebosity): Setting
                logging level of the Status logger. Default=Verbosity.INFO
        """
        if self.verbosity > Verbosity.DISABLE:
            if not data:
                data = {}
            # Define generic data.
            data["date"] = self.date
            data["time"] = self.time
            data["status"] = status_level_to_name.get(status_level, "RUNNING")
            data["verbosity"] = log_level_to_name.get(verbosity_level, "INFO")

            if message:
                data["message"] = message

            if self.categorical:
                data["categorical"] = self.categorical

            if self.graphical:
                data["graphical"] = self.graphical

            if self.kpi:
                data["kpi"] = self.kpi

            data_string = self.format_data(data)
            status_callback(data_string)
            self.log(verbosity_level, data_string)
            self.flush()


class StatusLogger(BaseLogger):
    """Simple logger to save the status file."""

    def __init__(self, filename=None,
                 verbosity=Verbosity.INFO,
                 append=True):
        """Logger to write out the status.

        Args:
            filename (str): Path to the log file.
            verbosity (str): Logging level. Default=INFO
            append (bool): Flag to open the log file in
                append mode or write mode. Default=True
        """
        super().__init__(verbosity=verbosity)
        self.log_path = os.path.realpath(filename)
        if os.path.exists(self.log_path):
            rank_zero_warn(
                f"Log file already exists at {self.log_path}"
            )
        # Open the file only if rank == 0.
        distributed = torch_distributed.is_initialized() and torch_distributed.is_available()
        global_rank_0 = (not distributed) or (distributed and torch_distributed.get_rank() == 0)
        if global_rank_0:
            self.l_file = open(self.log_path, "a" if append else "w")
            atexit.register(self.l_file.close)

    @rank_zero_only
    def log(self, level, string):
        """Log the data string.

        This method is implemented only for rank 0 process in a multiGPU
        session.

        Args:
            level (int): Log level requested.
            string (string): Message to be written.
        """
        if level >= self.verbosity:
            self.l_file.write(string + "\n")

    @rank_zero_only
    def flush(self):
        """Flush contents of the log file."""
        self.l_file.flush()

    @staticmethod
    def format_data(data):
        """Format the dictionary data.

        Args:
            data(dict): Dictionary data to be formatted to a json string.

        Returns
            data_string (str): json formatted string from a dictionary.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary and not type {type(data)}.")
        data_string = json.dumps(data)
        return data_string


# Define the logger here so it's static.
_STATUS_LOGGER = BaseLogger()


def set_status_logger(status_logger):
    """Set the status logger.

    This function also automatically enables dual logging, which forwards
    all standard Python logging calls to the status logger in addition to
    console output.

    Args:
        status_logger: An instance of the logger class.
    """
    global _STATUS_LOGGER  # pylint: disable=W0603
    _STATUS_LOGGER = status_logger

    # Automatically enable dual logging when status logger is set
    try:
        from nvidia_tao_pytorch.core.tlt_logging import enable_dual_logging  # pylint: disable=C0415
        enable_dual_logging()
    except Exception:
        # Silently fail if dual logging cannot be enabled (e.g., in non-pytorch environments)
        pass


def get_status_logger():
    """Get the status logger."""
    global _STATUS_LOGGER  # pylint: disable=W0602,W0603
    return _STATUS_LOGGER
