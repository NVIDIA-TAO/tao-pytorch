# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PointPillars export APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import logging

import h5py
import numpy as np


"""Logger for data export APIs."""
logger = logging.getLogger(__name__)


class TensorFile(io.RawIOBase):
    """Class to read/write multiple tensors to a file.

    The underlying implementation using an HDF5 database
    to store data.

    Note: this class does not support multiple writers to
    the same file.

    Args:
        filename (str): path to file.
        mode (str): mode to open file in.
            r   Readonly, file must exist
            r+  Read/write, file must exist
            w   Create file, truncate if exists
            w-  Create file, fail if exists
            a   Read/write if exists, create otherwise (default)
        enforce_same_shape (bool): whether to enforce that all tensors be the same shape.
    """

    DEFAULT_ARRAY_KEY = "_tensorfile_array_key_"
    GROUP_NAME_PREFIX = "_tensorfile_array_key_"

    def __init__(
        self, filename, mode="a", enforce_same_shape=True, *args, **kwargs
    ):  # pylint: disable=W1113
        """Init routine."""
        super(TensorFile, self).__init__(*args, **kwargs)

        logger.debug("Opening %s with mode=%s", filename, mode)

        self._enforce_same_shape = enforce_same_shape
        self._mode = mode

        # Open or create the HDF5 file.
        self._db = h5py.File(filename, mode)

        if "count" not in self._db.attrs:
            self._db.attrs["count"] = 0

        if "r" in mode:
            self._cursor = 0
        else:
            self._cursor = self._db.attrs["count"]

    def _get_group_name(cls, cursor):
        """Return the name of the H5 dataset to create, given a cursor index."""
        return "%s_%d" % (cls.GROUP_NAME_PREFIX, cursor)

    def _write_data(self, group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                self._write_data(group.create_group(key), value)
            elif isinstance(value, np.ndarray):
                if self._enforce_same_shape:
                    if "shape" not in self._db.attrs:
                        self._db.attrs["shape"] = value.shape
                    else:
                        expected_shape = tuple(self._db.attrs["shape"].tolist())
                        if expected_shape != value.shape:
                            raise ValueError(
                                "Shape mismatch: %s v.s. %s"
                                % (str(expected_shape), str(value.shape))
                            )
                group.create_dataset(key, data=value, compression="gzip")
            else:
                raise ValueError(
                    "Only np.ndarray or dicts can be written into a TensorFile."
                )

    def close(self):
        """Close this file."""
        self._db.close()

    # For python2.
    def next(self):
        """Return next element."""
        return self.__next__()

    # For python3.
    def __next__(self):
        """Return next element."""
        if self._cursor < self._db.attrs["count"]:
            return self.read()
        raise StopIteration()

    def _read_data(self, group):
        if isinstance(group, h5py.Group):
            data = {key: self._read_data(value) for key, value in group.items()}
        else:
            data = group[()]
        return data

    def read(self):
        """Read from current cursor.

        Return array assigned to current cursor, or ``None`` to indicate
        the end of the file.
        """
        if not self.readable():
            raise IOError("Instance is not readable.")

        group_name = self._get_group_name(self._cursor)

        if group_name in self._db:
            self._cursor += 1
            group = self._db[group_name]
            data = self._read_data(group)
            if list(data.keys()) == [self.DEFAULT_ARRAY_KEY]:
                # The only key in this group is the default key.
                # Return the numpy array directly.
                return data[self.DEFAULT_ARRAY_KEY]
            return data
        return None

    def readable(self):
        """Return whether this instance is readable."""
        return self._mode in ["r", "r+", "a"]

    def seekable(self):
        """Return whether this instance is seekable."""
        return True

    def seek(self, n):
        """Move cursor."""
        self._cursor = min(n, self._db.attrs["count"])
        return self._cursor

    def tell(self):
        """Return current cursor index."""
        return self._cursor

    def truncate(self, n):
        """Truncation is not supported."""
        raise IOError("Truncate operation is not supported.")

    def writable(self):
        """Return whether this instance is writable."""
        return self._mode in ["r+", "w", "w-", "a"]

    def write(self, data):
        """Write a Numpy array or a dictionary of numpy arrays into file."""
        if not self.writable():
            raise IOError("Instance is not writable.")

        if isinstance(data, np.ndarray):
            data = {self.DEFAULT_ARRAY_KEY: data}

        group_name = self._get_group_name(self._cursor)

        # Delete existing instance of datasets at this cursor position.
        if group_name in self._db:
            del self._db[group_name]

        group = self._db.create_group(group_name)

        self._write_data(group, data)

        self._cursor += 1

        if self._cursor > self._db.attrs["count"]:
            self._db.attrs["count"] = self._cursor
