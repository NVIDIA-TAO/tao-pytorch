# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Unpickler to avoild unsafe deserialization."""
import pickle
from io import BytesIO


class SafeUnpickler(pickle.Unpickler):
    """
    Custom unpickler that only allows deserialization of a specified class.
    """

    def __init__(self, serialized_data: bytes, class_name: str):
        """
        Initialize the unpickler with the serialized data and the name of the class to allow deserialization for.

        Args:
        serialized_data (bytes): The serialized data to be deserialized.
        class_name (string): The name of the class to be deserialized.
        """
        self.class_name = class_name
        super().__init__(BytesIO(serialized_data))

    def find_class(self, module: str, name: str) -> type:
        """
        Override the default find_class() method to only allow the specified class to be deserialized.

        Args:
        module (string): The module name.
        name (string): The class name.

        Returns:
        type: The specified class.
        """
        # Only allow the specified class to be deserialized
        if name == self.class_name:
            return globals()[name]
        # Raise an exception for all other classes
        raise pickle.UnpicklingError("Invalid class: %s.%s" % (module, name))
