# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""This module provides utility functions to create dataclass fields with customized metadata for various data types.

Each function in this module is designed to simplify the creation of dataclass fields with predefined metadata,
which can be further customized via keyword arguments. This approach facilitates the definition of models or
configurations where the properties of data fields need to be clearly specified, such as in settings for data
validation, serialization, or user interfaces.

Functions:
    STR_FIELD(value, **meta_args) - Returns a dataclass field for a string with customizable metadata.
    INT_FIELD(value, **meta_args) - Returns a dataclass field for an integer with customizable metadata.
    FLOAT_FIELD(value, **meta_args) - Returns a dataclass field for a float with customizable metadata.
    BOOL_FIELD(value, **meta_args) - Returns a dataclass field for a boolean with customizable metadata.
    LIST_FIELD(arrList, **meta_args) - Returns a dataclass field for a list with customizable metadata.
    DICT_FIELD(hashMap, **meta_args) - Returns a dataclass field for a dictionary with customizable metadata.
    DATACLASS_FIELD(hashMap, **meta_args) - Returns a dataclass field for dataclass instances with customizable
        metadata.
    UNION_FIELD(value, union_types, **meta_args) - Returns a dataclass field for Union types with customizable
        metadata.

Each function supports an extensive range of metadata options to define attributes like display name,
description, default values, examples, validation constraints, and dependency relationships among fields.

Usage:
    The module functions can be directly called to create fields in dataclasses, where each field's characteristics
    and behavior are dictated by the provided metadata and the initial value.
"""

from dataclasses import field


def STR_FIELD(value, **meta_args):
    """Create a field with string data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (str): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "string",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "option_weights": None,
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    if metadata["valid_options"] not in (None, ""):
        metadata["value_type"] = "categorical"
    return field(default=value, metadata=metadata)  # noqa pylint: disable=E3701


def INT_FIELD(value, **meta_args):
    """Create a field with integer data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (int): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "int",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "option_weights": None,
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["valid_options"] not in (None, ""):
        metadata["value_type"] = "ordered_int"
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)  # noqa pylint: disable=E3701


def FLOAT_FIELD(value, **meta_args):
    """Create a field with float data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (float): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "float",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)  # noqa pylint: disable=E3701


def BOOL_FIELD(value, **meta_args):
    """Create a field with boolean data type, initializing with default settings that can be overridden by kwargs.

    Args:
        value (bool): Default value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "bool",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value
    return field(default=value, metadata=metadata)  # noqa pylint: disable=E3701


def LIST_FIELD(arrList, **meta_args):
    """Create a field for a list, initializing with default settings that can be overridden by kwargs.

    Args:
        arrList (list): Default list to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value
            (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "list",
        "description": "",
        "default_value": arrList,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: arrList, metadata=metadata)  # noqa pylint: disable=E3701


def SUBSET_LIST_FIELD(arrList, **meta_args):
    """Create a field for subset list types that generates random subsets from valid_options.

    This field type is designed for AutoML scenarios where you want to generate random subsets
    of items from a predefined list of valid options. It's particularly useful for parameters
    like LoRA modules_to_save, target layers, etc.

    Args:
        arrList (list): Default list value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "subset_list",
        "description": "",
        "default_value": arrList,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: arrList, metadata=metadata)  # noqa pylint: disable=E3701


def OPTIONAL_LIST_FIELD(arrList, **meta_args):
    """Create a field for optional list types that generates either None or a list from valid_options.

    This field type generates either None or a list containing items from valid_options.
    Useful for parameters that can be either disabled (None) or enabled with specific values.

    Args:
        arrList (list): Default list value for the field.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "optional_list",
        "description": "",
        "default_value": arrList,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: arrList, metadata=metadata)  # noqa pylint: disable=E3701


def DICT_FIELD(hashMap, **meta_args):
    """Create a field for a dictionary, initializing with default settings that can be overridden by kwargs.

    Args:
        hashMap (dict): Default dictionary to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value
            (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "collection",
        "description": "",
        "default_value": hashMap,
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: hashMap, metadata=metadata)  # noqa pylint: disable=E3701


def DATACLASS_FIELD(hashMap, **meta_args):
    """Create a field representing a dataclass, initializing with default settings that can be overridden by kwargs.

    Args:
        hashMap (any): Default dataclass instance to initialize the field with.
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value
            (default factory if specified).
    """
    metadata = {
        "display_name": "",
        "value_type": "collection",
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "FALSE",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    return field(default_factory=lambda: hashMap, metadata=metadata)  # noqa pylint: disable=E3701


def UNION_FIELD(value, union_types, literal_values=None, **meta_args):
    """Create a field for Union types, initializing with default settings that can be overridden by kwargs.

    Args:
        value (any): Default value for the field.
        union_types (list): List of type names that this Union field can accept (e.g., ['bool', 'string']).
        literal_values (list, optional): List of specific literal string values allowed (for Literal types).
        **meta_args: Arbitrary keyword arguments for additional metadata attributes such as display
            name, description, etc.

    Returns:
        dataclasses.Field: Configured dataclass field with specified metadata and default value.
    """
    metadata = {
        "display_name": "",
        "value_type": "union",
        "union_types": union_types,
        "literal_values": literal_values,
        "description": "",
        "default_value": "",
        "examples": "",
        "valid_min": "",
        "valid_max": "",
        "valid_options": "",
        "required": "",
        "popular": "",
        "regex": "",
        "automl_enabled": "",
        "math_cond": "",
        "parent_param": "",
        "depends_on": "",
    }
    for k, v in meta_args.items():
        metadata[k] = v
    if metadata["default_value"] in (None, "") and value not in (None, ""):
        metadata["default_value"] = value

    # If literal_values are provided, add them to valid_options for validation
    if literal_values and not metadata.get("valid_options"):
        # Combine boolean options with literal values
        bool_options = []
        if "bool" in union_types or "boolean" in union_types:
            bool_options = ["true", "false"]
        all_options = bool_options + literal_values
        metadata["valid_options"] = ",".join(all_options)

    return field(default=value, metadata=metadata)  # noqa pylint: disable=E3701
