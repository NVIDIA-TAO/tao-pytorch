# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data structure containing the nvinfer element of the model."""

import yaml
from abc import abstractmethod
from dataclasses import asdict, dataclass, is_dataclass, field
from copy import deepcopy
from typing import List

VALID_CHANNEL_ORDERS = ["channels_first", "channels_last"]
VALID_BACKENDS = ["onnx"]
VALID_NETWORK_TYPES = [0, 1, 2, 3, 100]
SKIP_LIST = [
    "data_format"
]


def replace_key_characters(dictionary: dict, find_char: str = "-", replace_char: str = "_") -> dict:
    """Recursively replace characters in a key.

    Args:
      dictionary (dict): The dictionary to replace the string.
      find_string (str): The string to search for.
      replace_str (str): The character to replace in the key with.

    Returns:
      dictionary (dict): Return dictionary.
    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    for key, value in zip(keys, values):
        if isinstance(value, dict):
            value = replace_key_characters(value, find_char=find_char, replace_char=replace_char)
        # Handle list of dictionaries.
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    item = replace_key_characters(item, find_char=find_char, replace_char=replace_char)
                    value[value.index(item)] = item
        if find_char in key:
            updated_key = key.replace(find_char, replace_char)
            dictionary[updated_key] = deepcopy(dictionary[key])
            del dictionary[key]
    return dictionary


def recursively_join_list(dictionary: dict) -> dict:
    """Recursively traverse a dictionary and convert list to strings.

    Args:
      dictionary (dict): The input dictionary to edit.

    Returns:
      dictionary (dict): Return dictionary.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            value = recursively_join_list(value)
        if isinstance(value, list):
            updated_value = []
            if all(isinstance(item, dict) for item in value):
                for item in value:
                    if isinstance(item, dict):
                        item = recursively_join_list(item)
                    else:
                        pass
                    updated_value.append(item)
                value = updated_value
            else:
                value = ";".join([str(item) for item in value])
        dictionary[key] = value
    return dictionary


def remove_null_keys(dictionary: dict) -> dict:
    """Recursively remove null keys from the dictionary.

    Args:
      dictionary (dict): The dictionary to edit.

    Returns:
      dictionary (dict): Return dictionary.
    """
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    for key, value in zip(keys, values):
        if isinstance(value, dict):
            value = remove_null_keys(value)
        if value is None or key in SKIP_LIST:
            del dictionary[key]
    return dictionary


@dataclass
class BaseDSType:
    """Base class for the DeepStream metadata."""

    def as_dict(self) -> dict:
        """Write a member function to serialize this as a dictionary."""
        config_dictionary = asdict(self)
        config_dictionary = remove_null_keys(config_dictionary)
        return config_dictionary

    @abstractmethod
    def validate(self):
        """Function to validate the config element."""
        raise NotImplementedError("Base class doesn't implement this method.")

    def __str__(self) -> str:
        """String representation of the data structure."""
        self.validate()
        config_dictionary = recursively_join_list(
            self.as_dict()
        )
        config_dictionary = replace_key_characters(config_dictionary, find_char="_", replace_char="-")
        config_keys = list(config_dictionary.keys())
        if "property-field" in config_keys:
            config_dictionary["property"] = deepcopy(config_dictionary["property-field"])
            del config_dictionary["property-field"]
        return_string = yaml.safe_dump(config_dictionary)
        return return_string


@dataclass
class BaseNvDSPropertyConfig(BaseDSType):
    """Configuration element for an nvinfer ds plugin.

    This base class defines a data structure to encapsulate all model
    specific parameters defined as part of the [property] field in the
    DeepStream GST-Nvinfer config element. For more details about this parameter,
    refer to: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html

    Args:
        net_scale_factor: float = Scale factor to be applied to the input buffer to the model.
        offsets: List = The pixel-wise mean to be subtracted from each pixel in the input buffer.
            Note: The channel order depends on the model-color-format parameter.
        labelfile_path: str = Path to the labels file.
        onnx_file: str = Path to the onnx model file.
        cluster_mode: int = Clustering algorithm to use.
            Refer to the next table for configuring the algorithm specific parameters.
            Refer Clustering algorithms supported by nvinfer for more information
                0: OpenCV groupRectangles()
                1: DBSCAN
                2: Non Maximum Suppression
                3: DBSCAN + NMS Hybrid
                4: No clustering
        workspace_size: int = Size of the workspace for the TensorRT backend.
        gie_unique_id: int = Unique ID for the GIE instance.
        key: str = The encryption key to the .tlt model.
        network_type: int = Type of the network model. Classification, Detection, Segmentation or Embedding.
            0: Detector
            1: Classifier
            2: Segmentation
            3: Instance Segmentation
        network_mode: int = Data format to be used by inference
            0: FP32
            1: INT8
            2: FP16
            3: BEST
        maintain_aspect_ratio: int = Flag to enable image resize with aspect ratio maintained.
        output_tensor_meta: int = Gst-nvinfer attaches raw tensor output as Gst Buffer metadata.
        num_detected_classes: int = Number of classes to be detected by the network.
        model_color_format: str = Model format in "RGB", "BGR" or "L".
        data_format: str = Channel index of the input. ["channels_first", "channels_last"]
        infer_dims: List = List field containing the input dimensions.
            field(default_factory=lambda: [3, 544, 960])
    """

    net_scale_factor: float = 1
    offsets: List = field(default_factory=lambda: [123.675, 116.28, 103.53])
    labelfile_path: str = None
    onnx_file: str = None
    cluster_mode: int = 2
    workspace_size: int = 1048576
    gie_unique_id: int = 1
    tlt_model_key: str = None
    network_type: int = 100
    maintain_aspect_ratio: int = 1
    output_tensor_meta: int = 1
    output_blob_names: List = field(default_factory=lambda: ["pred_boxes", "pred_logits"])
    num_detected_classes: int = 19
    network_mode: int = 0
    model_color_format: int = 1
    data_format: str = "channels_first"
    infer_dims: List = field(default_factory=lambda: [3, 544, 960])

    def validate(self):
        """Validate base config file properties for the DeepStream graph."""
        assert self.network_type in VALID_NETWORK_TYPES, (
            f"Invalid Network type {self.network_type} requested. Supported network types: {VALID_NETWORK_TYPES}"
        )

        if self.data_format not in VALID_CHANNEL_ORDERS:
            raise NotImplementedError(
                f"Invalid data format {self.data_format} encountered."
                f"Valid data formats: {VALID_CHANNEL_ORDERS}"
            )
        channel_index = 0
        if self.data_format == "channels_last":
            channel_index = -1
        if self.infer_dims[channel_index] == 1:
            assert self.model_color_format == 2, "Model format should be 2"
        else:
            assert self.infer_dims[channel_index] == 3, (
                "Channel count mismatched with color_format. "
                f"Provided\ndata_format: {self.infer_dims[channel_index]}\n color_format: {self.model_color_format}"
            )
            assert len(self.offsets) == 3, "Offsets must be 3 channel input."
            assert self.model_color_format in [0, 1], "Model format should be `0` or `1`"


@dataclass
class BaseNVDSClassAttributes(BaseDSType):
    """Base class defining class-wise attributes in the GST-NVINFER config.

    Args:
        pre-cluster-threshold: int = The threshold to be applied to raw predictions.
    """

    pre_cluster_threshold: float = 0.5

    def validate(self):
        """Validate method for the config element."""
        pass


if __name__ == "__main__":
    """Wrapper to run the base tests."""

    base_config = BaseNvDSPropertyConfig()
    assert is_dataclass(base_config), "The instance of base_config is not a dataclass."
    print(str(base_config))
