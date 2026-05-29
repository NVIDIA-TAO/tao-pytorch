# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data source config class for DriveNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os


class DataSourceConfig(object):
    """Hold all data source related parameters."""

    def __init__(self,
                 dataset_type,
                 dataset_files,
                 image_dir,
                 minimum_target_class_imbalance=None,
                 num_duplicates=0,
                 skip_empty_frames=False,
                 ignored_classifiers_for_skip=None,
                 additional_conditions=None):
        """Constructor.

        Args:
            dataset_type (string): Currently only 'tfrecord' and 'sqlite' are supported.
            dataset_files (list): A list of absolute paths to dataset files. In case of
                tfrecords, a list of absolute paths to .tfrecord files.
            image_dir (string): Absolute path to images directory.
            minimum_target_class_imbalance (map<string, float>): Minimum ratio
                (#dominant_class_instances/#target_class_instances) criteria for duplication
                of frames. The string is the non-dominant class name and the float is the
                ratio for duplication.
            num_duplicates (int): Number of duplicates of frames to be added, if the frame
                satifies the minimum_target_class_imbalance.
            skip_empty_frames (bool): Whether to ignore empty frames (i.e frames without relevant
                features. By default, False, i.e all frames are returned.
            ignored_classifiers_for_skip (set): Names of classifiers to ignore when
                considering if frame is empty. I.e if frame only has these classes, it is still
                regarded as empty.
            additional_conditions (list): List of additional sql conditions for a 'where' clause.
                It's only for SqliteDataSource, and other data sources will ignore it.
        """
        self.dataset_type = dataset_type
        self.dataset_files = dataset_files
        self.image_dir = image_dir
        self.minimum_target_class_imbalance = minimum_target_class_imbalance
        self.num_duplicates = num_duplicates
        self.skip_empty_frames = skip_empty_frames
        self.ignored_classifiers_for_skip = ignored_classifiers_for_skip
        self.additional_conditions = additional_conditions


def build_data_source_lists(data_sources):  # Non-sharding data
    """Build training and validation data source lists from proto.

    Args:
        dataset_config
    Returns:
        training_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for training.
        validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for validation. Can be None.
    """
    # Prepare data source
    data_source_list = []

    if type(data_sources).__name__ == "DictConfig":
        data_sources = [data_sources]

    for data_source in data_sources:
        image_dir = data_source["image_dir"]
        _files = data_source["json_file"]
        extension = os.path.splitext(os.path.basename(_files))[1]

        if extension == '.json':  # use specific json file provided in spec file
            json_files = [_files]
        elif extension == "":  # grab all json file under the directory
            json_files = glob.glob(_files)
        else:
            raise NotImplementedError("Need to provide json_file in dataset_config with the format of either '/path/to/json_file/.json' or '/path/to/json_files/*' ")

        data_source_list.append(DataSourceConfig(
            dataset_type='json',
            dataset_files=json_files,
            image_dir=image_dir))

    return data_source_list


def build_data_source_lists_per_gpu(data_sources, global_rank, num_gpus):  # Sharded data
    """Build training and validation data source lists from proto.

    Args:
        data_sources
    Returns:
        training_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for training.
        validation_data_source_list (list): List of tuples (tfrecord_file_pattern,
            image_directory_path) to use for validation. Can be None.
    """
    # Prepare training and validation data source
    data_source_list = []

    if type(data_sources).__name__ == "DictConfig":
        data_sources = [data_sources]

    for data_source in data_sources:
        image_dir = data_source["image_dir"]
        _files = data_source["json_file"]
        extension = os.path.splitext(os.path.basename(_files))[1]

        if extension == '.json':  # use specific json file provided
            json_files = [_files]
        elif extension == "":  # grab all json file under the directory
            json_files = glob.glob(_files)
        else:
            raise NotImplementedError("Need to provide json_file in dataset_config with the format of either '/path/to/json_file/.json' or '/path/to/json_files/*' ")

        training_jsons_per_seq = []
        for shard_idx, json_file in enumerate(json_files):
            if (shard_idx % num_gpus) == global_rank:
                training_jsons_per_seq.append(json_file)

        data_source_list.append(DataSourceConfig(
            dataset_type='json',
            dataset_files=training_jsons_per_seq,
            image_dir=image_dir))

    return data_source_list
