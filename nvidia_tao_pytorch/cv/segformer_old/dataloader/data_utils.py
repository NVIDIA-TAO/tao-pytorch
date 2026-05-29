# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Original source taken from https://github.com/open-mmlab/mmskeleton

# Copyright 2019 OpenMMLAB

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Utils Module."""


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, label_id, train_id=None, color=None, train_name=None):
        """Constructor.

        Args:
            name (str): Name of the target class.
            label_id (str):original label id of every pixel of the mask
            train_id (str): The mapped train id of every pixel in the mask
        Raises:
            ValueError: On invalid input args.
        """
        self.name = name
        self.train_id = train_id
        self.label_id = label_id
        self.color = color
        self.train_name = train_name


def build_target_class_list(dataset_config):
    """Build a list of TargetClasses based on palette"""
    target_classes = []
    orig_class_label_id_map = {}
    color_mapping = {}
    for target_class in dataset_config["palette"]:
        orig_class_label_id_map[target_class["seg_class"]] = target_class["label_id"]
        color_mapping[target_class["seg_class"]] = target_class["rgb"]
    class_label_id_calibrated_map = orig_class_label_id_map.copy()
    for target_class in dataset_config["palette"]:
        label_name = target_class["seg_class"]
        train_name = target_class["mapping_class"]

        class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

    train_ids = sorted(list(set(class_label_id_calibrated_map.values())))
    train_id_calibrated_map = {}
    for idx, tr_id in enumerate(train_ids):
        train_id_calibrated_map[tr_id] = idx

    class_train_id_calibrated_map = {}
    for label_name, train_id in class_label_id_calibrated_map.items():
        class_train_id_calibrated_map[label_name] = train_id_calibrated_map[train_id]

    for target_class in dataset_config["palette"]:
        target_classes.append(
            TargetClass(target_class["seg_class"], label_id=target_class["label_id"],
                        train_id=class_train_id_calibrated_map[target_class["seg_class"]],
                        color=color_mapping[target_class["mapping_class"]],
                        train_name=target_class["mapping_class"]
                        ))

    return target_classes


def build_palette(target_classes):
    """Build palette, classes and label_map."""
    label_map = {}
    classes_color = {}
    id_color_map = {}
    classes = []
    palette = []
    for target_class in target_classes:
        label_map[target_class.label_id] = target_class.train_id
        if target_class.train_name not in classes_color.keys():
            classes_color[target_class.train_id] = (target_class.train_name, target_class.color)
            id_color_map[target_class.train_id] = target_class.color
    keylist = list(classes_color.keys())
    keylist.sort()
    for train_id in keylist:
        classes.append(classes_color[train_id][0])
        palette.append(classes_color[train_id][1])

    return palette, classes, label_map, id_color_map
