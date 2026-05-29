# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build target class list and palette."""


class TargetClass(object):
    """Target class parameters."""

    def __init__(self, name, label_id, train_id=None, color=None, train_name=None):
        """Constructor target class. Include name, label_id, train_id, color and train_name.

        Args:
            name (str): Name of the target class.
            label_id (str):original label id of every pixel of the mask
            train_id (str): The mapped train id of every pixel in the mask
            color (list): RGB color of the target class.
            train_name (str): The mapped train name of the target class.
        """
        self.name = name
        self.train_id = train_id
        self.label_id = label_id
        self.color = color
        self.train_name = train_name


def build_target_class_list(dataset_config):
    """Build a list of TargetClasses based on palette

    Args:
        dataset_config (dict): The dataset configuration.
        dataset_config["palette"] (list): A list of dictionaries containing the target class information
        dataset_config["palette"][i]["seg_class"] (str): The name of the target class.
        dataset_config["palette"][i]["label_id"] (int): The label id of the target class.
        dataset_config["palette"][i]["mapping_class"] (str): The name of the target class to map to.
        dataset_config["palette"][i]["rgb"] (list): The RGB color of the target class.
    """
    target_classes = []
    orig_class_label_id_map = {}
    color_mapping = {}

    # create every seg_class's label_id and color mapping
    for target_class in dataset_config["palette"]:
        orig_class_label_id_map[target_class["seg_class"]] = target_class["label_id"]
        color_mapping[target_class["seg_class"]] = target_class["rgb"]

    # map the seg_class to mapping_class's label_id
    class_label_id_calibrated_map = orig_class_label_id_map.copy()
    for target_class in dataset_config["palette"]:
        label_name = target_class["seg_class"]
        train_name = target_class["mapping_class"]
        class_label_id_calibrated_map[label_name] = orig_class_label_id_map[train_name]

    for target_class in dataset_config["palette"]:
        target_classes.append(
            TargetClass(
                target_class["seg_class"], label_id=target_class["label_id"],
                # the label_id of the target class is the label_id of the mapping class
                train_id=orig_class_label_id_map[target_class["seg_class"]],
                color=color_mapping[target_class["mapping_class"]],
                train_name=target_class["mapping_class"]
            )
        )
    return target_classes


def build_palette(target_classes):
    """ Build palette, classes and label_map. id_color_map is the one will be used when transfer RGB mask to train_id mask.

    Args:
        target_classes (list): A list of TargetClass objects.
    """
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
