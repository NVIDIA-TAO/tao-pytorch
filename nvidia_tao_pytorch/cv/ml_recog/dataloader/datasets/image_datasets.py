# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image Dataset for Metric Learning Recognition model training."""

import os
import logging
from typing import Callable, cast, Dict, List, Optional, Tuple

from torchvision.datasets.folder import (has_file_allowed_extension, find_classes,
                                         default_loader, IMG_EXTENSIONS, Any)
from torchvision.datasets import DatasetFolder


def _make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
        return_empty_class: Optional[bool] = False) -> List[Tuple[str, int]]:
    directory = os.path.expanduser(directory)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file_func(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
        is_valid_file = is_valid_file_func

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)
    # replace empty class error with warning
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        empty_class_report = ""
        for clas in empty_classes:
            empty_class_report += clas + " "
        logging.warning(f"Empty classes detected in {directory}: {empty_class_report}")
    if return_empty_class:
        return instances, empty_classes
    return instances


class MetricLearnImageFolder(DatasetFolder):
    """This class inherits from :class:`torchvision.datasets.DatasetFolder`.

    The functions are similar to `torchvision.datasets.ImageFolder` that it
    creates a dataloader from a classification dataset input, except
    that it allows the existance of empty class folders. Users can also assign
    custom `classes` and `class_to_idx` to the class.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_to_idx=None,
        classes=None,
        class_mapping=None,
    ):
        """Intiates Dataset for image folder input.

        Args:
            root (String): Root directory path.
            transform (Callable, Optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, transforms.RandomCrop
            target_transform (Callable, Optional): A function/transform that takes in the target and transforms it.
            loader (Callable, Optional): A function to load an image given its path.
            is_valid_file (Boolean, Optional): A function that takes path of an Image file and check if the file is a valid file (used to check of corrupt files)
            class_to_idx (Dict[str, int], Optional): Dictionary mapping each class to an index.
            classes (List[str], Optional): List of all classes.
            class_mapping (Dict[str, str], Optional): Dictionary mapping each class to a new class name.
        """
        super(DatasetFolder, self).__init__(root=root,
                                            transform=transform,
                                            target_transform=target_transform)

        self.loader = loader
        self.extensions = IMG_EXTENSIONS if is_valid_file is None else None

        default_classes, default_class_to_idx = self.find_classes(self.root)
        # check class assigned
        if classes:
            class_set = set(classes)
            for clas in default_classes:
                if clas not in class_set:
                    raise ValueError("The image folder classes should be a subset of the assigned classes.")
        if class_to_idx:
            for clas in default_class_to_idx:
                if clas not in class_to_idx:
                    raise ValueError("The image folder classes should be a subset of the assigned classes.")
        else:
            classes = default_classes
            class_to_idx = default_class_to_idx

        samples, empty_classes = self.make_dataset(self.root, class_to_idx,
                                                   self.extensions, is_valid_file)
        self.empty_classes = empty_classes
        self.classes = classes
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples

        if class_mapping:
            # check class mapping dict first:
            for class_name in class_mapping:
                if class_name not in self.class_to_idx:
                    raise ValueError(f"Class {class_name} is not in the dataset.")
            for class_name in self.class_to_idx:
                if class_name not in class_mapping:
                    raise ValueError(f"Class {class_name} is not in the class mapping dict.")
            self.class_dict = {self.class_to_idx[k]: class_mapping[k] for k in self.class_to_idx}
        else:
            self.class_dict = {self.class_to_idx[k]: k for k in self.class_to_idx}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (String): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (Optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (Optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)

        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        return _make_dataset(directory, class_to_idx,
                             extensions=extensions,
                             is_valid_file=is_valid_file,
                             return_empty_class=True)
