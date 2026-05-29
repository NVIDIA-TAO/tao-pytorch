# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom class for Market1501 dataset."""

import glob
import re
import os.path as osp
from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """Custom class for the Market1501 dataset.

    This class provides an interface to the Market1501 dataset and inherits from the BaseImageDataset class.

    """

    def __init__(self, experiment_spec, prepare_for_training, verbose=False):
        """Initialize the Market1501 dataset.

        Args:
            experiment_spec (dict): Specification of the experiment.
            prepare_for_training (bool): If True, prepare the dataset for training.
            verbose (bool, optional): If True, print verbose information. Defaults to False.

        """
        super(Market1501, self).__init__()
        self.prepare_for_training = prepare_for_training
        if self.prepare_for_training:
            self.train_dir = experiment_spec["dataset"]["train_dataset_dir"]
            self.query_dir = experiment_spec["dataset"]["query_dataset_dir"]
            self.gallery_dir = experiment_spec["dataset"]["test_dataset_dir"]
        elif experiment_spec["inference"]["query_dataset"] and experiment_spec["inference"]["test_dataset"]:
            self.query_dir = experiment_spec["inference"]["query_dataset"]
            self.gallery_dir = experiment_spec["inference"]["test_dataset"]
        elif experiment_spec["evaluate"]["query_dataset"] and experiment_spec["evaluate"]["test_dataset"]:
            self.query_dir = experiment_spec["evaluate"]["query_dataset"]
            self.gallery_dir = experiment_spec["evaluate"]["test_dataset"]
        self._check_before_run()

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if self.prepare_for_training:
            train = self._process_dir(self.train_dir, relabel=True)
            self.print_dataset_statistics(train, query, gallery)
        else:
            self.print_dataset_statistics(query, gallery)
        if self.prepare_for_training:
            self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = None
        if self.prepare_for_training:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if self.prepare_for_training and not osp.exists(self.train_dir):
            raise FileNotFoundError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise FileNotFoundError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise FileNotFoundError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        """Check the directory and return a dataset.

        Args:
            dir_path (str): Path to the directory.
            relabel (bool, optional): If True, relabel the data. Defaults to False.

        Returns:
            list: A list of tuples containing the image path, person ID, and camera ID.

        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'(\d+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501, "The number of person IDs should be between 0 and 1501."
            # assert 1 <= camid <= 6, "The number of camera IDs should be between 0 and 6."
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset
