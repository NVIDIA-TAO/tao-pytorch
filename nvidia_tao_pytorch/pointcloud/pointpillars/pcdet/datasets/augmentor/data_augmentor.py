# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data Augmentor."""
from functools import partial

import numpy as np
import omegaconf
from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    """Data Augmentor class."""

    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        """Initialize."""
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, (list, omegaconf.listconfig.ListConfig)) \
            else augmentor_configs.aug_config_list

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, (list, omegaconf.listconfig.ListConfig)):
                if cur_cfg.name in augmentor_configs.disable_aug_list:
                    continue
            # cur_augmentor = getattr(self, cur_cfg.name)(config=cur_cfg)
            func_map = {key: getattr(DataAugmentor, key) for key in vars(DataAugmentor) if not key.startswith("__")}
            if cur_cfg.name in func_map:
                cur_augmentor = func_map[cur_cfg.name](self, config=cur_cfg)
                self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        """Groundtruth sampling."""
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        """Get state."""
        d = dict(self.__dict__)
        if "logger" in d:
            del d['logger']
        return d

    def __setstate__(self, d):
        """Set state."""
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        """Random world flip."""
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['along_axis_list']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        """Random world rotation."""
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['world_rot_angle']
        if not isinstance(rot_range, (list, omegaconf.listconfig.ListConfig)):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        """Random world scaling."""
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['world_scale_range']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
