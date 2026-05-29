# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point Feature Encoder."""
import numpy as np


class PointFeatureEncoder(object):
    """Point Feature Encoder class."""

    def __init__(self, config, point_cloud_range=None):
        """Initialize."""
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        """Number of point features."""
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        func_map = {key: getattr(PointFeatureEncoder, key) for key in vars(PointFeatureEncoder) if not key.startswith("__") and key.endswith("_encoding")}
        if self.point_encoding_config.encoding_type in func_map:
            data_dict['points'], use_lead_xyz = func_map[self.point_encoding_config.encoding_type](self, data_dict['points'])
            data_dict['use_lead_xyz'] = use_lead_xyz
            return data_dict
        return None

    def absolute_coordinates_encoding(self, points=None):
        """Absolute coordinates encoding."""
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx + 1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True
