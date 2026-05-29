# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PointPillars model definition."""
from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    """PointPillars model class."""

    def __init__(self, model_cfg, num_class, dataset):
        """Initialize."""
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """Forward."""
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts

    def get_training_loss(self):
        """Get training loss."""
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
