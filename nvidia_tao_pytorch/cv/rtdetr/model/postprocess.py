# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Post Process """

import torch

from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops
from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import PostProcess

from nvidia_tao_pytorch.cv.rtdetr.dataloader.od_dataset import mscoco_label2category


class RTDETRPostProcess(PostProcess):
    """This module converts the model's output into the format expected by the coco api."""

    def __init__(self, num_select=100, remap_mscoco_category=False) -> None:
        """PostProcess constructor.

        Args:
            num_select (int): top K predictions to select from
        """
        super().__init__()
        self.num_select = num_select
        self.remap_mscoco_category = remap_mscoco_category

    @torch.no_grad()
    def forward(self, outputs, target_sizes, image_names):
        """ Perform the post-processing. Scale back the boxes to the original size.

        Args:
            outputs (dict[torch.Tensor]): raw outputs of the model
            target_sizes (torch.Tensor): tensor of dimension [batch_size x 2] containing the size of each images of the batch.
                For evaluation, this must be the original image size (before any data augmentation).
                For visualization, this should be the image size after data augment, but before padding.

        Returns:
            results (List[dict]): final predictions compatible with COCOEval format.
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="floor")
        labels = topk_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        if self.remap_mscoco_category:
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        # from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b, 'image_names': n, 'image_size': i}
                   for s, l, b, n, i in zip(scores, labels, boxes, image_names, target_sizes)]

        return results
