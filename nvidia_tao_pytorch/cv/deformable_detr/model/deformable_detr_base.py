# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Deformable DETR model. """

import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import warnings

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import (tensor_from_tensor_list, inverse_sigmoid)


def _get_clones(module, N):
    """Get clones of nn.Module.

    Args:
        module (nn.Module): torch module to clone.
        N (int): number of times to clone.

    Returns:
        nn.ModuleList of the cloned module.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    """Simple multi-layer perceptron (FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        """FFN Initialization.

        Args:
            input_dim (int): input dimension.
            hidden_dim (int): hidden dimension.
            output_dim (int): output dimension.
            num_layers (int): number of layers.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        """Forward function."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, position_embedding, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=True, export=False):
        """ Initializes the D-DETR model.

        Args:
            backbone (nn.Module): torch module of the backbone to be used. See backbone.py
            transformer (nn.Module): torch module of the transformer architecture. See transformer.py
            num_classes (int): number of object classes
            num_queries (int): number of object queries, ie detection slot. This is the maximal number of objects
                DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss (bool): True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine (bool): iterative bounding box refinement.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels

        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.position_embedding = position_embedding

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.export = export
        if self.export:
            warnings.warn("Setting aux_loss to be False for export")
            self.aux_loss = False

        self.with_box_refine = with_box_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

    def forward(self, samples):
        """ Forward function of DD Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]

        Returns:
            pred_logits (torch.Tensor): the classification logits (including no-object) for all queries.
                Shape = [batch_size x num_queries x (num_classes + 1)]
            pred_boxes (torch.Tensor): the normalized boxes coordinates for all queries, represented as (center_x, center_y, height, width).
        """
        if not isinstance(samples, torch.Tensor):
            samples = tensor_from_tensor_list(samples)

        features = self.backbone(samples)

        srcs = []
        masks = []
        for level, feat in enumerate(features):
            src = feat[0]
            mask = (feat[1].float()[:, 0].bool())
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for li in range(_len_srcs, self.num_feature_levels):
                if li == _len_srcs:
                    src = self.input_proj[li](features[-1][0])
                else:
                    src = self.input_proj[li](srcs[-1])
                srcs.append(src)

                if self.export:
                    m = torch.zeros((src.shape[0], 1, src.shape[2], src.shape[3]), dtype=src.dtype, device=src.device)
                else:
                    m = samples[:, 3:4]
                mask = F.interpolate(m.float(), size=src.shape[-2:]).to(torch.bool)
                masks.append(mask.float()[:, 0].bool())

        # build positional embedding
        pos = []
        for mask in masks:
            if self.export:
                N, H, W = mask.shape
                tensor_shape = torch.empty(N, H, W, device=src.device)
                pos.append(self.position_embedding(tensor_shape, src.device))
            else:
                not_mask = ~mask
                pos.append(self.position_embedding(not_mask, src.device))

        query_embeds = self.query_embed.weight
        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):

            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """A workaround as torchscript doesn't support dictionary with non-homogeneous values."""
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
