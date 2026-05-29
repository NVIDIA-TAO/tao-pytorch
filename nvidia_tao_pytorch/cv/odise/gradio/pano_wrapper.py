# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import torch
import torch.nn as nn
from detectron2.utils.visualizer import random_color
from detectron2.data import MetadataCatalog

class OpenPanopticInference(nn.Module):
    def __init__(
        self,
        model,
        labels,
        metadata=None,
        semantic_on=True,
        instance_on=True,
        panoptic_on=True,
        test_topk_per_image=100,
    ):
        super().__init__()
        self.model = model
        self.metadata = metadata
        self.labels = labels
        self.model.semantic_on = semantic_on
        self.model.instance_on = instance_on
        self.model.panoptic_on = panoptic_on
        self.model.test_topk_per_image = test_topk_per_image


    def forward(self, batched_inputs, labels=None, metadata=None):
        
        # TODO(@yuw): potentially remove self.x
        if hasattr(self.model, "category_head"):
            self.model.category_head.test_labels = labels
            
            category_overlapping_list = []
            train_labels = {l for label in self.model.train_labels for l in label}
            for test_label in self.model.category_head.test_labels:
                category_overlapping_list.append(not set(train_labels).isdisjoint(set(test_label)))
            self.model.category_overlapping_mask = torch.tensor(
                category_overlapping_list, dtype=torch.long, device=self.model.device
            )
        else:
            raise NotImplementedError

        assert not self.training

        self.model.metadata = metadata
        # Calculate the chunksizes and start index to support a more efficient ensemble_logits_with_labels implementation.
        num_templates = []
        for l in labels:
            num_templates.append(len(l))
        chunk_sizes_pyt = torch.tensor(num_templates, dtype=torch.int32, device='cuda')
        chunk_start_idx = [0,]
        for i in range(len(num_templates) - 1):
            chunk_start_idx.append(chunk_start_idx[i] + num_templates[i])
        chunk_start_idx_pyt = torch.tensor(chunk_start_idx, dtype=torch.int32, device='cuda')

        self.chunk_sizes_pyt = self.model.chunk_sizes_pyt
        self.chunk_start_idx_pyt = self.model.chunk_start_idx_pyt
        self.model.chunk_sizes_pyt = chunk_sizes_pyt
        self.model.chunk_start_idx_pyt = chunk_start_idx_pyt
        self.num_classes = self.model.sem_seg_head.num_classes
        self.model.sem_seg_head.num_classes = len(labels)
    
        results = self.model(batched_inputs)
        return results
