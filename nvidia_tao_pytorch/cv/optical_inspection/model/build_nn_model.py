# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The top model builder interface."""

import torch.nn as nn
import torch
from torch.nn import functional as F
from torchmetrics import Metric

from nvidia_tao_pytorch.core.tlt_logging import logging


class SiameseNetwork3(nn.Module):
    """Siamese Network model for finding defects."""

    def __init__(self, embedding_vectorsize=None,
                 num_lights=4, output_shape=[100, 100]):
        """Initialize the SiameseNetwork3 model.

        Args:
            embedding_vectorsize (int): The size of the embedding vector.
            num_lights (int): The number of lighting conditions.
            output_shape (list): The output shape of the model [height, width].
        """
        super(SiameseNetwork3, self).__init__()
        self.embedding = embedding_vectorsize

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        fc_ip_dim = 8 * num_lights * int(output_shape[0]) * int(output_shape[1])
        self.fc1 = nn.Sequential(
            nn.Linear(fc_ip_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, self.embedding))

    def forward_once(self, x):
        """Forward pass using one image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        """Forward pass using two images.

        Args:
            input1 (torch.Tensor): The first input image.
            input2 (torch.Tensor): The second input image.

        Returns:
            tuple: Tuple of output tensors from the forward pass.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class SiameseNetwork1(nn.Module):
    """Siamese Network model for finding defects."""

    def __init__(self, embedding_vectorsize=None,
                 num_lights=4, output_shape=[100, 100]):
        """Initialize the SiameseNetwork1 model.

        Args:
            embedding_vectorsize (int): The size of the embedding vector.
            num_lights (int): The number of lighting conditions.
            output_shape (list): The output shape of the model [height, width].
        """
        super(SiameseNetwork1, self).__init__()
        self.embedding = embedding_vectorsize

        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        fc_ip_dim = 8 * num_lights * int(output_shape[0]) * int(output_shape[1])
        self.fc1 = nn.Sequential(
            nn.Linear(fc_ip_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, self.embedding))

    def forward_once(self, x):
        """Forward pass using one image.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The output of the forward pass.
        """
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        """Forward pass using two images.

        Args:
            input1 (torch.Tensor): The first input image.
            input2 (torch.Tensor): The second input image.

        Returns:
            tuple: Tuple of output tensors from the forward pass.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss1(torch.nn.Module):
    """Contrastive Loss for comparing image embeddings.

    Args:
        margin (float): The margin used for contrastive loss.
    """

    def __init__(self, margin=2.0):
        """Initialize"""
        super(ContrastiveLoss1, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Compute the contrastive loss.

        Args:
            output1 (torch.Tensor): Embedding vector of the first image.
            output2 (torch.Tensor): Embedding vector of the second image.
            label (torch.Tensor): Label indicating if the images are similar or dissimilar.

        Returns:
            torch.Tensor: Contrastive loss value.
        """
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class AOIMetrics(Metric):
    """AOI Metrics"""

    def __init__(self, margin=2.0):
        """Intialize metrics"""
        super().__init__()
        self.add_state("match_fail", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tot_fail", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("match_pass", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("tot_pass", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mismatch_fail", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("mismatch_pass", default=torch.tensor(0), dist_reduce_fx="sum")
        self.margin = margin

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update the metrics based on the predictions and targets.

        Args:
            preds (torch.Tensor): Predicted distances.
            target (torch.Tensor): Target labels.
        """
        preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape
        # self.correct += torch.sum(preds == target)
        # self.total += target.numel()

        for k, euc_dist in enumerate(preds, 0):
            if euc_dist > float(self.margin):
                # Model Classified as FAIL
                # totmodelfail += 1
                if target.data[k].item() == 1:
                    self.match_fail += 1
                    self.tot_fail += 1
                else:
                    self.mismatch_pass += 1
                    self.tot_pass += 1
            else:
                # totmodelpass += 1
                # Model Classified as PASS
                if target.data[k].item() == 0:
                    self.match_pass += 1
                    self.tot_pass += 1
                else:
                    self.mismatch_fail += 1
                    self.tot_fail += 1

    def compute(self):
        """Compute the metrics.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        metric_collect = {}
        metric_collect['total_accuracy'] = ((self.match_pass + self.match_fail) / (self.tot_pass + self.tot_fail)) * 100
        metric_collect['defect_accuracy'] = torch.Tensor([0]) if self.tot_fail == 0 else (self.match_fail / self.tot_fail) * 100
        metric_collect['false_alarm'] = (self.mismatch_pass / (self.tot_pass + self.tot_fail)) * 100
        metric_collect['false_negative'] = (self.mismatch_fail / (self.tot_pass + self.tot_fail)) * 100

        return metric_collect

    def _input_format(self, preds, target):
        return preds, target


def cal_model_accuracy(euclidean_distance, label, match_cnts, total_cnts, margin):
    """Calculate Siamese model accuracy"""
    for j in range(euclidean_distance.size()[0]):
        if ((euclidean_distance.data[j].item() < margin)):
            if label.data[j].item() == 0:
                match_cnts += 1
                total_cnts += 1
            else:
                total_cnts += 1
        else:
            if label.data[j].item() == 1:
                match_cnts += 1
                total_cnts += 1
            else:
                total_cnts += 1

    return (match_cnts / total_cnts) * 100, match_cnts, total_cnts


def build_oi_model(experiment_config, imagenet_pretrained=True,
                   export=False):
    """Select and build the Siamese model based on the experiment configuration.

    Args:
        experiment_config (OmegaConf.DictConf): The experiment configuration.
        imagenet_pretrained (bool): Flag indicating whether to use ImageNet pre-trained weights. # TODO: @pgurumurthy to add support
        export (bool): Flag indicating whether to export the model.

    Returns:
        torch.nn.Module: The built Siamese model.
    """
    model_config = experiment_config["model"]
    embedding_vectorsize = model_config["embedding_vectors"]
    model_type = model_config["model_type"]
    image_width = experiment_config["dataset"]["image_width"]
    image_height = experiment_config["dataset"]["image_height"]
    num_lights = experiment_config["dataset"]["num_input"]
    model_backbone = model_config["model_backbone"]
    if model_backbone == "custom":
        logging.info("Starting training with custom backbone")
        if model_type == 'Siamese_3':
            model = SiameseNetwork3(embedding_vectorsize, num_lights, [image_height, image_width]).cuda()
        else:
            model = SiameseNetwork1(embedding_vectorsize, num_lights, [image_height, image_width]).cuda()
    # TODO: @pgurumurthy to add resnet/efficientnet support.
    # elif model_backbone == "resnet":
    #     print("Starting Siamese with ResNet backbone")
    #     # PlaceHolder for adding ResNet backbone####
    # elif model_backbone == "efficientet":
    #     print("Starting Siamese with EfficientNet backbone")
    #     # PlaceHolder for adding EfficientNet backbone####
    else:
        raise NotImplementedError(f"Invalid model backbone requested.: {model_backbone}")

    return model
