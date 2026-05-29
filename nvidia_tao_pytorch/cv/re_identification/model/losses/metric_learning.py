# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric Learning Classes for Re-Identification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math


class ContrastiveLoss(nn.Module):
    """A module for computing Contrastive Loss, which aims to minimize distance
    between similar embeddings and maximize it between dissimilar ones.
    """

    def __init__(self, margin=0.3, **kwargs):
        """Initializes the ContrastiveLoss module.

        Args:
            margin (float, optional): Margin for separating dissimilar pairs. Defaults to 0.3.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        """Computes the Contrastive Loss between input pairs.

        Args:
            inputs (torch.Tensor): Input samples, shape (batch_size, feature_size).
            targets (torch.Tensor): Target labels, shape (batch_size).

        Returns:
            torch.Tensor: Scalar tensor of the loss.
        """
        n = inputs.size(0)
        # Compute similarity matrix
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets == targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets != targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            neg_loss = 0

            pos_loss = torch.sum(-pos_pair_ + 1)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / n
        return loss


class CircleLoss(nn.Module):
    """Circle Loss, a loss function designed to maximize the similarity between samples of the same class and minimize
    it between samples of different classes.
    """

    def __init__(self, in_features, num_classes, s=256, m=0.25):
        """Initializes the CircleLoss module.

        Args:
            in_features (int): Number of input features.
            num_classes (int): Number of target classes.
            s (float, optional): A scale factor. Defaults to 256.
            m (float, optional): A margin parameter. Defaults to 0.25.
        """
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(num_classes, in_features))
        self.s = s
        self.m = m
        self._num_classes = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the module using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):
        """Computes the Circle Loss for given features and targets.

        Args:
            bn_feat (torch.Tensor): Input batch-normalized features, shape (batch_size, in_features).
            targets (torch.Tensor): Target labels, shape (batch_size).

        Returns:
            torch.Tensor: The computed Circle Loss for each input sample.
        """
        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.s * alpha_p * (sim_mat - delta_p)
        s_n = self.s * alpha_n * (sim_mat - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        return pred_class_logits


class Arcface(nn.Module):
    """Implementation of the ArcFace loss function. This loss focuses on maximizing
    the angular margin between the embeddings and their corresponding class centers
    in the hyperspherical space.

    The formula for the ArcFace loss is given by:
        s * cos(theta + m)

    where `theta` is the angle between the feature and the weight vector of the
    correct class, `s` is a scaling factor, and `m` is an additive angular margin.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False, ls_eps=0.0):
        """Initializes the Arcface module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            s (float, optional): Scaling factor. Defaults to 30.0.
            m (float, optional): Angular margin added to the cosine of the angle. Defaults to 0.30.
            easy_margin (bool, optional): Apply the easy margin setting. Defaults to False.
            ls_eps (float, optional): Label smoothing epsilon. It smoothens the one-hot target. Defaults to 0.0.

        Attributes:
            weight (torch.Tensor): Trainable parameters of the layer.
        """
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_tensor, label):
        """Computes the ArcFace loss for the given input_tensor and labels.

        Args:
            input_tensor (torch.Tensor): Input features, shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels, shape (batch_size).

        Returns:
            torch.Tensor: ArcFace loss for each input_tensor sample.
        """
        cosine = F.linear(F.normalize(input_tensor), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.type_as(cosine)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class Cosface(nn.Module):
    """Implementation of the CosFace loss function, a variant of softmax loss which
    focuses on maximizing the cosine similarity between the embedding and the weight
    of the correct class, while introducing a margin in the cosine space.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        """Initializes the CenterLoss module.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            s (float, optional): Scaling factor. Defaults to 30.0.
            m (float, optional): Margin introduced in the cosine space. Defaults to 0.30.

        Attributes:
            weight (torch.Tensor): Trainable parameters of the layer.
        """
        super(Cosface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_tensor, label):
        """Computes the CosFace loss for the given input_tensor and labels.

        Args:
            input_tensor (torch.Tensor): Input features, shape (batch_size, in_features).
            label (torch.Tensor): Ground truth labels, shape (batch_size).

        Returns:
            torch.Tensor: CosFace loss for each input_tensor sample.
        """
        cosine = F.linear(F.normalize(input_tensor.cpu()), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size())
        label = label.cpu()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output

    def __repr__(self):
        """Represents the Cosface module in a string format.

        Returns:
            str: A string representation of the Cosface module detailing its class name and primary parameters.
        """
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', s=' + str(self.s) \
            + ', m=' + str(self.m) + ')'


class AMSoftmax(nn.Module):
    """Angular Margin Softmax (AM-Softmax) loss layer, designed to optimize the
    angular distance between embeddings and class centers in the hypersphere.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.30):
        """
        Initializes the AMSoftmax module.

        Args:
            in_features (int): Size of each input feature.
            out_features (int): Size of each output feature.
            s (float, optional): Scaling factor. Defaults to 30.0.
            m (float, optional): Margin introduced to separate or push the features towards their respective centers. Defaults to 0.30.
        """
        super(AMSoftmax, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_features
        self.W = torch.nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, lb):
        """Forward pass for AMSoftmax.

        Args:
            x (torch.Tensor): Input features, shape (batch_size, in_features).
            lb (torch.Tensor): Ground truth labels, shape (batch_size).

        Returns:
            torch.Tensor: AM-Softmax loss for each input sample.
        """
        assert x.size()[0] == lb.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        lb_view = lb.view(-1, 1)
        delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, self.m)
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        return costh_m_s
