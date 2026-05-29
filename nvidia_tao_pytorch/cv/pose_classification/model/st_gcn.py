# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ST-GCN model architecture for pose classification."""
import torch
from torch import nn
from torch.nn import functional
import numpy as np


def zero(x):
    """
    Function that returns zero regardless of the input.

    Args:
        x (Any): Input to the function.

    Returns:
        int: Returns 0.
    """
    return 0


def iden(x):
    """
    Identity function.

    Args:
        x (Any): Input to the function.

    Returns:
        Any: Returns the input without any changes.
    """
    return x


def get_hop_distance(num_node, edge, max_hop=1):
    """
    Compute the hop distance between nodes in a graph.

    Args:
        num_node (int): The number of nodes in the graph.
        edge (list): A list of tuples representing the edges between nodes.
        max_hop (int, optional): The maximum hop distance to consider. Defaults to 1.

    Returns:
        numpy.ndarray: An adjacency matrix representing the hop distances between nodes.
    """
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    """
    Normalize a directed graph.

    Args:
        A (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The adjacency matrix of the normalized graph.
    """
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    """
    Normalize an undirected graph.

    Args:
        A (numpy.ndarray): The adjacency matrix of the graph.

    Returns:
        numpy.ndarray: The adjacency matrix of the normalized graph.
    """
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class Graph():
    """
    The Graph class models the skeletons extracted by openpose.

    Attributes:
        max_hop (int): The maximal distance between two connected nodes.
        dilation (int): Controls the spacing between the kernel points.
        A (numpy.ndarray): The adjacency matrix of the graph.
    """

    def __init__(self,
                 layout="nvidia",
                 strategy="spatial",
                 max_hop=1,
                 dilation=1):
        """
        Initialize a spatial-temporal graph.

        Args:
            layout (str, optional): The layout of the graph. Defaults to "nvidia".
                - nvidia: Is consists of 34 joints. For more information, please
                    refer to https://docs.nvidia.com/deeplearning/maxine/ar-sdk-programming-guide/index.html
                - openpose: Is consists of 18 joints. For more information, please
                    refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
                - human3.6m: Is consists of 17 joints. For more information, please
                    refer to http://vision.imar.ro/human3.6m/description.php
                - ntu-rgb+d: Is consists of 25 joints. For more information, please
                    refer to https://github.com/shahroudy/NTURGB-D
                - ntu_edge: Is consists of 24 joints. For more information, please
                    refer to https://github.com/shahroudy/NTURGB-D
                - coco: Is consists of 17 joints. For more information, please
                    refer to https://cocodataset.org/#home
            strategy (str, optional): The strategy used to construct the graph. Defaults to "spatial".
                - uniform: Uniform Labeling
                - distance: Distance Partitioning
                - spatial: Spatial Configuration
                For more information, please refer to the section "Partition Strategies" in the paper (https://arxiv.org/abs/1801.07455).
            max_hop (int, optional): The maximal distance between two connected nodes. Defaults to 1.
            dilation (int, optional): Controls the spacing between the kernel points. Defaults to 1.
        """
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        """
        String representation of the spatial-temporal graph.

        Returns:
            str: The adjacency matrix of the graph as a string.
        """
        return self.A

    def get_edge(self, layout):
        """
        Get the edge of the graph.

        Args:
            layout (str): The layout of the graph.
        """
        # edge is a list of [child, parent] pairs

        if layout == "nvidia":
            self.num_node = 34
            self.num_person = 1
            self.seq_length = 300
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
                             (4, 7), (5, 8), (7, 9), (8, 10), (9, 11), (10, 12),
                             (7, 13), (8, 14), (6, 15), (15, 16), (15, 17),
                             (16, 18), (17, 19), (1, 20), (2, 21), (6, 20),
                             (6, 21), (20, 22), (21, 23), (22, 24), (23, 25),
                             (24, 26), (25, 27), (24, 28), (25, 29), (24, 30),
                             (25, 31), (24, 32), (25, 33)]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == "openpose":
            self.num_node = 18
            self.num_person = 1
            self.seq_length = 300
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "human3.6m":
            self.num_node = 17
            self.num_person = 1
            self.seq_length = 3163
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(9, 10), (8, 9), (7, 8), (1, 7),
                             (4, 7), (0, 1), (0, 4), (9, 11),
                             (9, 14), (11, 12), (14, 15), (12, 13),
                             (15, 16), (1, 2), (2, 3), (4, 5), (5, 6)]
            self.edge = self_link + neighbor_link
            self.center = 8
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            self.num_person = 2
            self.seq_length = 300
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == "ntu_edge":
            self.num_node = 24
            self.num_person = 2
            self.seq_length = 300
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == "coco":
            self.num_node = 17
            self.num_person = 1
            self.seq_length = 300
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        # elif layout=="customer settings"
        #     pass
        else:
            raise NotImplementedError(f"Layout \"{layout}\" not supported. Please choose from "
                                      f"[\"nvidia\", \"openpose\", \"human3.6m\", \"ntu-rgb+d\", "
                                      f"\"ntu_edge\", \"coco\"].")

    def get_adjacency(self, strategy):
        """
        Get the adjacency of the graph.

        Args:
            strategy (str): The strategy used to construct the graph.
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise NotImplementedError(f"Strategy \"{strategy}\" not supported. Please choose from "
                                      f"[\"uniform\", \"distance\", \"spatial\"].")

    def get_num_node(self):
        """
        Get the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph.
        """
        return self.num_node

    def get_num_person(self):
        """
        Get the number of persons in the graph.

        Returns:
            int: The number of persons in the graph.
        """
        return self.num_person

    def get_seq_length(self):
        """
        Get the sequence length.

        Returns:
            int: The sequence length.
        """
        return self.seq_length


class ConvTemporalGraphical(nn.Module):
    """
    The basic module for applying a graph convolution.

    Args:
        input_channels (int): Number of channels in the input sequence data
        output_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, input_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, output_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        """
        Initializes a module for graph convolution.

        This module is a basic unit for applying a convolution operation on graph-structured data. It involves
        both spatial convolution (i.e., convolution on graph) and temporal convolution (i.e., convolution
        across time dimension).
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(input_channels,
                              output_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A):
        """
        Apply forward propagation.

        Args:
            x (torch.Tensor): The input graph sequence. It has a shape of :math:`(N, input_channels, T_{in}, V)`.
            A (torch.Tensor): The adjacency matrix of the graph. It has a shape of :math:`(K, V, V)`.

        Returns:
            torch.Tensor: The output graph sequence. It has a shape of :math:`(N, output_channels, T_{out}, V)`.
            torch.Tensor: The adjacency matrix of the graph for output data. It has a shape of :math:`(K, V, V)`.
        """
        assert A.size(0) == self.kernel_size, f"A.size(0) {A.size(0)} does not match "\
            f"self.kernel_size {self.kernel_size}."

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A


class ST_GCN_Block(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        input_channels (int): Number of channels in the input sequence data
        output_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, input_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, output_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        """
        Initializes a module for spatial temporal graph convolution.

        This module is a basic unit for applying a convolution operation on graph-structured data with consideration
        of both spatial and temporal information.
        """
        super().__init__()

        assert len(kernel_size) == 2, f"len(kernel_size) should be 2. Got {len(kernel_size)}."
        assert kernel_size[0] % 2 == 1, "kernel_size[0] should be odd. Got even."
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(input_channels, output_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (input_channels == output_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(input_channels,
                          output_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(output_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        """
        Apply forward propagation.

        Args:
            x (torch.Tensor): The input graph sequence. It has a shape of :math:`(N, input_channels, T_{in}, V)`.
            A (torch.Tensor): The adjacency matrix of the graph. It has a shape of :math:`(K, V, V)`.

        Returns:
            torch.Tensor: The output graph sequence. It has a shape of :math:`(N, output_channels, T_{out}, V)`.
            torch.Tensor: The adjacency matrix of the graph for output data. It has a shape of :math:`(K, V, V)`.
        """
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class ST_GCN(nn.Module):
    """
    Spatial temporal graph convolutional networks.

    Args:
        input_channels (int): Number of channels in the input data
        num_classes (int): Number of classes for the classification task
        graph_layout (str): The layout of the graph
        graph_strategy (str): The strategy of the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, input_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_classes)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self,
                 input_channels,
                 num_classes,
                 graph_layout,
                 graph_strategy,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        """
        Initializes the spatial-temporal graph convolution network.
        """
        super().__init__()

        # load graph
        self.graph = Graph(layout=graph_layout, strategy=graph_strategy)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer("A", A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(input_channels *
                                      A.size(1)) if data_bn else iden
        kwargs = {k: v for k, v in kwargs.items() if k != "model_type"}
        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList((
            ST_GCN_Block(input_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 128, kernel_size, 2, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 256, kernel_size, 2, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Apply forward propagation.

        Args:
            x (torch.Tensor): The input graph sequence. It has a shape of :math:`(N, input_channels, T_{in}, V_{in}, M_{in})`.

        Returns:
            torch.Tensor: The output sequence. It has a shape of :math:`(N, num_classes)`.
        """
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x_size = [int(s) for s in x.size()[2:]]
        x = functional.avg_pool2d(x, x_size)
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):
        """
        Extract features from the input.

        Args:
            x (torch.Tensor): The input graph sequence. It has a shape of :math:`(N, input_channels, T_{in}, V_{in}, M_{in})`.

        Returns:
            torch.Tensor: The output sequence. It has a shape of :math:`(N, num_classes, T_{out}, V_{out}, M_{out})`.
            torch.Tensor: The extracted feature from the input. It has a shape of :math:`(N, C_{out}, T_{out}, V_{out}, M_{out})`.
        """
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


def st_gcn(pretrained_weights,
           input_channels,
           num_classes,
           graph_layout,
           graph_strategy,
           edge_importance_weighting=True,
           data_bn=True,
           **kwargs):
    """
    Constructs an ST-GCN (Spatial Temporal Graph Convolutional Networks) model.

    Args:
        pretrained_weights (torch.nn.Module): A PyTorch module with pretrained weights.
            If provided, these weights are loaded into the model.
        input_channels (int): Number of channels in the input data.
        num_classes (int): Number of classes for the classification task.
        graph_layout (str): The layout of the graph.
        graph_strategy (str): The strategy of the graph.
        edge_importance_weighting (bool, optional): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: ``True``.
        data_bn (bool, optional): If ``True``, applies Batch Normalization on the input data. Default: ``True``.
        **kwargs (optional): Other parameters for graph convolution units.

    Returns:
        model (ST_GCN): An ST-GCN model configured with the given parameters and weights.
    """
    model = ST_GCN(input_channels=input_channels, num_classes=num_classes,
                   graph_layout=graph_layout, graph_strategy=graph_strategy,
                   edge_importance_weighting=edge_importance_weighting,
                   data_bn=data_bn, **kwargs)

    if pretrained_weights:
        model.load_state_dict(pretrained_weights)

    return model
