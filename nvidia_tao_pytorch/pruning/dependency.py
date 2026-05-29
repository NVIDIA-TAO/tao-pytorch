# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Prune modules as Monkey patch for torch-pruning. """

from torch_pruning import DependencyGraph
from torch_pruning import ops


class TAO_DependencyGraph(DependencyGraph):
    """Inherit DependencyGraph class from torch-pruning"""

    def update_index_mapping(self):
        """ Update all index mapping after pruning """
        for _, node in self.module2node.items():
            if node.type == ops.OPTYPE.CONCAT:
                # enable index mapping for the concat in FPN neck
                for node_out in node.outputs:
                    if "linear_fuse.conv" in node_out.name:
                        node.enable_index_mapping = True
                self._update_concat_index_mapping(node)
            if node.type == ops.OPTYPE.SPLIT:
                self._update_split_index_mapping(node)
            if node.type == ops.OPTYPE.RESHAPE:
                self._update_reshape_index_mapping(node)
