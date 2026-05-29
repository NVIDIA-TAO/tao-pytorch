# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Match finder for metric-learning recognition."""

from pytorch_metric_learning.utils.inference import (FaissKNN, try_gpu, return_results)
from pytorch_metric_learning.utils import common_functions as c_f


class EmbeddingKNN(FaissKNN):
    """Uses the faiss library to compute k-nearest-neighbors.

    Inherits from `pytorch_metric_learning.utils.inference.FaissKNN` but removes logging
    function when calling the object.

    Attributes:
        reset_before (Boolean): Reset the faiss index before knn is computed
        reset_after (Boolean): Reset the faiss index after knn is computed (good for clearing memory)
        index_init_fn (Callable, optional): A callable that takes in the embedding dimensionality and returns a faiss index. The default is faiss.IndexFlatL2
        gpus (List[int], optional): A list of gpu indices to move the faiss index onto. The default is to use all available gpus, if the input tensors are also on gpus
    """

    def __call__(
        self,
        query,
        k,
        reference=None,
        embeddings_come_from_same_source=False,
    ):
        """Calculates the K nearest neighghbors.

        Args:
            query (torch.Tensor): Query embeddings.
            k (int): The k in k-nearest-neighbors.
            reference (torch.Tensor,  optional): The embeddings to search.
            embeddings_come_from_same_source (Boolean, optional): Whether or not query and reference share datapoints.

        Returns:
            distances (torch.Tensor): the distances of k-nearest-neighbors in increasing order.
            indices (torch.Tensor): the indices of k-nearest-neighbors in dataset.
        """
        if embeddings_come_from_same_source:
            k = k + 1
        device = query.device
        is_cuda = query.is_cuda
        d = query.shape[1]
        if self.reset_before:
            self.index = self.index_init_fn(d)
        distances, indices = try_gpu(
            self.index,
            query,
            reference,
            k,
            is_cuda,
            self.gpus,
        )
        distances = c_f.to_device(distances, device=device)
        indices = c_f.to_device(indices, device=device)
        if self.reset_after:
            self.reset()
        return return_results(distances, indices, embeddings_come_from_same_source)
