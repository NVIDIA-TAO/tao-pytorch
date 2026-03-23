# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Distributed training utils. """

import torch.distributed as _dist


# TODO: Remove if not needed.
def get_rank():
    """Returns the rank of the current process in the distributed setting.

    Returns:
        int: Rank of the current process. Defaults to 0 if not in a distributed environment.
    """
    return _dist.get_rank() if _dist.is_initialized() else 0


def get_world_size():
    """Returns the total number of processes in the distributed environment.

    Returns:
        int: The world size (total processes). Defaults to 1 if not in a distributed environment.
    """
    return _dist.get_world_size() if _dist.is_initialized() else 1


def all_gather(tensor_list, tensor, group=None, async_op=False):
    """Gathers tensors from all processes and stores them in the provided list.

    Args:
        tensor_list (list): A list to store the gathered tensors.
        tensor (Tensor): The tensor to gather from each process.
        group (optional): The group of processes participating in the gather. Defaults to None.
        async_op (bool): Whether to perform the gather asynchronously. Defaults to False.
    """
    if _dist.is_initialized():
        _dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
    else:
        tensor_list[0].copy_(tensor)


def all_reduce(tensor, op=_dist.ReduceOp.SUM, group=None, async_op=False):
    """Performs an all-reduce operation on the input tensor across all processes.

    Args:
        tensor (Tensor): The tensor to reduce.
        op (ReduceOp, optional): The reduction operation (e.g., SUM). Defaults to ReduceOp.SUM.
        group (optional): The group of processes participating in the reduce. Defaults to None.
        async_op (bool): Whether to perform the reduction asynchronously. Defaults to False.
    """
    if _dist.is_initialized():
        _dist.all_reduce(tensor, op=op, group=group, async_op=async_op)
