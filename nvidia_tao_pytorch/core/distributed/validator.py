# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


"""Multinode configurations and validation"""

import os
import logging
import socket

import torch


def validate_configs(log: logging.Logger) -> None:
    """Validate the distributed training configurations"""
    num_nodes = int(os.environ.get("WORLD_SIZE")) if os.environ.get("WORLD_SIZE") else None
    node_rank = int(os.environ.get("NODE_RANK")) if os.environ.get("NODE_RANK") else None
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = int(os.environ.get("MASTER_PORT")) if os.environ.get("MASTER_PORT") else None
    num_gpus = int(os.environ.get("NUM_GPU_PER_NODE")) if os.environ.get("NUM_GPU_PER_NODE") else None

    if num_nodes is None:
        log.warning("Multinode training is not enabled through WORLD_SIZE, ignoring master address, port, and node rank")
        return

    # invalid parameters
    if num_nodes < 1:
        raise ValueError("Number of nodes must be greater than 0")

    if node_rank is not None and (node_rank < 0 or node_rank >= num_nodes):
        raise ValueError("Node rank must be between 0 and num_nodes - 1")

    if num_gpus is not None and (num_gpus < 1 or num_gpus > torch.cuda.device_count()):
        raise ValueError("Number of GPUs must be greater than 0 and less than the total number of GPUs")

    if master_port is not None and (master_port < 1024 or master_port > 65535):
        raise ValueError("Port must be between 1024 and 65535")

    # missing parameters
    if num_nodes is not None and num_nodes > 1:
        if master_addr is None or master_port is None or node_rank is None:
            raise ValueError("Master address, port, and node rank must be specified for multinode training")
    else:
        if master_addr is not None or master_port is not None or node_rank is not None:
            log.warning("Multinode training is not enabled through num_nodes, ignoring master address, port, and node rank")
        return

    # network validation
    if master_addr is not None:
        try:
            # Validate master address
            socket.gethostbyname(master_addr)
            log.info("Successfully validated connection to master worker.")
        except socket.gaierror:
            raise ValueError(f"Invalid master address: {master_addr}")

        # validate port
        if master_port is not None and node_rank is not None and node_rank == 0:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((master_addr, master_port))
                    s.listen(1)
                log.info("Successfully validated master node port binding.")
            except socket.error:
                raise ValueError(f"Port {master_port} is already in use")
