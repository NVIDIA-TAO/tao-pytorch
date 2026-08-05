#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

