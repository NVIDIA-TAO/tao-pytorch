# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter="???",
    amp=dict(
        enabled=False,
        opt_level=None,
    ),  # options for Automatic Mixed Precision
    grad_clip=None,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period="${train.checkpointer.period}",
    log_period=50,
    device="cuda",
    seed=42,
    # ...
    wandb=dict(
        enable_writer=False,
        resume=False,
        project="ODISE",
    ),
    cfg_name="",
    run_name="",
    run_tag="",
    reference_world_size=0,
)
