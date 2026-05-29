# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define entrypoint to run tasks for depth_net."""

import argparse
from nvidia_tao_pytorch.cv.depth_net import scripts
from nvidia_tao_pytorch.core.entrypoint import get_subtasks, launch, command_line_parser


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "depth_net",
        add_help=True,
        description="Train Adapt Optimize entrypoint for depth_net",
    )

    # Obtain the list of substasks
    subtasks = get_subtask_list()

    # Parse the arguments
    args, unknown_args = command_line_parser(parser, subtasks)

    # Launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="depth_net")


if __name__ == "__main__":
    main()
