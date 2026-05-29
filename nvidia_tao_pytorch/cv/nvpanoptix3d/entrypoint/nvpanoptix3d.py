# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define entrypoint to run tasks for NVPanoptix3D model."""

import argparse

from nvidia_tao_pytorch.core.entrypoint import get_subtasks, launch, command_line_parser
import nvidia_tao_pytorch.cv.nvpanoptix3d.scripts as scripts


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "nvpanoptix3d", add_help=True,
        description="Train Adapt Optimize entrypoint for NVPanoptix3D model",
    )

    # Build list of subtasks by inspecting the package.
    subtasks = get_subtask_list()

    # Parse the arguments
    args, unknown_args = command_line_parser(parser, subtasks)

    # Launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="nvpanoptix3d")


if __name__ == "__main__":
    main()
