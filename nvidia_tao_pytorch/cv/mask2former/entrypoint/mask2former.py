# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""'Entry point' script running subtasks related to Mask2former."""

import argparse

from nvidia_tao_pytorch.core.entrypoint import get_subtasks, launch, command_line_parser
import nvidia_tao_pytorch.cv.mask2former.scripts as scripts


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "mask2former",
        add_help=True,
        description="Train Adapt Optimize entrypoint for mask2former",
    )

    # Build list of subtasks by inspecting the package.
    subtasks = get_subtask_list()

    # Parse the arguments
    args, unknown_args = command_line_parser(parser, subtasks)

    # Launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="mask2former")


if __name__ == "__main__":
    main()
