# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CoDETR entrypoint."""

import argparse
from nvidia_tao_pytorch.cv.codetr import scripts
from nvidia_tao_pytorch.core.entrypoint import get_subtasks, launch, command_line_parser


def get_subtask_list():
    """Return list of available subtasks."""
    return get_subtasks(scripts)


def main():
    """CoDETR CLI entrypoint."""
    parser = argparse.ArgumentParser(
        "codetr", add_help=True,
        description="Train Adapt Optimize entrypoint for CoDETR"
    )
    subtasks = get_subtask_list()
    args, unknown_args = command_line_parser(parser, subtasks)
    launch(vars(args), unknown_args, subtasks, network="codetr")


if __name__ == "__main__":
    main()
