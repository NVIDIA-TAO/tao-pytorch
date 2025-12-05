# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""'Entry point' script running subtasks related to OneFormer."""

import argparse

from nvidia_tao_pytorch.core.entrypoint import get_subtasks, launch, command_line_parser
import nvidia_tao_pytorch.cv.oneformer.scripts as scripts


def get_subtask_list():
    """Return the list of subtasks by inspecting the scripts package."""
    return get_subtasks(scripts)


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "oneformer",
        add_help=True,
        description="Train Adapt Optimize entrypoint for oneformer",
    )

    # Build list of subtasks by inspecting the package.
    subtasks = get_subtask_list()

    # Parse the arguments
    args, unknown_args = command_line_parser(parser, subtasks)

    # Launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="oneformer")


if __name__ == "__main__":
    main()
