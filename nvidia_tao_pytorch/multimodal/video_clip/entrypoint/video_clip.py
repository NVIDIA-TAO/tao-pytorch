# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Define entrypoint to run tasks for video_clip."""

import argparse
import os
import pkgutil
import warnings
from pathlib import Path

_WARNING_FILTERS = (
    (
        "Importing from timm.models.layers is deprecated.*",
        FutureWarning,
        "Importing from timm.models.layers is deprecated",
    ),
    (
        "The video decoding and encoding capabilities of torchvision "
        "are deprecated.*",
        UserWarning,
        "The video decoding and encoding capabilities of torchvision "
        "are deprecated",
    ),
)


def _suppress_known_warning_noise():
    """Suppress known dependency deprecation warnings for this task."""
    for message, category, _ in _WARNING_FILTERS:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=category,
        )
    env_filters = [
        f"ignore:{env_message}:{category.__name__}"
        for _, category, env_message in _WARNING_FILTERS
    ]
    existing_filters = os.environ.get("PYTHONWARNINGS")
    if existing_filters:
        env_filters.append(existing_filters)
    os.environ["PYTHONWARNINGS"] = ",".join(env_filters)


_suppress_known_warning_noise()

from nvidia_tao_pytorch.multimodal.video_clip import scripts  # noqa: E402
from nvidia_tao_pytorch.core.entrypoint import launch, command_line_parser  # noqa: E402
from nvidia_tao_pytorch.core.utils import default_specs  # noqa: E402


def get_subtask_list():
    """Return supported subtasks without importing each script module."""
    script_path = Path(next(iter(scripts.__path__)))
    modules = {}
    for _, task, is_package in pkgutil.walk_packages(scripts.__path__):
        if is_package:
            continue
        modules[task] = {
            "module_name": f"{scripts.__name__}.{task}",
            "runner_path": str(script_path / f"{task}.py"),
        }
    modules["default_specs"] = {
        "module_name": "nvidia_tao_pytorch.core.utils.default_specs",
        "runner_path": str(Path(default_specs.__file__).resolve()),
    }
    return modules


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "video_clip",
        add_help=True,
        description="Train Adapt Optimize entrypoint for video_clip",
    )

    # Obtain the list of substasks
    subtasks = get_subtask_list()

    # Parse the arguments
    args, unknown_args = command_line_parser(parser, subtasks)

    # Launch the subtask.
    launch(vars(args), unknown_args, subtasks, network="video_clip")


if __name__ == "__main__":
    main()
