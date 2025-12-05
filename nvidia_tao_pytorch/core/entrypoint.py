# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Common utilities that could be used across all nlp models."""

import re
import ast
import importlib
import os
import pkgutil
import subprocess
import shlex
import sys
import torch
from time import time
import yaml
from contextlib import contextmanager

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.core.distributed.validator import validate_configs
from nvidia_tao_core.telemetry.nvml import get_device_details
from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

LIGHTNING_EXCLUDED_NETWORKS = [
    "bevfusion",
    "pointpillars",
    "rtdetr",
]


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the tasks in in the .scripts folder.

    Returns:
        subtasks (dict): Dictionary of files.

    """
    module_path = package.__path__
    modules = {}

    # Collect modules dynamically.
    for _, task, is_package in pkgutil.walk_packages(module_path):
        if is_package:
            continue
        module_name = package.__name__ + "." + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(
                importlib.import_module(module_name).__file__
            ),
        }
        modules[task] = module_details

    # Add download_specs command as a common subtask for all networks
    try:
        from nvidia_tao_pytorch.core.utils import default_specs
        modules["default_specs"] = {
            "module_name": "nvidia_tao_pytorch.core.utils.default_specs",
            "runner_path": os.path.abspath(default_specs.__file__),
        }
    except ImportError as e:
        logging.warning(f"Could not load default_specs: {e}")

    return modules


def command_line_parser(parser, subtasks):
    """Construct parser for CLI arguments"""
    parser.add_argument(
        "subtask",
        default="train",
        choices=subtasks.keys(),
        help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        default=None,
    )
    args, unknown_args = parser.parse_known_args()

    return args, unknown_args


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, "a") as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def launch(args, unknown_args, subtasks, network=None):
    """CLI function that executes subtasks.

    Args:
        parser: Created parser object for a given task.
        subtasks: list of subtasks for a given task.
        network (str): name of the network running.
    """
    # default_specs doesn't require an experiment spec file
    if args["subtask"] != "default_specs":
        # Make sure the user provides spec file.
        if args["experiment_spec_file"] is None:
            print(
                "ERROR: The subtask `{}` requires the following argument: -e/--experiment_spec_file".format(
                    args["subtask"]
                )
            )
            exit(1)

        # Make sure the file exists!
        if not os.path.exists(args["experiment_spec_file"]):
            print(
                "ERROR: The indicated experiment spec file `{}` doesn't exist!".format(
                    args["experiment_spec_file"]
                )
            )
            exit(1)

    script_args = ""

    # Handle default_specs separately - it doesn't use experiment_spec_file
    if args["subtask"] == "default_specs":
        # Add module_name argument (the network name)
        if network:
            script_args += f" module_name={network}"
        # Pass results_dir if provided
        if "results_dir" in args:
            script_args += " results_dir=" + args["results_dir"]
    else:
        # Split spec file_path into config path and config name.
        path, name = os.path.split(args["experiment_spec_file"])
        if path != "":
            script_args += " --config-path " + os.path.realpath(path)
        script_args += " --config-name " + name

        # This enables a results_dir arg to be passed from the microservice side,
        # but there is no --results_dir cmdline arg. Instead, the spec field must be used
        if "results_dir" in args:
            script_args += " results_dir=" + args["results_dir"]

    # Pass unknown args to call
    unknown_args_as_str = " " + " ".join(unknown_args)

    # Set gpus - overwrite if fields are inconsistent
    # Precedence for gpu setting: env > cmdline > specfile > default
    overrides = ["num_gpus", "gpu_ids", "cuda_blocking"]
    num_gpus = 1
    gpu_ids = [0]
    num_nodes = 1
    if args["subtask"] in ["train", "evaluate", "inference", "distill"]:
        # Parsing cmdline override
        if any(arg in unknown_args_as_str for arg in overrides):
            if "num_gpus" in unknown_args_as_str:
                num_gpus = int(
                    unknown_args_as_str.split("num_gpus=")[1].split()[0]
                )
            if "gpu_ids" in unknown_args_as_str:
                gpu_ids = ast.literal_eval(
                    unknown_args_as_str.split("gpu_ids=")[1].split()[0]
                )
            if "num_nodes" in unknown_args_as_str:
                num_nodes = (
                    unknown_args_as_str.split("num_nodes=")[1].split()[0]
                )
        # If no cmdline override, look at specfile
        else:
            if args["subtask"] == "distill":
                # distill looks at train.num_gpus and train.gpu_ids
                task = "train"
            else:
                task = args["subtask"]
            with open(args["experiment_spec_file"], "r") as spec:
                exp_config = yaml.safe_load(spec)
                if task in exp_config:
                    if "num_gpus" in exp_config[task]:
                        num_gpus = exp_config[task]["num_gpus"]
                    if "gpu_ids" in exp_config[task]:
                        gpu_ids = exp_config[task]["gpu_ids"]
                    if "num_nodes" in exp_config[task]:
                        num_nodes = exp_config[task]["num_nodes"]

    if num_gpus != len(gpu_ids):
        logging.warning(f"Number of gpus {num_gpus} != len({gpu_ids}).")
        num_gpus = max(num_gpus, len(gpu_ids))
        gpu_ids = list(range(num_gpus)) if len(gpu_ids) != num_gpus else gpu_ids
        logging.info(f"Using GPUs {gpu_ids} (total {num_gpus})")

    # Configure multinode
    multinode = [f"--nnodes={num_nodes}", f"--nproc-per-node={num_gpus}"]
    if os.environ.get("WORLD_SIZE"):
        try:
            validate_configs(logging)

            logging.warning("[Multinode] Overriding configs (num_nodes, num_gpus, gpu_ids) with environment variables.")
            num_nodes = int(os.environ.get("WORLD_SIZE"))
            num_gpus = int(os.environ.get("NUM_GPU_PER_NODE", torch.cuda.device_count()))
            gpu_ids = list(range(num_gpus))

            # Update multinode config in spec file
            with open(args["experiment_spec_file"], "r") as spec:
                exp_config = yaml.safe_load(spec)

            train_config = exp_config.get("train", {})
            if "num_nodes" in train_config:
                train_config["num_nodes"] = num_nodes
            if "num_gpus" in train_config:
                train_config["num_gpus"] = num_gpus
            if "gpu_ids" in train_config:
                train_config["gpu_ids"] = gpu_ids

            if train_config:
                exp_config["train"] = train_config
                with open(args["experiment_spec_file"], "w") as spec:
                    yaml.dump(exp_config, spec, default_flow_style=False)

            multinode = [
                f"--nnodes={num_nodes}",
                f"--nproc-per-node={num_gpus}",
                f"--node-rank={os.getenv('NODE_RANK') or os.getenv('RANK')}",
                f"--master-addr={os.getenv('MASTER_ADDR', 'localhost')}",
                f"--master-port={os.getenv('MASTER_PORT', '29500')}",
            ]
        except Exception as e:
            logging.warning(f"[Multinode] Error overriding configs: {e}")
            logging.warning("[Multinode] Using default configs.")
            num_nodes = 1
            num_gpus = torch.cuda.device_count()
            gpu_ids = list(range(num_gpus))
            multinode = [f"--nnodes={num_nodes}", f"--nproc-per-node={num_gpus}"]

    # All future logic will look at this envvar for guidance on which devices to use
    os.environ["TAO_VISIBLE_DEVICES"] = str(gpu_ids)[1:-1]

    # Find relevant module and pass args
    script = subtasks[args["subtask"]]["runner_path"]

    log_file = ""
    if os.getenv("JOB_ID"):
        logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
        log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

    # Create a system call.
    if network in LIGHTNING_EXCLUDED_NETWORKS:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["TAO_VISIBLE_DEVICES"]
        if not os.getenv("RANK") and os.getenv("NODE_RANK"):
            os.environ["RANK"] = os.getenv("NODE_RANK")
        call = (
            "torchrun" +
            f" {' '.join(multinode)} " +
            script +
            script_args +
            unknown_args_as_str
        )
    else:
        call = "python " + script + script_args + unknown_args_as_str

    process_passed = False
    user_error = False
    start = time()
    progress_bar_pattern = re.compile(r"Epoch \d+: \s*\d+%|\[.*\]")

    try:
        # Run the script.
        with dual_output(log_file) as (stdout_target, log_target):
            proc = subprocess.Popen(
                shlex.split(call),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            )
            last_progress_bar_line = None

            for line in proc.stdout:
                # Check if the line contains \r or matches the progress bar pattern
                if '\r' in line or progress_bar_pattern.search(line):
                    last_progress_bar_line = line.strip()
                    # Print the progress bar line to the terminal
                    stdout_target.write('\r' + last_progress_bar_line)
                    stdout_target.flush()
                else:
                    # Write the final progress bar line to the log file before a new log line
                    if last_progress_bar_line:
                        if log_target:
                            log_target.write(last_progress_bar_line + '\n')
                            log_target.flush()
                        last_progress_bar_line = None
                    stdout_target.write(line)
                    stdout_target.flush()
                    if log_target:
                        log_target.write(line)
                        log_target.flush()

            proc.wait()  # Wait for the process to complete
            # Write the final progress bar line after process completion
            if last_progress_bar_line and log_target:
                log_target.write(last_progress_bar_line + '\n')
                log_target.flush()
            if proc.returncode == 0:
                process_passed = True

    except (KeyboardInterrupt, SystemExit):
        logging.exception("Command was interrupted")
        process_passed = True
    except Exception as e:
        # Check if the exception is a user configuration error
        error_message = str(e)
        user_error = any(keyword in error_message for keyword in [
            "Configuration error",
            "Feature not implemented",
            "Parameter validation error",
            "File system error",
            "Schema validation error"
        ])

        logging.exception(e)
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    try:
        gpu_data = list()
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logging.info("Sending telemetry data.")
        send_telemetry_data(
            network,
            args["subtask"],
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed,
            user_error=user_error
        )
    except Exception as e:
        logging.warning(
            "Telemetry data couldn't be sent, but the command ran successfully."
        )
        logging.warning(f"[Error]: {e}")
        pass

    if not process_passed:
        logging.warning("Execution status: FAIL")
        sys.exit(1)

    logging.info("Execution status: PASS")
    sys.exit(0)
