# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
try:
    from nvidia_tao_core.telemetry.nvml import get_device_details
    from nvidia_tao_core.telemetry.telemetry import send_telemetry_data
    TELEMETRY_AVAILABLE = True
except ImportError:
    get_device_details = None
    send_telemetry_data = None
    TELEMETRY_AVAILABLE = False

LIGHTNING_EXCLUDED_NETWORKS = [
    "bevfusion",
    "pointpillars",
    "rtdetr",
]


def _native_dinov3_deft_requested(args, network):
    """Return whether the DINOv3 DEFT node-level Lightning path is selected."""
    return all((
        os.environ.get("TAO_REFINEMENT_LIGHTNING_LAUNCH") == "1",
        network == "dinov3",
        args["subtask"] == "train",
    ))


def _resolve_native_dinov3_deft_allocation(args, unknown_args):
    """Resolve the sealed train spec first, then overlay Hydra allocation args."""
    prefix = "Parameter validation error: DINOv3 DEFT"
    try:
        with open(args["experiment_spec_file"], "r", encoding="utf-8") as spec:
            experiment = yaml.safe_load(spec)
    except yaml.YAMLError as error:
        raise ValueError(f"{prefix} train spec is invalid YAML") from error
    if experiment is None:
        experiment = {}
    if not isinstance(experiment, dict):
        raise ValueError(f"{prefix} train spec root must be a mapping")
    train = experiment.get("train") or {}
    if not isinstance(train, dict):
        raise ValueError(f"{prefix} train must be a mapping")
    allocation = {
        "num_nodes": train.get("num_nodes", 1),
        "num_gpus": train.get("num_gpus", 1),
        "gpu_ids": train.get("gpu_ids", [0]),
    }
    supported = {
        "num_nodes": "num_nodes",
        "train.num_nodes": "num_nodes",
        "num_gpus": "num_gpus",
        "train.num_gpus": "num_gpus",
        "gpu_ids": "gpu_ids",
        "train.gpu_ids": "gpu_ids",
    }
    for argument in unknown_args:
        key, separator, value = argument.lstrip("+").partition("=")
        field = supported.get(key)
        if not separator or field is None:
            continue
        allocation[field] = (
            ast.literal_eval(value)
            if field == "gpu_ids"
            else int(value)
        )
    return allocation["num_nodes"], allocation["num_gpus"], allocation["gpu_ids"]


def _validate_native_dinov3_deft_launch(*, num_nodes, num_gpus, gpu_ids):
    """Fail closed on an incomplete or inconsistent DEFT node allocation."""
    prefix = "Parameter validation error: DINOv3 DEFT"

    def parse_integer(value, name):
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{prefix} {name} must be an integer") from error

    expected_nodes = parse_integer(num_nodes, "train.num_nodes")
    expected_gpus = parse_integer(num_gpus, "train.num_gpus")
    if expected_nodes <= 0 or expected_gpus <= 0:
        raise ValueError(f"{prefix} allocation must be positive")
    if len(gpu_ids) != expected_gpus:
        raise ValueError(
            f"{prefix} train.num_gpus must equal the length of train.gpu_ids"
        )
    try:
        configured_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{prefix} train.gpu_ids must contain integers") from error
    if configured_gpu_ids != list(range(expected_gpus)):
        raise ValueError(f"{prefix} requires contiguous local train.gpu_ids")

    worker_variables = {
        "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
        "ROLE_RANK", "ROLE_WORLD_SIZE",
    }
    inherited_workers = sorted(
        name for name in os.environ if (
            name in worker_variables or name.startswith("TORCHELASTIC_")
        )
    )
    if inherited_workers:
        raise ValueError(
            f"{prefix} must start at a node boundary; inherited worker "
            f"variables are forbidden: {inherited_workers}"
        )

    required = (
        "WORLD_SIZE", "NUM_GPU_PER_NODE", "NODE_RANK",
        "MASTER_ADDR", "MASTER_PORT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"{prefix} is missing allocation fields: {missing}")

    declared_nodes = parse_integer(os.environ["WORLD_SIZE"], "WORLD_SIZE")
    declared_gpus = parse_integer(
        os.environ["NUM_GPU_PER_NODE"], "NUM_GPU_PER_NODE"
    )
    node_rank = parse_integer(os.environ["NODE_RANK"], "NODE_RANK")
    master_port = parse_integer(os.environ["MASTER_PORT"], "MASTER_PORT")
    if declared_nodes != expected_nodes:
        raise ValueError(f"{prefix} WORLD_SIZE differs from train.num_nodes")
    if declared_gpus != expected_gpus:
        raise ValueError(f"{prefix} NUM_GPU_PER_NODE differs from train.num_gpus")
    if not 0 <= node_rank < expected_nodes:
        raise ValueError(f"{prefix} NODE_RANK is out of range")
    if not 1024 <= master_port <= 65535:
        raise ValueError(f"{prefix} MASTER_PORT is invalid")
    if expected_gpus > torch.cuda.device_count():
        raise ValueError(f"{prefix} requests more GPUs than are visible")
    # Retain TAO's DNS and rank-zero rendezvous-port validation.
    validate_configs(logging)


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
    # Ordinary launch: env > cmdline > specfile > default. DEFT instead requires
    # the allocation environment to agree with the effective Hydra/spec values.
    overrides = ["num_gpus", "gpu_ids", "cuda_blocking"]
    native_dinov3_deft = _native_dinov3_deft_requested(args, network)
    launch_validation_error = None
    num_gpus = 1
    gpu_ids = [0]
    num_nodes = 1
    if native_dinov3_deft:
        try:
            num_nodes, num_gpus, gpu_ids = _resolve_native_dinov3_deft_allocation(
                args, unknown_args
            )
            _validate_native_dinov3_deft_launch(
                num_nodes=num_nodes,
                num_gpus=num_gpus,
                gpu_ids=gpu_ids,
            )
        except (SyntaxError, TypeError, ValueError) as error:
            message = str(error)
            if not message.startswith("Parameter validation error:"):
                error = ValueError(
                    "Parameter validation error: DINOv3 DEFT allocation "
                    f"override is invalid: {message}"
                )
            launch_validation_error = error
    elif args["subtask"] in ["train", "evaluate", "inference", "distill"]:
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

    if num_gpus != len(gpu_ids) and not native_dinov3_deft:
        logging.warning(f"Number of gpus {num_gpus} != len({gpu_ids}).")
        num_gpus = max(num_gpus, len(gpu_ids))
        gpu_ids = list(range(num_gpus)) if len(gpu_ids) != num_gpus else gpu_ids
        logging.info(f"Using GPUs {gpu_ids} (total {num_gpus})")

    # Configure multinode
    multinode = [f"--nnodes={num_nodes}", f"--nproc-per-node={num_gpus}"]
    if os.environ.get("WORLD_SIZE") and not native_dinov3_deft:
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
        if launch_validation_error is not None:
            raise launch_validation_error
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
                logging.info("Command completed successfully with return code 0")
                process_passed = True
            else:
                logging.info(f"Command completed with return code {proc.returncode}")

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

        logging.exception("Command encountered an exception")
        process_passed = False

    end = time()
    time_lapsed = int(end - start)

    try:
        if not TELEMETRY_AVAILABLE:
            logging.info("Telemetry module not available, skipping telemetry data.")
            raise ImportError("nvidia_tao_core.telemetry is not available")
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
