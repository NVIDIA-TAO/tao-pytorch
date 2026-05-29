# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instantiate the TAO-pytorch docker container for developers."""

import argparse
import json
import os
import platform
import subprocess
import sys

from packaging import version

ROOT_DIR = os.getenv("NV_TAO_PYTORCH_TOP", os.getcwd())

with open(os.path.join(ROOT_DIR, "docker/manifest.json"), "r") as m_file:
    docker_config = json.load(m_file)

DOCKER_REGISTRY = docker_config["registry"]
DOCKER_REPOSITORY = docker_config["repository"]

# Platform keys for digest lookup
X86_KEY = "x86"
ARM_KEY = "arm"

# Handle both old and new manifest formats
if "digest" in docker_config:
    # Old format with single digest
    DOCKER_DIGEST = docker_config["digest"]
elif "digests" in docker_config:
    # New format with platform-specific digests
    arch = platform.machine()
    if arch == "x86_64":
        DOCKER_DIGEST = docker_config["digests"][X86_KEY]
    elif arch == "aarch64":
        DOCKER_DIGEST = docker_config["digests"][ARM_KEY]
    else:
        # Fallback to x86
        DOCKER_DIGEST = docker_config["digests"][X86_KEY]
else:
    raise ValueError("Invalid manifest format: missing 'digest' or 'digests' field")
DOCKER_COMMAND = "docker"
HOME_PATH = os.path.expanduser("~")
MOUNTS_PATH = os.path.join(HOME_PATH, ".tao_mounts.json")


def get_docker_mounts_from_file(mounts_file=MOUNTS_PATH):
    """Check for docker mounts in ~/.tao_mounts.json."""
    if not os.path.exists(mounts_file):
        return []
    with open(mounts_file, 'r') as mfile:
        data = json.load(mfile)
    assert "Mounts" in list(data.keys()), "Invalid json file. Requires Mounts key."
    return data["Mounts"]


def format_mounts(mount_points):
    """Format mount points to docker recognizable commands."""
    formatted_mounts = []
    # Traverse through mount points and add format them for the docker command.
    for mount_point in mount_points:
        assert "source" in list(mount_point.keys()), "destination" in list(mount_point.keys())
        mount = "{}:{}".format(mount_point["source"], mount_point["destination"])
        formatted_mounts.append(mount)
    return formatted_mounts


def check_image_exists(docker_image):
    """Check if the image exists locally."""
    check_command = '{} images | grep "\\<{}\\>" >/dev/null 2>&1'.format(DOCKER_COMMAND, docker_image)
    rc = subprocess.call(check_command, stdout=sys.stderr, shell=True)
    return rc == 0


def pull_base_container(docker_image):
    """Pull the default base container."""
    pull_command = "{} pull {}@{}".format(DOCKER_COMMAND, docker_image, DOCKER_DIGEST)
    rc = subprocess.call(pull_command, stdout=sys.stderr, shell=True)
    return rc == 0


def get_formatted_mounts(mount_file):
    """Simple function to get default mount points."""
    default_mounts = get_docker_mounts_from_file(mount_file)
    return format_mounts(default_mounts)


def check_mounts(formatted_mounts):
    """Check the formatted mount commands."""
    assert isinstance(formatted_mounts, list)
    for mounts in formatted_mounts:
        source_path = mounts.split(":")[0]
        if not os.path.exists(source_path):
            raise ValueError("Path doesn't exist: {}".format(source_path))
    return True


def get_docker_gpus_prefix(gpus):
    """Get the docker command gpu's prefix."""
    docker_version = (
        subprocess.check_output(
            ["docker", "version", "--format={{ .Server.APIVersion }}"]
        )
        .strip()
        .decode()
    )

    # Check if running on a tegra system like thor or jetson
    uname_output = subprocess.check_output(["uname", "-a"]).decode().strip()
    is_tegra = "tegra" in uname_output

    # Use nvidia runtime if docker version is old OR if on tao-thor/tegra system
    if version.parse(docker_version) <= version.parse("1.40") or is_tegra:
        # Stick to the older version of getting the gpu's using runtime=nvidia
        gpu_string = "--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all "
        if gpus != "none":
            gpu_string += "-e NVIDIA_VISIBLE_DEVICES={}".format(gpus)
    else:
        # You are using the latest version of docker using
        # --gpus instead of the nvidia runtime.
        gpu_string = "--gpus "
        if gpus == "all":
            gpu_string += "all"
        else:
            gpu_string += "\'\"device={}\"\'".format(gpus)
    return gpu_string


def instantiate_dev_docker(gpus, mount_file,
                           mount_cli_list,
                           env_var_list,
                           tag, command, ulimit=None,
                           shm_size="16G", run_as_user=False,
                           port_mapping=None,
                           tty=True,
                           cached=None):
    """Instiate the docker container."""
    docker_image = "{}/{}@{}".format(DOCKER_REGISTRY, DOCKER_REPOSITORY, DOCKER_DIGEST)
    if tag is not None:
        docker_image = "{}/{}:{}".format(DOCKER_REGISTRY, DOCKER_REPOSITORY, tag)

    if cached is not None:
        docker_image = cached

    # Invoking the nvidia docker.
    gpu_string = get_docker_gpus_prefix(gpus)

    # Prefix for the run command.
    if tty:
        run_command = "{} run -it --rm".format(DOCKER_COMMAND)
    else:
        run_command = "{} run --rm".format(DOCKER_COMMAND)

    # get default mount points.
    formatted_mounts = get_formatted_mounts(MOUNTS_PATH)

    # get mounts from cli mount file.
    formatted_mounts += get_formatted_mounts(mount_file)

    if mount_cli_list is not None:
        formatted_mounts.extend(mount_cli_list)

    assert check_mounts(formatted_mounts), "Mounts don't exists, Please make sure the paths all exist."

    mount_string = "-v {}:/tao-pt ".format(os.getenv("NV_TAO_PYTORCH_TOP", os.getcwd()))

    # Defining env variables.
    env_variables = "-e PYTHONPATH={}:$PYTHONPATH ".format("/tao-pt")
    for env in env_var_list:
        if "=" not in env:
            print(f"invalid env variable definition. skipping this {env}")
            continue
        env_variables += "-e {} ".format(env)

    for path in formatted_mounts:
        mount_string += "-v {} ".format(path)

    # Setting shared memory.
    shm_option = "--shm-size {}".format(shm_size)

    # Setting ulimits for host
    ulimit_options = ""
    if ulimit is not None:
        for param in ulimit:
            ulimit_options += "--ulimit {} ".format(param)

    user_option = ""
    if run_as_user:
        user_option = "--user {}:{}".format(os.getuid(), os.getgid())

    port_option = "--net=host"
    if port_mapping:
        port_option += f" -p {port_mapping}"

    final_command = "{} {} {} {} {} {} {} {} {} {}".format(
        run_command, gpu_string,
        mount_string, env_variables,
        shm_option, ulimit_options, user_option,
        port_option,
        docker_image, " ".join(command)
    )
    print(final_command)
    return subprocess.check_call(final_command, stdout=sys.stderr, shell=True)


def parse_cli_args(args=None):
    """Parse run container command line."""
    parser = argparse.ArgumentParser(prog="tao_pt", description="Tool to run the pytorch container.", add_help=True)

    parser.add_argument(
        "--gpus", default="all", type=str, help="Comma separated GPU indices to be exposed to the docker."
    )

    parser.add_argument("--volume", action="append", type=str, default=[], help="Volumes to bind.")

    parser.add_argument("--env", action="append", type=str, default=[], help="Environment variables to bind.")

    parser.add_argument("--no-tty", dest="tty", action="store_false")
    parser.set_defaults(tty=True)

    parser.add_argument("--mounts_file", help="Path to the mounts file.", default="", type=str)
    parser.add_argument("--shm_size", help="Shared memory size for docker", default="16G", type=str)
    parser.add_argument("--run_as_user", help="Flag to run as user", action="store_true", default=False)

    parser.add_argument("--tag", help="The tag value for the local dev docker.", default=None, type=str)
    parser.add_argument("--cached", help="The cached value for the local dev docker.", default=None, type=str)
    parser.add_argument("--ulimit", action='append', help="Docker ulimits for the host machine.")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Port mapping (e.g. 8889:8889)."
    )
    args = vars(parser.parse_args(args))
    return args


def main(cl_args=None):
    """Start docker container."""
    if "--" in cl_args:
        index = cl_args.index("--")
        # Split args to the tao docker wrapper and the command to be run inside the docker.
        tao_pt_args = cl_args[:index]
        command_args = cl_args[index + 1:]
    else:
        tao_pt_args = cl_args
        command_args = ""

    # parse command line args.
    args = parse_cli_args(tao_pt_args)
    docker_image = "{}/{}".format(DOCKER_REGISTRY, DOCKER_REPOSITORY)
    if args["tag"] is not None:
        docker_image = "{}:{}".format(docker_image, args["tag"])
    if not check_image_exists(docker_image):
        assert pull_base_container(docker_image), "The base container doesn't exist locally and " "the pull failed."
    try:
        instantiate_dev_docker(
            args["gpus"], args["mounts_file"],
            args["volume"], args["env"],
            args["tag"], command_args,
            args["ulimit"], args["shm_size"],
            args["run_as_user"],
            args['port'],
            args['tty'],
            args['cached']
        )
    except subprocess.CalledProcessError:
        # Do nothing - the errors are printed in entrypoint launch.
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
