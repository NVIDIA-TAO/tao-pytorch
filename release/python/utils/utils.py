# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper utils for packaging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import compileall
import glob
import os
import setuptools

from torch.utils.cpp_extension import CUDAExtension

# Rename all .py files to .py_tmp temporarily.
ignore_list = ['__init__.py', '__version__.py']

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))


def up_directory(dir_path, n=1):
    """Go up n directories from dir_path."""
    dir_up = dir_path
    for _ in range(n):
        dir_up = os.path.split(dir_up)[0]
    return dir_up


TOP_LEVEL_DIR = up_directory(LOCAL_DIR, 3)

def remove_prefix(dir_path):
    """Remove a certain prefix from path."""
    max_path = 8
    prefix = dir_path
    while max_path > 0:
        prefix = os.path.split(prefix)[0]
        if prefix.endswith('ai_infra'):
            return dir_path[len(prefix) + 1:]
        max_path -= 1
    return dir_path


def get_subdirs(path):
    """Get all subdirs of given path."""
    dirs = os.walk(path)
    return [remove_prefix(x[0]) for x in dirs]


def rename_py_files(path, ext, new_ext, ignore_files):
    """Rename all .ext files in a path to .new_ext except __init__ files."""
    files = glob.glob(path + '/*' + ext)
    for ignore_file in ignore_files:
        files = [f for f in files if ignore_file not in f]

    for filename in files:
        os.rename(filename, filename.replace(ext, new_ext))


def get_version_details():
    """Simple function to get packages for setup.py."""
    # Define env paths.
    LAUNCHER_SDK_PATH = os.path.join(TOP_LEVEL_DIR, "release/python/") 
    # Get current __version__.
    version_locals = {}
    with open(os.path.join(LAUNCHER_SDK_PATH, 'version.py')) as version_file:
        exec(version_file.read(), {}, version_locals)

    return  version_locals


def cleanup():
    """Cleanup directories after the build process."""
    req_subdirs = get_subdirs(TOP_LEVEL_DIR)
    # Cleanup. Rename all .py_tmp files back to .py and delete pyc files
    for dir_path in req_subdirs:
        dir_path = os.path.join(TOP_LEVEL_DIR, dir_path)
        # TODO: @vpraveen Think about removing python files before the final
        # release.
        rename_py_files(dir_path, '.py_tmp', '.py', ignore_list)
        pyc_list = glob.glob(dir_path + '/*.pyc')
        for pyc_file in pyc_list:
            os.remove(pyc_file)


def make_cuda_ext(name, module, sources, include_dirs=None, define_macros=None, extra_flags=None):
    """Build cuda extensions for custom ops.

    Args:
        name (str): Name of the op.
        module (str): Name of the module with the op.
        source (list): List of source files.
        extra_flags (dict): Any extra compile flags.

    Returns
        cuda_ext (torch.utils.cpp_extension.CUDAExtension): Cuda extension for wheeling.
    """
    kwargs = {"extra_compile_args": extra_flags}
    if include_dirs:
        kwargs["include_dirs"] = [
            os.path.join(os.path.relpath(TOP_LEVEL_DIR), *module.split('.'), dir)
            for dir in include_dirs
        ]
    if define_macros:
        kwargs["define_macros"] = define_macros

    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[
            os.path.join(os.path.relpath(TOP_LEVEL_DIR), *module.split('.'), src)
            for src in sources
        ],
        **kwargs,
    )
    return cuda_ext


def get_extra_compile_args():
    """Function to get extra compile arguments.

    Returns:
        extra_compile_args (dict): Dictionary of compile flags.
    """
    extra_compile_args = {"cxx": []}
    extra_compile_args["nvcc"] = [
        "-DCUDA_HAS_FP16=1",
        "-DCUTE_ARCH_MMA_SM90A_ENABLED=1",  # Enable SM90A architecture support
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    return extra_compile_args


def find_packages(package_name):
    """List of packages.

    Args:
        package_name (str): Name of the package.

    Returns:
        packages (list): List of packages.
    """
    packages = setuptools.find_packages(package_name)
    packages = [f"{package_name}.{f}" for f in packages]
    packages.append(package_name)
    return packages
