#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Description: Script responsible for generation of an obf_src wheel using pyarmor package.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/tao-pt' || REPO_ROOT="${WORKSPACE}"
echo "Building from ${REPO_ROOT}"

echo "Registering pyarmor"
pyarmor -d reg ${REPO_ROOT}/release/docker/pyarmor-regfile-1219.zip || exit $?

echo "Clearing build and dists"
python ${REPO_ROOT}/setup.py clean --all
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Obfuscating the code using pyarmor"
# This makes sure the non-py files are retained.
pyarmor cfg data_files=*
pyarmor -d gen --recursive --output /obf_src/ ${REPO_ROOT}/nvidia_tao_pytorch/ || exit $?

echo "Migrating codebase"
# Move sources to orig_src
rm -rf /orig_src
mkdir /orig_src
mv ${REPO_ROOT}/nvidia_tao_pytorch/* /orig_src/

# Move obf_src files to src
mv /obf_src/* ${REPO_ROOT}/

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?

echo "Restoring the original project structure"
# Move the obf_src files.
rm -rf ${REPO_ROOT}/nvidia_tao_pytorch/*

# Move back the original files
mv /orig_src/* ${REPO_ROOT}/nvidia_tao_pytorch/

# Remove the tmp folders.
rm -rf /orig_src
rm -rf /obf_src
rm -rf ${REPO_ROOT}/pyarmor_runtime_001219
