#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Description: Script responsible for building the nvidia_tao_pytorch wheel.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/tao-pt' || REPO_ROOT="${WORKSPACE}"
echo "Building from ${REPO_ROOT}"

echo "Clearing build and dists"
python ${REPO_ROOT}/setup.py clean --all
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?
