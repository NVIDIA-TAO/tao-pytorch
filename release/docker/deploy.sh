#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail
# cd "$( dirname "${BASH_SOURCE[0]}" )"

registry="nvcr.io"
pytorch_version="2.1.0"
# Keep in sync with release/python/version.py and release/docker/Dockerfile (TAO_TOOLKIT_VERSION).
tao_version="7.0.1"
repository="nvstaging/tao/tao-toolkit-pyt"
build_id="01"
tag="v${tao_version}-pyt${pytorch_version}-py3-${build_id}"

# Build parameters.
SKIP_SUBMODULE_INIT="1"
BUILD_DOCKER="0"
BUILD_WHEEL="0"
PUSH_DOCKER="0"
FORCE="0"
USE_CUDA_CACHE="0"

# Required for tao-core and tao-converter since they are submodules.
if [ $SKIP_SUBMODULE_INIT = "0" ]; then
    echo "Updating submodules ..."
    git submodule update --init --recursive
fi

wheel_dir=${NV_TAO_PYTORCH_TOP}/dist

# Setting up the environment.
source $NV_TAO_PYTORCH_TOP/scripts/envsetup.sh

# Parse command line.
while [[ $# -gt 0 ]]
    do
    key="$1"

    case $key in
        -b|--build)
        BUILD_DOCKER="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -w|--wheel)
        BUILD_WHEEL="1"
        RUN_DOCKER="0"
        shift # past argument
        ;;
        -p|--push)
        PUSH_DOCKER="1"
        shift # past argument
        ;;
        -f|--force)
        FORCE=1
        shift
        ;;
        --use-cache)
        USE_CUDA_CACHE="1"
        shift
        ;;
        -r|--run)
        RUN_DOCKER="1"
        BUILD_DOCKER="0"
        FORCE="0"
        PUSH_DOCKER="0"
        shift # past argument
        ;;
        --default)
        BUILD_DOCKER="0"
        RUN_DOCKER="1"
        FORCE="0"
        PUSH_DOCKER="0"
        shift # past argument
        ;;
        *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    esac
done


if [ $BUILD_DOCKER = "1" ]; then
    echo "Building base docker ..."
    if [ $FORCE = "1" ]; then
        echo "Forcing docker build without cache ..."
        NO_CACHE="--no-cache"
    else
        NO_CACHE=""
    fi
    if [ $BUILD_WHEEL = "1" ]; then
        if [ ! -d ${wheel_dir} ]; then
          mkdir -p $wheel_dir
        fi
        echo "Building source code wheel ..."
        if [ "$USE_CUDA_CACHE" = "1" ] && "$NV_TAO_PYTORCH_TOP/scripts/cuda_cache.sh" check; then
            echo "Restoring cached CUDA build artifacts ..."
            cache_build_succeeded="0"
            if "$NV_TAO_PYTORCH_TOP/scripts/cuda_cache.sh" restore \
                && tao_pt --no-tty -- python setup.py build_py \
                && tao_pt --no-tty -- python setup.py bdist_wheel --skip-build \
                && ls ${wheel_dir}/*.whl >/dev/null 2>&1; then
                cache_build_succeeded="1"
            fi
            if [ "$cache_build_succeeded" = "0" ]; then
                echo "WARNING: cached CUDA build did not produce a wheel; rebuilding from source ..."
                tao_pt --no-tty -- python setup.py bdist_wheel
                "$NV_TAO_PYTORCH_TOP/scripts/cuda_cache.sh" save
            fi
        else
            tao_pt --no-tty -- python setup.py bdist_wheel
            if [ "$USE_CUDA_CACHE" = "1" ]; then
                "$NV_TAO_PYTORCH_TOP/scripts/cuda_cache.sh" save
            fi
        fi
        # tao_pt swallows the container's exit code (see runner/tao_pt.py), so verify the
        # wheel was actually produced before trying to COPY it into the image.
        if ! ls ${wheel_dir}/*.whl >/dev/null 2>&1; then
            echo "ERROR: wheel build produced no artifact in ${wheel_dir}. Aborting." >&2
            exit 1
        fi
    else
        echo "Skipping wheel builds ..."
    fi

    docker build --pull -f $NV_TAO_PYTORCH_TOP/release/docker/Dockerfile -t $registry/$repository:$tag $NO_CACHE --network=host $NV_TAO_PYTORCH_TOP/.

    if [ $PUSH_DOCKER = "1" ]; then
        echo "Pusing docker ..."
        docker push $registry/$repository:$tag
    else
        echo "Skip pushing docker ..."
    fi

    if [ $BUILD_WHEEL = "1" ]; then
        echo "Cleaning wheels ..."
        # running cleanup
        tao_pt -- bash -c "'rm -rf *.egg-info'"
        tao_pt -- bash -c "'rm -rf build/ dist/ *_build '"
    else
        echo "Skipping wheel cleaning ..."
    fi
elif [ "$RUN_DOCKER" = "1" ]; then
    echo "Running docker interactively..."
    docker run --gpus all -v $HOME/tlt-experiments:/workspace/tlt-experiments \
                          --network=host \
                          --shm-size=30g \
                          --ulimit memlock=-1 \
                          --ulimit stack=67108864 \
                          --rm -it $registry/$repository:$tag /bin/bash
else
    echo "Usage: ./deploy.sh [--build] [--wheel] [--run] [--push] [--default] [--use-cache]"
fi
