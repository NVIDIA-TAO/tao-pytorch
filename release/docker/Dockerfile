# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:23c2385c3d9232cf0e8b4d37d13fc7e9c9db8fece5c0f6475e5f7924dc6ac06a
# Select base image based on architecture
ARG TARGETARCH # provided by Docker if using BuildKit. Else pass in explicitly.
ARG X86_DIGEST=224a161014b5c7ac864e5d1055600d4d27c08a2e23e89bc6e5102a474d2ffae2
ARG ARM64_DIGEST=2e095fb1db53458140769418dcd3c0cae46042a8554a94b5cb0bea86ec8811c0

FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:${ARM64_DIGEST} AS base-arm64
FROM nvcr.io/nvstaging/tao/tao_pytorch_base_image@sha256:${X86_DIGEST} AS base-amd64

FROM base-${TARGETARCH}

# Installing the TLT packages.
WORKDIR /opt/nvidia

# NGC Security scan related fixes.
# Low-level packages in apt are updated to resolve published security vulnerabilities ASAP
# We retrieve these updates and upgrade the critical/high vulnerability ones
RUN apt-get update && \
    apt --only-upgrade install linux-libc-dev -y && \
    apt --only-upgrade install curl -y && \
    apt --only-upgrade install libc-bin -y && \
    apt --only-upgrade install libc6-dev -y && \
    apt --only-upgrade install libc6 -y && \
    apt --only-upgrade install libc-dev-bin -y && \
    apt --only-upgrade install libcurl3-gnutls -y && \
    apt --only-upgrade install libcurl4 -y && \
    apt --only-upgrade install libcurl4-openssl-dev -y && \
    apt --only-upgrade install openssh-client -y && \
    rm -rf /opt/nvidia/entrypoint.d/*.txt
COPY release/docker/entrypoint.d/* /opt/nvidia/entrypoint.d/

# Installing TAO-Core
COPY tao-core tao-core
RUN cd tao-core && pip install --upgrade pip setuptools wheel && python setup.py bdist_wheel && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cp nvidia_tao_core/microservices/nginx.conf /etc/nginx/ && \
    cd .. && rm -rf tao-core

COPY dist/*.whl /opt/nvidia/wheels/

# Copying tlt-converter and building it in the docker.
# ENV TRT_LIB_PATH="/usr/lib/x86_64-linux-gnu"
# ENV TRT_INCLUDE_PATH="/usr/include/x86_64-linux-gnu"
# COPY tao-converter /opt/nvidia/tools
# WORKDIR /opt/nvidia/tools
# RUN make \
#     && rm -rf Makefile *.cc *.h *.o
# RUN ln -s tao-converter converter

ENV NVIDIA_PRODUCT_NAME="TAO Toolkit"
ENV TAO_TOOLKIT_VERSION="7.0.1"
ENV NVIDIA_TAO_TOOLKIT_VERSION="${TAO_TOOLKIT_VERSION}-PyTorch"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Defining the telemetry URL.
ENV TAO_TELEMETRY_SERVER="https://api.tao.ngc.nvidia.com"

EXPOSE 8000

# Microservices entrypoint
ENV FLASK_APP=nvidia_tao_core.microservices.app

ENV RUN_CLI=0

# Install the TAO wheels, fix LD_LIBRARY_PATH ordering, and strip git metadata (NGC
# security scan) in a SINGLE RUN layer. The base image already ships ~120 filesystem
# layers and overlay2 caps a container at 128, so every extra RUN/COPY layer counts;
# collapsing these three steps into one keeps the release image buildable.
RUN cd /opt/nvidia/wheels && ls ./*.whl | xargs -I'{}' python -m pip install '{}' && rm -f *.whl && \
    python -c 'import importlib.util; from importlib.metadata import version; import av, onnxscript; assert version("av") == "17.1.0"; assert version("onnxscript") == "0.7.1"; assert importlib.util.find_spec("decord") is None; print(f"FC video dependencies: av={av.__version__}, onnxscript={onnxscript.__version__}, decord=absent")' && \
    ARCH_LIB_PATH="/usr/lib/$(uname -m)-linux-gnu" && \
    sed -i "s|export LD_LIBRARY_PATH=${ARCH_LIB_PATH}:\${LD_LIBRARY_PATH}|export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${ARCH_LIB_PATH}|g" /etc/bash.bashrc && \
    { find / -type d -name ".git" -exec rm -rf {} + 2>/dev/null; \
      find / -type f -name ".git" -exec rm -f {} + 2>/dev/null; \
      find / -type f -name ".gitignore" -exec rm -f {} + 2>/dev/null; \
      find / -type d -name ".github" -exec rm -rf {} + 2>/dev/null; \
      find / -type d -name ".gitlab" -exec rm -rf {} + 2>/dev/null; \
      find / -type f -name ".gitlab-ci.yml" -exec rm -f {} + 2>/dev/null; \
      find / -type f -name "gitlab-ci.yml" -exec rm -f {} + 2>/dev/null; \
      true; }

CMD if [ "$RUN_CLI" = "1" ]; then \
        /bin/bash; \
    else \
        /bin/bash $(get-microservice-script); \
    fi
