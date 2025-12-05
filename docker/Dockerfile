ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${PYTORCH_BASE_IMAGE}

# Docker buildx provides these automatically for cross-platform detection
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
COPY docker/requirements-apt.txt requirements-apt.txt
RUN apt-get upgrade && apt-get update && \
  xargs apt-get install -y < requirements-apt.txt && \
  rm requirements-apt.txt && \
  rm -rf /var/lib/apt/lists/*

# TAO-CORE microservice dependencies
WORKDIR /workspace/k8s
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    pip install --disable-pip-version-check -r requirements.txt && \
    python setup.py install


COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install --upgrade pip \
  && pip install -r requirements-pip.txt \
  && rm requirements-pip.txt


# Set platform-specific library path using uname
RUN ARCH_LIB_PATH="/usr/lib/$(uname -m)-linux-gnu" && \
    echo "export LD_LIBRARY_PATH=${ARCH_LIB_PATH}:\${LD_LIBRARY_PATH}" >> /etc/bash.bashrc
ENV BASH_ENV=/etc/bash.bashrc

COPY docker/requirements-pip-pytorch.txt requirements-pip-pytorch.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-pytorch.txt \
  && rm requirements-pip-pytorch.txt

COPY docker/requirements-pip-odise.txt requirements-pip-odise.txt
RUN pip install --ignore-installed --no-deps -r requirements-pip-odise.txt \
  && rm requirements-pip-odise.txt

COPY docker/requirements-pip-quantization.txt requirements-pip-quantization.txt
RUN pip install --upgrade --no-deps -r requirements-pip-quantization.txt \
  && rm requirements-pip-quantization.txt

################## Build onnxruntime-gpu ##################
# We need to build onnxruntime-gpu from source because the official wheel is not available for CUDA 13.0
# TODO: Update to the official wheel when it is available for CUDA 13.0
ARG ONNXRUNTIME_VERSION=1.23.0
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BUILD_CONFIG=Release
ARG ONNXRUNTIME_BUILD_PARALLEL=8

# Adding support for Hopper (9.0/9.0a), Blackwell (10.0), Thor (11.0), and future architectures
# Used by both ONNX Runtime and xformers builds
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 9.0a 10.0 11.0 12.0+PTX"
ENV GPU_ARCHS="75 80 86 90 90a 100 110 120"

# Ensure CUDA headers (including CCCL) are found for all compilations
# This is critical for cross-platform builds where target-specific paths differ
# (e.g., x86_64-linux vs sbsa-linux on ARM)
ENV CPATH="/usr/local/cuda/include/cccl:${CPATH}"

# Clone and build ONNX Runtime
WORKDIR /workspace/onnxruntime_build
RUN git clone -b rel-${ONNXRUNTIME_VERSION} --recursive ${ONNXRUNTIME_REPO} onnxruntime && \
    find /workspace/onnxruntime_build -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/onnxruntime_build -name ".git" -type f -exec rm -f {} + 2>/dev/null || true && \
    find /workspace/onnxruntime_build -name ".github" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /workspace/onnxruntime_build -name ".gitlab" -type d -exec rm -rf {} + 2>/dev/null || true

WORKDIR /workspace/onnxruntime_build/onnxruntime

# Build ONNX Runtime
RUN CUDA_ARCHS=$(echo ${GPU_ARCHS} | sed 's/ /;/g' | sed 's/\([0-9]\+a\?\)/\1-real/g') && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then \
        CUDA_TARGET="sbsa-linux"; \
    else \
        CUDA_TARGET="${ARCH}-linux"; \
    fi && \
    CUDA_INCLUDE_DIRS="/usr/local/cuda/include/cccl;/usr/local/cuda/targets/${CUDA_TARGET}/include;/usr/local/cuda/targets/${CUDA_TARGET}/include/cccl" && \
    ./build.sh \
        --config ${ONNXRUNTIME_BUILD_CONFIG} \
        --parallel ${ONNXRUNTIME_BUILD_PARALLEL} \
        --skip_submodule_sync \
        --build_shared_lib \
        --compile_no_warning_as_error \
        --build_dir /workspace/onnxruntime_build/build \
        --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        --cmake_extra_defines CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES="${CUDA_INCLUDE_DIRS}" \
        --build_wheel \
        --update \
        --build \
        --use_cuda --cuda_home "/usr/local/cuda" \
        --cudnn_home "/usr" \
        --use_tensorrt --use_tensorrt_builtin_parser \
        --tensorrt_home "/usr/src/tensorrt" \
        --allow_running_as_root

# Install the built wheel
RUN pip install --no-deps /workspace/onnxruntime_build/build/${ONNXRUNTIME_BUILD_CONFIG}/dist/onnxruntime_gpu-*.whl

# Clean up build artifacts including .git folders from submodules
RUN rm -rf /workspace/onnxruntime_build

WORKDIR /workspace

##################
## build xformers
##################

# Xformers commit hash
ARG XFORMERS_COMMIT_HASH=c0eb310a0afbf782b8947501e4f835e30b5a47b1

# flash attention issue: https://nvbugspro.nvidia.com/bug/5440235
ENV NVCC_FLAGS="--ptxas-options=-O2 --ptxas-options=-v"

COPY docker/xformers xformers_src

RUN echo "======Start building xformers from vllm gitlab repository======" && \
  find xformers_src -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
  find xformers_src -name ".git" -type f -exec rm -f {} + 2>/dev/null || true && \
  find xformers_src -name ".github" -type d -exec rm -rf {} + 2>/dev/null || true && \
  find xformers_src -name ".gitlab" -type d -exec rm -rf {} + 2>/dev/null || true && \
  cd xformers_src && \
  echo "======Calculating MAX_JOBS to build xformers parallelly======" && \
  # TLT-5330: 90 GB overhead, 20 GB/job, 4 job min and ${max_cpu} job max
  total_mem_gb=$(grep MemFree /proc/meminfo | awk '{print int($2 / 1024 / 1024)}'); \
  max_jobs=$(( (total_mem_gb - 90) / 20 )); \
  max_jobs=$(( max_jobs < 4 ? 4 : max_jobs )); \
  max_jobs=$(( max_jobs > $(nproc) ? $(nproc) : max_jobs )); \
  echo "======Using MAX_JOBS=${max_jobs} to build xformers======" && \
  MAX_JOBS=${max_jobs} python setup.py bdist_wheel -d build/ && \
  pip install --no-deps build/xformers*.whl && \
  cd ../ && \
  rm -rf xformers_src && \
  echo "======Xformers built and installed successfully======"

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} taotoolkituser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash taotoolkituser
RUN usermod -aG sudo taotoolkituser
RUN echo 'taotoolkituser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown taotoolkituser /workspace

WORKDIR /tao-pt

# Security: Remove all git-related files and folders from the image
# This addresses security scan failures (credentials, internal URLs) and reduces image size
RUN find / -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true && \
    find / -type f -name ".git" -exec rm -f {} + 2>/dev/null || true && \
    find / -type f -name ".gitignore" -exec rm -f {} + 2>/dev/null || true && \
    find / -type d -name ".github" -exec rm -rf {} + 2>/dev/null || true && \
    find / -type d -name ".gitlab" -exec rm -rf {} + 2>/dev/null || true && \
    find / -type f -name ".gitlab-ci.yml" -exec rm -f {} + 2>/dev/null || true && \
    find / -type f -name "gitlab-ci.yml" -exec rm -f {} + 2>/dev/null || true

CMD [ "/bin/bash" ]