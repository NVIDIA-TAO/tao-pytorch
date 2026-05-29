ARG PYTORCH_BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3
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

COPY docker/requirements-pip.txt requirements-pip.txt
RUN pip install --upgrade pip \
  && pip install --no-build-isolation -r requirements-pip.txt \
  && rm requirements-pip.txt


# Set platform-specific library path using uname
RUN ARCH_LIB_PATH="/usr/lib/$(uname -m)-linux-gnu" && \
    echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:${ARCH_LIB_PATH}" >> /etc/bash.bashrc
ENV BASH_ENV=/etc/bash.bashrc

COPY docker/requirements-pip-pytorch.txt requirements-pip-pytorch.txt
RUN pip install --no-build-isolation --ignore-installed --no-deps -r requirements-pip-pytorch.txt \
  && rm requirements-pip-pytorch.txt

COPY docker/requirements-pip-odise.txt requirements-pip-odise.txt
RUN pip install --no-build-isolation --ignore-installed --no-deps -r requirements-pip-odise.txt \
  && rm requirements-pip-odise.txt

COPY docker/requirements-pip-quantization.txt requirements-pip-quantization.txt
RUN pip install --no-build-isolation --upgrade --no-deps -r requirements-pip-quantization.txt \
  && rm requirements-pip-quantization.txt

COPY docker/requirements-pip-nvpanoptix3d.txt requirements-pip-nvpanoptix3d.txt
RUN pip install meson-python meson ninja \
  && pip install --no-build-isolation --upgrade --no-deps -r requirements-pip-nvpanoptix3d.txt \
  && rm requirements-pip-nvpanoptix3d.txt

# sam3: installed --no-deps because its setup.py pins numpy<2, which collides
# with the 26.03 base (numpy 2.x) and opencv-python-headless 4.13 (numpy>=2).
# Importers in nvidia_tao_pytorch/cv/backbone_v2/sam3.py guard with try/except,
# so the install is best-effort — keep it after the bulk pip layers so a
# transient github outage doesn't invalidate them.
RUN pip install --no-build-isolation --no-deps \
        git+https://github.com/facebookresearch/sam3.git

# PyMarchingCubes: xheon's repo is unmaintained since Mar 2022 and its C
# extension uses NumPy 1.x-only constants (PyArray_DOUBLE/PyArray_ULONG) that
# NumPy 2.x removed. Patch to the NPY_* names before building.
ARG PYMARCHINGCUBES_COMMIT=a6ad0388187b50dc62032a39359463d811af6768
RUN git clone https://github.com/xheon/PyMarchingCubes.git /tmp/PyMarchingCubes \
  && cd /tmp/PyMarchingCubes \
  && git checkout ${PYMARCHINGCUBES_COMMIT} \
  && sed -i -e 's/PyArray_DOUBLE/NPY_DOUBLE/g' \
            -e 's/PyArray_ULONG/NPY_ULONG/g' \
            marching_cubes/src/pywrapper.cpp \
  && pip install --no-build-isolation --no-deps . \
  && cd / && rm -rf /tmp/PyMarchingCubes

################## Install WarpConvNet ##################
# Install PyTorch first (specify your CUDA version)
ENV CUDA=cu130

# Install core dependencies
RUN pip install --no-deps cupy-cuda13x==13.6.0  # use cupy-cuda13x for CUDA 13.x
# Build torch-scatter from source against the current torch/cuda stack.
RUN FORCE_CUDA=1 pip install --no-build-isolation --no-cache-dir --no-binary=torch-scatter torch-scatter

# Install patched WarpConvNet for CUDA 13.x
RUN git clone https://github.com/daocongtuyen2x/WarpConvNet.git
RUN cd WarpConvNet && \
    if [ -d .git ]; then git submodule sync --recursive && git submodule update --init --recursive; fi && \
    pip install --no-build-isolation .

################## Build onnxruntime-gpu ##################
# We need to build onnxruntime-gpu from source because the official wheel is not available for CUDA 13.0
# TODO: Update to the official wheel when it is available for CUDA 13.0
ARG ONNXRUNTIME_VERSION=1.25.1
ARG ONNXRUNTIME_REPO=https://github.com/microsoft/onnxruntime
ARG ONNXRUNTIME_BUILD_CONFIG=Release
ARG ONNXRUNTIME_BUILD_PARALLEL=8

# Adding support for Hopper (9.0/9.0a), Blackwell (10.0), Thor (11.0), and future architectures.
# Used by ONNX Runtime, torch-scatter, WarpConvNet, and pytorch3d (built via requirements-pip.txt).
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

# Patch CCCL device_transform.cuh for the CUDA 13.x + GCC 13 incompatibility.
# At file scope, the header declares a template specialization using a
# globally-qualified struct name:
#
#     template <typename T>
#     struct ::cuda::proclaims_copyable_arguments<…> : ::cuda::std::true_type {};
#
# GCC 13 rejects this with "global qualification of class name is invalid before
# ':' token". Wrapping the specialization in `namespace cuda { … }` removes the
# need for the leading `::cuda::` qualification and compiles cleanly.
# Mirrors docker/optimized/artifacts/Dockerfile.ort — keep both in sync.
RUN python3 <<'PY'
import re, glob
PATTERN = re.compile(
    r'template <(?:class|typename) T>\n'
    r'struct ::cuda::proclaims_copyable_arguments'
    r'<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n'
    r'\{\};'
)
REPLACEMENT = (
    'namespace cuda {\n'
    'template <typename T>\n'
    'struct proclaims_copyable_arguments'
    '<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n'
    '{};\n'
    '} // namespace cuda'
)
total = 0
for path in glob.glob('/usr/local/cuda*/**/cccl/cub/device/device_transform.cuh', recursive=True):
    with open(path) as f:
        src = f.read()
    new_src, n = PATTERN.subn(REPLACEMENT, src)
    if n:
        with open(path, 'w') as f:
            f.write(new_src)
        print(f'patched {n}x in {path}')
        total += n
    else:
        print(f'no match in {path}')
print(f'CCCL patch: {total} substitution(s) applied')
PY

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

####################
## install xformers
####################
# xformers comes from NVIDIA's internal GitLab PyPI registry. The local-version
# segment (+nv26.3.1.<pipeline>) ties the wheel's torch/CUDA ABI to the base
# image — bump in lockstep with PYTORCH_BASE_IMAGE upgrades.
ARG XFORMERS_VERSION=0.0.33+nv26.3.1.49215124
ARG XFORMERS_INDEX_URL=https://gitlab-master.nvidia.com/api/v4/projects/100660/packages/pypi/simple
RUN pip install --no-deps "xformers==${XFORMERS_VERSION}" \
    --index-url "${XFORMERS_INDEX_URL}"

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