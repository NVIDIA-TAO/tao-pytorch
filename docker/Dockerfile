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

# ---------------------------------------------------------------------------
# Codec-disabled, LGPL-only FFmpeg (TAO-2183 / FF-4) — identical allow-list to the
# tao-dataservices / tao-deploy images. The system `ffmpeg` apt package (Ubuntu GPL build,
# bundles libx264/libx265) was removed from requirements-apt.txt. NO software H.264/H.265/AAC
# codec ships: H.264 *input* decode is hardware-only via h264_cuvid (NVDEC); royalty-free VP9
# (libvpx-vp9) + mjpeg encode. libavformat/libavutil retained per Legal guidance (2026-06-12).
# NVIDIA_DRIVER_CAPABILITIES must include `video` for the h264_cuvid runtime path.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
# If imageio-ffmpeg is ever pulled in, force it to use our codec-disabled ffmpeg rather than
# downloading its own bundled GPL build with libx264 (TAO-2183 / FF-5, plan B.5).
ENV IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg
ARG FFMPEG_VERSION=8.1
ARG FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
# Mirror build ARGs into ENV so the quoted (<<'EOF') heredoc reads them from the shell env.
ENV FFMPEG_VERSION=${FFMPEG_VERSION} \
    FFMPEG_SHA256=${FFMPEG_SHA256} \
    NV_CODEC_HEADERS_VERSION=${NV_CODEC_HEADERS_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential yasm nasm pkg-config git curl ca-certificates xz-utils libvpx-dev libgnutls28-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build
git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" https://github.com/FFmpeg/nv-codec-headers.git
tar -C /tmp/ffmpeg-build --exclude='.git' \
    -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers
make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install
curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz
echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -
cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"
tar -xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"
./configure \
  --prefix=/usr/local \
  --disable-debug --disable-doc --disable-static --enable-shared --enable-pic \
  --enable-gnutls \
  --disable-nonfree --disable-gpl \
  --disable-encoders --disable-decoders --disable-everything \
  --enable-libvpx \
  --enable-encoder=libvpx_vp9 --enable-encoder=mjpeg \
  --enable-decoder=h264_cuvid --enable-decoder=vp8 --enable-decoder=vp9 --enable-decoder=mjpeg \
  --enable-nvenc --enable-nvdec \
  --enable-protocol=file \
  --enable-muxer=webm --enable-muxer=image2 \
  --enable-demuxer=mov --enable-demuxer=matroska --enable-demuxer=image2 \
  --enable-parser=h264 --enable-parser=vp8 --enable-parser=vp9 \
  --enable-bsf=h264_mp4toannexb --enable-bsf=vp9_superframe \
  --enable-filter=fps --enable-filter=scale --enable-filter=format \
  --extra-cflags="-I/usr/local/include" --extra-ldflags="-L/usr/local/lib"
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/ffmpeg-build
# ---- build-time codec assertions (fail the build, not the OSRB audit) ----
ffmpeg -hide_banner -version | head -n 3
! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"
for forbidden in libx264 libopenh264 libx265 h264 hevc aac; do
  if ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden encoder '$forbidden' leaked into the image (GPL/patent-encumbered)" >&2; exit 1; fi
done
for enc in libvpx-vp9 mjpeg; do
  ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$enc" >/dev/null \
    || { echo "ERROR: expected encoder '$enc' to be enabled" >&2; exit 1; }
done
for dec in h264_cuvid vp8 vp9 mjpeg; do
  ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$dec" >/dev/null \
    || { echo "ERROR: expected decoder '$dec' to be enabled" >&2; exit 1; }
done
h264_decoders="$(ffmpeg -hide_banner -decoders 2>/dev/null | awk '$2 ~ /^h264/ {print $2}' | sort -u)"
forbidden_h264_decoders="$(printf '%s\n' "${h264_decoders}" | awk 'NF && $0 != "h264_cuvid"')"
if [[ -n "${forbidden_h264_decoders}" ]]; then
  echo "ERROR: forbidden H.264 decoder(s) leaked into the image: ${forbidden_h264_decoders}" >&2; exit 1; fi
for forbidden in hevc aac; do
  if ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden decoder '$forbidden' leaked into the image" >&2; exit 1; fi
done
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx webm   >/dev/null || { echo "ERROR: muxer webm missing"   >&2; exit 1; }
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx image2 >/dev/null || { echo "ERROR: muxer image2 missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)matroska(,|$)' >/dev/null || { echo "ERROR: demuxer matroska,webm missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)mov(,|$)'      >/dev/null || { echo "ERROR: demuxer mov missing" >&2; exit 1; }
ffmpeg -hide_banner -protocols 2>/dev/null | awk '{print $1}' | grep -Fx file             >/dev/null || { echo "ERROR: protocol file missing" >&2; exit 1; }
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx h264_mp4toannexb  >/dev/null || { echo "ERROR: bsf h264_mp4toannexb missing" >&2; exit 1; }
# vp9_superframe is required to mux libvpx-vp9 output into WebM/Matroska (the FF-1 path).
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx vp9_superframe    >/dev/null || { echo "ERROR: bsf vp9_superframe missing (needed to mux VP9 into WebM)" >&2; exit 1; }
for filt in fps scale format; do
  ffmpeg -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -Fx "$filt" >/dev/null \
    || { echo "ERROR: filter '$filt' missing" >&2; exit 1; }
done
echo "FFmpeg codec allow-list verified: no libx264/libx265/openh264/SW-h264/hevc/aac; VP9+mjpeg encode, h264_cuvid+vp8/vp9/mjpeg decode."
EOF
RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*

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

# ---------------------------------------------------------------------------
# spatialai_data_utils (Sparse4D eval), installed --no-deps  (TAO-2183 / FF-5)
# ---------------------------------------------------------------------------
# 2.0.0 (not 1.0.0): the index's spatialai_data_utils==1.0.0 wheel was re-published and now
# omits visualization.COLOR_MAP (and the extrinsic2pyramid module), so it no longer matches
# the version the shipped release/7.0.1 image was built against and breaks the Sparse4D
# import. 2.0.0 has the full API (all Sparse4D imports resolve). Installed --no-deps (plus its
# numpy-2-compatible runtime deps) so its nuscenes-devkit dependency does NOT downgrade the
# base numpy 2.x — nuscenes' numpy<2 pin is overly conservative and it runs fine on numpy 2
# (verified). Placed late (after the heavy source builds) so it doesn't invalidate their cache.
RUN pip install --no-deps \
        --extra-index-url https://edge.urm.nvidia.com/artifactory/api/pypi/sw-metropolis-pypi/simple \
        spatialai_data_utils==2.0.0 nuscenes-devkit==1.2.0 \
 && pip install --no-build-isolation \
        motmetrics==1.4.0 descartes==1.1.0 tabulate==0.9.0 jsonschema==4.23.0 \
        pytz==2024.2 parameterized==0.9.0 lap==0.5.12 fvcore==0.1.5.post20221221 yacs==0.1.8

# ---------------------------------------------------------------------------
# OpenCV rebuilt WITH_FFMPEG=OFF (TAO-2183 / FF-3)
# ---------------------------------------------------------------------------
# Runs LAST among the pip layers so it overrides the opencv-python-headless 4.13.0.92 pulled
# by requirements-pip.txt, whose bundled libav* (inside opencv_python_headless.libs/) carries
# the royalty-bearing H.264/H.265/AAC codecs. Uninstall every opencv flavor, scrub leftover
# cv2/ + .libs/, then build from sdist with codec backends disabled. The embedded python check
# fails the build if cv2 reports FFMPEG: YES or links any libav*.
#
# VERSION NOTE: base/pinned wheel 4.13.0.92 is wheel-only (no public sdist), so we build the
# latest source-available release 4.12.0.88, mirroring paidf-auto-labeling. Immaterial API
# delta for TAO's cv2 usage; revisit if DL-frameworks ships a libav-free 4.13 wheel (FF-3).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake libjpeg-dev libpng-dev libtiff-dev libwebp-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
# Neutralize any base global PIP_CONSTRAINT; every install below is exact-pinned.
export PIP_CONSTRAINT=""
python -m pip uninstall -y \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "${SP}"/cv2 \
       "${SP}"/opencv_python*.libs \
       "${SP}"/opencv_python*.dist-info \
       "${SP}"/opencv_contrib_python*.dist-info \
       "${SP}"/opencv_contrib_python*.libs
# --ignore-installed: the base may ship some build tools (e.g. wheel) via apt/debian with no
# RECORD file, which pip cannot uninstall when upgrading; install over them instead.
python -m pip install --no-cache-dir --ignore-installed \
    "setuptools>=78.1.1" "wheel>=0.46.2" "scikit-build>=0.18,<1" "cmake>=3.27,<4"
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1
export CMAKE_ARGS="\
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF"
python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=opencv-python-headless \
    "opencv-python-headless==4.12.0.88"
python - <<'PY'
import re, subprocess
from pathlib import Path
import cv2
info = cv2.getBuildInformation()
if re.search(r"FFMPEG:\s+YES", info):
    raise SystemExit("ERROR: OpenCV unexpectedly built WITH FFmpeg")
cv2_pkg_dir = Path(cv2.__file__).resolve().parent
cv2_sos = sorted(cv2_pkg_dir.rglob("cv2*.so"))
if not cv2_sos:
    raise SystemExit(f"ERROR: could not locate cv2 compiled extension under {cv2_pkg_dir}")
forbidden = ("libavcodec", "libavdevice", "libavfilter",
             "libavformat", "libavutil", "libswresample", "libswscale")
for cv2_so in cv2_sos:
    ldd = subprocess.check_output(["ldd", str(cv2_so)], text=True)
    for lib in forbidden:
        if lib in ldd:
            raise SystemExit(f"ERROR: {cv2_so.name} still links {lib}")
print("opencv", cv2.__version__, "FFmpeg disabled, no libav* linkage in", [p.name for p in cv2_sos])
PY
leftovers="$(find "${SP}" \( \
    -path '*opencv_python*.libs/libavcodec*' -o \
    -path '*opencv_python*.libs/libavdevice*' -o \
    -path '*opencv_python*.libs/libavfilter*' -o \
    -path '*opencv_python*.libs/libavformat*' -o \
    -path '*opencv_python*.libs/libavutil*' -o \
    -path '*opencv_python*.libs/libswresample*' -o \
    -path '*opencv_python*.libs/libswscale*' \
\) -print 2>/dev/null || true)"
if [ -n "${leftovers}" ]; then echo "ERROR: bundled libav* leaked back in:"; echo "${leftovers}"; exit 1; fi
EOF
# NOTE: build-essential/cmake intentionally NOT purged (pytorch:26.03 base needs a runtime
# toolchain for torch.compile / extension JIT). Only build-only codec assemblers (yasm/nasm)
# were purged after the ffmpeg layer above.

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