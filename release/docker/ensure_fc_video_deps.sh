#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Make the release image's FC video dependencies match docker/requirements-pip.txt,
# whatever base image the release build was handed.
#
# Why this exists: release/docker/Dockerfile only *asserted* av/onnxscript, trusting
# the base image to carry them. The release CI does not honour the Dockerfile's
# X86_DIGEST/ARM64_DIGEST defaults -- it substitutes its own (flattened) base via
# BuildKit --build-context -- so a stale base pin in CI shipped a release image with
# no PyAV and onnxscript 0.6.2, and the assert killed the nightly
# (ModuleNotFoundError: No module named 'av'). A local `deploy.sh --build` could not
# reproduce it because it consumed the Dockerfile's (correct) digest.
#
# Rules (TAO-2183 / FF-4): PyAV must be built from source against the restricted,
# codec-disabled FFmpeg under ${FFMPEG_PREFIX} (default /usr/local). The PyPI wheel is
# never acceptable -- av.libs/ bundles libx264/libx265. Decord is forbidden for the
# same reason. If the base has no restricted FFmpeg this script fails the build
# instead of "fixing" it with a wheel.
#
# Usage:
#   ensure_fc_video_deps.sh [REQUIREMENTS_FILE]      # default: docker/requirements-pip.txt next to this script
#   ensure_fc_video_deps.sh --print-pins [FILE]       # print resolved pins and exit (no changes)
set -euo pipefail

FFMPEG_PREFIX="${FFMPEG_PREFIX:-/usr/local}"
# PyAV 17.1.0's pyio callback declarations do not compile with Cython 3.2.x; the base
# ships 3.2.4. Constrain only the isolated build env (same as docker/Dockerfile).
PYAV_BUILD_CYTHON="${PYAV_BUILD_CYTHON:-cython==3.1.6}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
print_only=0
if [[ "${1:-}" == "--print-pins" ]]; then
    print_only=1
    shift
fi
requirements="${1:-${script_dir}/requirements-pip.txt}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -r "${requirements}" ]] || die "requirements file not found: ${requirements}"

pin_of() {
    # Exact pin (name==version) for package $1 from the requirements file; strip comments.
    local name="$1" line
    line="$(grep -E "^${name}==" "${requirements}" | head -n 1 | sed -E 's/[[:space:]]+#.*$//; s/[[:space:]]+$//')"
    [[ -n "${line}" ]] || die "no exact pin '${name}==<version>' in ${requirements}"
    printf '%s\n' "${line}"
}

av_requirement="$(pin_of av)"
onnxscript_requirement="$(pin_of onnxscript)"
av_version="${av_requirement#av==}"
onnxscript_version="${onnxscript_requirement#onnxscript==}"

if (( print_only )); then
    echo "av=${av_version} onnxscript=${onnxscript_version}"
    exit 0
fi

site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

installed_version() {
    python - "$1" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    print("")
PY
}

av_links_restricted_ffmpeg() {
    # True when the installed PyAV extension resolves libavcodec from ${FFMPEG_PREFIX}/lib.
    python - "${FFMPEG_PREFIX}" <<'PY'
import glob, os, subprocess, sys
prefix = sys.argv[1]
try:
    import av  # noqa: F401
except Exception:
    sys.exit(1)
ext = glob.glob(os.path.join(os.path.dirname(av.__file__), "container", "core*.so"))
if not ext:
    sys.exit(1)
out = subprocess.run(["ldd", ext[0]], capture_output=True, text=True).stdout
line = next((l for l in out.splitlines() if "libavcodec" in l), "")
sys.exit(0 if os.path.join(prefix, "lib") + "/" in line else 1)
PY
}

# ---- Decord: forbidden (bundles its own libav* + libx264 under decord.libs/) ----
if python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("decord") else 1)'; then
    echo "Decord found in the base image; removing (this image decodes video through PyAV)."
    pip uninstall -y decord
fi

# ---- PyAV: exact pin, source-built against the restricted FFmpeg ----
current_av="$(installed_version av)"
if [[ "${current_av}" == "${av_version}" ]] && [[ ! -d "${site_packages}/av.libs" ]] && av_links_restricted_ffmpeg; then
    echo "PyAV ${current_av} already present, linked against ${FFMPEG_PREFIX}/lib -- keeping it."
else
    echo "PyAV state: installed='${current_av:-<absent>}' wanted='${av_version}' -> source-building against ${FFMPEG_PREFIX}."
    export PKG_CONFIG_PATH="${FFMPEG_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
    export LD_LIBRARY_PATH="${FFMPEG_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    pkg-config --exists libavcodec libavformat libavutil libswscale \
        || die "no restricted FFmpeg under ${FFMPEG_PREFIX} (libavcodec.pc missing). Refusing to install the PyPI PyAV wheel (bundles libx264/libx265). Rebuild the base image from docker/Dockerfile and update docker/manifest.json."
    ffmpeg_prefix_seen="$(pkg-config --variable=prefix libavcodec)"
    [[ "${ffmpeg_prefix_seen}" == "${FFMPEG_PREFIX}" ]] \
        || die "pkg-config resolved libavcodec from '${ffmpeg_prefix_seen}', expected '${FFMPEG_PREFIX}'."
    constraints="$(mktemp)"
    printf '%s\n' "${PYAV_BUILD_CYTHON}" > "${constraints}"
    pip uninstall -y av || true
    pip install --no-cache-dir --force-reinstall --no-deps --no-binary=av \
        --build-constraint "${constraints}" "${av_requirement}"
    rm -f "${constraints}"
fi

# ---- ONNXScript: exact pin (pure Python; its deps -- onnx, numpy, ml_dtypes, typing_extensions -- ship in the base) ----
current_onnxscript="$(installed_version onnxscript)"
if [[ "${current_onnxscript}" == "${onnxscript_version}" ]]; then
    echo "onnxscript ${current_onnxscript} already present -- keeping it."
else
    echo "onnxscript state: installed='${current_onnxscript:-<absent>}' wanted='${onnxscript_version}' -> installing pinned version."
    pip install --no-cache-dir --no-deps "${onnxscript_requirement}"
fi

# ---- Verify: exact versions, no decord, no bundled codec libraries, PyAV on the restricted FFmpeg ----
bad_libs="$(find "${site_packages}" \
    \( -path '*/av.libs/*' -o -path '*/decord.libs/*' \
       -o -name 'libx264*.so*' -o -name 'libx265*.so*' -o -name 'libopenh264*.so*' \) -print)"
[[ -z "${bad_libs}" ]] || die "forbidden bundled video codec libraries in ${site_packages}:"$'\n'"${bad_libs}"
av_links_restricted_ffmpeg || die "PyAV is not linked against ${FFMPEG_PREFIX}/lib/libavcodec."

AV_VERSION_WANTED="${av_version}" ONNXSCRIPT_VERSION_WANTED="${onnxscript_version}" python - <<'PY'
import importlib.util, os
from importlib.metadata import version
import av, onnxscript
assert version("av") == os.environ["AV_VERSION_WANTED"], version("av")
assert version("onnxscript") == os.environ["ONNXSCRIPT_VERSION_WANTED"], version("onnxscript")
assert importlib.util.find_spec("decord") is None
print(f"FC video dependencies: av={av.__version__}, onnxscript={onnxscript.__version__}, decord=absent")
PY
