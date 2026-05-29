#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# build.sh — driver for the optimized base-image build path.
#
#   1. Verify docker/optimized/artifacts.lock.json matches its inputs.
#   2. For each component (ort, torch_scatter, warpconvnet): query the registry
#      for the locked artifact image; build + push if absent.
#   3. Build the Tier-2 base image with Dockerfile.optimized, passing the three
#      artifact image refs as --build-args. xformers is pip-installed from the
#      internal GitLab index inside Dockerfile.optimized — no separate artifact.
#
# Usage:
#   docker/optimized/build.sh --build --x86                     # native local build, --load
#   docker/optimized/build.sh --build --x86 --push              # native build + push
#   docker/optimized/build.sh --build --arm --push              # cross-arch build + push
#   docker/optimized/build.sh --refresh-lock                    # update artifacts.lock.json
#   docker/optimized/build.sh --skip-artifacts --x86 --push     # only Tier-2 (Tier-1 images must exist)
#   docker/optimized/build.sh --component ort --x86 --push      # only one Tier-1 artifact, no Tier-2
#                                                               #   (used by Jenkins matrix cells)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCKFILE="${SCRIPT_DIR}/artifacts.lock.json"
LOCKHASH="${SCRIPT_DIR}/lockhash.sh"
DOCKERFILE_OPTIMIZED="${SCRIPT_DIR}/Dockerfile.optimized"

log_info()    { echo -e "\033[0;34m[INFO]\033[0m  $*"; }
log_success() { echo -e "\033[0;32m[OK]\033[0m    $*"; }
log_warn()    { echo -e "\033[0;33m[WARN]\033[0m  $*"; }
log_error()   { echo -e "\033[0;31m[ERROR]\033[0m $*"; }

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed -e 's/^# \?//' | sed '$d'
    exit 1
}

require() { command -v "$1" >/dev/null 2>&1 || { log_error "missing tool: $1"; exit 2; }; }
require docker
require jq

BUILD=0
PUSH=0
PLATFORM=""
ARCH_LABEL=""
SKIP_ARTIFACTS=0
SINGLE_COMPONENT=""
REFRESH_LOCK=0
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build)         BUILD=1; shift ;;
        -p|--push)          PUSH=1; shift ;;
        -f|--force)         NO_CACHE="--no-cache"; shift ;;
        --x86)              PLATFORM="linux/amd64"; ARCH_LABEL="x86"; shift ;;
        --arm)              PLATFORM="linux/arm64"; ARCH_LABEL="arm64"; shift ;;
        --platform)
            PLATFORM="$2"
            case "$2" in
                linux/amd64) ARCH_LABEL="x86" ;;
                linux/arm64) ARCH_LABEL="arm64" ;;
                *) log_error "--platform must be linux/amd64 or linux/arm64"; exit 1 ;;
            esac
            shift 2 ;;
        --skip-artifacts)   SKIP_ARTIFACTS=1; shift ;;
        --component)        SINGLE_COMPONENT="$2"; shift 2 ;;
        --refresh-lock)     REFRESH_LOCK=1; shift ;;
        -h|--help)          usage ;;
        *) log_error "unknown arg: $1"; usage ;;
    esac
done

if [ "$REFRESH_LOCK" = "1" ]; then
    "$LOCKHASH" --update
    exit 0
fi

if [ "$BUILD" = "0" ]; then usage; fi

if [ -z "$PLATFORM" ]; then
    case "$(uname -m)" in
        x86_64)  PLATFORM="linux/amd64"; ARCH_LABEL="x86" ;;
        aarch64) PLATFORM="linux/arm64"; ARCH_LABEL="arm64" ;;
        *)       PLATFORM="linux/amd64"; ARCH_LABEL="x86" ;;
    esac
fi
log_info "Target platform: $PLATFORM ($ARCH_LABEL)"

# Lockhash drift guard. Fails fast if Dockerfiles or pins changed without
# `lockhash.sh --update`, so CI can't push images with stale tags.
"$LOCKHASH" --verify

base_image=$(jq -r '.base_image'             "$LOCKFILE")
gpu_archs=$(jq -r '.gpu_archs'               "$LOCKFILE")
torch_arch=$(jq -r '.torch_cuda_arch_list'   "$LOCKFILE")

build_artifact() {
    local component="$1"
    local pin dockerfile build_arg image
    pin=$(jq -r       --arg c "$component" '.components[$c].pin'        "$LOCKFILE")
    dockerfile=$(jq -r --arg c "$component" '.components[$c].dockerfile' "$LOCKFILE")
    build_arg=$(jq -r --arg c "$component" '.components[$c].build_arg'  "$LOCKFILE")
    image=$("$LOCKHASH" --image "$component" "$ARCH_LABEL")

    # All status output goes to stderr — stdout is reserved for the resolved
    # image ref, so callers can do `image=$(build_artifact <c>)` without
    # capturing log noise.
    if [ "$NO_CACHE" = "" ] && docker buildx imagetools inspect "$image" >/dev/null 2>&1; then
        log_success "$component cache hit: $image" >&2
        printf '%s' "$image"
        return 0
    fi

    log_info "Building $component pin=$pin → $image" >&2
    # --network=host is required when running under the buildx docker-container
    # driver inside Kubernetes — the nested build container's bridge network
    # often can't resolve internal NVIDIA hosts (gitlab-master.nvidia.com etc.)
    # or has SSL CA-bundle mismatches with the host. Local builds work fine
    # with this too, so we always set it.
    if ! docker buildx build \
            --network=host \
            --platform "$PLATFORM" \
            -f "${REPO_ROOT}/${dockerfile}" \
            --build-arg PYTORCH_BASE_IMAGE="$base_image" \
            --build-arg "${build_arg}=${pin}" \
            --build-arg GPU_ARCHS="$gpu_archs" \
            --build-arg TORCH_CUDA_ARCH_LIST="$torch_arch" \
            $NO_CACHE \
            -t "$image" \
            --push \
            "${REPO_ROOT}" >&2; then
        log_error "buildx failed for $component ($ARCH_LABEL)" >&2
        return 1
    fi
    printf '%s' "$image"
}

# Single-component mode: build one artifact and exit (used by the CI matrix).
if [ -n "$SINGLE_COMPONENT" ]; then
    case "$SINGLE_COMPONENT" in
        ort|torch_scatter|warpconvnet) ;;
        *) log_error "--component must be one of: ort, torch_scatter, warpconvnet"; exit 1 ;;
    esac
    if ! image=$(build_artifact "$SINGLE_COMPONENT"); then
        log_error "Failed to build $SINGLE_COMPONENT for $ARCH_LABEL"
        exit 1
    fi
    log_success "Built $SINGLE_COMPONENT for $ARCH_LABEL: $image"
    exit 0
fi

if [ "$SKIP_ARTIFACTS" = "1" ]; then
    log_warn "Skipping Tier-1 artifact builds (--skip-artifacts)"
    ORT_IMAGE=$("$LOCKHASH"           --image ort           "$ARCH_LABEL")
    TORCH_SCATTER_IMAGE=$("$LOCKHASH" --image torch_scatter "$ARCH_LABEL")
    WARPCONVNET_IMAGE=$("$LOCKHASH"   --image warpconvnet   "$ARCH_LABEL")
else
    ORT_IMAGE=$(build_artifact ort)                     || { log_error "ort build failed";           exit 1; }
    TORCH_SCATTER_IMAGE=$(build_artifact torch_scatter) || { log_error "torch_scatter build failed"; exit 1; }
    WARPCONVNET_IMAGE=$(build_artifact warpconvnet)     || { log_error "warpconvnet build failed";   exit 1; }
fi

log_info "Tier-1 refs ($ARCH_LABEL):"
log_info "  ort:           $ORT_IMAGE"
log_info "  torch_scatter: $TORCH_SCATTER_IMAGE"
log_info "  warpconvnet:   $WARPCONVNET_IMAGE"

# Tier-2: final base image
final_repo="nvcr.io/nvstaging/tao/tao_pytorch_base_image"
local_tag="${USER:-local}"
dated_tag="${USER:-local}-$(date +%Y%m%d%H%M)"

OUT_FLAG="--load"
[ "$PUSH" = "1" ] && OUT_FLAG="--push"
[[ "$PLATFORM" == *","* ]] && OUT_FLAG="--push"  # multi-platform requires --push

log_info "Building Tier-2 image for $PLATFORM (output: ${OUT_FLAG#--})"
# --network=host: see comment in build_artifact above. Tier-2 also needs it
# because Dockerfile.optimized installs xformers from gitlab-master.nvidia.com.
docker buildx build \
    --network=host \
    --platform "$PLATFORM" \
    -f "$DOCKERFILE_OPTIMIZED" \
    --build-arg PYTORCH_BASE_IMAGE="$base_image" \
    --build-arg ORT_IMAGE="$ORT_IMAGE" \
    --build-arg TORCH_SCATTER_IMAGE="$TORCH_SCATTER_IMAGE" \
    --build-arg WARPCONVNET_IMAGE="$WARPCONVNET_IMAGE" \
    $NO_CACHE \
    -t "${final_repo}:${local_tag}" \
    -t "${final_repo}:${dated_tag}" \
    $OUT_FLAG \
    "${REPO_ROOT}"

log_success "Tier-2 build complete:"
log_success "  ${final_repo}:${local_tag}"
log_success "  ${final_repo}:${dated_tag}"
