#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# lockhash.sh — content-addressable hashing for the TAO PyTorch base-image artifacts.
#
# For each component (ort, xformers, torch-scatter, warpconvnet) we compute a
# stable lockhash from the inputs that affect its output:
#   sha256(base_digest, pin, gpu_archs, sha256(component Dockerfile))
# The lockhash becomes part of the image tag, so the same inputs always
# resolve to the same image; any input change forces a rebuild.
#
# docker/artifacts.lock.json is the committed index. This script keeps it in
# sync with the committed Dockerfiles + pins, and is invoked by build.sh and
# Jenkinsfile.base-image.
#
# Usage:
#   lockhash.sh --print                       # print the recomputed JSON to stdout
#   lockhash.sh --verify                      # exit 0 if committed file matches, 1 if drift
#   lockhash.sh --update                      # write the recomputed JSON to disk
#   lockhash.sh --image <component> [arch]    # print the image ref for one component
#                                             # arch ∈ {x86, arm64} appends -<arch> to the tag

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCKFILE="${SCRIPT_DIR}/artifacts.lock.json"

require() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "lockhash.sh: missing required tool '$1'" >&2
        exit 2
    }
}
require jq
require sha256sum

# Resolve the multi-arch manifest digest of an image reference. Tries buildx
# first, then docker manifest inspect, then a docker pull-and-inspect fallback.
#
# Each step extracts the first `sha256:...` token from the command's stdout
# instead of trusting the format flag — older `docker buildx` versions ignore
# `--format '{{.Manifest.Digest}}'` and fall back to the verbose human-readable
# output, which would otherwise be captured wholesale and corrupt the lockhash
# (see the Jenkins-only failure where the multi-line buildx output was treated
# as `base_digest`).
extract_sha256() {
    grep -oE 'sha256:[a-f0-9]{64}' | head -1
}

resolve_digest() {
    local ref="$1" digest=""
    if command -v docker >/dev/null 2>&1; then
        digest=$(docker buildx imagetools inspect "$ref" --format '{{.Manifest.Digest}}' 2>/dev/null \
            | extract_sha256 || true)
        if [ -z "$digest" ]; then
            # Default-format buildx output also begins with `Digest: sha256:...`.
            digest=$(docker buildx imagetools inspect "$ref" 2>/dev/null \
                | extract_sha256 || true)
        fi
        if [ -z "$digest" ]; then
            digest=$(docker manifest inspect "$ref" 2>/dev/null \
                | extract_sha256 || true)
        fi
        if [ -z "$digest" ]; then
            docker pull -q "$ref" >/dev/null 2>&1 || true
            digest=$(docker inspect "$ref" --format='{{index .RepoDigests 0}}' 2>/dev/null \
                | extract_sha256 || true)
        fi
    fi
    if [ -z "$digest" ]; then
        echo "lockhash.sh: failed to resolve digest for $ref" >&2
        exit 3
    fi
    echo "$digest"
}

sha256_short() {
    sha256sum | cut -c1-16
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

compute_component() {
    local component="$1" pin="$2" dockerfile="$3" \
          base_digest="$4" gpu_archs="$5" artifacts_repo="$6"
    local df_hash
    df_hash=$(sha256_file "${REPO_ROOT}/${dockerfile}")
    local lockhash
    lockhash=$(printf 'base_digest=%s\npin=%s\ngpu_archs=%s\ndockerfile_sha=%s\n' \
        "$base_digest" "$pin" "$gpu_archs" "$df_hash" \
        | sha256_short)
    local image="${artifacts_repo}:${component}-${lockhash}"
    jq -n --arg lockhash "$lockhash" --arg image "$image" \
        '{lockhash: $lockhash, image: $image}'
}

# Build the recomputed lockfile JSON on stdout.
recompute() {
    local committed
    committed=$(cat "$LOCKFILE")

    local base_image gpu_archs artifacts_repo
    base_image=$(echo "$committed"      | jq -r '.base_image')
    gpu_archs=$(echo "$committed"       | jq -r '.gpu_archs')
    artifacts_repo=$(echo "$committed"  | jq -r '.artifacts_repo')

    local base_digest
    if [ "${LOCKHASH_OFFLINE:-0}" = "1" ]; then
        base_digest=$(echo "$committed" | jq -r '.base_digest')
        if [ -z "$base_digest" ] || [ "$base_digest" = "null" ]; then
            echo "lockhash.sh: LOCKHASH_OFFLINE=1 but committed base_digest is empty" >&2
            exit 4
        fi
    else
        base_digest=$(resolve_digest "$base_image")
    fi

    # Walk the committed components in order, recompute lockhash + image.
    local components_json='{}'
    while IFS= read -r component; do
        local pin dockerfile extras
        pin=$(echo "$committed"        | jq -r --arg c "$component" '.components[$c].pin')
        dockerfile=$(echo "$committed" | jq -r --arg c "$component" '.components[$c].dockerfile')
        extras=$(echo "$committed"     | jq -c --arg c "$component" '.components[$c]
            | with_entries(select(.key | IN("pin","dockerfile","lockhash","image") | not))')

        local computed
        computed=$(compute_component "$component" "$pin" "$dockerfile" \
            "$base_digest" "$gpu_archs" "$artifacts_repo")

        components_json=$(jq -n \
            --argjson acc "$components_json" \
            --arg c "$component" \
            --arg pin "$pin" \
            --arg dockerfile "$dockerfile" \
            --argjson computed "$computed" \
            --argjson extras "$extras" \
            '$acc + {($c): ($extras + {pin: $pin, dockerfile: $dockerfile} + $computed)}')
    done < <(echo "$committed" | jq -r '.components | keys_unsorted[]')

    jq -n \
        --argjson committed "$committed" \
        --arg base_digest "$base_digest" \
        --argjson components "$components_json" \
        '$committed + {base_digest: $base_digest, components: $components}'
}

cmd=${1:-}
case "$cmd" in
    --print)
        recompute
        ;;
    --verify)
        recomputed=$(recompute)
        committed=$(jq -S . < "$LOCKFILE")
        recomputed_sorted=$(echo "$recomputed" | jq -S .)
        if [ "$committed" = "$recomputed_sorted" ]; then
            echo "lockhash.sh: $LOCKFILE is up to date"
            exit 0
        fi
        echo "lockhash.sh: $LOCKFILE is OUT OF DATE — run lockhash.sh --update" >&2
        diff <(echo "$committed") <(echo "$recomputed_sorted") || true
        exit 1
        ;;
    --update)
        recomputed=$(recompute)
        echo "$recomputed" | jq -S . > "$LOCKFILE.tmp"
        mv "$LOCKFILE.tmp" "$LOCKFILE"
        echo "lockhash.sh: wrote $LOCKFILE"
        ;;
    --image)
        component=${2:?usage: lockhash.sh --image <component> [arch]}
        arch=${3:-}
        ref=$(jq -r --arg c "$component" '.components[$c].image' < "$LOCKFILE")
        if [ -n "$arch" ]; then
            case "$arch" in
                x86|arm64) ref="${ref}-${arch}" ;;
                *) echo "lockhash.sh: arch must be x86 or arm64 (got: $arch)" >&2; exit 1 ;;
            esac
        fi
        echo "$ref"
        ;;
    *)
        sed -n '1,/^set -euo pipefail/p' "$0" | head -n 20
        exit 1
        ;;
esac
