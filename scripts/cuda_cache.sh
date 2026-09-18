#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

CACHE_DIR="${HOME}/.tao_pytorch_cuda_cache"
PROJECT_ROOT="${NV_TAO_PYTORCH_TOP:-$(pwd)}"
BUILD_DIR="${PROJECT_ROOT}/build"
HASH_FILE="${CACHE_DIR}/source.hash"

ensure_tree_writable() {
    local path="$1"
    [ ! -e "$path" ] && return 0
    local blocked
    blocked=$(find "$path" ! -writable -print -quit 2>/dev/null || true)
    if [ -n "$blocked" ]; then
        echo "ERROR: cache path is not writable: $blocked" >&2
        echo "Run cache operations inside the container that owns build artifacts, or remove the root-owned path." >&2
        return 1
    fi
}

generate_hash() {
    {
        find "$PROJECT_ROOT/nvidia_tao_pytorch" \
            \( -name "*.cu" -o -name "*.cpp" -o -name "*.cc" \
               -o -name "*.h" -o -name "*.cuh" \) -print0 2>/dev/null \
            | sort -z | xargs -0 -r md5sum 2>/dev/null || true
        [ -f "$PROJECT_ROOT/setup.py" ] \
            && md5sum "$PROJECT_ROOT/setup.py" 2>/dev/null || true
        [ -f "$PROJECT_ROOT/docker/manifest.json" ] \
            && sha256sum "$PROJECT_ROOT/docker/manifest.json" 2>/dev/null || true
        [ -f "$PROJECT_ROOT/docker/Dockerfile" ] \
            && grep -E '^ENV TORCH_CUDA_ARCH_LIST=' "$PROJECT_ROOT/docker/Dockerfile" || true
        printf 'TORCH_CUDA_ARCH_LIST=%s\n' "${TORCH_CUDA_ARCH_LIST-}"
    } | md5sum | cut -d' ' -f1
}

is_valid() {
    [ -d "$CACHE_DIR/build" ] \
        && [ -f "$HASH_FILE" ] \
        && [ "$(cat "$HASH_FILE" 2>/dev/null)" = "$(generate_hash)" ]
}

case "${1:-}" in
    restore)
        if ! is_valid; then
            echo "No valid cache found"
            exit 1
        fi
        ensure_tree_writable "$BUILD_DIR"
        rm -rf "$BUILD_DIR"
        cp -r "$CACHE_DIR/build" "$BUILD_DIR"
        find "$BUILD_DIR" \( -name "*.so" -o -name "*.o" -o \
            -name "build.ninja" -o -name ".ninja_*" \) -exec touch {} \; \
            2>/dev/null || true
        echo "Cache restored successfully"
        ;;
    save)
        if [ ! -d "$BUILD_DIR" ]; then
            echo "No build directory to cache"
            exit 1
        fi
        mkdir -p "$CACHE_DIR"
        ensure_tree_writable "$CACHE_DIR/build"
        rm -rf "$CACHE_DIR/build"
        cp -r "$BUILD_DIR" "$CACHE_DIR/"
        generate_hash > "$HASH_FILE"
        echo "Cache saved successfully"
        ;;
    check)
        if is_valid; then
            echo "Cache is valid"
            exit 0
        fi
        echo "Cache is invalid or missing"
        exit 1
        ;;
    clean)
        ensure_tree_writable "$CACHE_DIR"
        rm -rf "$CACHE_DIR"
        echo "Cache cleaned"
        ;;
    status)
        echo "Cache directory: $CACHE_DIR"
        if [ -d "$CACHE_DIR/build" ]; then
            echo "Cache size: $(du -sh "$CACHE_DIR/build" 2>/dev/null | cut -f1 || echo unknown)"
            if is_valid; then echo "Status: VALID"; else echo "Status: INVALID"; fi
        else
            echo "Status: EMPTY"
        fi
        echo "Current hash: $(generate_hash)"
        ;;
    *)
        echo "Usage: $0 {restore|save|check|clean|status}"
        exit 1
        ;;
esac
