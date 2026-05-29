#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail
cd "$( dirname "${BASH_SOURCE[0]}" )"

# Logging functions
log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

log_warn() {
    echo -e "\033[0;33m[WARN]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# Help function
show_help() {
    cat << EOF
Usage: ./build.sh [OPTIONS]

Build TAO PyTorch base Docker image with optional cross-platform support.

OPTIONS:
    -b, --build              Build the Docker image
    -p, --push               Push the Docker image to registry
    -f, --force              Force rebuild without cache
    -h, --help               Show this help message
    
    --default                Use default settings (build without push)
    
Platform Options:
    --x86                    Build for x86_64/AMD64 (linux/amd64)
    --arm                    Build for ARM64 (linux/arm64)
    --multiplatform          Build for both x86_64 and ARM64
    --platform <platform>    Specify platform(s) explicitly
                            Examples: linux/amd64, linux/arm64

EXAMPLES:
    # Build for x86_64/AMD64
    ./build.sh --build --x86

    # Build for ARM64
    ./build.sh --build --arm

    # Build for both platforms and push
    ./build.sh --build --multiplatform --push

    # Force rebuild without cache
    ./build.sh --build --force --x86

NOTES:
    Multi-platform builds REQUIRE the --push flag (buildx limitation)
    Docker buildx cannot load multiple architectures to local Docker
    For local testing, build a single platform at a time (use --x86 or --arm)
    Default platform is auto-detected based on host architecture (native build)
    
    Cross-platform builds (e.g., ARM on x86) automatically setup QEMU emulation
    QEMU setup persists on the host and is reused across builds

EOF
    exit 0
}

# Setup QEMU for cross-platform builds
setup_qemu() {
    local platform="$1"
    
    # Check if we need QEMU (building for ARM on x86 or vice versa)
    local host_arch=$(uname -m)
    local needs_qemu=false
    
    if [[ "$platform" == *"arm64"* ]] && [[ "$host_arch" == "x86_64" ]]; then
        needs_qemu=true
    elif [[ "$platform" == *"amd64"* ]] && [[ "$host_arch" == "aarch64" ]]; then
        needs_qemu=true
    fi
    
    if [ "$needs_qemu" = true ]; then
        log_info "Cross-platform build detected (host: $host_arch, target: $platform)"
        log_info "Checking QEMU setup for emulation..."
        
        # Check if QEMU is already registered
        if docker run --rm --platform "$platform" alpine uname -m > /dev/null 2>&1; then
            log_success "QEMU already configured"
        else
            log_warn "QEMU not configured. Setting up multi-architecture support..."
            log_info "Executing: docker run --rm --privileged multiarch/qemu-user-static --reset -p yes"
            
            if docker run --rm --privileged multiarch/qemu-user-static --reset -p yes > /dev/null 2>&1; then
                log_success "QEMU configured successfully"
            else
                log_error "Failed to setup QEMU"
                log_error "You may need to install qemu-user-static on your host:"
                log_error "  sudo apt-get install -y qemu qemu-user-static binfmt-support"
                exit 1
            fi
        fi
    fi
}

registry="nvcr.io"
repository="nvstaging/tao/tao_pytorch_base_image"

tag="$USER-$(date +%Y%m%d%H%M)"
local_tag="$USER"

# Detect native platform
HOST_ARCH=$(uname -m)
if [[ "$HOST_ARCH" == "x86_64" ]]; then
    DEFAULT_PLATFORM="linux/amd64"
elif [[ "$HOST_ARCH" == "aarch64" ]]; then
    DEFAULT_PLATFORM="linux/arm64"
else
    DEFAULT_PLATFORM="linux/amd64"  # Fallback to amd64
fi

# Build parameters.
BUILD_DOCKER="0"
PUSH_DOCKER="0"
FORCE="0"
PLATFORM="$DEFAULT_PLATFORM"  # Default to native platform, can be overridden


# Parse command line.
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    show_help
    ;;
    -b|--build)
    BUILD_DOCKER="1"
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
    --platform)
    if [[ -z "$2" || "$2" == -* ]]; then
        log_error "Missing value for --platform option"
        show_help
    fi
    PLATFORM="$2"
    shift # past argument
    shift # past value
    ;;
    --x86)
    PLATFORM="linux/amd64"
    shift
    ;;
    --arm)
    PLATFORM="linux/arm64"
    shift
    ;;
    --multiplatform)
    PLATFORM="linux/amd64,linux/arm64"
    shift
    ;;
    --default)
    BUILD_DOCKER="1"
    PUSH_DOCKER="0"
    FORCE="0"
    shift # past argument
    ;;
    *)    # unknown option
    log_warn "Unknown option: $1"
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

# Build docker
if [ $BUILD_DOCKER = "1" ]; then
    log_info "Starting Docker build process..."
    log_info "Platform(s): $PLATFORM"
    
    # Validate build configuration before doing any work
    if [[ "$PLATFORM" == *","* ]] && [ $PUSH_DOCKER != "1" ]; then
        log_error "Multi-platform builds require the --push flag"
        log_error "Docker buildx cannot load multiple architectures to local Docker simultaneously"
        log_info "Option 1: Add --push flag: ./build.sh --build --multiplatform --push"
        log_info "Option 2: Build for single platform: ./build.sh --build --x86  (or --arm)"
        exit 1
    fi
    
    # Setup QEMU for cross-platform builds if needed
    setup_qemu "$PLATFORM"

    if [ $FORCE = "1" ]; then
        log_warn "Force rebuild enabled - ignoring Docker cache"
        NO_CACHE="--no-cache"
    else
        log_info "Using Docker cache (if available)"
        NO_CACHE=""
    fi
    
    # Check if building for multiple platforms
    if [[ "$PLATFORM" == *","* ]]; then
        log_info "Multi-platform build detected - building and pushing for: $PLATFORM"
        log_info "Executing: DOCKER_BUILDKIT=1 docker buildx build --platform $PLATFORM -f $NV_TAO_PYTORCH_TOP/docker/Dockerfile -t $registry/$repository:$local_tag -t $registry/$repository:$tag $NO_CACHE --push $NV_TAO_PYTORCH_TOP/."
        
        DOCKER_BUILDKIT=1 docker buildx build --platform $PLATFORM \
            -f $NV_TAO_PYTORCH_TOP/docker/Dockerfile \
            -t $registry/$repository:$local_tag \
            -t $registry/$repository:$tag \
            $NO_CACHE \
            --push \
            $NV_TAO_PYTORCH_TOP/.
        
        log_success "Multi-platform build completed and pushed"
        
        log_info "Retrieving image digest..."
        log_info "Executing: docker buildx imagetools inspect $registry/$repository:$tag"
        digest=$(docker buildx imagetools inspect $registry/$repository:$tag --format '{{json .Manifest}}' | grep -o 'sha256:[a-f0-9]*' | head -1)
        log_warn "Update the digest in manifest.json to: $registry/$repository@$digest"
    else
        # Single platform build
        log_info "Building for single platform: $PLATFORM"
        log_info "Executing: DOCKER_BUILDKIT=0 docker buildx build --platform $PLATFORM -f $NV_TAO_PYTORCH_TOP/docker/Dockerfile -t $registry/$repository:$local_tag $NO_CACHE --load $NV_TAO_PYTORCH_TOP/."
        
        DOCKER_BUILDKIT=0 docker buildx build --platform $PLATFORM \
            -f $NV_TAO_PYTORCH_TOP/docker/Dockerfile \
            -t $registry/$repository:$local_tag \
            $NO_CACHE \
            --load \
            $NV_TAO_PYTORCH_TOP/.
        
        log_success "Docker build completed"
        
        if [ $PUSH_DOCKER = "1" ]; then
            log_info "Tagging image..."
            log_info "Executing: docker tag $registry/$repository:$local_tag $registry/$repository:$tag"
            docker tag $registry/$repository:$local_tag $registry/$repository:$tag
            
            log_info "Pushing image to registry..."
            log_info "Executing: docker push $registry/$repository:$tag"
            docker push $registry/$repository:$tag
            
            log_success "Image pushed successfully"
            
            log_info "Retrieving image digest..."
            log_info "Executing: docker inspect --format='{{index .RepoDigests 0}}' $registry/$repository:$tag"
            digest=$(docker inspect --format='{{index .RepoDigests 0}}' $registry/$repository:$tag)
            log_warn "Update the digest in manifest.json to: $digest"
        else
            log_info "Image built locally (use --push to push to registry)"
        fi
    fi
    
    log_success "All operations completed successfully!"
else
    log_error "No build action specified"
    show_help
fi
