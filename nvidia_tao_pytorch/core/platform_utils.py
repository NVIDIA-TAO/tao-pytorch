# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Platform utilities for detecting and handling platform-specific configurations."""

import platform


def get_platform():
    """
    Get the current platform architecture.

    Returns
    -------
    str
        Platform identifier - "x86" for x86_64/amd64 architectures,
        "arm" for ARM64/aarch64 architectures.

    Raises
    ------
    ValueError
        If the platform is not supported.
    """
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return "x86"
    elif machine in ("arm64", "aarch64"):
        return "arm"
    else:
        raise ValueError(f"Unsupported platform architecture: {machine}")


def get_platform_digest(manifest_config, platform_override=None):
    """
    Get the appropriate digest for the current or specified platform.

    Parameters
    ----------
    manifest_config : dict
        The manifest configuration containing platform-specific digests.
    platform_override : str, optional
        Override the platform detection with a specific value.
        Must be either "x86" or "arm". Default is None.

    Returns
    -------
    str
        The digest for the specified or detected platform.

    Raises
    ------
    ValueError
        If the platform is not supported or digest is not found.
    KeyError
        If the digests key is missing from manifest.
    """
    if platform_override is not None:
        if platform_override not in ("x86", "arm"):
            raise ValueError(f"Invalid platform override: {platform_override}. Must be 'x86' or 'arm'.")
        current_platform = platform_override
    else:
        current_platform = get_platform()

    if "digests" not in manifest_config:
        raise KeyError("The manifest must contain a 'digests' key with platform-specific digests.")

    digests = manifest_config["digests"]

    if current_platform not in digests:
        available_platforms = ", ".join(digests.keys())
        raise ValueError(
            f"No digest found for platform '{current_platform}'. "
            f"Available platforms: {available_platforms}"
        )

    return digests[current_platform]
