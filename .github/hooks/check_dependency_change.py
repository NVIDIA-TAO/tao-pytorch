#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block any commit that changes a dependency declaration.

Receives the staged files that matched the dependency patterns in
.pre-commit-config.yaml and always fails when there is at least one. There is
no allowlist and no bypass: dependency changes are routed to the TAO Infra
team instead.
"""

import sys

MESSAGE = "Please reach out to TAO Infra team for dependency change"


def main(paths):
    """Fail if any dependency-bearing file is part of this commit."""
    if not paths:
        return 0
    print("Dependency change blocked. This commit modifies:")
    for path in paths:
        print(f"  {path}")
    print()
    print(MESSAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
