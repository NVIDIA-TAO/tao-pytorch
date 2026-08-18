#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that source files carry an SPDX license header.

Takes file paths as arguments and exits non-zero if any of them is missing a
license header in its first few lines. Used by pre-commit; it can also be run
directly:

    python .github/hooks/check_license_header.py path/to/file.py
"""

import os
import sys

# How far into the file to look, so a shebang, encoding line, or short module
# preamble ahead of the header is tolerated.
MAX_HEADER_LINES = 10

LICENSE_MARKER = "SPDX-License-Identifier:"
COPYRIGHT_MARKERS = ("SPDX-FileCopyrightText:", "Copyright")

HOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HOOKS_DIR))
EXCLUDE_FILE = os.path.join(HOOKS_DIR, "license_header_exclude.txt")

EXPECTED_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES."
    " All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0"
)


def load_exclusions():
    """Return the set of repo-relative paths exempt from the SPDX requirement."""
    try:
        with open(EXCLUDE_FILE, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return set()
    entries = set()
    for line in lines:
        entry = line.split("#", 1)[0].strip()
        if entry:
            entries.add(os.path.normpath(entry))
    return entries


def relative_to_repo(path):
    """Return path relative to the repository root, for matching exclusions."""
    try:
        return os.path.normpath(os.path.relpath(os.path.abspath(path), REPO_ROOT))
    except ValueError:
        return os.path.normpath(path)


def has_license_header(path):
    """Return True if the file starts with both a copyright and a license line."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            head = "".join(next(handle, "") for _ in range(MAX_HEADER_LINES))
    except OSError as err:
        print(f"{path}: cannot be read ({err})")
        return False
    return LICENSE_MARKER in head and any(marker in head for marker in COPYRIGHT_MARKERS)


def main(paths):
    """Check every path and report the ones missing a header."""
    exclusions = load_exclusions()
    checked = [path for path in paths if relative_to_repo(path) not in exclusions]
    missing = [path for path in checked if not has_license_header(path)]
    if not missing:
        return 0

    print("License header missing or incomplete in:")
    for path in missing:
        print(f"  {path}")
    print("\nAdd the following at the top of each file, below any shebang line:\n")
    print(EXPECTED_HEADER)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
