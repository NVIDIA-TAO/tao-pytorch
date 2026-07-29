#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that source files carry an SPDX license header.

Takes file paths as arguments and exits non-zero if any of them is missing a
license header in its first few lines. Used by the pre-push hook and by
pre-commit; it can also be run directly:

    python3 .github/hooks/check_license_header.py path/to/file.py
"""

import sys

# How far into the file to look, so a shebang, encoding line, or short module
# preamble ahead of the header is tolerated.
MAX_HEADER_LINES = 10

LICENSE_MARKER = "SPDX-License-Identifier:"
COPYRIGHT_MARKERS = ("SPDX-FileCopyrightText:", "Copyright")

EXPECTED_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES."
    " All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0"
)


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
    missing = [path for path in paths if not has_license_header(path)]
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
