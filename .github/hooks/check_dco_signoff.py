#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that a commit message carries a Developer Certificate of Origin sign-off.

Takes the path to the commit message file as its only argument, as git supplies
it at the commit-msg stage, and exits non-zero if no sign-off trailer is present.
Reading the message file rather than `git log -1` matters: the latter inspects
the parent commit and would inherit its sign-off. It can also be run directly:

    python .github/hooks/check_dco_signoff.py .git/COMMIT_EDITMSG
"""

import sys

TRAILER = "Signed-off-by:"


def has_signoff(path):
    """Return True if any line of the commit message starts with the DCO trailer."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return any(line.startswith(TRAILER) for line in handle.read().splitlines())
    except OSError as err:
        print(f"{path}: cannot be read ({err})")
        return False


def main(argv):
    """Check the commit message file named in argv for a sign-off trailer."""
    if not argv:
        print("ERROR - no commit message file supplied to the DCO hook")
        return 1
    if has_signoff(argv[0]):
        return 0
    print("ERROR - Commit missing DCO sign-off. Use git commit -s")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
