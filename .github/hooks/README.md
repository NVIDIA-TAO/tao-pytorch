<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Git hooks

`pre-push` checks the files your push adds or modifies:

- **License headers** — every changed `.py` and `.sh` file must carry an SPDX
  header (`check_license_header.py`).
- **pylint**, **pydocstyle**, **flake8** — static analysis of changed Python
  files under the package source.

## Enable

Git does not enable repository-tracked hooks automatically. Run this once per
clone:

```bash
git config core.hooksPath .github/hooks
```

Install the linters if you do not have them:

```bash
pip install pylint pydocstyle flake8
```

## Bypass

The hook is a local convenience check, not an enforcement gate. To skip it for
one push:

```bash
git push --no-verify
# or
SKIP_LINT=1 git push
```

## Configuration

pylint reads `.pylintrc` at the repository root. The pydocstyle and flake8
ignore lists live at the top of `pre-push`, alongside the list of file suffixes
that require a license header.
