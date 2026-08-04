<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Git hooks

Scripts run by [pre-commit](https://pre-commit.com). The hook list lives in
`.pre-commit-config.yaml` at the repository root.

- `check_license_header.py` — every changed `.py` and `.sh` file must carry an
  SPDX header. Runs at the `pre-commit` stage.
- `check_dco_signoff.py` — every commit message must carry a `Signed-off-by`
  trailer. Runs at the `commit-msg` stage.

pylint, pydocstyle and flake8 run at the `pre-commit` stage as well, on changed
Python files under the package source.

## Enable

Git does not enable repository-tracked hooks automatically. Run this once per
clone:

```bash
pip install pre-commit
pre-commit install
```

`pre-commit install` wires up both the `pre-commit` and `commit-msg` hook types.

Install the linters if you do not have them:

```bash
pip install pylint pydocstyle flake8
```

### Migrating from the old pre-push hook

Earlier revisions shipped a `pre-push` script enabled with `git config
core.hooksPath .github/hooks`. That script has been removed and its checks now
run at commit time. `core.hooksPath` overrides `.git/hooks`, which is where
`pre-commit install` writes, so unset it or the new hooks will never run:

```bash
git config --unset core.hooksPath
pre-commit install
```

## Bypass

To skip the checks for one commit:

```bash
git commit --no-verify
```

## Configuration

Hook definitions, tool arguments and file scopes live in
`.pre-commit-config.yaml`. pylint reads `.pylintrc` at the repository root.
