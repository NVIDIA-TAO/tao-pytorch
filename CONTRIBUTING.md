# Contributing to TAO PyTorch

Thanks for contributing! This guide covers **how we work** — trunk-based development and how CI is triggered — and the two **requirements** CI enforces on every pull request: an **SPDX license header** on source files (the `license header` hook in **Static Tests**) and **DCO sign-off** on every commit (the **DCO** workflow). See [Running the checks locally](#running-the-checks-locally) to catch failures before you push.

## Trunk-based development and backports

We practice **trunk-based development**: all changes land on **`main`** first, then are backported to release branches as needed.

- **Open your PR against `main`**, not against a `release/*` branch.
- If a fix should also ship in a release, add a **`release/X.Y.Z`** label to the PR — the label name matches the release branch (e.g. `release/7.1.0`). Add several labels to backport to multiple releases.
- When the PR merges to `main`, **`tao-cherry-pick-bot`** cherry-picks the commit onto each labeled release branch:
  - **Clean** cherry-pick → pushed straight to the release branch.
  - **Conflict** → the bot opens a **draft PR** against the release branch and assigns + @-mentions you to resolve the conflicts, then mark it ready for review.
- A summary comment on your original PR shows where each backport landed (and notes any label whose branch doesn't exist).

If you open a PR directly against a `release/*` branch, `tao-pr-bot` will remind you to retarget it to `main` and add the label. Genuinely release-only fixes — code that no longer exists on `main` — are the exception: add the **`release-only`** label to keep such a PR on the release branch.

## Running CI (the `/build` command)

For security, CI does **not** run automatically on NVIDIA's runners — it is triggered per commit.

**Internal developers**

- Comment **`/build`** on your PR to run CI (`blossom-ci`) on the latest commit.
- CI is pinned to the head commit, so **re-run `/build` after every push** — a stale run won't count.
- Make sure `blossom-ci` (and the other checks) are green before merging.

**External contributors**

- You can't trigger CI yourself. Once an internal reviewer has reviewed and vetted your PR, a maintainer will run **`/build`** for you.
- No action is needed on your side beyond addressing review feedback — just wait for the maintainer to trigger CI after the review is complete.

## License headers

Every source file must carry an **SPDX license header** near the top. Add these two lines, **below** any shebang (`#!`) line, replacing `<year>` with the current year:

```
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

Use the comment style for the file's language. The check passes as long as **both** a copyright line (`SPDX-FileCopyrightText:` or `Copyright`) **and** an `SPDX-License-Identifier:` line appear within the **first 10 lines** of the file.

**Python / shell / YAML** (an optional shebang may sit above the header):

```python
#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

**C / C++ / JavaScript / Go** (`//` comments):

```c
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

Files that legitimately cannot carry a header (e.g. generated files or data) may be listed, one repo-relative path per line, in `.github/hooks/license_header_exclude.txt`.

## Signing your work (DCO)

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Already committed without a sign-off? Add one with:
  ```bash
  git commit --amend -s --no-edit      # fixes the most recent commit
  git rebase --signoff origin/main     # signs off a range of commits
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

## Running the checks locally

Both checks also run in CI, but you can catch failures before pushing:

```bash
pip install pre-commit

# check only the files your branch changed vs main:
pre-commit run --from-ref origin/main --to-ref HEAD

# or check everything:
pre-commit run --all-files
```

The `license header` and `dco-signoff` hooks (plus the lint/docstring checks) are defined in `.pre-commit-config.yaml`. Run `pre-commit install` once to have them run automatically on every `git commit`.
