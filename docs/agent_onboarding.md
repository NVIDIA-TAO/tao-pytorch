# Agent Onboarding

This guide is for a coding agent or new engineer who needs to work safely in
TAO PyTorch without reading the entire repository first.

## Mental Model

TAO PyTorch is organized around model-family commands such as `depth_net`,
`dino`, or `pointpillars`.

The normal flow is:

```text
console command
  -> nvidia_tao_pytorch/core/entrypoint.py
  -> <task package>/scripts/<subtask>.py
  -> Hydra experiment config
  -> model / dataloader / export / inference code
```

![Agent development loop](assets/agent_workflow.svg)

The important source roots are:

| Path | What to expect |
| :--- | :--- |
| `nvidia_tao_pytorch/core` | Shared launcher, Hydra, logging, telemetry, checkpoint, Lightning, quantization, and export utilities. |
| `nvidia_tao_pytorch/config` | Dataclass schemas for experiment specs. |
| `nvidia_tao_pytorch/cv`, `multimodal`, `ssl`, `sdg`, `pointcloud` | Model-family packages. |
| `tests` | Unit and integration tests grouped by domain and task. |
| `docker`, `runner`, `release` | Development launcher, base image, and release image tooling. |
| `tao-core` | Submodule for API, microservice, telemetry, and FTMS integration. |

## First Inspection Pass

Start with the command or task named by the user. For example, for `depth_net`
export:

```sh
rg -n "depth_net=" setup.py
find nvidia_tao_pytorch/cv/depth_net -maxdepth 3 -type f | sort
sed -n '1,220p' nvidia_tao_pytorch/cv/depth_net/entrypoint/depth_net.py
sed -n '1,260p' nvidia_tao_pytorch/cv/depth_net/scripts/export.py
```

Find the config schema and default specs:

```sh
find nvidia_tao_pytorch/config/depth_net -maxdepth 2 -type f | sort
find nvidia_tao_pytorch/cv/depth_net/experiment_specs -maxdepth 1 -type f | sort
```

Find tests:

```sh
find tests -path '*depth_net*' -type f | sort
rg -n "depth_net|DepthNet" tests nvidia_tao_pytorch/cv/depth_net
```

When the task is not known, list registered commands:

```sh
python tools/update_readme_supported_commands.py --check
rg -n "console_scripts" -n setup.py
```

## Safety Rules

* Check the worktree before editing. If normal `git status` trips over LFS,
  use:

  ```sh
  git -c filter.lfs.process= -c filter.lfs.required=false status --short
  ```

* Do not revert unrelated edits. Treat unrecognized changes as user work unless
  the user explicitly asks for cleanup.
* Prefer targeted searches with `rg` and `find`; do not bulk-read generated
  artifacts or large logs.
* Use the repo's existing task pattern instead of inventing a new layout.
* Keep docs and generated command tables current:

  ```sh
  python tools/update_readme_supported_commands.py --check
  ```

* When adding dependencies, check container requirements and release packaging;
  do not assume a local pip install is enough.
* For code changes, run the narrowest useful tests first, then broaden only when
  the change touches shared infrastructure.

## Common Agent Questions

| Question | Where to look |
| :--- | :--- |
| What command invokes this package? | `setup.py` `console_scripts` |
| What subtasks exist? | `<task>/scripts/*.py` and the README generated table |
| What config fields are valid? | `nvidia_tao_pytorch/config/<task>/default_config.py` and related files |
| Where does training start? | `<task>/scripts/train.py` |
| Where does model code live? | `<task>/model` |
| Where does dataloading live? | `<task>/dataloader` or `<task>/datasets` |
| Where is export implemented? | `<task>/scripts/export.py` and task `utils/export` files |
| How do I add a new network? | [Integrating a New Network](new_network_integration.md) |
| How do I run in containers? | [Container Power Users](container_power_users.md) |

## Before Finalizing Work

Run checks that match the blast radius:

```sh
python tools/update_readme_supported_commands.py --check
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/update_readme_supported_commands.py
pytest tests/cv_unit_test/<task>
```

If tests require GPU, TensorRT, datasets, or private checkpoints, state that
clearly and provide the narrower checks that did run.
