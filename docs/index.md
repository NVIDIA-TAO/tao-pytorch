# TAO PyTorch Developer Documentation

This directory is the deeper onboarding and operations guide for TAO PyTorch.
The root `README.md` is the landing page; these files are for people and agents
who need to develop, debug, extend, or run the repository in containers.

## Start Here

| Audience | Read |
| :--- | :--- |
| Coding agent or new contributor | [Agent Onboarding](agent_onboarding.md), then [Architecture](architecture.md) |
| Contributor modifying existing code | [Development Workflows](development_workflows.md), then [Testing And Debugging](testing_and_debugging.md) |
| Developer adding a model family | [Integrating a New Network](new_network_integration.md) |
| Power user running prebuilt containers | [Container Power Users](container_power_users.md) |
| Release or base-image maintainer | [Container Power Users](container_power_users.md), then the Docker sections in `README.md` |

## Documentation Map

* [Agent Onboarding](agent_onboarding.md): repo mental model, first inspection
  commands, and safety rules for autonomous coding work.
* [Architecture](architecture.md): command, config, task, model, dataloader,
  export, and shared-service flows.
* [Development Workflows](development_workflows.md): recipes for common source
  changes such as adding config fields, tracing checkpoint loading, modifying
  export behavior, and adding subtasks.
* [Testing And Debugging](testing_and_debugging.md): targeted test selection,
  smoke-test strategy, and common failure modes.
* [Container Power Users](container_power_users.md): `tao_pt`, direct
  `docker run`, prebuilt images, mounts, GPU flags, user mapping, CI jobs, and
  container troubleshooting.
* [Integrating a New Network](new_network_integration.md): canonical checklist
  for adding a new model family to TAO PyTorch and optionally to TAO Core/FTMS.

## Diagram Format

Architecture and workflow diagrams are checked in as SVG files under
`docs/assets/` and embedded with normal Markdown image links. SVG keeps diagrams
renderable in common Git hosting UIs while staying source-controlled and
diffable. The checked-in SVG files are the canonical editable sources; no PNG
conversion step is required unless a downstream publishing system does not
support SVG.

## Maintenance Rules

Keep the root `README.md` concise. Add detailed developer, architecture,
workflow, and container guidance here instead.

When adding or removing a console command, update the generated README command
table:

```sh
python tools/update_readme_supported_commands.py
```

For checks:

```sh
python tools/update_readme_supported_commands.py --check
git diff --check -- README.md docs/*.md docs/assets/*.svg tools/update_readme_supported_commands.py
```

The repo ships a `.pre-commit-config.yaml` that regenerates the README command
table automatically. Install it once per clone:

```sh
pip install pre-commit
pre-commit install
```

After install, the hook runs on every `git commit` that touches `setup.py`,
files under `nvidia_tao_pytorch/<task>/scripts/`, the generator, or `README.md`.
If the README is out of date, the hook regenerates it in place and aborts the
commit so the regenerated diff can be staged. CI runs the same check in
`--check` mode in the `static_tests` stage as the hard gate.
