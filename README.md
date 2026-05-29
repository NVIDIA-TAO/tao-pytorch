# TAO Toolkit - PyTorch Backend

## Overview

TAO Toolkit is a Python package hosted on the NVIDIA Python Package Index. It
uses TAO containers from the NVIDIA GPU Accelerated Container Registry (NGC) for
training, evaluation, export, and deployment workflows. The output of a TAO
workflow is a trained model that can be deployed for inference on NVIDIA devices
with DeepStream, TensorRT, or Triton.

This repository contains the PyTorch backend implementation for TAO Toolkit. It
includes model code, dataclass-based experiment schemas, task entrypoints,
Docker build tooling, release packaging, and tests for the TAO PyTorch model
families.

## Contents

* [Repository Map](#repository-map)
* [Developer And Power-User Docs](#developer-and-power-user-docs)
* [Supported Commands](#supported-commands)
* [Quickstart](#quickstart)
* [Requirements](#requirements)
* [Development Container](#development-container)
* [Running Tasks](#running-tasks)
* [Testing](#testing)
* [Updating the Base Docker](#updating-the-base-docker)
* [Building a Release Container](#building-a-release-container)
* [Troubleshooting](#troubleshooting)
* [Adding a New Network](#adding-a-new-network)
* [Contribution Guidelines](#contribution-guidelines)
* [License](#license)

## Repository Map

| Path | Purpose |
| :--- | :--- |
| `nvidia_tao_pytorch/core` | Shared runtime utilities: entrypoint launch, Hydra config loading, logging, telemetry, checkpoint handling, Lightning helpers, quantization, and export support. |
| `nvidia_tao_pytorch/config` | Structured dataclass configuration schemas used to validate experiment YAML files and generate default specs. |
| `nvidia_tao_pytorch/cv` | Computer vision task implementations, including model code, dataloaders, experiment specs, and task scripts. |
| `nvidia_tao_pytorch/multimodal` | Multimodal workflows such as CLIP and RADIO. |
| `nvidia_tao_pytorch/ssl` | Self-supervised learning workflows such as MAE and NVDINOv2. |
| `nvidia_tao_pytorch/sdg` | Synthetic data generation workflows, currently StyleGAN-XL. |
| `nvidia_tao_pytorch/pointcloud` | Point-cloud workflows, including PointPillars. |
| `tests` | Unit and integration tests grouped by task family. |
| `docker` | Development base-image Dockerfile, dependency lists, build scripts, and image manifest. |
| `release` | Release Dockerfile, wheel build scripts, and version metadata. |
| `runner/tao_pt.py` | Local developer launcher that runs commands inside the TAO PyTorch development container. |
| `tao-core` | TAO Core submodule with shared API, microservice, telemetry, and configuration infrastructure. |
| `tools` and `ci` | Documentation generation helpers, CI checks, changelog tooling, and release helpers. |

## Developer And Power-User Docs

The detailed developer and container-operation guides live in
[docs/index.md](docs/index.md). Start there for agent onboarding, architecture,
source-code workflows, testing/debugging, new-network integration, and prebuilt
container usage.

## Supported Commands

<!-- BEGIN GENERATED: supported-commands -->

The package installs one console command per model family. Each command discovers
its available subtasks from that model's `scripts` package. The `default_specs`
subtask is added by the shared TAO entrypoint for every registered model.

Run this command after adding or removing a model entrypoint:

```sh
python tools/update_readme_supported_commands.py
```

| Domain | Commands |
| :--- | :--- |
| Computer vision | `action_recognition`, `centerpose`, `classification_pyt`, `deformable_detr`, `depth_net`, `dino`, `grounding_dino`, `mal`, `mask2former`, `mask_grounding_dino`, `ml_recog`, `ocdnet`, `ocrnet`, `oneformer`, `optical_inspection`, `pose_classification`, `re_identification`, `rtdetr`, `segformer`, `visual_changenet` |
| 3D / point cloud | `bevfusion`, `pointpillars`, `sparse4d` |
| Multimodal | `clip`, `radio` |
| Self-supervised learning | `mae`, `nvdinov2` |
| Synthetic data generation | `stylegan_xl` |

| Command | Package | Subtasks |
| :--- | :--- | :--- |
| `action_recognition` | `cv.action_recognition` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `bevfusion` | `cv.bevfusion` | `dataset_convert`, `default_specs`, `evaluate`, `inference`, `train` |
| `centerpose` | `cv.centerpose` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `classification_pyt` | `cv.classification_pyt` | `default_specs`, `distill`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `clip` | `multimodal.clip` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `deformable_detr` | `cv.deformable_detr` | `convert`, `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `depth_net` | `cv.depth_net` | `convert`, `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `dino` | `cv.dino` | `convert`, `default_specs`, `distill`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `grounding_dino` | `cv.grounding_dino` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `mae` | `ssl.mae` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `mal` | `cv.mal` | `default_specs`, `evaluate`, `inference`, `train` |
| `mask2former` | `cv.mask2former` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `mask_grounding_dino` | `cv.mask_grounding_dino` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `ml_recog` | `cv.ml_recog` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `nvdinov2` | `ssl.nvdinov2` | `default_specs`, `export`, `inference`, `train` |
| `ocdnet` | `cv.ocdnet` | `default_specs`, `evaluate`, `export`, `inference`, `prune`, `quantize`, `train` |
| `ocrnet` | `cv.ocrnet` | `dataset_convert`, `default_specs`, `evaluate`, `export`, `inference`, `prune`, `quantize`, `train` |
| `oneformer` | `cv.oneformer` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `optical_inspection` | `cv.optical_inspection` | `dataset_convert`, `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `pointpillars` | `pointcloud.pointpillars` | `dataset_convert`, `default_specs`, `evaluate`, `export`, `inference`, `prune`, `train` |
| `pose_classification` | `cv.pose_classification` | `dataset_convert`, `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `radio` | `multimodal.radio` | `default_specs`, `distill` |
| `re_identification` | `cv.re_identification` | `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `rtdetr` | `cv.rtdetr` | `default_specs`, `distill`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `segformer` | `cv.segformer` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `sparse4d` | `cv.sparse4d` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |
| `stylegan_xl` | `sdg.stylegan_xl` | `dataset_convert`, `default_specs`, `evaluate`, `export`, `inference`, `train` |
| `visual_changenet` | `cv.visual_changenet` | `default_specs`, `evaluate`, `export`, `inference`, `quantize`, `train` |

For a specific model family, run the command with no subtask or with `-h` to see
the supported subtasks:

```sh
depth_net -h
depth_net train -h
```

<!-- END GENERATED: supported-commands -->

## Quickstart

Start by preparing the checkout and development shell:

```sh
git lfs install
git lfs pull
git submodule update --init --recursive
source scripts/envsetup.sh
```

The `envsetup.sh` script sets `NV_TAO_PYTORCH_TOP` and defines the `tao_pt`
function for launching the development container.

Launch an interactive container with the repository mounted at `/tao-pt`. Mount
a host workspace if you want specs, datasets, checkpoints, and results to
persist outside the container:

```sh
tao_pt --gpus all \
       --run_as_user \
       --volume /path/on/host/tao-workspace:/workspace \
       -- bash
```

Inside the container, generate a default experiment spec for a model family:

```sh
cd /tao-pt
depth_net default_specs results_dir=/workspace/specs/depth_net
```

Edit the generated `/workspace/specs/depth_net/experiment.yaml` with dataset,
checkpoint, and output paths, then run a task:

```sh
depth_net train -e /workspace/specs/depth_net/experiment.yaml
depth_net evaluate -e /workspace/specs/depth_net/experiment.yaml
depth_net export -e /workspace/specs/depth_net/experiment.yaml
```

Most task commands accept Hydra-style overrides after the experiment spec:

```sh
depth_net train -e /workspace/specs/depth_net/experiment.yaml train.num_gpus=1 train.gpu_ids=[0]
```

## Requirements

### Hardware

Minimum system configuration:

* 8 GB system RAM
* 4 GB GPU RAM
* 8 CPU cores
* 1 NVIDIA GPU
* 100 GB SSD space

Recommended system configuration:

* 32 GB system RAM
* 32 GB GPU RAM
* 8 CPU cores
* 1 NVIDIA GPU
* 100 GB SSD space

Model-specific requirements can be higher, especially for large transformer,
multimodal, 3D, and segmentation workloads.

### Host Software

| Software | Version |
| :--- | :--- |
| Ubuntu LTS | >= 18.04 |
| Python | >= 3.10 |
| Docker CE | > 19.03.5 |
| Docker API | >= 1.40 |
| `nvidia-container-toolkit` | > 1.3.0-1 |
| NVIDIA driver | > 535.85 |
| `python-pip` | > 21.06 |

The development and release containers provide the Python, CUDA, PyTorch, and
third-party packages used by TAO workflows. Prefer running tasks through
`tao_pt` unless you are intentionally developing against a local environment.

## Development Container

TAO Toolkit provides a base development Dockerfile at `docker/Dockerfile`. The
container keeps CUDA, PyTorch, TensorRT, and model dependencies consistent across
developer machines.

`tao_pt` usage:

```sh
usage: tao_pt [-h] [--gpus GPUS] [--volume VOLUME] [--env ENV]
              [--no-tty] [--mounts_file MOUNTS_FILE] [--shm_size SHM_SIZE]
              [--run_as_user] [--tag TAG] [--cached CACHED]
              [--ulimit ULIMIT] [--port PORT]
```

| Option | Description | Default |
| :--- | :--- | :--- |
| `--gpus` | Comma-separated GPU indices, `all`, or `none`. | `all` |
| `--volume` | Bind mount in `host_path:container_path` form. Can be repeated. | None |
| `--env` | Environment variable in `NAME=value` form. Can be repeated. | None |
| `--no-tty` | Run without allocating an interactive TTY. | TTY enabled |
| `--mounts_file` | Additional mounts file. | None |
| `--shm_size` | Docker shared memory size. | `16G` |
| `--run_as_user` | Run as the host UID/GID to avoid root-owned output files. | Disabled |
| `--tag` | Use a locally tagged development image instead of the manifest digest. | None |
| `--cached` | Use a fully specified cached image reference. | None |
| `--ulimit` | Docker ulimit option. Can be repeated. | None |
| `--port` | Port mapping such as `8889:8889`. The launcher also uses host networking. | None |

Example:

```sh
tao_pt --gpus all \
       --run_as_user \
       --volume /path/to/data:/data \
       --volume /path/to/results:/results \
       -- bash
```

### Mounts File

Large datasets often live outside the repository. You can define reusable Docker
mounts in `~/.tao_mounts.json` or pass another file with `--mounts_file`.

Example:

```json
{
  "Mounts": [
    {
      "source": "/path/to/your/experiments",
      "destination": "/workspace/tao-experiments"
    },
    {
      "source": "/path/to/config/files",
      "destination": "/workspace/tao-experiments/specs"
    }
  ]
}
```

## Running Tasks

The common task form is:

```sh
<model_command> <subtask> -e <experiment.yaml> [hydra.overrides]
```

Examples:

```sh
dino train -e /workspace/specs/dino.yaml
dino evaluate -e /workspace/specs/dino.yaml evaluate.checkpoint=/results/dino/model.tlt
dino export -e /workspace/specs/dino.yaml export.checkpoint=/results/dino/model.tlt
```

The launcher validates that an experiment spec exists, configures visible GPUs,
forwards Hydra overrides, streams logs, and records task telemetry. Some
distributed workflows are launched with `torchrun`; most PyTorch Lightning
workflows are launched directly with Python and manage devices through the
shared training utilities.

## Testing

Run tests from the repository root, preferably inside the development container:

```sh
pytest tests/cv_unit_test/depth_net
pytest tests/core
pytest -m "cv_unit and not tensorrt"
```

Useful targeted examples:

```sh
pytest tests/cv_unit_test/depth_net/test_export.py
pytest tests/multimodal_unit_test
pytest tests/ssl_unit_test
```

For static and CI-style checks, see the scripts in `ci/`, especially
`ci/run_static_tests.py`, `ci/test_changed_files.py`, and
`ci/run_functional_tests.py`.

To verify the generated README command table is up to date:

```sh
python tools/update_readme_supported_commands.py --check
```

## Updating the Base Docker

Use this flow when updating CUDA, PyTorch, system dependencies, or common Python
dependencies.

The base development Dockerfile is `docker/Dockerfile`. Python package
requirements live in `docker/requirements-pip*.txt`, and apt package
requirements live in `docker/requirements-apt.txt`.

The build script supports x86_64/AMD64 and ARM64. By default, it builds for
`linux/amd64`.

```sh
cd $NV_TAO_PYTORCH_TOP/docker

# Build for x86_64/AMD64.
./build.sh --build --x86

# Build for ARM64.
./build.sh --build --arm

# Build and push both platforms.
./build.sh --build --multiplatform --push
```

For more build options:

```sh
./build.sh --help
```

The build script tags the newly built base Docker image with the current
username. Test it with:

```sh
tao_pt --tag $USER -- bash
```

After validating the image, push it and update `docker/manifest.json` with the
new digest:

```sh
./build.sh --build --push --x86
./build.sh --build --push --multiplatform
```

To force a rebuild without using cache:

```sh
./build.sh --build --push --force --x86
```

## Building a Release Container

The release container is built on top of the TAO PyTorch base development image.
The release flow builds a wheel for `nvidia_tao_pytorch` and installs it into
the image defined by `release/docker/Dockerfile`.

```sh
git lfs install
git lfs pull
git submodule update --init --recursive
source scripts/envsetup.sh
cd $NV_TAO_PYTORCH_TOP
./release/docker/deploy.sh --build --wheel
```

Before cutting a release, confirm the wheel/package version in
`release/python/version.py` and the release Docker tag assembled in
`release/docker/deploy.sh`.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| `tao_pt` is not found | Run `source scripts/envsetup.sh` from the repository root. |
| Docker cannot pull the image | Run `docker login nvcr.io` and confirm access to the configured NGC registry. |
| Docker permission denied | Add your user to the `docker` group or run Docker with the required privileges. |
| GPUs are not visible in the container | Check the NVIDIA driver, `nvidia-container-toolkit`, and the `--gpus` argument. |
| Imports from `nvidia_tao_core` fail in local development | Initialize the `tao-core` submodule and source `scripts/envsetup.sh`. |
| Output files are owned by root | Launch with `tao_pt --run_as_user ...`. |
| Large assets or release files are missing | Run `git lfs install` and `git lfs pull`. |
| An experiment YAML fails schema validation | Generate a fresh spec with `<model> default_specs results_dir=<dir>` and compare fields. |

## Adding a New Network

See [Integrating a New Network in TAO PyTorch](docs/new_network_integration.md)
for the expected source layout, CLI registration steps, config schema patterns,
task-script conventions, tests, and TAO Core / FTMS integration notes.

## Contribution Guidelines

This repository is maintained as part of NVIDIA TAO Toolkit. Follow the existing
task structure when adding or updating a model family:

* Add or update dataclass schemas under `nvidia_tao_pytorch/config/<task>`.
* Keep model code, dataloaders, scripts, experiment specs, and entrypoints under
  the corresponding task package.
* Use shared utilities from `nvidia_tao_pytorch/core` when possible.
* Add focused tests under the matching `tests/*` task directory.
* Update this README or task-specific documentation when commands or workflows
  change.

## License

This project is licensed under the [Apache-2.0](./LICENSE) License.
