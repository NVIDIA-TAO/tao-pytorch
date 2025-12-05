# TAO Toolkit - PyTorch Backend

<!-- vscode-markdown-toc -->
* [Overview](#Overview)
* [Getting Started](#GettingStarted)
	* [Requirements](#Requirements)
		* [Hardware Requirements](#HardwareRequirements)
		* [Software Requirements](#SoftwareRequirements)
	* [Instantiating the development container](#Instantiatingthedevelopmentcontainer)
		* [Command line options](#Commandlineoptions)
		* [Using the mounts file](#Usingthemountsfile)
	* [Updating the base docker](#Updatingthebasedocker)
		* [Build base docker](#Buildbasedocker)
		* [Test the newly built base docker](#Testthenewlybuiltbasedocker)
		* [Update the new docker](#Updatethenewdocker)
* [Building a release container](#Buildingareleasecontainer)
* [Contribution Guidelines](#ContributionGuidelines)
* [License](#License)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

## <a name='Overview'></a>Overview

TAO Toolkit is a Python package hosted on the NVIDIA Python Package Index. It interacts with lower-level TAO dockers available from the NVIDIA GPU Accelerated Container Registry (NGC). The TAO containers come pre-installed with all dependencies required for training. The output of the TAO workflow is a trained model that can be deployed for inference on NVIDIA devices using DeepStream, TensorRT and Triton.

This repository contains the required implementation for the all the deep learning components and networks using the PyTorch backend. These routines are packaged as part of the TAO Toolkit PyTorch container in the Toolkit package. These source code here is compatible with PyTorch version > 2.0.0

## <a name='GettingStarted'></a>Getting Started

As soon as the repository is cloned, run the `envsetup.sh` file to check
if the build environment has the necessary dependencies, and the required
environment variables are set.

```sh
source ${PATH_TO_REPO}/scripts/envsetup.sh
```

We recommend adding this command to your local `~/.bashrc` file, so that every new terminal instance receives this.

### <a name='Requirements'></a>Requirements

#### <a name='HardwareRequirements'></a>Hardware Requirements

##### Minimum system configuration

* 8 GB system RAM
* 4 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

##### Recommended system configuration

* 32 GB system RAM
* 32 GB of GPU RAM
* 8 core CPU
* 1 NVIDIA GPU
* 100 GB of SSD space

#### <a name='SoftwareRequirements'></a>Software Requirements

| **Software**                     | **Version** |
| :--- | :--- |
| Ubuntu LTS                       | >=18.04     |
| python                           | >=3.10.x     |
| docker-ce                        | >19.03.5    |
| docker-API                       | 1.40        |
| `nvidia-container-toolkit`       | >1.3.0-1    |
| nvidia-container-runtime         | 3.4.0-1     |
| nvidia-docker2                   | 2.5.0-1     |
| nvidia-driver                    | >535.85     |
| python-pip                       | >21.06      |

### <a name='Instantiatingthedevelopmentcontainer'></a>Instantiating the development container

Inorder to maintain a uniform development environment across all users, TAO Toolkit provides a base environment Dockerfile in `docker/Dockerfile` that contains all
the required third party dependencies for the developers. For instantiating the docker, simply run the `tao_pt` CLI. The usage for the command line launcher is mentioned below.

```sh
usage: tao_pt [-h] [--gpus GPUS] [--volume VOLUME] [--env ENV]
              [--mounts_file MOUNTS_FILE] [--shm_size SHM_SIZE]
              [--run_as_user] [--tag TAG] [--ulimit ULIMIT] [--port PORT]

Tool to run the pytorch container.

optional arguments:
  -h, --help                show this help message and exit
  --gpus GPUS               Comma separated GPU indices to be exposed to the docker.
  --volume VOLUME           Volumes to bind.
  --env ENV                 Environment variables to bind.
  --mounts_file MOUNTS_FILE Path to the mounts file.
  --shm_size SHM_SIZE       Shared memory size for docker
  --run_as_user             Flag to run as user
  --tag TAG                 The tag value for the local dev docker.
  --ulimit ULIMIT           Docker ulimits for the host machine.
  --port PORT               Port mapping (e.g. 8889:8889).

```

A sample command to instantiate an interactive session in the base development docker is mentioned below.

```sh
tao_pt --gpus all \
       --volume /path/to/data/on/host:/path/to/data/on/container \
       --volume /path/to/results/on/host:/path/to/results/in/container \
       --env PYTHONPATH=/tao-pt
```

Running Deep Neural Networks implies working on large datasets. These datasets are usually stored on network share drives with significantly higher storage capacity. Since the `tao_pt` CLI wrapper uses docker containers under the hood, these drives/mount points need to be mapped to the docker.

There are 2 ways to configure the `tao_pt` CLI wrapper. 

1. Via the command line options
2. Via the mounts file. By default, at `~/.tao_mounts.json`.

#### <a name='Commandlineoptions'></a>Command line options

| **Option**      | **Description** | **Default** |
| :-- | :-- | :-- |
| `gpus`         | Comma separated GPU indices to be exposed to the docker | 1 | 
| `volume`       | Paths on the host machine to be exposed to the container. This is analogous to the `-v` option in the docker CLI. You may define multiple mount points by using the --volume option multiple times.  | None |  
| `env`          | Environment variables to defined inside the interactive container. You may set them as `--env VAR=<value>`. Multiple environment variables can be set by repeatedly defining the `--env` option. | None |
| `mounts_file`  | Path to the mounts file, explained more in the next section. | `~/.tao_mounts.json` | 
| `shm_size`     | Shared memory size for docker in Bytes. | 16G |
| `run_as_user`  | Flag to run as default user account on the host machine. This helps with maintaining permissions for all directories and artifacts created by the container. | 
| `tag`          | The tag value for the local dev docker | None |
| `ulimit`       | Docker ulimits for the host machine | 
| `port`         | Port mapping (e.g. 8889:8889) | None |

#### <a name='Usingthemountsfile'></a>Using the mounts file

The `tao_pt` CLI wrapper instance can be configured by using a mounts file. By default, the wrapper expects the mounts file to be at 
`~/.tao_mounts.json`. However, for multiple options, you may be able 

The launcher config file consists of three sections:

* `Mounts`

The `Mounts` parameter defines the paths in the local machine, that should be mapped to the docker. This is a list of `json` dictionaries containing the source path in the local machine and the destination path that is mapped for the CLI wrapper.

A sample config file containing 2 mount points and no docker options is as below.

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

### <a name='Updatingthebasedocker'></a>Updating the base docker

There will be situations where developers would be required to update the third party dependancies to newer versions, or upgrade CUDA etc. In such a case, please follow the steps below:

#### <a name='Buildbasedocker'></a>Build base docker

The base dev docker is defined in `$NV_TAO_PYTORCH_TOP/docker/Dockerfile`. The python packages required for the TAO dev is defined in `$NV_TAO_PYTORCH_TOP/docker/requirements-pip.txt` and the third party apt packages are defined in `$NV_TAO_PYTORCH_TOP/docker/requirements-apt.txt`. Once you have made the required change, please update the base docker using the build script in the same directory.

The build script supports building for different platforms (x86_64/AMD64 and ARM64). By default, it builds for `linux/amd64` (x86_64).

**Cross-platform builds:** When building for a different architecture than your host (e.g., ARM64 on x86_64), the script automatically detects and configures QEMU emulation. The QEMU setup persists on your system and is reused for future builds.

```sh
cd $NV_TAO_PYTORCH_TOP/docker
# Build for x86_64/AMD64 (default platform)
./build.sh --build --x86

# Build for ARM64 (automatically sets up QEMU if on x86_64 host)
./build.sh --build --arm

# Build for both platforms (requires --push flag)
./build.sh --build --multiplatform --push
```

For more build options and help, run:

```sh
./build.sh --help
```

#### <a name='Testthenewlybuiltbasedocker'></a>Test the newly built base docker

The build script tags the newly built base docker with the username of the account in the user's local machine. Therefore, the developers may tests their new docker by using the `tao_pt` command with the `--tag` option.

```sh
tao_pt --tag $USER -- script args
```

#### <a name='Updatethenewdocker'></a>Update the new docker

Once you are sufficiently confident about the newly built base docker, please do the following

1. Push the newly built base docker to the registry

    ```sh
    # Push for x86_64/AMD64 platform
    bash $NV_TAO_PYTORCH_TOP/docker/build.sh --build --push --x86
    
    # Or push multi-platform build (x86_64 and ARM64)
    bash $NV_TAO_PYTORCH_TOP/docker/build.sh --build --push --multiplatform
    ```
    
    **Note:** Multi-platform builds automatically push to the registry due to Docker buildx limitations. Single platform builds can be built locally without the `--push` flag for testing.

2. The above step produces a digest file associated with the docker. This is a unique identifier for the docker. So please note this, and update all references of the old digest in the repository with the new digest. You may find the old digest in the `$NV_TAO_PYTORCH_TOP/docker/manifest.json`.

Push you final updated changes to the repository so that other developers can leverage and sync with the new dev environment.

Please note that if for some reason you would like to force build the docker without using a cache from the previous docker, you may do so by using the `--force` option.

```sh
# Force rebuild without cache
bash $NV_TAO_PYTORCH_TOP/docker/build.sh --build --push --force --x86
```

## <a name='Buildingareleasecontainer'></a>Building a release container

The TAO docker is built on top of the TAO Pytorch base dev docker, by building a python wheel for the `nvidia_tao_pyt` module in this repository and installing the wheel in the Dockerfile defined in `release/docker/Dockerfile`. The whole build process is captured in a single shell script which may be run as follows:

```sh
git lfs install
git lfs pull
source scripts/envsetup.sh
cd $NV_TAO_PYTORCH_TOP
./release/docker/deploy.sh --build --wheel
```

In order to build a new docker, please edit the `deploy.sh` file in `$NV_TAO_PYTORCH_TOP/release/docker` to update the patch version and re-run the steps above.

## <a name='ContributionGuidelines'></a>Contribution Guidelines
TAO Toolkit PyTorch backend is not accepting contributions as part of the TAO 5.0 release, but will be open in the future.

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.
