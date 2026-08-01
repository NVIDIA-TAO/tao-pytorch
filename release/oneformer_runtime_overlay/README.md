# OneFormer TAO 7.1 source overlay

This overlay is for the pinned image:

```text
nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.1.0-rc-245-multiarch
/lustre/fsw/portfolios/edgeai/users/rarunachalam/nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh
sha256 e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2
size   28860358656
```

The installer validates every base-file hash directly against the immutable
package root in the pinned container, then writes patched files into a separate
ephemeral site-packages tree placed first on `PYTHONPATH`. It never audits the
empty output tree as if that were the base, never mutates the SQSH package
root, and fails before installation when `panopticapi` is unavailable.

## Reproducible build

Build only from a clean reviewed commit:

```bash
cd /localhome/local-rarunachalam/.tao/worktrees/tao-pytorch-oneformer-runtime-product-fixes
python release/oneformer_runtime_overlay/build_overlay.py \
  --output /localhome/local-rarunachalam/.tao/artifacts/oneformer-runtime-overlay.tar
sha256sum -c \
  /localhome/local-rarunachalam/.tao/artifacts/oneformer-runtime-overlay.tar.sha256
```

The tar archive is deterministic: file order, ownership, modes, and mtimes are
fixed. Rebuilding twice from the same clean commit must produce the same
SHA-256.

Extract and transfer the directory to Lustre without changing its contents:

```bash
mkdir -p /tmp/oneformer-overlay-extract
tar -xf /localhome/local-rarunachalam/.tao/artifacts/oneformer-runtime-overlay.tar \
  -C /tmp/oneformer-overlay-extract
rsync -a --checksum \
  /tmp/oneformer-overlay-extract/oneformer-runtime-overlay/ \
  <slurm-login>:/lustre/fsw/portfolios/edgeai/users/rarunachalam/artifacts/oneformer-runtime-overlay/
```

Verify the SQSH and overlay hashes on the SLURM login node before submission.
Do not launch if either differs from the committed manifest.

## Exact eight-GPU SQSH integration

The existing campaign generator may render the equivalent command. The
essential Pyxis/Enroot contract is:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --wait-all-nodes=1

set -euo pipefail

SQSH=/lustre/fsw/portfolios/edgeai/users/rarunachalam/nvcr.io_nvstaging_tao_tao-toolkit-pyt_7.1.0-rc-245-multiarch.sqsh
OVERLAY=/lustre/fsw/portfolios/edgeai/users/rarunachalam/artifacts/oneformer-runtime-overlay
RESULTS=/lustre/fsw/portfolios/edgeai/users/rarunachalam/results/<immutable-job-id>
SPEC=/lustre/fsw/portfolios/edgeai/users/rarunachalam/specs/<frozen-oneformer-spec>.yaml

test "$(sha256sum "$SQSH" | awk '{print $1}')" = \
  e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2
mkdir -p "$RESULTS"

srun \
  --container-image="$SQSH" \
  --container-mounts=/lustre,"$OVERLAY":/opt/oneformer-runtime-overlay:ro \
  --container-writable \
  bash -lc '
    set -euo pipefail
    OVERLAY_SITE="$(mktemp -d)/site-packages"
    python /opt/oneformer-runtime-overlay/install_overlay.py \
      --base-site-packages /usr/local/lib/python3.12/dist-packages \
      --site-packages "$OVERLAY_SITE" \
      --receipt "'"$RESULTS"'/oneformer-runtime-overlay-receipt.json"
    export PYTHONPATH="$OVERLAY_SITE${PYTHONPATH:+:$PYTHONPATH}"
    oneformer train -e "'"$SPEC"'" \
      train.num_gpus=8 \
      evaluate.task=panoptic \
      results_dir="'"$RESULTS"'"
  '
```

Submit through the platform SDK/SLURM skill using `sbatch --parsable`; preserve
the returned job ID and immutable manifest. The command above intentionally
uses all eight GPUs in the one-node allocation and the SQSH image directly.
For semantic campaigns, set `evaluate.task=semantic` and consume `mIoU`.
For COCO panoptic campaigns, set `evaluate.task=panoptic` and consume `PQ`.
