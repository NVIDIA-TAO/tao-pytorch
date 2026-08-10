# OneFormer TAO 7.1 source overlay

This overlay is for the pinned image:

```text
nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.1.0-rc-245-multiarch
sha256 e36640f9ae7a03bc80828cf7de93bd6bdbbb0fecf509a71a243be0ab5b497fc2
size   28860358656
```

The SQSH location, overlay output, and result directories are deployment
inputs; only the image digest and size above are part of the reproducibility
contract.

The installer validates every base-file hash directly against the immutable
package root in the pinned container, then writes patched files into a separate
ephemeral site-packages tree placed first on `PYTHONPATH`. It never audits the
empty output tree as if that were the base, never mutates the SQSH package
root, and fails before installation when `panopticapi` is unavailable.

## Reproducible build

Build only from a clean reviewed commit:

```bash
cd <tao-pytorch-checkout>
python release/oneformer_runtime_overlay/build_overlay.py \
  --output <overlay-output>/oneformer-runtime-overlay.tar
sha256sum -c \
  <overlay-output>/oneformer-runtime-overlay.tar.sha256
```

The tar archive is deterministic: file order, ownership, modes, and mtimes are
fixed. Rebuilding twice from the same clean commit must produce the same
SHA-256.

Extract and transfer the directory to Lustre without changing its contents:

```bash
mkdir -p <temporary-extract-dir>
tar -xf <overlay-output>/oneformer-runtime-overlay.tar \
  -C <temporary-extract-dir>
rsync -a --checksum \
  <temporary-extract-dir>/oneformer-runtime-overlay/ \
  <slurm-login>:<overlay-deployment-dir>/
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

SQSH=<sqsh-path>
OVERLAY=<overlay-deployment-dir>
RESULTS=<results-dir>/<immutable-job-id>
SPEC=<spec-path>

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
