#!/usr/bin/env bash
# S1 gate: 500-step LoRA smoke on one H100 node, then the G2 audit and the drift probe.
#
# This is also the FIRST validation of the LoRA path on fp16 + xformers + H100. The devbox
# gates ran on A100, and its fp32 numerical checks ran with custom attention disabled (the
# xformers path hard-casts q/k/v to .half()), so this combination is genuinely unexercised.
#
# Run inside a cluster-side tmux so a dropped laptop connection cannot kill it:
#   ssh cw-pdx 'tmux new -d -s smoke "bash <repo>/scripts/slurm/dinov3_lora_smoke.sh"'
#
# Step arithmetic: the 1k Foxconn subset at global batch 128 (8 GPU x 16) is ~7 steps/epoch,
# so 64 epochs lands at ~500 steps -- the same step count as the devbox Stage-2 smoke.

set -uo pipefail

LUSTRE_USER=/lustre/fsw/portfolios/edgeai/users/vpraveen
PROJ=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip

REPO_DIR="${REPO_DIR:-$LUSTRE_USER/repos/tao-pytorch}"
TAO_CORE_DIR="${TAO_CORE_DIR:-$REPO_DIR/tao-core}"
SPEC_DIR="$REPO_DIR/nvidia_tao_pytorch/ssl/dinov3/experiment_specs"
SPEC_NAME=train_dinov3_vitb_lora.yaml
RESULT_DIR="${RESULT_DIR:-$LUSTRE_USER/outputs/dinov3_lora/smoke}"
DATA_DIR="${DATA_DIR:-$PROJ/users/nnagrajrao/dinov3_test_subsample/foxconn_wds_1k}"
PRETRAINED="${PRETRAINED:-$PROJ/hf_cache/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/edgeai/users/yuw/docker/tao_evfm_2603.sqsh}"
TMP_ROOT="${TMP_ROOT:-$LUSTRE_USER/tmp}"
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-64}"
PARTITION="${PARTITION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
HOST_HOME="${HOME}"

mkdir -p "$RESULT_DIR" "$TMP_ROOT" "$REPO_DIR/logs/slurm"

echo "=============================================="
echo "DINOv3 LoRA smoke  (TAO-2492)"
echo "host      : $(hostname)"
echo "date UTC  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo      : $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null))"
echo "data      : $DATA_DIR ($(ls "$DATA_DIR" 2>/dev/null | grep -c '\.jpg$') jpg)"
echo "results   : $RESULT_DIR"
echo "gpus      : $GPUS   epochs: $EPOCHS"
echo "=============================================="

for p in "$REPO_DIR" "$TAO_CORE_DIR" "$SPEC_DIR/$SPEC_NAME" "$DATA_DIR" "$PRETRAINED" "$CONTAINER"; do
    [ -e "$p" ] || { echo "Missing required path: $p" >&2; exit 2; }
done

export TAO_SSL_CHECKPOINT_KEEP_LAST_N=5
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

MOUNTS="$TMP_ROOT:/root,/lustre:/lustre,$HOST_HOME:$HOST_HOME"

echo
echo "########## STEP 1: train 500 steps ##########"
srun --partition="$PARTITION" --account=edgeai_tao-ptm_image-foundation-model-clip \
    --job-name=dinov3_lora_smoke_TAO-2492 \
    --nodes=1 --ntasks="$GPUS" --ntasks-per-node="$GPUS" --gpus-per-node="$GPUS" \
    --cpus-per-task=8 --time="$TIME_LIMIT" \
    --container-image="$CONTAINER" \
    --container-mounts="$MOUNTS" \
    --no-container-entrypoint --container-remap-root --container-writable \
    bash -lc "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export HOME='$HOST_HOME'
        export PYTHONPATH='$REPO_DIR:$TAO_CORE_DIR:'\"\${PYTHONPATH:-}\"
        cd '$REPO_DIR'
        /usr/bin/python nvidia_tao_pytorch/ssl/dinov3/scripts/train.py \\
            --config-path '$SPEC_DIR' --config-name '$SPEC_NAME' \\
            results_dir='$RESULT_DIR' \\
            dataset.train_dataset.images_dir='$DATA_DIR' \\
            train.num_nodes=1 train.num_gpus=$GPUS \\
            train.num_epochs=$EPOCHS \\
            train.checkpoint_interval=250 train.checkpoint_interval_unit=step \\
            train.pretrained_model_path='$PRETRAINED'
    " 2>&1 | tee "$RESULT_DIR/smoke_train.log"
TRAIN_RC=${PIPESTATUS[0]}
echo "train exit=$TRAIN_RC"

echo
echo "########## STEP 2: loss finiteness (G2.2) ##########"
if grep -qiE 'nan|inf' "$RESULT_DIR/smoke_train.log"; then
    echo "WARNING: 'nan'/'inf' appears in the log; showing context:"
    grep -inE 'nan|inf' "$RESULT_DIR/smoke_train.log" | head -20
else
    echo "PASS: no nan/inf in the training log."
fi
echo "--- last loss lines ---"
grep -oE 'train_loss[^,]*|dino_[a-z_]*loss[^,]*|ibot[^,]*|koleo[^,]*' "$RESULT_DIR/smoke_train.log" | tail -12

CKPT="$(find "$RESULT_DIR/train" "$RESULT_DIR" -maxdepth 1 -type f -name 'model_*.pth' \
        -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -1 | cut -f2-)"
echo
echo "latest checkpoint: ${CKPT:-NONE}"
if [ -z "${CKPT:-}" ]; then
    echo "No checkpoint produced; stopping before audit." >&2
    exit 3
fi
echo "--- all checkpoints ---"
ls -lh "$RESULT_DIR"/train/*.pth "$RESULT_DIR"/*.pth 2>/dev/null | head -12

echo
echo "########## STEP 3: G2 audit (keys, rank, frozen base) ##########"
srun --partition="$PARTITION" --account=edgeai_tao-ptm_image-foundation-model-clip \
    --job-name=dinov3_lora_audit_TAO-2492 \
    --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 --time=00:30:00 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --no-container-entrypoint --container-remap-root --container-writable \
    bash -lc "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export HOME='$HOST_HOME'
        export PYTHONPATH='$REPO_DIR:$TAO_CORE_DIR:'\"\${PYTHONPATH:-}\"
        cd '$REPO_DIR'
        /usr/bin/python scripts/slurm/dinov3_lora_smoke_audit.py \\
            --checkpoint '$CKPT' --pretrained '$PRETRAINED' \\
            --output '$RESULT_DIR/audit.json'
    " 2>&1 | tee "$RESULT_DIR/smoke_audit.log"
AUDIT_RC=${PIPESTATUS[0]}
echo "audit exit=$AUDIT_RC"

echo
echo "########## STEP 4: drift probe (revised G4.5) ##########"
srun --partition="$PARTITION" --account=edgeai_tao-ptm_image-foundation-model-clip \
    --job-name=dinov3_lora_probe_TAO-2492 \
    --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=8 --time=00:30:00 \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    --no-container-entrypoint --container-remap-root --container-writable \
    bash -lc "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export HOME='$HOST_HOME'
        export PYTHONPATH='$REPO_DIR:$TAO_CORE_DIR:'\"\${PYTHONPATH:-}\"
        cd '$REPO_DIR'
        /usr/bin/python scripts/slurm/dinov3_lora_drift_probe.py \\
            --spec '$SPEC_DIR/$SPEC_NAME' --checkpoint '$CKPT' \\
            --images-dir '$DATA_DIR' --output '$RESULT_DIR/probe.json' \\
            --num-images 32 --arm smoke
    " 2>&1 | tee "$RESULT_DIR/smoke_probe.log"
PROBE_RC=${PIPESTATUS[0]}
echo "probe exit=$PROBE_RC"

echo
echo "=============================================="
echo "SMOKE SUMMARY"
echo "  train : exit $TRAIN_RC"
echo "  audit : exit $AUDIT_RC   ($RESULT_DIR/audit.json)"
echo "  probe : exit $PROBE_RC   ($RESULT_DIR/probe.json)"
echo "  logs  : $RESULT_DIR/smoke_*.log"
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================="
touch "$RESULT_DIR/smoke_done"
