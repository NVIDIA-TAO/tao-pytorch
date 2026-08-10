#!/usr/bin/env bash
# DINOv3 ViT-B LoRA + preservation continual pre-training on the Foxconn AOI pool.
# Stage-4 arms B/C/D/E -- see docs/plans/dinov3_lora_slurm_plan.md (repo: vpraveen/dinov3-lora).
#
# Derived from dinov3_foxconn_vitb.sbatch (nnagrajrao@dinov3-foxconn-slurm), deliberately kept
# as a small diff off that proven skeleton. What changed and why:
#   * 1 node instead of 10        -- LoRA arms are ~14k steps, and the plan mandates sequential
#                                    single-node submission for fairshare etiquette
#   * ARM selects TRAIN_OVERRIDES -- the four arms differ only in loss weights, so the arm is a
#                                    single positional input rather than four near-identical files
#   * JIRA ID in the job name     -- attribution per the plan
#   * post-train drift probe      -- writes probe.json into RESULT_DIR so every completed arm
#                                    carries its diagnostics without a separate job
# Everything else (container mounts, timeout 3.65h -> scontrol requeue, resume detection,
# TAO_SSL_CHECKPOINT_KEEP_LAST_N, WANDB toggles) is inherited unchanged.
#
# Submit through scripts/slurm/submit_lora_arm.sh -- the cw-pdx login shell is /bin/csh, so the
# `VAR=value sbatch ...` form does not work over SSH.

#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --job-name=dinov3_lora_TAO-2492
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

ARM="${ARM:?Set ARM to one of A, B, C, D, E.}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
TAO_CORE_DIR="${TAO_CORE_DIR:-${REPO_DIR}/tao-core}"
SPEC_DIR="${SPEC_DIR:-${REPO_DIR}/nvidia_tao_pytorch/ssl/dinov3/experiment_specs}"
SPEC_NAME="${SPEC_NAME:-train_dinov3_vitb_lora.yaml}"
DATA_DIR="${DATA_DIR:-/lustre/fsw/portfolios/edgeai/users/nnagrajrao/datasets/Foxconn_wds}"
RESULT_DIR="${RESULT_DIR:?Set RESULT_DIR to a durable output directory.}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/hf_cache/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/edgeai/users/yuw/docker/tao_evfm_2603.sqsh}"
TMP_ROOT="${TMP_ROOT:-/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/${USER}/tmp}"
TIMEOUT_DURATION="${TIMEOUT_DURATION:-3.65h}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-DINOv3 LoRA}"
WANDB_ENTITY="${WANDB_ENTITY:-metropolis}"
WANDB_NAME="${WANDB_NAME:-dinov3_lora_arm_${ARM}}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PROBE="${RUN_PROBE:-1}"
PROBE_NUM_IMAGES="${PROBE_NUM_IMAGES:-64}"

# Arm selection. The spec carries arm-D defaults, so arm D adds nothing; every other arm is a
# delta off it. Keeping the arms here rather than in four spec files guarantees that nothing
# except these weights differs between them -- which is the whole point of the ablation.
case "${ARM}" in
    # Arm A is the full-FT drift baseline: no LoRA, no anchoring. The slurm plan assumed
    # nnagrajrao's foxconn full-SSL run would supply it for free, but no such checkpoint was
    # ever retained (session-log finding S0-e), so it is ours to run -- and the plan's own
    # fallback applies: run it last, on the smallest schedule that shows drift.
    # It is the arm the retention ordering D >= C >= B > A is measured against.
    A)  ARM_OVERRIDES="model.lora.enable=false model.gram.enable=false model.gram.w_gram=0.0 model.preservation.enable=false" ;;
    B)  ARM_OVERRIDES="model.lora.enable=true model.gram.enable=false model.gram.w_gram=0.0 model.preservation.enable=false" ;;
    C)  ARM_OVERRIDES="model.lora.enable=true model.gram.enable=true model.gram.w_gram=1.0 model.preservation.enable=false" ;;
    D)  ARM_OVERRIDES="" ;;
    E)  ARM_OVERRIDES="model.lora.enable=true model.gram.enable=true model.gram.w_gram=2.0 model.preservation.enable=true model.preservation.cls_mse_weight=0.10 model.preservation.cls_cosine_weight=0.10" ;;
    *)  echo "Unknown ARM '${ARM}'. Expected one of: A B C D E." >&2; exit 2 ;;
esac

# Caller-supplied overrides win, so they come last on the command line.
TRAIN_OVERRIDES="${ARM_OVERRIDES} ${TRAIN_OVERRIDES:-}"

NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"
TOTAL_TASKS="$((NODE_COUNT * GPUS_PER_NODE))"
HOST_HOME="${HOME}"
PROBE_SCRIPT="${REPO_DIR}/scripts/slurm/dinov3_lora_drift_probe.py"

mkdir -p "${RESULT_DIR}" "${TMP_ROOT}" logs/slurm

for path in \
    "${REPO_DIR}" \
    "${TAO_CORE_DIR}" \
    "${SPEC_DIR}/${SPEC_NAME}" \
    "${PROBE_SCRIPT}" \
    "${DATA_DIR}" \
    "${PRETRAINED_MODEL_PATH}" \
    "${CONTAINER}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing required path: ${path}" >&2
        exit 2
    fi
done

if [[ -f "${RESULT_DIR}/train_done" ]]; then
    echo "Training already completed: ${RESULT_DIR}"
    exit 0
fi

# Resume-target selection, newest first but *validated*.
#
# `timeout` kills training at TIMEOUT_DURATION regardless of what it is doing, and a full
# Lightning checkpoint here is ~1.3 GB and takes tens of seconds to write. If the kill lands
# inside that window the newest model_*.pth is truncated, and resuming from it crashes the
# next cycle -- which would then requeue and crash again, indefinitely. Across ~9 requeues per
# arm and four arms that is enough boundary crossings for it to happen.
#
# Guard: walk checkpoints newest-first and take the first whose size is within 1% of the
# largest one present. Checkpoint sizes are near-identical (observed 1295438661 vs
# 1295438789, a 128-byte spread), so a truncated file is obvious by size alone and this needs
# no torch import to detect.
resume_args=()
mapfile -t ckpt_candidates < <(
    find "${RESULT_DIR}/train" "${RESULT_DIR}" -maxdepth 1 -type f \
        -name 'model_*.pth' -printf '%T@\t%s\t%p\n' 2>/dev/null \
        | sort -rn | cut -f2-
)

latest_ckpt=""
if [[ ${#ckpt_candidates[@]} -gt 0 ]]; then
    max_size=0
    for entry in "${ckpt_candidates[@]}"; do
        size="${entry%%$'\t'*}"
        (( size > max_size )) && max_size="${size}"
    done
    min_ok=$(( max_size * 99 / 100 ))
    for entry in "${ckpt_candidates[@]}"; do
        size="${entry%%$'\t'*}"
        path="${entry#*$'\t'}"
        if (( size >= min_ok )); then
            latest_ckpt="${path}"
            break
        fi
        echo "Skipping truncated checkpoint (${size} B < ${min_ok} B): ${path}" >&2
    done
fi

if [[ -n "${latest_ckpt}" ]]; then
    resume_args+=("train.resume_training_checkpoint_path=${latest_ckpt}")
fi

echo "Job        : ${SLURM_JOB_ID:-not-submitted}"
echo "Arm        : ${ARM}"
echo "Repository : ${REPO_DIR}"
echo "Spec       : ${SPEC_DIR}/${SPEC_NAME}"
echo "Data       : ${DATA_DIR}"
echo "Results    : ${RESULT_DIR}"
echo "Container  : ${CONTAINER}"
echo "Nodes/GPUs : ${NODE_COUNT}/${TOTAL_TASKS}"
echo "Resume     : ${latest_ckpt:-none}"
echo "Overrides  : ${TRAIN_OVERRIDES:-none}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run passed; no Slurm workload was started."
    exit 0
fi

export TAO_VISIBLE_DEVICES
TAO_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS_PER_NODE - 1))")"
export TAO_SSL_CHECKPOINT_KEEP_LAST_N=3
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export WANDB_ENABLE WANDB_PROJECT WANDB_ENTITY WANDB_NAME

set +e
timeout "${TIMEOUT_DURATION}" srun \
    --ntasks="${TOTAL_TASKS}" \
    --ntasks-per-node="${GPUS_PER_NODE}" \
    --container-image="${CONTAINER}" \
    --container-mounts="${TMP_ROOT}:/root,/lustre:/lustre,${HOST_HOME}:${HOST_HOME}" \
    --no-container-entrypoint \
    --container-remap-root \
    --container-writable \
    bash -lc "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export HOME='${HOST_HOME}'
        export PYTHONPATH='${REPO_DIR}:${TAO_CORE_DIR}:'\"\${PYTHONPATH:-}\"
        cd '${REPO_DIR}'
        wandb_overrides=()
        if [[ \"\${WANDB_ENABLE:-0}\" == \"1\" ]]; then
            wandb_overrides=(
                wandb.enable=true
                \"wandb.project=\${WANDB_PROJECT}\"
                \"wandb.entity=\${WANDB_ENTITY}\"
                \"wandb.name=\${WANDB_NAME}\"
            )
        fi
        /usr/bin/python nvidia_tao_pytorch/ssl/dinov3/scripts/train.py \\
            --config-path '${SPEC_DIR}' \\
            --config-name '${SPEC_NAME}' \\
            results_dir='${RESULT_DIR}' \\
            dataset.train_dataset.images_dir='${DATA_DIR}' \\
            train.num_nodes='${NODE_COUNT}' \\
            train.num_gpus='${GPUS_PER_NODE}' \\
            train.pretrained_model_path='${PRETRAINED_MODEL_PATH}' \\
            \"\${wandb_overrides[@]}\" \\
            ${resume_args[*]:-} \\
            ${TRAIN_OVERRIDES}
    "
status=$?
set -e

if [[ ${status} == 124 ]]; then
    echo "Training reached ${TIMEOUT_DURATION}; requeueing ${SLURM_JOB_ID}."
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
fi
if [[ ${status} != 0 ]]; then
    echo "Training failed with status ${status}" | tee "${RESULT_DIR}/failed" >&2
    exit "${status}"
fi

touch "${RESULT_DIR}/train_done"
echo "Training completed: ${RESULT_DIR}"

# ---------------------------------------------------------------------------------------------
# Post-training drift probe (revised gate G4.5).
#
# Runs on a single GPU inside the same container, right after training, so a completed arm is
# self-describing. Deliberately non-fatal: the arm's checkpoints are the expensive artifact and
# a probe failure must not mark a finished 14k-step run as failed. A missing probe.json is
# visible on the dashboard and the probe can be re-run standalone against the same checkpoint.
# ---------------------------------------------------------------------------------------------
if [[ "${RUN_PROBE}" != "1" ]]; then
    echo "Drift probe skipped (RUN_PROBE=${RUN_PROBE})."
    exit 0
fi

probe_ckpt="$(
    find "${RESULT_DIR}/train" "${RESULT_DIR}" -maxdepth 1 -type f \
        -name 'model_*.pth' -printf '%T@\t%p\n' 2>/dev/null \
        | sort -n | tail -n 1 | cut -f2- || true
)"
if [[ -z "${probe_ckpt}" ]]; then
    echo "Drift probe skipped: no model_*.pth found under ${RESULT_DIR}." >&2
    exit 0
fi

echo "Drift probe : ${probe_ckpt}"
set +e
srun --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 \
    --container-image="${CONTAINER}" \
    --container-mounts="${TMP_ROOT}:/root,/lustre:/lustre,${HOST_HOME}:${HOST_HOME}" \
    --no-container-entrypoint \
    --container-remap-root \
    --container-writable \
    bash -lc "
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export HOME='${HOST_HOME}'
        export PYTHONPATH='${REPO_DIR}:${TAO_CORE_DIR}:'\"\${PYTHONPATH:-}\"
        cd '${REPO_DIR}'
        /usr/bin/python scripts/slurm/dinov3_lora_drift_probe.py \\
            --spec '${SPEC_DIR}/${SPEC_NAME}' \\
            --checkpoint '${probe_ckpt}' \\
            --images-dir '${DATA_DIR}' \\
            --output '${RESULT_DIR}/probe.json' \\
            --num-images '${PROBE_NUM_IMAGES}' \\
            --arm '${ARM}'
    "
probe_status=$?
set -e

if [[ ${probe_status} != 0 ]]; then
    echo "Drift probe failed with status ${probe_status} (training output is intact)." >&2
else
    echo "Drift probe written: ${RESULT_DIR}/probe.json"
fi
exit 0
