#!/usr/bin/env bash
# Submit (or dry-run) one DINOv3 LoRA Stage-4 arm on cw-pdx.
#
# Why this wrapper exists: the cw-pdx login shell is /bin/csh, so the documented
#   ssh cluster 'RESULT_DIR=... ARM=D sbatch scripts/slurm/dinov3_lora_vitb.sbatch'
# fails -- inline `VAR=value cmd` assignment is bash syntax and csh rejects it with
# "Illegal variable name.". Everything the remote side needs is therefore positional:
#
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_lora_arm.sh D'          # submit
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_lora_arm.sh D 1'        # DRY_RUN
#
# Usage: submit_lora_arm.sh <ARM> [DRY_RUN] [TIMEOUT_DURATION] [EXTRA_TRAIN_OVERRIDES...]
#
#   ARM                 B | C | D | E
#   DRY_RUN             1 to validate paths and print the plan without submitting (default 0)
#   TIMEOUT_DURATION    passed through to the sbatch (default 3.65h). The requeue-resume
#                       drill uses a short value such as 15m to force one requeue cycle.

set -euo pipefail

ARM="${1:?Usage: submit_lora_arm.sh <ARM: B|C|D|E> [DRY_RUN] [TIMEOUT_DURATION] [overrides...]}"
DRY_RUN="${2:-0}"
TIMEOUT_DURATION="${3:-3.65h}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA_OVERRIDES="$*"

LUSTRE_USER="${LUSTRE_USER:-/lustre/fsw/portfolios/edgeai/users/vpraveen}"
REPO_DIR="${REPO_DIR:-${LUSTRE_USER}/repos/tao-pytorch}"
RESULT_DIR="${RESULT_DIR:-${LUSTRE_USER}/outputs/dinov3_lora/arm_${ARM}}"

case "${ARM}" in
    B|C|D|E) ;;
    *) echo "Unknown ARM '${ARM}'. Expected one of: B C D E." >&2; exit 2 ;;
esac

cd "${REPO_DIR}"

export ARM DRY_RUN TIMEOUT_DURATION REPO_DIR RESULT_DIR
export TRAIN_OVERRIDES="${EXTRA_OVERRIDES}"

echo "=== submit_lora_arm.sh ==="
echo "host       : $(hostname)"
echo "date (UTC) : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "arm        : ${ARM}"
echo "repo       : ${REPO_DIR}"
echo "results    : ${RESULT_DIR}"
echo "timeout    : ${TIMEOUT_DURATION}"
echo "dry run    : ${DRY_RUN}"
echo "extra ovr  : ${EXTRA_OVERRIDES:-none}"
echo "git commit : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo

# Fairshare is logged on every submission per the plan's operating rules -- it is the number
# that decides whether we should be queueing at all, so it belongs in the job's own record.
echo "=== fairshare (edgeai) ==="
sshare -al 2>/dev/null | grep -E 'Account|edgeai_tao-ptm.*vpraveen' || echo "(sshare returned nothing)"
echo

# One training job queued or running at a time (plan rule). Counted before we add another.
echo "=== our current queue ==="
squeue -u "${USER}" -o '%.10i %.24j %.9T %.10M %.6D %R' || true
running="$(squeue -u "${USER}" -h -o '%i' 2>/dev/null | wc -l)"
echo "jobs in queue: ${running}"
echo

if [[ "${DRY_RUN}" != "1" && "${running}" -gt 0 ]]; then
    echo "Refusing to submit: ${running} job(s) already queued/running." >&2
    echo "The plan allows one training job at a time. Cancel or wait, then retry." >&2
    exit 3
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== DRY RUN: executing the sbatch body without submitting ==="
    bash scripts/slurm/dinov3_lora_vitb.sbatch
    exit $?
fi

echo "=== submitting ==="
sbatch --job-name="dinov3_lora_${ARM}_TAO-2492" scripts/slurm/dinov3_lora_vitb.sbatch
