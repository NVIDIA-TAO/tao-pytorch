#!/usr/bin/env bash
# Submit the Stage-4 arms concurrently, one node each.
#
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_all_arms.sh B C D E'      # submit
#   ssh cw-pdx 'bash <repo>/scripts/slurm/submit_all_arms.sh --dry B C D E' # validate only
#
# The arms are independent experiments, so they parallelise ACROSS arms rather than within
# one. That matters: running four 1-node arms concurrently reuses the exact configuration
# validated in S1 (global batch 128, LR 3.2e-5, requeue+resume proven end to end), whereas
# giving one arm four nodes would change the global batch to 512 and force the LR to be
# re-derived, on the one experiment whose entire purpose is cross-arm comparability.
#
# Node budget is the number of arms submitted -- 4 arms = 4 nodes held.

set -uo pipefail

DRY=0
if [[ "${1:-}" == "--dry" ]]; then DRY=1; shift; fi

ARMS=("$@")
[[ ${#ARMS[@]} -gt 0 ]] || { echo "Usage: submit_all_arms.sh [--dry] <ARM> [ARM...]" >&2; exit 2; }

LUSTRE_USER="${LUSTRE_USER:-/lustre/fsw/portfolios/edgeai/users/vpraveen}"
REPO_DIR="${REPO_DIR:-${LUSTRE_USER}/repos/tao-pytorch}"
cd "${REPO_DIR}" || exit 5

echo "=== Stage-4 arm submission ==="
echo "date (UTC) : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit     : $(git rev-parse --short HEAD)"
echo "arms       : ${ARMS[*]}  (${#ARMS[@]} nodes total, 1 per arm)"
echo "dry run    : ${DRY}"
echo

echo "=== fairshare before submission ==="
sshare -al 2>/dev/null | grep -E "Account|edgeai_tao-ptm.*${USER}" || true
echo

# The per-arm wrapper enforces the concurrency limit; raise it to the number of arms so each
# submission sees room for the ones already queued.
export MAX_JOBS=${#ARMS[@]}

rc=0
for arm in "${ARMS[@]}"; do
    echo "---------- arm ${arm} ----------"
    bash scripts/slurm/submit_lora_arm.sh "${arm}" "${DRY}" || rc=$?
    echo
done

echo "=== queue after submission ==="
squeue -u "${USER}" -o '%.10i %.28j %.9T %.10M %.6D %R' || true
exit $rc
