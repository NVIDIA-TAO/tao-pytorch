#!/usr/bin/env bash
# S1 gate: prove the 4-hour requeue+resume loop preserves a LoRA run.
#
# Every Stage-4 arm will be requeued 2-5 times, so G2.6 (resume continuity) stops being a
# nice-to-have here and becomes load-bearing: if a resumed checkpoint silently dropped its
# lora_* keys, an arm would quietly continue as the frozen base and the ablation would be
# measuring nothing.
#
#   ssh cw-pdx 'bash <repo>/scripts/slurm/dinov3_lora_requeue_drill.sh submit'
#   ssh cw-pdx 'bash <repo>/scripts/slurm/dinov3_lora_requeue_drill.sh verify'
#
# Deliberately writes to its OWN results dir (drill_D, not arm_D) so the drill cannot
# contaminate the real arm-D run that Stage 4 will submit later.
#
# Mechanism: TIMEOUT_DURATION=15m instead of 3.65h, so `timeout` fires early, the script hits
# its exit-124 branch, calls `scontrol requeue`, and Slurm restarts the same job ID -- the
# identical code path as the real 4-hour boundary, just reached ~15x sooner.
# checkpoint_interval is dropped to 100 steps so a checkpoint definitely exists before the
# first timeout (the spec's 1000 would likely not be reached in 15 minutes at 512px).

set -uo pipefail

LUSTRE_USER=/lustre/fsw/portfolios/edgeai/users/vpraveen
REPO_DIR="${REPO_DIR:-$LUSTRE_USER/repos/tao-pytorch}"
RESULT_DIR="$LUSTRE_USER/outputs/dinov3_lora/drill_D"
ACTION="${1:-submit}"

cd "$REPO_DIR" || exit 5

case "$ACTION" in
submit)
    running="$(squeue -u "$USER" -h -o '%i' 2>/dev/null | wc -l)"
    if [ "$running" -gt 0 ]; then
        echo "Refusing: $running job(s) already queued/running (one at a time)." >&2
        squeue -u "$USER" -o '%.10i %.30j %.9T %.8M'
        exit 3
    fi
    mkdir -p "$RESULT_DIR"
    echo "=== requeue drill: submitting arm D with a 15m timeout ==="
    echo "results : $RESULT_DIR"
    echo "commit  : $(git rev-parse --short HEAD)"
    sshare -al 2>/dev/null | grep -E "edgeai_tao-ptm.*$USER" || true

    ARM=D \
    RESULT_DIR="$RESULT_DIR" \
    TIMEOUT_DURATION=15m \
    TRAIN_OVERRIDES="train.checkpoint_interval=100 train.checkpoint_interval_unit=step" \
    RUN_PROBE=0 \
    sbatch --job-name=dinov3_lora_drill_TAO-2492 scripts/slurm/dinov3_lora_vitb.sbatch
    ;;

verify)
    echo "=== requeue drill: verification ==="
    echo "results : $RESULT_DIR"
    echo
    echo "--- job history (RESTART/REQUEUE shows the loop fired) ---"
    sacct -u "$USER" --name=dinov3_lora_drill_TAO-2492 \
        --starttime now-1days --parsable2 --noheader \
        -o JobID,State,Elapsed,Start,End 2>/dev/null | grep -v '\.' | head -20
    echo
    echo "--- checkpoints (step numbers must continue ACROSS the requeue, not restart at 0) ---"
    find "$RESULT_DIR" -name 'model_*.pth' -printf '%TY-%Tm-%TdT%TH:%TM %10s %f\n' 2>/dev/null | sort
    echo
    echo "--- requeue markers in the log ---"
    grep -hE 'requeue|Requeue|reached 15m|Resume *:' "$REPO_DIR"/logs/slurm/dinov3_lora_drill_*.out 2>/dev/null | head -20
    echo
    echo "--- loss continuity across the boundary ---"
    grep -hoE 'epoch=[0-9]+|step=[0-9]+|train_loss=[0-9.]+|v_num=[^ ]*' \
        "$REPO_DIR"/logs/slurm/dinov3_lora_drill_*.out 2>/dev/null | tail -40
    ;;

*)
    echo "Usage: dinov3_lora_requeue_drill.sh [submit|verify]" >&2
    exit 2
    ;;
esac
