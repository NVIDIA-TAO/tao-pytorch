#!/usr/bin/env bash
# Measure how DINOv3 LoRA training scales with per-GPU batch size, on one H100 node.
#
#   ssh cw-pdx 'tmux new -d -s sweep bash <repo>/scripts/slurm/dinov3_lora_batch_sweep.sh'
#
# Motivation. The Stage-4 arms run at batch 16/GPU and sit at ~20 GB of 80 GB with
# utilization pinned at 100%. That combination does not mean "saturated": nvidia-smi's
# utilization.gpu is the fraction of time at least one kernel was resident, not achieved
# FLOPs, so there may be real headroom. Two separate questions follow, and this measures both:
#
#   throughput  -- does a larger batch raise images/s, or are the GEMMs already efficient?
#   capacity    -- where does it actually OOM? (activations scale ~linearly with batch, but
#                  fragmentation and the 512 px global crops make linear extrapolation
#                  optimistic)
#
# The accuracy question -- larger batches give Sinkhorn-Knopp centering, KoLeo and prototype
# assignment better statistics, and DINOv2/v3 train at batch 1024-3072 against our 128 -- is
# NOT answered here. This measures cost, so a v2 recipe can be chosen knowing what it buys.
#
# Runs on the 1k Foxconn subset so each point is minutes, and touches nothing the arms use.

set -uo pipefail

LUSTRE_USER=/lustre/fsw/portfolios/edgeai/users/vpraveen
PROJ=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip

REPO_DIR="${REPO_DIR:-$LUSTRE_USER/repos/tao-pytorch}"
TAO_CORE_DIR="$REPO_DIR/tao-core"
SPEC_DIR="$REPO_DIR/nvidia_tao_pytorch/ssl/dinov3/experiment_specs"
SPEC_NAME=train_dinov3_vitb_lora.yaml
SWEEP_ROOT="${SWEEP_ROOT:-$LUSTRE_USER/outputs/dinov3_lora/batch_sweep}"
DATA_DIR="${DATA_DIR:-$PROJ/users/nnagrajrao/dinov3_test_subsample/foxconn_wds_1k}"
PRETRAINED="${PRETRAINED:-$PROJ/hf_cache/hub/models--timm--vit_base_patch16_dinov3.lvd1689m/snapshots/c6a5fb7d12bbd3cf3b0079253141c3332aaed7da}"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/edgeai/users/yuw/docker/tao_evfm_2603.sqsh}"
TMP_ROOT="$LUSTRE_USER/tmp"
GPUS="${GPUS:-8}"
BATCHES="${BATCHES:-16 32 48 64}"
# One extra point at the arms' batch with LoRA OFF, so the LoRA-vs-full-FT cost comparison is
# measured rather than reasoned about. Theory says it should be close: LoRA saves optimizer
# state and gradient buffers (~1 GB of ~20 GB) but not activation memory, because gradients
# still flow through the frozen layers to reach adapters in early blocks. This checks that.
FULLFT_BATCH="${FULLFT_BATCH:-16}"
RUN_FULLFT="${RUN_FULLFT:-1}"
TARGET_STEPS="${TARGET_STEPS:-40}"
POOL=1000                       # images in the smoke subset
HOST_HOME="${HOME}"

mkdir -p "$SWEEP_ROOT"
COMBINED_LOG="$SWEEP_ROOT/sweep.log"
exec > >(tee -a "$COMBINED_LOG") 2>&1

echo "=============================================="
echo "DINOv3 LoRA batch-size sweep (TAO-2492)"
echo "date UTC  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo      : $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null)"
echo "batches   : $BATCHES  (per GPU, x${GPUS} GPUs)"
echo "data      : $DATA_DIR"
echo "=============================================="

for BS in $BATCHES; do
    RD="$SWEEP_ROOT/bs_${BS}"
    mkdir -p "$RD"
    GLOBAL=$(( BS * GPUS ))
    STEPS_PER_EPOCH=$(( POOL / GLOBAL ))
    [ "$STEPS_PER_EPOCH" -lt 1 ] && STEPS_PER_EPOCH=1
    EPOCHS=$(( (TARGET_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))

    echo
    echo "########## batch ${BS}/GPU  (global ${GLOBAL}, ~${STEPS_PER_EPOCH} steps/epoch, ${EPOCHS} epochs) ##########"

    export TAO_VISIBLE_DEVICES
    TAO_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"
    export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 HYDRA_FULL_ERROR=1

    srun --partition=interactive --account=edgeai_tao-ptm_image-foundation-model-clip \
        --job-name=dinov3_lora_sweep_TAO-2492 \
        --nodes=1 --ntasks="$GPUS" --ntasks-per-node="$GPUS" --gpus-per-node="$GPUS" \
        --cpus-per-task=8 --time=00:40:00 \
        --container-image="$CONTAINER" \
        --container-mounts="$TMP_ROOT:/root,/lustre:/lustre,$HOST_HOME:$HOST_HOME" \
        --no-container-entrypoint --container-remap-root --container-writable \
        bash -lc "
            set -uo pipefail
            export PYTHONUNBUFFERED=1
            export HOME='$HOST_HOME'
            export TAO_VISIBLE_DEVICES='$TAO_VISIBLE_DEVICES'
            export PYTHONPATH='$REPO_DIR:$TAO_CORE_DIR:'\"\${PYTHONPATH:-}\"
            cd '$REPO_DIR'
            # Rank 0 samples memory for the whole node while training runs. Sampling from
            # inside the allocation avoids juggling job ids from the login node, and peak
            # reserved memory is what decides whether a batch size actually fits.
            if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then
                nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \\
                    --loop-ms=2000 > '$RD/mem.csv' 2>/dev/null &
                SMI_PID=\$!
            fi
            /usr/bin/python nvidia_tao_pytorch/ssl/dinov3/scripts/train.py \\
                --config-path '$SPEC_DIR' --config-name '$SPEC_NAME' \\
                results_dir='$RD' \\
                dataset.train_dataset.images_dir='$DATA_DIR' \\
                dataset.batch_size=$BS \\
                train.num_nodes=1 train.num_gpus=$GPUS \\
                train.num_epochs=$EPOCHS \\
                train.checkpoint_interval=100000 train.checkpoint_interval_unit=step \\
                train.pretrained_model_path='$PRETRAINED'
            rc=\$?
            if [ -n \"\${SMI_PID:-}\" ]; then kill \$SMI_PID 2>/dev/null; fi
            exit \$rc
        " 2>&1 | tee "$RD/train.log"
    RC=${PIPESTATUS[0]}
    echo "batch ${BS}: exit ${RC}"

    if grep -qiE 'out of memory|CUDA error: out of memory' "$RD/train.log" 2>/dev/null; then
        echo "batch ${BS}: OOM" > "$RD/result.txt"
        echo "  -> OOM at batch ${BS}; stopping the sweep here (larger will also OOM)."
        break
    fi
    echo "exit=${RC}" > "$RD/result.txt"
done

# --- full-FT reference point -------------------------------------------------------------
if [ "$RUN_FULLFT" = "1" ]; then
    BS="$FULLFT_BATCH"
    RD="$SWEEP_ROOT/fullft_bs_${BS}"
    mkdir -p "$RD"
    GLOBAL=$(( BS * GPUS ))
    STEPS_PER_EPOCH=$(( POOL / GLOBAL )); [ "$STEPS_PER_EPOCH" -lt 1 ] && STEPS_PER_EPOCH=1
    EPOCHS=$(( (TARGET_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))

    echo
    echo "########## FULL-FT reference: batch ${BS}/GPU, LoRA disabled ##########"
    export TAO_VISIBLE_DEVICES
    TAO_VISIBLE_DEVICES="$(seq -s, 0 "$((GPUS - 1))")"

    srun --partition=interactive --account=edgeai_tao-ptm_image-foundation-model-clip \
        --job-name=dinov3_fullft_sweep_TAO-2492 \
        --nodes=1 --ntasks="$GPUS" --ntasks-per-node="$GPUS" --gpus-per-node="$GPUS" \
        --cpus-per-task=8 --time=00:40:00 \
        --container-image="$CONTAINER" \
        --container-mounts="$TMP_ROOT:/root,/lustre:/lustre,$HOST_HOME:$HOST_HOME" \
        --no-container-entrypoint --container-remap-root --container-writable \
        bash -lc "
            set -uo pipefail
            export PYTHONUNBUFFERED=1
            export HOME='$HOST_HOME'
            export TAO_VISIBLE_DEVICES='$TAO_VISIBLE_DEVICES'
            export PYTHONPATH='$REPO_DIR:$TAO_CORE_DIR:'\"\${PYTHONPATH:-}\"
            cd '$REPO_DIR'
            if [ \"\${SLURM_PROCID:-0}\" = \"0\" ]; then
                nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \\
                    --loop-ms=2000 > '$RD/mem.csv' 2>/dev/null &
                SMI_PID=\$!
            fi
            /usr/bin/python nvidia_tao_pytorch/ssl/dinov3/scripts/train.py \\
                --config-path '$SPEC_DIR' --config-name '$SPEC_NAME' \\
                results_dir='$RD' \\
                dataset.train_dataset.images_dir='$DATA_DIR' \\
                dataset.batch_size=$BS \\
                train.num_nodes=1 train.num_gpus=$GPUS \\
                train.num_epochs=$EPOCHS \\
                train.checkpoint_interval=100000 train.checkpoint_interval_unit=step \\
                train.pretrained_model_path='$PRETRAINED' \\
                model.lora.enable=false model.gram.enable=false model.preservation.enable=false
            rc=\$?
            if [ -n \"\${SMI_PID:-}\" ]; then kill \$SMI_PID 2>/dev/null; fi
            exit \$rc
        " 2>&1 | tee "$RD/train.log"
    echo "full-FT: exit ${PIPESTATUS[0]}"
    echo "exit=${PIPESTATUS[0]}" > "$RD/result.txt"
fi

echo
echo "=============================================="
echo "SWEEP SUMMARY"
printf '%-8s %-9s %-10s %-12s %-12s %s\n' BATCH GLOBAL "IT/S" "IMAGES/S" "PEAK_MEM_GB" NOTE
for BS in $BATCHES; do
    RD="$SWEEP_ROOT/bs_${BS}"
    [ -d "$RD" ] || continue
    GLOBAL=$(( BS * GPUS ))
    NOTE="$(cat "$RD/result.txt" 2>/dev/null || echo 'not run')"

    # Median of the last 10 rates, so warm-up and the final partial step do not skew it.
    ITS="$(tr '\r' '\n' < "$RD/train.log" 2>/dev/null \
        | grep -oE '[0-9.]+it/s' | tail -10 | tr -d 'its/' \
        | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')"
    IMGS=""
    [ -n "${ITS:-}" ] && IMGS="$(awk -v i="$ITS" -v g="$GLOBAL" 'BEGIN{printf "%.1f", i*g}')"

    PEAK="$(awk -F', *' 'NF>=2 && $2+0>m {m=$2+0} END{if(m)printf "%.1f", m/1024}' "$RD/mem.csv" 2>/dev/null)"

    printf '%-8s %-9s %-10s %-12s %-12s %s\n' \
        "$BS" "$GLOBAL" "${ITS:-—}" "${IMGS:-—}" "${PEAK:-—}" "$NOTE"
done
RD="$SWEEP_ROOT/fullft_bs_${FULLFT_BATCH}"
if [ -d "$RD" ]; then
    GLOBAL=$(( FULLFT_BATCH * GPUS ))
    ITS="$(tr '\r' '\n' < "$RD/train.log" 2>/dev/null | grep -oE '[0-9.]+it/s' | tail -10 \
        | tr -d 'its/' | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')"
    IMGS=""; [ -n "${ITS:-}" ] && IMGS="$(awk -v i="$ITS" -v g="$GLOBAL" 'BEGIN{printf "%.1f", i*g}')"
    PEAK="$(awk -F', *' 'NF>=2 && $2+0>m {m=$2+0} END{if(m)printf "%.1f", m/1024}' "$RD/mem.csv" 2>/dev/null)"
    printf '%-8s %-9s %-10s %-12s %-12s %s\n' \
        "${FULLFT_BATCH}*" "$GLOBAL" "${ITS:-—}" "${IMGS:-—}" "${PEAK:-—}" "FULL-FT (LoRA off)"
fi
echo
echo "* full-FT reference at the arms' batch. LoRA should be close on both time and memory:"
echo "  it saves optimizer state and gradient buffers (~1 GB of ~20 GB), not activation"
echo "  memory, since gradients still flow through the frozen layers to reach the adapters."
echo "Reference: the Stage-4 arms ran at batch 16/GPU (global 128), ~0.13 it/s, ~19.5-20.6 GB."
echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================="
touch "$SWEEP_ROOT/sweep_done"
