# DINOv3 LoRA — Devbox Execution & Test Plan (a4u8g-0146)

Runbook for executing the `dinov3_lora_design.md` proposal end-to-end on **`a4u8g-0146`**
(8×A100 80 GB PCIe, tracks `main`, direct internet). Every stage has explicit **success
gates** — quantitative, mechanically checkable criteria — so progress is measurable and a
failure is localized to the stage that introduced it.

Scope: LoRA + preservation training → checkpoint artifacts → convert/merge → downstream
consumption (classification_pyt, segformer, visual_changenet) → ONNX export.

---

## Stage 0 — Environment prep & frozen baseline numbers

**Goal:** working container + the reference metrics every later retention gate compares against.

### 0.1 Host setup

```sh
ssh local-vpraveen@10.63.138.157        # a4u8g-0146

# Workspace + branch
cd ~/Software/tao-pytorch
git fetch origin && git checkout feature/dinov3_lora   # branch created in Phase 0 (local dev)
source scripts/envsetup.sh

# Experiment root (scratch.metropolis4 = most free space)
export EXP=/media/scratch.metropolis4/users/vpraveen/dinov3_lora
mkdir -p $EXP/{baseline,smoke,arms,convert,downstream,logs}

# Check pre-existing artifacts BEFORE downloading anything
ls /media/scratch.metropolis4/users/vpraveen/dinov3_vitb_v1/          # ViT-B weights (expected present)
ls /media/scratch.metropolis3/zaid/imagenet-1k/ | head                 # ImageNet
find /media/scratch.metropolis* -maxdepth 3 -iname "*ade20k*" 2>/dev/null   # ADE20K: expected ABSENT

# Mounts hygiene (avoids Duplicate mount point)
[ -f ~/.tao_mounts.json ] && cp ~/.tao_mounts.json ~/.tao_mounts.json.bak && rm ~/.tao_mounts.json
```

### 0.2 Container launcher (staged script — never inline `bash -c`)

```sh
cat > ~/launch_dinov3_lora.sh << 'EOF'
#!/bin/bash
set -eo pipefail
export PYTHONPATH="${PYTHONPATH:-}"
cd ~/Software/tao-pytorch
source scripts/envsetup.sh
tao_pt --gpus all \
  --volume /media/scratch.metropolis4:/data/scratch \
  --volume /media/scratch.metropolis3:/data/scratch3 \
  --volume /media/projects.metropolis2:/data/projects2 \
  -- bash "$1"
EOF
chmod +x ~/launch_dinov3_lora.sh
# NOTE: no --run_as_user (crashes getpass on PyTorch 2.11+ dev images)
```

In-container one-time setup (~10–15 min CUDA compile — plan for it in every scripted pipeline):

```sh
pip install -e /tao-pt/tao-core/.
cd /tao-pt && python setup.py develop
```

Long jobs: always under `tmux new -s dinov3_lora "bash ~/launch_dinov3_lora.sh <inner>.sh 2>&1 | tee $EXP/logs/<name>.log"`.

### 0.3 Data prep

```sh
# ADE20K (not on 0146 shares; direct internet — no proxy dance)
cd $EXP && wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip -q ADEChallengeData2016.zip -d $EXP/datasets/

# ImageNet smoke subset (100 classes, ~130k imgs) — build file list, don't copy data
# Domain dataset for adaptation arms: pick from shares (candidates below) — decision gate D0
```

Domain-dataset candidates on mounted shares (pick one, unlabeled images suffice for SSL):
`/media/scratch.metropolis3/datasets/*`, `/media/projects.metropolis2/datasets/*`,
retail/ITS/warehouse verticals. **D0: confirm choice before Stage 4.**

### 0.4 Baseline metrics (stock DINOv3 ViT-B — the retention reference)

Run once, record in `$EXP/baseline/baseline.json`; all Stage-4/5 retention gates reference these:

| Metric | Protocol | Symbol |
|---|---|---|
| ImageNet-1k k-NN top-1 (k=20, CLS) | frozen backbone, train-set bank | `KNN_IN_0` |
| ImageNet-1k linear-probe top-1 | linear on CLS, short schedule | `LIN_IN_0` |
| ADE20K linear-seg mIoU | linear on patch tokens | `SEG_ADE_0` |
| Domain k-NN top-1 (if labels) or domain retrieval mAP | frozen backbone | `KNN_DOM_0` |

**Gate G0:** container builds; `dinov3 --help` resolves; baseline numbers within ~1 pt of
published DINOv3 ViT-B references (sanity that eval harness is trustworthy). ✅/❌ before Stage 1.

---

## Stage 1 — Static correctness (unit tests, <1 GPU-hour)

```sh
pytest tests/ssl_unit_test/dinov3/test_lora.py -v
pytest tests/ssl_unit_test/dinov3 -v          # no regressions in existing dinov3 tests
```

**Gates (all must pass):**

| # | Invariant | Test |
|---|---|---|
| G1.1 | **Identity**: injected LoRA (B=0) forward == stock forward, fp32 bit-exact tolerance ≤1e-6 | `test_injection_identity` |
| G1.2 | **Merge parity**: after random A/B, `forward(merge(m)) ≈ forward(m)` ≤1e-5 | `test_merge_parity` |
| G1.3 | **EMA alignment**: student/teacher `named_parameters()` name lists identical post-injection | `test_ema_zip_alignment` |
| G1.4 | **Freeze audit**: `requires_grad` true only for `lora_*`, heads, `mask_token` | `test_trainable_set` |
| G1.5 | **Key stability**: backbone state-dict keys == stock keys ∪ `lora_*`; `timm_to_tao` round-trip unaffected | `test_state_dict_keys` |
| G1.6 | **Gram-teacher sync** with LoRA-injected teacher (lora keys filtered) succeeds | `test_gram_sync_lora` |
| G1.7 | Optimizer groups: lora params land in correct `blocks.N` layer-id groups; no frozen params in any group | `test_optim_groups` |

---

## Stage 2 — Smoke training (1 GPU, ~1–2 h)

ViT-B, LoRA r=8 qkv+proj, gram+CLS preservation on, ImageNet 100-class subset, 500 steps,
`results_dir=$EXP/smoke`.

```sh
dinov3 train -e $EXP/specs/smoke_lora.yaml \
  model.lora.enable=true model.gram.enable=true model.preservation.enable=true \
  train.pretrained_model_path=/data/scratch/users/vpraveen/dinov3_vitb_v1
```

**Gates:**

| # | Check | How verified |
|---|---|---|
| G2.1 | **Zero-start preservation**: `losses/gram_loss`, `losses/cls_mse`, `losses/cls_cos` are ≈0 (<1e-6) at step 0 and grow smoothly | TensorBoard scalars — student==anchor at init (B=0); nonzero start ⇒ teacher-sync or remap bug |
| G2.2 | DINO/iBOT/KoLeo losses finite, no NaN/inf over 500 steps, `train_loss` trending down | TB + log grep |
| G2.3 | **Frozen-base hash**: SHA256 of all non-lora backbone tensors identical at step 0 vs step 500 (student AND teacher) | audit script `tools/lora_audit.py` (dump + compare) |
| G2.4 | **EMA correctness**: teacher `lora_*` ≈ EMA trajectory of student `lora_*`; equal at step 0 | audit script |
| G2.5 | Grad audit at step 100: nonzero grad-norms only on lora/heads/mask_token | audit script hook |
| G2.6 | Checkpoint artifacts: Lightning `.ckpt` + stripped `student_*.pth`/`teacher_*.pth` contain `lora_*` keys; resume from `.ckpt` continues loss curve without discontinuity | restart run, overlay curves |
| G2.7 | Trainable-param report logged: ~0.44 M lora (ViT-B r8 qkv+proj) + heads + mask_token | log grep |

**Diagnostics if gates fail:** G2.1 fails → check `_sync_gram_teacher` filter and injection order
vs `restore_pretrained_weights`; G2.3 fails → optimizer got frozen params (check G1.7) or
weight-decay applied outside groups; G2.4 fails → zip misalignment (G1.3 regressed under wrap).

---

## Stage 3 — Convert, merge & export (CPU/1 GPU, ~1 h)

```sh
# Merge + convert to timm format
dinov3 convert -e spec.yaml convert.checkpoint=$EXP/smoke/teacher_epoch_000_step_00500.pth \
  convert.output_path=$EXP/convert/dinov3_vitb_lora_merged.safetensors convert.validate=true

# ONNX export (merged)
dinov3 export -e spec.yaml export.checkpoint=... export.onnx_file=$EXP/convert/dinov3_vitb_lora.onnx
```

**Gates:**

| # | Check |
|---|---|
| G3.1 | `convert.validate=true` passes (merged dict loads strict into fresh timm `vit_base_patch16_dinov3`) |
| G3.2 | **End-to-end feature parity**: CLS + patch features of (a) in-container LoRA model, (b) converted backbone loaded via `backbone_v2.dinov3_vitb16(pretrained_backbone_path=...)` — cosine ≥0.9999, max-abs ≤1e-4 on 32 real images |
| G3.3 | Converted-minus-stock weight delta is nonzero exactly on lora target modules (qkv/proj), zero elsewhere — confirms merge touched only what it should |
| G3.4 | ONNX graph contains no LoRA ops (stock DINOv3 topology); onnxruntime forward parity vs torch ≤1e-3 |
| G3.5 | (optional, tao-deploy container) `gen_trt_engine` from the ONNX succeeds |

---

## Stage 4 — Adaptation arms & retention measurement (the science, 8 GPUs, ~2–3 days wall)

Domain dataset per **D0**. Each arm ~10–20k steps, DDP 8×A100, `NCCL_P2P_DISABLE=1`,
`train.distributed_strategy=ddp`. One tmux session per arm, sequential or 2×4-GPU pairs.

| Arm | lora | w_gram | cls w | Purpose |
|---|---|---|---|---|
| A | off | 0 | 0 | full-FT drift baseline |
| B | on | 0 | 0 | rank-bounding alone |
| C | on | 1.0 | 0 | + patch-geometry anchor |
| D | on | 1.0 | 0.05/0.05 | + global anchor (proposed default) |
| E | on | 2.0 | 0.10/0.10 | over-constraint probe |

After each arm: convert (Stage-3 pipeline) → evaluate the four Stage-0 metrics.

**Gates (define success of the whole proposal):**

| # | Criterion | Threshold (target, revisit after arm A/B land) |
|---|---|---|
| G4.1 | In-domain gain: `KNN_DOM(arm) > KNN_DOM_0` for all arms B–E | any improvement; D within 80% of B's gain |
| G4.2 | Global retention: `KNN_IN(D) ≥ KNN_IN_0 − 1.0 pt`; arm A expected to lose ≥2–3× more | retention ordering D ≥ C ≥ B > A |
| G4.3 | Dense retention: `SEG_ADE(D) ≥ SEG_ADE_0 − 1.0 mIoU`; C/D materially better than B | Gram term earns its keep, else drop it |
| G4.4 | Over-constraint: E's in-domain gain < D's | confirms weights are in the useful range |
| G4.5 | Monitored drift diagnostics logged every 500 steps: `cos(student_cls, anchor_cls)`, Gram-MSE — smooth, no cliffs | TB |

Deliverable: `$EXP/arms/results.md` table + recommended shipped defaults.

**Kill criteria** (stop and rethink, don't burn GPU): arm D shows no in-domain gain at 20k steps
(preservation too strong or domain too close to pretraining); arm B retention ≈ arm A
(rank-bounding does nothing → check freeze audit).

---

## Stage 5 — Downstream consumption (8 GPUs, ~1 day)

Use arm-D converted backbone from `$EXP/convert/`.

```sh
# 5a. classification_pyt: frozen-backbone linear probe on domain data
classification_pyt train -e cls_spec.yaml \
  model.backbone.type=dinov3_vitb16 \
  model.backbone.pretrained_backbone_path=$EXP/convert/dinov3_vitb_lora_merged.safetensors \
  model.backbone.freeze_at=-1   # per classifier freeze semantics

# 5b. segformer: frozen-backbone semantic seg (ADE20K or domain seg set)
segformer train -e seg_spec.yaml \
  model.backbone.type=vit_base_dinov3 \
  model.backbone.pretrained_backbone_path=$EXP/convert/dinov3_vitb_lora_merged.safetensors \
  model.backbone.freeze_backbone=true

# 5c. visual_changenet: load-and-train smoke with the same path
```

**Gates:**

| # | Check |
|---|---|
| G5.1 | All three consumers load the converted file with zero unexpected/missing keys in their load logs |
| G5.2 | classification frozen-probe on domain data: adapted backbone beats stock backbone (same spec, only `pretrained_backbone_path` differs) |
| G5.3 | segformer frozen-backbone: training converges; adapted ≥ stock on domain seg (if domain labels exist) and within 1 mIoU of stock on ADE20K (retention, mirrors G4.3 through a real head) |
| G5.4 | No code changes were required in any consumer (assert the design claim) |

G5.2/G5.3 are the **product-level success measure**: frozen-backbone downstream is where a
drifted backbone cannot hide behind the task head.

---

## Stage 6 — Wrap-up & MR

- Re-run full unit suite + `python tools/update_readme_supported_commands.py --check`.
- Ship spec: `experiment_specs/train_dinov3_vitb_lora.yaml` with arm-D defaults.
- Retrieve artifacts to local: two-hop (CIFS breaks rsync mkstemp) — `rsync` devbox→`~/staging/`,
  then `cp` from CIFS; or pull from local machine via
  `rsync -avz local-vpraveen@10.63.138.157:$EXP/arms/results.md ./`.
- MR per repo conventions (DCO `git commit -s`, HTTPS+token push through proxy per root CLAUDE.md).
- Report per remote-runner convention: command, host, exit status, key output lines.

---

## Summary of the gate ladder

```
G0  env + trusted baselines
G1  static invariants (identity, merge, EMA-zip, freeze, keys)
G2  dynamic invariants (zero-start preservation, frozen-base hash, EMA trajectory, resume)
G3  artifact invariants (convert parity, delta locality, ONNX topology)
G4  scientific outcome (in-domain gain vs retention ordering D ≥ C ≥ B > A)
G5  product outcome (frozen-backbone downstream wins, zero consumer changes)
```

Each gate depends only on the previous ones, so a red gate localizes the defect to one stage.
