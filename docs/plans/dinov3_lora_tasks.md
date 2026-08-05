# DINOv3 LoRA + Preservation — Task Breakdown & Execution Plan

Companion to `dinov3_lora_design.md`. Target branch: `feature/dinov3_lora` off `main`.
Execution machine: **`a4u8g-0146`** (8×A100 80 GB PCIe, tracks `main`, direct internet —
timm/HF DINOv3 weights download in place; `NCCL_P2P_DISABLE=1` for multi-GPU).

## Phase 0 — Scaffolding & unit tests (local / CI, no GPU)

| # | Task | Files | Acceptance |
|---|------|-------|------------|
| 0.1 | Key-preserving `LoRALinear(nn.Linear)` + `inject_lora(backbone, cfg)` + `merge_lora(backbone)` + trainable-param stats | `ssl/dinov3/model/lora.py` (new) | merged-vs-unmerged forward parity ≤ 1e-5 (fp32); state-dict keys = base keys ∪ `lora_*` |
| 0.2 | Extend `LoRAConfig` (dropout, `target_modules`, `num_last_blocks`); add `PreservationConfig` (enable, `cls_mse_weight`, `cls_cosine_weight`) | `config/dinov3/default_config.py` | `default_specs` generation passes; spec YAMLs updated |
| 0.3 | Injection lifecycle in train flow: inject into student+teacher backbones after `restore_pretrained_weights`, freeze base, keep heads + `mask_token` trainable; guards vs `distill.enable`, `gram.teacher_source='ema'`, FSDP | `ssl/dinov3/scripts/train.py`, `ssl/dinov3/model/pl_model.py` | unit test: student/teacher `named_parameters()` name lists identical (EMA zip safety); grads exist only on lora/heads/mask_token |
| 0.4 | `_sync_gram_teacher` lora-key filtering; decouple anchor-teacher construction from `gram.enable` (build when gram OR preservation enabled) | `ssl/dinov3/model/pl_model.py` | sync works pre- and post-injection |
| 0.5 | CLS preservation losses in `_extra_losses`, sharing the single anchor-teacher forward with Gram | `ssl/dinov3/model/pl_model.py`, `ssl/dinov3/model/loss.py` | loss values logged (`losses/cls_mse`, `losses/cls_cos`); zero when weights are 0 |
| 0.6 | `merge_lora_state_dict()` in checkpoint remap; wire into `convert` and ONNX `export` | `ssl/dinov3/utils/checkpoint_remap.py`, `scripts/convert.py`, `scripts/export.py` | convert `validate=True` passes against fresh timm model from a LoRA checkpoint |
| 0.7 | Unit tests for all of the above | `tests/ssl_unit_test/dinov3/test_lora.py` (new) | `pytest tests/ssl_unit_test/dinov3 -m "not gpu"` green |

Dependencies: 0.1 → 0.3 → {0.4, 0.5}; 0.1 → 0.6; tests last.

## Phase 1 — Single-GPU smoke (0146)

Setup on 0146 (once): `cd ~/Software/tao-pytorch && git fetch && git checkout feature/dinov3_lora`;
launch via a **staged launcher script** (never inline `bash -c`), back up `~/.tao_mounts.json`,
no `--run_as_user`. In-container: `pip install tao-core/. && python setup.py develop` (~10–15 min).

Assets already on shares (check before downloading):
- DINOv3 ViT-B weights: `/media/scratch.metropolis4/users/vpraveen/dinov3_vitb_v1` (+ `_smoke`), CI ckpts under `/media/scratch.metropolis2/tao_ci/testmon/feature_dinov3-*`
- ImageNet: `/media/scratch.metropolis3/zaid/imagenet-1k`, `/media/projects.metropolis2/public/imagenet2012`
- Outputs → `/media/scratch.metropolis4/users/vpraveen/dinov3_lora/` (~13 TB free)
- ADE20K: **not on 0146 shares** — download once for the dense-retention eval (direct internet)

| # | Task | Acceptance |
|---|------|------------|
| 1.1 | ViT-B + LoRA smoke, 1 GPU, ~200 steps on small ImageNet subset | all losses finite & trending down; `losses/gram_loss`, `losses/cls_*` logged |
| 1.2 | Grad audit at step N: dump `requires_grad` + grad-norm per param group | only lora/heads/mask_token nonzero |
| 1.3 | Checkpoint artifacts: Lightning ckpt + stripped `student_*.pth`/`teacher_*.pth` contain `lora_*` keys; resume-from-ckpt works | resume run continues loss curve |
| 1.4 | `dinov3 convert` from LoRA ckpt → merged timm backbone; load in `backbone_v2` registry | validate pass + feature parity vs in-training merged forward |
| 1.5 | Baseline reference: identical run with `lora.enable=false` (full FT) for later comparison | metrics recorded |

## Phase 2 — Ablations: does preservation earn its keep? (0146, 1–2 GPUs)

Protocol per arm (~10–20k steps, ViT-B, domain dataset TBD — pick one from shares, e.g. a
Metropolis vertical): measure **in-domain k-NN**, **ImageNet k-NN retention**, and
**VOC/ADE20K linear-seg mIoU retention** (dense geometry — the metric Gram anchoring protects).

| Arm | lora | w_gram | cls weights | Question |
|-----|------|--------|-------------|----------|
| A | off | 0 | 0 | full-FT drift baseline |
| B | on | 0 | 0 | how much does rank-bounding alone preserve? |
| C | on | 1.0 | 0 | Gram (patch geometry) contribution |
| D | on | 1.0 | 0.05/0.05 | + CLS (global geometry) contribution |
| E | on | 2.0 | 0.10/0.10 | over-constraint check (in-domain gain should shrink) |

Deliverable: table + recommendation for shipped defaults; rank sweep (4/8/16) on the winning arm
if time allows.

## Phase 3 — Multi-GPU + integration (0146)

| # | Task | Notes |
|---|------|-------|
| 3.1 | 8-GPU DDP run, `NCCL_P2P_DISABLE=1`, `distributed_strategy=ddp` | EMA zip correctness under DDP; throughput number |
| 3.2 | End-to-end: train → convert → downstream consumption via `pretrained_backbone_path` in all three consumers: `classification_pyt` (`dinov3_vitb16`), `segformer` (`vit_base_dinov3`, frozen-backbone mode), `visual_changenet` | feature parity: converted backbone through `backbone_v2` ≈ in-training merged student; frozen-backbone segformer run trains |
| 3.3 | Docs + spec YAMLs (`train_dinov3_vitb_lora.yaml`), README command-table check, DCO commits, MR | `python tools/update_readme_supported_commands.py --check` |

## Deferred (explicitly out of v1)

FSDP + LoRA (`use_orig_params=True`, ViT-H+/7B); distillation-mode LoRA; `gram.teacher_source='ema'`
with LoRA; LoRA on SwiGLU fc1/fc2 for H+/7B (blocked on FSDP anyway); tao-deploy TRT path
(unaffected — merged export is a stock DINOv3 ONNX).

## Estimated effort

Phase 0: ~2–3 days. Phase 1: 1 day (mostly container build + smoke iterations).
Phase 2: 2–3 days wall-clock (runs are short; analysis dominates). Phase 3: 1–2 days.
