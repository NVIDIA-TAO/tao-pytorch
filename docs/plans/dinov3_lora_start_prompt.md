# Start Prompt — DINOv3 LoRA Implementation Kickoff

Paste the block below into a fresh Claude session (Cowork with the `tao` folder mounted) to
begin execution. It assumes the three planning docs in `tao-pytorch/docs/plans/` exist.

---

We're implementing LoRA + preservation-loss continual pre-training for DINOv3 in tao-pytorch.
All design decisions are already made — do not re-derive them. Read, in order:

1. `tao-pytorch/docs/plans/dinov3_lora_design.md` — architecture & rationale
2. `tao-pytorch/docs/plans/dinov3_lora_tasks.md` — phased task breakdown
3. `tao-pytorch/docs/plans/dinov3_lora_devbox_plan.md` — execution runbook + gate ladder (G0–G5)

Key constraints you must honor (from the design doc):
- Key-preserving LoRA: `LoRALinear(nn.Linear)` subclass, base state-dict keys unchanged,
  `lora_A`/`lora_B` added. NOT the CLIP branch's `.original`-nesting wrapper.
- Inject into BOTH student and EMA-teacher backbones, after `restore_pretrained_weights`,
  before `trainer.fit` (EMA zip alignment).
- Trainable set: lora_A/B + dino_head + ibot_head + mask_token. Everything else in the
  backbone frozen.
- The frozen Gram teacher gets NO LoRA; `_sync_gram_teacher` filters `lora_*` keys; anchor
  teacher must build when gram OR preservation is enabled.
- CLS preservation piggybacks on the existing Gram-teacher forward in `_extra_losses`
  (one anchor forward per step, not two).
- Guards: assert against `distill.enable`, `gram.teacher_source='ema'`, and FSDP in v1.
- Merge-on-export in `convert` (timm) and `export` (ONNX); merged output must be stock
  DINOv3 topology.

Start with Phase 0 (task doc) on branch `feature/dinov3_lora` off `main`:
create `ssl/dinov3/model/lora.py`, extend `config/dinov3/default_config.py`
(LoRAConfig: + dropout, target_modules, num_last_blocks; new PreservationConfig), wire
injection in `ssl/dinov3/scripts/train.py` / `pl_model.py`, add
`tests/ssl_unit_test/dinov3/test_lora.py` covering gates G1.1–G1.7 from the devbox plan.
Follow existing dinov3 code patterns (see how GramLoss/gram_teacher are integrated). Run the
existing dinov3 unit tests before and after to prove no regression. DCO sign-off on commits
(`git commit -s`); don't touch unrelated worktree changes.

When Phase 0 is green locally, stop and give me: files changed, test results, and the exact
commands for Stage 0–2 of the devbox plan on a4u8g-0146.

## Project tracking (JIRA) and MR conventions

Before starting implementation, create JIRA tickets on `https://jirasw.nvidia.com` (use the
jira skill if available, else `ci/create_jira_issue.py`):

- 1 Epic: "DINOv3 LoRA + preservation continual pre-training"
- Tasks under it, one per deliverable chunk (matching `dinov3_lora_tasks.md`):
  T1 Phase 0 scaffolding + unit tests (0.1–0.7) · T2 Stage 0–2 devbox smoke ·
  T3 convert/merge/export (Stage 3) · T4 ablation arms + retention analysis (Stage 4) ·
  T5 downstream consumption (Stage 5) · T6 docs/specs/MR wrap-up
- Record the ticket IDs in `docs/plans/dinov3_lora_tasks.md` next to each phase.

MR titles are CI-validated (`validate_mr_title`) against:
`^\[(JIRA|TLT|TAO|TAOVN)-[0-9]+\]\[(CI|CD|Bugfix|Hotfix|CV|LLM|Feature|Model-Cards|Docs|Chore)\] .+$`

So every MR in the stack must be titled like `[TAO-<id>][Feature] DINOv3 LoRA: <component>`,
using the JIRA ID of the matching task. Suggested MR stack (small, reviewable, ordered):

1. `[TAO-<T1>][Feature] DINOv3 LoRA: injection, config, unit tests` — branch `feature/dinov3_lora`
2. `[TAO-<T3>][Feature] DINOv3 LoRA: merge-on-convert + ONNX export` — stacked on 1
3. `[TAO-<T6>][Docs] DINOv3 LoRA: design docs, specs, ablation results` — stacked on 2

Reference the JIRA ID in commit messages too (`git commit -s -m "[TAO-<id>] ..."`), and put
the gate results (G1–G5) from the devbox plan in each MR description as the test evidence.

---

## Devbox quick-setup (a4u8g-0146) — copy-paste

```sh
ssh local-vpraveen@10.63.138.157
cd ~/Software/tao-pytorch && git fetch origin && git checkout feature/dinov3_lora
source scripts/envsetup.sh
export EXP=/media/scratch.metropolis4/users/vpraveen/dinov3_lora
mkdir -p $EXP/{baseline,smoke,arms,convert,downstream,logs,specs,datasets}
[ -f ~/.tao_mounts.json ] && mv ~/.tao_mounts.json ~/.tao_mounts.json.bak

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

# Inner script pattern (container setup is ~10-15 min CUDA compile):
cat > ~/inner_setup.sh << 'EOF'
#!/bin/bash
set -eo pipefail
pip install -e /tao-pt/tao-core/.
cd /tao-pt && python setup.py develop
dinov3 --help
EOF
tmux new -s setup "bash ~/launch_dinov3_lora.sh ~/inner_setup.sh 2>&1 | tee $EXP/logs/setup.log"
```

Reminders: no `--run_as_user`; `NCCL_P2P_DISABLE=1` for multi-GPU; long jobs in tmux;
outputs to scratch.metropolis4; check for pre-existing weights at
`/media/scratch.metropolis4/users/vpraveen/dinov3_vitb_v1` before downloading.
