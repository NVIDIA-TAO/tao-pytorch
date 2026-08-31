# PAS V3.1.1 partial-negative alignment reproduction

Use `experiment_siglip2_pas_v3_1_dual_tower_lora_pa.yaml` to reproduce the
fixed SigLIP2-256 + standard LoRA + partial-negative alignment (PA) run. This
profile does not use SigLIP2-384, RandLoRA, AutoML, reranking, or query
expansion.

PA here means only the partial-negative component from Song et al., “Dual
alignment: Partial negative and soft-label alignment for text-to-image person
retrieval,” Information Fusion 127 (2026), DOI
`10.1016/j.inffus.2025.103644`. The soft-label alignment component is not
implemented, so this code must not be described as the paper's full method.

## Data contract

Mount the balanced PAS V3.1.1 derived dataset at `/workspace/data/pas`, or
change all data paths in the spec together. The directory must contain:

- `images/`
- `captions/`
- `train_list.txt` and `train_pairs.json`
- `val_list.txt` and `val_pairs.json`
- `attribute_vocab.json` and `accessory_vocab.json`

The fixed profile uses `val_list.txt` for model selection and
`scalar_plus_accessories` for both validation and evaluation. Do not select a
checkpoint using the held-out test split.

## Run

Set a new, empty `results_dir` in a copy of the spec, then run:

```bash
clip train \
  -e /path/to/experiment_siglip2_pas_v3_1_dual_tower_lora_pa.yaml
```

The recorded run used one node with 8 A100 GPUs, fp16, batch size 128 per GPU,
20 epochs, seed 1234, and selected the maximum `val/pas/overall_mAP` checkpoint.
Evaluate that checkpoint by setting `evaluate.checkpoint` and a new
`evaluate.results_dir`, then run:

```bash
clip evaluate -e /path/to/resolved_pa_evaluate.yaml
```

## Recorded reference

The historical fixed run selected `model_best_002.pth` and reported:

| Split | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Validation | 85.8242% | 83.9294% | 98.1736% | 99.3848% |
| Test | 83.7930% | 83.0727% | 97.7381% | 99.0349% |

Treat these values as a historical reference, not a unit-test tolerance. A
fair comparison must preserve the dataset version, train/validation/test
lists, metadata vocabularies, evaluation ground-truth mode, and checkpoint
selection rule.
