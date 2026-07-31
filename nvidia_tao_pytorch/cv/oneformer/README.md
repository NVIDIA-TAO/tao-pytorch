# OneFormer runtime contracts

## Transfer checkpoints

`train.pretrained_model` accepts a resolved full OneFormer checkpoint. TAO
Lightning (`state_dict` with `model.` keys), DDP/compiled wrappers, public
`state_dict`, and public `model` containers are normalized before shape-aware
loading. Incompatible transfer heads are reported and skipped; a checkpoint
with no compatible tensors fails closed. `train.pretrained_backbone` remains
the separate backbone-only path.

## Task-correct evaluation

Evaluation task selection is explicit and independent of visualization:

```yaml
evaluate:
  task: panoptic  # semantic | panoptic
```

- `semantic` reports `mIoU` and pixel accuracy.
- `panoptic` reports COCO-style `PQ`, `SQ`, `RQ`, and thing/stuff breakdowns.

`inference.mode` controls prediction visualization only; it does not select an
evaluation metric. Existing recipes explicitly retain `evaluate.task:
semantic`. A COCO panoptic campaign must set `evaluate.task: panoptic` and
consume `PQ`, never relabel semantic mIoU as PQ.

Panoptic data loading requires
[cocodataset/panopticapi](https://github.com/cocodataset/panopticapi). The
runtime exits with an actionable dependency error when it is unavailable.

For distributed validation and test, OneFormer sums semantic-confusion or PQ
sufficient statistics across all workers before computing nonlinear metrics.
All ranks log the same global values, and only global rank zero writes KPI
status records.
