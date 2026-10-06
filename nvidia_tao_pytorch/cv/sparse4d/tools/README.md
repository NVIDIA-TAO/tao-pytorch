# Sparse4D co-training artifact tools

For the complete real-world adaptation workflow, including TAO Data Services
preparation, released LTT checkpoint reuse or optional fitting, training
configuration, and evaluation, see
[2D-to-3D geometric distillation](../../../../docs/sparse4d_geometric_distillation.md).

For the released seven-class warehouse taxonomy, follow
[Get the LTT checkpoint](../../../../docs/sparse4d_geometric_distillation.md#get-the-ltt-checkpoint)
to download `_loose_to_tight_mlp.pth` from NGC `sparse4d_rn50:trainable_v3.0`.
Set `model.head.loose_to_tight.mlp_ckpt` to the downloaded path and skip the
geometry extraction and fitting commands in step 2 below. Keep the checkpoint's
ordered class taxonomy aligned with `dataset.classes`; use custom fitting for
a different taxonomy or when adapting the correction model to your data.

Calibration-free single-view (SV2D) training is currently unsupported. The
`build_sv2d_dataset` tool and SV2D configuration fields remain in the source
tree, but do not provide a supported training workflow. Use calibrated
multi-camera inputs for geometric distillation.

Run these portable producers from the `tao-pytorch` checkout root. They import
TAO's Loose-to-Tight geometry directly and do not require MMCV or MMDetection.

```bash
# 0. Index a .txt split (or PKL directory) before dataset.lazy_load is enabled.
python -m nvidia_tao_pytorch.cv.sparse4d.tools.build_lazy_index \
  /data/ov_train_split.txt --workers 16

# 1. Visible-2D-GT sidecars used while training on 3D-labelled scenes.
python -m nvidia_tao_pytorch.cv.sparse4d.tools.ltt_build_2dgt \
  --data-root /data/mtmc --train-split /data/ov_train_split.txt \
  --out-dir /data/ltt_2dgt

# 2. Optional custom LTT fitting; skip when using the released NGC checkpoint.
python -m nvidia_tao_pytorch.cv.sparse4d.tools.ltt_extract \
  --data-root /data/mtmc --train-split /data/ov_train_split.txt \
  --frame-stride 10 --out /data/ltt_training.npz
python -m nvidia_tao_pytorch.cv.sparse4d.tools.ltt_train \
  --data /data/ltt_training.npz --out /data/loose_to_tight_mlp.pth \
  --epochs 60 --device cuda

# 3. Real-scene RT-DETR KITTI archives -> one safe pseudo-label cache.
python -m nvidia_tao_pytorch.cv.sparse4d.tools.ltt_rtdetr_pseudo_labels \
  --rtdetr-dir /data/SceneA/rt-detr --out /data/rtdetr/SceneA__rtdetr2d.npz \
  --cam-map gopro1=GoPro1,gopro3=GoPro3
```

Use the same `--class-config` with the LTT and RT-DETR commands when
training a taxonomy other than the built-in warehouse-v4 order. The generated
cache records this exact order and runtime loading rejects a mismatch with
`dataset.classes`. `ltt_extract` reads NVSchema
`calibration.json` by default; select `--calib-mode aic24` for
`calibration_bevformer.json`.
Include each PKL once in the mixed 3D/real-scene training split. Use
`dataset.real_block_prob` to control aggregate 2D-vs-3D frequency.

`ltt_rtdetr_pseudo_labels` preserves the reviewed source script name; it is a thin
CLI alias for the TAO-native `rtdetr_pseudo_labels` implementation.

## Output-to-spec mapping

| Artifact | Numeric keys | Experiment-spec field |
|---|---|---|
| `<split>_lazy_index.pkl` | `frame_index`, `metadata`, `mtimes`, `pkl_cam_counts` | Required by `dataset.lazy_load: true`; discovered beside `dataset.train_dataset.ann_file` |
| `_pkl_cam_counts.pkl` | PKL-path to camera-count mapping | Optional legacy/override input via `dataset.pkl_cam_counts_path` |
| `<scene>__ltt2dgt.npz` | `frame_id`, `instance_id`, `class_id`, `cam`, `box2`, `box3`, `occ` | `dataset.ltt_2dgt_sidecar_dir` (the containing directory) |
| LTT training NPZ | `packed` (`N x 18`), `class_id`, source-frame `group_id` | Offline input to `ltt_train`; not read by training |
| `_loose_to_tight_mlp.pth` (NGC) or `loose_to_tight_mlp.pth` (custom fit) | model state + architecture metadata | `model.head.loose_to_tight.mlp_ckpt` |
| `<scene>__rtdetr2d.npz` | `frame_id`, `cam`, `class_id`, `box`, `score`; optional legacy-compatible validity arrays `valid_frame_id`, `valid_cam` | `dataset.rtdetr_2d_cache_dir`, or `dataset.rtdetr_2d_cache_path` for one cache |

All NPZ metadata is JSON encoded into a one-dimensional `uint8` `_meta` array;
runtime readers use `numpy.load(..., allow_pickle=False)`. The lazy-index
command reads trusted TAO annotation PKLs. It embeds camera counts and also
emits the sibling `_pkl_cam_counts.pkl` by default; with a new index, `pkl_sample_size` needs no `pkl_cam_counts_path`. Use
`--camera-counts-out` only when an explicit shared sidecar path is desired.

Enable `model.head.loose_to_tight.enable: true` and set its checkpoint path for
2D-GT correction. Enable `pseudo_enable: true` for RT-DETR geometric pseudo
losses on calibrated real scenes. `model.head.loose_to_tight.num_classes`
defaults to `0`, which derives the class count from `dataset.classes`.
The LTT checkpoint must record that same ordered `class_names` taxonomy.

For multi-GPU mixed 3D/2D co-training, set all of the following:

```yaml
model:
  cotrain_param_touch: false  # recommended: use TAO's find-unused DDP strategy
dataset:
  sync_route: true
  real_scene_keywords: ["SceneA"]
  scene_switch_iters: 100  # positive fixed cadence
```

Leave `cotrain_param_touch` at its default `false` to use TAO's
find-unused-parameters DDP strategy. Enable it only when standard DDP reducer
behavior is specifically required: it keeps route-specific parameters in the
autograd graph, but the explicit zero gradients still participate in optimizer
steps. With AdamW, otherwise inactive parameters can be weight-decayed and their
optimizer state can advance. TAO 7.2 timm's non-reentrant activation checkpointing
remains enabled in either mode. `sync_route` keeps every rank on the same
supervision branch. Choose scene keywords that match the actual calibrated real
scene names.

## Deliberately not ported

The reviewed branch's visualization/probe scripts and MMCV-only smoke hooks are
diagnostics, not artifact producers, and remain in the legacy repository. Its
`IterEMAHook` maps conceptually to TAO's
`nvidia_tao_pytorch.core.callbacks.ema.EMA` plus `EMAModelCheckpoint`
(`momentum=2e-4` -> `decay=0.9998`, interval -> `every_n_steps`, warm-up ->
`warmup_steps`; there is no `start_iter` equivalent). It is not force-enabled here:
Sparse4D currently uses normal checkpoint callbacks, while EMA resume expects the
matching `-EMA.pth` sibling.
