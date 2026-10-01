# Sparse4D co-training artifact tools

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

# 2. Geometry cache, then the frozen Loose-to-Tight MLP checkpoint.
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

# 4. Calibration-free COCO -> GT-less TAO info PKL + pseudo-label NPZ.
python -m nvidia_tao_pytorch.cv.sparse4d.tools.build_sv2d_dataset \
  --manifest /data/datasets.json --dataset all \
  --cache-dir /data/rtdetr --pkl-dir /data/sv2d \
  --class-config /data/classes.py
```

Use the same `--class-config` with the LTT, RT-DETR, and SV2D commands when
training a taxonomy other than the built-in warehouse-v4 order. The generated
cache records this exact order and runtime loading rejects a mismatch with
`dataset.classes`. `ltt_extract` reads NVSchema
`calibration.json` by default; select `--calib-mode aic24` for
`calibration_bevformer.json`. `build_sv2d_dataset` requires a user-provided
manifest containing a list (or `{"datasets": [...]}`) of portable `file`/`h5`
COCO sources; input and output locations have no repository-specific defaults.
`--suffix _smoke` changes the SV scene name and **both** output filenames, so a
smoke build cannot overwrite the full NPZ. `--dataset all` also writes a
one-PKL-per-line `<pkl-dir>/sv2d_train_split<suffix>.txt`; without a suffix this
remains `sv2d_train_split.txt`. Use `--split-out` to choose another path.
Append/merge those lines into the main 3D/real-scene training split for mixed
co-training. A sibling `.sv2d_weights.json` preserves the source manifest's
relative weights as advisory metadata. The current real-route sampler does not
consume per-SV weights: repeating PKLs would corrupt Sparse4D sequence grouping,
so use `dataset.real_block_prob` to control aggregate 2D-vs-3D frequency.

`ltt_rtdetr_pseudo_labels` preserves the reviewed source script name; it is a thin
CLI alias for the TAO-native `rtdetr_pseudo_labels` implementation.

## Output-to-spec mapping

| Artifact | Numeric keys | Experiment-spec field |
|---|---|---|
| `<split>_lazy_index.pkl` | `frame_index`, `metadata`, `mtimes`, `pkl_cam_counts` | Required by `dataset.lazy_load: true`; discovered beside `dataset.train_dataset.ann_file` |
| `_pkl_cam_counts.pkl` | PKL-path to camera-count mapping | Optional legacy/override input via `dataset.pkl_cam_counts_path` |
| `<scene>__ltt2dgt.npz` | `frame_id`, `instance_id`, `class_id`, `cam`, `box2`, `box3`, `occ` | `dataset.ltt_2dgt_sidecar_dir` (the containing directory) |
| LTT training NPZ | `packed` (`N x 18`), `class_id`, source-frame `group_id` | Offline input to `ltt_train`; not read by training |
| `loose_to_tight_mlp.pth` | model state + architecture metadata | `model.head.loose_to_tight.mlp_ckpt` |
| `<scene>__rtdetr2d.npz` | `frame_id`, `cam`, `class_id`, `box`, `score`; optional legacy-compatible validity arrays `valid_frame_id`, `valid_cam` | `dataset.rtdetr_2d_cache_dir`, or `dataset.rtdetr_2d_cache_path` for one cache |
| `SV2D..._infos_train.pkl` | TAO `infos` with one virtual camera and `gt_boxes=None` | Included through `dataset.train_dataset.ann_file` |
| `sv2d_train_split<suffix>.txt` | One entry per generated PKL | Merge into the split referenced by `dataset.train_dataset.ann_file` |
| `sv2d_train_split<suffix>.sv2d_weights.json` | Advisory scene-to-relative-weight map | Not consumed by the current real-route sampler |

All NPZ metadata is JSON encoded into a one-dimensional `uint8` `_meta` array;
runtime readers use `numpy.load(..., allow_pickle=False)`. The SV info PKL is the
existing TAO annotation format and should only be loaded from trusted storage.
The lazy-index command likewise reads trusted TAO annotation PKLs. It embeds
camera counts and also emits the sibling `_pkl_cam_counts.pkl` by default; with a
new index, `pkl_sample_size` needs no `pkl_cam_counts_path`. Use
`--camera-counts-out` only when an explicit shared sidecar path is desired.

Enable `model.head.loose_to_tight.enable: true` and set its checkpoint path for
2D-GT correction. Enable `pseudo_enable: true` for RT-DETR/SV geometric pseudo
losses. SV images also need `dataset.resize_to_canonical_2d: true`; the independent
classifier route is controlled by `model.sv_aux_head.enable` and
`model.sv_scene_keywords`. Both `num_classes` fields default to `0`, which
derives the exact class count and order from `dataset.classes`. An LTT checkpoint
must record that same ordered `class_names` taxonomy.

For multi-GPU mixed 3D/2D co-training, set all of the following:

```yaml
model:
  cotrain_param_touch: false  # recommended: use TAO's find-unused DDP strategy
  sv_scene_keywords: ["SV2D"]
dataset:
  sync_route: true
  real_scene_keywords: ["SV2D", "Zanker"]
  scene_switch_iters: 100  # positive fixed cadence
```

Leave `cotrain_param_touch` at its default `false` to use TAO's
find-unused-parameters DDP strategy. Enable it only when standard DDP reducer
behavior is specifically required: it keeps route-specific parameters in the
autograd graph, but the explicit zero gradients still participate in optimizer
steps. With AdamW, otherwise inactive parameters can be weight-decayed and their
optimizer state can advance. TAO 7.2 timm's non-reentrant activation checkpointing
remains enabled in either mode. `sync_route` keeps every rank on the same
supervision branch. Choose scene keywords that match the actual generated/real
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
