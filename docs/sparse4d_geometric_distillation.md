<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sparse4D: 2D-to-3D geometric distillation for real-world datasets

Use **2D-to-3D geometric distillation** to adapt Sparse4D to calibrated
multi-camera real-world imagery with 2D annotations or detector pseudo-labels
while retaining 3D-labeled data in the training mix. The term **2D-to-3D**
describes the direction of supervision: 2D targets supervise the 3D model.
During training, **3D-to-2D projection** maps Sparse4D's predicted 3D boxes
into camera images, where they can be compared with the 2D targets. Sparse4D
remains a multi-camera 3D detector and tracker; the deployed model outputs
3D predictions.

This guide requires a TAO PyTorch build containing
`model.head.loose_to_tight` and the co-training dataset
fields in the [experiment spec](../nvidia_tao_pytorch/cv/sparse4d/experiment_specs/experiment_spec.yaml).
The data-service commands require a companion TAO Data Services build with
`annotations sparse4d_prepare` and the
[annotation-free scene converter](https://github.com/NVIDIA-TAO/tao-data-services/pull/55). Check that the specs shipped with
your build expose `aicity.load_annotations` and `aicity.fps` before using these options. The optional LTT training losses are disabled by default.
See the [compatibility notes](../nvidia_tao_pytorch/cv/sparse4d/COMPATIBILITY.md) for
training and deployment behavior that changed with this implementation.

## How the geometric supervision works

A 3D cuboid's eight projected corners define a loose 2D bounding rectangle.
The actual object occupies a tighter image-space box. A **loose-to-tight
(LTT) MLP** learns this correction from labeled data using the class, box
extent, camera-relative orientation, projected box, image size, and distance.
The offline fitting target is the tight **amodal** 2D box (the object's full
extent), not just its visible fragment.

During Sparse4D training, the MLP is frozen and its four predicted edge scales
are detached. Applying those scales to the projected box preserves gradients
through the box projection into Sparse4D. On calibrated real scenes without
3D ground truth, corrected projections are matched to per-camera 2D detections
using Hungarian matching, followed by cross-camera consistency filtering.
Box losses combine GIoU, normalized L1, and visible-box containment;
classification losses supervise the matched classes and background queries.
The 2D teacher, such as RT-DETR, supplies offline detections.

| Data route | Inputs | Training signal |
| --- | --- | --- |
| 3D-labeled scenes | Synchronized images, intrinsics/extrinsics, 3D boxes and IDs; optional visible-2D sidecars | Standard 3D losses plus optional LTT visible-2D loss |
| Calibrated real scenes without 3D labels | Synchronized images, intrinsics/extrinsics, BEV groups, 2D teacher detections | LTT projected-box and classification pseudo-losses |

Preserve a 3D-supervised route and validate 3D accuracy independently.

## Prepare compatible inputs

Start with a complete Sparse4D fine-tuning spec and pretrained checkpoint.
Keep the backbone, input resolution, anchors, and class order compatible with
that checkpoint. Use container-visible paths for all artifacts.

* **Taxonomy:** use the same ordered `class_names` in preparation and
  `dataset.classes` in training. The LTT checkpoint and NPZ caches retain this
  order; readers reject mismatches. The example below uses the seven warehouse
  classes. Replace the taxonomy and aliases together for a custom model.
* **Geometry:** calibrated routes need separate camera intrinsics and rigid
  world-to-camera extrinsics, with consistent units, axes, box centers, and
  image coordinates. A projection-only `cameraMatrix` is insufficient for LTT
  extraction. Inspect projected boxes before producing a full cache.
* **Frame identity:** align camera names and source frame IDs across images,
  calibration, info PKLs, and 2D caches. Scene names must not contain the
  reserved `+` BEV-group separator. Check joins after conversion or sampling.
* **3D label availability:** real info PKLs must retain image paths, camera
  geometry, sequence metadata, and timestamps. Use `gt_boxes=None` when 3D
  annotations are unavailable. An empty annotated frame is a different case.
  Prepare these info PKLs with the conversion command below; the RT-DETR
  operation produces the corresponding detection caches.
* **Splits:** reserve separate scenes or sequences for validation before fitting
  the LTT adapter or selecting teacher thresholds. Do not mix held-out real
  scenes into pseudo-label training.

## Convert calibrated real scenes without 3D annotations

Use the [unlabeled-scene spec](https://github.com/NVIDIA-TAO/tao-data-services/blob/dbf26c3a4ba80f63e3460e3dcf06f59ecf184cde/nvidia_tao_ds/annotations/experiment_specs/aicity2ovpkl_unlabeled.yaml) from a TAO Data Services build
containing the annotation-free converter. Copy it to
`/specs/aicity2ovpkl_unlabeled.yaml` and set the ordered
`aicity.class_config.CLASS_LIST` to match your model and teacher caches.

Arrange each scene as `/data/real/train/<scene>/calibration.json` plus
`<camera>/rgb/000000000.jpg`, `000000001.jpg`, and so on. NVSchema calibration
must include every camera's intrinsic matrix and world-to-camera pose, with
camera IDs matching the image directories. The `rgb_00000.jpg` naming convention
also works. Use synchronized, constant-rate camera sequences that are
contiguous from zero, and supply their actual capture rate:

```bash
annotations convert -e /specs/aicity2ovpkl_unlabeled.yaml \
  aicity.root=/data/real aicity.split=train aicity.fps=30 \
  results_dir=/data/real_infos
```

The shipped spec sets `aicity.load_annotations=false`, disables recentering,
and uses all calibrated cameras as one group per scene. It writes
`/data/real_infos/train/<scene>_infos_train.pkl` with `gt_boxes=None`, no depth
paths, and timestamps in seconds. It requires neither `ground_truth.json` nor
depth files and skips anchor initialization; reuse the pretrained model's
anchors. Labeled conversion still defaults to loading 3D ground truth.

For HDF5 image arrays, set `aicity.rgb_format=h5` and provide `<camera>.h5`
files with HWC arrays at `rgb/rgb_00000.jpg`, `rgb/rgb_00001.jpg`, and so on.
Use `dataset.use_h5_file_for_rgb: true` when training on these inputs. Keep image paths
visible at the same location inside the training environment.

The converter rejects empty, gapped, or unequal camera sequences. Synchronize
irregular or dropped-frame input before conversion, then run the 2D teacher
on those same frames. Inspect projections before training. The existing
camera-grouping options can also generate multiple BEV groups; keep the
calibration and cache joins consistent with each group.

## Prepare artifacts with TAO Data Services

The [preparation entrypoint](https://github.com/NVIDIA-TAO/tao-data-services/blob/main/nvidia_tao_ds/annotations/scripts/sparse4d_prepare.py)
and [shipped spec](https://github.com/NVIDIA-TAO/tao-data-services/blob/main/nvidia_tao_ds/annotations/experiment_specs/sparse4d_prepare.yaml)
are maintained in TAO Data Services. In that environment, save the following
configuration as `/specs/sparse4d_prepare.yaml`. Replace the example paths, scene names, and
label aliases with your dataset's values. `ltt_data` and `ltt_2dgt` consume raw
AICity/MTMC-style annotations; convert other annotation formats first. In
`aic25` mode, LTT extraction reads the scene's NVSchema `calibration.json`.

```yaml
operation: ltt_data
results_dir: /results/sparse4d_prepare
overwrite: false
class_names: [person, gr1_t2, agility_digit, nova_carter, transporter, forklift, pallet_truck]
subclass_map:
  person: [Person, Human]
  gr1_t2: [FourierGR1T2, Fourier_GR1_T2_Humanoid]
  agility_digit: [AgilityDigit, Agility_Digit_Humanoid]
  nova_carter: [NovaCarter, Nova_Carter]
  transporter: [Transporter]
  forklift: [Forklift]
  pallet_truck: [Pallet_Truck]

ltt_data:
  selection:
    scene_dirs: [/data/labeled/SceneA]
  output_path: /data/ltt_training.npz
  calibration_mode: aic25
  annotation_version: v0.1
  frame_stride: 10

ltt_2dgt:
  selection:
    scene_dirs: [/data/labeled/SceneA]
  output_dir: /data/ltt_2dgt
  annotation_version: v0.1

rtdetr_2d:
  input_dir: /data/RealWarehouse/rt-detr
  output_path: /data/rtdetr/RealWarehouse__rtdetr2d.npz
  scene_name: RealWarehouse
  camera_map: {camera1: Camera1, camera2: Camera2}
  class_map: {pallet: null}
  confidence_threshold: 0.4

lazy_index:
  annotation_source: /data/mixed_train_split.txt
```

Only the block selected by `operation` is executed. Choose output paths that
do not overwrite existing artifacts, or explicitly opt into a rebuild.
The teacher threshold is an example and must be checked on your real data.

First extract LTT geometry and, if using the supervised LTT loss, visible-2D
sidecars for the 3D-labeled scenes:

```bash
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=ltt_data
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=ltt_2dgt
```

For each calibrated real scene, run RT-DETR offline and arrange its KITTI label
archives as `/data/RealWarehouse/rt-detr/<camera>/labels.tar.gz`. Map those
camera directory names to the exact names in the Sparse4D info PKL. Label rows
must contain the class, 2D box coordinates, and confidence score. Explicitly
alias or drop every teacher class outside the target taxonomy. Then convert:

```bash
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=rtdetr_2d
```

This command normalizes existing detections; it does not invoke RT-DETR. Run it
with the corresponding input, scene name, and output path for each real scene.
Do not create teacher caches for 3D-labeled scenes that should stay on the 3D
route when `dataset.rtdetr_2d_mark_real` is enabled.

Create `/data/mixed_train_split.txt` with one absolute, container-visible PKL
path per line for the existing 3D-labeled data and the prepared calibrated real
data. Include each PKL only once. Then build the index for lazy loading:

```bash
annotations sparse4d_prepare -e /specs/sparse4d_prepare.yaml operation=lazy_index
```

Rebuild the index after changing the split or moving data. Control aggregate
2D/3D sampling with `real_block_prob`.

| Artifact | Consumer |
| --- | --- |
| `ltt_training.npz` (`ltt_data/v2`) | Offline MLP fitting below |
| `<scene>__ltt2dgt.npz` (`ltt_2dgt/v1`) | `dataset.ltt_2dgt_sidecar_dir` |
| `<scene>__rtdetr2d.npz` (`ltt_rtdetr2d/v1`) | `dataset.rtdetr_2d_cache_dir`, or `rtdetr_2d_cache_path` for a single scene |
| 3D-labeled and calibrated real info PKLs | Entries in `dataset.train_dataset.ann_file` |
| Lazy index beside the split | `dataset.lazy_load: true` |

The LTT sidecar's `box2` and `box3` arrays both contain 2D boxes: `box2` is
amodal and `box3` is visible. NPZ metadata is JSON encoded in a `uint8` `_meta`
array and read with `allow_pickle=False`. Info PKLs and lazy indexes use the
existing trusted TAO annotation format.

The [native artifact tools](../nvidia_tao_pytorch/cv/sparse4d/tools/README.md)
provide an alternative when working entirely in a TAO PyTorch checkout.

## Fit the frozen LTT adapter

In the TAO PyTorch environment, train the MLP from the geometry cache:

```bash
python -m nvidia_tao_pytorch.cv.sparse4d.tools.ltt_train \
  --data /data/ltt_training.npz \
  --out /data/loose_to_tight_mlp.pth \
  --epochs 60 --device cuda
```

The cache supplies the ordered class taxonomy. Fitting separates source-frame
`group_id` values between its training and validation partitions, so camera
views of the same frame stay together. This internal split does not replace a
held-out scene evaluation. Compare per-class validation GIoU with the raw
projection baseline, and check class coverage and difficult viewpoints.

Retain the external MLP artifact for later training and resume: it is frozen
and deliberately excluded from Sparse4D model checkpoints. Evaluation,
inference, and export do not need this artifact.

## Configure and train Sparse4D

Merge this fragment into a complete spec for the selected pretrained model.
Set `train.pretrained_model_path` and retain the matching model, anchors,
image preprocessing, optimizer, validation, and evaluation settings.

```yaml
model:
  cotrain_param_touch: false
  head:
    loose_to_tight:
      enable: true
      mlp_ckpt: /data/loose_to_tight_mlp.pth
      num_classes: 0
      loss_weight: 0.1
      tight_l1_weight: 1.0
      containment_weight: 1.0
      pseudo_enable: true
      pseudo_box_weight: 0.1
      pseudo_cls_weight: 1.0
      giou_thr: 0.3
      min_cams: 1
      class_gate: true

dataset:
  classes: [person, gr1_t2, agility_digit, nova_carter, transporter, forklift, pallet_truck]
  train_dataset:
    ann_file: /data/mixed_train_split.txt
  lazy_load: true
  ltt_2dgt_sidecar_dir: /data/ltt_2dgt
  rtdetr_2d_cache_dir: /data/rtdetr
  rtdetr_2d_mark_real: true
  sync_route: true
  real_scene_keywords: [RealWarehouse]
  real_block_prob: 0.5
  scene_switch_iters: 100
```

Both `enable` and `pseudo_enable` must be true for geometric pseudo-label
supervision. `num_classes: 0` derives the taxonomy size from `dataset.classes`.
`min_cams: 1` permits supervision from one supporting view; increase it only
when camera overlap and teacher coverage support that requirement. Match
thresholds and loss weights need validation on the target data.

`real_scene_keywords` must identify all 2D-supervised scene names and exclude
3D-supervised scenes. Each batch must have homogeneous 2D/3D supervision.
For distributed mixed training, set `sync_route: true` and a positive
`scene_switch_iters` so ranks take compatible routes. `real_block_prob: 0.5`
is an example block probability; `-1` selects automatic scene-count weighting.
Leave `cotrain_param_touch: false` to use the training CLI's find-unused-
parameters DDP strategy. Enabling parameter touch can advance optimizer state
and apply weight decay to otherwise inactive parameters.

Use the regular training command inside the TAO PyTorch environment:

```bash
sparse4d train -e /specs/experiment_spec.yaml
```

There is no separate Sparse4D `distill` task. The standard training task selects
losses from the batch's supervision route and the configuration above.

## Verify supervision and evaluate adaptation

Run a short training job before a long adaptation run. Inspect source images,
corrected projections, teacher boxes, and sidecar joins. Check these signals:

| Route | Expected signal |
| --- | --- |
| 3D-labeled | Standard classification/regression losses; `loss_box_2d_*` when valid visible-2D sidecars are supplied |
| Calibrated real 2D | `loss_box_2d_pseudo_*` and `loss_cls_pseudo_*` on valid matched data |

A missing cache, unresolved frame, or incomplete camera join sets
`has_2d_pseudo=False` and skips geometric pseudo supervision for that sample.
An explicitly processed frame with no detections is different: validity arrays
preserve it and it can supervise background. Investigate join warnings and
unexpected zero box losses instead of treating them as convergence. Confirm
that teacher coverage is adequate before treating unmatched queries as
background.

Evaluate the baseline and adapted checkpoints on the same held-out real scenes
with independent 3D and tracking annotations. Report detection and tracking
metrics separately and evaluate a synthetic holdout for regressions. Neither
2D box agreement nor a successful short training run establishes a 3D accuracy
gain. Select confidence thresholds on validation data and keep test scenes
separate.

Use standard Sparse4D evaluation, inference, and export after adaptation. The
prediction stages do not construct the training criterion and need neither
the external LTT MLP nor the offline RT-DETR teacher. Deploy the exported
Sparse4D model with its matching anchors, class order, calibration,
preprocessing, and compatible TensorRT/plugin stack; see the
[deployment compatibility notes](../nvidia_tao_pytorch/cv/sparse4d/COMPATIBILITY.md#temporal-state-and-deployment).
