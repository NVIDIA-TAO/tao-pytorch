# Sparse4D co-training compatibility notes

The optional LTT, pseudo-label, SV-auxiliary, and parameter-touch branches are
disabled by default. This does **not** mean existing 3D training is numerically
unchanged:

- Focal classification loss now includes background queries (`target == C`).
  Previously they were excluded. This changes the loss scale and gradients even
  with co-training disabled; existing recipes may need retuning. Evaluation of
  an unchanged checkpoint alone cannot measure this training-objective change.
- Dense-depth targets retain the invalid `-1` sentinel instead of clipping it
  to 0.1 m. Invalid pixels no longer create artificial near-depth supervision.
  Existing `loss_dense_depth` values and gradients therefore change.
- PKL subset sampling now fails early for `pkl_sample_size > 0` without lazy
  loading, missing camera-count overrides, and lazy indexes lacking camera
  counts. Rebuild old indexes with tao-ds
  `annotations sparse4d_prepare -e <spec.yaml> operation=lazy_index lazy_index.annotation_source=<split.txt>`
  (provide `results_dir` in the spec). Remove stale overrides to use embedded
  counts. The native `python -m nvidia_tao_pytorch.cv.sparse4d.tools.build_lazy_index`
  remains available.
- Reloaded training samplers use `train.seed + current_epoch`: repeated runs at
  the same epoch are reproducible, but scene permutations vary between epochs.
  Distributed `sync_route` requires positive `scene_switch_iters`.
- A batch must be homogeneous in 2D/3D supervision. Mixed samples now raise
  rather than silently discarding pseudo-label supervision.
- Default training raises on non-finite losses and does not clip classification
  logits. `train.scrub_nan_gradients=true` explicitly opts into clipping,
  backward-safe loss replacement, and NaN-gradient scrubbing.
- `loss_param_touch` stays in the gradient graph when enabled, but is not
  published as a training KPI. Core's metric allowlist excludes it.
- `dataset.real_block_prob` accepts exactly `-1` for automatic weighting or
  values in `[0, 1]`. Fractional negative values are invalid.

## Companion data preparation and service schema

Core's generated Sparse4D schema must ship with a runtime image containing this
implementation. The core compatibility note documents the evaluation-default
change and the release-image gate; auxiliary artifact paths remain explicit
mounted paths, not automatically bound FTMS assets.

Native and data-service lazy-index producers preserve symlinked mount paths and
interpret relative split rows against the working directory. Duplicate PKL rows
are rejected. Prefer absolute container-visible split paths and rebuild indexes
after moving datasets.

RT-DETR labels outside the configured taxonomy now require an explicit alias or
drop rule. The native producer accepts `--class-map '{"Human":"person","pallet":null}'`;
data-services uses `rtdetr_2d.class_map`. Scene names must not contain the reserved
`+` BEV-group separator. LTT geometry requires separate camera intrinsics and
rigid world-to-camera extrinsics; projection-only `cameraMatrix` is rejected.
The two LTT producers share frame ordering and duplicate-ID validation.

## Temporal state and deployment

Partial group/scene boundaries reset only the affected GT mappings when temporal
alignment is enabled. Full temporal/DN caches reset when all slots cross a
boundary. Similarly, `reset_on_time_gap` clears full caches only when every slot
is invalid; otherwise the existing per-sample mask applies.

Temporal cache resets preserve the monotonically allocated tracking-ID counter,
including across cache misses/scenes, preventing unrelated tracks from reusing an
ID in a combined NVSchema stream. They do not preserve associations across the
boundary. `InstanceBank.reset()` remains the explicit start-of-run reset and
sets the counter to zero. Consumers must treat IDs as opaque identities rather
than expecting each scene to start at zero.
NVSchema writes IDs as strings without renumbering. HOTA preprocessing relabels
IDs contiguously within each sequence, so offsetting IDs alone does not change
the metric. Its current lookup allocation scales with the maximum numeric ID;
call the explicit full reset between independent inference runs.

Evaluation, inference, export, and quantization do not construct the training
criterion, so a saved LTT-enabled spec needs no MLP artifact for those stages.
Custom training code must instantiate
`Sparse4DPlModel(spec, build_training_losses=True)` before checkpoint restore and
DDP/optimizer setup. The training CLI does this; trainable SV auxiliary parameters
are registered before restoration, not created in the first training step.

Every ONNX export, including one from an unchanged 3D checkpoint, replaces
invalid MSDA sampling locations with zero. A compatible deployment plugin must
skip exactly-zero coordinates (strict `0 < coordinate < 1` bounds). The ONNX
graph differs from the previous export. Unit tests do not replace release-stack
`tao-deploy gen_trt_engine` and inference validation.
