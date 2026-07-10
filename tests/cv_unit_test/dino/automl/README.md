# DINO AutoML Test Template

This package is the reference pattern for testing TAO AutoML against a PyTorch
model before the model is onboarded into `tao-skills`.

The tests run real short DINO trainings on a generated dataset and exercise
AutoML algorithm contracts directly through `tao-automl`.

## Algorithms Covered

- `bayesian`: two real training experiments, search-range checks, completion
  through the AutoML controller, and best-checkpoint existence.
- `hyperband`: `automl_max_epochs=2`, `automl_reduction_factor=2`,
  `epoch_multiplier=1`. The first two experiments stop at epoch 1, the best
  result is promoted, and the promoted run resumes from epoch 1 to epoch 2.
- `asha`: asynchronous successive-halving coverage with the same promotion and
  resume-from-checkpoint checks as Hyperband.
- `pbt`: population-based training coverage with population size 2, two
  generations, exploit-from-best-checkpoint checks, and epoch-1 resume checks.

## `tao-automl` Dependency

The release image or test environment should install `tao-automl` so these tests
can run without a sibling checkout. For local multi-repo development, the harness
also supports:

- `TAO_AUTOML_SRC=/path/to/tao-automl/src`
- `../tao-automl/src` next to the `tao-pytorch` checkout

If neither an installed package nor a source checkout is available, pytest skips
the AutoML tests.

## Extending This Pattern To A New Model

Create a new package under the model test directory:

```text
tests/cv_unit_test/<model>/automl/
  __init__.py
  conftest.py
  dataset.py
  harness.py
  specs/<model>_minimal.yaml
  test_bayesian.py
  test_hyperband.py
  test_asha.py
  test_pbt.py
```

The new model should copy the DINO structure, then replace only the
model-specific pieces.

## Files To Create

`specs/<model>_minimal.yaml`

- Use nested YAML config for the model's minimal training spec.
- Keep runtime low: tiny input size, batch size 1, workers 0, one train batch,
  no validation batches, no pretrained weights unless required, and minimal
  layer/query/head settings that are still valid for the model.
- Define `automl.hyperparameters` with the parameter names passed to
  `tao-automl`, for example `train.optim.lr`.
- Define `automl.custom_param_ranges` with tight ranges for low-risk tests.

`dataset.py`

- Generate a deterministic tiny dataset under pytest's `tmp_path`.
- Do not commit generated images, annotations, checkpoints, or downloaded data.
- Match the model's real dataloader format. DINO uses generated COCO images and
  annotations.

`harness.py`

- Import the model's real `ExperimentConfig`, data module, and Lightning module.
- Implement a model harness with:
  - `network`
  - `metric`
  - `base_spec(results_dir, num_epochs)`
  - `train(rec, target_epoch, resume_checkpoint=None)`
  - `spec_dict()`
- In `train`, launch real training and record the epochs actually entered.
- Assign the checkpoint path as the AutoML job id with `rec.assign_job_id(...)`.
- Report the metric and success status back to the recommendation.
- Preserve deterministic ranking for algorithm assertions if needed, but derive
  the objective from an actual training metric.

`conftest.py`

- Create fixtures for the generated dataset and model harness.
- Skip when no GPU is available unless `TAO_AUTOML_ALLOW_CPU=1` is set.

`test_bayesian.py`

- Request exactly two recommendations.
- Run two real trainings.
- Verify parameter ranges, two records, controller completion, and best
  checkpoint existence.

`test_hyperband.py`

- Use the minimal `2,2,1` settings.
- Verify the first rung launches two experiments.
- Verify both first-rung trainings start at epoch 0 and stop after epoch 1.
- Verify the best result is promoted with `resume_from_job_id`.
- Resume from that checkpoint and assert the resumed run starts at epoch 1.
- Verify completion and best-checkpoint existence.

`test_asha.py` and `test_pbt.py`

- Add at least one non-Bayesian and non-Hyperband family.
- Keep settings minimal.
- Verify the algorithm-specific correctness contract, especially checkpoint
  reuse, promotion or exploit behavior, resume epoch, completion, and best
  checkpoint existence.

## Test Image Requirements

For CI or image-built validation, ensure the image contains:

- `tao-pytorch` and compiled model ops required by the model.
- `tao-automl`.
- Any model-specific runtime dependencies.

The DINO harness has a fallback for source-mounted validation: if the mounted
source tree does not contain the compiled deformable-attention extension, it
loads the compiled extension from the installed release-image package.

## Run Before Pushing

Use the release PyTorch image, not a local virtual environment:

```bash
docker run --rm --gpus all --ipc=host --entrypoint /bin/bash \
  -v /path/to/tao-pytorch:/workspace/tao-pytorch \
  -v /path/to/tao-automl:/workspace/tao-automl \
  -w /workspace/tao-pytorch \
  nvcr.io/nvidia/tao/tao-toolkit:7.0.0-pyt \
  -lc 'pytest -q -s tests/cv_unit_test/<model>/automl'
```

Also run:

```bash
git diff --check -- tests/cv_unit_test/<model>/automl pytest.ini
```

## Commit Checklist

Commit only source and test files:

- `tests/cv_unit_test/<model>/automl/**`
- `pytest.ini` marker additions, if the model marker is new
- image or test dependency files only if `tao-automl` is intentionally added to
  the built test image

Do not commit:

- `__pycache__`
- generated datasets
- checkpoints
- temp result directories
- unrelated dirty submodules or user changes

Example:

```bash
git switch -c rarunachalam/<model>-automl-tests
git add tests/cv_unit_test/<model>/automl pytest.ini
git commit -m "test: add <model> AutoML integration coverage"
git push -u origin rarunachalam/<model>-automl-tests
```
