# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NVDINOv2 Trainer Unit Tests
"""
import os
import torch
import pytest
import tempfile
import numpy as np
from PIL import Image

from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.nvdinov2.default_config import ExperimentConfig
from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel
from nvidia_tao_pytorch.ssl.nvdinov2.dataloader.pl_dinov2_data_module import DinoV2DataModule

BATCH_SIZE = 2
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
SAMPLES = 100

@pytest.fixture()
def _test_dir_obj():
    tmp_obj = tempfile.TemporaryDirectory()
    check_and_create(tmp_obj.name)
    for sample in range(SAMPLES):
        test_data = np.random.rand(IMAGE_HEIGHT, IMAGE_WIDTH, 3) * 255
        test_data = test_data.astype(np.uint8)
        im = Image.fromarray(test_data)
        save_name = f'test_{sample}.jpg'
        im.save(os.path.join(tmp_obj.name, save_name))

    yield tmp_obj

@pytest.fixture
def _test_exp_spec(_test_dir_obj):
    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config.dataset.train_dataset.images_dir = _test_dir_obj.name
    experiment_config.dataset.batch_size = BATCH_SIZE
    experiment_config.results_dir = _test_dir_obj.name
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1
    experiment_config.train.num_epochs = 3
    yield experiment_config

@pytest.mark.skipif(
    os.getenv("CI_PROJECT_DIR", None) is not None,
    reason='Skipping running on CI.'
)
@pytest.mark.train
@pytest.mark.ssl_unit
def test_trainer_fit(_test_dir_obj, _test_exp_spec):
    acc_flag = 'auto'
    precision = '16-mixed'

    dm = DinoV2DataModule(_test_exp_spec)
    model = DinoV2PlModel(_test_exp_spec)
    print(_test_exp_spec.train.num_gpus)
    trainer = Trainer(devices=_test_exp_spec.train.num_gpus,
                      num_nodes=_test_exp_spec.train.num_nodes,
                      max_epochs=_test_exp_spec.train.num_epochs,
                      check_val_every_n_epoch=_test_exp_spec.train.validation_interval,
                      default_root_dir=_test_exp_spec.results_dir,
                      accelerator='gpu',
                      strategy=acc_flag,
                      precision=precision,
                      )

    trainer.fit(model, dm, ckpt_path=None)

    _test_dir_obj.cleanup()


@pytest.mark.ssl_unit
def test_collate_predictions_multi_gpu_flattens_and_pairs():
    """Multi-GPU predict collation flattens [world, N, D] and pairs each row with its path.

    Regression for bug 6469109: the old code never reshaped the all_gather'd [world, N, D]
    feature tensor (so it emitted one row per RANK) and never actually gathered the string
    paths, producing mismatched column lengths that crashed/hung rank 0 with no CSV written.
    """
    feat = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)  # [world=2, N=3, D=4]
    paths = [["a.jpg", "b.jpg", "c.jpg"], ["d.jpg", "e.jpg", "f.jpg"]]     # per-rank lists
    rows = DinoV2PlModel._collate_predictions(feat, paths, distributed=True)

    assert [r["input_path"] for r in rows] == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"]
    flat = feat.reshape(-1, 4)
    assert [r["features"] for r in rows] == [str(flat[i].tolist()) for i in range(6)]


@pytest.mark.ssl_unit
def test_collate_predictions_single_gpu():
    """Single-device collation yields one row per sample (no world axis)."""
    feat = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    paths = ["a.jpg", "b.jpg", "c.jpg"]
    rows = DinoV2PlModel._collate_predictions(feat, paths, distributed=False)
    assert [r["input_path"] for r in rows] == paths
    assert [r["features"] for r in rows] == [str(feat[i].tolist()) for i in range(3)]


@pytest.mark.ssl_unit
def test_collate_predictions_dedupes_sampler_padding():
    """Duplicate paths from the predict DistributedSampler's padding collapse to one row each."""
    feat = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
    paths = [["a.jpg", "b.jpg"], ["c.jpg", "a.jpg"]]  # rank 1 padded by repeating 'a.jpg'
    rows = DinoV2PlModel._collate_predictions(feat, paths, distributed=True)
    assert [r["input_path"] for r in rows] == ["a.jpg", "b.jpg", "c.jpg"]  # one row per image


def _predict_gather_worker(rank, world_size, init_file, ret_dict):
    """gloo worker: run the real gather + collate that fixes the multi-GPU predict hang."""
    import torch.distributed as dist
    from nvidia_tao_pytorch.ssl.nvdinov2.model.pl_model import DinoV2PlModel
    dist.init_process_group("gloo", rank=rank, world_size=world_size, init_method=f"file://{init_file}")
    try:
        n, d = 3, 4
        feat = torch.arange(rank * n * d, (rank + 1) * n * d, dtype=torch.float32).reshape(n, d)
        # rank 1's last entry repeats rank 0's first path -> mimics the DistributedSampler padding.
        paths = ["a.jpg", "b.jpg", "c.jpg"] if rank == 0 else ["d.jpg", "e.jpg", "a.jpg"]
        gathered_feat, paths_per_rank = DinoV2PlModel._all_gather_predictions(feat, paths, world_size)
        if rank == 0:
            rows = DinoV2PlModel._collate_predictions(gathered_feat, paths_per_rank, distributed=True)
            ret_dict["paths"] = [r["input_path"] for r in rows]
        dist.barrier()  # both ranks reach here iff no collective desynced
        if rank == 0:
            ret_dict["both_returned"] = True
    finally:
        dist.destroy_process_group()


@pytest.mark.ssl_unit
def test_predict_gather_two_process_gloo(tmp_path):
    """2-process gloo run of the actual on_predict_epoch_end collective + collate.

    Regression for bug 6469109: asserts (a) both ranks return without hanging and (b) rank 0
    assembles one row per unique image across ranks (sampler-padding duplicate dropped). This is
    the only test that would catch a re-desync of the multi-GPU predict path.
    """
    import time
    import torch.multiprocessing as mp
    mgr = mp.Manager()
    ret = mgr.dict()
    ctx = mp.spawn(
        _predict_gather_worker,
        args=(2, str(tmp_path / "pg_init"), ret),
        nprocs=2, join=False,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if ctx.join(timeout=2):  # returns True once all ranks joined; raises if one errored
            break
    else:
        for p in ctx.processes:
            p.terminate()
        pytest.fail("multi-GPU predict gather hung -- ranks desynced (bug 6469109 regression)")

    assert ret.get("both_returned") is True
    assert sorted(ret["paths"]) == ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"]  # one row per image


@pytest.mark.ssl_unit
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nvdinov2_nonfinite_train_loss_raises(bad_value):
    """A non-finite epoch train loss must raise so @monitor_status records FAILURE.

    Regression for the silent-PASS bug (6460915): train_loss is the sole in-loop AutoML KPI,
    written to status.json but previously never validated, so a NaN loss was reported as PASS.
    The guard lives on the shared base, so every SSL model (nvdinov2, dinov3, ...) inherits it.
    """
    from types import SimpleNamespace
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(bad_value)})
    )
    with pytest.raises(ValueError, match="non-finite"):
        DinoV2PlModel._assert_train_loss_finite(fake)


@pytest.mark.ssl_unit
def test_nvdinov2_finite_train_loss_passes():
    """A finite epoch train loss must not raise."""
    from types import SimpleNamespace
    fake = SimpleNamespace(
        trainer=SimpleNamespace(logged_metrics={"train_loss_epoch": torch.tensor(10.5)})
    )
    DinoV2PlModel._assert_train_loss_finite(fake)  # should not raise


@pytest.mark.ssl_unit
def test_nvdinov2_unsupported_backbone_type_raises():
    """The shared backbone-name validator raises a clear ValueError on an unsupported arch.

    Regression for bug 6460904: lives on the base so every SSL family inherits it. Uses the
    subclass's own param_map (its keys are the supported archs) as the single source of truth.
    """
    from nvidia_tao_pytorch.config.nvdinov2.default_config import map_params
    good = sorted(map_params["depth"].keys())[0]
    with pytest.raises(ValueError, match="Unsupported model.backbone"):
        DinoV2PlModel._validate_backbone_types(
            {"teacher_type": good, "student_type": "not_a_real_arch"}, map_params
        )
    # A fully-supported pair must not raise.
    DinoV2PlModel._validate_backbone_types(
        {"teacher_type": good, "student_type": good}, map_params
    )
