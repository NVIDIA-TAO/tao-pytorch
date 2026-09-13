# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from platform import machine

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.sparse4d.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.sparse4d.model.sparse4d_pl_model import Sparse4DPlModel
from nvidia_tao_pytorch.cv.sparse4d.utils import onnx_export
from nvidia_tao_pytorch.cv.sparse4d.utils.onnx_export import Sparse4DExporter


pytestmark = pytest.mark.skipif(
    ("aarch64" in machine().lower()) or ("arm" in machine().lower()),
    reason="Sparse4D tests take very long (~12 hours) on ARM architecture.",
)


@pytest.fixture
def _test_experiment_spec(tmp_path):
    """Creates a minimal ExperimentConfig for testing export."""
    # The exporter consumes 900 learned 11-D box anchors and caches the first
    # 600 temporal instances. A deterministic local anchor file exercises the
    # same loading and ONNX graph paths without relying on TAO CI datasets.
    anchors = np.zeros((900, 11), dtype=np.float32)
    anchors[:, 7] = 1.0  # Valid identity yaw: [sin(yaw), cos(yaw)] = [0, 1].
    anchor_path = tmp_path / "anchors.npy"
    np.save(anchor_path, anchors)

    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.head.instance_bank.anchor = str(anchor_path)
    cfg.model.head.deformable_model.use_camera_embed = True
    cfg.dataset.classes = [
        "person",
        "gr1_t2",
        "agility_digit",
        "nova_carter",
        "transporter",
        "forklift",
        "pallet_truck",
    ]
    OmegaConf.resolve(cfg)
    return cfg


@pytest.fixture
def _restore_exporter_methods():
    """Restore class methods that the legacy exporter patches globally."""
    methods = (
        (onnx_export.SparseBox3DKeyPointsGenerator, "forward"),
        (onnx_export.SparseBox3DRefinementModule, "forward"),
        (onnx_export.DeformableAggregationFunction, "symbolic"),
        (onnx_export.DeformableFeatureAggregation, "project_points"),
    )
    missing = object()
    # Read the raw class dictionaries so staticmethod descriptors and inherited
    # attributes are restored exactly, rather than as newly bound methods.
    originals = [owner.__dict__.get(name, missing) for owner, name in methods]
    try:
        yield
    finally:
        for (owner, name), original in zip(methods, originals):
            if original is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, original)


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.export
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Sparse4D ONNX export requires CUDA",
)
def test_sparse4d_onnx_export(
    _test_experiment_spec,
    _restore_exporter_methods,
    tmp_path,
):
    """Unit test for ONNX export on Sparse4D model."""
    cfg = _test_experiment_spec
    output_file = tmp_path / "sparse4d.onnx"
    on_cpu = cfg.export.on_cpu

    # Instantiate the model first
    model = Sparse4DPlModel(cfg)

    model.eval()
    if not on_cpu:
        model.cuda()

    sparse4d_exporter = Sparse4DExporter(model)
    sparse4d_exporter.export_model(cfg, model, str(output_file))
    sparse4d_exporter.check_onnx(str(output_file))
    assert output_file.is_file(), "ONNX file was not generated properly!"


@pytest.mark.cv_unit
@pytest.mark.sparse4d
@pytest.mark.export
def test_sparse4d_refinement_module():
    """Simple test for the refinement module forward pass."""
    from nvidia_tao_pytorch.cv.sparse4d.model.detection3d.detection3d_blocks import (
        SparseBox3DRefinementModule,
    )
    from nvidia_tao_pytorch.cv.sparse4d.utils.onnx_export import Sparse4DExporter

    # Create a simple refinement module
    embed_dims = 256
    output_dim = 11
    num_cls = 7

    module = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=output_dim,
        num_cls=num_cls,
        refine_yaw=True,
        normalize_yaw=True,
        with_quality_estimation=True,
    )
    module.eval()

    # Create test inputs
    batch_size = 2
    num_anchors = 10

    instance_feature = torch.randn(batch_size, num_anchors, embed_dims)
    anchor = torch.randn(
        batch_size, num_anchors, 11
    )  # [x,y,z,w,l,h,sin_yaw,cos_yaw,vx,vy,vz]
    anchor_embed = torch.randn(batch_size, num_anchors, embed_dims)
    time_interval = torch.tensor(0.1)

    # Test original forward vs static forward method for export
    with torch.no_grad():
        # Original forward
        original_output, original_cls, original_quality = module(
            instance_feature, anchor, anchor_embed, time_interval, return_cls=True
        )

        # Static forward method for ONNX export
        (
            static_output,
            static_cls,
            static_quality,
        ) = Sparse4DExporter.sparse_box_3d_refinement_module_forward(
            module,
            instance_feature,
            anchor,
            anchor_embed,
            time_interval,
            return_cls=True,
        )

    # Check if outputs match across original forward methods and static forward method for export
    torch.testing.assert_close(original_output, static_output, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(original_cls, static_cls, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(original_quality, static_quality, rtol=1e-4, atol=1e-5)
