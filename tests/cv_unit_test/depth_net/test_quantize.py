# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test cases for DepthNet quantization functionality."""

import pytest

from nvidia_tao_pytorch.config.depth_net.dataset import (
    DepthNetDatasetConfig,
    QuantCalibrationDataset,
)
from nvidia_tao_pytorch.config.depth_net.default_config import ExperimentConfig


@pytest.mark.cv_unit
def test_dataset_config_has_quant_calibration_dataset():
    """Test that DepthNetDatasetConfig has quant_calibration_dataset field."""
    data_config = DepthNetDatasetConfig()
    assert hasattr(data_config, "quant_calibration_dataset")
    assert isinstance(data_config.quant_calibration_dataset, QuantCalibrationDataset)


@pytest.mark.cv_unit
def test_quant_calibration_dataset_has_images_dir():
    """Test that QuantCalibrationDataset has images_dir field."""
    calib_config = QuantCalibrationDataset()
    assert hasattr(calib_config, "images_dir")
    assert calib_config.images_dir == ""


@pytest.mark.cv_unit
def test_experiment_config_has_quantize():
    """Test that ExperimentConfig has quantize field."""
    exp_config = ExperimentConfig()
    assert hasattr(exp_config, "quantize")


@pytest.mark.cv_unit
def test_quantize_config_has_required_fields():
    """Test that quantize config has all required fields."""
    exp_config = ExperimentConfig()
    quantize_config = exp_config.quantize
    assert hasattr(quantize_config, "backend")
    assert hasattr(quantize_config, "mode")
    assert hasattr(quantize_config, "algorithm")
    assert hasattr(quantize_config, "model_path")
    assert hasattr(quantize_config, "results_dir")


# ---------------------------------------------------------------------------
# Stereo (FoundationStereo / FastFoundationStereo) calibration path
# ---------------------------------------------------------------------------


def _make_stereo_onnx(tmp_path, names=("left_image", "right_image"), hw=(480, 736), dynamic_batch=True):
    """Write a two-input ONNX graph (Add) with optional dynamic batch axis and return its path."""
    import onnx
    from onnx import TensorProto, helper

    batch = "batch" if dynamic_batch else 1
    inputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [batch, 3, hw[0], hw[1]])
        for name in names
    ]
    output = helper.make_tensor_value_info("disparity", TensorProto.FLOAT, [batch, 3, hw[0], hw[1]])
    node = helper.make_node("Add", list(names), ["disparity"], name="/add/Add")
    graph = helper.make_graph([node], "stereo_stub", inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "stereo_stub.onnx"
    onnx.save(model, str(path))
    return str(path)


def _fake_stereo_loader(num_batches=2, batch_size=1, hw=(480, 736)):
    """Yield dict batches shaped like StereoDataset samples."""
    import torch

    batches = []
    for idx in range(num_batches):
        batches.append({
            "image": torch.full((batch_size, 3, hw[0], hw[1]), float(idx)),
            "right_image": torch.full((batch_size, 3, hw[0], hw[1]), float(idx) + 0.5),
            "disparity": torch.zeros((batch_size, hw[0], hw[1])),
            "image_path": [f"/fake/{idx}.png"] * batch_size,
        })
    return batches


@pytest.mark.cv_unit
def test_quant_calibration_dataset_has_stereo_fields():
    """Test that QuantCalibrationDataset carries the stereo calibration fields."""
    calib_config = QuantCalibrationDataset()
    assert calib_config.data_sources is None
    assert calib_config.num_samples == 128
    assert calib_config.batch_size == 1
    assert calib_config.workers == 4
    assert calib_config.pin_memory is True
    assert list(calib_config.augmentation.input_mean) == [0.485, 0.456, 0.406]


@pytest.mark.cv_unit
def test_is_stereo_model():
    """Test the stereo/mono dispatch keyed on model.model_type."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import is_stereo_model

    exp_config = ExperimentConfig()
    for model_type, expected in [
        ("FastFoundationStereo", True),
        ("FoundationStereo", True),
        ("MetricDepthAnything", False),
        ("RelativeDepthAnything", False),
    ]:
        exp_config.model.model_type = model_type
        assert is_stereo_model(exp_config) is expected


@pytest.mark.cv_unit
def test_get_onnx_input_dims_reports_dynamic_axes(tmp_path):
    """Test that ONNX graph inputs are read with dynamic axes reported as None."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import get_onnx_input_dims

    onnx_path = _make_stereo_onnx(tmp_path)
    dims = get_onnx_input_dims(onnx_path)
    assert dims == {
        "left_image": [None, 3, 480, 736],
        "right_image": [None, 3, 480, 736],
    }


@pytest.mark.cv_unit
def test_resolve_stereo_onnx_inputs_prefers_exporter_names():
    """Test the batch-key -> ONNX-input mapping for the exporter's names."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import resolve_stereo_onnx_inputs

    onnx_inputs = {"right_image": [None, 3, 480, 736], "left_image": [None, 3, 480, 736]}
    assert resolve_stereo_onnx_inputs(onnx_inputs) == {"image": "left_image", "right_image": "right_image"}


@pytest.mark.cv_unit
def test_resolve_stereo_onnx_inputs_positional_fallback_and_errors():
    """Test positional mapping for foreign input names and errors for unsupported graphs."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import resolve_stereo_onnx_inputs

    foreign = {"img_l": [1, 3, 480, 736], "img_r": [1, 3, 480, 736]}
    assert resolve_stereo_onnx_inputs(foreign) == {"image": "img_l", "right_image": "img_r"}

    with pytest.raises(ValueError, match="non-image inputs"):
        resolve_stereo_onnx_inputs({"left_image": [1, 3, 480, 736], "right_image": [1, 3, 480, 736], "iters": []})
    with pytest.raises(ValueError, match="exactly two"):
        resolve_stereo_onnx_inputs({"image": [1, 3, 480, 736]})


@pytest.mark.cv_unit
def test_collect_stereo_calibration_data_keys_and_shapes(tmp_path):
    """Test that calibration pairs are concatenated per ONNX input with the right keys and shapes."""
    import numpy as np

    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import (
        collect_stereo_calibration_data,
        get_onnx_input_dims,
    )

    onnx_inputs = get_onnx_input_dims(_make_stereo_onnx(tmp_path))
    data = collect_stereo_calibration_data(_fake_stereo_loader(num_batches=2), onnx_inputs)

    assert set(data.keys()) == {"left_image", "right_image"}
    assert data["left_image"].shape == (2, 3, 480, 736)
    assert data["right_image"].shape == (2, 3, 480, 736)
    assert data["left_image"].dtype == np.float32
    # Left/right stay paired and in dataset order.
    assert np.allclose(data["left_image"][1], 1.0)
    assert np.allclose(data["right_image"][1], 1.5)


@pytest.mark.cv_unit
def test_collect_stereo_calibration_data_resizes_to_static_hw(tmp_path):
    """Test that full-resolution pairs are resized to the ONNX static H/W."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import (
        collect_stereo_calibration_data,
        get_onnx_input_dims,
    )

    onnx_inputs = get_onnx_input_dims(_make_stereo_onnx(tmp_path, hw=(480, 736)))
    data = collect_stereo_calibration_data(_fake_stereo_loader(num_batches=1, hw=(994, 1476)), onnx_inputs)
    assert data["left_image"].shape == (1, 3, 480, 736)
    assert data["right_image"].shape == (1, 3, 480, 736)


@pytest.mark.cv_unit
def test_collect_stereo_calibration_data_rejects_mixed_shapes_on_dynamic_hw(tmp_path):
    """Test that mixed-size pairs are rejected when the graph cannot pin them to a static H/W."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import collect_stereo_calibration_data

    onnx_inputs = {"left_image": [None, 3, None, None], "right_image": [None, 3, None, None]}
    loader = _fake_stereo_loader(num_batches=1, hw=(32, 64)) + _fake_stereo_loader(num_batches=1, hw=(48, 64))
    with pytest.raises(ValueError, match="mixed shapes"):
        collect_stereo_calibration_data(loader, onnx_inputs)


@pytest.mark.cv_unit
def test_collect_stereo_calibration_data_rejects_empty_loader(tmp_path):
    """Test that an empty calibration dataset is reported instead of silently calibrating on nothing."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import (
        collect_stereo_calibration_data,
        get_onnx_input_dims,
    )

    onnx_inputs = get_onnx_input_dims(_make_stereo_onnx(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        collect_stereo_calibration_data([], onnx_inputs)


@pytest.mark.cv_unit
def test_stereo_pair_loader_and_wrapper_roundtrip():
    """Test the PyTorch-backend adapters: channel-concat loader feeds a split-and-forward wrapper."""
    import torch
    import torch.nn as nn

    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import StereoCalibrationWrapper, StereoPairLoader

    class _Stereo(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, left, right, iters=12, flow_init=None, test_mode=False):
            self.calls.append((tuple(left.shape), tuple(right.shape), iters, test_mode))
            return left - right

    loader = StereoPairLoader(_fake_stereo_loader(num_batches=3, hw=(32, 64)))
    assert len(loader) == 3
    model = _Stereo()
    wrapped = StereoCalibrationWrapper(model, iters=8)
    for x in loader:
        assert x.shape == (1, 6, 32, 64)
        out = wrapped(x)
        assert torch.allclose(out, torch.full_like(out, -0.5))
    assert model.calls == [((1, 3, 32, 64), (1, 3, 32, 64), 8, True)] * 3


@pytest.mark.cv_unit
def test_stereo_pair_loader_resizes_to_target_hw():
    """Test that StereoPairLoader resizes pairs to target_hw (as the deploy dataloader does) and is a no-op otherwise."""
    from nvidia_tao_pytorch.cv.depth_net.scripts.quantize import StereoPairLoader

    resized = list(StereoPairLoader(_fake_stereo_loader(num_batches=2, hw=(30, 46)), target_hw=(32, 64)))
    assert [tuple(x.shape) for x in resized] == [(1, 6, 32, 64)] * 2
    same = list(StereoPairLoader(_fake_stereo_loader(num_batches=1, hw=(32, 64)), target_hw=(32, 64)))
    assert tuple(same[0].shape) == (1, 6, 32, 64)
    untouched = list(StereoPairLoader(_fake_stereo_loader(num_batches=1, hw=(30, 46))))
    assert tuple(untouched[0].shape) == (1, 6, 30, 46)


@pytest.mark.cv_unit
def test_stereo_data_module_calibration_stage(tmp_path, monkeypatch):
    """Test that the stereo DataModule builds a capped, inference-transformed calibration dataset."""
    import torch
    from omegaconf import OmegaConf

    from nvidia_tao_pytorch.cv.depth_net.dataloader import pl_stereo_data_module as module

    class _FakeDataset(torch.utils.data.Dataset):
        def __init__(self, data_file, transform, max_disparity):
            with open(data_file, "r", encoding="utf-8") as handle:
                self.lines = handle.read().splitlines()
            self.transform = transform
            self.max_disparity = max_disparity

        def __len__(self):
            return len(self.lines)

        def __getitem__(self, idx):
            return {"image": torch.zeros(3, 8, 8), "right_image": torch.ones(3, 8, 8)}

    captured = {}

    def _fake_build(data_sources, transform, max_disparity=None):
        captured["data_sources"] = data_sources
        return _FakeDataset(data_sources[0]["data_file"], transform, max_disparity)

    def _fake_transforms(aug_config, max_disparity=None, split="train"):
        captured["split"] = split
        return None

    monkeypatch.setattr(module, "build_stereo_dataset", _fake_build)
    monkeypatch.setattr(module, "build_stereo_transforms", _fake_transforms)

    list_file = tmp_path / "train.txt"
    list_file.write_text("\n".join(f"l{i}.png r{i}.png d{i}.pfm" for i in range(10)))

    dataset_cfg = OmegaConf.structured(DepthNetDatasetConfig())
    dataset_cfg.quant_calibration_dataset.data_sources = [
        {"dataset_name": "Middlebury", "data_file": str(list_file)}
    ]
    dataset_cfg.quant_calibration_dataset.num_samples = 4
    dataset_cfg.quant_calibration_dataset.workers = 0

    dm = module.StereoDepthNetDataModule(dataset_cfg)
    with pytest.raises(ValueError, match="setup"):
        dm.calib_dataloader()
    dm.setup(stage="calibration")
    loader = dm.calib_dataloader()

    assert captured["split"] == "infer"
    assert captured["data_sources"][0]["data_file"] == str(list_file)
    assert len(dm.calib_dataset) == 4
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0]["image"].shape == (1, 3, 8, 8)
    assert batches[0]["right_image"].shape == (1, 3, 8, 8)


@pytest.mark.cv_unit
def test_stereo_data_module_calibration_falls_back_to_test_dataset(tmp_path, monkeypatch):
    """Test that an unset quant_calibration_dataset.data_sources falls back to test_dataset."""
    from omegaconf import OmegaConf

    from nvidia_tao_pytorch.cv.depth_net.dataloader import pl_stereo_data_module as module

    captured = {}
    monkeypatch.setattr(module, "build_stereo_transforms", lambda *a, **k: None)

    def _fake_build(data_sources, transform, max_disparity=None):
        captured["data_sources"] = data_sources
        return list(range(3))

    monkeypatch.setattr(module, "build_stereo_dataset", _fake_build)

    dataset_cfg = OmegaConf.structured(DepthNetDatasetConfig())
    dataset_cfg.test_dataset.data_sources = [{"dataset_name": "Middlebury", "data_file": "/x/test.txt"}]
    dm = module.StereoDepthNetDataModule(dataset_cfg)
    dm.setup(stage="calibration")
    assert captured["data_sources"][0]["data_file"] == "/x/test.txt"
    assert len(dm.calib_dataset) == 3  # num_samples (128) > len -> no Subset

    dataset_cfg.test_dataset.data_sources = None
    with pytest.raises(ValueError, match="data_sources"):
        module.StereoDepthNetDataModule(dataset_cfg).setup(stage="calibration")
