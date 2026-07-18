# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Classification_PL Export Unit Tests
"""
import os
import onnx
import subprocess
import sys
import torch
import pytest
import tempfile
import onnxruntime as ort
from omegaconf import OmegaConf
import numpy as np
from PIL import Image

from nvidia_tao_pytorch.core.utilities import check_and_create
from nvidia_tao_pytorch.config.classification_pyt.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.classification_pyt.model.classifier_pl_model import ClassifierPlModel
from nvidia_tao_pytorch.cv.classification_pyt.utils.onnx_export import ONNXExporter


tmp_top_obj = tempfile.TemporaryDirectory()
tmp_top_dir = tmp_top_obj.name
# always use batch size 1 for onnx export, otherwise nvdinov2 will thorw gridsample not found error.
BATCH_SIZE = 1
OUTPUT_SHAPE = 224
SAMPLES = 10
NUM_CLASSES = 10
INPUT_SHAPE = 600
OUTPUT_SHAPE = 224
DATASET = 'CLDataset'

TEST_TOPOLOGIES = [
    # ConvNeXtV2.
    ("convnextv2_atto"),
    # DINOV2.
    ("vit_large_patch14_dinov2_swiglu"),
    # FAN.
    ("fan_small_12_p16_224"),
    ("fan_small_12_p4_hybrid"),
    ("fan_small_12_p16_224_se_attn"),
    # FasterViT.
    ("faster_vit_1_224"),
    # GCViT.
    ("gc_vit_xxtiny"),
    # OpenCLIP.
    ("vit_l_14_siglip_clipa_336"),
    # RADIO.
    ("c_radio_v2_vit_base_patch16"),
]
LARGE_BACKBONE_TOPOLOGIES = [
    # DINOV2.
    ("vit_giant_patch14_reg4_dinov2_swiglu"),
    # RADIO.
    ("c_radio_p3_vit_huge_patch16_mlpnorm"),
]
if not os.getenv("CI_PROJECT_DIR", None):
    TEST_TOPOLOGIES.extend(LARGE_BACKBONE_TOPOLOGIES)


@pytest.fixture
def _test_dir():
    # set this as dataset folder name
    splits = ['train', 'val', 'test']
    img_paths = []

    if not os.path.exists(tmp_top_dir):
        os.makedirs(tmp_top_dir)
    tmp_img_dir = os.path.join(tmp_top_dir)
    check_and_create(tmp_img_dir)

    # write the class.txt to tmp_img_dir, which consists of class names
    class_file = os.path.join(tmp_img_dir, 'classes.txt')
    with open(class_file, 'w') as f:
        for i in range(NUM_CLASSES):
            f.write(str(i) + '\n')

    for split in splits:
        tmp_split_img_dir = os.path.join(tmp_img_dir, split)
        check_and_create(tmp_split_img_dir)
        img_paths.append(tmp_split_img_dir)

    #Input images
    test_data = np.random.rand(INPUT_SHAPE, INPUT_SHAPE, 3) * 255
    test_data = test_data.astype(np.uint8)
    im = Image.fromarray(test_data)

    total_samples = SAMPLES
    for sample in range(total_samples):
        for img_path in img_paths:
            if 'test' in img_path:
                im.save(os.path.join(img_path, str(sample)+'.png'))
            else:
                for class_id in range(NUM_CLASSES):
                    class_dir = os.path.join(img_path, str(class_id))
                    check_and_create(class_dir)
                    # randomly scale the images
                    scale1 = np.random.uniform(0.5, 1.5)
                    scale2 = np.random.uniform(0.5, 1.5)
                    im_resized = im.resize((int(INPUT_SHAPE*scale1), int(INPUT_SHAPE*scale2)))
                    im_resized.save(os.path.join(class_dir, str(sample)+'.png'))

@pytest.fixture
def _test_experiment_spec():

    experiment_config = OmegaConf.structured(ExperimentConfig())
    experiment_config["dataset"]["root_dir"] = tmp_top_dir
    experiment_config["dataset"]["train_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "train")
    experiment_config["dataset"]["val_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "val")
    experiment_config["dataset"]["test_dataset"]["images_dir"] = os.path.join(tmp_top_dir, "test")
    experiment_config["dataset"]["dataset"] = DATASET
    experiment_config["dataset"]["img_size"] = OUTPUT_SHAPE
    experiment_config["dataset"]["batch_size"] = BATCH_SIZE
    experiment_config["dataset"]["num_classes"] = NUM_CLASSES
    experiment_config["dataset"]["classes_file"] = os.path.join(tmp_top_dir, "classes.txt")

    experiment_config["results_dir"] = tmp_top_dir

    experiment_config.train.num_epochs = 1
    experiment_config.train.num_gpus = 1
    experiment_config.train.num_nodes = 1

    yield experiment_config


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [17])
def test_classifier_onnx_compare_output(_test_dir, _test_experiment_spec, backbone, batch_size, opset_version):
    """Unit test for ONNX export on Classifier model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input']
    output_names = ['output']

    model = ClassifierPlModel(_test_experiment_spec).model
    model.eval()

    dummy_input = torch.ones(1, input_channel, input_height, input_width, device='cpu')
    print(dummy_input.dtype)
    with torch.no_grad():
        torch_output = model(dummy_input)
    print(torch_output[0].dtype)
    print("input shape",dummy_input.shape)
    print("output shape",torch_output.shape)
    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}"))
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}", f"{backbone}_opset{opset_version}.onnx")
    print("==========================", input_batch_size, batch_size)
    onnx_export = ONNXExporter()
    onnx_export.export_model(
        model, batch_size,
        onnx_path,
        dummy_input,
        input_names=input_names,
        opset_version=opset_version,
        output_names=output_names,
        do_constant_folding=True,
        verbose=_test_experiment_spec.export.verbose
    )

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"

    # Load ONNX and ONNXRuntime
    ort.set_seed(47)
    onnx.checker.check_model(onnx_path)
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )

    dummy_input2 = torch.ones(1, input_channel, input_height, input_width, device='cpu')
    with torch.no_grad():
        torch_output = model(dummy_input2)

    ort_inputs = {sess.get_inputs()[0].name: dummy_input2.detach().cpu().numpy()}

    onnx_output = sess.run(None, ort_inputs)
    print("onnx_len", len(onnx_output))
    print(f"Output shapes: {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    if not (list(torch_output.shape) == list(onnx_output[0].shape)):
        raise ValueError(f"Size Mismatch {list(torch_output.shape)} vs {list(onnx_output[0].shape)}")
    else:
        torch.testing.assert_close(torch_output.cpu(), torch.from_numpy(onnx_output[0]), rtol=5e-4, atol=1e-3)


@pytest.mark.cv_unit
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("batch_size", [-1])
@pytest.mark.parametrize("opset_version", [17])
def test_classifier_onnx_export(_test_dir, _test_experiment_spec, backbone, batch_size, opset_version):
    """Unit test for ONNX export on Classifier model.
    Includes unit test for onnxruntime inference and comparison with torch output
    Includes unit test for trt engine generation.
    """

    _test_experiment_spec.model.backbone.type = backbone

    # Define input and output of ONNX
    if batch_size == -1:
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    input_channel, input_height, input_width = 3, OUTPUT_SHAPE, OUTPUT_SHAPE
    input_names = ['input']
    output_names = ['output']

    model = ClassifierPlModel(_test_experiment_spec).model
    model.eval()
    model.cuda()

    dummy_input = torch.ones(input_batch_size, input_channel, input_height, input_width, device='cuda')

    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}"))
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}", f"{backbone}_opset{opset_version}.onnx")

    onnx_export = ONNXExporter()
    onnx_export.export_model(
        model, batch_size,
        onnx_path,
        dummy_input,
        input_names=input_names,
        opset_version=opset_version,
        output_names=output_names,
        do_constant_folding=True,
        verbose=_test_experiment_spec.export.verbose
    )

    onnx_export.check_onnx(onnx_path)

    assert os.path.exists(onnx_path), "ONNX file was not generated properly!"


@pytest.mark.cv_unit
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("backbone", TEST_TOPOLOGIES)
@pytest.mark.parametrize("opset_version", [17])
@pytest.mark.skip(reason="flaky test to be fixed")
def test_cls_trtexec(_test_dir, _test_experiment_spec, backbone, batch_size, opset_version):
    check_and_create(tmp_top_dir)

    check_and_create(os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}"))
    onnx_path = os.path.join(tmp_top_dir, f"{backbone}_opset{opset_version}", f"{backbone}_opset{opset_version}.onnx")

    input_height, input_width = OUTPUT_SHAPE, OUTPUT_SHAPE
    # Test TensorRT engine generation for dynamic batch size ONNX
    call = (
        f"trtexec --onnx={onnx_path} "
        f"--minShapes=input:{batch_size}x3x{input_height}x{input_width} "
        f"--optShapes=input:{batch_size}x3x{input_height}x{input_width} "
        f"--maxShapes=input:{batch_size}x3x{input_height}x{input_width} "
    )
    print(call)

    # Run the call as subprocess.
    subprocess.check_call(call, shell=True, stdout=sys.stdout, stderr=sys.stdout)


@pytest.mark.cv_unit
def tmp_obj_cleanup():
    tmp_top_obj.cleanup()
