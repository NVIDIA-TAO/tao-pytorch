# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export StyleGAN-XL model to ONNX."""

import os
import torch
import torch.nn.functional as F
import numpy as np
import PIL
import onnxruntime

from nvidia_tao_pytorch.config.stylegan_xl.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.utilities import encrypt_onnx
from nvidia_tao_pytorch.sdg.stylegan_xl.utils.onnx_export import ONNXExporter, patch_affine_grid_generator
from nvidia_tao_pytorch.sdg.stylegan_xl.model.sx_pl_model import StyleganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.model.bg_pl_model import BigdatasetganPlModel
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_sx_data_module import SXDataModule
from nvidia_tao_pytorch.sdg.stylegan_xl.dataloader.pl_bg_data_module import BGDataModule


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="export", schema=ExperimentConfig
)
@monitor_status(name="StyleGAN-XL", mode="export")
def main(cfg: ExperimentConfig) -> None:
    """CLI wrapper to run export.
    This function parses the command line interface for tlt-export, instantiates the respective
    exporter and serializes the trained model to an etlt file. The tools also runs optimization
    to the int8 backend.

    Args:
        cl_args(list): Arguments to parse.

    Returns:
        No explicit returns.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run_export(cfg)


def run_export(experiment_config):
    """Wrapper to run export of tlt models.

    Args:
        args (dict): Dictionary of parsed arguments to run export.

    Returns:
        No explicit returns.
    """
    gpu_id = experiment_config.export.gpu_id
    torch.cuda.set_device(gpu_id)

    # Parsing command line arguments.
    model_path = experiment_config.export.checkpoint
    key = experiment_config.encryption_key
    # set the encryption key:
    TLTPyTorchCookbook.set_passphrase(key)

    output_file = experiment_config.export.onnx_file
    opset_version = experiment_config.export.opset_version
    batch_size = experiment_config.export.batch_size

    if experiment_config.export.on_cpu:
        device = "cpu"
    else:
        device = "cuda"
    if batch_size is None or batch_size == -1:
        # Using an input batch size of 1 as a sample for JIT tracing with dynamic batch ONNX.
        # Ideally, using different input batch sizes should not affect the dynamic batching capability of the generated ONNX model.
        input_batch_size = 1
    else:
        input_batch_size = batch_size

    # Set default output filename if the filename
    # isn't provided over the command line.
    if output_file is None:
        split_name = os.path.splitext(model_path)[0]
        output_file = "{}.onnx".format(split_name)

    # Warn the user if an exported file already exists.
    assert not os.path.exists(output_file), "Default onnx file {} already "\
        "exists".format(output_file)

    # Make an output directory if necessary.
    output_root = os.path.dirname(os.path.realpath(output_file))
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    if experiment_config.task == 'stylegan':
        # build dataloader
        dm = SXDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        pl_model = StyleganPlModel.load_from_checkpoint(model_path,
                                                        map_location=device,
                                                        experiment_spec=experiment_config,
                                                        dm=dm
                                                        )
        model = pl_model.G_ema
        c_dim = model.c_dim
        z_dim = model.z_dim
        input_names = ['z', 'labels']
        output_names = ['output_image']

    elif experiment_config.task == 'bigdatasetgan':
        # build dataloader
        dm = BGDataModule(experiment_config.dataset)
        # build model and load from the given checkpoint
        pl_model = BigdatasetganPlModel.load_from_checkpoint(model_path,
                                                             map_location="cpu",
                                                             experiment_spec=experiment_config,
                                                             dm=dm
                                                             )
        model = pl_model.model
        c_dim = model.feature_extractor.model.c_dim
        z_dim = model.feature_extractor.model.z_dim
        input_names = ['z', 'labels']
        output_names = ['output_image', 'output_mask']
    else:
        raise NotImplementedError("Task {} is not implemented".format(experiment_config.task))

    # Only export generator
    model.eval()
    model.to(device)

    # Patch affine_grid due to unsupported op for torch.nn.functional.affine_grid when export
    patch_affine_grid_generator()

    # Set your class and batch size
    seed = np.random.randint(0, 10000)

    # Step 1: Generate random noise and labels
    z = np.random.RandomState(seed).randn(input_batch_size, z_dim)
    z_tensor = torch.from_numpy(z).to(device)  # Convert to tensor and move to GPU
    if c_dim > 0:
        class_idx = np.random.randint(0, c_dim)
        class_indices = torch.full((input_batch_size,), class_idx).unsqueeze(1).to(device)  # Unsqueeze in order to meet dynamic batch size
        labels = F.one_hot(class_indices.squeeze(1), c_dim)
        onnx_inputs = (z_tensor.to(torch.float32), labels.to(torch.float32))
    else:
        onnx_inputs = (z_tensor.to(torch.float32), None)
    # Step 2: Export the model to ONNX
    # Some file operations for encrpytion
    if output_file.endswith('.etlt'):
        tmp_onnx_file = output_file.replace('.etlt', '.onnx')
    else:
        tmp_onnx_file = output_file
    # Main export function
    onnx_export = ONNXExporter()
    model.forward = model.onnx_forward
    onnx_export.export_model(model, batch_size,
                             tmp_onnx_file,
                             onnx_inputs,
                             input_names=input_names,
                             opset_version=opset_version,
                             output_names=output_names,
                             do_constant_folding=True,
                             verbose=experiment_config.export.verbose)
    onnx_export.check_onnx(tmp_onnx_file)

    if output_file.endswith('.etlt') and key:
        # Encrypt the onnx if and only if key is provided and output file name ends with .etlt
        encrypt_onnx(tmp_file_name=tmp_onnx_file,
                     output_file_name=output_file,
                     key=key)
        os.remove(tmp_onnx_file)
        print(f"ONNX file stored at {output_file}")
        return  # encrypted onnx does not involve onnxruntime sample test
    else:
        print(f"ONNX file stored at {output_file}")

    # Only run the following code when `test_onnxruntime` flag is True
    if not experiment_config.export.onnxruntime.test_onnxruntime:
        return

    # Step 3: Run inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(output_file)

    # Prepare the input for ONNX Runtime
    # Step 4: Generate random noise and labels
    z = np.random.RandomState(experiment_config.export.onnxruntime.runtime_seed).randn(experiment_config.export.onnxruntime.runtime_batch_size, z_dim)
    z_tensor = torch.from_numpy(z)  # Convert to tensor and move to GPU
    if c_dim > 0:
        class_indices = torch.full((experiment_config.export.onnxruntime.runtime_batch_size,), experiment_config.export.onnxruntime.runtime_class_dix).unsqueeze(1)
        labels = F.one_hot(class_indices.squeeze(1), c_dim)
        ort_inputs = {
            "z": z.astype(np.float32),
            "labels": labels.cpu().numpy().astype(np.float32)  # Make sure labels are in float32
        }
    else:
        ort_inputs = {
            "z": z.astype(np.float32),
            "labels": None  # Make sure labels are in float32
        }

    # Step 5: Run inference and save the results
    ort_outs = ort_session.run(None, ort_inputs)
    # Make an output directory if necessary and save the results for different task
    onnxruntime_root = experiment_config.results_dir
    if not onnxruntime_root:
        onnxruntime_root = os.path.join(experiment_config.export.onnxruntime.sample_result_dir)
    if experiment_config.task == 'bigdatasetgan':
        os.makedirs(os.path.join(onnxruntime_root, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(onnxruntime_root, 'images'), exist_ok=True)

        img, mask = ort_outs[0], ort_outs[1]
        labels = mask.argmax(axis=1)
        assert len(labels) == len(img)
        for batch_idx in range(len(labels)):
            # img
            img_np = (img[batch_idx] + 1) * 255 / 2  # Scale to [0, 255]
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # Clamp values to uint8 range
            img_np = img_np.transpose(1, 2, 0)

            # mask label
            labels_np = labels[batch_idx].squeeze()
            labels_np = (labels_np * 255).astype(np.uint8)

            print("Saving", f'{onnxruntime_root}/images/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')
            print("Saving", f'{onnxruntime_root}/masks/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')
            PIL.Image.fromarray(img_np, mode='RGB').save(f'{onnxruntime_root}/images/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')
            PIL.Image.fromarray(labels_np, mode='L').save(f'{onnxruntime_root}/masks/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')

    elif experiment_config.task == 'stylegan':
        os.makedirs(onnxruntime_root, exist_ok=True)

        img = ort_outs[0]
        for batch_idx in range(len(img)):
            # img
            img_np = (img[batch_idx] + 1) * 255 / 2  # Scale to [0, 255]
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # Clamp values to uint8 range
            img_np = img_np.transpose(1, 2, 0)

            print("Saving", f'{onnxruntime_root}/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')
            PIL.Image.fromarray(img_np, mode='RGB').save(f'{onnxruntime_root}/onnxruntime_seed{experiment_config.export.onnxruntime.runtime_seed:04d}_batchIdx{batch_idx}.png')

    else:
        raise NotImplementedError("Task {} is not implemented".format(experiment_config.task))


if __name__ == "__main__":
    main()
