# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PointPillars models."""
from collections import namedtuple
import os
import tempfile

from easydict import EasyDict
import numpy as np
import torch

from .detectors import build_detector
from eff.core.codec import decrypt_stream


def decrypt_pytorch(input_file_name, output_file_name, key):
    """Decrypt the TLT model to Pytorch model"""
    with open(input_file_name, "rb") as open_temp_file, open(output_file_name, "wb") as open_encoded_file:
        decrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def model_config_lower_case(cfg):
    """Convert cfg key to lower cases."""
    new_cfg = EasyDict()
    for k, v in cfg.items():
        if isinstance(v, (dict, EasyDict)):
            new_cfg[k.lower()] = model_config_lower_case(v)
        else:
            if k == "NMS_PRE_MAXSIZE":
                new_k = "nms_pre_max_size"
            elif k == "NMS_POST_MAXSIZE":
                new_k = "nms_post_max_size"
            else:
                new_k = k.lower()
            new_cfg[new_k] = v
    return new_cfg


def load_checkpoint(model_path, key, to_cpu=False):
    """Helper function to load a saved checkpoint."""
    loc_type = torch.device('cpu') if to_cpu else None
    if model_path.endswith(".tlt"):
        handle, temp_name = tempfile.mkstemp(".pth")
        os.close(handle)
        decrypt_pytorch(model_path, temp_name, key)
        loaded_model = torch.load(temp_name, map_location=loc_type, weights_only=False)
        os.remove(temp_name)
    else:
        loaded_model = torch.load(model_path, map_location=loc_type, weights_only=False)
    epoch = it = 0
    opt_state = None
    # It can be a dict or a bare-metal model
    if isinstance(loaded_model, dict):
        if "model" in loaded_model and loaded_model["model"] is not None:
            model = loaded_model["model"]
        else:
            raise KeyError(f"Key `model` not found in model loaded from {model_path}")
        # Load optimizer states
        if (
            "optimizer_state" in loaded_model and
            loaded_model["optimizer_state"] is not None
        ):
            opt_state = loaded_model["optimizer_state"]
        epoch = loaded_model.get('epoch', 0)
        it = loaded_model.get('it', 0.0)
    else:
        model = loaded_model
    # It can be a DDP wrapper
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    # For backward compatibility fo old PP PTMs
    model.model_cfg = model_config_lower_case(model.model_cfg)
    model.dense_head.model_cfg = model_config_lower_case(model.dense_head.model_cfg)
    return model, opt_state, epoch, it


def build_network(model_cfg, num_class, dataset):
    """Build the network."""
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def build_model_and_optimizer(
    model_cfg, num_class, dataset,
    pruned_model_path, resume_training_checkpoint_path,
    pretrained_model_path, to_cpu, logger,
    key
):
    """Build model and optimizer."""
    epoch = 0
    it = 0.
    opt_state = None
    if resume_training_checkpoint_path:
        # Case 1: resume an interrupted training
        model, opt_state, epoch, it = load_checkpoint(
            resume_training_checkpoint_path, key, to_cpu
        )
        logger.info(f"Model resumed from: {resume_training_checkpoint_path}")
    elif pruned_model_path:
        # Case 2: retrain a possibly pruned model
        # No optimizer states in pruned model
        model = load_checkpoint(pruned_model_path, key, to_cpu)[0]
        logger.info(f"Pruned model loaded from: {pruned_model_path}")
    else:
        # Case 3: Build a new model from scratch
        model = build_network(model_cfg, num_class, dataset)
        # Case 4: Using pretrained weights
        if pretrained_model_path:
            pretrained_model = load_checkpoint(pretrained_model_path, key, to_cpu)[0]
            loaded_state_dict = pretrained_model.state_dict()
            current_model_dict = model.state_dict()
            new_state_dict = dict()
            for k in current_model_dict:
                if (
                    k in loaded_state_dict and
                    loaded_state_dict[k].size() == current_model_dict[k].size()
                ):
                    new_state_dict.update({k: loaded_state_dict[k]})
            model.load_state_dict(new_state_dict, strict=False)
            new_model_dict = model.state_dict()
            loaded_layers = []
            unloaded_layers = []
            for n in new_model_dict:
                if n in new_state_dict:
                    loaded_layers.append(n)
                else:
                    unloaded_layers.append(n)
            logger.info("Layers initialized from pretrained model:")
            logger.info("=" * 30)
            for m in loaded_layers:
                logger.info(m)
            logger.info("=" * 30)
            logger.info("Layers initialized randomly:")
            logger.info("=" * 30)
            for m in unloaded_layers:
                logger.info(m)
            logger.info("=" * 30)
            logger.info(f"Pretrained weights loaded from: {pretrained_model_path}")
    return model, opt_state, epoch, it


def load_data_to_gpu(batch_dict):
    """Load data to GPU."""
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    """Model function decorator for training."""
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        """Custom model function."""
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
