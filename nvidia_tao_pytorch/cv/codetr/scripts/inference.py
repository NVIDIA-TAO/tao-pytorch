# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native inference with CoDETR PyTorch checkpoint."""

import os

import torch
from pytorch_lightning import Trainer

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import initialize_inference_experiment
from nvidia_tao_pytorch.core.tlt_logging import logging

from nvidia_tao_pytorch.cv.deformable_detr.dataloader.pl_od_data_module import ODDataModule
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.codetr.model.utils import map_codetr_checkpoint


def build_and_load_codetr_model(experiment_config, model_path):
    """Build a CoDETRPlModel and load weights from a .pth/.tlt checkpoint.

    Handles both TAO-format and reference Co-DETR checkpoints (auto-detected),
    drops EMA keys, pads ``label_enc`` for the background token, and skips
    size-mismatched keys (e.g. differing collab head conv shapes).

    Args:
        experiment_config: Hydra experiment spec (with model/dataset/etc).
        model_path (str): path to a ``.pth`` or ``.tlt`` checkpoint file.

    Returns:
        CoDETRPlModel: model with checkpoint weights loaded (still on CPU).

    Raises:
        NotImplementedError: if ``model_path`` is not a ``.pth`` or ``.tlt`` file.
    """
    if not (model_path.endswith('.tlt') or model_path.endswith('.pth')):
        if model_path.endswith('.engine'):
            raise NotImplementedError(
                "TensorRT inference is supported through tao-deploy. "
                "Please use tao-deploy to generate TensorRT engine and run inference."
            )
        raise NotImplementedError("Model path format is only supported for .tlt or .pth")

    model = CoDETRPlModel(experiment_spec=experiment_config)
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Filter out EMA keys (flattened with underscores; not usable directly)
    ema_keys = [k for k in state_dict if k.startswith("ema_")]
    if ema_keys:
        logging.info("Dropping %d EMA keys from checkpoint.", len(ema_keys))
        for k in ema_keys:
            del state_dict[k]

    # Detect checkpoint format: TAO checkpoints already have the model.
    # prefix; reference Co-DETR checkpoints start with backbone./query_head.
    is_reference_ckpt = any(
        k.startswith(("backbone.", "query_head.", "neck.", "bbox_head."))
        for k in state_dict
    )
    if is_reference_ckpt:
        logging.info("Detected reference Co-DETR checkpoint — mapping keys to TAO format.")
        dec_layers = experiment_config.model.dec_layers
        # num_backbone_stages: for FPN-style necks derive from neck.convs;
        # for SFP-style necks (ViT) this is unused (SFP maps to backbone).
        num_backbone_stages = len({
            k.split('.')[2] for k in state_dict if k.startswith('neck.convs.')
        })
        state_dict = map_codetr_checkpoint(
            state_dict,
            num_backbone_stages=num_backbone_stages,
            dec_layers=dec_layers,
        )

    # Pad label_enc from num_classes to num_classes+1 (background token)
    label_key = "model.model.model.label_enc.weight"
    if label_key in state_dict and label_key in model.state_dict():
        ckpt_w = state_dict[label_key]
        model_w = model.state_dict()[label_key]
        if ckpt_w.shape[0] == model_w.shape[0] - 1:
            state_dict[label_key] = torch.cat(
                [ckpt_w, torch.zeros(1, ckpt_w.shape[1], device=ckpt_w.device)], dim=0
            )

    # Filter out size-mismatched keys (e.g. collab head conv differences)
    model_sd = model.state_dict()
    to_remove = []
    for k, v in state_dict.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            logging.info("Skipping size-mismatched key %s: ckpt %s vs model %s",
                         k, list(v.shape), list(model_sd[k].shape))
            to_remove.append(k)
    for k in to_remove:
        del state_dict[k]

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        # Categorize missing keys: collab/aux heads + buffers are expected;
        # decoder/encoder/backbone parameters are NOT and indicate a mapping bug.
        expected_prefixes = (
            "model.collab_heads.",
            "model.downsample.",
        )
        # Buffers / non-trainable state we don't load from a Co-DETR ckpt:
        expected_substr = (
            "rope_win.freqs",
            "rope_glb.freqs",
            "label_enc.weight",
            "num_batches_tracked",
            "transformer.decoder.bbox_embed",
            "transformer.decoder.class_embed",
        )
        unexpected_missing = [
            k for k in result.missing_keys
            if not k.startswith(expected_prefixes) and not any(s in k for s in expected_substr)
        ]
        logging.info(
            "Missing keys: %d total (%d expected for collab heads / buffers, %d unexpected)",
            len(result.missing_keys), len(result.missing_keys) - len(unexpected_missing),
            len(unexpected_missing),
        )
        if unexpected_missing:
            logging.warning("UNEXPECTED missing keys (mapping bug?): %s",
                            unexpected_missing[:20])
    if result.unexpected_keys:
        logging.warning("Unexpected keys (in checkpoint, not in model): %s",
                        result.unexpected_keys[:10])

    return model


def run_experiment(experiment_config, key):
    """Execute CoDETR inference."""
    model_path, trainer_kwargs = initialize_inference_experiment(experiment_config, key)

    # start_from_one=False: the reference Co-DETR checkpoint uses 0-indexed class labels
    dm = ODDataModule(experiment_config.dataset, subtask_config=experiment_config.inference,
                      start_from_one=False)
    dm.setup(stage="predict")

    model = build_and_load_codetr_model(experiment_config, model_path)
    trainer = Trainer(**trainer_kwargs)
    trainer.predict(model, datamodule=dm)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="inference", schema=ExperimentConfig
)
@monitor_status(name="CoDETR", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run CoDETR inference."""
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
