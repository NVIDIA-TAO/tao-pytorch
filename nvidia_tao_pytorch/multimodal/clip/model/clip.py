# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLIP model builder."""

import open_clip

from nvidia_tao_pytorch.multimodal.clip.utils.model_configs import (
    map_clip_model_cfg,
    radio_model_configs,
    siglip2_model_configs,
    openclip_model_configs,
)
from nvidia_tao_pytorch.multimodal.clip.model.builders import (
    build_radio_model,
    build_siglip2_model,
    build_openclip_model,
)


class CLIPModelPreProcess():
    """Encapsulates the CLIP model, tokenizer, and preprocessing functions
    for training and validation.

    Args:
        model: The initialized CLIP model.
        tokenizer: Tokenizer associated with the CLIP model.
        preprocess_train: Preprocessing function for training data.
        preprocess_val: Preprocessing function for validation data.
    """

    def __init__(self, model, tokenizer, preprocess_train, preprocess_val):
        """Initialize CLIPModelPreProcess."""
        self.model = model
        self.tokenizer = tokenizer
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val


def build_model(experiment_config,
                export=False):
    """Build a CLIP-based model with preprocessing and tokenizer based on
    the given configuration.

    Args:
        experiment_config: Configuration object containing model and
            dataset parameters.
        export (bool): Flag to indicate if the model is prepared for
            ONNX export. Defaults to False.

    Returns:
        CLIPModelPreProcess: A wrapper object containing the model,
            tokenizer, and preprocessing functions.
    """
    model_name = experiment_config.model.type
    aug_cfg = experiment_config.dataset.augmentation
    freeze_vision = getattr(
        experiment_config.model, 'freeze_vision_encoder', False)
    freeze_text = getattr(
        experiment_config.model, 'freeze_text_encoder', False)
    canonicalize_text = getattr(
        experiment_config.model, 'canonicalize_text', False)

    image_size = experiment_config.model.image_size

    # Infer logit scale/bias from loss_type if not explicitly set
    loss_type = getattr(experiment_config.train, 'loss_type', 'clip')
    init_logit_scale = experiment_config.model.init_logit_scale
    init_logit_bias = experiment_config.model.init_logit_bias
    if init_logit_scale is None:
        init_logit_scale = 2.3026 if loss_type == 'siglip' else 2.6592
    if init_logit_bias is None:
        init_logit_bias = -10.0 if loss_type == 'siglip' else 0.0

    if model_name in radio_model_configs:
        adaptor_name = getattr(
            experiment_config.model, 'adaptor_name', None)
        model, preprocess_train, preprocess_val, tokenizer = (
            build_radio_model(
                model_version=model_name,
                adaptor_name=adaptor_name,
                aug_cfg=dict(aug_cfg) if aug_cfg is not None else None,
                freeze_vision_encoder=freeze_vision,
                freeze_text_encoder=freeze_text,
                image_size=image_size,
                logit_scale_init=init_logit_scale,
                logit_bias_init=init_logit_bias,
                canonicalize_text=canonicalize_text,
            ))

    elif model_name in siglip2_model_configs:
        model, preprocess_train, preprocess_val, tokenizer = (
            build_siglip2_model(
                model_version=model_name,
                aug_cfg=dict(aug_cfg) if aug_cfg is not None else None,
                freeze_vision_encoder=freeze_vision,
                freeze_text_encoder=freeze_text,
                image_size=image_size,
                logit_scale_init=init_logit_scale,
                logit_bias_init=init_logit_bias,
                canonicalize_text=canonicalize_text,
            ))

    elif model_name in openclip_model_configs:
        model, preprocess_train, preprocess_val, tokenizer = (
            build_openclip_model(
                model_version=model_name,
                aug_cfg=dict(aug_cfg) if aug_cfg is not None else None,
                freeze_vision_encoder=freeze_vision,
                freeze_text_encoder=freeze_text,
                image_size=image_size,
                logit_scale_init=init_logit_scale,
                logit_bias_init=init_logit_bias,
                canonicalize_text=canonicalize_text,
            ))

    else:
        # Fallback: Build standard CLIP model using open_clip directly
        # This path is for models not in any config dict
        # (e.g., pretrained OpenCLIP models)
        model_config = map_clip_model_cfg.get(model_name)

        # Handle customized clip config
        if model_config is not None:
            open_clip.factory._MODEL_CONFIGS[model_name] = model_config

        model, preprocess_train, preprocess_val = (
            open_clip.create_model_and_transforms(
                model_name,
                aug_cfg=dict(aug_cfg) if aug_cfg is not None else None
            ))
        tokenizer = open_clip.get_tokenizer(model_name)

    return CLIPModelPreProcess(
        model, tokenizer, preprocess_train, preprocess_val)
