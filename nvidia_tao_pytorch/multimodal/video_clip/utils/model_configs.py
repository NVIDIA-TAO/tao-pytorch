# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

""" CLIP supported model configs. """

internvideo2clip_model_configs = {
    "internvideo2-clip-l14": {
        "model_type": "internvideo2clip",
        "description": "OpenGVLab InternVideo2-CLIP L14",
        "image_size": 224,
        "num_frames": 8,
        "embed_dim": 512,
        "init_logit_scale": 4.605170185988092,
        "init_logit_bias": 0.0,
    },
}

# Map model config for CLIP model - clip training
map_clip_model_cfg = {
    "ViT-H-14-SigLIP-CLIPA-84": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 84,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 16,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-H-14-SigLIP-CLIPA-224": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-H-14-SigLIP-CLIPA-336": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 336,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-H-14-SigLIP-CLIPA-574": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 574,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-L-14-SigLIP-CLIPA-84": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 84,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 16,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 768,
            "heads": 12,
            "layers": 12,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-L-14-SigLIP-CLIPA-224": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 768,
            "heads": 12,
            "layers": 12,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-L-14-SigLIP-CLIPA-336": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 336,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 256,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 768,
            "heads": 12,
            "layers": 12,
            "pool_type": "last",
            "no_causal_mask": True
        }
    }
}

# SigLIP2 model configurations (standalone, not via RADIO).
# Uses Google's SigLIP2 vision + text encoders from HuggingFace.
siglip2_model_configs = {
    # NaFlex (dynamic resolution)
    "siglip2-so400m-patch16-naflex": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch16-naflex",
        "description": "SigLIP2 SO400M model (NaFlex, dynamic resolution)",
        "image_size": 384,
        "patch_size": 16,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    # Fixed resolution patch14 variants
    "siglip2-so400m-patch14-224": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch14-224",
        "description": "SigLIP2 SO400M model with patch14, 224x224 resolution",
        "image_size": 224,
        "patch_size": 14,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "siglip2-so400m-patch14-384": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch14-384",
        "description": "SigLIP2 SO400M model with patch14, 384x384 resolution",
        "image_size": 384,
        "patch_size": 14,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    # Fixed resolution patch16 variants
    "siglip2-so400m-patch16-256": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch16-256",
        "description": "SigLIP2 SO400M model with patch16, 256x256 resolution",
        "image_size": 256,
        "patch_size": 16,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "siglip2-so400m-patch16-384": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch16-384",
        "description": "SigLIP2 SO400M model with patch16, 384x384 resolution",
        "image_size": 384,
        "patch_size": 16,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "siglip2-so400m-patch16-512": {
        "model_type": "siglip2",
        "hf_model": "google/siglip2-so400m-patch16-512",
        "description": "SigLIP2 SO400M model with patch16, 512x512 resolution",
        "image_size": 512,
        "patch_size": 16,
        "embed_dim": 1152,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
}

# RADIO model configurations (loaded via torch.hub from NVlabs/RADIO)
# User-facing adaptor names: 'siglip' (default) or 'clip' (DFN CLIP).
# 'siglip' auto-resolves to the correct internal name per model version.
# Recommended settings per adaptor:
#   'siglip': loss_type=siglip, init_logit_scale=2.3026, init_logit_bias=-10.0
#   'clip':   loss_type=clip,   init_logit_scale=2.6592, init_logit_bias=0.0
radio_model_configs = {
    "c-radio_v3-h": {
        "model_type": "radio",
        "version": "c-radio_v3-h",
        "adaptor_name": "siglip2-g",
        "description": "C-RADIOv3-H model (ViT-H/16) - Commercial License",
        "image_size": 224,
        "embed_dim": 1280,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "c-radio_v3-l": {
        "model_type": "radio",
        "version": "c-radio_v3-l",
        "adaptor_name": "siglip2",
        "description": "C-RADIOv3-L model (ViT-L/16) - Commercial License",
        "image_size": 224,
        "embed_dim": 1024,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "c-radio_v3-b": {
        "model_type": "radio",
        "version": "c-radio_v3-b",
        "adaptor_name": "siglip2",
        "description": "C-RADIOv3-B model (ViT-B/16) - Commercial License",
        "image_size": 224,
        "embed_dim": 768,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "c-radio_v3-g": {
        "model_type": "radio",
        "version": "c-radio_v3-g",
        "adaptor_name": "siglip2",
        "description": "C-RADIOv3-g model (ViT-g/14) - Commercial License",
        "image_size": 224,
        "embed_dim": 1536,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
}

# OpenCLIP model configurations (using backbone_v2)
# These models use the backbone_v2/open_clip.py implementation
openclip_model_configs = {
    "ViT-L-14-SigLIP-CLIPA-224": {
        "model_type": "openclip",
        "description": "ViT-L/14 with SigLIP CLIPA training, 224x224 images",
        "image_size": 224,
        "embed_dim": 768,
        "init_logit_scale": 2.3026,  # np.log(10) for SigLIP
        "init_logit_bias": -10.0,
    },
    "ViT-L-14-SigLIP-CLIPA-336": {
        "model_type": "openclip",
        "description": "ViT-L/14 with SigLIP CLIPA training, 336x336 images",
        "image_size": 336,
        "embed_dim": 768,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "ViT-H-14-SigLIP-CLIPA-224": {
        "model_type": "openclip",
        "description": "ViT-H/14 with SigLIP CLIPA training, 224x224 images",
        "image_size": 224,
        "embed_dim": 1024,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "ViT-H-14-SigLIP-CLIPA-336": {
        "model_type": "openclip",
        "description": "ViT-H/14 with SigLIP CLIPA training, 336x336 images",
        "image_size": 336,
        "embed_dim": 1024,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "ViT-H-14-SigLIP-CLIPA-574": {
        "model_type": "openclip",
        "description": "ViT-H/14 with SigLIP CLIPA training, 574x574 images",
        "image_size": 574,
        "embed_dim": 1024,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "ViT-L-14-SigLIP-CLIPA-84": {
        "model_type": "openclip",
        "description": "ViT-L/14 SigLIP CLIPA, 84x84 (low-res)",
        "image_size": 84,
        "embed_dim": 768,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
    "ViT-H-14-SigLIP-CLIPA-84": {
        "model_type": "openclip",
        "description": "ViT-H/14 SigLIP CLIPA, 84x84 (low-res)",
        "image_size": 84,
        "embed_dim": 1024,
        "init_logit_scale": 2.3026,
        "init_logit_bias": -10.0,
    },
}

all_model_configs = {
    **map_clip_model_cfg,
    **radio_model_configs,
    **siglip2_model_configs,
    **openclip_model_configs,
}


# Registries that build_model (see model/video_clip.py) dispatches on, by
# membership, in this order. map_clip_model_cfg is intentionally excluded: it
# holds raw open_clip architecture defs for the fallback path and legitimately
# shares names with openclip_model_configs.
_DISPATCH_REGISTRIES = {
    "internvideo2clip": internvideo2clip_model_configs,
    "radio": radio_model_configs,
    "siglip2": siglip2_model_configs,
    "openclip": openclip_model_configs,
}

_REQUIRED_CONFIG_KEYS = {"model_type", "image_size", "embed_dim"}


def _validate_model_registries():
    """Fail fast on registry mistakes that would otherwise mis-dispatch silently.

    Guarantees that (1) every dispatched entry declares the core fields
    build_model and the adapters rely on and the expected ``model_type``, and
    (2) the dispatch registries are pairwise disjoint, so a model name can never
    resolve to two different backbones depending on check order.
    """
    for expected_type, registry in _DISPATCH_REGISTRIES.items():
        for name, entry in registry.items():
            missing = _REQUIRED_CONFIG_KEYS - entry.keys()
            if missing:
                raise ValueError(
                    f"Model config '{name}' is missing required "
                    f"key(s): {sorted(missing)}."
                )
            if entry["model_type"] != expected_type:
                raise ValueError(
                    f"Model config '{name}' has model_type "
                    f"'{entry['model_type']}' but lives in the "
                    f"'{expected_type}' registry."
                )

    items = list(_DISPATCH_REGISTRIES.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (name_a, reg_a), (name_b, reg_b) = items[i], items[j]
            overlap = reg_a.keys() & reg_b.keys()
            if overlap:
                raise ValueError(
                    f"Model name(s) {sorted(overlap)} appear in both the "
                    f"'{name_a}' and '{name_b}' registries; build_model "
                    f"dispatch would be order-dependent."
                )


_validate_model_registries()
