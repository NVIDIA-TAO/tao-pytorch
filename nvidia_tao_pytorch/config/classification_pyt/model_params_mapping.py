# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model Parameters Mapping Module"""

map_params = {"head": {"in_channels": {
    "fan_tiny_8_p4_hybrid": 192,  # FAN
    "fan_small_12_p4_hybrid": 384,
    "fan_base_16_p4_hybrid": 448,
    "fan_large_16_p4_hybrid": 480,
    "fan_Xlarge_16_p4_hybrid": 768,
    "fan_base_18_p16_224": 448,
    "fan_tiny_12_p16_224": 192,
    "fan_small_12_p16_224_se_attn": 384,
    "fan_small_12_p16_224": 384,
    "fan_large_24_p16_224": 480,
    "gc_vit_xxtiny": 512,  # GCViT
    "gc_vit_xtiny": 512,
    "gc_vit_tiny": 512,
    "gc_vit_small": 768,
    "gc_vit_base": 1024,
    "gc_vit_large": 1536,
    "gc_vit_large_384": 1536,
    "faster_vit_0_224": 512,  # FasterViT
    "faster_vit_1_224": 640,
    "faster_vit_2_224": 768,
    "faster_vit_3_224": 1024,
    "faster_vit_4_224": 1568,
    "faster_vit_5_224": 2560,
    "faster_vit_6_224": 2560,
    "faster_vit_4_21k_224": 1568,
    "faster_vit_4_21k_384": 1568,
    "faster_vit_4_21k_512": 1568,
    "faster_vit_4_21k_768": 1568,
    "ViT-L-14": 768,
    "ViT-B-16": 512,
    "ViT-L-14-336": 768,
    "ViT-g-14": 1024,
    "ViT-H-14": 1024,
    "EVA02-E-14-plus": 1024,
    "EVA02-E-14": 1024,
    "EVA02-L-14-336": 768,
    "EVA02-L-14": 768,
    "ViT-B-32": 512,
    "vit_large_patch14_dinov2_swiglu": 1024,
    "vit_giant_patch14_reg4_dinov2_swiglu": 1536,
    "ViT-H-14-SigLIP-CLIPA-224": 1024,
    "ViT-L-14-SigLIP-CLIPA-336": 768,
    "ViT-L-14-SigLIP-CLIPA-224": 768,
    "c_radio_p1_vit_huge_patch16_224_mlpnorm": 3840,
    "c_radio_p2_vit_huge_patch16_224_mlpnorm": 5120,
    "c_radio_p3_vit_huge_patch16_224_mlpnorm": 3840
}}}

# Map input resolution for different backbones
map_input_lr_head = {
    "faster_vit_4_21k_384": 384,
    "faster_vit_4_21k_512": 512,
    "faster_vit_4_21k_768": 768,
    "ViT-L-14-336": 336,
    "EVA02-L-14-336": 336,
    "gc_vit_large_384": 384,
    "ViT-L-14-SigLIP-CLIPA-336": 336
}

# Map model config for CLIP model
map_clip_model_cfg = {
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
    }
}
