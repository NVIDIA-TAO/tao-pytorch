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

"""InternVideo2 asset resolution utilities."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default HF repo for the InternVideo2-CLIP L14 assets, pinned to a commit so a
# repo-side change cannot silently alter the weights we train and ship on. Point
# model.internvideo2clip_hf_id (or the per-component fields) at your own repo to
# opt out; the pin is only applied to the default repo below.
DEFAULT_INTERNVIDEO2CLIP_HF_ID = "OpenGVLab/InternVideo2_distillation_models"
DEFAULT_INTERNVIDEO2CLIP_REVISION = (
    "449f7ea1d7d3b70b6b5630e70d238b44d3b7aaac"
)

# Known filenames inside the InternVideo2-CLIP HF repo / MobileCLIP release.
VISION_FILENAME = "stage1/L14/L14_dist_1B_stage2/pytorch_model.bin"
CLIP_HEAD_FILENAME = "clip/L14/pytorch_model.bin"
TEXT_FILENAME = "mobileclip_blt.pt"


class AttrDict(dict):
    """Recursive dict with attribute access for OpenGVLab configs."""

    def __init__(self, value: dict[str, Any] | None = None, **kwargs):
        """Initialize from dict/kwargs."""
        super().__init__()
        if value:
            self.update(value)
        if kwargs:
            self.update(kwargs)

    def __getattr__(self, name):
        """Read attribute as dict key."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        """Set attribute as dict key."""
        self[name] = self._wrap(value)

    def __setitem__(self, key, value):
        """Set key with recursive wrapping."""
        super().__setitem__(key, self._wrap(value))

    def update(self, value=None, **kwargs):  # type: ignore[override]
        """Update recursively."""
        if value:
            for key, item in value.items():
                self[key] = item
        for key, item in kwargs.items():
            self[key] = item

    @classmethod
    def _wrap(cls, value):
        """Wrap nested containers."""
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


def _get(cfg, key, default=None):
    """Read from dict/OmegaConf/dataclass-like config."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _require_file(path, label):
    """Validate a file path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def _hf_download(repo_id, filename, label):
    """Download (or resolve from the ambient HF_HOME cache) an HF file.

    Requests for the default InternVideo2-CLIP repo are pinned to
    ``DEFAULT_INTERNVIDEO2CLIP_REVISION``; a user-supplied repo id resolves at
    its own ``main`` because we cannot know a valid commit for it.
    """
    from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

    revision = (
        DEFAULT_INTERNVIDEO2CLIP_REVISION
        if repo_id == DEFAULT_INTERNVIDEO2CLIP_HF_ID
        else None
    )
    if revision:
        logger.info(
            "Resolving %s from %s at pinned revision %s",
            filename, repo_id, revision,
        )
    # No cache_dir/local_files_only: rely on ambient HF_HOME; cache-aware
    # (downloads only if missing). Token via the standard HF_TOKEN env if set.
    resolved = hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision,
    )
    return _require_file(resolved, label)


def _resolve_component(value, default_repo, filename, label):
    """Resolve one component to a local file path.

    ``value`` may be an existing local file (used directly) or an HF repo id
    (the component's known ``filename`` is pulled from it). When ``value`` is
    empty, fall back to ``default_repo`` (if given). Returns ``(path, source)``
    where ``source`` is a human-readable provenance string for logging.
    """
    if value and Path(value).exists():
        return _require_file(value, label), f"local:{value}"
    if value:
        # Treat as an HF repo id (a non-existent local path falls here and will
        # surface a clear HF error).
        return _hf_download(value, filename, label), f"hf:{value}/{filename}"
    if default_repo:
        return _hf_download(default_repo, filename, label), \
            f"hf:{default_repo}/{filename}"
    return None, None


def resolve_internvideo2_l14_assets(model_cfg):
    """Resolve InternVideo2-CLIP L14 checkpoint assets from the new fields.

    Fields (each a local file OR an HF repo id; HF uses the ambient HF_HOME cache):
    - ``vision_encoder`` / ``clip_head``: default to ``internvideo2clip_hf_id``.
    - ``text_encoder``: required (MobileCLIP is not in the IV2CLIP repo).
    Logs each component's source and the overwrite precedence.
    """
    default_repo = _get(model_cfg, "internvideo2clip_hf_id",
                        DEFAULT_INTERNVIDEO2CLIP_HF_ID)
    component_values = (
        default_repo,
        _get(model_cfg, "vision_encoder"),
        _get(model_cfg, "text_encoder"),
        _get(model_cfg, "clip_head"),
    )
    if not any(component_values):
        logger.info(
            "All InternVideo2-CLIP weight sources are null; constructing the "
            "architecture without loading pretrained assets."
        )
        return {
            "vision_ckpt": None,
            "text_ckpt": None,
            "extra_ckpt": None,
        }

    text_value = _get(model_cfg, "text_encoder")
    if not text_value:
        raise ValueError(
            "InternVideo2-CLIP L14 requires model.text_encoder (MobileCLIP "
            "local path or HF id) when model.pretrained_ckpt is not set."
        )

    vision, vision_src = _resolve_component(
        _get(model_cfg, "vision_encoder"), default_repo, VISION_FILENAME,
        "InternVideo2 vision encoder",
    )
    clip_head, clip_src = _resolve_component(
        _get(model_cfg, "clip_head"), default_repo, CLIP_HEAD_FILENAME,
        "InternVideo2 CLIP head",
    )
    text, text_src = _resolve_component(
        text_value, None, TEXT_FILENAME, "InternVideo2 text encoder",
    )

    logger.info("InternVideo2-CLIP component checkpoints resolved:")
    logger.info("  vision_encoder: %s", vision_src)
    logger.info("  text_encoder:   %s", text_src)
    logger.info("  clip_head:      %s", clip_src)
    logger.warning(
        "Weight precedence: clip_head is applied LAST and OVERWRITES any "
        "overlapping vision_encoder.*/text_encoder.* keys (clip_head wins on "
        "overlap). A model.pretrained_ckpt, if set, overrides all of the above."
    )
    return {"vision_ckpt": vision, "text_ckpt": text, "extra_ckpt": clip_head}


def build_internvideo2_l14_config(
    assets,
    num_frames=8,
    image_size=224,
    freeze_vision_encoder=False,
    freeze_text_encoder=False,
    use_flash_attn=False,
    use_fused_rmsnorm=False,
    use_fused_mlp=False,
):
    """Build the OpenGVLab config object for InternVideo2-CLIP L14."""
    return AttrDict({
        "model": {
            "model_cls": "InternVideo2_CLIP_small",
            "vision_encoder": {
                "name": "internvideo2",
                "in_chans": 3,
                "patch_size": 14,
                "img_size": image_size,
                "qkv_bias": False,
                "drop_path_rate": 0.0,
                "head_drop_path_rate": 0.0,
                "embed_dim": 1024,
                "num_heads": 16,
                "mlp_ratio": 4,
                "init_values": 0.1,
                "qk_normalization": True,
                "depth": 24,
                "use_flash_attn": use_flash_attn,
                "use_fused_rmsnorm": use_fused_rmsnorm,
                "use_fused_mlp": use_fused_mlp,
                "fused_mlp_heuristic": 1,
                "drop_cls_token": False,
                "attn_pool_num_heads": 16,
                "clip_embed_dim": 768,
                "layerscale_no_force_fp32": True,
                "num_frames": num_frames,
                "tubelet_size": 1,
                "sep_pos_embed": False,
                "use_checkpoint": False,
                "checkpoint_num": 0,
                "align_dim": 512,
            },
            "text_encoder": {"name": "mobileclip_b"},
            "temp": 1 / 100.0,
            "temp_min": 1 / 100.0,
            "freeze_vision": freeze_vision_encoder,
            "open_vision_clip_projector": True,
            "freeze_text": freeze_text_encoder,
            "open_text_projection": False,
            "vision_ckpt_path": assets["vision_ckpt"],
            "load_vision_ckpt_from_internvideo2_stage2": False,
            "text_ckpt_path": assets["text_ckpt"],
            "extra_ckpt_path": assets["extra_ckpt"],
        },
        "use_half_precision": True,
        "use_bf16": True,
    })
