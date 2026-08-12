# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional state-dict key remapping between timm DINOv3 and TAO ``ssl/dinov3``.

Single source of truth for the DINOv3 ViT key renames, so the two directions can never
drift apart:

* **timm -> TAO** (:func:`timm_to_tao`) — used by ``DinoV3PlModel`` to load public DINOv3
  weights into the SSL backbone at the start of continual pre-training.
* **TAO -> timm** (:func:`tao_to_timm`) — used by the ``dinov3 convert`` subtask to export an
  SSL-trained backbone into the timm layout that the ``cv/backbone_v2`` ``dinov3_vitb16``
  registry entry (and downstream supervised tasks) consume.

The two ViTs are numerically identical (verified by the feature-parity smoke test), so this
is purely a key-naming translation; no weights are transformed.
"""

import os

import timm
import torch

# Exact full-key renames, timm name -> TAO name.
_EXACT_TIMM_TO_TAO = {
    "reg_token": "register_tokens",
}
# Per-block suffix renames, timm suffix -> TAO suffix (LayerScale: raw gammas vs ls modules).
_SUFFIX_TIMM_TO_TAO = {
    ".gamma_1": ".ls1.gamma",
    ".gamma_2": ".ls2.gamma",
}
# Inverses.
_EXACT_TAO_TO_TIMM = {v: k for k, v in _EXACT_TIMM_TO_TAO.items()}
_SUFFIX_TAO_TO_TIMM = {v: k for k, v in _SUFFIX_TIMM_TO_TAO.items()}

# TAO-side params/buffers absent from a timm DINOv3 (inference) checkpoint.
# ``mask_token`` is an iBOT parameter; ``rope.periods`` is a non-persistent buffer (already
# excluded from state dicts, but guarded here for robustness).
TAO_ONLY_KEYS = ("mask_token",)

# TAO backbone arch -> timm DINOv3 model name (architecture only; pretrained=False).
_TIMM_MODEL_BY_ARCH = {
    "vit_s": "vit_small_patch16_dinov3",
    "vit_s_plus": "vit_small_plus_patch16_dinov3",
    "vit_b": "vit_base_patch16_dinov3",
    "vit_l": "vit_large_patch16_dinov3",
    "vit_h_plus": "vit_huge_plus_patch16_dinov3",
    "vit_7b": "vit_7b_patch16_dinov3",
}


def timm_model_name_for_arch(arch):
    """Return the timm DINOv3 model name for a TAO backbone arch (e.g. ``vit_b``).

    Args:
        arch (str): TAO backbone type (``vit_b`` / ``vit_l`` / ``vit_h_plus``).

    Returns:
        str: The corresponding timm model name.
    """
    if arch not in _TIMM_MODEL_BY_ARCH:
        raise ValueError(
            f"No timm DINOv3 model name known for arch '{arch}'. Known: {list(_TIMM_MODEL_BY_ARCH)}"
        )
    return _TIMM_MODEL_BY_ARCH[arch]


def timm_to_tao(key):
    """Translate a timm DINOv3 ViT parameter name to the TAO ``ssl/dinov3`` name.

    Args:
        key (str): timm-side parameter name.

    Returns:
        str: TAO-side parameter name (unchanged if no rule applies).
    """
    if key in _EXACT_TIMM_TO_TAO:
        return _EXACT_TIMM_TO_TAO[key]
    for src, dst in _SUFFIX_TIMM_TO_TAO.items():
        if key.endswith(src):
            return key[: -len(src)] + dst
    return key


def tao_to_timm(key):
    """Translate a TAO ``ssl/dinov3`` ViT parameter name to the timm DINOv3 name.

    Args:
        key (str): TAO-side parameter name.

    Returns:
        str: timm-side parameter name (unchanged if no rule applies).
    """
    if key in _EXACT_TAO_TO_TIMM:
        return _EXACT_TAO_TO_TIMM[key]
    for src, dst in _SUFFIX_TAO_TO_TIMM.items():
        if key.endswith(src):
            return key[: -len(src)] + dst
    return key


# ---------------------------------------------------------------------------------------
# HuggingFace ``DINOv3ViTModel`` -> TAO
#
# The HF export carries the *same weights* as the timm release -- verified bit-exact:
# ``cat([q_proj, k_proj, v_proj])`` equals timm's fused ``attn.qkv.weight`` with zero
# difference, as do ``patch_embed`` and ``cls_token``. Only the serialization differs, in two
# ways the timm remapper cannot express:
#
#   1. Different names throughout (``embeddings.*`` / ``layer.N.*`` rather than bare / ``blocks.N.*``).
#      Under the timm rules exactly two keys coincide -- ``norm.weight`` / ``norm.bias`` -- which
#      is why an HF checkpoint used to load 2 of 211 tensors and train from a near-random backbone.
#   2. Attention stored as **separate** q/k/v projections rather than a fused ``qkv``. That is a
#      tensor transform, not a rename, so it needs its own pass (cf. ``fuse_timm_swiglu_fc1``).
#
# Bonus: the HF export includes ``mask_token``, which the timm release omits (TAO otherwise
# initializes it randomly for iBOT).
# ---------------------------------------------------------------------------------------

# Exact full-key renames, HF name -> TAO name.
_EXACT_HF_TO_TAO = {
    "embeddings.cls_token": "cls_token",
    "embeddings.mask_token": "mask_token",
    "embeddings.register_tokens": "register_tokens",
    "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
    "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
}
# Per-block suffix renames, HF suffix -> TAO suffix (applied after ``layer.N`` -> ``blocks.N``).
_SUFFIX_HF_TO_TAO = {
    ".attention.o_proj": ".attn.proj",
    ".layer_scale1.lambda1": ".ls1.gamma",
    ".layer_scale2.lambda1": ".ls2.gamma",
    ".mlp.up_proj": ".mlp.fc1",
    ".mlp.down_proj": ".mlp.fc2",
}
# HF attention biases that TAO/timm deliberately do not have. DINOv3 ships these as all-zero
# (the config advertises query/value bias, but the tensors are zero), so dropping them is
# lossless -- ``convert_hf_to_tao`` asserts that rather than assuming it.
_HF_DROPPED_BIASES = (".attention.q_proj.bias", ".attention.k_proj.bias", ".attention.v_proj.bias")


def is_hf_dinov3_state_dict(state_dict):
    """Whether a state dict is a HuggingFace ``DINOv3ViTModel`` export.

    Args:
        state_dict (Mapping): Candidate state dict.

    Returns:
        bool: True for the HF layout (``embeddings.*`` / ``layer.N.*`` keys).
    """
    keys = list(state_dict)
    return any(k.startswith("embeddings.") for k in keys) and any(
        k.startswith("layer.") for k in keys
    )


def hf_to_tao(key):
    """Translate a HuggingFace DINOv3 parameter name to the TAO ``ssl/dinov3`` name.

    Does not handle the q/k/v fusion, which is not a rename; see :func:`convert_hf_to_tao`.

    Args:
        key (str): HF-side parameter name.

    Returns:
        str: TAO-side parameter name (unchanged if no rule applies).
    """
    if key in _EXACT_HF_TO_TAO:
        return _EXACT_HF_TO_TAO[key]
    if key.startswith("layer."):
        key = "blocks." + key[len("layer."):]
    for src, dst in _SUFFIX_HF_TO_TAO.items():
        if src in key:
            key = key.replace(src, dst)
            break
    return key


def convert_hf_to_tao(hf_state_dict, strict_zero_bias=True):
    """Convert a HuggingFace DINOv3 state dict into TAO ``ssl/dinov3`` naming and layout.

    Renames every key, fuses the separate ``q_proj``/``k_proj``/``v_proj`` weights into the
    single ``blocks.N.attn.qkv.weight`` that the TAO ViT expects, and drops the q/k/v biases
    that the TAO/timm architecture does not carry.

    Args:
        hf_state_dict (Mapping): State dict in HuggingFace ``DINOv3ViTModel`` layout.
        strict_zero_bias (bool): If True, assert the dropped q/k/v biases are all zero. DINOv3
            ships them zero, so a nonzero value means the checkpoint genuinely needs a
            bias-carrying attention and silently discarding it would corrupt the model.

    Returns:
        dict: State dict in TAO naming, loadable into ``DinoV3VisionTransformer``.

    Raises:
        ValueError: If ``strict_zero_bias`` and a dropped bias is nonzero, or if a block has an
            incomplete q/k/v triple.
    """
    out = {}
    block_ids = sorted(
        {int(k.split(".")[1]) for k in hf_state_dict if k.startswith("layer.")}
    )

    for block in block_ids:
        parts = []
        for proj in ("q_proj", "k_proj", "v_proj"):
            name = f"layer.{block}.attention.{proj}.weight"
            if name not in hf_state_dict:
                raise ValueError(
                    f"HF checkpoint is missing {name}; cannot fuse block {block}'s attention."
                )
            parts.append(hf_state_dict[name])
        out[f"blocks.{block}.attn.qkv.weight"] = torch.cat(parts, dim=0).contiguous()

    for key, weight in hf_state_dict.items():
        if ".attention.q_proj.weight" in key or ".attention.k_proj.weight" in key or \
                ".attention.v_proj.weight" in key:
            continue  # already fused above
        if any(dropped in key for dropped in _HF_DROPPED_BIASES):
            if strict_zero_bias and weight.abs().max().item() > 0:
                raise ValueError(
                    f"HF checkpoint has a nonzero {key} but the TAO/timm DINOv3 attention has no "
                    "qkv bias. Dropping it would change the model; refusing to convert silently."
                )
            continue
        out[hf_to_tao(key)] = weight.contiguous()

    return out


def fuse_timm_swiglu_fc1(timm_state_dict):
    """Fuse timm's split SwiGLU projections into the ``GluMlp`` fused ``fc1``.

    This is timm -> TAO path. Concatenates ``[fc1_g, fc1_x]`` into ``fc1`` for
    both weight and bias. Plain-MLP checkpoints (ViT-B/L) pass through
    unchanged.

    Args:
        timm_state_dict (dict): Source timm DINOv3 state dict.

    Returns:
        dict: A new state dict with split SwiGLU projections fused into ``fc1``.
    """
    fused = dict(timm_state_dict)
    gate_weight_keys = [k for k in timm_state_dict if k.endswith("mlp.fc1_g.weight")]
    for gate_weight_key in gate_weight_keys:
        prefix = gate_weight_key[: -len("fc1_g.weight")]  # e.g. 'blocks.0.mlp.'
        for suffix in ("weight", "bias"):
            g_key = f"{prefix}fc1_g.{suffix}"
            x_key = f"{prefix}fc1_x.{suffix}"
            if g_key in fused and x_key in fused:
                fused[f"{prefix}fc1.{suffix}"] = torch.cat(
                    [fused.pop(g_key), fused.pop(x_key)], dim=0
                ).contiguous()
    return fused


def split_fused_swiglu_fc1(timm_state_dict, reference):
    """Split a fused ``GluMlp`` back into timm's ``mlp.fc1_g``/``mlp.fc1_x``.

    This is TAO -> timm path. Splits each fused ``mlp.fc1`` into ``fc1_g`` and
    ``fc1_x``. Applied only when the target ``reference`` actually uses split
    projections (ViT-H+/7B); plain-MLP targets (ViT-B/L) keep ``fc1`` unchanged.

    Args:
        timm_state_dict (dict): State dict in timm naming, possibly with a fused
            ``mlp.fc1``.
        reference (Mapping): Target timm model state dict (or any key iterable);
            its keys decide whether to split.

    Returns:
        dict: State dict with fused ``mlp.fc1`` split into
            ``mlp.fc1_g``/``mlp.fc1_x`` when the target is SwiGLU; otherwise the
            input unchanged.
    """
    if not any(k.endswith("mlp.fc1_g.weight") for k in reference):
        return timm_state_dict
    out = dict(timm_state_dict)
    fused_weight_keys = [k for k in timm_state_dict if k.endswith("mlp.fc1.weight")]
    for fused_weight_key in fused_weight_keys:
        prefix = fused_weight_key[: -len("fc1.weight")]  # e.g. 'blocks.0.mlp.'
        for suffix in ("weight", "bias"):
            fused = out.pop(f"{prefix}fc1.{suffix}", None)
            if fused is None:
                continue
            half = fused.shape[0] // 2
            out[f"{prefix}fc1_g.{suffix}"] = fused[:half].contiguous()
            out[f"{prefix}fc1_x.{suffix}"] = fused[half:].contiguous()
    return out


def merge_lora_state_dict(state_dict):
    """Fold any LoRA adapters in a backbone state dict into their base weights.

    Turns ``{W, lora_A, lora_B, lora_scaling}`` into ``{W + scaling * (B @ A)}`` and drops the
    adapter keys, so the result is a stock DINOv3 state dict that a plain ViT (or timm) loads
    strictly. Without this step ``convert`` either fails validation on the unexpected ``lora_*``
    keys or -- with ``validate=False`` -- writes them straight through, and ``export`` builds a
    plain backbone that ignores them: in both cases the artifact is the *frozen base* model, not
    the adapted one, with nothing in the output to say so.

    The scale is read from the ``lora_scaling`` buffer stored alongside each adapter, so a
    checkpoint carries everything needed to merge it and cannot be folded with the wrong alpha.

    A state dict without adapters passes through untouched, so this is safe to call
    unconditionally.

    Args:
        state_dict (Mapping): Backbone-level state dict in TAO naming, with or without adapters.

    Returns:
        dict: State dict with adapters folded in and all ``lora_*`` keys removed.

    Raises:
        ValueError: If an adapter is incomplete, or carries no scale to merge with.
    """
    merged = dict(state_dict)
    prefixes = sorted({k[: -len("lora_A")] for k in state_dict if k.endswith("lora_A")})

    for prefix in prefixes:
        a_key, b_key = f"{prefix}lora_A", f"{prefix}lora_B"
        scale_key, weight_key = f"{prefix}lora_scaling", f"{prefix}weight"
        if b_key not in merged:
            raise ValueError(f"{a_key} has no matching {b_key}; refusing to merge a partial adapter.")
        if weight_key not in merged:
            raise ValueError(f"{a_key} has no base weight at {weight_key}.")
        if scale_key not in merged:
            raise ValueError(
                f"{a_key} has no {scale_key}; the alpha/rank scale is unknown, and guessing it "
                "would silently produce a wrongly-scaled backbone."
            )

        lora_a = merged.pop(a_key)
        lora_b = merged.pop(b_key)
        scaling = merged.pop(scale_key).to(torch.float32).item()
        base = merged[weight_key]
        delta = (lora_b.to(torch.float32) @ lora_a.to(torch.float32)) * scaling
        merged[weight_key] = (base.to(torch.float32) + delta).to(base.dtype).contiguous()

    # Defensive: nothing lora-shaped may survive into a "stock" state dict.
    leftover = [k for k in merged if ".lora_" in k or k.endswith(("lora_A", "lora_B"))]
    if leftover:
        raise ValueError(f"LoRA keys survived the merge: {sorted(leftover)}")
    return merged


def load_checkpoint_file(path):
    """Load a ``.safetensors`` / ``.pth`` / ``.ckpt`` checkpoint into a dict.

    Args:
        path (str): Path to the checkpoint file.

    Returns:
        dict: The loaded object (a state dict, or a container with ``state_dict``).
    """
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu")


def save_state_dict(state_dict, path):
    """Save a flat state dict as ``.safetensors`` (by extension) or ``.pth``.

    Args:
        state_dict (dict): Parameter-name -> tensor mapping.
        path (str): Output path; ``.safetensors`` extension selects the safetensors writer.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.detach().contiguous().cpu() for k, v in state_dict.items()}, path)
    else:
        torch.save(state_dict, path)


def is_full_checkpoint(raw):
    """Return whether a checkpoint contains named backbone branches.

    Args:
        raw (dict): Loaded checkpoint, optionally wrapped in a ``state_dict`` container.

    Returns:
        bool: True for full SSL checkpoints containing ``<source>.backbone.*`` keys.
    """
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    return any(".backbone." in key for key in state_dict)


def extract_backbone_state_dict(raw, source="teacher"):
    """Normalize any DINOv3 SSL checkpoint into a backbone-level state dict (TAO naming).

    Handles two input shapes:

    * a **stripped backbone file** (``student_*.pth`` / ``teacher_*.pth`` from
      ``CustomModelCheckpoint``) whose keys are already backbone-level, or
    * a **full Lightning checkpoint** with ``<source>.backbone.*`` (and head / gram-teacher)
      keys, from which the chosen source's backbone is selected and de-prefixed.

    Args:
        raw (dict): Loaded checkpoint (a state dict, or a container with ``state_dict``).
        source (str): Which sub-model's backbone to extract from a full checkpoint
            (``student`` / ``teacher`` / ``student_ema``). Ignored for stripped files.

    Returns:
        dict: Backbone-level state dict in TAO naming.
    """
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    if is_full_checkpoint(state_dict):
        prefix = f"{source}.backbone."
        extracted = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if not extracted:
            available = sorted({k.split(".backbone.")[0] for k in state_dict if ".backbone." in k})
            raise ValueError(
                f"No keys with prefix '{prefix}' in checkpoint. Available backbone sources: {available}"
            )
        return extracted

    # Already a stripped backbone file; drop any stray (non-backbone) head keys defensively.
    return {k: v for k, v in state_dict.items() if not k.startswith(("dino_head", "ibot_head"))}


def remap_tao_backbone_to_timm(tao_state_dict):
    """Rename a TAO backbone state dict into timm DINOv3 layout, dropping TAO-only keys.

    Args:
        tao_state_dict (dict): Backbone-level state dict in TAO naming.

    Returns:
        dict: State dict in timm naming (contiguous tensors).
    """
    out = {}
    for key, weight in tao_state_dict.items():
        if key in TAO_ONLY_KEYS or key.endswith("rope.periods"):
            continue
        out[tao_to_timm(key)] = weight.contiguous()
    return out


def validate_against_timm(timm_state_dict, timm_model_name="vit_base_patch16_dinov3", reference=None):
    """Assert a state dict matches a fresh timm DINOv3 model's keys and shapes.

    Args:
        timm_state_dict (dict): Candidate state dict in timm naming.
        timm_model_name (str): timm model whose architecture defines the expected keys.
        reference (dict, optional): Pre-built reference state dict (e.g. shared with
            :func:`split_fused_swiglu_fc1`). When ``None``, a fresh timm model is created;
            passing it avoids building large models (e.g. ViT-7B) twice.

    Raises:
        ValueError: If any key is missing, unexpected, or shape-mismatched.
    """
    if reference is None:
        reference = timm.create_model(timm_model_name, pretrained=False).state_dict()
    missing = sorted(set(reference) - set(timm_state_dict))
    unexpected = sorted(set(timm_state_dict) - set(reference))
    shape_mismatch = [
        k for k in timm_state_dict
        if k in reference and tuple(reference[k].shape) != tuple(timm_state_dict[k].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "Converted state dict is not timm-compatible:\n"
            f"  missing={missing}\n  unexpected={unexpected}\n  shape_mismatch={shape_mismatch}"
        )


def convert_ssl_to_timm(src_path, dst_path, source="teacher", validate=True,
                        timm_model_name="vit_base_patch16_dinov3"):
    """Convert a TAO DINOv3 SSL checkpoint into a timm-format backbone file.

    The output is loadable by ``timm.create_model(..., checkpoint_path=dst_path)`` and hence by
    the ``cv/backbone_v2`` ``dinov3_vitb16`` registry entry for downstream supervised tasks.

    Args:
        src_path (str): SSL checkpoint (stripped backbone ``.pth`` or full Lightning ``.ckpt``).
        dst_path (str): Output path (``.safetensors`` or ``.pth``).
        source (str): Which backbone to export (``teacher`` recommended — the EMA teacher).
        validate (bool): If True, validate the result against a fresh timm DINOv3 model.
        timm_model_name (str): timm model name for validation / target layout.

    Returns:
        dict: The converted timm-format state dict.
    """
    raw = load_checkpoint_file(src_path)
    backbone_state_dict = extract_backbone_state_dict(raw, source=source)
    # Fold LoRA adapters into the base weights before translation, so the exported backbone is
    # the *adapted* model in stock DINOv3 topology. No-op for full-fine-tune checkpoints.
    backbone_state_dict = merge_lora_state_dict(backbone_state_dict)
    timm_state_dict = remap_tao_backbone_to_timm(backbone_state_dict)
    reference = timm.create_model(timm_model_name, pretrained=False).state_dict()
    timm_state_dict = split_fused_swiglu_fc1(timm_state_dict, reference)
    if validate:
        validate_against_timm(
            timm_state_dict,
            timm_model_name=timm_model_name,
            reference=reference,
        )
    save_state_dict(timm_state_dict, dst_path)
    return timm_state_dict
