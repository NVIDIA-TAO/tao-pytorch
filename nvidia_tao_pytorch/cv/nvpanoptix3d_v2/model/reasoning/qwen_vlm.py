# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL-4B wrapper for the embedding-as-mask reasoning pipeline.

The wrapper adds a ``[SEG]`` token, freezes the vision tower, optionally applies
LoRA, and exposes the last-layer ``[SEG]`` hidden states for the SAM3 prompt
projector.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_4B_HIDDEN_SIZE = 2560
ASSISTANT_PREFIX = "<|im_start|>assistant\n"
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
LOCAL_MODEL_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json")


@dataclass
class Qwen3VLReasonerOutput:
    """Stable output contract consumed by the rest of NVPanoptix3Dv2-Reasoning.

    Attributes:
        lm_logits:    ``[B, T, V]`` autoregressive token logits (for text CE).
        hidden:       ``[B, T, d_llm]`` last-layer hidden states.
        input_ids:    ``[B, T]`` token ids (used to locate ``[SEG]`` positions).
        attention_mask: ``[B, T]`` padding mask.
        labels:       ``[B, T]`` text-CE targets (prompt tokens set to ``-100``),
                      or ``None`` at inference.
    """

    lm_logits: Tensor
    hidden: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    labels: Optional[Tensor] = None


def extract_seg_hidden(
    hidden: Tensor,
    input_ids: Tensor,
    seg_token_id: int,
) -> Tuple[Tensor, Tensor]:
    """Gather last-layer hidden states at every ``[SEG]`` position.

    Pure function so it is testable without a model. Works with a variable number
    of ``[SEG]`` tokens per sample (0, 1, or many): the returned ``batch_index``
    maps each gathered row back to its sample, which the SAM3 prompt projector
    uses to scatter rows into per-sample prompts.

    Args:
        hidden:    ``[B, T, d]`` last-layer hidden states.
        input_ids: ``[B, T]`` token ids.
        seg_token_id: vocabulary id of the ``[SEG]`` token.

    Returns:
        ``(h_seg, batch_index)`` where ``h_seg`` is ``[N, d]`` (N = total number of
        ``[SEG]`` tokens across the batch) and ``batch_index`` is ``[N]`` (long).
        Both are empty (N=0) when no ``[SEG]`` token is present.
    """
    if hidden.dim() != 3:
        raise ValueError(f"hidden must be [B, T, d]; got shape {tuple(hidden.shape)}")
    if input_ids.shape != hidden.shape[:2]:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} must match hidden[:2] "
            f"{tuple(hidden.shape[:2])}"
        )
    seg_pos = input_ids == seg_token_id          # [B, T] bool
    batch_index, _ = torch.where(seg_pos)        # [N], [N]
    h_seg = hidden[seg_pos]                       # [N, d]
    return h_seg, batch_index.long()


def bind_seg_to_views(
    batch_index: Tensor,
    seg_view_indices: Sequence[Sequence[int]],
    num_views: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Map each ``[SEG]`` row to a ``(sample, view)`` pair.

    :func:`extract_seg_hidden` returns rows row-major over ``(sample, token-pos)``, so within a
    sample the ``[SEG]`` rows are already in answer (ascending picture) order. The j-th ``[SEG]``
    of sample ``b`` therefore maps to ``seg_view_indices[b][j]``.

    Args:
        batch_index: ``[M]`` long, sample id per ``[SEG]`` row (from :func:`extract_seg_hidden`).
        seg_view_indices: per-sample target views (ascending), length ``B``.
        num_views: number of views ``S`` actually present in the batch tensors.

    Returns:
        ``(sample_idx[N], view_idx[N], keep[N])`` for the ``N`` rows that bind to a valid view.
        Surplus ``[SEG]`` tokens (count > available views) and out-of-range views are dropped
        (with a one-time warning), so ``N`` may be < ``M``.
    """
    device = batch_index.device
    sample_rows: List[int] = []
    view_rows: List[int] = []
    keep_rows: List[int] = []
    seen: Dict[int, int] = {}
    dropped = 0
    for row, b_raw in enumerate(batch_index.tolist()):
        b = int(b_raw)
        j = seen.get(b, 0)
        seen[b] = j + 1
        sv = seg_view_indices[b] if 0 <= b < len(seg_view_indices) else []
        if j < len(sv) and 0 <= int(sv[j]) < int(num_views):
            sample_rows.append(b)
            view_rows.append(int(sv[j]))
            keep_rows.append(row)
        else:
            dropped += 1
    if dropped:
        warnings.warn(
            f"bind_seg_to_views: dropped {dropped} [SEG] token(s) with no matching/in-range "
            f"view (num_views={num_views}).",
            RuntimeWarning,
        )
    return (
        torch.tensor(sample_rows, dtype=torch.long, device=device),
        torch.tensor(view_rows, dtype=torch.long, device=device),
        torch.tensor(keep_rows, dtype=torch.long, device=device),
    )


def find_answer_start(input_ids: Tensor, prefix_ids: Sequence[int]) -> int:
    """Return the position immediately after the last assistant prefix."""
    ids = input_ids.tolist()
    prefix = list(prefix_ids)
    for start in range(len(ids) - len(prefix), -1, -1):
        if ids[start:start + len(prefix)] == prefix:
            return start + len(prefix)
    raise ValueError("Qwen3-VL input does not contain an assistant prefix.")


def resolve_model_source(model_id: str) -> Tuple[str, bool]:
    """Resolve a Hub model ID or validate a local model directory."""
    model_id = str(model_id)
    path = Path(model_id).expanduser()
    if path.is_dir():
        missing = [name for name in LOCAL_MODEL_FILES if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Qwen3-VL model directory {str(path)!r} is missing: {', '.join(missing)}"
            )
        return str(path), True

    if path.is_absolute() or model_id.startswith((".", "~")):
        raise FileNotFoundError(
            f"Qwen3-VL model directory not found: {str(path)!r}. "
            "Check the container mount and model.reasoning.qwen.model_id."
        )

    return model_id, False


class Qwen3VLReasoner(nn.Module):
    """Qwen3-VL-4B with segmentation tokens and optional LoRA."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        seg_token: str = "[SEG]",
        freeze_vision: bool = True,
        # fp32 + sdpa is the safe runnable default (no autocast/flash-attn deps);
        # switch to bf16 + flash_attention_2 once the pipeline runs, for memory.
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "sdpa",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: Sequence[str] = DEFAULT_LORA_TARGETS,
        use_lora: bool = True,
    ):
        super().__init__()
        self.processor, self.model = self.load_backbone(
            model_id, dtype, attn_implementation,
        )
        self.tokenizer = self.processor.tokenizer
        self._assistant_prefix_ids = self.tokenizer(
            ASSISTANT_PREFIX,
            add_special_tokens=False,
        )["input_ids"]

        # Extend the vocabulary with the segmentation token.
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [seg_token]}
        )
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self._seg_token_id = self.tokenizer.convert_tokens_to_ids(seg_token)

        if freeze_vision:
            self.freeze_vision_tower()

        if use_lora:
            self.apply_lora(lora_r, lora_alpha, lora_dropout, list(lora_targets))

    @staticmethod
    def load_backbone(model_id, dtype, attn_implementation):
        """Load and validate the native Qwen3-VL-4B model."""
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Qwen3-VL requires `transformers>=4.57.0,<5.0.0`."
            ) from e

        model_source, is_local = resolve_model_source(model_id)
        load_kwargs = {"local_files_only": True} if is_local else {}
        processor = Qwen3VLProcessor.from_pretrained(model_source, **load_kwargs)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_source,
            dtype=dtype,
            attn_implementation=attn_implementation,
            **load_kwargs,
        )
        hidden_size = int(model.config.text_config.hidden_size)
        if hidden_size != QWEN3_VL_4B_HIDDEN_SIZE:
            raise ValueError(
                "NVPanoptix3Dv2 requires Qwen3-VL-4B "
                f"(hidden_size={QWEN3_VL_4B_HIDDEN_SIZE}), but {model_id!r} "
                f"has hidden_size={hidden_size}."
            )

        return processor, model

    def freeze_vision_tower(self) -> None:
        """Freeze the Qwen3-VL visual encoder."""
        for param in self.model.visual.parameters():
            param.requires_grad = False

    def apply_lora(self, r: int, alpha: int, dropout: float, targets: List[str]) -> None:
        """Attach trainable LoRA adapters to the configured Qwen modules."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "peft is required for LoRA fine-tuning of Qwen3-VL-4B. "
                "Install with `pip install peft`."
            ) from e

        lora_cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            # Keep the (resized) token embeddings + LM head fully trainable so the
            # the new [SEG] row can learn; mirrors LISA's embed_tokens/lm_head.
            modules_to_save=["embed_tokens", "lm_head"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_cfg)

    @property
    def hidden_size(self) -> int:
        """Qwen3-VL-4B language hidden width."""
        return QWEN3_VL_4B_HIDDEN_SIZE

    def build_inputs(
        self,
        images: Sequence[Sequence[object]],
        instructions: Sequence[str],
        answers: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Tensor]:
        """Build processor inputs for a batch of (multi-image, instruction[, answer]).

        Args:
            images:       per-sample list of PIL images (the ``S`` views).
            instructions: per-sample query string.
            answers:      per-sample assistant answer (must contain ``[SEG]``);
                          when provided, ``labels`` are built (teacher forcing).
            device:       move tensors here.

        Returns:
            dict with ``input_ids``, ``attention_mask``, the model's image inputs
            (e.g. ``pixel_values`` / ``image_grid_thw``), and ``labels`` (if
            ``answers`` is given). Prompt tokens in ``labels`` are ``-100``.
        """
        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")
        if answers is not None and len(answers) != len(instructions):
            raise ValueError("answers and instructions must have the same batch size")

        def content(imgs, text):
            return [{"type": "image", "image": im} for im in imgs] + [
                {"type": "text", "text": text}
            ]

        messages = []
        for i, (imgs, instr) in enumerate(zip(images, instructions)):
            user = {"role": "user", "content": content(imgs, instr)}
            conversation = [user]
            if answers is not None:
                conversation.append({"role": "assistant", "content": answers[i]})
            messages.append(conversation)

        # add_vision_id=True prepends a per-image identifier ("Picture i:") before
        # each image, so the multi-view answer can reference "Picture i" and bind
        # each [SEG] to its image (matches the reasonseg_mv data convention).
        texts = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=answers is None,
                add_vision_id=True,
            )
            for message in messages
        ]
        flat_images = [list(imgs) for imgs in images]
        enc = self.processor(
            text=texts,
            images=flat_images,
            return_tensors="pt",
            padding=True,
        )
        if answers is not None:
            labels = enc["input_ids"].clone()
            for row, input_ids in enumerate(enc["input_ids"]):
                answer_start = find_answer_start(input_ids, self._assistant_prefix_ids)
                labels[row, :answer_start] = -100
            labels[enc["attention_mask"] == 0] = -100
            enc["labels"] = labels
        return self.to_device(enc, device)

    @staticmethod
    def to_device(enc, device):
        """Move tensor values in a processor encoding to ``device``."""
        if device is None:
            return dict(enc)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

    def forward(self, inputs: Dict[str, Tensor]) -> Qwen3VLReasonerOutput:
        """Run the VLM and return logits + last-layer hidden states.

        ``inputs`` is the dict from :meth:`build_inputs`. ``labels`` are carried
        through to the output but the text-CE loss is computed in
        the segmentation loss (so weighting/diagnostics stay in one place).
        """
        labels = inputs.get("labels", None)
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        out = self.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        return Qwen3VLReasonerOutput(
            lm_logits=out.logits,
            hidden=hidden,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )

    def extract_seg_hidden(self, out: Qwen3VLReasonerOutput) -> Tuple[Tensor, Tensor]:
        """Convenience wrapper around :func:`extract_seg_hidden`."""
        return extract_seg_hidden(out.hidden, out.input_ids, self._seg_token_id)
