# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen SigLIP text encoder for open-vocabulary panoptic segmentation."""

import os

import torch
from torch import nn
from transformers import AutoTokenizer, SiglipTextModel


MODEL_ID = "google/siglip-base-patch16-224"
EMBED_DIM = 768
PROMPT_TEMPLATE = "This is a photo of {}."


class TextEncoder(nn.Module):
    """Produce L2-normalized SigLIP embeddings for class names.

    Args:
        fixed_vocab: When ``True``, ``set_vocab`` must be called once
            up-front to pre-cache embeddings; subsequent ``forward(classes)``
            calls become dictionary look-ups (no SigLIP forward pass per
            training step). When ``False``, ``forward(classes)`` re-encodes
            the text every call -- useful for variable-vocabulary inference.
    """

    def __init__(self, fixed_vocab: bool = False):
        super().__init__()
        self.embed_dim = EMBED_DIM
        self.fixed_vocab = fixed_vocab
        if self.fixed_vocab:
            self.class_embeddings = {}
        else:
            self.model, self.tokenizer = self.get_model()

    def get_model(self):
        """Load the frozen SigLIP text model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = SiglipTextModel.from_pretrained(MODEL_ID)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        return model, tokenizer

    def encode_batch(self, model, tokenizer, texts, bs=256):
        """Encode a flat list of text strings into pooled embeddings."""
        embs = []
        total = len(texts)
        with torch.no_grad():
            for i in range(0, total, bs):
                inputs = tokenizer(
                    texts[i:i + bs], return_tensors='pt',
                    padding='max_length',
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                lang_emb = model(**inputs).pooler_output.detach()
                embs.append(lang_emb)
        return torch.cat(embs)

    def embed_classes(self, model, tokenizer, classes, bs=256):
        """Encode class names with the SigLIP prompt template."""
        texts = [PROMPT_TEMPLATE.format(c) for c in classes]
        return self.encode_batch(model, tokenizer, texts, bs=bs)

    def set_vocab(self, classes: list[str], device=None):
        """Pre-encode and cache embeddings keyed by exact text."""
        model, tokenizer = self.get_model()

        # Prefer CUDA even when the pre-Trainer caller reports CPU, since this
        # is a one-shot startup encode. Respect an explicit CUDA device for
        # inference; otherwise select torchrun's local rank.
        if torch.cuda.is_available():
            requested = torch.device(device) if device is not None else None
            if requested is not None and requested.type == "cuda":
                target_device = requested
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                target_device = torch.device("cuda", local_rank)
        elif device is not None:
            target_device = torch.device(device)
        else:
            target_device = torch.device("cpu")
        model.to(target_device)

        lang_emb = self.embed_classes(model, tokenizer, classes)
        self.class_embeddings = {c: emb for c, emb in zip(classes, lang_emb)}

        del model, tokenizer
        torch.cuda.empty_cache()

    def forward(self, classes: list[str]):
        """Return the language embeddings for a list of class names."""
        if self.fixed_vocab:
            missing = [
                class_name
                for class_name in classes
                if class_name not in self.class_embeddings
            ]
            if missing:
                raise ValueError(
                    "Missing classes in fixed vocabulary: "
                    f"{', '.join(missing)}. Call set_vocab() before forward."
                )
            lang_emb = torch.stack([self.class_embeddings[c] for c in classes])
        else:
            lang_emb = self.embed_classes(self.model, self.tokenizer, classes)

        out = lang_emb / lang_emb.norm(dim=-1, keepdim=True)

        return out
