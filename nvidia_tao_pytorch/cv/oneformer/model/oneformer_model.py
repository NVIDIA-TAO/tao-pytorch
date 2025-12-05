# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""OneFormer model implementation for unified panoptic segmentation."""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from nvidia_tao_pytorch.cv.oneformer.model.backbone.swin import D2SwinTransformer
from nvidia_tao_pytorch.cv.oneformer.model.transformer_decoder.text_transformer import (
    TextTransformer,
)
from nvidia_tao_pytorch.cv.oneformer.model.transformer_decoder.oneformer_transformer_decoder import (
    MLP,
)
from nvidia_tao_pytorch.cv.oneformer.model.tokenizer import Tokenize


class Postprocessor(nn.Module):
    """OneFormer postprocessor."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.test_topk_per_image = cfg.model.test.test_topk_per_image
        self.num_classes = cfg.model.sem_seg_head.num_classes
        self.num_queries = cfg.model.one_former.num_object_queries
        self.object_mask_threshold = cfg.model.test.object_mask_threshold
        self.overlap_threshold = cfg.model.test.overlap_threshold

        # Task settings
        self.semantic_on = cfg.model.test.semantic_on
        self.instance_on = cfg.model.test.instance_on
        self.panoptic_on = cfg.model.test.panoptic_on
        self.detection_on = cfg.model.test.detection_on

    def semantic_inference(self, mask_cls, mask_pred):
        """Semantic segmentation inference."""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred, metadata):
        """Panoptic segmentation inference."""
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = (
                    pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                )
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        """Instance segmentation inference."""
        mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.num_classes, device=mask_cls.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(
            self.test_topk_per_image, sorted=False
        )
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        mask_pred = mask_pred[topk_indices]

        # Keep scores above threshold
        keep = scores_per_image > self.object_mask_threshold
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        mask_pred = mask_pred[keep]

        pred_masks = (mask_pred > 0).float()
        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)
        ).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
        final_scores = scores_per_image * mask_scores_per_image

        return pred_masks, final_scores, labels_per_image

    def batch_semantic_inference(self, mask_cls, mask_pred):
        """Batch semantic segmentation inference."""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return semseg

    def forward(self, outputs, metadata=None):
        """Forward pass."""
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # Process each image in batch
        processed_results = []

        for mask_cls_result, mask_pred_result in zip(
            mask_cls_results, mask_pred_results
        ):
            result = {}

            if self.semantic_on:
                semantic_result = self.semantic_inference(
                    mask_cls_result, mask_pred_result
                )
                result["sem_seg"] = semantic_result

            if self.panoptic_on and metadata is not None:
                panoptic_result = self.panoptic_inference(
                    mask_cls_result, mask_pred_result, metadata
                )
                result["panoptic_seg"] = panoptic_result

            if self.instance_on:
                instance_result = self.instance_inference(
                    mask_cls_result, mask_pred_result
                )
                result["instances"] = instance_result

            processed_results.append(result)

        return processed_results


class OneFormerHead(nn.Module):
    """OneFormer Head."""

    def __init__(self, cfg, input_shape):
        super().__init__()
        from .pixel_decoder.fpn import build_pixel_decoder
        from .transformer_decoder.oneformer_transformer_decoder import (
            build_transformer_decoder,
        )

        self.pixel_decoder = build_pixel_decoder(cfg, input_shape)

        # Figure out transformer predictor input channels
        if cfg.model.one_former.transformer_in_feature == "transformer_encoder":
            transformer_predictor_in_channels = cfg.model.sem_seg_head.convs_dim
        elif cfg.model.one_former.transformer_in_feature == "pixel_embedding":
            transformer_predictor_in_channels = cfg.model.sem_seg_head.mask_dim
        elif cfg.model.one_former.transformer_in_feature == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.model.sem_seg_head.convs_dim
        else:
            transformer_predictor_in_channels = input_shape[
                cfg.model.one_former.transformer_in_feature
            ].channels

        self.predictor = build_transformer_decoder(
            cfg, transformer_predictor_in_channels, mask_classification=True
        )
        self.transformer_in_feature = cfg.model.one_former.transformer_in_feature
        self.num_classes = cfg.model.sem_seg_head.num_classes

    def forward(self, features, tasks, mask=None):
        """Forward pass."""
        mask_features, transformer_encoder_features, multi_scale_features = (
            self.pixel_decoder.forward_features(features)
        )

        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(
                multi_scale_features, mask_features, tasks, mask
            )
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(
                    transformer_encoder_features, mask_features, tasks, mask
                )
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, tasks, mask)
            else:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features, tasks, mask
                )
        return predictions


class OneFormerModel(nn.Module):
    """OneFormer Model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = self.build_backbone(cfg)
        self.sem_seg_head = OneFormerHead(cfg, self.backbone_feature_shape)

        # Task processing
        self.task_mlp = MLP(
            cfg.dataset.task_seq_len,
            cfg.model.one_former.hidden_dim,
            cfg.model.one_former.hidden_dim,
            2,
        )

        # Text processing (for inference only)
        if (
            hasattr(cfg.model, "text_encoder") and
            cfg.model.text_encoder.context_length > 0
        ):
            self.text_encoder = TextTransformer(
                context_length=cfg.model.text_encoder.context_length,
                width=cfg.model.text_encoder.width,
                layers=cfg.model.text_encoder.num_layers,
                vocab_size=cfg.model.text_encoder.vocab_size,
            )
            self.text_projector = MLP(
                self.text_encoder.width,
                cfg.model.one_former.hidden_dim,
                cfg.model.one_former.hidden_dim,
                cfg.model.text_encoder.proj_num_layers,
            )
            if cfg.model.text_encoder.n_ctx > 0:
                self.prompt_ctx = nn.Embedding(
                    cfg.model.text_encoder.n_ctx, cfg.model.text_encoder.width
                )
            else:
                self.prompt_ctx = None
        else:
            self.text_encoder = None
            self.text_projector = None
            self.prompt_ctx = None

        # Tokenizers
        self.text_tokenizer = Tokenize(max_seq_len=cfg.dataset.max_seq_len)
        self.task_tokenizer = Tokenize(max_seq_len=cfg.dataset.task_seq_len)

        # Postprocessor
        use_post_processor = getattr(cfg.model, "export", False)
        self.post_processor = Postprocessor(cfg) if use_post_processor else None

        # Pixel normalization
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1), False
        )

    def build_backbone(self, cfg):
        """Build backbone."""
        backbone_type = cfg.model.backbone.name

        if backbone_type == "D2SwinTransformer":
            backbone = D2SwinTransformer(cfg, input_shape=None)
            self.backbone_feature_shape = backbone.output_shape()
        # elif backbone_type == "D2RADIO":
        #     backbone = D2RADIO(cfg, input_shape=None)
        #     self.backbone_feature_shape = backbone.output_shape()
        else:
            raise NotImplementedError(f"Backbone {backbone_type} not supported!")

        return backbone

    def encode_text(self, text):
        """Encode text input."""
        if self.text_encoder is None:
            return {"texts": None}

        assert text.ndim in [2, 3], text.ndim
        text.shape[0]
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, "b n l -> (b n) l", n=num_text)
            squeeze_dim = True

        x = self.text_encoder(text)
        text_x = self.text_projector(x)
        if squeeze_dim:
            text_x = rearrange(text_x, "(b n) c -> b n c", n=num_text)
            if self.prompt_ctx is not None:
                text_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(
                    text_x.shape[0], 1, 1
                )
                text_x = torch.cat([text_x, text_ctx], dim=1)

        return {"texts": text_x}

    def forward(self, inputs, tasks, texts=None):
        """Forward pass."""
        # Process tasks
        if tasks is not None:
            if isinstance(tasks, list):
                tasks = torch.stack(
                    [self.task_tokenizer(task).float() for task in tasks]
                )
            # Move tasks to the same device as the model
            tasks = tasks.to(self.pixel_mean.device)
            tasks = self.task_mlp(tasks)

        # Process texts (if provided)
        if texts is not None and self.text_encoder is not None:
            if isinstance(texts, list):
                texts = torch.stack([self.text_tokenizer(text) for text in texts])

            texts = texts.to(self.pixel_mean.device)

            text_features = self.encode_text(texts)
        else:
            text_features = None

        # Backbone forward
        features = self.backbone(inputs)

        # Head forward
        outputs = self.sem_seg_head(features, tasks)
        if text_features is not None:
            outputs.update(text_features)

        return outputs
