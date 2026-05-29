# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Grounding DINO model. """

import copy
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import tensor_from_tensor_list, inverse_sigmoid
from nvidia_tao_pytorch.cv.dino.model.model_utils import MLP
from nvidia_tao_pytorch.cv.dino.model.dn_components import prepare_for_cdn, dn_post_process

from nvidia_tao_pytorch.cv.grounding_dino.utils import get_tokenlizer
from nvidia_tao_pytorch.cv.grounding_dino.model.bertwraper import BertModelWraper
from nvidia_tao_pytorch.cv.grounding_dino.model.model_utils import ContrastiveEmbed


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        position_embedding,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        dn_number=0,
        dn_box_noise_scale=1.0,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=256,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        log_scale=None,
        class_embed_bias=False,
        export=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        if self.dn_number > 0:
            self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWraper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        self.backbone = backbone

        self.input_proj = self.prepare_channel_mapper(num_feature_levels, hidden_dim, two_stage_type)
        self.position_embedding = position_embedding

        self.aux_loss = aux_loss

        self.aux_loss = aux_loss
        self.export = export
        if self.export:
            warnings.warn("Setting aux_loss to be False for export")
            self.aux_loss = False

        self.box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        # Bias True is what's different for MM pipeline
        _class_embed = ContrastiveEmbed(max_text_len=self.max_text_len, log_scale=log_scale, bias=class_embed_bias)

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def prepare_channel_mapper(self, num_feature_levels, hidden_dim, two_stage_type):
        """Create Channel Mapper style for DETR-based model.

        Args:
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            two_stage_type (str): type of two stage in Grounding DINO.
            hidden_dim (int): size of the hidden dimension.

        Returns:
            nn.ModuleList of input projection.
        """
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            return nn.ModuleList(input_proj_list)

        assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.backbone.num_channels[-1], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        """initialize reference point"""
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(self, samples: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                text_self_attention_masks: Optional[torch.Tensor] = None,
                targets: Optional[dict] = None):
        """Forward function of Grounding DINO Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]
            targets (dict): batched annotations

        Returns:
            pred_logits (torch.Tensor): the classification logits (including no-object) for all queries. Shape= [batch_size x num_queries x (num_classes + 1)]
            pred_boxes (torch.Tensor): The normalized boxes coordinates for all queries, represented as(center_x, center_y, height, width)
        """
        tokenized = {}

        tokenized["input_ids"] = input_ids
        tokenized["attention_mask"] = text_self_attention_masks
        tokenized["position_ids"] = position_ids
        tokenized["token_type_ids"] = token_type_ids

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        if not isinstance(samples, torch.Tensor):
            samples = tensor_from_tensor_list(samples)

        features = self.backbone(samples)

        srcs = []
        masks = []
        for level, feat in enumerate(features):
            src = feat[0]
            mask = (feat[1].float()[:, 0].bool())
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for li in range(_len_srcs, self.num_feature_levels):
                if li == _len_srcs:
                    src = self.input_proj[li](features[-1][0])
                else:
                    src = self.input_proj[li](srcs[-1])
                srcs.append(src)

                if self.export:
                    m = torch.zeros((src.shape[0], 1, src.shape[2], src.shape[3]), dtype=src.dtype, device=src.device)
                else:
                    m = samples[:, 3:4]
                mask = F.interpolate(m.float(), size=src.shape[-2:]).to(torch.bool)
                masks.append(mask.float()[:, 0].bool())

        # build positional embedding
        pos = []
        for mask in masks:
            if self.export:
                N, H, W = mask.shape
                tensor_shape = torch.tensor([N, H, W], device=src.device)
                pos.append(self.position_embedding(tensor_shape, src.device))
            else:
                not_mask = ~mask
                pos.append(self.position_embedding(not_mask, src.device))

        if self.dn_number > 0 and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training, num_queries=self.num_queries, num_classes=self.max_text_len,
                                hidden_dim=self.hidden_dim, label_enc=self.label_enc)
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, pos, input_query_label, attn_mask, text_dict
        )

        if self.dn_number > 0 and targets is not None:
            # In case num object=0
            hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for _, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        if not self.export:
            # Used to calculate losses
            bs, len_td = text_dict['text_token_mask'].shape
            out['text_mask'] = torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
                samples.device
            )

            out['text_mask'][:, :len_td] = text_dict['text_token_mask']
            for b in range(bs):
                for j in range(len_td):
                    if text_dict['text_token_mask'][b][j] is True:
                        out['text_mask'][b][j] = True

            # for intermediate outputs
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            if not self.export:
                out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
                out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}
                out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
