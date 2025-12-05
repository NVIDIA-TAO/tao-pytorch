# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Mask Grounding DINO model. """

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import tensor_from_tensor_list, inverse_sigmoid
from nvidia_tao_pytorch.cv.dino.model.model_utils import MLP

from nvidia_tao_pytorch.cv.grounding_dino.model.groundingdino import GroundingDINO
from nvidia_tao_pytorch.cv.mask_grounding_dino.utils.vl_utils import create_positive_map, create_positive_map_from_span

from nvidia_tao_pytorch.cv.mask_grounding_dino.model.model_utils import (
    aligned_bilinear, compute_locations, parse_dynamic_params,
    MaskHeadConv,
)


class MaskGroundingDINO(GroundingDINO):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        *args,
        has_mask=True,
        rela=None,
        **kwargs,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            rela: ReLA module. See rela.py
        """
        super().__init__(*args, **kwargs)

        # Mask Branch
        self.has_mask = has_mask
        self.rela = None
        if has_mask:
            self.in_channels = self.hidden_dim // 32
            self.dynamic_mask_channels = 8
            self.controller_layers = 3
            self.max_insts_num = 100
            self.mask_out_stride = 4
            self.up_rate = 8 // self.mask_out_stride
            self.rel_coord = True
            self.rela = rela
            # dynamic_mask_head params
            weight_nums, bias_nums = [], []
            for index in range(self.controller_layers):
                if index == 0:
                    if self.rel_coord:
                        weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                    else:
                        weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                    bias_nums.append(self.dynamic_mask_channels)
                elif index == self.controller_layers - 1:
                    weight_nums.append(self.dynamic_mask_channels * 1)
                    bias_nums.append(1)
                else:
                    weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                    bias_nums.append(self.dynamic_mask_channels)

            self.weight_nums = weight_nums
            self.bias_nums = bias_nums
            self.num_gen_params = sum(weight_nums) + sum(bias_nums)

            self.controller = MLP(self.hidden_dim, self.hidden_dim, self.num_gen_params, 3)

            for contr in self.controller.layers:
                nn.init.xavier_uniform_(contr.weight)
                nn.init.zeros_(contr.bias)
            self.use_raft = False
            self.mask_head = MaskHeadConv(self.hidden_dim, None, self.hidden_dim, use_raft=self.use_raft, up_rate=self.up_rate)

    def forward(self, samples: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                text_self_attention_masks: Optional[torch.Tensor] = None,
                one_hot_token=None,
                is_training=False,
                cat_list=None,
                captions=None,
                targets=None):
        """Forward function of Mask Grounding DINO Model

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
        if self.export:
            is_training = False
            tensor_shapes = samples.shape
            image_sizes = [tensor_shapes[-2:]] * int(tensor_shapes[0])
        else:
            image_sizes = [t['size'] for t in targets]

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
        spatial_shapes = []
        for level, feat in enumerate(features):
            src = feat[0]
            mask = (feat[1].float()[:, 0].bool())
            src_proj_l = self.input_proj[level](src)
            srcs.append(src_proj_l)
            masks.append(mask)
            _, _, h, w = src_proj_l.shape
            spatial_shapes.append((h, w))

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

        input_query_bbox = input_query_label = attn_mask = None
        hs, inter_references, hs_enc, ref_enc, init_box_proposal, memory, spatial_shapes = self.transformer(
            srcs, masks, input_query_bbox, pos, input_query_label, attn_mask, text_dict
        )
        # hs: 6 of [4, 900, 256]
        # memory: 4 of [37150, 256], /sum(hi*wi)

        # deformable-detr-like anchor update
        outputs = {}
        outputs_class_list = []
        outputs_coord_list = []
        outputs_mask_list = []
        outputs_mask_feature_list = []
        indices_list = []
        enc_lay_num = len(hs)
        assert enc_lay_num == 6
        if is_training:
            for lvl in range(enc_lay_num):
                layer_ref_sig = inter_references[lvl]
                layer_ref_unsig = inverse_sigmoid(layer_ref_sig)
                tmp = self.bbox_embed[lvl](hs[lvl])
                assert layer_ref_unsig.shape[-1] == 4
                tmp += layer_ref_unsig
                outputs_coord = tmp.sigmoid()
                outputs_coord_list.append(outputs_coord)
                outputs_class = self.class_embed[lvl](hs[lvl], text_dict)
                outputs_class_list.append(outputs_class)
                outputs_layer = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}

                if self.has_mask:
                    dynamic_mask_head_params = self.controller(hs[lvl])    # [bs, num_quries, num_params]
                    # caption == wine glass . cat . hot dog . ....
                    # cat_list == [['wine glass', 'cat', 'hot dog', ...
                    label_map_list = []
                    indices = []
                    for j in range(len(cat_list)):  # bs
                        if len(targets[j].get("positive_tokens", [])) > 0:
                            tokens_positive = targets[j]["positive_tokens"]  # List[List[(char_start, char_end)]]
                            label_map = create_positive_map_from_span(one_hot_token[j], [tokens_positive], empty=targets[j].get('empty', False))
                        else:
                            label_map = []
                            for i in range(len(cat_list[j])):
                                label_id = torch.tensor([i])
                                per_label = create_positive_map(one_hot_token[j], label_id, cat_list[j], captions[j], empty=targets[j].get('empty', False))
                                label_map.append(per_label)
                            label_map = torch.stack(label_map, dim=0).squeeze(1)
                        label_map_list.append(label_map)
                    # label_map.shape == [80, 256]
                    for j in range(len(cat_list)):  # bs
                        if targets[j].get('empty', False):
                            # inds = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64))]
                            inds = [(torch.zeros((0,), dtype=torch.int64), torch.zeros((0,), dtype=torch.int64))]
                        else:
                            for_match = {
                                "pred_logits": outputs_layer['pred_logits'][j].unsqueeze(0),
                                "pred_boxes": outputs_layer['pred_boxes'][j].unsqueeze(0)
                            }

                            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
                        indices.extend(inds)

                    # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
                    # - index_i is the indices of the selected predictions (in order)
                    # - index_j is the indices of the corresponding selected targets (in order)
                    indices_list.append(indices)
                    reference_points, mask_head_params, num_insts = [], [], []
                    for i, index in enumerate(indices):
                        pred_i, _ = index  # _ = tgt_j
                        num_insts.append(len(pred_i))
                        mask_head_params.append(dynamic_mask_head_params[i, pred_i].unsqueeze(0))

                        # This is the image size after data augmentation (so as the gt boxes & masks)

                        orig_h, orig_w = image_sizes[i]
                        orig_h = torch.as_tensor(orig_h).to(layer_ref_unsig)
                        orig_w = torch.as_tensor(orig_w).to(layer_ref_unsig)
                        scale_f = torch.stack([orig_w, orig_h], dim=0)

                        ref_cur_f = layer_ref_unsig[i].sigmoid()
                        ref_cur_f = ref_cur_f[:, :2]
                        ref_cur_f = ref_cur_f * scale_f[None, :]
                        reference_points.append(ref_cur_f[pred_i].unsqueeze(0))

                    # reference_points: [1, \sum{selected_insts}, 2]
                    # mask_head_params: [1, \sum{selected_insts}, num_params]
                    reference_points = torch.cat(reference_points, dim=1)
                    mask_head_params = torch.cat(mask_head_params, dim=1)

                    # mask prediction
                    has_mask_list = ["masks" in x.keys() for x in targets]
                    assert len(set(has_mask_list)) == 1  # must be "all True" or "all False"
                    if has_mask_list[0]:
                        outputs_layer = self.forward_mask_head(
                            outputs_layer, memory, spatial_shapes,
                            reference_points, mask_head_params, num_insts)
                    else:
                        # avoid unused parameters
                        dummy_output = torch.sum(mask_head_params)
                        for p in self.mask_head.parameters():
                            dummy_output += p.sum()
                        outputs_layer['pred_masks'] = 0.0 * dummy_output
                    outputs_mask_list.append(outputs_layer['pred_masks'])
                    outputs_mask_feature_list.append(outputs_layer['mask_features'])
            outputs_class = torch.stack(outputs_class_list)
            outputs_coord = torch.stack(outputs_coord_list)
            outputs['pred_logits'] = outputs_class[-1]
            outputs['pred_boxes'] = outputs_coord[-1]
        else:
            lvl = enc_lay_num - 1
            layer_ref_sig = inter_references[lvl - 1]
            layer_ref_unsig = inverse_sigmoid(layer_ref_sig)
            outputs_class = self.class_embed[lvl](hs[lvl], text_dict)

            tmp = self.bbox_embed[lvl](hs[lvl])
            tmp += layer_ref_unsig
            outputs_coord = tmp.sigmoid()

            outputs['pred_logits'] = outputs_class
            outputs['pred_boxes'] = outputs_coord

        if self.has_mask:
            if is_training:
                outputs["pred_masks"] = outputs_mask_list[-1]
                outputs["mask_features"] = outputs_mask_feature_list[-1]
            else:
                # mask infer
                outputs['reference_points'] = inter_references[-2][:, :, :2]
                dynamic_mask_head_params = self.controller(hs[-1])    # [bs, num_quries, num_params]

                bs, num_queries, _ = dynamic_mask_head_params.shape
                if self.export and isinstance(bs, torch.Tensor):
                    bs = int(bs.cpu())
                num_insts = [num_queries for i in range(bs)]
                reference_points = []
                for i, image_size_i in enumerate(image_sizes):
                    orig_h, orig_w = image_size_i
                    orig_h = torch.as_tensor(orig_h).to(outputs['reference_points'][i])
                    orig_w = torch.as_tensor(orig_w).to(outputs['reference_points'][i])
                    scale_f = torch.stack([orig_w, orig_h], dim=0)
                    ref_cur_f = outputs['reference_points'][i] * scale_f[None, :]
                    reference_points.append(ref_cur_f.unsqueeze(0))
                # reference_points: [1, N * num_queries, 2]
                # mask_head_params: [1, N * num_queries, num_params]
                reference_points = torch.cat(reference_points, dim=1)
                mask_head_params = dynamic_mask_head_params.reshape(1, bs * self.num_queries, dynamic_mask_head_params.shape[-1])
                # mask prediction
                outputs = self.forward_mask_head(outputs, memory, spatial_shapes,
                                                 reference_points, mask_head_params, num_insts)
                # outputs['pred_masks']: [bs, num_queries, num_frames, H/4, W/4]
                outputs['pred_masks'] = torch.cat(outputs['pred_masks'], dim=0)
                outputs.pop('reference_points')
        if not self.export:
            # Used to calculate losses
            bs, len_td = text_dict['text_token_mask'].shape
            outputs['text_mask'] = torch.zeros(bs, self.max_text_len, dtype=torch.bool).to(
                samples.device
            )

            outputs['text_mask'][:, :len_td] = text_dict['text_token_mask']

            # for intermediate outputs
            if self.aux_loss:
                outputs['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_mask_list)

        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            if not self.export:
                outputs['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
                outputs['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        if self.rela is not None:
            # Split memory into per-level sequences: [(B, H*W, C), ...]
            spatial_shapes = spatial_shapes.tolist()
            memory_split = memory.split([h * w for h, w in spatial_shapes], dim=1)

            # Flatten positional embeddings from (B, C, H, W) → (B, C, HW)
            rela_pos = [p.flatten(2) for p in pos]

            # Forward pass through ReLA module
            rela_outputs = self.rela(
                srcs=memory_split,
                pos_embeds=rela_pos,
                mask_features=outputs['mask_features'],
                lang_feat=text_dict['encoded_text'],
                spatial_shapes=spatial_shapes,
            )

            # Use only the last layer outputs
            outputs['no_targets'] = rela_outputs['no_targets'][-1]
            outputs['union_mask_logits'] = rela_outputs['union_mask_logits'][-1]
            if self.training:
                outputs['minimaps'] = rela_outputs['minimaps'][-1]
                outputs['valid_masks'] = samples[:, 3:4]
        outputs.pop('mask_features')
        return outputs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_mask):
        """Set aux loss."""
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if len(outputs_mask) > 0:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_masks': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_mask[:-1])]
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def mask_heads_forward(self, features, weights, biases, num_insts):
        """
        :param features [bs*nq*(8+2), H/8, W/8]
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        """
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def upsample_preds(self, pred, mask):
        """ Upsample pred [N, 1, H/8, W/8] -> [N, 1, H, W] using convex combination """
        N, _, H, W = pred.shape
        mask = mask.view(1, 1, 9, self.up_rate, self.up_rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_pred = F.unfold(pred, [3, 3], padding=1)
        up_pred = up_pred.view(N, 1, 9, 1, 1, H, W)

        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, 1, self.up_rate * H, self.up_rate * W)

    def dynamic_mask_with_coords(self, mask_feats, reference_points, mask_head_params, num_insts,
                                 mask_feat_stride, rel_coord=True, up_masks=None):
        """Mask prediction with dynamic conv kernels."""
        # mask_feats: [N, C/32, H/8, W/8]
        # reference_points: [1, \sum{selected_insts}, 2]
        # mask_head_params: [1, \sum{selected_insts}, num_params]
        # return:
        #     mask_logits: [1, \sum{num_queries}, H/8, W/8]
        device = mask_feats.device

        _, in_channels, H, W = mask_feats.size()
        num_insts_all = reference_points.shape[1]
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            device=device, stride=mask_feat_stride)
        # locations: [H*W, 2]

        if rel_coord:
            instance_locations = reference_points
            # instance_locations: [1, num_insts_all, 2]
            # locations: [H*W, 2]
            # import pdb;pdb.set_trace()
            # relative_coords = locations.reshape(1, 1, H, W, 2).repeat(1,num_insts_all,1,1,1)
            relative_coords = instance_locations.reshape(1, num_insts_all, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            # relative_coords: [1, num_insts_all, H, W, 2]
            # # coords normalization
            # scale = torch.tensor([W, H],device=device)
            # relative_coords = relative_coords.float() / scale[None, None, None, None, :]
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            # relative_coords: [1, num_insts_all, 2, H*W]
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                # [1, num_queries * (C/32+2), H/8 * W/8]
                relative_coords_b = relative_coords[:, inst_st: inst_st + num_inst, :, :]  # torch.Size([1, 900, 2, 8160])
                mask_feats_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)  # torch.Size([1, 900, 8, 8160])
                mask_head_b = torch.cat([relative_coords_b, mask_feats_b], dim=2)

                mask_head_inputs.append(mask_head_b)
                inst_st += num_inst

        else:
            mask_head_inputs = []
            inst_st = 0
            for i, num_inst in enumerate(num_insts):
                mask_head_b = mask_feats[i].reshape(1, in_channels, H * W).unsqueeze(1).repeat(1, num_inst, 1, 1)
                mask_head_b = mask_head_b.reshape(1, -1, H, W)
                mask_head_inputs.append(mask_head_b)

        # mask_head_inputs: [1, \sum{num_queries}, (C/32+2)}, H/8* W/8]
        mask_head_inputs = torch.cat(mask_head_inputs, dim=1)
        # mask_head_inputs: [1, \sum{num_queries * (C/32+2)}, H/8, W/8], C=256
        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1)

        if num_insts_all != 0:
            weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

            mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        else:
            mask_logits = mask_head_inputs + torch.sum(mask_head_params) * 0.0
            return mask_logits
        # mask_logits: [1, num_insts_all, H/8, W/8]
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        # upsample predicted masks
        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0

        if self.use_raft:
            assert up_masks is not None
            inst_idx = 0
            mask_logits_output = []
            for b, n in enumerate(num_insts):
                mask_logits_output.append(self.upsample_preds(mask_logits[inst_idx: inst_idx + n], up_masks[b: b + 1]))
                inst_idx += n
            mask_logits = torch.cat(mask_logits_output, dim=0)
        else:
            mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        mask_logits = mask_logits.reshape(1, -1, mask_logits.shape[-2], mask_logits.shape[-1])
        # mask_logits: [1, num_insts_all, H/4, W/4]

        return mask_logits

    def forward_mask_head(self, outputs, feats, spatial_shapes, reference_points, mask_head_params, num_insts):
        """Mask branch forward pass.

        feats (memory from encoder): [bs, /sum{hi*wi}, C], C=256
        mask_head_params: [1, nq*bs, 169]
        """
        bs, _, c = feats.shape

        # encod_feat_l: num_layers x [bs, C, num_frames, hi, wi]
        encod_feat_l = []
        spatial_indx = 0
        for feat_l in range(self.num_feature_levels - 1):
            h, w = spatial_shapes[feat_l]
            mem_l = feats[:, spatial_indx: spatial_indx + 1 * h * w, :].reshape(bs, 1, h, w, c).permute(0, 4, 1, 2, 3)
            encod_feat_l.append(mem_l)
            spatial_indx += 1 * h * w

        pred_masks = []
        for iframe in range(1):
            encod_feat_f = []
            for lvl in range(self.num_feature_levels - 1):
                encod_feat_f.append(encod_feat_l[lvl][:, :, iframe, :, :])  # [bs, C, hi, wi]

            mask_head_outputs = self.mask_head(encod_feat_f, fpns=None)
            decod_feat_f = mask_head_outputs['fused_x']
            fused_x_fpn = mask_head_outputs['fused_x_fpn']
            if self.use_raft:
                up_masks = mask_head_outputs['up_masks']
            else:
                up_masks = None

            mask_logits = self.dynamic_mask_with_coords(decod_feat_f, reference_points, mask_head_params,
                                                        num_insts=num_insts,
                                                        mask_feat_stride=8,
                                                        rel_coord=self.rel_coord, up_masks=up_masks)
            # mask_logits: [1, num_queries_all, H/4, W/4]

            mask_f = []
            inst_st = 0
            for num_inst in num_insts:
                # [1, selected_queries, 1, H/4, W/4]
                mask_f.append(mask_logits[:, inst_st: inst_st + num_inst, :, :].unsqueeze(2))
                inst_st += num_inst

            pred_masks.append(mask_f)

        output_pred_masks = []
        for i, num_inst in enumerate(num_insts):
            out_masks_b = [m[i] for m in pred_masks]
            output_pred_masks.append(torch.cat(out_masks_b, dim=2))

        outputs['pred_masks'] = output_pred_masks
        outputs['mask_features'] = fused_x_fpn
        return outputs
