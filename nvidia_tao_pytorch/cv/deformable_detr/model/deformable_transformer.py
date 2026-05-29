# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Deformable Transformer module. """
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_
import torch.utils.checkpoint as checkpoint

from nvidia_tao_pytorch.core.modules.activation.activation import MultiheadAttention
from nvidia_tao_pytorch.cv.deformable_detr.utils.misc import inverse_sigmoid
from nvidia_tao_pytorch.cv.deformable_detr.model.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    """Deformable Transfromer module."""

    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.3,
                 activation="relu", return_intermediate_dec=True,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 export=False, export_format='onnx', activation_checkpoint=True):
        """Initialize Deformable Transformer Module.

        Args:
            d_model (int): size of the hidden dimension.
            nhead (int): number of heads.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            dim_feedforward (int): dimension of the feedforward layer.
            dropout (float): probability for the dropout layer.
            activation (str): type of activation layer.
            return_intermediate_dec (bool): return intermediate decoder layers.
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            dec_n_points (int): number of reference points in the decoder.
            enc_n_points (int): number of reference points in the encoder.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            export_format (str): format for exporting (e.g. 'onnx' or 'xdl')
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.export = export
        self.export_format = export_format
        self.activation_checkpoint = activation_checkpoint

        encoder_args = {
            "d_model": d_model,
            "dropout": dropout,
            "d_ffn": dim_feedforward,
            "activation": activation,
            "n_levels": num_feature_levels,
            "n_heads": nhead,
            "n_points": enc_n_points,
            "export": self.export,
        }

        decoder_args = dict(encoder_args)
        decoder_args["n_points"] = dec_n_points
        decoder_args["export_format"] = self.export_format

        self.encoder = DeformableTransformerEncoder(num_encoder_layers, encoder_args,
                                                    export=self.export, activation_checkpoint=self.activation_checkpoint)
        self.decoder = DeformableTransformerDecoder(num_decoder_layers, decoder_args,
                                                    return_intermediate=return_intermediate_dec,
                                                    export=self.export,
                                                    activation_checkpoint=self.activation_checkpoint)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset parmaeters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """Compute the valid ratio from given mask."""
        _, H, W = mask.shape
        temp_mask = mask.bool()
        valid_H = torch.sum((~temp_mask).float()[:, :, 0], 1)
        valid_W = torch.sum((~temp_mask).float()[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """Forward function."""
        assert query_embed is not None

        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        if self.export:
            spatial_shapes = []
        else:
            spatial_shapes = torch.empty(len(srcs), 2, dtype=torch.int32, device=srcs[0].device)

        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            if self.export:  # Input shaped is fixed for export in onnx/tensorRT
                spatial_shapes.append(torch.tensor([[h, w]], dtype=torch.int32, device=srcs[0].device))
            else:  # Used for dynamic input shape
                spatial_shapes[lvl, 0], spatial_shapes[lvl, 1] = h, w

            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        if isinstance(spatial_shapes, list):
            spatial_shapes = torch.cat(spatial_shapes, 0)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])).type(torch.int32)
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed)

        inter_references_out = inter_references

        return hs, init_reference_out, inter_references_out


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable Transfromer Encoder Layer module."""

    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.3, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, export=False):
        """Initializes the Transformer Encoder Layer.

        Args:
            d_model (int): size of the hidden dimension.
            d_ffn (int): dimension of the feedforward layer.
            dropout (float): probability for the dropout layer.
            activation (str): type of activation layer.
            n_heads (int): number of heads.
            n_points (int): number of encoder layers.
            export (bool): flag to indicate if the current model is being used for ONNX export.
        """
        super().__init__()
        self.export = export
        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional Embedding to the tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        """Forward ffn."""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        """Forward function for Encoder Layer."""
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, export=self.export)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transfromer Encoder module"""

    def __init__(self, num_layers, encoder_args={}, export=False, activation_checkpoint=True):
        """Initializes the Transformer Encoder Module.

        Args:
            num_layers (int): number of encoder layers.
            encoder_args (dict): additional arguments.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
        """
        super().__init__()
        self.layers = _get_clones(DeformableTransformerEncoderLayer, num_layers, **encoder_args)
        self.num_layers = num_layers
        self.export = export
        self.activation_checkpoint = activation_checkpoint

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device, export=False):
        """Get reference points."""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            if export:  # Fixed dimensions for export in onnx
                H_, W_ = int(H_), int(W_)
            else:
                H_, W_ = spatial_shapes[lvl, 0], spatial_shapes[lvl, 1]

            torch._check(int(H_ * W_) != int(W_))
            torch._check(int(W_) > 1)

            range_y = torch.arange(H_, dtype=torch.int32, device=device).float() + 0.5
            range_x = torch.arange(W_, dtype=torch.int32, device=device).float() + 0.5

            ref_y, ref_x = torch.meshgrid(range_y, range_x)

            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)

            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        torch._check(int(reference_points.shape[1]) < 9223372036854775807)

        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None):
        """Forward function for Encoder Module."""
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device, export=self.export)
        for _, layer in enumerate(self.layers):
            if self.export or not self.activation_checkpoint:
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
            else:
                output = checkpoint.checkpoint(layer,
                                               output,
                                               pos,
                                               reference_points,
                                               spatial_shapes,
                                               level_start_index,
                                               use_reentrant=True)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """ Deformable Transfromer Decoder Layer module """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.3, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 export=False, export_format='onnx'):
        """Initializes the Transformer Decoder Layer.

        Args:
            d_model (int): size of the hidden dimension.
            d_ffn (int): dimension of the feedforward layer.
            dropout (float): probability for the dropout layer.
            activation (str): type of activation layer.
            n_heads (int): number of heads.
            n_points (int): number of encoder layers.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            export_format (str): format for exporting (e.g. 'onnx' or 'xdl')
        """
        super().__init__()
        self.export = export
        self.export_format = export_format
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        if self.export and self.export_format != "xdl":
            # Starting from PyT 1.14, _scaled_dot_product_attention has been switched to C++ backend
            # which is not exportable as ONNX operator
            # However, the training / eval time can be greatly optimized by Torch selecting the optimal
            # attention mechanism under the hood
            self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional Embedding to the tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Forward ffn."""
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index):
        """Forward function for Decoder Layer."""
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, export=self.export)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


# pylint:disable=E1136
class DeformableTransformerDecoder(nn.Module):
    """ Deformable Transfromer Decoder module """

    def __init__(self, num_layers, decoder_args={}, return_intermediate=False, export=False, activation_checkpoint=True):
        """Initializes the Transformer Decoder Module.

        Args:
            num_layers (int): number of decoder layers.
            decoder_args (dict): additional arguments.
            return_intermediate (bool): flat to indicate if intermediate outputs to be returned.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
        """
        super().__init__()
        self.export = export
        self.activation_checkpoint = activation_checkpoint
        self.layers = _get_clones(DeformableTransformerDecoderLayer, num_layers, **decoder_args)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None):
        """Forward function for Decoder Module."""
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            if self.export or not self.activation_checkpoint:
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index)
            else:
                output = checkpoint.checkpoint(layer,
                                               output,
                                               query_pos,
                                               reference_points_input,
                                               src,
                                               src_spatial_shapes,
                                               src_level_start_index,
                                               use_reentrant=True)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module_class, N, **kwargs):
    """Get clones of nn.Module.

    Args:
        module_class (nn.Module): torch module to clone.
        N (int): number of times to clone.

    Returns:
        nn.ModuleList of the cloned module_class.
    """
    return nn.ModuleList([module_class(**kwargs) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string.

    Args:
        activation (str): type of activation function.

    Returns:
        PyTorch activation layer.

    Raises:
        RuntimeError: if unsupported activation type is provided.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")
