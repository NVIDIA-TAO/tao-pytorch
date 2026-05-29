# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build CoDETR model."""

import torch.nn as nn

from nvidia_tao_pytorch.cv.dino.model.build_nn_model import DINOModel
from nvidia_tao_pytorch.cv.codetr.model.collaborative_head import ATSSCollaborativeHead

# Strides corresponding to backbone stage indices.
# Swin / ResNet stage index → feature map stride.
# Index 4 is the extra level created via stride-2 conv on the last backbone stage.
_STRIDE_MAP = {0: 4, 1: 8, 2: 16, 3: 32, 4: 64}


class CoDETRModel(nn.Module):
    """CoDETR model.

    Wraps DINOModel (backbone + deformable transformer + DETR head) and adds
    ATSS collaborative auxiliary heads for training. At export time the
    collaborative heads are omitted — the exported graph is identical to DINO.

    Args:
        dino_kwargs (dict): keyword arguments forwarded to ``DINOModel``.
        collab_kwargs (dict): keyword arguments forwarded to each
            ``ATSSCollaborativeHead``. The optional ``num_co_heads`` entry
            controls how many collaborative heads are instantiated.
        export (bool): whether to build the model for export. When ``True``,
            collaborative heads are skipped.
    """

    def __init__(self, dino_kwargs, collab_kwargs, export=False):
        """Initialize CoDETRModel.

        Args:
            dino_kwargs (dict): keyword arguments forwarded to DINOModel.
            collab_kwargs (dict): keyword arguments for ATSSCollaborativeHead.
            export (bool): if True skip instantiating collaborative heads.
        """
        super().__init__()
        self.export = export
        self.model = DINOModel(**dino_kwargs)

        if not export:
            hidden_dim = dino_kwargs.get('hidden_dim', 256)
            num_co_heads = collab_kwargs.pop("num_co_heads", 1)
            self.collab_heads = nn.ModuleList(
                [ATSSCollaborativeHead(**collab_kwargs) for _ in range(num_co_heads)]
            )
            # Downsample module to create 6th feature level from last encoder output
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                nn.GroupNorm(32, hidden_dim),
            )
        else:
            self.collab_heads = None

    def forward(self, x, targets=None):
        """Forward pass.

        Args:
            x (Tensor): input image tensor [B, 3+1, H, W] (3 RGB + 1 mask channel).
            targets (list[dict] | None): ground-truth annotations for training.

        Returns:
            out (dict): DINO-compatible output dict. During training includes
                'collab_outputs' key with list of (cls_scores, bbox_preds, centernesses).
        """
        out = self.model(x, targets)

        if self.training and self.collab_heads is not None and 'enc_memory' in out:
            # Reshape encoder memory back to per-level spatial feature maps
            enc_memory = out['enc_memory']           # [bs, sum(h_i*w_i), C]
            spatial_shapes = out['enc_spatial_shapes']  # [num_levels, 2]
            enc_features = []
            start = 0
            for lvl in range(spatial_shapes.shape[0]):
                h, w = spatial_shapes[lvl]
                end = start + h * w
                feat = enc_memory[:, start:end, :].permute(0, 2, 1).contiguous()
                enc_features.append(feat.reshape(-1, enc_memory.shape[2], h, w))
                start = end
            # Add 6th level via stride-2 downsample of last encoder feature
            enc_features.append(self.downsample(enc_features[-1]))
            collab_outputs = [head(enc_features) for head in self.collab_heads]
            out['collab_outputs'] = collab_outputs

        return out


def build_model(experiment_config, export=False):
    """Build CoDETR model from experiment config.

    Args:
        experiment_config (OmegaConf): experiment configuration.
        export (bool): flag for ONNX export mode.

    Returns:
        model (CoDETRModel): the constructed CoDETR model.
    """
    model_config = experiment_config.model
    dataset_config = experiment_config.dataset
    num_classes = dataset_config.num_classes

    return_interm_indices = list(model_config.return_interm_indices)
    strides = [_STRIDE_MAP[i] for i in return_interm_indices if i in _STRIDE_MAP]
    # Collab heads receive encoder features + one downsampled level (stride 2x last)
    collab_strides = strides + [strides[-1] * 2]

    dino_kwargs = dict(
        num_classes=num_classes,
        hidden_dim=model_config.hidden_dim,
        pretrained_backbone_path=model_config.pretrained_backbone_path,
        backbone=model_config.backbone,
        train_backbone=model_config.train_backbone,
        num_feature_levels=model_config.num_feature_levels,
        nheads=model_config.nheads,
        enc_layers=model_config.enc_layers,
        dec_layers=model_config.dec_layers,
        dim_feedforward=model_config.dim_feedforward,
        dec_n_points=model_config.dec_n_points,
        enc_n_points=model_config.enc_n_points,
        num_queries=model_config.num_queries,
        aux_loss=model_config.aux_loss,
        dilation=model_config.dilation,
        dropout_ratio=model_config.dropout_ratio,
        export=export,
        activation_checkpoint=experiment_config.train.activation_checkpoint,
        return_interm_indices=return_interm_indices,
        decoder_sa_type=model_config.decoder_sa_type,
        embed_init_tgt=model_config.embed_init_tgt,
        use_dn=model_config.use_dn,
        dn_number=model_config.dn_number,
        dn_box_noise_scale=model_config.dn_box_noise_scale,
        dn_label_noise_ratio=model_config.dn_label_noise_ratio,
        pe_temperatureH=model_config.pe_temperatureH,
        pe_temperatureW=model_config.pe_temperatureW,
        lsj_resolution=experiment_config.dataset.augmentation.fixed_random_crop,
        pre_norm=model_config.pre_norm,
        two_stage_type=model_config.two_stage_type,
        fix_refpoints_hw=model_config.fix_refpoints_hw,
        # Co-DETR (mmdet) uses INDEPENDENT per-layer reg/cls branches in the
        # decoder. The DINOModel default is to share them, which causes the
        # per-layer cls_branches/reg_branches weights from a Co-DETR checkpoint
        # to be silently collapsed (only the last decoder layer's weights end
        # up loaded). Override the defaults to match Co-DETR's architecture so
        # all 6 decoder layers get their own independently-loaded weights.
        dec_pred_class_embed_share=False,
        dec_pred_bbox_embed_share=False,
    )

    collab_kwargs = dict(
        num_co_heads=model_config.num_co_heads,
        num_classes=num_classes,
        in_channels=model_config.hidden_dim,
        feat_channels=model_config.hidden_dim,
        num_convs=model_config.co_head_num_convs,
        strides=collab_strides,
    )

    return CoDETRModel(dino_kwargs=dino_kwargs, collab_kwargs=collab_kwargs, export=export)
