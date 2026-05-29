# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from copy import deepcopy
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level='INFO')
logger = logging.getLogger(__name__)


def auto_scale_workers(cfg, num_workers: int):
    """
    When the config is defined for certain number of workers (according to
    ``cfg.train.reference_world_size``) that's different from the number of
    workers currently in use, returns a new cfg where the total batch size
    is scaled so that the per-GPU batch size stays the same as the
    original ``total_batch_size // reference_world_size``.

    Other config options are also scaled accordingly:
    * training steps and warmup steps are scaled inverse proportionally.
    * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.

    For example, with the original config like the following:

    .. code-block:: yaml

        dataloader.train.total_batch_size: 16
        optimizer.lr: 0.1
        train.reference_world_size: 8
        train.max_iter: 5000
        train.checkpointer.period: 1000

    When this config is used on 16 GPUs instead of the reference number 8,
    calling this method will return a new config with:

    .. code-block:: yaml

        dataloader.train.total_batch_size: 32
        optimizer.lr: 0.2
        train.reference_world_size: 16
        train.max_iter: 2500
        train.checkpointer.period: 500

    Note that both the original config and this new config can be trained on 16 GPUs.
    It's up to user whether to enable this feature (by setting ``reference_world_size``).

    Returns:
        CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
    """
    old_world_size = cfg.train.reference_world_size
    if old_world_size == 0 or old_world_size == num_workers:
        return cfg
    cfg = deepcopy(cfg)

    assert cfg.dataloader.train.total_batch_size % old_world_size == 0, (
        f"Invalid reference_world_size in config! "
        f"{cfg.dataloader.train.total_batch_size} % {old_world_size} != 0"
    )
    scale = num_workers / old_world_size
    bs = cfg.dataloader.train.total_batch_size = int(
        round(cfg.dataloader.train.total_batch_size * scale)
    )
    lr = cfg.optimizer.lr = cfg.optimizer.lr * scale
    max_iter = cfg.train.max_iter = int(round(cfg.train.max_iter / scale))
    cfg.train.eval_period = int(round(cfg.train.eval_period / scale))
    cfg.train.checkpointer.period = int(round(cfg.train.checkpointer.period / scale))
    cfg.train.reference_world_size = num_workers  # maintain invariant
    logger.info(
        "Auto-scaling the config to batch_size=%s, learning_rate=%s, max_iter=%s.",
        bs, lr, max_iter
    )
    return cfg


def override_default_cfg(results_dir, cfg, hydra_cfg, world_size=1):
    """Override the default LazyConfig parameters with Hydra configs."""
    cfg.train.cfg_name = 'odise_' + hydra_cfg.model.type + '_' + hydra_cfg.model.name
    cfg.train.run_name = ("${train.cfg_name}_bs${dataloader.train.total_batch_size}")
    cfg.train.reference_world_size = hydra_cfg.reference_world_size
    cfg.dataloader.train.total_batch_size = hydra_cfg.dataset.total_batch_size
    cfg.train.eval_period = hydra_cfg.train.checkpoint_interval
    cfg.train.checkpointer.period = hydra_cfg.train.checkpoint_interval
    cfg.optimizer.lr = hydra_cfg.train.learning_rate
    cfg.train.max_iter = hydra_cfg.train.max_iter
    cfg = auto_scale_workers(cfg, world_size)
    cfg.train.output_dir = results_dir
    cfg.train.amp.enabled = hydra_cfg.train.use_amp
    cfg.train.log_dir = results_dir
    cfg.train.init_checkpoint = hydra_cfg.train.checkpoint
    cfg.train.grad_clip = hydra_cfg.train.grad_clip
    cfg.optimizer.weight_decay = hydra_cfg.train.weight_decay

    cfg.model.alpha = hydra_cfg.model.alpha
    cfg.model.beta = hydra_cfg.model.beta
    cfg.model.backbone.model_name = hydra_cfg.model.name
    cfg.model.backbone.pretrained = hydra_cfg.model.pretrained_weights
    if hydra_cfg.model.type == 'category':
        cfg.model.category_head.clip_model_name = hydra_cfg.model.name
        cfg.model.category_head.pretrained = hydra_cfg.model.pretrained_weights
        cfg.model.category_head.projection_dim = hydra_cfg.model.text_proj_dim
        cfg.model.category_head.labels.label_file = hydra_cfg.dataset.train.prompt_eng_file
        cfg.model.category_head.labels.dataset = hydra_cfg.dataset.train.name
    cfg.model.num_queries = hydra_cfg.model.num_queries
    cfg.model.object_mask_threshold = hydra_cfg.model.object_mask_threshold
    cfg.model.overlap_threshold = hydra_cfg.model.overlap_threshold
    cfg.model.test_topk_per_image = hydra_cfg.model.test_topk_per_image

    if hydra_cfg.model.type == 'category':
        cfg.model.sem_seg_head.num_classes = hydra_cfg.model.num_classes
    else:
        cfg.model.sem_seg_head.num_classes = 1
    if hydra_cfg.model.name == 'convnext_large_d_320':
        cfg.model.sem_seg_head.transformer_predictor.post_mask_embed.projection_dim = 768
    elif hydra_cfg.model.name == 'convnext_xxlarge':
        cfg.model.sem_seg_head.transformer_predictor.post_mask_embed.projection_dim = 1024
    else:
        raise NotImplementedError
    cfg.model.sem_seg_head.transformer_predictor.mask_dim = hydra_cfg.model.mask_dim
    cfg.model.sem_seg_head.transformer_predictor.hidden_dim = hydra_cfg.model.hidden_dim
    cfg.model.metadata.name = hydra_cfg.dataset.train.name + "_with_sem_seg"

    cfg.dataloader.train.dataset.names = hydra_cfg.dataset.train.name + "_with_sem_seg"
    cfg.dataloader.test.dataset.names = hydra_cfg.dataset.val.name + "_with_sem_seg"
    cfg.dataloader.wrapper.labels.label_file = hydra_cfg.dataset.train.prompt_eng_file
    cfg.dataloader.wrapper.labels.dataset = hydra_cfg.dataset.train.name
    return cfg
