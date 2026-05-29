# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config template and utils for PointPillars."""
from pathlib import Path

import yaml
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    """Log config to file."""
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:  # noqa: E722
            value = v

        if (not isinstance(value, type(d[subkey]))) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif isinstance(value, type(d[subkey])) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert isinstance(value, type(d[subkey])), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def merge_new_config(config, new_config):
    """Merge new config."""
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)  # nosec
            except:  # noqa: E722
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    """Parse config from yaml file."""
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)  # nosec
        merge_new_config(config=config, new_config=new_config)
    # Set defaults for optional parameters
    if not hasattr(config.train, "resume_training_checkpoint_path"):
        config.train.resume_training_checkpoint_path = None
    if not hasattr(config.model, "pretrained_model_path"):
        config.model.pretrained_model_path = None
    if not hasattr(config.train, "pruned_model_path"):
        config.train.pruned_model_path = None
    if not hasattr(config.train, "random_seed"):
        config.train.random_seed = None
    if not hasattr(config, "results_dir"):
        config.results_dir = None
    if not hasattr(config, "class_names"):
        config.class_names = config.dataset.class_names
    if not hasattr(config, "export"):
        config.export = EasyDict()
    if not hasattr(config.export, "onnx_file"):
        config.export.onnx_file = None
    if not hasattr(config.export, "checkpoint"):
        config.export.checkpoint = None
    if not hasattr(config.export, "gpu_id"):
        config.export.gpu_id = None
    if not hasattr(config, "prune"):
        config.prune = EasyDict()
    if not hasattr(config.prune, "model"):
        config.prune.model = None
    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
