# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" Build Dataloader for OCRNet """

import argparse
import torch
from nvidia_tao_pytorch.cv.ocrnet.dataloader.ocr_dataset import (LmdbDataset,
                                                                 RawGTDataset,
                                                                 AlignCollate,
                                                                 AlignCollateVal)


def translate_dataset_config(experiment_spec):
    """Translate experiment spec to match with CLOVA"""
    parser = argparse.ArgumentParser()
    opt, _ = parser.parse_known_args()

    opt.exp_name = experiment_spec.results_dir
    # 1. Init dataset params
    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    # Support single dataset source now
    # Shall we check it with output feature length to avoid Nan in CTC Loss?
    # (image_width // stride) >= 2 * max_label_length - 1
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel
    if dataset_config.augmentation.keep_aspect_ratio:
        opt.PAD = True
    else:
        opt.PAD = False

    if model_config.input_channel == 3:
        opt.rgb = True
    else:
        opt.rgb = False
    # load character list:
    # Don't convert the characters to lower case
    with open(dataset_config.character_list_file, "r") as f:
        characters = "".join([ch.strip() for ch in f.readlines()])
    opt.character = characters

    # hardcode the data_filtering_off to be True.
    # And there will be KeyError when encoding the labels if
    # the labels and character list cannot match
    opt.data_filtering_off = True

    opt.workers = dataset_config.workers
    opt.batch_size = dataset_config.batch_size

    return opt


def build_dataloader(experiment_spec, data_path, shuffle=True, gt_file=None):
    """Build dataloader for training and validation.

    Args:
        experiment_spec (dict): A dictionary of experiment specifications.
        data_path (str): The path to the dataset.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        gt_file (str, optional): The path to the ground truth file. Default is None.

    Returns:
        torch.utils.data.DataLoader: A dataloader for the dataset.
    """
    opt = translate_dataset_config(experiment_spec)

    if shuffle:
        AlignCollate_func = AlignCollate(experiment_spec=experiment_spec, imgH=opt.imgH, imgW=opt.imgW,
                                         keep_ratio_with_pad=opt.PAD)
    else:
        AlignCollate_func = AlignCollateVal(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    if gt_file is not None:
        dataset = RawGTDataset(gt_file, data_path, opt)
    else:
        dataset = LmdbDataset(data_path, opt)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_func, pin_memory=True)

    return data_loader
