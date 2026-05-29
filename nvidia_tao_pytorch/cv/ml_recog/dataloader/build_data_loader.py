# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build torch data loader."""

import yaml
from torch.utils.data import DataLoader
from pytorch_metric_learning import samplers

from nvidia_tao_pytorch.cv.ml_recog.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.ml_recog.dataloader.datasets.inference_datasets import InferenceImageFolder, InferenceImageDataset
from nvidia_tao_pytorch.cv.ml_recog.dataloader.datasets.image_datasets import MetricLearnImageFolder


def build_dataloader(cfg, mode="train"):
    """Build torch dataloader.

    Args:
        cfg (DictConfig): Hydra config object
        mode (str): Choice between 'train', 'eval', or 'inference'

    Returns:
        train_loader (Dataloader): Train dataloader
        query_loader (Dataloader): Val dataloader, used for query jobs in validation or test
        gallery_loader (Dataloader): Val dataloader, used for reference job in validation or test
        dataset_dict (Dict): a dictionary of train, query and gallery datasets

    """
    assert mode in ["train", "eval", "inference"], "mode can only be train, eval or inference"
    dataset_configs = cfg["dataset"]
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = dataset_configs["workers"]
    train_loader, query_loader, gallery_loader = None, None, None
    train_dataset = None
    query_dataset = None
    class_mapping = None

    if dataset_configs["class_map"]:
        with open(dataset_configs["class_map"], "r") as f:
            class_mapping = yaml.load(f, Loader=yaml.FullLoader)

    if mode == "train":
        train_transforms = build_transforms(cfg, is_train=True)

        train_dataset = MetricLearnImageFolder(dataset_configs['train_dataset'],
                                               transform=train_transforms,
                                               class_mapping=class_mapping)

        sampler = samplers.MPerClassSampler(train_dataset.targets,
                                            m=cfg["dataset"]["num_instance"],
                                            batch_size=cfg["train"]["batch_size"],
                                            length_before_new_iter=len(train_dataset))

        train_loader = DataLoader(
            train_dataset, batch_size=cfg["train"]["batch_size"],
            sampler=sampler,
            num_workers=num_workers
        )
        val_batch_size = cfg["train"]["val_batch_size"]

    elif mode == "inference":
        val_batch_size = cfg["inference"]["batch_size"]

    elif mode == "eval":
        val_batch_size = cfg["evaluate"]["batch_size"]

    gallery_dataset = MetricLearnImageFolder(dataset_configs["val_dataset"]["reference"],
                                             transform=val_transforms,
                                             class_mapping=class_mapping)

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=num_workers
    )

    if mode in ("eval", "train"):
        # inference mode has query folder as inference.input_path
        query_dataset = MetricLearnImageFolder(dataset_configs["val_dataset"]["query"],
                                               transform=val_transforms,
                                               class_to_idx=gallery_dataset.class_to_idx,
                                               classes=gallery_dataset.classes,
                                               class_mapping=class_mapping)

        query_loader = DataLoader(
            query_dataset, batch_size=val_batch_size,
            shuffle=False, num_workers=num_workers
        )

    dataset_dict = {
        "train": train_dataset,
        "query": query_dataset,
        "gallery": gallery_dataset
    }

    return train_loader, query_loader, gallery_loader, dataset_dict


def build_inference_dataloader(cfg):
    """Create a dataloader for inference task.

    Args:
        cfg (DictConfig): Hydra config object

    Returns:
        dataloader (InferenceImageFolder / InferenceImageDataset): If
            cfg.inference.input_path is a classification folder and
            cfg.inference.inference_input_type is correctly marked as
            `classification_folder`, it returns a InferenceImageFolder. If
            cfg.inference.input_path is a folder of images and
            cfg.inference.inference_input_type is correctly marked as `image_folder`,
            it returns a InferenceImageDataset
    """
    image_folder_type = cfg["inference"]["inference_input_type"]

    if image_folder_type == "classification_folder":
        dataset_builder = InferenceImageFolder

    elif image_folder_type == "image_folder":
        dataset_builder = InferenceImageDataset

    val_transform = build_transforms(cfg, is_train=False)
    inference_dataset = dataset_builder(
        cfg['inference']['input_path'],
        transform=val_transform)

    dataloader = DataLoader(inference_dataset,
                            batch_size=cfg['inference']['batch_size'],
                            shuffle=False)

    return dataloader
