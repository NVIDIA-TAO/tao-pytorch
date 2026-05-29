# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build torch data loader."""
import os
import numpy as np
import random
from functools import partial
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from nvidia_tao_pytorch.cv.action_recognition.dataloader.ar_dataset import MotionDataset, SpatialDataset, FuseDataset
from nvidia_tao_pytorch.cv.action_recognition.dataloader.frame_sampler import (random_interval_sample,
                                                                               random_consecutive_sample,
                                                                               test_interval_sample,
                                                                               test_consecutive_sample,
                                                                               joint_random_interval_sample,
                                                                               joint_random_consecutive_sample,
                                                                               joint_test_interval_sample,
                                                                               joint_test_consecutive_sample)
from nvidia_tao_pytorch.cv.action_recognition.utils.group_transforms import (GroupNormalize,
                                                                             GroupWorker,
                                                                             GroupRandomHorizontalFlip,
                                                                             GroupRandomCrop,
                                                                             MultiScaleCrop,
                                                                             ToNumpyNDArray,
                                                                             ToTorchFormatTensor,
                                                                             GroupJointWorker,
                                                                             GroupJointRandomCrop,
                                                                             JointMultiScaleCrop,
                                                                             GroupJointRandomHorizontalFlip,
                                                                             JointWorker,
                                                                             GroupJointNormalize)


def split_shad_dataset(top_dir, val_percent):
    """Randomly split the orginal SHAD train dataset to train/val.
    And validation dataset takes val_ratio part of the whole dataset.
    """
    action_set = os.listdir(top_dir)
    sample_list = []
    for action in action_set:
        action_root_path = os.path.join(top_dir, action)
        for video in os.listdir(action_root_path):
            video_path = os.path.join(action_root_path, video)
            for sample in os.listdir(video_path):
                sample_path = os.path.join(video_path, sample)
                sample_list.append((sample_path, action))

    total_sample_cnt = len(sample_list)
    index_list = list(range(total_sample_cnt))
    random.shuffle(index_list)
    val_sample_cnt = int(total_sample_cnt * val_percent)
    train_samples = {}
    val_samples = {}
    for idx in index_list[:val_sample_cnt]:
        sample_path, action_label = sample_list[idx]
        val_samples[sample_path] = action_label

    for idx in index_list[val_sample_cnt:total_sample_cnt]:
        sample_path, action_label = sample_list[idx]
        train_samples[sample_path] = action_label

    return train_samples, val_samples


def split_dataset(top_dir, val_percent):
    """Randomly split the original train dataset into train and validation sets.

    This function takes the top-level directory of the dataset and the validation percentage as input. It randomly
    splits the original train dataset into train and validation sets, where the validation set takes up val_percent
    of the whole dataset. The function returns the file paths of the train and validation sets as lists.

    Args:
        top_dir (str): The top-level directory of the dataset.
        val_percent (float): The percentage of the dataset to be used for validation.

    Returns:
        tuple: A tuple containing the file paths of the train and validation sets as lists.
    """
    action_set = os.listdir(top_dir)
    sample_list = []
    for action in action_set:
        action_root_path = os.path.join(top_dir, action)
        for video in os.listdir(action_root_path):
            video_path = os.path.join(action_root_path, video)
            sample_list.append((video_path, action))

    total_sample_cnt = len(sample_list)
    index_list = list(range(total_sample_cnt))
    random.shuffle(index_list)
    val_sample_cnt = int(total_sample_cnt * val_percent)
    train_samples = {}
    val_samples = {}
    for idx in index_list[:val_sample_cnt]:
        sample_path, action_label = sample_list[idx]
        val_samples[sample_path] = action_label

    for idx in index_list[val_sample_cnt:total_sample_cnt]:
        sample_path, action_label = sample_list[idx]
        train_samples[sample_path] = action_label

    return train_samples, val_samples


def list_dataset(top_dir):
    """Generate the sample_dict from top_dir.
    Args:
        top_dir (str): The top-level directory of the dataset.

    Returns:
        dict: A dictionary containing video patch paths as keys and action labels as values.
    """
    action_set = os.listdir(top_dir)
    sample_dict = {}
    for action in action_set:
        action_root_path = os.path.join(top_dir, action)
        for video in os.listdir(action_root_path):
            video_path = os.path.join(action_root_path, video)
            sample_dict[video_path] = action

    return sample_dict


def get_clips_list(sample_dict, seq_len, eval_mode,
                   sampler_strategy="random_interval",
                   sample_rate=1, num_segments=1, dir_path="rgb"):
    """Get a list of clips covering all the frames in the dataset.

    This function takes a sample dictionary, sequence length, evaluation mode, sampler strategy, sample rate,
    number of segments, and directory path as input. It generates a list of clips covering all the frames in the
    dataset based on the specified parameters. The function returns two lists: one containing the file paths of
    the clips, and one containing the corresponding image IDs of the frames in the clips.

    Args:
        sample_dict (dict): A dictionary containing video patch paths as keys and action labels as values.
        seq_len (int): The sequence length of the clips.
        eval_mode (str): The evaluation mode, either "conv" or "all".
        sampler_strategy (str, optional): The sampler strategy, either "random_interval" or "consecutive".
            Defaults to "random_interval".
        sample_rate (int, optional): The sample rate for the frames. Defaults to 1.
        num_segments (int, optional): The number of segments for the clips when using conv mode. Defaults to 1.
        dir_path (str, optional): The directory path of the frames. Defaults to "rgb".

    Returns:
        tuple: A tuple containing two lists: one containing the file paths of the clips, and one containing
            the corresponding image IDs of the frames in the clips.

    Raises:
        ValueError: If the sampler strategy is not supported.
    """
    sample_path_list = list(sample_dict.keys())

    sample_clips_path = []
    # tuple of image_ids
    sample_clips_ids = []
    if sampler_strategy == "random_interval":
        for sample_path in sample_path_list:
            total_frames = len(os.listdir(os.path.join(sample_path, dir_path)))
            # interval
            seq_interval = total_frames // seq_len
            if seq_interval > 0:
                for j in range(seq_interval):
                    sample_clips_path.append(sample_path)
                    sample_clips_ids.append(np.sort((np.arange(seq_len) * seq_interval + j)))
            else:
                sample_clips_path.append(sample_path)
                img_ids = np.arange(seq_len)
                img_ids = np.minimum(img_ids, total_frames - 1)
                sample_clips_ids.append(img_ids)
    # num segments and eval_mode only works for consecutive sampler strategy
    elif sampler_strategy == "consecutive":
        if eval_mode == "conv":
            for sample_path in sample_path_list:
                total_frames = len(os.listdir(os.path.join(sample_path, dir_path)))
                orig_len = seq_len * sample_rate
                if total_frames > orig_len - 1:
                    tick = (total_frames - orig_len + 1) / float(num_segments)
                    offsets = np.array([int(tick / 2.0 + tick * x)
                                        for x in range(num_segments)])
                else:
                    offsets = np.zeros((1,))

                for offset in offsets:
                    sample_clips_path.append(sample_path)
                    img_ids = offset + np.arange(seq_len) * sample_rate
                    # img_ids = np.minimum(img_ids, total_frames-1)
                    img_ids = np.mod(img_ids, total_frames)
                    img_ids = img_ids.astype(np.int32)
                    sample_clips_ids.append(img_ids)
        elif eval_mode == "all":
            for sample_path in sample_path_list:
                total_frames = len(os.listdir(os.path.join(sample_path, dir_path)))
                orig_len = seq_len * sample_rate
                num_clips = total_frames // orig_len
                if num_clips > 0:
                    for j in range(num_clips):
                        for i in range(sample_rate):
                            sample_clips_path.append(sample_path)
                            sample_clips_ids.append(list(range(j * orig_len + i, (j + 1) * orig_len, sample_rate)))
                else:
                    sample_clips_path.append(sample_path)
                    img_ids = np.arange(seq_len) * sample_rate
                    # img_ids = np.minimum(img_ids, total_frames-1)
                    img_ids = np.mod(img_ids, total_frames)
                    sample_clips_ids.append(img_ids)
    else:
        raise ValueError("Only supports sample strategy [random_interval, consecutive].")

    return sample_clips_path, sample_clips_ids


def build_joint_augmentation_pipeline(output_shape, augmentation_config,
                                      dataset_mode="train"):
    """Build an augmentation pipeline for a joint model.

    This function takes the output shape, augmentation configuration, and dataset mode as input. It builds an
    augmentation pipeline for a joint model based on the specified parameters. The function returns the pipeline
    as a `transforms.Compose` object.

    Args:
        output_shape (tuple): The output shape of the joint model.
        augmentation_config (dict): The augmentation configuration for the joint model.
        dataset_mode (str, optional): The dataset mode, either "train", "val", or "inf". Defaults to "train".

    Returns:
        transforms: The augmentation pipeline as a `transforms.Compose` object.

    Raises:
        ValueError: If the dataset mode is not supported.
    """
    rgb_input_mean = list(augmentation_config["rgb_input_mean"])
    rgb_input_std = list(augmentation_config["rgb_input_std"])
    of_input_mean = list(augmentation_config["of_input_mean"])
    of_input_std = list(augmentation_config["of_input_std"])

    output_height, output_width = output_shape
    smaller_edge = augmentation_config["crop_smaller_edge"]

    transforms_list = []
    if dataset_mode in ["val", "inf"]:
        if augmentation_config["val_center_crop"]:
            # resize to smaller size 256
            transforms_list.append(GroupJointWorker(transforms.Resize(int(smaller_edge))))
            # center crop :
            transforms_list.append(GroupJointWorker(transforms.CenterCrop([output_height, output_width])))
            # transforms_list.append(GroupThreeCrop([output_height, output_width]))  # [3*64, 256, 256, 3]
        else:
            # simply resize to target size
            transforms_list.append(GroupJointWorker(transforms.Resize([output_height, output_width])))
    elif dataset_mode == "train":
        if augmentation_config["train_crop_type"] == "multi_scale_crop":
            scales = augmentation_config["scales"]
            transforms_list.append(JointMultiScaleCrop([output_width, output_height],
                                                       scales))
        elif augmentation_config["train_crop_type"] == "random_crop":
            # @TODO(tylerz): enable joint training experiments with .png, remove 340 later
            transforms_list.append(GroupJointWorker(transforms.Resize(int(smaller_edge))))
            # transforms_list.append(GroupJointWorker(transforms.Resize((int(smaller_edge), 340))))
            transforms_list.append(GroupJointRandomCrop((output_height, output_width)))
        else:
            transforms_list.append(GroupJointWorker(transforms.Resize([output_height, output_width])))

        if augmentation_config["horizontal_flip_prob"] > 0.0:
            prob = min(1.0, augmentation_config["horizontal_flip_prob"])
            transforms_list.append(GroupJointRandomHorizontalFlip(prob))
    else:
        raise ValueError('There are only train, val, inf mode.')

    transforms_list.append(JointWorker(ToNumpyNDArray()))
    transforms_list.append(JointWorker(ToTorchFormatTensor()))
    if (len(rgb_input_mean) != 0 or len(rgb_input_std) != 0 or
       len(of_input_mean) != 0 or len(of_input_std) != 0):
        transforms_list.append(GroupJointNormalize(rgb_input_mean, rgb_input_std,
                                                   of_input_mean, of_input_std))

    transform = transforms.Compose(transforms_list)
    return transform


def build_single_augmentation_pipeline(output_shape, augmentation_config,
                                       dataset_type="rgb", dataset_mode="train"):
    """Build a single stream augmentation pipeline.

    This function takes the output shape, augmentation configuration, dataset type, and dataset mode as input. It
    builds a single stream augmentation pipeline based on the specified parameters. The function returns the
    pipeline as a `transforms.Compose` object.

    Args:
        output_shape (tuple): The output shape of the single stream model.
        augmentation_config (dict): The augmentation configuration for the single stream model.
        dataset_type (str, optional): The dataset type, either "rgb" or "flow". Defaults to "rgb".
        dataset_mode (str, optional): The dataset mode, either "train", "val", or "inf". Defaults to "train".

    Returns:
        transforms.Compose: The augmentation pipeline as a `transforms.Compose` object.

    Raises:
        ValueError: If the dataset type or mode is not supported.
    """
    if dataset_type == "rgb":
        input_mean = list(augmentation_config["rgb_input_mean"])
        input_std = list(augmentation_config["rgb_input_std"])
    elif dataset_type == "of":
        input_mean = list(augmentation_config["of_input_mean"])
        input_std = list(augmentation_config["of_input_std"])
    else:
        ValueError(("Only the type in [of, rgb] is supported for single input pipeline"))

    output_height, output_width = output_shape
    smaller_edge = augmentation_config["crop_smaller_edge"]

    transforms_list = []
    if dataset_mode in ["val", "inf"]:
        if augmentation_config["val_center_crop"]:
            # resize to smaller size 256
            # transforms_list.append(GroupWorker(transforms.Resize(int(smaller_edge * 256 / 224))))
            transforms_list.append(GroupWorker(transforms.Resize(int(smaller_edge))))
            # center crop :
            transforms_list.append(GroupWorker(transforms.CenterCrop([output_height, output_width])))
            # transforms_list.append(GroupThreeCrop([output_height, output_width]))  # [3*64, 256, 256, 3]
        else:
            # simply resize to target size
            transforms_list.append(GroupWorker(transforms.Resize([output_height, output_width])))
    elif dataset_mode == "train":
        if augmentation_config["train_crop_type"] == "multi_scale_crop":
            transforms_list.append(GroupWorker(transforms.Resize(int(smaller_edge))))
            scales = augmentation_config["scales"]
            transforms_list.append(MultiScaleCrop([output_width, output_height],
                                                  scales))
        elif augmentation_config["train_crop_type"] == "random_crop":
            transforms_list.append(GroupWorker(transforms.Resize(int(smaller_edge))))
            transforms_list.append(GroupRandomCrop((output_height, output_width)))
        else:
            transforms_list.append(GroupWorker(transforms.Resize([output_height, output_width])))

        if augmentation_config["horizontal_flip_prob"] > 0.0:
            prob = min(1.0, augmentation_config["horizontal_flip_prob"])
            if dataset_type == "rgb":
                transforms_list.append(GroupRandomHorizontalFlip(prob))
            elif dataset_type == "of":
                transforms_list.append(GroupRandomHorizontalFlip(flip_prob=prob, is_flow=True))
            else:
                raise ValueError("Single branch augmentation pipeline only supports rgb, of.")
    else:
        raise ValueError('There are only train, val, inf mode.')

    transforms_list.append(ToNumpyNDArray())
    transforms_list.append(ToTorchFormatTensor())
    if len(input_mean) != 0 or len(input_std) != 0:
        transforms_list.append(GroupNormalize(input_mean, input_std))

    transform = transforms.Compose(transforms_list)
    return transform


def build_single_sampler(sampler_strategy, sample_rate=1, all_frames_3d=False):
    """Build a frames sampler for a single branch model.

    This function takes the sampler strategy, sample rate, and all frames 3D flag as input. It builds a frames
    sampler for a single branch model based on the specified parameters. The function returns two samplers: one
    for training and one for testing.

    Args:
        sampler_strategy (str): The sampler strategy, either "random_interval" or "consecutive".
        sample_rate (int, optional): The sample rate for the frames. Defaults to 1.
        all_frames_3d (bool, optional): The flag indicating whether to take all frames for 3D model. Defaults to False.

    Returns:
        tuple: A tuple containing two samplers: one for training and one for testing.

    Raises:
        ValueError: If the sampler strategy is not supported.
    """
    if sampler_strategy == "random_interval":
        train_sampler = random_interval_sample
        test_sampler = test_interval_sample
    elif sampler_strategy == "consecutive":
        train_sampler = partial(random_consecutive_sample, sample_rate=sample_rate)
        test_sampler = partial(test_consecutive_sample,
                               sample_rate=sample_rate,
                               all_frames_3d=all_frames_3d)
    else:
        raise ValueError("Only supports [random_interval, consecutive] sample strategy")

    return train_sampler, test_sampler


def build_joint_sampler(sampler_strategy, sample_rate=1, all_frames_3d=False):
    """Build frames sampler for joint model."""
    if sampler_strategy == "random_interval":
        train_sampler = joint_random_interval_sample
        test_sampler = joint_test_interval_sample
    elif sampler_strategy == "consecutive":
        train_sampler = partial(joint_random_consecutive_sample, sample_rate=sample_rate)
        test_sampler = partial(joint_test_consecutive_sample,
                               sample_rate=sample_rate,
                               all_frames_3d=all_frames_3d)
    else:
        raise ValueError("Only supports [random_interval, consecutive] sample strategy")

    return train_sampler, test_sampler


def build_dataloader(sample_dict, model_config,
                     output_shape, label_map, augmentation_config,
                     dataset_mode="inf", batch_size=1, workers=4,
                     input_type="2d", shuffle=False, pin_mem=False,
                     eval_mode="center", num_segments=1,
                     clips_per_video=1):
    """Build a torch dataloader.

    This function takes the sample dictionary, model configuration, output shape, label map, augmentation
    configuration, dataset mode, batch size, number of workers, input type, shuffle flag, pin memory flag,
    evaluation mode, number of segments, and clips per video as input. It builds a torch dataloader based on
    the specified parameters and returns it.

    Args:
        sample_dict (dict): A dictionary containing video patch paths as keys and action labels as values.
        model_config (dict): The model configuration.
        output_shape (tuple): The output shape.
        label_map (dict): A dictionary mapping labels to their corresponding indices.
        augmentation_config (dict): The augmentation configuration.
        dataset_mode (str, optional): The dataset mode, could be "train", "val" or "inf". Defaults to "inf".
        batch_size (int, optional): The batch size. Defaults to 1.
        workers (int, optional): The number of workers. Defaults to 4.
        input_type (str, optional): The input type, either "2d" or "3d". Defaults to "2d".
        shuffle (bool, optional): The shuffle flag. Defaults to False.
        pin_mem (bool, optional): The pin memory flag. Defaults to False.
        eval_mode (str, optional): The evaluation mode for evaluation dataset, either "conv", "center" or "all". Defaults to "center".
        num_segments (int, optional): The number of segments when using full clip. Defaults to 1.
        clips_per_video (int, optional): The number of clips to be extracted from each video. Defaults to 1.

    Returns:
        torch.utils.data.DataLoader: The torch dataloader.

    Raises:
        ValueError: If the dataset type is not supported.
    """
    dataset_type = model_config["model_type"]
    train_sampler, test_sampler = build_single_sampler(model_config['sample_strategy'],
                                                       model_config['sample_rate'])
    sample_clips_path = None
    sample_clips_id = None
    full_clip = False

    if dataset_type == "of":
        aug_transform = build_single_augmentation_pipeline(output_shape=output_shape,
                                                           augmentation_config=augmentation_config,
                                                           dataset_mode=dataset_mode,
                                                           dataset_type="of")

        if eval_mode == "conv":
            full_clip = True
            sample_clips_path, sample_clips_id = \
                get_clips_list(sample_dict,
                               model_config["of_seq_length"],
                               eval_mode=eval_mode,
                               sampler_strategy=model_config['sample_strategy'],
                               sample_rate=model_config['sample_rate'],
                               num_segments=num_segments,
                               dir_path="u")
        elif eval_mode == "all":
            if input_type == "2d":
                full_clip = True
                sample_clips_path, sample_clips_id = \
                    get_clips_list(sample_dict,
                                   model_config["of_seq_length"],
                                   eval_mode=eval_mode,
                                   sampler_strategy=model_config['sample_strategy'],
                                   sample_rate=model_config['sample_rate'],
                                   dir_path="u")
            elif input_type == "3d":
                train_sampler, test_sampler = \
                    build_single_sampler(model_config['sample_strategy'],
                                         model_config['sample_rate'],
                                         all_frames_3d=True)

        dataset = MotionDataset(sample_dict=sample_dict,
                                output_shape=output_shape,
                                seq_length=model_config["of_seq_length"],
                                input_type=input_type,
                                label_map=label_map,
                                mode=dataset_mode,
                                full_clip=full_clip,
                                clips_per_video=clips_per_video,
                                transform=aug_transform,
                                train_sampler=train_sampler,
                                test_sampler=test_sampler,
                                sample_clips_path=sample_clips_path,
                                sample_clips_ids=sample_clips_id)
    elif dataset_type == "rgb":
        aug_transform = build_single_augmentation_pipeline(output_shape=output_shape,
                                                           augmentation_config=augmentation_config,
                                                           dataset_mode=dataset_mode,
                                                           dataset_type="rgb")
        if eval_mode == "conv":
            full_clip = True
            sample_clips_path, sample_clips_id = \
                get_clips_list(sample_dict,
                               model_config["rgb_seq_length"],
                               eval_mode=eval_mode,
                               sampler_strategy=model_config['sample_strategy'],
                               sample_rate=model_config['sample_rate'],
                               num_segments=num_segments,
                               dir_path="rgb")
        elif eval_mode == "all":
            if input_type == "2d":
                full_clip = True
                sample_clips_path, sample_clips_id = \
                    get_clips_list(sample_dict,
                                   model_config["rgb_seq_length"],
                                   eval_mode=eval_mode,
                                   sampler_strategy=model_config['sample_strategy'],
                                   sample_rate=model_config['sample_rate'],
                                   dir_path="rgb")
            elif input_type == "3d":
                train_sampler, test_sampler = \
                    build_single_sampler(model_config['sample_strategy'],
                                         model_config['sample_rate'],
                                         all_frames_3d=True)

        dataset = SpatialDataset(sample_dict=sample_dict,
                                 output_shape=output_shape,
                                 seq_length=model_config["rgb_seq_length"],
                                 input_type=input_type,
                                 label_map=label_map,
                                 mode=dataset_mode,
                                 full_clip=full_clip,
                                 clips_per_video=clips_per_video,
                                 transform=aug_transform,
                                 train_sampler=train_sampler,
                                 test_sampler=test_sampler,
                                 sample_clips_path=sample_clips_path,
                                 sample_clips_ids=sample_clips_id)
    elif dataset_type == "joint":
        train_sampler, test_sampler = \
            build_joint_sampler(model_config['sample_strategy'],
                                model_config['sample_rate'])

        aug_transform = \
            build_joint_augmentation_pipeline(output_shape=output_shape,
                                              augmentation_config=augmentation_config,
                                              dataset_mode=dataset_mode)

        larger_seq = max(model_config["rgb_seq_length"],
                         model_config["of_seq_length"])
        if eval_mode == "conv":
            full_clip = True
            sample_clips_path, sample_clips_id = \
                get_clips_list(sample_dict,
                               larger_seq,
                               eval_mode=eval_mode,
                               sampler_strategy=model_config['sample_strategy'],
                               sample_rate=model_config['sample_rate'],
                               num_segments=num_segments,
                               dir_path="u")
        elif eval_mode == "all":
            if input_type == "2d":
                full_clip = True
                sample_clips_path, sample_clips_id = \
                    get_clips_list(sample_dict,
                                   larger_seq,
                                   eval_mode=eval_mode,
                                   sampler_strategy=model_config['sample_strategy'],
                                   sample_rate=model_config['sample_rate'],
                                   dir_path="u")
            elif input_type == "3d":
                train_sampler, test_sampler = \
                    build_joint_sampler(model_config['sample_strategy'],
                                        model_config['sample_rate'],
                                        all_frames_3d=True)

        dataset = FuseDataset(sample_dict=sample_dict,
                              output_shape=output_shape,
                              of_seq_length=model_config["of_seq_length"],
                              rgb_seq_length=model_config["rgb_seq_length"],
                              input_type=input_type,
                              label_map=label_map,
                              mode=dataset_mode,
                              full_clip=full_clip,
                              clips_per_video=clips_per_video,
                              transform=aug_transform,
                              train_sampler=train_sampler,
                              test_sampler=test_sampler,
                              sample_clips_path=sample_clips_path,
                              sample_clips_ids=sample_clips_id)
    else:
        raise ValueError("Only the type in [of, rgb, joint] is supported")

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=workers,
                            pin_memory=pin_mem)
    return dataloader
