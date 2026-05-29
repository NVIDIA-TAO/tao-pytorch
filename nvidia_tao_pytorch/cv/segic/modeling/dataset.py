# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gives the dataset class for inference."""

import os

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import open_clip
import numpy as np


def resize_item(sample, size=(320, 320)):
    """Resizes the image and label to the given size."""
    imidx, image, label = sample['imidx'], sample['image'], sample['label']
    ori_size = size
    size = max(ori_size)
    scale_factor = size / max(image.shape[-2:])
    if image.shape[-2:] != label.shape[-2:]:
        print(imidx)
    image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0),
                                        scale_factor=scale_factor,
                                        mode='bilinear'), dim=0)
    label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0),
                                        scale_factor=scale_factor,
                                        mode='bilinear'), dim=0)
    padding_h = max(size - image.size(1), 0)
    padding_w = max(size - image.size(2), 0)
    ori_size_prepad = image.shape[-2:]
    image = F.pad(image, [0, padding_w, 0, padding_h], value=128)
    label = F.pad(label, [0, padding_w, 0, padding_h], value=0)

    aug_sample = {'imidx': imidx, 'image': image, 'label': label,
                  'shape': torch.tensor(ori_size)}
    aug_sample.update(class_name=sample.get('class_name', ''))
    aug_sample['ori_size'] = torch.tensor(ori_size_prepad)

    sample.update(aug_sample)
    return sample


class InferenceDataset(Dataset):
    """Creates Dataset for Gradio inference."""

    def __init__(self, image_input, seg_input=None, meta_input=None, size=(320, 320)):
        """
        args:
            image_input: path to the image file or directory
            seg_input: path to the segmentation file or directory
            meta_input: a list of meta data for each image
        """
        if isinstance(image_input, list):
            if isinstance(seg_input, list):
                assert len(image_input) == len(seg_input), \
                    "image_input and seg_input should have the same length"
                self.image_files_list = image_input
                self.seg_files_list = seg_input
            else:
                self.seg_files_list = [None] * len(image_input)

            if meta_input is not None:
                assert len(meta_input) == len(image_input), \
                    "meta data should be provided for each image"
                self.meta_list = meta_input
            else:
                self.meta_list = ['instance'] * len(image_input)

        elif os.path.isfile(image_input):
            self.image_files_list = [image_input]
            if seg_input is None:
                self.seg_files_list = [None]
            elif os.path.isfile(seg_input):
                self.seg_files_list = [seg_input]
            else:
                raise ValueError("seg_input should be a file or None when image_input is a file")

            if meta_input is not None:
                self.meta_list = meta_input
            else:
                self.meta_list = ['instance']

        else:
            raise ValueError("image_input should be a file or a list of files")

        self.size = size

        self.tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP')

    def __len__(self):
        """Length of the dataset."""
        return len(self.image_files_list)

    def __getitem__(self, index):
        """Returns the item at the given index."""
        image_file = self.image_files_list[index]
        meta = self.meta_list[index]

        np_image = np.array(Image.open(image_file).convert('RGB'))
        img = torch.tensor(np_image, dtype=torch.float32).permute(2, 0, 1)

        if self.seg_files_list[index] is None:
            mask = torch.zeros_like(img)[:1]
        else:
            seg_file = self.seg_files_list[index]
            np_mask = np.array(Image.open(seg_file).convert('L'))
            mask = torch.tensor(np_mask, dtype=torch.float32).unsqueeze(0)

        shape = img.shape[-2:]
        imidx = os.path.basename(image_file).split('.')[0]
        sample = {'image': img,
                  'label': mask,
                  'imidx': imidx,
                  'shape': shape,
                  'ori_label': mask,
                  'ori_img': np_image,
                  'ori_size': shape,
                  }

        sample['is_inst'] = meta == 'instance'
        sample['class_name'] = '' if meta == 'instance' else meta
        sample['token'] = self.prepare_prompt_feature(sample['class_name'])
        sample = resize_item(sample, self.size)

        return sample

    def prepare_prompt_feature(self, class_name):
        """
        Prepares the meta description for the prompt and tokenizes it.

        Args:
            class_name(str): The class name to be segmented.

        Returns:
            torch.Tensor: The tokenized prompt.
        """
        if class_name is None or class_name == '':
            prompt = ['please segment the instances']
        else:
            prompt = [f'a photo of a {class_name}']

        return self.tokenizer(prompt).squeeze(0)
