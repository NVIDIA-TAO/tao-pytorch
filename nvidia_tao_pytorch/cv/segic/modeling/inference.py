# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gives Inference Module for SegIC."""

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

from nvidia_tao_pytorch.cv.segic.modeling import utils
from nvidia_tao_pytorch.cv.segic.modeling.dataset import InferenceDataset


class SegicTRTInferencer:
    """Runs in-context segmentation by SegIC."""

    def __init__(self, feature_extract_trt, segic_trt, inference_size=(896, 896), device='cuda'):
        """Initializes the SegIC TRT Inferencer.

        Args:
            feature_extract_trt (str): Path to the prompt feature extractor TRT engine.
            segic_trt (str): Path to the SegIC TRT engine.
            inference_size (Tuple[Int]): Size of the inference image.
            device (str): Device to run inference on.
        """
        self.feature_extract_trt = feature_extract_trt
        self.segic_trt = segic_trt
        self.inference_size = inference_size
        self.device = device
        self.data_meta = None

        # mask decoding on target images
        load_segic_engine = EngineFromBytes(BytesFromPath(self.segic_trt))
        self.trt_engine_runner = TrtRunner(load_segic_engine())

        load_prompt_engine = EngineFromBytes(BytesFromPath(self.feature_extract_trt))
        self.trt_engine_runner_prompt = TrtRunner(load_prompt_engine())

    def inference_batch(self, target_input):
        """
        Run batch inference on target images using the extracted prompt features.

        Args:
            target_input (str/List(str)): Path to the target image or a list of image paths.

        Returns:
            List(np.ndarray): List of images with segmentation masks overlayed.
        """
        if self.data_meta is None:
            raise ValueError("Please extract prompt features before running inference.")
        inference_dataset = InferenceDataset(target_input, None, None, size=self.inference_size)
        inference_dl = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=32)

        results = []
        for data_inf in inference_dl:
            if self.device == 'cuda' and torch.cuda.is_available():
                data_inf = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in
                            data_inf.items()}
            with self.trt_engine_runner as runner:
                output = runner.infer(feed_dict={'images': data_inf['image'],
                                                 'ori_sizes': data_inf['ori_size'].int(),
                                                 'input_prompts': self.data_meta[0],
                                                 'inst_feats': self.data_meta[1]}
                                      )
            masks_hq = output["masks_pred"]

            # Resize the masks to the original size
            h, w = data_inf['ori_size'][0]
            masks_hq = masks_hq[:, :, :h, :w]

            labels_ori = data_inf['ori_label']
            masks_pred = F.interpolate(masks_hq, labels_ori.shape[-2:], mode='bilinear') > 0

            # for i in range(len(masks_pred)):
            imidx = data_inf['imidx']
            if isinstance(imidx, torch.Tensor):
                imidx = imidx.item()
            if isinstance(imidx, Sequence):
                imidx = imidx[0]

            # search for mask with largest segmentation area
            areas = masks_pred.view(masks_pred.shape[0], -1).sum(dim=1)
            max_area_index = areas.argmax().item()
            mask_pred = masks_pred[max_area_index] * 255

            ori_image = data_inf['ori_img'][0].cpu().numpy()

            image_with_masks = utils.overlay_masks(ori_image.copy(),
                                                   mask_pred.cpu().numpy().astype(np.int32),
                                                   color=(0, 255, 0), alpha=0.5)
            results.append(image_with_masks)
        return results

    def extract_prompt_features(self, image_input, seg_input, meta_input, batch_size):
        """Extracts visual prompts features from masked input images and meta descriptions.

        Args:
            image_input (str): Path to the prompt image.
            seg_input (str): Path to the prompt mask.
            meta_input (List(str)): Meta description of the in-context object.
        """
        if meta_input == "":
            meta_input = None
        elif isinstance(meta_input, str):
            meta_input = meta_input.split(";")
        dataset = InferenceDataset(image_input, seg_input, meta_input, size=self.inference_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)

        # Extracting prompt features
        prompt_feats_list, inst_feats_list = [], []

        for data in dataloader:
            with self.trt_engine_runner_prompt as runner:
                output = runner.infer(feed_dict={'images': data['image'],
                                                 'labels': data['label'],
                                                 'tokens': data['token'].int(),
                                                 'ori_sizes': data['ori_size'].int()})

            prompt_feats_list.append(output['input_prompts'])
            inst_feats_list.append(output['inst_feats'])

        prompt_feats_list = np.concatenate(prompt_feats_list)
        inst_feats_list = np.concatenate(inst_feats_list)
        data_meta = prompt_feats_list, inst_feats_list
        self.data_meta = data_meta
