# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import glob
import itertools
import numpy as np
import os
import tempfile
import time
import tqdm
import warnings
from contextlib import ExitStack
import cv2
import nltk

import torch
from torch import nn

from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.engine import create_ddp_model
from detectron2.evaluation import inference_context
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer, random_color
from detectron2.utils.visualizer import _PanopticPrediction, _create_text_labels, GenericMask

from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.odise.checkpoint import ODISECheckpointer
from nvidia_tao_pytorch.cv.odise.config import instantiate_odise
from nvidia_tao_pytorch.cv.odise.config.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.odise.config.utils import override_default_cfg
from nvidia_tao_pytorch.cv.odise.data import get_openseg_labels
from nvidia_tao_pytorch.cv.odise.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
from nvidia_tao_pytorch.cv.odise.engine.defaults import get_model_from_module

nltk.download("popular", quiet=True)
nltk.download("universal_tagset", quiet=True)
spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']
setup_logger()
logger = setup_logger(name="odise")

COCO_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 1
]
COCO_THING_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
COCO_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("coco_panoptic", True))
    if COCO_CATEGORIES[idx]["isthing"] == 0
]
COCO_STUFF_COLORS = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

ADE_THING_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
]
ADE_THING_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
ADE_STUFF_CLASSES = [
    label
    for idx, label in enumerate(get_openseg_labels("ade20k_150", True))
    if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
]
ADE_STUFF_COLORS = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

LVIS_CLASSES = get_openseg_labels("lvis_1203", True)
# use beautiful coco colors
LVIS_COLORS = list(
    itertools.islice(itertools.cycle([c["color"] for c in COCO_CATEGORIES]), len(LVIS_CLASSES))
)


def draw_instance_predictions(self, predictions, jitter_color=False):
    """
    Draw instance-level prediction results on an image.
    Args:
        predictions (Instances): the output of an instance detection/segmentation
            model. Following fields will be used to draw:
            "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        jitter_color (bool): whether to jitter colors
    Returns:
        output (VisImage): image object with visualizations.
    """
    boxes = None

    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
    else:
        masks = None
    if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        if jitter_color:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
        else:
            colors = [
                [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
            ]
        alpha = 0.8
    else:
        colors = None
        alpha = 0.5

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(
            self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
                if predictions.has("pred_masks")
                else None
            )
        )
        alpha = 0.3

    self.overlay_instances(
        masks=masks,
        boxes=boxes,
        labels=labels,
        keypoints=keypoints,
        assigned_colors=colors,
        alpha=alpha,
    )
    return self.output


def overlay_instances(
    self,
    *,
    boxes=None,
    labels=None,
    masks=None,
    keypoints=None,
    assigned_colors=None,
    alpha=0.5,
):
    """
    Args:
        boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
            or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
            or a :class:`RotatedBoxes`,
            or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
            for the N objects in a single image,
        labels (list[str]): the text to be displayed for each instance.
        masks (masks-like object): Supported types are:
            * :class:`detectron2.structures.PolygonMasks`,
                :class:`detectron2.structures.BitMasks`.
            * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                The first level of the list corresponds to individual instances. The second
                level to all the polygon that compose the instance, and the third level
                to the polygon coordinates. The third level should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
            * list[ndarray]: each ndarray is a binary mask of shape (H, W).
            * list[dict]: each dict is a COCO-style RLE.
        keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
            where the N is the number of instances and K is the number of keypoints.
            The last dimension corresponds to (x, y, visibility or score).
        assigned_colors (list[matplotlib.colors]): a list of colors, where each color
            corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
            for full list of formats that the colors are accepted in.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
    Returns:
        output (VisImage): image object with visualizations.
    """
    num_instances = 0
    if boxes is not None:
        boxes = self._convert_boxes(boxes)
        num_instances = len(boxes)
    if masks is not None:
        masks = self._convert_masks(masks)
        if num_instances:
            assert len(masks) == num_instances
        else:
            num_instances = len(masks)
    if keypoints is not None:
        if num_instances:
            assert len(keypoints) == num_instances
        else:
            num_instances = len(keypoints)
        keypoints = self._convert_keypoints(keypoints)
    if labels is not None:
        assert len(labels) == num_instances
    if assigned_colors is None:
        assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
    if num_instances == 0:
        return self.output
    if boxes is not None and boxes.shape[1] == 5:
        return self.overlay_rotated_instances(
            boxes=boxes, labels=labels, assigned_colors=assigned_colors
        )

    # Display in largest to smallest order to reduce occlusion.
    areas = None
    if boxes is not None:
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    elif masks is not None:
        areas = np.asarray([x.area() for x in masks])

    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs] if boxes is not None else None
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
        keypoints = keypoints[sorted_idxs] if keypoints is not None else None

    for i in range(num_instances):
        color = assigned_colors[i]
        if boxes is not None:
            self.draw_box(boxes[i], edge_color=color)

        if masks is not None:
            for segment in masks[i].polygons:
                self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

        if labels is not None:
            # first get a box
            if boxes is not None:
                x0, y0, x1, y1 = boxes[i]
                text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                horiz_align = "left"
            elif masks is not None:
                # skip small mask without polygon
                if len(masks[i].polygons) == 0:
                    continue

                x0, y0, x1, y1 = masks[i].bbox()

                # draw text in the center (defined by median) when box is not drawn
                # median is less sensitive to outliers.
                text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                horiz_align = "center"
            else:
                continue  # drawing the box confidence for keypoints isn't very useful.
            # for small objects, draw text at the side to avoid occlusion
            instance_area = (y1 - y0) * (x1 - x0)
            if (
                instance_area < 1000 * self.output.scale
                or y1 - y0 < 40 * self.output.scale
            ):
                if y1 >= self.output.height - 5:
                    text_pos = (x1, y0)
                else:
                    text_pos = (x0, y1)

            height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            self.draw_text(
                labels[i],
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
            )

    # draw keypoints
    if keypoints is not None:
        for keypoints_per_instance in keypoints:
            self.draw_and_connect_keypoints(keypoints_per_instance)

    return self.output


def draw_panoptic_seg(self,
                      panoptic_seg,
                      segments_info,
                      area_threshold=None,
                      alpha=0.7,
                      jitter_color=False):
    """
    Draw panoptic prediction annotations or results.

    Args:
        panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
            segment.
        segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
            If it is a ``list[dict]``, each dict contains keys "id", "category_id".
            If None, category id of each pixel is computed by
            ``pixel // metadata.label_divisor``.
        area_threshold (int): stuff segments with less than `area_threshold` are not drawn.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        jitter_color (bool): whether to jitter colors

    Returns:
        output (VisImage): image object with visualizations.
    """
    pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

    if self._instance_mode == ColorMode.IMAGE_BW:
        self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

    # draw mask for all semantic segments first i.e. "stuff"
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        try:
            mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
        except AttributeError:
            mask_color = None

        text = self.metadata.stuff_classes[category_idx]
        self.draw_binary_mask(
            mask,
            color=mask_color,
            edge_color=(1.0, 1.0, 240.0 / 255),  # off_white 
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )

    # draw mask for all instances second
    all_instances = list(pred.instance_masks())
    if len(all_instances) == 0:
        return self.output
    masks, sinfo = list(zip(*all_instances))
    category_ids = [x["category_id"] for x in sinfo]

    try:
        scores = [x["score"] for x in sinfo]
    except KeyError:
        scores = None
    labels = _create_text_labels(
        category_ids, scores, self.metadata.thing_classes, [x.get("iscrowd", 0) for x in sinfo]
    )

    try:
        if jitter_color:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in category_ids
            ]
        else:
            colors = [
                [x / 255 for x in self.metadata.thing_colors[c]] for c in category_ids
            ]
    except AttributeError:
        colors = None
    self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

    return self.output

Visualizer.overlay_instances = overlay_instances
Visualizer.draw_instance_predictions = draw_instance_predictions
Visualizer.draw_panoptic_seg = draw_panoptic_seg


def get_nouns(caption, with_preposition):
    if with_preposition:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    else:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    tokenized = nltk.word_tokenize(caption)
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk


class VisualizationDemo(object):
    def __init__(self, model, metadata, aug, instance_mode=ColorMode.IMAGE):
        """
        Args:
            model (nn.Module):
            metadata (MetadataCatalog): image metadata.
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.model = model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

    def predict(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        height, width = original_image.shape[:2]
        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        predictions = self.model([inputs])[0]
        return predictions

    def run_on_image(self, path):
        """
        Args:
            path (str): path to the original image
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # use PIL, to be consistent with evaluation
        image = utils.read_image(path, format="RGB")
        start_time = time.time()
        predictions = self.predict(image)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="spec", schema=ExperimentConfig
)
def run_inference(hydra_cfg: ExperimentConfig):
    """ODISE inference."""
    results_dir = hydra_cfg.inference.results_dir or os.path.join(hydra_cfg.results_dir, 'inference')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    if hydra_cfg.model.type == 'category':
        default_config_file = 'config/common/category_odise.py'
    elif hydra_cfg.model.type == 'caption':
        default_config_file = 'config/common/caption_odise.py'
    else:
        raise NotImplementedError(f"Only `caption` and `category` are supported. Got {hydra_cfg.model.type} instead.")

    cfg = LazyConfig.load(os.path.join(spec_root, default_config_file))
    cfg = override_default_cfg(results_dir, cfg, hydra_cfg, comm.get_world_size())
    cfg.model.is_inference = True
    cfg.model.precision = hydra_cfg.inference.precision
    cfg.model.overlap_threshold = hydra_cfg.inference.overlap_threshold
    cfg.model.object_mask_threshold = hydra_cfg.inference.object_mask_threshold

    seed_all_rng(42)
    dataset_cfg = cfg.dataloader.test
    wrapper_cfg = cfg.dataloader.wrapper
    extra_classes = []

    if hydra_cfg.inference.vocab:
        for words in hydra_cfg.inference.vocab.split(";"):
            extra_classes.append([word.strip() for word in words.split(",")])

    if hydra_cfg.inference.caption:
        caption_words = []
        caption_words.extend(get_nouns(hydra_cfg.inference.caption, True))
        caption_words.extend(get_nouns(hydra_cfg.inference.caption, False))
        for word in list(set(caption_words)):
            extra_classes.append([word.strip()])

    logger.info(f"extra classes: {extra_classes}")
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if "COCO" in hydra_cfg.inference.label_set:
        demo_thing_classes += COCO_THING_CLASSES
        demo_stuff_classes += COCO_STUFF_CLASSES
        demo_thing_colors += COCO_THING_COLORS
        demo_stuff_colors = COCO_STUFF_COLORS
    if "ADE" in hydra_cfg.inference.label_set:
        demo_thing_classes += ADE_THING_CLASSES
        demo_stuff_classes += ADE_STUFF_CLASSES
        demo_thing_colors += ADE_THING_COLORS
        demo_stuff_colors += ADE_STUFF_COLORS
    if "LVIS" in hydra_cfg.inference.label_set:
        demo_thing_classes += LVIS_CLASSES
        demo_thing_colors += LVIS_COLORS

    demo_metadata = MetadataCatalog.get("odise_demo_metadata")
    demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
    demo_metadata.stuff_classes = [
        *demo_metadata.thing_classes,
        *[c[0] for c in demo_stuff_classes],
    ]
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id_value_tensor = \
        torch.tensor(list(demo_metadata.thing_dataset_id_to_contiguous_id.values()), device='cuda')

    wrapper_cfg.labels = demo_thing_classes + demo_stuff_classes
    wrapper_cfg.metadata = demo_metadata

    # Calculate the chunksizes and start index to support a more efficient ensemble_logits_with_labels implementation.
    num_templates = []
    for l in wrapper_cfg.labels:
        num_templates.append(len(l))
    chunk_sizes_pyt = torch.tensor(num_templates, dtype=torch.int32, device='cuda')
    chunk_start_idx = [0,]
    for i in range(len(num_templates) - 1):
        chunk_start_idx.append(chunk_start_idx[i] + num_templates[i])
    chunk_start_idx_pyt = torch.tensor(chunk_start_idx, dtype=torch.int32, device='cuda')

    demo_metadata.chunk_sizes_pyt = chunk_sizes_pyt
    demo_metadata.chunk_start_idx_pyt = chunk_start_idx_pyt

    # Initialize cfgs
    aug = instantiate(dataset_cfg.mapper).augmentations

    model = instantiate_odise(cfg.model)
    model.to(cfg.train.device)
    ODISECheckpointer(model).load(hydra_cfg.inference.checkpoint)
    # look for the last wrapper
    while "model" in wrapper_cfg:
        wrapper_cfg = wrapper_cfg.model
    wrapper_cfg.model = get_model_from_module(model)

    inference_model = create_ddp_model(instantiate(cfg.dataloader.wrapper))
    with ExitStack() as stack:
        if isinstance(inference_model, nn.Module):
            stack.enter_context(inference_context(inference_model))
        stack.enter_context(torch.no_grad())
        demo = VisualizationDemo(inference_model, demo_metadata, aug)

        imgpath_list = [os.path.join(hydra_cfg.inference.image_dir, imgname)
            for imgname in sorted(os.listdir(hydra_cfg.inference.image_dir))
            if os.path.splitext(imgname)[1].lower()
            in supported_img_format]

        for path in tqdm.tqdm(imgpath_list):
            predictions, visualized_output = demo.run_on_image(path)
            output_path = os.path.join(results_dir, os.path.basename(path))
            logger.info(f"Saving to {output_path}")
            visualized_output.save(output_path)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_inference()
