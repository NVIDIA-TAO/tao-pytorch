# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Post processing utilities for test/inference."""

import os
from typing import List, Dict, Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from nvidia_tao_pytorch.cv.deformable_detr.model.post_process import get_key
from nvidia_tao_pytorch.cv.deformable_detr.utils import box_ops


def threshold_predictions(
    predictions: List[Dict],
    conf_threshold: float = 0.0,
    nms_threshold: float = 0.0,
    no_targets: Optional[Any] = None,
) -> List[Dict]:
    """Filter predictions by confidence and NMS.

    Args:
        predictions (List[Dict]): List of predictions from the model.
        conf_threshold (float): Confidence score threshold.
        nms_threshold (float): IoU threshold for NMS.
        no_targets (Optional[Any]): No targets flag.
    Returns:
        List[Dict]: Filtered predictions, one dict per image.
    """
    filtered_predictions = []

    for i, pred in enumerate(predictions):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        pred_masks = pred["masks"]
        pred_text_mask = pred["text_mask"]
        image_size = pred["image_size"]
        image_names = pred["image_names"]
        device = pred_scores.device

        assert pred_boxes.shape[0] == pred_scores.shape[0]

        if no_targets is not None:
            skip = no_targets[i].bool()
        else:
            skip = False

        if pred_boxes.shape[0] == 0 or skip:
            filtered_predictions.append(
                {
                    "image_size": image_size,
                    "image_names": image_names,
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.float32, device=device),
                    "masks": torch.empty(
                        (0, *pred["image_size"]), dtype=torch.long, device=device
                    ),
                    "text_mask": torch.empty((0,), dtype=torch.float32, device=device),
                }
            )
            continue

        keep = pred_scores >= conf_threshold
        if keep.sum() == 0:
            filtered_predictions.append(
                {
                    "image_size": image_size,
                    "image_names": image_names,
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.float32, device=device),
                    "masks": torch.empty(
                        (0, *pred["image_size"]), dtype=torch.long, device=device
                    ),
                    "text_mask": torch.empty((0,), dtype=torch.float32, device=device),
                }
            )
            continue

        if nms_threshold > 0:
            pred_text_mask = pred_text_mask[keep]
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            pred_masks = pred_masks[keep]
            keep = nms(pred_boxes, pred_scores, nms_threshold)

        filtered_predictions.append(
            {
                "image_size": image_size,
                "image_names": image_names,
                "boxes": pred_boxes[keep],
                "scores": pred_scores[keep],
                "labels": pred_labels[keep],
                "masks": pred_masks[keep],
                "text_mask": pred_text_mask[keep],
            }
        )

    return filtered_predictions


def get_valid_mask(union_mask_logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5):
    """Validate predicted masks against union mask logits.

    Args:
        union_mask_logits (Tensor): Shape (B, 2, H, W), logits for fg/bg mask.
        masks (Tensor): Shape (B, N, H, W), binary masks.
        threshold (float): Min intersection-over-mask ratio to keep.

    Returns:
        List[Tensor]: List of boolean masks indicating valid predictions.
    """
    bsz, channels, _, _ = union_mask_logits.shape
    assert channels == 2, f"Expected 2 channels, got {channels}"
    _, h, w = masks.shape[1:]

    preds = union_mask_logits.argmax(dim=1, keepdim=True).float()
    preds_up = F.interpolate(preds, size=(h, w), mode="bilinear", align_corners=False)
    preds_bin = (preds_up.sigmoid() > 0.5).float()

    keep_mask = []
    for b in range(bsz):
        pred_mask = preds_bin[b]  # (1, H, W)
        gt_masks = masks.sigmoid().gt(0.5).float()[b]  # (N, H, W)
        inter = (pred_mask * gt_masks).flatten(1).sum(1)
        area = gt_masks.flatten(1).sum(1) + 1e-6
        ratio = inter / area
        keep_mask.append(ratio > threshold)

    return keep_mask


def get_phrase_from_expression(predictions, tokenizer, tokenized):
    """Convert text mask predictions into human-readable phrases.

    Args:
        predictions (List[Dict]): List of predictions from the model.
        tokenizer: Tokenizer instance.
        tokenized (Dict): Tokenized inputs with ``input_ids``.

    Returns:
        List[Dict]: Predictions with added "phrase" field.
    """
    input_ids = tokenized["input_ids"]

    for b, pred in enumerate(predictions):
        pred_text_mask = pred.pop("text_mask")
        phrase_list = []

        for phrase_mask in pred_text_mask:
            non_zero_idx = phrase_mask.nonzero(as_tuple=True)[0].tolist()
            token_ids = input_ids[b, non_zero_idx].tolist()
            decoded = tokenizer.decode(
                [tid for tid in token_ids if tid not in tokenizer.all_special_ids],
                skip_special_tokens=True,
            )
            phrase_list.append(decoded)

        pred["phrase"] = phrase_list

    return predictions


def save_inference_prediction(
    predictions,
    output_dir,
    label_map=None,
    color_map=None,
    is_internal: bool = False,
    outline_width: int = 3,
    use_phrases: bool = False,
    save_masks: bool = True,
    mask_alpha: float = 0.4,
):
    """
    Save annotated images (bboxes + masks) and label files.

    Args:
        predictions (List[Dict]): Model predictions.
        output_dir (str): Directory to save outputs.
        label_map (Dict, optional): Mapping of class indices to labels.
        color_map (Dict, optional): Mapping of label names to RGB tuples.
        is_internal (bool): Save in nested folder structure if True.
        outline_width (int): Bounding box outline thickness.
        use_phrases (bool): Use prediction['phrase'] instead of class labels.
        save_masks (bool): Overlay masks if available.
        mask_alpha (float): Mask transparency (0=transparent, 1=opaque).
    """
    color_map = color_map or {}

    for pred in predictions:
        image_path = pred["image_names"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_masks = pred.get("masks")

        # Labels
        if use_phrases:
            pred_labels = pred["phrase"]
        else:
            assert label_map is not None, "label_map must be provided if use_phrases=False"
            pred_labels = pred["labels"]

        # Paths
        basename, extension = os.path.splitext(os.path.basename(image_path))
        if is_internal:
            folder_name = image_path.split(os.sep)[-3]
            output_label_root = os.path.join(output_dir, folder_name, "labels")
            output_annot_root = os.path.join(output_dir, folder_name, "images_annotated")
        else:
            output_label_root = os.path.join(output_dir, "labels")
            output_annot_root = os.path.join(output_dir, "images_annotated")

        os.makedirs(output_label_root, exist_ok=True)
        os.makedirs(output_annot_root, exist_ok=True)

        output_label_file = os.path.join(output_label_root, f"{basename}.txt")
        output_image_file = os.path.join(output_annot_root, f"{basename}{extension}")

        # Open image
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)
        W, H = pil_img.size

        # Layers
        bbox_img = pil_img.copy()  # drawing bboxes/text
        draw = ImageDraw.Draw(bbox_img)

        mask_overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # transparent layer for masks

        # Handle no-target case
        if pred_boxes.shape[0] == 0:
            with open(output_label_file, "w") as f:
                f.write('Empty\n')
            pil_img.save(output_image_file)
            continue

        assert (
            pred_boxes.shape[0] == len(pred_scores) == len(pred_labels)
        ), "Mismatch in number of boxes, scores, and labels"

        with open(output_label_file, "w") as f:
            for k, box in enumerate(pred_boxes.tolist()):
                # Label resolution
                if use_phrases:
                    label_name = pred_labels[k]
                else:
                    class_key = get_key(label_map, pred_labels[k])
                    label_name = class_key if class_key is not None else None

                if label_name is None:
                    continue

                x1, y1, x2, y2 = map(float, box)

                # Write label file
                bbox_str = f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f}"
                f.write(
                    f"{label_name} 0.00 0 0.00 {bbox_str}"
                    f" 0.00 0.00 0.00 0.00 0.00 0.00 0.00 {pred_scores[k]:.3f}\n"
                )

                # Draw bbox
                color = color_map.get(label_name, (255, 0, 0))
                draw.rectangle([int(x1), int(y1), int(x2), int(y2)],
                               outline=color, width=outline_width)

                # Draw label text
                if not is_internal:
                    draw.rectangle([x1, y1 - 10, x2, y1], fill=color)
                    draw.text((x1, y1 - 10), f"{label_name}: {pred_scores[k]:.2f}")

                # Overlay mask
                if save_masks and pred_masks is not None and k < pred_masks.shape[0]:
                    mask = (pred_masks[k].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0]

                    mask_pil = Image.fromarray(mask * 255).resize((W, H), Image.NEAREST)

                    # Convert mask to RGBA & accumulate it without destroying previous drawings
                    colored_mask = Image.new("RGBA", (W, H), (*color, int(mask_alpha * 255)))
                    mask_overlay = Image.composite(colored_mask, mask_overlay, mask_pil)

        final_img = Image.alpha_composite(bbox_img.convert("RGBA"), mask_overlay)
        final_img = final_img.convert("RGB")  # remove alpha
        final_img.save(output_image_file)


class PostProcess(nn.Module):
    """Convert model outputs into COCO-style predictions."""

    def __init__(self, num_select: int = 100, has_mask: bool = True) -> None:
        """Initialize GDINO PostProcess module."""
        super().__init__()
        self.num_select = num_select
        self.has_mask = has_mask

    @torch.no_grad()
    def forward(
        self,
        outputs,
        target_sizes,
        image_names,
        input_sizes=None,
        label_positive_map=None,
        text_threshold: float = 0.0,
        ioi_threshold: float = 0.5,
    ):
        """Convert raw model outputs to post-processed predictions.

        Args:
            outputs (Dict): Raw model outputs.
            target_sizes (Tensor): Shape [B, 2], original image sizes.
            image_names (List[str]): List of image paths.
            input_sizes (Tensor, optional): Shape [B, 2], padded input sizes.
            label_positive_map (Tensor, optional): Label-to-token positive map.
            text_threshold (float): Threshold for text masks.
            ioi_threshold (float): Intersection over Instance (IoI) threshold for mask validation.

        Returns:
            Tuple[List[Dict], Optional[Tensor]]: Predictions and no_targets flag.
        """
        img_h, img_w = target_sizes.unbind(1)
        device = outputs["pred_logits"].device

        if input_sizes is not None:
            inp_h, inp_w = input_sizes.unbind(1)
        else:
            self.has_mask = False

        num_select = self.num_select
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        prob_to_token = out_logits.sigmoid()

        if label_positive_map is not None:
            prob = prob_to_token @ label_positive_map.T
        else:
            prob = prob_to_token

        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode="trunc")
        labels = topk_indexes % prob.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        text_masks = prob_to_token > text_threshold
        _, _, vocab_size = text_masks.shape
        topk_boxes_expanded = topk_boxes.unsqueeze(-1).expand(-1, -1, vocab_size)
        text_masks = torch.gather(text_masks, 1, topk_boxes_expanded)

        masks = []
        valid_masks = None

        if "no_targets" in outputs:
            if outputs["no_targets"].shape[-1] == 2:
                no_targets = torch.argmax(outputs["no_targets"], dim=-1)
            else:
                no_targets = (outputs["no_targets"].sigmoid() > 0.5).long()
        else:
            no_targets = None

        if self.has_mask:
            pred_masks = outputs["pred_masks"].squeeze(2).to(device)
            bsz, _, mh, mw = pred_masks.shape
            topk_mask_idx = (
                topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mh, mw).to(torch.int64)
            )
            pred_masks = torch.gather(pred_masks, 1, topk_mask_idx)
            if ioi_threshold > 0 and "union_mask_logits" in outputs:
                valid_masks = get_valid_mask(outputs["union_mask_logits"], pred_masks, threshold=ioi_threshold)
            for i in range(bsz):
                h, w = inp_h[i].item(), inp_w[i].item()
                resized_mask = F.interpolate(pred_masks[i:i + 1], size=(mh * 4, mw * 4),
                                             mode="bilinear", align_corners=False)
                resized_mask = resized_mask[:, :, :h, :w]
                resized_mask = F.interpolate(resized_mask, size=(img_h[i].item(), img_w[i].item()),
                                             mode="bilinear", align_corners=False)
                masks.append(resized_mask.sigmoid()[0])

        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        if self.has_mask:
            if valid_masks is not None:
                results = [
                    {
                        "text_mask": t[vm],
                        "scores": s[vm],
                        "labels": l[vm],
                        "boxes": b[vm],
                        "masks": m[vm],
                        "image_names": n,
                        "image_size": i,
                    }
                    for t, s, l, b, vm, m, n, i in zip(
                        text_masks, scores, labels, boxes,
                        valid_masks, masks, image_names, target_sizes
                    )
                ]
            else:
                results = [
                    {
                        "text_mask": t,
                        "scores": s,
                        "labels": l,
                        "boxes": b,
                        "masks": m,
                        "image_names": n,
                        "image_size": i,
                    }
                    for t, s, l, b, m, n, i in zip(
                        text_masks, scores, labels, boxes,
                        masks, image_names, target_sizes
                    )
                ]
        else:
            results = [
                {
                    "text_mask": t,
                    "scores": s,
                    "labels": l,
                    "boxes": b,
                    "image_names": n,
                    "image_size": i,
                }
                for t, s, l, b, n, i in zip(
                    text_masks, scores, labels, boxes, image_names, target_sizes
                )
            ]

        return results, no_targets
