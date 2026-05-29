# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections import defaultdict
import functools
import json
import multiprocessing as mp
import numpy as np
import os
import time
from fvcore.common.download import download
from panopticapi.utils import rgb2id
from PIL import Image


def load_coco_caption(caption_json):
    id2caption = defaultdict(list)
    with open(caption_json, 'r') as f:
        obj = json.load(f)
        for ann in obj["annotations"]:
            id2caption[int(ann["image_id"])].append(ann["caption"])
    return id2caption


def create_annotation_with_caption(input_json, output_json, caption_json):
    id2coco_caption = load_coco_caption(caption_json)

    with open(input_json) as f:
        obj = json.load(f)

    coco_count = 0

    print(f"Starting to add captions to {input_json} ...")
    print(f"Total images: {len(obj['annotations'])}")
    for ann in obj["annotations"]:
        image_id = int(ann["image_id"])
        if image_id in id2coco_caption:
            ann["coco_captions"] = id2coco_caption[image_id]
            coco_count += 1
    print(f"Found {coco_count} captions from COCO ")

    print(f"Start writing to {output_json} ...")
    with open(output_json, "w") as f:
        json.dump(obj, f)


def _process_panoptic_to_semantic(input_panoptic, output_semantic, segments, id_map):
    panoptic = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
    panoptic = rgb2id(panoptic)
    output = np.zeros_like(panoptic, dtype=np.uint8) + 255
    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id
    os.makedirs(os.path.dirname(output_semantic), exist_ok=True)
    Image.fromarray(output).save(output_semantic)


def separate_coco_semantic_from_panoptic(panoptic_json, panoptic_root, sem_seg_root, categories):
    """
    Create semantic segmentation annotations from panoptic segmentation
    annotations, to be used by PanopticFPN.
    It maps all thing categories to class 0, and maps all unlabeled pixels to class 255.
    It maps all stuff categories to contiguous ids starting from 1.
    Args:
        panoptic_json (str): path to the panoptic json file, in COCO's format.
        panoptic_root (str): a directory with panoptic annotation files, in COCO's format.
        sem_seg_root (str): a directory to output semantic annotation files
        categories (list[dict]): category metadata. Each dict needs to have:
            "id": corresponds to the "category_id" in the json annotations
            "isthing": 0 or 1
    """
    os.makedirs(sem_seg_root, exist_ok=True)

    id_map = {}  # map from category id to id in the output semantic annotation
    assert len(categories) <= 254
    for i, k in enumerate(categories):
        id_map[k["id"]] = i
    # what is id = 0?
    # id_map[0] = 255
    print(id_map)

    with open(panoptic_json) as f:
        obj = json.load(f)

    pool = mp.Pool(processes=max(mp.cpu_count() // 2, 4))

    def iter_annotations():
        for anno in obj["annotations"]:
            file_name = anno["file_name"]
            segments = anno["segments_info"]
            input = os.path.join(panoptic_root, file_name)
            output = os.path.join(sem_seg_root, file_name)
            yield input, output, segments

    print("Start writing to {} ...".format(sem_seg_root))
    start = time.time()
    pool.starmap(
        functools.partial(_process_panoptic_to_semantic, id_map=id_map),
        iter_annotations(),
        chunksize=100,
    )
    print("Finished. time: {:.2f}s".format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The data preprocessing script for ODISE pipelines."
    )
    parser.add_argument('--pan_json', type=str,
                        help="JSON file in COCO Panoptic format.")
    parser.add_argument('--semsegm_dir', type=str,
                        help="Directory to save semantic segmentation images.")
    parser.add_argument('--pan_dir', type=str, default=None, 
                        help="Folder with panoptic COCO format segmentations.")
    parser.add_argument('--cat_json', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--caption_json', type=str,
                        default=None,
                        help="JSON file in COCO caption format")
    parser.add_argument('--pan_caption_json', type=str,
                        default=None,
                        help="JSON file in COCO caption format")
    args = parser.parse_args()
    with open(args.cat_json, 'r') as f:
        categories_list = json.load(f)
    separate_coco_semantic_from_panoptic(
        args.pan_json,
        args.pan_dir,
        args.semsegm_dir,
        categories_list,
    )
    if args.caption_json:
        create_annotation_with_caption(
            args.pan_json,
            args.pan_caption_json,
            args.caption_json
        )
