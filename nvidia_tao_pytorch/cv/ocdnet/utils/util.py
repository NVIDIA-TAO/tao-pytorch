#
# **************************************************************************
# Modified from github (https://github.com/WenmuZhou/DBNet.pytorch)
# Copyright (c) WenmuZhou
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# https://github.com/WenmuZhou/DBNet.pytorch/blob/master/LICENSE.md
# **************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""utility module."""
import json
import logging
import pathlib
import time
import tempfile
import os
import glob
from natsort import natsorted
import cv2
import matplotlib.pyplot as plt
import numpy as np
from eff.codec import encrypt_stream, decrypt_stream
import torch
from nvidia_tao_pytorch.core.path_utils import expand_path


def get_file_list(folder_path: str, p_postfix: list = None, sub_dir: bool = True) -> list:
    """Get file list

    Args:
        folder_path: the folder path
        p_postfix: postfix
        sub_dir: check the subfolder

    Returns:
        Return file list
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path), "Please set valid input_folder in yaml file."
    if p_postfix is None:
        p_postfix = ['.jpg']
    if isinstance(p_postfix, str):
        p_postfix = [p_postfix]
    file_list = [x for x in glob.glob(folder_path + '/*.*') if
                 os.path.splitext(x)[-1] in p_postfix or '.*' in p_postfix]
    return natsorted(file_list)


def setup_logger(log_file_path: str = None):
    """setup logger."""
    logging._warn_preinit_stderr = 0
    logger = logging.getLogger('OCDNet.pytorch')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file_path:
        file_handle = logging.FileHandler(log_file_path)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
    logger.setLevel(logging.INFO)
    return logger


def exe_time(func):
    """exe time."""
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("{} cost {:.3f}s".format(func.__name__, time.time() - t0))
        return back

    return newFunc


def load(file_path: str):
    """load file."""
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _load_txt, '.json': _load_json, '.list': _load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def _load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def _load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def save(data, file_path):
    """save file."""
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': _save_txt, '.json': _save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def _save_txt(data, file_path):
    """Write the list into a txt file"""
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def _save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_img(imgs: np.ndarray, title='img'):
    """show img."""
    color = (len(imgs.shape) == 3 and imgs.shape[-1] == 3)
    imgs = np.expand_dims(imgs, axis=0)
    for i, img in enumerate(imgs):
        plt.figure()
        plt.title('{}_{}'.format(title, i))
        plt.imshow(img, cmap=None if color else 'gray')
    plt.show()


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    """draw bbox."""
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)
        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


def cal_text_score(texts, gt_texts, training_masks, running_metric_text, thred=0.5):
    """cal text score."""
    training_masks = training_masks.data.cpu().numpy()
    pred_text = texts.data.cpu().numpy() * training_masks
    pred_text[pred_text <= thred] = 0
    pred_text[pred_text > thred] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def order_points_clockwise(pts):
    """order points clockwise."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def order_points_clockwise_list(pts):
    """order points clockwise list."""
    pts = pts.tolist()
    pts.sort(key=lambda x: (x[1], x[0]))
    pts[:2] = sorted(pts[:2], key=lambda x: x[0])
    pts[2:] = sorted(pts[2:], key=lambda x: -x[0])
    pts = np.array(pts)
    return pts


def get_datalist(train_data_path):
    """Get train data list and val data list"""
    train_data = []
    for p in train_data_path:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(expand_path(line[0].strip(' ')))
                        label_path = pathlib.Path(expand_path(line[1].strip(' ')))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        else:
            img_dir = os.path.join(p, "img")
            label_dir = os.path.join(p, "gt")
            for img in os.listdir(img_dir):
                img_file = os.path.join(img_dir, img)
                label = "gt_" + os.path.splitext(img)[0] + ".txt"
                label_file = os.path.join(label_dir, label)
                assert os.path.exists(label_file), (
                    f"Cannot find label file for image: {img_file}"
                )
                train_data.append((img_file, label_file))
    return sorted(train_data)


def get_datalist_uber(train_data_path):
    """Get uber train data list and val data list"""
    train_data = []
    for p in train_data_path:
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(expand_path(line[0].strip(' ')))
                        label_path = pathlib.Path(expand_path(line[1].strip(' ')))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        else:
            img_dir = os.path.join(p, "img")
            label_dir = os.path.join(p, "gt")
            for img in os.listdir(img_dir):
                img_file = os.path.join(img_dir, img)
                label = "truth_" + img.split('.')[0] + ".txt"
                label_file = os.path.join(label_dir, label)
                assert os.path.exists(label_file), (
                    f"Cannot find label file for image: {img_file}"
                )
                train_data.append((img_file, label_file))
    return sorted(train_data)


def parse_config(config: dict) -> dict:
    """parse config."""
    import anyconfig
    base_file_list = config.pop('base')
    base_config = {}
    for base_file in base_file_list:
        tmp_config = anyconfig.load(open(base_file, 'rb'))
        if 'base' in tmp_config:
            tmp_config = parse_config(tmp_config)
        anyconfig.merge(tmp_config, base_config)
        base_config = tmp_config
    anyconfig.merge(base_config, config)
    return base_config


def save_result(result_path, box_list, score_list, is_output_polygon):
    """save result."""
    if is_output_polygon:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                score = score_list[i]
                res.write(result + ',' + str(score) + "\n")
    else:
        with open(result_path, 'wt') as res:
            for i, box in enumerate(box_list):
                score = score_list[i]
                box = box.reshape(-1).tolist()
                result = ",".join([str(int(x)) for x in box])
                res.write(result + ',' + str(score) + "\n")


def expand_polygon(polygon):
    """Expand bbox which has only one character."""
    (x, y), (w, h), angle = cv2.minAreaRect(np.float32(polygon))
    if angle < -45:
        w, h = h, w
        angle += 90
    new_w = w + h
    box = ((x, y), (new_w, h), angle)
    points = cv2.boxPoints(box)
    return order_points_clockwise(points)


def mkdir(path):
    """ make directory if not existing """
    if not os.path.exists(path):
        os.makedirs(path)


def encrypt_pytorch(tmp_file_name, output_file_name, key):
    """Encrypt the pytorch model"""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def save_checkpoint(state, filename, key):
    """Save the checkpoint."""
    handle, temp_name = tempfile.mkstemp(".tlt")
    os.close(handle)
    torch.save(state, temp_name)
    encrypt_pytorch(temp_name, filename, key)
    os.remove(temp_name)


def decrypt_pytorch(input_file_name, output_file_name, key):
    """Decrypt the TLT model to Pytorch model"""
    with open(input_file_name, "rb") as open_temp_file, open(output_file_name,
                                                             "wb") as open_encoded_file:
        decrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )


def load_checkpoint(model_path, to_cpu=False, only_state_dict=True):
    """Helper function to load a saved checkpoint."""
    loc_type = torch.device('cpu') if to_cpu else None
    loaded_state = torch.load(model_path, map_location=loc_type, weights_only=False)
    if only_state_dict:
        state_dict = {}
        ema_state_dict = {}
        if isinstance(loaded_state, dict):
            for key, value in loaded_state["state_dict"].items():
                if 'model_ema' in key:
                    ema_state_dict[key.replace('model_ema.module.', '')] = value
                else:
                    state_dict[key.replace("model.", "")] = value
            if len(ema_state_dict) > 0:
                torch.save(ema_state_dict, f"{model_path.replace('.pth', '_ema.pth')}")
                print(f"Extract EMA state_dict and save to {model_path.replace('.pth', '_ema.pth')}")
        else:
            state_dict = loaded_state.state_dict()
        return state_dict

    return loaded_state


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    """Create logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
