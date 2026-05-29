# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

import os
import lmdb
import cv2

import numpy as np

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.path_utils import expand_path
from nvidia_tao_pytorch.config.ocrnet.default_config import ExperimentConfig


def checkImageIsValid(imageBin):
    """Check if the image is valid.

    Args:
        imageBin : the encoded image data.

    Returns:
        bool : True if the image is valid else False.
    """
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    """Write the cache to LMDB

    Args:
        env (lmdb.Environment): the LMDB environment to save the content.
        cache (dict): the content to be writed in LMDB.
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """Create LMDB dataset for training and evaluation.

    Args:
        inputPath (string): input folder path where starts imagePath
        outputPath (string): LMDB output path
        gtFile (string): list of image path and label
        checkValid (bool): if true, check the validity of every image
    """
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    gtFile = expand_path(gtFile)
    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = expand_path(f"{inputPath}/{imagePath}")

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print(f'{imagePath} does not exist')
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print(f'{imagePath} is not a valid image')
                    continue
            except Exception:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment",
    schema=ExperimentConfig)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        if "train_gt_file" in cfg["dataset"]:
            if cfg["dataset"]["train_gt_file"] == "":
                cfg["dataset"]["train_gt_file"] = None
        if "val_gt_file" in cfg["dataset"]:
            if cfg["dataset"]["val_gt_file"] == "":
                cfg["dataset"]["val_gt_file"] = None
        if cfg.dataset_convert.results_dir:
            results_dir = cfg.dataset_convert.results_dir
        else:
            results_dir = os.path.join(cfg.results_dir, "dataset_convert", "lmdb")
            cfg.dataset_convert.results_dir = results_dir

        os.makedirs(results_dir, exist_ok=True)
        # Set status logging
        status_file = os.path.join(results_dir, "status.json")
        status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file,
                                                                     append=True))
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.STARTED,
            message="Starting OCRNet dataset_convert"
        )
        inputPath = expand_path(cfg.dataset_convert.input_img_dir)
        createDataset(inputPath=inputPath,
                      gtFile=cfg.dataset_convert.gt_file,
                      outputPath=results_dir)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.RUNNING,
            message="Dataset convert finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Dataset convert was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == '__main__':
    main()
