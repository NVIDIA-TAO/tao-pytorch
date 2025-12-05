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

""" Download pretrained modules of StyleGAN-XL """

import pickle
import sys
import torch
import os
import tempfile

from nvidia_tao_core.microservices.handlers.cloud_handlers import utils
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from nvidia_tao_pytorch.core.distributed.comm import synchronize


def traverse_up(path, n_levels=0):
    """Traverse directories up by a specified number of levels.

    Args:
        path (str): The starting file or directory path.
        n_levels (int, optional): Number of levels to traverse up. Default is 0.

    Returns:
        str: The resulting path after moving up n_levels directories.
    """
    assert n_levels > 0, "Should atleast be 1 level."
    for _ in range(n_levels):
        path = os.path.dirname(path)
    return path


@synchronize
@rank_zero_only
def download_and_convert_pretrained_modules():
    """Download and convert pretrained modules for StyleGAN-XL.

    Raises:
        RuntimeError: If there is an issue loading the downloaded pickle files.
    """
    stylegan_root = traverse_up(os.path.abspath(__file__), n_levels=2)
    path_to_pretrained_modules = os.path.join(stylegan_root, "pretrained_modules")

    if os.path.exists(os.path.join(path_to_pretrained_modules, "InceptionV3.pth")) and os.path.exists(os.path.join(path_to_pretrained_modules, "tf_efficientnet_lite0_embed.pth")):
        return
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
            tmp_path = os.path.join(tmpdirname, "stylegan-xl")

            TAO_STYLEGAN_HF_URL = os.getenv("TAO_STYLEGAN_HF_URL", "https://github.com/autonomousvision/stylegan-xl.git")
            TAO_STYLEGAN_INCEPTION_URL = os.getenv("TAO_STYLEGAN_INCEPTION_URL", "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl")

            utils.download_from_git_link(TAO_STYLEGAN_HF_URL, tmp_path)
            utils.download_from_https_link(TAO_STYLEGAN_INCEPTION_URL, tmp_path)

            InceptionV3_file_path = os.path.join(tmp_path, "inception-2015-12-05.pkl")
            tf_efficientnet_lite0_embed_file_path = os.path.join(tmp_path, "in_embeddings/tf_efficientnet_lite0.pkl")

            # Try loading the checkpoint using pickle
            sys.path.append(tmp_path)  # Add system path of stylegan-xl source repo to run pickle properly
            with open(InceptionV3_file_path, 'rb') as f:
                InceptionV3 = pickle.load(f)
            with open(tf_efficientnet_lite0_embed_file_path, 'rb') as f:
                tf_efficientnet_lite0_embed = pickle.load(f)

            torch.save(InceptionV3.state_dict(), os.path.join(path_to_pretrained_modules, "InceptionV3.pth"))
            torch.save(tf_efficientnet_lite0_embed['embed'].state_dict(), os.path.join(path_to_pretrained_modules, "tf_efficientnet_lite0_embed.pth"))
