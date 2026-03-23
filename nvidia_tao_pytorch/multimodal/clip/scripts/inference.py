# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Inference for CLIP model - extract image and text embeddings."""

import os
from typing import List, Optional, Tuple, Any

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from nvidia_tao_pytorch.core.decorators.workflow import monitor_status
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.core.initialize_experiments import (
    initialize_inference_experiment,
)
from nvidia_tao_pytorch.core.tlt_logging import logging, obfuscate_logs

from nvidia_tao_core.config.clip.default_config import (
    CLIPExperimentConfig as ExperimentConfig,
)
from nvidia_tao_pytorch.multimodal.clip.model.pl_clip_model import (
    CLIPPlModel,
)
from nvidia_tao_pytorch.multimodal.clip.utils.utils import (
    load_model_from_checkpoint,
)


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def get_image_files(image_dir: str) -> List[str]:
    """Get sorted list of image files from directory.

    Recursively searches the directory for supported image formats.

    Parameters
    ----------
    image_dir : str
        Directory path to search for images.

    Returns
    -------
    List[str]
        Sorted list of absolute paths to image files.
    """
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def load_and_preprocess_batch(
    batch_files: List[str],
    preprocess_fn,
    device: torch.device
) -> Tuple[Optional[Any], List[str]]:
    """Load and preprocess a batch of images.

    Parameters
    ----------
    batch_files : List[str]
        List of image file paths to process.
    preprocess_fn : callable
        Preprocessing function to apply to each image.
    device : torch.device
        Device to move tensors to (CPU or CUDA).

    Returns
    -------
    batch : torch.Tensor or dict or None
        Preprocessed images, or None if no images loaded.
        For SigLIP2 models, returns a dict with 'pixel_values' key.
    valid_paths : List[str]
        List of paths for successfully loaded images.
    """
    images = []
    valid_paths = []

    for img_path in batch_files:
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = preprocess_fn(image)
            images.append(image_tensor)
            valid_paths.append(img_path)
        except Exception as e:
            logging.warning("Failed to load %s: %s", img_path, e)
            continue

    if not images:
        return None, []

    if isinstance(images[0], dict):
        batch = {
            k: torch.stack([img[k] for img in images]).to(device)
            for k in images[0]
        }
    else:
        batch = torch.stack(images).to(device)

    return batch, valid_paths


def extract_image_features(model, batch) -> torch.Tensor:
    """Extract image features from model.

    Parameters
    ----------
    model : CLIPPlModel
        CLIP model instance.
    batch : torch.Tensor or Dict[str, torch.Tensor]
        Preprocessed image batch.

    Returns
    -------
    torch.Tensor
        Image feature tensor of shape (B, D) where B is batch size and D is
        embedding dimension.
    """
    image_features = model.model(image=batch)
    if isinstance(image_features, dict):
        return image_features["image_features"]
    return image_features[0]


def save_embeddings(
    items: List[str],
    embeddings: np.ndarray,
    results_dir: str,
    embedding_type: str = 'image'
) -> None:
    """Save embeddings to HDF5 file.

    HDF5 datasets:
    - 'embeddings': Float32 array of shape (N, D).
    - 'image_paths' or 'texts': UTF-8 source strings.

    Parameters
    ----------
    items : List[str]
        Image paths or text strings for each embedding.
    embeddings : np.ndarray
        Numpy array of embeddings with shape (N, D).
    results_dir : str
        Directory path to save output file.
    embedding_type : str
        'image' or 'text'. Determines output filename.
    """
    if embedding_type == 'image':
        filename = "image_embeddings.h5"
        items_key = 'image_paths'
        count_key = 'num_images'
    else:
        filename = "text_embeddings.h5"
        items_key = 'texts'
        count_key = 'num_texts'

    embeddings_file = os.path.join(results_dir, filename)

    with h5py.File(embeddings_file, 'w') as f:
        # Store embeddings as float32
        f.create_dataset(
            'embeddings',
            data=embeddings.astype(np.float32),
            compression='gzip',
            compression_opts=4
        )

        # Store items (paths or texts) as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        items_dataset = f.create_dataset(items_key, (len(items),), dtype=dt)
        for i, item in enumerate(items):
            items_dataset[i] = item

        # Store metadata
        f.attrs[count_key] = len(items)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['embedding_type'] = embedding_type

    logging.info(
        "%s embeddings saved to %s",
        embedding_type.capitalize(), embeddings_file,
    )


def load_text_file(text_file: str) -> List[str]:
    """Load text prompts from file.

    Parameters
    ----------
    text_file : str
        Path to text file with one text prompt per line.

    Returns
    -------
    List[str]
        List of text prompts (non-empty, stripped of whitespace).
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def extract_text_features(
    model, texts: List[str], device: torch.device,
) -> torch.Tensor:
    """Extract text features from model.

    Parameters
    ----------
    model : CLIPPlModel
        CLIP model instance.
    texts : List[str]
        List of text strings to encode.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    torch.Tensor
        Text feature tensor of shape (B, D) where B is batch size and D is
        embedding dimension.
    """
    # Tokenize texts - all tokenizers return [dict]
    tokenized = model.tokenizer(texts)[0]

    # Move to device
    tokenized = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tokenized.items()
    }

    # Extract features
    text_output = model.model(text=tokenized)
    if isinstance(text_output, dict):
        return text_output["text_features"]
    return text_output[1]


def run_image_inference(
    model, inference_cfg, results_dir: str,
    device: torch.device,
) -> None:
    """Run image embedding extraction.

    Parameters
    ----------
    model : CLIPPlModel
        Loaded model.
    inference_cfg : InferenceExpConfig
        Inference configuration.
    results_dir : str
        Output directory.
    device : torch.device
        Device to run on.
    """
    batch_size = max(1, inference_cfg.batch_size)

    # Collect images from all datasets
    image_files = []
    for dataset_cfg in inference_cfg.datasets:
        image_dir = dataset_cfg.image_dir
        files = get_image_files(image_dir)
        image_files.extend(files)
        logging.info(f"Found {len(files)} images in {image_dir}")

    if not image_files:
        logging.warning("No images found in any dataset")
        return

    logging.info(f"Total: {len(image_files)} images")

    all_embeddings = []
    all_paths = []
    num_batches = (len(image_files) + batch_size - 1) // batch_size

    with torch.no_grad():
        pbar = tqdm(
            range(0, len(image_files), batch_size),
            total=num_batches, desc="Image embeddings",
        )
        for i in pbar:
            batch_files = image_files[i:i + batch_size]

            batch, valid_paths = load_and_preprocess_batch(
                batch_files, model.preprocess_val, device
            )

            if batch is None:
                continue

            image_features = extract_image_features(model, batch)

            all_embeddings.append(image_features.cpu().numpy())
            all_paths.extend(valid_paths)

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        save_embeddings(
            all_paths, embeddings_array,
            results_dir, embedding_type='image',
        )
        logging.info(f"Extracted embeddings for {len(all_paths)} images")
    else:
        logging.warning("No image embeddings were extracted")


def run_text_inference(
    model, inference_cfg, results_dir: str, device: torch.device
) -> None:
    """Run text embedding extraction.

    Parameters
    ----------
    model : CLIPPlModel
        Loaded model.
    inference_cfg : InferenceExpConfig
        Inference configuration.
    results_dir : str
        Output directory.
    device : torch.device
        Device to run on.
    """
    text_file = inference_cfg.text_file
    batch_size = max(1, inference_cfg.batch_size)

    texts = load_text_file(text_file)
    if not texts:
        logging.warning(f"No texts found in {text_file}")
        return

    logging.info(f"Found {len(texts)} text prompts in {text_file}")

    all_embeddings = []
    all_texts = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), batch_size),
            total=num_batches,
            desc="Extracting text embeddings"
        ):
            batch_texts = texts[i:i + batch_size]

            text_features = extract_text_features(model, batch_texts, device)

            all_embeddings.append(text_features.cpu().numpy())
            all_texts.extend(batch_texts)

    if all_embeddings:
        embeddings_array = np.concatenate(all_embeddings, axis=0)
        save_embeddings(
            all_texts, embeddings_array, results_dir, embedding_type='text'
        )
        logging.info(f"Extracted embeddings for {len(all_texts)} texts")
    else:
        logging.warning("No text embeddings were extracted")


def run_experiment(experiment_config, key):
    """Run inference experiment to extract image and/or text embeddings.

    Loads a trained CLIP model and extracts embeddings for images and/or text.
    - image_dir: extracts image embeddings to image_embeddings.h5
    - text_file: extracts text embeddings to text_embeddings.h5
    - Both can be provided to extract both.

    Parameters
    ----------
    experiment_config : ExperimentConfig
        Experiment configuration object containing inference settings.
    key : str
        Encryption key (unused, kept for TAO API compatibility).
    """
    del key  # Unused but required by TAO API

    model_path, _ = initialize_inference_experiment(
        experiment_config, experiment_config.encryption_key
    )
    model = load_model_from_checkpoint(
        model_path, experiment_config, CLIPPlModel
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Get inference config
    inference_cfg = experiment_config.inference
    datasets = getattr(inference_cfg, 'datasets', None) or []
    text_file = getattr(inference_cfg, 'text_file', None)
    results_dir = experiment_config.results_dir or inference_cfg.results_dir

    if not datasets and not text_file:
        raise ValueError(
            "At least one of inference.datasets or inference.text_file "
            "must be specified"
        )

    os.makedirs(results_dir, exist_ok=True)

    # Run image inference if datasets provided
    if datasets:
        run_image_inference(model, inference_cfg, results_dir, device)

    # Run text inference if text_file provided
    if text_file:
        run_text_inference(model, inference_cfg, results_dir, device)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="experiment_spec",
    schema=ExperimentConfig
)
@monitor_status(name="CLIP", mode="inference")
def main(cfg: ExperimentConfig) -> None:
    """Run the inference process.

    Parameters
    ----------
    cfg : ExperimentConfig
        Hydra configuration object populated from experiment spec.
    """
    obfuscate_logs(cfg)
    run_experiment(experiment_config=cfg, key=cfg.encryption_key)


if __name__ == "__main__":
    main()
