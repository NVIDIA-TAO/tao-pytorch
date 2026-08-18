# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom filesystem-based image-text dataset loader for CLIP.

This module provides a DataLoader for datasets where images and their
corresponding text captions are stored as individual files on disk.
Used when dataset type is 'custom' in the config.

Optional: when each dataset config includes train_pairs_file and balance_query_types
is set, training uses balanced sampling across the query types present in the
combined JSON metadata. If train_pairs_file includes a caption field per row,
batches can optionally be built so each caption appears at most once per batch.
Multiple images may share the same caption across rows; only one such row per
batch is selected so CLIP/SigLIP contrastive loss does not treat other valid
pairs as negatives. For very large datasets, set unique_caption_per_batch=False
to avoid the expensive greedy unique-caption batch construction while keeping
query-type frequency balancing. Otherwise the loader falls back to
inverse-frequency query-type weighting only.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import (
    DataLoader,
    distributed,
    RandomSampler,
    BatchSampler,
    Dataset,
)

from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.multimodal.clip.dataloader.sampler import (
    BalancedByQueryTypeSampler,
    BalancedUniqueCaptionBatchSampler,
    query_types_order_from_pairs,
    query_types_order_from_types,
)
from nvidia_tao_pytorch.multimodal.clip.utils.attribute_metadata import (
    load_missing_match_value_ids,
    normalize_missing_match_values,
)


_ATTRIBUTE_METADATA_FIELDS = ("image_attr_values", "text_attr_values")
_ACCESSORY_METADATA_FIELDS = ("image_accessory_ids", "text_accessory_ids")
_PAS_ROW_INDEX_FIELD = "pas_row_index"


# New dataloader that takes a list of dataset sources
class ImageTextDataset(Dataset):
    """Image Text dataloader for fine-tuning."""

    def __init__(self, datasets: List[dict], transform: Callable = None,
                 tokenizer: Callable = None, zero_shot_eval=False, mapping=None,
                 mode='train', include_attribute_metadata: bool = False):
        """
        Initializes the ImageTextDataset.

        Args:
            datasets (List[dict]): List of dataset configurations.
            transform (Callable): Transform function for images.
            tokenizer (Callable): Tokenizer function for texts.
            zero_shot_eval (bool): Flag for zero-shot evaluation.
            mapping (Optional[dict]): Mapping for text transformations.
            mode (str): Dataset mode ('train' or 'val').
            include_attribute_metadata (bool): If True, return image/text
                attribute vectors from split-aligned pairs metadata.
        """
        self.transform = transform
        self.tokenizer = tokenizer
        self.zero_shot_eval = zero_shot_eval
        self.mapping = mapping
        self.mode = mode
        self.attribute_metadata = None

        self.image_text_pairs = []
        attribute_metadata_rows = []
        attribute_width = None
        attribute_vocab_digest = None
        accessory_metadata_present = None
        accessory_vocab_digest = None
        if len(datasets) > 1 and self.zero_shot_eval:
            raise NotImplementedError(
                "Validation currently only supports a single dataset as input")

        # Supported image file extensions
        supported_extensions = ['*.jpg', '*.jpeg', '*.png']
        # Iterate over each dataset configuration
        for dataset in datasets:
            image_dir = Path(dataset['image_dir'])
            caption_dir = Path(dataset.get('caption_dir') or image_dir)
            image_list_file = dataset.get('image_list_file')
            caption_file_suffix = dataset.get('caption_file_suffix', '.txt')

            # If image_list_file is provided, read image list from the text file
            if image_list_file:
                with open(image_list_file, 'r') as file:
                    image_list = [
                        line.strip() for line in file if line.strip()
                    ]
                if include_attribute_metadata:
                    (
                        metadata_rows,
                        attribute_width,
                        source_attribute_vocab_digest,
                        source_accessory_vocab_digest,
                    ) = _load_attribute_metadata(
                        dataset=dataset,
                        mode=mode,
                        image_list=image_list,
                        expected_width=attribute_width,
                        require_attribute_vocab=(
                            len(datasets) > 1 or mode != 'train'
                        ),
                    )
                    if attribute_vocab_digest is None:
                        attribute_vocab_digest = source_attribute_vocab_digest
                    elif attribute_vocab_digest != (
                        source_attribute_vocab_digest
                    ):
                        raise ValueError(
                            "All datasets with attribute metadata must use "
                            "the same ordered attribute vocabulary."
                        )
                    source_has_accessories = (
                        source_accessory_vocab_digest is not None
                    )
                    if accessory_metadata_present is None:
                        accessory_metadata_present = source_has_accessories
                        accessory_vocab_digest = source_accessory_vocab_digest
                    elif accessory_metadata_present != source_has_accessories:
                        raise ValueError(
                            "All datasets must consistently provide accessory "
                            "metadata when include_attribute_metadata is enabled."
                        )
                    elif (
                        source_has_accessories
                        and accessory_vocab_digest
                        != source_accessory_vocab_digest
                    ):
                        raise ValueError(
                            "All datasets with accessory metadata must use the "
                            "same accessory vocabulary."
                        )
                    attribute_metadata_rows.extend(metadata_rows)
                # Trust the image list file - skip existence checks for speed
                for image_name in image_list:
                    image_path = image_dir / image_name
                    text_path = caption_dir / \
                        Path(image_name).with_suffix(caption_file_suffix)
                    self.image_text_pairs.append((image_path, text_path))
            else:
                if include_attribute_metadata:
                    raise ValueError(
                        "include_attribute_metadata requires image_list_file "
                        "for every custom dataset."
                    )
                # No image list - glob for files and verify existence
                logging.info(
                    f"image_list_file not provided. Using all images with "
                    f"extensions {supported_extensions} from {image_dir}"
                )
                image_list = [
                    p.name for ext in supported_extensions for p in image_dir.glob(ext)]
                for image_name in image_list:
                    image_path = image_dir / image_name
                    text_path = caption_dir / \
                        Path(image_name).with_suffix(caption_file_suffix)
                    if text_path.exists():
                        self.image_text_pairs.append((image_path, text_path))
        logging.info(
            f"Loaded {len(self.image_text_pairs)} image-text pairs ({self.mode})")
        if not self.image_text_pairs:
            raise ValueError("No valid image-text pairs found across datasets")
        if include_attribute_metadata:
            if len(attribute_metadata_rows) != len(self.image_text_pairs):
                raise ValueError(
                    "Attribute metadata has "
                    f"{len(attribute_metadata_rows)} items but dataset has "
                    f"{len(self.image_text_pairs)}."
                )
            self.attribute_metadata = {
                field: torch.tensor(
                    [row[field] for row in attribute_metadata_rows],
                    dtype=torch.long,
                )
                for field in _ATTRIBUTE_METADATA_FIELDS
            }
            if accessory_metadata_present:
                for field in _ACCESSORY_METADATA_FIELDS:
                    padded_width = max(
                        1,
                        max(len(row[field]) for row in attribute_metadata_rows),
                    )
                    self.attribute_metadata[field] = torch.tensor(
                        [
                            row[field]
                            + [0] * (padded_width - len(row[field]))
                            for row in attribute_metadata_rows
                        ],
                        dtype=torch.long,
                    )
            if self.mode != 'train':
                self.attribute_metadata[_PAS_ROW_INDEX_FIELD] = torch.arange(
                    len(attribute_metadata_rows),
                    dtype=torch.long,
                )
            logging.info(
                "Loaded attribute metadata for "
                f"{len(self.image_text_pairs)} samples ({self.mode}), "
                f"width={attribute_width}, "
                f"accessories={bool(accessory_metadata_present)}."
            )

    def __len__(self):
        """Returns the number of image-text pairs in the dataset."""
        return len(self.image_text_pairs)

    def __getitem__(self, idx):
        """
        Retrieves an image-text pair from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Image, str]: A tuple containing the transformed image and the corresponding text.
        """
        image_path, text_path = self.image_text_pairs[idx]

        image = Image.open(image_path).convert('RGB')
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().strip()

        if self.transform:
            image = self.transform(image)
        if self.zero_shot_eval and self.mapping:
            # For zero-shot eval, map text to class index BEFORE tokenization
            text = self.mapping.get(text, text)
            # If mapping found, text is now an integer class index - don't tokenize
        elif self.tokenizer:
            text = self.tokenizer(text)[0]

        if self.attribute_metadata is not None:
            metadata = {
                field: values[idx]
                for field, values in self.attribute_metadata.items()
            }
            return image, text, metadata
        return image, text


def _read_image_list(image_list_file: str) -> List[str]:
    """Read image basenames from an image_list_file, matching ImageTextDataset order."""
    with open(image_list_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def _infer_pairs_file_from_image_list(image_list_file: str) -> Path:
    """Infer ``*_pairs.json`` from a split-aligned ``*_list.txt`` path."""
    image_list_path = Path(image_list_file)
    suffix = "_list.txt"
    if not image_list_path.name.endswith(suffix):
        raise ValueError(
            "Validation attribute metadata requires attribute_pairs_file or "
            f"an image_list_file ending with {suffix!r}, got "
            f"{image_list_path}."
        )
    return image_list_path.with_name(
        image_list_path.name[:-len(suffix)] + "_pairs.json"
    )


def _resolve_attribute_pairs_file(dataset: dict, mode: str) -> Path:
    """Resolve split-aligned pairs metadata used for attribute tensors."""
    if mode == 'train':
        pairs_file = dataset.get('train_pairs_file')
        if pairs_file:
            return Path(pairs_file)
        raise ValueError(
            "include_attribute_metadata requires train_pairs_file for every "
            "training dataset."
        )

    pairs_file = dataset.get('attribute_pairs_file')
    if pairs_file:
        return Path(pairs_file)
    image_list_file = dataset.get('image_list_file')
    if not image_list_file:
        raise ValueError(
            "Validation attribute metadata requires attribute_pairs_file or "
            "image_list_file for every validation dataset."
        )
    return _infer_pairs_file_from_image_list(image_list_file)


def _validate_attribute_vector(
    value,
    field: str,
    row_idx: int,
    pairs_file: Path,
    expected_width: Optional[int],
    missing_ids_by_attr=None,
) -> Tuple[List[int], int]:
    """Validate one attribute vector and return integer values plus width."""
    if not isinstance(value, list):
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' must be a list."
        )
    if not value:
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' must not be empty."
        )
    if expected_width is not None and len(value) != expected_width:
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' has length "
            f"{len(value)}, expected {expected_width}."
        )
    if any(type(attribute_id) is not int for attribute_id in value):
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' must contain integers."
        )
    values = normalize_missing_match_values(value, missing_ids_by_attr)
    return values, len(values)


def _load_attribute_vocab_contract(
    pairs_file: Path,
    required: bool,
) -> Tuple[Optional[str], Optional[int]]:
    """Load the ordered scalar-attribute vocabulary identity."""
    vocab_path = pairs_file.with_name("attribute_vocab.json")
    if not vocab_path.is_file():
        if required:
            raise ValueError(
                "Multi-dataset attribute metadata requires "
                f"{vocab_path}."
            )
        return None, None

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict):
        raise ValueError(f"{vocab_path} must contain a JSON object.")

    attributes = vocab.get("attributes")
    if (
        not isinstance(attributes, list)
        or not attributes
        or any(
            not isinstance(attribute, str) or not attribute.strip()
            for attribute in attributes
        )
        or len(attributes) != len(set(attributes))
    ):
        raise ValueError(
            f"{vocab_path} must contain a non-empty ordered list of unique "
            "attribute names."
        )

    canonical = json.dumps(
        vocab,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), len(attributes)


def _load_accessory_vocab(pairs_file: Path) -> Tuple[set, str]:
    """Load the accessory vocabulary used to validate and compare ID lists."""
    vocab_path = pairs_file.with_name("accessory_vocab.json")
    if not vocab_path.is_file():
        raise ValueError(
            f"Accessory metadata in {pairs_file} requires {vocab_path}."
        )
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict) or vocab.get("unknown_id") != 0:
        raise ValueError(
            f"{vocab_path} must be an object with unknown_id set to 0."
        )
    value_to_id = vocab.get("value_to_id")
    if not isinstance(value_to_id, dict):
        raise ValueError(f"{vocab_path} must contain a value_to_id mapping.")

    valid_ids = set()
    for value, accessory_id in value_to_id.items():
        if (
            isinstance(accessory_id, bool)
            or not isinstance(accessory_id, int)
            or accessory_id < 0
        ):
            raise ValueError(
                f"{vocab_path} maps {value!r} to invalid ID {accessory_id!r}."
            )
        if accessory_id > 0:
            valid_ids.add(accessory_id)
    canonical = json.dumps(
        vocab,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return valid_ids, hashlib.sha256(canonical).hexdigest()


def _validate_accessory_ids(
    value,
    field: str,
    row_idx: int,
    pairs_file: Path,
    valid_ids: set,
) -> List[int]:
    """Validate one unpadded accessory-ID list."""
    if not isinstance(value, list):
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' must be a list."
        )
    if any(
        isinstance(accessory_id, bool)
        or not isinstance(accessory_id, int)
        or accessory_id <= 0
        for accessory_id in value
    ):
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' must contain "
            "positive integer IDs."
        )
    if len(value) != len(set(value)):
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' contains duplicate IDs."
        )
    unknown_ids = sorted(set(value) - valid_ids)
    if unknown_ids:
        raise ValueError(
            f"{pairs_file} row {row_idx} field '{field}' contains IDs absent "
            f"from accessory_vocab.json: {unknown_ids}."
        )
    return list(value)


def _validate_validation_pairs_alignment(
    dataset: dict,
    pairs_file: Path,
    pairs: List[dict],
    image_list: List[str],
) -> None:
    """Verify validation pairs describe the image and caption at each row."""
    caption_dir = Path(
        dataset.get('caption_dir') or dataset['image_dir']
    )
    caption_file_suffix = dataset.get('caption_file_suffix', '.txt')

    for row_idx, (pair, image_name) in enumerate(
        zip(pairs, image_list)
    ):
        if not isinstance(pair, dict):
            raise ValueError(
                f"{pairs_file} row {row_idx} must be a JSON object."
            )

        expected_name = Path(image_name).as_posix()
        pair_name = Path(
            str(pair.get("unique_name") or "").strip()
        ).as_posix()
        if pair_name != expected_name:
            raise ValueError(
                f"{pairs_file} row {row_idx} unique_name "
                f"{pair_name!r} does not match image list entry "
                f"{expected_name!r}."
            )

        caption_path = (
            caption_dir / Path(image_name).with_suffix(caption_file_suffix)
        )
        try:
            caption = caption_path.read_text(encoding='utf-8').strip()
        except OSError as error:
            raise ValueError(
                f"{pairs_file} row {row_idx} could not read caption file "
                f"{caption_path}: {error}"
            ) from error
        pair_caption = str(pair.get("caption") or "").strip()
        if pair_caption != caption:
            raise ValueError(
                f"{pairs_file} row {row_idx} caption {pair_caption!r} "
                f"does not match caption file {caption_path} content "
                f"{caption!r}."
            )


def _load_attribute_metadata(
    dataset: dict,
    mode: str,
    image_list: List[str],
    expected_width: Optional[int] = None,
    require_attribute_vocab: bool = False,
) -> Tuple[List[Dict[str, List[int]]], int, Optional[str], Optional[str]]:
    """Load image/text attribute vectors aligned with one image list."""
    pairs_file = _resolve_attribute_pairs_file(dataset, mode)
    if not pairs_file.is_file():
        raise ValueError(
            "include_attribute_metadata requires a valid pairs metadata file, "
            f"got {pairs_file}."
        )

    with open(pairs_file, 'r') as f:
        pairs = json.load(f)
    if len(pairs) != len(image_list):
        raise ValueError(
            f"{pairs_file} has {len(pairs)} items but image list has "
            f"{len(image_list)}."
        )
    if mode != 'train':
        _validate_validation_pairs_alignment(
            dataset,
            pairs_file,
            pairs,
            image_list,
        )

    attribute_vocab_digest, attribute_vocab_width = (
        _load_attribute_vocab_contract(
            pairs_file,
            required=require_attribute_vocab,
        )
    )
    if attribute_vocab_digest is None:
        vocab_path = pairs_file.with_name("attribute_vocab.json")
        logging.warning(
            "Attribute metadata is enabled for %s, but %s is missing. "
            "Values such as 'not visible' cannot be normalized and will be "
            "treated as literal attribute IDs.",
            pairs_file,
            vocab_path,
        )
    metadata_rows = []
    width = expected_width
    missing_ids_by_attr = load_missing_match_value_ids(pairs_file)
    accessory_metadata_present = None
    valid_accessory_ids = set()
    accessory_vocab_digest = None
    for row_idx, pair in enumerate(pairs):
        accessory_fields_present = [
            field in pair for field in _ACCESSORY_METADATA_FIELDS
        ]
        if any(accessory_fields_present) and not all(accessory_fields_present):
            raise ValueError(
                f"{pairs_file} row {row_idx} must provide both "
                "image_accessory_ids and text_accessory_ids."
            )
        row_has_accessories = all(accessory_fields_present)
        if accessory_metadata_present is None:
            accessory_metadata_present = row_has_accessories
            if row_has_accessories:
                valid_accessory_ids, accessory_vocab_digest = (
                    _load_accessory_vocab(pairs_file)
                )
        elif accessory_metadata_present != row_has_accessories:
            raise ValueError(
                f"{pairs_file} has inconsistent accessory metadata fields."
            )
        row = {}
        for field in _ATTRIBUTE_METADATA_FIELDS:
            if field not in pair:
                raise ValueError(
                    f"{pairs_file} row {row_idx} missing '{field}'."
                )
            values, field_width = _validate_attribute_vector(
                pair[field],
                field,
                row_idx,
                pairs_file,
                width,
                missing_ids_by_attr,
            )
            if width is None:
                width = field_width
            if attribute_vocab_width is not None and (
                field_width != attribute_vocab_width
            ):
                raise ValueError(
                    f"{pairs_file} field '{field}' has width {field_width}, "
                    "but attribute_vocab.json defines "
                    f"{attribute_vocab_width} attributes."
                )
            row[field] = values
        if accessory_metadata_present:
            for field in _ACCESSORY_METADATA_FIELDS:
                row[field] = _validate_accessory_ids(
                    pair[field],
                    field,
                    row_idx,
                    pairs_file,
                    valid_accessory_ids,
                )
            if not set(row["text_accessory_ids"]).issubset(
                row["image_accessory_ids"]
            ):
                raise ValueError(
                    f"{pairs_file} row {row_idx} text accessories are not "
                    "contained in the paired image accessories."
                )
        metadata_rows.append(row)

    return (
        metadata_rows,
        width,
        attribute_vocab_digest,
        accessory_vocab_digest,
    )


def _load_train_pairs_metadata(
    pairs_file: Path,
    image_list: List[str],
    load_captions: bool = True,
) -> Optional[Tuple[List[str], Optional[List[str]], Tuple[str, ...]]]:
    """Load query_type and optional caption metadata aligned with one image_list."""
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)
    if len(pairs) != len(image_list):
        logging.warning(
            f"{pairs_file} has {len(pairs)} items but image list has "
            f"{len(image_list)}; balanced sampling disabled."
        )
        return None

    query_types: List[str] = []
    captions: List[str] = []
    has_all_captions = True
    for p in pairs:
        query_types.append(p.get('query_type', 'easy'))
        if not load_captions:
            continue
        cap = p.get('caption')
        if cap is None:
            has_all_captions = False
        elif has_all_captions:
            captions.append(cap)

    if not load_captions:
        captions = None
    elif not has_all_captions:
        logging.warning(
            f"{pairs_file} entries missing 'caption'; falling back to "
            "query-type frequency balancing only."
        )
        captions = None

    return query_types, captions, query_types_order_from_pairs(pairs)


def _resolve_train_pairs_file(
    dataset: dict,
    fallback_train_pairs_file: Optional[str],
    dataset_count: int,
) -> Optional[Path]:
    """Resolve train_pairs_file from a dataset config, keeping legacy single-dataset fallback."""
    train_pairs_file = dataset.get('train_pairs_file')
    if train_pairs_file:
        return Path(train_pairs_file)
    if fallback_train_pairs_file and dataset_count == 1:
        return Path(fallback_train_pairs_file)
    return None


def _load_combined_train_pairs_metadata(
    datasets: List[dict],
    fallback_train_pairs_file: Optional[str] = None,
    load_captions: bool = True,
) -> Optional[Tuple[List[str], Optional[List[str]], Tuple[str, ...]]]:
    """Load and concatenate train_pairs metadata for every dataset config in dataloader order."""
    all_query_types: List[str] = []
    all_captions: Optional[List[str]] = []
    dataset_count = len(datasets)

    if fallback_train_pairs_file and dataset_count > 1:
        logging.warning(
            "The get_custom_dataloader train_pairs_file argument is ignored for "
            "multi-dataset training; set train_pairs_file inside each dataset block."
        )

    for dataset_idx, dataset in enumerate(datasets):
        image_list_file = dataset.get('image_list_file')
        if not image_list_file:
            logging.warning(
                "Balanced query type sampling requires image_list_file for every "
                f"training dataset; dataset[{dataset_idx}] is missing it."
            )
            return None

        pairs_file = _resolve_train_pairs_file(
            dataset, fallback_train_pairs_file, dataset_count
        )
        if not pairs_file or not pairs_file.is_file():
            logging.warning(
                "Balanced query type sampling requires a valid train_pairs_file for "
                f"every training dataset; dataset[{dataset_idx}] has {pairs_file}."
            )
            return None

        image_list = _read_image_list(image_list_file)
        metadata = _load_train_pairs_metadata(
            pairs_file, image_list, load_captions=load_captions
        )
        if metadata is None:
            return None

        query_types, captions, _ = metadata
        all_query_types.extend(query_types)
        if captions is None:
            all_captions = None
        elif all_captions is not None:
            all_captions.extend(captions)

    if not all_query_types:
        logging.warning("No train_pairs metadata loaded; balanced sampling disabled.")
        return None

    return (
        all_query_types,
        all_captions,
        query_types_order_from_types(all_query_types),
    )


def get_custom_dataloader(
    datasets: List[dict],
    batch_size: int = 32,
    transform: Callable = None,
    tokenizer: Callable = None,
    num_workers: int = 0,
    seed: int = 42,
    zero_shot_eval: bool = False,
    mapping=None,
    shuffle=True,
    pin_memory=True,
    is_distributed=None,
    mode='train',
    train_pairs_file: Optional[str] = None,
    balance_query_types: bool = False,
    unique_caption_per_batch: bool = True,
    include_attribute_metadata: bool = False,
):
    """
    Creates a DataLoader for custom filesystem-based image-text datasets.

    Args:
        datasets (List[dict]): List of dataset configurations.
        batch_size (int): Size of batches.
        transform (Callable): Transform function for images.
        tokenizer (Callable): Tokenizer function for texts.
        num_workers (int): Number of subprocesses to use for data loading.
        seed (int): Random seed for reproducibility.
        zero_shot_eval (bool): Flag for zero-shot evaluation.
        mapping (Optional[dict]): Mapping for text transformations.
        shuffle (bool): Flag to shuffle data.
        pin_memory (bool): Flag to pin memory.
        is_distributed (Optional[bool]): Flag for distributed training.
        mode (str): Mode for the DataLoader ('train' or 'val').
        train_pairs_file (Optional[str]): Legacy single-dataset fallback path to train_pairs.json.
        balance_query_types (bool): If True and train_pairs_file metadata is set, use balanced sampling over query types.
        unique_caption_per_batch (bool): If True, balanced batches keep each caption string unique.
        include_attribute_metadata (bool): If True, include image/text attribute
            tensors from pairs metadata as a third batch item.

    Returns:
        DataLoader: A DataLoader for the specified datasets.
    """
    # Set the random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = ImageTextDataset(
        datasets=datasets,
        transform=transform,
        tokenizer=tokenizer,
        zero_shot_eval=zero_shot_eval,
        mapping=mapping,
        mode=mode,
        include_attribute_metadata=include_attribute_metadata,
    )
    dataloader_kwargs = {}

    use_balanced = mode == 'train' and balance_query_types
    query_type_per_index = None
    caption_per_index = None
    query_types_order = None
    pairs_metadata = None
    if use_balanced:
        pairs_metadata = _load_combined_train_pairs_metadata(
            datasets,
            train_pairs_file,
            load_captions=unique_caption_per_batch,
        )
        if pairs_metadata is not None:
            query_type_per_index, caption_per_index, query_types_order = pairs_metadata
            if len(query_type_per_index) != len(dataset):
                logging.warning(
                    "Combined train_pairs metadata has "
                    f"{len(query_type_per_index)} items but dataset has {len(dataset)}; "
                    "balanced sampling disabled."
                )
                query_type_per_index = None
                caption_per_index = None
                query_types_order = None
        if query_type_per_index is None:
            use_balanced = False
            logging.warning(
                "Balanced query type sampling disabled (missing or invalid train_pairs_file)."
            )
        else:
            logging.info(
                f"Loaded train_pairs metadata for {len(datasets)} dataset(s), "
                f"{len(query_type_per_index)} samples."
            )

    if mode == 'train':
        # MetadataMaskedSigLipLoss gather mode requires the same local batch
        # shape on every rank, so every custom training path emits only full
        # batches.
        train_batch_sampler_set = False
        if (
            use_balanced
            and unique_caption_per_batch
            and caption_per_index is not None
        ):
            num_replicas = 1
            rank = 0
            if is_distributed and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            try:
                dataloader_kwargs['batch_sampler'] = BalancedUniqueCaptionBatchSampler(
                    num_samples=len(dataset),
                    query_type_per_index=query_type_per_index,
                    caption_per_index=caption_per_index,
                    batch_size=batch_size,
                    num_replicas=num_replicas,
                    rank=rank,
                    seed=seed,
                    query_types_order=query_types_order,
                )
                train_batch_sampler_set = True
                logging.info(
                    "Using balanced batches with at most one row per caption string per batch "
                    f"(query types: {', '.join(query_types_order)})."
                )
            except ValueError as e:
                logging.warning(
                    f"Cannot use unique-caption balanced batches ({e}); "
                    "falling back to query-type frequency balancing."
                )
                caption_per_index = None
        if (
            not train_batch_sampler_set
            and use_balanced
            and query_type_per_index is not None
        ):
            num_replicas = 1
            rank = 0
            if is_distributed and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            balanced_sampler = BalancedByQueryTypeSampler(
                num_samples=len(dataset),
                query_type_per_index=query_type_per_index,
                num_replicas=num_replicas,
                rank=rank,
                seed=seed,
                replacement=True,
            )
            dataloader_kwargs['batch_sampler'] = BatchSampler(
                balanced_sampler, batch_size, drop_last=True
            )
            train_batch_sampler_set = True
            logging.info(
                "Using balanced sampling across query types "
                "(unique-caption batching disabled)."
            )
        if not train_batch_sampler_set:
            if is_distributed:
                dataloader_kwargs['batch_sampler'] = BatchSampler(
                    distributed.DistributedSampler(dataset, shuffle=True),
                    batch_size,
                    drop_last=True,
                )
            else:
                dataloader_kwargs['batch_sampler'] = BatchSampler(
                    RandomSampler(dataset), batch_size, drop_last=True
                )
    else:
        dataloader_kwargs['batch_size'] = batch_size
        dataloader_kwargs['shuffle'] = shuffle

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataloader_kwargs
    )
    return dataloader
