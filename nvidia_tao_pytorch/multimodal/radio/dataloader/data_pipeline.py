# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline assembly for WebDataset-based distillation training.

Builds a complete data pipeline in three parts — I/O, decode, augment —
and wraps the result in a WebLoader + optional GPU prefetcher and
LongDatasetAdaptor for fixed-length epochs.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
import hashlib
import io
import json
import os
import random
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Union,
)
from logging import getLogger, Logger
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.distributed as dist

import webdataset as wds

from nvidia_tao_pytorch.multimodal.radio.dataloader.adapter import LongDatasetAdaptor
from nvidia_tao_pytorch.multimodal.radio.dataloader.filters.hash_count_filter import (
    get_hash_database,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.filters.native_resolution_filter import (
    NativeResolutionFilter,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.filters.uniform_color_filter import (
    UniformColorFilter,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.ops.fast_to_tensor import fast_to_tensor
from nvidia_tao_pytorch.multimodal.radio.dataloader.misc.mixture_emitter import (
    MixtureEmitter,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.prefetcher import IterablePrefetcher
from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.multi_pipe_sampler import (
    MultiPipeSampler,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.multi_stream_shuffle import (
    MultiStreamShuffle,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.sample_weight_injector import (
    SampleWeightInjector,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.stages.utils import (
    SharedEpoch,
    _SAMPLE_SHUFFLE_SIZE,
    identity,
    ignore_and_log,
    md5_str_to_bytes,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.base import ScopedRNG
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.defer_image import DeferImage
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.equivariant_collate import (
    equivariant_collate,
)
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.pipeline import get_pipeline
from nvidia_tao_pytorch.multimodal.radio.dataloader.transforms.quad import Quad


# ---------------------------------------------------------------------------
# Listing helpers
# ---------------------------------------------------------------------------

def _normalize_listing(
    data_listing: Union[str, Iterable[Union[str, Tuple[str, str]]]],
) -> Iterable[Tuple[str, float]]:
    """Normalise dataset listing to ``[(path, rate), ...]`` form.

    Accepts comma-separated strings (``"path1:0.5,path2:0.5"``), lists of
    strings, or lists of ``[path, rate]`` pairs.
    """
    data_listing = deepcopy(data_listing)

    if isinstance(data_listing, str):
        data_listing = data_listing.split(',')

    for i in range(len(data_listing)):
        listing = data_listing[i]
        if isinstance(listing, str):
            parts = listing.split(':')
            if len(parts) == 1:
                data_path = parts[0]
                rate = 1
            else:
                data_path = parts[0]
                rate = float(parts[1])
            data_listing[i] = [data_path, rate]
        else:
            data_path = listing[0]
            if len(listing) > 1:
                rate = float(listing[1])
            data_listing[i] = [data_path, rate]

    return data_listing


def _simplify_listing(
    data_listing: Iterable[Tuple[str, float]],
):
    """Deduplicate paths and normalise rates to sum to 1.0."""
    unique_listings = dict()
    total_rate = 0
    for data_path, rate in data_listing:
        if data_path in unique_listings:
            unique_listings[data_path] += rate
        else:
            unique_listings[data_path] = rate
        total_rate += rate
    return [(k, v / total_rate) for k, v in unique_listings.items()]


def _expand_listing(
    data_listing: List[Tuple[str, float]],
    batch_size: int,
) -> Iterable[Tuple[List[str], float]]:
    """Distribute tar URLs across distributed ranks.

    Uses weighted proportional splitting with ``MixtureEmitter`` for
    remainder assignment so that every rank gets a fair share of shards.
    """
    ret = []

    path_rank_hist_map = defaultdict(dict)

    data_listing.sort()
    for i in range(len(data_listing)):
        data_listing[i] = (data_listing[i][0], data_listing[i][1] * batch_size)

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        all_listings = [None for _ in range(world_size)]
        dist.all_gather_object(all_listings, data_listing)

        for curr_rank, listings in enumerate(all_listings):
            for data_path, rate in listings:
                path_rank_hist_map[data_path][curr_rank] = rate
    else:
        rank = 0
        world_size = 1
        for data_path, rate in data_listing:
            path_rank_hist_map[data_path][rank] = rate

    rng = np.random.default_rng(42)

    for data_path, rank_rate in path_rank_hist_map.items():
        if os.path.isfile(data_path):
            urls = [data_path]
        else:
            urls = glob(f'{data_path}/**/*.tar', recursive=True)
        if not urls:
            raise ValueError(f'No tar files found in {data_path}')

        if world_size > 1:
            urls.sort()
            rng.shuffle(urls)

            total_rate = sum(v for v in rank_rate.values())
            world_rate = {k: v / total_rate for k, v in rank_rate.items()}

            my_urls = []
            offset = 0
            for curr_rank, prob in world_rate.items():
                num_keep = int(prob * len(urls))
                if num_keep == 0:
                    warnings.warn(
                        f'Rank {curr_rank} has no urls assigned for '
                        f'dataset {data_path}'
                    )
                if curr_rank == rank:
                    end = max(offset + num_keep, offset + 1)
                    my_urls.extend(urls[offset:end])
                offset += num_keep
            if offset < len(urls):
                prob_dist = np.zeros((world_size,), dtype=np.float32)
                for curr_rank, prob in world_rate.items():
                    prob_dist[curr_rank] = prob

                chooser = MixtureEmitter(prob_dist.tolist())
                for i in range(offset, len(urls)):
                    dest_rank = chooser.pick()
                    if dest_rank == rank:
                        my_urls.append(urls[i])
        else:
            my_urls = urls

        if my_urls:
            my_urls.sort()
            ret.append((data_path, my_urls, rank_rate[rank] / batch_size))

    return ret


def _reduce_urls_to_worker(
    urls: List[str],
    rng: random.Random,
) -> List[Tuple[int, str]]:
    """Partition URLs across DataLoader workers within a rank."""
    w_info = torch.utils.data.get_worker_info()

    if w_info is not None:
        w_id = w_info.id
        w_num = w_info.num_workers
    else:
        w_id = 0
        w_num = 1

    if len(urls) > w_num:
        ret = list((i, urls[i]) for i in range(w_id, len(urls), w_num))
    else:
        if w_id == 0:
            rank = dist.get_rank() if dist.is_initialized() else 0
            warnings.warn(
                f'Not enough urls ({len(urls)}) for workers ({w_num}) '
                f'in rank {rank}'
            )
        c = rng.randint(0, len(urls) - 1)
        ret = [(c, urls[c])]

    return ret


# ---------------------------------------------------------------------------
# Sample processing helpers
# ---------------------------------------------------------------------------

def _remove_pkl(data):
    """Strip pickle data from webdataset samples."""
    for sample in data:
        if 'pkl' in sample:
            del sample['pkl']
        yield sample


def _source_enricher(data_path: str):
    """Add ``__dataset__`` field to each sample for source tracking."""
    def _stage(data):
        for sample in data:
            sample['__dataset__'] = data_path
            yield sample
    return _stage


_SUPPORTED_IMAGE_FORMATS = 'jpg;jpeg;png;gif;webp;bmp;tiff;img'


def _sample_hook():
    """Extract image key (SHA256 / MD5 / UID) from sample JSON metadata."""
    formats = _SUPPORTED_IMAGE_FORMATS.split(';')

    def _stage(data):
        for sample in data:
            js = sample.get('json', None)
            skey = None
            if js is not None:
                sha = js.get('sha256', None)
                if sha is not None:
                    skey = sha
                else:
                    md5 = js.get('md5', None)
                    if md5 is not None:
                        skey = md5
                    else:
                        skey = js.get('uid', None)
            if skey is None:
                for fmt in formats:
                    if fmt in sample:
                        skey = hashlib.sha256(
                            sample[fmt].tobytes()
                        ).hexdigest()
                        break
            sample['image_key'] = skey

            if js is not None and 'caption' in js and 'txt' not in sample:
                sample['txt'] = js['caption']

            yield sample
    return _stage


def _image_filter(formats: str = _SUPPORTED_IMAGE_FORMATS):
    """Keep only samples that contain a supported image format."""
    keys = frozenset(formats.split(';'))

    def _stage(data):
        for sample in data:
            if not keys.isdisjoint(sample.keys()):
                yield sample
            else:
                for k in sample.keys():
                    final_part = k.split('.')[-1]
                    if final_part in keys:
                        sample[final_part] = sample[k]
                        yield sample
                        break

    return _stage


def _img_format_handler(img_idx: int = 0, mode='RGB', error_handler=None):
    """Convert raw image bytes to PIL Image (safety net after wds.decode)."""
    mode = mode.upper()

    def _stage(data):
        for sample in data:
            img = sample[img_idx]
            if isinstance(img, bytes):
                try:
                    with io.BytesIO(img) as stream:
                        img = Image.open(stream)
                        img.load()
                        img = img.convert(mode)
                except UnidentifiedImageError as e:
                    if error_handler is not None:
                        error_handler(e)
                    continue
            ret = sample[:img_idx] + (img,) + sample[img_idx + 1:]
            yield ret
    return _stage


def _prepare_image(num_replicas: int):
    """Convert PIL → tensor and wrap in ``(DeferImage, Quad)`` tuples.

    Replicates ``num_replicas`` times (1 student + N-1 teachers).
    """
    def _stage(data):
        for sample in data:
            img = fast_to_tensor(sample[0])
            ret = tuple(
                (DeferImage(img), Quad(img))
                for _ in range(num_replicas)
            ) + sample[1:]
            yield ret
    return _stage


def _sample_identity_dump(batch_size: int):
    """Dump ``__key__`` and ``__url__`` of raw samples for parity testing.

    Activated by env var ``PARITY_DUMP_DIR``.  Captures the first
    ``PARITY_DUMP_BATCHES * batch_size`` sample identities to a JSON
    file, proving both pipelines read the same samples in the same order.
    Inserted at the very start of decode stages (samples are still dicts).
    """
    dump_dir = os.environ.get("PARITY_DUMP_DIR")
    if not dump_dir:
        return identity

    max_samples = int(os.environ.get("PARITY_DUMP_BATCHES", "5")) * batch_size
    id_dir = os.path.join(dump_dir, "identity")
    os.makedirs(id_dir, exist_ok=True)
    logger = getLogger("parity_identity_dump")

    collected = []

    def _stage(data):
        for sample in data:
            if len(collected) < max_samples:
                collected.append({
                    "idx": len(collected),
                    "key": sample.get("__key__", ""),
                    "url": sample.get("__url__", ""),
                })
                if len(collected) == 1:
                    logger.info(
                        "Parity identity dump: capturing %d sample keys to %s",
                        max_samples, id_dir,
                    )
                if len(collected) == max_samples:
                    with open(os.path.join(id_dir, "sample_keys.json"), "w") as f:
                        json.dump(collected, f, indent=2)
                    logger.info("Parity identity dump: wrote %d keys", max_samples)
            yield sample
    return _stage


def _raw_sample_dump(batch_size: int):
    """Dump raw image tensors before augmentation for parity testing.

    Activated by env var ``PARITY_DUMP_DIR``.  Saves the first
    ``PARITY_DUMP_BATCHES * batch_size`` raw samples.
    """
    dump_dir = os.environ.get("PARITY_DUMP_DIR")
    if not dump_dir:
        return identity

    max_samples = int(os.environ.get("PARITY_DUMP_BATCHES", "5")) * batch_size
    raw_dir = os.path.join(dump_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    logger = getLogger("parity_raw_dump")

    def _stage(data):
        sample_idx = 0
        for sample in data:
            if sample_idx < max_samples:
                raw_tensor = sample[0][0].image.clone().cpu()
                torch.save(
                    {"sample_idx": sample_idx, "raw_img": raw_tensor},
                    os.path.join(raw_dir, f"raw_sample_{sample_idx:06d}.pt"),
                )
                if sample_idx == 0:
                    logger.info(
                        "Parity raw dump: saving %d samples to %s",
                        max_samples, raw_dir,
                    )
                sample_idx += 1
            yield sample
    return _stage


def _reset_rng(scoped_rng: ScopedRNG, batch_size: int):
    """Reset the shared ``ScopedRNG`` every *batch_size* samples.

    Keeps stochastic resolution in sync across ranks when using
    ``wds.batched``.
    """
    def _stage(data):
        ctr = 0
        for sample in data:
            if ctr == batch_size:
                scoped_rng.reset_seed()
                ctr = 0
            ctr += 1
            yield sample
    return _stage


# ---------------------------------------------------------------------------
# Pipeline config & state
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Pipeline configuration parameters."""

    steps_per_epoch: int = 2000
    workers: int = 8


class LoaderState:
    """Checkpointable state for training resumption.

    Wraps visit counters from ``MultiStreamShuffle`` and supports
    distributed ``state_dict()`` / ``restore()`` across ranks.
    """

    def __init__(self, state_objects):
        self.state_objects = state_objects

    def state_dict(self):
        """Return the per-rank loader state, gathered across ranks when distributed."""
        ret = [so.get_state() for so in self.state_objects]

        if dist.is_initialized():
            all_ranks = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_ranks, ret)
            ret = all_ranks
        else:
            ret = [ret]

        return ret

    def restore(self, checkpoint_state):
        """Restore the loader state from a previously-saved checkpoint."""
        rank = 0
        world_size = 1
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        if len(checkpoint_state) != world_size:
            warnings.warn(
                'The world size has changed between the checkpoint and now'
            )
            return False

        checkpoint_state = checkpoint_state[rank]

        if len(checkpoint_state) != len(self.state_objects):
            warnings.warn(
                "The checkpoint state doesn't match the number of "
                "loader states"
            )

        for chk_state, so in zip(checkpoint_state, self.state_objects):
            so.load_state(chk_state)
        return True


# ---------------------------------------------------------------------------
# I/O pipeline builder
# ---------------------------------------------------------------------------

def _get_dataset_pipeline(
    data_listing: Union[str, Iterable[Union[str, Tuple[str, str]]]],
    batch_size: int,
    is_train: bool,
    data_weight_mode: str,
    seed: int,
    shared_epoch: SharedEpoch,
    logger: Logger,
    split_by_node: bool = True,
):
    """Build I/O pipeline stages for reading shards from disk.

    For training: ``MultiStreamShuffle`` → ``SampleWeightInjector``
    → ``MultiPipeSampler`` (if multi-source).

    For validation: ``SimpleShardList`` → ``split_by_worker``
    → ``tarfile_to_samples``.
    """
    data_listing: Iterable[Tuple[str, float]] = _normalize_listing(data_listing)
    data_listing = _simplify_listing(data_listing)
    data_listing = _expand_listing(data_listing, batch_size)

    rank = dist.get_rank() if dist.is_initialized() else 0
    log_msg = f'Rank {rank}\n'
    for ds_path, urls, rate in data_listing:
        log_msg += f'\tRate {rate:.3f} - Path: {ds_path}\n'

    if rank == 0:
        logger.info('Datasets')
    if dist.is_initialized():
        for i in range(dist.get_world_size()):
            if i == rank:
                logger.info(log_msg)
            dist.barrier()
    else:
        logger.info(log_msg)

    state_objects = []

    io_pipelines, rates = [], []
    bufsize = max(min(batch_size * 100, _SAMPLE_SHUFFLE_SIZE), 1024)

    for data_path, urls, rate in data_listing:
        if is_train:
            multi_stream_shuffle = MultiStreamShuffle(
                urls=urls,
                seed=seed,
                epoch=shared_epoch,
                deterministic=True,
                reduce_urls_fn=_reduce_urls_to_worker,
                num_streams=1,
            )
            state_objects.append(multi_stream_shuffle)

            io_stages = [
                multi_stream_shuffle,
            ]

            hash_count_db = get_hash_database(data_path)
            if hash_count_db is not None and data_weight_mode is not None:
                injector = SampleWeightInjector(
                    database=hash_count_db,
                    key_extractor=md5_str_to_bytes(),
                    weight_mode=data_weight_mode,
                    batch_size=batch_size,
                    logger=logger,
                )
                io_stages.append(injector)

            io_stages.append(_source_enricher(data_path))
            io_pipeline = wds.DataPipeline(*io_stages)

        else:
            io_stages = [
                wds.SimpleShardList(urls),
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=ignore_and_log(logger)),
            ]

            io_stages.append(_source_enricher(data_path))
            io_pipeline = wds.DataPipeline(*io_stages)

        io_pipelines.append(io_pipeline)
        rates.append(rate)

    if len(io_pipelines) > 1:
        multi_io_pipeline = MultiPipeSampler(
            io_pipelines,
            rates,
            epoch=shared_epoch,
            seed=seed,
            bufsize=bufsize,
        )
        io_pipeline = wds.DataPipeline(multi_io_pipeline)
    else:
        io_pipeline = io_pipelines[0]

    return io_pipeline, state_objects


# ---------------------------------------------------------------------------
# Top-level pipeline builder
# ---------------------------------------------------------------------------

def get_data_pipeline(
    args: PipelineConfig,
    ds_listing: Union[str, Iterable[Union[str, Tuple[str, str]]]],
    input_sizes: List[Union[int, Tuple[int, int]]],
    patch_sizes: List[int],
    batch_size: int,
    is_train: Union[bool, List[bool]],
    epoch: Optional[Union[int, SharedEpoch]],
    seed: int,
    upsample_factors: List[int] = None,
    data_weight_mode: str = 'inv_frequency',
    prefetch: bool = True,
    label_extractor: Optional[Callable] = None,
    label_key: Optional[str] = None,
    split_by_node: bool = True,
    full_equivariance: bool = False,
    shift_equivariance: bool = False,
    stochastic_size_args: Optional[Dict[str, Any]] = None,
    stochastic_teachers: Optional[List[bool]] = None,
    include_keys: bool = False,
    include_dataset_source: bool = False,
    aug_config: Optional[Dict[str, Any]] = None,
    native_resolution_filter: Optional[Dict[str, Any]] = None,
):
    """Build the complete data pipeline (I/O → decode → augment).

    Returns ``(loader, shared_epoch, loader_state)`` where *loader* is the
    final iterable (WebLoader → IterablePrefetcher → LongDatasetAdaptor).
    """
    if isinstance(is_train, bool):
        is_train = [is_train] + ([False] * (len(input_sizes) - 1))

    if stochastic_teachers is None:
        stochastic_teachers = [False] * (len(input_sizes) - 1)

    max_input_size = max(
        t if isinstance(t, int) else max(t)
        for t in input_sizes
    )

    if upsample_factors is None:
        upsample_factors = [1 for _ in range(len(input_sizes) - 1)]

    rng = np.random.default_rng(seed)

    steps_per_epoch = args.steps_per_epoch if is_train[0] else None
    l_name = 'train' if is_train else 'val'
    logger = getLogger(f'{l_name}_data_pipeline')

    unified_seed = rng.bit_generator.random_raw()

    if epoch is None:
        epoch = -1
    if isinstance(epoch, int):
        epoch = SharedEpoch(epoch)

    # -- I/O stages --
    io_pipeline, state_objects = _get_dataset_pipeline(
        data_listing=ds_listing,
        batch_size=batch_size,
        is_train=is_train[0],
        data_weight_mode=data_weight_mode,
        seed=seed,
        shared_epoch=epoch,
        logger=logger,
        split_by_node=split_by_node,
    )

    # -- Decode stages --
    decode_stages: list = [
        _sample_identity_dump(batch_size),
        _remove_pkl,
        wds.decode('pilrgb', handler=ignore_and_log(logger)),
    ]

    addl_tuple = tuple()
    addl_map = dict()
    if include_keys:
        addl_map[len(addl_tuple)] = 'image_key'
        addl_tuple = addl_tuple + ('image_key',)

    if include_dataset_source:
        addl_map[len(addl_tuple)] = 'dataset'
        addl_tuple = addl_tuple + ('__dataset__',)

    if label_extractor is not None and label_key is not None:
        decode_stages.append(label_extractor)
        addl_map[len(addl_tuple)] = 'label'
        addl_tuple = addl_tuple + (label_key,)

    decode_stages.extend([
        _image_filter(),
        _sample_hook(),
        wds.to_tuple(
            _SUPPORTED_IMAGE_FORMATS, *addl_tuple,
            handler=ignore_and_log(logger),
        ),
        _img_format_handler(error_handler=ignore_and_log(logger)),
        UniformColorFilter(),
    ])
    if native_resolution_filter is not None:
        decode_stages.append(NativeResolutionFilter(**native_resolution_filter))
    decode_stages.extend([
        _prepare_image(num_replicas=len(input_sizes)),
    ])

    # -- Raw sample dump (parity testing, before augmentation) --
    decode_stages.append(_raw_sample_dump(batch_size))

    # -- Augmentation stages --
    scoped_rng = ScopedRNG(epoch.get_value())

    transforms = [
        get_pipeline(
            input_sizes[0], input_size, patch_size,
            t, i > 0, max_img_size=max_input_size,
            shift_equivariance=shift_equivariance,
            full_equivariance=full_equivariance,
            stochastic_size_args=stochastic_size_args,
            rng=rng, unified_seed=unified_seed,
            scoped_rng=scoped_rng,
            stochastic_teacher=(
                False if i == 0 else stochastic_teachers[i - 1]
            ),
            student_patch_size=(
                patch_sizes[0] if i == 0
                else patch_sizes[0] // upsample_factors[i - 1]
            ),
            aug_config=aug_config,
        )
        for i, (input_size, patch_size, t) in enumerate(
            zip(input_sizes, patch_sizes, is_train)
        )
    ]

    eqfn = equivariant_collate(
        len(input_sizes),
        patch_sizes=patch_sizes,
        extra_map=addl_map,
    )

    aug_stages: list = [
        _reset_rng(scoped_rng, batch_size),
        wds.map_tuple(*transforms, identity),
        wds.batched(
            batch_size, partial=not is_train[0], collation_fn=eqfn,
        ),
    ]

    # -- Assemble final pipeline --
    assert isinstance(io_pipeline, wds.DataPipeline)
    combined_stages = list(io_pipeline.pipeline)
    combined_stages.extend(decode_stages)
    combined_stages.extend(aug_stages)

    combined_pipeline = wds.DataPipeline(combined_stages)

    dl_generator = torch.Generator()
    dl_generator.manual_seed(seed)

    loader = wds.WebLoader(
        combined_pipeline,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        prefetch_factor=4 if args.workers > 0 else None,
        generator=dl_generator,
    )

    if prefetch:
        loader = IterablePrefetcher(loader)

    if steps_per_epoch is not None:
        loader = LongDatasetAdaptor(
            loader,
            steps_per_epoch,
            reset_each_epoch=not is_train[0],
            batch_size=batch_size,
        )

    return loader, epoch, LoaderState(state_objects)
