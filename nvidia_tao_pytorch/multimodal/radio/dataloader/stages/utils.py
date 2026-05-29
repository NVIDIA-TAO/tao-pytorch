# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundation utilities for the equivariant data pipeline.

SharedEpoch, deterministic seeding, URL expansion, and misc helpers
used throughout the pipeline stages.
"""

import braceexpand
import hashlib
import logging
from multiprocessing import Value
from typing import Any, Callable, Iterator, Iterable, List, Union

from torch.utils.data import get_worker_info

import webdataset as wds

logger = logging.getLogger(__name__)

# From OpenCLIP
# https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py#L238
_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 512 * 30
_SAMPLE_SHUFFLE_INITIAL = 512 * 10


class SharedEpoch:
    """Process-shared epoch counter backed by ``multiprocessing.Value``."""

    def __init__(self, epoch: int = 0):
        """Initialize the shared epoch counter."""
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        """Overwrite the shared epoch counter with ``epoch``."""
        self.shared_epoch.value = epoch

    def increment(self, ct: int = 1):
        """Atomically add ``ct`` to the shared epoch counter."""
        with self.shared_epoch.get_lock():
            self.shared_epoch.value += ct

    def get_value(self):
        """Return the current value of the shared epoch counter."""
        return self.shared_epoch.value


def ignore_and_log(log: logging.Logger):
    """Return a webdataset-compatible handler that logs and swallows exceptions."""
    def _ignore_exceptions(exn):
        log.warning(exn)
        return True
    return _ignore_exceptions


def identity(x):
    """Return ``x`` unchanged."""
    return x


def seed_from_tuple(*args):
    """Deterministically derive a 32-bit seed from arbitrary positional args."""
    b = repr(args).encode("utf-8")
    h = hashlib.sha256(b).hexdigest()
    return int(h, 16) % (2**32)


def pytorch_worker_seed(increment=0):
    """Get dataloader worker seed from pytorch."""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def expand_urls(urls, weights=None):
    """Expand brace-style URL listings and align per-shard sampling weights."""
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), \
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def iterator_exhauster(gen_iter_fn: Callable[[], Iterator]):
    """Repeatedly call ``gen_iter_fn`` to produce and drain iterators in sequence."""
    while True:
        curr_iter = gen_iter_fn()
        if curr_iter is None:
            break

        while True:
            try:
                v = next(curr_iter)
                yield v
            except StopIteration:
                break


def compute_md5_hash(value: Union[str, bytes]):
    """Return an ``md5`` hasher populated with the (possibly utf-8 encoded) value."""
    if value is None:
        logger.warning('Encountered null url')
        value = ''

    md5_hash = hashlib.md5()

    if isinstance(value, str):
        value = value.encode('utf-8')

    md5_hash.update(value)
    return md5_hash


def extract_caption(gen: Iterable):
    """Ensure each sample has a ``txt`` caption derived from its JSON metadata."""
    for data in gen:
        if 'txt' not in data:
            json = data['json']
            txt = json['text'] if 'text' in json else json['caption']
            data['txt'] = txt
        # Ignore empty captions
        if data['txt']:
            yield data


def _coerce_ints(items: List[str]):
    ret = []
    for item in items:
        try:
            item = int(item)
        except ValueError:
            pass
        ret.append(item)
    return ret


def get_item_recursive(obj, path: List[Union[str, int]]):
    """Walk ``path`` through ``obj`` using attribute or item access at each step."""
    if not path:
        return obj

    next_p = path[0]

    if not isinstance(next_p, int):
        try:
            next_obj = getattr(obj, next_p)
        except AttributeError:
            next_obj = obj[next_p]
    else:
        next_obj = obj[next_p]

    return get_item_recursive(next_obj, path[1:])


def set_item_recursive(obj, path: List[Union[str, int]], value: Any):
    """Assign ``value`` at ``path`` within ``obj`` using attribute or item access."""
    obj = get_item_recursive(obj, path[:-1])

    k = path[-1]

    if hasattr(obj, k):
        setattr(obj, k, value)
    else:
        obj[k] = value


def get_attribute_path(attr_path: str):
    """Split a dotted ``attr_path`` into a list, coercing numeric segments to ints."""
    return _coerce_ints(attr_path.split('.'))


def extract_dict_field(src_path: str, dest_path: str):
    """Return a pipeline stage that copies ``src_path`` to ``dest_path`` per sample."""
    src_path = get_attribute_path(src_path)
    dest_path = get_attribute_path(dest_path)

    def run(gen: Iterable):
        for data in gen:
            src_val = get_item_recursive(data, src_path)
            set_item_recursive(data, dest_path, src_val)
            yield data
    return run


def _get_json_key_extractor(key_names, decode=True):
    """Extract a value from raw JSON bytes without full parsing.

    Scans for ``"key":"value"`` or ``"key": "value"`` patterns in the raw
    bytes, avoiding a full json.loads() for throughput.  Will move to
    filters/ module in a later phase.
    """
    if not isinstance(key_names, (list, tuple)):
        key_names = [key_names]

    start_keys = [
        b'"' + key_name + suffix
        for key_name in key_names
        for suffix in [b'":"', b'": "']
    ]
    end_key = b'"'

    def key_extractor(data):
        j_data = data['json']

        key_start = -1
        matched_key = None
        for start_key in start_keys:
            key_start = j_data.find(start_key)
            if key_start != -1:
                matched_key = start_key
                break

        if key_start == -1 or matched_key is None:
            return None

        # Move past the matched start_key to reach the value
        val_start = key_start + len(matched_key)
        key_end = j_data.find(end_key, val_start)
        b_extract = j_data[val_start:key_end]
        if decode:
            return b_extract.decode()
        return b_extract

    return key_extractor


def md5_str_to_bytes():
    """Return a function that reads ``md5``/``sha256`` JSON keys as raw bytes."""
    key_extractor = _get_json_key_extractor([b'md5', b'sha256'])

    def to_bytes(sample):
        key = key_extractor(sample)
        ret = bytes.fromhex(key)
        return ret
    return to_bytes
