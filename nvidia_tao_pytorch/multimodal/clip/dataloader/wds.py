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

"""CLIP wds dataloader module."""

import io
import random
import urllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import boto3
import braceexpand
import webdataset as wds
from botocore.config import Config
from torch.utils.data import DataLoader, IterableDataset
from webdataset.tariterators import (
    base_plus_ext,
    gopen,
    tar_file_expander,
    valid_sample,
)

from nvidia_tao_pytorch.multimodal.clip.utils.logger import RankedLogger

logger = RankedLogger(__name__)


class ResumableShardList(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls: list[str] = None,
        root: str | Path = None,
        samples_per_file: int = 10000,
        batch_size: int = 32,
        seed: int = 42,
        resume_step: int | None = None,
    ):
        """Initialize the ResumableShardList.

        :param urls: a list of URLs as a Python list or brace notation string
        :param root: the root directory if urls are not provided
        :param samples_per_file: number of samples per file
        :param batch_size: total batch size (batch size per GPU * number of GPUs)
        :param seed: random seed for shuffling
        :param resume_step: step to resume training from
        """
        super().__init__()

        if urls is None and root is None:
            raise ValueError("Either 'urls' or 'root' must be provided")

        if urls is not None:
            if isinstance(urls, str):
                if root is None:
                    root = Path(urls).parent

                with open(urls) as stream:
                    urls = [
                        str(Path(root) / line.strip())
                        for line in stream
                        if line.strip()
                    ]

            self.urls = [
                i for url in urls for i in braceexpand.braceexpand(url)]
        else:
            self.urls = [str(i) for i in Path(root).rglob("*.tar")]

        if not self.urls or not isinstance(self.urls[0], str):
            raise ValueError("No valid shard URLs found")
        logger.info(f"Found {len(self.urls)} shards")

        self.samples_per_file = samples_per_file
        self.batch_size = batch_size
        self.seed = seed
        self.resume_step = resume_step
        self.is_first_epoch = True
        self.current_epoch = 0
        self.skip_files = 0

        if self.resume_step is not None:
            # we need to handle resume
            # suppose our resume step is 10000 (10000 optimizer updates)
            # we will have processed 10000 * batch_size samples
            # thus our current epoch is 10000 * batch_size // samples_per_file
            total_samples = len(self.urls) * self.samples_per_file
            seen_samples = int(self.resume_step) * self.batch_size
            self.current_epoch += seen_samples // total_samples
            self.skip_files = (seen_samples %
                               total_samples) // self.samples_per_file

    @property
    def num_samples(self):
        """Return the total number of samples across all shards."""
        return len(self.urls) * self.samples_per_file

    def __len__(self):
        """Return the number of URLs in the dataset."""
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards.

        This method shuffles the URLs and yields them for processing,
        taking into account the current epoch and any files to skip.
        """
        urls = self.urls.copy()

        skip_files = self.skip_files if self.is_first_epoch else 0

        # shuffle urls
        urls.sort()
        random.Random(self.seed + self.current_epoch).shuffle(urls)
        logger.info(
            f"WDS next epoch {self.current_epoch} skip_files {skip_files}")

        urls = urls[skip_files:]

        for url in urls:
            yield {"url": url}

        self.is_first_epoch = False
        self.current_epoch += 1


class InfinityResumableShardList(ResumableShardList):
    """An infinite iterable dataset that repeats the ResumableShardList."""

    def __iter__(self):
        """Yield from the parent iterator indefinitely."""
        while True:
            yield from super().__iter__()


def filter_no_caption_or_no_image(sample):
    """Filter samples to ensure they contain both captions and images.

    :param sample: a dictionary containing sample data
    :return: True if the sample contains both caption and image data, False otherwise
    """
    has_caption = "txt" in sample or "text" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        if not isinstance(filesample, dict):
            raise TypeError(f"Expected dict, got {type(filesample).__name__}")
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None or
            prefix != current_sample["__key__"] or
            suffix in current_sample  # pylint: disable=E1135
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = {
                "__key__": prefix,
                "__url__": filesample["__url__"],
            }
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


@lru_cache()
def get_s3_client(
    username: str, password: str, endpoint_url: str = None, region_name: str = None
):
    """Create an S3 client using provided credentials.

    :param username: AWS access key ID
    :param password: AWS secret access key
    :param endpoint_url: optional custom S3 endpoint URL
    :param region_name: optional AWS region name
    :return: configured S3 client
    """
    config = Config(connect_timeout=120, read_timeout=1200,
                    retries={"max_attempts": 5})

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=username,
        aws_secret_access_key=password,
        config=config,
        region_name=region_name,
    )


def get_s3_client_by_uri(url) -> tuple[boto3.client, str, str]:
    """Extract S3 client and bucket/key from a given S3 URL."""
    url = urllib.parse.urlparse(url)
    bucket = url.hostname
    key = url.path[1:]
    qs = urllib.parse.parse_qs(url.query)
    endpoint = qs.get("endpoint", ["https://pdx.s8k.io"])[0]
    region = qs.get("region", ["us-east-1"])[0]

    return get_s3_client(url.username, url.password, endpoint, region), bucket, key


def url_opener(
    data: Iterable[dict[str, Any]],
    handler: Callable[[Exception], bool] = log_and_continue,
    **kw: dict[str, Any],
):
    """Open URLs and yield a stream of url+stream pairs.

    Args:
        data: iterator over dict(url=...)
        handler: exception handler.
        kw: keyword arguments for gopen.gopen.

    Yields:
        a stream of url+stream pairs.
    """
    for sample in data:
        if not isinstance(sample, dict):
            raise TypeError(f"Expected dict, got {type(sample).__name__}: {sample}")
        if "url" not in sample:
            raise KeyError("Sample missing required 'url' key")
        url = sample["url"]

        try:
            if url.startswith("s3://"):
                client, bucket, key = get_s3_client_by_uri(url)
                obj = client.get_object(Bucket=bucket, Key=key)
                buffer = obj["Body"].read()

                with io.BytesIO(buffer) as stream:
                    sample.update(stream=stream)
                    yield sample
            else:
                stream = gopen.gopen(url, **kw)
                sample.update(stream=stream)
                yield sample
        except Exception as exn:
            exn.args = exn.args + (url,)
            if handler(exn):
                continue
            else:
                break


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    """Extract samples from a tarfile without throwing exceptions.

    :param src: source iterator providing tarfile data
    :param handler: exception handler
    :yield: extracted samples as dictionaries
    """
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def get_train_dataloader(
    urls: list[str] = None,
    root: str | Path = None,
    batch_size: int = 32,
    samples_per_file: int = 10000,
    transform: Callable | None = None,
    seed: int = 42,
    resume_step: int | None = None,
    num_workers: int = 0,
    world_size: int = 1,
    pin_memory=True
):
    """Create a DataLoader for training from URLs or directory.

    :param urls: list of URLs for the dataset
    :param root: root directory if URLs are not provided
    :param batch_size: total batch size (per GPU * number of GPUs)
    :param samples_per_file: number of samples per file
    :param transform: optional transformation function for samples
    :param seed: random seed for shuffling
    :param resume_step: step to resume training from
    :param num_workers: number of worker threads for data loading
    :param world_size: number of GPUs
    :param pin_memory: whether to pin memory
    :return: DataLoader for the dataset
    """
    if transform is None:

        def transform(x):
            return x

    base = InfinityResumableShardList(
        urls=urls,
        root=root,
        # The batch size is per GPU, so we need to multiply it by the number of GPUs
        batch_size=batch_size * world_size,
        samples_per_file=samples_per_file,
        seed=seed,
        resume_step=resume_step,
    )

    dataset = wds.DataPipeline(
        base,
        wds.split_by_node,
        wds.split_by_worker,
        tarfile_to_samples_nothrow,
        wds.shuffle(
            bufsize=5000,
            initial=1000,
        ),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.to_tuple("jpg;jpeg;png;webp", "txt;text",
                     handler=log_and_continue),
        wds.map(lambda x: transform(x), handler=log_and_continue),
        wds.batched(batch_size, partial=False),
    )

    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
    )
