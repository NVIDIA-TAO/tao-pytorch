# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Video frame decoding for the video_clip dataloader.

A single public entry point, :func:`load_video_frames`, samples a fixed number
of PIL RGB frames from a clip. It tries PyAV first -- the in-process decoder the
TAO image builds against its restricted FFmpeg -- then the ffmpeg CLI, then
OpenCV, and logs which backend won.
"""

import json

import numpy as np
from PIL import Image

from nvidia_tao_pytorch.core.tlt_logging import logging


_DECODE_BACKEND_LOGGED = set()


def _log_decode_backend(name):
    """Log the winning decode backend once per process.

    A silent fall back from PyAV to the ffmpeg CLI is ~10x slower but otherwise
    invisible, which makes any throughput measurement unattributable. One line
    per process makes the active backend explicit.
    """
    if name in _DECODE_BACKEND_LOGGED:
        return
    _DECODE_BACKEND_LOGGED.add(name)
    logging.info("video_clip: decoding video with the %s backend", name)


def _linspace_indices(total_frames, num_frames):
    """Select a fixed number of frames from a clip."""
    if total_frames <= 0:
        raise ValueError("Cannot sample from an empty video clip")
    if total_frames >= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int)
    indices = np.full((num_frames,), total_frames - 1, dtype=int)
    indices[:total_frames] = np.arange(total_frames)
    return indices


def _pyav_stream_rate(stream):
    """Return a stream's average frame rate as a float, or 0.0 if unknown."""
    for candidate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        if candidate:
            return float(candidate)
    return 0.0


def _pyav_stream_length(stream, rate):
    """Return a stream's frame count, deriving it from duration when absent.

    Container metadata often omits the frame count (the MJPEG/AVI clips in the
    KPI corpus do), in which case decord computes it internally. Derive the same
    quantity from the stream (or container) duration so the frame range matches.
    """
    if stream.frames:
        return int(stream.frames)
    duration = stream.duration
    time_base = stream.time_base
    if duration and time_base and rate:
        return int(round(float(duration * time_base) * rate))
    container_duration = getattr(stream.container, "duration", None)
    if container_duration and rate:
        return int(round(container_duration / 1e6 * rate))
    return 0


def _pyav_frame_index(frame, stream, rate):
    """Map a decoded frame back to its absolute frame index, or None."""
    pts = frame.pts
    if pts is None or not rate or stream.time_base is None:
        return None
    origin = stream.start_time or 0
    return int(round(float((pts - origin) * stream.time_base) * rate))


def _nearest_index(available, wanted):
    """Return the decoded index closest to ``wanted``."""
    return min(available, key=lambda known: abs(known - wanted))


def _pyav_seek_to_index(container, stream, rate, index):
    """Seek so that decoding resumes at or before ``index``."""
    origin = stream.start_time or 0
    target = origin + int(index / rate / float(stream.time_base))
    container.seek(target, stream=stream, backward=True, any_frame=False)


def _pyav_decode_indices_intra(container, stream, rate, frame_indices):
    """Decode wanted indices on an intra-only stream, seeking to each.

    Every frame of an intra-only codec (MJPEG) is independently decodable, so a
    seek lands exactly where we want and no forward decoding is wasted. Scanning
    instead measured 2.57x slower than decord on the MJPEG/AVI clips.
    """
    decoded = {}
    for index in sorted({int(i) for i in frame_indices}):
        _pyav_seek_to_index(container, stream, rate, index)
        for frame in container.decode(stream):
            got = _pyav_frame_index(frame, stream, rate)
            if got is None or got >= index:
                decoded[index] = frame.to_rgb().to_image()
                break
    return decoded


def _pyav_decode_indices(container, stream, rate, frame_indices):
    """Decode the requested absolute frame indices into a dict.

    Seeks to the first wanted frame rather than scanning from zero: the corpus
    addresses full-length videos by chunk range, so scanning would repeat the
    ffmpeg-CLI fallback's cost. Falls back to a sequential count when the
    container exposes no usable timestamps.
    """
    if (rate and stream.time_base is not None and
            getattr(stream.codec_context.codec, "intra_only", False)):
        try:
            decoded = _pyav_decode_indices_intra(
                container, stream, rate, frame_indices
            )
            if len(decoded) == len({int(i) for i in frame_indices}):
                return decoded
        except Exception:  # pylint: disable=broad-except
            pass  # fall through to the generic path

    wanted = {int(index) for index in frame_indices}
    first = min(wanted)
    last = max(wanted)
    decoded = {}

    seek_ok = False
    if rate and stream.time_base is not None and first > 0:
        origin = stream.start_time or 0
        target = origin + int(first / rate / float(stream.time_base))
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
            seek_ok = True
        except Exception:  # pylint: disable=broad-except
            container.seek(0, stream=stream, backward=True, any_frame=False)

    counter = -1
    for frame in container.decode(stream):
        counter += 1
        index = _pyav_frame_index(frame, stream, rate)
        if index is None:
            # No usable pts: only a scan from the start can be trusted.
            if seek_ok:
                return _pyav_decode_indices_by_scan(
                    container, stream, frame_indices
                )
            index = counter
        if index in wanted:
            decoded[index] = frame.to_rgb().to_image()
        if index >= last:
            break
    return decoded


def _pyav_decode_indices_by_scan(container, stream, frame_indices):
    """Decode wanted indices by counting frames from the start of the stream."""
    wanted = {int(index) for index in frame_indices}
    last = max(wanted)
    decoded = {}
    container.seek(0, stream=stream, backward=True, any_frame=False)
    for counter, frame in enumerate(container.decode(stream)):
        if counter in wanted:
            decoded[counter] = frame.to_rgb().to_image()
        if counter >= last:
            break
    return decoded


def _load_with_pyav(video_path, num_frames, start_time_sec, end_time_sec,
                    start_frame, end_frame):
    """Load frames with PyAV, the codec-compliant in-process decoder.

    PyAV is built against the image's restricted FFmpeg (see docker/Dockerfile),
    so H.264 resolves to h264_cuvid/NVDEC and no bundled software codec ships.
    The frame range, ordering and duplicate padding match the decord backend
    this replaces.
    """
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = _pyav_stream_rate(stream)
        actual_frames = _pyav_stream_length(stream, rate)
        if actual_frames <= 0:
            raise ValueError(f"Could not determine frame count for {video_path}")
        start = 0
        end = actual_frames
        if start_frame is not None and end_frame is not None:
            start = max(0, min(int(start_frame), actual_frames - 1))
            end = min(int(end_frame), actual_frames)
        elif start_time_sec is not None and end_time_sec is not None:
            start = max(
                0,
                min(int(round(float(start_time_sec) * rate)), actual_frames - 1),
            )
            end = min(int(round(float(end_time_sec) * rate)), actual_frames)
        if end <= start:
            raise ValueError(
                f"Invalid video range for {video_path}: [{start}, {end})"
            )
        frame_indices = _linspace_indices(end - start, num_frames) + start
        decoded = _pyav_decode_indices(container, stream, rate, frame_indices)

    if not decoded:
        raise ValueError(f"Decoded no frames from {video_path}")
    available = sorted(decoded)
    frames = []
    for index in frame_indices:
        index = int(index)
        if index not in decoded:
            # Timestamp rounding can miss an exact hit; take the nearest frame
            # that was decoded rather than dropping a sample.
            index = _nearest_index(available, index)
        frames.append(decoded[index].convert("RGB"))
    return frames


def _clip_frame_range(total_frames, fps, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Resolve temporal metadata to a frame range."""
    start = 0
    end = total_frames
    if start_frame is not None and end_frame is not None:
        start = int(start_frame)
        end = int(end_frame)
    elif start_time_sec is not None and end_time_sec is not None and fps > 0:
        start = int(round(float(start_time_sec) * fps))
        end = int(round(float(end_time_sec) * fps))
    start = max(0, min(start, total_frames - 1))
    end = max(start + 1, min(end, total_frames))
    return start, end


def _parse_frame_rate(value):
    """Parse ffprobe frame-rate strings like 30000/1001."""
    if not value or value == "0/0":
        return 0.0
    if "/" in str(value):
        numerator, denominator = value.split("/", 1)
        denominator = float(denominator)
        if denominator == 0.0:
            return 0.0
        return float(numerator) / denominator
    return float(value)


def _probe_video_with_ffmpeg(video_path):
    """Read basic video stream metadata with ffprobe."""
    import subprocess

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream = streams[0]
    fps = _parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )
    duration = float(stream.get("duration") or 0.0)
    total_frames = stream.get("nb_frames")
    total_frames = int(total_frames) if total_frames else 0
    # pylint: disable=chained-comparison  # three independent > 0 tests
    if total_frames <= 0 and fps > 0.0 and duration > 0.0:
        total_frames = int(round(fps * duration))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Could not probe video geometry for {video_path}")
    return total_frames, fps, width, height


def _load_with_ffmpeg(video_path, num_frames, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Load frames with the ffmpeg CLI as a dependency-light fallback."""
    import subprocess

    total_frames, fps, width, height = _probe_video_with_ffmpeg(video_path)
    start, end = _clip_frame_range(
        total_frames,
        fps,
        start_time_sec,
        end_time_sec,
        start_frame,
        end_frame,
    )
    frame_indices = _linspace_indices(end - start, num_frames) + start
    unique_indices = []
    for index in frame_indices:
        index = int(index)
        if index not in unique_indices:
            unique_indices.append(index)
    selector = "+".join(f"eq(n\\,{index})" for index in unique_indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select={selector}",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    frame_bytes = height * width * 3
    decoded_frames = len(result.stdout) // frame_bytes
    if decoded_frames <= 0:
        raise ValueError(f"No frames decoded from {video_path}")
    if len(result.stdout) % frame_bytes != 0:
        logging.warning(
            "Dropping partial decoded frame bytes from %s: got %d bytes",
            video_path,
            len(result.stdout),
        )
    if decoded_frames != len(unique_indices):
        logging.warning(
            "Decoded %d/%d requested frames from %s; padding with the last "
            "decoded frame.",
            decoded_frames,
            len(unique_indices),
            video_path,
        )
    usable_bytes = decoded_frames * frame_bytes
    array = np.frombuffer(result.stdout[:usable_bytes], dtype=np.uint8)
    array = array.reshape(decoded_frames, height, width, 3)
    frames = [Image.fromarray(frame).convert("RGB") for frame in array]
    frames_by_index = {
        frame_index: frames[min(pos, decoded_frames - 1)]
        for pos, frame_index in enumerate(unique_indices)
    }
    return [frames_by_index[int(index)] for index in frame_indices]


def _load_with_opencv(video_path, num_frames, start_time_sec, end_time_sec,
                      start_frame, end_frame):
    """Load frames with OpenCV when PyAV and ffmpeg are unavailable."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0:
            raise ValueError(f"No frames found in {video_path}")

        start, end = _clip_frame_range(
            total_frames,
            fps,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
        frame_indices = _linspace_indices(end - start, num_frames) + start
        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(
                    f"Could not decode frame {frame_index} from {video_path}"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame).convert("RGB"))
        return frames
    finally:
        cap.release()


def load_video_frames(video_path, num_frames, start_time_sec=None,
                      end_time_sec=None, start_frame=None, end_frame=None):
    """Load a fixed number of PIL RGB frames from a video clip."""
    pyav_error = None
    try:
        frames = _load_with_pyav(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
        _log_decode_backend("pyav")
        return frames
    except Exception as exc:
        pyav_error = exc
        logging.debug("pyav decode failed for %s: %s", video_path, exc)

    ffmpeg_error = None
    try:
        frames = _load_with_ffmpeg(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
        _log_decode_backend("ffmpeg-cli")
        return frames
    except Exception as exc:
        ffmpeg_error = exc
        logging.debug("ffmpeg decode failed for %s: %s", video_path, exc)

    try:
        frames = _load_with_opencv(
            video_path,
            num_frames,
            start_time_sec,
            end_time_sec,
            start_frame,
            end_frame,
        )
        _log_decode_backend("opencv")
        return frames
    except ImportError as exc:
        raise ImportError(
            "Video decoding requires PyAV, ffmpeg/ffprobe, or OpenCV in "
            "the TAO environment."
        ) from pyav_error or ffmpeg_error or exc
    except Exception as exc:
        raise RuntimeError(
            "All video decoding backends failed. PyAV is the supported "
            "decoder in the TAO image; the container ffmpeg is codec-disabled "
            "so the CLI fallback cannot pipe raw frames and OpenCV is built "
            "WITH_FFMPEG=OFF. Check that the clip's container and codec are in "
            "the image's allow-list (see docker/Dockerfile)."
        ) from pyav_error or ffmpeg_error or exc
