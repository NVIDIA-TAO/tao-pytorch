# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the release image's FC video dependency contract (PyAV / ONNXScript)."""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "docker" / "requirements-pip.txt"
RELEASE_DOCKERFILE = REPO_ROOT / "release" / "docker" / "Dockerfile"
ENSURE_SCRIPT = REPO_ROOT / "release" / "docker" / "ensure_fc_video_deps.sh"


def _exact_pin(name):
    """Return the exact ``name==version`` pin from docker/requirements-pip.txt."""
    pattern = re.compile(rf"^{re.escape(name)}==(\S+)")
    for line in REQUIREMENTS.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1)
    raise AssertionError(f"{name} is not exact-pinned in {REQUIREMENTS}")


def _print_pins(requirements):
    return subprocess.run(
        ["bash", str(ENSURE_SCRIPT), "--print-pins", str(requirements)],
        capture_output=True, text=True, check=False,
    )


def test_video_dependencies_are_exact_pinned():
    """The release image contract needs exact av/onnxscript pins to reproduce."""
    assert re.fullmatch(r"\d+(\.\d+)+", _exact_pin("av"))
    assert re.fullmatch(r"\d+(\.\d+)+", _exact_pin("onnxscript"))


def test_ensure_script_reads_the_requirements_pins():
    """The script resolves the same pins the base image was built from."""
    result = _print_pins(REQUIREMENTS)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"av={_exact_pin('av')} onnxscript={_exact_pin('onnxscript')}"


def test_ensure_script_fails_closed_without_exact_pins(tmp_path):
    """A requirements file without an exact pin must fail the build, not guess a version."""
    loose = tmp_path / "requirements-pip.txt"
    loose.write_text("av>=17\nonnxscript\n")
    result = _print_pins(loose)
    assert result.returncode != 0
    assert "no exact pin 'av==<version>'" in result.stderr


@pytest.mark.parametrize("copied", ["docker/requirements-pip.txt", "release/docker/ensure_fc_video_deps.sh"])
def test_release_dockerfile_copies_contract_inputs_with_the_wheels(copied):
    """Both inputs ride the existing wheel COPY so the release image gains no filesystem layer."""
    dockerfile = RELEASE_DOCKERFILE.read_text()
    copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY ") and "dist/*.whl" in line]
    assert len(copy_lines) == 1, copy_lines
    assert copied in copy_lines[0]


def test_release_dockerfile_runs_the_contract_after_installing_wheels():
    """The dependency fix-up runs in the wheel RUN layer, after the TAO wheels are installed."""
    dockerfile = RELEASE_DOCKERFILE.read_text()
    install = dockerfile.index("python -m pip install '{}'")
    ensure = dockerfile.index("ensure_fc_video_deps.sh /opt/nvidia/wheels/requirements-pip.txt")
    assert install < ensure
    # The stopgap (PyPI wheel, bundles libx264/libx265) must not come back.
    assert not re.search(r"pip install ['\"]?av==", dockerfile)
