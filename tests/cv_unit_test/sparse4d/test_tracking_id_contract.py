# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracking consumers accept monotonically allocated, non-zero-based IDs."""

import json

import numpy as np

from spatialai_data_utils.converters.nusc_results_to_nvschema import convert_sparse4d_to_nvschema
from spatialai_data_utils.eval.tracking.hota.datasets.mtmc_challenge_3d_bbox import MTMCChallenge3DBBox
from spatialai_data_utils.eval.tracking.hota.metrics.hota import HOTA


def test_nvschema_preserves_ids_across_scenes(tmp_path):
    """Scene-separated NVSchema output preserves opaque global identities."""
    predictions = {}
    for scene, track_id in (("SceneA", 41), ("SceneB", 73)):
        predictions[f"{scene}+bev-sensor-1__0"] = [{
            "translation": [0, 0, 0],
            "size": [1, 1, 1],
            "rotation": [1, 0, 0, 0],
            "tracking_name": "person",
            "tracking_score": 0.9,
            "tracking_id": str(track_id),
        }]
    source = tmp_path / "predictions.json"
    source.write_text(json.dumps({"results": predictions}), encoding="utf-8")
    output = tmp_path / "nvschema"
    convert_sparse4d_to_nvschema(str(source), str(output), {"person": "Person"})
    for scene, track_id in (("SceneA", 41), ("SceneB", 73)):
        frame = json.loads((output / f"{scene}+bev-sensor-1.json").read_text())
        assert frame["objects"][0]["id"] == str(track_id)


def test_hota_is_invariant_to_constant_tracking_id_offset():
    """Real HOTA preprocessing renumbers identities per sequence before scoring."""
    dataset = MTMCChallenge3DBBox.__new__(MTMCChallenge3DBBox)
    dataset.benchmark = "MOT15"
    dataset.do_preproc = False
    dataset.class_name_to_class_id = {
        "class": 1, "box": 2, "static_person": 3, "distractor": 4, "reflection": 5,
    }
    def evaluate(offset):
        raw = {
            "seq": "synthetic", "num_timesteps": 2,
            "gt_ids": [np.array([5, 8]) for _ in range(2)],
            "tracker_ids": [np.array([0, 1]) + offset for _ in range(2)],
            "gt_classes": [np.ones(2, dtype=int) for _ in range(2)],
            "tracker_classes": [np.ones(2, dtype=int) for _ in range(2)],
            "gt_extras": [{"zero_marked": np.ones(2)} for _ in range(2)],
            "gt_dets": [np.zeros((2, 7)) for _ in range(2)],
            "tracker_dets": [np.zeros((2, 7)) for _ in range(2)],
            "tracker_confidences": [np.ones(2) for _ in range(2)],
            "similarity_scores": [np.eye(2) for _ in range(2)],
        }
        processed = dataset.get_preprocessed_seq_data(raw, "class")
        assert processed["num_tracker_ids"] == 2
        for ids in processed["tracker_ids"]:
            np.testing.assert_array_equal(ids, [0, 1])
        return HOTA().eval_sequence(processed)
    original, offset = evaluate(0), evaluate(1000)
    for key in ("HOTA", "DetA", "AssA", "LocA"):
        np.testing.assert_array_equal(original[key], offset[key])
