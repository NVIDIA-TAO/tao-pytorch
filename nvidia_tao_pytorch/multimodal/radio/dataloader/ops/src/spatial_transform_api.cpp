/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Perspective warp using 3x3 homography matrices.
 * Dispatches to CPU (OpenMP) or CUDA based on input tensor device.
 * Ported from EVFM (libs/pytorch_image_ops/spatial_transform/).
 */

#include "spatial_transform.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spatial_transform", &spatialtransform, "Perspective warp (CPU/CUDA)",
          py::arg("inputs"), py::arg("stms"), py::arg("output_width"),
          py::arg("output_height"), py::arg("method"), py::arg("background"),
          py::arg("verbose"));
}
