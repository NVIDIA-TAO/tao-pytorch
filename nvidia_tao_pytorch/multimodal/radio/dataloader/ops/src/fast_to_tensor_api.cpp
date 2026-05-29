/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PYBIND11 entry point for the FastToTensor C++ extension.
 *
 * Exposes:
 *   fast_to_tensor_cpu(input: Tensor) -> Tensor
 *     Fused cast+normalize: HWC uint8 -> CHW float32 in [0,1].
 *     Single pass over the image data, processing 4 pixels per iteration
 *     using SSE4/AVX2 SIMD intrinsics on x86_64.
 *
 *   is_available() -> bool
 *     Returns true on x86_64 (SIMD kernel compiled), false otherwise.
 *     The Python wrapper checks this to decide whether to use the C++
 *     path or fall back to pure-Python.
 */

#include <torch/extension.h>

torch::Tensor fast_to_tensor_cpu(torch::Tensor input);
bool fast_to_tensor_is_available();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_to_tensor_cpu", &fast_to_tensor_cpu,
          "Fused cast+normalize to CHW (CPU, 1-pass, 4-pixel AVX2)");
    m.def("is_available", &fast_to_tensor_is_available,
          "Returns true if the x86_64 SIMD kernel is available");
}
