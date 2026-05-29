/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>

torch::Tensor spatialtransform_gpu(torch::Tensor inputs, torch::Tensor stms, int output_width,
                                   int output_height, std::string method, float background,
                                   bool verbose);
