/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "spatial_transform_cpu.h"
#include "spatial_transform_gpu.h"

torch::Tensor spatialtransform(torch::Tensor inputs, torch::Tensor stms, int output_width,
                               int output_height, std::string method, float background,
                               bool verbose)
{
    if (inputs.is_cuda())
    {
        if (! stms.is_cuda())
            throw std::runtime_error("Inputs are CUDA tensors, but stms aren't.");

        return spatialtransform_gpu(std::move(inputs), std::move(stms),
                                    output_width, output_height,
                                    std::move(method), background, verbose);
    }
    else
    {
        if (stms.is_cuda())
            throw std::runtime_error("Inputs are on CPU, but stms are CUDA.");

        return spatialtransform_cpu(std::move(inputs), std::move(stms),
                                    output_width, output_height,
                                    std::move(method), background, verbose);
    }
}
