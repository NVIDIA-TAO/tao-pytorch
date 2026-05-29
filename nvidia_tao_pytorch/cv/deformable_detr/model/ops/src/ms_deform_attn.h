/*
 * Copyright (c) 2020 SenseTime.
 *
 * Original source taken from https://github.com/huggingface/transformers/blob/main/src/transformers/kernels/deformable_detr/ms_deform_attn.h
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "torch/script.h"
#include "ms_deform_attn_cpu.h"

#ifdef WITH_CUDA
#include "ms_deform_attn_cuda.h"
#endif

at::Tensor
ms_deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight)
{
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

std::vector<at::Tensor>
ms_deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int64_t im2col_step)
{
    if (value.is_cuda())
    {
#ifdef WITH_CUDA
        return ms_deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR("Not implemented on the CPU");
}

static auto registry = torch::RegisterOperators("nvidia::MultiscaleDeformableAttnPlugin_TRT", &ms_deform_attn_forward);
static auto registry_backward = torch::RegisterOperators("nvidia::DMHA_backward", &ms_deform_attn_backward);