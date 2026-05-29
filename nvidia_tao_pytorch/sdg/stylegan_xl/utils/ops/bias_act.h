/*
 * Original source taken from https://github.com/autonomousvision/stylegan-xl
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//------------------------------------------------------------------------
// CUDA kernel parameters.

struct bias_act_kernel_params
{
    const void* x;      // [sizeX]
    const void* b;      // [sizeB] or NULL
    const void* xref;   // [sizeX] or NULL
    const void* yref;   // [sizeX] or NULL
    const void* dy;     // [sizeX] or NULL
    void*       y;      // [sizeX]

    int64_t         grad;
    int64_t         act;
    double       alpha;
    double       gain;
    double       clamp;

    int64_t         sizeX;
    int64_t         sizeB;
    int64_t         stepB;
    int64_t         loopX;
};

//------------------------------------------------------------------------
// CUDA kernel selection.

template <class T> void* choose_bias_act_kernel(const bias_act_kernel_params& p);

//------------------------------------------------------------------------
