/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "spatial_transform_gpu.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
enum FilterMode { FILTER_MODE_NEAREST = 0, FILTER_MODE_BILINEAR, FILTER_MODE_BICUBIC };

__inline__ __device__ int reflect_offset(int v, int bound)
{
    v = abs(v);

    // Count the number of folds
    int m = v / bound;

    int rem = v - (m * bound);

    // For an odd number of folds, we need to work the frame from right to left
    if ((m % 2) == 1) {
        rem = bound - 1 - rem;
    }

    // NOTE: This isn't entirely correct.
    // When v = 2*bound - 1, then rem=0
    // But also, v = 2*bound, then rem=0

    return rem;

    // // TODO(mranzinger): There must exist a version of this that doesn't
    // // use a loop to compute the pixel coordinate. Essentially, we want to "fold"
    // // down the excess values until the remainder falls within (-bound, bound).
    // // These folds are periodic.
    // int v2;
    // if (v >= bound) {
    //     auto rem = v - bound;
    //     v2 = bound - 1 - rem;
    // } else if (v < 0) {
    //     v2 = -v;
    // } else {
    //     return v;
    // }
    // return reflect_offset(v2, bound);
}

template <typename I, typename O>
static __inline__ __device__ void _SpatialTransformKernel(
    int x, int y, const I* in, const float* mat, O* out, int num_channels, int height, int width,
    int output_height, int output_width, FilterMode filter_mode, float background,
    bool input_channels_first, bool output_channels_first) {
    int input_x_stride, input_y_stride, input_c_stride;
    if (input_channels_first) {
        input_x_stride = 1;
        input_y_stride = width;
        input_c_stride = width * height;
    } else {
        input_x_stride = num_channels;
        input_y_stride = width * num_channels;
        input_c_stride = 1;
    }
    int output_x_stride, output_y_stride, output_c_stride;
    if (output_channels_first) {
        output_x_stride = 1;
        output_y_stride = output_width;
        output_c_stride = output_width * output_height;
    } else {
        output_x_stride = num_channels;
        output_y_stride = output_width * num_channels;
        output_c_stride = 1;
    }

    // Add 0.5 to output pixel coordinates to move the sampling point (=pixel center) to
    // half integers. This matches OpenGL rasterization rules (OpenGL 4.5 spec, Chapter 14).
    float fx = static_cast<float>(x) + 0.5f;
    float fy = static_cast<float>(y) + 0.5f;
    // Transform output pixel coordinate into input image space. Note that this allows for
    // perspective warps by transforming from 2D output image plane into 3D, followed by
    // projection back to 2D.
    // See: https://github.com/ssloy/tinyrenderer/wiki/Lesson-4:-Perspective-projection#..
    // ..wait-a-minute-may-i-touch-this-magical-bottom-row-of-the-3x3-matrix
    float ifx = mat[0] * fx + mat[3] * fy + mat[6];
    float ify = mat[1] * fx + mat[4] * fy + mat[7];
    float ifz = mat[2] * fx + mat[5] * fy + mat[8];
    // Project to 2D input image plane.
    ifx /= ifz;
    ify /= ifz;
    // Subtract pixel center. This is needed to avoid half a pixel shift so that identity
    // transform produces an identical image with bilinear sampling mode. This is not done
    // for nearest filter mode as we're flooring the coordinate. This matches OpenGL
    // texture filtering rules (OpenGL 4.5 spec, section 8.14 Texture Minification).
    if (filter_mode != FILTER_MODE_NEAREST) {
        ifx -= 0.5f;
        ify -= 0.5f;
    }

    // Compute integer input image coordinates by flooring.
    int ix = static_cast<int>(floor(ifx));
    int iy = static_cast<int>(floor(ify));
    // Compute distance from the exact sampling point to the integer coordinates. This is
    // used for weighting the adjacent samples to produce a filtered result.
    float bx = ifx - static_cast<float>(ix);
    float by = ify - static_cast<float>(iy);
    float ibx = 1.0f - bx;
    float iby = 1.0f - by;

    // Compute filter kernel weights based on sampling mode.
    float wx[4], wy[4];
    int kernel_size;
    switch (filter_mode) {
        case FILTER_MODE_NEAREST:
            // Read 1 pixel at the sampling point.
            wx[0] = 1.0f;
            wy[0] = 1.0f;
            kernel_size = 1;
            break;
        case FILTER_MODE_BILINEAR:
            // Read 2x2 pixels around the sampling point.
            wx[0] = ibx;
            wx[1] = bx;
            wy[0] = iby;
            wy[1] = by;
            kernel_size = 2;
            break;
        case FILTER_MODE_BICUBIC:
        default:
            // Read 4x4 pixels around the sampling point. Note that while in general bicubic
            // gives higher quality upsampling than bilinear, it introduces a slight blur to
            // the image, and thus identity mapping doesn't produce an identical image.
            // Bicubic weights reference http://vec3.ca/bicubic-filtering-in-fewer-taps
            wx[0] = 1.0f / 6.0f * ibx * ibx * ibx;
            wx[1] = 1.0f / 6.0f * (4.0f + 3.0f * bx * bx * bx - 6.0f * bx * bx);
            wx[2] = 1.0f / 6.0f * (4.0f + 3.0f * ibx * ibx * ibx - 6.0f * ibx * ibx);
            wx[3] = 1.0f / 6.0f * bx * bx * bx;

            wy[0] = 1.0f / 6.0f * iby * iby * iby;
            wy[1] = 1.0f / 6.0f * (4.0f + 3.0f * by * by * by - 6.0f * by * by);
            wy[2] = 1.0f / 6.0f * (4.0f + 3.0f * iby * iby * iby - 6.0f * iby * iby);
            wy[3] = 1.0f / 6.0f * by * by * by;

            ix--;
            iy--;
            kernel_size = 4;
            break;
    }

    // Sample and filter.
    for (int c = 0; c < num_channels; c++) {
        float o = 0.0f;
        for (int t = 0; t < kernel_size; t++) {
            int yy = iy + t;
            for (int s = 0; s < kernel_size; s++) {
                int xx = ix + s;
                // Pixels outside the input image are set to background value. Note that this
                // code correctly handles cases where the filter kernel is partially in and
                // partially out of the image.
                float sample = background;
                auto ryy = yy;
                auto rxx = xx;
                if (background == -1234.0f) {
                    ryy = reflect_offset(yy, height);
                    rxx = reflect_offset(xx, width);
                }
                if (rxx >= 0 && rxx < width && ryy >= 0 && ryy < height) {
                    int in_offset = ryy * input_y_stride + rxx * input_x_stride + c * input_c_stride;
                    sample = static_cast<float>(in[in_offset]);
                }
                o += sample * wx[s] * wy[t];
            }
        }
        int out_offset = y * output_y_stride + x * output_x_stride + c * output_c_stride;
        out[out_offset] = static_cast<O>(o);
    }
}

template <typename I, typename O>
__global__ void SpatialTransformKernel(const I* __restrict__ input_images,
                                       const float* __restrict__ transformation_matrices,
                                       O* __restrict__ output_images, int nbatch, int num_channels,
                                       int height, int width, int output_height, int output_width,
                                       FilterMode filter_mode, float background,
                                       bool input_channels_first, bool output_channels_first) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= output_height) {
        return;
    }
    int b = x / output_width;
    if (b >= nbatch) {
        return;
    }
    x %= output_width;

    if (x >= output_width) {
        return;
    }

    const I* in = input_images + b * height * width * num_channels;
    const float* mat = transformation_matrices + b * 3 * 3;
    O* out = output_images + b * output_height * output_width * num_channels;

    _SpatialTransformKernel<I, O>(x, y, in, mat, out, num_channels, height, width, output_height,
                                  output_width, filter_mode, background, input_channels_first,
                                  output_channels_first);
}
}  // End of namespace

at::Tensor spatialtransform_gpu(at::Tensor inputs, at::Tensor stms, int output_width,
                                int output_height, std::string method, float background,
                                bool verbose) {
    const auto nbatch = inputs.sizes()[0];
    const auto channels = inputs.sizes()[1];
    const auto height = inputs.sizes()[2];
    const auto width = inputs.sizes()[3];

    // A few sanity checks on input dimensions.
    assert(stms.sizes()[0] == nbatch);

    auto output = at::empty({nbatch, channels, output_height, output_width}, inputs.options());

    if (verbose) {
        std::cout << "Input Shape: " << inputs.sizes() << std::endl;
        std::cout << "Output Shape: " << output.sizes() << std::endl;
    }

    dim3 dimBlock(8, 8);
    dim3 dimGrid(((output_width * nbatch) + dimBlock.x - 1) / dimBlock.x,
                 (output_height + dimBlock.y - 1) / dimBlock.y);

    FilterMode filter_mode;
    if (!method.compare("nearest")) {
        filter_mode = FILTER_MODE_NEAREST;
    } else if (!method.compare("bilinear")) {
        filter_mode = FILTER_MODE_BILINEAR;
    } else if (!method.compare("bicubic")) {
        filter_mode = FILTER_MODE_BICUBIC;
    } else {
        assert(false);  // Unknown method.
    }

    // TODO(jrasanen) These could be hardcoded.
    const bool input_channels_first = true;
    const bool output_channels_first = true;

    cudaStream_t currStream = at::cuda::getCurrentCUDAStream();

    // TODO(jrasanen) Separate input and output types?
    // TODO(williamz): Use packed accessors
    // https://pytorch.org/tutorials/advanced/cpp_extension.html#using-accessors
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, inputs.scalar_type(), "spatialtransform_cuda_forward", ([&] {
            SpatialTransformKernel<scalar_t, scalar_t><<<dimGrid, dimBlock, 0, currStream>>>(
                inputs.data_ptr<scalar_t>(), stms.data_ptr<float>(), output.data_ptr<scalar_t>(), nbatch,
                channels, height, width, output_height, output_width, filter_mode, background,
                input_channels_first, output_channels_first);
        }));

    return output;
}
