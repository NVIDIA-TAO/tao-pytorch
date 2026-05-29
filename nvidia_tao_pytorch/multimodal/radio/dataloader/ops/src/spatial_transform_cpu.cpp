/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "spatial_transform_cpu.h"

#include <ATen/Parallel.h>
#include <ATen/ParallelOpenMP.h>

// If 1, use at::parallel_for to parallelize the op along batch and height dimensions. If 0, use
// single threaded processing.
#define USE_MULTITHREADING 1
#ifdef USE_MULTITHREADING
#include <omp.h>
#endif

enum FilterMode { FILTER_MODE_NEAREST = 0, FILTER_MODE_BILINEAR, FILTER_MODE_BICUBIC };

template <typename I, typename O>
static inline void _SpatialTransformKernelNearest(int output_base_offset, int y, const I* in,
                                                      const float* mat, O* out, int num_channels,
                                                      int height, int width, float background,
                                                      int input_x_stride, int input_y_stride,
                                                      int input_c_stride, int output_x_stride,
                                                      int output_c_stride, int output_width) {
    float fy = static_cast<float>(y) + 0.5f;
    float mx = mat[0] * 0.5f + mat[3] * fy + mat[6];
    float my = mat[1] * 0.5f + mat[4] * fy + mat[7];
    float mz = mat[2] * 0.5f + mat[5] * fy + mat[8];

    for (auto x = 0; x < output_width; x++) {
        float ifx = mx / mz;  // Note: this should not have -0.5f. See the CUDA code for docs.
        float ify = my / mz;
        mx += mat[0];
        my += mat[1];
        mz += mat[2];
        int ix = static_cast<int>(floorf(ifx));
        int iy = static_cast<int>(floorf(ify));

        int output_offset = output_base_offset;
        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
            // Sample in.
            int input_offset = iy * input_y_stride + ix * input_x_stride;
            for (int c = 0; c < num_channels; c++) {
                out[output_offset] = static_cast<O>(in[input_offset]);
                output_offset += output_c_stride;
                input_offset += input_c_stride;
            }
        } else {
            // Sample out.
            for (int c = 0; c < num_channels; c++) {
                out[output_offset] = static_cast<O>(background);
                output_offset += output_c_stride;
            }
        }
        output_base_offset += output_x_stride;
    }
}

template <typename I, typename O>
static inline void _SpatialTransformKernelBilinear(int output_base_offset, int y, const I* in,
                                                       const float* mat, O* out, int num_channels,
                                                       int height, int width, float background,
                                                       int input_x_stride, int input_y_stride,
                                                       int input_c_stride, int output_x_stride,
                                                       int output_c_stride, int output_width) {
    float fy = static_cast<float>(y) + 0.5f;
    float mx = mat[0] * 0.5f + mat[3] * fy + mat[6];
    float my = mat[1] * 0.5f + mat[4] * fy + mat[7];
    float mz = mat[2] * 0.5f + mat[5] * fy + mat[8];

    for (auto x = 0; x < output_width; x++) {
        float ifx = mx / mz - 0.5f;  // Note: this should have -0.5f. See the CUDA code for docs.
        float ify = my / mz - 0.5f;
        mx += mat[0];
        my += mat[1];
        mz += mat[2];
        int ix = static_cast<int>(floorf(ifx));
        int iy = static_cast<int>(floorf(ify));

        int output_offset = output_base_offset;
        if (ix + 1 < 0 || ix >= width || iy + 1 < 0 || iy >= height) {
            // All samples out.
            for (int c = 0; c < num_channels; c++) {
                out[output_offset] = static_cast<O>(background);
                output_offset += output_c_stride;
            }
        } else {
            float bx = ifx - static_cast<float>(ix);
            float by = ify - static_cast<float>(iy);
            float ibx = 1.0f - bx;
            float iby = 1.0f - by;

            if (ix >= 0 && ix + 1 < width && iy >= 0 && iy + 1 < height) {
                // All samples in.
                int input_offset = iy * input_y_stride + ix * input_x_stride;
                float w00 = ibx * iby;
                float w10 = bx * iby;
                float w01 = ibx * by;
                float w11 = bx * by;
                for (int c = 0; c < num_channels; c++) {
                    float o;
                    o = static_cast<float>(in[input_offset]) * w00;
                    o += static_cast<float>(in[input_offset + input_x_stride]) * w10;
                    o += static_cast<float>(in[input_offset + input_y_stride]) * w01;
                    o += static_cast<float>(in[input_offset + input_x_stride + input_y_stride]) *
                         w11;
                    out[output_offset] = static_cast<O>(o);
                    output_offset += output_c_stride;
                    input_offset += input_c_stride;
                }
            } else {
                // Some samples in, some out.
                float wx[2], wy[2];
                wx[0] = ibx;
                wx[1] = bx;
                wy[0] = iby;
                wy[1] = by;
                for (int c = 0; c < num_channels; c++) {
                    float o = 0.0f;
                    for (int t = 0; t < 2; t++) {
                        int yy = iy + t;
                        for (int s = 0; s < 2; s++) {
                            int xx = ix + s;
                            float sample = background;
                            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                                int input_offset =
                                    yy * input_y_stride + xx * input_x_stride + c * input_c_stride;
                                sample = static_cast<float>(in[input_offset]);
                            }
                            o += sample * wx[s] * wy[t];
                        }
                    }
                    out[output_offset] = static_cast<O>(o);
                    output_offset += output_c_stride;
                }
            }
        }
        output_base_offset += output_x_stride;
    }
}

template <typename I, typename O>
static inline void _SpatialTransformKernelBicubic(int output_base_offset, int y, const I* in,
                                                      const float* mat, O* out, int num_channels,
                                                      int height, int width, float background,
                                                      int input_x_stride, int input_y_stride,
                                                      int input_c_stride, int output_x_stride,
                                                      int output_c_stride, int output_width) {
    float fy = static_cast<float>(y) + 0.5f;
    float mx = mat[0] * 0.5f + mat[3] * fy + mat[6];
    float my = mat[1] * 0.5f + mat[4] * fy + mat[7];
    float mz = mat[2] * 0.5f + mat[5] * fy + mat[8];

    for (auto x = 0; x < output_width; x++) {
        float ifx = mx / mz - 0.5f;  // Note: this should have -0.5f. See the CUDA code for docs.
        float ify = my / mz - 0.5f;
        mx += mat[0];
        my += mat[1];
        mz += mat[2];
        int ix = static_cast<int>(floorf(ifx));
        int iy = static_cast<int>(floorf(ify));
        float bx = ifx - static_cast<float>(ix);
        float by = ify - static_cast<float>(iy);
        float ibx = 1.0f - bx;
        float iby = 1.0f - by;

        int output_offset = output_base_offset;
        ix--;
        iy--;
        if (ix + 3 < 0 || ix >= width || iy + 3 < 0 || iy >= height) {
            // All samples out.
            for (int c = 0; c < num_channels; c++) {
                out[output_offset] = static_cast<O>(background);
                output_offset += output_c_stride;
            }
        } else {
            // Compute filter kernel weights based on sampling mode.
            // Read 4x4 pixels around the sampling point. Note that while in general bicubic
            // gives higher quality upsampling than bilinear, it introduces a slight blur to
            // the image, and thus identity mapping doesn't produce an identical image.
            // Bicubic weights reference http://vec3.ca/bicubic-filtering-in-fewer-taps
            float wx[4], wy[4];
            wx[0] = 1.0f / 6.0f * ibx * ibx * ibx;
            wx[1] = 1.0f / 6.0f * (4.0f + 3.0f * bx * bx * bx - 6.0f * bx * bx);
            wx[2] = 1.0f / 6.0f * (4.0f + 3.0f * ibx * ibx * ibx - 6.0f * ibx * ibx);
            wx[3] = 1.0f / 6.0f * bx * bx * bx;

            wy[0] = 1.0f / 6.0f * iby * iby * iby;
            wy[1] = 1.0f / 6.0f * (4.0f + 3.0f * by * by * by - 6.0f * by * by);
            wy[2] = 1.0f / 6.0f * (4.0f + 3.0f * iby * iby * iby - 6.0f * iby * iby);
            wy[3] = 1.0f / 6.0f * by * by * by;

            if (ix >= 0 && ix + 3 < width && iy >= 0 && iy + 3 < height) {
                // All samples in.
                for (int c = 0; c < num_channels; c++) {
                    float o = 0.0f;
                    for (int t = 0; t < 4; t++) {
                        int yy = iy + t;
                        for (int s = 0; s < 4; s++) {
                            int xx = ix + s;
                            int input_offset =
                                yy * input_y_stride + xx * input_x_stride + c * input_c_stride;
                            o += static_cast<float>(in[input_offset]) * wx[s] * wy[t];
                        }
                    }
                    out[output_offset] = static_cast<O>(o);
                    output_offset += output_c_stride;
                }
            } else {
                // Some samples in, some out.
                for (int c = 0; c < num_channels; c++) {
                    float o = 0.0f;
                    for (int t = 0; t < 4; t++) {
                        int yy = iy + t;
                        for (int s = 0; s < 4; s++) {
                            int xx = ix + s;
                            float sample = background;
                            if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                                int input_offset =
                                    yy * input_y_stride + xx * input_x_stride + c * input_c_stride;
                                sample = static_cast<float>(in[input_offset]);
                            }
                            o += sample * wx[s] * wy[t];
                        }
                    }
                    out[output_offset] = static_cast<O>(o);
                    output_offset += output_c_stride;
                }
            }
        }
        output_base_offset += output_x_stride;
    }
}

template <typename I, typename O>
void SpatialTransformKernel(const I* input_images, const float* transformation_matrices,
                            O* output_images, int nbatch, int num_channels, int height, int width,
                            int output_height, int output_width, FilterMode filter_mode,
                            float background, bool input_channels_first, bool output_channels_first,
                            bool verbose) {
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

#if USE_MULTITHREADING
    if (verbose)
        std::cout << "SpatialTransform cpu multithreaded: max threads=" << omp_get_max_threads()
                  << std::endl;

    at::parallel_for(0, nbatch * output_height, 1, [&](int64_t begin, int64_t end) {
        if (verbose)
            // Note: not using std::cout since the output would be messed up with multithreading.
            printf("  thread %d/%d: processing range [%ld, %ld[\n", omp_get_thread_num(),
                   omp_get_num_threads(), begin, end);
        for (auto w = begin; w < end; w++) {
            int b = static_cast<int>(w) / output_height;
            int y = static_cast<int>(w) - b * output_height;
            const I* in = input_images + b * height * width * num_channels;
            const float* mat = transformation_matrices + b * 3 * 3;
            O* out = output_images + b * output_height * output_width * num_channels;
#else
    for (auto b = 0; b < nbatch; b++) {
        const I* in = input_images + b * height * width * num_channels;
        const float* mat = transformation_matrices + b * 3 * 3;
        O* out = output_images + b * output_height * output_width * num_channels;
        for (auto y = 0; y < output_height; y++) {
#endif
            int output_base_offset = y * output_y_stride;
            switch (filter_mode) {
                case FILTER_MODE_NEAREST:
                    _SpatialTransformKernelNearest<I, O>(
                        output_base_offset, y, in, mat, out, num_channels, height, width,
                        background, input_x_stride, input_y_stride, input_c_stride, output_x_stride,
                        output_c_stride, output_width);
                    break;
                case FILTER_MODE_BILINEAR:
                    _SpatialTransformKernelBilinear<I, O>(
                        output_base_offset, y, in, mat, out, num_channels, height, width,
                        background, input_x_stride, input_y_stride, input_c_stride, output_x_stride,
                        output_c_stride, output_width);
                    break;
                case FILTER_MODE_BICUBIC:
                    _SpatialTransformKernelBicubic<I, O>(
                        output_base_offset, y, in, mat, out, num_channels, height, width,
                        background, input_x_stride, input_y_stride, input_c_stride, output_x_stride,
                        output_c_stride, output_width);
                    break;
            }
        }
#if USE_MULTITHREADING
    });
#else
    }
#endif
}

torch::Tensor spatialtransform_cpu(torch::Tensor inputs, torch::Tensor stms, int output_width,
                                   int output_height, std::string method, float background,
                                   bool verbose) {
    const auto nbatch = inputs.sizes()[0];
    const auto channels = inputs.sizes()[1];
    const auto height = inputs.sizes()[2];
    const auto width = inputs.sizes()[3];

    // A few sanity checks on input dimensions.
    assert(stms.sizes()[0] == nbatch);

    auto output = torch::empty({nbatch, channels, output_height, output_width}, inputs.options());

    if (verbose) {
        std::cout << "Input Shape: " << inputs.sizes() << std::endl;
        std::cout << "Output Shape: " << output.sizes() << std::endl;
    }

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

    // TODO(jrasanen) Separate input and output types?
    AT_DISPATCH_ALL_TYPES_AND(
        at::ScalarType::Half, inputs.scalar_type(), "spatialtransform_cpp_forward", ([&] {
            SpatialTransformKernel<scalar_t, scalar_t>(
                inputs.data_ptr<scalar_t>(), stms.data_ptr<float>(), output.data_ptr<scalar_t>(), nbatch,
                channels, height, width, output_height, output_width, filter_mode, background,
                input_channels_first, output_channels_first, verbose);
        }));

    return output;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("forward", &spatialtransform_cpp, "SpatialTransform C++", py::arg("inputs"),
//           py::arg("stms"), py::arg("output_width"), py::arg("output_height"), py::arg("method"),
//           py::arg("background"), py::arg("verbose"));
// }
