/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused HWC uint8 -> CHW float32 conversion with 1/255 normalization.
 * Uses SSE4/AVX2 SIMD intrinsics on x86_64 for ~4x throughput over scalar.
 * Ported from EVFM (data/misc/fast_to_tensor.cpp).
 */

#include <torch/extension.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

bool fast_to_tensor_is_available() { return true; }

torch::Tensor fast_to_tensor_cpu(torch::Tensor input) {
    TORCH_CHECK(input.dtype() == torch::kUInt8, "Input must be uint8");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 3, "Expected HWC input");

    const auto H = input.size(0);
    const auto W = input.size(1);
    const size_t HW = H * W;  // total number of pixels

    // Output tensor: CHW float32 normalized
    auto out = torch::empty({3, H, W}, torch::dtype(torch::kFloat32));
    const uint8_t* in_ptr = input.data_ptr<uint8_t>();
    float* out_ptr = out.data_ptr<float>();

    // Scale factor: 1/255 applied to all converted floats
    constexpr float scale = 1.0f / 255.0f;
    const __m128 scale_vec = _mm_set1_ps(scale);

    // Shuffle masks for deinterleaving 4 RGB pixels (12 bytes total)
    // RGBRGBRGBRGB → [R0 R1 R2 R3] / [G0 G1 G2 G3] / [B0 B1 B2 B3]
    const __m128i shuf_r = _mm_setr_epi8(
        0, 3, 6, 9,  // bytes for R
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    );
    const __m128i shuf_g = _mm_setr_epi8(
        1, 4, 7, 10, // bytes for G
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    );
    const __m128i shuf_b = _mm_setr_epi8(
        2, 5, 8, 11, // bytes for B
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    );

    size_t i = 0;
    const size_t step = 4;  // 4 pixels processed per loop

    for (; i + step <= HW; i += step) {
        const uint8_t* p = in_ptr + i * 3;  // pointer to current pixel block (HWC interleaved)

        // Load 16 bytes (we only need first 12 bytes, last 4 are unused but harmless)
        __m128i px = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));

        // Shuffle to extract each channel:
        // r8 = [R0, R1, R2, R3, 0, 0, ...]
        __m128i r8 = _mm_shuffle_epi8(px, shuf_r);
        __m128i g8 = _mm_shuffle_epi8(px, shuf_g);
        __m128i b8 = _mm_shuffle_epi8(px, shuf_b);

        // Zero-extend to 32-bit, then convert to float32
        __m128 r = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(r8));
        __m128 g = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(g8));
        __m128 b = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(b8));

        // Normalize to [0,1] range
        r = _mm_mul_ps(r, scale_vec);
        g = _mm_mul_ps(g, scale_vec);
        b = _mm_mul_ps(b, scale_vec);

        // Store directly into CHW layout:
        // [0:HW]   = R channel
        // [HW:2HW] = G channel
        // [2HW:]   = B channel
        _mm_storeu_ps(out_ptr + i, r);
        _mm_storeu_ps(out_ptr + HW + i, g);
        _mm_storeu_ps(out_ptr + 2 * HW + i, b);
    }

    // Handle remaining pixels (tail <4) with scalar fallback
    for (; i < HW; i++) {
        size_t idx = i * 3;
        out_ptr[i]          = static_cast<float>(in_ptr[idx])     * scale;
        out_ptr[HW + i]     = static_cast<float>(in_ptr[idx + 1]) * scale;
        out_ptr[2 * HW + i] = static_cast<float>(in_ptr[idx + 2]) * scale;
    }

    return out;
}

#else
#warning "fast_to_tensor: x86_64 SIMD not available on this architecture. C++ kernel disabled; Python fallback will be used at runtime."

// On non-x86 the extension still builds successfully, but is_available()
// returns false so the Python wrapper uses the pure-Python path.
// When ARM support is needed, add a proper NEON implementation as a new
// #elif branch.
bool fast_to_tensor_is_available() { return false; }

torch::Tensor fast_to_tensor_cpu(torch::Tensor /*input*/) {
    TORCH_CHECK(false,
        "fast_to_tensor C++ SIMD kernel requires x86_64. "
        "This should not be called; use is_available() to check first.");
    return {};
}
#endif
