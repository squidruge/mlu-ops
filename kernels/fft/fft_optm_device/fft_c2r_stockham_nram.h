/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#pragma once
#include "kernels/fft/fft_optm_device/fft_generic_butterfly.h"
#include "kernels/fft/fft_optm_device/fft_vector_butterfly.h"

template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageC2R(
    DT *output, DT *input, int large_in_stride, int section_num,
    const DT *twiddles, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, int dir, int nfft, int last_stage) {
    // printf("call of computeLargeButterflyFirststage\n");

}

template <typename DT>
__mlu_func__ void computeLargeButterflyOtherstagesC2R(
    DT *output, DT *input, const DT *cur_large_twiddles, const DT *_twiddles,
    const DT *dft_matrix, int large_section_num, int large_butterfly_num,
    int large_in_stride, void *nram_buf, const int *small_factors, int nfft,
    int dir, int last_stage) {

}

template <typename DT>
__mlu_func__ void computeLargeButterflyLaststageC2R(
    DT *output, DT *input, const DT *cur_large_twiddles, const DT *_twiddles,
    const DT *dft_matrix, int large_section_num, int large_butterfly_num,
    int large_in_stride, void *nram_buf, const int *small_factors, int nfft,
    int dir) {

}