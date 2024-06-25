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

#include "kernels/fft/fft_optm_device/fft_butterfly_ops.h"

extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

template <typename DT>
__mlu_func__ void reverse(DT *dst, const DT *src, int length) {
    DT *_dst = dst;
    DT *_src =(DT *)src + length;
    for(int i = 0; i < length; i++) {
        _dst[0] = _src[0];
        _dst++;
        _src--;
    }
}

template <typename DT>
__mlu_func__ void concatHarizontal(
  DT *dst, DT *src1, DT *src2, int target_width, int target_height,
  int width1, int width2, int height) {
    __bang_pad(dst, src1, 1, height, width1,
               0, target_height - height, 0, target_width - width1);
    dst += width1;
    __memcpy(dst, src2, width2, NRAM2NRAM, target_width, width2, height - 1);
}


// include completing, transposing and padding
template <typename DT>
__mlu_func__ void completeInputMatrix(
  DT *output_r, DT *output_i, DT *input_r, DT *input_i,
  DT *trans_r, DT *trans_i, DT *mirrored_r, DT *mirrored_i,
  int length, int length_logical, int radix, int align_K, int align_N) {
    int butterfly_num = length_logical / radix;
    int butterfly_num_half = (butterfly_num >> 1) + 1;
    // int length_completed = butterfly_num_half * radix;
    int turn_part_A = length % butterfly_num; // 转折部分里，前一半不需要翻转的部分。
    int turn_part_B = butterfly_num_half - turn_part_A; // 转折部分里，后一半需要翻转的部分。
    int unturned_times = length / butterfly_num; // 无需翻转的部分搬运次数
    int turning_time = turn_part_A > 0 ? 1 : 0; // 中间部分
    int turned_times = radix - unturned_times - turning_time; //需要翻转的部分搬运次数

    DT *reversePoint = NULL;
    DT *in_r = input_r;
    DT *in_i = input_i;
    DT *out_r = output_r;
    DT *out_i = output_i;

    DT *trans_unturned_r = trans_r;
    DT *trans_turned_r =
      trans_unturned_r + (unturned_times + turning_time) * butterfly_num_half;

    DT *trans_unturned_i = trans_i;
    DT *trans_turned_i =
      trans_unturned_i + (unturned_times + turning_time) * butterfly_num_half;

    // overlap
    DT *compute_buffer = trans_r;

    for (int i = 0; i < unturned_times; i++) {
      __memcpy(out_r, in_r, butterfly_num_half * sizeof(DT), NRAM2NRAM);
      __memcpy(out_i, in_i, butterfly_num_half * sizeof(DT), NRAM2NRAM);
      in_r += butterfly_num;
      in_i += butterfly_num;
      out_r += butterfly_num_half;
      out_i += butterfly_num_half;
    }
    reversePoint = out_i;
    if (turning_time) {
      __memcpy(out_r, in_r, turn_part_A * sizeof(DT), NRAM2NRAM);
      __memcpy(out_i, in_i, turn_part_A * sizeof(DT), NRAM2NRAM);
      in_r -= turn_part_B;
      in_i -= turn_part_B;
      out_r += turn_part_A;
      out_i += turn_part_A;
      reversePoint = out_i;
      __memcpy(compute_buffer, in_r,
               turn_part_B * sizeof(DT), NRAM2NRAM);
      __memcpy(compute_buffer + turn_part_B, in_i,
               turn_part_B * sizeof(DT), NRAM2NRAM);
      reverse(out_r, compute_buffer, turn_part_B);
      reverse(out_i, compute_buffer, turn_part_B);
      out_r += turn_part_B;
      out_i += turn_part_B;
    } else {
      in_r -= (butterfly_num - butterfly_num_half);
      in_i -= (butterfly_num - butterfly_num_half);
    }
    in_r -= butterfly_num;
    in_i -= butterfly_num;
    for (int i = 0; i < turned_times; i++) {
      __memcpy(out_r, in_r, butterfly_num_half * sizeof(DT), NRAM2NRAM);
      __memcpy(out_i, in_i, butterfly_num_half * sizeof(DT), NRAM2NRAM);
      in_r -= butterfly_num;
      in_i -= butterfly_num;
      out_r += butterfly_num_half;
      out_i += butterfly_num_half;
    }

    // neg
    __bang_mul_scalar(reversePoint, reversePoint, -1, output_i - reversePoint);

    // vector reverse
    __bang_transpose(trans_unturned_r, output_r,
      unturned_times + turning_time, butterfly_num_half);
    __bang_transpose(trans_unturned_i, output_i,
      unturned_times + turning_time, butterfly_num_half);
    __bang_transpose(trans_turned_r, 
      output_r + (unturned_times + turning_time) * butterfly_num_half,
      turned_times, butterfly_num_half);
    __bang_transpose(trans_turned_i, 
      output_i + (unturned_times + turning_time) * butterfly_num_half,
      turned_times, butterfly_num_half);
    
    __bang_mirror(mirrored_r, trans_turned_r, butterfly_num_half, turned_times);
    __bang_mirror(mirrored_i, trans_turned_i, butterfly_num_half, turned_times);

    concatHarizontal(output_r, trans_unturned_r, mirrored_r,
                     align_K, align_N, unturned_times + turning_time,
                     turned_times, butterfly_num_half);
    concatHarizontal(output_i, trans_unturned_i, mirrored_i,
                     align_K, align_N, unturned_times + turning_time,
                     turned_times, butterfly_num_half);
}


template <typename DT>
__mlu_func__ void computeGenericButterflyFirststageMatC2R(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, int section_num, int butterfly_num,
    int in_stride, int radix) {
  // outplace(nram)
  // origin: M = radix, K = radix, N =butterfly_num
  // pad_up:
  const int para_num = butterfly_num;
  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  // prepare for completeInputMatrix
  int length_logical = radix * butterfly_num;
  int length = length_logical / 2 + 1;

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;
  DT *wram_sratch = (DT *)wram_buffer;

  FFT_CPX_T<DT> in_wram = {
      &wram_sratch[wram_scratch_offset],
      &wram_sratch[wram_scratch_offset + align_N * align_K]};

  wram_scratch_offset += (align_N * align_K * 2);

  // overlap
  FFT_CPX_T<DT> in_align = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);

  // overlap
  FFT_CPX_T<DT> in_align2 = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + align_M * align_N]};

  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_mirrored =  {&nram_scratch[nram_scratch_offset], 
                                &nram_scratch[nram_scratch_offset + align_M * align_N]};
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_trans = {nram_out_r, nram_out_i};

  FFT_CPX_T<DT> dftmtx;

  if (align_K != radix) {
    dftmtx = {&nram_scratch[nram_scratch_offset],
              &nram_scratch[nram_scratch_offset + align_M * align_K]};
    nram_scratch_offset += (align_M * align_K * 2);
    __bang_pad(dftmtx.r, nram_dftmtx, 1, radix, radix, 0, 0, 0,
               align_K - radix);
    __bang_pad(dftmtx.i, &nram_dftmtx[radix * radix], 1, radix, radix, 0, 0, 0,
               align_K - radix);

  } else {
    dftmtx = {nram_dftmtx, &nram_dftmtx[radix * radix]};
  }

  DT *RR = &nram_scratch[nram_scratch_offset];
  DT *RI = &nram_scratch[nram_scratch_offset + align_K * align_N];
  DT *IR = &nram_scratch[nram_scratch_offset + align_K * align_N * 2];
  DT *II = &nram_scratch[nram_scratch_offset + align_K * align_N * 3];

  completeInputMatrix(in_align.r, in_align.i, nram_in_r, nram_in_i,
                      in_trans.r, in_trans.i, in_mirrored.r, in_mirrored.i,
                      length, length_logical, radix, align_K, align_N);

  __bang_reshape_filter(in_align2.r, in_align.r, align_N, 1, 1, align_K);
  __bang_reshape_filter(in_align2.i, in_align.i, align_N, 1, 1, align_K);

  __memcpy(in_wram.r, in_align2.r, align_N * align_K * sizeof(DT), NRAM2WRAM);
  __memcpy(in_wram.i, in_align2.i, align_N * align_K * sizeof(DT), NRAM2WRAM);

  __bang_matmul((float *)RR, (float *)dftmtx.r, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_matmul((float *)II, (float *)dftmtx.i, (float *)in_wram.i, align_M,
                align_K, align_N);

  __bang_sub(out.r, RR, II, align_M * align_N);
  __bang_transpose(out_trans.r, out.r, align_M, align_N);
  __memcpy(nram_out_r, out_trans.r, radix * butterfly_num * sizeof(DT),
           NRAM2NRAM);

  __bang_matmul((float *)RI, (float *)dftmtx.r, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_matmul((float *)IR, (float *)dftmtx.i, (float *)in_wram.r, align_M,
                align_K, align_N);

  __bang_add(out.i, RI, IR, align_M * align_N);
  __bang_transpose(out_trans.i, out.i, align_M, align_N);
  __memcpy(nram_out_i, out_trans.i, radix * butterfly_num * sizeof(DT),
           NRAM2NRAM);
}

template <typename DT>
__mlu_func__ void  computeGenericButterflyOtherstagesMatC2R(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, int section_num,
    int butterfly_num, int para_large_butterfly, int in_stride,
    int radix) {

  const int para_num = butterfly_num * section_num * para_large_butterfly;

  const int align_M = radix;  // no align
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int align_N = 64 * ((para_num + 64 - 1) / 64);

  FFT_CPX_T<DT> scratch_tw = {nram_tw, &nram_tw[butterfly_num * (radix - 1)]};

  int nram_scratch_offset = 0;
  int wram_scratch_offset = 0;

  // overlap
  FFT_CPX_T<DT> in_align2 = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  FFT_CPX_T<DT> out = {&nram_scratch[nram_scratch_offset],
                       &nram_scratch[nram_scratch_offset + align_M * align_N]};
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> Fin = in_align2;
  TRANSPOSE_XYZ2YXZ_PAIR(Fin.r, Fin.i, nram_in_r, nram_in_i,
                        para_large_butterfly, radix,
                        butterfly_num * section_num, DT)

  DT *wram_sratch = (DT *)wram_buffer;
  FFT_CPX_T<DT> in_wram = {
      &wram_sratch[wram_scratch_offset],
      &wram_sratch[wram_scratch_offset + align_N * align_K]};
  wram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_align = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + align_N * align_K]};
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> in_trans = {
      &nram_scratch[nram_scratch_offset],
      &nram_scratch[nram_scratch_offset + para_num * radix]};
  nram_scratch_offset += (para_num * radix * 2);

  FFT_CPX_T<DT> in_mirrored =  {&nram_scratch[nram_scratch_offset], 
                                &nram_scratch[nram_scratch_offset + align_M * align_N]};
  nram_scratch_offset += (align_N * align_K * 2);

  FFT_CPX_T<DT> dftmtx;
  if (align_K != radix) {
    dftmtx = {&nram_scratch[nram_scratch_offset],
              &nram_scratch[nram_scratch_offset + align_M * align_K]};
    nram_scratch_offset += (align_M * align_K * 2);
    __bang_pad(dftmtx.r, nram_dftmtx, 1, radix, radix, 0, 0, 0,
               align_K - radix);
    __bang_pad(dftmtx.i, &nram_dftmtx[radix * radix], 1, radix, radix, 0, 0, 0,
               align_K - radix);
  } else {
    dftmtx = {nram_dftmtx, &nram_dftmtx[radix * radix]};
  }

  DT *RR = &nram_scratch[nram_scratch_offset];
  DT *RI = &nram_scratch[nram_scratch_offset + align_K * align_N];
  DT *IR = &nram_scratch[nram_scratch_offset + align_K * align_N * 2];
  DT *II = &nram_scratch[nram_scratch_offset + align_K * align_N * 3];

  nram_scratch_offset += (align_K * 4 * align_N);

  int nram_in_offset = para_num;
  for (int i = 1; i < radix; i++, nram_in_offset += para_num) {
    __bang_cycle_mul(&RR[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.r[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&RI[(i - 1) * para_num], &Fin.r[nram_in_offset],
                     &scratch_tw.i[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&IR[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.r[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
    __bang_cycle_mul(&II[(i - 1) * para_num], &Fin.i[nram_in_offset],
                     &scratch_tw.i[(i - 1) * butterfly_num], para_num,
                     butterfly_num);
  }

  __bang_sub(&Fin.r[para_num], RR, II, para_num * (radix - 1));
  __bang_add(&Fin.i[para_num], RI, IR, para_num * (radix - 1));

  __bang_transpose(in_trans.i, Fin.i, radix, para_num);
  __bang_transpose(in_trans.r, Fin.r, radix, para_num);

  int logical_length = butterfly_num * radix;

  // complete the input matrix, in_mirrored is not found
  completeInputMatrix(in_align.r, in_align.i, nram_in_r, nram_in_i,
                      in_trans.r, in_trans.i, in_mirrored.r, in_mirrored.i,
                      logical_length / 2 + 1, logical_length, 
                      radix, align_K, align_N);

  __bang_reshape_filter(in_align2.r, in_align.r, align_N, 1, 1, align_K);
  __bang_reshape_filter(in_align2.i, in_align.i, align_N, 1, 1, align_K);

  __memcpy(in_wram.r, in_align2.r, align_N * align_K * sizeof(DT), NRAM2WRAM);
  __memcpy(in_wram.i, in_align2.i, align_N * align_K * sizeof(DT), NRAM2WRAM);

  __bang_matmul((float *)RR, (float *)dftmtx.r, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_matmul((float *)II, (float *)dftmtx.i, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_sub(out.r, RR, II, align_M * align_N);

  __bang_matmul((float *)RI, (float *)dftmtx.r, (float *)in_wram.i, align_M,
                align_K, align_N);
  __bang_matmul((float *)IR, (float *)dftmtx.i, (float *)in_wram.r, align_M,
                align_K, align_N);
  __bang_add(out.i, RI, IR, align_M * align_N);

  {
    int src_stride0 = butterfly_num * sizeof(DT);
    int src_segnum1 = para_large_butterfly * section_num - 1;
    int src_stride1 = align_N * sizeof(DT);
    int src_segnum2 = radix - 1;

    int dst_stride0 = radix * butterfly_num * sizeof(DT);
    int dst_segnum1 = para_large_butterfly * section_num - 1;
    int dst_stride1 = butterfly_num * sizeof(DT);
    int dst_segnum2 = radix - 1;

    __memcpy(nram_out_r, out.r, sizeof(DT) * butterfly_num, NRAM2NRAM,
             dst_stride0, dst_segnum1, dst_stride1, dst_segnum2, src_stride0,
             src_segnum1, src_stride1, src_segnum2);
    __memcpy(nram_out_i, out.i, sizeof(DT) * butterfly_num, NRAM2NRAM,
             dst_stride0, dst_segnum1, dst_stride1, dst_segnum2, src_stride0,
             src_segnum1, src_stride1, src_segnum2);
  }
}

template <typename DT>
__mlu_func__ void  computeGenericButterflyLaststageMatC2R(
    DT *nram_out_r, DT *nram_out_i, DT *nram_in_r, DT *nram_in_i,
    DT *nram_scratch, DT *nram_dftmtx, DT *nram_tw, int section_num,
    int butterfly_num, int para_large_butterfly, int in_stride,
    int radix) {
  computeGenericButterflyOtherstagesMatC2R(
      nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch, nram_dftmtx,
      nram_tw, section_num, butterfly_num, para_large_butterfly, in_stride,
      radix);
}