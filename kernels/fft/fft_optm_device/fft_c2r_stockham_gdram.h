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
#include "kernels/fft/fft_optm_device/fft_c2r_stockham_nram.h"

extern __nram__ char
    nram_buffer[MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024 - FFT_MAXFACTORS * 4];
__mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];
extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

template <typename DT>
__mlu_func__ void computeMutiStageOnchipC2R(DT *input, DT *output, int *factors,
                                            DT *twiddles,
                                            const DT *twiddles_end,
                                            const DT *dft_matrix, DT *buffer,
                                            int batch, int fft_flag) {
  // const int direction = FFT_BACKWARD;
  int total_num = batch;
  int repeat_num = total_num / taskDim;
  int remain_num = total_num % taskDim;

  char *nram_buf = nram_buffer;

  // Each core needs to process "t_len" blocks, "remain_num" is evenly
  // assigned to the previous "remian_num" cores.
  int t_len = repeat_num + ((remain_num > 0 && taskId < remain_num) ? 1 : 0);
  // Calculate the offset of the block at each core.
  int t_start = taskId - remain_num <= 0 ? taskId * (repeat_num + 1)
                                         : (remain_num * (repeat_num + 1) +
                                            (taskId - remain_num) * repeat_num);
  int t_end = (t_start + t_len);

  MLULOG(
      "taskId: %d, repeat_num: %d, "
      "remain_num: %d, t_len: %d, t_start: %d, t_end: %d\n",
      taskId, repeat_num, remain_num, t_len, t_start, t_end);

  int radix, section_num, butterfly_num, in_stride, stage_count, value_mul,
      out_stride, small_factors_offset;

  int *small_factors;
  int last_stage;

  __nram__ int nram_factors[FFT_MAXFACTORS];

  int sram_offset = 0;
  int *sram_factors = (int *)(sram_buffer + sram_offset);
  sram_offset += FFT_MAXFACTORS * sizeof(int);

  DT *sram_twiddles = (DT *)(sram_buffer + sram_offset);
  const int twiddles_size = twiddles_end - twiddles;
  sram_offset += twiddles_size * sizeof(DT);

  // int sram_dftmtx_size = 0;
  DT *sram_dftmtx = (DT *)(sram_buffer + sram_offset);

  const int _stage_count = factors[0];
  const int nfft = factors[1];

  // first stage
  radix = factors[5 * _stage_count + 0];
  section_num = factors[5 * _stage_count + 1];
  butterfly_num = factors[5 * _stage_count + 2];
  out_stride = factors[5 * _stage_count + 3];
  in_stride = butterfly_num;
  small_factors_offset = factors[5 * _stage_count + 4];

  stage_count = 1;
  last_stage = (_stage_count == 1);

  if (__is_mpu()) {
    __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                   GDRAM2SRAM);
    if (twiddles_size) {
      __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                     GDRAM2SRAM);
    }

    const dft_table_entry *dft_table_gdram =
        (const dft_table_entry *)dft_matrix;
    int dft_matrix_offset = dft_table_gdram[0].offset;

    if (dft_matrix_offset != -1) {
      // copy the table
      __memcpy(sram_dftmtx, dft_matrix, sizeof(DT) * 2 * dft_matrix_offset,
               GDRAM2SRAM);
      const dft_table_entry *dft_table = (const dft_table_entry *)sram_dftmtx;

      for (int entry = 0;; entry++) {
        if (dft_table[entry + 1].radix == -1) {
          int last_radix = dft_table[entry].radix;
          int last_offset = dft_table[entry].offset;

          const int K_num = 64 / sizeof(DT);
          int align_K = K_num * ((last_radix + K_num - 1) / K_num);
          __memcpy_async(sram_dftmtx, dft_matrix,
                         sizeof(DT) * 2 * (last_radix * align_K + last_offset),
                         GDRAM2SRAM);
          break;
        }
      }
    }
    // factors = sram_factors;
  }

  __sync_cluster();
  if (__is_ipu()) {
    __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
             SRAM2NRAM);
    factors = nram_factors;
    twiddles = sram_twiddles;
  }

  if (__is_mpu()) {
    return;
  }

  DT *_twiddles = twiddles;

  int cur_radix, butterfly_num_stage;
  for (int loop_stage = 2; loop_stage < _stage_count; loop_stage++) {
    cur_radix = factors[5 * loop_stage];
    butterfly_num_stage = factors[5 * loop_stage + 2];
    twiddles += (cur_radix - 1) * (butterfly_num_stage / 2 + 1) * 2;
  }

  DT *extra_buffer;
  if (__is_ipu()) {
    extra_buffer = buffer + batch * (nfft << 1);  // for in_place temp buffer

    // c2r:       input -> output (1 stage)
    //            input -> buffer -> output (2 stage)
    //            input -> buffer -> extra_buffer -> output (3 stage)
    //            input -> buffer -> extra_buffer -> buffer -> output (4 stage)
    //            input -> buffer -> extra_buffer -> buffer -> extra_buffer ->
    //            output (5 stage)

    // _stage_count = stage_count;

    // if (_stage_count != 1) FFT_SWAP_PTR(buffer, output);
    if (_stage_count == 1) buffer = output;

    small_factors = factors + small_factors_offset;
    if (repeat_num > 0 || taskId < remain_num) {
      computeLargeButterflyFirststageBatchPingpongC2R<DT>(
          buffer, input, twiddles, _twiddles, sram_dftmtx, section_num,
          butterfly_num, out_stride, (void *)nram_buf, small_factors, nfft,
          t_start, t_end, FFT_BACKWARD, last_stage);
      printf("first stage_count: %d, radix: %d\n", stage_count, radix);
    }

    // __sync();
  }

  if (stage_count == _stage_count) {
    return;
  }
  stage_count++;

  // value_mul = 10;
  for (; stage_count < _stage_count; stage_count++) {
    value_mul = (_stage_count - stage_count + 1) * 5;

    // update parameter

    radix = factors[value_mul];
    section_num = factors[value_mul + 1];
    butterfly_num = factors[value_mul + 2];
    in_stride = butterfly_num;
    out_stride = factors[value_mul + 3];
    small_factors_offset = factors[value_mul + 4];
    twiddles -= (radix - 1) * (butterfly_num / 2 + 1) * 2;
    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      if (repeat_num > 0 || taskId < remain_num) {
        computeLargeButterflyOtherstagesBatchPingpongC2R<DT>(
            extra_buffer, buffer, (DT *)twiddles, _twiddles, sram_dftmtx,
            section_num, butterfly_num, out_stride, (void *)nram_buf,
            small_factors, nfft, t_start, t_end, 0);
        FFT_SWAP_PTR(extra_buffer, buffer);
        printf("other stage_count: %d, radix: %d\n", stage_count, radix);
      }
    }
  }  // for (stage_count)

  // last stage
  {
    // update parameter
    radix = factors[5];
    section_num = factors[6];
    butterfly_num = factors[7];
    in_stride = butterfly_num;
    out_stride = factors[8];
    small_factors_offset = factors[9];
    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      if (repeat_num > 0 || taskId < remain_num) {
        // printf("computeLargeButterflyLaststageBatchPingpongC2R\n");
        computeLargeButterflyLaststageBatchPingpongC2R(
            output, buffer, out_stride, section_num, _twiddles, sram_dftmtx,
            (void *)nram_buf, small_factors, nfft, t_start, t_end);

        printf("last stage_count: %d, radix: %d\n", stage_count, radix);
      }
    }
  }
}

template <typename DT>
__mlu_func__ void computeMutiStageOnchipC2RColumn(DT *input, DT *output,
                                            int *factors, DT *twiddles,
                                            const DT *twiddles_end,
                                            const DT *dft_matrix, DT *buffer,
                                            int batch, int fft_flag, int nb) {
  int total_num = batch;
  int repeat_num = total_num / taskDim;
  int remain_num = total_num % taskDim;

  char *nram_buf = nram_buffer;

  // [BATCHES OF CORES ALLOCATE]
  int t_len = repeat_num + ((remain_num > 0 && taskId < remain_num) ? 1 : 0);
  int t_start = taskId - remain_num <= 0 ? taskId * (repeat_num + 1)
                                         : (remain_num * (repeat_num + 1) +
                                            (taskId - remain_num) * repeat_num);
  int t_end = (t_start + t_len);

  // [CHECK POINT] use of the parameter "out_stride"
  int radix, section_num, butterfly_num, in_stride, stage_count, value_mul,
      small_factors_offset, out_stride;

  int *small_factors;
  int last_stage;

  __nram__ int nram_factors[FFT_MAXFACTORS];
  int sram_offset = 0;

  // [SRAM MEMORY ALLOCTACE] Factors
  int *sram_factors = (int *)sram_buffer;
  sram_offset += FFT_MAXFACTORS * sizeof(int);

  // [SRAM MEMORY ALLOCTACE] Twiddles
  DT *sram_twiddles = (DT *)(sram_buffer + sram_offset);
  const int twiddles_size = twiddles_end - twiddles;
  sram_offset += twiddles_size * sizeof(DT);

  // [SRAM MEMORY ALLOCTACE] DFT_Matrix
  DT *sram_dftmtx = (DT *)(sram_buffer + sram_offset);

  // [PARAMETERS PREPARE] Global
  const int _stage_count = factors[0];
  const int nfft = factors[1];

  // [PARAMETERS PREPARE] Firststage
  radix = factors[5 * _stage_count + 0];
  section_num = factors[5 * _stage_count + 1];
  butterfly_num = factors[5 * _stage_count + 2];
  out_stride = factors[5 * _stage_count + 3];
  in_stride = butterfly_num;
  small_factors_offset = factors[5 * _stage_count + 4];

  stage_count = 1;  // stage_count ↑↑, _stage_count ↓↓
  last_stage = (_stage_count == 1);

  // [MEMORY COPY G2S] Factors & Twiddles & DFT_Matrix
  if (__is_mpu()) {
    __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                   GDRAM2SRAM);
    if (twiddles_size) {
      __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                     GDRAM2SRAM);
    }

    const dft_table_entry *dft_table_gdram =
        (const dft_table_entry *)dft_matrix;
    int dft_matrix_offset = dft_table_gdram[0].offset;

    if (dft_matrix_offset != -1) {
      __memcpy(sram_dftmtx, dft_matrix, sizeof(DT) * 2 * dft_matrix_offset,
               GDRAM2SRAM);
      const dft_table_entry *dft_table = (const dft_table_entry *)sram_dftmtx;

      for (int entry = 0;; entry++) {
        if (dft_table[entry + 1].radix == -1) {
          int last_radix = dft_table[entry].radix;
          int last_offset = dft_table[entry].offset;

          const int K_num = 64 / sizeof(DT);
          int align_K = K_num * ((last_radix + K_num - 1) / K_num);
          __memcpy_async(sram_dftmtx, dft_matrix,
                         sizeof(DT) * 2 * (last_radix * align_K + last_offset),
                         GDRAM2SRAM);
          break;
        }
      }
    }
  }
  __sync_cluster();

  // [MEMORY COPY S2N] Factors
  if (__is_ipu()) {
    __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
             SRAM2NRAM);
    factors = nram_factors;
    twiddles = sram_twiddles;
  }

  if (__is_mpu()) {
    return;
  }

  DT *_twiddles = twiddles;

  // [CHECK POINT] use of the parameter "cur_radix", "butterfly_num_stage"
  int cur_radix, butterfly_num_stage;
  for (int loop_stage = 2; loop_stage < _stage_count; loop_stage++) {
    cur_radix = factors[5 * loop_stage];
    butterfly_num_stage = factors[5 * loop_stage + 2];
    twiddles += (cur_radix - 1) * (butterfly_num_stage / 2 + 1) * 2;
  }

  DT *odd_extra_buffer;
  int max_para_batch;

  if (__is_ipu()) {
    odd_extra_buffer = buffer + batch * (nfft << 1);

    // [CHECK POINT]
    if (_stage_count == 1) FFT_SWAP_PTR(buffer, output);

    if (repeat_num > 0 || taskId < remain_num) {
      small_factors = factors + small_factors_offset;
      max_para_batch = (6144 / radix) > batch ? batch : (6144 / radix);
      for (int t = t_start; t < t_end; t += max_para_batch) {
          int para_batch =
              (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);
          DT *input_batch = input + t * 2;
          DT *output_batch;
          if (last_stage) {
            output_batch = output + t * 2;
          } else {
            output_batch = output + t * nfft * 2;
          }
        computeLargeButterflyFirststageC2RColumn<DT>(
          output_batch, input_batch, twiddles, dft_matrix,
          butterfly_num, nram_buf, small_factors,
          nfft, last_stage, para_batch, nb);
      }
    }
    stage_count--;
    if (stage_count == 0) {
      return;
    }
  }
}
