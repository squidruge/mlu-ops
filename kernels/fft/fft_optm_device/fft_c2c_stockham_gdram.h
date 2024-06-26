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
#include "kernels/fft/fft_optm_device/fft_c2c_stockham_nram.h"

extern __nram__ char
    nram_buffer[MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024 - FFT_MAXFACTORS * 4];
__mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];
extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

template <typename DT>
__mlu_func__ void computeMutiStageOnchip(DT *input, DT *output, int *factors,
                                         DT *twiddles, const DT *twiddles_end,
                                         const DT *dft_matrix, DT *buffer,
                                         int batch, int fft_flag,
                                         int direction) {
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
      small_factors_offset;

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
  radix = factors[5 + 0];
  section_num = factors[5 + 1];
  in_stride = factors[5 + 3];
  small_factors_offset = factors[5 + 4];

  // small_factors = factors + small_factors_offset;

  stage_count = _stage_count;
  last_stage = (stage_count == 1);

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
  DT *odd_extra_buffer;
  if (__is_ipu()) {
    // FFT_CPX_T<DT> *odd_extra_buffer = (FFT_CPX_T<DT> *)buffer + batch * nfft;
    // // for in_place temp buffer
    odd_extra_buffer =
        buffer + batch * (nfft << 1);  // for in_place temp buffer
    // out_place: input -> output (1 stage)
    //            input -> buffer -> output (2 stage)
    //            input -> buffer -> odd_extra_buffer -> output (3 stage)
    //            input -> buffer -> output -> buffer -> output (4 stage)
    //            input -> buffer -> output -> buffer -> odd_extra_buffer ->
    //            output (5 stage)

    // _stage_count = stage_count;

    if (_stage_count != 1) FFT_SWAP_PTR(buffer, output);
    small_factors = factors + small_factors_offset;
    if (repeat_num > 0 || taskId < remain_num) {
      if (0) {
        for (int t = t_start; t < t_end; t++) {
          // MLULOG("taskId: %d, batchId: %d\n", taskId, t);
          DT *input_batch = input + t * (nfft << 1);
          DT *output_batch = output + t * (nfft << 1);

          // DT *buffer_batch = buffer + t * (nfft * 2);
          // DT *odd_extra_buffer_batch = odd_extra_buffer + t * (nfft * 2);

          // first stage

          computeLargeButterflyFirststage<DT>(
              output_batch, input_batch, in_stride, section_num, twiddles,
              sram_dftmtx, (void *)nram_buf, small_factors, direction, nfft,
              last_stage);
        }
      } else {
        // printf("computeLargeButterflyFirststageBatchPingpong\n");
        computeLargeButterflyFirststageBatchPingpong<DT>(
            output, input, in_stride, section_num, twiddles, sram_dftmtx,
            (void *)nram_buf, small_factors, direction, nfft, last_stage,
            t_start, t_end);
      }
    }

    // __sync();
  }

  // sram_large_tw
  stage_count--;
  if (stage_count == 0) {
    // continue;
    return;
  }

  // if (__is_mpu()) {
  //   return;
  // }

  // sram_large_tw
  value_mul = 10;
  for (; stage_count > 1; stage_count--) {
    // fft_swap_ptr<DT>(&buffer, &output);
    // FFT_SWAP_PTR(buffer, output);
    FFT_SWAP_PTR(buffer, output);

    // if (is_two_stages && is_in_place) {
    //   if (_stage_count % 2 == 0) {
    //     if ((_stage_count - stage_count) == 2)
    //       fft_swap_ptr<DT>(&odd_extra_buffer, &output);
    //   } else {
    //     if ((_stage_count - stage_count) == 3)
    //       fft_swap_ptr<DT>(&odd_extra_buffer, &output);
    //   }
    // }

    if (stage_count == 2 && _stage_count % 2) {
      // fft_swap_ptr<DT>(&odd_extra_buffer, &output);
      FFT_SWAP_PTR(odd_extra_buffer, output);
    }

    // value_mul = (_stage_count - stage_count + 1) * 5;

    // update parameter
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul++];

    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      // MLULOG("other stage radix: %d \n", radix);

      if (repeat_num > 0 || taskId < remain_num) {
        if (6000 / radix > repeat_num && 0) {
          for (int t = t_start; t < t_end; t++) {
            DT *output_batch = output + t * (nfft << 1);
            DT *buffer_batch = buffer + t * (nfft << 1);

            computeLargeButterflyOtherstages<DT>(
                output_batch, buffer_batch, (DT *)twiddles, _twiddles,
                sram_dftmtx, section_num, butterfly_num, in_stride,
                (void *)nram_buf, small_factors, nfft, direction, 0);

            // __sync();
          }
        } else {
          computeLargeButterflyOtherstagesBatchPingpong<DT>(
              output, buffer, (DT *)twiddles, _twiddles, sram_dftmtx,
              section_num, butterfly_num, in_stride, (void *)nram_buf,
              small_factors, nfft, t_start, t_end, direction, 0);
        }
      }
    }
    twiddles += butterfly_num * (radix - 1) * 2;  // 2 for complex
  }                                               // for (stage_count)

  // __mlu_shared__ DT *sram_tw[2048];  // radix-1024
  // __mlu_shared__ DT *sram_tw[64];  // radix-1024
  // last stage
  {
    if ((_stage_count % 2 == 1)) {
      FFT_SWAP_PTR(odd_extra_buffer, buffer);
    }

    // fft_swap_ptr<DT>(&buffer, &output);
    FFT_SWAP_PTR(buffer, output);

    // update parameter
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul];

    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      if (repeat_num > 0 || taskId < remain_num) {
        if (0) {
          for (int t = t_start; t < t_end; t++) {
            DT *output_batch = output + t * (nfft << 1);
            DT *buffer_batch = buffer + t * (nfft << 1);

            computeLargeButterflyLaststage<DT>(
                output_batch, buffer_batch, (DT *)twiddles, _twiddles,
                sram_dftmtx, section_num, butterfly_num, in_stride,
                (void *)nram_buf, small_factors, nfft, direction);
          }
        } else {
          computeLargeButterflyLaststageBatchPingpong(
              output, buffer, (DT *)twiddles, _twiddles, sram_dftmtx,
              section_num, butterfly_num, in_stride, (void *)nram_buf,
              small_factors, nfft, t_start, t_end, direction);
        }
      }
    }
  }
}

template <typename DT>
__mlu_func__ void computeMutiStageOnchipColumn(DT *input, DT *output,
                                               int *factors, DT *twiddles,
                                               DT *twiddles_end,
                                               const DT *dft_matrix, DT *buffer,
                                               int batch, int fft_flag,
                                               int direction, int nb) {
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
      small_factors_offset;
  // const int is_two_stages = (stage_count == 2);  // the variable for
  // twostages
  // const int is_in_place = (input == output);
  // const DT *_twiddles = twiddles;
  int *small_factors;
  int last_stage;
  __nram__ int nram_factors[FFT_MAXFACTORS];

  int sram_offset = 0;
  int *sram_factors = (int *)(sram_buffer + sram_offset);
  sram_offset += FFT_MAXFACTORS * sizeof(int);
  DT *sram_dftmtx = (DT *)(sram_buffer + sram_offset);
  sram_offset += DFT_TABLE_SIZE * sizeof(DT);
  DT *sram_twiddles = (DT *)(sram_buffer + sram_offset);
  const int twiddles_size = twiddles_end - twiddles;
  sram_offset += twiddles_size * sizeof(DT);

  const int _stage_count = factors[0];
  const int nfft = factors[1];

  // first stage
  radix = factors[5 + 0];
  section_num = factors[5 + 1];
  in_stride = factors[5 + 3];
  small_factors_offset = factors[5 + 4];

  stage_count = _stage_count;
  last_stage = (stage_count == 1);

  if (__is_mpu()) {
    __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                   GDRAM2SRAM);
    if (twiddles_size) {
      __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                     GDRAM2SRAM);
    }
    // _small_stage_count = small_factors[0];

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
    // twiddles = sram_twiddles;
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

  DT *odd_extra_buffer;
  // TODO(zrg): find largest radix, 6000/ largest
  // int max_para_batch = (6144 / radix) > batch ? batch : (6144 / radix);
  int max_para_batch;

  //  int max_para_batch = 6000/64;

  if (__is_ipu()) {
    // FFT_CPX_T<DT> *odd_extra_buffer = (FFT_CPX_T<DT> *)buffer + batch * nfft;
    // // for in_place temp buffer
    odd_extra_buffer =
        buffer + batch * (nfft << 1);  // for in_place temp buffer
    // out_place: input -> output (1 stage)
    //            input -> buffer -> output (2 stage)
    //            input -> buffer -> odd_extra_buffer -> output (3 stage)
    //            input -> buffer -> output -> buffer -> output (4 stage)
    //            input -> buffer -> output -> buffer -> odd_extra_buffer ->
    //            output (5 stage)

    // _stage_count = stage_count;

    if (_stage_count != 1) FFT_SWAP_PTR(buffer, output);

    if (repeat_num > 0 || taskId < remain_num) {
      small_factors = factors + small_factors_offset;
      // max_para_batch = small_factors[3] > batch ? batch : small_factors[3];
      max_para_batch = (6144 / radix) > batch ? batch : (6144 / radix);
      for (int t = t_start; t < t_end; t += max_para_batch) {
        // MLULOG("taskId: %d, batchId: %d\n", taskId, t);
        int para_batch =
            (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);
        DT *input_batch = input + t * 2;
        DT *output_batch;
        if (last_stage) {
          output_batch = output + t * 2;
        } else {
          output_batch = output + t * nfft * 2;
        }
        // DT *buffer_batch = buffer + t * (nfft * 2);
        // DT *odd_extra_buffer_batch = odd_extra_buffer + t * (nfft * 2);

        // first stage
        // int radix

        // MLULOG("para_batch: %d\n", para_batch);

        computeLargeButterflyFirststageColumn<DT>(
            output_batch, input_batch, in_stride, section_num, twiddles,
            sram_dftmtx, (void *)nram_buf, small_factors, direction, nfft,
            last_stage, para_batch, nb);
      }
    }
    // __sync();
    stage_count--;
    if (stage_count == 0) {
      // continue;

      return;
    }
  }

  // sram_large_tw

  // sram_large_tw
  value_mul = 10;
  for (; stage_count > 1; stage_count--) {
    // fft_swap_ptr<DT>(&buffer, &output);
    // FFT_SWAP_PTR(buffer, output);
    FFT_SWAP_PTR(buffer, output);

    if (stage_count == 2 && _stage_count % 2) {
      // fft_swap_ptr<DT>(&odd_extra_buffer, &output);
      FFT_SWAP_PTR(odd_extra_buffer, output);
    }

    // value_mul = (_stage_count - stage_count + 1) * 5;

    // update parameter
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul++];

    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      // MLULOG("other stage radix: %d \n", radix);
      // int max_para_batch = (6000 + radix - 1) / radix;
      if (repeat_num > 0 || taskId < remain_num) {
        // for (int t = t_start; t < t_end; t++) {
        //   // DT *output_batch = output + t * 2;
        //   // DT *buffer_batch = buffer + t * 2;

        //     computeLargeButterflyOtherstagesColumn<DT>(
        //         output_batch, buffer_batch, (DT *)twiddles, _twiddles,
        //         sram_dftmtx, section_num, butterfly_num, in_stride,
        //         (void *)nram_buf, small_factors, nfft, direction, 0,
        //         last_stage, para_batch, nb0, nb1);
        // }

        for (int t = t_start; t < t_end; t += max_para_batch) {
          // MLULOG("taskId: %d, batchId: %d\n", taskId, t);
          DT *output_batch = output + t * (nfft * 2);
          DT *buffer_batch = buffer + t * (nfft * 2);
          // DT *buffer_batch = buffer + t * (nfft * 2);
          // DT *odd_extra_buffer_batch = odd_extra_buffer + t * (nfft * 2);

          // first stage
          // int radix
          int para_batch =
              (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);

          computeLargeButterflyOtherstagesColumn<DT>(
              output_batch, buffer_batch, (DT *)twiddles, _twiddles,
              sram_dftmtx, section_num, butterfly_num, in_stride,
              (void *)nram_buf, small_factors, nfft, direction, 0, para_batch,
              nb);
        }
      }
    }
    twiddles += butterfly_num * (radix - 1) * 2;  // 2 for complex
  }                                               // for (stage_count)

  // __mlu_shared__ DT *sram_tw[2048];  // radix-1024
  // __mlu_shared__ DT *sram_tw[64];  // radix-1024
  // last stage
  {
    if ((_stage_count % 2 == 1)) {
      FFT_SWAP_PTR(odd_extra_buffer, buffer);
    }

    // fft_swap_ptr<DT>(&buffer, &output);
    FFT_SWAP_PTR(buffer, output);

    // update parameter
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul];

    small_factors = factors + small_factors_offset;

    // DT * sram_la = sram_large_tw

    // if (__is_mpu()) {
    //   __memcpy(sram_tw, twiddles, sizeof(DT) * 2 * butterfly_num * (radix -
    //   1),
    //            GDRAM2SRAM);
    // }

    // __sync_cluster();
    // // imag
    // __memcpy(
    //     sram_large_tw + butterfly_num * (radix - 1),
    //     twiddles + + butterfly_num * (radix - 1),
    //     sizeof(DT) * para_ldst_num, GDRAM2NRAM,
    //     sizeof(DT) * para_ldst_num, large_out_stride * sizeof(DT),
    //     large_radix - 2);

    if (__is_ipu()) {
      MLULOG("last stage radix: %d, section_num: %d\n", radix, section_num);

      if (repeat_num > 0 || taskId < remain_num) {
        // for (int t = t_start; t < t_end; t++) {
        //   DT *output_batch = output + t * (nfft << 1);
        //   DT *buffer_batch = buffer + t * (nfft << 1);

        //   computeLargeButterflyLaststageColumn<DT>(
        //       output_batch, buffer_batch, (DT *)twiddles, _twiddles,
        //       sram_dftmtx, section_num, butterfly_num, in_stride,
        //       (void *)nram_buf, small_factors, nfft, direction,
        //       last_stage, para_batch, nb0, nb1);
        // }
        // int max_para_batch = (6000 + radix - 1) / radix;
        for (int t = t_start; t < t_end; t += max_para_batch) {
          // MLULOG("taskId: %d, batchId: %d\n", taskId, t);
          DT *output_batch = output + t * 2;
          DT *buffer_batch = buffer + t * (nfft * 2);
          // DT *buffer_batch = buffer + t * (nfft * 2);
          // DT *odd_extra_buffer_batch = odd_extra_buffer + t * (nfft * 2);

          // first stage
          // int radix
          int para_batch =
              (max_para_batch < (t_end - t)) ? max_para_batch : (t_end - t);

          computeLargeButterflyLaststageColumn<DT>(
              output_batch, buffer_batch, (DT *)twiddles, _twiddles,
              sram_dftmtx, section_num, butterfly_num, in_stride,
              (void *)nram_buf, small_factors, nfft, direction, para_batch, nb);
        }
      }
    }
  }
}
