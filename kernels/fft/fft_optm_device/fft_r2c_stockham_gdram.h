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
#include "kernels/fft/fft_optm_device/fft_r2c_stockham_nram.h"

extern __nram__ char
    nram_buffer[MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024 - FFT_MAXFACTORS * 4];
__mlu_shared__ char sram_buffer[MAX_SRAM_SIZE];
extern __wram__ char wram_buffer[MAX_WRAM_SIZE];

template <typename DT>
__mlu_func__ void computeMutiStageR2COnchip(DT *input, DT *output, int *factors,
                                         const DT *twiddles, const DT *twiddles_end,
                                         const DT *dft_matrix, DT *buffer,
                                         int batch, int fft_flag) {
  int total_num = batch;
  int repeat_num = total_num / taskDim;
  int remain_num = total_num % taskDim;
 
  //printf("taskDim=%d\n", taskDim);
  //if(__is_ipu()){
  //  for(int i = 0; i < 32; i++)
  //    printf("input[%d]=%f\n", i, input[i]);
  //}
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

  int *small_factors;
  int last_stage;
  // __sync_io();
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
  if (__is_ipu()) MLULOG("nfft: %d\n", nfft);

  // first stage
  radix = factors[5 + 0];
  section_num = factors[5 + 1];
  in_stride = factors[5 + 3];
  small_factors_offset = factors[5 + 4];

  //small_factors = factors + small_factors_offset;

  stage_count = _stage_count;
  last_stage = (stage_count == 1);
  
  //if (dft_matrix != NULL) { 
      if (__is_mpu()) {
        __memcpy_async(sram_factors, factors, FFT_MAXFACTORS * sizeof(int),
                       GDRAM2SRAM);
        if(twiddles_size){
          __memcpy_async(sram_twiddles, twiddles, twiddles_size * sizeof(DT),
                       GDRAM2SRAM);
        }

        const dft_table_entry *dft_table_gdram =
            (const dft_table_entry *)dft_matrix;
        int dft_matrix_offset = dft_table_gdram[0].offset;
      
        if (dft_matrix_offset != -1) {
          // copy the table
          __memcpy(sram_dftmtx, dft_matrix, sizeof(DT) * 2 * dft_matrix_offset,
                   GDRAM2SRAM);         //R2C FFT sizeof(DT) * 2 * dft_matrix_offset
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
              // last_radix * last_radix < last_radix * 64
              //sram_dftmtx_size = sizeof(DT) * 2 * (last_radix * 64 + last_offset);
              //__memcpy_async(sram_dftmtx, dft_matrix, sram_dftmtx_size, GDRAM2SRAM);
              //__memcpy_async(
              //    sram_dftmtx, dft_matrix,
              //    sizeof(DT) * 2 * (last_radix * last_radix + last_offset),
              //    GDRAM2SRAM);   //R2C FFT
              break;
            }
          }
        }
        //factors = sram_factors;
      }
  //}
  __sync_cluster();
  
  
  if (__is_ipu()) {
    __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
             SRAM2NRAM);
    factors = nram_factors;
    // __memcpy(nram_factors, sram_factors, FFT_MAXFACTORS * sizeof(int),
    //          SRAM2NRAM);
    // factors = sram_factors;
    twiddles = sram_twiddles;
  }

  if (__is_mpu()) {
    return;
  }

  //DT *buffer2;
  //buffer2 = buffer + batch * (nfft << 2);

  const DT *_twiddles = twiddles;
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
      for (int t = t_start; t < t_end; t++) {
        // MLULOG("taskId: %d, batchId: %d\n", taskId, t);
        DT *input_batch = input + t * nfft;
        DT *output_batch = output + t * (nfft * 2);
        // DT *buffer_batch = buffer + t * (nfft * 2);
        // DT *odd_extra_buffer_batch = odd_extra_buffer + t * (nfft * 2);

        // first stage

        computeLargeButterflyFirststageR2C<DT>(
            output_batch, input_batch, in_stride, section_num, twiddles,
            sram_dftmtx, (void *)nram_buf, small_factors, nfft, last_stage);
      }

    }
    // __sync();
  } else {
    stage_count = _stage_count;
    last_stage = (stage_count == 1);
  }
  // return;

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
    FFT_SWAP_PTR(buffer, output);
    //FFT_SWAP_PTR(buffer, buffer2);

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
        for (int t = t_start; t < t_end; t++) {
          DT *output_batch = output + t * (nfft << 1);
          DT *buffer_batch = buffer + t * (nfft << 1);

          computeLargeButterflyOtherstagesR2C<DT>(
              output_batch, buffer_batch, (DT *)twiddles, _twiddles,
              sram_dftmtx, section_num, butterfly_num, in_stride,
              (void *)nram_buf, small_factors, nfft, 0);

           __sync();
        }
      }
    }
    twiddles += ((butterfly_num + 2) / 2) * (radix - 1) * 2;  // 2 for complex
  }                                               // for (stage_count)

  // last stage
  {
    if ((_stage_count % 2 == 1)) {
      FFT_SWAP_PTR(odd_extra_buffer, buffer);
    }

    // fft_swap_ptr<DT>(&buffer, &output);
    FFT_SWAP_PTR(buffer, output);
    //FFT_SWAP_PTR(buffer, buffer2);

    // update parameter
    radix = factors[value_mul++];
    section_num = factors[value_mul++];
    butterfly_num = factors[value_mul++];
    in_stride = factors[value_mul++];
    small_factors_offset = factors[value_mul];

    small_factors = factors + small_factors_offset;

    if (__is_ipu()) {
      MLULOG("last stage radix: %d, section_num: %d\n", radix, section_num);

      if (repeat_num > 0 || taskId < remain_num) {
        for (int t = t_start; t < t_end; t++) {
          DT *output_batch = output + t * (nfft << 1);
          DT *buffer_batch = buffer + t * (nfft << 1);

          computeLargeButterflyLaststageR2C<DT>(
              output_batch, buffer_batch, (DT *)twiddles, _twiddles,
              sram_dftmtx, section_num, butterfly_num, in_stride,
              (void *)nram_buf, small_factors, nfft);
        }
      }
    }
  }
}
