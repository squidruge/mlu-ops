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
#include "kernels/fft/fft_optm_device/irfft_generic_butterfly.h"
#include "kernels/fft/fft_optm_device/fft_vector_butterfly.h"


template <typename DT>
__mlu_func__ void computeLargeButterflyFirststageC2R(
    DT *output, DT *input, int large_in_stride, int section_num,
    const DT *twiddles, const DT *dft_matrix, void *nram_buf,
    const int *small_factors, int dir, int nfft, int last_stage) {
  const dft_table_entry *dft_table = (const dft_table_entry *)dft_matrix;
  // network info
  int radix, small_in_stride, small_stage_count, large_radix,
      _small_stage_count;
  int small_section_num, small_butterfly_num, value_mul;
  int tw_offset;
  // int max_radix = small_factors[4];

  printf("[debug]checkpoint1 \n");
  _small_stage_count = small_factors[0];
  large_radix = small_factors[1];
  tw_offset = small_factors[2];

  printf("[debug]checkpoint2 \n");
  const int max_para_ldst_num = (4096 + large_radix - 1) / large_radix;
  // const int max_para_ldst_num = 1;
  const DT *small_twiddles = twiddles + tw_offset * 2;  // complex

  // TODO(zrg): save nram space.
  // assign nram space
  int nram_buf_offset = 0;
  // sizeof(DT) * 2 * large_radix * max_para_ldst_num * 2
  DT *nram_in_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_in_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_r = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  DT *nram_out_i = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num;

  // parallel load/store space
  DT *nram_para_load_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_load_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_ping = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *nram_para_store_pong = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * max_para_ldst_num * 2;  // complex

  DT *_nram_tw = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += large_radix * 2;  // complex

  // load dftmtx sample
  int ld_dft_radix = -1;
  const int max_radix = 64;
  DT *nram_dftmtx = (DT *)nram_buf + nram_buf_offset;
  nram_buf_offset += max_radix * max_radix * 2;  // complex

  DT *nram_scratch = (DT *)nram_buf + nram_buf_offset;

  printf("[debug]checkpoint3 \n");
  MLULOG("nram used: %d bytes.\n",
         (int)((size_t)nram_scratch - (size_t)nram_buffer));
  
  __memcpy_async(_nram_tw, small_twiddles, large_radix * sizeof(DT) * 2,
                 SRAM2NRAM);

  // ceil
  int repeat_num = (section_num + max_para_ldst_num - 1) / max_para_ldst_num;

  // loop at the section level
  for (int repeat_id = 0; repeat_id < repeat_num + 2; ++repeat_id) {
    // pipeline: load-stage
    if (repeat_id < repeat_num) {
      // MLULOG("pipeline: load-stage.\n");
      int i = max_para_ldst_num * repeat_id;
      DT *nram_para_load =
          (repeat_id % 2 == 0) ? nram_para_load_ping : nram_para_load_pong;

      // DT *nram_dftmtx =
      //     (repeat_id % 2 == 0) ? nram_dftmtx_ping : nram_dftmtx_pong;
      int para_load_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;
      if (section_num == 1) {
        __memcpy_async(nram_para_load, input, sizeof(DT) * large_radix,
                       GDRAM2NRAM);     //Real FFT
      } else {
        // gather load
        // 2d memcpy
        // 0 1 2 3 4 ... 1023
        // GDRAM -> NRAM
        // 8bytes radix-1024
        // 64bytes

        __memcpy_async(nram_para_load, input + i,
                       sizeof(DT) * para_load_num, GDRAM2NRAM,
                       sizeof(DT) * para_load_num,
                       large_in_stride * sizeof(DT), large_radix - 1);
      }       //Real FFT
    }

    // pipeline: store-stage
    if (repeat_id >= 2) {
      // MLULOG("pipeline: store-stage.\n");
      int i = max_para_ldst_num * (repeat_id - 2);

      int para_store_num = (max_para_ldst_num > (section_num - i))
                               ? (section_num - i)
                               : max_para_ldst_num;

      DT *nram_para_store =
          (repeat_id % 2 == 0) ? nram_para_store_ping : nram_para_store_pong;

      if (last_stage) {           //last_stage ?
        if (section_num == 1) {
          __memcpy_async(output, nram_para_store, sizeof(DT) * 2 * large_radix,
                         NRAM2GDRAM);
        } else {
          // scatter-store
          __memcpy_async(output + i * large_radix * 2, nram_para_store,
                         sizeof(DT) * 2 * para_store_num * large_radix,
                         NRAM2GDRAM);
        }
      } else {
        // real
        __memcpy_async(output + i * large_radix, nram_para_store,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
        // imag
        __memcpy_async(output + i * large_radix + nfft,
                       nram_para_store + max_para_ldst_num * large_radix,
                       para_store_num * large_radix * sizeof(DT), NRAM2GDRAM);
      }
    }

    // pipeline: compute-stage

    if (repeat_id >= 1 && repeat_id < repeat_num + 1) {
      int i = max_para_ldst_num * (repeat_id - 1);

      DT *nram_para_load =
          (repeat_id % 2 != 0) ? nram_para_load_ping : nram_para_load_pong;
      DT *nram_para_store =
          (repeat_id % 2 != 0) ? nram_para_store_ping : nram_para_store_pong;

      int para_ldst_num = (max_para_ldst_num > (section_num - i))
                              ? (section_num - i)
                              : max_para_ldst_num;

      // DT *nram_transpose_load = nram_in_r;
      __bang_transpose(nram_in_r, nram_para_load, large_radix * para_ldst_num,
                       2);
      // [large_radix, para_ldst_num] -> [para_ldst_num, large_radix]
 
      // Firststage
      for (int compute_id = 0; compute_id < para_ldst_num;
           compute_id += para_ldst_num) {
        // load real & imag

        radix = small_factors[4];
        small_section_num = small_factors[5];
        small_in_stride = small_factors[7];
        small_stage_count = _small_stage_count;

        // first stage
        if (ld_dft_radix != radix) {
          ld_dft_radix = radix;
          for (int entry = 0;; entry++) {
            if (dft_table[entry].radix == ld_dft_radix) {
              __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                       sizeof(DT) * 2 * ld_dft_radix * ld_dft_radix, SRAM2NRAM);
              break;
            }

            if (dft_table[entry].radix == -1) {
              break;
            }
          }
        }
        printf("[debug]checkpoint4 \n");
        MLULOG("computeFirststageMatC2R: %d.\n", radix);
        computeGenericButterflyFirststageMatC2R(
            nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
            nram_dftmtx, small_section_num * para_ldst_num,
            small_section_num * para_ldst_num, 1, radix);
        printf("[debug]checkpoint5 \n");
        small_stage_count--;
        if (small_stage_count == 0) {
          // nram to gdram

          if (last_stage) {
            //  [2, para_ldst_num, large_radix] -> [para_ldst_num, large_radix,
            //  2]
            // DT* nram_transpose_store = nram_in_r;

            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);

          } else {
            //  [2, para_ldst_num, large_radix] -> [2, para_ldst_num,
            //  large_radix]
            // TODO(zrg): redundant move
            __memcpy(nram_para_store, nram_out_r,
                     para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store + max_para_ldst_num * large_radix,
                     nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }

          continue;
        }

        // [small_section_num, para_ldst_num, radix] -> [para_ldst_num,
        // small_section_num, radix]

        FFT_SWAP_PTR(nram_out_r, nram_in_r);
        FFT_SWAP_PTR(nram_out_i, nram_in_i);

        TRANSPOSE_XYZ2YXZ_PAIR(nram_out_r, nram_out_i, nram_in_r, nram_in_i,
                               small_section_num, para_ldst_num, radix, DT)

        value_mul = 8;
        // DT *sram_tw = (DT *)sram_buffer;
        DT *nram_tw = _nram_tw;

        // Otherstages
        for (; small_stage_count > 1; small_stage_count--) {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          // value_mul = (_small_stage_count - small_stage_count + 1) << 2;
          // // update parameter
          radix = small_factors[value_mul++];
          small_section_num = small_factors[value_mul++];
          small_butterfly_num = small_factors[value_mul++];
          small_in_stride = small_factors[value_mul++];

          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * ld_dft_radix,
                         SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }

              // computeGenericButterflyOtherstages(Fout, buffer, twiddles,
              // radix, section_num, butterfly_num, in_stride, 0, dir);
              // in_section_length is not found
              MLULOG("computeGenericButterflyOtherstagesMatC2R: %d.\n", radix);
              computeGenericButterflyOtherstagesMatC2R(
                  nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                  nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                  para_ldst_num, small_in_stride, radix);

          nram_tw += small_butterfly_num * (radix - 1) * 2;
        }  // for (stage_count)

        // Laststage
        {
          FFT_SWAP_PTR(nram_out_r, nram_in_r);
          FFT_SWAP_PTR(nram_out_i, nram_in_i);

          radix = small_factors[value_mul++];
          small_section_num = small_factors[value_mul++];
          small_butterfly_num = small_factors[value_mul++];
          small_in_stride = small_factors[value_mul];

          if (ld_dft_radix != radix) {
            ld_dft_radix = radix;
            for (int entry = 0;; entry++) {
              if (dft_table[entry].radix == ld_dft_radix) {
                __memcpy(nram_dftmtx, &dft_matrix[dft_table[entry].offset * 2],
                         sizeof(DT) * 2 * ld_dft_radix * ld_dft_radix,
                         SRAM2NRAM);
                break;
              }

              if (dft_table[entry].radix == -1) {
                break;
              }
            }
          }
          // in_section_length is not found
            MLULOG("computeGenericButterflyLaststageMatC2R: %d.\n", radix);
            computeGenericButterflyLaststageMatC2R(
                nram_out_r, nram_out_i, nram_in_r, nram_in_i, nram_scratch,
                nram_dftmtx, nram_tw, small_section_num, small_butterfly_num,
                para_ldst_num, small_in_stride, radix);
            MLULOG("computeGenericButterflyLaststageMatC2R: %d End.\n", radix);
          
          if (last_stage) {
            __bang_transpose(nram_para_store, nram_out_r, 2,
                             max_para_ldst_num * large_radix);

          } else {
            __memcpy(nram_para_store, nram_out_r,
                     para_ldst_num * large_radix * sizeof(DT), NRAM2NRAM);
            __memcpy(nram_para_store + max_para_ldst_num * large_radix,
                     nram_out_i, para_ldst_num * large_radix * sizeof(DT),
                     NRAM2NRAM);
          }
        }
      }
    }

    __sync();
  }
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