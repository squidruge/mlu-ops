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
#include <string>
#include "kernels/fft/fft.h"
#include "kernels/fft/rfft/rfft.h"
#include "kernels/fft/irfft/irfft.h"
#include "kernels/fft/c2c_fft/c2c_fft.h"

// May be use a common function is a better choice?
static inline bool supportFloatConv(mluOpHandle_t handle) {
  switch (handle->arch) {
    case MLUOP_MLU370:
      return true;
    default:
      return true;
  }
}

// Calculate whether the optimization strategy can be
// entered(CNFFT_FUNC_STOCKHAM and CNFFT_FUNC_COOLEY_TUKEY). If it can enter,
// select the optimal strategy and calculate corresponding parameters.
mluOpStatus_t selectFFTStrategy(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                                const std::string make_plan_api) {
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  fft_plan->fft_strategy = CNFFT_FUNC_MATMUL;
  // The basic conditions for entering the optimization.
  if (fft_plan->n[0] > 4096) {
    bool find_stockham = 0;
    // CNFFT_FUNC_STOCKHAM optimizaion currently has more retrictions as
    // follows:
    if (handle->arch >= 300 &&
        (fft_plan->execution_dtype == MLUOP_DTYPE_HALF ||
         fft_plan->execution_dtype == MLUOP_DTYPE_FLOAT)) {
      find_stockham = true;
    }
    // strategy_status: 0 means select MLUOP_FUNC_STOCKHAM, 1 means selelct
    // COOLEY_TUKEY,
    //                  -1 means still select CNFFT_FUNC_MATMUL.
    int strategy_status =
        findFFTOptLimit(handle, fft_plan->n[0], fft_plan->m, fft_plan->L,
                        fft_plan->s, fft_plan->L_sub, find_stockham);
    if (strategy_status == 1) {
      fft_plan->fft_strategy = CNFFT_FUNC_COOLEY_TUKEY;
    } else if (strategy_status == 0) {
      fft_plan->fft_strategy = CNFFT_FUNC_STOCKHAM;
    }
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpCreateFFTPlan(mluOpFFTPlan_t *fft_plan) {
  mluOpFFTStruct *ts = new (std::nothrow) mluOpFFTStruct();
  if (ts == nullptr) {
    LOG(ERROR) << "[mluOpCreateFFTPlan]: alloc failed";
    return MLUOP_STATUS_ALLOC_FAILED;
  }
  ts->factors = new int[FFT_MAXFACTORS];
  ts->factors_2d = new int[FFT_MAXFACTORS];
  *fft_plan = ts;
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddlesLine(
    void *_twiddles, const int butterfly_num, const int section_num,
    const int radix, const int nfft, const int dir) {
  int j, k;
  DT phase;
  DT *twiddles = (DT *)_twiddles;
  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (j = 0; j < butterfly_num; j++) {
    // phase = 1 when k = 0
    for (k = 1; k < radix; k++) {
      phase = sign * 2 * (DT)FFT_PI * section_num * k * j / nfft;
      twiddles[(butterfly_num * (k - 1) + j)] = (DT)cos(phase);  // r
      twiddles[(butterfly_num * (k - 1) + j) + butterfly_num * (radix - 1)] =
          (DT)sin(phase);  // i
      // twiddles[(butterfly_num * (k - 1) + j) * 2] = (DT)cos(phase);     // r
      // twiddles[(butterfly_num * (k - 1) + j) * 2 + 1] = (DT)sin(phase); // i
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddlesLineColumn(
    void *_twiddles, const int butterfly_num, const int section_num,
    const int radix, const int nfft, const int dir) {
  int j, k;
  DT phase;
  DT *twiddles = (DT *)_twiddles;
  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (j = 1; j < radix; j++) {
    // phase = 1 when k = 0
    for (k = 0; k < butterfly_num; k++) {
      phase = sign * 2 * (DT)FFT_PI * section_num * k * j / nfft;
      twiddles[((radix - 1) * k + (j - 1))] = (DT)cos(phase);  // r
      twiddles[((radix - 1) * k + (j - 1)) + butterfly_num * (radix - 1)] =
          (DT)sin(phase);  // i
      // twiddles[(butterfly_num * (k - 1) + j) * 2] = (DT)cos(phase);     // r
      // twiddles[(butterfly_num * (k - 1) + j) * 2 + 1] = (DT)sin(phase); // i
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

/**
 * @details
 * @brief   The control interfaces of the generation of FFT's twiddles.
 * @param[in]   generator       twiddle generation function pointer.
 * @param[out]  *twiddles       stores the twiddles information generated in
 * this function, first stage's is ignored.
 * @param[in]   *factors         the way plan factoring the length of inpue
 * sequence.
 * @param[in]   nfft            the length of FFT.
 * @param[in]   dir             the flag for FFT/IFFT.
 * @return none
 */

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddles(mluOpFFTPlan_t fft_plan,
                                                void *&_twiddles,
                                                void *&_twiddles_end,
                                                int *factors, const int _nfft,
                                                const int dir) {
  // twiddles = _twiddles;
  DT *twiddles = new DT[_nfft * 2 * 2];  // complex *2(large+small)
  _twiddles = twiddles;
  int stage_count = factors[0];
  int cur_large_radix, cur_small_radix, section_num, butterfly_num,
      loop_stage;  // current radix
  int tw_offset = 0;
  int small_stage_count, small_loop_stage, small_factors_offset;

  // for other stage, ignore first stage
  for (loop_stage = 2; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    section_num = factors[5 * loop_stage + 1];
    butterfly_num = factors[5 * loop_stage + 2];

    switch (fft_plan->fft_type) {
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_FLOAT2COMPLEX_FLOAT: {
        // R2C
      } break;
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_FLOAT2FLOAT: {
        // C2R
        butterfly_num = butterfly_num / 2 + 1;
      } break;
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
        // C2C
      }; break;
      default: {
      }
    }
    fftGenerateTwiddlesLine<DT>(twiddles, butterfly_num, section_num,
                                cur_large_radix, _nfft, dir);

    twiddles += butterfly_num * (cur_large_radix - 1) * 2;
    tw_offset += butterfly_num * (cur_large_radix - 1);
  }  // stage_count

  // do not ignore first stage
  for (loop_stage = 1; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    small_factors_offset = factors[5 * loop_stage + 4];
    small_stage_count = factors[small_factors_offset];
    factors[small_factors_offset + 2] = tw_offset;
    // cur_radix = factors[4 * loop_stage];
    // section_num = factors[4 * loop_stage + 1];
    // butterfly_num = factors[4 * loop_stage + 2];
    // butterfly_num = factors[4 * loop_stage + 2];
    // generator(twiddles, butterfly_num, section_num, cur_radix, _nfft, dir);
    // twiddles += butterfly_num * (cur_radix - 1);

    for (small_loop_stage = 2; small_loop_stage <= small_stage_count;
         small_loop_stage++) {
      cur_small_radix = factors[small_factors_offset + 4 * small_loop_stage];
      section_num = factors[small_factors_offset + 4 * small_loop_stage + 1];
      butterfly_num = factors[small_factors_offset + 4 * small_loop_stage + 2];
      // butterfly_num = factors[small_factors_offset + 4 * small_loop_stage +
      // 2];
      fftGenerateTwiddlesLine<DT>(twiddles, butterfly_num, section_num,
                                  cur_small_radix, cur_large_radix, dir);
      twiddles += butterfly_num * (cur_small_radix - 1) * 2;
      tw_offset +=
          butterfly_num * (cur_small_radix - 1);  // complex element offset
    }                                             // small_stage_count
  }                                               // stage_count

  _twiddles_end = (void *)((DT *)_twiddles + tw_offset * 2);
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateTwiddlesColumn(
    mluOpFFTPlan_t fft_plan, void *&_twiddles, void *&_twiddles_end,
    int *factors, const int _nfft, const int dir) {
  // twiddles = _twiddles;
  DT *twiddles = new DT[_nfft * 2 * 2];  // complex *2(large+small)
  _twiddles = twiddles;
  int stage_count = factors[0];
  int cur_large_radix, cur_small_radix, section_num, butterfly_num,
      loop_stage;  // current radix
  int tw_offset = 0;
  int small_stage_count, small_loop_stage, small_factors_offset;

  // for other stage, ignore first stage
  for (loop_stage = 2; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    section_num = factors[5 * loop_stage + 1];
    butterfly_num = factors[5 * loop_stage + 2];
    fftGenerateTwiddlesLineColumn<DT>(twiddles, butterfly_num, section_num,
                                      cur_large_radix, _nfft, dir);
    twiddles += butterfly_num * (cur_large_radix - 1) * 2;
    tw_offset += butterfly_num * (cur_large_radix - 1);
  }  // stage_count

  // do not ignore first stage
  for (loop_stage = 1; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    small_factors_offset = factors[5 * loop_stage + 4];
    small_stage_count = factors[small_factors_offset];
    factors[small_factors_offset + 2] = tw_offset;
    // cur_radix = factors[4 * loop_stage];
    // section_num = factors[4 * loop_stage + 1];
    // butterfly_num = factors[4 * loop_stage + 2];
    // butterfly_num = factors[4 * loop_stage + 2];
    // generator(twiddles, butterfly_num, section_num, cur_radix, _nfft, dir);
    // twiddles += butterfly_num * (cur_radix - 1);

    for (small_loop_stage = 2; small_loop_stage <= small_stage_count;
         small_loop_stage++) {
      cur_small_radix = factors[small_factors_offset + 4 * small_loop_stage];
      section_num = factors[small_factors_offset + 4 * small_loop_stage + 1];
      butterfly_num = factors[small_factors_offset + 4 * small_loop_stage + 2];
      // butterfly_num = factors[small_factors_offset + 4 * small_loop_stage +
      // 2];
      fftGenerateTwiddlesLine<DT>(twiddles, butterfly_num, section_num,
                                  cur_small_radix, cur_large_radix, dir);
      twiddles += butterfly_num * (cur_small_radix - 1) * 2;
      tw_offset +=
          butterfly_num * (cur_small_radix - 1);  // complex element offset
    }                                             // small_stage_count
  }                                               // stage_count

  _twiddles_end = (void *)((DT *)_twiddles + tw_offset * 2);
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateDftMatrixKernel(DT *dft_matrix,
                                                       const int radix,
                                                       const int dir) {
  int j, k;
  DT phase;
  // const int k_align = 64;
  const int K_num = 64 / sizeof(DT);
  const int align_K = K_num * ((radix + K_num - 1) / K_num);
  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (k = 0; k < radix; k++) {
    for (j = 0; j < align_K; j++) {
      if (j < radix) {
        phase = sign * 2 * (DT)FFT_PI * k * j / radix;
        dft_matrix[align_K * k + j] = (DT)cos(phase);                    // r
        dft_matrix[align_K * k + j + align_K * radix] = (DT)sin(phase);  // i
      } else {
        dft_matrix[align_K * k + j] = (DT)0.0;                    // r
        dft_matrix[align_K * k + j + align_K * radix] = (DT)0.0;  // i
      }
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateDftMatrixKernelNoPad(DT *dft_matrix,
                                                            const int radix,
                                                            const int dir) {
  int j, k;
  DT phase;

  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (k = 0; k < radix; k++) {
    for (j = 0; j < radix; j++) {
      phase = sign * 2 * (DT)FFT_PI * k * j / radix;
      dft_matrix[radix * k + j] = (DT)cos(phase);                  // r
      dft_matrix[radix * k + j + radix * radix] = (DT)sin(phase);  // i
    }                                                              // radix
  }  // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API
fftGenerateC2RDftMatrixKernelNoPad(DT *dft_matrix, const int radix) {
  int j, k;
  DT phase;
  int half = (radix / 2 + 1);
  const int sign = 1;  // backward
  for (k = 0; k < radix; k++) {
    for (j = 0; j < half; j++) {
      phase = sign * 2 * (DT)FFT_PI * k * j / radix;
      if (j == 0 || j == half - 1) {
        dft_matrix[2 * half * k + j] = (DT)cos(phase);          // r
        dft_matrix[2 * half * k + j + half] = -(DT)sin(phase);  // i neg
      } else {
        dft_matrix[2 * half * k + j] = 2 * (DT)cos(phase);          // r
        dft_matrix[2 * half * k + j + half] = -2 * (DT)sin(phase);  // i neg
      }
    }  // radix
  }    // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateHalfDftMatrixKernelNoPad(DT *dft_matrix,
                                                                const int radix,
                                                                const int dir) {
  int j, k;
  DT phase;
  int rows = radix / 2 + 1;
  const int sign = (dir == FFT_FORWARD) ? -1 : 1;
  for (k = 0; k < rows; k++) {
    for (j = 0; j < radix; j++) {
      phase = sign * 2 * (DT)FFT_PI * k * j / radix;
      dft_matrix[radix * k + j] = (DT)cos(phase);                 // r
      dft_matrix[radix * k + j + radix * rows] = (DT)sin(phase);  // i
    }                                                             // radix
  }  // butterfly_num
  return MLUOP_STATUS_SUCCESS;
}

template <typename DT>
mluOpStatus_t MLUOP_WIN_API fftGenerateDftMatrix(void *&_dft_matrix,
                                                 int *factors, const int _nfft,
                                                 const int dir) {
  // allocate space for dft_matrix_table and dft_matrix
  const std::string api = "[fftGenerateDftMatrix]";

  const int K_num = 64 / sizeof(DT);
  DT *dft_matrix = new DT[DFT_TABLE_SIZE];  // complex *2(large+small)
  dft_table_entry *dft_matrix_table = (dft_table_entry *)dft_matrix;
  _dft_matrix = dft_matrix;
  int align_K = 0;
  dft_table_entry *dft_matrix_table_end =
      dft_matrix_table + MAX_DFT_MATRIX_NR + 1;
  dft_matrix = (DT *)dft_matrix_table_end;

  // radix == -1 means the end of table
  // init table
  for (int i = 0; i < MAX_DFT_MATRIX_NR; i++) {
    dft_matrix_table[i] = {-1, -1};
  }
  // dft_matrix_table =
  int stage_count = factors[0];
  int cur_large_radix, cur_small_radix, section_num, butterfly_num,
      loop_stage;  // current radix
  int tw_offset = 0;
  int small_stage_count, small_loop_stage, small_factors_offset;

  // initialize offset as the end of table
  // transform  dft_table_entry to complex DT
  // int cur_offset =
  //     (MAX_DFT_MATRIX_NR + 1) * sizeof(dft_table_entry) / (sizeof(DT) * 2);

  int cur_offset = ((DT *)dft_matrix - (DT *)_dft_matrix) / 2;

  // dft_matrix += cur_offset * 2;
  int cur_table_entry = 0;
  for (loop_stage = 1; loop_stage <= stage_count; loop_stage++) {
    cur_large_radix = factors[5 * loop_stage];
    small_factors_offset = factors[5 * loop_stage + 4];
    small_stage_count = factors[small_factors_offset];

    for (small_loop_stage = 1; small_loop_stage <= small_stage_count;
         small_loop_stage++) {
      cur_small_radix = factors[small_factors_offset + 4 * small_loop_stage];
      section_num = factors[small_factors_offset + 4 * small_loop_stage + 1];
      butterfly_num = factors[small_factors_offset + 4 * small_loop_stage + 2];

      for (int entry = 0;; entry++) {
        if (dft_matrix_table[entry].radix == -1) {
          // DT *dft_matrix_real = dft_matrix;
          align_K = K_num * ((cur_small_radix + K_num - 1) / K_num);
          fftGenerateDftMatrixKernel<DT>(dft_matrix, cur_small_radix, dir);
          dft_matrix += cur_small_radix * align_K * 2;

          dft_matrix_table[cur_table_entry] = {cur_small_radix, cur_offset};
          cur_table_entry++;
          cur_offset += cur_small_radix * align_K;

          if (cur_table_entry == MAX_DFT_MATRIX_NR) {
            LOG(ERROR) << api << ": too much dft matrices.";
            return MLUOP_STATUS_NOT_SUPPORTED;
          }
          break;
        }

        if (dft_matrix_table[entry].radix == cur_small_radix) {
          break;
        }
      }
    }  // small_stage_count
  }    // stage_count

  return MLUOP_STATUS_SUCCESS;
}

// data struct
// factors[0]: stage_count
// factors[1]: nfft
// factors[2]: null
// factors[3]: null
// factors[4]: null
// factors[5]: null

// i-th large radix info:
// factors[5*(i+1)+0]: radix
// factors[5*(i+1)+1]: section_num
// factors[5*(i+1)+2]: butterfly_num
// factors[5*(i+1)+3]: in_stride
// factors[5*(i+1)+4]: small_factors_offset

// factors[small_factors_offset+0]: small_stage_count
// factors[small_factors_offset+1]: large radix
// factors[small_factors_offset+2]: tw_offset

// i-th large radix, j-th small radix info:
// factors[small_factors_offset+4*(j+1)+0]: radix
// factors[small_factors_offset+4*(j+1)+1]: section_num
// factors[small_factors_offset+4*(j+1)+2]: butterfly_num
// factors[small_factors_offset+4*(j+1)+3]: in_stride

mluOpStatus_t MLUOP_WIN_API fftFactor(const int _n, int *facbuf,
                                      int &small_factors_offset) {
  int n = _n;

  int r, in_stride, section_num, stage_num = 0, out_stride = 1;
  int radix_basic[] = {3, 4, 5, 6, 7, 8, 9, 11, 13, 16};

  int large_radix = 1;
  facbuf += small_factors_offset;
  // 256 = 16 * 16
  // 512 = 32 * 16
  // 1024 = 32 * 32
  // 2048 = 64 * 32
  // while (n > 1) {
  //   if ((n % 64) == 0 && 0) {
  //     r = 64;
  //     // set small factors
  //   } else if ((n % 32) == 0 && 0) {
  //     r = 32;
  //   } else if ((n % 16) == 0 ) {
  //     r = 16;
  //   } else if ((n % 8) == 0 && 0) {
  //     r = 8;
  //     // set small factors
  //   } else if ((n % 27) == 0) {
  //     r = 27;
  //   } else if ((n % 9) == 0) {
  //     r = 9;
  //   } else if ((n % 3) == 0) {
  //     r = 3;
  //   } else {
  //     // prime
  //     r = n;
  //   }

  while (n > 1) {
    // if (n % 16 ==0) {
    //   r = 16;
    // } else if ((n % 8) == 0) {
    //   r = 8;
    // } else if ((n % 25) == 0) {
    //   r = 25;
    // } else if ((n % 28) == 0) {
    //   r = 28;
    // } else {
    //   // prime
    //   r = n;
    // }

    switch (_n) {
      case 128:
        if (n % 16 == 0) {
          r = 16;
        } else if ((n % 8) == 0) {
          r = 8;
        }
        break;

      case 12:
        if (n % 4 == 0) {
          r = 4;
        } else if ((n % 3) == 0) {
          r = 3;
        }
        break;

      case 140:
        if (n % 14 == 0) {
          r = 14;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      case 160:
        if (n % 16 == 0) {
          r = 16;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      case 200:
        if (n % 20 == 0) {
          r = 20;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      case 256:
        if (n % 16 == 0) {
          r = 16;
        }
        break;

      case 275:
        if (n % 25 == 0) {
          r = 25;
        } else if ((n % 11) == 0) {
          r = 11;
        }
        break;

      case 280:
        if (n % 20 == 0) {
          r = 20;
        } else if ((n % 14) == 0) {
          r = 14;
        }
        break;

      case 300:
        if (n % 30 == 0) {
          r = 30;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      case 320:
        if (n % 20 == 0) {
          r = 20;
        } else if ((n % 16) == 0) {
          r = 16;
        }
        break;

      case 350:
        if (n % 25 == 0) {
          r = 25;
        } else if ((n % 14) == 0) {
          r = 14;
        }
        break;

      case 400:
        if (n % 25 == 0) {
          r = 25;
        } else if ((n % 16) == 0) {
          r = 16;
        }
        break;

      case 500:
        if (n % 25 == 0) {
          r = 25;
        } else if ((n % 20) == 0) {
          r = 20;
        }
        break;

      case (32 * 17):
        if (n % 32 == 0) {
          r = 32;
        } else if ((n % 17) == 0) {
          r = 17;
        }
        break;

      case 600:
        if (n % 30 == 0) {
          r = 30;
        } else if ((n % 20) == 0) {
          r = 20;
        }
        break;

      case 650:
        if (n % 25 == 0) {
          r = 25;
        } else if ((n % 26) == 0) {
          r = 26;
        }
        break;

      case 2048:
        if (n % 16 == 0) {
          r = 16;
        } else if ((n % 8) == 0) {
          r = 8;
        }
        break;

      case 4096:
        if (n % 16 == 0) {
          r = 16;
        }
        break;

      case 6000:
        if (n % 30 == 0) {
          r = 30;
        } else if ((n % 20) == 0) {
          r = 20;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      case 7000:
        if (n % 50 == 0) {
          r = 50;
        } else if ((n % 14) == 0) {
          r = 14;
        } else if ((n % 10) == 0) {
          r = 10;
        }
        break;

      default:
        if (_n <= 64) {
          r = _n;
          break;
        } else {
          for (int cur_r = 64; cur_r > 1; cur_r--) {
            if (n % cur_r == 0) {
              r = cur_r;
              break;
            }
          }
        }

        break;
    }

    n /= r;
    in_stride = _n / r;
    section_num = n;
    stage_num++;

    facbuf[4 * stage_num + 0] = r;
    facbuf[4 * stage_num + 1] = section_num;
    facbuf[4 * stage_num + 2] = out_stride;
    facbuf[4 * stage_num + 3] = in_stride;

    out_stride *= r;
  }

  facbuf[0] = stage_num;
  facbuf[1] = _n;
  facbuf[2] = 0;  // tw_offset
  // facbuf[3] = 0;  // max_parallel_num has been set

  if (stage_num > 21) {
    // Since nfft is openfft_int32_t, stage_num can never be greater than 21,
    // because 3^21 > 2^32 printf("ERROR, unsupported length for int32 type.
    // __FILE__: %s, __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  }

  small_factors_offset += (stage_num + 1) * 4;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API fftTwoStepFactor(mluOpFFTPlan_t fft_plan,
                                             const int _n, int *facbuf,
                                             int is_row_major) {
  int n = _n;
  // if ((facbuf == NULL) || (n <= 0))
  // {
  //   printf("ERROR, facbuf is NULL or n smaller and equal to 0.  __FILE__: %s,
  //   __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  // }

  int r, in_stride, section_num, stage_num = 0, out_stride = 1;
  int radix_basic[] = {3, 4, 5, 6, 7, 8, 9, 11, 13, 16};

  int large_radix = 1;
  int small_factors_offset = 22 * 5;
  // is_row_major = 0;
  while (n > 1) {
    if (is_row_major) {
      switch (_n) {
        case ((32 * 17) * (32 * 17)):
          if (n % (32 * 17) == 0) {
            r = (32 * 17);
          }
          break;

        case ((13 * 17) * (13 * 17)* (13 * 17)):
          if (n % (13 * 17) == 0) {
            r = (13 * 17);
          }
          break;

        case ((25) * (25)* (25)):
          if (n % (25) == 0) {
            r = (25);
          }
          break;

        case ((32 * 17)):
          if (n % (32 * 17) == 0) {
            r = (32 * 17);
          }
          break;

        case ((23 * 300)):
          if (n % (23) == 0) {
            r = (23);
          } else if (n % (300) == 0) {
            r = (300);
          }
          break;

        case ((32) * (32 * 17)):
          if (n % (32 * 17) == 0) {
            r = (32 * 17);
          } else if (n % (32) == 0) {
            r = (32);
          }
          break;

        case ((58 * 17) * (33)):
          if (n % (58 * 17) == 0) {
            r = (58 * 17);
          } else if (n % (33) == 0) {
            r = (33);
          }
          break;

        case (200):
          r = 200;
          break;

        case (600):
          r = 600;
          break;

        case (256):
          r = 256;
          break;

        case 1024:
          if (n % 32 == 0) {
            r = 32;
          }
          break;

        case 2048:
          if (n % 64 == 0) {
            r = 64;
          } else if ((n % 32) == 0) {
            r = 32;
          }
          break;

        case 4096:
          4096;
          break;

        case 6000:
          r = 6000;
          break;
          // case 6000:
          //   if (n % 300 == 0) {
          //     r = 300;
          //   } else if ((n % 20) == 0) {
          //     r = 20;
          //   }
          //   break;

        case 7000:
          r = 7000;
          break;

          // case 7000:
          //   if (n % 280 == 0) {
          //     r = 280;
          //   } else if ((n % 25) == 0) {
          //     r = 25;
          //   }
          //   break;

        case 8000:
          if (n % 160 == 0) {
            r = 160;
          } else if ((n % 50) == 0) {
            r = 50;
          }
          break;

        case 9000:
          if (n % 500 == 0) {
            r = 500;
          } else if ((n % 18) == 0) {
            r = 18;
          }
          break;

        case 10000:
          if (n % 500 == 0) {
            r = 500;
          } else if ((n % 20) == 0) {
            r = 20;
          }
          break;

        case 11000:
          if (n % 275 == 0) {
            r = 275;
          } else if ((n % 40) == 0) {
            r = 40;
          }
          break;

        case 12000:
          if (n % 400 == 0) {
            r = 400;
          } else if ((n % 30) == 0) {
            r = 30;
          }
          break;

        case 13000:
          if (n % 650 == 0) {
            r = 650;
          } else if ((n % 20) == 0) {
            r = 20;
          }
          break;

          // case 14000:
          //   if (n % 40 == 0) {
          //     r = 40;
          //   } else if ((n % 25) == 0) {
          //     r = 25;
          //   } else if ((n % 14) == 0) {
          //     r = 14;
          //   }
          //   break;

        case 14000:
          if (n % 350 == 0) {
            r = 350;
          } else if ((n % 40) == 0) {
            r = 40;
          }
          break;

        default:
          if (n <= 64) {
            r = n;
          } else {
            int *cur_facbuf = &facbuf[small_factors_offset];
            searchLargeRadix(fft_plan, r, cur_facbuf, stage_num + 1, n);
          }
          break;
      }
    } else {
      // column major

      for (int cur_r = 64; cur_r > 1; cur_r--) {
        if (n % cur_r == 0) {
          r = cur_r;
          break;
        }
      }
    }
    n /= r;

    switch (fft_plan->fft_type) {
      // r2c
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_FLOAT2COMPLEX_FLOAT:
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_FLOAT2FLOAT: {
        if ((n * r) != _n) {
          in_stride = (((out_stride / 2) + 1) * section_num) / r;

        } else {
          in_stride = _n / r;
        }
      } break;

      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
        in_stride = _n / r;
      } break;

      default:
        break;
    }

    section_num = n;
    stage_num++;

    facbuf[5 * stage_num + 0] = r;
    facbuf[5 * stage_num + 1] = section_num;
    facbuf[5 * stage_num + 2] = out_stride;
    facbuf[5 * stage_num + 3] = in_stride;
    facbuf[5 * stage_num + 4] = small_factors_offset;
    int *cur_facbuf = &facbuf[small_factors_offset];
    fftFactor(r, facbuf, small_factors_offset);
    setMaxParallelNum(fft_plan, cur_facbuf, stage_num);

    // facbuf[6*stage_num+4] = small_stage_count;
    // facbuf[6*stage_num+5] = small_factors_offset;

    // facbuf[4 * stage_num] = r;
    // facbuf[4 * stage_num + 1] = section_num;
    // facbuf[4 * stage_num + 2] = out_stride;
    // facbuf[4 * stage_num + 3] = in_stride;

    out_stride *= r;
  }

  facbuf[0] = stage_num;
  facbuf[1] = _n;
  facbuf[2] = 0;
  facbuf[3] = 0;
  facbuf[4] = 0;

  VLOG(5) << "stage_num: " << stage_num << " _n: " << _n;

  if (stage_num > 21) {
    // Since nfft is openfft_int32_t, stage_num can never be greater than 21,
    // because 3^21 > 2^32 printf("ERROR, unsupported length for int32 type.
    // __FILE__: %s, __LINE__: %d. \n", __FILE__, __LINE__); exit(1);
  }

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API searchLargeRadix(mluOpFFTPlan_t fft_plan,
                                             int &large_radix, int *facbuf,
                                             int large_stage_id, int _n) {
  large_radix = 1;

  int cur_stage_num = 0, cur_large_radix = 1;
  int section_num = 0;
  int stage_num = 0, out_stride = 1;
  int n = _n;
  int small_radix;
  while (n > 1) {
    for (small_radix = 64; small_radix > 1; small_radix--) {
      if (n % small_radix == 0) {
        cur_stage_num = stage_num + 1;

        facbuf[4 * cur_stage_num + 0] = small_radix;
        facbuf[4 * cur_stage_num + 1] = large_radix;
        facbuf[4 * cur_stage_num + 2] = out_stride * small_radix;

        facbuf[0] = cur_stage_num;
        facbuf[1] = large_radix * small_radix;
        int parallel_num_lb = 0;

        calParallelNumLowBound(fft_plan, facbuf, large_stage_id,
                               parallel_num_lb);
        if (parallel_num_lb > 0) {
          out_stride *= small_radix;
          large_radix *= small_radix;
          section_num = n / small_radix;
          stage_num++;
          n /= small_radix;
          break;
        } else {
          facbuf[0] = stage_num;
          facbuf[1] = large_radix;
        }
      }
    }

    if (small_radix == 1) {
      break;
    }
  }

  return MLUOP_STATUS_SUCCESS;
}

// low bound
mluOpStatus_t MLUOP_WIN_API calParallelNumLowBound(mluOpFFTPlan_t fft_plan,
                                                   int *facbuf, int stage,
                                                   int &parallel_num_lb) {
  const size_t nram_space_size =
      (MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024 - FFT_MAXFACTORS * 4);
  size_t workspace_size = 0;
  size_t reservespace_size = 0;
  const int max_radix = 64;
  size_t TYPE_SIZE = 0;
  parallel_num_lb = 0;
  int nram_space_need = 0;
  int nram_space_need_tw = 0;
  int nram_space_need_dftmtx = (stage == 1)
                                   ? max_radix * max_radix * 2 * 2
                                   : max_radix * max_radix * 2;  // complex
  // int nram_space_need_dftmtx_align = 0;
  int space_need_matmul = 0;
  int space_need_matmul_tmp = 0;
  int small_stage_num = facbuf[0];
  int _n = facbuf[1];
  int radix = 0;
  int section_num = 0;
  int butterfly_num = 0;
  int para_num = 0;
  int K_num = 0;
  int align_M = 0;
  int align_K = 0;
  int align_N = 0;

  mluOpStatus_t status;

  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      TYPE_SIZE = 4;
      K_num = 64 / TYPE_SIZE;

      if (stage == 1) {
        nram_space_need = 0;
        // nram_para_load_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex

      } else {
        nram_space_need = 0;
        // nram_para_load_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_tw
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // // _nram_tw
        // nram_space_need += _n * 2 * TYPE_SIZE;  // complex
      }

      space_need_matmul = 0;
      if (stage != 1) {
        space_need_matmul = _n * 4 * TYPE_SIZE;
      }
      for (int small_stage_id = 1; small_stage_id <= small_stage_num;
           small_stage_id++) {
        radix = facbuf[small_stage_id * 4 + 0];
        section_num = facbuf[small_stage_id * 4 + 1];
        butterfly_num = facbuf[small_stage_id * 4 + 2];
        if (small_stage_id == 1) {
          para_num = section_num * 2;
          align_M = radix;
          align_K = K_num * ((radix + K_num - 1) / K_num);
          align_N = para_num;
          // align_N = 64 * ((para_num + 64 - 1) / 64);

          space_need_matmul_tmp =
              ((align_M * 2 > align_K) ? (align_M * 2) : align_K) * align_N *
              2 * TYPE_SIZE;
          // nram_space_need_dftmtx_align =
          //     (align_K != radix) ? (align_M * align_K * 2 * TYPE_SIZE) : 0;
        } else {
          para_num = butterfly_num * section_num;
          align_M = radix;
          align_K = K_num * ((radix + K_num - 1) / K_num);
          // align_N = 64 * ((para_num + 64 - 1) / 64);
          align_N = para_num;

          space_need_matmul_tmp = 0;
          space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (para_num * radix * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (align_K * 4 * align_N * TYPE_SIZE);
          // nram_space_need_dftmtx_align =
          //     (align_K != radix) ? (align_M * align_K * 2 * TYPE_SIZE) : 0;
        }

        // nram_space_need_dftmtx_align =
        //     (align_K != radix) ? (align_M * 64 * 2 * TYPE_SIZE) : 0;
        space_need_matmul = (space_need_matmul > space_need_matmul_tmp)
                                ? space_need_matmul
                                : space_need_matmul_tmp;
      }

      // nram_space_need += space_need_matmul;
      nram_space_need_tw = _n * 2 * TYPE_SIZE;  // complex
      const size_t nram_space_remain =
          (nram_space_size - nram_space_need_tw - nram_space_need_dftmtx);
      parallel_num_lb =
          nram_space_remain / (nram_space_need + space_need_matmul);
    }; break;
  }

  // return status;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API setMaxParallelNum(mluOpFFTPlan_t fft_plan,
                                              int *facbuf, int stage) {
  const std::string make_plan_api = "[setMaxParallelNum]";

  const size_t nram_space_size =
      (MAX_NRAM_SIZE + REM_FOR_STACK - 32 * 1024 - FFT_MAXFACTORS * 4);
  size_t workspace_size = 0;
  size_t reservespace_size = 0;
  const int max_radix = 64;
  size_t TYPE_SIZE = 0;
  int max_parallel_num = 0;
  int nram_space_need = 0;
  int nram_space_need_tw = 0;
  int nram_space_need_dftmtx = (stage == 1)
                                   ? max_radix * max_radix * 2 * 2
                                   : max_radix * max_radix * 2;  // complex
  // int nram_space_need_dftmtx_align = 0;
  int space_need_matmul = 0;
  int space_need_matmul_tmp = 0;
  int small_stage_num = facbuf[0];
  int _n = facbuf[1];
  int radix = 0;
  int section_num = 0;
  int butterfly_num = 0;
  int para_num = 0;
  int K_num = 0;
  int align_M = 0;
  int align_K = 0;
  int align_N = 0;

  mluOpStatus_t status;
  // space(large_radix) * para_num > space(para_num * large_radix)
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      TYPE_SIZE = 4;
      K_num = 64 / TYPE_SIZE;

      if (stage == 1) {
        nram_space_need = 0;
        // nram_para_load_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex

      } else {
        nram_space_need = 0;
        // nram_para_load_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_ping
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_store_pong
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // nram_para_load_tw
        nram_space_need += _n * 2 * TYPE_SIZE;  // complex
        // // _nram_tw
        // nram_space_need += _n * 2 * TYPE_SIZE;  // complex
      }

      space_need_matmul = 0;
      if (stage != 1) {
        space_need_matmul = _n * 4 * TYPE_SIZE;
      }
      for (int small_stage_id = 1; small_stage_id <= small_stage_num;
           small_stage_id++) {
        radix = facbuf[small_stage_id * 4 + 0];
        section_num = facbuf[small_stage_id * 4 + 1];
        butterfly_num = facbuf[small_stage_id * 4 + 2];
        if (small_stage_id == 1) {
          para_num = section_num * 2;
          align_M = radix;
          align_K = K_num * ((radix + K_num - 1) / K_num);
          align_N = para_num;
          // align_N = 64 * ((para_num + 64 - 1) / 64);

          space_need_matmul_tmp =
              ((align_M * 2 > align_K) ? (align_M * 2) : align_K) * align_N *
              2 * TYPE_SIZE;
          // nram_space_need_dftmtx_align =
          //     (align_K != radix) ? (align_M * align_K * 2 * TYPE_SIZE) : 0;
        } else {
          para_num = butterfly_num * section_num;
          align_M = radix;
          align_K = K_num * ((radix + K_num - 1) / K_num);
          // align_N = 64 * ((para_num + 64 - 1) / 64);
          align_N = para_num;

          space_need_matmul_tmp = 0;
          space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (para_num * radix * 2 * TYPE_SIZE);
          space_need_matmul_tmp += (align_K * 4 * align_N * TYPE_SIZE);
          // nram_space_need_dftmtx_align =
          //     (align_K != radix) ? (align_M * align_K * 2 * TYPE_SIZE) : 0;
        }

        // nram_space_need_dftmtx_align =
        //     (align_K != radix) ? (align_M * 64 * 2 * TYPE_SIZE) : 0;
        space_need_matmul = (space_need_matmul > space_need_matmul_tmp)
                                ? space_need_matmul
                                : space_need_matmul_tmp;
      }

      // nram_space_need += space_need_matmul;
      nram_space_need_tw = _n * 2 * TYPE_SIZE;  // complex
      const size_t nram_space_remain =
          (nram_space_size - nram_space_need_tw - nram_space_need_dftmtx);
      max_parallel_num =
          nram_space_remain / (nram_space_need + space_need_matmul);

      // _nram_tw
      // calParallelNumLowBound(fft_plan, facbuf, stage, max_parallel_num);

      while (1) {
        space_need_matmul = 0;

        for (int small_stage_id = 1; small_stage_id <= small_stage_num;
             small_stage_id++) {
          radix = facbuf[small_stage_id * 4 + 0];
          section_num = facbuf[small_stage_id * 4 + 1];
          butterfly_num = facbuf[small_stage_id * 4 + 2];
          if (small_stage_id == 1) {
            para_num = section_num * 2 * max_parallel_num;
            align_M = radix;
            align_K = K_num * ((radix + K_num - 1) / K_num);
            align_N = 64 * ((para_num + 64 - 1) / 64);

            space_need_matmul_tmp =
                ((align_M * 2 > align_K) ? (align_M * 2) : align_K) * align_N *
                2 * TYPE_SIZE;

          } else {
            para_num = butterfly_num * section_num * max_parallel_num;
            align_M = radix;
            align_K = K_num * ((radix + K_num - 1) / K_num);
            align_N = 64 * ((para_num + 64 - 1) / 64);

            space_need_matmul_tmp = 0;
            space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
            space_need_matmul_tmp += (align_N * align_K * 2 * TYPE_SIZE);
            space_need_matmul_tmp += (para_num * radix * 2 * TYPE_SIZE);
            space_need_matmul_tmp += (align_K * 4 * align_N * TYPE_SIZE);
          }

          space_need_matmul = (space_need_matmul > space_need_matmul_tmp)
                                  ? space_need_matmul
                                  : space_need_matmul_tmp;
        }

        if (nram_space_remain >
                (nram_space_need * max_parallel_num + space_need_matmul) &&
            ((max_parallel_num * _n % 64) == 0 || max_parallel_num < 64)) {
          break;
        } else {
          max_parallel_num--;
        }
      }
    }; break;
  }

  if (max_parallel_num <= 0) {
    status = MLUOP_STATUS_ALLOC_FAILED;
  } else {
    facbuf[3] = max_parallel_num;
    status = MLUOP_STATUS_SUCCESS;
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAllocateC2C1D(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                   mluOpTensorDescriptor_t input_desc,
                   mluOpTensorDescriptor_t output_desc, const int nfft) {
  const std::string make_plan_api = "[mluOpAllocateC2C1D]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);

  int batch = fft_plan->batch;

  size_t buffer_size = batch * in_c_dtype_size * nfft;

  workspace_size = buffer_size * 2;
  workspace_size +=
      (fft_plan->is_input_contiguous || fft_plan->is_batch_contiguous)
          ? 0
          : buffer_size;
  workspace_size +=
      (fft_plan->is_output_contiguous || fft_plan->is_batch_contiguous)
          ? 0
          : buffer_size;

  size_t twiddles_size = in_c_dtype_size * nfft * 2;
  reservespace_size = sizeof(int) * (FFT_MAXFACTORS)            /* factors */
                      + twiddles_size * 2 + DFT_TABLE_SIZE * 2; /* twiddles */

  fft_plan->workspace_size = workspace_size;
  fft_plan->reservespace_size = reservespace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpAllocateC2C2D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc) {
  const std::string make_plan_api = "[mluOpAllocateC2C2D]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);

  int batch = fft_plan->batch;
  const int _n0 = fft_plan->n[0];
  const int _n1 = fft_plan->n[1];

  size_t buffer_size = batch * in_c_dtype_size * _n0 * _n1;

  size_t twiddles_size = in_c_dtype_size * _n0;
  size_t twiddles_size_2d = in_c_dtype_size * _n1;

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    reservespace_size =
        (in_c_dtype_size * _n0 * _n0 + in_c_dtype_size * _n1 * _n1) *
        2; /* DFT matrix */
    workspace_size = buffer_size * 6;
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    reservespace_size = sizeof(int) * (FFT_MAXFACTORS) /* factors */
                        + sizeof(int) * (FFT_MAXFACTORS) + twiddles_size * 2 +
                        DFT_TABLE_SIZE * 2 + twiddles_size_2d * 2 +
                        DFT_TABLE_SIZE * 2; /* twiddles */
    workspace_size = buffer_size * 2;
    workspace_size += (fft_plan->is_input_contiguous) ? 0 : buffer_size;
    workspace_size += (fft_plan->is_output_contiguous) ? 0 : buffer_size;
  }

  fft_plan->workspace_size = workspace_size;
  fft_plan->reservespace_size = reservespace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API
mluOpAllocateC2R1D(mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
                   mluOpTensorDescriptor_t input_desc,
                   mluOpTensorDescriptor_t output_desc, const int nfft) {
  const std::string make_plan_api = "[mluOpAllocateC2R1D]";
  size_t workspace_size = 0;
  size_t reservespace_size = 0;

  mluOpDataType_t in_c_dtype = fft_plan->input_dtype;
  size_t in_c_dtype_size = mluOpDataTypeBytes(in_c_dtype);

  int batch = fft_plan->batch;

  size_t buffer_size = batch * in_c_dtype_size * nfft;

  workspace_size = buffer_size * 2;
  workspace_size += (fft_plan->is_input_contiguous) ? 0 : buffer_size;
  workspace_size += (fft_plan->is_output_contiguous) ? 0 : buffer_size;

  size_t twiddles_size = in_c_dtype_size * nfft * 2;
  reservespace_size = sizeof(int) * (FFT_MAXFACTORS)            /* factors */
                      + twiddles_size * 2 + DFT_TABLE_SIZE * 2; /* twiddles */

  fft_plan->workspace_size = workspace_size;
  fft_plan->reservespace_size = reservespace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpAllocateRFFT2D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int _n0, const int _n1) {
  const std::string make_plan_api = "[mluOpAllocateRFFT2D]";
  size_t workspace_size = 0, reservespace_size = 0;
  size_t CPX_TYPE_SIZE = 0, REAL_TYPE_SIZE = 0;

  switch (fft_plan->fft_type) {
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF: {
      REAL_TYPE_SIZE = 2;
    } break;
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      REAL_TYPE_SIZE = 4;
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid r2c 2d fft type.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  CPX_TYPE_SIZE = REAL_TYPE_SIZE * 2;

  int batch = fft_plan->batch;

  size_t buffer_size = batch * sizeof(CPX_TYPE_SIZE) * _n0 * _n1;

  workspace_size = buffer_size * 3;

  // int padded_input_num = batch * n;
  const int trans_dim_num = 3;
  int trans_input_dims[trans_dim_num] = {fft_plan->n[0], fft_plan->n[1],
                                         COMPLEX};
  int trans_output_dims[trans_dim_num] = {fft_plan->n[1], fft_plan->n[0],
                                          COMPLEX};
  int trans_permute[trans_dim_num] = {1, 0, 2};
  size_t trans_workspace_size;
  mluOpStatus_t status = fftGetTransposeWorkspaceSize(
      handle, trans_workspace_size, trans_dim_num, trans_input_dims,
      trans_permute, fft_plan->input_dtype, make_plan_api);

  workspace_size = (workspace_size > trans_workspace_size)
                       ? workspace_size
                       : trans_workspace_size;

  status = fftGetTransposeWorkspaceSize(
      handle, trans_workspace_size, trans_dim_num, trans_output_dims,
      trans_permute, fft_plan->input_dtype, make_plan_api);

  workspace_size = (workspace_size > trans_workspace_size)
                       ? workspace_size
                       : trans_workspace_size;

  size_t twiddles_size = CPX_TYPE_SIZE * _n0;
  size_t twiddles_size_2d = CPX_TYPE_SIZE * _n1;

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    reservespace_size =
        CPX_TYPE_SIZE * _n0 * _n0 + CPX_TYPE_SIZE * _n1 * _n1; /* DFT matrix */
    workspace_size = CPX_TYPE_SIZE * _n1 * _n0 * batch * 6;
  } else if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    reservespace_size = sizeof(int) * (FFT_MAXFACTORS) /* factors */
                        + sizeof(int) * (FFT_MAXFACTORS) + twiddles_size +
                        DFT_TABLE_SIZE + twiddles_size_2d +
                        DFT_TABLE_SIZE; /* twiddles */
  }

  fft_plan->workspace_size = workspace_size;
  fft_plan->reservespace_size = reservespace_size;

  // printf("%ld, %ld\n\n", workspace_size, reservespace_size);
  return status;
}

/**
 * @degroup C2C_PLAN Floating Complex-to-Complex FFT plan
 */

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2C1D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n) {
  // reservespace_addr_ = mlu_runtime_.allocate(reservespace_size_)
  // st = CNAME(openfft_allocate_c2c_plan_1d)(nfft, fin, fout, dir);
  printf("is_batch_contiguous\n");
  fft_plan->is_batch_contiguous = 0;
  // (fft_plan->idist == 1 && fft_plan->odist == 1);

  // std::cout<< "mluOpAllocateC2C1D"<<std::endl;
  mluOpAllocateC2C1D(handle, fft_plan, input_desc, output_desc, n[0]);
  // std::cout<< "mluOpAllocateC2C1D"<<std::endl;
  int is_row_major = !fft_plan->is_batch_contiguous;
  fftTwoStepFactor(fft_plan, n[0], fft_plan->factors, is_row_major);

  switch (fft_plan->fft_type) {
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
      if (!fft_plan->is_batch_contiguous) {
        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                   fft_plan->twiddles_end, fft_plan->factors,
                                   n[0], FFT_FORWARD);
        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles_inv,
                                   fft_plan->twiddles_inv_end,
                                   fft_plan->factors, n[0], FFT_BACKWARD);
      } else {
        fftGenerateTwiddlesColumn<float>(fft_plan, fft_plan->twiddles,
                                         fft_plan->twiddles_end,
                                         fft_plan->factors, n[0], FFT_FORWARD);
        fftGenerateTwiddlesColumn<float>(fft_plan, fft_plan->twiddles_inv,
                                         fft_plan->twiddles_inv_end,
                                         fft_plan->factors, n[0], FFT_BACKWARD);
      }

      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  FFT_FORWARD);
      fftGenerateDftMatrix<float>(fft_plan->idft_matrix, fft_plan->factors,
                                  n[0], FFT_BACKWARD);
      break;
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:

      if (!fft_plan->is_batch_contiguous) {
        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                   fft_plan->twiddles_end, fft_plan->factors,
                                   n[0], FFT_FORWARD);
        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles_inv,
                                   fft_plan->twiddles_inv_end,
                                   fft_plan->factors, n[0], FFT_BACKWARD);
      } else {
        fftGenerateTwiddlesColumn<float>(fft_plan, fft_plan->twiddles,
                                         fft_plan->twiddles_end,
                                         fft_plan->factors, n[0], FFT_FORWARD);
        fftGenerateTwiddlesColumn<float>(fft_plan, fft_plan->twiddles_inv,
                                         fft_plan->twiddles_inv_end,
                                         fft_plan->factors, n[0], FFT_BACKWARD);
      }

      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  FFT_FORWARD);
      fftGenerateDftMatrix<float>(fft_plan->idft_matrix, fft_plan->factors,
                                  n[0], FFT_BACKWARD);
      break;
    default:
      break;
  }

  // if(fft_plan->twiddles == NULL){
  //   std::cout<<" \n\n\n fft_plan->twiddles == NULL \n\n\n"<<std::endl;
  // }
  // CNAME(openfft_generate_twiddles)(st->twiddles, st->factors, nfft, st->dir);

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2R1D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n) {
  mluOpAllocateC2R1D(handle, fft_plan, input_desc, output_desc, n[0]);
  int is_row_major = 1;
  fftTwoStepFactor(fft_plan, n[0], fft_plan->factors, is_row_major);

  switch (fft_plan->fft_type) {
    case CNFFT_COMPLEX_FLOAT2FLOAT:
      fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                 fft_plan->twiddles_end, fft_plan->factors,
                                 n[0], FFT_BACKWARD);
      // for(int i=0; i< 32; i++) {
      //   printf("twiddles: %f\n", ((float*)(fft_plan->twiddles))[i]);
      // }
      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  FFT_BACKWARD);
      break;
    case CNFFT_COMPLEX_HALF2HALF:
      fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                 fft_plan->twiddles_end, fft_plan->factors,
                                 n[0], FFT_BACKWARD);

      fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors, n[0],
                                  FFT_BACKWARD);

      break;
    default:
      break;
  }

  return MLUOP_STATUS_SUCCESS;
}

/**
 * @degroup C2C_PLAN Floating Complex-to-Complex FFT plan
 */

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2C2D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n) {
  fft_plan->is_batch_contiguous =
      (fft_plan->idist == 1 && fft_plan->odist == 1);

  if (fft_plan->is_batch_contiguous && fft_plan->istride == fft_plan->batch &&
      fft_plan->inembed[0] == fft_plan->n[0] &&
      fft_plan->onembed[0] == fft_plan->n[0] &&
      fft_plan->inembed[1] == fft_plan->n[1] &&
      fft_plan->onembed[1] == fft_plan->n[1]) {
    fft_plan->fft_strategy = CNFFT_FUNC_MANY_DIST1_2D;
  } else {
    fft_plan->fft_strategy = CNFFT_FUNC_TWO_LEVEL_STOCKHAM;
  }

  mluOpAllocateC2C2D(handle, fft_plan, input_desc, output_desc);

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    switch (fft_plan->fft_type) {
      case CNFFT_FLOAT2COMPLEX_FLOAT:
      case CNFFT_COMPLEX_FLOAT2FLOAT:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
        fft_plan->dft_matrix = new float[n[1] * n[1] * 2];
        fft_plan->dft_matrix_2d = new float[n[0] * n[0] * 2];
        fft_plan->idft_matrix = new float[n[1] * n[1] * 2];
        fft_plan->idft_matrix_2d = new float[n[0] * n[0] * 2];

        fftGenerateDftMatrixKernelNoPad<float>((float *)fft_plan->dft_matrix,
                                               n[1], FFT_FORWARD);
        fftGenerateDftMatrixKernelNoPad<float>((float *)fft_plan->dft_matrix_2d,
                                               n[0], FFT_FORWARD);
        fftGenerateDftMatrixKernelNoPad<float>((float *)fft_plan->idft_matrix,
                                               n[1], FFT_BACKWARD);
        fftGenerateDftMatrixKernelNoPad<float>(
            (float *)fft_plan->idft_matrix_2d, n[0], FFT_BACKWARD);
        break;
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
        // TODO(zrg): need to copy twiddles to device, and convert to half.

        break;
      default:
        break;
    }
  }

  if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
    fftTwoStepFactor(fft_plan, n[1], fft_plan->factors, 1);
    fftTwoStepFactor(fft_plan, n[0], fft_plan->factors_2d, 0);

    switch (fft_plan->fft_type) {
      case CNFFT_FLOAT2COMPLEX_FLOAT:
      case CNFFT_COMPLEX_FLOAT2FLOAT:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:

        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                   fft_plan->twiddles_end, fft_plan->factors,
                                   n[1], FFT_FORWARD);
        fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors,
                                    n[1], FFT_FORWARD);
        fftGenerateTwiddlesColumn<float>(
            fft_plan, fft_plan->twiddles_2d, fft_plan->twiddles_2d_end,
            fft_plan->factors_2d, n[0], FFT_FORWARD);
        fftGenerateDftMatrix<float>(fft_plan->dft_matrix_2d,
                                    fft_plan->factors_2d, n[0], FFT_FORWARD);

        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles_inv,
                                   fft_plan->twiddles_inv_end,
                                   fft_plan->factors, n[1], FFT_BACKWARD);
        fftGenerateDftMatrix<float>(fft_plan->idft_matrix, fft_plan->factors,
                                    n[1], FFT_BACKWARD);
        fftGenerateTwiddlesColumn<float>(
            fft_plan, fft_plan->twiddles_inv_2d, fft_plan->twiddles_inv_2d_end,
            fft_plan->factors_2d, n[0], FFT_BACKWARD);
        fftGenerateDftMatrix<float>(fft_plan->idft_matrix_2d,
                                    fft_plan->factors_2d, n[0], FFT_BACKWARD);

        break;
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
        // TODO(zrg): need to copy twiddles to device, and convert to half.
        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles,
                                   fft_plan->twiddles_end, fft_plan->factors,
                                   n[1], FFT_FORWARD);
        fftGenerateDftMatrix<float>(fft_plan->dft_matrix, fft_plan->factors,
                                    n[1], FFT_FORWARD);
        fftGenerateTwiddlesColumn<float>(
            fft_plan, fft_plan->twiddles_2d, fft_plan->twiddles_2d_end,
            fft_plan->factors_2d, n[0], FFT_FORWARD);
        fftGenerateDftMatrix<float>(fft_plan->dft_matrix_2d,
                                    fft_plan->factors_2d, n[0], FFT_FORWARD);

        fftGenerateTwiddles<float>(fft_plan, fft_plan->twiddles_inv,
                                   fft_plan->twiddles_inv_end,
                                   fft_plan->factors, n[1], FFT_BACKWARD);
        fftGenerateDftMatrix<float>(fft_plan->idft_matrix, fft_plan->factors,
                                    n[1], FFT_BACKWARD);
        fftGenerateTwiddlesColumn<float>(
            fft_plan, fft_plan->twiddles_inv_2d, fft_plan->twiddles_inv_2d_end,
            fft_plan->factors_2d, n[0], FFT_BACKWARD);
        fftGenerateDftMatrix<float>(fft_plan->idft_matrix_2d,
                                    fft_plan->factors_2d, n[0], FFT_BACKWARD);
        break;
      default:
        break;
    }
  }

  return MLUOP_STATUS_SUCCESS;
}

/**
 * @degroup R2C_PLAN Floating Real-to-Complex FFT plan
 */

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanR2C2D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n) {
  if (fft_plan->idist == 1 && fft_plan->odist == 1 &&
      fft_plan->istride == fft_plan->batch) {
    fft_plan->fft_strategy = CNFFT_FUNC_MANY_DIST1_2D;
  } else {
    fft_plan->fft_strategy = CNFFT_FUNC_TWO_LEVEL_STOCKHAM;
  }

  mluOpAllocateRFFT2D(handle, fft_plan, input_desc, output_desc, n[0], n[1]);

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    switch (fft_plan->fft_type) {
      case CNFFT_FLOAT2COMPLEX_FLOAT:
      case CNFFT_COMPLEX_FLOAT2FLOAT:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
        fft_plan->dft_matrix = new float[n[1] * (n[1] / 2 + 1) * 2];
        fft_plan->dft_matrix_2d = new float[n[0] * n[0] * 2];

        fftGenerateHalfDftMatrixKernelNoPad<float>(
            (float *)fft_plan->dft_matrix, n[1], FFT_FORWARD);
        fftGenerateDftMatrixKernelNoPad<float>((float *)fft_plan->dft_matrix_2d,
                                               n[0], FFT_FORWARD);
        break;
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
        // TODO(zrg): need to copy twiddles to device, and convert to half.

        break;
      default:
        break;
    }
  }

  // if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
  // }

  return MLUOP_STATUS_SUCCESS;
}

/**
 * @degroup R2C_PLAN Floating Real-to-Complex FFT plan
 */

mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanC2R2D(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n) {
  if (fft_plan->idist == 1 && fft_plan->odist == 1 &&
      fft_plan->istride == fft_plan->batch) {
    fft_plan->fft_strategy = CNFFT_FUNC_MANY_DIST1_2D;
  } else {
    fft_plan->fft_strategy = CNFFT_FUNC_TWO_LEVEL_STOCKHAM;
  }

  mluOpAllocateRFFT2D(handle, fft_plan, input_desc, output_desc, n[0], n[1]);

  if (fft_plan->fft_strategy == CNFFT_FUNC_MANY_DIST1_2D) {
    switch (fft_plan->fft_type) {
      case CNFFT_FLOAT2COMPLEX_FLOAT:
      case CNFFT_COMPLEX_FLOAT2FLOAT:
      case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
        fft_plan->dft_matrix = new float[n[1] * (n[1] / 2 + 1) * 2];
        fft_plan->dft_matrix_2d = new float[n[0] * n[0] * 2];

        fftGenerateC2RDftMatrixKernelNoPad<float>((float *)fft_plan->dft_matrix,
                                                  n[1]);
        fftGenerateDftMatrixKernelNoPad<float>((float *)fft_plan->dft_matrix_2d,
                                               n[0], FFT_BACKWARD);
        break;
      case CNFFT_HALF2COMPLEX_HALF:
      case CNFFT_COMPLEX_HALF2HALF:
      case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
        // TODO(zrg): need to copy twiddles to device, and convert to half.

        break;
      default:
        break;
    }
  }

  // if (fft_plan->fft_strategy == CNFFT_FUNC_TWO_LEVEL_STOCKHAM) {
  // }

  return MLUOP_STATUS_SUCCESS;
}

/**
 * This function
 * 1. receives parameters from user;
 * 2. checks the validity of the parameters;
 * 3. picks up the supported circumstances;
 * 4. make decisions which strategy should use.
 */
mluOpStatus_t MLUOP_WIN_API mluOpMakeFFTPlanMany(
    mluOpHandle_t handle, mluOpFFTPlan_t fft_plan,
    mluOpTensorDescriptor_t input_desc, mluOpTensorDescriptor_t output_desc,
    const int rank, const int *n, size_t *reservespace_size,
    size_t *workspace_size) {
  // bad param check
  const std::string make_plan_api = "[mluOpMakeFFTPlanMany]";
  // plan NULL check
  PARAM_CHECK_NE(make_plan_api, handle, NULL);
  if (fft_plan == NULL) {
    LOG(ERROR) << make_plan_api + ": plan is not allocated.";
    return MLUOP_STATUS_NOT_INITIALIZED;
  }
  PARAM_CHECK_NE(make_plan_api, input_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, output_desc, NULL);
  PARAM_CHECK_NE(make_plan_api, n, NULL);
  PARAM_CHECK_NE(make_plan_api, reservespace_size, NULL);
  PARAM_CHECK_NE(make_plan_api, workspace_size, NULL);

  // plan rank can only be 1, 2, 3
  if (rank < 1 || rank > FFT_DIM_MAX) {
    LOG(ERROR) << make_plan_api + ": invalid rank, should be 1, 2 or 3. Now is "
               << rank << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }
  for (auto i = 0; i < rank; i++) {
    if (n[i] <= 0) {
      LOG(ERROR)
          << make_plan_api +
                 ": fourier transform length should be greater than 0. Now n["
          << i << "] is " << n[i] << ".";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }
  fft_plan->rank = rank;
  for (auto i = 0; i < rank; i++) {
    fft_plan->n[i] = n[i];
  }

  // dimension check
  fft_plan->idim = input_desc->dim;
  fft_plan->odim = output_desc->dim;
  fft_plan->inum = mluOpGetTensorElementNum(input_desc);
  fft_plan->onum = mluOpGetTensorElementNum(output_desc);
  PARAM_CHECK_GT(make_plan_api, input_desc->dim, 0);
  PARAM_CHECK_EQ(make_plan_api, fft_plan->idim, fft_plan->odim,
                 ": input and output dimension mismatch.");

  if (!(fft_plan->idim == rank || fft_plan->idim == rank + 1)) {
    LOG(ERROR) << make_plan_api << ": invalid input dimension, should be "
               << rank << " or " << rank + 1 << ". Now is " << fft_plan->idim
               << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  // batch check
  if (fft_plan->idim == rank) {
    fft_plan->batch = 1;
  } else {  // idim == rank + 1
    fft_plan->batch = input_desc->dims[0];
    PARAM_CHECK_EQ(make_plan_api, fft_plan->batch, output_desc->dims[0],
                   ": batch size mismatch.");
  }

  // if (rank == 2) {
  //   fft_plan->batch_2d = input_desc->dims[2];
  //   fft_plan->batch = input_desc->dims[1];
  // }

  // The FFT Struct is designed after cufftXtMakePlanMany.
  // An element of coordinates [z, y, x] in signal number b in the batch will
  // be associated with the following addresses in the memory
  // 1-D:
  //   input[b * idist + x * istride]
  //   output[b * odist + x * ostride]
  // 2-D:
  //   input[b * idist + (x * inembed[1] + y) * istride]
  //   output[b * odist + (x * onembed[1] + y) * istride]
  // 3-D:
  //   input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
  //   output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
  // Thus, cufft and fftw advanced data layout is a subset of mluOp advanced
  // data layout with tensor dim strides. 2-D and 3-D should pay attention.
  // stride check, if an in-place fft is adopted check, `istride` should be
  // equal to `ostride`.
  fft_plan->istride = input_desc->strides[fft_plan->idim - 1];
  fft_plan->ostride = output_desc->strides[fft_plan->odim - 1];

  PARAM_CHECK_GE(make_plan_api, fft_plan->istride, 0,
                 ": input stride should be greater than or equal to 0.");
  PARAM_CHECK_GE(make_plan_api, fft_plan->ostride, 0,
                 ": output stride should be greater than or equal to 0.");

  for (auto i = 0; i < fft_plan->rank; i++) {
    fft_plan->inembed[i] = input_desc->dims[fft_plan->idim - rank + i];
    fft_plan->onembed[i] = output_desc->dims[fft_plan->odim - rank + i];
  }
  if (fft_plan->idim == rank + 1) {
    fft_plan->idist = input_desc->strides[0];
    fft_plan->odist = output_desc->strides[0];
  } else {  // batch == 1
    fft_plan->idist = mluOpGetTensorElementNum(input_desc) / fft_plan->batch;
    fft_plan->odist = mluOpGetTensorElementNum(output_desc) / fft_plan->batch;
  }
  fft_plan->is_input_contiguous = !mluop::ifNeedTensorStrideProcess(input_desc);
  fft_plan->is_output_contiguous =
      !mluop::ifNeedTensorStrideProcess(output_desc);

  // dtype check
  mluOpDataType_t input_dtype = input_desc->dtype;
  mluOpDataType_t output_dtype = output_desc->dtype;
  const mluOpDataType_t f_c_dtype = MLUOP_DTYPE_COMPLEX_FLOAT;
  const mluOpDataType_t f_r_dtype = MLUOP_DTYPE_FLOAT;
  const mluOpDataType_t hf_c_dtype = MLUOP_DTYPE_COMPLEX_HALF;
  const mluOpDataType_t hf_r_dtype = MLUOP_DTYPE_HALF;
  if (input_dtype == hf_r_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2COMPLEX_HALF;
  } else if (input_dtype == hf_c_dtype && output_dtype == hf_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_HALF2HALF;
  } else if (input_dtype == f_r_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_c_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT;
  } else if (input_dtype == f_c_dtype && output_dtype == f_r_dtype) {
    fft_plan->fft_type = CNFFT_COMPLEX_FLOAT2FLOAT;
  } else {
    LOG(ERROR) << make_plan_api
               << ": invalid data type combination. Now input data type is "
               << mluOpGetNameOfDataType(input_dtype)
               << ", and output data type is "
               << mluOpGetNameOfDataType(output_dtype) << ".";
    return MLUOP_STATUS_BAD_PARAM;
  }

  fft_plan->input_dtype = input_desc->dtype;
  fft_plan->output_dtype = output_desc->dtype;
  fft_plan->execution_dtype = input_desc->onchip_dtype;

  VLOG(5) << "input data type: "
          << mluOpGetNameOfDataType(fft_plan->input_dtype);
  VLOG(5) << "output data type: "
          << mluOpGetNameOfDataType(fft_plan->output_dtype);
  VLOG(5) << "execution data type: "
          << mluOpGetNameOfDataType(fft_plan->execution_dtype);

  // fft length check
  for (auto i = 0; i < fft_plan->rank - 1; i++) {  // except the last dim
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->inembed[i],
        ": the signal lengths of fft and input mismatch in dimention ", i, ".");
    PARAM_CHECK_EQ(
        make_plan_api, n[i], fft_plan->onembed[i],
        ": the signal lengths of fft and output mismatch in dimension ", i,
        ".");
  }
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1] / 2 + 1,
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in R2C.");
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2C.");
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      PARAM_CHECK_EQ(make_plan_api, fft_plan->n[rank - 1],
                     fft_plan->onembed[rank - 1],
                     ": the signal lengths of fft and output last dimention "
                     "mismatch in C2R.");
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid fft type.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  mluOpDataType_t execution_dtype = fft_plan->execution_dtype;
  switch (fft_plan->fft_type) {
    // half
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_HALF2HALF: {
      if (supportFloatConv(handle)) {
        if (!(execution_dtype == hf_r_dtype ||
              execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      } else {
        if (!(execution_dtype == MLUOP_DTYPE_INT16)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    // float
    case CNFFT_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (supportFloatConv(handle)) {
        if (!(execution_dtype == f_r_dtype ||
              execution_dtype == MLUOP_DTYPE_INT31)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      } else {
        if (!(execution_dtype == MLUOP_DTYPE_INT31)) {
          LOG(ERROR) << make_plan_api << ": invalid execution dtype "
                     << mluOpGetNameOfDataType(fft_plan->execution_dtype)
                     << ".";
          return MLUOP_STATUS_BAD_PARAM;
        }
      }
    }; break;
    default: {
      LOG(ERROR) << make_plan_api << ": invalid execution dtype.";
      return MLUOP_STATUS_BAD_PARAM;
    }
  }

  // unsupported param
  if (fft_plan->rank != 1 && fft_plan->rank != 2) {
    LOG(ERROR) << make_plan_api
               << ": 3-dimensional FFT are not supported currently.";
    return MLUOP_STATUS_NOT_SUPPORTED;
  }

  if (fft_plan->fft_type == CNFFT_HALF2COMPLEX_HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2HALF ||
      fft_plan->fft_type == CNFFT_COMPLEX_HALF2COMPLEX_HALF) {
    if ((n[0] & (n[0] - 1)) != 0) {
      LOG(ERROR) << make_plan_api
                 << ": the signal lengths of half-precision FFT are"
                 << " restriced to power of two only, but now is " << n[0]
                 << ".";
      return MLUOP_STATUS_NOT_SUPPORTED;
    }
  }

  // create input and output descriptor for gen_case
  // because mluOpExecFFT don't have input and output descriptor
  mluOpTensorDescriptor_t fft_input_desc, fft_output_desc;
  INTERNAL_CHECK(make_plan_api, mluOpCreateTensorDescriptor(&fft_input_desc) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api, mluOpCreateTensorDescriptor(&fft_output_desc) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_input_desc, input_desc->layout, input_desc->dtype,
                     input_desc->dim, input_desc->dims,
                     input_desc->strides) == MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api, mluOpSetTensorDescriptorOnchipDataType(
                                    fft_input_desc, input_desc->onchip_dtype) ==
                                    MLUOP_STATUS_SUCCESS);
  INTERNAL_CHECK(make_plan_api,
                 mluOpSetTensorDescriptorEx_v2(
                     fft_output_desc, output_desc->layout, output_desc->dtype,
                     output_desc->dim, output_desc->dims,
                     output_desc->strides) == MLUOP_STATUS_SUCCESS);
  fft_plan->input_desc = fft_input_desc;
  fft_plan->output_desc = fft_output_desc;

  VLOG(5) << "into make FFT1d Policy";
  int n0 = n[0];
  // int prime = 0;
  fft_plan->prime = 0;
  int r = 0;
  while (n0 > 1) {
    for (r = 64; r > 1; r--) {
      if (n0 % r == 0) {
        n0 /= r;
        break;
      }
    }
    if (r == 1) {
      fft_plan->prime = n0;
      break;
    }
  }

  /*
   * decision part
   */
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        VLOG(5) << "into make RFFT1d Policy";
        status = makeRFFT1dPolicy(handle, fft_plan);
      } else if (rank == 2) {
        status = mluOpMakeFFTPlanR2C2D(handle, fft_plan, input_desc,
                                       output_desc, rank, n);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (rank == 1) {
        if (fft_plan->prime == 0) {
          status = mluOpMakeFFTPlanC2R1D(handle, fft_plan, input_desc,
                                         output_desc, rank, n);

        } else {
          VLOG(5) << "into make IRFFT1d Policy";
          status = makeIRFFT1dPolicy(handle, fft_plan);
        }
      } else if (rank == 2) {
        status = mluOpMakeFFTPlanC2R2D(handle, fft_plan, input_desc,
                                       output_desc, rank, n);
      }
    }; break;
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (rank == 1) {
        if (fft_plan->prime == 0) {
          printf("mluOpMakeFFTPlanC2C1D\n");
          status = mluOpMakeFFTPlanC2C1D(handle, fft_plan, input_desc,
                                         output_desc, rank, n);

        } else {
          printf("makeFFT1dPolicy\n");

          status = makeFFT1dPolicy(handle, fft_plan);
        }

        // C2C 1D

      } else if (rank == 2) {
        VLOG(5) << "into make FFT2d Policy";

        // status = makeFFT1dPolicy(handle, fft_plan);
        // C2C 1D
        status = mluOpMakeFFTPlanC2C2D(handle, fft_plan, input_desc,
                                       output_desc, rank, n);
      }
    }; break;
  }

  if (status != MLUOP_STATUS_SUCCESS) {
    return status;
  }

  *reservespace_size = fft_plan->reservespace_size;
  *workspace_size = fft_plan->workspace_size;

  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpDestroyFFTPlan(mluOpFFTPlan_t fft_plan) {
  const std::string destroy_api = "[mluOpDestroyFFTPlan]";
  PARAM_CHECK_NE("[mluOpDestroyFFTPlan]", fft_plan, NULL);
  if (fft_plan->input_desc != NULL) {
    INTERNAL_CHECK(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->input_desc) ==
                       MLUOP_STATUS_SUCCESS);
  }
  if (fft_plan->output_desc != NULL) {
    INTERNAL_CHECK(destroy_api,
                   mluOpDestroyTensorDescriptor(fft_plan->output_desc) ==
                       MLUOP_STATUS_SUCCESS);
  }
  if (fft_plan->factors != NULL) {
    delete fft_plan->factors;
  }

  if (fft_plan->twiddles != NULL) {
    delete (char *)fft_plan->twiddles;
  }

  if (fft_plan->twiddles_inv != NULL) {
    delete (char *)fft_plan->twiddles_inv;
  }

  if (fft_plan->factors_2d != NULL) {
    delete fft_plan->factors_2d;
  }

  if (fft_plan->twiddles_2d != NULL) {
    delete (char *)fft_plan->twiddles_2d;
  }

  if (fft_plan->twiddles_inv_2d != NULL) {
    delete (char *)fft_plan->twiddles_inv_2d;
  }

  if (fft_plan->dft_matrix != NULL) {
    delete (char *)fft_plan->dft_matrix;
  }

  if (fft_plan->dft_matrix_2d != NULL) {
    delete (char *)fft_plan->dft_matrix_2d;
  }

  if (fft_plan->idft_matrix != NULL) {
    delete (char *)fft_plan->idft_matrix;
  }

  if (fft_plan->idft_matrix_2d != NULL) {
    delete (char *)fft_plan->idft_matrix_2d;
  }

  delete fft_plan;
  return MLUOP_STATUS_SUCCESS;
}

mluOpStatus_t MLUOP_WIN_API mluOpSetFFTReserveArea(mluOpHandle_t handle,
                                                   mluOpFFTPlan_t fft_plan,
                                                   void *reservespace) {
  const std::string api = "[mluOpSetReserveArea]";
  PARAM_CHECK_NE(api, handle, NULL);
  PARAM_CHECK_NE(api, fft_plan, NULL);
  PARAM_CHECK_NE(api, reservespace, NULL);
  fft_plan->reservespace_addr = reservespace;
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        status = setRFFT1dReserveArea(handle, fft_plan, api);
      } else if (fft_plan->rank == 2) {
        // status = setFFT1dReserveArea(handle, fft_plan, api);
        status = setFFT2dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if (fft_plan->rank == 1) {
        if (fft_plan->prime) {
          status = setFFT1dReserveArea(handle, fft_plan, api);
        } else {
          status = setFFT1dReserveArea_v2(handle, fft_plan, api);
        }

      } else if (fft_plan->rank == 2) {
        // status = setFFT1dReserveArea(handle, fft_plan, api);
        status = setFFT2dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (fft_plan->rank == 1) {
        if (fft_plan->prime) {
          status = setIRFFT1dReserveArea(handle, fft_plan, api);
        } else {
          status = setIRFFT1dReserveArea_v2(handle, fft_plan, api);
        }
      } else if (fft_plan->rank == 2) {
        // status = setFFT1dReserveArea(handle, fft_plan, api);
        status = setFFT2dReserveArea(handle, fft_plan, api);
      } else {
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }
  return status;
}

mluOpStatus_t MLUOP_WIN_API mluOpExecFFT(
    mluOpHandle_t handle, const mluOpFFTPlan_t fft_plan, const void *input,
    const float scale_factor, void *workspace, void *output, int direction) {
  const std::string exec_api = "[mluOpExecFFT]";
  PARAM_CHECK_NE(exec_api, handle, NULL);
  PARAM_CHECK_NE(exec_api, fft_plan, NULL);
  VLOG(5) << "input contiguous ? " << fft_plan->is_input_contiguous;
  VLOG(5) << "output contiguous ? " << fft_plan->is_output_contiguous;

  if (fft_plan->batch == 0) {
    VLOG(5) << "[mluOpExecFFT] Skip zero element tensor";
    return MLUOP_STATUS_SUCCESS;
  }
  // generate mluOpFFTExec prototxt start!
  {
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->input_desc),
                     LARGE_TENSOR_NUM, "");
    TENSOR_NUM_CHECK("[mluOpFft]",
                     mluOpGetTensorElementNum(fft_plan->output_desc),
                     LARGE_TENSOR_NUM, "");
  }

  if (MLUOP_GEN_CASE_ON_NEW) {
    GEN_CASE_START("fft", "FFT");
    GEN_CASE_HANDLE(handle);
    GEN_CASE_DATA(true, "input", input, fft_plan->input_desc, 1, 0);
    GEN_CASE_DATA(false, "output", output, fft_plan->output_desc, 0, 0);
    GEN_CASE_OP_PARAM_SINGLE(0, "fft", "rank", fft_plan->rank);
    GEN_CASE_OP_PARAM_ARRAY(1, "fft", "n", fft_plan->n, fft_plan->rank);
    GEN_CASE_OP_PARAM_SINGLE(1, "fft", "direction", direction);
    GEN_CASE_OP_PARAM_SINGLE(2, "fft", "scale_factor", scale_factor);
    GEN_CASE_TEST_PARAM_NEW(true, true, false, 0.003, 0.003, 0);
  }

  if (fft_plan->workspace_size > 0) {
    PARAM_CHECK_NE(exec_api, workspace, NULL);
  }
  if (fft_plan->inum > 0) {
    PARAM_CHECK_NE(exec_api, input, NULL);
  }
  PARAM_CHECK_NE(exec_api, output, NULL);
  mluOpStatus_t status = MLUOP_STATUS_SUCCESS;

  bool is_in_place = (input == output);
  VLOG(5) << exec_api << ": in place ? " << is_in_place;
  switch (fft_plan->fft_type) {
    // r2c
    case CNFFT_HALF2COMPLEX_HALF:
    case CNFFT_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < (fft_plan->odist * 2)) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "real-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execRFFT1d(handle, fft_plan, input, scale_factor, workspace,
                            output);
      } else if (fft_plan->rank == 2) {
        status = execFFT2d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2c
    case CNFFT_COMPLEX_HALF2COMPLEX_HALF:
    case CNFFT_COMPLEX_FLOAT2COMPLEX_FLOAT: {
      if ((fft_plan->idist < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-complex fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        status = execFFT1d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 2) {
        status = execFFT2d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
    // c2r
    case CNFFT_COMPLEX_HALF2HALF:
    case CNFFT_COMPLEX_FLOAT2FLOAT: {
      if (((fft_plan->idist * 2) < fft_plan->odist) && is_in_place) {
        LOG(ERROR)
            << exec_api
            << ": output overwritten may occur during an in-place "
               "complex-to-real fft "
               "execution, input needs to be slightly padding. Please refer to "
               "mluOpExecFFT document for detail.";
        status = MLUOP_STATUS_BAD_PARAM;
      }
      if (fft_plan->rank == 1) {
        printf("execIRFFT1d_v2\n");
        status = execIRFFT1d_v2(handle, fft_plan, input, scale_factor,
                                workspace, output);
      } else if (fft_plan->rank == 2) {
        status = execFFT2d(handle, fft_plan, input, scale_factor, workspace,
                           output, direction);
      } else if (fft_plan->rank == 3) {
        // TODO(who)
        status = MLUOP_STATUS_NOT_SUPPORTED;
      }
    }; break;
  }

  GEN_CASE_END();
  return status;
}
