/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino
*/ 

#include "pulp_train_utils_fp32.h"
#include "pulp_im2col_fp32.h"

/**
 * @brief IM2ROW with padding and stride
 * 
 * @param im2col_args 
 */
void pulp_im2row_fp32(void * im2col_args){

  // unpack args
  struct im2col_args * args = (struct im2col_args *) im2col_args;
  struct blob * input = args->input;
  struct blob * coeff = args->c;
  struct blob * output = args->output;

  float * i2c_buf = args->pBuffer;

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;
  // Flag to activate the DMA version of the IM2COL
  uint8_t USE_DMA = args->USE_DMA;
  uint8_t HWC = args->HWC;

  // activations dimensions, w/o padding
  uint32_t Win = input->W;
  uint32_t Hin = input->H;
  uint32_t Cin = input->C;
  // kernel dimensions
  uint32_t Wk = coeff->W;
  uint32_t Hk = coeff->H;
  // input channels size
  uint32_t Wo = output->W;
  uint32_t Ho = output->H;
  uint32_t Co = output->C;

  // Set up internal variables (simpify external interface)
  //Ho = Hin - Hk + 1;
  //Wo = Win - Wk + 1;

  // Set up im2col variables for padding and stride
  uint32_t Htot=0, Wtot=0;
  Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
  Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;
  //printf("(im2row) input = 0x%x, input->W = %d\n", input, input->W);
  //printf("(im2row) [Cin, Hin , Win] = [%d, %d, %d], [Co, Htot, Wtot] = [%d, %d, %d], [Hk, Wk] = [%d, %d]\n", Cin, Hin, Win, Co, Htot, Wtot, Wk, Hk);

  // Partial im2row variables
  uint32_t ht_start = args->htile_start;
  uint32_t ht_stop = args->htile_end;
  uint32_t wt_start = args->wtile_start;
  uint32_t wt_stop = args->wtile_end;
  // Check bindings
  //printf("\n(im2row) ht = [%d, %d], wt = [%d, %d]", ht_start, ht_stop, wt_start, wt_stop);
  if (ht_start < 0) printf("\nInvalid partial im2col boundary on the upper side!! (have ht_start = %d < 0)\n", ht_start);
  if (ht_stop > Htot) printf("\nInvalid partial im2col boundary on the lower side!! (have ht_stop = %d > %d)\n", ht_stop, Htot);
  if (wt_start < 0) printf("\nInvalid partial im2col boundary on the left!! (have wt_start = %d < 0)\n", wt_start);
  if (wt_stop > Wtot) printf("\nInvalid partial im2col boundary on the right!! (have wt_stop = %d > %d)\n", wt_stop, Wtot);

  #if NUM_CORES > 1
  // Definitions for parallelism
  uint32_t blockSize=0, start=0, stop=0;
  if (HWC == 0 && mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
  }
  else if (HWC == 0 && mod == 1) {
    blockSize = (Co+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Co ? Co : start+blockSize;
  }
  else if (HWC == 1 && mod == 0) {
    blockSize = (Htot+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Htot ? Htot : start+blockSize;
  }
  else if (HWC == 1 && mod == 1) {
    blockSize = (Hin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Hin ? Hin : start+blockSize;
  }
  #else
  uint32_t start=0, stop=0; 
  if (HWC == 0 && mod == 0) {
    start = 0;
    stop = Cin;    
  }
  else if (HWC == 0 && mod == 1) {
    start = 0;
    stop = Co;
  }
  else if (HWC == 1 && mod == 0) {
    start = 0;
    stop = Htot;
  }
  else if (HWC == 1 && mod == 1) {
    start = 0;
    stop = Hin;
  }
  #endif

  /**
   * USE CHW FORMAT (ADJACENT ELEMENTS ARE ROW ELEMENTS OF THE INPUT OR OUTPUT MATRIX)
   */
  if (HWC == 0) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;
        // Partial im2row indices
        uint32_t Wout_diff = wt_stop - wt_start;
        uint32_t pho = 0; uint32_t pwo = 0;

        if (padding == 0) {
          //for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (uint32_t ho=ht_start; ho<ht_stop/*Ho+2*pad*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Hk*Wk;
                uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
                uint32_t partial_segment_idx = pwo*Hk*Wk*Cin; // + pho*Hk*Wk*Cin*(Wout_diff);
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr) + (ho*Hstr)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;

                    //printf("(im2row, pho, pwo = [%d, %d]) i2c_buf[%d = %d + %d + %d] = %f\n", pho, pwo, kernel_idx+partial_segment_idx+i2c_inner_idx, 
                    //          kernel_idx, partial_segment_idx, i2c_inner_idx, input->data[receptive_field_idx+in_inner_idx]);
                    i2c_buf[kernel_idx+partial_segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                    //printf("(ho=%d, wo=%d) i2c_buf[%d] = %f, indata[%d] = %f\n", ho, wo, kernel_idx+segment_idx+i2c_inner_idx, 
                    //          i2c_buf[kernel_idx+segment_idx+i2c_inner_idx], receptive_field_idx+in_inner_idx, input->data[receptive_field_idx+in_inner_idx]);
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }          
        }

        else {
          //for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (uint32_t ho=ht_start; ho<ht_stop/*Ho+2*pad*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Hk*Wk;
                //uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
                uint32_t segment_idx = pwo*Hk*Wk*Cin + pho*Hk*Wk*Cin*(Wtot);
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk + hk*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;
                    // Padding condition
                    uint32_t w_pad_cond = wk + wo*Wstr;
                    uint32_t h_pad_cond = hk + ho*Hstr;

                    if ((padding>0)&&((h_pad_cond<Upad) || (w_pad_cond<Lpad) || (h_pad_cond>Ho+(Hk)-Dpad) || (w_pad_cond>Wo+(Wk)-Rpad))) {
                      // Padding
                      i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                      //printf("(pad) i2c_buf[%d]=%f                        kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], kernel_idx, segment_idx, ho);
                    }
                    else {
                      // Fill IM2COL buffer
                      i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                      //printf("(i2c) i2c_buf[%d]=%f (indata=%f)      kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], input->data[receptive_field_idx], kernel_idx, segment_idx, ho);
                    }
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }
        }

      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        // Partial im2row indices
        uint32_t phi = 0; uint32_t pwi = 0;        
        
        //for (uint32_t hi=0; hi<Hin; hi++) {
        for (uint32_t hi=ht_start; hi<ht_stop; hi++) {
          //for (uint32_t wi=0; wi<Win; wi++) {
          for (uint32_t wi=wt_start; wi<wt_stop; wi++) {
            for (uint32_t co=start; co<stop; co++) {
              // IM2COL buffer coordinates
              uint32_t kernel_idx = co*Hk*Wk;
              //uint32_t segment_idx = wi*Hk*Wk*Co + hi*Hk*Wk*Co*Win;
              uint32_t segment_idx = pwi*Hk*Wk*Co + phi*Hk*Wk*Co*Win;
              // Output grad tensor coordinates
              int ho_rf = hi - (Hk-1);
              int wo_rf = wi - (Wk-1);
              int receptive_field_idx = wo_rf + ho_rf*Wox + co*Hox*Wox;

              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  // IM2COl buffer coordinates
                  uint32_t i2c_inner_idx = wk + hk*Wk;
                  // Output grad tensor coordinates
                  uint32_t out_inner_idx = wk + hk*Wox;
                  // Padding condition
                  int w_pad_cond = wk + wo_rf;
                  int h_pad_cond = hk + ho_rf;

                  if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                    // Padding
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                  }
                  else {
                    // Fill IM2COL buffer
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
                  }
                }
              }
            }
            pwi++;
            if (pwi == wt_stop) pwi = 0;
          }
          phi++;
        }
      }
    }

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32: 414] Invalid USE_DMA parameter (not 0 or 1)\n");
    }
  }

  /**
   * USE HWC FORMAT (ADJACENT ELEMENTS ARE CHANNEL ELEMENTS IN THE INPUT OR OUTPUT MATRIX)
   */
  else if (HWC == 1) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;
        // Partial im2row indices
        uint32_t pho = 0; uint32_t pwo = 0;
        // Recompute parallel indices per core
        int blockSize = ((ht_stop-ht_start+1)+NUM_CORES-1) / NUM_CORES;
        start = pi_core_id()*blockSize > ht_start ? ht_start : pi_core_id()*blockSize;
        stop = start+blockSize > ht_stop ? ht_stop : start+blockSize;

        if (padding == 0) {

          for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wtot*/; wo++) {
              // Im2Col indices
              //uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              uint32_t segment_idx = pwo*Hk*Wk*Cin + pho*Hk*Wk*Cin*(Wtot);
              // Input activation indices
              uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  for (uint32_t ci=0; ci<Cin; ci++) {
                    // Im2Col indices
                    uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
                    // Input activation indices                    
                    uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
                    // Fill im2col buffer
                    i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }

        }
        else {

          printf("\n[pulp_im2col_fp32.c:] Padding not implemented for HWC im2col without DMA!\n");

          // for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
          //   for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
          //     // Im2Col indices
          //     uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
          //     // Input activation indices
          //     uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
          //     for (uint32_t hk=0; hk<Hk; hk++) {
          //       for (uint32_t wk=0; wk<Wk; wk++) {
          //         for (uint32_t ci=0; ci<Cin; ci++) {
          //           // Im2Col indices
          //           uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
          //           // Input activation indices                    
          //           uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
          //           // Fill im2col buffer
          //           i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
          //         }
          //       }
          //     }
          //   }
          // }

        }
      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        // Partial im2row indices
        uint32_t phi = 0; uint32_t pwi = 0;  
        // Recompute start and stop indices per core
        int blockSize = ((ht_stop-ht_start+1)+NUM_CORES-1) / NUM_CORES;
        start = pi_core_id()*blockSize > ht_start ? ht_start : pi_core_id()*blockSize;
        stop = start+blockSize > ht_stop ? ht_stop : start+blockSize;

        for (uint32_t hi=start/*0*/; hi<stop/*Hin*/; hi++) {
          //for (uint32_t wi=0; wi<Win; wi++) {
          for (uint32_t wi=wt_start; wi<wt_stop; wi++) {
            // Padding variables
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);

            for (uint32_t hk=0; hk<Hk; hk++) {
              for (uint32_t wk=0; wk<Wk; wk++) {
                // Padding conditions
                int w_pad_cond = wk + wo_rf;
                int h_pad_cond = hk + ho_rf;    

                // Set padding loop
                if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                  for (uint32_t co=0; co<Co; co++) {
                    // IM2COL buffer coordinates
                    //uint32_t segment_idx = wi*Co*Hk*Wk + hi*Co*Hk*Wk*Win;
                    uint32_t segment_idx = pwi*Co*Hk*Wk + phi*Co*Hk*Wk*Win;
                    uint32_t kernel_idx = wk*Co + hk*Co*Wk;
                    uint32_t i2c_inner_idx = co;  

                    // Fill with zeroes  
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0.0f;             
                  }
                }
                else {
                  // Non-padded iteration
                  for (uint32_t co=0; co<Co; co++) {
                    // OutDiff coordinates
                    int receptive_field_idx = (wo_rf+wk)*Co + (ho_rf+hk)*Co*Wox;
                    uint32_t out_inner_idx = co;

                    // IM2COL buffer coordinates
                    uint32_t segment_idx = pwi*Co*Hk*Wk + phi*Co*Hk*Wk*Win;
                    uint32_t kernel_idx = wk*Co + hk*Co*Wk;
                    uint32_t i2c_inner_idx = co;

                    // Fill IM2COL buffer
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
                  }
                                           
                }
              }
            }
            pwi++;
            if (pwi == wt_stop) pwi = 0;
          }
          phi++;
        }
      }
    }

    /**
     * IM2COL FROM L2 DATA TO L1 IM2COL_BUFFER
     */
    else if (USE_DMA == 1) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;

        if (padding == 0) {

          for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
            for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
              // Im2Col indices
              uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
              // Input activation indices
              uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;

              // DMA Copy structures
              pi_cl_dma_copy_2d_t dma_i2cfw;

              // Load first data into L1A
              dma_i2cfw.dir = PI_CL_DMA_DIR_EXT2LOC;
              dma_i2cfw.merge = 0;
              dma_i2cfw.stride = 4*Cin*Win;
              dma_i2cfw.length = 4*Cin*Wk;
              dma_i2cfw.size = 4*Hk*Wk*Cin;
              dma_i2cfw.id = pi_core_id();
              dma_i2cfw.ext = (uint32_t) (input->data + input_idx);
              dma_i2cfw.loc = (uint32_t) &i2c_buf[segment_idx];
              pi_cl_dma_memcpy_2d(&dma_i2cfw);  

              pi_cl_dma_wait(&dma_i2cfw);  
            }
          }

        }
        else {

          printf("\n[pulp_im2col_fp32.c:] Padding not implemented for HWC im2col with DMA!\n");

          // for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
          //   for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
          //     // Im2Col indices
          //     uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
          //     // Input activation indices
          //     uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;

          //     // DMA Copy structures
          //     pi_cl_dma_copy_2d_t dma_i2cfw;

          //     // Load first data into L1A
          //     dma_i2cfw.dir = PI_CL_DMA_DIR_EXT2LOC;
          //     dma_i2cfw.merge = 0;
          //     dma_i2cfw.stride = 4*Cin*Win;
          //     dma_i2cfw.length = 4*Cin*Wk;
          //     dma_i2cfw.size = 4*Hk*Wk*Cin;
          //     dma_i2cfw.id = pi_core_id();
          //     dma_i2cfw.ext = (uint32_t) (input->data + input_idx);
          //     dma_i2cfw.loc = (uint32_t) &i2c_buf[segment_idx];
          //     pi_cl_dma_memcpy_2d(&dma_i2cfw);  

          //     pi_cl_dma_wait(&dma_i2cfw);  
          //   }
          // }
                    
        }
      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        
        printf("\n[pulp_im2col_fp32:] HWC Im2Col for IN GRAD not implemented!!\n");

        for (uint32_t hi=start; hi<stop; hi++) {
          for (uint32_t wi=0; wi<Win; wi++) {
            for (uint32_t hk=0; hk<Hk; hk++) {
              for (uint32_t wk=0; wk<Wk; wk++) {
                for (uint32_t co=0; co<Co; co++) {
                                                        
                }
              }
            }
          }
        }
      }
    }

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32: 414] Invalid USE_DMA parameter (not 0 or 1)\n");
    }
    
  }

  // ERROR SIGNAL
  else {
    printf("[pulp_im2col_fp32:] Invalid HWC parameter (not 0 or 1)\n");
  }

}







/**
 * @brief IM2COL with padding and stride
 * 
 * @param im2col_args 
 */
void pulp_im2col_fp32(void * im2col_args){

  // unpack args
  struct im2col_args * args = (struct im2col_args *) im2col_args;
  struct blob * input = args->input;
  struct blob * coeff = args->c;
  struct blob * output = args->output;

  float * i2c_buf = args->pBuffer;

  uint8_t Lpad = args->Lpad;
  uint8_t Rpad = args->Rpad;
  uint8_t Upad = args->Upad;
  uint8_t Dpad = args->Dpad;
  uint8_t mod = args->mod;
  uint8_t Hstr = args->stride_h;
  uint8_t Wstr = args->stride_w;
  // Flag to activate the DMA version of the IM2COL
  uint8_t USE_DMA = args->USE_DMA;
  uint8_t HWC = args->HWC;

  // activations dimensions, w/o padding
  uint32_t Win = input->W;
  uint32_t Hin = input->H;
  uint32_t Cin = input->C;
  // kernel dimensions
  uint32_t Wk = coeff->W;
  uint32_t Hk = coeff->H;
  // input channels size
  uint32_t Wo = output->W;
  uint32_t Ho = output->H;
  uint32_t Co = output->C;

  // Set up internal variables (simpify external interface)
  Ho = Hin - Hk + 1;
  Wo = Win - Wk + 1;

  // Set up im2col variables for padding and stride
  uint32_t Htot=0, Wtot=0;
  Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
  Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

  // Partial im2row variables
  uint32_t ht_start = args->htile_start;
  uint32_t ht_stop = args->htile_end;
  uint32_t wt_start = args->wtile_start;
  uint32_t wt_stop = args->wtile_end;
  // Check bindings
  if (ht_start < 0) printf("Invalid partial im2col boundary on the upper side!!\n");
  if (ht_stop > Htot) printf("Invalid partial im2col boundary on the lower side!!\n");
  if (wt_start < 0) printf("Invalid partial im2col boundary on the left!!\n");
  if (wt_stop > Wtot) printf("Invalid partial im2col boundary on the right!!\n");

  #if NUM_CORES > 1
  // Definitions for parallelism
  uint32_t blockSize=0, start=0, stop=0;
  if (HWC == 0 && mod == 0) {
    blockSize = (Cin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Cin ? Cin : start+blockSize;
  }
  else if (HWC == 0 && mod == 1) {
    blockSize = (Co+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Co ? Co : start+blockSize;
  }
  else if (HWC == 1 && mod == 0) {
    blockSize = (Htot+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Htot ? Htot : start+blockSize;
  }
  else if (HWC == 1 && mod == 1) {
    blockSize = (Hin+NUM_CORES-1) / NUM_CORES;
    start = pi_core_id()*blockSize;
    stop = start+blockSize > Hin ? Hin : start+blockSize;
  }
  #else
  uint32_t start=0, stop=0; 
  if (HWC == 0 && mod == 0) {
    start = 0;
    stop = Cin;    
  }
  else if (HWC == 0 && mod == 1) {
    start = 0;
    stop = Co;
  }
  else if (HWC == 1 && mod == 0) {
    start = 0;
    stop = Htot;
  }
  else if (HWC == 1 && mod == 1) {
    start = 0;
    stop = Hin;
  }
  #endif

  /**
   * USE CHW FORMAT (ADJACENT ELEMENTS ARE ROW ELEMENTS OF THE INPUT OR OUTPUT MATRIX)
   */
  if (HWC == 0) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;
        // Partial im2row indices
        uint32_t pho = 0; uint32_t pwo = 0;

        if (padding == 0) {
          //for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (uint32_t ho=ht_start; ho<ht_stop/*Ho+2*pad*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Htot*Wtot*Hk*Wk;
                //uint32_t segment_idx = wo + ho*Wtot;
                uint32_t segment_idx = pwo + pho*Wtot;
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk*Htot*Wtot + hk*Htot*Wtot*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;

                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                    printf("(ho=%d, wo=%d) \ti2c_buf[%d] = %f, \tindata[%d] = %f\n", ho, wo, kernel_idx+segment_idx+i2c_inner_idx, 
                              i2c_buf[kernel_idx+segment_idx+i2c_inner_idx], receptive_field_idx+in_inner_idx, input->data[receptive_field_idx+in_inner_idx]);
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }          
        }

        else {
          //for (uint32_t ho=0; ho<Htot/*Ho+2*pad*/; ho++) {
          for (uint32_t ho=ht_start; ho<ht_stop/*Ho+2*pad*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wo+2*pad*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wo+2*pad*/; wo++) {
              for (uint32_t ci=start; ci<stop; ci++) {
                // IM2COL buffer coordinates
                uint32_t kernel_idx = ci*Htot*Wtot*Hk*Wk;
                //uint32_t segment_idx = wo + ho*Wtot;
                uint32_t segment_idx = pwo + pho*Wtot;
                // Input tensor coordinates
                uint32_t receptive_field_idx = (wo*Wstr-Lpad) + (ho*Hstr-Upad)*Win + ci*Hin*Win;
                for (uint32_t hk=0; hk<Hk; hk++) {
                  for (uint32_t wk=0; wk<Wk; wk++) {
                    // IM2COl buffer coordinate update
                    uint32_t i2c_inner_idx = wk*Htot*Wtot + hk*Htot*Wtot*Wk;
                    // Input tensor coordinate update
                    uint32_t in_inner_idx = wk + hk*Win;
                    // Padding condition
                    uint32_t w_pad_cond = wk + wo*Wstr;
                    uint32_t h_pad_cond = hk + ho*Hstr;

                    if ((padding>0)&&((h_pad_cond<Upad) || (w_pad_cond<Lpad) || (h_pad_cond>Ho+(Hk)-Dpad) || (w_pad_cond>Wo+(Wk)-Rpad))) {
                      // Padding
                      i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                      //printf("(pad) i2c_buf[%d]=%f                        kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], kernel_idx, segment_idx, ho);
                    }
                    else {
                      // Fill IM2COL buffer
                      i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = input->data[receptive_field_idx+in_inner_idx];
                      //printf("(i2c) i2c_buf[%d]=%f (indata=%f)      kernel_idx=%d, segment_idx=%d, ho=%d\n", kernel_idx+segment_idx, i2c_buf[kernel_idx+segment_idx], input->data[receptive_field_idx], kernel_idx, segment_idx, ho);
                    }
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }
        }

      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        // Partial im2row indices
        uint32_t phi = 0; uint32_t pwi = 0;  
        
        //for (uint32_t hi=0; hi<Hin; hi++) {
        for (uint32_t hi=ht_start; hi<ht_stop; hi++) {
          //for (uint32_t wi=0; wi<Win; wi++) {
          for (uint32_t wi=wt_start; wi<wt_stop; wi++) {
            for (uint32_t co=start; co<stop; co++) {
              // IM2COL buffer coordinates
              uint32_t kernel_idx = co*Hin*Win*Hk*Wk;
              //uint32_t segment_idx = wi + hi*Win;
              uint32_t segment_idx = pwi + phi*Win;
              // Output grad tensor coordinates
              int ho_rf = hi - (Hk-1);
              int wo_rf = wi - (Wk-1);
              int receptive_field_idx = wo_rf + ho_rf*Wox + co*Hox*Wox;

              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  // IM2COl buffer coordinates
                  uint32_t i2c_inner_idx = wk*Hin*Win + hk*Hin*Win*Wk;
                  // Output grad tensor coordinates
                  uint32_t out_inner_idx = wk + hk*Wox;
                  // Padding condition
                  int w_pad_cond = wk + wo_rf;
                  int h_pad_cond = hk + ho_rf;

                  if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                    // Padding
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0;
                  }
                  else {
                    // Fill IM2COL buffer
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
                  }
                }
              }
            }
            pwi++;
            if (pwi == wt_stop) pwi = 0;
          }
          phi++;
        }
      }
    }
    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32] DMA not implemented!\n");
    }
  }

  /**
   * USE HWC FORMAT (ADJACENT ELEMENTS ARE CHANNEL ELEMENTS IN THE INPUT OR OUTPUT MATRIX)
   */
  else if (HWC == 1) {
    /**
     * LOCAL L1 IM2COL (FROM L1 TENSOR TO L1 IM2COL_BUFFER)
     */
    if (USE_DMA == 0) {
      // FORWARD & WEIGHT GRAD
      if (mod==0)
      {
        if ((Hin-Hk+Upad+Dpad+Hstr) % Hstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid H stride (non multiple H sizes): have H_in=%d, H_ker=%d, U_pad=%d, D_pad=%d, H_stride=%d, remainder=%d", Hin, Hk, Upad, Dpad, Hstr, (Hin-Hk+Upad+Dpad+Hstr) % Hstr); return;}
        else                                        Htot = (Hin-Hk+Upad+Dpad+Hstr)/Hstr;
        if ((Win-Wk+Lpad+Rpad+Wstr) % Wstr > 0)     {printf("\n[pulp_im2col_fp32: 243] Invalid W stride (non multiple W sizes): have W_in=%d, W_ker=%d, L_pad=%d, R_pad=%d, W_stride=%d, remainder=%d", Win, Wk, Lpad, Rpad, Wstr, (Win-Wk+Lpad+Rpad+Wstr) % Wstr); return;}
        else                                        Wtot = (Win-Wk+Lpad+Rpad+Wstr)/Wstr;

        uint32_t padding = Lpad + Rpad + Upad + Dpad;
        // Partial im2row indices
        uint32_t pho = 0; uint32_t pwo = 0;
        // Recompute parallel indices per core
        int blockSize = ((ht_stop-ht_start+1)+NUM_CORES-1) / NUM_CORES;
        start = pi_core_id()*blockSize > ht_start ? ht_start : pi_core_id()*blockSize;
        stop = start+blockSize > ht_stop ? ht_stop : start+blockSize;

        if (padding == 0) {

          for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
            //for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
            for (uint32_t wo=wt_start; wo<wt_stop/*Wtot*/; wo++) {
              // Im2Col indices
              //uint32_t segment_idx = wo + ho*Wtot;
              uint32_t segment_idx = pwo + pho*Wtot;
              // Input activation indices
              uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
              for (uint32_t hk=0; hk<Hk; hk++) {
                for (uint32_t wk=0; wk<Wk; wk++) {
                  for (uint32_t ci=0; ci<Cin; ci++) {
                    // Im2Col indices
                    uint32_t i2c_inner_idx = ci*Htot*Wtot + wk*Htot*Wtot*Cin + hk*Htot*Wtot*Cin*Wk;
                    // Input activation indices                    
                    uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
                    // Fill im2col buffer
                    i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
                  }
                }
              }
              pwo++;
              if (pwo == wt_stop) pwo = 0;
            }
            pho++;
          }

        }
        else {

          printf("\n[pulp_im2col_fp32.c:] Padding not implemented for HWC im2col without DMA!\n");

          // for (uint32_t ho=start; ho<stop/*Htot*/; ho++) {
          //   for (uint32_t wo=0; wo<Wtot/*Wtot*/; wo++) {
          //     // Im2Col indices
          //     uint32_t segment_idx = wo*Hk*Wk*Cin + ho*Hk*Wk*Cin*(Wtot);
          //     // Input activation indices
          //     uint32_t input_idx = (wo*Wstr-Lpad)*Cin + (ho*Hstr-Upad)*Cin*Win;
          //     for (uint32_t hk=0; hk<Hk; hk++) {
          //       for (uint32_t wk=0; wk<Wk; wk++) {
          //         for (uint32_t ci=0; ci<Cin; ci++) {
          //           // Im2Col indices
          //           uint32_t i2c_inner_idx = ci + wk*Cin + hk*Cin*Wk;
          //           // Input activation indices                    
          //           uint32_t act_idx = ci + wk*Cin + hk*Cin*Win;
          //           // Fill im2col buffer
          //           i2c_buf[segment_idx+i2c_inner_idx] = input->data[input_idx+act_idx];
          //         }
          //       }
          //     }
          //   }
          // }

        }
      }
      else // IN GRAD
      {
        uint32_t Hox = output->H;
        uint32_t Wox = output->W;
        // Partial im2row indices
        uint32_t phi = 0; uint32_t pwi = 0;  
        // Recompute start and stop indices per core
        int blockSize = ((ht_stop-ht_start+1)+NUM_CORES-1) / NUM_CORES;
        start = pi_core_id()*blockSize > ht_start ? ht_start : pi_core_id()*blockSize;
        stop = start+blockSize > ht_stop ? ht_stop : start+blockSize;

        for (uint32_t hi=start/*0*/; hi<stop/*Hin*/; hi++) {
          //for (uint32_t wi=0; wi<Win; wi++) {
          for (uint32_t wi=wt_start; wi<wt_stop; wi++) {
            // Padding variables
            int ho_rf = hi - (Hk-1);
            int wo_rf = wi - (Wk-1);

            for (uint32_t hk=0; hk<Hk; hk++) {
              for (uint32_t wk=0; wk<Wk; wk++) {
                // Padding conditions
                int w_pad_cond = wk + wo_rf;
                int h_pad_cond = hk + ho_rf;  

                // Set padding loop
                if ((h_pad_cond<0) || (w_pad_cond<0) || (h_pad_cond>=(int)Hox) || (w_pad_cond>=(int)Wox)) {
                  for (uint32_t co=0; co<Co; co++) {
                    // IM2COL buffer coordinates
                    //uint32_t segment_idx = wi + hi*Win;
                    uint32_t segment_idx = pwi + phi*Win;
                    uint32_t kernel_idx = wk*Co*Hin*Win + hk*Co*Hin*Win*Wk;
                    uint32_t i2c_inner_idx = co*Hin*Win;  

                    // Fill with zeroes  
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = 0.0f;             
                  }
                }
                else {
                  // Non-padded iteration
                  for (uint32_t co=0; co<Co; co++) {
                    // OutDiff coordinates
                    int receptive_field_idx = (wo_rf+wk)*Co + (ho_rf+hk)*Co*Wox;
                    uint32_t out_inner_idx = co;

                    // IM2COL buffer coordinates
                    //uint32_t segment_idx = wi + hi*Win;
                    uint32_t segment_idx = pwi + phi*Win;
                    uint32_t kernel_idx = wk*Co*Hin*Win + hk*Co*Hin*Win*Wk;
                    uint32_t i2c_inner_idx = co*Hin*Win;

                    // Fill IM2COL buffer
                    i2c_buf[kernel_idx+segment_idx+i2c_inner_idx] = output->diff[receptive_field_idx+out_inner_idx];
                  }
                                           
                }
              }
            }
            pwi++;
            if (pwi == wt_stop) pwi = 0;
          }
          phi++;
        }
      }
    }

    // ERROR SIGNAL
    else {
      printf("\n[pulp_im2col_fp32] DMA not implemented!\n");
    }
    
  }

  // ERROR SIGNAL
  else {
    printf("[pulp_im2col_fp32:] Invalid HWC parameter (not 0 or 1)\n");
  }
}







void pulp_blocktransp_fp32 (void * blocktransp_args)
{
  struct blocktransp_args * args = (struct blocktransp_args *) blocktransp_args;
  float * weights = args->weights;
  float * bt_weights = args->bt_weights;
  uint32_t Cin = args->Cin;
  uint32_t Cout = args->Cout;
  uint32_t Hk = args->Hk;
  uint32_t Wk = args->Wk;
  uint8_t HWC_layout = args->HWC;

  uint32_t HW = Hk*Wk;

  uint32_t blockSize = (Cout+NUM_CORES-1) / NUM_CORES;
  uint32_t start = pi_core_id()*blockSize;
  uint32_t stop = start+blockSize > Cout ? Cout : start+blockSize;

  // USE CHW LAYOUT
  if (HWC_layout == 0) {
    // Block tranposition
    // for (uint32_t k=0; k<Cout; k++)
    for (uint32_t k=start; k<stop; k++)
    {
      for (uint32_t c=0; c<Cin; c++)
      {
        for (uint32_t i=0; i<Hk*Wk; i++)
        {
          // OTHER MATRIX
          //bt_weights[i+k*HW+c*Cout*HW] = weights[i+c*HW+k*Cin*HW];
          bt_weights[i+k*HW+c*Cout*HW] = weights[(HW-1-i)+c*HW+k*Cin*HW];
        }
      }
    } 
  }

  // USE HWC LAYOUT
  else if (HWC_layout == 1) {
    for (uint32_t co=0; co<Cout; co++) {
      for (uint32_t hk=0; hk<Hk; hk++) {
        for (uint32_t wk=0; wk<Wk; wk++) {
          for (uint32_t ci=0; ci<Cin; ci++) {
            bt_weights[ci*Hk*Wk*Cout + wk*Cout + hk*Wk*Cout + co] = weights[ci + (Wk-1-wk)*Cin + (Hk-1-hk)*Wk*Cin + co*Wk*Hk*Cin];
          }
        }
      }
    }
  }

  // LAYOUT ERROR
  else {
    printf("[pulp_blocktransp_fp32.c] Invalid data layout (not 0 or 1)!!\n");
  }
}
