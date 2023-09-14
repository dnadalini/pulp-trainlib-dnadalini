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

#include "pmsis.h"
#include "pulp_train_utils_fp32.h"
#include "pulp_matmul_fp32.h"


int verify_tensor(float * tensor_out, float * tensor_ref, int size, float tolerance){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > tolerance ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



void transpose(void * void_args) 
{
    struct transp_args args = *((struct transp_args *)void_args);
    float * matrix = args.matrix;
    float * transp_matrix = args.transp_matrix;
    int N = args.N;
    int M = args.M;

    // Parallelize on N or M depending on the wides available dimension
    if (N > M) 
    {
        int blockSize = (N+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > N ? N : start+blockSize;

        for (int i=start; i<stop; i++)
        {
            for (int j=0; j<M; j++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }
    else 
    {
        int blockSize = (M+NUM_CORES-1) / NUM_CORES;
        int start = pi_core_id()*blockSize;
        int stop = start+blockSize > M ? M : start+blockSize;

        for (int j=start; j<stop; j++)
        {
            for (int i=0; i<N; i++)
            {
                transp_matrix[j*N+i] = matrix[i*M+j];
            }
        }
    }
}



void copy (void * void_args)
{
  struct copy_args args = *((struct copy_args *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for(int i=start; i<stop; i++)
    args.to[i] = args.from[i];
}



void set_to_value (void * void_args)
{
  struct set_to_value_args args = *((struct set_to_value_args *)void_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for(int i=start; i<stop; i++)
    args.to[i] = args.value;  
}



void vect_sum (void * vect_sum_args)
{
  struct vect_sum_args * args = (struct vect_sum_args*) vect_sum_args;
  float * op_1 = args->op_1;
  float * op_2 = args->op_2;
  float * dest = args->dest;
  int size = args->size;

  int blockSize = (size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > size ? size : start+blockSize;

  for (int i=start; i<stop; i++) {
      dest[i] = op_1[i] + op_2[i];
  }   
}



void cast_fp16_tensor_to_fp32 (void * cast_16t32_args) 
{
  struct cast_16t32_args args = *((struct cast_16t32_args *)cast_16t32_args);
  int blockSize = (args.size+NUM_CORES-1) / NUM_CORES;
  int start = pi_core_id()*blockSize;
  int stop = start+blockSize > args.size ? args.size : start+blockSize;

  for (int i=start; i<stop; i++) {
    args.destination[i] = (float) args.source[i];
  }
}




void pad_tensor (void * pad_args) 
{
    struct pad_args * args = (struct pad_args*) pad_args;
    float * source = args->source;
    float * dest = args->dest;
    int C = args->C;
    int H = args->H;
    int W = args->W;
    int L_PAD = args->T_LPAD;
    int R_PAD = args->T_RPAD;
    int U_PAD = args->T_UPAD;
    int D_PAD = args->T_DPAD;
    int HWC = args->HWC_lay;
    
    int H_out = H + U_PAD + D_PAD;
    int W_out = W + L_PAD + R_PAD;

    int blockSize = (C+NUM_CORES-1) / NUM_CORES;
    int start = pi_core_id()*blockSize;
    int stop = start+blockSize > C ? C : start+blockSize;

    if (HWC == 0) 
    {
        for (int ch=0; ch<C; ch++) 
        {
            for (int ht=0; ht<H_out; ht++) 
            {
                for (int wt=0; wt<W_out; wt++) 
                {
                    // Compute matrix idx
                    int in_t_idx = (wt-L_PAD) + (ht-U_PAD)*W + ch*H*W;
                    int out_t_idx = wt + ht*W_out + ch*H_out*W_out;
                    // Padding conditions
                    int zero_cond = (wt < L_PAD || wt > W) || (ht < U_PAD || ht > H);
                    if (zero_cond == 1) { dest[out_t_idx] = 0; }
                    else 
                    {
                        dest[out_t_idx] = source[in_t_idx];
                    }
                }
            }
        }
    }
    else 
    {
        printf("[pad_tensor] HWC layout not implemented!!");
    }
}



void HWC_to_CHW (void * layout_args) 
{
    struct layout_args * args = (struct layout_args *) layout_args;
    float * data = args->tensor->data;
    float * grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    float * buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args tr_args;
    struct copy_args cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        tr_args.matrix = data;
        tr_args.transp_matrix = buff;
        tr_args.N = H*W;
        tr_args.M = C;
        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }

    if (transpose_grad == 1) {
        // Transpose grad
        tr_args.matrix = grad;
        tr_args.transp_matrix = buff;
        tr_args.N = H*W;
        tr_args.M = C;
        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);    
    }
}




void CHW_to_HWC (void * layout_args) 
{
    struct layout_args * args = (struct layout_args *) layout_args;
    float * data = args->tensor->data;
    float * grad = args->tensor->diff;
    uint16_t C = args->tensor->C;
    uint16_t H = args->tensor->H;
    uint16_t W = args->tensor->W;
    float * buff = args->transp_buffer;
    uint8_t transpose_data = args->transpose_data;
    uint8_t transpose_grad = args->transpose_grad;

    struct transp_args tr_args;
    struct copy_args cpy_args;

    if (transpose_data == 1) {
        // Transpose data
        tr_args.matrix = data;
        tr_args.transp_matrix = buff;
        tr_args.N = C;
        tr_args.M = H*W;
        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = data;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);
    }

    if (transpose_grad == 1)  {
        // Transpose grad
        tr_args.matrix = grad;
        tr_args.transp_matrix = buff;
        tr_args.N = C;
        tr_args.M = H*W;
        pi_cl_team_fork(NUM_CORES, transpose, &tr_args);
        cpy_args.from = buff;
        cpy_args.to = grad;
        cpy_args.size = C*H*W;
        pi_cl_team_fork(NUM_CORES, copy, &cpy_args);    
    }
}




/**
 * Choose the user-selected matmul for the chosen layer.
 */
void mm_manager (void * void_args)
{
    struct mm_manager_args* args = (struct mm_manager_args *) void_args;
    
    struct matMul_args *matMul_args = args->mm_args;    
    struct matMul_DW_args *matMul_DW_args = args->mm_dw_args;
    int layer_type = args->layer_type;
    int step_type = args->step_type;
    int matmul_type = args->matmul_type;

    #ifdef DEBUG
    printf("Running layer %d, step %d, matmul %d\n", layer_type, step_type, matmul_type);
    #endif
    
// =====> CONV2D
    if (layer_type == LAYER_CONV2D) 
    {
        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection

    }

// =====> DEPTHWISE CONVOLUTION
    else if (layer_type == LAYER_DW_CONV) 
    {

        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_u2((void *) matMul_DW_args);}
            else if (matmul_type == 2)      { mm_dw_u3((void *) matMul_DW_args);}
            else if (matmul_type == 3)      { mm_dw_unroll_1x2((void *) matMul_DW_args);}
            else if (matmul_type == 4)      { mm_dw_unroll_1x4((void *) matMul_DW_args);}
            else if (matmul_type == 5)      { mm_dw_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_unroll_1x4_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_u2((void *) matMul_DW_args);}
            else if (matmul_type == 2)      { mm_dw_u3((void *) matMul_DW_args);}
            else if (matmul_type == 3)      { mm_dw_unroll_1x2((void *) matMul_DW_args);}
            else if (matmul_type == 4)      { mm_dw_unroll_1x4((void *) matMul_DW_args);}
            else if (matmul_type == 5)      { mm_dw_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_unroll_1x4_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            if      (matmul_type == 0)      { mm_dw_in_grad((void *) matMul_DW_args); }
            else if (matmul_type == 1)      { mm_dw_in_grad_u2((void *) matMul_DW_args); }
            else if (matmul_type == 2)      { mm_dw_in_grad_u3((void *) matMul_DW_args); }
            else if (matmul_type == 3)      { mm_dw_in_grad_unroll_1x2((void *) matMul_DW_args); }
            else if (matmul_type == 4)      { mm_dw_in_grad_unroll_1x4((void *) matMul_DW_args); }
            else if (matmul_type == 5)      { mm_dw_in_grad_unroll_1x2_u2((void *) matMul_DW_args);}
            else if (matmul_type == 6)      { mm_dw_in_grad_unroll_1x4_u2((void *) matMul_DW_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
        
    }

// =====> POINTWISE CONVOLUTION
    else if (layer_type == LAYER_PW_CONV)
    {

        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
        
    }

// =====> LINEAR LAYER
    else if (layer_type == LAYER_LINEAR)
    {
        // Select step type
        if (step_type == STEP_FW)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_WGT_GRAD) 
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }

        else if (step_type == STEP_IN_GRAD)
        {
            // Select matmul type
            // Naive
            if      (matmul_type == 0)      { mm((void *) matMul_args); }
            else if (matmul_type == 1)      { mm_M((void *) matMul_args);}
            // Parallelism on N
            else if (matmul_type == 2)      { mm_u2((void *) matMul_args);}
            else if (matmul_type == 3)      { mm_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 4)      { mm_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 5)      { mm_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 6)      { mm_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 7)      { mm_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 8)      { mm_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 9)      { mm_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 10)     { mm_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 11)     { mm_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 12)     { mm_unroll_4x4((void *) matMul_args);}
            // Parallelism on M
            else if (matmul_type == 13)     { mm_M_u2((void *) matMul_args);}
            else if (matmul_type == 14)     { mm_M_unroll_1x2((void *) matMul_args);}
            else if (matmul_type == 15)     { mm_M_unroll_1x4((void *) matMul_args);}
            else if (matmul_type == 16)     { mm_M_unroll_1x8((void *) matMul_args);}
            else if (matmul_type == 17)     { mm_M_unroll_2x1((void *) matMul_args);}
            else if (matmul_type == 18)     { mm_M_unroll_4x1((void *) matMul_args);}
            else if (matmul_type == 19)     { mm_M_unroll_8x1((void *) matMul_args);}
            else if (matmul_type == 20)     { mm_M_unroll_2x2((void *) matMul_args);}
            else if (matmul_type == 21)     { mm_M_unroll_2x4((void *) matMul_args);}
            else if (matmul_type == 22)     { mm_M_unroll_4x2((void *) matMul_args);}
            else if (matmul_type == 23)     { mm_M_unroll_4x4((void *) matMul_args);}
            else
            {
                printf("\nWrong matmul selection!\n");
            }
            // End of matmul type selection
        }
        
        else
        {
            printf("\nWrong step selection!!\n");
        }
        // End step selection
        
    }

// =====> WRONG LAYER SELECTION
    else
    {
        printf("\nWrong layer_type selection!!\n");
    }

}

static inline float
fasterpow2 (float p)
{
  float clipp = (p < -126) ? -126.0f : p;
  union { uint32_t i; float f; } v = { (uint32_t) ( (1 << 23) * (clipp + 126.94269504f) ) };
  return v.f;
}

static inline float
fasterexp (float p)
{
  return fasterpow2 (1.442695040f * p);
}

void exponential(void* void_args){
    struct softmax_args *args = ((struct softmax_args *)void_args);
    float* input = args->input;
    float* output = args->output;
    int dim = args->dim;

    const uint32_t blockSize = (dim+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > dim ? dim : start+blockSize;

    for(int i = start; i < stop; i++){
        output[i] = fasterexp(input[i]);
    }
}

void softmax(void* void_args){
    struct softmax_args *args = ((struct softmax_args *)void_args);
    float* output = args->output;
    int dim = args->dim;
    float sum = args->sum;

    const uint32_t blockSize = (dim+NUM_CORES-1) / NUM_CORES;
    const uint32_t start = pi_core_id()*blockSize;
    const uint32_t stop = start+blockSize > dim ? dim : start+blockSize;

    for(int i = start; i < stop; i++){
        output[i] = output[i] / sum;
    }
}
