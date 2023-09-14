#include "pulp_train.h"

#include "stats.h"
#include "net.h"

PI_L1 float IN_DATA_FP32[Tin_C*Tin_H*Tin_W];
PI_L1 float PADDED_DATA_FP32[Tin_C*Tout_H*Tout_W];

static void init_matrix ()
{
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++)
    {
        IN_DATA_FP32[i] = 0.1 + (float) i/10;
    }
}

// Main function
void net_step () 
{
    #ifdef PROF_NET
    INIT_STATS();
    PRE_START_STATS();
    #endif

    init_matrix();

    printf("Padding a matrix with sizes C = %d, H = %d, W = %d.\n", Tin_C, Tin_H, Tin_W);
    printf("Executing on %d cores.\n", NUM_CORES);

// ---------FP32-------------------------------

    // INITIALIZE STRUCTURE
    struct pad_args args;
    args.source = IN_DATA_FP32;
    args.dest = PADDED_DATA_FP32;
    args.C = Tin_C;
    args.H = Tin_H;
    args.W = Tin_W;
    args.T_LPAD = LPAD;
    args.T_RPAD = RPAD;
    args.T_UPAD = UPAD;
    args.T_DPAD = DPAD;
    args.HWC_lay = 0;

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nHello, starting to pad in FP32!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_W))) printf("\n");
        if (!(i%(Tin_H*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP32[i]);
    }
    #else
    printf("\nHello, starting to pad in FP32!\n");
    printf("\nINPUT MATRIX:\n");
    for (int i=0; i<Tin_C*Tin_H*Tin_W; i++) 
    {
        if (!(i%(Tin_Cin))) printf("\n");
        if (!(i%(Tin_Cin*Tin_W))) printf("\n");
        printf("%f ", IN_DATA_FP32[i]);
    }    
    #endif
    #endif
    printf("\n\nFP32 Stats:\n");

    #ifdef PROF_NET
    START_STATS();
    #endif

    pi_cl_team_fork(NUM_CORES, pad_tensor, &args);

    #ifdef PROF_NET
    STOP_STATS();
    #endif

    #ifdef PRINT_MATS
    #if HWC_LAYOUT == 0
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tout_W))) printf("\n");
        if (!(i%(Tout_H*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP32[i]);
    }
    printf("\n\n");
    #else
    printf("\nTRANSPOSED MATRIX:\n");
    for (int i=0; i<Tin_C*Tout_H*Tout_W; i++) 
    {
        if (!(i%(Tin_C))) printf("\n");
        if (!(i%(Tin_C*Tout_W))) printf("\n");
        printf("%f ", PADDED_DATA_FP32[i]);
    }
    printf("\n\n");    
    #endif
    #endif
        
    return;
}
