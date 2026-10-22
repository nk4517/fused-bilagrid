#ifndef _CONFIG_H_INC
#define _CONFIG_H_INC

// RGB to gray constants
#define kC2G_r 0.299f
#define kC2G_g 0.587f
#define kC2G_b 0.114f

// per-channel guidance weights: C==1 is Y-only (identity), C==3 is RGB-to-gray
#ifdef __CUDACC__
template<int C>
__device__ __forceinline__ float guidance_weight(int c) {
    return (C == 1) ? 1.0f : (c == 0 ? kC2G_r : c == 1 ? kC2G_g : kC2G_b);
}
#endif

#if 0
#include "stdio.h"
#define CHECK_IDX(idx, max_size) \
    if ((idx) < 0 || (idx) >= (max_size)) { \
        printf("out of bound: %s @ %d, %d/%d\n", __FILE__, __LINE__, idx, max_size); \
    }
#endif

#endif  // _CONFIG_H_INC
