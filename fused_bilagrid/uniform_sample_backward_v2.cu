#include "config.h"


template<int C>
__global__ void bilagrid_uniform_sample_backward_v2_kernel(
    const float* __restrict__ bilagrid,  // [N,C*(C+1),L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,C]
    const float* __restrict__ v_output,  // [N,m,h,w,C]
    float* __restrict__ v_bilagrid,  // [N,C*(C+1),L,H,W]
    float* __restrict__ v_rgb,  // [N,m,h,w,C]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    constexpr int NC = C * (C + 1);
    #if 0
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    #else
    // reduces number of threads writing to the same address at a time in atomicAdd
    int wi = threadIdx.x * ((w+blockDim.x-1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h+blockDim.y-1) / blockDim.y) + blockIdx.y;
    #endif

    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    bool inside = (wi < w && hi < h && idx < (N*m));
    if (!inside) return;
    int mi = idx % m;
    int ni = idx / m;

    // load RGB colors
    int g_off = (((ni*m + mi)*h + hi)*w + wi) * C;
    float s[C];
    #pragma unroll
    for (int c = 0; c < C; c++) {
        float sc = rgb[g_off+c];
        s[c] = isfinite(sc) ? sc : 0.5f;
    }

    // grid coords
    float x = (float)wi / (float)(w-1) * (float)(W-1);
    float y = (float)hi / (float)(h-1) * (float)(H-1);
    float gz = 0.0f;
    #pragma unroll
    for (int c = 0; c < C; c++) gz += guidance_weight<C>(c) * s[c];
    float z = gz * (L-1);

    // floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = min(x0+1, W-1);
    int y1 = min(y0+1, H-1);
    int z1 = z0 + 1;
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;

    // read rgb coeffs and upstream gradient
    float gd[C];
    #pragma unroll
    for (int c = 0; c < C; c++) {
        float dc = v_output[g_off+c];
        gd[c] = isfinite(dc) ? dc : 0.0f;
    }
    float vs[C] = {0.0f};

    // spatial derivatives for coords

    float gz_grad = 0.f;

    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;

        float dfdz = ((corner & 1) ? fx : (1-fx)) *
            ((corner & 2) ? fy : (1-fy)) * ((corner & 4) ? 1 : -1);
        float f = dfdz * ((corner & 4) ? fz : (fz-1));

        float trilerp = 0.f;
        #pragma unroll
        for (int ci = 0; ci < NC; ++ci) {
            int bidx = (((ni*NC + ci)*L + zi)*H + yi)*W + xi;
            int si = ci % (C + 1), di = ci / (C + 1);

            float r_coeff = (si < C ? s[si] : 1.f);
            float gout = gd[di];

            float v = bilagrid[bidx];

            if (si < C)
                vs[si] += v * f * gout;

            float grad_weight = r_coeff * gout;
            trilerp += v * grad_weight;
            atomicAdd(v_bilagrid+bidx, f * grad_weight);
        }
        gz_grad += dfdz * (L-1) * trilerp;
    }

    // save gradient, with discontinuity masking
    gz_grad *= (float)(z0 != z && z1 != z);
    #pragma unroll
    for (int c = 0; c < C; c++)
        v_rgb[g_off+c] = vs[c] + guidance_weight<C>(c) * gz_grad;
}


void bilagrid_uniform_sample_backward_v2(
    const float* bilagrid,
    const float* rgb,
    const float* v_output,
    float* v_bilagrid,
    float* v_rgb,
    int N, int L, int H, int W,
    int m, int h, int w, int C
) {
    dim3 block = { 16, 16, 1 };
    dim3 bounds = {
        (w +block.x-1)/block.x,
        (h +block.y-1)/block.y,
        (N*m +block.z-1)/block.z
    };
    if (C == 1)
        bilagrid_uniform_sample_backward_v2_kernel<1><<<bounds, block>>>(
            bilagrid, rgb, v_output,
            v_bilagrid, v_rgb,
            N, L, H, W, m, h, w
        );
    else
        bilagrid_uniform_sample_backward_v2_kernel<3><<<bounds, block>>>(
            bilagrid, rgb, v_output,
            v_bilagrid, v_rgb,
            N, L, H, W, m, h, w
        );
}
