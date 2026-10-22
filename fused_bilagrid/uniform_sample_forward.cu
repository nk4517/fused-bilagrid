#include "config.h"


template<int C>
__global__ void bilagrid_uniform_sample_forward_kernel(
    const float* __restrict__ bilagrid, // [N,C*(C+1),L,H,W]
    const float* __restrict__ rgb,  // [N,m,h,w,C]
    float* __restrict__ output,  // [N,m,h,w,C]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    constexpr int NC = C * (C + 1);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * m * h * w;
    if (idx >= total) return;

    int tmp = idx;
    int wi = tmp % w; tmp /= w;
    int hi = tmp % h; tmp /= h;
    int mi = tmp % m; tmp /= m;
    int ni = tmp;

    // read coords
    int g_offset = (((ni * m + mi) * h + hi) * w + wi) * C;

    // input and output colors
    float s[C];
    #pragma unroll
    for (int c = 0; c < C; c++) s[c] = rgb[g_offset+c];
    float d[C] = {0.0f};

    // grid coords
    float gx = (float)wi / (float)(w-1);
    float gy = (float)hi / (float)(h-1);
    float gz = 0.0f;
    #pragma unroll
    for (int c = 0; c < C; c++) gz += guidance_weight<C>(c) * s[c];
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);

    // find corner indices
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int z0 = (int)floorf(z);
    int x1 = min(x0+1, W-1);
    int y1 = min(y0+1, H-1);
    int z1 = z0 + 1;
    z0 = min(max(z0, 0), L-1);
    z1 = min(max(z1, 0), L-1);

    // interpolation parameters
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float fz = z - (float)z0;

    // interpolate and and affine in one loop
    #pragma unroll
    for (int ci = 0; ci < NC; ci++) {
        // base pointer for this volume
        int base = (ni*NC + ci)*L*H*W;

        // fetch 8 corners
        auto v000 = bilagrid[base+(z0*H+y0)*W+x0];
        auto v001 = bilagrid[base+(z0*H+y0)*W+x1];
        auto v010 = bilagrid[base+(z0*H+y1)*W+x0];
        auto v011 = bilagrid[base+(z0*H+y1)*W+x1];
        auto v100 = bilagrid[base+(z1*H+y0)*W+x0];
        auto v101 = bilagrid[base+(z1*H+y0)*W+x1];
        auto v110 = bilagrid[base+(z1*H+y1)*W+x0];
        auto v111 = bilagrid[base+(z1*H+y1)*W+x1];

        // trilinear interp
        float c00 = v000*(1.0f-fx) + v001*fx;
        float c01 = v010*(1.0f-fx) + v011*fx;
        float c10 = v100*(1.0f-fx) + v101*fx;
        float c11 = v110*(1.0f-fx) + v111*fx;
        float c0 = c00*(1.0f-fy) + c01*fy;
        float c1 = c10*(1.0f-fy) + c11*fy;
        float val = c0*(1.0f-fz) + c1*fz;

        // affine transform
        int si = ci % (C + 1);
        int di = ci / (C + 1);
        d[di] += val * (si < C ? s[si] : 1.0f);
    }

    #pragma unroll
    for (int c = 0; c < C; c++)
        output[g_offset+c] = isfinite(d[c]) ? d[c] : 0.5f;
}


void bilagrid_uniform_sample_forward(
    const float* bilagrid,
    const float* rgb,
    float* output,
    int N, int L, int H, int W,
    int m, int h, int w, int C
) {
    int total = N * m * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (C == 1)
        bilagrid_uniform_sample_forward_kernel<1><<<blocks, threads>>>(
            bilagrid, rgb, output,
            N, L, H, W, m, h, w
        );
    else
        bilagrid_uniform_sample_forward_kernel<3><<<blocks, threads>>>(
            bilagrid, rgb, output,
            N, L, H, W, m, h, w
        );
    // cudaDeviceSynchronize();
}
