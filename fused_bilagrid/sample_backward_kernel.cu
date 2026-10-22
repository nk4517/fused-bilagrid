#include "config.h"


// template<int compute_coords_grad>
#ifdef COMPUTE_COORDS_GRAD
template<int C>
__global__ void bilagrid_sample_backward_kernel_cg(
#else
template<int C>
__global__ void bilagrid_sample_backward_kernel(
#endif
    const float* __restrict__ bilagrid,  // [N,C*(C+1),L,H,W]
    const float* __restrict__ coords,  // [N,m,h,w,2]
    const float* __restrict__ rgb,  // [N,m,h,w,C]
    const float* __restrict__ v_output,  // [N,m,h,w,C]
    float* __restrict__ v_bilagrid,  // [N,C*(C+1),L,H,W]
    #ifdef COMPUTE_COORDS_GRAD
    float* __restrict__ v_coords,  // [N,m,h,w,2]
    #endif
    float* __restrict__ v_rgb,  // [N,m,h,w,C]
    int N, int L, int H, int W,
    int m, int h, int w
) {
    constexpr int NC = C * (C + 1);
    #if 0
    // faster when coords are random
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    #else
    // faster when coords is a regular grid
    // reduces number of threads writing to the same address at a time in atomicAdd
    int wi = threadIdx.x * ((w+blockDim.x-1) / blockDim.x) + blockIdx.x;
    int hi = threadIdx.y * ((h+blockDim.y-1) / blockDim.y) + blockIdx.y;
    #endif

    int idx = blockIdx.z * blockDim.z + threadIdx.z;
    bool inside = (wi < w && hi < h && idx < (N*m));
    if (!inside) return;
    int mi = idx % m;
    int ni = idx / m;

    // grid coords
    int g_off = (((ni*m + mi)*h + hi)*w + wi);
    float s[C];
    #pragma unroll
    for (int c = 0; c < C; c++) s[c] = rgb[C*g_off+c];
    float gx = coords[2*g_off+0];
    float gy = coords[2*g_off+1];
    float gz = 0.0f;
    #pragma unroll
    for (int c = 0; c < C; c++) gz += guidance_weight<C>(c) * s[c];
    float x = gx * (W - 1);
    float y = gy * (H - 1);
    float z = gz * (L - 1);

    // floor + ceil, clamped
    int x0 = floorf(x), y0 = floorf(y), z0 = floorf(z);
    int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
    x0 = min(max(x0,0), W-1); x1 = min(max(x1,0), W-1);
    y0 = min(max(y0,0), H-1); y1 = min(max(y1,0), H-1);
    z0 = min(max(z0,0), L-1); z1 = min(max(z1,0), L-1);

    // fractional parts
    float fx = x - x0, fy = y - y0, fz = z - z0;
    float f000 = (1-fx)*(1-fy)*(1-fz);
    float f001 = fx*(1-fy)*(1-fz);
    float f010 = (1-fx)*fy*(1-fz);
    float f011 = fx*fy*(1-fz);
    float f100 = (1-fx)*(1-fy)*fz;
    float f101 = fx*(1-fy)*fz;
    float f110 = (1-fx)*fy*fz;
    float f111 = fx*fy*fz;

    // read rgb coeffs and upstream gradient
    float gd[C];
    #pragma unroll
    for (int c = 0; c < C; c++) gd[c] = v_output[C*g_off+c];
    float vs[C] = {0.0f};

    // accumulate bilagrid gradient over all channels
    #pragma unroll
    for (int ci = 0; ci < NC; ++ci) {
        // weight from rgb channel
        int si = ci % (C + 1), di = ci / (C + 1);
        float r_coeff = (si < C ? s[si] : 1.f);
        float gout = gd[di];
        float grad_weight = r_coeff * gout;

        // scatter back into the eight corners
        // accounts for >90% of run time, needs optimization
        int base = ((ni*NC + ci)*L*H*W);
        atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x0, f000 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y0)*W + x1, f001 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y1)*W + x0, f010 * grad_weight);
        atomicAdd(v_bilagrid + base + (z0*H + y1)*W + x1, f011 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y0)*W + x0, f100 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y0)*W + x1, f101 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y1)*W + x0, f110 * grad_weight);
        atomicAdd(v_bilagrid + base + (z1*H + y1)*W + x1, f111 * grad_weight);

        // gradient w.r.t. rgb coefficients
        if (si < C) {
            float val =
                ( ( (bilagrid[base + (z0*H + y0)*W + x0]*(1-fx) + bilagrid[base + (z0*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[base + (z0*H + y1)*W + x0]*(1-fx) + bilagrid[base + (z0*H + y1)*W + x1]*fx)*fy )*(1-fz)
                + ( (bilagrid[base + (z1*H + y0)*W + x0]*(1-fx) + bilagrid[base + (z1*H + y0)*W + x1]*fx)*(1-fy)
                    + (bilagrid[base + (z1*H + y1)*W + x0]*(1-fx) + bilagrid[base + (z1*H + y1)*W + x1]*fx)*fy )*fz
                );
            vs[si] += val * gout;
        }
    }

    // spatial derivatives for coords
    // ∂w000/∂x = -(1-fy)*(1-fz), ∂w001/∂x = +(1-fy)*(1-fz), etc...
#ifdef COMPUTE_COORDS_GRAD
    float dwdx[8] = {
        -(1-fy)*(1-fz),  (1-fy)*(1-fz),
        -fy*(1-fz),      fy*(1-fz),
        -(1-fy)*fz,      (1-fy)*fz,
        -fy*fz,          fy*fz
    };
    float dwdy[8] = {
        -(1-fx)*(1-fz), -fx*(1-fz),
         (1-fx)*(1-fz),  fx*(1-fz),
        -(1-fx)*fz,     -fx*fz,
         (1-fx)*fz,      fx*fz
    };
#endif
    float dwdz[8] = {
        -(1-fx)*(1-fy), -fx*(1-fy),
        -(1-fx)*fy,     -fx*fy,
         (1-fx)*(1-fy),  fx*(1-fy),
         (1-fx)*fy,      fx*fy
    };

    // accumulate gradient into coords (chain through bilagrid values and rgb)
    #ifdef COMPUTE_COORDS_GRAD
    float gx_grad = 0.f, gy_grad = 0.f;
    #endif
    float gz_grad = 0.f;
    #pragma unroll
    for (int corner = 0; corner < 8; ++corner) {
        int xi = (corner & 1) ? x1 : x0;
        int yi = (corner & 2) ? y1 : y0;
        int zi = (corner & 4) ? z1 : z0;
        float trilerp = 0.f;
        // gather the corresponding bilagrid value for each of the 12 channels
        #pragma unroll
        for (int ci = 0; ci < NC; ++ci) {
            const float* vol = bilagrid + ((ni*NC + ci)*L*H*W);
            float v = vol[(zi*H + yi)*W + xi];
            int si = ci % (C + 1), di = ci / (C + 1);
            float r_coeff = (si < C ? s[si] : 1.f);
            float gout = gd[di];
            trilerp += v * r_coeff * gout;
        }
        #ifdef COMPUTE_COORDS_GRAD
        gx_grad += dwdx[corner] * (W-1) * trilerp;
        gy_grad += dwdy[corner] * (H-1) * trilerp;
        #endif
        gz_grad += dwdz[corner] * (L-1) * trilerp;
    }
    // save gradient, with discontinuty masking
    #ifdef COMPUTE_COORDS_GRAD
    v_coords[2*g_off+0] = gx_grad * (float)(x0 != x && x1 != x);
    v_coords[2*g_off+1] = gy_grad * (float)(y0 != y && y1 != y);
    #endif
    gz_grad *= (float)(z0 != z && z1 != z);
    #pragma unroll
    for (int c = 0; c < C; c++)
        v_rgb[C*g_off+c] = vs[c] + guidance_weight<C>(c) * gz_grad;
}
