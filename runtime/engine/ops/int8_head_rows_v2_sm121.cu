// Four to twelve rows share one traversal of the packed INT8 head (v2).
// x is staged once per block in shared memory (read as 4-element words) and every
// thread owns COLS columns so each x read feeds COLS FMAs. Per (row, column) the
// FMA order (group = part + 8k, word, byte) and the eight-part reduction order are
// exactly those of gptq_head_gemv / _rows, so outputs are bit-identical.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

static __device__ __forceinline__ float bf16_lo(uint32_t v) { return __uint_as_float(v << 16); }
static __device__ __forceinline__ float bf16_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

template<int R, int COLS>
__global__ void __launch_bounds__(256) head_rows_v2(const __nv_bfloat16* x, const uint32_t* weight,
                                                    const __half* scales, __nv_bfloat16* out,
                                                    int n, int groups) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int lane = threadIdx.x, part = threadIdx.y, tid = part * 32 + lane;
    {
        const uint4* src = reinterpret_cast<const uint4*>(x);
        uint4* dst = reinterpret_cast<uint4*>(smem);
        for (int i = tid; i < R * 2560 / 8; i += 256) dst[i] = src[i];
    }
    __syncthreads();
    const uint2* xs = reinterpret_cast<const uint2*>(smem);
    const int col0 = blockIdx.x * 32 * COLS + lane;
    float sum[R][COLS];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int c = 0; c < COLS; c++) sum[r][c] = 0.0f;
    for (int group = part; group < groups; group += 8) {
        float sf[COLS];
        #pragma unroll
        for (int c = 0; c < COLS; c++) sf[c] = __half2float(scales[group * n + col0 + 32 * c]);
        #pragma unroll 2
        for (int word = 0; word < 32; word++) {
            uint32_t bits[COLS];
            #pragma unroll
            for (int c = 0; c < COLS; c++) bits[c] = weight[(group * 32 + word) * n + col0 + 32 * c];
            float w[4][COLS];
            #pragma unroll
            for (int byte = 0; byte < 4; byte++)
                #pragma unroll
                for (int c = 0; c < COLS; c++)
                    w[byte][c] = (float)((int)((bits[c] >> (byte * 8)) & 255) - 128) * sf[c];
            const int quad = (group * 128 + word * 4) >> 2;
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const uint2 v = xs[r * 640 + quad];
                const float x0 = bf16_lo(v.x), x1 = bf16_hi(v.x), x2 = bf16_lo(v.y), x3 = bf16_hi(v.y);
                #pragma unroll
                for (int c = 0; c < COLS; c++) {
                    sum[r][c] = fmaf(x0, w[0][c], sum[r][c]);
                    sum[r][c] = fmaf(x1, w[1][c], sum[r][c]);
                    sum[r][c] = fmaf(x2, w[2][c], sum[r][c]);
                    sum[r][c] = fmaf(x3, w[3][c], sum[r][c]);
                }
            }
        }
    }
    __syncthreads();
    float* partial = reinterpret_cast<float*>(smem);  // [R][8][32*COLS], reuses the x stage
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int c = 0; c < COLS; c++) partial[(r * 8 + part) * (32 * COLS) + lane + 32 * c] = sum[r][c];
    __syncthreads();
    if (part == 0) {
        #pragma unroll
        for (int r = 0; r < R; r++)
            #pragma unroll
            for (int c = 0; c < COLS; c++) {
                float total = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; i++) total += partial[(r * 8 + i) * (32 * COLS) + lane + 32 * c];
                out[r * n + col0 + 32 * c] = __float2bfloat16_rn(total);
            }
    }
}

template<int R, int COLS>
static int launch(const void* x, const void* weight, const void* scales, void* out, int n, int groups, void* stream) {
    if (n % (32 * COLS)) return -3;
    const int bytes = R * 2560 * 2 > R * 8 * 32 * COLS * 4 ? R * 2560 * 2 : R * 8 * 32 * COLS * 4;
    static bool configured = false;
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(head_rows_v2<R, COLS>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return (int)e;
        configured = true;
    }
    head_rows_v2<R, COLS><<<n / (32 * COLS), dim3(32, 8), bytes, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x, (const uint32_t*)weight, (const __half*)scales, (__nv_bfloat16*)out, n, groups);
    return (int)cudaGetLastError();
}

template<int COLS>
static int dispatch(const void* x, const void* w, const void* s, void* o, int n, int g, int rows, void* stream) {
    switch (rows) {
        case 4: return launch<4, COLS>(x, w, s, o, n, g, stream);
        case 5: return launch<5, COLS>(x, w, s, o, n, g, stream);
        case 6: return launch<6, COLS>(x, w, s, o, n, g, stream);
        case 7: return launch<7, COLS>(x, w, s, o, n, g, stream);
        case 8: return launch<8, COLS>(x, w, s, o, n, g, stream);
        case 9: return launch<9, COLS>(x, w, s, o, n, g, stream);
        case 10: return launch<10, COLS>(x, w, s, o, n, g, stream);
        case 11: return launch<11, COLS>(x, w, s, o, n, g, stream);
        case 12: return launch<12, COLS>(x, w, s, o, n, g, stream);
        default: return -2;
    }
}

extern "C" int gptq_head_gemv_rows_v2(const void* x, const void* weight, const void* scales,
                                      void* out, int k, int n, int rows, int cols, void* stream) {
    if (k != 2560 || n <= 0) return -1;
    switch (cols) {
        case 1: return dispatch<1>(x, weight, scales, out, n, k / 128, rows, stream);
        case 2: return dispatch<2>(x, weight, scales, out, n, k / 128, rows, stream);
        case 4: return dispatch<4>(x, weight, scales, out, n, k / 128, rows, stream);
        default: return -4;
    }
}
