// GB10 blockwise FP8 activation quantization bring-up; 128 values per warp.
// The selectable arithmetic mode is temporary differential-test instrumentation.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

__global__ void quantize_groups(const __nv_bfloat16* x, __nv_fp8_e4m3* q,
                               float* scales, int rows, int cols, int mode) {
    const int group = blockIdx.x * blockDim.y + threadIdx.y;
    if (group >= rows*(cols/128)) return;
    const int lane = threadIdx.x;
    const int base = group * 128;
    float values[4];
    float amax = 1e-10f;
    #pragma unroll
    for (int i=0; i<4; ++i) {
        values[i] = __bfloat162float(x[base+lane+i*32]);
        amax = fmaxf(amax, fabsf(values[i]));
    }
    #pragma unroll
    for (int step=16; step>0; step/=2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, step));
    const int groups_per_row = cols/128;
    const float scale = __fdiv_rn(amax,448.0f);
    if (lane == 0)
        scales[(group%groups_per_row)*rows + group/groups_per_row] = scale;
    #pragma unroll
    for (int i=0; i<4; ++i) {
        float scaled;
        if (mode==0) scaled = values[i] * __fdividef(448.0f, amax);
        else if (mode==1) scaled = __fdividef(values[i]*448.0f, amax);
        else if (mode==2) scaled = values[i] / amax * 448.0f;
        else if (mode==3) scaled = values[i] * 448.0f / amax;
        else scaled = __fdiv_rn(values[i],scale);
        q[base+lane+i*32] = __nv_fp8_e4m3(fminf(448.0f,fmaxf(-448.0f,scaled)));
    }
}

extern "C" int fp8_quantize(const void* x, void* q, float* scales,
                            int rows, int cols, int mode, void* stream) {
    if (rows<=0 || cols<=0 || cols%128 || mode<0 || mode>4) return -1;
    // Decode needs many small blocks to cover the SMs; prefill amortizes
    // block scheduling by placing four independent group warps per block.
    const int groups = rows*(cols/128);
    const int warps = groups>=128 ? 4 : 1;
    quantize_groups<<<(groups+warps-1)/warps,dim3(32,warps),0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(__nv_fp8_e4m3*)q,scales,rows,cols,mode);
    return (int)cudaGetLastError();
}
