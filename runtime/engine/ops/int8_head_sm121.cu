// Single-token symmetric GPTQ INT8 head, group128, desc_act=false.
// Reads the checkpoint's original [K/4,N] uint32 packing without expanding W.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<int KWARPS>
__global__ void head_gemv(const __nv_bfloat16* x,const uint32_t* weight,
                          const __half* scales,__nv_bfloat16* out,
                          int n,int groups) {
    const int lane=threadIdx.x, part=threadIdx.y;
    const int col=blockIdx.x*32+lane;
    float sum=0.0f;
    for(int group=part;group<groups;group+=KWARPS) {
        const float sf=__half2float(scales[group*n+col]);
        #pragma unroll 4
        for(int word=0;word<32;word++) {
            const uint32_t bits=weight[(group*32+word)*n+col];
            #pragma unroll
            for(int byte=0;byte<4;byte++) {
                const float w=(float)((int)((bits>>(byte*8))&255)-128)*sf;
                const float value=__bfloat162float(x[group*128+word*4+byte]);
                sum=fmaf(value,w,sum);
            }
        }
    }
    __shared__ float partial[KWARPS][32];
    partial[part][lane]=sum;
    __syncthreads();
    if(part==0) {
        float total=0.0f;
        #pragma unroll
        for(int i=0;i<KWARPS;i++)total+=partial[i][lane];
        out[col]=__float2bfloat16_rn(total);
    }
}

extern "C" int gptq_head_gemv(const void* x,const void* weight,const void* scales,
                              void* out,int k,int n,void* stream) {
    if(k!=2560 || n<=0 || n%32)return -1;
    head_gemv<8><<<n/32,dim3(32,8),0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(const uint32_t*)weight,
        (const __half*)scales,(__nv_bfloat16*)out,n,k/128);
    return (int)cudaGetLastError();
}

// Two consecutive verification rows share one traversal of the packed head.
// Each row retains the ordinary GEMV's FMA and split reduction order.
template<int KWARPS>
__global__ void head_gemv_two(const __nv_bfloat16* x,const uint32_t* weight,
                             const __half* scales,__nv_bfloat16* out,
                             int n,int groups) {
    const int lane=threadIdx.x, part=threadIdx.y;
    const int col=blockIdx.x*32+lane;
    float sum0=0.0f, sum1=0.0f;
    for(int group=part;group<groups;group+=KWARPS) {
        const float sf=__half2float(scales[group*n+col]);
        #pragma unroll 4
        for(int word=0;word<32;word++) {
            const uint32_t bits=weight[(group*32+word)*n+col];
            #pragma unroll
            for(int byte=0;byte<4;byte++) {
                const float w=(float)((int)((bits>>(byte*8))&255)-128)*sf;
                const int offset=group*128+word*4+byte;
                sum0=fmaf(__bfloat162float(x[offset]),w,sum0);
                sum1=fmaf(__bfloat162float(x[2560+offset]),w,sum1);
            }
        }
    }
    __shared__ float partial0[KWARPS][32],partial1[KWARPS][32];
    partial0[part][lane]=sum0;
    partial1[part][lane]=sum1;
    __syncthreads();
    if(part==0) {
        float total0=0.0f,total1=0.0f;
        #pragma unroll
        for(int i=0;i<KWARPS;i++) {
            total0+=partial0[i][lane];
            total1+=partial1[i][lane];
        }
        out[col]=__float2bfloat16_rn(total0);
        out[n+col]=__float2bfloat16_rn(total1);
    }
}

extern "C" int gptq_head_gemv_two(const void* x,const void* weight,const void* scales,
                                  void* out,int k,int n,void* stream) {
    if(k!=2560 || n<=0 || n%32)return -1;
    head_gemv_two<8><<<n/32,dim3(32,8),0,(cudaStream_t)stream>>>(
        (const __nv_bfloat16*)x,(const uint32_t*)weight,
        (const __half*)scales,(__nv_bfloat16*)out,n,k/128);
    return (int)cudaGetLastError();
}
