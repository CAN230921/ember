// GB10 fixed shared expert: explicit BF16 materialization boundaries.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

__global__ void silu_mul(const __nv_bfloat16* input, __nv_bfloat16* output, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 640) return;
    const int row = i / 640, col = i % 640;
    const float gate = __bfloat162float(input[row * 1280 + col]);
    const float up = __bfloat162float(input[row * 1280 + 640 + col]);
    const float silu = gate / (1.0f + expf(-gate));
    const float rounded = __bfloat162float(__float2bfloat16_rn(silu));
    output[i] = __float2bfloat16_rn(rounded * up);
}

__global__ void finalize_shared(const __nv_bfloat16* routed, const __nv_bfloat16* shared,
    const __nv_bfloat16* gate, __nv_bfloat16* output, int rows, int gate_columns) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 2560) return;
    const float logit = __bfloat162float(gate[gate_columns == 1 ? i / 2560 : i]);
    const float sigmoid = 1.0f / (1.0f + expf(-logit));
    const float narrowed = __bfloat162float(__float2bfloat16_rn(sigmoid));
    const float product = __bfloat162float(__float2bfloat16_rn(__bfloat162float(shared[i]) * narrowed));
    output[i] = __float2bfloat16_rn(__bfloat162float(routed[i]) + product);
}

extern "C" int sm121_shared_silu(const void* input, void* output, int rows, void* stream) {
    if (rows < 1 || rows > 8) return -1;
    silu_mul<<<(rows * 640 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, rows);
    return (int)cudaGetLastError();
}

extern "C" int sm121_shared_finalize(const void* routed, const void* shared,
    const void* gate, void* output, int rows, int gate_columns, void* stream) {
    if (rows < 1 || rows > 8 || (gate_columns != 1 && gate_columns != 2560)) return -1;
    finalize_shared<<<(rows * 2560 + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)routed, (const __nv_bfloat16*)shared,
        (const __nv_bfloat16*)gate, (__nv_bfloat16*)output, rows, gate_columns);
    return (int)cudaGetLastError();
}
