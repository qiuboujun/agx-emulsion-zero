#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void reduce_sum_and_sq(const float* data, float* out, size_t size) {
    extern __shared__ float shared[];

    float* s_sum = shared;
    float* s_sq = &shared[blockDim.x];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;

    float val = (tid < size) ? data[tid] : 0.0f;
    s_sum[lane] = val;
    s_sq[lane] = val * val;

    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) {
            s_sum[lane] += s_sum[lane + s];
            s_sq[lane] += s_sq[lane + s];
        }
        __syncthreads();
    }

    if (lane == 0) {
        atomicAdd(&out[0], s_sum[0]);
        atomicAdd(&out[1], s_sq[0]);
    }
} 