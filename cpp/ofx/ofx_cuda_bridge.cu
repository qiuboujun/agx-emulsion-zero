#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void ofxExposureKernel(const float* __restrict__ inRGBA,
                                              float* __restrict__ outRGBA,
                                              int width,
                                              int height,
                                              float gain)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * 4;
    float r = inRGBA[idx + 0];
    float g = inRGBA[idx + 1];
    float b = inRGBA[idx + 2];
    float a = inRGBA[idx + 3];
    outRGBA[idx + 0] = r * gain;
    outRGBA[idx + 1] = g * gain;
    outRGBA[idx + 2] = b * gain;
    outRGBA[idx + 3] = a;
}

extern "C" void RunCudaOfxBridge(int width, int height, float exposureEV,
                                 const float* hostInRGBA, float* hostOutRGBA)
{
    const size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t numFloats = numPixels * 4;
    const size_t bytes = numFloats * sizeof(float);

    float* d_in = nullptr;
    float* d_out = nullptr;

    cudaError_t err = cudaMalloc(&d_in, bytes);
    if (err != cudaSuccess) return;
    err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) { cudaFree(d_in); return; }

    err = cudaMemcpy(d_in, hostInRGBA, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return; }

    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x,
                (height + threads.y - 1) / threads.y);

    float gain = powf(2.0f, exposureEV);
    ofxExposureKernel<<<blocks, threads>>>(d_in, d_out, width, height, gain);
    cudaDeviceSynchronize();

    err = cudaMemcpy(hostOutRGBA, d_out, bytes, cudaMemcpyDeviceToHost);
    (void)err;
    cudaFree(d_in);
    cudaFree(d_out);
}
