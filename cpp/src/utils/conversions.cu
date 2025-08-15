#include "conversions.hpp"
#include "NumCpp.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

namespace agx {
namespace utils {

namespace {

struct D2LParams { int rows, cols, lrows, lcols, lsize; };

__global__ void k_density_to_light(const float* dens,
				   const float* light,
				   float* out,
				   D2LParams p)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = p.rows * p.cols;
	if (idx >= total) return;

	int r = idx / p.cols;
	int c = idx % p.cols;

	double t = pow(10.0, -(double)dens[idx]);
	double L = 1.0;

	if (p.lsize == 1) {
		L = (double)light[0];
	} else if (p.lrows == p.rows && p.lcols == p.cols) {
		L = (double)light[idx];
	} else if (p.lsize == p.cols || (p.lrows == 1 && p.lcols == p.cols) || (p.lcols == 1 && p.lrows == p.cols)) {
		L = (double)light[c];
	} else if (p.lsize == p.rows || (p.lrows == p.rows && p.lcols == 1) || (p.lcols == p.rows && p.lrows == 1)) {
		L = (double)light[r];
	} else if (p.lsize > 1 && (p.cols % p.lsize == 0)) {
		int blockSize = p.lsize;
		int k = c % blockSize;
		L = (double)light[k];
	} else if (total == p.lsize) {
		L = (double)light[idx];
	} else {
		L = (double)light[0];
	}

	out[idx] = (float)(t * L);
}

} // namespace

bool density_to_light_gpu(const nc::NdArray<float>& density,
			  const nc::NdArray<float>& light,
			  nc::NdArray<float>& out)
{
	const auto rows = density.shape().rows;
	const auto cols = density.shape().cols;
	out = nc::NdArray<float>(rows, cols);

	const float* d_host = density.data();
	const float* l_host = light.data();
	float* o_host = out.data();

	size_t d_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
	size_t l_count = static_cast<size_t>(light.shape().rows) * static_cast<size_t>(light.shape().cols);

	float *d_d = nullptr, *d_l = nullptr, *d_o = nullptr;
	cudaError_t err;

	if ((err = cudaMalloc(&d_d, d_count * sizeof(float))) != cudaSuccess) {
		std::cout << "CUDA malloc d_d failed: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	if ((err = cudaMalloc(&d_l, l_count * sizeof(float))) != cudaSuccess) {
		std::cout << "CUDA malloc d_l failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d);
		return false;
	}
	if ((err = cudaMalloc(&d_o, d_count * sizeof(float))) != cudaSuccess) {
		std::cout << "CUDA malloc d_o failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l);
		return false;
	}

	if ((err = cudaMemcpy(d_d, d_host, d_count * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
		std::cout << "CUDA memcpy d_d failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
		return false;
	}
	if ((err = cudaMemcpy(d_l, l_host, l_count * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
		std::cout << "CUDA memcpy d_l failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
		return false;
	}

	const int N = static_cast<int>(rows * cols);
	int threads = 256;
	int blocks = (N + threads - 1) / threads;

	D2LParams p{(int)rows, (int)cols,
		   (int)light.shape().rows, (int)light.shape().cols,
		   (int)l_count};

	// Debug inputs
	std::cout << "CUDA-DEBUG: Input validation before kernel launch:" << std::endl;
	std::cout << "  density shape: " << rows << "x" << cols << " (" << d_count << " elements)" << std::endl;
	std::cout << "  light shape: " << light.shape().rows << "x" << light.shape().cols << " (" << l_count << " elements)" << std::endl;
	std::cout << "  density values [0-2]: ";
	for (int i = 0; i < ((int)d_count < 3 ? (int)d_count : 3); ++i) std::cout << d_host[i] << " ";
	std::cout << std::endl;
	std::cout << "  light values [0-2]: ";
	for (int i = 0; i < ((int)l_count < 3 ? (int)l_count : 3); ++i) std::cout << l_host[i] << " ";
	std::cout << std::endl;
	std::cout << "  kernel params: N=" << N << ", blocks=" << blocks << ", threads=" << threads << std::endl;

	k_density_to_light<<<blocks, threads>>>(d_d, d_l, d_o, p);
	if ((err = cudaGetLastError()) != cudaSuccess) {
		std::cout << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
		return false;
	}
	if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
		std::cout << "CUDA kernel sync failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
		return false;
	}

	if ((err = cudaMemcpy(o_host, d_o, d_count * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
		std::cout << "CUDA memcpy result failed: " << cudaGetErrorString(err) << std::endl;
		cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
		return false;
	}

	// Debug outputs
	std::cout << "CUDA-DEBUG: Output validation after kernel:" << std::endl;
	std::cout << "  output values [0-2]: ";
	for (int i = 0; i < ((int)d_count < 3 ? (int)d_count : 3); ++i) std::cout << o_host[i] << " ";
	std::cout << std::endl;

	cudaFree(d_d); cudaFree(d_l); cudaFree(d_o);
	return true;
}

bool density_to_light_cuda(const nc::NdArray<float>& density,
			   const nc::NdArray<float>& light,
			   nc::NdArray<float>& out)
{
	std::cout << "CUDA-DEBUG: density_to_light_cuda called with density "
		  << density.shape().rows << "x" << density.shape().cols
		  << " and light " << light.shape().rows << "x" << light.shape().cols << std::endl;
	return density_to_light_gpu(density, light, out);
}

} // namespace utils
} // namespace agx


