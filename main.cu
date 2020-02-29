#include <iostream>

#include <array>

#include <cuda_runtime.h>

__global__
void saxpy(float* a_ptr, float* b_ptr, int N) {
	const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (thread_index >= N) { return; }
	a_ptr[thread_index] = a_ptr[thread_index] + b_ptr[thread_index];
}

int main() {
	std::cout << "Hello, World!" << std::endl;
	
	constexpr int N = 1000;
	
	std::array<float, N> a = {};
	std::array<float, N> b = {};
	
	
	for (int i = 0; i < N; i++) {
		a.at(i) = static_cast<float>(i);
		b.at(i) = 42.0f;
	}
	
	void* a_dev_ptr = nullptr;
	void* b_dev_ptr = nullptr;
	cudaMalloc(&a_dev_ptr, sizeof(a));
	cudaMemcpy(a_dev_ptr, a.data(), sizeof(a), cudaMemcpyHostToDevice);
	cudaMalloc(&b_dev_ptr, sizeof(b));
	cudaMemcpy(b_dev_ptr, b.data(), sizeof(b), cudaMemcpyHostToDevice);
	
	saxpy<<<10, 100>>>(reinterpret_cast<float*>(a_dev_ptr), reinterpret_cast<float*>(b_dev_ptr), N);
	
	cudaMemcpy(a.data(), a_dev_ptr, sizeof(a), cudaMemcpyDeviceToHost);
	
	
	for (const auto e: a) {
		std::cout << e << '\n';
	}
	
	return 0;
}
