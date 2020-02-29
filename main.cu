#include <iostream>

#include <array>

#include <cuda_runtime.h>

__global__
void saxpy(float* a_ptr, float* b_ptr, int N) {
	const int thread_index = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (thread_index >= N) { return; }
	a_ptr[thread_index] = a_ptr[thread_index] + b_ptr[thread_index];
}

__global__ void reduce(float* a_dev_ptr) {
	extern __shared__ float shared_data[];
	
	const unsigned int tid = threadIdx.x;
	const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	
	shared_data[tid] = a_dev_ptr[i];
	__syncthreads();
	
	for (unsigned int s = 1; s < blockDim.x; s*=2) {
		if (tid % (2*s) == 0) {
			shared_data[tid] += shared_data[tid + s];
		}
		
		__syncthreads();
	}
	
	if (tid == 0) {
		a_dev_ptr[blockIdx.x] = shared_data[0];
	}
	
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

	
	reduce<<<10, 100, sizeof(float)*100>>>(reinterpret_cast<float*>(a_dev_ptr));
	
	cudaMemcpy(a.data(), a_dev_ptr, sizeof(a), cudaMemcpyDeviceToHost);
	
	
	for (const auto e: a) {
		std::cout << e << '\n';
	}
	
	float sum = 0;
	
	for (int i=0; i < 10; i++) {
		sum += a.at(i);
	}
	
	std::cout << "Total sum: " << sum << '\n';
	
	return 0;
}
