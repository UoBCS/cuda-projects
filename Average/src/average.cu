#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__host__ __device__ int h_d_max(int a, int b)
{
	return a > b ? a : b;
}

__host__ __device__ float h_d_average(const float *vec, int n)
{
	float sum = 0;
	int i;

	for (i = 0; i < n; i++)
		sum += vec[i];

	return sum / n;
}

void h_win_average(const float *A, float **B, size_t size, size_t n)
{
	int i;
	float v_els[n];

	for (i = 0; i < size; i++) {
		int j;
		for (j = 0; j < n; j++)
			v_els[j] = A[h_d_max(0, i - n + j)];

		(*B)[i] = h_d_average(v_els, n);
	}
}

__device__ int get_global_idx_2d_2d()
{
	int block_id = blockIdx.x + blockIdx.y * gridDim.x;
	int thread_id = block_id * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return thread_id;
}

__device__ void d_average(const float *g_idata, float *g_odata, unsigned int size)
{
	extern __shared__ float s_data[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	s_data[tid] = g_idata[i];
	__syncthreads();

	unsigned int s;
	for (s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			s_data[tid] += s_data[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
		g_odata[blockIdx.x] = s_data[0] / size;
}

__global__ void d_naive_win_average(const float *A, float *B, size_t n_els, size_t n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x; // get_global_idx_2d_2d();
	float sum = 0;

	if (i < n_els) {
		int j;
		for (j = 0; j < n; j++) {
			__syncthreads();
			sum += A[h_d_max(0, i - n + j)];
		}

		B[i] = sum / n; //h_average(v_els, n);
	}
}


int main(void)
{
	cudaError_t err = cudaSuccess;
	int win_size = 2;
	int num_el = 100000;
	size_t size = num_el * sizeof(float);
	float *h_vec = (float*)malloc(size);

	// Timer
	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (h_vec == NULL) {
		fprintf(stderr, "Failed to allocate host vector!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vector
	for (int i = 0; i < num_el; ++i)
		h_vec[i] = rand() / (float)RAND_MAX;

	// Run host version
	float *h_avg = (float*)malloc(sizeof(float) * num_el); //[num_el];
	h_win_average(h_vec, &h_avg, num_el, win_size);

	printf("HOST VERSION:\n");
	int i;
//	for (i = 0; i < num_el; i++)
//		printf("%f ", h_vec[i]);

	printf("\n");

//	for (i = 0; i < num_el; i++)
//		printf("%f ", h_avg[i]);

	printf("\n");

	// ===============================================================================================================================

	printf("DEVICE NAIVE VERSION:\n");
	// Allocate the device input vector h_vec
	float *d_vec = NULL;
	err = cudaMalloc((void **)&d_vec, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector h_vec (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	float *d_avg = NULL;
	err = cudaMalloc((void **)&d_avg, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector h_avg (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector h_vec from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int threads_per_block = 256;
	int blocks_per_grid = 1 + ((num_el - 1) / threads_per_block);

	cudaEventRecord(start, 0);
	d_naive_win_average<<<blocks_per_grid, threads_per_block>>>(d_vec, d_avg, num_el, win_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy the device result vector in device memory to the host result vector in host memory.
	err = cudaMemcpy(h_avg, d_avg, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector d_avg from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

//	for (i = 0; i < num_el; i++)
//		printf("%f ", h_avg[i]);

	printf("\n");
	printf("Time elapsed for naive parallel implementation: %f\n", time);

	// --------------------------------------------------

	cudaEventRecord(start, 0);
	d_naive_win_average<<<blocks_per_grid, threads_per_block>>>(d_vec, d_avg, num_el, win_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy the device result vector in device memory to the host result vector in host memory.
	err = cudaMemcpy(h_avg, d_avg, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector d_avg from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

//	for (i = 0; i < num_el; i++)
//		printf("%f ", h_avg[i]);

	printf("\n");
	printf("Time elapsed for naive parallel implementation: %f\n", time);


	// ===============================================================================================================================

	/*dim3 threads_per_block_2d(32 , 32);
	dim3 blocks_per_grid_2d(num_el / threads_per_block_2d.x, num_el / threads_per_block_2d.y);

	printf("x: %d, y: %d\n\n", blocks_per_grid_2d.x, blocks_per_grid_2d.y);

	cudaEventRecord(start, 0);
	d_naive_win_average<<<threads_per_block_2d , blocks_per_grid_2d>>>(d_vec, d_avg, num_el, win_size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Copy the device result vector in device memory to the host result vector in host memory.
	err = cudaMemcpy(h_avg, d_avg, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to copy vector d_avg from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

//	for (i = 0; i < num_el; i++)
//		printf("%f ", h_avg[i]);

	printf("\n");
	printf("Time elapsed for naive parallel implementation 2D: %f\n", time);*/

	return 0;
}
