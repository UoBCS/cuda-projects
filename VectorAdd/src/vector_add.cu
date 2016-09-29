#include <memory>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <cuda_runtime.h>

float random_at_most(long max) {
	unsigned long
	// max <= RAND_MAX < ULONG_MAX, so this is okay.
	num_bins = (unsigned long) max + 1, num_rand = (unsigned long) RAND_MAX + 1,
			bin_size = num_rand / num_bins, defect = num_rand % num_bins;

	long x;
	do {
		x = random();
	}
	// This is carefully written not to overflow
	while (num_rand - defect <= (unsigned long) x);

	// Truncated division is intentional
	return (float) (x / bin_size);
}

void generate_vector(float *vector, int sz) {
	//float *vector; //= (float*) malloc(sizeof(float) * sz);
	//cudaMalloc( (void **)&vector, sizeof(float)*sz );

	for (int i = 0; i < sz; i++)
		vector[i] = random_at_most(10);
}

void print_vector(const float *v, int sz) {
	for (int i = 0; i < sz; i++)
		printf("%f, ", v[i]);
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("%d\n, ", i);
	if (i < n) {
		C[i] = A[i] + B[i];
	}
}

__global__ void printa()
{
	printf("helloooo\n\n");
}

int main(int argc, char **argv) {
	int numElements = 2000;
	float *d_A, *d_B, *d_C;
	float a[numElements], b[numElements], c[numElements];
	int threadsPerBlock = 256;
	int blocksPerGrid = 1 + (numElements - 1) / threadsPerBlock;

	generate_vector(a, numElements);
	generate_vector(b, numElements);

	cudaMalloc( (void **)&d_A, sizeof(float)*numElements );
	cudaMalloc( (void **)&d_B, sizeof(float)*numElements );
	cudaMalloc( (void **)&d_C, sizeof(float)*numElements );

	cudaMemcpy( d_A, a, sizeof(float)*numElements, cudaMemcpyHostToDevice );
	cudaMemcpy( d_B, b, sizeof(float)*numElements, cudaMemcpyHostToDevice );

	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

	cudaMemcpy( c, d_C, sizeof(float)*numElements, cudaMemcpyDeviceToHost );

	print_vector(a, numElements);
	printf("\n\n");
	print_vector(b, numElements);
	printf("\n\n");
	print_vector(c, numElements);
	printf("\n\n");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
