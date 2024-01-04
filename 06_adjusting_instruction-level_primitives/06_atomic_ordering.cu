#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

__global__ void atomics(int* shared_var, int* values_read, int size, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    values_read[tid] = atomicAdd(shared_var, 1);

    for (int i = 0; i < iterations; i++)
    {
        atomicAdd(shared_var, 1);
    }
}

__global__ void unsafe(int* shared_var, int* values_read, int size, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    int old          = *shared_var;
    *shared_var      = old + 1;
    values_read[tid] = old;

    for (int i = 0; i < iterations; i++)
    {
        old         = *shared_var;
        *shared_var = old + 1;
    }
}

int main(int argc, char const* argv[])
{
    int size        = 64;
    int nRuns       = 30;
    int nKernelIter = 100000;

    int *d_shared_var, h_shared_var_atomic, h_shared_var_unsafe;
    int *d_values_read_atomic, *d_values_read_unsafe, *h_value_read;
    ERROR_CHECK(cudaMalloc((void**)&d_shared_var, sizeof(int)));
    ERROR_CHECK(cudaMalloc((void**)&d_values_read_atomic, size * sizeof(int)));
    ERROR_CHECK(cudaMalloc((void**)&d_values_read_unsafe, size * sizeof(int)));
    h_value_read = (int*)malloc(size * sizeof(int));

    float atomic_mean_time = 0.0f, unsafe_mean_time = 0.0f;

    float elapsed_time = 0.0f;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < nRuns; i++)
    {
        ERROR_CHECK(cudaMemset(d_shared_var, 0, sizeof(int)));
        ERROR_CHECK(cudaEventRecord(start));
        atomics<<<2, 32>>>(d_shared_var, d_values_read_atomic, size, nKernelIter);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        atomic_mean_time += elapsed_time;
        ERROR_CHECK(cudaMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));

        ERROR_CHECK(cudaMemset(d_shared_var, 0, sizeof(int)));
        ERROR_CHECK(cudaEventRecord(start));
        unsafe<<<2, 32>>>(d_shared_var, d_values_read_unsafe, size, nKernelIter);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        unsafe_mean_time += elapsed_time;
        ERROR_CHECK(cudaMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int), cudaMemcpyDeviceToHost));
    }

    printf("In total, %d runs using atomic operations took %f ms\n", nRuns, atomic_mean_time);
    printf("  Using atomic operations also produced an output of %d\n", h_shared_var_atomic);
    printf("In total, %d runs using unsafe operations took %f ms\n", nRuns, unsafe_mean_time);
    printf("  Using unsafe operations also produced an output of %d\n", h_shared_var_unsafe);

    ERROR_CHECK(cudaMemcpy(h_value_read, d_values_read_atomic, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Threads performing atomic operations read values");
    for (int i = 0; i < 10; i++)
    {
        printf(" %d", h_value_read[i]);
    }
    printf("\n");

    ERROR_CHECK(cudaMemcpy(h_value_read, d_values_read_unsafe, 10 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("Threads performing unsafe operations read values");
    for (int i = 0; i < 10; i++)
    {
        printf(" %d", h_value_read[i]);
    }
    printf("\n");

    return 0;
}
