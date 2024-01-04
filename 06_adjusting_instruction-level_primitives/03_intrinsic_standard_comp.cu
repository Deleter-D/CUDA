#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

__global__ void standardKernel(float a, float* out, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (int i = 0; i < iterations; i++)
        {
            tmp = powf(a, 2.0f);
        }

        *out = tmp;
    }
}

__global__ void intrinsicKernel(float a, float* out, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (int i = 0; i < iterations; i++)
        {
            tmp = __powf(a, 2.0f);
        }

        *out = tmp;
    }
}

int main(int argc, char const* argv[])
{
    setDevice();

    int nRuns        = 30;
    int nKernelIters = 1000;

    float *d_standard_out, h_standard_out;
    ERROR_CHECK(cudaMalloc((void**)&d_standard_out, sizeof(float)));

    float *d_intrinsic_out, h_intrinsic_out;
    ERROR_CHECK(cudaMalloc((void**)&d_intrinsic_out, sizeof(float)));

    float input_value = 4283.14;

    float mean_standard_time  = 0.0f;
    float mean_intrinsic_time = 0.0f;

    float elapase_time = 0.0f;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < nRuns; i++)
    {
        ERROR_CHECK(cudaEventRecord(start));
        standardKernel<<<1, 32>>>(input_value, d_standard_out, nKernelIters);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapase_time, start, stop));
        mean_standard_time += elapase_time;

        ERROR_CHECK(cudaEventRecord(start));
        intrinsicKernel<<<1, 32>>>(input_value, d_intrinsic_out, nKernelIters);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapase_time, start, stop));
        mean_intrinsic_time += elapase_time;
    }

    ERROR_CHECK(cudaMemcpy(&h_standard_out, d_standard_out, sizeof(float), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float), cudaMemcpyDeviceToHost));
    float host_value = powf(input_value, 2.0f);

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_standard_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n", host_value == h_standard_out ? "Yes" : "No", fabs(host_value - h_standard_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n", host_value == h_intrinsic_out ? "Yes" : "No", fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n", h_standard_out == h_intrinsic_out ? "Yes" : "No", fabs(h_standard_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard function powf:    %f ms\n", mean_standard_time);
    printf("Mean execution time for intrinsic function __powf: %f ms\n", mean_intrinsic_time);
    return 0;
}
