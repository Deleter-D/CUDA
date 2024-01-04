#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令编译fmad优化的PTX代码
    nvcc --ptx --fmad=true -o 04_manipulation_instruction_generation.ptx /path/to/04_manipulation_instruction_generation.cu

    使用如下命令编译禁用fmad优化的PTX代码
    nvcc --ptx --fmad=false -o 04_manipulation_instruction_generation.ptx /path/to/04_manipulation_instruction_generation.cu
*/

__global__ void division(float a, float b, float* c, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (int i = 0; i < iterations; i++)
        {
            tmp = a / b;
        }

        *c = tmp;
    }
}

__global__ void divisionIntrinsic(float a, float b, float* c, int iterations)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        float tmp;

        for (int i = 0; i < iterations; i++)
        {
            tmp = __fdividef(a, b);
        }

        *c = tmp;
    }
}

__global__ void fmad(float* ptr)
{
    *ptr = (*ptr) * (*ptr) + (*ptr);
}

int main(int argc, char const* argv[])
{
    setDevice();

    int nRuns        = 30;
    int nKernelIters = 1000;

    float *d_operator_out, h_operator_out;
    ERROR_CHECK(cudaMalloc((void**)&d_operator_out, sizeof(float)));

    float *d_intrinsic_out, h_intrinsic_out;
    ERROR_CHECK(cudaMalloc((void**)&d_intrinsic_out, sizeof(float)));

    float a = 4283.14f, b = 1.45f;

    float mean_standard_time  = 0.0f;
    float mean_intrinsic_time = 0.0f;

    float elapase_time = 0.0f;
    cudaEvent_t start, stop;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < nRuns; i++)
    {
        ERROR_CHECK(cudaEventRecord(start));
        division<<<1, 32>>>(a, b, d_operator_out, nKernelIters);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapase_time, start, stop));
        mean_standard_time += elapase_time;

        ERROR_CHECK(cudaEventRecord(start));
        divisionIntrinsic<<<1, 32>>>(a, b, d_intrinsic_out, nKernelIters);
        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        ERROR_CHECK(cudaEventElapsedTime(&elapase_time, start, stop));
        mean_intrinsic_time += elapase_time;
    }

    ERROR_CHECK(cudaMemcpy(&h_operator_out, d_operator_out, sizeof(float), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaMemcpy(&h_intrinsic_out, d_intrinsic_out, sizeof(float), cudaMemcpyDeviceToHost));
    float host_value = a / b;

    printf("Host calculated\t\t\t%f\n", host_value);
    printf("Standard Device calculated\t%f\n", h_operator_out);
    printf("Intrinsic Device calculated\t%f\n", h_intrinsic_out);
    printf("Host equals Standard?\t\t%s diff=%e\n", host_value == h_operator_out ? "Yes" : "No", fabs(host_value - h_operator_out));
    printf("Host equals Intrinsic?\t\t%s diff=%e\n", host_value == h_intrinsic_out ? "Yes" : "No", fabs(host_value - h_intrinsic_out));
    printf("Standard equals Intrinsic?\t%s diff=%e\n", h_operator_out == h_intrinsic_out ? "Yes" : "No", fabs(h_operator_out - h_intrinsic_out));
    printf("\n");
    printf("Mean execution time for standard operator /:\t\t%f ms\n", mean_standard_time);
    printf("Mean execution time for intrinsic function __fdividef:\t%f ms\n", mean_intrinsic_time);

    return 0;
}
