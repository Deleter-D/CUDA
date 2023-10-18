#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"

__device__ float devData;

__global__ void checkGlobalVariable()
{
    printf("Device:\tthe value of the global variable is %f\n", devData);
    devData += 2.0f;
}

int main(int argc, char const *argv[])
{
    setDevice();

    float value = 3.14f;
    ERROR_CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:\tcopied %f to the global variable\n", value);

    checkGlobalVariable<<<1, 1>>>();

    ERROR_CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:\tthe value changed by the kernel to %f\n", value);

    ERROR_CHECK(cudaDeviceReset());
    return 0;
}
