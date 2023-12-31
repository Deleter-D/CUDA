#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析代码
    nsys nvprof /path/to/03_page_locked_memory
*/

int main(int argc, char const *argv[])
{
    setDevice();

    unsigned int size  = 1 << 22;
    unsigned int bytes = size * sizeof(float);

    float *h_a;
    ERROR_CHECK(cudaMallocHost((void **)&h_a, bytes));

    initializeData<float>(h_a, size);

    float *d_a;
    ERROR_CHECK(cudaMalloc((void **)&d_a, bytes));

    ERROR_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    ERROR_CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));

    ERROR_CHECK(cudaFree(d_a));
    ERROR_CHECK(cudaFreeHost(h_a));

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
