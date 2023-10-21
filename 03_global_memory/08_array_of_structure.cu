#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析全局加载和存储效率
    sudo ncu --target-processes all -k testInnerStruct --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/out/08_array_of_structure
    sudo ncu --target-processes all -k testInnerStruct --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/out/08_array_of_structure
    sudo ncu --target-processes all -k testInnerStruct --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/out/08_array_of_structure
*/

#define LEN 1 << 20

struct innerStruct
{
    float x;
    float y;
};

void testInnerStructHost(innerStruct *data, innerStruct *result, const int size)
{
    for (int i = 0; i < size; i++)
    {
        innerStruct temp = data[i];
        temp.x += 10.f;
        temp.y += 20.f;
        result[i] = temp;
    }
}

__global__ void testInnerStruct(innerStruct *data, innerStruct *result, const int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        innerStruct temp = data[idx];
        temp.x += 10.f;
        temp.y += 20.f;
        result[idx] = temp;
    }
}

void initializeInnerStruct(innerStruct *ptr, const int size)
{
    for (int i = 0; i < size; i++)
    {
        ptr[i].x = (float)(rand() & 0xFF) / 100.0f;
        ptr[i].y = (float)(rand() & 0xFF) / 100.0f;
    }
}

void checkInnerStruct(innerStruct *hostRef, innerStruct *gpuRef, const int size)
{
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i].x - gpuRef[i].x) > epsilon)
        {
            match = false;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
            break;
        }
        if (abs(hostRef[i].y - gpuRef[i].y) > epsilon)
        {
            match = false;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i].x, gpuRef[i].x);
            break;
        }
    }
    if (!match)
        printf("Arrays do not match.\n\n");
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size = LEN;
    size_t bytes = size * sizeof(innerStruct);

    innerStruct *h_A, *hostRef, *gpuRef;
    h_A = (innerStruct *)malloc(bytes);
    hostRef = (innerStruct *)malloc(bytes);
    gpuRef = (innerStruct *)malloc(bytes);

    initializeInnerStruct(h_A, size);

    testInnerStructHost(h_A, hostRef, size);

    innerStruct *d_A, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    ERROR_CHECK(cudaEventRecord(start));
    warmupKernelDo();
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    int blocksize = 128;
    if (argc > 1)
        blocksize = atoi(argv[1]);

    dim3 block(blocksize);
    dim3 grid((size + block.x - 1) / block.x);

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    testInnerStruct<<<grid, block>>>(d_A, d_C, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("innerStruct<<<%d, %d>>>\telapsed %f ms\n", grid.x, block.x, elapsedTime);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkInnerStruct(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
