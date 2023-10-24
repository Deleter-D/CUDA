#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析全局加载和存储效率
    sudo ncu --target-processes all -k testInnerArray --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/09_structure_of_array
    sudo ncu --target-processes all -k testInnerArray --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/09_structure_of_array
    sudo ncu --target-processes all -k testInnerArray --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct /path/09_structure_of_array
*/

#define LEN 1 << 20

// SoA排布方式
struct innerArray
{
    float x[LEN];
    float y[LEN];
};

// 主机端操作
void testInnerArrayHost(innerArray *data, innerArray *result, const int size)
{
    for (int i = 0; i < size; i++)
    {
        result->x[i] = data->x[i] + 10.f;
        result->y[i] = data->y[i] + 20.f;
    }
}

// 相同操作的核函数
__global__ void testInnerArray(innerArray *data, innerArray *result, const int size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        float tempx = data->x[idx];
        float tempy = data->y[idx];

        tempx += 10.f;
        tempy += 20.f;

        result->x[idx] = tempx;
        result->y[idx] = tempy;
    }
}

void initilalizeInnerArray(innerArray *ptr, int size)
{
    for (int i = 0; i < size; i++)
    {
        ptr->x[i] = (float)(rand() & 0xFF) / 100.0f;
        ptr->y[i] = (float)(rand() & 0xFF) / 100.0f;
    }
}

void checkInnerArray(innerArray *hostRef, innerArray *gpuRef, const int size)
{
    double epsilon = 1.0E-8;
    bool match = true;
    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef->x[i] - gpuRef->x[i]) > epsilon)
        {
            match = false;
            printf("different on x %dth element: host %f gpu %f\n", i, hostRef->x[i], gpuRef->x[i]);
            break;
        }

        if (abs(hostRef->y[i] - gpuRef->y[i]) > epsilon)
        {
            match = false;
            printf("different on y %dth element: host %f gpu %f\n", i, hostRef->y[i], gpuRef->y[i]);
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
    size_t bytes = sizeof(innerArray);

    innerArray *h_A, *hostRef, *gpuRef;
    h_A = (innerArray *)malloc(bytes);
    hostRef = (innerArray *)malloc(bytes);
    gpuRef = (innerArray *)malloc(bytes);

    initilalizeInnerArray(h_A, size);

    testInnerArrayHost(h_A, hostRef, size);

    innerArray *d_A, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    int blocksize = 128;
    if (argc > 1)
        blocksize = atoi(argv[1]);

    dim3 block(blocksize);
    dim3 grid((size + block.x - 1) / block.x);

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    testInnerArray<<<grid, block>>>(d_A, d_C, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(start));
    testInnerArray<<<grid, block>>>(d_A, d_C, size);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("innerArray<<<%d, %d>>>\telapsed %f ms\n", grid.x, block.x, elapsedTime);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkInnerArray(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_C));
    free(h_A);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
