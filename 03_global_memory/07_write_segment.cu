#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令分析全局存储效率和全局存储事务
    sudo ncu --target-processes all -k sumArraysWriteOffset --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum /path/out/07_write_segment 0
    sudo ncu --target-processes all -k sumArraysWriteOffset --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum /path/out/07_write_segment 11
    sudo ncu --target-processes all -k sumArraysWriteOffset --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum /path/out/07_write_segment 128
*/

void sumArraysHost(float *A, float *B, float *C, const int size, int offset)
{
    for (int i = 0, j = offset; j < size; i++, j++)
        C[j] = A[i] + B[i];
}

__global__ void sumArraysWriteOffset(float *A, float *B, float *C, const int size, int offset)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = tid + offset;
    if (j < size)
        C[j] = A[tid] + B[tid];
}

int main(int argc, char const *argv[])
{
    setDevice();

    int size = 1 << 22;
    printf("Array size: %d\n", size);
    size_t bytes = size * sizeof(float);

    int offset = 0;
    if (argc > 1)
        offset = atoi(argv[1]);

    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    hostRef = (float *)malloc(bytes);
    gpuRef = (float *)malloc(bytes);

    initializeData<float>(h_A, size);
    initializeData<float>(h_B, size);

    sumArraysHost(h_A, h_B, hostRef, size, offset);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_B, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    sumArraysWriteOffset<<<grid, block>>>(d_A, d_B, d_C, size, offset);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    sumArraysWriteOffset<<<grid, block>>>(d_A, d_B, d_C, size, offset);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("writeOffset<<<%d, %d>>>\toffset %4d\telapsed %f ms\n", grid.x, block.x, offset, elapsedTime);

    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, size);

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}