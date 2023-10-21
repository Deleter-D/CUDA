#include <stdio.h>
#include <cuda_runtime.h>
#include "../utils/common.cuh"
#include "../utils/data.cuh"

/*
    使用如下命令编译代码
    nvcc -O3 /path/to/03_parallelism.cu -o /path/to/out/03_parallelism -arch sm_89

    使用ncu分析不同block设计下的占用率：
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics sm__warps_active.avg.pct_of_peak_sustained_active /path/to/out/03_parallelism 32 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics sm__warps_active.avg.pct_of_peak_sustained_active /path/to/out/03_parallelism 32 16
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics sm__warps_active.avg.pct_of_peak_sustained_active /path/to/out/03_parallelism 16 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics sm__warps_active.avg.pct_of_peak_sustained_active /path/to/out/03_parallelism 16 16

    使用ncu分析不同block设计下的内存读取效率：
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /path/to/out/03_parallelism 32 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /path/to/out/03_parallelism 32 16
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /path/to/out/03_parallelism 16 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second /path/to/out/03_parallelism 16 16

    使用ncu分析不同block设计下的全局加载效率：
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct /path/to/out/03_parallelism 32 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct /path/to/out/03_parallelism 32 16
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct /path/to/out/03_parallelism 16 32
    sudo ncu --target-processes all -k "sumMatrixOnGPU2D" --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct /path/to/out/03_parallelism 16 16
*/

__global__ void sumMatrixOnGPU2D(float *A, float *B, float *C, int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(int argc, char const *argv[])
{
    int dimx, dimy;
    if (argc != 3)
    {
        printf("argument invalid!\n\tUsage: parallelism blockDim.x blockDim.y\n");
        return 1;
    }
    else
    {
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }

    setDevice();

    int nx = 1 << 14; // 16384
    int ny = 1 << 14; // 16384

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nx * ny * sizeof(float));
    h_B = (float *)malloc(nx * ny * sizeof(float));
    h_C = (float *)malloc(nx * ny * sizeof(float));

    initializeData<float>(h_A, nx * ny);
    initializeData<float>(h_B, nx * ny);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, nx * ny * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_B, nx * ny * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_C, nx * ny * sizeof(float)));

    ERROR_CHECK(cudaMemcpy(d_A, h_A, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    warmupKernelDo();
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("sumMatrixOnGPU2D <<<(%d, %d), (%d, %d)>>> elapsed %g ms\n", grid.x, grid.y, block.x, block.y, elapsedTime);

    ERROR_CHECK(cudaMemcpy(h_C, d_C, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();

    return 0;
}
