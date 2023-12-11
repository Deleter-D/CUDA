#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixGPU(float *A, float *B, float *C, const int nx, const int ny)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char const *argv[])
{
    setDevice();

    int nx, ny;
    nx = ny = 1 << 12;
    int bytes = nx * ny * sizeof(float);
    printf("Matrix size: %d x %d\tTotal: %d\n", nx, ny, nx * ny);

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));
    printf("tasks\t\ttime\n");

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(bytes);
    h_B = (float *)malloc(bytes);
    hostRef = (float *)malloc(bytes);
    gpuRef = (float *)malloc(bytes);

    ERROR_CHECK(cudaEventRecord(start));
    initializeData<float>(h_A, nx * ny);
    initializeData<float>(h_B, nx * ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("initData\t%f ms\n", elapsedTime);

    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("cpuSum\t\t%f ms\n", elapsedTime);

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_B, bytes));
    ERROR_CHECK(cudaMalloc((void **)&d_C, bytes));

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("memcpyHtD\t%f ms\n", elapsedTime);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixGPU<<<grid, block>>>(d_A, d_B, d_C, 1, 1);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    memset(gpuRef, 0, bytes);
    ERROR_CHECK(cudaMemset(d_C, 0, bytes));

    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixGPU<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("sum matrix manual<<<(%d, %d), (%d, %d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, elapsedTime);

    ERROR_CHECK(cudaEventRecord(start));
    ERROR_CHECK(cudaMemcpy(gpuRef, d_C, bytes, cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("memcpyDtH\t%f ms\n", elapsedTime);

    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, nx * ny);

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