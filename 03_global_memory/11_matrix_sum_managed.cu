#include <stdio.h>
#include <cuda_runtime.h>
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
    printf("Matrix size: %d x %d\tTotal: %d\n\n", nx, ny, nx * ny);

    cudaEvent_t start, stop;
    float elapsedTime;
    ERROR_CHECK(cudaEventCreate(&start));
    ERROR_CHECK(cudaEventCreate(&stop));
    printf("tasks\t\ttime\n");

    float *A, *B, *hostRef, *gpuRef;
    ERROR_CHECK(cudaMallocManaged((void **)&A, bytes));
    ERROR_CHECK(cudaMallocManaged((void **)&B, bytes));
    ERROR_CHECK(cudaMallocManaged((void **)&hostRef, bytes));
    ERROR_CHECK(cudaMallocManaged((void **)&gpuRef, bytes));

    ERROR_CHECK(cudaEventRecord(start));
    initializeData<float>(A, nx * ny);
    initializeData<float>(B, nx * ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("initData\t%f ms\n", elapsedTime);

    memset(hostRef, 0, bytes);
    memset(gpuRef, 0, bytes);

    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixOnHost(A, B, hostRef, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("cpuSum\t\t%f ms\n", elapsedTime);

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // 预热
    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, 1, 1);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    ERROR_CHECK(cudaEventRecord(start));
    sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
    ERROR_CHECK(cudaEventRecord(stop));
    ERROR_CHECK(cudaEventSynchronize(stop));
    ERROR_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("sum matrix managed<<<(%d, %d), (%d, %d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, elapsedTime);

    ERROR_CHECK(cudaDeviceSynchronize());

    checkResult<float>(hostRef, gpuRef, nx * ny);

    ERROR_CHECK(cudaFree(A));
    ERROR_CHECK(cudaFree(B));
    ERROR_CHECK(cudaFree(hostRef));
    ERROR_CHECK(cudaFree(gpuRef));

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
