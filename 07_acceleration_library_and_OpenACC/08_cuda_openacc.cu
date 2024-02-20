#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

int main(int argc, char const *argv[])
{
    setDevice();

    int M = 1024;
    int N = 1024;
    int P = 1024;

    float *__restrict__ d_A;
    float *__restrict__ d_B;
    float *__restrict__ d_C;

    float *d_row_sum;
    float total_sum;

    curandGenerator_t rand_state = 0;
    ERROR_CHECK_CURAND(curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT));

    cublasHandle_t cublas_handle = 0;
    ERROR_CHECK_CUBLAS(cublasCreate(&cublas_handle));

    ERROR_CHECK(cudaMalloc((void **)&d_A, M * N * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_B, N * P * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_C, M * P * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_row_sum, M * sizeof(float)));

    ERROR_CHECK_CURAND(curandGenerateUniform(rand_state, d_A, M * N));
    ERROR_CHECK_CURAND(curandGenerateUniform(rand_state, d_B, N * P));

#pragma acc parallel loop gang deviceptr(d_A, d_B, d_C)
    for (int i = 0; i < M; i++)
    {
#pragma acc loop worker vector
        for (int j = 0; j < P; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
            {
                sum += d_A[i * N + k] * d_B[k * P + j];
            }
            d_C[i * P + j] = sum;
        }
    }

    ERROR_CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    for (int i = 0; i < M; i++)
    {
        ERROR_CHECK_CUBLAS(cublasSasum(cublas_handle, P, d_C + (i * P), 1, d_row_sum + i));
    }

    ERROR_CHECK_CUBLAS(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    ERROR_CHECK_CUBLAS(cublasSasum(cublas_handle, M, d_row_sum, 1, &total_sum));
    ERROR_CHECK(cudaDeviceSynchronize());

    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));
    ERROR_CHECK(cudaFree(d_row_sum));

    printf("Total sum = %f\n", total_sum);

    return 0;
}
