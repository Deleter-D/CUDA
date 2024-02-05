#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

// 行主序的GEMM
void hostMMRow(const void *alpha, float *matA, float *matB, const void *beta, float *matC,
               const int n1, const int n3, const int n2)
{
    float alpha_f32 = *((float *)alpha);
    float beta_f32  = *((float *)beta);
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n3; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < n2; k++)
            {
                sum += matA[i * n2 + k] * matB[k * n3 + j];
            }
            matC[i * n3 + j] = alpha_f32 * sum + beta_f32 * matC[i * n3 + j];
        }
    }
}

// 列主序的GEMM
void hostMMCol(const void *alpha, float *matA, float *matB, const void *beta, float *matC,
               const int n1, const int n3, const int n2)
{
    float alpha_val = *((float *)alpha);
    float beta_val  = *((float *)beta);

    for (int j = 0; j < n3; j++)
    {
        for (int i = 0; i < n1; i++)
        {
            float sum = 0.0;
            for (int k = 0; k < n2; k++)
            {
                sum += matA[k * n1 + i] * matB[j * n2 + k];
            }
            matC[j * n1 + i] = alpha_val * sum + beta_val * matC[j * n1 + i];
        }
    }
}

int main(int argc, char const *argv[])
{
    setDevice();

    // A矩阵m * k，B矩阵k * n
    int m = 1024;
    int k = 512;
    int n = 1024;
    // A、B、C三个矩阵的leading_dimension
    int lda = m;
    int ldb = k;
    int ldc = m;

    float alpha = 3.0f;
    float beta  = 4.0f;

    float *A, *B, *C; // 矩阵A、B、C
    A = (float *)malloc(m * k * sizeof(float));
    B = (float *)malloc(k * n * sizeof(float));
    C = (float *)malloc(m * n * sizeof(float));
    initializeData<float>(A, m * k);
    initializeData<float>(B, k * n);
    initializeData<float>(C, m * n);

    float *host_C = (float *)malloc(m * n * sizeof(float));
    memcpy(host_C, C, m * n * sizeof(float));

    float *d_A, *d_B, *d_C;
    ERROR_CHECK(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

    ERROR_CHECK(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_C, C, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    ERROR_CHECK_CUBLAS(cublasCreate(&handle));

    // 执行GEMM计算
    ERROR_CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));

    // 销毁handle
    ERROR_CHECK_CUBLAS(cublasDestroy(handle));

    // 错误检查
    ERROR_CHECK(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    hostMMCol(&alpha, A, B, &beta, host_C, m, n, k);
    checkResult<float>(host_C, C, m * n, 2);

    // 资源释放
    ERROR_CHECK(cudaFree(d_A));
    ERROR_CHECK(cudaFree(d_B));
    ERROR_CHECK(cudaFree(d_C));
    free(host_C);
    free(A);
    free(B);
    free(C);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
