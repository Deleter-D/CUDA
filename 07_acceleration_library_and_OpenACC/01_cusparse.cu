#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <stdio.h>

#include "../utils/common.cuh"
#include "../utils/data.cuh"

void hostMV(const void *alpha, float *matA, float *vecX, const void *beta, float *vecY, int rows, int cols)
{
    float *alpha_f32 = (float *)alpha;
    float *beta_f32  = (float *)beta;
    printf("Host:  alpha: %f, beta: %f\n", *alpha_f32, *beta_f32);
    float *temp = (float *)malloc(rows * sizeof(float));
    memset(temp, 0, rows * sizeof(float));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            temp[i] += (*alpha_f32) * matA[i * rows + j] * vecX[j];
        }
        temp[i] += (*beta_f32) * vecY[i];
    }
    memcpy(vecY, temp, rows * sizeof(float));
    free(temp);
}

int main(int argc, char const *argv[])
{
    setDevice();

    int rows    = 1024;    // 矩阵行数
    int columns = 1024;    // 矩阵列数
    int ld      = columns; // 矩阵leading_dimension

    // 本示例采用CSR格式存储稀疏矩阵

    float *A, *X, *Y; // 矩阵A，向量X、Y
    A = (float *)malloc(rows * columns * sizeof(float));
    X = (float *)malloc(columns * sizeof(float));
    Y = (float *)malloc(rows * sizeof(float));
    initializeDataSparse<float>(A, rows * columns);
    initializeData<float>(X, columns);
    initializeData<float>(Y, rows);

    float *host_Y = (float *)malloc(rows * sizeof(float)); // 存储主机端结果
    memcpy(host_Y, Y, rows * sizeof(float));

    float *d_A, *d_X, *d_Y;
    int *d_row_offsets_A; // CSR中的行偏移数组
    ERROR_CHECK(cudaMalloc((void **)&d_A, rows * columns * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_X, columns * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_Y, rows * sizeof(float)));
    ERROR_CHECK(cudaMalloc((void **)&d_row_offsets_A, (rows + 1) * sizeof(int)));

    ERROR_CHECK(cudaMemcpy(d_A, A, rows * columns * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_X, X, columns * sizeof(float), cudaMemcpyHostToDevice));
    ERROR_CHECK(cudaMemcpy(d_Y, Y, rows * sizeof(float), cudaMemcpyHostToDevice));

    // 创建cuSPARSE句柄
    cusparseHandle_t handle;
    ERROR_CHECK_CUSPARSE(cusparseCreate(&handle));

    // 创建稠密矩阵
    cusparseDnMatDescr_t dn_mat;
    ERROR_CHECK_CUSPARSE(cusparseCreateDnMat(&dn_mat, rows, columns, ld, d_A, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // 创建稀疏矩阵
    cusparseSpMatDescr_t sp_mat;
    ERROR_CHECK_CUSPARSE(cusparseCreateCsr(&sp_mat, rows, columns, 0, d_row_offsets_A, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // 若有必要，为转换工作申请额外的buffer
    size_t buffer_size = 0;
    void *d_buffer     = NULL;
    ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size)); // 该函数返回所需的buffer大小
    printf("%ld bytes external buffer for DenseToSparse will be allocated.\n", buffer_size);
    ERROR_CHECK(cudaMalloc(&d_buffer, buffer_size));

    // 分析矩阵中的非零元素个数
    ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));
    // 获取非零元素个数
    int64_t rows_tmp, cols_tmp, nnz;
    ERROR_CHECK_CUSPARSE(cusparseSpMatGetSize(sp_mat, &rows_tmp, &cols_tmp, &nnz));
    printf("The sparse matrix has %ld rows and %ld columns, and the number of non-zero elements is %ld\n", rows_tmp, cols_tmp, nnz);

    // 申请CSR中的列索引数组和值数组
    int *d_column_indices_A;
    float *d_values_A;
    ERROR_CHECK(cudaMalloc((void **)&d_column_indices_A, nnz * sizeof(int)));
    ERROR_CHECK(cudaMalloc((void **)&d_values_A, nnz * sizeof(float)));

    // 为稀疏矩阵设置各个数组指针
    ERROR_CHECK_CUSPARSE(cusparseCsrSetPointers(sp_mat, d_row_offsets_A, d_column_indices_A, d_values_A));

    // 执行稠密矩阵到稀疏矩阵的转换
    ERROR_CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dn_mat, sp_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

    // 创建稠密向量
    cusparseDnVecDescr_t dn_vec_X, dn_vec_Y;
    ERROR_CHECK_CUSPARSE(cusparseCreateDnVec(&dn_vec_X, columns, d_X, CUDA_R_32F));
    ERROR_CHECK_CUSPARSE(cusparseCreateDnVec(&dn_vec_Y, rows, d_Y, CUDA_R_32F));

    // 若有必要，为SpMV计算申请额外的buffer
    float alpha = 3.0f;
    float beta  = 4.0f;
    size_t spmv_buffer_size;
    void *d_spmv_buffer;
    ERROR_CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sp_mat, dn_vec_X, &beta, dn_vec_Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size));
    printf("%ld bytes external buffer for SpMV will be allocated.\n", spmv_buffer_size);
    ERROR_CHECK(cudaMalloc(&d_spmv_buffer, spmv_buffer_size));

    // 执行SpMV计算
    ERROR_CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sp_mat, dn_vec_X, &beta, dn_vec_Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer));

    // 销毁矩阵、向量以及handle
    ERROR_CHECK_CUSPARSE(cusparseDestroyDnMat(dn_mat));
    ERROR_CHECK_CUSPARSE(cusparseDestroySpMat(sp_mat));
    ERROR_CHECK_CUSPARSE(cusparseDestroyDnVec(dn_vec_X));
    ERROR_CHECK_CUSPARSE(cusparseDestroyDnVec(dn_vec_Y));
    ERROR_CHECK_CUSPARSE(cusparseDestroy(handle));

    // 错误检查
    ERROR_CHECK(cudaMemcpy(Y, d_Y, rows * sizeof(float), cudaMemcpyDeviceToHost));
    ERROR_CHECK(cudaDeviceSynchronize());
    hostMV(&alpha, A, X, &beta, host_Y, rows, columns);
    checkResult<float>(host_Y, Y, rows, 2);

    // 资源释放
    ERROR_CHECK(cudaFree(d_spmv_buffer));
    ERROR_CHECK(cudaFree(d_values_A));
    ERROR_CHECK(cudaFree(d_column_indices_A));
    ERROR_CHECK(cudaFree(d_buffer));
    ERROR_CHECK(cudaFree(d_row_offsets_A));
    ERROR_CHECK(cudaFree(d_X));
    ERROR_CHECK(cudaFree(d_Y));
    ERROR_CHECK(cudaFree(d_A));
    free(host_Y);
    free(X);
    free(Y);
    free(A);

    ERROR_CHECK(cudaDeviceReset());

    return 0;
}
