#include <stdio.h>
#include <stdlib.h>

/*
    使用PIG编译器编译该文件
    pgcc -acc -Minfo=accel path/to/05_openacc_kernels.c -o /path/to/out/05_openacc_kernels
*/

int main(int argc, char const *argv[])
{
    size_t N = 1024;

    int *restrict A = (int *)malloc(N * sizeof(int));
    int *restrict B = (int *)malloc(N * sizeof(int));
    int *restrict C = (int *)malloc(N * sizeof(int));
    int *restrict D = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

#pragma acc kernels if (N > 128) async(0)
    {
        for (int i = 0; i < N; i++)
        {
            C[i] = A[i] + B[i];
        }
    }

    int flag = acc_async_test(0);
    printf("%s\n", flag ? "finished." : "not finished.");

#pragma acc kernels wait(0) async(1)
    {
        for (int i = 0; i < N; i++)
        {
            D[i] = C[i] * A[i];
        }
    }

#pragma acc wait(1)

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }
    printf("...\n");

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
