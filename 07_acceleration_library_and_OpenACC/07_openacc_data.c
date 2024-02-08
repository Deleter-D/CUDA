#include <stdio.h>
#include <stdlib.h>

/*
    使用PIG编译器编译该文件
    pgcc -acc -Minfo=accel path/to/07_openacc_data.c -o /path/to/out/07_openacc_data
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

#pragma acc data copyin(A[0 : N], B[0 : N]) copyout(C[0 : N], D[0 : N])
    {
#pragma acc parallel
        {
#pragma acc loop
            for (int i = 0; i < N; i++)
            {
                C[i] = A[i] + B[i];
            }
#pragma acc loop
            for (int i = 0; i < N; i++)
            {
                D[i] = C[i] * A[i];
            }
        }
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", D[i]);
    }
    printf("...\n");

    int *restrict vec1 = (int *)malloc(N * sizeof(int));
    int *restrict vec2 = (int *)malloc(N * sizeof(int));
    int *restrict vec3 = (int *)malloc(N * sizeof(int));
    int *restrict vec4 = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        vec1[i] = 0;
        vec2[i] = i;
    }

#pragma acc enter data copyin(vec1[0 : N], vec2[0 : N]) async(0)

    for (int i = 0; i < N; i++)
    {
        vec3[i] = 2 * i;
    }

#pragma acc kernels wait(0) async(1)
    {
        for (int i = 0; i < N; i++)
        {
            vec1[i] = vec2[i] + 1;
        }
    }

#pragma acc exit data copyout(vec1[0 : N]) wait(1) async(2)

    for (int i = 0; i < N; i++)
    {
        vec4[i] = -i;
    }

#pragma acc wait(2)

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", vec1[i]);
    }
    printf("...\n");
    return 0;
}
