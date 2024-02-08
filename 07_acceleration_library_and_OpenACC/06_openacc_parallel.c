#include <stdio.h>
#include <stdlib.h>

/*
    使用PIG编译器编译该文件
    pgcc -acc -Minfo=accel path/to/06_openacc_parallel.c -o /path/to/out/06_openacc_parallel
*/

int main(int argc, char const *argv[])
{
    // #pragma acc parallel num_gangs(32) num_workers(32) vector_length(64)
    //     {
    //         // ...
    //     }

    size_t N = 1024;

    int *restrict vec = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        vec[i] = 1;
    }

    int result = 0;
#pragma acc parallel reduction(+ : result)
    {
        for (int i = 0; i < N; i++)
        {
            result += vec[i];
        }
    }

    printf("sum: %d\n", result);

    int a = 5;
#pragma acc parallel private(a)
    {
        printf("private a: %d\n", a);
        a = 10;
    }
    printf("a value after private block: %d\n", a);

#pragma acc parallel firstprivate(a)
    {
        printf("firstprivate a: %d\n", a);
        a = 10;
    }
    printf("a value after firstprivate block: %d\n", a);

    int *restrict A = (int *)malloc(N * sizeof(int));
    int *restrict B = (int *)malloc(N * sizeof(int));
    int *restrict C = (int *)malloc(N * sizeof(int));
    int *restrict D = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = 2 * i;
    }

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

    int *restrict vec2 = (int *)malloc(N * sizeof(int));

#pragma acc parallel
    {
        int a = 1;

#pragma acc loop gang
        for (int i = 0; i < N; i++)
        {
            vec2[i] = a;
        }
    }

    free(A);
    free(B);
    free(C);
    free(D);

    return 0;
}
