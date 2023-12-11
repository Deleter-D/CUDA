#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "../utils/common.cuh"

/*
    性能分析方法：
    - CPU计时：简单粗暴的计时方法
    - 事件计时：可以为主机和设备代码计时
    - NVVP Visual Profiler
    - nvprof
    - Nsight Systems
    - Nsight Compute
*/

#define REPEATS 10

int main(int argc, char const *argv[])
{
    // >>>>> CPU计时 >>>>>
    struct timeval tp;
    gettimeofday(&tp, NULL);
    double start_cpu = (double)tp.tv_sec + (double)(tp.tv_usec * 1.e-6);

    // 需要计时的代码

    gettimeofday(&tp, NULL);
    ERROR_CHECK(cudaDeviceSynchronize());
    double elapsed_time_cpu = (double)tp.tv_sec + (double)(tp.tv_usec * 1.e-6) - start_cpu;

    printf("Elapsed time (CPU): %g ms.\n", elapsed_time_cpu);
    // <<<<< CPU计时 <<<<<

    // >>>>> 事件计时方法 >>>>>
    float total_time;
    for (int i = 0; i <= REPEATS; i++) // for循环重复调用多次来取平均时间
    {
        cudaEvent_t start, stop;
        ERROR_CHECK(cudaEventCreate(&start));
        ERROR_CHECK(cudaEventCreate(&stop));
        ERROR_CHECK(cudaEventRecord(start));
        /*
            cudaEventQuery()在TCC模式下可以省略，但WDDM模式必须存在
            为了兼容性写上为好（两个模式介绍见main函数之后）
            它的功能是查询最近一次调用cudaEventRecord()之前的所有设备工作的状态
            - 如果设备已成功完成此工作，或者如果尚未在事件中调用cudaEventRecord()，则返回cudaSuccess
            - 如果设备尚未完成此工作，则返回cudaErrorNotReady
            所以当他不返回cudaSuccess的时候不代表程序出错了，故不适合在此处使用错误检查
        */
        cudaEventQuery(start);

        // 需要计时的代码

        ERROR_CHECK(cudaEventRecord(stop));
        ERROR_CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        ERROR_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        // 由于设备需要启动预热，故不计入第一次核函数执行的时间
        if (i > 0)
            total_time += elapsed_time;
        ERROR_CHECK(cudaEventDestroy(start));
        ERROR_CHECK(cudaEventDestroy(stop));
    }

    printf("Elapsed time (Event): %g ms.\n", total_time / REPEATS); // 获取平均时间
    // <<<<< 事件计时方法 <<<<<

    return 0;
}

/*
    两种驱动模式：
    - TCC（Tesla Compute Cluster）：GPU完全用于计算，不能作为本地显示输出
    - WDDM（Windows Display Driver Model）：GPU即用于计算，又用于本地显示输出

    使用nvidia-smi -dm 0 -i ${DeviceID}命令切换为WDDM模式
    使用nvidia-smi -dm 1 -i ${DeviceID}命令切换为TCC模式
*/
