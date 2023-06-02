
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>

#include <cmath>           
#include <iomanip>           
#include <limits>           
#include <stdexcept>           
#include <string>           

#include <main.cuh>




int main(int argc, char **argv)
{
    long long N = 25 * ((long long)1e9);
    size_t size = N * sizeof(float);
    printf("size allocated: %lld GB\n", ((long long)size) / ((long long)1e9));
    
    float *arr;

    cudaMallocManaged(&arr, size);

    int n_blocks = 32;
    int n_threads_per_block = 128;
    kernel1<<<n_blocks, n_threads_per_block>>>(arr, N, n_threads_per_block * n_blocks);

    cudaDeviceSynchronize();
    printf("kernel done\n");

    float sum = 0;
    for(long long i = 0; i < N; i++){
        sum += arr[i];
    }

    printf("gpu: %f\n", sum);
    cudaFree(arr);
    


    

    return 0;
}



