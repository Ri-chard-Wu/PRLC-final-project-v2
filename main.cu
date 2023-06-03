
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

typedef unsigned char BYTE;

__global__ void kernel1(float *arr, long long N, int n_total_threads){

    int gid = blockIdx.x * blockDim.x + threadIdx.x;


    // if(threadIdx.x != 0) return;
    
    for(long long i = 0; i < N; i += n_total_threads){
        arr[gid + i] = (gid + i) * 0.00001;
    }
}


void kernel1_cpu(long long N){
    float sum = 0.0;
    for(long long i = 0; i < N; i++){
        sum += i * 0.00001;
    }    

    printf("cpu: %f\n", sum);
}




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
    
    kernel1_cpu(N);

    

    return 0;
}

