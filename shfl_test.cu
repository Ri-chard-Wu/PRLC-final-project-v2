#include <iostream>
#include <random>

#define N 1024
typedef unsigned int WORD;
typedef unsigned char BYTE;
using namespace std;


template<typename T>
T CPU_reduction(T *arr, int n) {

    T sum = 0;

    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }

    return sum;
}

template<typename T>
T fRand(T fMin, T fMax)
{
    T f = (T)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

template<typename T>
void generate_random_numbers(T *arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = fRand(0., 10000.);
    }
}



#define FULL_MASK 0xffffffff

template<typename T>
__inline__ __device__
T warpReduceSum(T val) {

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {

        T val_other = __shfl_down_sync(FULL_MASK, val, offset);
        val = val + val_other;

    }

    return val;
}




template<typename T>
__inline__ __device__
T blockReduceSum(T val) {

    __shared__ T shared[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum<T>(val);  

    if (lane == 0) shared[wid] = val; 

    __syncthreads();             

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid==0) val = warpReduceSum<T>(val); 

    return val;
}


template<typename T>
__global__ void cuda_reduction(T *arr, int n, T *ret) {

    unsigned int tid = threadIdx.x;
    T val = arr[tid];
    
    __syncthreads();

    val = blockReduceSum<T>(val);

    if (tid == 0) *ret = val;
}




template<typename T>
void f(){

    srand(time(0));

    T *ret = new T;
    T *arr = new T[N];
    
    generate_random_numbers<T>(arr, N);

    T *arr_dev, *ret_dev;
    cudaMalloc(&arr_dev, N * sizeof(T));
    cudaMalloc(&ret_dev, 1 * sizeof(T));
    cudaMemcpy((BYTE *)arr_dev, (BYTE *)arr,
                            N * sizeof(T), cudaMemcpyHostToDevice);
                                
    cuda_reduction<T><<<1, N>>>(arr_dev, N, ret_dev);

    cudaDeviceSynchronize();

    cudaMemcpy((BYTE *)ret, (BYTE *)ret_dev,
                            1 * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "[main] (cuda) The minimum value: " << *ret << '\n';
    *ret = CPU_reduction<T>(arr, N);
    std::cout << "[main] (cpu) The minimum value: " << *ret << '\n';
    
    delete ret;
    delete [] arr;    
}


int main() {

    f<float>();
    return 0;
}