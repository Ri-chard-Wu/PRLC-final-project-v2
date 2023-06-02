typedef unsigned char BYTE;



__global__ void kernel1(float *arr, long long N, int n_total_threads){

    int gid = blockIdx.x * blockDim.x + threadIdx.x;


    // if(threadIdx.x != 0) return;
    
    for(long long i = 0; i < N; i += n_total_threads){
        arr[gid + i] = (gid + i) * 0.00001;
    }
}


