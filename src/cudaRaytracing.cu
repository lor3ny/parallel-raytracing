#include "cudaRaytracing.h"

__global__
void test(int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
     i = i*2;
}