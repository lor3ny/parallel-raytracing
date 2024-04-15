#include "CUDA_Renderer.h"


__global__
void CUDA_Render(){//unsigned char* buffer, SceneHandler& scene, Renderer& renderer){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i*2;
}