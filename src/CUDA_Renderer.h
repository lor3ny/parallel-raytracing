#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "Scene.h"

__global__
void SceneRaycastCUDA(const Scene& scene, Ray& r, Intersection& isec);