# Tiny Parallel Raytracer 

The project tries to build different parallel approaches to obtain optimized raytracing algorithm. The aim is not the raytracing itself but the parallelization techniques.]

### Multi-core version:

Is built with MPI and OpenMP, the algorithm divide the the computation in k parallel batches, k identifies the number of nodes in a cluster (if It is a single-node one, It reason by cores). Every node parallelize the loops. When the computation is endeed the master node gather the final results from every node.

### Single-gpu version:

*Work in progress*

### Multi-gpu version:

*Work in progress*

## Dependencies

Tiny Obj Loader
stb_image

## Requirements


OpenMPI
CUDA
