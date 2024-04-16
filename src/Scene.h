#pragma once

#include "fwd.hpp"
#include "tiny_obj_loader.h"
#include "vec3.h"

#include <string>
#include <vector>
#include <cuda_runtime.h>


using namespace std;


class triangle {

    public:
        __host__ __device__ vec3 vertices[3];
        __host__ __device__ vec3 normal;
        __host__ __device__ vec3 pos;

        __host__ __device__ triangle() {}
        __host__ __device__ triangle(vec3& p0, vec3& p1, vec3& p2, vec3& nor) {
            vertices[0] = p0;
            vertices[1] = p1;
            vertices[2] = p2;

            pos = (p0 + p1 + p2) / 3.0f;

            normal = nor;
        }

};

class shape {
    public:
        __host__ __device__ triangle* triangles;
        __host__ __device__ vec3 color;
        __host__ __device__ int trianglesCount;

        __host__ __device__ shape() {}
        __host__ __device__ shape(int size) {
            trianglesCount = size;
            triangles = new triangle[size];
        }
        ~shape() {}

        void freeTriangles(){
             delete [] triangles;
        }
        
};

class Scene {

    private:
        int LoadTinyObjScene(const string& path, const string& materials_path);

    public:

        tinyobj::attrib_t attrib;
        vector<tinyobj::material_t> materials;
        vector<tinyobj::shape_t> shapes;

        __host__ __device__ shape* raw_shapes;

        __host__ __device__ int shapesCount;

        Scene() {};
        ~Scene(){
            this->materials.clear();
            this->shapes.clear();

            for(int i = 0; i < shapesCount; i++){
                 delete [] raw_shapes[i].triangles;
            }

            delete [] raw_shapes;
        }

    int LoadScene(const string& path, const string& materials_path);

};