#pragma once

#include "tiny_obj_loader.h"
#include "vec3.h"

#include <string>
#include <vector>
#include <cuda_runtime.h>


using namespace std;


class triangle {

    public:
        vec3 vertices[3];
        vec3 normal;
        vec3 pos;

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

        triangle* triangles;
        vec3 color;
        int trianglesCount;

        __host__ __device__ shape() {}
        __host__ __device__ shape(int size) {
            trianglesCount = size;
            triangles = new triangle[size];
        }
        ~shape() {}

        void freeTriangles(){
             delete [] triangles;
        }

        __host__ __device__
        int ShapeSize(){
            return trianglesCount*sizeof(triangle) + sizeof(shape);
        }
        
};

class Scene {

    public:

        shape* raw_shapes;

        int shapesCount;


        Scene() {};
        ~Scene(){

            for(int i = 0; i < shapesCount; i++){
                 delete [] raw_shapes[i].triangles;
            }

            delete [] raw_shapes;
        }

        int LoadScene(const string& path, const string& materials_path);

        __host__ __device__ 
        int SceneSize(){


            int ShapesSize = 0;

            for(int i = 0; i<shapesCount; i++){
                ShapesSize += raw_shapes[i].ShapeSize();
            }
    
            return ShapesSize + sizeof(Scene);
        }

};