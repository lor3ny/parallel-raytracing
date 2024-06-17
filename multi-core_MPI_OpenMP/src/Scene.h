#pragma once

#include "tiny_obj_loader.h"
#include "vec3.h"

#include <string>
#include <vector>


using namespace std;


class triangle {

    public:
        vec3 vertices[3];
        vec3 normal;
        vec3 pos;

         triangle() {}
         triangle(vec3& p0, vec3& p1, vec3& p2, vec3& nor) {
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

         shape() {}
         shape(int size) {
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

        shape* raw_shapes;

        int shapesCount;

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