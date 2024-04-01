#pragma once

#include "Log.h"
#include "SceneHandler.h"

#include <cstddef>
#include <cstdlib>
#include <fwd.hpp>
#include <geometric.hpp>
#include <glm.hpp>
#include <vector>


struct Ray{
    glm::vec3 o; // origin
    glm::vec3 dir; // direction
    glm::vec3 auxPoint;
};

struct Intersection{
    bool isNull;
    int faceMaterialIdx;
    glm::vec3 faceNormal;
    float dist;
};


class Camera {

    private:

        glm::vec3 origin;
        float frameDistance;
        int frameWidth;
        int frameHeight;

    public:


        const inline int GetWidth(){ return this->frameWidth;}
        const inline int GetHeight(){ return this->frameHeight;}

        Camera(glm::vec3& o, float d, int width, int height) 
            : origin(o), frameDistance(d), frameWidth(width), frameHeight(height) {}   

        glm::vec3 PixelToPoint(int i , int j){

            float u = (i+0.5f)/frameWidth;
            float v = (j+0.5f)/frameHeight;

            glm::vec3 point;

            point = origin + glm::vec3((u-0.5)*frameWidth, -(v-0.5)*frameHeight, frameDistance);

            return point;
        } 

        Ray generateRay(glm::vec3& point){

            Ray tmpRay;
            tmpRay.o = this->origin;
            tmpRay.dir = (point - tmpRay.o) / glm::distance(point, origin);
            tmpRay.auxPoint = point;
            return tmpRay;
        }

};


class Renderer {

    private:
        Camera* cam;

        bool intersectTriangle(Ray& r, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2, float& dist){
            
            const double EPSILON = 0.000001; // error i think

            auto edge1 = p1-p0;
            auto edge2  = p2-p0;

            auto pvec = glm::cross(r.dir, edge2);
            auto det = glm::dot(edge1, pvec);
            if(det > -EPSILON && det < EPSILON){
                //std::cout << det << std::endl;
                return false;
            }

            auto idet = 1.0f / det;
            auto tvec = r.o - p0;
            auto u = glm::dot(tvec, pvec) * idet;
            if(u < 0.0 || u > 1.0){
                //Log::Print("u ~ 0");
                return false;
            }

            auto qvec = glm::cross(tvec, edge1);
            auto v = glm::dot(r.dir, qvec) * idet;
            if(v < 0.0 || u+v > 1.0){
                //Log::Print("u+v ~ 0");
                return false;
            }

            auto t = glm::dot(edge2, qvec) * idet;

            // Results
            glm::vec2 uv = {u, v};
            dist = t;
            return t > EPSILON; 
        }

        Intersection Raytrace(Ray& ray, SceneHandler& scene){

            int faceMaterial = -1;
            bool raycast = false;

            Intersection isec;
            isec.isNull = true;

            for (size_t s = 0; s < scene.shapes.size(); s++) {
                size_t index_offset = 0; 
                for (size_t f = 0; f < scene.shapes[s].mesh.num_face_vertices.size(); f++) {
                    
                    size_t fv = size_t(scene.shapes[s].mesh.num_face_vertices[f]);
                    if(fv != 3)
                        Log::PrintError("Robust Triangulation is not working. Raytracing stopped");          
                    
                    tinyobj::index_t idx0 = scene.shapes[s].mesh.indices[index_offset + 0];
                    tinyobj::index_t idx1 = scene.shapes[s].mesh.indices[index_offset + 1];
                    tinyobj::index_t idx2 = scene.shapes[s].mesh.indices[index_offset + 2];

                    

                    float p0x = scene.attrib.vertices[3*size_t(idx0.vertex_index)+0];
                    float p0y = scene.attrib.vertices[3*size_t(idx0.vertex_index)+1];
                    float p0z = scene.attrib.vertices[3*size_t(idx0.vertex_index)+2];
            
                    float p1x = scene.attrib.vertices[3*size_t(idx1.vertex_index)+0];
                    float p1y = scene.attrib.vertices[3*size_t(idx1.vertex_index)+1];
                    float p1z = scene.attrib.vertices[3*size_t(idx1.vertex_index)+2];

                    float p2x = scene.attrib.vertices[3*size_t(idx2.vertex_index)+0];
                    float p2y = scene.attrib.vertices[3*size_t(idx2.vertex_index)+1];
                    float p2z = scene.attrib.vertices[3*size_t(idx2.vertex_index)+2];

                    glm::vec3 p0(p0x,p0y,p0z);
                    glm::vec3 p1(p1x,p1y,p1z);
                    glm::vec3 p2(p2x,p2y,p2z);

                    float dist;
                    bool raycast = intersectTriangle(ray, p0, p1, p2, dist);
                    if(raycast){

                        // VETTORI NORMALI: Non corrispono a quelli realmente presenti nel file .obj, perche`?
                        /*
                        tinyobj::real_t n0x, n0y, n0z, n1x, n1y, n1z, n2x, n2y, n2z = 0;

                        if(idx0.normal_index >= 0){
                            n0x = std::abs(scene.attrib.normals[3*size_t(idx0.vertex_index)+0]);
                            n0y = std::abs(scene.attrib.normals[3*size_t(idx0.vertex_index)+1]);
                            n0z = std::abs(scene.attrib.normals[3*size_t(idx0.vertex_index)+2]);
                        }

                        if(idx1.normal_index >= 0){
                            n1x = std::abs(scene.attrib.normals[3*size_t(idx1.vertex_index)+0]);
                            n1y = std::abs(scene.attrib.normals[3*size_t(idx1.vertex_index)+1]);
                            n1z = std::abs(scene.attrib.normals[3*size_t(idx1.vertex_index)+2]);
                        }

                        if(idx2.normal_index >= 0){
                            n2x = scene.attrib.normals[3*size_t(idx2.vertex_index)+0];
                            n2y = scene.attrib.normals[3*size_t(idx2.vertex_index)+1];
                            n2z = scene.attrib.normals[3*size_t(idx2.vertex_index)+2];
                        }

                        r = (n0x + n1x + n2x)/glm::sqrt((n0x*n0x+n1x*n1x+n2x*n2x));
                        g = (n0y + n1y + n2y)/glm::sqrt((n0y*n0y+n1y*n1y+n2y*n2y));
                        b = (n0z + n1z + n2z)/glm::sqrt((n0z*n0z+n1z*n1z+n2z*n2z));
                        */
                        // VETTORI NORMALI

                        isec.isNull = false;
                        isec.dist = dist;
                        isec.faceNormal = glm::vec3(0,0,0);
                        isec.faceMaterialIdx = scene.shapes[s].mesh.material_ids[f];
                        return isec;
                    }

                    index_offset += fv;
                }
            }

            return isec;
        }

    public: 

        // Generalize the constructor with camera data

        Renderer(int width, int height){
            glm::vec3 o = {250.0f,250.0f,-1000.0f};
            cam = new Camera(o,1000, width, height);
        }
        ~Renderer(){
            delete [] cam;
        }


        void Render(unsigned char* buffer, SceneHandler& scene){
            for(int i = 0; i < cam->GetHeight(); i++){
                for(int j = 0; j < cam->GetWidth(); j++){

                    glm::vec3 q = cam->PixelToPoint(j, i);
                    Ray qRay = cam->generateRay(q);
                    tinyobj::real_t r = 0, g=0, b=0;

                    Intersection isec;
                    isec = Raytrace(qRay, scene);

                    // GENERALIZE IN SHADING FUNCTIONS

                    if(!isec.isNull){
                        buffer[(i * cam->GetWidth() + j) * 3 + 0] = scene.materials[isec.faceMaterialIdx].diffuse[0]*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 1] = scene.materials[isec.faceMaterialIdx].diffuse[1]*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 2] = scene.materials[isec.faceMaterialIdx].diffuse[2]*255;
                    } else {
                        buffer[(i * cam->GetWidth() + j) * 3 + 0] = 0.0;
                        buffer[(i * cam->GetWidth() + j) * 3 + 1] = 0.0;
                        buffer[(i * cam->GetWidth() + j) * 3 + 2] = 0.0;
                    }

                    // GENERALIZE IN SHADING FUNCTIONS
                }
            }
        }
};