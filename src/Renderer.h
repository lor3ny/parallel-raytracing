#pragma once

#include "Log.h"
#include "SceneHandler.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fwd.hpp>
#include <geometric.hpp>
#include <glm.hpp>
#include <iostream>
#include <vector>


struct Ray{
    glm::vec3 o; // origin
    glm::vec3 dir; // direction
    glm::vec3 auxPoint;
};

struct Intersection{
    bool isNull = true;
    int faceMaterialIdx;
    glm::vec3 faceNormal;
    float dist = 0;
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

            for (size_t s = 0; s < scene.shapes.size(); s++) {

                size_t v_index_offset = 0; 
                for (size_t f = 0; f < scene.shapes[s].mesh.num_face_vertices.size(); f++) {
                    
                    size_t fv = size_t(scene.shapes[s].mesh.num_face_vertices[f]);
                    if(fv != 3)
                        Log::PrintError("Robust Triangulation is not working. Raytracing stopped");          
                    
                    tinyobj::index_t idx0 = scene.shapes[s].mesh.indices[v_index_offset + 0];
                    tinyobj::index_t idx1 = scene.shapes[s].mesh.indices[v_index_offset + 1];
                    tinyobj::index_t idx2 = scene.shapes[s].mesh.indices[v_index_offset + 2];
                    

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

                    v_index_offset += fv;

                    float dist;
                    bool raycast = intersectTriangle(ray, p0, p1, p2, dist);
                    if(raycast){


                        if(dist>isec.dist && isec.dist > 0)
                            continue;

                        int idxFace = ceil(v_index_offset/6)-1;  // Blender export normals
                        
                        if(scene.shapes[s].name == "tall_block"){
                            std::cout << "palle" << std::endl;
                        }

                        isec.isNull = false;
                        isec.dist = dist;
                        glm::vec3 norm = glm::vec3(scene.attrib.normals[3*idx0.normal_index+0],scene.attrib.normals[3*idx0.normal_index+1], scene.attrib.normals[3*idx0.normal_index+2]);
                        isec.faceNormal = norm;
                        isec.faceMaterialIdx = scene.shapes[s].mesh.material_ids[f];
                        break;
                    }
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
                    /*
                    if(!isec.isNull){
                        buffer[(i * cam->GetWidth() + j) * 3 + 0] = scene.materials[isec.faceMaterialIdx].diffuse[0]*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 1] = scene.materials[isec.faceMaterialIdx].diffuse[1]*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 2] = scene.materials[isec.faceMaterialIdx].diffuse[2]*255;
                    } else {
                        buffer[(i * cam->GetWidth() + j) * 3 + 0] = 0.0;
                        buffer[(i * cam->GetWidth() + j) * 3 + 1] = 0.0;
                        buffer[(i * cam->GetWidth() + j) * 3 + 2] = 0.0;
                    }
                    */

                    if(!isec.isNull){
                        //cout << isec.faceNormal.x << " " << isec.faceNormal.y << " " << isec.faceNormal.z << endl;
                        buffer[(i * cam->GetWidth() + j) * 3 + 0] = abs(isec.faceNormal.x)*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 1] = abs(isec.faceNormal.y)*255;
                        buffer[(i * cam->GetWidth() + j) * 3 + 2] = abs(isec.faceNormal.z)*255;
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