#pragma once


#include "Log.h"
#include "SceneHandler.h"


#include <cstddef>
#include <fwd.hpp>
#include <geometric.hpp>
#include <glm.hpp>
#include <iostream>


struct Ray{
    glm::vec3 o; // origin
    glm::vec3 dir; // direction
    glm::vec3 auxPoint;
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

    public: 
        Camera* cam;

        Renderer(int width, int height){
            glm::vec3 o = {.0f,.0f,.0f};
            cam = new Camera(o,-20, width, height);
        }
        ~Renderer(){
            delete [] cam;
        }


        int intersectTriangle(Ray& r, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2){
            
            auto edge1 = p1-p0;
            auto edge2  = p2-p0;

            auto pvec = glm::cross(r.dir, edge2);
            auto det = glm::dot(edge1, pvec);

            if(det==0)
                return 1;

            auto idet = 1.0f / det;
            auto tvec = r.o - p0;
            auto u = glm::dot(tvec, pvec) * idet;

            if(u<0 || u>1) 
                return 1;

            auto qvec = glm::cross(tvec, edge1);
            auto v = glm::dot(r.dir, qvec) * idet;
            if(v<0 || v>1)
                return 1;

            auto t = glm::dot(edge2, qvec) * idet;
            // if( t<r.tmin || t>r.tmax) return 0

            // Results
            glm::vec2 uv = {u, v};
            auto dist = t;
            return 0;
        }


        void Raytrace(unsigned char* buffer, SceneHandler scene){
            for(int i = 0; i < cam->GetHeight(); i++){
                for(int j = 0; j < cam->GetWidth(); j++){

                    glm::vec3 q = cam->PixelToPoint(j, i);
                    Ray qRay = cam->generateRay(q);

                    for (size_t s = 0; s < scene.shapes.size(); s++) {
                        
                        size_t index_offset = 0; 
                        for (size_t f = 0; f < scene.shapes[s].mesh.num_face_vertices.size(); f++) {
                            
                            size_t fv = size_t(scene.shapes[s].mesh.num_face_vertices[f]);
                            if(fv != 3)
                                Log::PrintError("Robust Triangulation is not working. Raytracing stopped");       

                            // VERTEX 0: scene.shapes[s].mesh.num_face_vertices[0]
                            // VERTEX 1: scene.shapes[s].mesh.num_face_vertices[1]
                            // VERTEX 2: scene.shapes[s].mesh.num_face_vertices[2]        
                            
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

                            // For now i take only one color, but maybe next is interpolation between the three vertecies is needed

                            int ret = intersectTriangle(qRay, p0, p1, p2);
                            tinyobj::real_t r, g, b;
                            if(ret == 0){
                                r = scene.attrib.colors[3*size_t(idx0.vertex_index)+0];
                                g = scene.attrib.colors[3*size_t(idx0.vertex_index)+1];
                                b = scene.attrib.colors[3*size_t(idx0.vertex_index)+2];
                                std::cout << g << std::endl;
                            } else {
                                r = .0f; g = .0f; b = .0f;
                            }

                            buffer[(i * cam->GetWidth() + j) * 3 + 0] = r*255;
                            buffer[(i * cam->GetWidth() + j) * 3 + 1] = g*255;
                            buffer[(i * cam->GetWidth() + j) * 3 + 2] = b*255;
                        
                            index_offset += fv;

                            //Log::Print(fv);
                        }
                    }
                }
            }
        }
};