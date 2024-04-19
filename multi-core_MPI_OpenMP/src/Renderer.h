#pragma once

#include "vec3.h"
#include "Scene.h"
#include <omp.h>


class Camera {

    private:

        vec3 origin;
        float frameDistance;
        int frameWidth;
        int frameHeight;

    public:


        inline int GetWidth() const { return this->frameWidth;}
        inline int GetHeight() const { return this->frameHeight;}

        Camera(vec3& o, float d, int width, int height) 
            : origin(o), frameDistance(d), frameWidth(width), frameHeight(height) {}   

        vec3 PixelToPoint(int i , int j);

        Ray generateRay(vec3& point);

};


class Renderer {

    private:
        Intersection SceneRaycast(const Scene& scene, Ray& r);
        vec3 Shade(const Scene& scene, Ray& ray);

    public: 

        // Generalize the constructor with camera data 

        int start_index;
        int batchSize;

        Camera* cam;

        Renderer(Camera* camera, int& start_idx, int batch){
            cam = camera;
            start_index = start_idx;
            batchSize = batch;
        }
        ~Renderer(){}


        void Render(unsigned char* buffer, Scene& scene);

        __host__ __device__
        static bool intersectTriangle(Ray& r, vec3& p0, vec3& p1, vec3& p2, float& dist){ //, vec2& pos){
    
            const double EPSILON = 0.000001; // error i think

            auto edge1 = p1-p0;
            auto edge2  = p2-p0;

            auto pvec = cross(r.dir, edge2);
            auto det = dot(edge1, pvec);
            if(det > -EPSILON && det < EPSILON){
                return false;
            }

            auto idet = 1.0f / det;
            auto tvec = r.o - p0;
            auto u = dot(tvec, pvec) * idet;
            if(u < 0.0 || u > 1.0){
                return false;
            }

            auto qvec = cross(tvec, edge1);
            auto v = dot(r.dir, qvec) * idet;
            if(v < 0.0 || u+v > 1.0){
                return false;
            }

            auto t = dot(edge2, qvec) * idet;

            dist = t;
            //pos = {u, v};
            return t > EPSILON; 
        }
};