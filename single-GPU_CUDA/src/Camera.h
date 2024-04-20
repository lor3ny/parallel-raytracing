#pragma once

#include "vec3.h"

class Camera {

    private:

        vec3 origin;
        float frameDistance;

    public:

        int frameWidth;
        int frameHeight;


        Camera(vec3& o, float d, int width, int height) 
            : origin(o), frameDistance(d), frameWidth(width), frameHeight(height) {}   


        __device__ vec3 PixelToPoint(int i, int j){
            float u = (i+0.5f)/frameWidth;
            float v = ((j+0.5f)/frameHeight);

            vec3 point;

            point = origin + vec3((u-0.5)*frameWidth, -(v-0.5)*frameHeight, frameDistance);

            return point;
        }

        __device__ Ray generateRay(vec3& point){
            Ray tmpRay;
            tmpRay.o = this->origin;
            tmpRay.dir = (point - tmpRay.o) / distance(point, origin);
            return tmpRay;
        }

};