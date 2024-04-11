#pragma once

#include "SceneHandler.h"

#include <fwd.hpp>
#include <geometric.hpp>
#include <glm.hpp>
#include <string>


struct Ray{
    glm::vec3 o; // origin
    glm::vec3 dir; // direction
};

struct Intersection{
    std::string name;
    bool hasHit;
    int materialIdx;
    glm::vec3 normal;
    glm::vec3 trianglePos;
    glm::vec2 hitPos;
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

        glm::vec3 PixelToPoint(int i , int j);

        Ray generateRay(glm::vec3& point);

};


class Renderer {

    private:
        Camera* cam;
        int maxBounces = 3;


        bool intersectTriangle(Ray& r, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2, float& dist, glm::vec2& pos);
        Intersection SceneRaycast(const SceneHandler& scene, Ray& r);
        glm::vec3 Shade(const SceneHandler& scene, Ray& ray, int bounce);

    public: 

        // Generalize the constructor with camera data

        Renderer(int width, int height){
            glm::vec3 o = {250.0f,250.0f,-1000.0f};
            cam = new Camera(o,1000, width, height);
        }
        ~Renderer(){
            delete [] cam;
        }


        void Render(unsigned char* buffer, SceneHandler& scene);
};