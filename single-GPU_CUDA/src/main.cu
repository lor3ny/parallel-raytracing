#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#define STB_IMAGE_WRITE_IMPLEMENTATION

//#include <chrono>
//#include <iostream>

#include "Scene.h"
#include "Camera.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "vec3.h"


#define WIDTH 1920
#define HEIGHT 1080

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__device__
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

__device__
Intersection SceneRaycast(const Scene& scene, Ray& r){
    
    Intersection isec;
    isec.hasHit = false;

    for (int s = 0; s < scene.shapesCount; s++) {

        for (int t = 0; t < scene.raw_shapes[s].trianglesCount; t++) {

            float dist;
            //glm::vec2 position;
            bool rayHit = intersectTriangle(r, scene.raw_shapes[s].triangles[t].vertices[0], scene.raw_shapes[s].triangles[t].vertices[1], scene.raw_shapes[s].triangles[t].vertices[2], dist);
            
            if(!rayHit)
                continue;

            if(dist>isec.dist && isec.dist > 0)
                    continue;

            isec.hasHit = true;
            isec.dist = dist;
            isec.normal = scene.raw_shapes[s].triangles[t].normal;
            isec.color = scene.raw_shapes[s].color;
        }
    }

    return isec;
}

__device__
vec3 Shade(const Scene& scene, Ray& ray){

    Intersection hit = SceneRaycast(scene, ray);

    if(!hit.hasHit)
        return vec3{0,20,80};

    return hit.color * dot(-ray.dir, hit.normal);
}

__global__
void Render(unsigned char* buffer, Scene& scene, Camera& cam){

    for(int i = 0; i < cam.frameHeight; i++){
        for(int j = 0; j < cam.frameWidth; j++){

            vec3 q = cam.PixelToPoint(j, i);
            Ray qRay = cam.generateRay(q);

            vec3 pixelValue = Shade(scene, qRay);

            //std::cout << pixelValue.x() << " - " << pixelValue.y() << " - " << pixelValue.z() << endl;

            buffer[(i * cam.frameWidth + j) * 3 + 0] = pixelValue.r() * 255;
            buffer[(i * cam.frameWidth + j) * 3 + 1] = pixelValue.g() * 255;
            buffer[(i * cam.frameWidth + j) * 3 + 2] = pixelValue.b() * 255;

        }
    }
}

int main(int argc, char *argv[]){

    unsigned char* bigBuff = new unsigned char[WIDTH*HEIGHT*3];
    Scene* scene;
    Camera* cam;

    (*scene).LoadScene("../test/cornell_box.obj", "../test/");
    vec3 origin = {250,250,-1000};
    *cam = Camera(origin, 1000, WIDTH, HEIGHT);

    cout << "Scene Loaded, Rendering started." << endl;

    cudaMalloc(&bigBuff, WIDTH*HEIGHT*3*sizeof(char));
    cudaMalloc(&scene, sizeof(Scene)); //Needs an internal sizeof function
    cudaMalloc(&cam, sizeof(Camera)); 

    Render<<<1,1>>>(bigBuff, *scene, *cam);

    cudaFree(&cam);
    cudaFree(&scene);
    cudaFree(&bigBuff);

    cudaFree(&bigBuff);

    cout << "Rendering done." << endl;

    stbi_write_png("res.png", WIDTH, HEIGHT, 3, bigBuff, WIDTH*3);
    
}