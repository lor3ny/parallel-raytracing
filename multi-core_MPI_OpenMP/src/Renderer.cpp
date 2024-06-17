#include "Renderer.h"


// CAMERA CLASS

vec3 Camera::PixelToPoint(int i, int j){
    float u = (i+0.5f)/frameWidth;
    float v = ((j+0.5f)/frameHeight);

    vec3 point;

    point = origin + vec3((u-0.5)*frameWidth, -(v-0.5)*frameHeight, frameDistance);

    return point;
}

Ray Camera::generateRay(vec3& point){

    Ray tmpRay;
    tmpRay.o = this->origin;
    tmpRay.dir = (point - tmpRay.o) / distance(point, origin);
    return tmpRay;
}

Intersection Renderer::SceneRaycast(const Scene& scene, Ray& r){
    
    Intersection isec;
    isec.hasHit = false;

    for (int s = 0; s < scene.shapesCount; s++) {

        for (int t = 0; t < scene.raw_shapes[s].trianglesCount; t++) {

            float dist;
            //glm::vec2 position;
            bool rayHit = Renderer::intersectTriangle(r, scene.raw_shapes[s].triangles[t].vertices[0], scene.raw_shapes[s].triangles[t].vertices[1], scene.raw_shapes[s].triangles[t].vertices[2], dist);
            
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

vec3 Renderer::Shade(const Scene& scene, Ray& ray){

    Intersection hit = SceneRaycast(scene, ray);

    if(!hit.hasHit)
        return vec3{0,20,80};

    return hit.color * dot(-ray.dir, hit.normal);
}


void Renderer::Render(unsigned char* buffer, Scene& scene){

    // about 315 seconds on laptop with 4 cores and no OpenMP
    // Try Galileo with 1 node 4 cores
    // Try Galileo multiple nodes 1 core each without OpenMP
    // Try Galileo multiple nodes 1 core each with OpenMP


    //#pragma omp parallel for num_threads(2)
    for(int i = 0; i < batchSize; i++){
        std::cout << i << std::endl;
        for(int j = 0; j < cam->GetWidth(); j++){

            int p_x = start_index+i;

            vec3 q = cam->PixelToPoint(j, p_x);
            Ray qRay = cam->generateRay(q);

            vec3 pixelValue = Shade(scene, qRay);

            //std::cout << pixelValue.x << " " << pixelValue.y << " " << pixelValue.z << endl;

            buffer[(i * cam->GetWidth() + j) * 3 + 0] = pixelValue.r() * 255;
            buffer[(i * cam->GetWidth() + j) * 3 + 1] = pixelValue.g() * 255;
            buffer[(i * cam->GetWidth() + j) * 3 + 2] = pixelValue.b() * 255;

        }
    }
}