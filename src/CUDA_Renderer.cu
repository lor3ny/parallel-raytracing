#include "CUDA_Renderer.h"
#include "Renderer.h"


__global__
void SceneRaycastCUDA(const Scene& scene, Ray& r, Intersection& isec){
    
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
}

