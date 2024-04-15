#include "Renderer.h"
#include "fwd.hpp"
#include "geometric.hpp"
#include <iostream>
#include <ostream>
#include <random>


// CAMERA CLASS

glm::vec3 Camera::PixelToPoint(int i, int j){
    float u = (i+0.5f)/frameWidth;
    float v = ((j+0.5f)/frameHeight);

    glm::vec3 point;

    point = origin + glm::vec3((u-0.5)*frameWidth, -(v-0.5)*frameHeight, frameDistance);

    return point;
}

Ray Camera::generateRay(glm::vec3& point){

    Ray tmpRay;
    tmpRay.o = this->origin;
    tmpRay.dir = (point - tmpRay.o) / glm::distance(point, origin);
    return tmpRay;
}


// RENDERER CLASS
bool Renderer::intersectTriangle(Ray& r, glm::vec3& p0, glm::vec3& p1, glm::vec3& p2, float& dist, glm::vec2& pos){
    
    const double EPSILON = 0.000001; // error i think

    auto edge1 = p1-p0;
    auto edge2  = p2-p0;

    auto pvec = glm::cross(r.dir, edge2);
    auto det = glm::dot(edge1, pvec);
    if(det > -EPSILON && det < EPSILON){
        return false;
    }

    auto idet = 1.0f / det;
    auto tvec = r.o - p0;
    auto u = glm::dot(tvec, pvec) * idet;
    if(u < 0.0 || u > 1.0){
        return false;
    }

    auto qvec = glm::cross(tvec, edge1);
    auto v = glm::dot(r.dir, qvec) * idet;
    if(v < 0.0 || u+v > 1.0){
        return false;
    }

    auto t = glm::dot(edge2, qvec) * idet;

    dist = t;
    pos = {u, v};
    return t > EPSILON; 
}

Intersection Renderer::SceneRaycast(const SceneHandler& scene, Ray& r){
    
    Intersection isec;
    isec.hasHit = false;

    for (size_t s = 0; s < scene.shapes.size(); s++) {

        size_t v_index_offset = 0; 
        for (size_t f = 0; f < scene.shapes[s].mesh.num_face_vertices.size(); f++) {
            
            size_t fv = size_t(scene.shapes[s].mesh.num_face_vertices[f]);
            if(fv != 3)
                cerr << "Robust Triangulation is not working. Raytracing stopped" << endl;       
            
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
            glm::vec2 position;
            bool rayHit = intersectTriangle(r, p0, p1, p2, dist, position);
            
            if(!rayHit)
                continue;

            if(dist>isec.dist && isec.dist > 0)
                    continue;


            isec.hasHit = true;
            isec.dist = dist;
            isec.normal = glm::vec3(scene.attrib.normals[3*idx0.normal_index+0],scene.attrib.normals[3*idx0.normal_index+1], scene.attrib.normals[3*idx0.normal_index+2]);
            isec.materialIdx = scene.shapes[s].mesh.material_ids[f];
            isec.trianglePos = (p0 + p1 + p2) / 3.0f;
            isec.hitPos = position;
            isec.name = scene.shapes[s].name;
        }
    }

    return isec;
}


glm::vec3 transform_position(glm::vec2 uv, glm::vec3 triangle){
    return {triangle.x * (1-uv.x-uv.y), triangle.y*uv.x, triangle.z*uv.y};
}

glm::vec3 randomSample() {
  // Use a Mersenne Twister engine for randomness
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis_x(-1.0f, 1.0f);
  static std::uniform_real_distribution<float> dis_z(-1.0f, 1.0f);

  return glm::vec3(dis_x(gen), 0, dis_z(gen));
}


glm::vec3 Renderer::Shade(const SceneHandler& scene, Ray& ray){

    Intersection hit = SceneRaycast(scene, ray);
    if(!hit.hasHit)
        return glm::vec3{0,20,80};

    auto color = glm::vec3{scene.materials[hit.materialIdx].diffuse[0],
                                scene.materials[hit.materialIdx].diffuse[1],
                                scene.materials[hit.materialIdx].diffuse[2]};
    auto radiance = glm::vec3{scene.materials[hit.materialIdx].ambient[0],
                                scene.materials[hit.materialIdx].ambient[1],
                                scene.materials[hit.materialIdx].ambient[2]};


    return color * glm::dot(-ray.dir, hit.normal);
}


void Renderer::Render(unsigned char* buffer, SceneHandler& scene){
    for(int i = 0; i < batchSize; i++){
        for(int j = 0; j < cam->GetWidth(); j++){

            int p_x = start_index+i;

            glm::vec3 q = cam->PixelToPoint(j, p_x);
            Ray qRay = cam->generateRay(q);

            auto pixelValue = Shade(scene, qRay);

            //std::cout << pixelValue.x << " " << pixelValue.y << " " << pixelValue.z << endl;

            buffer[(i * cam->GetWidth() + j) * 3 + 0] = pixelValue.x * 255;
            buffer[(i * cam->GetWidth() + j) * 3 + 1] = pixelValue.y * 255;
            buffer[(i * cam->GetWidth() + j) * 3 + 2] = pixelValue.z * 255;

        }
    }
}