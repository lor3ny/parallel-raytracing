#include "Scene.h"
#include <iostream>


int Scene::LoadScene(const string& path, const string& materials_path){

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = materials_path;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(path, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << reader.Error() << std::endl;
        }
        return 1;
    }

    if (!reader.Warning().empty()) {
        std::cerr << reader.Warning() << std::endl;
    }

    tinyobj::attrib_t attrib = reader.GetAttrib();
    vector<tinyobj::material_t> materials = reader.GetMaterials();
    vector<tinyobj::shape_t> shapes = reader.GetShapes();

    raw_shapes = new shape[shapes.size()];
    shapesCount = shapes.size();

    for (size_t s = 0; s < shapes.size(); s++) {

        int trianglesCount = shapes[s].mesh.num_face_vertices.size();
        shape shape(trianglesCount);

        size_t v_index_offset = 0; 
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

            
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if(fv != 3)
                cerr << "Robust Triangulation is not working. Raytracing stopped" << endl;       
            
            tinyobj::index_t idx0 = shapes[s].mesh.indices[v_index_offset + 0];
            tinyobj::index_t idx1 = shapes[s].mesh.indices[v_index_offset + 1];
            tinyobj::index_t idx2 = shapes[s].mesh.indices[v_index_offset + 2];
            

            vec3 p0(attrib.vertices[3*size_t(idx0.vertex_index)+0],attrib.vertices[3*size_t(idx0.vertex_index)+1],attrib.vertices[3*size_t(idx0.vertex_index)+2]);
            vec3 p1(attrib.vertices[3*size_t(idx1.vertex_index)+0],attrib.vertices[3*size_t(idx1.vertex_index)+1],attrib.vertices[3*size_t(idx1.vertex_index)+2]);
            vec3 p2(attrib.vertices[3*size_t(idx2.vertex_index)+0],attrib.vertices[3*size_t(idx2.vertex_index)+1],attrib.vertices[3*size_t(idx2.vertex_index)+2]);

            vec3 normal(attrib.normals[3*idx0.normal_index+0], attrib.normals[3*idx0.normal_index+1], attrib.normals[3*idx0.normal_index+2]);

            int materialIdx = shapes[s].mesh.material_ids[f];

            triangle tr(p0, p1, p2, normal);
            shape.triangles[f] = tr;
            raw_shapes[s] = shape;
            raw_shapes[s].color = vec3{materials[materialIdx].diffuse[0], materials[materialIdx].diffuse[1], materials[materialIdx].diffuse[2]};
            v_index_offset += fv;
        }
    }

    return 0;
}