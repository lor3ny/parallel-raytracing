#pragma once

#include "tiny_obj_loader.h"

#include <string>
#include <vector>

using namespace std;

class SceneHandler {

    private:

        int scene_id;


    public:

        tinyobj::attrib_t attrib;
        vector<tinyobj::material_t> materials;
        vector<tinyobj::shape_t> shapes;

        SceneHandler();
        ~SceneHandler();


    int LoadScene(const string& path, const string& materials_path);

};