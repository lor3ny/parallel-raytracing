#pragma once

#include "tiny_obj_loader.h"

#include "Log.h"

#include <string>
#include <vector>

using namespace std;

class SceneHandler {

    private:

        int scene_id;
        const tinyobj::attrib_t* attrib;
        const vector<tinyobj::material_t>* materials;
        const vector<tinyobj::shape_t>* shapes;


    public:

    SceneHandler();
    ~SceneHandler();


    int LoadScene(const string& path, const string& materials_path);

};