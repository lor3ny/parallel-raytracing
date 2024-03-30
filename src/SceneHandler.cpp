#include "SceneHandler.h"

SceneHandler::SceneHandler(){
    this->attrib = nullptr;
    this->materials = nullptr;
    this->shapes = nullptr;
    this->scene_id = 0;
}

SceneHandler::~SceneHandler(){
    this->attrib = nullptr;
    this->materials = nullptr;
    this->shapes = nullptr;
    this->scene_id = 0;
}


int SceneHandler::LoadScene(const string& path, const string& materials_path){

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = materials_path;

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(path, reader_config)) {
        if (!reader.Error().empty()) {
            Log::PrintError(reader.Error());
        }
        return 1;
    }

    if (!reader.Warning().empty()) {
        Log::PrintError(reader.Warning());
    }

    attrib = &reader.GetAttrib();
    shapes = &reader.GetShapes();
    materials = &reader.GetMaterials();

    return 0;
}