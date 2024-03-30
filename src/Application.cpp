#define TINYOBJLOADER_IMPLEMENTATION

#include "SceneHandler.h"
#include "Log.h"


int main(){

    SceneHandler scene;

    scene.LoadScene("test/CornellBox-Original.obj", "test/");

    Log::Print("tutt'apposto");

    return 0;
}