#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "SceneHandler.h"
#include "Log.h"
#include "Renderer.h"
#include "stb_image.h"
#include "stb_image_write.h"


#define WIDTH 1920
#define HEIGHT 1080

int main(){

    SceneHandler scene;

    scene.LoadScene("../test/cornell_box.obj", "../test/");


    Log::Print("tutt'apposto");


    Renderer renderer(WIDTH, HEIGHT);

    unsigned char* buff = new unsigned char[WIDTH*HEIGHT*3];
    renderer.Render(buff, scene);

    // Save the image as a PNG file
    stbi_write_png("res.png", WIDTH, HEIGHT, 3, buff, WIDTH*3);

    // Free the image data
    delete[] buff;

    //std::cout << "Image saved successfully!" << std::endl;

    return 0;
}