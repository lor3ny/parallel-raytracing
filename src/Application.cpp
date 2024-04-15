#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <mpi.h>

#include "SceneHandler.h"
#include "Log.h"
#include "Renderer.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "cudaRaytracing.h"


#define WIDTH 1920
#define HEIGHT 1080

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    double start = MPI_Wtime();

    unsigned char* buff = new unsigned char[WIDTH*HEIGHT*3];
    Renderer renderer(WIDTH, HEIGHT);
    SceneHandler scene;

    scene.LoadScene("../test/cornell_box.obj", "../test/");

    Log::Print("Scene loaded.");
    Log::Print("Rendering...");

    renderer.Render(buff, scene);

    stbi_write_png("res.png", WIDTH, HEIGHT, 3, buff, WIDTH*3);

    Log::Print("Image saved.");

    delete[] buff;

    double end = MPI_Wtime();

    std::cout << "Time spent: " << end-start << "s" << endl;

    MPI_Finalize();
}