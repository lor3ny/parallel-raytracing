#include "fwd.hpp"
#include <iostream>
#include <queue>
#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <mpi.h>

#include "Scene.h"
#include "Renderer.h"
#include "stb_image.h"
#include "stb_image_write.h"


#define WIDTH 1920
#define HEIGHT 1080

using namespace std;

int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);

    double start = MPI_Wtime();

    int id, count;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &count);

    int batchSize = HEIGHT/count;
    unsigned char* smallBuff = new unsigned char[WIDTH*batchSize*3];
    unsigned char* bigBuff;

    Scene scene;
    //scene.LoadScene("../test/cornell_box.obj", "../test/");
    scene.LoadScene("../test/cornell_box.obj", "../test/");

    vec3 origin = {250,250,-1000};
    Camera camera(origin, 1000, WIDTH, HEIGHT);

    int camera_start = batchSize * id;
    int camera_end = batchSize * (id+1);

    if(id==0){
        bigBuff = new unsigned char[WIDTH*HEIGHT*3];

        cout << id << ": scene loaded" << endl;

        Renderer renderer(&camera, camera_start, batchSize);

        cout << id << ": Rendering started." << endl;
        renderer.Render(smallBuff, scene);

        cout << id << ": Rendering done." << endl;

        MPI_Gather(smallBuff, WIDTH*batchSize*3, MPI_UNSIGNED_CHAR,
               bigBuff, WIDTH*batchSize*3, MPI_UNSIGNED_CHAR, 0,
               MPI_COMM_WORLD);

        cout << id << ": Data received." << endl;

    } else {

        cout << id << ": Rendering started." << endl;

        Renderer renderer(&camera, camera_start, batchSize);
        renderer.Render(smallBuff, scene);

        cout << id << ": Rendering done." << endl;

        MPI_Gather(smallBuff, WIDTH*batchSize*3, MPI_UNSIGNED_CHAR,
               nullptr, WIDTH*batchSize*3, MPI_UNSIGNED_CHAR, 0,
               MPI_COMM_WORLD);

        cout << id << ": Data sent." << endl;
    }

    if(id==0){
        stbi_write_png("res.png", WIDTH, HEIGHT, 3, bigBuff, WIDTH*3);
        double end = MPI_Wtime();
        std::cout << "Time spent: " << end-start << "s" << endl;
    }

    delete[] smallBuff;
    if(id==0){
        delete[] bigBuff;
    }

    MPI_Finalize();
}