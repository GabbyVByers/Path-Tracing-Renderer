#include "interopRenderer.h"
#include <iostream>
#include "random.h"
#include "dataStructures.h"

int main()
{
    InteropRenderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

    // Spheres
    int numSpheres = 5;
    Sphere* hostSpheres = nullptr;
    hostSpheres = new Sphere[numSpheres];
    hostSpheres[0] = { {  0.0f, -90.0f,   0.0f },  90.0f, { 0.2f, 0.2f, 1.0f} };
    hostSpheres[1] = { { -8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 1.0f, 1.0f} };
    hostSpheres[2] = { { -2.6f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 0.2f} };
    hostSpheres[3] = { {  2.6f,   1.0f,   0.0f },   2.5f, { 0.2f, 1.0f, 0.2f} };
    hostSpheres[4] = { {  8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 1.0f} };
    Sphere* devSpheres = nullptr; 
    cudaMalloc((void**)&devSpheres, sizeof(Sphere) * numSpheres);
    cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);

    // Camera
    Camera camera;
    camera.raysPerPixel = 1;
    camera.lightDirection = { 1.0f, 1.0f, 1.0f };
    normalize(camera.lightDirection);
    camera.position  = { 11.3f, 8.0f, -10.0f };
    camera.direction = { -0.5f, -0.5f, 0.7f };
    camera.depth     = 1.5f;
    camera.uniqeFrameHash = 42069;
    camera.bufferSize = 0;
    camera.bufferLimit = 100;
    fixCamera(camera);

    while (!glfwWindowShouldClose(renderer.window))
    {
        hash_uint32(camera.uniqeFrameHash);
        renderer.launchCudaKernel(devSpheres, numSpheres, camera);
        renderer.processKeyboardMouseInput(camera);
        renderer.renderTexturedQuad(camera);
    }

    return 0;
}

