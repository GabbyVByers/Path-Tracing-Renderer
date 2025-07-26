#include "interopRenderer.h"
#include <iostream>
#include "random.h"
#include "dataStructures.h"

int main()
{
    InteropRenderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

    // Spheres
    int numSpheres = 6;
    Sphere* hostSpheres = nullptr;
    hostSpheres = new Sphere[numSpheres];

    hostSpheres[0] = { false, {  0.0f, -90.0f,   0.0f },  90.0f, randomVec3(0.3f, 1.0f) };
    hostSpheres[1] = { false, { -8.0f,   1.0f,   0.0f },   2.0f, randomVec3(0.3f, 1.0f) };
    hostSpheres[2] = { false, { -2.6f,   1.0f,   0.0f },   2.0f, randomVec3(0.3f, 1.0f) };
    hostSpheres[3] = { false, {  2.6f,   1.0f,   0.0f },   2.0f, randomVec3(0.3f, 1.0f) };
    hostSpheres[4] = { false, {  8.0f,   1.0f,   0.0f },   2.0f, randomVec3(0.3f, 1.0f) };
    hostSpheres[5] = { true,  {  0.0f,  30.0f, 150.0f }, 100.0f, randomVec3(0.3f, 1.0f) };

    Sphere* devSpheres = nullptr; 
    cudaMalloc((void**)&devSpheres, sizeof(Sphere) * numSpheres);
    cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);

    // Camera
    Camera camera;
    camera.position  = { 11.3f, 8.0f, -10.0f };
    camera.direction = { -0.5f, -0.5f, 0.7f };
    camera.depth     = 2.0f;
    fixCamera(camera);

    // Controls
    int raysPerPixel = 2;
    int maxBounceLimit = 2;

    // bobbing spheres
    float theta = 0.0f;

    while (!glfwWindowShouldClose(renderer.window))
    {

        // bobbing spheres
        theta += 0.2;
        for (int i = 1; i < 5; i++)
        {
            hostSpheres[i].position.y = 2.0f * sin(theta + (float)i) + 1.0f;
        }
        if (theta >= 2.0f * 3.14159265)
            theta = 0.0f;
        cudaMemcpy(devSpheres, hostSpheres, sizeof(Sphere) * numSpheres, cudaMemcpyHostToDevice);


        renderer.launchCudaKernel(devSpheres,
                                  numSpheres,
                                  camera,
                                  raysPerPixel,
                                  maxBounceLimit);

        renderer.processKeyboardInput(camera);
        renderer.processMouseInput(camera);
        renderer.renderTexturedQuad(raysPerPixel, maxBounceLimit, camera);
    }

    return 0;
}

 