
#include <iostream>
#include "input.h"
#include "selector.h"
#include "framerate.h"

int main()
{
    Opengl opengl;
    const bool FULLSCREEN = false;
    setupOpengl(opengl, 1920, 1080, "CUDA-Powered Path-Tracing", FULLSCREEN);
    setupImgui(opengl.window);

    World world;
    fixCamera(world.camera);
    buildHashArrayAndFrameBuffer(world.metadata, screenSize(opengl));

    Sphere sphere;
    world.spheres.add(sphere);
    world.spheres.updateHostToDevice();

    Box box;
    world.boxes.add(box);
    world.boxes.updateHostToDevice();

    FrameRateTracker frameRateTracker;
    char fileName[24] = "";

    while (!glfwWindowShouldClose(opengl.window))
    {
        frameRateTracker.update();
        selectSphere(opengl, world);
        launchCudaKernel(opengl, world);
        processKeyboardMouseInput(opengl, world);
        renderScreen(opengl);
        drawImgui(world, fileName, frameRateTracker.frameRate);
        finishRendering(opengl);
    }

    world.spheres.free();
    world.boxes.free();
    freeOpengl(opengl);
    return 0;
}

