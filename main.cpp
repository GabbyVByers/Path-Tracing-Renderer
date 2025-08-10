
#include <iostream>
#include "input.h"
#include "selector.h"
#include "framerate.h"

int main()
{
    const bool FULLSCREEN = false;
    Opengl opengl;
    setupOpengl(opengl, 1920, 1080, "CUDA-Powered Path-Tracing", FULLSCREEN);
    setupImgui(opengl.window);

    World world;
    fixCamera(world.camera);
    buildHashArrayAndFrameBuffer(world.buffer, screenSize(opengl));
    initializeBoxes(world.boxes);

    Sphere sphere;
    world.spheres.add(sphere);
    world.spheres.updateHostToDevice();

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

    freeBoxes(world.boxes);
    freeOpengl(opengl);
    return 0;
}

