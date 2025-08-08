
#include <iostream>
#include "input.h"
#include "selector.h"
#include "framerate.h"

int main()
{
    const bool FULLSCREEN = true;
    Opengl opengl;
    setupOpengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setupImgui(opengl.window);

    World world;
    fixCamera(world.camera);
    buildHashArrayAndFrameBuffer(world.buffer, screenSize(opengl));
    initializeSpheres(world.spheres);
    initializeBoxes(world.boxes);

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

    freeSpheres(world.spheres);
    freeOpengl(opengl);
    return 0;
}

