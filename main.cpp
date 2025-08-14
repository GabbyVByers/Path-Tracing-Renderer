
#include <iostream>
#include "input.h"
#include "selector.h"
#include "framerate.h"

int main()
{
    Opengl opengl;
    const bool FULLSCREEN = true;
    setupOpengl(opengl, 1920, 1080, "CUDA-Powered Path-Tracing", FULLSCREEN);
    setupImgui(opengl.window);

    World world;
    fixCamera(world.camera);
    buildHashArrayAndFrameBuffer(world.global, screenSize(opengl));

    Sphere sphere;
    world.spheres.add(sphere);
    world.spheres.updateHostToDevice();

    Box box;
    world.boxes.add(box);
    world.boxes.updateHostToDevice();

    FrameRateTracker FPSTracker;
    char fileName[24] = "";

    while (!glfwWindowShouldClose(opengl.window))
    {
        FPSTracker.update();
        selectSphere(opengl, world);
        launchCudaKernel(opengl, world);
        processKeyboardMouseInput(opengl, world);
        renderScreen(opengl);
        drawImgui(opengl.enableGUI, opengl.window, world, fileName, FPSTracker.getFPS());
    }

    world.spheres.free();
    world.boxes.free();
    freeOpengl(opengl);
    return 0;
}

