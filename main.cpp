
#include <iostream>
#include "input.h"
#include "selector.h"

int main()
{
    Opengl opengl;
    const bool FULLSCREEN = false;
    setup_opengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setup_imgui(opengl.window);

    World world;
    fix_camera(world.camera);
    build_hash_array_and_frame_buffer(world.buffer, screen_size(opengl));
    initialize_spheres(world.spheres);

    char file_name[24] = "";
    while (!glfwWindowShouldClose(opengl.window))
    {
        select_sphere(opengl, world);
        launch_cuda_kernel(opengl, world);
        process_keyboard_mouse_input(opengl, world);
        render_screen(opengl);
        draw_imgui(world, file_name);
        finish_rendering(opengl);
    }

    free_spheres(world.spheres);
    free_opengl(opengl);
    return 0;
}

