
#include <iostream>
#include "input.h"

/*

TODO

Select & Edit Sphere.
Add Sphere.
Delete Sphere.

*/

int main()
{
    Opengl opengl;
    const bool FULLSCREEN = true;
    setup_opengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setup_imgui(opengl.window);

    World world;
    fix_camera(world.camera);
    build_hash_array_and_frame_buffer(world.buffer, screen_size(opengl));
    initialize_spheres(world.spheres);
    update_spheres_on_gpu(world.spheres);

    while (!glfwWindowShouldClose(opengl.window))
    {
        launch_cuda_kernel(opengl, world);
        process_keyboard_mouse_input(opengl, world);
        render_screen(opengl);
        draw_imgui(world);
        finish_rendering(opengl);
    }

    free_spheres(world.spheres);
    free_opengl(opengl);
    return 0;
}

