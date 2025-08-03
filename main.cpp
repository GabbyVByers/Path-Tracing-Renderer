
#include <iostream>
#include "input.h"
#include "random.h"

int main()
{
    Opengl opengl;
    const bool FULLSCREEN = false;
    setup_opengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setup_imgui(opengl.window);

    // Camera
    Camera camera;
    fix_camera(camera);

    // World
    World world;
    normalize(world.light_direction);

    unsigned int* hostHashArray = nullptr;
    hostHashArray = new unsigned int[screen_size(opengl)];
    for (int i = 0; i < screen_size(opengl); i++)
    {
        unsigned int hash = i;
        hash_uint32(hash);
        hostHashArray[i] = hash;
    }
    cudaMalloc((void**)&world.device_hash_array, sizeof(unsigned int) * screen_size(opengl));
    cudaMemcpy(world.device_hash_array, hostHashArray, sizeof(unsigned int) * screen_size(opengl), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&world.accumulated_frame_buffer, sizeof(Vec3) * screen_size(opengl));

    // Spheres
    world.num_spheres = 5;
    Sphere* host_spheres = nullptr;
    host_spheres = new Sphere[world.num_spheres];
    host_spheres[0] = { {  0.0f, -90.0f,   0.0f }, 90.0f, { 1.0f, 1.0f, 1.0f}, 0.3f };
    host_spheres[1] = { { -8.0f,   1.0f,   0.0f },  2.5f, { 0.2f, 0.2f, 1.0f}, 1.0f };
    host_spheres[2] = { { -2.6f,   1.0f,   0.0f },  2.5f, { 1.0f, 0.2f, 0.2f}, 1.0f };
    host_spheres[3] = { {  2.6f,   1.0f,   0.0f },  2.5f, { 0.2f, 1.0f, 0.2f}, 1.0f };
    host_spheres[4] = { {  8.0f,   1.0f,   0.0f },  2.5f, { 1.0f, 0.2f, 1.0f}, 1.0f };
    cudaMalloc((void**)&world.device_spheres, sizeof(Sphere) * world.num_spheres);
    cudaMemcpy(world.device_spheres, host_spheres, sizeof(Sphere) * world.num_spheres, cudaMemcpyHostToDevice);

    while (!glfwWindowShouldClose(opengl.window))
    {
        launch_cuda_kernel(opengl, world, camera);
        process_keyboard_mouse_input(opengl, world, camera);
        render_screen(opengl, camera);
        draw_imgui(world, camera);
        finish_rendering(opengl);
    }

    free_opengl(opengl);
    return 0;
}

