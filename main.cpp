
#include <iostream>
#include "cuda.h"
#include "input.h"

int main()
{
    opengl opengl;
    const bool FULLSCREEN = false;
    setup_opengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setup_imgui(opengl.window);

    

    // Camera
    camera camera;
    camera.position  = { 11.3f, 8.0f, -10.0f };
    camera.direction = { -0.5f, -0.5f, 0.7f };
    camera.depth     = 1.5f;
    fix_camera(camera);

    // World
    world world;
    world.rays_per_pixel = 1;
    world.light_direction = { 1.0f, 1.0f, 1.0f };
    normalize(world.light_direction);
    world.buffer_size = 0;
    world.buffer_limit = 1000;
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
    cudaMalloc((void**)&world.device_true_frame_buffer, sizeof(vec3) * screen_size(opengl));

    // Spheres
    world.num_spheres = 5;
    sphere* host_spheres = nullptr;
    host_spheres = new sphere[world.num_spheres];
    host_spheres[0] = { {  0.0f, -90.0f,   0.0f },  90.0f, { 0.2f, 0.2f, 1.0f} };
    host_spheres[1] = { { -8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 1.0f, 1.0f} };
    host_spheres[2] = { { -2.6f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 0.2f} };
    host_spheres[3] = { {  2.6f,   1.0f,   0.0f },   2.5f, { 0.2f, 1.0f, 0.2f} };
    host_spheres[4] = { {  8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 1.0f} };
    cudaMalloc((void**)&world.device_spheres, sizeof(sphere) * world.num_spheres);
    cudaMemcpy(world.device_spheres, host_spheres, sizeof(sphere) * world.num_spheres, cudaMemcpyHostToDevice);

    while (!glfwWindowShouldClose(opengl.window))
    {
        launch_cuda_kernel(opengl, world, camera);
        process_keyboard_input(opengl, world, camera);
        process_mouse_input(opengl, world, camera);
        render_screen(opengl, camera);
        draw_imgui(world, camera);
        finish_rendering(opengl);
    }

    free_opengl(opengl);
    return 0;
}

