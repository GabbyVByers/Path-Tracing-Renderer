
#include <iostream>
#include "cuda_interop.h"

int main()
{
    opengl opengl;
    const bool FULLSCREEN = false;
    setup_opengl(opengl, 1920, 1080, "CUDA-Powered Ray-Tracing", FULLSCREEN);
    setup_imgui(opengl.window);

    // Spheres
    int num_spheres = 5;
    sphere* host_spheres = nullptr;
    host_spheres = new sphere[num_spheres];
    host_spheres[0] = { {  0.0f, -90.0f,   0.0f },  90.0f, { 0.2f, 0.2f, 1.0f} };
    host_spheres[1] = { { -8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 1.0f, 1.0f} };
    host_spheres[2] = { { -2.6f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 0.2f} };
    host_spheres[3] = { {  2.6f,   1.0f,   0.0f },   2.5f, { 0.2f, 1.0f, 0.2f} };
    host_spheres[4] = { {  8.0f,   1.0f,   0.0f },   2.5f, { 1.0f, 0.2f, 1.0f} };
    sphere* dev_spheres = nullptr; 
    cudaMalloc((void**)&dev_spheres, sizeof(sphere) * num_spheres);
    cudaMemcpy(dev_spheres, host_spheres, sizeof(sphere) * num_spheres, cudaMemcpyHostToDevice);

    // Camera
    camera camera;
    camera.rays_per_pixel = 1;
    camera.light_direction = { 1.0f, 1.0f, 1.0f };
    normalize(camera.light_direction);
    camera.position  = { 11.3f, 8.0f, -10.0f };
    camera.direction = { -0.5f, -0.5f, 0.7f };
    camera.depth     = 1.5f;
    camera.buffer_size = 0;
    camera.buffer_limit = 1000;
    fix_camera(camera);

    unsigned int* hostHashArray = nullptr;
    hostHashArray = new unsigned int[screen_size(opengl)];
    for (int i = 0; i < screen_size(opengl); i++)
    {
        unsigned int hash = i;
        hash_uint32(hash);
        hostHashArray[i] = hash;
    }
    cudaMalloc((void**)&camera.device_hash_array, sizeof(unsigned int) * screen_size(opengl));
    cudaMemcpy(camera.device_hash_array, hostHashArray, sizeof(unsigned int) * screen_size(opengl), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&camera.device_true_frame_buffer, sizeof(vec3) * screen_size(opengl));

    while (!glfwWindowShouldClose(opengl.window))
    {
        launch_cuda_kernel(opengl, dev_spheres, num_spheres, camera);
        process_keyboard_mouse_input(opengl, camera);
        draw_imgui(camera);
        render_textured_quad(opengl, camera);
    }

    free_opengl(opengl);
    return 0;
}

