#include "interopRenderer.h"
#include <iostream>
#include "random.h"
#include "dataStructures.h"

int main()
{
    interop_renderer renderer(1920, 1080, "CUDA OpenGL Path Tracer", false);

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
    camera cam;
    cam.rays_per_pixel = 1;
    cam.light_direction = { 1.0f, 1.0f, 1.0f };
    normalize(cam.light_direction);
    cam.position  = { 11.3f, 8.0f, -10.0f };
    cam.direction = { -0.5f, -0.5f, 0.7f };
    cam.depth     = 1.5f;
    cam.buffer_size = 0;
    cam.buffer_limit = 500;
    fix_camera(cam);

    unsigned int* hostHashArray = nullptr;
    hostHashArray = new unsigned int[renderer.screen_size()];
    for (int i = 0; i < renderer.screen_size(); i++)
    {
        unsigned int hash = i;
        hash_uint32(hash);
        hostHashArray[i] = hash;
    }
    cudaMalloc((void**)&cam.device_hash_array, sizeof(unsigned int) * renderer.screen_size());
    cudaMemcpy(cam.device_hash_array, hostHashArray, sizeof(unsigned int) * renderer.screen_size(), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cam.device_true_frame_buffer, sizeof(vec3) * renderer.screen_size());

    while (!glfwWindowShouldClose(renderer.window))
    {
        renderer.launch_cuda_kernel(dev_spheres, num_spheres, cam);
        renderer.process_keyboard_mouse_input(cam);
        renderer.render_textured_quad(cam);
    }

    return 0;
}

