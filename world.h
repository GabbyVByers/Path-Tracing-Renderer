#pragma once

#include "vec3.h"
#include "spheres.h"
#include "utilities.h"

struct Camera
{
    Vec3 position = { 11.3f, 8.0f, -10.0f };
    Vec3 direction = { -0.5f, -0.5f, 0.7f };
    Vec3 up;
    Vec3 right;
    float depth = 1.5f;
};

struct Sky
{
    Vec3 sun_direction = return_normalized({ 1.0f, 1.0f, 1.0f });
    float sun_intensity = 35.0f;
    float sun_exponent = 75.0f;
    float horizon_exponent = 0.35f;

    Vec3 color_sun     = rgb(255, 255, 255);
    Vec3 color_zenith  = rgb(255, 255, 255);
    Vec3 color_horizon = rgb( 63, 195, 235);
    Vec3 color_ground  = rgb( 79, 112,  76);
};

struct Buffer
{
    int num_accumulated_frames = 0;
    Vec3* accumulated_frame_buffer;
    unsigned int* device_hash_array = nullptr;
};

struct World
{   
    uchar4* pixels = nullptr;
    int screen_width = 0;
    int screen_height = 0;

    Camera camera;
    Sky sky;
    Spheres spheres;
    Buffer buffer;
};

inline void fix_camera(Camera& camera)
{
    const Vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = cross(camera.direction, up);
    normalize(camera.right);
    camera.up = cross(camera.right, camera.direction);
    normalize(camera.up);
}

inline void build_hash_array_and_frame_buffer(Buffer& buffer, int screen_size)
{
    unsigned int* host_hash_array = nullptr;
    host_hash_array = new unsigned int[screen_size];
    for (int i = 0; i < screen_size; i++)
    {
        unsigned int hash = i;
        hash_uint32(hash);
        host_hash_array[i] = hash;
    }
    cudaMalloc((void**)&buffer.device_hash_array, sizeof(unsigned int) * screen_size);
    cudaMemcpy(buffer.device_hash_array, host_hash_array, sizeof(unsigned int) * screen_size, cudaMemcpyHostToDevice);
    delete[] host_hash_array;
    cudaMalloc((void**)&buffer.accumulated_frame_buffer, sizeof(Vec3) * screen_size);
}

