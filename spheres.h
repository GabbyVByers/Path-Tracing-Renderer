#pragma once

#include "vec3.h"

struct Sphere
{
    Vec3 position;
    float radius = 0.0f;
    Vec3 color;
    float roughness = 1.0f;
    bool is_selected = false;
};

struct Spheres
{
    int num_spheres = 0;
    Sphere* host_spheres = nullptr;
    Sphere* device_spheres = nullptr;
};

inline void initialize_spheres(Spheres& spheres)
{
    spheres.num_spheres = 5;
    spheres.host_spheres = nullptr;
    spheres.host_spheres = new Sphere[spheres.num_spheres];
    spheres.host_spheres[0] = { {  0.0f, -90.0f,   0.0f }, 89.0f, rgb( 33,  45, 255), 0.0f, false };
    spheres.host_spheres[1] = { { -8.0f,   1.0f,   0.0f },  2.5f, rgb( 33, 255,  89), 1.0f, false };
    spheres.host_spheres[2] = { { -2.6f,   1.0f,   0.0f },  2.5f, rgb(255,  67, 201), 0.0f, false };
    spheres.host_spheres[3] = { {  2.6f,   1.0f,   0.0f },  2.5f, rgb(255,   0,   0), 0.0f, false };
    spheres.host_spheres[4] = { {  8.0f,   1.0f,   0.0f },  2.5f, rgb(255, 255, 255), 1.0f, false };
    cudaMalloc((void**)&spheres.device_spheres, sizeof(Sphere) * spheres.num_spheres);
}

inline void update_spheres_on_gpu(Spheres& spheres)
{
    cudaMemcpy(spheres.device_spheres, spheres.host_spheres, sizeof(Sphere) * spheres.num_spheres, cudaMemcpyHostToDevice);
}

inline void free_spheres(Spheres& spheres)
{
    delete[] spheres.host_spheres;
    cudaFree(spheres.device_spheres);
}

