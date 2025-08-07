#pragma once

#include "vec3.h"

struct Sphere
{
    Vec3 position;
    float radius = 0.0f;
    Vec3 color;
    float roughness = 1.0f;
    bool is_selected = false;
    bool is_light_source = false;
    float light_intensity = 1.0f;
};

struct Spheres
{
    int num_spheres = 0;
    Sphere* host_spheres = nullptr;
    Sphere* device_spheres = nullptr;
};

inline void update_spheres_on_gpu(Spheres& spheres)
{
    cudaMemcpy(spheres.device_spheres, spheres.host_spheres, sizeof(Sphere) * spheres.num_spheres, cudaMemcpyHostToDevice);
}

inline void free_spheres(Spheres& spheres)
{
    delete[] spheres.host_spheres;
    cudaFree(spheres.device_spheres);
}

inline void initialize_spheres(Spheres& spheres)
{
    spheres.num_spheres = 1;
    spheres.host_spheres = nullptr;
    spheres.host_spheres = new Sphere[spheres.num_spheres];
    spheres.host_spheres[0] = { { 0.0f, 0.0f, 0.0f }, 1.0f, rgb(255, 255, 255), 1.0f, false };
    cudaMalloc((void**)&spheres.device_spheres, sizeof(Sphere) * spheres.num_spheres);
    update_spheres_on_gpu(spheres);
}

