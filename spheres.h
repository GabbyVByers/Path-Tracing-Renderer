#pragma once

#include "vec3.h"

struct Sphere
{
    Vec3 position = { 0.0f, 0.0f, 0.0f };
    float radius = 1.0f;
    Vec3 color = rgb(255, 255, 255);
    float roughness = 0.5f;
    bool isSelected = false;
    bool isLightSource = false;
    float lightIntensity = 1.0f;
};

struct Spheres
{
    int numSpheres = 0;
    Sphere* hostSpheres = nullptr;
    Sphere* deviceSpheres = nullptr;
};

inline void updateSpheresOnGpu(Spheres& spheres)
{
    cudaMemcpy(spheres.deviceSpheres, spheres.hostSpheres, sizeof(Sphere) * spheres.numSpheres, cudaMemcpyHostToDevice);
}

inline void freeSpheres(Spheres& spheres)
{
    delete[] spheres.hostSpheres;
    cudaFree(spheres.deviceSpheres);
}

inline void initializeSpheres(Spheres& spheres)
{
    spheres.numSpheres = 1;
    spheres.hostSpheres = nullptr;
    spheres.hostSpheres = new Sphere[spheres.numSpheres];
    cudaMalloc((void**)&spheres.deviceSpheres, sizeof(Sphere) * spheres.numSpheres);
    updateSpheresOnGpu(spheres);
}

