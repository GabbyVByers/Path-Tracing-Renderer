#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vec3.h"
#include "camera.h"
#include "quaternions.h"
#include "dataStructures.h"
#include "random.h"
#include <cfloat>

using uint8 = unsigned char;

__device__ inline HitInfo raySpheresIntersection(const Ray& ray, const Sphere* devSpheres, const int& numSpheres)
{
    HitInfo info = { false };
    float closest_t = FLT_MAX;
    
    for (int i = 0; i < numSpheres; i++)
    {
        vec3 V = ray.origin - devSpheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (devSpheres[i].radius * devSpheres[i].radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        info.didHit = true;
        
        if (t < closest_t)
        {
            closest_t = t;
            info.didHitLightSource = devSpheres[i].isLightSource;
            info.hitColor = devSpheres[i].color;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitNormal = info.hitLocation - devSpheres[i].position;
            normalize(info.hitNormal);
        }
    }

    return info;
}

__device__ inline vec3 skyColor(const vec3& direction)
{
    vec3 lightDirection = { 0.57735f, 0.57735f, 0.57735f };
    float intensity = (dot(lightDirection, direction) + 1.0f) / 2.0f;
    return { intensity, intensity, intensity };
}

__device__ inline vec3 calculateIncomingLight(Ray ray,
                                              const Sphere* devSpheres,
                                              const int& numSpheres,
                                              const int& maxBounceLimit,
                                              uint32& randomState)
{
    vec3 rayColor = { 1.0f, 1.0f, 1.0f };
    vec3 incomingLight = { 0.0f, 0.0f, 0.0f };

    for (int i = 0; i < maxBounceLimit; i++)
    {
        HitInfo info = raySpheresIntersection(ray, devSpheres, numSpheres);

        if (!info.didHit)
            break;
        
        if (info.didHitLightSource)
        {
            incomingLight = multiply(rayColor, info.hitColor);
            break;
        }

        //if (!info.didHit)
        //{
        //    incomingLight = multiply(rayColor, skyColor(ray.direction));
        //    break;
        //}

        rayColor *= info.hitColor;

        ray.origin = info.hitLocation;
        ray.direction = randomHemisphereDirection(info.hitNormal, randomState);
    }

    return incomingLight;
}

__global__ inline void renderKernel(uchar4* pixels,
                                    int width,
                                    int height,
                                    Sphere* devSpheres,
                                    int numSpheres,
                                    Camera camera)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint32 randomState = idx;

    float u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    float v = (y / (float)height) * 2.0f - 1.0f;

    vec3 incomingLight = { 0.0f, 0.0f, 0.0f };
    Ray ray = { camera.position, (camera.direction * camera.depth) + (camera.up * v) + (camera.right * u) };
    normalize(ray.direction);
    for (int i = 0; i < camera.raysPerPixel; i++)
    {
        incomingLight += calculateIncomingLight(ray, devSpheres, numSpheres, camera.maxBounceLimit, randomState);
    }
    incomingLight /= camera.raysPerPixel;

    uint8 r = incomingLight.x * 255.0f;
    uint8 g = incomingLight.y * 255.0f;
    uint8 b = incomingLight.z * 255.0f;

    pixels[idx] = make_uchar4(r, g, b, 255);
    return;
}

