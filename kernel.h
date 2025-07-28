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
            info.hitColor = devSpheres[i].color;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitNormal = info.hitLocation - devSpheres[i].position;
            normalize(info.hitNormal);
        }
    }

    return info;
}

__device__ inline vec3 skyColor(const vec3& direction, const vec3& lightDirection)
{
    vec3 skyWhite    = rgb(255, 255, 255);
    vec3 skyBlue     = rgb(57, 162, 237);
    vec3 groundColor = rgb(143, 136, 130);

    if (direction.y < 0.0f)
        return groundColor;

    if (dot(lightDirection, direction) > 0.997f)
        return skyWhite;
    
    return skyBlue;
}

__device__ inline vec3 calculateIncomingLight(Ray ray, const Sphere* devSpheres, const int& numSpheres, uint32& randomState, const Camera& camera)
{
    vec3 rayColor = { 1.0f, 1.0f, 1.0f };

    HitInfo info = raySpheresIntersection(ray, devSpheres, numSpheres);

    if (!info.didHit)
    {
        return skyColor(ray.direction, camera.lightDirection);
    }

    rayColor *= info.hitColor;


    ray.origin = info.hitLocation + (info.hitNormal * 0.001f);
    ray.direction = camera.lightDirection + (randomDirection(randomState) * 0.15f);

    HitInfo shadowInfo = raySpheresIntersection(ray, devSpheres, numSpheres);

    if (shadowInfo.didHit)
    {
        return multiply(rayColor, { 0.3f, 0.3f, 0.3f });
    }

    return rayColor;
}

__global__ inline void renderKernel(uchar4* pixels, int width, int height, Sphere* devSpheres, int numSpheres, Camera camera)
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
        incomingLight += calculateIncomingLight(ray, devSpheres, numSpheres, randomState, camera);
    }
    incomingLight /= camera.raysPerPixel;

    uint8 r = incomingLight.x * 255.0f;
    uint8 g = incomingLight.y * 255.0f;
    uint8 b = incomingLight.z * 255.0f;

    pixels[idx] = make_uchar4(r, g, b, 255);
    return;
}

