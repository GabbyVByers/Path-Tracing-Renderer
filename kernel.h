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

__device__ inline vec3 skyColor(const vec3& direction)
{
    vec3 lightDirection = { 1.0f, 1.0f, 1.0f };
    normalize(lightDirection);

    //float intensity = pow((dot(lightDirection, direction) + 1.0f) / 2.0f, 10.0f);
    //
    //return { intensity, intensity, intensity };

    if (dot(lightDirection, direction) > 0.9f)
    {
        return { 1.0f, 1.0f, 1.0f };
    }
    
    return { 0.25f, 0.25f, 0.5f };


    //vec3 lightDirection = { 1.0f, 1.0f, 1.0f };
    //normalize(lightDirection);
    //
    //vec3 skyBlue = { 0.325f, 0.667f, 0.9f };
    //vec3 skyWhite = { 0.9f, 0.9f, 0.9f };
    //vec3 groundColor = { 0.57f, 0.529f, 0.57f };
    //
    //if (dot(lightDirection, direction) > 0.997f)
    //{
    //    return skyWhite;
    //}
    //
    //if (direction.y < 0.0f)
    //{
    //    return groundColor;
    //}
    //
    //return skyBlue;

    //return { 1.0f, 1.0f, 1.0f };
}

__device__ inline vec3 calculateIncomingLight(Ray ray, const Sphere* devSpheres, const int& numSpheres, const int& maxBounceLimit, uint32& randomState)
{
    vec3 rayColor = { 1.0f, 1.0f, 1.0f };
    vec3 incomingLight = { 0.0f, 0.0f, 0.0f };

    bool hitAnything = false;
    HitInfo info;

    for (int i = 0; i < maxBounceLimit; i++)
    {
        info = raySpheresIntersection(ray, devSpheres, numSpheres);

        if (info.didHit)
        {
            hitAnything = true;
            rayColor *= info.hitColor;
            ray.origin = info.hitLocation;
            ray.direction = randomHemisphereDirection(info.hitNormal, randomState);
            continue;
        }

        break;
    }

    if (!hitAnything)
    {
        return skyColor(ray.direction);
    }

    vec3 lightDirection = { 1.0f, 1.0f, 1.0f };
    normalize(lightDirection);

    ray.origin += info.hitNormal * 0.01;
    ray.direction = lightDirection + (randomDirection(randomState) * 0.15f);

    HitInfo shadowInfo = raySpheresIntersection(ray, devSpheres, numSpheres);

    if (shadowInfo.didHit)
    {
        return multiply(rayColor, { 0.3f, 0.3f, 0.3f });
    }

    return rayColor;

    // if we hit nothing at all (aka sky is directly visible from camera)
    // then we return some arbitary sky color, probably some function of the ray's direction vector

    // otherwise:
    // cast a ray from last hit location towards light source + a random offset
    // return rayColor multiplied by pure white if we do not hit anything
    // return rayColor multiplied by some ambient lighting

    //for (int i = 0; i < maxBounceLimit; i++)
    //{
    //    HitInfo info = raySpheresIntersection(ray, devSpheres, numSpheres);
    //
    //    //if (!info.didHit)
    //    //    break;
    //    //
    //    //if (info.didHitLightSource)
    //    //{
    //    //    incomingLight = multiply(rayColor, info.hitColor);
    //    //    break;
    //    //}
    //
    //    if (!info.didHit)
    //    {
    //
    //
    //
    //        incomingLight = multiply(rayColor, skyColor(ray.direction));
    //        break;
    //    }
    //
    //    rayColor *= info.hitColor;
    //
    //    ray.origin = info.hitLocation;
    //    ray.direction = randomHemisphereDirection(info.hitNormal, randomState);
    //}
    //
    //return incomingLight;
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
        incomingLight += calculateIncomingLight(ray, devSpheres, numSpheres, camera.maxBounceLimit, randomState);
    }
    incomingLight /= camera.raysPerPixel;

    uint8 r = incomingLight.x * 255.0f;
    uint8 g = incomingLight.y * 255.0f;
    uint8 b = incomingLight.z * 255.0f;

    pixels[idx] = make_uchar4(r, g, b, 255);
    return;
}

