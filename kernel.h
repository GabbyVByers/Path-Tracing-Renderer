#pragma once

#include "thread.h"
#include "world.h"
#include <cfloat>

struct HitInfo
{
    float closest_t = FLT_MAX;
    bool didHit = false;
    Vec3 hitLocation;
    Vec3 hitColor;
    Vec3 hitNormal;
    float hitRoughness;
    bool hitIsLightSource;
    float hitLightIntensity;
};

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

__device__ inline HitInfo rayBoxesIntersection(const Ray& ray, const World& world)
{
    HitInfo info;

    for (int i = 0; i < world.boxes.size; i++)
    {
        Box& box = world.boxes.devicePointer[i];
        Vec3& position = box.position;
        Vec3& size = box.size;

        Vec3 min = position - size;
        Vec3 max = position + size;

        float normalSignX = -1.0f;
        float normalSignY = -1.0f;
        float normalSignZ = -1.0f;

        float tminx = (min.x - ray.origin.x) / ray.direction.x;
        float tmaxx = (max.x - ray.origin.x) / ray.direction.x;
        if (tminx > tmaxx) { float temp = tminx; tminx = tmaxx; tmaxx = temp; normalSignX = 1.0f; }

        float tminy = (min.y - ray.origin.y) / ray.direction.y;
        float tmaxy = (max.y - ray.origin.y) / ray.direction.y;
        if (tminy > tmaxy) { float temp = tminy; tminy = tmaxy; tmaxy = temp; normalSignY = 1.0f; }

        float tminz = (min.z - ray.origin.z) / ray.direction.z;
        float tmaxz = (max.z - ray.origin.z) / ray.direction.z;
        if (tminz > tmaxz) { float temp = tminz; tminz = tmaxz; tmaxz = temp; normalSignZ = 1.0f; }

        float tmin = fmaxf(fmaxf(tminx, tminy), tminz);
        float tmax = fminf(fminf(tmaxx, tmaxy), tmaxz);

        if (tmin > tmax)
            continue;

        if (tmin < 0.0f)
            continue;

        Vec3 normal;

        if (tmin == tminx)
        {
            normal = { 1.0f, 0.0f, 0.0f };
            normal *= normalSignX;
        }

        if (tmin == tminy)
        {
            normal = { 0.0f, 1.0f, 0.0f };
            normal *= normalSignY;
        }

        if (tmin == tminz)
        {
            normal = { 0.0f, 0.0f, 1.0f };
            normal *= normalSignZ;
        }
        
        if (tmin < info.closest_t)
        {
            info.didHit = true;
            info.closest_t = tmin;
            info.hitLocation = ray.origin + (ray.direction * tmin);
            info.hitColor = world.boxes.devicePointer[i].color;
            info.hitNormal = normal;
            info.hitRoughness = world.boxes.devicePointer[i].roughness;
            info.hitIsLightSource = world.boxes.devicePointer[i].isLightSource;
            info.hitLightIntensity = world.boxes.devicePointer[i].lightIntensity;
        }
    }

    return info;
}

__device__ inline HitInfo raySpheresIntersection(const Ray& ray, World& world)
{
    HitInfo info;

    for (int i = 0; i < world.spheres.size; i++)
    {
        Sphere& deviceSpherePointer = world.spheres.devicePointer[i];

        Vec3 V = ray.origin - deviceSpherePointer.position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (deviceSpherePointer.radius * deviceSpherePointer.radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        if (t < info.closest_t)
        {
            info.closest_t = t;
            info.didHit = true;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitColor = deviceSpherePointer.color;
            info.hitNormal = info.hitLocation - deviceSpherePointer.position; normalize(info.hitNormal);
            info.hitRoughness = deviceSpherePointer.roughness;
            info.hitIsLightSource = deviceSpherePointer.isLightSource;
            info.hitLightIntensity = deviceSpherePointer.lightIntensity;
        }
    }

    return info;
}

__device__ inline Vec3 environmentLight(const Ray& ray, const World& world)
{
    float zenithHorizonGradient = pow(smoothstep(0.0f, 0.5f, ray.direction.y), world.sky.horizonExponent);
    float sunMask = pow(fmaxf(0.0f, dot(ray.direction, world.sky.sunDirection)), world.sky.sunExponent);
    float groundSkyGradient = smoothstep(-0.01f, 0.0f, ray.direction.y);
    Vec3 colorSky = lerpVec3(world.sky.colorZenith, world.sky.colorHorizon, zenithHorizonGradient);
    colorSky = colorSky + ((world.sky.colorSun * sunMask) * world.sky.sunIntensity);
    return lerpVec3(world.sky.colorGround, colorSky, groundSkyGradient);
}

__device__ inline Vec3 calculateIncomingLight(Ray ray, Thread& thread, World& world)
{
    Vec3 color, light;
    color = 1.0f;
    light = 0.0f;

    for (int i = 0; i < world.global.maxBounceLimit; i++)
    {
        HitInfo info = raySpheresIntersection(ray, world);
        HitInfo boxInfo = rayBoxesIntersection(ray, world);
        if (boxInfo.closest_t < info.closest_t)
            info = boxInfo;
    
        if (!info.didHit)
        {
            if (world.sky.toggleSky)
                light = environmentLight(ray, world);
            break;
        }
    
        if (info.hitIsLightSource)
        {
            light = info.hitColor * info.hitLightIntensity;
            break;
        }
    
        color = color * info.hitColor;
    
        ray.origin = info.hitLocation + (info.hitNormal * 0.0001f);
    
        Vec3 diffuseDirection = randomHemisphereDirection(info.hitNormal, *thread.hashPtr) + randomDirection(*thread.hashPtr);
        Vec3 specularDirection = reflect(ray.direction, info.hitNormal);
    
        ray.direction = lerpVec3(diffuseDirection, specularDirection, info.hitRoughness);
        normalize(ray.direction);
    }

    return color * light;
}

__device__ inline void frameAccumulation(Vec3 newColor, const Thread& thread, World& world)
{
    Vec3& accumulatedColor = world.global.accumulatedFrameBuffer[thread.index];
    
    if (world.global.numAccumulatedFrames == 0)
        accumulatedColor = 0.0f;

    accumulatedColor += newColor;
    Vec3 thisColor = accumulatedColor / (world.global.numAccumulatedFrames + 1);

    unsigned char r = fminf(255.0f, thisColor.x * 255.0f);
    unsigned char g = fminf(255.0f, thisColor.y * 255.0f);
    unsigned char b = fminf(255.0f, thisColor.z * 255.0f);
    world.global.pixels[thread.index] = make_uchar4(r, g, b, 255);
}

__global__ inline void mainKernel(World world)
{
    Thread thread = getThread(world.global.screenWidth, world.global.screenHeight, world.global.deviceHashArray);
    if (thread.index == -1)
        return;

    Ray ray;
    ray.origin = world.camera.position;
    ray.direction = (world.camera.direction * world.camera.depth) + (world.camera.right * thread.u) + (world.camera.up * thread.v);
    ray.direction += randomDirection(*thread.hashPtr) * world.global.antiAliasing;
    normalize(ray.direction);

    Vec3 color = calculateIncomingLight(ray, thread, world);
    frameAccumulation(color, thread, world);
    return;
}

