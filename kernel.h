#pragma once

#include "thread.h"
#include "world.h"
#include <cfloat>

struct HitInfo {
    bool didHit = false;
    Vec3 hitLocation;
    Vec3 hitColor;
    Vec3 hitNormal;
    float hitRoughness;
    bool hitIsLightSource;
    float hitLightIntensity;
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

__device__ inline HitInfo raySpheresIntersection(const Ray& ray, const World& world) {
    HitInfo info;
    float closest_t = FLT_MAX;

    for (int i = 0; i < world.spheres.numSpheres; i++) {
        Vec3 V = ray.origin - world.spheres.deviceSpheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (world.spheres.deviceSpheres[i].radius * world.spheres.deviceSpheres[i].radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        info.didHit = true;

        if (t < closest_t) {
            closest_t = t;
            info.hitColor = world.spheres.deviceSpheres[i].color;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitNormal = info.hitLocation - world.spheres.deviceSpheres[i].position;
            normalize(info.hitNormal);
            info.hitRoughness = world.spheres.deviceSpheres[i].roughness;
            info.hitIsLightSource = world.spheres.deviceSpheres[i].isLightSource;
            info.hitLightIntensity = world.spheres.deviceSpheres[i].lightIntensity;
        }
    }

    return info;
}

__device__ inline Vec3 environmentLight(const Ray& ray, const World& world) {
    float zenithHorizonGradient = pow(smoothstep(0.0f, 0.5f, ray.direction.y), world.sky.horizonExponent);
    float sunMask = pow(fmaxf(0.0f, dot(ray.direction, world.sky.sunDirection)), world.sky.sunExponent);
    float groundSkyGradient = smoothstep(-0.01f, 0.0f, ray.direction.y);
    Vec3 colorSky = lerpVec3(world.sky.colorZenith, world.sky.colorHorizon, zenithHorizonGradient);
    colorSky = colorSky + ((world.sky.colorSun * sunMask) * world.sky.sunIntensity);
    return lerpVec3(world.sky.colorGround, colorSky, groundSkyGradient);
}

__device__ inline Vec3 calculateIncomingLight(Ray ray, Thread& thread, const World& world) {
    Vec3 color, light;
    color = 1.0f;
    light = 0.0f;

    Ray cameraRay = ray;

    for (int i = 0; i < 50; i++) {
        HitInfo info = raySpheresIntersection(ray, world);

        if (!info.didHit) {
            if (world.sky.toggleSky)
                light = environmentLight(ray, world);
            break;
        }

        if (info.hitIsLightSource) {
            light = info.hitColor * info.hitLightIntensity;
            break;
        }

        color = color * info.hitColor;

        ray.origin = info.hitLocation;

        Vec3 diffuseDirection = randomHemisphereDirection(info.hitNormal, *thread.hashPtr) + randomDirection(*thread.hashPtr);
        Vec3 specularDirection = reflect(ray.direction, info.hitNormal);

        ray.direction = lerpVec3(diffuseDirection, specularDirection, info.hitRoughness);
        normalize(ray.direction);
    }

    return color * light;
}

__device__ inline void frameAccumulation(Vec3 newColor, const Thread& thread, World& world) {
    Vec3& accumulatedColor = world.buffer.accumulatedFrameBuffer[thread.index];
    
    if (world.buffer.numAccumulatedFrames == 0)
        accumulatedColor = 0.0f;

    accumulatedColor += newColor;
    Vec3 thisColor = accumulatedColor / (world.buffer.numAccumulatedFrames + 1);

    unsigned char r = fminf(255.0f, thisColor.x * 255.0f);
    unsigned char g = fminf(255.0f, thisColor.y * 255.0f);
    unsigned char b = fminf(255.0f, thisColor.z * 255.0f);
    world.pixels[thread.index] = make_uchar4(r, g, b, 255);
}

__global__ inline void mainKernel(World world) {
    Thread thread = getThread(world.screenWidth, world.screenHeight, world.buffer.deviceHashArray);
    if (thread.index == -1)
        return;

    Ray ray;
    ray.origin = world.camera.position;
    ray.direction = (world.camera.direction * world.camera.depth) + (world.camera.right * thread.u) + (world.camera.up * thread.v);
    ray.direction += randomDirection(*thread.hashPtr) * 0.001f;
    normalize(ray.direction);

    Vec3 color = calculateIncomingLight(ray, thread, world);
    frameAccumulation(color, thread, world);
    return;
}

