#pragma once

#include "thread.h"
#include "world.h"
#include <cfloat>

struct HitInfo {
    float closest_t = FLT_MAX;
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

__device__ inline Vec3 getBoxNormal(Vec3 hitLocation, Vec3 A, Vec3 B) {
    const float epsilon = 0.0001f;

    if (fabs(hitLocation.x - A.x) < epsilon) return { -1.0f, 0.0f, 0.0f };
    if (fabs(hitLocation.x - B.x) < epsilon) return {  1.0f, 0.0f, 0.0f };

    if (fabs(hitLocation.y - A.y) < epsilon) return { 0.0f, -1.0f, 0.0f };
    if (fabs(hitLocation.y - B.y) < epsilon) return { 0.0f,  1.0f, 0.0f };

    if (fabs(hitLocation.z - A.z) < epsilon) return { 0.0f, 0.0f, -1.0f };
    if (fabs(hitLocation.z - B.z) < epsilon) return { 0.0f, 0.0f,  1.0f };

    // something terrible has happened
    return { 0.0f, 0.0f, 0.0f };
}


__device__ inline HitInfo rayBoxesIntersection(const Ray& ray, const World& world) {
    HitInfo info;

    for (int i = 0; i < world.boxes.numBoxes; i++) {
        const Vec3& A = world.boxes.deviceBoxes[i].boxMin;
        const Vec3& B = world.boxes.deviceBoxes[i].boxMax;
        
        float tmin = -FLT_MAX;
        float tmax = FLT_MAX;

        // X axis
        if (ray.direction.x != 0.0f) {
            float invD = 1.0f / ray.direction.x;
            float t0 = (A.x - ray.origin.x) * invD;
            float t1 = (B.x - ray.origin.x) * invD;
            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.x < A.x || ray.origin.x > B.x) {
            return info;
        }

        // Y axis
        if (ray.direction.y != 0.0f) {
            float invD = 1.0f / ray.direction.y;
            float t0 = (A.y - ray.origin.y) * invD;
            float t1 = (B.y - ray.origin.y) * invD;
            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.y < A.y || ray.origin.y > B.y) {
            return info;
        }

        // Z axis
        if (ray.direction.z != 0.0f) {
            float invD = 1.0f / ray.direction.z;
            float t0 = (A.z - ray.origin.z) * invD;
            float t1 = (B.z - ray.origin.z) * invD;
            if (invD < 0.0f) {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.z < A.z || ray.origin.z > B.z) {
            return info;
        }

        if (tmax < tmin)
            continue;

        if (tmin < info.closest_t) {
            info.didHit = true;
            info.closest_t = tmin;
            info.hitLocation = ray.origin + (ray.direction * tmin);
            info.hitColor = world.boxes.deviceBoxes[i].color;
            info.hitNormal = getBoxNormal(info.hitLocation, A, B);
            info.hitRoughness = world.boxes.deviceBoxes[i].roughness;
            info.hitIsLightSource = world.boxes.deviceBoxes[i].isLightSource;
            info.hitLightIntensity = world.boxes.deviceBoxes[i].lightIntensity;
        }
    }

    return info;
}

__device__ inline HitInfo raySpheresIntersection(const Ray& ray, const World& world) {
    HitInfo info;

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

        if (t < info.closest_t) {
            info.closest_t = t;
            info.didHit = true;
            info.hitLocation = ray.origin + (ray.direction * t);
            info.hitColor = world.spheres.deviceSpheres[i].color;
            info.hitNormal = info.hitLocation - world.spheres.deviceSpheres[i].position; normalize(info.hitNormal);
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
        HitInfo info = rayBoxesIntersection(ray, world);
    
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

