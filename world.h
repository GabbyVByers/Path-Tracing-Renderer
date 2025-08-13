#pragma once

#include "vec3.h"
#include "utilities.h"
#include "sharedarray.h"

struct Sphere
{
    Vec3 position = { -5.0f, 0.0f, 0.0f };
    float radius = 2.0f;
    Vec3 color = rgb(100, 255, 150);
    float roughness = 0.0f;
    bool isSelected = false;
    bool isLightSource = false;
    float lightIntensity = 1.0f;
};

struct Box
{
    Vec3 boxMin = { -2.0f, -2.0f, -2.0f };
    Vec3 boxMax = { 2.0f, 2.0f, 2.0f };
    Vec3 color = rgb(255, 100, 255);
    float roughness = 0.0f;
    bool isSelected = false;
    bool isLightSource = false;
    float lightIntensity = 1.0f;
};

struct Camera
{
    Vec3 position = { -8.55f, 3.9f, -8.17f };
    Vec3 direction = { 0.678f, -0.204f, 0.697f };
    Vec3 up;
    Vec3 right;
    float depth = 1.5f;
};

struct Sky
{
    bool toggleSky = true;
    Vec3 sunDirection = returnNormalized({ 1.0f, 1.0f, 1.0f });
    float sunIntensity = 20.0f;
    float sunExponent = 50.0f;
    float horizonExponent = 0.35f;

    Vec3 colorSun     = rgb(255, 255, 255);
    Vec3 colorZenith  = rgb(255, 255, 255);
    Vec3 colorHorizon = rgb( 63, 195, 235);
    Vec3 colorGround  = rgb( 79, 112,  76);
};

struct GlobalUtilities
{
    uchar4* pixels = nullptr;
    int screenWidth = 0;
    int screenHeight = 0;
    int numAccumulatedFrames = 0;
    Vec3* accumulatedFrameBuffer;
    unsigned int* deviceHashArray = nullptr;
    float antiAliasing = 0.001f;
    int maxBounceLimit = 50;
};

struct World
{   
    SharedArray<Sphere> spheres;
    SharedArray<Box> boxes;
    Camera camera;
    Sky sky;
    GlobalUtilities global;
};

inline void fixCamera(Camera& camera)
{
    const Vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = cross(camera.direction, up);
    normalize(camera.right);
    camera.up = cross(camera.right, camera.direction);
    normalize(camera.up);
}

inline void buildHashArrayAndFrameBuffer(GlobalUtilities& global, int screenSize)
{
    unsigned int* hostHashArray = nullptr;
    hostHashArray = new unsigned int[screenSize];
    for (int i = 0; i < screenSize; i++)
    {
        unsigned int hash = i;
        hash_uint32(hash);
        hostHashArray[i] = hash;
    }
    cudaMalloc((void**)&global.deviceHashArray, sizeof(unsigned int) * screenSize);
    cudaMemcpy(global.deviceHashArray, hostHashArray, sizeof(unsigned int) * screenSize, cudaMemcpyHostToDevice);
    delete[] hostHashArray;
    cudaMalloc((void**)&global.accumulatedFrameBuffer, sizeof(Vec3) * screenSize);
}

