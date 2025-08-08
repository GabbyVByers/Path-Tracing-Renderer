#pragma once

#include "spheres.h"
#include "utilities.h"

struct Camera {
    Vec3 position = { 0.0f, 0.0f, -5.0f };
    Vec3 direction = { 0.0f, 0.0f, 1.0f };
    Vec3 up;
    Vec3 right;
    float depth = 1.5f;
};

struct Sky {
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

struct Buffer {
    int numAccumulatedFrames = 0;
    Vec3* accumulatedFrameBuffer;
    unsigned int* deviceHashArray = nullptr;
};

struct World {   
    uchar4* pixels = nullptr;
    int screenWidth = 0;
    int screenHeight = 0;

    Camera camera;
    Sky sky;
    Spheres spheres;
    Buffer buffer;
};

inline void fixCamera(Camera& camera) {
    const Vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = cross(camera.direction, up);
    normalize(camera.right);
    camera.up = cross(camera.right, camera.direction);
    normalize(camera.up);
}

inline void buildHashArrayAndFrameBuffer(Buffer& buffer, int screenSize) {
    unsigned int* hostHashArray = nullptr;
    hostHashArray = new unsigned int[screenSize];
    for (int i = 0; i < screenSize; i++) {
        unsigned int hash = i;
        hash_uint32(hash);
        hostHashArray[i] = hash;
    }
    cudaMalloc((void**)&buffer.deviceHashArray, sizeof(unsigned int) * screenSize);
    cudaMemcpy(buffer.deviceHashArray, hostHashArray, sizeof(unsigned int) * screenSize, cudaMemcpyHostToDevice);
    delete[] hostHashArray;
    cudaMalloc((void**)&buffer.accumulatedFrameBuffer, sizeof(Vec3) * screenSize);
}

