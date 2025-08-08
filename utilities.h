#pragma once

#include "vec3.h"
#include <cmath>

inline float randomFloat(float min, float max) {
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline Vec3 randomVec3(float min, float max) {
    return Vec3 {
        randomFloat(min, max),
        randomFloat(min, max),
        randomFloat(min, max)
    };
}

__host__ __device__ inline void hash_uint32(unsigned int& state) {
    state ^= state >> 17;
    state *= 0xed5ad4bb;
    state ^= state >> 11;
    state *= 0xac4c1b51;
    state ^= state >> 15;
    state *= 0x31848bab;
    state ^= state >> 14;
}

__host__ __device__ inline float randomFloatFromInt(unsigned int& state) {
    hash_uint32(state);
    return state / (float)UINT_MAX;
}

__host__ __device__ inline float randomFloatNormalDistribution(unsigned int& state) {
    float theta = 2.0f * 3.1415927f * randomFloatFromInt(state);
    return sqrt(-2.0f * log(randomFloatFromInt(state))) * cos(theta);
}

__host__ __device__ inline Vec3 randomHemisphereDirection(const Vec3& normal, unsigned int& state) {
    float rx = randomFloatNormalDistribution(state);
    float ry = randomFloatNormalDistribution(state);
    float rz = randomFloatNormalDistribution(state);

    Vec3 direction = { rx, ry, rz };
    normalize(direction);

    if (dot(direction, normal) < 0.0f)
        direction *= -1.0f;

    return direction;
}

__host__ __device__ inline Vec3 randomDirection(unsigned int& state) {
    float rx = randomFloatNormalDistribution(state);
    float ry = randomFloatNormalDistribution(state);
    float rz = randomFloatNormalDistribution(state);

    Vec3 direction = { rx, ry, rz };

    return direction;
}

__device__ inline float smoothstep(float a, float b, float x) {
    x = fmaxf(a, fminf(x, b));
    float t = (x - a) / (b - a);
    return t * t * (3.0f - 2.0f * t);
}

