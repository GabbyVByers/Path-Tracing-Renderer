#pragma once

#include "vec3.h"
using uint32 = unsigned int;

// Used on the CPU

inline float randomFloat(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline vec3 randomVec3(float min, float max)
{
    return vec3
    {
        randomFloat(min, max),
        randomFloat(min, max),
        randomFloat(min, max)
    };
}

// Used on the GPU

__host__ __device__ inline void hash_uint32(uint32& state)
{
    state ^= state >> 17;
    state *= 0xed5ad4bb;
    state ^= state >> 11;
    state *= 0xac4c1b51;
    state ^= state >> 15;
    state *= 0x31848bab;
    state ^= state >> 14;
}

__host__ __device__ inline float randomFloatFromInt(uint32& state)
{
    hash_uint32(state);
    return state / (float)UINT_MAX;
}

__host__ __device__ inline float randomFloatNormalDistribution(uint32& state)
{
    float theta = 2.0f * 3.1415927f * randomFloatFromInt(state);
    return sqrt(-2.0f * log(randomFloatFromInt(state))) * cos(theta);
}

__host__ __device__ inline vec3 randomHemisphereDirection(const vec3& normal, uint32& state)
{
    float rx = randomFloatNormalDistribution(state);
    float ry = randomFloatNormalDistribution(state);
    float rz = randomFloatNormalDistribution(state);

    vec3 direction = { rx, ry, rz };
    normalize(direction);

    if (dot(direction, normal) < 0.0f)
    {
        direction *= -1.0f;
    }

    return direction;
}

__host__ __device__ inline vec3 randomDirection(uint32& state)
{
    float rx = randomFloatNormalDistribution(state);
    float ry = randomFloatNormalDistribution(state);
    float rz = randomFloatNormalDistribution(state);

    vec3 direction = { rx, ry, rz };
    //normalize(direction);

    return direction;
}
