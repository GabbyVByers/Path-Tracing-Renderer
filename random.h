#pragma once

#include "vec3.h"

inline float random_float(float min, float max)
{
    return ((rand() / (float)RAND_MAX) * (max - min)) + min;
}

inline vec3 random_vec3(float min, float max)
{
    return vec3
    {
        random_float(min, max),
        random_float(min, max),
        random_float(min, max)
    };
}

__host__ __device__ inline void hash_uint32(unsigned int& state)
{
    state ^= state >> 17;
    state *= 0xed5ad4bb;
    state ^= state >> 11;
    state *= 0xac4c1b51;
    state ^= state >> 15;
    state *= 0x31848bab;
    state ^= state >> 14;
}

__host__ __device__ inline float random_float_from_int(unsigned int& state)
{
    hash_uint32(state);
    return state / (float)UINT_MAX;
}

__host__ __device__ inline float random_float_normal_distribution(unsigned int& state)
{
    float theta = 2.0f * 3.1415927f * random_float_from_int(state);
    return sqrt(-2.0f * log(random_float_from_int(state))) * cos(theta);
}

__host__ __device__ inline vec3 random_hemisphere_direction(const vec3& normal, unsigned int& state)
{
    float rx = random_float_normal_distribution(state);
    float ry = random_float_normal_distribution(state);
    float rz = random_float_normal_distribution(state);

    vec3 direction = { rx, ry, rz };
    normalize(direction);

    if (dot(direction, normal) < 0.0f)
        direction *= -1.0f;

    return direction;
}

__host__ __device__ inline vec3 randomDirection(unsigned int& state)
{
    float rx = random_float_normal_distribution(state);
    float ry = random_float_normal_distribution(state);
    float rz = random_float_normal_distribution(state);

    vec3 direction = { rx, ry, rz };

    return direction;
}

