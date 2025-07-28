#pragma once

#include "vec3.h"

struct Camera
{
    vec3 position;
    vec3 direction;
    vec3 up;
    vec3 right;
    float depth;

    int raysPerPixel;
    vec3 lightDirection;
};

inline void fixCamera(Camera& camera)
{
    const vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = camera.direction * up;
    normalize(camera.right);
    camera.up = camera.right * camera.direction;
    normalize(camera.up);
}

