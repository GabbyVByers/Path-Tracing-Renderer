#pragma once

#include "vec3.h"

struct camera
{
    vec3 position;
    vec3 direction;
    vec3 up;
    vec3 right;
    vec3 light_direction;

    float depth;
    int rays_per_pixel;
    int buffer_size;
    int buffer_limit;
    unsigned int* device_hash_array;
    vec3* device_true_frame_buffer;
};

inline void fix_camera(camera& cam)
{
    const vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(cam.direction);
    cam.right = cam.direction * up;
    normalize(cam.right);
    cam.up = cam.right * cam.direction;
    normalize(cam.up);
}

