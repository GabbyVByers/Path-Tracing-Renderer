#pragma once

#include "vec3.h"

struct sphere
{
    vec3 position;
    float radius;
    vec3 color;
};

struct hit_info
{
    bool did_hit;
    bool did_hit_light_source;
    vec3 hit_location;
    vec3 hit_color;
    vec3 hit_normal;
};

struct ray
{
    vec3 origin;
    vec3 direction;
};

struct camera
{
    vec3 position;
    vec3 direction;
    vec3 up;
    vec3 right;
    float depth;
};

inline void fix_camera(camera& camera)
{
    const vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = camera.direction * up;
    normalize(camera.right);
    camera.up = camera.right * camera.direction;
    normalize(camera.up);
}

struct world
{
    vec3 light_direction;
    int rays_per_pixel;
    int buffer_size;
    int buffer_limit;
    unsigned int* device_hash_array;
    vec3* device_true_frame_buffer;
    sphere* device_spheres;
    int num_spheres;
};

