#pragma once

#include "vec3.h"

struct Sphere
{
    Vec3 position;
    float radius = 0.0f;
    Vec3 color;
    float roughness = 1.0f;
};

struct Hit_info
{
    bool did_hit = false;
    Vec3 hit_location;
    Vec3 hit_color;
    Vec3 hit_normal;
    float hit_roughness;
};

struct Ray
{
    Vec3 origin;
    Vec3 direction;
};

struct Camera
{
    Vec3 position = { 11.3f, 8.0f, -10.0f };
    Vec3 direction = { -0.5f, -0.5f, 0.7f };
    Vec3 up;
    Vec3 right;
    float depth = 2.0f;
};

inline void fix_camera(Camera& camera)
{
    const Vec3 up = { 0.0f, 1.0f, 0.0f };
    normalize(camera.direction);
    camera.right = cross(camera.direction, up);
    normalize(camera.right);
    camera.up = cross(camera.right, camera.direction);
    normalize(camera.up);
}

struct World
{
    uchar4* pixels = nullptr;
    int width = 0;
    int height = 0;
    Vec3 light_direction = { 1.0f, 1.0f, 1.0f };
    float sun_intensity = 10.0f;
    float ambient_brightness = 0.33f;
    int max_bounce_limit = 10;
    int buffer_size = 0;
    unsigned int* device_hash_array = nullptr;
    Vec3* accumulated_frame_buffer;
    Sphere* device_spheres = nullptr;
    int num_spheres = 0;
};

