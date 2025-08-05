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
    float depth = 1.5f;
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

struct Sky_parameters
{
    Vec3 sun_direction = { 1.0f, 1.0f, 1.0f };
    float sun_intensity = 35.0f;
    float sun_exponent = 75.0f;
    float horizon_exponent = 0.35f;

    Vec3 color_sun     = rgb(255, 255, 255);
    Vec3 color_zenith  = rgb(255, 255, 255);
    Vec3 color_horizon = rgb( 63, 195, 235);
    Vec3 color_ground  = rgb( 79, 112,  76);
};

struct World
{
    uchar4* pixels = nullptr;
    int width = 0;
    int height = 0;

    Sky_parameters sky;

    int max_bounce_limit = 50;
    int num_accumulated_frames = 0;
    unsigned int* device_hash_array = nullptr;
    Vec3* accumulated_frame_buffer;
    Sphere* device_spheres = nullptr;
    int num_spheres = 0;
};

