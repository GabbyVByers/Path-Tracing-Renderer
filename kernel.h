#pragma once

#include "thread.h"
#include "world.h"
#include <cfloat>

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

__device__ inline Hit_info ray_spheres_intersection(const Ray& ray, const World& world)
{
    Hit_info info;
    float closest_t = FLT_MAX;

    for (int i = 0; i < world.spheres.num_spheres; i++)
    {
        Vec3 V = ray.origin - world.spheres.device_spheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (world.spheres.device_spheres[i].radius * world.spheres.device_spheres[i].radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        info.did_hit = true;

        if (t < closest_t)
        {
            closest_t = t;
            info.hit_color = world.spheres.device_spheres[i].color;
            info.hit_location = ray.origin + (ray.direction * t);
            info.hit_normal = info.hit_location - world.spheres.device_spheres[i].position;
            normalize(info.hit_normal);
            info.hit_roughness = world.spheres.device_spheres[i].roughness;
        }
    }

    return info;
}

__device__ inline Vec3 environment_light(const Ray& ray, const World& world)
{
    float zenith_horizon_gradient = pow(smoothstep(0.0f, 0.5f, ray.direction.y), world.sky.horizon_exponent);
    float sun_mask = pow(fmaxf(0.0f, dot(ray.direction, world.sky.sun_direction)), world.sky.sun_exponent);
    float ground_sky_gradient = smoothstep(-0.01f, 0.0f, ray.direction.y);
    Vec3 color_sky = lerp_vec3(world.sky.color_zenith, world.sky.color_horizon, zenith_horizon_gradient);
    color_sky = color_sky + ((world.sky.color_sun * sun_mask) * world.sky.sun_intensity);
    return lerp_vec3(world.sky.color_ground, color_sky, ground_sky_gradient);
}

__device__ inline Vec3 calculate_incoming_light(Ray ray, Thread& thread, const World& world)
{
    Vec3 color, light;
    color = 1.0f;
    light = 0.0f;

    Ray camera_ray = ray;

    for (int i = 0; i < 50; i++)
    {
        Hit_info info = ray_spheres_intersection(ray, world);

        if (!info.did_hit)
        {
            light = environment_light(ray, world);
            break;
        }

        color = color * info.hit_color;

        ray.origin = info.hit_location;

        Vec3 diffuse_direction = random_hemisphere_direction(info.hit_normal, *thread.hash_ptr) + random_direction(*thread.hash_ptr);
        Vec3 specular_direction = reflect(ray.direction, info.hit_normal);

        ray.direction = lerp_vec3(diffuse_direction, specular_direction, info.hit_roughness);
        normalize(ray.direction);
    }

    return color * light;
}

__device__ inline void frame_accumulation(Vec3 new_color, const Thread& thread, World& world)
{
    Vec3& accumulated_color = world.buffer.accumulated_frame_buffer[thread.index];
    
    if (world.buffer.num_accumulated_frames == 0)
        accumulated_color = 0.0f;

    accumulated_color += new_color;
    Vec3 this_color = accumulated_color / (world.buffer.num_accumulated_frames + 1);

    unsigned char r = fmin(255.0f, this_color.x * 255.0f);
    unsigned char g = fmin(255.0f, this_color.y * 255.0f);
    unsigned char b = fmin(255.0f, this_color.z * 255.0f);
    world.pixels[thread.index] = make_uchar4(r, g, b, 255);
}

__global__ inline void main_kernel(World world)
{
    Thread thread = get_thread(world.screen_width, world.screen_height, world.buffer.device_hash_array);
    if (thread.index == -1)
        return;

    Ray ray;
    ray.origin = world.camera.position;
    ray.direction = (world.camera.direction * world.camera.depth) + (world.camera.right * thread.u) + (world.camera.up * thread.v);
    ray.direction += random_direction(*thread.hash_ptr) * 0.001f;
    normalize(ray.direction);

    Vec3 color = calculate_incoming_light(ray, thread, world);
    frame_accumulation(color, thread, world);
    return;
}

