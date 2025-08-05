#pragma once

#include "thread.h"
#include "structs.h"
#include "random.h"
#include <cfloat>

__device__ inline Hit_info ray_spheres_intersection(const Ray& ray, const World& world)
{
    Hit_info info;
    float closest_t = FLT_MAX;
    
    for (int i = 0; i < world.num_spheres; i++)
    {
        Vec3 V = ray.origin - world.device_spheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (world.device_spheres[i].radius * world.device_spheres[i].radius);

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
            info.hit_color = world.device_spheres[i].color;
            info.hit_location = ray.origin + (ray.direction * t);
            info.hit_normal = info.hit_location - world.device_spheres[i].position;
            normalize(info.hit_normal);
            info.hit_roughness = world.device_spheres[i].roughness;
        }
    }

    return info;
}

__device__ inline Vec3 environment_light(const Ray& ray, const World& world)
{
    float zenith_horizon_gradient = pow(smoothstep(0.0f, 0.5f, ray.direction.y), world.sky.horizon_exponent);
    Vec3 color_sky = lerp_between_vectors(world.sky.color_zenith, world.sky.color_horizon, zenith_horizon_gradient);
    float sun_mask = pow(fmaxf(0, dot(ray.direction, world.sky.sun_direction)), world.sky.sun_exponent);
    color_sky = color_sky + ((world.sky.color_sun * sun_mask) * world.sky.sun_intensity);
    
    float ground_sky_gradient = smoothstep(-0.01f, 0.0f, ray.direction.y);
    Vec3 composite = lerp_between_vectors(world.sky.color_ground, color_sky, ground_sky_gradient);

    return composite;
}

__device__ inline Vec3 calculate_incoming_light(Ray ray, Thread& thread, const World& world)
{
    Vec3 color, light;
    color = 1.0f;
    light = 0.0f;

    Ray camera_ray = ray;

    for (int i = 0; i < world.max_bounce_limit; i++)
    {
        Hit_info info = ray_spheres_intersection(ray, world);

        if (!info.did_hit)
        {
            light = environment_light(ray, world);
            break;
        }

        color = color * info.hit_color;

        ray.origin = info.hit_location;
        ray.direction = random_hemisphere_direction(info.hit_normal, *thread.hash_ptr);
        normalize(ray.direction);
    }

    return color * light;
}

__device__ inline void frame_accumulation(Vec3 new_color, const Thread& thread, World& world)
{
    Vec3& accumulated_color = world.accumulated_frame_buffer[thread.index];
    
    if (world.num_accumulated_frames == 0)
        accumulated_color = 0.0f;

    accumulated_color += new_color;
    Vec3 this_color = accumulated_color / (world.num_accumulated_frames + 1);

    unsigned char r = fmin(255.0f, this_color.x * 255.0f);
    unsigned char g = fmin(255.0f, this_color.y * 255.0f);
    unsigned char b = fmin(255.0f, this_color.z * 255.0f);
    world.pixels[thread.index] = make_uchar4(r, g, b, 255);
}

__global__ inline void main_kernel(World world, Camera camera)
{
    Thread thread = get_thread(world.width, world.height, world.device_hash_array);
    if (thread.index == -1)
        return;

    Ray ray;
    ray.origin = camera.position;
    ray.direction = (camera.direction * camera.depth) + (camera.right * thread.u) + (camera.up * thread.v);
    ray.direction += random_direction(*thread.hash_ptr) * 0.001f;
    normalize(ray.direction);

    Vec3 color = calculate_incoming_light(ray, thread, world);
    frame_accumulation(color, thread, world);
    return;
}

