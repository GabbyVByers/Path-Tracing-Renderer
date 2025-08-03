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
        {
            continue;
        }

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
        {
            continue;
        }

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

__device__ inline Vec3 hit_color_considering_shadow(const Hit_info& info, Thread& thread, const World& world)
{
    Ray ray;
    ray.origin = info.hit_location + (info.hit_normal * 0.001f);
    ray.direction = world.light_direction + (random_direction(*thread.hash_ptr) * world.random_offset_magnitude);
    normalize(ray.direction);

    Vec3 color = info.hit_color;

    Vec3 noisy_normal = info.hit_normal + (random_direction(*thread.hash_ptr) * world.random_offset_magnitude);
    normalize(noisy_normal);

    float cos_weight = dot(noisy_normal, world.light_direction);
    Hit_info shadow_info = ray_spheres_intersection(ray, world);

    if (cos_weight <= 0.0f)
    {
        cos_weight = 0.0f;
    }

    if (shadow_info.did_hit)
    {
        cos_weight = 0.0f;
    }

    float real_dimming_factor = ((1.0f - world.shadow_dimming_factor) * cos_weight) + world.shadow_dimming_factor;
    return color * real_dimming_factor;
}

__device__ inline Vec3 skybox(const Ray& ray, const World& world)
{
    Vec3 skyWhite    = rgb(255, 255, 255);
    Vec3 skyBlue     = rgb(57, 162, 237);
    Vec3 groundColor = rgb(143, 136, 130);

    if (ray.direction.y < 0.0f)
    {
        return groundColor;
    }

    Vec3 dir = ray.direction;
    normalize(dir);
    if (dot(world.light_direction, dir) > 0.997f)
    {
        return skyWhite;
    }
    
    return skyBlue;
}

__device__ inline Vec3 calculate_incoming_light(Ray camera_ray, Thread& thread, const World& world)
{
    Hit_info initial_hit_info = ray_spheres_intersection(camera_ray, world);

    if (!initial_hit_info.did_hit)
    {
        return skybox(camera_ray, world);
    }

    Vec3 color = hit_color_considering_shadow(initial_hit_info, thread, world);

    Ray ray;
    ray.origin = initial_hit_info.hit_location;
    Vec3 diffuse_direction = random_hemisphere_direction(initial_hit_info.hit_normal, *thread.hash_ptr);
    Vec3 specular_direction = reflect(ray.direction, initial_hit_info.hit_normal);
    ray.direction = lerp_between_vectors(diffuse_direction, specular_direction, initial_hit_info.hit_roughness);
    normalize(ray.direction);

    for (int i = 0; i < world.max_bounce_limit; i++)
    {
        Hit_info info = ray_spheres_intersection(ray, world);

        if (!info.did_hit)
        {
            break;
        }

        Vec3 next_color = hit_color_considering_shadow(info, thread, world);

        color = (color * (float)(i + 1)) + next_color;
        color /= (float)(i + 2);

        ray.origin = info.hit_location;
        ray.direction = random_hemisphere_direction(info.hit_normal, *thread.hash_ptr);
    }

    return color;
}

__device__ inline void frame_accumulation(Vec3 new_color, const Thread& thread, World& world)
{
    if (world.buffer_size >= world.buffer_limit)
    {
        return;
    }

    Vec3 curr_color = world.accumulated_frame_buffer[thread.index];
    Vec3 average_color = (curr_color * (float)world.buffer_size + new_color) / ((float)world.buffer_size + 1.0f);

    world.accumulated_frame_buffer[thread.index] = average_color;

    unsigned char r = average_color.x * 255.0f;
    unsigned char g = average_color.y * 255.0f;
    unsigned char b = average_color.z * 255.0f;
    world.pixels[thread.index] = make_uchar4(r, g, b, 255);
}

__global__ inline void main_kernel(World world, Camera camera)
{
    Thread thread = get_thread(world.width, world.height, world.device_hash_array);
    if (thread.index == -1)
        return;

    Ray ray;
    ray.origin = camera.position;
    ray.direction = camera.direction + (camera.right * thread.u) + (camera.up * thread.v);

    Vec3 color = calculate_incoming_light(ray, thread, world);
    frame_accumulation(color, thread, world);
    return;
}

