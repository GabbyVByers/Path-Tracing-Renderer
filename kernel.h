#pragma once

#include "thread.h"
#include "structs.h"
#include "random.h"
#include <cfloat>

__device__ inline hit_info ray_spheres_intersection(const ray& ray, const world& world)
{
    hit_info info = { false };
    float closest_t = FLT_MAX;
    
    for (int i = 0; i < world.num_spheres; i++)
    {
        vec3 V = ray.origin - world.device_spheres[i].position;
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
        }
    }

    return info;
}

__device__ inline vec3 sky_color(const vec3& direction, const vec3& light_direction)
{
    vec3 skyWhite    = rgb(255, 255, 255);
    vec3 skyBlue     = rgb(57, 162, 237);
    vec3 groundColor = rgb(143, 136, 130);

    if (direction.y < 0.0f)
    {
        return groundColor;
    }

    if (dot(light_direction, direction) > 0.997f)
    {
        return skyWhite;
    }
    
    return skyBlue;
}

__device__ inline vec3 calculate_incoming_light(ray ray, unsigned int& randomState, const world& world, const camera& camera)
{
    vec3 rayColor = { 1.0f, 1.0f, 1.0f };

    hit_info info = ray_spheres_intersection(ray, world);

    if (!info.did_hit)
    {
        return sky_color(ray.direction, world.light_direction);
    }

    rayColor *= info.hit_color;


    ray.origin = info.hit_location + (info.hit_normal * 0.001f);
    ray.direction = world.light_direction + (randomDirection(randomState) * 0.15f);

    hit_info shadowInfo = ray_spheres_intersection(ray, world);

    if (shadowInfo.did_hit)
    {
        return multiply(rayColor, { 0.3f, 0.3f, 0.3f });
    }

    return rayColor;
}

__global__ inline void main_kernel(uchar4* pixels, int width, int height, world world, camera camera)
{
    thread thread = get_thread(width, height);
    if (thread.index == -1)
        return;

    unsigned int& random_state = world.device_hash_array[thread.index];

    

    vec3 incoming_light = { 0.0f, 0.0f, 0.0f };
    ray ray = { camera.position, (camera.direction * camera.depth) + (camera.up * thread.v) + (camera.right * thread.u) };
    normalize(ray.direction);
    for (int i = 0; i < world.rays_per_pixel; i++)
    {
        incoming_light += calculate_incoming_light(ray, random_state, world, camera);
    }
    incoming_light /= world.rays_per_pixel;

    if (world.buffer_size > world.buffer_limit)
    {
        return;
    }

    vec3 new_color = incoming_light;
    vec3 curr_color = world.device_true_frame_buffer[thread.index];
    vec3 average_color = (curr_color * (float)world.buffer_size + new_color) / ((float)world.buffer_size + 1.0f);

    world.device_true_frame_buffer[thread.index] = average_color;

    unsigned char r = average_color.x * 255.0f;
    unsigned char g = average_color.y * 255.0f;
    unsigned char b = average_color.z * 255.0f;
    pixels[thread.index] = make_uchar4(r, g, b, 255);

    return;
}

