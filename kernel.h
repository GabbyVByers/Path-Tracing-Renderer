#pragma once

#include "thread.h"
#include "structs.h"
#include "random.h"
#include <cfloat>

__device__ inline hit_info ray_spheres_intersection(const ray& ray, const sphere* dev_spheres, const int& num_spheres)
{
    hit_info info = { false };
    float closest_t = FLT_MAX;
    
    for (int i = 0; i < num_spheres; i++)
    {
        vec3 V = ray.origin - dev_spheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (dev_spheres[i].radius * dev_spheres[i].radius);

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
            info.hit_color = dev_spheres[i].color;
            info.hit_location = ray.origin + (ray.direction * t);
            info.hit_normal = info.hit_location - dev_spheres[i].position;
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

__device__ inline vec3 calculate_incoming_light(ray ray, const sphere* devSpheres, const int& numSpheres, unsigned int& randomState, const camera& camera)
{
    vec3 rayColor = { 1.0f, 1.0f, 1.0f };

    hit_info info = ray_spheres_intersection(ray, devSpheres, numSpheres);

    if (!info.did_hit)
    {
        return sky_color(ray.direction, camera.light_direction);
    }

    rayColor *= info.hit_color;


    ray.origin = info.hit_location + (info.hit_normal * 0.001f);
    ray.direction = camera.light_direction + (randomDirection(randomState) * 0.15f);

    hit_info shadowInfo = ray_spheres_intersection(ray, devSpheres, numSpheres);

    if (shadowInfo.did_hit)
    {
        return multiply(rayColor, { 0.3f, 0.3f, 0.3f });
    }

    return rayColor;
}

__global__ inline void main_kernel(uchar4* pixels, int width, int height, sphere* dev_spheres, int num_spheres, camera camera)
{
    thread thread = get_thread(width, height);
    if (thread.index == -1)
        return;

    unsigned int& random_state = camera.device_hash_array[thread.index];

    

    vec3 incoming_light = { 0.0f, 0.0f, 0.0f };
    ray ray = { camera.position, (camera.direction * camera.depth) + (camera.up * thread.v) + (camera.right * thread.u) };
    normalize(ray.direction);
    for (int i = 0; i < camera.rays_per_pixel; i++)
    {
        incoming_light += calculate_incoming_light(ray, dev_spheres, num_spheres, random_state, camera);
    }
    incoming_light /= camera.rays_per_pixel;

    if (camera.buffer_size > camera.buffer_limit)
    {
        return;
    }

    vec3 new_color = incoming_light;
    vec3 curr_color = camera.device_true_frame_buffer[thread.index];
    vec3 average_color = (curr_color * (float)camera.buffer_size + new_color) / ((float)camera.buffer_size + 1.0f);

    camera.device_true_frame_buffer[thread.index] = average_color;

    unsigned char r = average_color.x * 255.0f;
    unsigned char g = average_color.y * 255.0f;
    unsigned char b = average_color.z * 255.0f;
    pixels[thread.index] = make_uchar4(r, g, b, 255);

    return;
}

