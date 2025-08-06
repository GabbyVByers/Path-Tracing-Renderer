#pragma once

#include "opengl.h"
#include <iostream>

inline int mouse_spheres_intersection(const Ray& ray, const World& world)
{
    Hit_info info;
    float closest_t = FLT_MAX;
    int index = -1;

    for (int i = 0; i < world.spheres.num_spheres; i++)
    {
        Vec3 V = ray.origin - world.spheres.host_spheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (world.spheres.host_spheres[i].radius * world.spheres.host_spheres[i].radius);

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
            index = i;
        }
    }

    return index;
}

inline void select_sphere(Opengl& opengl, World& world)
{
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS)
        return;

	for (int i = 0; i < world.spheres.num_spheres; i++)
		world.spheres.host_spheres[i].is_selected = false;
	
	int screen_width = opengl.screen_width;
	int screen_height = opengl.screen_height;
	
	double mouse_x;
	double mouse_y;
	glfwGetCursorPos(opengl.window, &mouse_x, &mouse_y);

    float v = (((screen_height - mouse_y) / screen_height) * 2.0f) - 1.0f;
    float u = (((mouse_x / screen_width) * 2.0f) - 1.0f) * (screen_width / (float)screen_height);

	Ray mouse_ray;
	mouse_ray.origin = world.camera.position;
	mouse_ray.direction = (world.camera.direction * world.camera.depth) + (world.camera.right * u) + (world.camera.up * v);
	
	int mouse_sphere_index = mouse_spheres_intersection(mouse_ray, world);
	
	if (mouse_sphere_index != -1)
		world.spheres.host_spheres[mouse_sphere_index].is_selected = true;
	
	update_spheres_on_gpu(world.spheres);
}

