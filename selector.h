#pragma once

#include "opengl.h"

inline void select_sphere(const Opengl& opengl, World& world)
{
	int screen_width = opengl.screen_width;
	int screen_height = opengl.screen_height;

	int mouse_x;
	int mouse_y;

	Ray mouse_ray;
	mouse_ray.origin = world.camera.position;
	mouse_ray.direction = (world.camera.direction * world.camera.depth);
}

