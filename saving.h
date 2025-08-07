#pragma once

#include "world.h"
#include <fstream>
#include <string>

inline void save_spheres(World& world, char file_name[24])
{
	size_t size_bytes = sizeof(Sphere) * world.spheres.num_spheres;
	unsigned char* save_data = new unsigned char[size_bytes];
	memcpy(save_data, world.spheres.host_spheres, size_bytes);

	std::string real_file_name(&file_name[0], sizeof(file_name));
	real_file_name = "saves/" + real_file_name + ".bin";

	std::ofstream out_file(real_file_name, std::ios::binary);
	if (out_file.is_open())
	{
		out_file.write(reinterpret_cast<const char*>(save_data), size_bytes);
		out_file.close();
	}

	delete[] save_data;
}

inline void load_spheres(World& world, char file_name[24])
{
	std::string real_file_name(file_name, sizeof(file_name));
	real_file_name = "saves/" + real_file_name + ".bin";
	std::ifstream in_file(real_file_name, std::ios::binary);

	if (!in_file.is_open())
		return;

	in_file.seekg(0, std::ios::end);
	size_t size_bytes = in_file.tellg();
	in_file.seekg(0, std::ios::beg);

	char* save_data = new char[size_bytes];
	in_file.read(save_data, size_bytes);

	int num_spheres = size_bytes / sizeof(Sphere);

	free_spheres(world.spheres);

	world.spheres.num_spheres = num_spheres;
	world.spheres.host_spheres = new Sphere[num_spheres];
	cudaMalloc((void**)&world.spheres.device_spheres, sizeof(Sphere) * num_spheres);

	memcpy(world.spheres.host_spheres, save_data, size_bytes);
	update_spheres_on_gpu(world.spheres);
}

