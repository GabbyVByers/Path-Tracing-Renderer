#pragma once

#include "world.h"
#include <fstream>

struct Meta_data
{
	int num_spheres;
};

inline void save_spheres(World& world)
{
	unsigned char* save_data = nullptr;

	size_t size_bytes = sizeof(Meta_data) + (sizeof(Sphere) * world.spheres.num_spheres);

	save_data = new unsigned char[size_bytes];

	memcpy(save_data, world.spheres.host_spheres, size_bytes);

	std::ofstream out_file("save_file.bin", std::ios::binary);

	if (out_file.is_open())
	{
		out_file.write(reinterpret_cast<const char*>(save_data), size_bytes);
		out_file.close();
	}

	delete[] save_data;
}

inline void load_spheres()
{

}