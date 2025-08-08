#pragma once

#include "world.h"
#include <fstream>
#include <string>

inline void saveSpheres(World& world, char fileName[24])
{
	size_t sizeBytes = sizeof(Sphere) * world.spheres.numSpheres;
	unsigned char* saveData = new unsigned char[sizeBytes];
	memcpy(saveData, world.spheres.hostSpheres, sizeBytes);

	std::string realFileName(&fileName[0], sizeof(fileName));
	realFileName = "saves/" + realFileName + ".bin";

	std::ofstream outFile(realFileName, std::ios::binary);
	if (outFile.is_open()) {
		outFile.write(reinterpret_cast<const char*>(saveData), sizeBytes);
		outFile.close();
	}

	delete[] saveData;
}

inline void loadSpheres(World& world, char fileName[24])
{
	std::string realFileName(fileName, sizeof(fileName));
	realFileName = "saves/" + realFileName + ".bin";
	std::ifstream inFile(realFileName, std::ios::binary);

	if (!inFile.is_open())
		return;

	inFile.seekg(0, std::ios::end);
	size_t sizeBytes = inFile.tellg();
	inFile.seekg(0, std::ios::beg);

	char* saveData = new char[sizeBytes];
	inFile.read(saveData, sizeBytes);

	int numSpheres = sizeBytes / sizeof(Sphere);

	freeSpheres(world.spheres);

	world.spheres.numSpheres = numSpheres;
	world.spheres.hostSpheres = new Sphere[numSpheres];
	cudaMalloc((void**)&world.spheres.deviceSpheres, sizeof(Sphere) * numSpheres);

	memcpy(world.spheres.hostSpheres, saveData, sizeBytes);
	updateSpheresOnGpu(world.spheres);
}

