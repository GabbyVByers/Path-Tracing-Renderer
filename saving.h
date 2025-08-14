#pragma once

#include "world.h"
#include <fstream>
#include <string>

struct MetaData
{
	int sizeBytes = -1;
	int numSpheres = -1;
	int numBoxes = -1;

	Camera camera;
};

inline void saveSpheres(World& world, char fileName[24])
{
	int numSpheres = world.spheres.size;
	int numBoxes = world.boxes.size;
	int sizeBytes = sizeof(MetaData) + (numSpheres * sizeof(Sphere)) + (numBoxes * sizeof(Box));

	MetaData meta;
	meta.sizeBytes = sizeBytes;
	meta.numSpheres = numSpheres;
	meta.numBoxes = numBoxes;
	meta.camera = world.camera;

	char* buffer = new char[sizeBytes];
	Sphere* spheres = world.spheres.hostPointer;
	Box* boxes = world.boxes.hostPointer;

	memcpy(buffer, &meta, sizeof(MetaData));
	memcpy(buffer + sizeof(MetaData), spheres, numSpheres * sizeof(Sphere));
	memcpy(buffer + sizeof(MetaData) + (numSpheres * sizeof(Sphere)), boxes, numBoxes * sizeof(Box));

	std::string realFileName(&fileName[0], sizeof(fileName));
	realFileName = "saves/" + realFileName + ".bin";

	std::ofstream outFile(realFileName, std::ios::binary);
	if (outFile.is_open())
	{
		outFile.write(reinterpret_cast<const char*>(buffer), sizeBytes);
		outFile.close();
	}

	delete[] buffer;
}

inline void loadSpheres(World& world, char fileName[24])
{
	std::string realFileName(fileName, sizeof(fileName));
	realFileName = "saves/" + realFileName + ".bin";

	std::ifstream inFile(realFileName, std::ios::binary);
	if (!inFile.is_open())
		return;

	MetaData meta;
	inFile.read((char*)&meta, sizeof(MetaData));
	world.camera = meta.camera;

	char* buffer = new char[meta.sizeBytes];
	inFile.seekg(0, std::ios::beg);
	inFile.read(buffer, meta.sizeBytes);

	world.spheres.clear();
	world.boxes.clear();

	for (int i = 0; i < meta.numSpheres; i++)
	{
		Sphere sphere = ((Sphere*)(buffer + sizeof(MetaData)))[i];
		world.spheres.add(sphere);
	}

	for (int i = 0; i < meta.numBoxes; i++)
	{
		Box box = ((Box*)(buffer + sizeof(MetaData) + (meta.numSpheres * sizeof(Sphere))))[i];
		world.boxes.add(box);
	}

	world.spheres.updateHostToDevice();
	world.boxes.updateHostToDevice();
}

