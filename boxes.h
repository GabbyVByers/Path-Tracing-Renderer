#pragma once

#include "vec3.h"

struct Box
{
    Vec3 boxMin = { -2.0f, -2.0f, -2.0f };
	Vec3 boxMax = { 2.0f, -1.9f, 2.0f };
    Vec3 color = rgb(255, 33, 255);
    float roughness = 0.8f;
    bool isSelected = false;
    bool isLightSource = false;
    float lightIntensity = 1.0f;
};

struct Boxes
{
    int numBoxes;
    Box* hostBoxes = nullptr;
    Box* deviceBoxes = nullptr;
};

inline void updateBoxesOnGpu(Boxes& boxes)
{
    cudaMemcpy(boxes.deviceBoxes, boxes.hostBoxes, sizeof(Box) * boxes.numBoxes, cudaMemcpyHostToDevice);
}

inline void freeBoxes(Boxes& boxes)
{
    delete[] boxes.hostBoxes;
    cudaFree(boxes.deviceBoxes);
}

inline void initializeBoxes(Boxes& boxes)
{
    boxes.numBoxes = 1;
    boxes.hostBoxes = nullptr;
    boxes.hostBoxes = new Box[boxes.numBoxes];
    cudaMalloc((void**)&boxes.deviceBoxes, sizeof(Box) * boxes.numBoxes);
    updateBoxesOnGpu(boxes);
}

