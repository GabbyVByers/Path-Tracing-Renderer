#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Thread {
    int index = -1;
    int screenWidth = 0;
    int screenHeight = 0;
    float u = 0.0f;
    float v = 0.0f;
    unsigned int* hashPtr = nullptr;
};

__device__ inline Thread getThread(const int& width, const int height, unsigned int* deviceHashArray) {
    Thread thread;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    thread.index = -1;
    if ((x < width) && (y < height))
    {
        thread.index = y * width + x;
    }
    thread.screenWidth = width;
    thread.screenHeight = height;
    thread.u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    thread.v = (y / (float)height) * 2.0f - 1.0f;
    thread.hashPtr = &deviceHashArray[thread.index];
    return thread;
}

