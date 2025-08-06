#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Thread
{
    int index = -1;
    int screen_width = 0;
    int screen_height = 0;
    float u = 0.0f;
    float v = 0.0f;
    unsigned int* hash_ptr = nullptr;
};

__device__ inline Thread get_thread(const int& width, const int height, unsigned int* device_hash_array)
{
    Thread thread;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    thread.index = -1;
    if ((x < width) && (y < height))
    {
        thread.index = y * width + x;
    }
    thread.screen_width = width;
    thread.screen_height = height;
    thread.u = ((x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    thread.v = (y / (float)height) * 2.0f - 1.0f;
    thread.hash_ptr = &device_hash_array[thread.index];
    return thread;
}

