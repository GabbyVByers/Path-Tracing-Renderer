#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct thread // TODO: give every struct default values, remove unnessasary struct initializations
{
    int index = -1;
    int x = -1;
    int y = -1;
    float u = 0.0f;
    float v = 0.0f;
};

__device__ inline thread get_thread(const int& width, const int& height)
{
    thread thread;
    thread.x = blockIdx.x * blockDim.x + threadIdx.x;
    thread.y = blockIdx.y * blockDim.y + threadIdx.y;
    thread.index = -1;
    if ((thread.x < width) && (thread.y < height))
    {
        thread.index = thread.y * width + thread.x;
    }
    thread.u = ((thread.x / (float)width) * 2.0f - 1.0f) * (width / (float)height);
    thread.v = (thread.y / (float)height) * 2.0f - 1.0f;
    return thread;
}

