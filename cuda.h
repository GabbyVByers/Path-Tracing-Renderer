#pragma once

#include "opengl.h"
#include "structs.h"
#include "kernel.h"

inline void launch_cuda_kernel(opengl& opengl, world world, camera camera)
{
    uchar4* dev_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &opengl.cuda_pbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, opengl.cuda_pbo);

    dim3 GRID = opengl.grid;
    dim3 BLOCK = opengl.block;
    main_kernel <<<GRID, BLOCK>>> (dev_ptr, opengl.screen_width, opengl.screen_height, world, camera);
}

