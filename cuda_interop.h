#pragma once

#include "opengl_manager.h"
#include "dataStructures.h"
#include "kernel.h"

inline void launch_cuda_kernel(opengl& opengl, sphere* dev_spheres, int num_spheres, camera camera)
{
    uchar4* dev_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &opengl.cuda_pbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, opengl.cuda_pbo);

    dim3 GRID = opengl.grid;
    dim3 BLOCK = opengl.block;
    main_kernel <<<GRID, BLOCK>>> (dev_ptr, opengl.screen_width, opengl.screen_height, dev_spheres, num_spheres, camera);
}

