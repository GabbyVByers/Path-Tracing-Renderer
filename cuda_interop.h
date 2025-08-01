#pragma once

#include "opengl_manager.h"

void launch_cuda_kernel(opengl& opengl, sphere* dev_spheres, int num_spheres, camera camera)
{
    uchar4* dev_ptr;
    size_t size;
    cudaGraphicsMapResources(1, &opengl.cuda_pbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, opengl.cuda_pbo);

    dim3 GRID = opengl.grid;
    dim3 BLOCK = opengl.block;
    renderKernel <<<GRID, BLOCK>>> (dev_ptr, width, height, dev_spheres, num_spheres, camera);
}

