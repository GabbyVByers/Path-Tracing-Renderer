#pragma once

#include "cuda_runtime.h"

/*
	A utility for managing an array of elements between the Host and Device
	The Host is responsible for adding or removing elements from the array.
	The Device need only concern herself with reading or writing data to the existing array.
	Accessing an element with operator "example_array[index]" returns a pointer to that element.
	Call the appropriate update function if you modify the element with the pointer returned by accessing.
*/

// I am looking through the callstack,
// and i think the deconstructor is being called when world is passed to mainKernel
// should try converting this from a class to a struct but it's 1 am would rather go to bed.

template<typename type>

class SharedArray
{

public:

	size_t size = 0;
	size_t capacity = 1;
	type* hostPointer = nullptr;
	type* devicePointer = nullptr;

	SharedArray()
	{
		hostPointer = new type[capacity];
		cudaMalloc((void**)&devicePointer, sizeof(type) * capacity);
	}

	~SharedArray()
	{
		delete[] hostPointer;
		cudaFree(devicePointer);
	}

	void doubleCapacity()
	{
		type* newHostPointer = nullptr;
		type* newDevicePointer = nullptr;

		newHostPointer = new type[capacity * 2];
		cudaMalloc((void**)&newDevicePointer, sizeof(type) * capacity * 2);

		memcpy(newHostPointer, hostPointer, sizeof(type) * capacity);
		cudaMemcpy(newDevicePointer, devicePointer, sizeof(type) * capacity, cudaMemcpyDeviceToDevice);

		delete[] hostPointer;
		cudaFree(devicePointer);

		hostPointer = newHostPointer;
		devicePointer = newDevicePointer;

		capacity = capacity * 2;
	}

	void remove(size_t index)
	{
		if (index >= size)
			return;

		if (index == size - 1)
		{
			size--;
			return;
		}

		for (size_t i = index + 1; i < size; i++)
		{
			hostPointer[i - 1] = hostPointer[i];
		}
		size--;

		updateHostToDevice();
	}

	void add(type element)
	{
		if (size == capacity)
			doubleCapacity();

		hostPointer[size] = element;
		size++;

		updateHostToDevice();
	}

	void clear()
	{
		delete[] hostPointer;
		cudaFree(devicePointer);

		size_t size = 0;
		size_t capacity = 1;

		hostPointer = new type[capacity];
		cudaMalloc((void**)&devicePointer, sizeof(type) * capacity);
	}

	void updateHostToDevice()
	{
		cudaMemcpy(devicePointer, hostPointer, size * sizeof(type), cudaMemcpyHostToDevice);
	}

	void updateDeviceToHost()
	{
		cudaMemcpy(hostPointer, devicePointer, size * sizeof(type), cudaMemcpyDeviceToHost);
	}

	type* getHostPtrAtIndex(size_t index)
	{
		return &hostPointer[index];
	}

	__device__ type* getDevicePtrAtIndex(size_t index)
	{
		return &devicePointer[index];
	}

	__host__ __device__ size_t getSize() const
	{
		return size;
	}
};

