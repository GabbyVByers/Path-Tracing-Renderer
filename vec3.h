#pragma once

#include <math.h>

struct vec3
{
	float x;
	float y;
	float z;

	__host__ __device__ vec3 operator + (const vec3& vec) const
	{
		return { x + vec.x, y + vec.y, z + vec.z };
	}

	__host__ __device__ vec3 operator - (const vec3& vec) const
	{
		return { x - vec.x, y - vec.y, z - vec.z };
	}

	__host__ __device__ vec3 operator * (const vec3& vec) const
	{
		return { ((y * vec.z) - (z * vec.y)), ((z * vec.x) - (x * vec.z)), ((x * vec.y) - (y * vec.x)) };
	}

	__host__ __device__ vec3 operator * (const float& value) const
	{
		return { x * value, y * value, z * value };
	}

	__host__ __device__ vec3 operator / (const float& value) const
	{
		return { x / value, y / value, z / value };
	}

	__host__ __device__ vec3& operator += (const vec3& vec)
	{
		x += vec.x; y += vec.y; z += vec.z;
		return *this;
	}

	__host__ __device__ vec3& operator -= (const vec3& vec)
	{
		x -= vec.x; y -= vec.y; z -= vec.z;
		return *this;
	}

	__host__ __device__ vec3& operator *= (const vec3& vec)
	{
		x = x * vec.x; y = y * vec.y; z = z * vec.z;
		return *this;
	}

	__host__ __device__ vec3& operator *= (const float& value)
	{
		x = x * value; y = y * value; z = z * value;
		return *this;
	}

	__host__ __device__ vec3& operator /= (const float& value)
	{
		x = x / value; y = y / value; z = z / value;
		return *this;
	}

	__host__ __device__ vec3& operator = (const float& value)
	{
		x = value; y = value; z = value;
		return *this;
	}

	__host__ __device__ vec3& operator = (const vec3& vec)
	{
		x = vec.x; y = vec.y; z = vec.z;
		return *this;
	}
};

__host__ __device__ inline float dot(const vec3& a, const vec3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__host__ __device__ inline void normalize(vec3& vec)
{
	float lengthSq = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	vec /= sqrt(lengthSq);
}

__host__ __device__ inline vec3 multiply(const vec3& a, const vec3& b)
{
	return { a.x * a.x, a.y * a.y, a.z * a.z };
}

