#pragma once

struct Vec3
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	__host__ __device__ Vec3 operator + (const Vec3& vec) const
	{
		return
		{
			x + vec.x,
			y + vec.y,
			z + vec.z
		};
	}

	__host__ __device__ Vec3 operator - (const Vec3& vec) const
	{
		return
		{
			x - vec.x,
			y - vec.y,
			z - vec.z
		};
	}

	__host__ __device__ Vec3 operator * (const Vec3& vec) const
	{
		return
		{
			x * vec.x,
			y * vec.y,
			z * vec.z
		};
	}

	__host__ __device__ Vec3 operator * (const float& value) const
	{
		return
		{
			x * value,
			y * value,
			z * value
		};
	}

	__host__ __device__ Vec3 operator / (const float& value) const
	{
		return
		{
			x / value,
			y / value,
			z / value
		};
	}

	__host__ __device__ Vec3& operator += (const Vec3& vec)
	{
		x += vec.x;
		y += vec.y;
		z += vec.z;
		return *this;
	}

	__host__ __device__ Vec3& operator -= (const Vec3& vec)
	{
		x -= vec.x;
		y -= vec.y;
		z -= vec.z;
		return *this;
	}

	__host__ __device__ Vec3& operator *= (const Vec3& vec)
	{
		x *= vec.x;
		y *= vec.y;
		z *= vec.z;
		return *this;
	}

	__host__ __device__ Vec3& operator *= (const float& value)
	{
		x *= value;
		y *= value;
		z *= value;
		return *this;
	}

	__host__ __device__ Vec3& operator /= (const float& value)
	{
		x /= value;
		y /= value;
		z /= value;
		return *this;
	}

	__host__ __device__ Vec3& operator = (const float& value)
	{
		x = value;
		y = value;
		z = value;
		return *this;
	}

	__host__ __device__ Vec3& operator = (const Vec3& vec)
	{
		x = vec.x;
		y = vec.y;
		z = vec.z;
		return *this;
	}
};

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b)
{
	return
	{
		((a.y * b.z) - (a.z * b.y)),
		((a.z * b.x) - (a.x * b.z)),
		((a.x * b.y) - (a.y * b.x))
	};
}

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b)
{
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

__host__ __device__ inline void normalize(Vec3& vec)
{
	float lengthSq = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	vec /= sqrt(lengthSq);
}

__host__ __device__ inline Vec3 returnNormalized(Vec3 vec)
{
	float lengthSq = (vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z);
	return vec /= sqrt(lengthSq);
}

__host__ __device__ inline Vec3 rgb(const unsigned char& r, const unsigned char& g, const unsigned char& b)
{
	return
	{
		(r / 255.99f),
		(g / 255.99f),
		(b / 255.99f)
	};
}

__host__ __device__ inline Vec3 lerpVec3(const Vec3& a, const Vec3& b, float weight)
{
	return
	{
		(a.x * (1.0f - weight)) + (b.x * weight),
		(a.y * (1.0f - weight)) + (b.y * weight),
		(a.z * (1.0f - weight)) + (b.z * weight)
	};
}

__host__ __device__ inline Vec3 reflect(const Vec3& a, const Vec3& n)
{
	float s = 2.0f * dot(a, n);
	return a - (n * s);
}

