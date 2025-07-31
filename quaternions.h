#pragma once

#include "vec3.h"

struct Quaternion
{
    float w, x, y, z;

    Quaternion operator * (const Quaternion& b) const
    {
        return
        {
            (w * b.w) - (x * b.x) - (y * b.y) - (z * b.z),
            (w * b.x) + (x * b.w) + (y * b.z) - (z * b.y),
            (w * b.y) - (x * b.z) + (y * b.w) + (z * b.x),
            (w * b.z) + (x * b.y) - (y * b.x) + (z * b.w)
        };
    }
};

inline vec3 rotate(const vec3& a, const vec3& b, const float& theta)
{
    float half = theta * 0.5f;
    float s = sin(half);

    Quaternion q = { cos(half), b.x * s, b.y * s, b.z * s };
    Quaternion qinv = { q.w, -q.x, -q.y, -q.z };
    Quaternion p = { 0.0f, a.x, a.y, a.z };
    Quaternion qp = q * p;
    Quaternion result = qp * qinv;

    return { result.x, result.y, result.z };
}

