#pragma once

#include "vec3.h"

struct quaternion {
    float w, x, y, z;

    quaternion operator * (const quaternion& b) const {
        return {
            (w * b.w) - (x * b.x) - (y * b.y) - (z * b.z),
            (w * b.x) + (x * b.w) + (y * b.z) - (z * b.y),
            (w * b.y) - (x * b.z) + (y * b.w) + (z * b.x),
            (w * b.z) + (x * b.y) - (y * b.x) + (z * b.w)
        };
    }
};

inline Vec3 rotate(const Vec3& a, const Vec3& b, const float& theta) {
    float half = theta * 0.5f;
    float s = sin(half);

    quaternion q = { cos(half), b.x * s, b.y * s, b.z * s };
    quaternion qinv = { q.w, -q.x, -q.y, -q.z };
    quaternion p = { 0.0f, a.x, a.y, a.z };
    quaternion qp = q * p;
    quaternion result = qp * qinv;

    return { result.x, result.y, result.z };
}

