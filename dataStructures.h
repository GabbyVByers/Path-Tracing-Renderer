#pragma once

#include "vec3.h"

struct Sphere
{
    bool isLightSource;
    vec3 position;
    float radius;
    vec3 color;
};

struct HitInfo
{
    bool didHit;
    bool didHitLightSource;
    vec3 hitLocation;
    vec3 hitColor;
    vec3 hitNormal;
};

struct Ray
{
    vec3 origin;
    vec3 direction;
};

