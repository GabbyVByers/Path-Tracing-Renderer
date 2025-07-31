#pragma once

#include "vec3.h"

struct sphere {
    vec3 position;
    float radius;
    vec3 color;
};

struct hit_info {
    bool did_hit;
    bool did_hit_light_source;
    vec3 hit_location;
    vec3 hit_color;
    vec3 hit_normal;
};

struct ray {
    vec3 origin;
    vec3 direction;
};

