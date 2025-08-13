#pragma once

#include "opengl.h"

struct SimpleHitInfo
{
    float hit_t = FLT_MAX;
    int hit_index = -1;
};

inline SimpleHitInfo mouseBoxesIntersection(const Ray& ray, const World& world)
{
    SimpleHitInfo info;

    for (int i = 0; i < world.boxes.size; i++)
    {
        Box& box = world.boxes.hostPointer[i];
        Vec3& position = box.position;
        Vec3& size = box.size;

        Vec3 min = position - size;
        Vec3 max = position + size;

        float tminx = (min.x - ray.origin.x) / ray.direction.x;
        float tmaxx = (max.x - ray.origin.x) / ray.direction.x;
        if (tminx > tmaxx) { float temp = tminx; tminx = tmaxx; tmaxx = temp; }

        float tminy = (min.y - ray.origin.y) / ray.direction.y;
        float tmaxy = (max.y - ray.origin.y) / ray.direction.y;
        if (tminy > tmaxy) { float temp = tminy; tminy = tmaxy; tmaxy = temp; }

        float tminz = (min.z - ray.origin.z) / ray.direction.z;
        float tmaxz = (max.z - ray.origin.z) / ray.direction.z;
        if (tminz > tmaxz) { float temp = tminz; tminz = tmaxz; tmaxz = temp; }

        float tmin = fmaxf(fmaxf(tminx, tminy), tminz);
        float tmax = fminf(fminf(tmaxx, tmaxy), tmaxz);

        if (tmin > tmax)
            continue;

        if (tmin < 0.0f)
            continue;

        if (tmin < info.hit_t)
        {
            info.hit_t = tmin;
            info.hit_index = i;
        }
    }

    return info;
}

inline SimpleHitInfo mouseSpheresIntersection(const Ray& ray, World& world)
{
    SimpleHitInfo info;

    for (int i = 0; i < world.spheres.size; i++)
    {
        Sphere& hostSpherePointer = world.spheres.hostPointer[i];

        Vec3 V = ray.origin - hostSpherePointer.position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (hostSpherePointer.radius * hostSpherePointer.radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        if (t < info.hit_t)
        {
            info.hit_t = t;
            info.hit_index = i;
        }
    }

    return info;
}

inline void selectSphere(Opengl& opengl, World& world)
{
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS)
        return;

    for (int i = 0; i < world.spheres.size; i++)
        world.spheres.hostPointer[i].isSelected = false;

    for (int i = 0; i < world.boxes.size; i++)
        world.boxes.hostPointer[i].isSelected = false;

    int screenWidth = opengl.screenWidth;
    int screenHeight = opengl.screenHeight;
    double mouseX;
    double mouseY;
    glfwGetCursorPos(opengl.window, &mouseX, &mouseY);
    float v = (((screenHeight - mouseY) / screenHeight) * 2.0f) - 1.0f;
    float u = (((mouseX / screenWidth) * 2.0f) - 1.0f) * (screenWidth / (float)screenHeight);
    Ray mouseRay;
    mouseRay.origin = world.camera.position;
    mouseRay.direction = (world.camera.direction * world.camera.depth) + (world.camera.right * u) + (world.camera.up * v);

    SimpleHitInfo sphereInfo = mouseSpheresIntersection(mouseRay, world);
    SimpleHitInfo boxInfo = mouseBoxesIntersection(mouseRay, world);
    
    if (sphereInfo.hit_t < boxInfo.hit_t)
    {
        world.spheres.hostPointer[sphereInfo.hit_index].isSelected = true;
        world.spheres.updateHostToDevice();
        return;
    }

    else if (boxInfo.hit_t < sphereInfo.hit_t)
    {
        world.boxes.hostPointer[boxInfo.hit_index].isSelected = true;
        world.boxes.updateHostToDevice();
    }

    return;
}

