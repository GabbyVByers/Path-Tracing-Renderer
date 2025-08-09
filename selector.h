#pragma once

#include "opengl.h"

inline int mouseBoxesIntersection(const Ray& ray, const World& world)
{
    float closest_t = FLT_MAX;
    int index = -1;

    for (int i = 0; i < world.boxes.numBoxes; i++)
    {
        const Vec3& A = world.boxes.hostBoxes[i].boxMin;
        const Vec3& B = world.boxes.hostBoxes[i].boxMax;

        float tmin = -FLT_MAX;
        float tmax = FLT_MAX;

        if (ray.direction.x != 0.0f)
        {
            float invD = 1.0f / ray.direction.x;
            float t0 = (A.x - ray.origin.x) * invD;
            float t1 = (B.x - ray.origin.x) * invD;
            if (invD < 0.0f)
            {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.x < A.x || ray.origin.x > B.x)
            return -1;

        if (ray.direction.y != 0.0f)
        {
            float invD = 1.0f / ray.direction.y;
            float t0 = (A.y - ray.origin.y) * invD;
            float t1 = (B.y - ray.origin.y) * invD;
            if (invD < 0.0f)
            {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.y < A.y || ray.origin.y > B.y)
            return -1;

        if (ray.direction.z != 0.0f)
        {
            float invD = 1.0f / ray.direction.z;
            float t0 = (A.z - ray.origin.z) * invD;
            float t1 = (B.z - ray.origin.z) * invD;
            if (invD < 0.0f)
            {
                float temp = t0;
                t0 = t1;
                t1 = temp;
            }
            tmin = fmaxf(tmin, t0);
            tmax = fminf(tmax, t1);
        }
        else if (ray.origin.z < A.z || ray.origin.z > B.z)
            return -1;

        if (tmax < tmin)
            continue;

        if (tmin < 0.0f)
            continue;

        if (tmin < closest_t)
        {
            closest_t = tmin;
            index = i;
        }
    }

    return index;
}

inline int mouseSpheresIntersection(const Ray& ray, const World& world)
{
    float closest_t = FLT_MAX;
    int index = -1;

    for (int i = 0; i < world.spheres.numSpheres; i++)
    {
        Vec3 V = ray.origin - world.spheres.hostSpheres[i].position;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(V, ray.direction);
        float c = dot(V, V) - (world.spheres.hostSpheres[i].radius * world.spheres.hostSpheres[i].radius);

        float discriminant = (b * b) - (4.0f * a * c);
        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-b) + sqrt(discriminant)) / (2.0f * a);
        float t2 = ((-b) - sqrt(discriminant)) / (2.0f * a);
        float t = fmin(t1, t2);

        if (t <= 0.0f)
            continue;

        if (t < closest_t)
        {
            closest_t = t;
            index = i;
        }
    }

    return index;
}

inline void selectSphere(Opengl& opengl, World& world)
{
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS)
        return;

    for (int i = 0; i < world.spheres.numSpheres; i++)
        world.spheres.hostSpheres[i].isSelected = false;

    for (int i = 0; i < world.boxes.numBoxes; i++)
        world.boxes.hostBoxes[i].isSelected = false;

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

    int mouseSphereIndex = mouseSpheresIntersection(mouseRay, world);
    if (mouseSphereIndex != -1)
        world.spheres.hostSpheres[mouseSphereIndex].isSelected = true;

    int mouseBoxesIndex = mouseBoxesIntersection(mouseRay, world);
    if (mouseBoxesIndex != -1)
        world.boxes.hostBoxes[mouseBoxesIndex].isSelected = true;

    updateSpheresOnGpu(world.spheres);
    updateBoxesOnGpu(world.boxes);
}

