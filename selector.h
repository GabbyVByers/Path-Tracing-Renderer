#pragma once

#include "opengl.h"

inline int mouseSpheresIntersection(const Ray& ray, const World& world) {
    HitInfo info;
    float closest_t = FLT_MAX;
    int index = -1;

    for (int i = 0; i < world.spheres.numSpheres; i++) {
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

        info.didHit = true;

        if (t < closest_t) {
            closest_t = t;
            index = i;
        }
    }

    return index;
}

inline void selectSphere(Opengl& opengl, World& world) {
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_MIDDLE) != GLFW_PRESS)
        return;

    for (int i = 0; i < world.spheres.numSpheres; i++)
        world.spheres.hostSpheres[i].isSelected = false;

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

    updateSpheresOnGpu(world.spheres);
}
