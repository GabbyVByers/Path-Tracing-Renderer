#pragma once

#include "opengl.h"

inline void processKeyboardInput(Opengl& opengl, World& world)
{
    Vec3 forward = { world.camera.direction.x, 0.0f, world.camera.direction.z };
    normalize(forward);
    Vec3 right = cross(world.camera.direction, world.camera.up);
    Vec3 up = { 0.0f, 1.0f, 0.0f };

    float slow = 1.0f;
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
        slow = 0.1f;

    if (glfwGetKey(opengl.window, GLFW_KEY_W) == GLFW_PRESS)
    {
        world.camera.position += forward * slow;
        world.buffer.numAccumulatedFrames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_S) == GLFW_PRESS)
    {
        world.camera.position -= forward * slow;
        world.buffer.numAccumulatedFrames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_D) == GLFW_PRESS)
    {
        world.camera.position += right * slow;
        world.buffer.numAccumulatedFrames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_A) == GLFW_PRESS)
    {
        world.camera.position -= right * slow;
        world.buffer.numAccumulatedFrames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        world.camera.position += up * slow;
        world.buffer.numAccumulatedFrames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        world.camera.position -= up * slow;
        world.buffer.numAccumulatedFrames = 0;
    }
}

inline void processMouseInput(Opengl& opengl, World& world)
{
    double currMouseX;
    double currMouseY;
    glfwGetCursorPos(opengl.window, &currMouseX, &currMouseY);
    double mouseRelX = currMouseX - opengl.prevMouseX;
    double mouseRelY = currMouseY - opengl.prevMouseY;
    opengl.prevMouseX = currMouseX;
    opengl.prevMouseY = currMouseY;
    Vec3 up = { 0.0f, 1.0f, 0.0f };

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        world.buffer.numAccumulatedFrames = 0;
        return;
    }

    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
        return;

    if (mouseRelX != 0.0f)
    {
        world.camera.direction = rotate(world.camera.direction, up, 0.005f * -mouseRelX);
        fixCamera(world.camera);
        world.buffer.numAccumulatedFrames = 0;
    }

    if (mouseRelY != 0.0f)
    {
        world.camera.direction = rotate(world.camera.direction, world.camera.right, 0.005f * -mouseRelY);
        fixCamera(world.camera);
        world.buffer.numAccumulatedFrames = 0;
    }
}

inline void processKeyboardMouseInput(Opengl& opengl, World& world)
{
    world.buffer.numAccumulatedFrames++;
    processKeyboardInput(opengl, world);
    processMouseInput(opengl, world);
}

