#pragma once

#include "opengl.h"

inline void process_keyboard_input(Opengl& opengl, World& world)
{
    Vec3 forward = { world.camera.direction.x, 0.0f, world.camera.direction.z };
    normalize(forward);
    Vec3 right = cross(world.camera.direction, world.camera.up);
    Vec3 up = { 0.0f, 1.0f, 0.0f };

    float slow = 1.0f;
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        slow = 0.1f;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_W) == GLFW_PRESS)
    {
        world.camera.position += forward * slow;
        world.buffer.num_accumulated_frames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_S) == GLFW_PRESS)
    {
        world.camera.position -= forward * slow;
        world.buffer.num_accumulated_frames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_D) == GLFW_PRESS)
    {
        world.camera.position += right * slow;
        world.buffer.num_accumulated_frames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_A) == GLFW_PRESS)
    {
        world.camera.position -= right * slow;
        world.buffer.num_accumulated_frames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        world.camera.position += up * slow;
        world.buffer.num_accumulated_frames = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        world.camera.position -= up * slow;
        world.buffer.num_accumulated_frames = 0;
    }
}

inline void process_mouse_input(Opengl& opengl, World& world)
{
    double curr_mouse_x;
    double curr_mouse_y;
    glfwGetCursorPos(opengl.window, &curr_mouse_x, &curr_mouse_y);
    double mouse_rel_x = curr_mouse_x - opengl.prev_mouse_x;
    double mouse_rel_y = curr_mouse_y - opengl.prev_mouse_y;
    opengl.prev_mouse_x = curr_mouse_x;
    opengl.prev_mouse_y = curr_mouse_y;
    Vec3 up = { 0.0f, 1.0f, 0.0f };

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        world.buffer.num_accumulated_frames = 0;
        return;
    }

    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
    {
        return;
    }

    if (mouse_rel_x != 0.0f)
    {
        world.camera.direction = rotate(world.camera.direction, up, 0.005f * -mouse_rel_x);
        fix_camera(world.camera);
        world.buffer.num_accumulated_frames = 0;
    }

    if (mouse_rel_y != 0.0f)
    {
        world.camera.direction = rotate(world.camera.direction, world.camera.right, 0.005f * -mouse_rel_y);
        fix_camera(world.camera);
        world.buffer.num_accumulated_frames = 0;
    }
}

inline void process_keyboard_mouse_input(Opengl& opengl, World& world)
{
    world.buffer.num_accumulated_frames++;
    process_keyboard_input(opengl, world);
    process_mouse_input(opengl, world);
}

