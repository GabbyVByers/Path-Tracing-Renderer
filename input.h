#pragma once

#include "opengl.h"

inline void process_keyboard_input(opengl& opengl, camera& camera)
{
    camera.buffer_size++;
    vec3 forward = { camera.direction.x, 0.0f, camera.direction.z };
    normalize(forward);
    vec3 right = camera.direction * camera.up;
    vec3 up = { 0.0f, 1.0f, 0.0f };

    float slow = 1.0f;
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        slow = 0.1f;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_W) == GLFW_PRESS)
    {
        camera.position += forward * slow;
        camera.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_S) == GLFW_PRESS)
    {
        camera.position -= forward * slow;
        camera.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_D) == GLFW_PRESS)
    {
        camera.position += right * slow;
        camera.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_A) == GLFW_PRESS)
    {
        camera.position -= right * slow;
        camera.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        camera.position += up * slow;
        camera.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        camera.position -= up * slow;
        camera.buffer_size = 0;
    }
}

inline void process_mouse_input(opengl& opengl, camera& camera)
{
    double curr_mouse_x;
    double curr_mouse_y;
    glfwGetCursorPos(opengl.window, &curr_mouse_x, &curr_mouse_y);
    double mouse_rel_x = curr_mouse_x - opengl.prev_mouse_x;
    double mouse_rel_y = curr_mouse_y - opengl.prev_mouse_y;
    opengl.prev_mouse_x = curr_mouse_x;
    opengl.prev_mouse_y = curr_mouse_y;
    vec3 up = { 0.0f, 1.0f, 0.0f };

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        camera.buffer_size = 0;
        return;
    }

    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
    {
        return;
    }

    if (mouse_rel_x != 0.0f)
    {
        camera.direction = rotate(camera.direction, up, 0.005f * -mouse_rel_x);
        fix_camera(camera);
        camera.buffer_size = 0;
    }

    if (mouse_rel_y != 0.0f)
    {
        camera.direction = rotate(camera.direction, camera.right, 0.005f * -mouse_rel_y);
        fix_camera(camera);
        camera.buffer_size = 0;
    }
}