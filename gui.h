#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "camera.h"

inline void setup_imgui(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    float scale = 1.5f;
    io.FontGlobalScale = scale;
    ImGui::GetStyle().ScaleAllSizes(scale);
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void draw_imgui(camera& camera)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Debugger");
    ImGui::SliderInt("Buffer Limit", &camera.buffer_limit, 0, 500);
    ImGui::SliderInt("Rays Per Pixel", &camera.rays_per_pixel, 0, 100);
    ImGui::SliderFloat("Light Source x:", &camera.light_direction.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Source y:", &camera.light_direction.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Source z:", &camera.light_direction.z, -1.0f, 1.0f);
    normalize(camera.light_direction);
    ImGui::Text("Camera Position x:%.2f, y:%.2f, z:%.2f", camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("Camera Direction x:%.2f, y:%.2f, z:%.2f", camera.direction.x, camera.direction.y, camera.direction.z);
    ImGui::Text("Camera Up x:%.2f, y:%.2f, z:%.2f", camera.up.x, camera.up.y, camera.up.z);
    ImGui::Text("Camera Right x:%.2f, y:%.2f, z:%.2f", camera.right.x, camera.right.y, camera.right.z);
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}