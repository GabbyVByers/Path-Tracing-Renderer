#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "structs.h"

inline void setup_imgui(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    //float scale = 1.5f;
    //io.FontGlobalScale = scale;
    //ImGui::GetStyle().ScaleAllSizes(scale);
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void draw_imgui(World& world, Camera& camera)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::Begin("Debugger");
    ImGui::Text("Accumulated Frames: %d", world.num_accumulated_frames);
    ImGui::SliderInt("Recursive Bounce Limit", &world.max_bounce_limit, 0, 100);
    ImGui::SliderFloat("Light Source x:", &world.light_direction.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Source y:", &world.light_direction.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Light Source z:", &world.light_direction.z, -1.0f, 1.0f);
    normalize(world.light_direction);
    ImGui::Text("Camera Position x:%.2f, y:%.2f, z:%.2f", camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("Camera Direction x:%.2f, y:%.2f, z:%.2f", camera.direction.x, camera.direction.y, camera.direction.z);
    ImGui::Text("Camera Up x:%.2f, y:%.2f, z:%.2f", camera.up.x, camera.up.y, camera.up.z);
    ImGui::Text("Camera Right x:%.2f, y:%.2f, z:%.2f", camera.right.x, camera.right.y, camera.right.z);
    ImGui::SliderFloat("Sun Intensity:", &world.sun_intensity, 0.0f, 20.0f);
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

