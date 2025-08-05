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
    float scale = 2.0f;
    io.FontGlobalScale = scale;
    ImGui::GetStyle().ScaleAllSizes(scale);
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

inline void draw_imgui(World& world, Camera& camera)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::Begin("DEBUGGER");
    ImGui::Text("Camera POS   x:%.2f, y:%.2f, z:%.2f", camera.position.x, camera.position.y, camera.position.z);
    ImGui::Text("Camera DIR   x:%.2f, y:%.2f, z:%.2f", camera.direction.x, camera.direction.y, camera.direction.z);
    ImGui::Text("Camera UP    x:%.2f, y:%.2f, z:%.2f", camera.up.x, camera.up.y, camera.up.z);
    ImGui::Text("Camera RIGHT x:%.2f, y:%.2f, z:%.2f", camera.right.x, camera.right.y, camera.right.z);
    ImGui::Text("Accumulated Frames: %d", world.num_accumulated_frames);
    
    ImGui::Text(" ");
    ImGui::Text("SUN DIRECTION");
    ImGui::SliderFloat("Sun Dir.x", &world.sky.sun_direction.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Sun Dir.y", &world.sky.sun_direction.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Sun Dir.z", &world.sky.sun_direction.z, -1.0f, 1.0f);
    normalize(world.sky.sun_direction);

    ImGui::Text(" ");
    ImGui::Text("SKY PARAMETERS");
    ImGui::SliderFloat("Sun Int", &world.sky.sun_intensity, 0.0f, 100.0f);
    ImGui::SliderFloat("Sun Exp", &world.sky.sun_exponent, 1.0f, 150.0f);
    ImGui::SliderFloat("Hor Exp", &world.sky.horizon_exponent, 0.0f, 1.0f);

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

