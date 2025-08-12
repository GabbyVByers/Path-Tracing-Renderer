#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "saving.h"
#include "framerate.h"

inline void setupImgui(GLFWwindow* window)
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

inline void drawImgui(World& world, char fileName[24], int fps)
{
    int indexSphere = -1;
    for (int i = 0; i < world.spheres.size; i++)
    {
        if (world.spheres.hostPointer[i].isSelected == true)
        {
            indexSphere = i;
            break;
        }
    }

    int indexBox = -1;
    for (int i = 0; i < world.boxes.size; i++)
    {
        if (world.boxes.hostPointer[i].isSelected == true)
        {
            indexBox = i;
            break;
        }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    
    ImGui::Begin("Main Menu");

    ImGui::Text(" ");
    ImGui::Text("Frames Per Second: %d", fps);
    ImGui::Text("Accumulated Frames: %d", world.metadata.numAccumulatedFrames);

    ImGui::Text(" ");
    ImGui::Text("Camera Properties");
    ImGui::DragFloat3("Position", (float*)&world.camera.position, 0.05f);
    ImGui::DragFloat3("Direction", (float*)&world.camera.direction, 0.05f);
    fixCamera(world.camera);

    ImGui::Text(" ");
    if (ImGui::Button("Toggle Enviroment Lighting"))
        world.sky.toggleSky = !world.sky.toggleSky;
    
    if (ImGui::Button("Enable VSYNC"))
        glfwSwapInterval(1);

    ImGui::SameLine();
    if (ImGui::Button("Disable VSYNC"))
        glfwSwapInterval(0);

    ImGui::Text(" ");
    ImGui::DragFloat3("Sun Direction", (float*)&world.sky.sunDirection, 0.05f);
    normalize(world.sky.sunDirection);

    ImGui::Text(" ");
    ImGui::Text("Sky Parameters");
    ImGui::SliderFloat("Sun Int", &world.sky.sunIntensity, 0.0f, 100.0f);
    ImGui::SliderFloat("Sun Exp", &world.sky.sunExponent, 1.0f, 150.0f);
    ImGui::SliderFloat("Hor Exp", &world.sky.horizonExponent, 0.0f, 1.0f);

    ImGui::Text(" ");
    ImGui::Text("Save or Load World Geometry");
    ImGui::InputText("File Name", fileName, IM_ARRAYSIZE(fileName));
    if (ImGui::Button("Save Geometry"))
        saveSpheres(world, fileName);
    ImGui::SameLine();
    if (ImGui::Button("Load Geometry"))
        loadSpheres(world, fileName);

    if (indexSphere != -1)
    {
        ImGui::Begin("Selected Sphere");

        if (ImGui::Button("Deselect Sphere"))
            world.spheres.hostPointer[indexSphere].isSelected = false;

        ImGui::Text(" ");
        ImGui::SliderFloat("Radius", &world.spheres.hostPointer[indexSphere].radius, 0.1f, 20.0f);
        ImGui::DragFloat3("Position", (float*)&world.spheres.hostPointer[indexSphere].position, 0.05f);
        
        ImGui::Text(" ");
        ImGui::SliderFloat("Roughness", &world.spheres.hostPointer[indexSphere].roughness, 0.0f, 1.0f);
        ImGui::ColorEdit3("Color", (float*)&world.spheres.hostPointer[indexSphere].color);

        ImGui::Text(" ");
        if (ImGui::Button("Toggle Light Source"))
            world.spheres.hostPointer[indexSphere].isLightSource = !world.spheres.hostPointer[indexSphere].isLightSource;
        ImGui::SliderFloat("Intensity", &world.spheres.hostPointer[indexSphere].lightIntensity, 0.0f, 35.0f);

        ImGui::Text(" ");
        if (ImGui::Button("Delete"))
            world.spheres.remove(indexSphere);
        ImGui::SameLine();
        if (ImGui::Button("Duplicate"))
        {
            Sphere newSphere = world.spheres.hostPointer[indexSphere];
            newSphere.position.y += newSphere.radius * 2.0f;
            world.spheres.add(newSphere);
        }

        world.spheres.updateHostToDevice();
        ImGui::End();
    }

    if (indexBox != -1)
    {
        ImGui::Begin("Selected Box");

        if (ImGui::Button("Deselect Box"))
            world.boxes.hostPointer[indexBox].isSelected = false;

        ImGui::Text(" ");
        ImGui::DragFloat3("Position Min", (float*)&world.boxes.hostPointer[indexBox].boxMin, 0.05f);
        ImGui::DragFloat3("Position Max", (float*)&world.boxes.hostPointer[indexBox].boxMax, 0.05f);

        ImGui::Text(" ");
        ImGui::SliderFloat("Roughness", &world.boxes.hostPointer[indexBox].roughness, 0.0f, 1.0f);
        ImGui::ColorEdit3("Color", (float*)&world.boxes.hostPointer[indexBox].color);

        ImGui::Text(" ");
        if (ImGui::Button("Toggle Light Source"))
            world.boxes.hostPointer[indexBox].isLightSource = !world.boxes.hostPointer[indexBox].isLightSource;
        ImGui::SliderFloat("Intensity", &world.boxes.hostPointer[indexBox].lightIntensity, 0.0f, 35.0f);

        ImGui::Text(" ");
        if (ImGui::Button("Delete"))
            world.boxes.remove(indexBox);
        ImGui::SameLine();
        if (ImGui::Button("Duplicate"))
        {
            Box newBox = world.boxes.hostPointer[indexBox];
            Vec3 offset = (newBox.boxMax - newBox.boxMin) * 1.2f;
            newBox.boxMax += offset;
            newBox.boxMin += offset;
            world.boxes.add(newBox);
        }

        world.boxes.updateHostToDevice();
        ImGui::End();
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

