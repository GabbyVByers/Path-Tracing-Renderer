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

inline void drawImgui(bool& enableGUI, GLFWwindow* window, World& world, char fileName[24], int fps)
{
    if (!enableGUI)
    {
        glfwSwapBuffers(window);
        glfwPollEvents();
        std::cout << "Rays: " << world.global.numAccumulatedFrames << "\n";
        return;
    }

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

    if (ImGui::Button("Disable GUI"))
        enableGUI = false;
    ImGui::Text("Press P to Reopen GUI");

    ImGui::Text(" ");
    ImGui::Text("Frames Per Second: %d", fps);
    ImGui::Text("Accumulated Frames: %d", world.global.numAccumulatedFrames);

    ImGui::Text(" ");
    if (ImGui::Button("Toggle Enviroment Lighting"))
        world.sky.toggleSky = !world.sky.toggleSky;
    
    if (ImGui::Button("Enable VSYNC"))
        glfwSwapInterval(1);

    ImGui::SameLine();
    if (ImGui::Button("Disable VSYNC"))
        glfwSwapInterval(0);

    ImGui::Text(" ");
    ImGui::SliderInt("Bounce Limit", &world.global.maxBounceLimit, 0, 32);

    ImGui::Text(" ");
    ImGui::Text("Camera Properties");
    ImGui::DragFloat3("Position", (float*)&world.camera.position, 0.05f);
    ImGui::DragFloat3("Direction", (float*)&world.camera.direction, 0.005f);
    fixCamera(world.camera);

    ImGui::Text(" ");
    ImGui::Text("Sky Parameters");
    ImGui::DragFloat3("Sun Direction", (float*)&world.sky.sunDirection, 0.005f);
    normalize(world.sky.sunDirection);
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
        ImGui::DragFloat3("Position", (float*)&world.spheres.hostPointer[indexSphere].position, 0.05f);
        ImGui::DragFloat("Radius", &world.spheres.hostPointer[indexSphere].radius, 0.05f);
        
        ImGui::Text(" ");
        ImGui::DragFloat("Roughness", &world.spheres.hostPointer[indexSphere].roughness, 0.005f, 0.0f, 1.0f);
        ImGui::ColorEdit3("Color", (float*)&world.spheres.hostPointer[indexSphere].color);

        ImGui::Text(" ");
        if (ImGui::Button("Toggle Light Source"))
            world.spheres.hostPointer[indexSphere].isLightSource = !world.spheres.hostPointer[indexSphere].isLightSource;
        ImGui::DragFloat("Intensity", &world.spheres.hostPointer[indexSphere].lightIntensity, 0.005f, 0.0f, 35.0f);

        ImGui::Text(" ");
        if (ImGui::Button("Delete"))
            world.spheres.remove(indexSphere);
        ImGui::SameLine();
        if (ImGui::Button("Duplicate"))
        {
            Sphere newSphere = world.spheres.hostPointer[indexSphere];
            newSphere.position.y += newSphere.radius * 2.4f;
            newSphere.isSelected = true;
            world.spheres.hostPointer[indexSphere].isSelected = false;
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
        ImGui::DragFloat3("Position", (float*)&world.boxes.hostPointer[indexBox].position, 0.05f);
        ImGui::DragFloat3("Size", (float*)&world.boxes.hostPointer[indexBox].size, 0.05f);

        ImGui::Text(" ");
        ImGui::DragFloat("Roughness", &world.boxes.hostPointer[indexBox].roughness, 0.005f, 0.0f, 1.0f);
        ImGui::ColorEdit3("Color", (float*)&world.boxes.hostPointer[indexBox].color);

        ImGui::Text(" ");
        if (ImGui::Button("Toggle Light Source"))
            world.boxes.hostPointer[indexBox].isLightSource = !world.boxes.hostPointer[indexBox].isLightSource;
        ImGui::DragFloat("Intensity", &world.boxes.hostPointer[indexBox].lightIntensity, 0.005f, 0.0f, 35.0f);

        ImGui::Text(" ");
        if (ImGui::Button("Delete"))
            world.boxes.remove(indexBox);
        ImGui::SameLine();
        if (ImGui::Button("Duplicate"))
        {
            Box newBox = world.boxes.hostPointer[indexBox];
            newBox.position.y += newBox.size.y * 2.4f;
            newBox.isSelected = true;
            world.boxes.hostPointer[indexBox].isSelected = false;
            world.boxes.add(newBox);
        }

        world.boxes.updateHostToDevice();
        ImGui::End();
    }

    ImGui::Begin("Exit Menu");
    if (ImGui::Button("Exit Application"))
        glfwSetWindowShouldClose(window, true);
    ImGui::End();

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    glfwPollEvents();
}

