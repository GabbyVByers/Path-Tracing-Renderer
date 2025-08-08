#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "saving.h"
#include "framerate.h"

inline void setupImgui(GLFWwindow* window) {
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

inline void drawImgui(World& world, char fileName[24], int fps) {
    Sphere* selectedHostSphere = nullptr;
    int indexSelectedSphere = -1;

    for (int i = 0; i < world.spheres.numSpheres; i++) {
        if (world.spheres.hostSpheres[i].isSelected == true) {
            selectedHostSphere = &world.spheres.hostSpheres[i];
            indexSelectedSphere = i;
            break;
        }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    
    ImGui::Begin("DEBUGGER");
    ImGui::Text("Frames Per Second: %d", fps);
    ImGui::Text("Accumulated Frames: %d", world.buffer.numAccumulatedFrames);

    ImGui::Text(" ");
    if (ImGui::Button("Toggle Enviroment Lighting"))
        world.sky.toggleSky = !world.sky.toggleSky;
    
    ImGui::Text(" ");
    ImGui::Text("SUN DIRECTION");
    ImGui::SliderFloat("Sun.dir.x", &world.sky.sunDirection.x, -1.0f, 1.0f);
    ImGui::SliderFloat("Sun.dir.y", &world.sky.sunDirection.y, -1.0f, 1.0f);
    ImGui::SliderFloat("Sun.dir.z", &world.sky.sunDirection.z, -1.0f, 1.0f);
    normalize(world.sky.sunDirection);

    ImGui::Text(" ");
    ImGui::Text("SKY PARAMETERS");
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

    if (selectedHostSphere != nullptr) {
        ImGui::Begin("Selected Sphere");

        if (ImGui::Button("Deselect Sphere"))
            selectedHostSphere->isSelected = false;

        ImGui::Text(" ");
        ImGui::SliderFloat("Roughness", &selectedHostSphere->roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("Radius", &selectedHostSphere->radius, 0.1f, 20.0f);
        

        ImGui::Text(" ");  if (ImGui::Button("+X")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++X")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++X") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---X") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--X")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-X")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.x -= 0.01f; }

                           if (ImGui::Button("+Y")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++Y")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++Y") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---Y") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--Y")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-Y")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.y -= 0.01f; }

                           if (ImGui::Button("+Z")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++Z")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++Z") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---Z") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--Z")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-Z")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selectedHostSphere->position.z -= 0.01f; }
        
        ImGui::Text(" ");
        ImGui::ColorEdit3("Color", (float*)&selectedHostSphere->color);

        
        if (ImGui::Button("Toggle Light Source"))
            selectedHostSphere->isLightSource = !selectedHostSphere->isLightSource;
        ImGui::SliderFloat("Intensity", &selectedHostSphere->lightIntensity, 0.0f, 35.0f);

        ImGui::End();

        updateSpheresOnGpu(world.spheres);
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

