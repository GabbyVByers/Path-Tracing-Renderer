#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "saving.h"

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

inline void draw_imgui(World& world)
{

    Sphere* selected_host_sphere = nullptr;
    int index_selected_sphere = -1;
    for (int i = 0; i < world.spheres.num_spheres; i++)
    {
        if (world.spheres.host_spheres[i].is_selected == true)
        {
            selected_host_sphere = &world.spheres.host_spheres[i];
            index_selected_sphere = i;
            break;
        }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    
    ImGui::Begin("DEBUGGER");
    ImGui::Text("Accumulated Frames: %d", world.buffer.num_accumulated_frames);
    
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

    ImGui::Text(" ");
    if (ImGui::Button("Toggle Enviroment Lighting"))
        world.sky.toggle_sky = !world.sky.toggle_sky;
    if (ImGui::Button("Save Spheres"))
        save_spheres(world);
    if (ImGui::Button("Load Spheres"))
        load_spheres(world);

    if (selected_host_sphere != nullptr)
    {
        ImGui::Begin("Selected Sphere");

        if (ImGui::Button("Deselect Sphere"))
            selected_host_sphere->is_selected = false;

        ImGui::Text(" ");
        ImGui::SliderFloat("Roughness", &selected_host_sphere->roughness, 0.0f, 1.0f);
        ImGui::SliderFloat("Radius", &selected_host_sphere->radius, 0.1f, 20.0f);
        

        ImGui::Text(" ");  if (ImGui::Button("+X")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++X")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++X") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---X") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--X")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-X")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.x -= 0.01f; }

                           if (ImGui::Button("+Y")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++Y")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++Y") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---Y") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--Y")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-Y")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.y -= 0.01f; }

                           if (ImGui::Button("+Z")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z += 0.01f; }
        ImGui::SameLine(); if (ImGui::Button("++Z")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z += 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("+++Z") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z += 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("---Z") || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z -= 1.0f;  }
        ImGui::SameLine(); if (ImGui::Button("--Z")  || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z -= 0.1f;  }
        ImGui::SameLine(); if (ImGui::Button("-Z")   || (ImGui::IsItemActive() && ImGui::IsMouseDown(0))) { selected_host_sphere->position.z -= 0.01f; }
        
        ImGui::Text(" ");
        ImGui::ColorEdit3("Change Color", (float*)&selected_host_sphere->color);
        if (ImGui::Button("Toggle Light Source"))
            selected_host_sphere->is_light_source = !selected_host_sphere->is_light_source;
        ImGui::SliderFloat("Light Intensity", &selected_host_sphere->light_intensity, 0.0f, 35.0f);

        ImGui::End();

        update_spheres_on_gpu(world.spheres);
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

