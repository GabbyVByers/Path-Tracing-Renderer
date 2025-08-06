#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <string>
#include "gui.h"
#include "quaternions.h"
#include "kernel.h"

#include <iostream>

struct Opengl
{
    int screen_width = 0;
    int screen_height = 0;
    GLuint pbo = 0;
    GLuint texture_id = 0;
    cudaGraphicsResource* cuda_pbo = nullptr;
    GLFWmonitor* primary = nullptr;
    GLFWwindow* window = nullptr;
    GLuint shader = 0;
    GLuint quad_vao = 0;
    GLuint quad_vbo = 0;
    dim3 block = 0;
    dim3 grid = 0;
    double prev_mouse_x = 0.0f;
    double prev_mouse_y = 0.0f;
};

inline void setup_opengl(Opengl& opengl, int screen_width, int screen_height, std::string title, bool full_screen)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    if (full_screen)
    {
        opengl.primary = glfwGetPrimaryMonitor();
        opengl.screen_width = glfwGetVideoMode(opengl.primary)->width;
        opengl.screen_height = glfwGetVideoMode(opengl.primary)->height;
        opengl.window = glfwCreateWindow(opengl.screen_width, opengl.screen_height, title.c_str(), opengl.primary, nullptr);
    }
    else
    {
        opengl.screen_width = screen_width;
        opengl.screen_height = screen_height;
        opengl.window = glfwCreateWindow(opengl.screen_width, opengl.screen_height, title.c_str(), nullptr, nullptr);
    }

    opengl.block = dim3(32, 32);
    opengl.grid = dim3((opengl.screen_width / 32) + 1, (opengl.screen_height / 32) + 1);
    glfwMakeContextCurrent(opengl.window);
    glfwSwapInterval(0);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glGenBuffers(1, &opengl.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, opengl.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, opengl.screen_width * opengl.screen_height * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&opengl.cuda_pbo, opengl.pbo, cudaGraphicsMapFlagsWriteDiscard);
    glGenTextures(1, &opengl.texture_id);
    glBindTexture(GL_TEXTURE_2D, opengl.texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, opengl.screen_width, opengl.screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    const char* vertex_shader_src =
        R"glsl(
        #version 330 core
        layout (location = 0) in vec2 a_pos;
        layout (location = 1) in vec2 a_tex;
        out vec2 tex_coord;
        void main()
        {
            gl_Position = vec4(a_pos.xy, 0.0, 1.0);
            tex_coord = a_tex;
        }
    )glsl";

    const char* fragment_shader_src =
        R"glsl(
        #version 330 core
        in vec2 tex_coord;
        out vec4 frag_color;
        uniform sampler2D screen_texture;
        void main()
        {
            frag_color = texture(screen_texture, tex_coord);
        }
    )glsl";

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertex_shader_src, nullptr);
    glCompileShader(vert);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragment_shader_src, nullptr);
    glCompileShader(frag);
    opengl.shader = glCreateProgram();
    glAttachShader(opengl.shader, vert);
    glAttachShader(opengl.shader, frag);
    glLinkProgram(opengl.shader);
    glDeleteShader(vert);
    glDeleteShader(frag);

    float quad_vertices[] =
    {
        -1.0f,  1.0f,   0.0f, 1.0f,
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,

        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f
    };

    glGenVertexArrays(1, &opengl.quad_vao);
    glGenBuffers(1, &opengl.quad_vbo);
    glBindVertexArray(opengl.quad_vao);
    glBindBuffer(GL_ARRAY_BUFFER, opengl.quad_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    return;
}

inline void free_opengl(Opengl& opengl)
{
    cudaGraphicsUnregisterResource(opengl.cuda_pbo);
    glDeleteBuffers(1, &opengl.pbo);
    glDeleteTextures(1, &opengl.texture_id);
    glDeleteVertexArrays(1, &opengl.quad_vao);
    glDeleteBuffers(1, &opengl.quad_vbo);
    glDeleteProgram(opengl.shader);
    glfwDestroyWindow(opengl.window);
    glfwTerminate();
}

inline void launch_cuda_kernel(Opengl& opengl, World& world)
{
    size_t size;
    cudaGraphicsMapResources(1, &opengl.cuda_pbo, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&world.pixels, &size, opengl.cuda_pbo);

    dim3 GRID = opengl.grid;
    dim3 BLOCK = opengl.block;
    world.screen_width = opengl.screen_width;
    world.screen_height = opengl.screen_height;
    main_kernel <<<GRID, BLOCK>>> (world);
}

inline void render_screen(Opengl& opengl)
{
    cudaGraphicsUnmapResources(1, &opengl.cuda_pbo, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, opengl.pbo);
    glBindTexture(GL_TEXTURE_2D, opengl.texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, opengl.screen_width, opengl.screen_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(opengl.shader);
    glBindVertexArray(opengl.quad_vao);
    glBindTexture(GL_TEXTURE_2D, opengl.texture_id);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

inline void finish_rendering(Opengl& opengl)
{
    glfwSwapBuffers(opengl.window);
    glfwPollEvents();
    if (glfwGetKey(opengl.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(opengl.window, true);
    }
}

inline int screen_size(const Opengl& opengl)
{
    return opengl.screen_width * opengl.screen_height;
}

