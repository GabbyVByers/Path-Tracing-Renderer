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
    int screenWidth = 0;
    int screenHeight = 0;
    GLuint textureId = 0;
    GLuint PBO = 0;
    cudaGraphicsResource* cudaPBO = nullptr;
    GLFWmonitor* primary = nullptr;
    GLFWwindow* window = nullptr;
    GLuint shader = 0;
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;
    dim3 block = 0;
    dim3 grid = 0;
    double prevMouseX = 0.0f;
    double prevMouseY = 0.0f;
};

inline void setupOpengl(Opengl& opengl, int screenWidth, int screenHeight, std::string title, bool fullScreen)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    if (fullScreen)
    {
        opengl.primary = glfwGetPrimaryMonitor();
        opengl.screenWidth = glfwGetVideoMode(opengl.primary)->width;
        opengl.screenHeight = glfwGetVideoMode(opengl.primary)->height;
        opengl.window = glfwCreateWindow(opengl.screenWidth, opengl.screenHeight, title.c_str(), opengl.primary, nullptr);
    }
    else
    {
        opengl.screenWidth = screenWidth;
        opengl.screenHeight = screenHeight;
        opengl.window = glfwCreateWindow(opengl.screenWidth, opengl.screenHeight, title.c_str(), nullptr, nullptr);
    }

    opengl.block = dim3(32, 32);
    opengl.grid = dim3((opengl.screenWidth / 32) + 1, (opengl.screenHeight / 32) + 1);
    glfwMakeContextCurrent(opengl.window);
    glfwSwapInterval(0);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glGenBuffers(1, &opengl.PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, opengl.PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, opengl.screenWidth * opengl.screenHeight * 4, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&opengl.cudaPBO, opengl.PBO, cudaGraphicsMapFlagsWriteDiscard);
    glGenTextures(1, &opengl.textureId);
    glBindTexture(GL_TEXTURE_2D, opengl.textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, opengl.screenWidth, opengl.screenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    const char* vertexShaderSrc = R"glsl(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTex;
        out vec2 texCoord;
        void main() {
            gl_Position = vec4(aPos.xy, 0.0, 1.0);
            texCoord = aTex;
        }
    )glsl";

    const char* fragmentShaderSrc = R"glsl(
        #version 330 core
        in vec2 texCoord;
        out vec4 fragColor;
        uniform sampler2D screenTexture;
        void main() {
            fragColor = texture(screenTexture, texCoord);
        }
    )glsl";

    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert, 1, &vertexShaderSrc, nullptr);
    glCompileShader(vert);
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag, 1, &fragmentShaderSrc, nullptr);
    glCompileShader(frag);
    opengl.shader = glCreateProgram();
    glAttachShader(opengl.shader, vert);
    glAttachShader(opengl.shader, frag);
    glLinkProgram(opengl.shader);
    glDeleteShader(vert);
    glDeleteShader(frag);

    float quadVertices[] =
    {
        -1.0f,  1.0f,   0.0f, 1.0f,
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
         1.0f,  1.0f,   1.0f, 1.0f
    };

    glGenVertexArrays(1, &opengl.quadVAO);
    glGenBuffers(1, &opengl.quadVBO);
    glBindVertexArray(opengl.quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, opengl.quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    glfwSwapInterval(1);
    return;
}

inline void freeOpengl(Opengl& opengl)
{
    cudaGraphicsUnregisterResource(opengl.cudaPBO);
    glDeleteBuffers(1, &opengl.PBO);
    glDeleteTextures(1, &opengl.textureId);
    glDeleteVertexArrays(1, &opengl.quadVAO);
    glDeleteBuffers(1, &opengl.quadVBO);
    glDeleteProgram(opengl.shader);
    glfwDestroyWindow(opengl.window);
    glfwTerminate();
}

inline void launchCudaKernel(Opengl& opengl, World& world)
{
    size_t size;
    cudaGraphicsMapResources(1, &opengl.cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&world.metadata.pixels, &size, opengl.cudaPBO);

    dim3 GRID = opengl.grid;
    dim3 BLOCK = opengl.block;
    world.metadata.screenWidth = opengl.screenWidth;
    world.metadata.screenHeight = opengl.screenHeight;
    mainKernel <<<GRID, BLOCK>>> (world);
}

inline void renderScreen(Opengl& opengl)
{
    cudaGraphicsUnmapResources(1, &opengl.cudaPBO, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, opengl.PBO);
    glBindTexture(GL_TEXTURE_2D, opengl.textureId);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, opengl.screenWidth, opengl.screenHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(opengl.shader);
    glBindVertexArray(opengl.quadVAO);
    glBindTexture(GL_TEXTURE_2D, opengl.textureId);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

inline void finishRendering(Opengl& opengl)
{
    glfwSwapBuffers(opengl.window);
    glfwPollEvents();
    if (glfwGetKey(opengl.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(opengl.window, true);
}

inline int screenSize(const Opengl& opengl)
{
    return opengl.screenWidth * opengl.screenHeight;
}

