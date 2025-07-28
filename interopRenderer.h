#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <string>
#include "kernel.h"
#include "dataStructures.h"
#include "quaternions.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

class InteropRenderer
{

public:

    const char* vertexShaderSrc = R"glsl(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTex;
        out vec2 TexCoord;
        void main()
        {
            gl_Position = vec4(aPos.xy, 0.0, 1.0);
            TexCoord = aTex;
        }
    )glsl";

    const char* fragmentShaderSrc = R"glsl(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D screenTexture;
        void main()
        {
            FragColor = texture(screenTexture, TexCoord);
        }
    )glsl";

    int width = 1920;
    int height = 1080;

    GLuint pbo = 0;
    GLuint textureID = 0;
    cudaGraphicsResource* cudaPBO;

    GLFWmonitor* primary = nullptr;
    GLFWwindow* window = nullptr;
    GLuint shader;
    GLuint quadVAO, quadVBO;

    dim3 block;
    dim3 grid;

    double prevMouseX;
    double prevMouseY;

    InteropRenderer(int screenWidth, int screenHeight, std::string title, bool fullScreen)
    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        if (fullScreen)
        {
            primary = glfwGetPrimaryMonitor();
            width = glfwGetVideoMode(primary)->width;
            height = glfwGetVideoMode(primary)->height;
            window = glfwCreateWindow(width, height, title.c_str(), primary, nullptr);
        }
        else
        {
            width = screenWidth;
            height = screenHeight;
            window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        }

        block = dim3(32, 32);
        grid = dim3((width / 32) + 1, (height / 32) + 1);

        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        createPBO();
        createTexture();
        shader = createShaderProgram();
        createFullscreenQuad(quadVAO, quadVBO);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        float scale = 1.5f;
        io.FontGlobalScale = scale;
        //io.IniFilename = nullptr;
        ImGui::GetStyle().ScaleAllSizes(scale);
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        prevMouseX = 0.0f;
        prevMouseY = 0.0f;
    }

    ~InteropRenderer()
    {
        cudaGraphicsUnregisterResource(cudaPBO);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &textureID);
        glDeleteVertexArrays(1, &quadVAO);
        glDeleteBuffers(1, &quadVBO);
        glDeleteProgram(shader);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createPBO()
    {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&cudaPBO, pbo, cudaGraphicsMapFlagsWriteDiscard);
    }

    void createTexture()
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    GLuint createShaderProgram()
    {
        auto compileShader = [](GLenum type, const char* src)
        {
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);
            return shader;
        };

        GLuint vert = compileShader(GL_VERTEX_SHADER, vertexShaderSrc);
        GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSrc);
        GLuint program = glCreateProgram();
        glAttachShader(program, vert);
        glAttachShader(program, frag);
        glLinkProgram(program);
        glDeleteShader(vert);
        glDeleteShader(frag);
        return program;
    }

    void createFullscreenQuad(GLuint& VAO, GLuint& VBO)
    {
        float quadVertices[] =
        {
            -1.0f,  1.0f,   0.0f, 1.0f,
            -1.0f, -1.0f,   0.0f, 0.0f,
             1.0f, -1.0f,   1.0f, 0.0f,

            -1.0f,  1.0f,   0.0f, 1.0f,
             1.0f, -1.0f,   1.0f, 0.0f,
             1.0f,  1.0f,   1.0f, 1.0f
        };
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }

    void launchCudaKernel(Sphere* devSpheres, int numSpheres, Camera camera)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        uchar4* devPtr;
        size_t size;
        cudaGraphicsMapResources(1, &cudaPBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, cudaPBO);
        renderKernel <<<grid, block>>> (devPtr, width, height, devSpheres, numSpheres, camera);
    }

    void processKeyboardInput(Camera& camera)
    {
        vec3 forward = { camera.direction.x, 0.0f, camera.direction.z };
        normalize(forward);
        
        vec3 right = camera.direction * camera.up;
        vec3 up = { 0.0f, 1.0f, 0.0f };

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
            camera.position += forward;

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            camera.position -= forward;

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
            camera.position += right;

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
            camera.position -= right;

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
            camera.position += up;

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
            camera.position -= up;
    }

    void processMouseInput(Camera& camera)
    {
        double currMouseX;
        double currMouseY;
        glfwGetCursorPos(window, &currMouseX, &currMouseY);

        double mouseRelX = currMouseX - prevMouseX;
        double mouseRelY = currMouseY - prevMouseY;

        prevMouseX = currMouseX;
        prevMouseY = currMouseY;

        ImGuiIO& io = ImGui::GetIO();
        bool mouseFree = !io.WantCaptureMouse;

        if (!mouseFree)
            return;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
            return;

        vec3 up = { 0.0f, 1.0f, 0.0f };
        float magicNum = 0.005f;

        if (mouseRelX != 0.0f)
        {
            camera.direction = rotate(camera.direction, up, magicNum * -mouseRelX);
            fixCamera(camera);
        }

        if (mouseRelY != 0.0f)
        {
            camera.direction = rotate(camera.direction, camera.right, magicNum * -mouseRelY);
            fixCamera(camera);
        }
    }

    void renderTexturedQuad(Camera& camera)
    {
        cudaGraphicsUnmapResources(1, &cudaPBO, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(quadVAO);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Debugger");
        ImGui::SliderInt("Rays Per Pixel", &camera.raysPerPixel, 0, 15);

        ImGui::SliderFloat("Light Source x:", &camera.lightDirection.x, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source y:", &camera.lightDirection.y, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source z:", &camera.lightDirection.z, -1.0f, 1.0f);
        normalize(camera.lightDirection);

        ImGui::Text("Camera Position x:%.2f, y:%.2f, z:%.2f", camera.position.x, camera.position.y, camera.position.z);
        ImGui::Text("Camera Direction x:%.2f, y:%.2f, z:%.2f", camera.direction.x, camera.direction.y, camera.direction.z);
        ImGui::Text("Camera Up x:%.2f, y:%.2f, z:%.2f", camera.up.x, camera.up.y, camera.up.z);
        ImGui::Text("Camera Right x:%.2f, y:%.2f, z:%.2f", camera.right.x, camera.right.y, camera.right.z);

        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
};

