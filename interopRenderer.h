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

class interop_renderer {

public:

    const char* vertex_shader_src = R"glsl(
        #version 330 core
        layout (location = 0) in vec2 a_pos;
        layout (location = 1) in vec2 a_tex;
        out vec2 tex_coord;
        void main() {
            gl_Position = vec4(a_pos.xy, 0.0, 1.0);
            tex_coord = a_tex;
        }
    )glsl";

    const char* fragment_shader_src = R"glsl(
        #version 330 core
        in vec2 tex_coord;
        out vec4 frag_color;
        uniform sampler2D screen_texture;
        void main() {
            frag_color = texture(screen_texture, tex_coord);
        }
    )glsl";

    int width = 1920;
    int height = 1080;

    GLuint pbo = 0;
    GLuint texture_id = 0;
    cudaGraphicsResource* cuda_pbo;

    GLFWmonitor* primary = nullptr;
    GLFWwindow* window = nullptr;
    GLuint shader;
    GLuint quad_vao, quad_vbo;

    dim3 block;
    dim3 grid;

    double prev_mouse_x;
    double prev_mouse_y;

    interop_renderer(int screen_width, int screen_height, std::string title, bool full_screen) {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        if (full_screen) {
            primary = glfwGetPrimaryMonitor();
            width = glfwGetVideoMode(primary)->width;
            height = glfwGetVideoMode(primary)->height;
            window = glfwCreateWindow(width, height, title.c_str(), primary, nullptr);
        } else {
            width = screen_width;
            height = screen_height;
            window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        }

        block = dim3(32, 32);
        grid = dim3((width / 32) + 1, (height / 32) + 1);

        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        create_pbo();
        create_texture();
        shader = create_shader_program();
        create_fullscreen_quad(quad_vao, quad_vbo);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        float scale = 1.5f;
        io.FontGlobalScale = scale;
        ImGui::GetStyle().ScaleAllSizes(scale);
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        prev_mouse_x = 0.0f;
        prev_mouse_y = 0.0f;
    }

    ~interop_renderer() {
        cudaGraphicsUnregisterResource(cuda_pbo);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &texture_id);
        glDeleteVertexArrays(1, &quad_vao);
        glDeleteBuffers(1, &quad_vbo);
        glDeleteProgram(shader);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void create_pbo() {
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 4, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
    }

    void create_texture() {
        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    GLuint create_shader_program() {
        auto compile_shader = [](GLenum type, const char* src) { // EXTRACT ME
            GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &src, nullptr);
            glCompileShader(shader);
            return shader;
        };

        GLuint vert = compile_shader(GL_VERTEX_SHADER, vertex_shader_src);
        GLuint frag = compile_shader(GL_FRAGMENT_SHADER, fragment_shader_src);
        GLuint program = glCreateProgram();
        glAttachShader(program, vert);
        glAttachShader(program, frag);
        glLinkProgram(program);
        glDeleteShader(vert);
        glDeleteShader(frag);
        return program;
    }

    void create_fullscreen_quad(GLuint& VAO, GLuint& VBO) {
        float quad_vertices[] = {
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
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
    }

    void launch_cuda_kernel(sphere* dev_spheres, int num_spheres, camera cam) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        uchar4* dev_ptr;
        size_t size;
        cudaGraphicsMapResources(1, &cuda_pbo, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, cuda_pbo);
        renderKernel <<<grid, block>>> (dev_ptr, width, height, dev_spheres, num_spheres, cam);
    }

    void process_keyboard_mouse_input(camera& cam) {
        cam.buffer_size++;
        vec3 forward = { cam.direction.x, 0.0f, cam.direction.z };
        normalize(forward);
        vec3 right = cam.direction * cam.up;
        vec3 up = { 0.0f, 1.0f, 0.0f };

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cam.position += forward;
            cam.buffer_size = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cam.position -= forward;
            cam.buffer_size = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cam.position += right;
            cam.buffer_size = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cam.position -= right;
            cam.buffer_size = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            cam.position += up;
            cam.buffer_size = 0;
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
            cam.position -= up;
            cam.buffer_size = 0;
        }

        double curr_mouse_x;
        double curr_mouse_y;
        glfwGetCursorPos(window, &curr_mouse_x, &curr_mouse_y);
        double mouse_rel_x = curr_mouse_x - prev_mouse_x;
        double mouse_rel_y = curr_mouse_y - prev_mouse_y;
        prev_mouse_x = curr_mouse_x;
        prev_mouse_y = curr_mouse_y;
        ImGuiIO& io = ImGui::GetIO();

        if (io.WantCaptureMouse)
            return;

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
            return;

        if (mouse_rel_x != 0.0f) {
            cam.direction = rotate(cam.direction, up, 0.005f * -mouse_rel_x);
            fix_camera(cam);
            cam.buffer_size = 0;
        }

        if (mouse_rel_y != 0.0f) {
            cam.direction = rotate(cam.direction, cam.right, 0.005f * -mouse_rel_y);
            fix_camera(cam);
            cam.buffer_size = 0;
        }
    }

    void render_textured_quad(camera& cam) {
        cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shader);
        glBindVertexArray(quad_vao);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Debugger");
        ImGui::SliderInt("Buffer Limit", &cam.buffer_limit, 0, 500);
        ImGui::SliderInt("Rays Per Pixel", &cam.rays_per_pixel, 0, 100);
        ImGui::SliderFloat("Light Source x:", &cam.light_direction.x, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source y:", &cam.light_direction.y, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source z:", &cam.light_direction.z, -1.0f, 1.0f);
        normalize(cam.light_direction);
        ImGui::Text("Camera Position x:%.2f, y:%.2f, z:%.2f", cam.position.x, cam.position.y, cam.position.z);
        ImGui::Text("Camera Direction x:%.2f, y:%.2f, z:%.2f", cam.direction.x, cam.direction.y, cam.direction.z);
        ImGui::Text("Camera Up x:%.2f, y:%.2f, z:%.2f", cam.up.x, cam.up.y, cam.up.z);
        ImGui::Text("Camera Right x:%.2f, y:%.2f, z:%.2f", cam.right.x, cam.right.y, cam.right.z);
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    int screen_size() {
        return width * height;
    }
};

