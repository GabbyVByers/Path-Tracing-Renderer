#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <string>
#include "gui.h"
#include "quaternions.h"

struct opengl
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

inline void setup_opengl(opengl& opengl, int screen_width, int screen_height, std::string title, bool full_screen)
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

inline void free_opengl(opengl& opengl)
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

inline int screen_size(const opengl& opengl)
{
    return opengl.screen_width * opengl.screen_height;
}

inline void process_keyboard_mouse_input(opengl& opengl, camera& cam)
{
    cam.buffer_size++;
    vec3 forward = { cam.direction.x, 0.0f, cam.direction.z };
    normalize(forward);
    vec3 right = cam.direction * cam.up;
    vec3 up = { 0.0f, 1.0f, 0.0f };

    float slow = 1.0f;
    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        slow = 0.1f;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_W) == GLFW_PRESS)
    {
        cam.position += forward * slow;
        cam.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_S) == GLFW_PRESS)
    {
        cam.position -= forward * slow;
        cam.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_D) == GLFW_PRESS)
    {
        cam.position += right * slow;
        cam.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_A) == GLFW_PRESS)
    {
        cam.position -= right * slow;
        cam.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        cam.position += up * slow;
        cam.buffer_size = 0;
    }

    if (glfwGetKey(opengl.window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    {
        cam.position -= up * slow;
        cam.buffer_size = 0;
    }

    double curr_mouse_x;
    double curr_mouse_y;
    glfwGetCursorPos(opengl.window, &curr_mouse_x, &curr_mouse_y);
    double mouse_rel_x = curr_mouse_x - opengl.prev_mouse_x;
    double mouse_rel_y = curr_mouse_y - opengl.prev_mouse_y;
    opengl.prev_mouse_x = curr_mouse_x;
    opengl.prev_mouse_y = curr_mouse_y;
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        cam.buffer_size = 0;
        return;
    }

    if (glfwGetMouseButton(opengl.window, GLFW_MOUSE_BUTTON_LEFT) != GLFW_PRESS)
    {
        return;
    }

    if (mouse_rel_x != 0.0f)
    {
        cam.direction = rotate(cam.direction, up, 0.005f * -mouse_rel_x);
        fix_camera(cam);
        cam.buffer_size = 0;
    }

    if (mouse_rel_y != 0.0f)
    {
        cam.direction = rotate(cam.direction, cam.right, 0.005f * -mouse_rel_y);
        fix_camera(cam);
        cam.buffer_size = 0;
    }
}

inline void render_textured_quad(opengl& opengl, camera& cam)
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

    glfwSwapBuffers(opengl.window);
    glfwPollEvents();

    if (glfwGetKey(opengl.window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(opengl.window, true);
    }
}

