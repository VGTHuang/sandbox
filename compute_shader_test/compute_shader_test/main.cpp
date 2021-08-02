#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "stb_image.h"
#include "shader_s.h"
#include <texture_loader.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include <stdlib.h>
#include <iostream>
#include <vector>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void CreateSSBO(GLuint &inSSBO, const int arraySize, Shader shader);
float* GetTextureData(GLuint width, GLuint height, GLuint channels, GLuint texID);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Shader *computeShader;

int main()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	std::string glsl_version = "";
#ifdef __APPLE__
	// GL 3.2 + GLSL 150
	glsl_version = "#version 150";
	glfwWindowHint( // required on Mac OS
		GLFW_OPENGL_FORWARD_COMPAT,
		GL_TRUE
	);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
#elif __linux__
	// GL 3.2 + GLSL 150
	glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#elif _WIN32
	// GL 3.0 + GLSL 130 (???)
	glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
#endif
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Sandbox", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	std::cout << glGetString(GL_VERSION) << std::endl;


	// compute shader
	computeShader = new Shader("ComputeShader.glsl");

	GLuint inSSBO; // Shader Storage Buffer Object

	const int arraySize = 8;
	float arrayIn[arraySize];
	CreateSSBO(inSSBO, arraySize, *computeShader);

	// compute
	float *inputData = new float[arraySize];
	float *outputData = new float[arraySize];

	int i;
	for (i = 0; i < arraySize; i++) {
		inputData[i] = i;
	}

	computeShader->use();


	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	//system("pause");
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

float* GetTextureData(GLuint width, GLuint height, GLuint channels, GLuint texID) {
	float* data = new float[width * height * channels];
	glBindTexture(GL_TEXTURE_2D, texID);
	if (channels == 1)    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
	if (channels == 3) glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, data);
	if (channels == 4) glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data);
	glBindTexture(GL_TEXTURE_2D, 0);
	return data;
}

void CreateSSBO(GLuint &inSSBO, const int arraySize, Shader shader)
{
	float *data = new float[arraySize];
	/*
	for (int i = 0; i < arraySize; i++) {
		data[i] = i * 10;
	}
	*/

	glGenBuffers(1, &inSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, inSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, arraySize * sizeof(float), &data[0], GL_DYNAMIC_DRAW);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, inSSBO);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	shader.use();

	int ssbo_binding = 0;
	int block_index = glGetProgramResourceIndex(shader.ID, GL_SHADER_STORAGE_BLOCK, "SSBO");
	glShaderStorageBlockBinding(shader.ID, block_index, ssbo_binding);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_binding, inSSBO);

	glDispatchCompute(4, 1, 1);

	//Synchronize all writes to the framebuffer image
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	GLfloat* read_data = (GLfloat*)glMapBuffer(GL_SHADER_STORAGE_BUFFER,
		GL_READ_ONLY);

	for (int i = 0; i < arraySize; i++) {
		printf("%f ", read_data[i]);
	}
	printf("\n");

	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	/*
	for (int i = 0; i < arraySize; i++) {
		DEBUG(buffer_data[i]);
	}
	*/
	assert(glGetError() == GL_NO_ERROR);

	// Reset bindings
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssbo_binding, 0);
	glBindImageTexture(0, 0, 0, false, 0, GL_READ_WRITE, GL_RGBA32F);
	glUseProgram(0);

	int work_grp_cnt[3];

	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &work_grp_cnt[0]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &work_grp_cnt[1]);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &work_grp_cnt[2]);

	printf("max global (total) work group counts x:%i y:%i z:%i\n",
		work_grp_cnt[0], work_grp_cnt[1], work_grp_cnt[2]);

}