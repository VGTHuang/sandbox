#version 430 core

layout(std430, binding = 0) buffer SSBO {
	float data[];
};

layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

void main() {
	uint ident = gl_GlobalInvocationID.x;
	data[ident] = float(gl_WorkGroupID.x);
}