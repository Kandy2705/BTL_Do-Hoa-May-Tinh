#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

uniform mat4 projection;
uniform mat4 modelview;

out vec3 vColor;
out vec3 vNormal;
out vec3 vPos;

void main() {
    vec4 pos4 = modelview * vec4(position, 1.0);
    mat3 normal_matrix = mat3(transpose(inverse(modelview)));

    vColor = color;
    vNormal = normalize(normal_matrix * normal);
    vPos = pos4.xyz;

    gl_Position = projection * pos4;
}
