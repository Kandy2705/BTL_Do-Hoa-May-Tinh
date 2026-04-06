#version 330 core

uniform float u_near;
uniform float u_far;

out vec4 frag_color;

float linearize_depth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * u_near * u_far) / (u_far + u_near - z * (u_far - u_near));
}

void main() {
    float linear = linearize_depth(gl_FragCoord.z);
    float normalized = clamp((linear - u_near) / (u_far - u_near), 0.0, 1.0);
    frag_color = vec4(vec3(normalized), 1.0);
}
