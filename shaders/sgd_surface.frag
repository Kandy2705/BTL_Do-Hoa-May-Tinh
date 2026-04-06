#version 330 core

in vec3 vColor;
in vec3 vNormal;
in vec3 vPos;

out vec4 fragColor;

uniform vec3 u_light_dir;
uniform vec3 u_view_dir;
uniform float u_ambient_strength;
uniform float u_diffuse_strength;
uniform float u_specular_strength;
uniform float u_shininess;
uniform float u_tint_mix;
uniform vec3 u_tint_color;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_view_dir);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), u_shininess);

    // Mục tiêu ở đây là kiểu trực quan scientific:
    // màu bề mặt vẫn do colormap quyết định, ánh sáng chỉ phụ để nhìn khối mượt hơn.
    vec3 base = vColor * (u_ambient_strength + u_diffuse_strength * diff)
              + vec3(u_specular_strength * spec);
    vec3 lit = mix(base, u_tint_color, u_tint_mix);

    fragColor = vec4(lit, 1.0);
}
