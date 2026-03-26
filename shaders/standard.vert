#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 texcoord;

uniform mat4 projection, modelview;
uniform vec3 light_pos;
uniform int u_render_mode; // 0: Nội suy màu, 1: Gouraud, 2: Phong

out vec3 vertPos;
out vec3 normalInterp;
out vec3 colorInterp;
out vec2 texcoordInterp;
out vec3 gouraudLighting; // Biến truyền ánh sáng Gouraud sang Fragment

void main() {
    colorInterp = color;
    texcoordInterp = texcoord;

    vec4 vertPos4 = modelview * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;

    mat4 normal_matrix = transpose(inverse(modelview));
    normalInterp = vec3(normal_matrix * vec4(normal, 0.0));

    // --- TÍNH TOÁN ÁNH SÁNG GOURAUD TẠI ĐỈNH ---
    if (u_render_mode == 1) {
        vec3 N = normalize(normalInterp);
        vec3 L = normalize(light_pos - vertPos);
        vec3 V = normalize(-vertPos);
        vec3 R = reflect(-L, N);

        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

        float diff = max(dot(N, L), 0.0);
        vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

        float specularStrength = 0.5;
        float spec = pow(max(dot(V, R), 0.0), 32.0);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

        gouraudLighting = ambient + diffuse + specular;
    } else {
        gouraudLighting = vec3(1.0, 1.0, 1.0);
    }

    gl_Position = projection * vertPos4;
}