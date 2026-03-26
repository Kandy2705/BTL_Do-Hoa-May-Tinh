#version 330 core
precision mediump float;

in vec3 vertPos;
in vec3 normalInterp;
in vec3 colorInterp;
in vec2 texcoordInterp;
in vec3 gouraudLighting; // Nhận kết quả Gouraud từ Vertex

out vec4 fragColor;

// --- CÁC CÔNG TẮC ĐIỀU KHIỂN TỪ PYTHON ---
uniform bool u_use_flat_color;
uniform vec3 u_flat_color;
uniform bool u_use_texture;
uniform sampler2D u_texture;

uniform int u_render_mode; // 0: Nội suy màu, 1: Gouraud, 2: Phong
uniform vec3 light_pos;

void main() {
    // [A] MÀU PHẲNG
    if (u_use_flat_color) {
        fragColor = vec4(u_flat_color, 1.0);
        return;
    }

    vec4 baseColor = vec4(colorInterp, 1.0);

    // [D] TEXTURE
    if (u_use_texture) {
        vec4 texColor = texture(u_texture, texcoordInterp);
        baseColor = baseColor * texColor;
    }

    // [C] LỰA CHỌN CHẾ ĐỘ CHIẾU SÁNG
    if (u_render_mode == 1) {
        // GOURAUD: Lấy ánh sáng đã tính sẵn ở Vertex x Màu vật thể
        fragColor = vec4(gouraudLighting * baseColor.rgb, baseColor.a);
    } 
    else if (u_render_mode == 2) {
        // PHONG: Tự tính toán chi tiết ánh sáng tại từng Pixel
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

        vec3 lighting = ambient + diffuse + specular;
        fragColor = vec4(lighting * baseColor.rgb, baseColor.a);
    } 
    else {
        // KHÔNG CÓ ĐÈN (Color Interpolation): Xuất màu/texture gốc
        fragColor = baseColor;
    }
}