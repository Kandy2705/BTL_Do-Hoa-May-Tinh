#version 330 core
precision mediump float;

in vec3 vertPos;
in vec3 normalInterp;
in vec3 colorInterp;
in vec2 texcoordInterp;
in vec3 gouraudLighting;

out vec4 fragColor;

uniform bool u_use_flat_color;
uniform vec3 u_flat_color;
uniform bool u_use_texture;
uniform sampler2D u_texture;
uniform int u_render_mode;
uniform int u_display_mode; // 0: RGB, 1: Depth Map
uniform float u_cam_far;    // Tầm nhìn xa của Camera để chuẩn hóa màu

// --- HỆ THỐNG NHIỀU ĐÈN (TỐI ĐA 4 ĐÈN) ---
#define MAX_LIGHTS 4
uniform int u_num_lights;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform bool u_light_active[MAX_LIGHTS];

void main() {
    if (u_use_flat_color) {
        fragColor = vec4(u_flat_color, 1.0);
        return;
    }

    vec4 baseColor = vec4(colorInterp, 1.0);

    if (u_use_texture) {
        vec4 texColor = texture(u_texture, texcoordInterp);
        baseColor = baseColor * texColor;
    }
    
    // --- CHỌN CHẾ ĐỘ HIỂN THỊ ---
    if (u_display_mode == 1) {
        // --- HIỂN THỊ DEPTH MAP ---
        float dist = -vertPos.z;
        float normalized_depth = clamp(dist / u_cam_far, 0.0, 1.0);
        float color_val = 1.0 - normalized_depth;
        fragColor = vec4(vec3(color_val), 1.0);
        return;
    }
    
    // --- HỆ THỐNG NHIỀU ĐÈN (TỐI ĐA 4 ĐÈN) ---
    vec3 ambient = vec3(0.1, 0.1, 0.1);
    vec3 final_lighting = vec3(0.0, 0.0, 0.0);
    
    vec3 N = normalize(normalInterp);
    vec3 V = normalize(-vertPos);
    
    for (int i = 0; i < u_num_lights; i++) {
        if (!u_light_active[i]) continue;
        
        vec3 L = normalize(u_light_pos[i] - vertPos);
        vec3 R = reflect(-L, N);
        
        float diff = max(dot(N, L), 0.0);
        vec3 diffuse = diff * u_light_color[i] * u_light_intensity[i];
        
        float spec = pow(max(dot(V, R), 0.0), 32.0);
        vec3 specular = 0.5 * spec * u_light_color[i] * u_light_intensity[i];
        
        final_lighting += diffuse + specular;
    }
    
    vec3 lighting = ambient + final_lighting;
    fragColor = vec4(lighting * baseColor.rgb, baseColor.a);
}