#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 texcoord;

uniform mat4 projection, modelview;
uniform mat4 view;
uniform int u_render_mode;

// --- HỆ THỐNG NHIỀU ĐÈN ---
#define MAX_LIGHTS 4
uniform int u_num_lights;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform bool u_light_active[MAX_LIGHTS];
uniform float u_light_range;          // ← THÊM: khoảng cách tối đa của đèn
uniform float u_shininess;            // ← THÊM: độ bóng

out vec3 vertPos;
out vec3 normalInterp;
out vec3 colorInterp;
out vec2 texcoordInterp;
out vec3 gouraudLighting;

void main() {
    if (u_render_mode == 3) {
        vec3 normPos = normalize(position);
        colorInterp = (normPos + 1.0) / 2.0;
    } else {
        colorInterp = color; 
    }

    texcoordInterp = texcoord;

    vec4 vertPos4 = modelview * vec4(position, 1.0);
    vertPos = vec3(vertPos4) / vertPos4.w;

    mat4 normal_matrix = transpose(inverse(modelview));
    normalInterp = vec3(normal_matrix * vec4(normal, 0.0));

    // ==================== GOURAUD LIGHTING ====================
    if (u_render_mode == 1) {
        vec3 N = normalize(normalInterp);
        vec3 V = normalize(-vertPos);
        vec3 final_lighting = vec3(0.0);
        
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

        for(int i = 0; i < MAX_LIGHTS; i++) {
            if (i >= u_num_lights) break;
            if (!u_light_active[i]) continue;
            
            vec4 lightPosView = view * vec4(u_light_pos[i], 1.0);
            vec3 L = normalize(lightPosView.xyz - vertPos);
            
            // === ATTENUATION + RANGE CUTOFF ===
            float distance = length(lightPosView.xyz - vertPos);
            float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
            attenuation *= clamp(1.0 - distance / u_light_range, 0.0, 1.0);  // fade to 0

            vec3 R = reflect(-L, N);
            
            float diff = max(dot(N, L), 0.0);
            vec3 diffuse = diff * u_light_color[i] * u_light_intensity[i] * attenuation;
            
            float spec = pow(max(dot(V, R), 0.0), u_shininess);
            vec3 specular = 0.5 * spec * u_light_color[i] * u_light_intensity[i] * attenuation;
            
            final_lighting += diffuse + specular;
        }
        gouraudLighting = ambient + final_lighting;
    } else {
        gouraudLighting = vec3(1.0, 1.0, 1.0);
    }

    gl_Position = projection * vertPos4;
}