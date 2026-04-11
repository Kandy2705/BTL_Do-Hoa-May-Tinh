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
uniform int u_display_mode;
uniform float u_cam_far;
uniform mat4 view;
uniform float u_shininess;   //sự sáng bóng của 1 vật

// --- HỆ THỐNG NHIỀU ĐÈN ---
#define MAX_LIGHTS 4
uniform int u_num_lights;
uniform vec3 u_light_pos[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform bool u_light_active[MAX_LIGHTS];
uniform float u_light_range;          // ← THÊM: khoảng cách tối đa của đèn

vec3 compute_face_normal() {
    // dFdx / dFdy cho biết fragment hiện tại thay đổi theo hai hướng màn hình như thế nào.
    // Với bề mặt 3D, tích có hướng cross(dFdx, dFdy) sẽ cho vector pháp tuyến của mặt.
    // Đây là cách rất tiện để tạo "flat normal" theo từng tam giác ngay trong fragment shader.
    vec3 dx = dFdx(vertPos);
    vec3 dy = dFdy(vertPos);
    vec3 face_normal = normalize(cross(dx, dy));

    if (!gl_FrontFacing) {
        face_normal = -face_normal;
    }

    if (length(face_normal) < 1e-5) {
        face_normal = normalize(normalInterp);
    }

    return face_normal;
}

vec3 compute_phong_lighting(vec3 surface_normal) {
    // Công thức Phong ở đây tách làm 3 phần:
    // 1. ambient  : ánh sáng nền tối thiểu
    // 2. diffuse  : max(dot(N, L), 0) theo định luật Lambert
    // 3. specular : pow(max(dot(R, V), 0), shininess) để tạo highlight bóng
    vec3 N = normalize(surface_normal);
    vec3 V = normalize(-vertPos);

    vec3 ambient = vec3(0.46, 0.46, 0.46);
    vec3 final_lighting = vec3(0.0, 0.0, 0.0);

    for (int i = 0; i < u_num_lights; i++) {
        if (!u_light_active[i]) continue;

        vec4 lightPosView = view * vec4(u_light_pos[i], 1.0);
        vec3 L = normalize(lightPosView.xyz - vertPos);

        // Soft attenuation: viewport should feel like broad daylight, not a tiny point lamp.
        float distance = length(lightPosView.xyz - vertPos);
        float lightRange = max(u_light_range, 1.0);
        float rangeFalloff = clamp(1.0 - distance / lightRange, 0.0, 1.0);
        float attenuation = mix(0.58, 1.0, rangeFalloff * rangeFalloff);

        vec3 R = reflect(-L, N);

        float diff = max(dot(N, L), 0.0);
        float wrappedDiff = 0.28 + 0.72 * diff;
        vec3 diffuse = wrappedDiff * u_light_color[i] * u_light_intensity[i] * attenuation;

        float spec = pow(max(dot(V, R), 0.0), u_shininess);
        vec3 specular = 0.28 * spec * u_light_color[i] * u_light_intensity[i] * attenuation;

        final_lighting += diffuse + specular;
    }

    return clamp(ambient + final_lighting, 0.0, 1.35);
}

void main() {
    vec3 finalBaseColor = u_use_flat_color ? u_flat_color : colorInterp;
    vec4 baseColor = vec4(finalBaseColor, 1.0);

    if (u_use_texture) {
        vec4 texColor = texture(u_texture, texcoordInterp);
        baseColor = baseColor * texColor;
    }
    
    if (u_display_mode == 1) {
        // Depth map ở đây lấy độ sâu theo trục nhìn của camera.
        // Sau đó chuẩn hóa về [0, 1] bằng far plane để đổi thành mức xám.
        float dist = -vertPos.z;
        // chuẩn hoá (0,1) u_cam_far là khoảng xa nhất camera nhìn thấy; 
        // dist là độ sâu hiện tại
        float normalized_depth = clamp(dist / u_cam_far, 0.0, 1.0);
        float color_val = 1.0 - normalized_depth;
        fragColor = vec4(vec3(color_val), 1.0);
        return;
    }
    
    vec3 lighting;
    
    if (u_use_flat_color) {
        // Flat shading đúng nghĩa: vẫn dùng lighting,
        // nhưng normal được lấy theo từng mặt chứ không nội suy theo đỉnh.
        lighting = compute_phong_lighting(compute_face_normal());
    }
    else if (u_render_mode == 1) {
        lighting = gouraudLighting;
    } 
    else if (u_render_mode == 2 || u_render_mode == 3) {
        lighting = compute_phong_lighting(normalInterp);
    } 
    else {
        lighting = vec3(1.0, 1.0, 1.0);
    }
    
    fragColor = vec4(clamp(lighting * baseColor.rgb, 0.0, 1.0), baseColor.a);
}
