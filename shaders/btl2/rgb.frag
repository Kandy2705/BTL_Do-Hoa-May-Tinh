#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_texcoord;

uniform vec3 u_base_color;
uniform bool u_use_texture;
uniform sampler2D u_texture;
uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_ambient_strength;

out vec4 frag_color;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(-u_light_dir);
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);

    // Half-Lambert keeps silhouettes readable and makes the synthetic scene feel daylight-lit.
    float diffuse = max(dot(normal, light_dir), 0.0);
    float wrapped_diffuse = 0.42 + 0.58 * diffuse;
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 24.0);
    vec3 albedo = u_base_color;
    if (u_use_texture) {
        albedo = texture(u_texture, v_texcoord).rgb;
    }

    vec3 ambient_term = max(u_ambient_strength, 0.46) * albedo;
    vec3 diffuse_term = wrapped_diffuse * u_light_intensity * u_light_color * albedo * 0.74;
    vec3 specular_tint = mix(albedo, vec3(1.0), 0.18);
    vec3 specular_term = 0.035 * specular * specular_tint * u_light_color;

    vec3 color = clamp(ambient_term + diffuse_term + specular_term, 0.0, 1.0);
    frag_color = vec4(color, 1.0);
}
