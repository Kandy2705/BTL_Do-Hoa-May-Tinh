#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;

uniform vec3 u_base_color;
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

    float diffuse = max(dot(normal, light_dir), 0.0);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);

    vec3 ambient_term = u_ambient_strength * u_base_color;
    vec3 diffuse_term = diffuse * u_light_intensity * u_light_color * u_base_color;
    vec3 specular_term = 0.20 * specular * u_light_color;

    frag_color = vec4(ambient_term + diffuse_term + specular_term, 1.0);
}
