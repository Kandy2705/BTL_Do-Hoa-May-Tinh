#version 330 core

precision mediump float;  // Độ chính xác trung bình cho floating point

// Input từ vertex shader (đã được nội suy)
in vec3 normal_interp;  // Vector pháp tuyến bề mặt
in vec3 vertPos;        // Vị trí vertex
in vec3 colorInterp;    // Màu gốc của vertex

// Biến toàn cục từ CPU
uniform mat3 K_materials;     // Thuộc tính vật liệu
uniform mat3 I_light;        // Cường độ ánh sáng

uniform int mode;             // Chế độ render
uniform float shininess;      // Độ bóng của bề mặt
uniform vec3 light_pos;       // Vị trí nguồn sáng
uniform int u_display_mode;   // 0: RGB, 1: Bản đồ độ sâu
uniform float u_cam_far;      // Mặt phẳng xa của camera để chuẩn hóa độ sâu

out vec4 fragColor;  // Màu đầu ra cho màn hình

void main() {
  // --- CHẾ ĐỘ BẢN ĐỒ ĐỘ SÂU ---
  if (u_display_mode == 1) {
    float dist = -vertPos.z;  // Khoảng cách đến camera
    float normalized_depth = clamp(dist / u_cam_far, 0.0, 1.0);  // Chuẩn hóa về [0,1]
    float color_val = 1.0 - normalized_depth;  // Đảo ngược: gần = sáng, xa = tối
    fragColor = vec4(vec3(color_val), 1.0);
    return;
  }
  
  // --- CHẾ ĐỘ PHONG SHADING ---
  
  // Chuẩn hóa các vector
  vec3 N = normalize(normal_interp);  // Vector pháp tuyến
  vec3 L = normalize(light_pos - vertPos);  // Vector đến nguồn sáng
  vec3 R = reflect(-L, N);      // Vector phản chiếu ánh sáng
  vec3 V = normalize(-vertPos); // Vector đến người xem

  // Tính các thành phần chiếu sáng theo mô hình Phong
  float specAngle = max(dot(R, V), 0.0);      // Góc phản chiếu
  float specular = pow(specAngle, shininess);    // Cường độ phản chiếu
  vec3 g = vec3(max(dot(L, N), 0.0), specular, 1.0);  // [khuếch tán, phản chiếu, môi trường]
  
  // Kết hợp vật liệu và ánh sáng
  vec3 rgb = 0.5*matrixCompMult(K_materials, I_light) * g + 0.5*colorInterp;

  // Xuất màu cuối cùng
  fragColor = vec4(rgb, 1.0);
}
